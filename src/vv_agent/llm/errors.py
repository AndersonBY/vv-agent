"""Provider error classification shared by model callers and compaction."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import import_module
from typing import Any

from vv_agent.llm.base import LlmRequestError

MAX_PTL_RETRIES = 3


@lru_cache(maxsize=1)
def _http_types() -> tuple[tuple[type, ...], tuple[type, ...], tuple[type, ...]]:
    responses, status_errors, transports = [], [], []
    for name in ("httpx2", "httpx"):
        try:
            module = import_module(name)
        except ImportError:
            continue
        responses.append(module.Response)
        status_errors.append(module.HTTPStatusError)
        transports.extend((module.TransportError, module.DecodingError))
    return tuple(responses), tuple(status_errors), tuple(transports)


def _provider_connection_types() -> tuple[type, ...]:
    types = []
    for name in ("openai", "anthropic"):
        try:
            candidate = import_module(name).APIConnectionError
        except Exception:
            continue
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            types.append(candidate)
    return tuple(types)


def _safe_message(message: str) -> str:
    # Even prose without credential markers can echo private input or a bare key.
    return "Model provider request failed."


def _safe_code(value: object) -> str | None:
    if type(value) is int:
        value = str(value)
    if (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value)
        and not re.search(r"(?i)api[_.-]?key|token|secret|authorization|credential|bearer|sk-", value)
    ):
        return value
    return None


class ModelRequestRejected(LlmRequestError):
    """An adapter confirms the provider rejected a request before execution.

    Only 4xx other than 408 qualify. Explicit retryable overrides routing policy;
    otherwise 429 retries, and 401/403/404 retry only with multiple frozen endpoints.
    Never use this after stream output.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        provider_code: str | None = None,
        provider_type: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        if type(status_code) is not int or not 400 <= status_code < 500 or status_code == 408:
            raise ValueError("a definitive rejection requires HTTP 4xx other than 408")
        if retryable is not None and type(retryable) is not bool:
            raise ValueError("retryable must be a boolean")
        self.status_code = status_code
        self._prompt_too_long = _matches_prompt_too_long(
            status_code, (" ".join(str(value or "") for value in (message, provider_code, provider_type)),)
        )
        self.message = _safe_message(message)
        self.provider_code = _safe_code(provider_code)
        self.provider_type = _safe_code(provider_type)
        self._retryable_override = retryable
        self.retryable = status_code == 429 if retryable is None else retryable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "provider_code": self.provider_code,
            "provider_type": self.provider_type,
            "message": self.message,
            "retryable": self.retryable,
        }


def model_response_rejection(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Only completed provider conclusions qualify; never retain response prose."""
    reason = metadata.get("finish_reason_normalized")
    if reason != "content_filter" and not (reason == "refusal" and not metadata.get("refusal")):
        return None
    return {
        "status_code": 200,
        "provider_code": reason,
        "provider_type": _safe_code(metadata.get("finish_reason")),
        "message": "Model provider response was rejected.",
        "retryable": False,
    }


class ModelStreamInterrupted(LlmRequestError):
    """The provider stream has no confirmed complete result; execution is ambiguous."""


def _provider_error_detail(error: BaseException, response: Any) -> dict[str, Any]:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        try:
            body = response.json()
        except (ValueError, RuntimeError):
            # An unread error body must not hide the received HTTP status or text.
            body = None
    detail = body.get("error", body) if isinstance(body, dict) else {}
    return detail if isinstance(detail, dict) else {}


def _exception_graph(error: BaseException) -> list[BaseException]:
    graph, stack, seen = [], [error], set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.append(current)
        stack.extend(item for item in (current.__cause__, current.__context__, *current.args) if isinstance(item, BaseException))
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
    return graph


def _uncertain(graph: list[BaseException]) -> bool:
    responses, status_errors, transports = _http_types()
    for current in graph:
        if isinstance(
            current, (ModelStreamInterrupted, TimeoutError, ConnectionError, *transports, *_provider_connection_types())
        ):
            return True
        observation = getattr(current, "_vv_http_observation", None)
        if observation is not None and (observation.uncertain or observation.outcomes) and not observation.all_rejected:
            return True
        response = getattr(current, "response", None)
        if response is not None and isinstance(response, responses):
            observed = response.extensions.get("vv_agent_http_observation")
            if observed is not None and not observed.all_rejected:
                return True
            status = response.status_code if isinstance(current, status_errors) else getattr(current, "status_code", None)
            if type(status) is not int or status != response.status_code or not 400 <= status < 500 or status == 408:
                return True
    return False


def safe_model_error(error: Exception) -> str:
    """A payload-free diagnostic, including for unobservable repair callbacks."""
    for current in _exception_graph(error):
        if isinstance(current, ModelRequestRejected):
            return f"HTTP {current.status_code}: {current.message}"
        response = getattr(current, "response", None)
        if response is not None and isinstance(response, _http_types()[0]):
            return f"HTTP {response.status_code}: Model provider request failed."
    return "Model provider request failed."


def _explicit_rejection(error: BaseException) -> ModelRequestRejected | None:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelRequestRejected):
            return current
        current = current.__cause__
    return None


def model_request_rejection(error: Exception, *, endpoint_count: int = 1) -> ModelRequestRejected | None:
    """Recognize rejection evidence after the caller checks prompt-too-long recovery."""
    graph = _exception_graph(error)
    if _uncertain(graph):
        return None
    explicit = _explicit_rejection(error)
    if explicit is not None:
        if explicit._retryable_override is None and explicit.status_code in {401, 403, 404} and endpoint_count > 1:
            return ModelRequestRejected(
                explicit.status_code,
                explicit.message,
                provider_code=explicit.provider_code,
                provider_type=explicit.provider_type,
                retryable=True,
            )
        return explicit
    # Only explicit causes provide positive evidence. Context/args/groups veto
    # uncertainty but cannot assert that this invocation was rejected.
    observations = []
    unobserved = False
    seen: set[int] = set()
    current: BaseException | None = error
    responses, status_errors, _ = _http_types()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        observation = getattr(current, "_vv_http_observation", None)
        if observation is not None:
            if observation.all_rejected:
                observations.append(observation)
            unobserved |= not observation.outcomes
        response = getattr(current, "response", None)
        status = getattr(current, "status_code", None)
        if response is None:
            current = current.__cause__
            continue
        if isinstance(current, status_errors) and isinstance(response, responses):
            status = response.status_code
        if response is not None and isinstance(response, responses):
            observed = response.extensions.get("vv_agent_http_observation")
            if unobserved or not any(observed is item for item in observations):
                current = current.__cause__
                continue
            if type(status) is not int or status != response.status_code:
                return None
            if not 400 <= status < 500 or status == 408:
                return None
            detail = _provider_error_detail(current, response)
            message = detail.get("message")
            if not isinstance(message, str):
                message = response.reason_phrase
            return ModelRequestRejected(
                status,
                message if isinstance(message, str) else response.reason_phrase,
                provider_code=detail.get("code"),
                provider_type=detail.get("type"),
                retryable=status == 429 or (status in {401, 403, 404} and endpoint_count > 1),
            )
        current = current.__cause__
    return None


_PTL_429_PATTERNS = (
    "prompt is too long",
    "prompt_too_long",
    "context_length_exceeded",
    "maximum context length",
)
_PTL_EXPLICIT_PATTERNS = (*_PTL_429_PATTERNS, "context window exceeded", "model context window is")
_PTL_GENERIC_PATTERNS = ("request too large", "too many tokens")
_PTL_QUOTA_SIGNAL = re.compile(
    r"\b(?:per\s+(?:day|hour|minute|min|second|sec|month)|tokens\s+per\s+min|"
    r"tpm|tpd|rpm|quota|rate\s+limit|rate_limit|insufficient_quota)\b",
    re.IGNORECASE,
)


def _matches_prompt_too_long(status: int | None, texts: tuple[str, ...], *, code: object = None, param: object = None) -> bool:
    if status == 413:
        return True
    # Other string limits (tool names, ids, etc.) cannot be fixed by compaction.
    if (
        code == "string_above_max_length"
        and isinstance(param, str)
        and (param in {"prompt", "input"} or re.fullmatch(r"(?:messages|input)\[\d+\]\.content(?:\[\d+\]\.text)?", param))
    ):
        return True
    patterns = _PTL_429_PATTERNS if status == 429 else _PTL_EXPLICIT_PATTERNS
    for text in texts:
        text = text.lower()
        if any(pattern in text for pattern in patterns):
            return True
        if status != 429 and not _PTL_QUOTA_SIGNAL.search(text) and any(pattern in text for pattern in _PTL_GENERIC_PATTERNS):
            return True
    return False


def is_prompt_too_long_error(error: Exception) -> bool:
    graph = _exception_graph(error)
    if _uncertain(graph):
        return False
    if _explicit_rejection(error) is None:
        observations = [getattr(item, "_vv_http_observation", None) for item in graph]
        for current in graph:
            response = getattr(current, "response", None)
            if response is not None and isinstance(response, _http_types()[0]):
                observed = response.extensions.get("vv_agent_http_observation")
                if observed is None or not any(observed is item for item in observations) or not observed.all_rejected:
                    return False
    matched = False
    responses, _, _ = _http_types()
    # Text-only wrappers must not turn a 429's generic wording into input-size evidence.
    wrapper_status = (
        429
        if any(
            (isinstance(current, ModelRequestRejected) and current.status_code == 429)
            or (
                isinstance(response := getattr(current, "response", None), responses)
                and getattr(response, "status_code", None) == 429
            )
            for current in graph
        )
        else None
    )
    for current in graph:
        response = getattr(current, "response", None)
        detail = {}
        status = wrapper_status
        if response is not None and isinstance(response, responses):
            status = response.status_code
            detail = _provider_error_detail(current, response)
        elif isinstance(current, ModelRequestRejected):
            status = current.status_code
            detail = {"code": current.provider_code, "type": current.provider_type, "message": current.message}
            matched |= current._prompt_too_long

        matched |= _matches_prompt_too_long(
            status,
            (str(current), " ".join(str(detail.get(key, "")) for key in ("code", "type", "message"))),
            code=detail.get("code"),
            param=detail.get("param"),
        )

    return matched
