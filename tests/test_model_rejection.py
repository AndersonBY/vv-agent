"""Conservative HTTP evidence and adapter-owned rejection receipts."""

from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from model_context_cases import CONTEXT_QUOTA_CASES
from session.test_model_rejection import context_http_error, http_error, observe_error

from vv_agent.llm.errors import ModelRequestRejected, ModelStreamInterrupted, is_prompt_too_long_error, model_request_rejection


@pytest.mark.parametrize("name,status,detail,expected", CONTEXT_QUOTA_CASES)
@pytest.mark.parametrize("provider", ["httpx2", "openai", "anthropic"])
def test_context_quota_rules_on_real_transport(name, status, detail, expected, provider):
    from importlib import import_module

    from vv_agent.llm.http_observation import HttpObservation, ObservedHttpClient

    observation = HttpObservation()
    with ObservedHttpClient(
        observation, transport=httpx2.MockTransport(lambda request: httpx2.Response(status, json={"error": detail}))
    ) as client:
        response = client.post("https://provider.invalid")
        if provider == "httpx2":
            with pytest.raises(httpx2.HTTPStatusError) as caught:
                response.raise_for_status()
            error = caught.value
        else:
            error = import_module(provider).APIStatusError("Rejected", response=response, body={"error": detail})
        cast(Any, error)._vv_http_observation = observation
        assert observation.all_rejected
        assert is_prompt_too_long_error(error) is expected
        explicit = ModelRequestRejected(
            status, detail.get("message", "Rejected"), provider_code=detail.get("code"), provider_type=detail.get("type")
        )
        assert explicit._prompt_too_long is expected
        assert is_prompt_too_long_error(explicit) is expected


@pytest.mark.parametrize(
    "signal",
    [
        "per day",
        "per hour",
        "per minute",
        "per min",
        "per second",
        "per sec",
        "per month",
        "tokens per min",
        "TPM",
        "TPD",
        "RPM",
        "quota",
        "rate limit",
        "rate_limit",
        "insufficient_quota",
    ],
)
@pytest.mark.parametrize("pattern", ["too many tokens", "request too large"])
def test_generic_context_patterns_reject_time_windows_and_quota(signal, pattern):
    assert not is_prompt_too_long_error(ModelRequestRejected(400, f"{pattern}: {signal}"))


@pytest.mark.parametrize("suffix", ["per request", "per minimum", "per monthly", "quotable", "TPMs", "RPMs", "rate limited"])
def test_quota_signals_require_word_boundaries(suffix):
    assert is_prompt_too_long_error(ModelRequestRejected(400, f"too many tokens {suffix}"))


@pytest.mark.parametrize(
    "pattern", ["prompt is too long", "prompt_too_long", "context_length_exceeded", "maximum context length"]
)
@pytest.mark.parametrize("status", [400, 429])
def test_explicit_context_patterns_win_over_quota(status, pattern):
    error = ModelRequestRejected(status, f"{pattern}: quota per day")
    assert error._prompt_too_long
    assert is_prompt_too_long_error(error)


@pytest.mark.parametrize(
    "pattern",
    [
        "too many tokens",
        "request too large",
        "too many tokens per request",
        "context window exceeded",
        "model context window is 128k",
    ],
)
def test_429_only_accepts_context_allowlist_even_through_wrappers(pattern):
    error = http_error(429, pattern)
    wrapper = RuntimeError(pattern)
    wrapper.__cause__ = error
    assert not is_prompt_too_long_error(wrapper)


def test_quota_veto_stays_local_to_matching_text():
    error = RuntimeError("too many tokens per request")
    error.__cause__ = ModelRequestRejected(400, "quota")
    assert is_prompt_too_long_error(error)


@pytest.mark.parametrize("field", ["message", "code", "type"])
@pytest.mark.parametrize("status", [400, 429])
def test_quota_veto_applies_to_provider_fields(field, status):
    error = http_error(status, "Rejected")
    error.body = {"error": {field: "request too large: tokens per min (TPM)"}}
    assert not is_prompt_too_long_error(error)


@pytest.mark.parametrize(
    "param,expected",
    [
        ("prompt", True),
        ("input", True),
        ("messages[0].content", True),
        ("input[12].content[0].text", True),
        ("tools[0].function.name", False),
        (None, False),
        ("messages[0].content_extra", False),
    ],
)
def test_429_string_limit_preserves_content_param_rule(param, expected):
    error = http_error(429, "quota")
    error.body = {"error": {"code": "string_above_max_length", "param": param, "message": "quota"}}
    assert is_prompt_too_long_error(error) is expected


def test_413_and_bare_text_share_quota_rules():
    assert ModelRequestRejected(413, "quota per day")._prompt_too_long
    assert is_prompt_too_long_error(ModelRequestRejected(413, "quota per day"))
    assert not is_prompt_too_long_error(RuntimeError("too many tokens per day"))
    assert is_prompt_too_long_error(RuntimeError("context window exceeded: quota"))


@pytest.mark.parametrize("transport", ["httpx2", "httpx"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("kind", ["openai", "typed", "anthropic", "moonshot", "payload", "content_string"])
def test_real_http_context_errors_are_prompt_too_long(transport, wrapped, kind):
    pytest.importorskip(transport)
    assert is_prompt_too_long_error(context_http_error(kind, transport=transport, wrapped=wrapped))


def test_string_limit_on_unrelated_field_is_not_prompt_too_long():
    error = context_http_error("tool_string")
    assert not is_prompt_too_long_error(error)
    assert model_request_rejection(error) is not None


def test_payload_status_without_error_body_still_requires_smaller_input():
    request = httpx2.Request("POST", "https://provider.invalid")
    error = httpx2.HTTPStatusError("Payload exceeds limit", request=request, response=httpx2.Response(413))
    observe_error(error, 413)
    assert is_prompt_too_long_error(error)


def test_explicit_context_rejection_uses_provider_code():
    assert is_prompt_too_long_error(ModelRequestRejected(400, "Input exceeds limit", provider_code="context_length_exceeded"))


@pytest.mark.parametrize(
    "status,message,too_long",
    [(400, "prompt is too long", True), (413, "Payload exceeds limit", True), (400, "Empty content", False)],
)
def test_unread_http_error_body_does_not_hide_text_or_status(status, message, too_long):
    request = httpx2.Request("POST", "https://provider.invalid")
    response = httpx2.Response(status, request=request, stream=httpx2.ByteStream(b"unread error body"))
    error = observe_error(httpx2.HTTPStatusError(message, request=request, response=response), status)
    try:
        assert is_prompt_too_long_error(error) is too_long
        if not too_long:
            rejection = model_request_rejection(error)
            assert rejection is not None and rejection.message == "Model provider request failed."
        assert not response.is_stream_consumed
    finally:
        response.close()


@pytest.mark.parametrize("kind", ["timeout", "connection", "transport", "stream", "408", "500"])
def test_ambiguity_overrides_prompt_too_long_evidence(kind):
    inner = context_http_error("openai")
    errors = {
        "timeout": TimeoutError("maximum context length"),
        "connection": ConnectionError("prompt is too long"),
        "transport": httpx2.ReadError("too many tokens"),
        "stream": ModelStreamInterrupted("request too large"),
        "408": httpx2.HTTPStatusError(
            "prompt is too long", request=httpx2.Request("POST", "https://provider.invalid"), response=httpx2.Response(408)
        ),
        "500": httpx2.HTTPStatusError(
            "prompt is too long", request=httpx2.Request("POST", "https://provider.invalid"), response=httpx2.Response(500)
        ),
    }
    outer = errors[kind]
    outer.__cause__ = inner
    assert not is_prompt_too_long_error(outer)
    wrapper = RuntimeError("maximum context length")
    wrapper.__cause__ = outer
    assert not is_prompt_too_long_error(wrapper)


def test_explicit_adapter_rejection_is_validated_and_sanitized():
    error = ModelRequestRejected(409, "Busy Authorization: Bearer sk-secret", retryable=True, provider_code="conflict")
    assert model_request_rejection(error) is error
    assert error.retryable is True
    assert error.provider_code == "conflict"
    assert "sk-secret" not in str(error)
    for status in [True, "400", 200, 408, 500]:
        with pytest.raises(ValueError):
            ModelRequestRejected(cast(Any, status), "invalid")


@pytest.mark.parametrize("endpoint_count", [0, 1, 2, 3])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429, 499])
def test_default_retry_classification_uses_frozen_endpoint_count(status, endpoint_count):
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://provider.invalid"))
    error = observe_error(httpx2.HTTPStatusError("Rejected", request=response.request, response=response), status)
    rejection = model_request_rejection(error, endpoint_count=endpoint_count)
    assert rejection is not None
    expected = status == 429 or (status in {401, 403, 404} and endpoint_count > 1)
    assert rejection.retryable is expected
    explicit_default = model_request_rejection(ModelRequestRejected(status, "Rejected"), endpoint_count=endpoint_count)
    assert explicit_default is not None and explicit_default.retryable is expected
    for override in (False, True):
        explicit = ModelRequestRejected(status, "Rejected", retryable=override)
        assert model_request_rejection(explicit, endpoint_count=endpoint_count) is explicit
        assert explicit.retryable is override


def test_only_explicit_cause_chain_is_followed():
    rejected = ModelRequestRejected(400, "invalid")
    outer = RuntimeError("wrapper")
    outer.__cause__ = rejected
    assert model_request_rejection(outer) is rejected
    outer.__cause__ = outer
    assert model_request_rejection(outer) is None
    outer.__cause__ = None
    outer.__context__ = rejected
    assert model_request_rejection(outer) is None
    for error in [TimeoutError("timeout"), ConnectionError("lost"), ModelStreamInterrupted("partial")]:
        error.__cause__ = rejected
        assert model_request_rejection(error) is None


@pytest.mark.parametrize("status,response", [(400, None), ("400", None), (True, None), (400, SimpleNamespace(status_code=400))])
def test_bare_or_forged_status_is_not_http_evidence(status, response):
    error: Any = RuntimeError("invalid")
    error.status_code = status
    error.response = response
    assert model_request_rejection(error) is None


@pytest.mark.parametrize("transport", ["httpx2", "httpx"])
def test_actual_http_status_error(transport):
    module = pytest.importorskip(transport)
    response = module.Response(
        422,
        request=module.Request("POST", "https://user:password@example.invalid/?api_key=secret"),
        json={"error": {"code": "invalid", "type": "request_error", "message": "Empty user message"}},
    )
    error = module.HTTPStatusError("unsafe request dump", request=response.request, response=response)
    observe_error(error, error.response.status_code)
    rejected = model_request_rejection(error)
    assert rejected is not None
    assert rejected.message == "Model provider request failed."
    assert rejected.provider_code == "invalid"
    assert rejected.provider_type == "request_error"


def test_mismatched_status_stays_unknown():
    error: Any = RuntimeError("invalid")
    error.status_code = 400
    error.response = httpx2.Response(500)
    assert model_request_rejection(error) is None


def test_secrets_are_removed_before_message_truncation():
    error = ModelRequestRejected(
        401,
        "API key=supersecret\nAuthorization: Bearer topsecret\nsk-secret-token\n" + "z" * 700,
        provider_code="sk-secret-code",
        provider_type="Authorization: Bearer supersecret",
    )
    assert len(error.message) <= 500
    assert all(secret not in error.message for secret in ["supersecret", "topsecret", "sk-secret-token", "Bearer"])
    assert error.provider_code is None and error.provider_type is None


def test_exception_request_dump_is_never_the_diagnostic():
    from openai import APIStatusError

    response = httpx2.Response(400, request=httpx2.Request("POST", "https://provider.invalid"))
    error = APIStatusError("Authorization: Bearer private; complete request body", response=response, body=None)
    observe_error(error, error.response.status_code)
    rejected = model_request_rejection(error)
    assert rejected is not None and rejected.message == "Model provider request failed."


def test_multiline_request_body_is_omitted():
    error = ModelRequestRejected(400, 'Invalid user. Request body: {\n"messages": ["private content"]\n}')
    assert error.message == "Model provider request failed."


@pytest.mark.parametrize("body", ['{"model":"m", "messages":["private input"]}', '{"headers":{"X-Custom":"private header"}}'])
def test_echoed_request_json_is_omitted(body):
    error = ModelRequestRejected(400, f"Invalid user: {body}")
    assert "private" not in error.message
    assert error.message == "Model provider request failed."


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_status_exception_and_vv_llm_wrapper(provider):
    from importlib import import_module

    from vv_llm.types import classify_exception

    cls = import_module(provider).APIStatusError
    response = httpx2.Response(403, request=httpx2.Request("POST", "https://provider.invalid"))
    error = cls("unsafe dump", response=response, body={"error": {"code": 1234, "type": "denied", "message": "Denied"}})
    wrapper = classify_exception(error)
    # vv-llm middleware/registry raises the classified wrapper from its source.
    wrapper.__cause__ = error
    observe_error(wrapper, 403)
    rejected = model_request_rejection(wrapper)
    assert rejected is not None
    assert rejected.to_dict() == {
        "status_code": 403,
        "provider_code": "1234",
        "provider_type": "denied",
        "message": "Model provider request failed.",
        "retryable": False,
    }
