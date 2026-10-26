from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from vv_agent.agent import Agent
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import (
    ApprovalDecision as WireApprovalDecision,
)
from vv_agent.app_server.protocol import (
    ApprovalRequestParams,
    AppServerError,
    AppServerErrorCode,
    ModelListRequest,
    ModelListResponse,
    ModelSummary,
    RequestId,
)
from vv_agent.approval import ApprovalDecision as RuntimeApprovalDecision
from vv_agent.approval import ApprovalProvider, ApprovalRequest
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.base import LLMClient
from vv_agent.model import resolve_model_selection
from vv_agent.run_config import RunConfig


@dataclass(frozen=True, slots=True)
class AgentResolutionRequest:
    thread_id: str
    agent_key: str
    cwd: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class RunConfigResolutionRequest:
    thread_id: str
    agent_key: str
    cwd: str | None = None
    metadata: dict[str, object] | None = None


class AppServerHost(Protocol):
    def authorize(self, connection_id: str, method: str, params: dict[str, Any]) -> bool:
        raise NotImplementedError

    def resolve_model(self, request: dict[str, Any]) -> tuple[LLMClient, ResolvedModelConfig]:
        raise NotImplementedError

    def resolve_agent(self, request: AgentResolutionRequest) -> Agent:
        raise NotImplementedError

    def build_run_config(self, request: RunConfigResolutionRequest) -> RunConfig:
        raise NotImplementedError

    def list_models(self, request: ModelListRequest) -> ModelListResponse:
        raise NotImplementedError


class DefaultAppServerHost:
    def __init__(
        self,
        *,
        agent: Agent | None = None,
        run_config: RunConfig | None = None,
        models: list[ModelSummary] | None = None,
    ) -> None:
        self._agent = agent
        self._run_config = run_config
        self._models = list(models or [])

    def authorize(self, connection_id: str, method: str, params: dict[str, Any]) -> bool:
        """Local, single-user host: all initialized connections are trusted."""
        return True

    def resolve_agent(self, request: AgentResolutionRequest) -> Agent:
        if self._agent is not None:
            return self._agent
        return Agent(
            name=request.agent_key or "assistant",
            instructions="You are the default vv-agent App Server assistant.",
        )

    def build_run_config(self, request: RunConfigResolutionRequest) -> RunConfig:
        if self._run_config is not None:
            return self._run_config
        return RunConfig(workspace=request.cwd, metadata={"agent_key": request.agent_key})

    def resolve_model(self, request: dict[str, Any]) -> tuple[LLMClient, ResolvedModelConfig]:
        provider = self._run_config.model_provider if self._run_config else None
        if provider is None:
            raise ValueError("ModelProvider is required for model selection")
        return resolve_model_selection(provider, request)

    def list_models(self, request: ModelListRequest) -> ModelListResponse:
        return ModelListResponse(models=[m for m in self._models if request.provider is None or m.provider == request.provider])


def model_summary(resolved: ResolvedModelConfig) -> ModelSummary:
    """Only allowlisted catalog fields enter the public wire."""
    c = resolved.capabilities

    def known_bool(key):
        value = c.get(key)
        return value if isinstance(value, bool) else None

    def known_strings(key):
        value = c.get(key)
        return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else None

    tools = known_bool("function_call_available")
    if tools is None and resolved.function_call_available:
        tools = True
    vision = known_bool("native_multimodal")
    if vision is None and resolved.native_multimodal:
        vision = True
    modalities = known_strings("input_modalities")
    if modalities is not None:
        vision = "image" in modalities
    if modalities is None and vision is not None:
        modalities = ["text", "image"] if vision else ["text"]
    return ModelSummary(
        id=resolved.model_id,
        provider=resolved.backend,
        display_name=resolved.selected_model,
        context_length=resolved.context_length,
        max_output_tokens=resolved.max_output_tokens,
        supports_tools=tools,
        supports_vision=vision,
        input_modalities=modalities,
        supports_reasoning=known_bool("reasoning_available"),
        supported_reasoning_efforts=known_strings("supported_reasoning_efforts"),
        default_reasoning_effort=c.get("default_reasoning_effort")
        if isinstance(c.get("default_reasoning_effort"), str)
        else None,
        supports_prompt_cache=known_bool("prompt_cache_available"),
        supports_continuation=known_bool("continuation_available"),
        metadata={"requestedModel": resolved.requested_model},
    )


class AppServerApprovalProvider(ApprovalProvider):
    def __init__(
        self,
        *,
        connection_id: str,
        thread_id: str,
        turn_id: str,
        router: OutgoingRouter,
        timeout_seconds: float | None,
        has_durable_answer: Callable[[ApprovalRequest], bool],
    ) -> None:
        self._connection_id = connection_id
        self._thread_id = thread_id
        self._turn_id = turn_id
        self._router = router
        self._timeout_seconds = timeout_seconds
        self._has_durable_answer = has_durable_answer

    def should_request(self, request: ApprovalRequest) -> bool:
        del request
        return True

    def decide(self, request: ApprovalRequest) -> RuntimeApprovalDecision | None:
        if not self._router.is_registered(self._connection_id):
            return None
        pending = self._router.send_server_request(
            self._connection_id,
            "approval/request",
            ApprovalRequestParams(
                request_id=request.request_id,
                thread_id=self._thread_id,
                turn_id=self._turn_id,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                preview=f"Approval required for tool {request.tool_name}.",
                arguments=dict(request.arguments),
            ).to_dict(),
            request_id=RequestId(request.request_id),
        )
        try:
            timeout = request.metadata.get("timeout_seconds", self._timeout_seconds)
            deadline = time.monotonic() + timeout if timeout is not None else None
            delay = 0.05
            while True:
                if self._has_durable_answer(request):
                    self._router.cancel_server_request(pending.request_id)
                    return None
                remaining = max(0, deadline - time.monotonic()) if deadline is not None else delay
                try:
                    result = pending.result(timeout=min(delay, remaining))
                    break
                except TimeoutError:
                    delay = min(1.0, delay * 2)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
        except TimeoutError:
            self._router.cancel_server_request(
                pending.request_id,
                AppServerError(AppServerErrorCode.INTERNAL_ERROR, "approval_timeout"),
            )
            return RuntimeApprovalDecision.timeout("Approval request timed out.")
        except RuntimeError:
            return None

        return self._decision_from_payload(result or {}, pending.request_id.require_wire())

    def _decision_from_payload(self, payload: dict[str, object], server_request_id: str | int) -> RuntimeApprovalDecision:
        raw_decision = str(payload.get("decision") or "deny")
        message = str(payload.get("reason") or payload.get("message") or "")
        raw_metadata = payload.get("metadata")
        metadata = {
            **(dict(raw_metadata) if isinstance(raw_metadata, dict) else {}),
            "server_request_id": server_request_id,
        }
        try:
            decision = WireApprovalDecision.from_wire(raw_decision)
        except ValueError:
            decision = WireApprovalDecision.DENY
        if decision is WireApprovalDecision.ALLOW:
            return RuntimeApprovalDecision(action=decision.value, reason=message, metadata=metadata)
        if decision is WireApprovalDecision.ALLOW_SESSION:
            return RuntimeApprovalDecision(action=decision.value, reason=message, metadata=metadata)
        if decision is WireApprovalDecision.TIMEOUT:
            return RuntimeApprovalDecision(
                action=decision.value,
                reason=message or "Approval request timed out.",
                metadata=metadata,
            )
        return RuntimeApprovalDecision(
            action=decision.value,
            reason=message or "Approval denied by client.",
            metadata=metadata,
        )
