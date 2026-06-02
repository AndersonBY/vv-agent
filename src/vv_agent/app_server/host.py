from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vv_agent.agent import Agent
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import (
    ApprovalRequestParams,
    AppServerError,
    AppServerErrorCode,
    ModelListRequest,
    ModelListResponse,
    ModelSummary,
)
from vv_agent.approval import ApprovalDecision, ApprovalProvider, ApprovalRequest
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

    def list_models(self, request: ModelListRequest) -> ModelListResponse:
        del request
        return ModelListResponse(models=list(self._models))


class AppServerApprovalProvider(ApprovalProvider):
    def __init__(
        self,
        *,
        connection_id: str,
        thread_id: str,
        turn_id: str,
        router: OutgoingRouter,
        timeout_seconds: float | None,
    ) -> None:
        self._connection_id = connection_id
        self._thread_id = thread_id
        self._turn_id = turn_id
        self._router = router
        self._timeout_seconds = timeout_seconds

    def should_request(self, request: ApprovalRequest) -> bool:
        del request
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
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
        )
        try:
            result = pending.result(timeout=self._timeout_seconds)
        except TimeoutError:
            self._router.cancel_server_request(
                pending.request_id,
                AppServerError(AppServerErrorCode.INTERNAL_ERROR, "approval_timeout"),
            )
            return ApprovalDecision.timeout("Approval request timed out.")
        except RuntimeError as exc:
            return ApprovalDecision.deny(str(exc) or "client_disconnected")

        return self._decision_from_payload(result or {}, pending.request_id.to_wire())

    def _decision_from_payload(self, payload: dict[str, object], server_request_id: str | int) -> ApprovalDecision:
        decision = str(payload.get("decision") or "deny").strip().lower()
        message = str(payload.get("message") or "")
        metadata = {"server_request_id": server_request_id}
        if decision in {"allow", "approve", "approved"}:
            return ApprovalDecision(action="allow", reason=message, metadata=metadata)
        if decision == "allow_session":
            return ApprovalDecision(action="allow_session", reason=message, metadata=metadata)
        if decision == "timeout":
            return ApprovalDecision.timeout(message or "Approval request timed out.")
        return ApprovalDecision.deny(message or "Approval denied by client.")
