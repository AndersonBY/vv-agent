from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vv_agent.agent import Agent
from vv_agent.app_server.protocol import ModelListRequest, ModelListResponse, ModelSummary
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
