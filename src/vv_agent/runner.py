from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from vv_agent.agent import Agent
from vv_agent.events import (
    RunEvent,
)
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig, effective_run_config
from vv_agent.run_handle import RunHandle
from vv_agent.types import (
    AgentTask,
)


class Runner:
    @classmethod
    def configured(cls, default_run_config: RunConfig | None = None) -> ConfiguredRunner:
        return ConfiguredRunner(default_run_config=default_run_config or RunConfig())

    @classmethod
    def run_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return cls.start(agent, input, run_config=run_config).result()

    @classmethod
    def stream_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        handle = cls.start(agent, input, run_config=run_config)
        yield from handle.events()
        handle.result()

    @classmethod
    def start(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        from vv_agent.session.surfaces import SessionDriver

        config = effective_run_config(agent, run_config or RunConfig())
        driver = SessionDriver()
        sid = uuid.uuid4().hex
        seed = {"messages": [m.to_dict() for m in config.initial_messages or []], "shared_state": config.shared_state or {}}
        driver.create(sid, str(config.workspace or Path.cwd()), {"seed": seed})
        return driver.start(sid, agent, config, input, input_id="initial")

    @classmethod
    def _run_compiled_sync(cls, agent: Agent, input: str, *, task: AgentTask, run_config: RunConfig | None = None) -> RunResult:
        return cls._start_compiled(agent, input, task=task, run_config=run_config).result()

    @classmethod
    def _start_compiled(cls, agent: Agent, input: str, *, task: AgentTask, run_config: RunConfig | None = None) -> RunHandle:
        from vv_agent.session.surfaces import SessionDriver

        config = effective_run_config(agent, run_config or RunConfig())
        driver = SessionDriver()
        sid = uuid.uuid4().hex
        driver.create(sid, str(config.workspace or Path.cwd()))
        return driver.start(sid, agent, config, input, task=task)

    @classmethod
    def resume(cls, session_id: str, turn_id: str) -> RunResult:
        from vv_agent.session.surfaces import resume_turn

        return resume_turn(session_id, turn_id)


@dataclass(frozen=True, slots=True)
class ConfiguredRunner:
    default_run_config: RunConfig = field(default_factory=RunConfig)

    def run_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return self.start(agent, input, run_config=run_config).result()

    def stream_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        handle = self.start(agent, input, run_config=run_config)
        yield from handle.events()
        handle.result()

    def start(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        config = effective_run_config(agent, run_config or RunConfig(), runner_defaults=self.default_run_config)
        return Runner.start(agent, input, run_config=config)

    def resume(self, session_id: str, turn_id: str) -> RunResult:
        return Runner.resume(session_id, turn_id)
