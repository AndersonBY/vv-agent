from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from vv_agent.agent import Agent
from vv_agent.events import (
    RunEvent,
)
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig, effective_run_config
from vv_agent.run_handle import RunHandle
from vv_agent.session.runtime import static_session_requirements, static_worker_supports
from vv_agent.types import (
    AgentStatus,
    AgentTask,
)


def _terminal(result: RunResult) -> bool:
    return result.status in {AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.MAX_CYCLES}


def _finish_run(handle: RunHandle) -> RunResult:
    try:
        result = handle.result()
        if _terminal(result):
            retain_owner = handle.kernel._wait_for_background_work(handle.session_id)
            if retain_owner:
                result = handle.result(timeout=0)
            else:
                handle.close()
                result._session_driver = None
        return result
    except BaseException:
        handle.close()
        raise


def _configure_owned_driver(
    driver,
    agent: Agent,
    config: RunConfig,
) -> SessionRequirements:
    """Publish only static worker declarations before constructing a Runtime."""

    handler_version = driver.default_handler_version
    extra_capabilities = driver.capabilities
    requirements = static_session_requirements(
        agent,
        config,
        resolved=None,
        handler_version=handler_version,
        extra_capabilities=extra_capabilities,
    )
    supports = static_worker_supports(
        agent,
        config,
        resolved=None,
        extra_capabilities=extra_capabilities,
    )
    driver._configure_static_worker(
        level=requirements.level,
        capabilities=supports,
        requirements=requirements,
    )
    return requirements


class _SyncEventStream(Iterator[RunEvent]):
    """Own a sync stream until it is exhausted or the caller drops it."""

    def __init__(self, handle: RunHandle) -> None:
        self._handle = handle
        self._events = handle.events()
        self._result: RunResult | None = None
        self._result_observed = False
        self._closed = False

    def __iter__(self) -> _SyncEventStream:
        return self

    def __next__(self) -> RunEvent:
        try:
            return next(self._events)
        except StopIteration:
            if self._result is None:
                self._result = _finish_run(self._handle)
            self._closed = True
            raise

    @property
    def result(self) -> RunResult:
        if self._result is None:
            raise RuntimeError("stream result is available after the iterator is exhausted")
        self._result_observed = True
        return self._result

    def _close_unobserved_result(self) -> None:
        result = self._result
        if result is None or self._result_observed:
            return
        owner = result._session_driver
        if owner is not None:
            owner.close()
            result._session_driver = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with suppress(StopIteration):
                self._events.close()
            if self._handle.done():
                self._result = _finish_run(self._handle)
            else:
                self._handle.kernel.close()
        # Exhaustion and caller disposal are different boundaries. An
        # unobserved waiting result still owns its driver, so an explicit
        # close after exhaustion must release that owner as well.
        self._close_unobserved_result()

    def __del__(self) -> None:
        with suppress(Exception):
            if not self._closed:
                self.close()
            self._close_unobserved_result()


class Runner:
    @classmethod
    def configured(cls, default_run_config: RunConfig | None = None) -> ConfiguredRunner:
        return ConfiguredRunner(default_run_config=default_run_config or RunConfig())

    @classmethod
    def run_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return _finish_run(cls.start(agent, input, run_config=run_config))

    @classmethod
    def stream_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        return _SyncEventStream(cls.start(agent, input, run_config=run_config))

    @classmethod
    def start(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        from vv_agent.session.surfaces import SessionDriver

        config = effective_run_config(agent, run_config or RunConfig())
        driver = SessionDriver()
        sid = uuid.uuid4().hex
        try:
            _configure_owned_driver(driver, agent, config)
            seed = {
                "messages": [m.to_dict() for m in config.initial_messages or []],
                "shared_state": config.shared_state or {},
            }
            driver.create(
                sid,
                str(config.workspace or Path.cwd()),
                {"seed": seed},
            )
            handle = driver.start(sid, agent, config, input, input_id="initial")
        except BaseException:
            driver.close()
            raise
        handle._owns_driver = True
        return handle

    @classmethod
    def _run_compiled_sync(cls, agent: Agent, input: str, *, task: AgentTask, run_config: RunConfig | None = None) -> RunResult:
        return _finish_run(cls._start_compiled(agent, input, task=task, run_config=run_config))

    @classmethod
    def _start_compiled(cls, agent: Agent, input: str, *, task: AgentTask, run_config: RunConfig | None = None) -> RunHandle:
        from vv_agent.session.surfaces import SessionDriver

        config = effective_run_config(agent, run_config or RunConfig())
        driver = SessionDriver()
        sid = uuid.uuid4().hex
        try:
            _configure_owned_driver(driver, agent, config)
            driver.create(
                sid,
                str(config.workspace or Path.cwd()),
            )
            handle = driver.start(sid, agent, config, input, task=task)
        except BaseException:
            driver.close()
            raise
        handle._owns_driver = True
        return handle

    @classmethod
    def resume(cls, session_id: str, turn_id: str) -> RunResult:
        from vv_agent.session.surfaces import _resume_turn_handle

        _, result = _resume_turn_handle(session_id, turn_id)
        return result


@dataclass(frozen=True, slots=True)
class ConfiguredRunner:
    default_run_config: RunConfig = field(default_factory=RunConfig)

    def run_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return _finish_run(self.start(agent, input, run_config=run_config))

    def stream_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        return _SyncEventStream(self.start(agent, input, run_config=run_config))

    def start(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        config = effective_run_config(agent, run_config or RunConfig(), runner_defaults=self.default_run_config)
        return Runner.start(agent, input, run_config=config)

    def resume(self, session_id: str, turn_id: str) -> RunResult:
        return Runner.resume(session_id, turn_id)
