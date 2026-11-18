from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.runtime import background_sessions as background_runtime
from vv_agent.runtime.processes import CapturedProcess
from vv_agent.session.environments import (
    EnvironmentCapabilities,
    InvalidLookupBinding,
    IsolationMode,
    LookupBinding,
    LookupStatus,
    NetworkMode,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.providers import Accepted, Definitive, Unknown

TENANT = "tenant-a"
SESSION = "session/a"
OPERATION = "operation/start"
ENVIRONMENT = "environment/a"
PROVIDER = "provider/local-test"
COMMAND_DIGEST = "3" * 64


class _FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class _FakeEnvironment:
    provider_id = PROVIDER

    def __init__(self, tmp_path: Path, process: _FakeProcess) -> None:
        self.capabilities = EnvironmentCapabilities.create(
            environment_id=ENVIRONMENT,
            generation=0,
            isolation=IsolationMode.NONE,
            network=NetworkMode.OPEN,
            enforces_network_policy=False,
            enforces_resource_limits=False,
            persistent=False,
            environment_capabilities=("command",),
            network_policy_ref_digest=None,
            isolation_policy_ref_digest=None,
        )
        self._tmp_path = tmp_path
        self._process = process
        self.starts = 0

    def start_process(self, command: list[str], **_: Any) -> CapturedProcess:
        del command
        self.starts += 1
        output_path = self._tmp_path / f"background-{self.starts}.log"
        output_path.write_text("terminal output\n", encoding="utf-8")
        return CapturedProcess(
            process=cast(subprocess.Popen[str], self._process),
            output_path=output_path,
            started_at=time.monotonic(),
        )


def _binding(
    *,
    attempt: int = 1,
    request: Any = 1,
    command_digest: str = COMMAND_DIGEST,
    provider_id: str = PROVIDER,
) -> LookupBinding:
    operation_key = stable_operation_key(SESSION, OPERATION)
    return LookupBinding(
        tenant_id=TENANT,
        session_id=SESSION,
        operation_id=OPERATION,
        attempt=attempt,
        request_digest=canonical_json_sha256(request),
        idempotency_key=operation_key,
        execution_id=execution_id_for(TENANT, ENVIRONMENT, operation_key),
        command_digest=command_digest,
        environment_id=ENVIRONMENT,
        generation=0,
        provider_id=provider_id,
    )


def _manager(tmp_path: Path, process: _FakeProcess, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeEnvironment]:
    environment = _FakeEnvironment(tmp_path, process)
    manager = background_runtime.BackgroundSessionManager(cast(Any, environment))
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    return manager, environment


def test_start_reuses_stable_execution_across_attempts_and_rejects_changed_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, environment = _manager(tmp_path, _FakeProcess(), monkeypatch)
    monkeypatch.setattr(background_runtime, "process_tree_is_running", lambda process: True)
    first_binding = _binding()

    session_id = manager.start(command="echo stable", cwd=tmp_path, binding=first_binding)
    retry_binding = _binding(attempt=2, request={"request": 2})
    duplicate_session_id = manager.start(command="echo stable", cwd=tmp_path, binding=retry_binding)

    assert duplicate_session_id == session_id
    assert environment.starts == 1
    state = manager._sessions[session_id]
    assert state.execution_id == first_binding.execution_id
    assert state.command_digest == first_binding.command_digest
    assert state.idempotency_key == first_binding.idempotency_key
    assert state.binding == first_binding

    running = manager.lookup(first_binding.execution_id, retry_binding)
    assert running.status is LookupStatus.RUNNING
    assert isinstance(running.outcome, Accepted)
    assert running.outcome.handle["attempt"] == retry_binding.attempt
    assert running.outcome.handle["request_digest"] == retry_binding.request_digest
    assert running.outcome.handle["session_id"] == session_id

    with pytest.raises(background_runtime.BackgroundExecutionConflict, match="different command"):
        manager.start(command="echo changed", cwd=tmp_path, binding=_binding(command_digest="4" * 64))
    assert environment.starts == 1


def test_lookup_rejects_binding_and_execution_mismatches_without_touching_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess()
    manager, _ = _manager(tmp_path, process, monkeypatch)
    monkeypatch.setattr(background_runtime, "process_tree_is_running", lambda process: True)
    binding = _binding()
    manager.start(command="echo stable", cwd=tmp_path, binding=binding)

    with pytest.raises(InvalidLookupBinding, match="execution_id"):
        manager.lookup("f" * 64, binding)
    with pytest.raises(InvalidLookupBinding, match="committed execution binding"):
        manager.lookup(binding.execution_id, _binding(command_digest="4" * 64))
    with pytest.raises(InvalidLookupBinding, match="manager provider"):
        manager.lookup(binding.execution_id, _binding(provider_id="provider/other"))

    assert process.poll() is None


def test_process_local_loss_returns_unknown_without_process_adoption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, environment = _manager(tmp_path, _FakeProcess(), monkeypatch)
    binding = _binding()
    session_id = manager.start(command="echo stable", cwd=tmp_path, binding=binding)
    manager._sessions.pop(session_id)

    result = manager.lookup(binding.execution_id, binding)

    assert result.status is LookupStatus.UNKNOWN
    assert isinstance(result.outcome, Unknown)
    assert "execution state is unavailable" in result.outcome.reason
    assert environment.starts == 1


def test_terminal_process_maps_to_definitive_execution_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()
    manager, _ = _manager(tmp_path, process, monkeypatch)
    binding = _binding()
    session_id = manager.start(command="echo stable", cwd=tmp_path, binding=binding)
    process.returncode = 0
    monkeypatch.setattr(background_runtime, "process_tree_is_running", lambda process: False)

    result = manager.lookup(binding.execution_id, binding)

    assert result.status is LookupStatus.EXITED
    assert isinstance(result.outcome, Definitive)
    assert result.outcome.result["session_id"] == session_id
    assert result.outcome.result["status"] == "completed"
    assert result.outcome.result["exit_code"] == 0
    assert "terminal output" in result.outcome.result["output"]
