from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.runtime import background_sessions as background_runtime
from vv_agent.runtime.local_environment import LocalEnvironment
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
        self.start_kwargs: dict[str, Any] = {}

    def start_process(self, command: list[str], **kwargs: Any) -> CapturedProcess:
        del command
        self.starts += 1
        self.start_kwargs = kwargs
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
    environment_id: str = ENVIRONMENT,
    session_id: str = SESSION,
    operation_id: str = OPERATION,
) -> LookupBinding:
    operation_key = stable_operation_key(session_id, operation_id)
    return LookupBinding(
        tenant_id=TENANT,
        session_id=session_id,
        operation_id=operation_id,
        attempt=attempt,
        request_digest=canonical_json_sha256(request),
        idempotency_key=operation_key,
        execution_id=execution_id_for(TENANT, environment_id, operation_key),
        command_digest=command_digest,
        environment_id=environment_id,
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
    assert environment.start_kwargs["expected_capabilities"] is environment.capabilities
    assert environment.start_kwargs["expected_environment_id"] == ENVIRONMENT
    assert environment.start_kwargs["expected_generation"] == 0
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
    with pytest.raises(background_runtime.BackgroundExecutionConflict, match="different command"):
        manager.start(command="echo stable", cwd=tmp_path, binding=_binding(environment_id="environment/b"))
    assert environment.starts == 1


def test_adopt_running_process_accepts_and_saves_frozen_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _manager(tmp_path, _FakeProcess(), monkeypatch)
    process = _FakeProcess()
    binding = _binding()
    output_path = tmp_path / "adopted.log"
    output_path.write_text("adopted output\n", encoding="utf-8")

    session_id = manager.adopt_running_process(
        command="echo adopted",
        cwd=tmp_path,
        timeout_seconds=5,
        process=cast(subprocess.Popen[str], process),
        output_path=output_path,
        binding=binding,
    )

    state = manager._sessions[session_id]
    assert state.execution_id == binding.execution_id
    assert state.command_digest == binding.command_digest
    assert state.idempotency_key == binding.idempotency_key
    assert state.binding == binding


def test_adopt_running_process_is_idempotent_only_for_the_registered_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = _manager(tmp_path, _FakeProcess(), monkeypatch)
    binding = _binding()
    process = _FakeProcess()
    first_output = tmp_path / "first.log"
    first_output.write_text("first\n", encoding="utf-8")
    session_id = manager.adopt_running_process(
        command="echo adopted",
        cwd=tmp_path,
        timeout_seconds=5,
        process=cast(subprocess.Popen[str], process),
        output_path=first_output,
        binding=binding,
    )

    assert (
        manager.adopt_running_process(
            command="echo adopted",
            cwd=tmp_path,
            timeout_seconds=5,
            process=cast(subprocess.Popen[str], process),
            output_path=first_output,
            binding=_binding(attempt=2, request={"request": 2}),
        )
        == session_id
    )

    rejected_process = _FakeProcess()
    rejected_output = tmp_path / "rejected.log"
    rejected_output.write_text("rejected\n", encoding="utf-8")
    killed: list[_FakeProcess] = []

    def fake_kill(candidate: _FakeProcess) -> bool:
        killed.append(candidate)
        candidate.returncode = -9
        return True

    monkeypatch.setattr(background_runtime, "kill_process_tree", fake_kill)

    with pytest.raises(background_runtime.BackgroundExecutionConflict, match="different process"):
        manager.adopt_running_process(
            command="echo adopted",
            cwd=tmp_path,
            timeout_seconds=5,
            process=cast(subprocess.Popen[str], rejected_process),
            output_path=rejected_output,
            binding=_binding(attempt=3, request={"request": 3}),
        )

    assert killed == [rejected_process]
    assert rejected_process.returncode == -9
    assert not rejected_output.exists()
    assert manager._identity_sessions == {(TENANT, binding.idempotency_key): session_id}
    assert manager._sessions == {session_id: manager._sessions[session_id]}
    assert manager._sessions[session_id].process is process

    with pytest.raises(background_runtime.BackgroundExecutionConflict, match="different command"):
        manager.adopt_running_process(
            command="echo changed",
            cwd=tmp_path,
            timeout_seconds=5,
            process=cast(subprocess.Popen[str], process),
            output_path=first_output,
            binding=_binding(command_digest="4" * 64),
        )
    assert process.returncode is None
    assert manager._identity_sessions == {(TENANT, binding.idempotency_key): session_id}


def test_adopt_running_process_reports_rejected_process_cleanup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _manager(tmp_path, _FakeProcess(), monkeypatch)
    binding = _binding()
    output_path = tmp_path / "first.log"
    output_path.write_text("first\n", encoding="utf-8")
    manager.adopt_running_process(
        command="echo adopted",
        cwd=tmp_path,
        timeout_seconds=5,
        process=cast(subprocess.Popen[str], _FakeProcess()),
        output_path=output_path,
        binding=binding,
    )
    rejected_process = _FakeProcess()
    rejected_output = tmp_path / "rejected.log"
    rejected_output.write_text("rejected\n", encoding="utf-8")
    monkeypatch.setattr(background_runtime, "kill_process_tree", lambda process: False)

    with pytest.raises(background_runtime.BackgroundExecutionConflict, match=r"cleanup failed.*not confirm"):
        manager.adopt_running_process(
            command="echo adopted",
            cwd=tmp_path,
            timeout_seconds=5,
            process=cast(subprocess.Popen[str], rejected_process),
            output_path=rejected_output,
            binding=_binding(attempt=2, request={"request": 2}),
        )

    assert rejected_process.returncode is None
    assert rejected_output.exists()
    assert len(manager._sessions) == 1


def test_watcher_start_failure_rolls_back_only_the_new_session_and_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, environment = _manager(tmp_path, _FakeProcess(), monkeypatch)
    existing_binding = _binding()
    existing_output = tmp_path / "existing.log"
    existing_output.write_text("existing\n", encoding="utf-8")
    existing_process = _FakeProcess()
    existing_session_id = manager.adopt_running_process(
        command="echo existing",
        cwd=tmp_path,
        timeout_seconds=5,
        process=cast(subprocess.Popen[str], existing_process),
        output_path=existing_output,
        binding=existing_binding,
    )

    rejected_process = _FakeProcess()
    environment._process = rejected_process
    rejected_binding = _binding(session_id="session/b", operation_id="operation/start")
    killed: list[_FakeProcess] = []

    def fake_kill(candidate: _FakeProcess) -> bool:
        killed.append(candidate)
        candidate.returncode = -9
        return True

    def fail_watcher(session_id: str) -> None:
        raise RuntimeError(f"cannot watch {session_id}")

    monkeypatch.setattr(background_runtime, "kill_process_tree", fake_kill)
    monkeypatch.setattr(manager, "_start_watch_thread", fail_watcher)

    with pytest.raises(RuntimeError, match="cannot watch"):
        manager.start(
            command="echo rejected",
            cwd=tmp_path,
            binding=rejected_binding,
        )

    assert killed == [rejected_process]
    assert environment.starts == 1
    assert not (tmp_path / "background-1.log").exists()
    assert manager._sessions == {existing_session_id: manager._sessions[existing_session_id]}
    assert manager._identity_sessions == {(TENANT, existing_binding.idempotency_key): existing_session_id}
    assert manager._sessions[existing_session_id].process is existing_process


def test_local_environment_has_stable_provider_identity_and_rejects_mismatch_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[list[str]] = []

    def fail_start(command: list[str], **_: Any) -> CapturedProcess:
        started.append(command)
        raise AssertionError("provider mismatch must be rejected before process spawn")

    environment = LocalEnvironment(process_starter=fail_start)
    manager = background_runtime.BackgroundSessionManager(environment)
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)

    assert environment.provider_id == PROVIDER
    with pytest.raises(InvalidLookupBinding, match="manager provider"):
        manager.start(
            command="echo rejected",
            cwd=tmp_path,
            binding=_binding(environment_id="local", provider_id="provider/other"),
        )

    assert started == []


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
    with pytest.raises(background_runtime.BackgroundExecutionConflict, match="lost process-local session state"):
        manager.start(command="echo stable", cwd=tmp_path, binding=binding)
    assert environment.starts == 1


def test_terminal_process_maps_to_definitive_execution_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()
    manager, environment = _manager(tmp_path, process, monkeypatch)
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

    duplicate_session_id = manager.start(command="echo stable", cwd=tmp_path, binding=binding)
    assert duplicate_session_id == session_id
    assert environment.starts == 1
