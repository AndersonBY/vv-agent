from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.prompt import PromptBundle, PromptSection
from vv_agent.runtime import local_environment as local_environment_runtime
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.local_environment import (
    LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY,
    LocalEnvironment,
    LocalEnvironmentBindingError,
    LocalEnvironmentPolicyError,
    expected_capabilities_for,
)
from vv_agent.session._dispatch_context import context as dispatch_context
from vv_agent.session.environments import EnvironmentCapabilities, IsolationMode, NetworkMode
from vv_agent.session.records import Record
from vv_agent.session.runtime import Runtime, TrustedEnvironmentBinding
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers import bash as bash_handler
from vv_agent.tools.handlers import search as search_handler
from vv_agent.tools.handlers import workspace_io
from vv_agent.types import AgentTask, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


def _isolated_capabilities() -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id="env-level-3",
        generation=9,
        isolation=IsolationMode.CONTAINER,
        network=NetworkMode.NONE,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=("command", "find_files", "search_files"),
        network_policy_ref_digest="1" * 64,
        isolation_policy_ref_digest="2" * 64,
    )


def _binding() -> TrustedEnvironmentBinding:
    return TrustedEnvironmentBinding(
        capabilities=_isolated_capabilities(),
        descriptor={"provider": "container"},
        operational_policy={
            "network_policy_digest": canonical_json_sha256({"network": "none"}),
            "isolation_policy_digest": canonical_json_sha256({"isolation": "container"}),
        },
    )


def _tool_context(
    tmp_path: Path,
    *,
    environment: TrustedEnvironmentBinding | None,
) -> ToolContext:
    task = AgentTask(
        task_id="level-3-tools",
        model="test",
        prompt_bundle=PromptBundle((PromptSection("instructions", "Use tools."),)),
        user_prompt="search",
        metadata={LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY: LocalEnvironment().capabilities},
    )
    plan = Record(
        "plan",
        "op_planned",
        "session",
        "turn",
        "tool",
        1,
        {
            "op_kind": "tool",
            "purpose": None,
            "request": {
                "id": "call",
                "name": "search_files",
                "arguments": {"pattern": "needle"},
                "tools": [{"function": {"name": "search_files"}}],
            },
            "dependencies": ["model/2"],
            "idempotency_key": "key",
        },
    )
    runtime = SimpleNamespace(
        agent=Agent("level-3-tools", "Use tools.", model="test"),
        config=RunConfig(workspace=tmp_path),
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        run_context=lambda task_id: SimpleNamespace(run_id=task_id, agent_name="level-3-tools"),
        turn_environment_binding=lambda: environment,
    )
    return dispatch_context(
        cast(Runtime, runtime),
        task,
        plan,
        CancellationToken(),
        approved=False,
        shared_state={"todo_list": []},
        store=None,
        session_id=None,
    )


def test_local_environment_declares_only_non_enforcing_host_capabilities() -> None:
    capabilities = LocalEnvironment().capabilities

    assert capabilities.isolation is IsolationMode.NONE
    assert capabilities.network is NetworkMode.OPEN
    assert capabilities.enforces_network_policy is False
    assert capabilities.enforces_resource_limits is False
    assert capabilities.persistent is False
    assert capabilities.capabilities_digest is not None


def test_local_environment_rejects_unsupported_policy_and_binding_requests() -> None:
    environment = LocalEnvironment(environment_id="local-test", generation=4)

    with pytest.raises(LocalEnvironmentPolicyError, match="does not enforce network policy"):
        environment.validate_binding(require_enforced_network_policy=True)
    with pytest.raises(LocalEnvironmentPolicyError, match="cannot enforce network policy 'none'"):
        environment.validate_binding(required_network_policy=NetworkMode.NONE)
    with pytest.raises(LocalEnvironmentBindingError, match="environment_id"):
        environment.validate_binding(expected_environment_id="other")
    with pytest.raises(LocalEnvironmentBindingError, match="generation"):
        environment.validate_binding(expected_generation=5)

    different_generation = EnvironmentCapabilities.create(
        environment_id="local-test",
        generation=5,
        isolation=IsolationMode.NONE,
        network=NetworkMode.OPEN,
        enforces_network_policy=False,
        enforces_resource_limits=False,
        persistent=False,
        environment_capabilities=environment.capabilities.environment_capabilities,
        network_policy_ref_digest=None,
        isolation_policy_ref_digest=None,
    )
    with pytest.raises(LocalEnvironmentBindingError, match="capabilities"):
        environment.validate_binding(expected_capabilities=different_generation)

    environment.validate_binding(
        expected_capabilities=environment.capabilities,
        expected_environment_id="local-test",
        expected_generation=4,
        required_network_policy=NetworkMode.OPEN,
    )


def test_local_environment_preserves_explicit_process_environment() -> None:
    captured: dict[str, Any] = {}

    def start(command: list[str], **kwargs: Any):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(process=object(), output_path=Path("capture.log"), started_at=1.0)

    explicit_env = {"CALLER_VALUE": "unchanged"}
    environment = LocalEnvironment(process_starter=start)

    environment.start_process(["echo", "ok"], cwd=Path.cwd(), stdin_text="input", env=explicit_env)

    assert captured["command"] == ["echo", "ok"]
    assert captured["stdin_text"] == "input"
    assert captured["env"] is explicit_env


def test_dispatch_context_uses_only_trusted_turn_environment_capabilities(tmp_path: Path) -> None:
    binding = _binding()
    context = _tool_context(tmp_path, environment=binding)

    assert expected_capabilities_for(context) is binding.capabilities
    assert context.task_metadata[LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY] is binding.capabilities
    assert context.ctx is not None
    assert context.ctx.metadata[LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY] is binding.capabilities


def test_dispatch_context_discards_self_reported_capabilities_without_binding(tmp_path: Path) -> None:
    context = _tool_context(tmp_path, environment=None)

    assert expected_capabilities_for(context) is None
    assert LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY not in context.task_metadata
    assert context.ctx is not None
    assert LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY not in context.ctx.metadata


def test_level3_environment_mismatch_fails_closed_without_starting_host_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")
    context = _tool_context(tmp_path, environment=_binding())
    started: list[list[str]] = []

    def fail_process_start(*args: Any, **kwargs: Any):
        del kwargs
        started.append(args[0])
        raise AssertionError("host process must not start")

    monkeypatch.setattr(bash_handler, "start_captured_process", fail_process_start)
    monkeypatch.setattr(local_environment_runtime.subprocess, "Popen", fail_process_start)
    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")

    bash_result = bash_handler.run_bash_command(context, {"command": "echo blocked"})
    search_result = search_handler.search_files(context, {"path": ".", "pattern": "needle"})
    find_result = workspace_io.find_files(context, {"path": ".", "glob": "**/*.txt"})

    assert started == []
    for result in (bash_result, search_result, find_result):
        assert result.status_code is ToolResultStatus.ERROR
        assert result.error_code == "environment_binding_mismatch"
        assert json.loads(result.content)["error_code"] == "environment_binding_mismatch"


def test_ordinary_local_environment_path_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")
    context = _tool_context(tmp_path, environment=None)
    commands: list[list[str]] = []

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        del kwargs
        commands.append(command)
        if "--json" in command:
            output = (
                "\n".join(
                    (
                        '{"type":"begin","data":{"path":{"text":"visible.txt"}}}',
                        '{"type":"match","data":{"path":{"text":"visible.txt"},'
                        '"lines":{"text":"needle\\n"},"line_number":1,'
                        '"submatches":[{"start":0,"end":6}]}}',
                        '{"type":"summary","data":{"stats":{"searches":1}}}',
                    )
                )
                + "\n"
            )
            return _FakeProcess(io.StringIO(output))
        return _FakeProcess(io.BytesIO(b"visible.txt\x00"))

    monkeypatch.setattr(local_environment_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")

    find_result = workspace_io.find_files(context, {"path": ".", "glob": "**/*.txt"})
    search_result = search_handler.search_files(context, {"path": ".", "pattern": "needle"})

    assert find_result.status_code is ToolResultStatus.SUCCESS
    assert json.loads(find_result.content)["files"] == ["visible.txt"]
    assert search_result.status_code is ToolResultStatus.SUCCESS
    assert search_result.metadata["summary"]["total_matches"] == 1
    assert len(commands) == 2
