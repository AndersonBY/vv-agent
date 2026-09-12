from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from support import FixedModelProvider, require_tool_result

from vv_agent import Agent, CheckpointConfig, RunConfig, Runner, ToolPolicy, function_tool
from vv_agent.checkpoint import OperationState
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.llm.base import LlmRequest
from vv_agent.runtime import background_sessions as background_runtime
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.tools import ToolContext, ToolRegistry, build_default_registry
from vv_agent.tools.handlers import background as background_handler
from vv_agent.tools.handlers import bash as bash_handler
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolExecutionResult, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend

CONTRACT = json.loads((Path(__file__).parent / "fixtures/parity/bash_process_management.json").read_text())


class ReceiptSqliteStore(SqliteCheckpointStore):
    """Observe committed real receipts before the normal cycle journal compaction."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.receipts: dict[str, Any] = {}

    def record_tool_receipt(self, checkpoint, **kwargs):
        recorded = super().record_tool_receipt(checkpoint, **kwargs)
        if recorded:
            stored = self.load_checkpoint(checkpoint.checkpoint_key)
            assert stored is not None
            for entry in stored.tool_journal:
                if entry.result is not None and entry.result_digest is not None:
                    assert entry.tool_call_id is not None
                    previous = self.receipts.get(entry.tool_call_id)
                    if previous is not None:
                        assert previous.result == entry.result
                        assert previous.result_digest == entry.result_digest
                    self.receipts[entry.tool_call_id] = deepcopy(entry)
        return recorded


@pytest.fixture
def manager(monkeypatch):
    manager = background_runtime.BackgroundSessionManager()
    monkeypatch.setattr(bash_handler, "background_session_manager", manager)
    monkeypatch.setattr(background_handler, "background_session_manager", manager)
    yield manager
    for session in list(manager._sessions.values()):
        manager.stop_for_tool(
            session.session_id,
            session.artifact_backend or LocalWorkspaceBackend(Path(session.owner_workspace)),
            session.owner_task_id,
            "cleanup",
            workspace=Path(session.owner_workspace),
        )


def context(workspace: Path, task_id: str = "owner") -> ToolContext:
    return ToolContext(
        workspace=workspace, workspace_backend=LocalWorkspaceBackend(workspace), task_id=task_id, shared_state={}, cycle_index=1
    )


def call(ctx: ToolContext, name: str, arguments: dict[str, Any], call_id: str = "call") -> ToolExecutionResult:
    ctx.tool_call_id = call_id
    return require_tool_result(build_default_registry().execute(ToolCall(id=call_id, name=name, arguments=arguments), ctx))


def runner_registry() -> ToolRegistry:
    """Expose the actual default executors through the public Runner registry hook."""
    native = build_default_registry()
    registry = ToolRegistry()
    for name in ("bash", "check_background_command", "stop_background_command"):
        registry.register_executor(native.get_executor(name))
    return registry


def python_command(workspace: Path, body: str) -> str:
    script = workspace / "child.py"
    script.write_text(body, encoding="utf-8")
    return f'"{sys.executable}" -u "{script.name}"'


def wait_until(predicate, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "local child did not reach expected state"
        time.sleep(0.01)


def process_exited(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}
    except FileNotFoundError:
        return True


def assert_running(result: ToolExecutionResult) -> str:
    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive.value == CONTRACT["running_receipt"]["directive"]
    assert result.metadata["status"] == "running"
    for name in CONTRACT["running_receipt"]["forbidden_metadata"]:
        assert name not in result.metadata
    body = json.loads(result.content)
    assert body["status"] == "running"
    assert body["session_id"] == result.metadata["session_id"]
    return body["session_id"]


INVALID = [("bash", case) for case in CONTRACT["invalid_bash_arguments"]] + [
    (name, case)
    for name in ("check_background_command", "stop_background_command")
    for case in CONTRACT["invalid_management_arguments"]
]


@pytest.mark.parametrize(("tool_name", "case"), INVALID, ids=[f"{tool}-{case['name']}" for tool, case in INVALID])
def test_contract_rejects_invalid_arguments_before_execution(tmp_path, manager, monkeypatch, tool_name, case):
    def never_spawn(*args, **kwargs):
        pytest.fail("invalid arguments reached process spawn")

    monkeypatch.setattr(bash_handler, "start_captured_process", never_spawn)
    result = call(context(tmp_path), tool_name, case["arguments"])
    assert result.status_code is ToolResultStatus.ERROR
    assert result.error_code == CONTRACT["invalid_arguments_error_code"]
    assert manager._sessions == {}


@pytest.mark.parametrize("launch_case", CONTRACT["launch_cases"], ids=lambda case: case["name"])
def test_real_runner_checkpoint_continues_after_start_query_and_stop(tmp_path, manager, launch_case):
    body = """from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        print('日志' + self.path, flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'LOCAL_HTTP_OK')
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', 0), Handler)
print('头' + 'x' * 15000 + '尾', flush=True)
Path('port').write_text(str(server.server_port))
server.serve_forever()
"""
    command = python_command(tmp_path, body)
    store = ReceiptSqliteStore(tmp_path / "checkpoint.sqlite")
    key = f"bash-{launch_case['name']}"
    seen: list[str] = []
    receipts: dict[str, Any] = {}
    session_ids: list[str] = []

    @function_tool(name="batch_marker")
    def batch_marker() -> str:
        seen.append("mixed-tool")
        return "MIXED_TOOL_OK"

    def receipt(call_id: str):
        entry = store.receipts[call_id]
        assert entry.state is not OperationState.AMBIGUOUS
        assert entry.result is not None and entry.result_digest is not None
        return entry

    def query_after_start(request: LlmRequest) -> LLMResponse:
        seen.append("model-after-start")
        tool_message = next(message for message in request.messages if message.tool_call_id == "start")
        assert "session_id" in json.loads(tool_message.content), tool_message.content
        session_id = json.loads(tool_message.content)["session_id"]
        session_ids.append(session_id)
        assert receipt("start").result["status_code"] == "SUCCESS"
        assert receipt("batch").result["status_code"] == "SUCCESS"
        receipts["start"] = deepcopy(receipt("start"))
        wait_until(lambda: (tmp_path / "port").exists())
        connection = http.client.HTTPConnection("127.0.0.1", int((tmp_path / "port").read_text()), timeout=1)
        try:
            connection.request("GET", "/before-query")
            assert connection.getresponse().read() == b"LOCAL_HTTP_OK"
        finally:
            connection.close()
        return LLMResponse(
            content="", tool_calls=[ToolCall(id="query", name="check_background_command", arguments={"session_id": session_id})]
        )

    def stop_after_query(request: LlmRequest) -> LLMResponse:
        seen.append("model-after-query")
        entry = receipt("query")
        assert entry.result["status_code"] == "SUCCESS"
        assert entry.result["metadata"]["status"] == "running"
        assert entry.result["truncated"] is True
        assert any("日志/before-query" in message.content for message in request.messages if message.tool_call_id == "query")
        receipts["query"] = deepcopy(entry)
        return LLMResponse(
            content="", tool_calls=[ToolCall(id="stop", name="stop_background_command", arguments={"session_id": session_ids[0]})]
        )

    def finish_after_stop(request: LlmRequest) -> LLMResponse:
        seen.append("model-after-stop")
        assert any(message.tool_call_id == "stop" for message in request.messages)
        stopped = receipt("stop").result
        assert stopped["metadata"]["status"] == "stopped"
        assert stopped["metadata"]["exit_code"] == manager._sessions[session_ids[0]].process.returncode
        assert stopped["status_code"] == "ERROR"  # The observed signal is not rewritten as success.
        return LLMResponse(content="NEXT_MODEL_CYCLE_OK")

    endpoint = EndpointConfig(endpoint_id="scripted", api_key="local-test", api_base="https://example.invalid")
    resolved = ResolvedModelConfig(
        backend="scripted",
        requested_model="local",
        selected_model="local",
        model_id="local",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="local")],
        function_call_available=True,
    )
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="start",
                        name="bash",
                        arguments={"command": command, "yield_time_ms": launch_case["yield_time_ms"], "timeout_seconds": 10},
                    ),
                    ToolCall(id="batch", name="batch_marker", arguments={}),
                ],
            ),
            query_after_start,
            stop_after_query,
            finish_after_stop,
        ]
    )
    result = Runner.run_sync(
        Agent(
            name="bash-runner",
            instructions="Use the local command tools.",
            model="local",
            tools=[batch_marker],
            tool_policy=ToolPolicy(allowed_tools=["bash", "check_background_command", "stop_background_command", "batch_marker"]),
        ),
        "Start, inspect and stop the local HTTP server.",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=FixedModelProvider(llm, resolved),
            max_cycles=5,
            no_tool_policy="finish",
            tool_registry_factory=runner_registry,
            checkpoint_config=CheckpointConfig(
                key=key,
                store=store,
                capability_refs={
                    "workspace": {"id": "test.local-workspace", "version": "1"},
                    "tool_registry_factory": {"id": "test.real-bash-registry", "version": "21"},
                },
            ),
        ),
    )
    assert result.status is AgentStatus.COMPLETED, result.raw_result
    assert result.final_output == "NEXT_MODEL_CYCLE_OK"
    assert seen == ["mixed-tool", "model-after-start", "model-after-query", "model-after-stop"]
    for call_id, original in receipts.items():
        assert receipt(call_id).result == original.result
        assert receipt(call_id).result_digest == original.result_digest
        stored = store.load_checkpoint(key)
        assert stored is not None
        retained = next(tool for cycle in stored.cycles for tool in cycle.tool_results if tool.tool_call_id == call_id)
        assert retained.to_dict() == original.result
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", int((tmp_path / "port").read_text())), timeout=0.2)
    store.close()


def test_execution_deadline_is_original_start_and_watchdog_needs_no_query(tmp_path, manager):
    ctx = context(tmp_path)
    command = python_command(tmp_path, "import time\nprint('ready', flush=True)\ntime.sleep(10)\n")
    result = call(ctx, "bash", {"command": command, "yield_time_ms": 800, "timeout_seconds": 1})
    session_id = assert_running(result)
    session = manager._sessions[session_id]
    original_start = session.started_at
    assert time.monotonic() - original_start >= 0.8
    call(ctx, "check_background_command", {"session_id": session_id})
    assert session.started_at == original_start
    assert session.done.wait(max(0, original_start + 1.6 - time.monotonic()))
    terminal = call(ctx, "check_background_command", {"session_id": session_id})
    assert terminal.status_code is ToolResultStatus.ERROR
    assert terminal.metadata["status"] == "timeout"
    assert terminal.metadata["exit_code"] == session.process.returncode
    assert "ready" in terminal.content


def test_initial_yield_rechecks_early_unsignaled_wait(tmp_path, manager, monkeypatch):
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    ctx = context(tmp_path)
    command = python_command(tmp_path, "import time\ntime.sleep(10)\n")
    session_id = assert_running(call(ctx, "bash", {"command": command, "yield_time_ms": 0}))
    session = manager._sessions[session_id]
    wait = session.done.wait
    attempts = []

    def interrupted(timeout=None):
        attempts.append(timeout)
        return False if len(attempts) == 1 else wait(timeout)

    monkeypatch.setattr(session.done, "wait", interrupted)
    manager.wait(session_id, 200)
    assert time.monotonic() - session.started_at >= 0.2
    assert session.process.poll() is None


def test_slow_live_artifact_does_not_block_execution_deadline(tmp_path, manager, monkeypatch):
    ctx = context(tmp_path)
    command = python_command(tmp_path, "import time\nprint('x' * 13000, flush=True)\ntime.sleep(10)\n")
    session_id = assert_running(call(ctx, "bash", {"command": command, "yield_time_ms": 0, "timeout_seconds": 1}))
    session = manager._sessions[session_id]
    wait_until(lambda: session.output_path.stat().st_size > 12000)
    entered, release = Event(), Event()
    persist = background_runtime.persist_captured_text_artifact
    observations = []

    def slow_persist(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5), "test did not release its own artifact writer"
        return persist(*args, **kwargs)

    monkeypatch.setattr(background_runtime, "persist_captured_text_artifact", slow_persist)
    query = Thread(target=lambda: observations.append(call(ctx, "check_background_command", {"session_id": session_id})))
    query.start()
    try:
        assert entered.wait(2)
        assert session.done.wait(2), "live artifact I/O blocked the process deadline"
        assert session.process.poll() is not None
        assert session.status == "timeout"
    finally:
        release.set()
        query.join(5)
    assert not query.is_alive()
    assert len(observations) == 1


@pytest.mark.parametrize("tool_name", ["check_background_command", "stop_background_command"])
@pytest.mark.parametrize("wrong_identity", ["task", "workspace"])
def test_foreign_owner_has_zero_process_output_artifact_or_stop_effects(
    tmp_path, manager, monkeypatch, tool_name, wrong_identity
):
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    ctx = context(tmp_path)
    command = python_command(tmp_path, "import time\ntime.sleep(10)\n")
    session_id = assert_running(call(ctx, "bash", {"command": command, "yield_time_ms": 0}))
    state = manager._sessions[session_id]
    foreign = context(tmp_path, "foreign")
    if wrong_identity == "workspace":
        other = tmp_path / "other"
        other.mkdir()
        foreign = context(other)
    calls: list[str] = []

    def unexpected(*args, **kwargs):
        calls.append("effect")
        raise AssertionError("foreign owner touched the process or output")

    with monkeypatch.context() as scoped:
        scoped.setattr(state.process, "poll", unexpected)
        scoped.setattr(background_runtime, "snapshot_captured_output", unexpected)
        scoped.setattr(background_runtime, "persist_captured_text_artifact", unexpected)
        scoped.setattr(background_runtime, "kill_process_tree", unexpected)
        result = call(foreign, tool_name, {"session_id": session_id})
    assert result.error_code == CONTRACT["forbidden_owner_error_code"]
    assert result.status_code is ToolResultStatus.ERROR
    assert calls == []
    assert state.owner_task_id == ctx.task_id
    assert state.artifact_task_id == ctx.task_id
    assert state.process.poll() is None


def test_missing_and_unconfirmed_observations_do_not_invent_exit_codes(tmp_path, manager, monkeypatch):
    ctx = context(tmp_path)
    for tool_name in ("check_background_command", "stop_background_command"):
        missing = call(ctx, tool_name, {"session_id": "lost-after-manager-restart"})
        assert missing.status_code is ToolResultStatus.ERROR
        assert missing.metadata["status"] == "missing"
        assert "exit_code" not in json.loads(missing.content)
        assert "exit_code" not in missing.metadata
    command = python_command(tmp_path, "import time\ntime.sleep(10)\n")
    session_id = assert_running(call(ctx, "bash", {"command": command, "yield_time_ms": 0}))
    with monkeypatch.context() as scoped:
        scoped.setattr(background_runtime, "kill_process_tree", lambda process: False)
        pending = call(ctx, "stop_background_command", {"session_id": session_id})
        assert pending.status_code is ToolResultStatus.SUCCESS
        assert pending.metadata["status"] == "stopping"
        assert "exit_code" not in pending.metadata
        assert "exit_code" not in json.loads(pending.content)
        assert "unconfirmed" in json.loads(pending.content)["message"]
    state = manager._sessions[session_id]
    with monkeypatch.context() as scoped:

        def unavailable():
            raise OSError("observation temporarily unavailable")

        scoped.setattr(state.process, "poll", unavailable)
        unknown = call(ctx, "check_background_command", {"session_id": session_id})
        assert unknown.status_code is ToolResultStatus.SUCCESS
        assert unknown.metadata["status"] == "unknown"
        assert "exit_code" not in json.loads(unknown.content)
        assert "unconfirmed" in json.loads(unknown.content)["message"]
    stopped = call(ctx, "stop_background_command", {"session_id": session_id})
    assert stopped.metadata["exit_code"] == state.process.returncode
    assert stopped.metadata["status"] == "stopped"


def test_unread_large_stdin_does_not_block_yield_or_deadline(tmp_path, manager):
    ctx = context(tmp_path)
    command = python_command(tmp_path, "import time\ntime.sleep(10)\n")
    started = time.monotonic()
    session_id = assert_running(
        call(ctx, "bash", {"command": command, "stdin": "x" * 2_000_000, "yield_time_ms": 0, "timeout_seconds": 1})
    )
    assert time.monotonic() - started < 0.75
    assert manager._sessions[session_id].done.wait(2)
    assert call(ctx, "check_background_command", {"session_id": session_id}).metadata["status"] == "timeout"


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("output_size", [4, 12001])
def test_zero_yield_returns_a_truthful_handle_even_if_process_already_exited(
    tmp_path, manager, monkeypatch, exit_code, output_size
):
    ctx = context(tmp_path)
    command = python_command(tmp_path, f"import sys\nsys.stdout.write('x' * {output_size})\nsys.exit({exit_code})\n")
    real_start = bash_handler.start_captured_process

    def already_exited(*args, **kwargs):
        captured = real_start(*args, **kwargs)
        captured.process.wait(timeout=2)
        return captured

    monkeypatch.setattr(bash_handler, "start_captured_process", already_exited)
    result = call(ctx, "bash", {"command": command, "yield_time_ms": 0})
    body = json.loads(result.content)
    assert body["session_id"] == result.metadata["session_id"]
    assert body["status"] == ("completed" if exit_code == 0 else "failed")
    assert body["exit_code"] == exit_code
    assert result.status_code is (ToolResultStatus.SUCCESS if exit_code == 0 else ToolResultStatus.ERROR)
    if output_size > 12000:
        assert result.artifact is not None
        assert ctx.workspace_backend.read_text(result.artifact.path) == "x" * output_size
        assert result.visible_bytes == len(result.content.encode())
        assert result.original_bytes is not None and result.visible_bytes is not None
        assert result.original_bytes >= result.visible_bytes


def test_running_unicode_output_has_live_tail_and_immutable_complete_artifacts(tmp_path, manager):
    ctx = context(tmp_path)
    command = python_command(
        tmp_path,
        """from pathlib import Path
import sys, time
print('头' + 'A' * 13000 + '尾', flush=True)
Path('ready').touch()
while not Path('more').exists(): time.sleep(.01)
print('追加😀尾部', flush=True)
Path('updated').touch()
while not Path('finish').exists(): time.sleep(.01)
""",
    )
    session_id = assert_running(call(ctx, "bash", {"command": command, "yield_time_ms": 0}))
    assert manager._sessions[session_id].timeout_seconds is None
    wait_until(lambda: (tmp_path / "ready").exists())
    first = call(ctx, "check_background_command", {"session_id": session_id}, "first")
    assert_running(first)
    assert first.truncated and first.artifact is not None
    full_first = ctx.workspace_backend.read_text(first.artifact.path)
    assert full_first == "头" + "A" * 13000 + "尾\n"
    assert first.visible_bytes == len(first.content.encode())
    assert first.original_bytes is not None and first.visible_bytes is not None
    assert first.original_bytes >= first.visible_bytes
    assert first.artifact.sha256 == hashlib.sha256(full_first.encode()).hexdigest()
    (tmp_path / "more").touch()
    wait_until(lambda: (tmp_path / "updated").exists())
    second = call(ctx, "check_background_command", {"session_id": session_id}, "second")
    assert_running(second)
    assert json.loads(second.content)["output"].endswith("追加😀尾部\n")
    assert second.artifact is not None and second.artifact != first.artifact
    assert ctx.workspace_backend.read_text(first.artifact.path) == full_first
    assert ctx.workspace_backend.read_text(second.artifact.path) == full_first + "追加😀尾部\n"
    (tmp_path / "finish").touch()
    assert manager._sessions[session_id].done.wait(2)
    terminal = call(ctx, "check_background_command", {"session_id": session_id})
    repeated = call(ctx, "check_background_command", {"session_id": session_id})
    assert terminal.artifact == repeated.artifact
    assert terminal.content == repeated.content


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group and signal semantics")
def test_stop_waits_for_child_that_outlives_its_shell_parent(tmp_path, manager):
    command = python_command(
        tmp_path,
        """import os, signal, time
from pathlib import Path
Path('parent-pid').write_text(str(os.getpid()))
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path('child-ready').write_text(str(os.getpid()))
    while True: time.sleep(.05)
while not Path('child-ready').exists(): time.sleep(.01)
os._exit(0)
""",
    )
    ctx = context(tmp_path)
    session_id = assert_running(call(ctx, "bash", {"command": command, "yield_time_ms": 20}))
    wait_until(lambda: (tmp_path / "child-ready").exists())
    wait_until(lambda: process_exited(int((tmp_path / "parent-pid").read_text())))
    active = call(ctx, "check_background_command", {"session_id": session_id})
    assert_running(active)
    stopped = call(ctx, "stop_background_command", {"session_id": session_id})
    assert stopped.metadata["status"] == "stopped"
    assert stopped.metadata["exit_code"] == 0
    pid = int((tmp_path / "child-ready").read_text())
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        assert proc_stat.read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux detached-child and procfs evidence")
@pytest.mark.parametrize(
    ("operation", "double_fork", "parent_signal"),
    [
        ("stop", False, False),
        ("stop", True, True),
        ("timeout", True, False),
        ("complete", True, False),
    ],
)
def test_detached_child_remains_managed_after_parent_exit(tmp_path, manager, operation, double_fork, parent_signal):
    command = "exec " + python_command(
        tmp_path,
        """import os, signal, time
from pathlib import Path
Path('parent-pid').write_text(str(os.getpid()))
if os.fork() == 0:
    os.setsid()
    if DOUBLE_FORK and os.fork() != 0: os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('DETACHED_READY' + 'x' * 13000, flush=True)
    Path('child-ready').write_text(str(os.getpid()))
    while not Path('finish').exists(): time.sleep(.01)
    print('DETACHED_DONE', flush=True)
    os._exit(0)
while not Path('child-ready').exists(): time.sleep(.01)
if PARENT_SIGNAL: os.kill(os.getpid(), signal.SIGTERM)
os._exit(7)
""".replace("DOUBLE_FORK", repr(double_fork)).replace("PARENT_SIGNAL", repr(parent_signal)),
    )
    ctx = context(tmp_path)
    arguments = {"command": command, "yield_time_ms": 0}
    if operation == "timeout":
        arguments["timeout_seconds"] = 1
    started = call(ctx, "bash", arguments)
    session_id = started.metadata["session_id"]
    wait_until(lambda: (tmp_path / "child-ready").exists())
    pid = int((tmp_path / "child-ready").read_text())
    try:
        parent = int((tmp_path / "parent-pid").read_text())

        wait_until(lambda: process_exited(parent))
        active = call(ctx, "check_background_command", {"session_id": session_id})
        print("DETACHED_OBSERVATION", active.metadata, "capture_exists", manager._sessions[session_id].output_path.exists())
        assert_running(active)
        assert "DETACHED_READY" in active.content
        assert active.artifact is not None
        if operation == "stop":
            stopped = call(ctx, "stop_background_command", {"session_id": session_id})
        else:
            if operation == "complete":
                (tmp_path / "finish").touch()
            assert manager._sessions[session_id].done.wait(2)
            stopped = call(ctx, "check_background_command", {"session_id": session_id})
        assert stopped.metadata["status"] == {"stop": "stopped", "timeout": "timeout", "complete": "failed"}[operation]
        assert stopped.metadata["exit_code"] == (-15 if parent_signal else 7)
        assert process_exited(pid)
        assert stopped.artifact is not None
        terminal_text = ctx.workspace_backend.read_text(stopped.artifact.path)
        assert "DETACHED_READY" in terminal_text
        if operation == "complete":
            assert "DETACHED_DONE" in terminal_text
        assert "DETACHED_DONE" not in ctx.workspace_backend.read_text(active.artifact.path)
    finally:
        # A private release file cleans up only this test child, even on failure.
        (tmp_path / "finish").touch()


@pytest.mark.parametrize("output_size", [17, 13001])
def test_terminal_capture_read_failure_is_retryable_without_duplicate_notification(tmp_path, manager, monkeypatch, output_size):
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    ctx = context(tmp_path)
    command = python_command(tmp_path, f"print('x' * {output_size}, end='')\n")
    started = call(ctx, "bash", {"command": command, "yield_time_ms": 0})
    session_id = started.metadata["session_id"]
    state = manager._sessions[session_id]
    notifications = []
    manager.subscribe(session_id, notifications.append)
    state.process.wait(timeout=2)
    saved = tmp_path / "saved-capture"
    state.output_path.rename(saved)
    failed = call(ctx, "check_background_command", {"session_id": session_id})
    assert failed.status_code is ToolResultStatus.ERROR
    assert failed.metadata["status"] == "completed"
    assert len(notifications) == 1
    assert saved.read_text() == "x" * output_size
    saved.rename(state.output_path)
    recovered = call(ctx, "check_background_command", {"session_id": session_id})
    assert recovered.status_code is ToolResultStatus.SUCCESS
    assert recovered.metadata["exit_code"] == 0
    assert state.output_error is None
    assert len(notifications) == 1
    if output_size > 12000:
        assert recovered.artifact is not None
        assert ctx.workspace_backend.read_text(recovered.artifact.path) == "x" * output_size
    else:
        assert recovered.content == "x" * output_size
    # A repeated receipt must not delete a later file at the released path.
    state.output_path.write_text("REPLACEMENT_CAPTURE")
    try:
        repeated = call(ctx, "check_background_command", {"session_id": session_id})
        assert repeated.content == recovered.content
        assert repeated.artifact == recovered.artifact
        assert state.output_path.read_text() == "REPLACEMENT_CAPTURE"
    finally:
        state.output_path.unlink(missing_ok=True)


def test_live_capture_read_failure_is_retryable(tmp_path, manager, monkeypatch):
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    ctx = context(tmp_path)
    command = python_command(tmp_path, "import time\nprint('LIVE_RECOVERED', flush=True)\ntime.sleep(10)\n")
    session_id = call(ctx, "bash", {"command": command, "yield_time_ms": 0}).metadata["session_id"]
    state = manager._sessions[session_id]
    wait_until(lambda: "LIVE_RECOVERED" in state.output_path.read_text())
    saved = tmp_path / "saved-capture"
    state.output_path.rename(saved)
    try:
        failed = call(ctx, "check_background_command", {"session_id": session_id})
        assert failed.status_code is ToolResultStatus.ERROR
        assert failed.metadata["status"] == "running"
    finally:
        saved.rename(state.output_path)
    recovered = call(ctx, "check_background_command", {"session_id": session_id})
    assert_running(recovered)
    assert "LIVE_RECOVERED" in recovered.content


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor evidence")
def test_lost_supervisor_does_not_claim_its_signal_as_command_exit(tmp_path, manager, monkeypatch):
    import signal

    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    ctx = context(tmp_path)
    command = python_command(
        tmp_path,
        """import os, time
from pathlib import Path
Path('root-pid').write_text(str(os.getpid()))
print('KEEP_CAPTURE', flush=True)
while not Path('finish').exists(): time.sleep(.01)
""",
    )
    session_id = call(ctx, "bash", {"command": command, "yield_time_ms": 0}).metadata["session_id"]
    state = manager._sessions[session_id]
    wait_until(lambda: "KEEP_CAPTURE" in state.output_path.read_text())
    root_pid = int((tmp_path / "root-pid").read_text())
    try:
        os.kill(state.process.pid, signal.SIGKILL)
        state.process.wait(timeout=2)
        result = call(ctx, "check_background_command", {"session_id": session_id})
        assert result.status_code is ToolResultStatus.SUCCESS
        assert result.metadata["status"] == "unknown"
        assert "exit_code" not in result.metadata
        assert "unconfirmed" in json.loads(result.content)["message"]
        assert state.output_path.exists()
        assert not process_exited(root_pid)
        assert not state.done.is_set()
    finally:
        (tmp_path / "finish").touch()
        wait_until(lambda: process_exited(root_pid))
        state.output_path.unlink(missing_ok=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper isolation")
def test_supervisor_does_not_adopt_unrelated_children_or_inherit_host_sockets(tmp_path):
    import ctypes
    import subprocess

    from vv_agent.runtime.processes import kill_process_tree, remove_captured_output, start_captured_process

    def subreaper_value():
        value = ctypes.c_int()
        assert ctypes.CDLL(None).prctl(37, ctypes.byref(value), 0, 0, 0) == 0
        return value.value

    before = subreaper_value()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(.1);raise SystemExit(23)"])
    started = start_captured_process([sys.executable, "-c", "import time;time.sleep(10)"], cwd=tmp_path)
    listener.close()
    try:
        assert subreaper_value() == before
        with socket.socket() as replacement:
            replacement.bind(("127.0.0.1", port))
        assert unrelated.wait(timeout=2) == 23
        assert started.process.poll() is None
        assert kill_process_tree(started.process)
        assert started.process.returncode == -15
    finally:
        kill_process_tree(started.process)
        remove_captured_output(started.output_path)
        unrelated.wait(timeout=2)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor descriptor isolation")
def test_supervisor_control_survives_closed_application_stdio(tmp_path):
    import subprocess

    script = tmp_path / "closed_stdio_probe.py"
    script.write_text(
        """import json, os, sys
from pathlib import Path
from vv_agent.runtime.processes import start_captured_process, kill_process_tree, remove_captured_output
for fd in (0, 1, 2): os.close(fd)
try:
    captured = start_captured_process([sys.executable, '-c', "print('CLOSED_STDIO_OK')"], cwd=Path.cwd())
    try:
        code = captured.process.wait(timeout=3)
        result = {'code': code, 'confirmed': kill_process_tree(captured.process), 'output': captured.output_path.read_text()}
    finally:
        kill_process_tree(captured.process)
        remove_captured_output(captured.output_path)
except BaseException as exc:
    result = {'error': repr(exc)}
Path('closed-stdio-result.json').write_text(json.dumps(result))
os._exit(0)
""",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True, timeout=5)
    assert json.loads((tmp_path / "closed-stdio-result.json").read_text()) == {
        "code": 0,
        "confirmed": True,
        "output": "CLOSED_STDIO_OK\n",
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor startup protocol")
@pytest.mark.parametrize("phase", ["before_start", "partial_ready"])
def test_supervisor_handshake_timeout_keeps_owned_handle_and_cleans_after_release(tmp_path, manager, monkeypatch, phase):
    from vv_agent.runtime import processes as process_runtime

    helper = Path(process_runtime.__file__).with_name("processes_supervisor.py")
    gate = tmp_path / "release-supervisor"
    wrapper = tmp_path / "delayed_supervisor.py"
    wrapper.write_text(
        "import runpy, socket, time\nfrom pathlib import Path\n"
        + f"gate = Path({str(gate)!r})\nphase = {phase!r}\n"
        + """if phase == 'before_start':
    while not gate.exists(): time.sleep(.01)
else:
    class PartialReadySocket(socket.socket):
        def sendall(self, data, *args, **kwargs):
            if data[:1] == b'R':
                super().sendall(data[:3], *args, **kwargs)
                while not gate.exists(): time.sleep(.01)
                data = data[3:]
            return super().sendall(data, *args, **kwargs)
    socket.socket = PartialReadySocket
"""
        + f"runpy.run_path({str(helper)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    real_popen = process_runtime.subprocess.Popen

    def delayed_popen(command, **kwargs):
        command = list(command)
        assert command[4] == str(helper)
        command[4] = str(wrapper)
        return real_popen(command, **kwargs)

    monkeypatch.setattr(process_runtime.subprocess, "Popen", delayed_popen)
    monkeypatch.setattr(process_runtime, "_SUPERVISOR_START_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(manager, "_start_watch_thread", lambda session_id: None)
    ctx = context(tmp_path)
    started_at = time.monotonic()
    command = python_command(tmp_path, "import time\nprint('STARTUP_CHILD', flush=True)\ntime.sleep(10)\n")
    try:
        result = call(ctx, "bash", {"command": command, "yield_time_ms": 0})
        assert time.monotonic() - started_at < 1
        assert result.status_code is ToolResultStatus.SUCCESS
        assert result.metadata["status"] == "unknown"
        assert "exit_code" not in result.metadata
        session_id = result.metadata["session_id"]
        state = manager._sessions[session_id]
        assert state.output_path.exists()
        denied = call(context(tmp_path, "another-owner"), "stop_background_command", {"session_id": session_id})
        assert denied.status_code is ToolResultStatus.ERROR
        pending = call(ctx, "check_background_command", {"session_id": session_id})
        assert pending.metadata["status"] == "unknown"
        assert "unconfirmed" in json.loads(pending.content)["message"]
    finally:
        gate.touch()
    terminal = None

    def completed():
        nonlocal terminal
        terminal = call(ctx, "check_background_command", {"session_id": session_id})
        return terminal.metadata["status"] not in {"running", "stopping", "unknown"}

    wait_until(completed)
    assert terminal is not None
    assert terminal.metadata["exit_code"] < 0
    assert state.done.is_set()
    assert not state.output_path.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor disconnected startup")
def test_supervisor_owner_disconnect_during_ready_still_reaps_detached_tree(tmp_path):
    import subprocess

    from vv_agent.runtime import processes as process_runtime

    helper = Path(process_runtime.__file__).with_name("processes_supervisor.py")
    gate = tmp_path / "send-ready"
    wrapper = tmp_path / "disconnect_supervisor.py"
    wrapper.write_text(
        "import runpy, socket, time\nfrom pathlib import Path\n"
        + f"gate = Path({str(gate)!r})\n"
        + """class DelayedReadySocket(socket.socket):
    def sendall(self, data, *args, **kwargs):
        if data[:1] == b'R':
            while not gate.exists(): time.sleep(.01)
        return super().sendall(data, *args, **kwargs)
socket.socket = DelayedReadySocket
"""
        + f"runpy.run_path({str(helper)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    child_script = tmp_path / "disconnect_child.py"
    child_script.write_text(
        """import os, signal, time
from pathlib import Path
if os.fork() == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path('detached-pid').write_text(str(os.getpid()))
    while not Path('child-release').exists(): time.sleep(.01)
    os._exit(0)
while not Path('detached-pid').exists(): time.sleep(.01)
os._exit(7)
""",
        encoding="utf-8",
    )
    control, child_control = socket.socketpair()
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", str(wrapper), str(child_control.fileno()), sys.executable, str(child_script)],
        cwd=tmp_path,
        pass_fds=(child_control.fileno(),),
        start_new_session=True,
    )
    child_control.close()
    try:
        wait_until(lambda: (tmp_path / "detached-pid").exists())
        pid = int((tmp_path / "detached-pid").read_text())
        control.close()
        gate.touch()
        assert process.wait(timeout=3) == 7
        assert process_exited(pid)
    finally:
        control.close()
        gate.touch()
        (tmp_path / "child-release").touch()
        process.wait(timeout=3)
