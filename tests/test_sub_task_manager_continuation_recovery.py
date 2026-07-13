from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.agent import RunContext
from vv_agent.approval import ApprovalBroker
from vv_agent.config import ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.events import SubRunCompletedEvent, SubRunStartedEvent
from vv_agent.interactive import AgentSessionRun, InteractiveAgentDefinition, create_agent_session
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import AgentRuntime, ExecutionContext, InMemoryStateStore, SubTaskManager, get_sub_agent_session
from vv_agent.runtime.engine import register_sub_agent_session, unregister_sub_agent_session
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    LLMResponse,
    Message,
    SubAgentConfig,
    SubTaskOutcome,
    SubTaskRequest,
    ToolCall,
)
from vv_agent.workspace import MemoryWorkspaceBackend

CONFIGURED_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent_v1.json"


def _configured_contract() -> dict[str, Any]:
    return json.loads(CONFIGURED_CONTRACT_PATH.read_text(encoding="utf-8"))


def _agent_result(*, messages: list[Message] | None = None) -> AgentResult:
    return AgentResult(
        status=AgentStatus.COMPLETED,
        messages=list(messages or []),
        cycles=[],
        final_answer="done",
        shared_state={"todo_list": []},
    )


class _Run:
    def __init__(self, result: AgentResult | None = None) -> None:
        self.result = result or _agent_result()


class _Session:
    def __init__(self, continuation: Callable[[str], AgentResult] | None = None) -> None:
        self._continuation = continuation or (lambda _prompt: _agent_result())
        self._messages: list[Message] = []
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self.continuation_calls = 0

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def replace_messages(self, messages: list[Message]) -> None:
        self._messages = list(messages)
        for listener in list(self._listeners):
            listener("session_messages_replaced", {"message_count": len(messages)})

    def subscribe(self, listener: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def continue_run(self, prompt: str) -> _Run:
        self.continuation_calls += 1
        for listener in list(self._listeners):
            listener("session_run_start", {"prompt": prompt})
        result = self._continuation(prompt)
        for listener in list(self._listeners):
            listener("session_run_end", {"status": result.status.value})
        return _Run(result)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(event, payload)


class _FailingSubscribeSession(_Session):
    def __init__(self) -> None:
        super().__init__()
        self.subscribe_calls = 0

    def subscribe(self, listener: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]:
        self.subscribe_calls += 1
        if self.subscribe_calls == 1:
            raise RuntimeError("subscribe failed")
        return super().subscribe(listener)


class _DeadThread:
    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        del timeout


class _RecordingApprovalBroker(ApprovalBroker):
    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0

    def reset_cancelled(self) -> None:
        self.reset_calls += 1
        super().reset_cancelled()


def _manager(
    *,
    register_session: Callable[[str, Any], None] | None = None,
    unregister_session: Callable[[str, Any | None], None] | None = None,
) -> SubTaskManager:
    return SubTaskManager(
        register_session=register_session or (lambda _session_id, _session: None),
        unregister_session=unregister_session or (lambda _session_id, _session=None: None),
    )


def _attach(
    manager: SubTaskManager,
    session: _Session,
    *,
    task_id: str = "retained-task",
    event_forwarder: Callable[[str, dict[str, Any]], None] | None = None,
) -> SubTaskOutcome:
    manager.attach_session(
        task_id=task_id,
        session_id=f"{task_id}-session",
        agent_name="researcher",
        task_title="original title",
        workspace_backend=MemoryWorkspaceBackend(),
        session=session,
        parent_run_id="parent-run-a",
        parent_tool_call_id="parent-call-a",
        event_forwarder=event_forwarder,
    )
    outcome = SubTaskOutcome(
        task_id=task_id,
        session_id=f"{task_id}-session",
        agent_name="researcher",
        status=AgentStatus.COMPLETED,
        final_answer="original result",
    )
    manager.record_outcome(task_id, outcome)
    return outcome


def _tool_context(
    *,
    execution_context: ExecutionContext | None = None,
    run_context: RunContext[Any] | None = None,
    tool_call_id: str = "continue-call",
    metadata: dict[str, Any] | None = None,
    task_metadata: dict[str, Any] | None = None,
) -> ToolContext:
    return ToolContext(
        workspace=Path.cwd(),
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
        ctx=execution_context,
        run_context=run_context,
        tool_call_id=tool_call_id,
        metadata=dict(metadata or {}),
        task_metadata=dict(task_metadata or {}),
    )


def test_manager_direct_task_ids_are_opaque_exact_keys() -> None:
    task_id = "  opaque task id  "
    manager = _manager()
    session = _Session()
    _attach(manager, session, task_id=task_id)

    assert _configured_contract()["manager"]["task_id_policy"] == "opaque_exact_key"
    assert manager.get(task_id) is not None
    assert manager.get(task_id.strip()) is None
    with pytest.raises(KeyError, match="Sub-task opaque task id not found"):
        manager.continue_task(task_id=task_id.strip(), prompt="continue")

    manager.continue_task(task_id=task_id, prompt="continue")
    completed = manager.wait(task_id, timeout=2)
    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.COMPLETED


def test_continuation_thread_start_failure_restores_admission_and_session_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    unresolved = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "unfinished", "name": "read_file", "arguments": {}}],
    )
    original_messages = [Message(role="user", content="keep me"), unresolved]
    session = _Session()
    session.replace_messages(original_messages)
    outcome = _attach(manager, session)
    record = manager.get("retained-task")
    assert record is not None
    previous_thread = _DeadThread()
    record.thread = previous_thread
    record.recent_activity = "original activity"
    record.updated_at = "original-updated-at"

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    context = _tool_context(
        execution_context=ExecutionContext(metadata={"_vv_agent_run_id": "parent-run-b"}),
        run_context=RunContext(run_id="parent-run-b"),
        tool_call_id="parent-call-b",
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        manager._continue_task_with_context(
            task_id="retained-task",
            prompt="new continuation title",
            context=context,
        )

    restored = manager.get("retained-task")
    assert restored is record
    assert restored.task_title == "original title"
    assert restored.outcome is outcome
    assert restored.active is False
    assert restored.parent_run_id == "parent-run-a"
    assert restored.parent_tool_call_id == "parent-call-a"
    assert restored.thread is previous_thread
    assert restored.recent_activity == "original activity"
    assert restored.updated_at == "original-updated-at"
    assert session.messages == original_messages
    assert not restored.is_running()


@pytest.mark.parametrize("reuse_existing", [False, True])
def test_submit_thread_start_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing: bool,
) -> None:
    manager = _manager()
    previous: SubTaskOutcome | None = None
    previous_backend: MemoryWorkspaceBackend | None = None
    previous_thread: _DeadThread | None = None
    previous_record: Any | None = None
    if reuse_existing:
        previous = SubTaskOutcome(
            task_id="submit-task",
            session_id="old-session",
            agent_name="old-agent",
            status=AgentStatus.COMPLETED,
            final_answer="old result",
        )
        manager.record_outcome("submit-task", previous)
        record = manager.get("submit-task")
        assert record is not None
        previous_record = record
        previous_backend = MemoryWorkspaceBackend()
        previous_thread = _DeadThread()
        record.task_title = "old title"
        record.workspace_backend = previous_backend
        record.parent_run_id = "old-parent-run"
        record.parent_tool_call_id = "old-parent-call"
        record.thread = previous_thread
        record.updated_at = "old-updated-at"

    monkeypatch.setattr(
        threading.Thread,
        "start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("submit start failed")),
    )

    with pytest.raises(RuntimeError, match="submit start failed"):
        manager.submit(
            task_id="submit-task",
            session_id="new-session",
            agent_name="new-agent",
            task_title="new title",
            workspace_backend=MemoryWorkspaceBackend(),
            runner=lambda: SubTaskOutcome(
                task_id="submit-task",
                agent_name="new-agent",
                status=AgentStatus.COMPLETED,
            ),
            parent_run_id="new-parent-run",
            parent_tool_call_id="new-parent-call",
        )

    record = manager.get("submit-task")
    if not reuse_existing:
        assert record is None
        return
    assert record is not None
    assert record is previous_record
    assert _configured_contract()["manager"]["spawn_failure_restores_previous_generation"] is True
    assert record.session_id == "old-session"
    assert record.agent_name == "old-agent"
    assert record.task_title == "old title"
    assert record.workspace_backend is previous_backend
    assert record.outcome is previous
    assert record.parent_run_id == "old-parent-run"
    assert record.parent_tool_call_id == "old-parent-call"
    assert record.thread is previous_thread
    assert record.updated_at == "old-updated-at"
    assert not record.active


def test_terminal_task_id_submit_reuse_starts_fresh_generation_and_ignores_old_session_events() -> None:
    manager_contract = _configured_contract()["manager"]
    manager = _manager()
    old_session = _Session()
    old_outcome = _attach(manager, old_session, task_id="reused-task")
    old_record = manager.get("reused-task")
    assert old_record is not None
    old_record.resolved = {"backend": "old-backend", "model_id": "old-model"}
    old_session.emit("cycle_llm_response", {"cycle": 4, "assistant_preview": "old activity"})

    started = threading.Event()
    release = threading.Event()

    def run_new_generation() -> SubTaskOutcome:
        started.set()
        release.wait(timeout=2)
        return SubTaskOutcome(
            task_id="reused-task",
            session_id="new-session",
            agent_name="new-agent",
            status=AgentStatus.COMPLETED,
            final_answer="new result",
        )

    new_backend = MemoryWorkspaceBackend()
    submitted = manager.submit(
        task_id="reused-task",
        session_id="new-session",
        agent_name="new-agent",
        task_title="new title",
        workspace_backend=new_backend,
        runner=run_new_generation,
        parent_run_id="new-parent-run",
        parent_tool_call_id="new-parent-call",
    )
    assert started.wait(timeout=1)
    current = manager.get("reused-task")

    try:
        assert manager_contract["terminal_task_id_reuse"] == "new_generation"
        assert current is submitted
        assert current is not old_record
        assert current.session_id == "new-session"
        assert current.agent_name == "new-agent"
        assert current.task_title == "new title"
        assert current.workspace_backend is new_backend
        assert current.session is None
        assert current.outcome is None
        assert current.resolved == {}
        assert current.current_cycle_index is None
        assert current.recent_activity is None
        assert current.latest_cycle is None
        assert current.latest_tool_call is None
        assert current.session_generation == 0
        assert current.parent_run_id == "new-parent-run"
        assert current.parent_tool_call_id == "new-parent-call"
        assert current.generation_token is not old_record.generation_token

        new_session = _Session()
        manager.attach_session(
            task_id="reused-task",
            session_id="new-session",
            agent_name="new-agent",
            task_title="new title",
            workspace_backend=new_backend,
            session=new_session,
        )
        old_session.emit("cycle_llm_response", {"cycle": 99, "assistant_preview": "stale activity"})
        assert current.current_cycle_index is None
        assert current.recent_activity is None
    finally:
        release.set()

    completed = manager.wait("reused-task", timeout=2)
    assert completed is current
    assert completed.outcome is not None
    assert completed.outcome.final_answer == "new result"
    assert old_record.outcome is old_outcome


@pytest.mark.parametrize("failing_hook", ["register", "unregister"])
def test_registry_hook_failure_records_failed_outcome_clears_active_and_allows_retry(failing_hook: str) -> None:
    failures_remaining = 1

    def register_session(_session_id: str, _session: Any) -> None:
        nonlocal failures_remaining
        if failing_hook == "register" and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("register failed")

    def unregister_session(_session_id: str, _session: Any | None = None) -> None:
        nonlocal failures_remaining
        if failing_hook == "unregister" and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("unregister failed")

    manager = _manager(register_session=register_session, unregister_session=unregister_session)
    session = _Session()
    _attach(manager, session)

    manager.continue_task(task_id="retained-task", prompt="first attempt")
    failed = manager.wait("retained-task", timeout=2)

    assert failed is not None and failed.outcome is not None
    assert failed.outcome.status == AgentStatus.FAILED
    assert failed.outcome.error_code == "sub_task_failed"
    assert failed.outcome.error is not None and f"{failing_hook} failed" in failed.outcome.error
    assert not failed.active
    assert not failed.is_running()

    manager.continue_task(task_id="retained-task", prompt="retry")
    retried = manager.wait("retained-task", timeout=2)

    assert retried is not None and retried.outcome is not None
    assert retried.outcome.status == AgentStatus.COMPLETED
    assert not retried.is_running()


def test_turn_snapshot_projects_current_execution_capabilities_only() -> None:
    old_provider = object()
    old_model_provider = object()
    old_broker = ApprovalBroker()
    old_state_store = InMemoryStateStore()
    current_provider = object()
    current_model_provider = object()
    current_broker = ApprovalBroker()
    current_state_store = InMemoryStateStore()
    memory_provider = object()
    app_state = object()
    current_run_context = RunContext(context=app_state, run_id="parent-run-b")
    old_context = ExecutionContext(
        state_store=old_state_store,
        metadata={
            "_vv_agent_approval_provider": old_provider,
            "_vv_agent_approval_broker": old_broker,
            "_vv_agent_approval_timeout_seconds": 99.0,
            "_vv_agent_model_provider": old_model_provider,
            "private_old_value": "must not leak",
        },
    )
    current_context = ExecutionContext(
        state_store=current_state_store,
        metadata={
            "_vv_agent_approval_provider": current_provider,
            "_vv_agent_approval_broker": current_broker,
            "_vv_agent_approval_timeout_seconds": 3.5,
            "_vv_agent_memory_providers": [memory_provider],
            "_vv_agent_model_provider": current_model_provider,
            "_vv_agent_trace_context": {"traceparent": "current"},
            "private_current_value": "must not leak",
        },
    )
    snapshot = SubTaskManager._capture_turn_snapshot(
        _tool_context(
            execution_context=current_context,
            run_context=current_run_context,
            task_metadata={
                "_vv_agent_allowed_tools": ["task_finish", "read_file"],
                "_vv_agent_disallowed_tools": ["bash"],
            },
        )
    )

    projected = AgentRuntime._context_from_turn_snapshot(base_ctx=old_context, snapshot=snapshot)

    assert projected.state_store is current_state_store
    assert projected.metadata["_vv_agent_approval_provider"] is current_provider
    assert projected.metadata["_vv_agent_approval_broker"] is current_broker
    assert projected.metadata["_vv_agent_approval_timeout_seconds"] == 3.5
    assert projected.metadata["_vv_agent_memory_providers"] == [memory_provider]
    assert projected.metadata["_vv_agent_model_provider"] is current_model_provider
    assert projected.metadata["_vv_agent_trace_context"] == {"traceparent": "current"}
    assert projected.metadata["_vv_agent_run_context"] is current_run_context
    assert projected.metadata["_vv_agent_allowed_tools"] == ["task_finish", "read_file"]
    assert projected.metadata["_vv_agent_disallowed_tools"] == ["bash"]
    assert "private_old_value" not in projected.metadata
    assert "private_current_value" not in projected.metadata


def test_agent_session_resets_only_its_owned_approval_broker(tmp_path: Path) -> None:
    injected = _RecordingApprovalBroker()

    def execute_run(**_kwargs: Any) -> AgentSessionRun:
        return AgentSessionRun(
            agent_name="researcher",
            result=_agent_result(),
            resolved=ResolvedModelConfig(
                backend="test",
                requested_model="test-model",
                selected_model="test-model",
                model_id="test-model",
                endpoint_options=[],
            ),
        )

    injected_session = create_agent_session(
        execute_run=execute_run,
        session_id="injected-broker-session",
        agent_name="researcher",
        definition=InteractiveAgentDefinition(description="Research", model="test-model"),
        workspace=tmp_path,
        approval_broker=injected,
    )
    injected_session.prompt("use injected broker", auto_follow_up=False)
    assert injected.reset_calls == 0

    owned_session = create_agent_session(
        execute_run=execute_run,
        session_id="owned-broker-session",
        agent_name="researcher",
        definition=InteractiveAgentDefinition(description="Research", model="test-model"),
        workspace=tmp_path,
    )
    owned = _RecordingApprovalBroker()
    owned_session._approval_broker = owned
    owned_session._owns_approval_broker = True
    owned_session.prompt("use owned broker", auto_follow_up=False)
    assert owned.reset_calls == 1


def _finish(message: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
    )


class _TurnRoutingLLM:
    def __init__(self) -> None:
        self.task_id = ""
        self.parent_calls = 0
        self.child_calls = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: Any = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, stream_callback, model_settings
        if request_metadata and request_metadata.get("is_sub_task") is True:
            self.child_calls += 1
            return _finish(f"child result {self.child_calls}", f"child-finish-{self.child_calls}")

        self.parent_calls += 1
        if self.parent_calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="delegate-a",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={"agent_id": "researcher", "task_description": "retain child"},
                    )
                ],
            )
        if self.parent_calls == 3:
            assert self.task_id
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="continue-b",
                        name="sub_task_status",
                        arguments={
                            "task_ids": [self.task_id],
                            "message": "continue in turn b",
                            "wait_for_response": True,
                        },
                    )
                ],
            )
        return _finish(f"parent result {self.parent_calls}", f"parent-finish-{self.parent_calls}")


def _parent_task(task_id: str) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        model="test-model",
        system_prompt="Parent prompt",
        user_prompt="Manage retained child",
        max_cycles=2,
        sub_agents={
            "researcher": SubAgentConfig(
                model="test-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=1,
            )
        },
    )


def test_retained_session_forwards_continuation_events_only_to_current_parent_turn(tmp_path: Path) -> None:
    llm = _TurnRoutingLLM()
    manager = _manager()
    turn_a_logs: list[tuple[str, dict[str, Any]]] = []
    turn_b_logs: list[tuple[str, dict[str, Any]]] = []
    runtime_a = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        log_handler=lambda event, payload: turn_a_logs.append((event, payload)),
    )
    first = runtime_a.run(
        _parent_task("parent-turn-a"),
        ctx=ExecutionContext(
            metadata={
                "_vv_agent_run_id": "parent-run-a",
                "_vv_agent_trace_id": "trace-a",
            }
        ),
        sub_task_manager=manager,
    )
    llm.task_id = str(first.cycles[0].tool_results[0].metadata["task_id"])
    assert manager.wait(llm.task_id, timeout=2) is not None
    turn_a_logs.clear()

    runtime_b = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        log_handler=lambda event, payload: turn_b_logs.append((event, payload)),
    )
    second = runtime_b.run(
        _parent_task("parent-turn-b"),
        ctx=ExecutionContext(
            metadata={
                "_vv_agent_run_id": "parent-run-b",
                "_vv_agent_trace_id": "trace-b",
            }
        ),
        sub_task_manager=manager,
    )

    assert second.status == AgentStatus.COMPLETED
    assert manager.wait(llm.task_id, timeout=2) is not None
    assert not [event for event, _payload in turn_a_logs if event.startswith("sub_agent_")]
    turn_b_sub_events = [event for event, _payload in turn_b_logs if event.startswith("sub_agent_")]
    assert "sub_agent_session_run_start" in turn_b_sub_events
    assert "sub_agent_session_run_end" in turn_b_sub_events


def test_listener_exception_isolated_and_agent_session_running_state_recovers(tmp_path: Path) -> None:
    execute_calls = 0

    def execute_run(**_kwargs: Any) -> AgentSessionRun:
        nonlocal execute_calls
        execute_calls += 1
        return AgentSessionRun(
            agent_name="researcher",
            result=_agent_result(),
            resolved=ResolvedModelConfig(
                backend="test",
                requested_model="test-model",
                selected_model="test-model",
                model_id="test-model",
                endpoint_options=[],
            ),
        )

    session = create_agent_session(
        execute_run=execute_run,
        session_id="listener-session",
        agent_name="researcher",
        definition=InteractiveAgentDefinition(description="Research", model="test-model"),
        workspace=tmp_path,
    )
    delivered: list[str] = []

    def broken_listener(_event: str, _payload: dict[str, Any]) -> None:
        raise RuntimeError("listener failed")

    session.subscribe(broken_listener)
    session.subscribe(lambda event, _payload: delivered.append(event))

    run = session.prompt("do work", auto_follow_up=False)

    assert run.result.status == AgentStatus.COMPLETED
    assert execute_calls == 1
    assert not session.running
    assert "session_run_start" in delivered
    assert "session_run_end" in delivered


def test_terminal_task_releases_parent_sink_and_plain_continuation_does_not_reuse_it() -> None:
    initial_events: list[str] = []
    session = _Session()
    manager = _manager()
    _attach(manager, session, event_forwarder=lambda event, _payload: initial_events.append(event))
    record = manager.get("retained-task")
    assert record is not None
    assert record.initial_event_forwarder is None
    assert record.turn_event_handler is None
    record.parent_run_id = "sidecar-parent-run"
    record.parent_tool_call_id = "sidecar-parent-call"

    manager.continue_task(task_id="retained-task", prompt="plain continuation")
    continued = manager.wait("retained-task", timeout=2)

    assert continued is not None and continued.outcome is not None
    assert continued.outcome.status == AgentStatus.COMPLETED
    assert initial_events == []
    assert continued.initial_event_forwarder is None
    assert continued.turn_event_handler is None
    assert continued.use_turn_event_handler is True
    assert continued.parent_run_id == "parent-run-a"
    assert continued.parent_tool_call_id == "parent-call-a"


def test_session_listeners_do_not_retain_manager() -> None:
    session = _Session()
    manager = _manager()
    manager.attach_session(
        task_id="weak-manager-task",
        session_id="weak-manager-session",
        agent_name="researcher",
        task_title="weak listener",
        workspace_backend=MemoryWorkspaceBackend(),
        session=session,
        event_forwarder=lambda _event, _payload: None,
    )
    manager_ref = weakref.ref(manager)

    del manager

    assert manager_ref() is None
    session.replace_messages([])


def test_listener_subscription_failure_retries_and_old_session_events_are_ignored() -> None:
    manager = _manager()
    failing = _FailingSubscribeSession()

    with pytest.raises(RuntimeError, match="subscribe failed"):
        manager.attach_session(
            task_id="listener-task",
            session_id="listener-session-a",
            agent_name="researcher",
            task_title="initial",
            workspace_backend=MemoryWorkspaceBackend(),
            session=failing,
        )

    manager.attach_session(
        task_id="listener-task",
        session_id="listener-session-a",
        agent_name="researcher",
        task_title="retry",
        workspace_backend=MemoryWorkspaceBackend(),
        session=failing,
    )
    assert failing.subscribe_calls >= 2
    failing.replace_messages([])

    replacement = _Session()
    manager.attach_session(
        task_id="listener-task",
        session_id="listener-session-b",
        agent_name="researcher",
        task_title="replacement",
        workspace_backend=MemoryWorkspaceBackend(),
        session=replacement,
    )
    failing.replace_messages([])
    record = manager.get("listener-task")
    assert record is not None
    replacement_generation = record.session_generation
    previous_updated_at = record.updated_at
    failing.replace_messages([Message(role="user", content="stale")])

    assert record.session is replacement
    assert record.session_generation == replacement_generation
    assert record.updated_at == previous_updated_at


def test_running_worker_hides_early_recorded_outcome_until_exit(tmp_path: Path) -> None:
    manager = _manager()
    recorded = threading.Event()
    release = threading.Event()

    def run() -> SubTaskOutcome:
        early = SubTaskOutcome(
            task_id="worker-task",
            session_id="worker-session",
            agent_name="researcher",
            status=AgentStatus.COMPLETED,
            final_answer="early terminal",
        )
        manager.record_outcome("worker-task", early)
        recorded.set()
        release.wait(timeout=2)
        return early

    manager.submit(
        task_id="worker-task",
        session_id="worker-session",
        agent_name="researcher",
        task_title="worker",
        workspace_backend=MemoryWorkspaceBackend(),
        runner=run,
    )
    assert recorded.wait(timeout=1)
    snapshot = manager.get("worker-task")
    assert snapshot is not None and snapshot.is_running()
    assert snapshot.outcome is None
    result = build_default_registry().get("sub_task_status").handler(
        ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=MemoryWorkspaceBackend(),
            sub_task_manager=manager,
        ),
        {"task_ids": ["worker-task"]},
    )
    payload = json.loads(result.content)
    assert payload["tasks"][0]["status"] == AgentStatus.RUNNING.value
    assert "final_answer" not in payload["tasks"][0]

    release.set()
    completed = manager.wait("worker-task", timeout=2)
    assert completed is not None and completed.outcome is not None
    assert completed.outcome.final_answer == "early terminal"


class _RecordingPersistenceSession:
    def __init__(
        self,
        *,
        session_id: str,
        items: list[Message],
        sequence: list[str],
        fail: bool,
    ) -> None:
        self.session_id = session_id
        self._items = list(items)
        self._sequence = sequence
        self._fail = fail

    def get_items(self, limit: int | None = None) -> list[Message]:
        items = list(self._items)
        if limit is None:
            return items
        return items[-max(limit, 0) :] if limit else []

    def add_items(self, items: list[Message]) -> None:
        self._sequence.append("persist")
        if self._fail:
            raise RuntimeError("continuation persistence failed")
        self._items.extend(items)

    def pop_item(self) -> Message | None:
        return self._items.pop() if self._items else None

    def clear(self) -> None:
        self._items.clear()

    def clear_session(self) -> None:
        self.clear()


def _retained_configured_child(
    tmp_path: Path,
    *,
    manager: SubTaskManager,
    events: list[Any],
    continuation_response: LLMResponse | None = None,
) -> str:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                _finish("initial child done", "initial-child-finish"),
                continuation_response or _finish("continued child done", "continued-child-finish"),
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
            )
        },
    )
    outcome = runtime._run_sub_task(
        parent_task=task,
        workspace_path=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
        parent_shared_state={},
        request=SubTaskRequest(
            agent_name="researcher",
            task_description="retain child",
            metadata={"parent_tool_call_id": "delegate"},
        ),
        sub_task_manager=manager,
        ctx=ExecutionContext(
            metadata={
                "_vv_agent_run_id": "parent-run-a",
                "_vv_agent_trace_id": "trace-a",
                "_vv_agent_emit_event": events.append,
            }
        ),
    )
    assert outcome.status == AgentStatus.COMPLETED
    return outcome.task_id


def test_sync_child_manager_stays_running_and_rejected_continuation_does_not_unregister(
    tmp_path: Path,
) -> None:
    child_started = threading.Event()
    release_child = threading.Event()
    unregister_calls: list[str] = []
    child_task_id: dict[str, str] = {}

    def register_session(session_id: str, session: Any) -> None:
        register_sub_agent_session(session_id, session)

    def unregister_session(session_id: str, session: Any | None = None) -> None:
        unregister_calls.append(session_id)
        unregister_sub_agent_session(session_id, session)

    def block_child(request: Any) -> LLMResponse:
        child_task_id["value"] = str(request.metadata["task_id"])
        child_started.set()
        if not release_child.wait(timeout=3):
            raise TimeoutError("sync child was not released")
        return _finish("sync child done", "sync-child-finish")

    manager = _manager(register_session=register_session, unregister_session=unregister_session)
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[block_child]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="sync-parent",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )
    result: dict[str, SubTaskOutcome] = {}

    def run_sync_child() -> None:
        result["outcome"] = runtime._run_sub_task(
            parent_task=task,
            workspace_path=tmp_path,
            workspace_backend=MemoryWorkspaceBackend(),
            parent_shared_state={},
            request=SubTaskRequest(agent_name="researcher", task_description="block"),
            sub_task_manager=manager,
        )

    worker = threading.Thread(target=run_sync_child)
    worker.start()
    try:
        assert child_started.wait(timeout=2)
        task_id = child_task_id["value"]
        record = manager.get(task_id)
        assert record is not None and record.is_running()
        active_session = get_sub_agent_session(session_id=task_id)
        assert active_session is record.session

        with pytest.raises(RuntimeError, match="already running"):
            manager.continue_task(task_id=task_id, prompt="competing continuation")

        assert get_sub_agent_session(session_id=task_id) is active_session
        assert unregister_calls == []
    finally:
        release_child.set()
        worker.join(timeout=3)

    assert not worker.is_alive()
    assert result["outcome"].status == AgentStatus.COMPLETED
    completed = manager.get(result["outcome"].task_id)
    assert completed is not None and not completed.is_running()
    assert get_sub_agent_session(session_id=result["outcome"].session_id or "") is None
    assert unregister_calls == []


@pytest.mark.parametrize("persistence_fails", [False, True], ids=["success", "failure"])
def test_continuation_completion_follows_persistence_and_failure_emits_only_failed_completion(
    tmp_path: Path,
    persistence_fails: bool,
) -> None:
    registered: dict[str, Any] = {}

    def register_session(session_id: str, session: Any) -> None:
        registered[session_id] = session

    def unregister_session(session_id: str, session: Any | None = None) -> None:
        if session is None or registered.get(session_id) is session:
            registered.pop(session_id, None)

    manager = _manager(register_session=register_session, unregister_session=unregister_session)
    initial_events: list[Any] = []
    task_id = _retained_configured_child(tmp_path, manager=manager, events=initial_events)
    record = manager.get(task_id)
    assert record is not None and record.session is not None
    retained_session = record.session
    retained_session_any = cast(Any, retained_session)
    backing_session = retained_session_any._session
    sequence: list[str] = []
    retained_session_any._session = _RecordingPersistenceSession(
        session_id=backing_session.session_id,
        items=backing_session.get_items(),
        sequence=sequence,
        fail=persistence_fails,
    )
    continuation_events: list[Any] = []

    def event_sink(event: Any) -> None:
        continuation_events.append(event)
        if isinstance(event, SubRunStartedEvent):
            sequence.append("started")
        elif isinstance(event, SubRunCompletedEvent):
            sequence.append(f"completed:{event.status}")

    manager._continue_task_with_context(
        task_id=task_id,
        prompt="continue after persistence",
        context=_tool_context(
            execution_context=ExecutionContext(
                metadata={
                    "_vv_agent_emit_event": event_sink,
                    "_vv_agent_trace_id": "trace-b",
                }
            ),
            run_context=RunContext(run_id="parent-run-b"),
            tool_call_id="continue-b",
        ),
    )
    continued = manager.wait(task_id, timeout=3)

    assert continued is not None and continued.outcome is not None
    lifecycle = [
        event
        for event in continuation_events
        if isinstance(event, SubRunStartedEvent | SubRunCompletedEvent)
    ]
    assert [event.type for event in lifecycle] == ["sub_run_started", "sub_run_completed"]
    completion = lifecycle[1]
    assert isinstance(completion, SubRunCompletedEvent)
    expected_status = AgentStatus.FAILED if persistence_fails else AgentStatus.COMPLETED
    assert continued.outcome.status == expected_status
    assert completion.status == expected_status.value
    assert sequence == ["started", "persist", f"completed:{expected_status.value}"]
    if persistence_fails:
        assert continued.outcome.error is not None
        assert "continuation persistence failed" in continued.outcome.error
        assert completion.error is not None
        assert "continuation persistence failed" in completion.error
    assert registered == {}


def test_continuation_started_sink_failure_emits_one_failed_completion_and_cleans_up(
    tmp_path: Path,
) -> None:
    registered: dict[str, Any] = {}
    unregister_calls: list[str] = []

    def register_session(session_id: str, session: Any) -> None:
        registered[session_id] = session

    def unregister_session(session_id: str, session: Any | None = None) -> None:
        unregister_calls.append(session_id)
        if session is None or registered.get(session_id) is session:
            registered.pop(session_id, None)

    manager = _manager(register_session=register_session, unregister_session=unregister_session)
    initial_events: list[Any] = []
    task_id = _retained_configured_child(tmp_path, manager=manager, events=initial_events)
    unregister_calls.clear()
    continuation_events: list[Any] = []

    def failing_started_sink(event: Any) -> None:
        continuation_events.append(event)
        if isinstance(event, SubRunStartedEvent):
            raise RuntimeError("started event sink failed")

    manager._continue_task_with_context(
        task_id=task_id,
        prompt="continue with broken started sink",
        context=_tool_context(
            execution_context=ExecutionContext(
                metadata={
                    "_vv_agent_emit_event": failing_started_sink,
                    "_vv_agent_trace_id": "trace-b",
                }
            ),
            run_context=RunContext(run_id="parent-run-b"),
            tool_call_id="continue-b",
        ),
    )
    failed = manager.wait(task_id, timeout=3)

    assert failed is not None and failed.outcome is not None
    assert failed.outcome.status == AgentStatus.FAILED
    assert failed.outcome.error is not None and "started event sink failed" in failed.outcome.error
    lifecycle = [
        event
        for event in continuation_events
        if isinstance(event, SubRunStartedEvent | SubRunCompletedEvent)
    ]
    assert [event.type for event in lifecycle] == ["sub_run_started", "sub_run_completed"]
    completion = lifecycle[1]
    assert isinstance(completion, SubRunCompletedEvent)
    assert completion.status == AgentStatus.FAILED.value
    assert completion.error is not None and "started event sink failed" in completion.error
    assert registered == {}
    assert unregister_calls == [failed.session_id]
    assert not failed.is_running()


def test_continuation_completion_sink_failure_does_not_skip_manager_cleanup(
    tmp_path: Path,
) -> None:
    registered: dict[str, Any] = {}

    def register_session(session_id: str, session: Any) -> None:
        registered[session_id] = session

    def unregister_session(session_id: str, session: Any | None = None) -> None:
        if session is None or registered.get(session_id) is session:
            registered.pop(session_id, None)

    manager = _manager(register_session=register_session, unregister_session=unregister_session)
    task_id = _retained_configured_child(tmp_path, manager=manager, events=[])
    continuation_events: list[Any] = []

    def failing_completion_sink(event: Any) -> None:
        continuation_events.append(event)
        if isinstance(event, SubRunCompletedEvent):
            raise RuntimeError("completion event sink failed")

    manager._continue_task_with_context(
        task_id=task_id,
        prompt="continue with broken completion sink",
        context=_tool_context(
            execution_context=ExecutionContext(
                metadata={
                    "_vv_agent_emit_event": failing_completion_sink,
                    "_vv_agent_trace_id": "trace-b",
                }
            ),
            run_context=RunContext(run_id="parent-run-b"),
            tool_call_id="continue-b",
        ),
    )
    completed = manager.wait(task_id, timeout=3)

    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.COMPLETED
    lifecycle = [
        event
        for event in continuation_events
        if isinstance(event, SubRunStartedEvent | SubRunCompletedEvent)
    ]
    assert [event.type for event in lifecycle] == ["sub_run_started", "sub_run_completed"]
    assert registered == {}
    assert not completed.is_running()
