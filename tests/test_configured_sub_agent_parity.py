from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.agent import Agent, RunContext
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalRequest
from vv_agent.config import ResolvedModelConfig
from vv_agent.constants import (
    CREATE_SUB_TASK_TOOL_NAME,
    FIND_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
)
from vv_agent.events import (
    AssistantDeltaEvent,
    ModelToolCallProgressEvent,
    ModelToolCallStartedEvent,
    ReasoningDeltaEvent,
    RunCompletedEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
)
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.model import ScriptedModelProvider
from vv_agent.model_settings import ModelSettings
from vv_agent.run_config import RunConfig, ToolPolicy
from vv_agent.runner import Runner
from vv_agent.runtime import (
    AgentRuntime,
    CancellationToken,
    ExecutionContext,
    InMemoryStateStore,
    SubTaskManager,
    get_sub_agent_session,
)
from vv_agent.runtime.backends.thread import ThreadBackend
from vv_agent.runtime.engine import register_sub_agent_session, unregister_sub_agent_session
from vv_agent.tools import ToolContext, build_default_registry, function_tool
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.tools.handlers.sub_task_status import sub_task_status
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
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)
from vv_agent.workspace import (
    INVALID_EXCLUDE_FILES_PATTERN_CODE,
    INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
    DiscoveryFilteredWorkspaceBackend,
    MemoryWorkspaceBackend,
    compile_portable_workspace_regex,
)

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent_v1.json"
EVENT_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent_events_v1.jsonl"
CONTRACT_SHA256 = "deb4f20c995c51f76f71590f91756ec6ab35f99128889bfa774cd2635d07106d"
EVENT_CONTRACT_SHA256 = "c2816a3962a44a3c0f5172edbffe4c88352142fee13f457da9a0667ceef996b0"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _event_contract() -> list[dict[str, Any]]:
    return [json.loads(line) for line in EVENT_CONTRACT_PATH.read_text(encoding="utf-8").splitlines() if line]


def _manager() -> SubTaskManager:
    return SubTaskManager(
        register_session=lambda _session_id, _session: None,
        unregister_session=lambda _session_id, _session=None: None,
    )


class _ManagerRun:
    def __init__(self, result: AgentResult) -> None:
        self.result = result


class _ManagerSession:
    def __init__(self, continuation: Callable[[str], AgentResult]) -> None:
        self._continuation = continuation
        self._messages: list[Message] = []
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def replace_messages(self, messages: list[Message]) -> None:
        self._messages = list(messages)

    def subscribe(self, listener: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def continue_run(self, prompt: str) -> _ManagerRun:
        return _ManagerRun(self._continuation(prompt))


class _SnapshotManagerSession(_ManagerSession):
    def __init__(self, snapshots: list[Any]) -> None:
        super().__init__(lambda _prompt: _manager_result())
        self._snapshots = snapshots

    def _continue_run_with_snapshot(self, prompt: str, snapshot: Any) -> _ManagerRun:
        assert prompt == "continue from approved parent turn"
        self._snapshots.append(snapshot)
        if snapshot.stream_callback is not None:
            snapshot.stream_callback({"event": "assistant_delta", "delta": "continued"})
        return _ManagerRun(_manager_result())


def _manager_result(
    *,
    status: AgentStatus = AgentStatus.COMPLETED,
    error: str | None = None,
) -> AgentResult:
    return AgentResult(
        status=status,
        messages=[],
        cycles=[],
        final_answer="done" if status == AgentStatus.COMPLETED else None,
        error=error,
        shared_state={"todo_list": []},
    )


def _attach_continuable_manager_task(
    manager: SubTaskManager,
    *,
    task_id: str,
    session: _ManagerSession,
    status: AgentStatus = AgentStatus.COMPLETED,
) -> None:
    manager.attach_session(
        task_id=task_id,
        session_id=f"{task_id}-session",
        agent_name="researcher",
        task_title="initial task",
        workspace_backend=MemoryWorkspaceBackend(),
        session=session,
    )
    manager.record_outcome(
        task_id,
        SubTaskOutcome(
            task_id=task_id,
            session_id=f"{task_id}-session",
            agent_name="researcher",
            status=status,
            final_answer="initial done" if status == AgentStatus.COMPLETED else None,
        ),
    )


def _finish(message: str, *, tool_call_id: str = "finish") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=tool_call_id, name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
    )


def _delegate(arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="delegate", name=CREATE_SUB_TASK_TOOL_NAME, arguments=arguments)],
    )


def _sub_events(events: list[Any]) -> list[SubRunStartedEvent | SubRunCompletedEvent]:
    return [event for event in events if isinstance(event, SubRunStartedEvent | SubRunCompletedEvent)]


def _lifecycle_is_fully_paired(events: list[Any]) -> bool:
    grouped: dict[str, list[str]] = {}
    for event in _sub_events(events):
        grouped.setdefault(event.run_id, []).append(event.type)
    return bool(grouped) and all(sequence == ["sub_run_started", "sub_run_completed"] for sequence in grouped.values())


def _assert_cancellation_contract(mode: str) -> dict[str, Any]:
    cancellation = _contract()["cancellation"]
    assert cancellation["modes"] == ["sync", "async", "batch", "continuation"]
    assert mode in cancellation["modes"]
    assert cancellation["direction"] == "parent_to_child"
    return cancellation


def _normalize_sub_event(event: SubRunStartedEvent | SubRunCompletedEvent) -> dict[str, Any]:
    payload = event.to_dict()
    failed_pair = event.parent_tool_call_id == "delegate-failed"
    task_id = "child-task-failed" if failed_pair else "child-task"
    session_id = "child-session-failed" if failed_pair else "child-session"
    payload["event_id"] = "evt_dynamic"
    payload["run_id"] = "run_dynamic"
    payload["session_id"] = session_id
    payload["child_session_id"] = session_id
    payload["task_id"] = task_id
    payload["created_at"] = 0.0
    return payload


def test_configured_sub_agent_shared_fixtures_have_expected_hashes() -> None:
    assert _contract()["version"] == "v1"
    assert all(event["version"] == _contract()["version"] for event in _event_contract())
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(EVENT_CONTRACT_PATH.read_bytes()).hexdigest() == EVENT_CONTRACT_SHA256


def test_portable_workspace_regex_cases_match_shared_contract() -> None:
    cases = _contract()["workspace_filter"]["portable_cases"]

    for case in cases["accepted"]:
        regex = compile_portable_workspace_regex(case["pattern"])
        assert all(regex.search(path) is not None for path in case["matches"])
        assert all(regex.search(path) is None for path in case["misses"])

    for pattern in cases["rejected"]:
        with pytest.raises(ValueError, match=INVALID_EXCLUDE_FILES_PATTERN_MESSAGE):
            compile_portable_workspace_regex(pattern)

    for positive_pattern, negative_pattern in ((r"^\w$", r"^\W$"), (r"^\s$", r"^\S$")):
        case = next(item for item in cases["accepted"] if item["pattern"] == positive_pattern)
        negative_regex = compile_portable_workspace_regex(negative_pattern)
        assert all(negative_regex.search(value) is None for value in case["matches"])
        assert all(negative_regex.search(value) is not None for value in case["misses"])


class _RawPathWorkspaceBackend(MemoryWorkspaceBackend):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self._raw_paths = list(paths)

    def list_files(self, base: str, glob: str) -> list[str]:
        del base, glob
        return list(self._raw_paths)


def test_workspace_filter_normalizes_for_matching_but_preserves_custom_backend_raw_paths() -> None:
    fixture = _contract()["workspace_filter"]["custom_backend_path_normalization"]
    backend = _RawPathWorkspaceBackend(fixture["raw_paths"])
    filtered = DiscoveryFilteredWorkspaceBackend(backend, fixture["pattern"])

    visible = filtered.list_files(".", "**/*")

    assert visible == fixture["visible_paths"]
    assert (visible[0] == fixture["raw_paths"][1]) is fixture["preserve_non_matching_raw_paths"]


def _parent_task() -> AgentTask:
    return AgentTask(
        task_id="parent-task",
        model="parent-model",
        system_prompt="Parent prompt",
        user_prompt="Parent task",
        max_cycles=6,
        memory_compact_threshold=64_000,
        memory_threshold_percentage=80,
        use_workspace=True,
        agent_type="computer",
        extra_tool_names=["custom_tool"],
        exclude_tools=["parent_excluded"],
        model_settings=ModelSettings(temperature=0.25, max_tokens=512),
        metadata={
            "language": "en-US",
            "available_skills": [{"name": "review"}],
            "active_skills": ["review"],
            "bash_shell": "bash",
        },
    )


def test_configured_sub_agent_task_projection_matches_shared_contract(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent = _parent_task()
    child = SubAgentConfig(
        model="child-model",
        description="Research facts",
        system_prompt="Child prompt",
        max_cycles=4,
        exclude_tools=["sub_excluded"],
        metadata={
            "sub_config_value": "sub",
            **{key: f"sub-agent-override-{key}" for key in _contract()["reserved_metadata_keys"]},
        },
    )
    parent.sub_agents["researcher"] = child
    request = SubTaskRequest(
        agent_name="researcher",
        task_description="Collect facts",
        output_requirements="Return JSON",
        metadata={
            "request_value": "request",
            **{key: f"request-override-{key}" for key in _contract()["reserved_metadata_keys"]},
        },
    )

    task = runtime._build_sub_agent_task(
        parent_task=parent,
        sub_task_id="child-task",
        sub_session_id="child-session",
        sub_agent_name="researcher",
        sub_agent=child,
        resolved_model_id="child-model",
        resolved_native_multimodal=True,
        child_run_id="child-run",
        trace_id="trace-parity",
        parent_run_id="parent-run",
        parent_tool_call_id="delegate",
        request=request,
        parent_shared_state={},
        workspace_path=tmp_path,
    )
    projection = {
        "model": task.model,
        "user_prompt": task.user_prompt,
        "max_cycles": task.max_cycles,
        "memory_compact_threshold": task.memory_compact_threshold,
        "memory_threshold_percentage": task.memory_threshold_percentage,
        "no_tool_policy": task.no_tool_policy,
        "allow_interruption": task.allow_interruption,
        "use_workspace": task.use_workspace,
        "has_sub_agents": task.has_sub_agents,
        "agent_type": task.agent_type,
        "native_multimodal": task.native_multimodal,
        "extra_tool_names": task.extra_tool_names,
        "exclude_tools": task.exclude_tools,
        "model_settings": task.model_settings.to_dict() if task.model_settings else None,
        "initial_messages": [message.to_dict() for message in task.initial_messages],
        "initial_shared_state": task.initial_shared_state,
    }
    metadata_projection = {
        key: ("<workspace>" if key == "workspace" else task.metadata[key]) for key in _contract()["metadata_projection"]
    }

    assert projection == _contract()["task_projection"]
    assert metadata_projection == _contract()["metadata_projection"]
    canonical_metadata = _contract()["metadata_projection"]
    for key in _contract()["reserved_metadata_keys"]:
        if key not in canonical_metadata:
            assert key not in task.metadata
            continue
        expected = str(tmp_path) if key == "workspace" else canonical_metadata[key]
        assert task.metadata[key] == expected


def test_agent_task_wire_round_trips_model_settings_messages_and_state() -> None:
    task = _parent_task()
    task.initial_messages = [Message(role="user", content="persisted")]
    task.initial_shared_state = {"scope": "child"}

    restored = AgentTask.from_dict(task.to_dict())

    assert restored.model_settings == task.model_settings
    assert restored.initial_messages == task.initial_messages
    assert restored.initial_shared_state == task.initial_shared_state


def test_sub_agent_model_normalization_and_outcome_wire_match_contract() -> None:
    validation = _contract()["validation"]
    config = SubAgentConfig(
        model=validation["normalized_model_input"],
        description="Research",
    )
    outcome = SubTaskOutcome(
        task_id="child-task",
        agent_name="researcher",
        status=AgentStatus.FAILED,
        error="failed",
    )

    assert config.model == validation["normalized_model_value"]
    assert "error_code" not in outcome.to_dict()


def test_sub_agent_config_uses_shared_portable_whitespace_contract() -> None:
    portable = _contract()["validation"]["portable_whitespace"]

    config = SubAgentConfig(model=portable["model_input"], description="Research")

    assert config.model == portable["model_value"]
    with pytest.raises(ValueError, match=_contract()["validation"]["empty_model_message"]):
        SubAgentConfig(model=portable["blank_model_input"], description="Research")
    with pytest.raises(ValueError, match=_contract()["validation"]["empty_system_prompt_message"]):
        SubAgentConfig(
            model="child-model",
            description="Research",
            system_prompt=portable["blank_system_prompt_input"],
        )

    mutated = SubAgentConfig(model="child-model", description="Research", system_prompt="Child prompt")
    mutated.system_prompt = portable["blank_system_prompt_input"]
    with pytest.raises(ValueError, match=_contract()["validation"]["empty_system_prompt_message"]):
        AgentRuntime._validate_sub_agent_config(mutated)


@pytest.mark.parametrize(
    ("field", "invalid_value", "error_code", "message"),
    [
        ("model", " ", "invalid_sub_agent_model", "sub-agent model cannot be empty"),
        (
            "system_prompt",
            " ",
            "invalid_sub_agent_system_prompt",
            "sub-agent system_prompt cannot be empty when provided",
        ),
    ],
)
def test_runtime_revalidates_mutated_sub_agent_config_and_pairs_events(
    tmp_path: Path,
    field: str,
    invalid_value: str,
    error_code: str,
    message: str,
) -> None:
    sub_agent = SubAgentConfig(model="shared-model", description="Research", system_prompt="Child prompt")
    setattr(sub_agent, field, invalid_value)
    events: list[Any] = []
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[_delegate({"agent_id": "researcher", "task_description": "Collect facts"})]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=1,
        sub_agents={"researcher": sub_agent},
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )
    manager = _manager()

    result = runtime.run(task, ctx=context, sub_task_manager=manager)

    tool_result = result.cycles[0].tool_results[0]
    payload = json.loads(tool_result.content)
    assert tool_result.error_code == error_code
    assert payload["error"] == message
    assert payload["error_code"] == error_code
    assert [event.type for event in _sub_events(events)] == _contract()["lifecycle"]["event_sequence"]
    assert _sub_events(events)[-1].status == AgentStatus.FAILED.value


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "", "description": "Research"}, "sub-agent model cannot be empty"),
        (
            {"model": "shared-model", "description": "Research", "system_prompt": ""},
            "sub-agent system_prompt cannot be empty when provided",
        ),
    ],
)
def test_sub_agent_config_constructor_validation_matches_contract(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as error:
        SubAgentConfig(**kwargs)

    assert str(error.value) == message


def test_sub_agent_config_from_wire_normalizes_model() -> None:
    fixture = _contract()["validation"]

    restored = SubAgentConfig.from_dict(
        {
            "model": fixture["normalized_model_input"],
        }
    )

    assert restored.model == fixture["normalized_model_value"]
    assert {
        "description": restored.description,
        "backend": restored.backend,
        "system_prompt": restored.system_prompt,
        "max_cycles": restored.max_cycles,
        "exclude_tools": restored.exclude_tools,
        "metadata": restored.metadata,
    } == fixture["wire_defaults"]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "  "},
        {"model": "child-model", "system_prompt": " \n "},
    ],
)
def test_sub_agent_config_from_wire_rejects_invalid_values(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SubAgentConfig.from_dict(payload)


@pytest.mark.parametrize(("field", "fixture_key"), [("backend", "backend_non_string"), ("max_cycles", "max_cycles_negative")])
def test_sub_agent_config_from_wire_rejects_shared_type_and_range_corpus(
    field: str,
    fixture_key: str,
) -> None:
    payload = {"model": "child-model", field: _contract()["validation"]["wire_rejections"][fixture_key]}

    with pytest.raises((TypeError, ValueError)):
        SubAgentConfig.from_dict(payload)


def test_configured_sub_agent_wire_ignores_unknown_top_level_fields() -> None:
    assert _contract()["validation"]["unknown_top_level_fields"] == "ignore"
    config = SubAgentConfig.from_dict({"model": "child-model", "backned": "ignored"})
    task = AgentTask.from_dict(
        {
            "task_id": "task",
            "model": "model",
            "system_prompt": "system",
            "user_prompt": "user",
            "runtime_metadata": {"trace_id": "legacy"},
        }
    )

    assert "backned" not in config.to_dict()
    assert "runtime_metadata" not in task.to_dict()


def test_runtime_boundary_uses_normalized_model_without_mutating_config(tmp_path: Path) -> None:
    provider = _ChildModelProvider(ScriptedLLM(steps=[]))
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    sub_agent = SubAgentConfig(model="child-model", description="Research")
    sub_agent.model = "  child-model  "
    context = ExecutionContext(metadata={"_vv_agent_model_provider": provider})

    runtime._validate_sub_agent_config(sub_agent)
    runtime._resolve_sub_agent_client(
        parent_task=_parent_task(),
        sub_agent=sub_agent,
        ctx=context,
    )

    assert provider.resolved_models == ["child-model"]
    assert sub_agent.model == "  child-model  "


def test_unknown_configured_sub_agent_uses_shared_failure_code(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )

    outcome = runtime._run_sub_task(
        parent_task=_parent_task(),
        workspace_path=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
        parent_shared_state={},
        request=SubTaskRequest(agent_name="missing", task_description="Collect facts"),
    )

    assert outcome.status == AgentStatus.FAILED
    assert outcome.error_code == _contract()["lifecycle"]["failure_error_code_fallback"]


def test_request_metadata_cannot_assign_framework_owned_child_identity(tmp_path: Path) -> None:
    identity = _contract()["identity"]
    events: list[Any] = []
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )
    outcome = AgentRuntime(
        llm_client=ScriptedLLM(steps=[_finish("done")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )._run_sub_task(
        parent_task=task,
        workspace_path=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
        parent_shared_state={},
        request=SubTaskRequest(
            agent_name="researcher",
            task_description="Collect facts",
            metadata={
                "task_id": "spoof-task",
                "session_id": "spoof-session",
                "run_id": "spoof-run",
                "parent_tool_call_id": "delegate",
            },
        ),
        sub_task_manager=_manager(),
        ctx=ExecutionContext(metadata={"_vv_agent_emit_event": events.append}),
    )

    pair = _sub_events(events)
    assert identity["request_metadata_controls_identity"] is False
    assert identity["run_id_generated_by_runtime"] is True
    assert outcome.task_id != "spoof-task"
    assert outcome.session_id != "spoof-session"
    assert all(event.task_id == outcome.task_id for event in pair)
    assert all(event.session_id == outcome.session_id for event in pair)
    assert all(event.run_id != "spoof-run" for event in pair)


def test_runtime_lineage_prefers_public_then_execution_then_request_without_task_fallback(
    tmp_path: Path,
) -> None:
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )

    def run_with_context(context: ExecutionContext, request_parent_run_id: str | None) -> tuple[Any, Any]:
        events: list[Any] = []
        context.metadata["_vv_agent_emit_event"] = events.append
        request_metadata = {"parent_tool_call_id": "delegate"}
        if request_parent_run_id is not None:
            request_metadata["parent_run_id"] = request_parent_run_id
        manager = _manager()
        outcome = AgentRuntime(
            llm_client=ScriptedLLM(steps=[_finish("done")]),
            tool_registry=build_default_registry(),
            default_workspace=tmp_path,
        )._run_sub_task(
            parent_task=task,
            workspace_path=tmp_path,
            workspace_backend=MemoryWorkspaceBackend(),
            parent_shared_state={},
            request=SubTaskRequest(
                agent_name="researcher",
                task_description="Collect facts",
                metadata=request_metadata,
            ),
            sub_task_manager=manager,
            ctx=context,
        )
        return _sub_events(events)[0], manager.get(outcome.task_id)

    fixture = _contract()["manager"]
    cases = [
        (
            ExecutionContext(
                metadata={
                    "_vv_agent_run_context": RunContext(run_id="public-run"),
                    "_vv_agent_run_id": "execution-run",
                }
            ),
            "request-run",
            "public-run",
        ),
        (ExecutionContext(metadata={"_vv_agent_run_id": "execution-run"}), "request-run", "execution-run"),
        (ExecutionContext(metadata={}), "request-run", "request-run"),
        (ExecutionContext(metadata={}), None, None),
    ]

    assert fixture["parent_lineage_precedence"] == [
        "run_context",
        "execution_context",
        "request_metadata",
    ]
    for context, request_parent_run_id, expected_parent_run_id in cases:
        event, record = run_with_context(context, request_parent_run_id)
        assert event.parent_run_id == expected_parent_run_id
        assert record is not None and record.parent_run_id == expected_parent_run_id
        if expected_parent_run_id is None:
            assert (record.parent_run_id == record.task_id) is fixture["fabricates_parent_run_id_from_task_id"]
            assert record.task_id.startswith("parent-task_sub_")


@pytest.mark.parametrize(
    ("source_label", "execution_trace_id", "public_trace_id", "task_trace_id", "expected_trace_id"),
    [
        ("execution_context", "execution-trace", "public-trace", "task-trace", "execution-trace"),
        ("run_context", None, "public-trace", "task-trace", "public-trace"),
        ("task_metadata", None, None, "task-trace", "task-trace"),
        ("child_run_id", None, None, None, None),
    ],
)
def test_runtime_trace_identity_precedence_and_child_run_fallback(
    tmp_path: Path,
    source_label: str,
    execution_trace_id: str | None,
    public_trace_id: str | None,
    task_trace_id: str | None,
    expected_trace_id: str | None,
) -> None:
    events: list[Any] = []
    task_metadata = {"trace_id": task_trace_id} if task_trace_id is not None else {}
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        metadata=task_metadata,
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )
    public_metadata = {"trace_id": public_trace_id} if public_trace_id is not None else {}
    context_metadata: dict[str, Any] = {
        "_vv_agent_emit_event": events.append,
        "_vv_agent_run_context": RunContext(run_id="parent-run", metadata=public_metadata),
    }
    if execution_trace_id is not None:
        context_metadata["_vv_agent_trace_id"] = execution_trace_id

    outcome = AgentRuntime(
        llm_client=ScriptedLLM(steps=[_finish("done")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )._run_sub_task(
        parent_task=task,
        workspace_path=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
        parent_shared_state={},
        request=SubTaskRequest(
            agent_name="researcher",
            task_description="Collect facts",
            metadata={"parent_tool_call_id": "delegate"},
        ),
        sub_task_manager=_manager(),
        ctx=ExecutionContext(metadata=context_metadata),
    )

    assert outcome.status == AgentStatus.COMPLETED
    pair = _sub_events(events)
    identity_contract = _contract()["identity"]
    source_index = ["execution_context", "run_context", "task_metadata", "child_run_id"].index(source_label)
    assert source_label == identity_contract["trace_precedence"][source_index]
    if source_label == identity_contract["trace_fallback"]:
        expected_trace_id = pair[0].run_id
    assert [event.trace_id for event in pair] == [expected_trace_id, expected_trace_id]


def test_public_run_context_private_trace_id_precedes_public_trace_id(tmp_path: Path) -> None:
    events: list[Any] = []
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_emit_event": events.append,
            "_vv_agent_run_context": RunContext(
                run_id="parent-run",
                metadata={
                    "_vv_agent_trace_id": "public-private-trace",
                    "trace_id": "public-trace",
                },
            ),
        }
    )

    outcome = AgentRuntime(
        llm_client=ScriptedLLM(steps=[_finish("done")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )._run_sub_task(
        parent_task=task,
        workspace_path=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
        parent_shared_state={},
        request=SubTaskRequest(agent_name="researcher", task_description="Collect facts"),
        sub_task_manager=_manager(),
        ctx=context,
    )

    assert outcome.status == AgentStatus.COMPLETED
    assert [event.trace_id for event in _sub_events(events)] == [
        "public-private-trace",
        "public-private-trace",
    ]


@pytest.mark.parametrize("invalid_value", _contract()["identity"]["non_string_metadata_values"])
def test_create_sub_task_handler_ignores_non_string_parent_lineage_sources(
    tmp_path: Path,
    invalid_value: Any,
) -> None:
    captured_requests: list[SubTaskRequest] = []

    def capture_request(request: SubTaskRequest) -> SubTaskOutcome:
        captured_requests.append(request)
        return SubTaskOutcome(
            task_id=f"child-{len(captured_requests)}",
            session_id=f"child-session-{len(captured_requests)}",
            agent_name=request.agent_name,
            status=AgentStatus.COMPLETED,
            final_answer="done",
        )

    registry = build_default_registry()

    def build_context(
        *,
        execution_run_id: Any,
        public_run_context: RunContext[Any] | None,
        tool_call_id: str,
    ) -> ToolContext:
        return ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=MemoryWorkspaceBackend(),
            sub_task_runner=capture_request,
            ctx=ExecutionContext(metadata={"_vv_agent_run_id": execution_run_id}),
            run_context=public_run_context,
            tool_call_id=tool_call_id,
        )

    fallback_context = build_context(
        execution_run_id="execution-run",
        public_run_context=RunContext(run_id=invalid_value),
        tool_call_id="fallback-call",
    )
    missing_context = build_context(
        execution_run_id=invalid_value,
        public_run_context=None,
        tool_call_id="missing-call",
    )

    fallback_result = registry.execute(
        ToolCall(
            id="fallback-call",
            name=CREATE_SUB_TASK_TOOL_NAME,
            arguments={"agent_id": "researcher", "task_description": "Use execution lineage"},
        ),
        fallback_context,
    )
    missing_result = registry.execute(
        ToolCall(
            id="missing-call",
            name=CREATE_SUB_TASK_TOOL_NAME,
            arguments={"agent_id": "researcher", "task_description": "Use no run lineage"},
        ),
        missing_context,
    )

    assert fallback_result.status_code == ToolResultStatus.SUCCESS
    assert missing_result.status_code == ToolResultStatus.SUCCESS
    assert captured_requests[0].metadata == {
        "parent_run_id": "execution-run",
        "parent_tool_call_id": "fallback-call",
    }
    assert captured_requests[1].metadata == {"parent_tool_call_id": "missing-call"}


@pytest.mark.parametrize("invalid_value", _contract()["identity"]["non_string_metadata_values"])
def test_non_string_identity_metadata_is_ignored_and_falls_through(
    tmp_path: Path,
    invalid_value: Any,
) -> None:
    identity_contract = _contract()["identity"]
    assert identity_contract["non_string_metadata_policy"] == "ignore_and_fall_through"
    assert invalid_value in identity_contract["non_string_metadata_values"]
    events: list[Any] = []
    captured_child_contexts: list[ToolContext] = []
    manager = _manager()
    registry = build_default_registry()

    def inspect_identity(context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        captured_child_contexts.append(context)
        return ToolExecutionResult(tool_call_id="", content="captured")

    registry.register_tool("inspect_identity", inspect_identity, "Inspect canonical child identity")
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        metadata={"trace_id": invalid_value},
        extra_tool_names=["inspect_identity"],
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_context": RunContext(
                run_id=invalid_value,
                metadata={"trace_id": invalid_value},
            ),
            "_vv_agent_run_id": invalid_value,
            "_vv_agent_trace_id": invalid_value,
            "trace_id": invalid_value,
            "_vv_agent_emit_event": events.append,
        }
    )

    outcome = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="inspect", name="inspect_identity", arguments={})],
                ),
                _finish("done"),
            ]
        ),
        tool_registry=registry,
        default_workspace=tmp_path,
    )._run_sub_task(
        parent_task=task,
        workspace_path=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
        parent_shared_state={},
        request=SubTaskRequest(
            agent_name="researcher",
            task_description="Collect facts",
            metadata={
                "parent_run_id": invalid_value,
                "parent_tool_call_id": invalid_value,
            },
        ),
        sub_task_manager=manager,
        ctx=context,
    )

    pair = _sub_events(events)
    assert outcome.status == AgentStatus.COMPLETED
    assert [event.trace_id for event in pair] == [pair[0].run_id, pair[0].run_id]
    assert all(event.parent_run_id is None for event in pair)
    assert all(event.parent_tool_call_id == "" for event in pair)
    record = manager.get(outcome.task_id)
    assert record is not None
    assert record.parent_run_id is None
    assert record.parent_tool_call_id is None
    assert len(captured_child_contexts) == 1
    child_tool_context = captured_child_contexts[0]
    child_ctx = child_tool_context.ctx
    assert child_ctx is not None
    canonical_trace_id = pair[0].run_id
    assert child_ctx.metadata["_vv_agent_trace_id"] == canonical_trace_id
    assert child_ctx.metadata["trace_id"] == canonical_trace_id
    child_run_context = child_ctx.metadata["_vv_agent_run_context"]
    assert child_tool_context.run_context is child_run_context
    assert child_run_context.metadata["trace_id"] == canonical_trace_id


@pytest.mark.parametrize("pattern", [r"(?=secret)", r"(a)\1", r"\p{Greek}"])
def test_runtime_boundary_rejects_non_portable_regex_before_lifecycle(tmp_path: Path, pattern: str) -> None:
    events: list[Any] = []
    manager = _manager()
    workspace_backend = MemoryWorkspaceBackend()
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )

    outcome = runtime._run_sub_task(
        parent_task=task,
        workspace_path=tmp_path,
        workspace_backend=workspace_backend,
        parent_shared_state={},
        request=SubTaskRequest(
            agent_name="researcher",
            task_description="Collect facts",
            exclude_files_pattern=pattern,
            metadata={"parent_tool_call_id": "delegate"},
        ),
        sub_task_manager=manager,
        ctx=context,
    )

    assert outcome.status == AgentStatus.FAILED
    assert outcome.error_code == INVALID_EXCLUDE_FILES_PATTERN_CODE
    assert outcome.error == INVALID_EXCLUDE_FILES_PATTERN_MESSAGE
    assert _sub_events(events) == []
    record = manager.get(outcome.task_id)
    assert record is not None
    assert record.outcome is outcome
    assert record.outcome.error_code == INVALID_EXCLUDE_FILES_PATTERN_CODE
    assert record.workspace_backend is workspace_backend
    assert record.parent_run_id == "parent-run"
    assert record.parent_tool_call_id == "delegate"


def test_configured_sub_agent_uses_parent_workspace_backend_and_emits_sub_run_events(
    tmp_path: Path,
) -> None:
    backend = MemoryWorkspaceBackend()
    backend.write_text("virtual.txt", "only in memory backend")
    child_requests: list[LlmRequest] = []

    def finish_child(request: LlmRequest) -> LLMResponse:
        child_requests.append(request)
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="child-finish",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "child done"},
                )
            ],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="delegate",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={"agent_id": "researcher", "task_description": "Read virtual.txt"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "virtual.txt"})],
            ),
            finish_child,
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="parent-finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )
    events: list[Any] = []
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        workspace_backend=backend,
    )
    task = AgentTask(
        task_id="parent",
        model="shared-model",
        system_prompt="Parent",
        user_prompt="Delegate",
        max_cycles=4,
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Read files")},
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )
    manager = _manager()

    result = runtime.run(task, ctx=context, sub_task_manager=manager)

    assert result.status == AgentStatus.COMPLETED
    assert child_requests
    assert any(message.role == "tool" and "only in memory backend" in message.content for message in child_requests[0].messages)
    sub_events = _sub_events(events)
    assert [event.type for event in sub_events] == _contract()["lifecycle"]["event_sequence"]
    assert sub_events[0].parent_run_id == "parent-run"
    assert sub_events[0].parent_tool_call_id == "delegate"
    assert sub_events[1].run_id == sub_events[0].run_id


def test_real_sub_run_events_normalize_line_by_line_to_shared_fixture(tmp_path: Path) -> None:
    events: list[Any] = []
    outcomes: dict[str, SubTaskOutcome] = {}
    registry = build_default_registry()

    def contract_delegate(context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        assert context.sub_task_runner is not None
        outcome = context.sub_task_runner(
            SubTaskRequest(
                agent_name="researcher",
                task_description="Collect facts",
                metadata={
                    "parent_run_id": "parent-run",
                    "parent_tool_call_id": "delegate",
                },
            )
        )
        outcomes[context.tool_call_id] = outcome
        return ToolExecutionResult(tool_call_id="", content=json.dumps(outcome.to_dict()))

    registry.register_tool("contract_delegate", contract_delegate, "Run the configured sub-agent contract request")
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="delegate", name="contract_delegate", arguments={})],
                ),
                _finish("child done", tool_call_id="child-finish"),
                _finish("parent done", tool_call_id="parent-finish"),
            ]
        ),
        tool_registry=registry,
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="child-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        extra_tool_names=["contract_delegate"],
        sub_agents={
            "researcher": SubAgentConfig(
                model="child-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=4,
            )
        },
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )

    result = runtime.run(task, ctx=context, sub_task_manager=_manager())

    assert result.status == AgentStatus.COMPLETED
    invalid_child = SubAgentConfig(
        model="child-model",
        description="Research",
        system_prompt="Child prompt",
        max_cycles=4,
    )
    invalid_child.system_prompt = " \n "
    failure_registry = build_default_registry()

    def contract_failure(context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        assert context.sub_task_runner is not None
        outcome = context.sub_task_runner(
            SubTaskRequest(
                agent_name="researcher",
                task_description="Collect invalid facts",
                metadata={
                    "parent_run_id": "parent-run",
                    "parent_tool_call_id": "delegate-failed",
                },
            )
        )
        outcomes[context.tool_call_id] = outcome
        return ToolExecutionResult(tool_call_id="", content=json.dumps(outcome.to_dict()))

    failure_registry.register_tool(
        "contract_failure",
        contract_failure,
        "Run the configured sub-agent failure contract request",
    )
    failure_runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="delegate-failed", name="contract_failure", arguments={})],
                ),
                _finish("parent recovered", tool_call_id="parent-finish-failed"),
            ]
        ),
        tool_registry=failure_registry,
        default_workspace=tmp_path,
    )
    failure_task = AgentTask(
        task_id="parent-task",
        model="child-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate invalid child",
        max_cycles=2,
        extra_tool_names=["contract_failure"],
        sub_agents={"researcher": invalid_child},
    )

    failure_result = failure_runtime.run(failure_task, ctx=context, sub_task_manager=_manager())

    assert failure_result.status == AgentStatus.COMPLETED
    sub_events = _sub_events(events)
    assert len(sub_events) == 4
    assert len({event.event_id for event in sub_events}) == 4
    assert all(event.event_id.startswith("evt_") for event in sub_events)
    assert all(event.created_at > 0 for event in sub_events)
    assert all(left.created_at <= right.created_at for left, right in pairwise(sub_events))
    assert sub_events[0].run_id == sub_events[1].run_id
    assert sub_events[2].run_id == sub_events[3].run_id
    assert sub_events[0].run_id != sub_events[2].run_id
    assert all(event.run_id != "parent-run" for event in sub_events)
    for started, completed in zip(sub_events[::2], sub_events[1::2], strict=True):
        outcome = outcomes[started.parent_tool_call_id]
        assert started.parent_tool_call_id == completed.parent_tool_call_id
        assert started.task_id and started.task_id == completed.task_id == outcome.task_id
        assert started.session_id and started.session_id == completed.session_id == outcome.session_id
        assert started.child_session_id == started.session_id
        assert completed.child_session_id == completed.session_id
    failed = cast(SubRunCompletedEvent, sub_events[3])
    lifecycle_contract = _contract()["lifecycle"]
    successful = cast(SubRunCompletedEvent, sub_events[1])
    assert successful.token_usage is not None
    assert (
        successful.token_usage["total_tokens"] == 0
    ) is lifecycle_contract["preserve_successful_zero_token_usage"]
    assert (
        failed.metadata.get("error_code") == "invalid_sub_agent_system_prompt"
    ) is lifecycle_contract["failure_error_code_in_metadata"]
    assert (failed.token_usage is None) is lifecycle_contract["omit_token_usage_when_unavailable"]
    assert [_normalize_sub_event(event) for event in sub_events] == _event_contract()


class _ChildModelProvider:
    def __init__(self, llm: ScriptedLLM) -> None:
        self.llm = llm
        self.resolved_models: list[str] = []

    def resolve(self, model_ref: Any) -> ResolvedModelConfig:
        assert model_ref.backend_name() is None
        model = model_ref.model()
        self.resolved_models.append(model)
        return ResolvedModelConfig(
            backend="test",
            requested_model=model,
            selected_model=model,
            model_id=model,
            endpoint_options=[],
            native_multimodal=True,
        )

    def client(self, _resolved: ResolvedModelConfig) -> ScriptedLLM:
        return self.llm


class _ExplicitBackendModelProvider:
    def __init__(self, llm: ScriptedLLM) -> None:
        self.llm = llm
        self.refs: list[Any] = []

    def resolve(self, model_ref: Any) -> ResolvedModelConfig:
        self.refs.append(model_ref)
        return ResolvedModelConfig(
            backend=str(model_ref.backend_name() or "default-backend"),
            requested_model=model_ref.model(),
            selected_model=model_ref.model(),
            model_id=model_ref.model(),
            endpoint_options=[],
            context_length=_contract()["model_resolution"]["resolved_token_limits"]["context_length"],
            max_output_tokens=_contract()["model_resolution"]["resolved_token_limits"]["max_output_tokens"],
        )

    def client(self, _resolved: ResolvedModelConfig) -> ScriptedLLM:
        return self.llm


@pytest.mark.parametrize(
    ("child_metadata", "expected_context", "expected_output"),
    [
        ({}, 32_000, 4_096),
        (
            {"model_context_window": 12_345, "reserved_output_tokens": 678},
            12_345,
            678,
        ),
    ],
    ids=["resolved-limits", "explicit-child-metadata-wins"],
)
def test_explicit_backend_and_resolved_limits_reach_real_child_request(
    tmp_path: Path,
    child_metadata: dict[str, int],
    expected_context: int,
    expected_output: int,
) -> None:
    model_contract = _contract()["model_resolution"]
    observed_requests: list[LlmRequest] = []

    def capture_child(request: LlmRequest) -> LLMResponse:
        observed_requests.append(request)
        return _finish("child done", tool_call_id="explicit-child-finish")

    parent_llm = ScriptedLLM(
        steps=[
            _delegate({"agent_id": "researcher", "task_description": "Resolve child model"}),
            _finish("parent done", tool_call_id="explicit-parent-finish"),
        ]
    )
    child_llm = ScriptedLLM(steps=[capture_child])
    provider = _ExplicitBackendModelProvider(child_llm)
    manager = _manager()
    events: list[Any] = []
    runtime = AgentRuntime(
        llm_client=parent_llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="explicit-backend-parent",
        model="child-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=2,
        sub_agents={
            "researcher": SubAgentConfig(
                model="child-model",
                backend="child-backend",
                description="Research",
                system_prompt="Child prompt",
                metadata=child_metadata,
            )
        },
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_model_provider": provider,
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-model-contract",
            "_vv_agent_emit_event": events.append,
        }
    )

    result = runtime.run(task, ctx=context, sub_task_manager=manager)

    assert result.status == AgentStatus.COMPLETED
    assert model_contract["explicit_backend_requires_resolver"] is True
    assert len(provider.refs) == 1
    assert provider.refs[0].backend_name() == "child-backend"
    assert provider.refs[0].model() == "child-model"
    assert observed_requests
    request_metadata = observed_requests[0].metadata
    assert request_metadata["model_context_window"] == expected_context
    assert request_metadata["reserved_output_tokens"] == expected_output
    assert (
        request_metadata["model_context_window"],
        request_metadata["reserved_output_tokens"],
    ) == (expected_context, expected_output)
    assert model_contract["resolved_token_limits"]["explicit_child_metadata_wins"] is True
    started = next(event for event in _sub_events(events) if isinstance(event, SubRunStartedEvent))
    record = manager.get(started.task_id or "")
    assert record is not None
    assert record.resolved["backend"] == "child-backend"
    assert record.resolved["model_id"] == "child-model"


def test_same_model_parent_client_reuse_inherits_parent_task_token_limits(tmp_path: Path) -> None:
    model_contract = _contract()["model_resolution"]
    portable = _contract()["validation"]["portable_whitespace"]
    expected_metadata = model_contract["resolved_token_limits"]["task_metadata"]
    child_requests: list[LlmRequest] = []

    def capture_child(request: LlmRequest) -> LLMResponse:
        child_requests.append(request)
        return _finish("child done", tool_call_id="reused-child-finish")

    llm = ScriptedLLM(
        steps=[
            _delegate({"agent_id": "researcher", "task_description": "Reuse parent client"}),
            capture_child,
            _finish("parent done", tool_call_id="reused-parent-finish"),
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        default_backend="parent-backend",
    )
    task = AgentTask(
        task_id="parent-client-reuse",
        model="child-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=2,
        metadata=dict(expected_metadata),
        sub_agents={
            "researcher": SubAgentConfig(
                model="child-model",
                backend=portable["blank_backend_input"],
                description="Research",
                system_prompt="Child prompt",
            )
        },
    )

    result = runtime.run(task, sub_task_manager=_manager())

    assert result.status == AgentStatus.COMPLETED
    assert model_contract["blank_backend_treated_as_absent"] is True
    assert model_contract["same_model_reuses_parent_client_only_without_explicit_backend"] is True
    assert model_contract["same_model_parent_client_inherits_token_limits"] is True
    assert child_requests[0].metadata["model_context_window"] == expected_metadata["model_context_window"]
    assert child_requests[0].metadata["reserved_output_tokens"] == expected_metadata["reserved_output_tokens"]


def _async_tool_context(tmp_path: Path, manager: SubTaskManager) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
        sub_task_runner=lambda request: SubTaskOutcome(
            task_id="assigned-by-handler",
            agent_name=request.agent_name,
            status=AgentStatus.COMPLETED,
            final_answer="done",
        ),
        sub_task_manager=manager,
        task_id="async-parent",
    )


def test_async_single_submit_exception_uses_shared_error_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    monkeypatch.setattr(manager, "submit", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("spawn failed")))

    result = create_sub_task(
        _async_tool_context(tmp_path, manager),
        {
            "agent_id": "researcher",
            "task_description": "single submit",
            "wait_for_completion": False,
        },
    )
    payload = json.loads(result.content)

    assert _contract()["manager"]["async_submission"]["single_error_code"] == "sub_task_submit_failed"
    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "sub_task_submit_failed"
    assert payload == {
        "ok": False,
        "error": "spawn failed",
        "error_code": "sub_task_submit_failed",
    }


@pytest.mark.parametrize("accepted_titles", [{"second"}, set()], ids=["partial-success", "all-failed"])
def test_async_batch_submit_exceptions_continue_and_report_shared_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_titles: set[str],
) -> None:
    manager = _manager()
    original_submit = manager.submit

    def selective_submit(**kwargs: Any) -> Any:
        if kwargs["task_title"] not in accepted_titles:
            raise RuntimeError(f"spawn failed: {kwargs['task_title']}")
        return original_submit(**kwargs)

    monkeypatch.setattr(manager, "submit", selective_submit)
    result = create_sub_task(
        _async_tool_context(tmp_path, manager),
        {
            "agent_id": "researcher",
            "tasks": [
                {"task_description": "first"},
                {"task_description": "second"},
                {"task_description": "third"},
            ],
            "wait_for_completion": False,
        },
    )
    payload = json.loads(result.content)
    details = payload["details"] if not accepted_titles else payload
    expected_accepted = len(accepted_titles)

    assert _contract()["manager"]["async_submission"]["batch_continues_after_item_failure"] is True
    assert details["summary"] == {"total": 3, "accepted": expected_accepted, "failed": 3 - expected_accepted}
    assert len(details["results"]) == 3
    assert [item["index"] for item in details["results"]] == [0, 1, 2]
    assert all(
        item.get("error_code") == "sub_task_submit_failed"
        for item in details["results"]
        if item["status"] == AgentStatus.FAILED.value
    )
    assert len(details["task_ids"]) == expected_accepted
    if accepted_titles:
        assert result.status_code == ToolResultStatus.SUCCESS
        assert manager.wait(details["task_ids"][0], timeout=2) is not None
    else:
        assert result.status_code == ToolResultStatus.ERROR
        assert result.error_code == _contract()["manager"]["async_submission"]["all_failed_error_code"]
        assert payload["error"] == "All batch sub-tasks failed"


def test_public_runtime_projects_capabilities_and_fresh_child_identity(tmp_path: Path) -> None:
    child_requests: list[LlmRequest] = []
    captured_contexts: list[ToolContext] = []
    forwarded_streams: list[dict[str, Any]] = []
    events: list[Any] = []

    def child_capture_request(request: LlmRequest) -> LLMResponse:
        child_requests.append(request)
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="capture", name="capture_context", arguments={})],
        )

    def child_finish_request(request: LlmRequest) -> LLMResponse:
        child_requests.append(request)
        return _finish("child done", tool_call_id="child-finish")

    llm = ScriptedLLM(
        steps=[
            _delegate({"agent_id": "researcher", "task_description": "Inspect capabilities"}),
            child_capture_request,
            child_finish_request,
            _finish("parent done", tool_call_id="parent-finish"),
        ]
    )
    provider = _ChildModelProvider(llm)
    registry = build_default_registry()

    def capture_context(context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        captured_contexts.append(context)
        assert context.ctx is not None
        assert context.ctx.stream_callback is not None
        context.ctx.stream_callback({"event": "assistant_delta", "content_delta": "capability probe"})
        return ToolExecutionResult(tool_call_id="", content="captured")

    registry.register_tool("capture_context", capture_context, "Capture child context")
    manager = _manager()
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=registry,
        default_workspace=tmp_path,
        workspace_backend=MemoryWorkspaceBackend(),
    )
    parent_settings = ModelSettings(temperature=0.1)
    effective_settings = ModelSettings(temperature=0.25, max_tokens=512)
    task = AgentTask(
        task_id="parent-task",
        model="parent-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        extra_tool_names=["capture_context"],
        model_settings=parent_settings,
        sub_agents={
            "researcher": SubAgentConfig(
                model="child-model",
                description="Research",
                backend=_contract()["model_resolution"]["blank_backend_input"],
                system_prompt="Child prompt",
                metadata={key: f"override-{key}" for key in _contract()["reserved_metadata_keys"]},
            )
        },
    )
    app_state = object()
    parent_run_context = RunContext(
        context=app_state,
        run_id="parent-run",
        agent_name="parent",
        model="parent-model",
        workspace=tmp_path,
        metadata={"parent_only": True},
    )
    parent_token = CancellationToken()
    state_store = InMemoryStateStore()
    approval_provider = object()
    approval_broker = ApprovalBroker()
    memory_providers = [object()]
    trace_context = object()
    event_sink = events.append
    stream_sink = forwarded_streams.append
    parent_session = object()
    context = ExecutionContext(
        cancellation_token=parent_token,
        stream_callback=stream_sink,
        state_store=state_store,
        metadata={
            "_vv_agent_agent_name": "parent",
            "_vv_agent_approval_provider": approval_provider,
            "_vv_agent_approval_broker": approval_broker,
            "_vv_agent_approval_timeout_seconds": 12.5,
            "_vv_agent_emit_event": event_sink,
            "_vv_agent_input": "parent input",
            "_vv_agent_memory_providers": memory_providers,
            "_vv_agent_model_provider": provider,
            "_vv_agent_model_settings": effective_settings,
            "_vv_agent_run_context": parent_run_context,
            "_vv_agent_run_id": "execution-run",
            "_vv_agent_session": parent_session,
            "_vv_agent_session_id": "parent-session",
            "_vv_agent_trace_context": trace_context,
            "_vv_agent_trace_id": "trace-parity",
            "trace_id": "trace-parity",
        },
    )

    result = runtime.run(
        task,
        shared_state={"parent_secret": "must not leak"},
        ctx=context,
        sub_task_manager=manager,
    )

    assert result.status == AgentStatus.COMPLETED
    assert provider.resolved_models == ["child-model"]
    assert child_requests and all(request.model_settings == effective_settings for request in child_requests)
    assert task.model_settings == parent_settings
    assert captured_contexts
    child_tool_context = captured_contexts[0]
    child_ctx = child_tool_context.ctx
    assert child_ctx is not None
    assert child_ctx.state_store is state_store
    assert child_ctx.cancellation_token is not None
    assert child_ctx.cancellation_token is not parent_token
    assert child_ctx.metadata["_vv_agent_approval_provider"] is approval_provider
    assert child_ctx.metadata["_vv_agent_approval_broker"] is approval_broker
    assert child_ctx.metadata["_vv_agent_approval_timeout_seconds"] == 12.5
    assert child_ctx.metadata["_vv_agent_emit_event"] is event_sink
    assert child_ctx.metadata["_vv_agent_memory_providers"] is memory_providers
    assert child_ctx.metadata["_vv_agent_model_provider"] is provider
    assert child_ctx.metadata["_vv_agent_trace_context"] is trace_context
    assert child_ctx.metadata["_vv_agent_parent_run_id"] == "parent-run"
    assert child_ctx.metadata["_vv_agent_parent_tool_call_id"] == "delegate"
    assert child_ctx.metadata["_vv_agent_agent_name"] == "researcher"
    assert child_ctx.metadata["_vv_agent_session_id"] != "parent-session"
    assert child_ctx.metadata["_vv_agent_run_id"] != "parent-run"
    assert "_vv_agent_input" not in child_ctx.metadata
    assert "_vv_agent_session" not in child_ctx.metadata
    child_run_context = child_ctx.metadata["_vv_agent_run_context"]
    assert child_run_context is not parent_run_context
    assert child_run_context.context is app_state
    assert child_run_context.run_id == child_ctx.metadata["_vv_agent_run_id"]
    assert child_run_context.agent_name == "researcher"
    assert child_run_context.model == "child-model"
    assert child_run_context.metadata["trace_id"] == "trace-parity"
    assert child_run_context.metadata["parent_run_id"] == "parent-run"
    assert child_run_context.metadata["parent_tool_call_id"] == "delegate"
    assert child_run_context.metadata["is_sub_task"] is True
    assert child_run_context.metadata["parent_task_id"] == "parent-task"
    assert child_run_context.metadata["sub_agent_name"] == "researcher"
    assert child_run_context.metadata["session_memory_enabled"] is False
    assert child_run_context.metadata["workspace"] == str(tmp_path)
    assert child_run_context.metadata["task_id"] == child_run_context.metadata["session_id"]
    assert child_run_context.metadata["session_id"] == child_ctx.metadata["_vv_agent_session_id"]
    canonical_metadata = _contract()["metadata_projection"]
    for key in _contract()["reserved_metadata_keys"]:
        if key not in canonical_metadata:
            assert key not in child_run_context.metadata
            continue
        assert not str(child_run_context.metadata[key]).startswith("override-")
    assert "parent_only" not in child_run_context.metadata
    assert child_tool_context.run_context is child_run_context
    assert child_tool_context.session is None
    assert "parent_secret" not in child_tool_context.shared_state
    assert forwarded_streams[-1]["event"] == "assistant_delta"
    observed_capabilities = {
        "inherited": [
            name
            for name, present in {
                "app_state": child_run_context.context is app_state,
                "approval_broker": child_ctx.metadata.get("_vv_agent_approval_broker") is approval_broker,
                "approval_provider": child_ctx.metadata.get("_vv_agent_approval_provider") is approval_provider,
                "approval_timeout": child_ctx.metadata.get("_vv_agent_approval_timeout_seconds") == 12.5,
                "event_sink": child_ctx.metadata.get("_vv_agent_emit_event") is event_sink,
                "memory_providers": child_ctx.metadata.get("_vv_agent_memory_providers") is memory_providers,
                "model_provider": child_ctx.metadata.get("_vv_agent_model_provider") is provider,
                "state_store": child_ctx.state_store is state_store,
                "stream_sink": forwarded_streams[-1].get("event") == "assistant_delta",
                "trace_context": child_ctx.metadata.get("_vv_agent_trace_context") is trace_context,
            }.items()
            if present
        ],
        "derived": [
            name
            for name, present in {
                "agent_name": child_run_context.agent_name == "researcher",
                "cancellation_token": (
                    child_ctx.cancellation_token is not None
                    and child_ctx.cancellation_token is not parent_token
                ),
                "run_context": child_run_context is not parent_run_context,
                "run_id": child_run_context.run_id == child_ctx.metadata.get("_vv_agent_run_id"),
                "session_id": child_run_context.metadata.get("session_id") == child_ctx.metadata.get("_vv_agent_session_id"),
            }.items()
            if present
        ],
        "isolated": [
            name
            for name, isolated in {
                "input": "_vv_agent_input" not in child_ctx.metadata,
                "parent_run_context": child_run_context is not parent_run_context,
                "session": child_tool_context.session is None and "_vv_agent_session" not in child_ctx.metadata,
                "shared_state": "parent_secret" not in child_tool_context.shared_state,
            }.items()
            if isolated
        ],
        "lineage": [
            name
            for name, present in {
                "parent_run_id": child_run_context.metadata.get("parent_run_id") == "parent-run",
                "parent_tool_call_id": child_run_context.metadata.get("parent_tool_call_id") == "delegate",
            }.items()
            if present
        ],
    }
    assert observed_capabilities == _contract()["capability_projection"]

    sub_event_pair = _sub_events(events)
    assert [event.type for event in sub_event_pair] == ["sub_run_started", "sub_run_completed"]
    assert all(event.parent_run_id == "parent-run" for event in sub_event_pair)
    assert all(event.parent_tool_call_id == "delegate" for event in sub_event_pair)
    record = manager.get(sub_event_pair[0].task_id or "")
    assert record is not None
    assert record.session is not None
    assert (
        record.parent_run_id == "parent-run" and record.parent_tool_call_id == "delegate"
    ) is _contract()["manager"]["persists_parent_lineage"]
    assert _contract()["model_resolution"]["blank_backend_treated_as_absent"] is True
    assert record.resolved == _contract()["model_resolution"]["resolved_without_endpoint"]
    child_session = cast(Any, record.session)
    assert child_session.definition.native_multimodal is True
    assert child_session._approval_broker is approval_broker


class _MaliciousChildStreamLLM:
    def __init__(self) -> None:
        self.parent_calls = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, model_settings
        if request_metadata and request_metadata.get("is_sub_task") is True:
            assert stream_callback is not None
            for event in (
                {
                    "event": "assistant_delta",
                    "content_delta": "child delta",
                    "content_chars": 11,
                    "estimated_tokens": 2,
                    "cycle": 999,
                    "run_id": "spoof-run",
                    "trace_id": "spoof-trace",
                    "agent_name": "spoof-agent",
                    "session_id": "spoof-session",
                    "status": "completed",
                    "final_output": "spoof-output",
                    "metadata": {"spoof": True},
                },
                {
                    "event": "reasoning_delta",
                    "reasoning_delta": "child plan",
                    "reasoning_chars": 10,
                    "estimated_tokens": 3,
                },
                {
                    "event": "tool_call_started",
                    "tool_call_id": "child-stream-finish",
                    "tool_call_index": 0,
                    "function_name": TASK_FINISH_TOOL_NAME,
                    "arguments_chars": 0,
                    "estimated_tokens": 0,
                },
                {
                    "event": "tool_call_progress",
                    "tool_call_id": "child-stream-finish",
                    "tool_call_index": 0,
                    "function_name": TASK_FINISH_TOOL_NAME,
                    "arguments_chars": 24,
                    "estimated_tokens": 6,
                },
            ):
                stream_callback(event)
            stream_callback(
                {
                    "event": "run_completed",
                    "run_id": "spoof-run",
                    "status": "completed",
                    "final_output": "spoof-output",
                }
            )
            return _finish("child done", tool_call_id="child-stream-finish")

        self.parent_calls += 1
        if self.parent_calls == 1:
            return _delegate({"agent_id": "researcher", "task_description": "Stream safely"})
        return _finish("parent done", tool_call_id="parent-stream-finish")


class _ObserverPanicChildStreamLLM:
    def __init__(self) -> None:
        self.parent_calls = 0
        self.child_calls = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, model_settings
        if request_metadata and request_metadata.get("is_sub_task") is True:
            self.child_calls += 1
            assert stream_callback is not None
            stream_callback({"event": "assistant_delta", "content_delta": f"child delta {self.child_calls}"})
            return _finish(f"child answer {self.child_calls}", tool_call_id=f"child-finish-{self.child_calls}")

        self.parent_calls += 1
        if self.parent_calls == 1:
            return _delegate({"agent_id": "researcher", "task_description": "Stream through observer"})
        return _finish("parent done", tool_call_id="parent-observer-finish")


def test_configured_child_stream_observer_failure_isolated_and_child_remains_continuable(tmp_path: Path) -> None:
    llm = _ObserverPanicChildStreamLLM()
    manager = _manager()
    lifecycle: list[Any] = []
    observer_calls = 0

    def broken_observer(_event: dict[str, Any]) -> None:
        nonlocal observer_calls
        observer_calls += 1
        raise RuntimeError("caller stream observer failed")

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="observer-panic-parent",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=2,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
            )
        },
    )
    context = ExecutionContext(
        stream_callback=broken_observer,
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-stream-panic",
            "_vv_agent_emit_event": lifecycle.append,
        },
    )

    result = runtime.run(task, ctx=context, sub_task_manager=manager)

    assert result.status == AgentStatus.COMPLETED
    task_id = str(result.cycles[0].tool_results[0].metadata["task_id"])
    initial = manager.get(task_id)
    assert initial is not None and initial.outcome is not None
    assert initial.outcome.status == AgentStatus.COMPLETED
    assert initial.outcome.final_answer == "child answer 1"

    continuation_context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
        ctx=context,
        tool_call_id="continue-observer-child",
    )
    manager._continue_task_with_context(
        task_id=task_id,
        prompt="continue after observer failure",
        context=continuation_context,
    )
    continued = manager.wait(task_id, timeout=2)
    assert continued is not None and continued.outcome is not None
    assert continued.outcome.status == AgentStatus.COMPLETED
    assert continued.outcome.final_answer == "child answer 2"
    assert observer_calls == 2
    assert [event.type for event in _sub_events(lifecycle)] == [
        "sub_run_started",
        "sub_run_completed",
        "sub_run_started",
        "sub_run_completed",
    ]
    assert _contract()["stream_forwarding"]["stream_observer_failure_isolated"] is True


def test_configured_child_stream_is_allowlisted_and_keeps_canonical_typed_identity(tmp_path: Path) -> None:
    stream_contract = _contract()["stream_forwarding"]
    llm = _MaliciousChildStreamLLM()
    provider = ScriptedModelProvider(
        backend="test",
        default_model="shared-model",
        llm=llm,
        context_length=None,
        max_output_tokens=None,
    )
    task = AgentTask(
        task_id="parent-stream-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=2,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
            )
        },
    )
    raw_streams: list[dict[str, Any]] = []
    typed_events: list[Any] = []

    result = Runner._run_compiled_sync(
        Agent(name="parent", instructions="Delegate.", model="shared-model"),
        "Delegate",
        task=task,
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            runtime_stream_callback=raw_streams.append,
            stream=typed_events.append,
            tracing={"trace_id": "trace-stream-contract"},
        ),
    )

    assert result.status == AgentStatus.COMPLETED
    assert len(raw_streams) == 4
    assert [raw["event"] for raw in raw_streams] == stream_contract["allowed_events"]
    identity_fields = set(stream_contract["canonical_identity_fields"])
    cycle_field = stream_contract["canonical_cycle_field"]
    for raw in raw_streams:
        producer_fields = set(stream_contract["producer_fields"][raw["event"]])
        assert set(raw).issubset(producer_fields | identity_fields | {cycle_field})
        assert (set(raw) - identity_fields - {cycle_field}).isdisjoint(
            stream_contract["reserved_producer_fields"]
        )
        assert raw[cycle_field] == 1
    assert raw_streams[0]["content_delta"] == "child delta"
    assert raw_streams[0]["content_chars"] == 11
    assert raw_streams[0]["estimated_tokens"] == 2

    started = next(event for event in result.events if isinstance(event, SubRunStartedEvent))
    for raw in raw_streams:
        assert raw["run_id"] == raw["child_run_id"] == started.run_id
        assert raw["session_id"] == raw["child_session_id"] == started.session_id
        assert raw["task_id"] == started.task_id
        assert raw["agent_name"] == raw["sub_agent_name"] == "researcher"
        assert raw["trace_id"] == "trace-stream-contract"
        assert raw["parent_run_id"] == result.run_id
        assert raw["parent_tool_call_id"] == "delegate"

    child_stream_types = (
        AssistantDeltaEvent,
        ReasoningDeltaEvent,
        ModelToolCallStartedEvent,
        ModelToolCallProgressEvent,
    )
    typed_streams = [event for event in typed_events if isinstance(event, child_stream_types)]
    assert [event.type for event in typed_streams] == [
        stream_contract["typed_wire_types"][raw["event"]] for raw in raw_streams
    ]
    for typed_event, raw in zip(typed_streams, raw_streams, strict=True):
        assert typed_event.run_id == started.run_id
        assert typed_event.trace_id == "trace-stream-contract"
        assert typed_event.agent_name == "researcher"
        assert typed_event.session_id == started.session_id
        assert typed_event.parent_run_id == result.run_id
        assert typed_event.cycle_index == 1
        assert typed_event.metadata == raw

    parent_terminals = [
        event
        for event in typed_events
        if isinstance(event, RunCompletedEvent) and event.run_id == result.run_id
    ]
    assert len(parent_terminals) == 1
    assert stream_contract["untrusted_terminal_cannot_suppress_real_terminal"] is True


class _PolicyContinuationLLM:
    def __init__(self, child_task_id: dict[str, str]) -> None:
        self.child_task_id = child_task_id
        self.parent_calls = 0
        self.child_calls = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, stream_callback, model_settings
        if request_metadata and request_metadata.get("is_sub_task") is True:
            self.child_calls += 1
            if self.child_calls == 1:
                return _finish("initial child done", tool_call_id="initial-child-finish")
            if self.child_calls in {2, 4, 6, 8, 10}:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"dangerous-{self.child_calls}",
                            name="dangerous_action",
                            arguments={"value": self.child_calls},
                        )
                    ],
                )
            return _finish(
                f"continued child done {self.child_calls}",
                tool_call_id=f"continued-child-finish-{self.child_calls}",
            )

        self.parent_calls += 1
        if self.parent_calls == 1:
            return _delegate({"agent_id": "researcher", "task_description": "Create retained child"})
        if self.parent_calls in {2, 4, 6, 8, 10, 12}:
            return _finish(f"parent done {self.parent_calls}", tool_call_id=f"parent-finish-{self.parent_calls}")
        assert self.child_task_id.get("value")
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"continue-child-{self.parent_calls}",
                    name="sub_task_status",
                    arguments={
                        "task_ids": [self.child_task_id["value"]],
                        "message": f"continue from parent turn {self.parent_calls}",
                        "wait_for_response": True,
                    },
                )
            ],
        )


class _PolicyApprovalProvider:
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    def should_request(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.tool_name == "dangerous_action":
            return ApprovalDecision.deny("dangerous action denied")
        return ApprovalDecision.allow()


def test_configured_child_continuation_uses_current_turn_predicate_and_approval_policy(tmp_path: Path) -> None:
    policy_contract = _contract()["tool_policy_projection"]
    assert policy_contract["child_may_only_tighten"] is True
    assert policy_contract["schema_and_execution_enforced"] is True
    approval_modes = policy_contract["approval_modes"]
    assert approval_modes["values"] == ["default", "always", "never", "on_request"]
    assert approval_modes["default_is_merge_sentinel"] is True
    assert approval_modes["on_request_is_explicit_override"] is True
    assert approval_modes["tool_declaration_modes"] == ["default", "on_request"]
    assert approval_modes["predicate_bypass_modes"] == ["always", "never"]

    child_task_id: dict[str, str] = {}
    llm = _PolicyContinuationLLM(child_task_id)
    provider = ScriptedModelProvider(
        backend="test",
        default_model="shared-model",
        llm=llm,
        context_length=None,
        max_output_tokens=None,
    )
    manager = _manager()
    approval_provider = _PolicyApprovalProvider()
    approval_broker = ApprovalBroker()
    executed: list[int] = []
    approval_predicate_calls: list[int] = []

    def needs_dangerous_approval(_context: ToolContext, arguments: dict[str, Any]) -> bool:
        approval_predicate_calls.append(int(arguments["value"]))
        return True

    @function_tool(needs_approval=needs_dangerous_approval)
    def dangerous_action(value: int) -> str:
        """Record a dangerous action."""
        executed.append(value)
        return "executed"

    dangerous_action.metadata["configured_tool"] = "dangerous_action"

    agent = Agent(
        name="parent",
        instructions="Delegate and manage the retained child.",
        model="shared-model",
        tools=[dangerous_action],
    )

    def runtime_task(turn: int) -> AgentTask:
        return AgentTask(
            task_id=f"parent-policy-turn-{turn}",
            model="shared-model",
            system_prompt="Parent prompt",
            user_prompt="Manage child",
            max_cycles=2,
            extra_tool_names=["dangerous_action"],
            sub_agents={
                "researcher": SubAgentConfig(
                    model="shared-model",
                    description="Research",
                    system_prompt="Child prompt",
                    max_cycles=2,
                )
            },
        )

    first = Runner._run_compiled_sync(
        agent,
        "create child",
        task=runtime_task(1),
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            tool_policy=ToolPolicy(approval="never"),
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )
    first_payload = json.loads(first.raw_result.cycles[0].tool_results[0].content)
    child_task_id["value"] = first_payload["task_id"]

    second = Runner._run_compiled_sync(
        agent,
        "continue with predicate",
        task=runtime_task(2),
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            tool_policy=ToolPolicy(
                can_use_tool=lambda name, _arguments: name != "dangerous_action",
            ),
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )
    assert second.status == AgentStatus.COMPLETED
    assert executed == []
    assert all(request.tool_name != "dangerous_action" for request in approval_provider.requests)

    approval_provider.requests.clear()
    third = Runner._run_compiled_sync(
        agent,
        "continue with approval",
        task=runtime_task(3),
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            tool_policy=ToolPolicy(approval="on_request"),
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )

    assert third.status == AgentStatus.COMPLETED
    assert executed == []
    dangerous_requests = [request for request in approval_provider.requests if request.tool_name == "dangerous_action"]
    assert len(dangerous_requests) == 1
    assert dangerous_requests[0].metadata["tool_metadata"] == {"configured_tool": "dangerous_action"}

    approval_provider.requests.clear()
    fourth = Runner._run_compiled_sync(
        agent,
        "continue with default approval",
        task=runtime_task(4),
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            tool_policy=ToolPolicy(approval="default"),
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )
    assert fourth.status == AgentStatus.COMPLETED
    assert executed == []
    assert [request.tool_name for request in approval_provider.requests].count("dangerous_action") == 1

    approval_provider.requests.clear()
    fifth = Runner._run_compiled_sync(
        agent,
        "continue with forced approval",
        task=runtime_task(5),
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            tool_policy=ToolPolicy(approval="always"),
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )
    assert fifth.status == AgentStatus.COMPLETED
    assert executed == []
    assert [request.tool_name for request in approval_provider.requests].count("dangerous_action") == 1

    approval_provider.requests.clear()
    sixth = Runner._run_compiled_sync(
        agent,
        "continue without approval",
        task=runtime_task(6),
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            tool_policy=ToolPolicy(approval="never"),
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )
    assert sixth.status == AgentStatus.COMPLETED
    assert executed == [10]
    assert all(request.tool_name != "dangerous_action" for request in approval_provider.requests)
    assert approval_predicate_calls == [4, 6]
    record = manager.get(child_task_id["value"])
    assert record is not None and record.outcome is not None
    assert record.outcome.status == AgentStatus.COMPLETED


def test_manual_approval_resume_runs_sub_task_status_with_accepting_turn_snapshot(tmp_path: Path) -> None:
    manager = _manager()
    snapshots: list[Any] = []
    task_id = "retained-child"
    _attach_continuable_manager_task(
        manager,
        task_id=task_id,
        session=_SnapshotManagerSession(snapshots),
    )
    token = CancellationToken()
    raw_stream: list[dict[str, Any]] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="approved-status-call",
                                name="sub_task_status",
                                arguments={
                                    "task_ids": [task_id],
                                    "message": "continue from approved parent turn",
                                    "wait_for_response": True,
                                },
                            )
                        ],
                    )
                ]
            ),
                ResolvedModelConfig(
                    backend="test",
                    requested_model="approval-model",
                    selected_model="approval-model",
                    model_id="approval-model",
                    endpoint_options=[],
                ),
        )

    runtime_task = AgentTask(
        task_id="parent-approval-turn",
        model="approval-model",
        system_prompt="Manage retained child.",
        user_prompt="Continue child.",
        max_cycles=1,
        sub_agents={
            "researcher": SubAgentConfig(model="approval-model", description="Retained child")
        },
        metadata={"_vv_agent_tool_use_behavior": "stop_on_first_tool"},
    )
    interrupted = Runner._run_compiled_sync(
        Agent(name="parent", instructions="Manage child.", model="approval-model"),
        "continue child",
        task=runtime_task,
        run_config=RunConfig(
            model_provider=model_provider,
            workspace=tmp_path,
            sub_task_manager=manager,
            cancellation_token=token,
            runtime_stream_callback=raw_stream.append,
            metadata={"trace_id": "trace-approved-status"},
            tool_policy=ToolPolicy(
                allowed_tools=["task_finish", "sub_task_status"],
                approval="always",
                can_use_tool=lambda name, _arguments: name in {"task_finish", "sub_task_status"},
            ),
        ),
    )
    assert interrupted.status == AgentStatus.WAIT_USER
    assert snapshots == []
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.status == AgentStatus.COMPLETED
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.cancellation_token is token
    assert snapshot.trace_id == interrupted.trace_id
    assert snapshot.parent_run_id == interrupted.run_id
    assert snapshot.parent_tool_call_id == "approved-status-call"
    assert snapshot.allowed_tools == ("task_finish", "sub_task_status")
    assert snapshot.disallowed_tools is None
    assert callable(snapshot.can_use_tool)
    assert snapshot.approval == "always"
    assert callable(snapshot.event_sink)
    assert callable(snapshot.stream_callback)
    assert raw_stream == [{"event": "assistant_delta", "delta": "continued"}]
    record = manager.get(task_id)
    assert record is not None and record.outcome is not None
    assert record.outcome.status == AgentStatus.COMPLETED
    assert record.parent_run_id == interrupted.run_id
    assert record.parent_tool_call_id == "approved-status-call"


def test_continuation_trace_falls_back_to_parent_task_metadata() -> None:
    context = ToolContext(
        workspace=Path.cwd(),
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
        ctx=ExecutionContext(metadata={}),
        task_metadata={"_vv_agent_trace_id": "trace-from-task"},
        run_context=RunContext(metadata={}),
    )

    snapshot = SubTaskManager._capture_turn_snapshot(context)

    assert snapshot.trace_id == "trace-from-task"


def test_model_resolution_failure_emits_started_before_resolution_and_one_completed(tmp_path: Path) -> None:
    events: list[Any] = []
    observed_at_resolution: list[str] = []
    resolution_requests: list[tuple[str, str]] = []

    def failing_builder(*_args: Any, **kwargs: Any) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        observed_at_resolution.extend(event.type for event in _sub_events(events))
        resolution_requests.append((kwargs["backend"], kwargs["model"]))
        raise RuntimeError("child model unavailable")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[_delegate({"agent_id": "researcher", "task_description": "Collect facts"})]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=tmp_path / "settings.py",
        default_backend="test",
        llm_builder=failing_builder,
    )
    task = AgentTask(
        task_id="parent-task",
        model="parent-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=1,
        sub_agents={
            "researcher": SubAgentConfig(
                model="child-model",
                description="Research",
                backend=_contract()["model_resolution"]["blank_backend_input"],
            )
        },
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )
    manager = _manager()

    result = runtime.run(task, ctx=context, sub_task_manager=manager)

    assert observed_at_resolution == ["sub_run_started"]
    assert resolution_requests == [("test", "child-model")]
    assert _contract()["model_resolution"]["blank_backend_treated_as_absent"] is True
    pair = _sub_events(events)
    assert (observed_at_resolution == ["sub_run_started"]) is _contract()["lifecycle"][
        "started_before_model_resolution"
    ]
    assert [event.type for event in pair] == ["sub_run_started", "sub_run_completed"]
    assert _lifecycle_is_fully_paired(events) is _contract()["lifecycle"]["completed_after_every_started"]
    assert pair[1].run_id == pair[0].run_id
    assert pair[1].status == _contract()["lifecycle"]["resolution_failure_status"]
    assert isinstance(pair[1], SubRunCompletedEvent)
    assert pair[1].error == "child model unavailable"
    assert pair[1].metadata["error_code"] == _contract()["lifecycle"]["failure_error_code_fallback"]
    assert (pair[1].token_usage is None) is _contract()["lifecycle"]["omit_token_usage_when_unavailable"]
    failed_record = manager.get(pair[0].task_id or "")
    assert failed_record is not None
    assert failed_record.parent_run_id == "parent-run"
    assert failed_record.parent_tool_call_id == "delegate"
    assert failed_record.workspace_backend is not None
    tool_result = result.cycles[0].tool_results[0]
    assert tool_result.error_code == "sub_task_failed"
    assert json.loads(tool_result.content)["error"] == "child model unavailable"


def test_failed_child_preserves_usage_after_a_completed_cycle(tmp_path: Path) -> None:
    events: list[Any] = []

    def fail_second_child_request(_request: LlmRequest) -> LLMResponse:
        raise RuntimeError("second child request failed")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                _delegate({"agent_id": "researcher", "task_description": "Collect facts"}),
                LLMResponse(
                    content="continue",
                    raw={
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                        }
                    },
                ),
                fail_second_child_request,
                _finish("parent handled failure", tool_call_id="parent-finish"),
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
        max_cycles=2,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=3,
            )
        },
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )

    result = runtime.run(task, ctx=context, sub_task_manager=_manager())

    assert result.status == AgentStatus.COMPLETED
    completed = cast(SubRunCompletedEvent, _sub_events(events)[1])
    assert completed.status == "failed"
    assert completed.metadata["error_code"] == _contract()["lifecycle"]["failure_error_code_fallback"]
    assert _contract()["lifecycle"]["preserve_failed_usage_after_completed_cycle"] is True
    assert completed.token_usage is not None
    assert completed.token_usage["prompt_tokens"] == 11
    assert completed.token_usage["completion_tokens"] == 7
    assert completed.token_usage["total_tokens"] == 18
    assert completed.token_usage["cycles"][0]["cycle_index"] == 1


def test_child_runtime_inherits_settings_file_and_default_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "settings.py"
    settings_file.write_text("LLM_SETTINGS = {}\n", encoding="utf-8")
    observed_runtime_config: list[tuple[Path | None, str | None, bool]] = []
    original_build_memory_manager = AgentRuntime._build_memory_manager

    def capture_runtime_config(
        self: AgentRuntime,
        *,
        task: AgentTask,
        workspace_path: Path,
        ctx: Any = None,
    ) -> Any:
        observed_runtime_config.append(
            (self.settings_file, self.default_backend, bool(task.metadata.get("is_sub_task")))
        )
        return original_build_memory_manager(
            self,
            task=task,
            workspace_path=workspace_path,
            ctx=ctx,
        )

    monkeypatch.setattr(AgentRuntime, "_build_memory_manager", capture_runtime_config)
    llm = ScriptedLLM(
        steps=[
            _delegate({"agent_id": "researcher", "task_description": "Inspect runtime config"}),
            _finish("child done", tool_call_id="child-finish"),
            _finish("parent done", tool_call_id="parent-finish"),
        ]
    )

    def builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        return (
            llm,
            ResolvedModelConfig(
                backend=backend,
                requested_model=model,
                selected_model=model,
                model_id=model,
                endpoint_options=[],
            ),
        )

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="test",
        llm_builder=builder,
    )
    task = AgentTask(
        task_id="parent-task",
        model="parent-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        sub_agents={"researcher": SubAgentConfig(model="child-model", description="Research")},
    )

    result = runtime.run(task, sub_task_manager=_manager())

    assert result.status == AgentStatus.COMPLETED
    parent_config = next(config for config in observed_runtime_config if not config[2])
    child_config = next(config for config in observed_runtime_config if config[2])
    inheritance = _contract()["model_resolution"]["child_runtime_inherits"]
    assert (child_config[0] == parent_config[0] == settings_file.resolve()) is inheritance["settings_file"]
    assert (child_config[1] == parent_config[1] == "test") is inheritance["default_backend"]


@pytest.mark.parametrize(
    ("child_response", "expected_status"),
    [
        (
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="wait", name="wait_child", arguments={})],
            ),
            AgentStatus.WAIT_USER,
        ),
        (LLMResponse(content="still working"), AgentStatus.MAX_CYCLES),
    ],
)
def test_wait_user_and_max_cycles_each_complete_started_lifecycle(
    tmp_path: Path,
    child_response: LLMResponse,
    expected_status: AgentStatus,
) -> None:
    events: list[Any] = []
    registry = build_default_registry()
    registry.register_tool(
        "wait_child",
        lambda _context, _arguments: ToolExecutionResult(
            tool_call_id="",
            content="Need input",
            directive=ToolDirective.WAIT_USER,
            metadata={"question": "Need input"},
        ),
        "Wait for input",
    )
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                _delegate({"agent_id": "researcher", "task_description": "Collect facts"}),
                child_response,
            ]
        ),
        tool_registry=registry,
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=1,
        extra_tool_names=["wait_child"],
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=1,
            )
        },
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )
    manager = _manager()

    result = runtime.run(task, ctx=context, sub_task_manager=manager)

    outcome = json.loads(result.cycles[0].tool_results[0].content)
    assert outcome["status"] == expected_status.value
    pair = _sub_events(events)
    assert [event.type for event in pair] == ["sub_run_started", "sub_run_completed"]
    assert _lifecycle_is_fully_paired(events) is _contract()["lifecycle"]["completed_after_every_started"]
    assert pair[1].status == expected_status.value
    if expected_status == AgentStatus.MAX_CYCLES:
        rejected = False
        with pytest.raises(RuntimeError, match="reached max cycles"):
            manager.continue_task(task_id=outcome["task_id"], prompt="continue")
        rejected = True
        assert rejected is _contract()["manager"]["continue_max_cycles_rejected"]


class _BlockingConfiguredLLM:
    def __init__(self, create_arguments: dict[str, Any]) -> None:
        self.create_arguments = create_arguments
        self.child_started: queue.Queue[None] = queue.Queue()
        self.release = threading.Event()
        self._parent_calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Any = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, tools, stream_callback, model_settings, request_metadata
        if messages and messages[0].role == "system" and messages[0].content == "Child prompt":
            self.child_started.put(None)
            if not self.release.wait(timeout=5):
                raise TimeoutError("test child was not released")
            return _finish("child done", tool_call_id="child-finish")

        with self._lock:
            self._parent_calls += 1
            parent_call = self._parent_calls
        if parent_call == 1:
            return _delegate(self.create_arguments)
        return _finish("parent done", tool_call_id="parent-finish")


def _blocked_runtime(
    tmp_path: Path,
    create_arguments: dict[str, Any],
    *,
    execution_backend: ThreadBackend | None = None,
    workspace_backend: MemoryWorkspaceBackend | None = None,
) -> tuple[
    _BlockingConfiguredLLM,
    AgentRuntime,
    AgentTask,
    ExecutionContext,
    CancellationToken,
    SubTaskManager,
    list[Any],
]:
    llm = _BlockingConfiguredLLM(create_arguments)
    manager = _manager()
    events: list[Any] = []
    token = CancellationToken()
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        execution_backend=execution_backend,
        workspace_backend=workspace_backend,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=2,
            )
        },
    )
    context = ExecutionContext(
        cancellation_token=token,
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        },
    )
    return llm, runtime, task, context, token, manager, events


def test_parent_cancellation_reaches_sync_configured_sub_agent(tmp_path: Path) -> None:
    cancellation = _assert_cancellation_contract("sync")
    llm, runtime, task, context, token, manager, events = _blocked_runtime(
        tmp_path,
        {"agent_id": "researcher", "task_description": "Inspect cancellation"},
    )
    holder: dict[str, Any] = {}

    def run_parent() -> None:
        try:
            holder["result"] = runtime.run(task, ctx=context, sub_task_manager=manager)
        except BaseException as exc:
            holder["error"] = exc

    run_thread = threading.Thread(target=run_parent)
    run_thread.start()
    llm.child_started.get(timeout=2)

    token.cancel()
    llm.release.set()
    run_thread.join(timeout=3)

    assert not run_thread.is_alive()
    assert "error" not in holder
    parent_result = holder.get("result")
    assert parent_result is not None
    assert parent_result.status == AgentStatus.FAILED
    assert parent_result.error == "Operation was cancelled"
    create_payload = json.loads(parent_result.cycles[0].tool_results[0].content)
    assert create_payload["status"] == AgentStatus.FAILED.value
    pair = _sub_events(events)
    assert [event.type for event in pair] == ["sub_run_started", "sub_run_completed"]
    assert pair[1].status == AgentStatus.FAILED.value
    record = manager.get(pair[0].task_id or "")
    assert record is not None and record.outcome is not None
    assert record.outcome.status.value == cancellation["terminal_status"]
    assert record.outcome.error is not None and "cancel" in record.outcome.error.lower()


def test_parent_cancellation_reaches_async_configured_sub_agent_and_preserves_lineage(tmp_path: Path) -> None:
    cancellation = _assert_cancellation_contract("async")
    llm, runtime, task, context, token, manager, events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "task_description": "Inspect async cancellation",
            "wait_for_completion": False,
        },
    )

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    create_payload = json.loads(parent_result.cycles[0].tool_results[0].content)
    task_id = create_payload["task_id"]
    llm.child_started.get(timeout=2)
    running = manager.get(task_id)
    assert parent_result.status == AgentStatus.COMPLETED
    assert running is not None and running.is_running() and running.outcome is None
    assert _contract()["lifecycle"]["async_may_finish_after_parent"] is True

    token.cancel()
    llm.release.set()
    record = manager.wait(task_id, timeout=3)

    assert record is not None and record.outcome is not None
    assert record.outcome.status.value == cancellation["terminal_status"]
    assert record.outcome.error is not None and "cancel" in record.outcome.error.lower()
    pair = _sub_events(events)
    assert [event.type for event in pair] == ["sub_run_started", "sub_run_completed"]
    assert all(event.parent_run_id == "parent-run" for event in pair)
    assert all(event.parent_tool_call_id == "delegate" for event in pair)


def test_initial_async_lineage_ignores_task_metadata_and_keeps_preallocated_identity(tmp_path: Path) -> None:
    manager_contract = _contract()["manager"]
    llm, runtime, task, context, _token, manager, events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "task_description": "Inspect trusted initial lineage",
            "wait_for_completion": False,
        },
    )
    task.metadata.update(
        {
            "_vv_agent_run_id": "spoof-task-run",
            "parent_run_id": "spoof-parent-run",
            "parent_tool_call_id": "spoof-parent-tool",
        }
    )
    context.metadata["_vv_agent_run_id"] = "execution-run"

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    create_payload = json.loads(parent_result.cycles[0].tool_results[0].content)
    task_id = create_payload["task_id"]
    session_id = create_payload["session_id"]
    llm.child_started.get(timeout=2)
    running = manager.get(task_id)
    started = next(event for event in _sub_events(events) if isinstance(event, SubRunStartedEvent))

    assert manager_contract["initial_lineage_ignores_task_metadata"] is True
    assert create_payload["status"] == AgentStatus.RUNNING.value
    assert task_id == session_id == started.task_id == started.session_id
    assert running is not None and running.is_running()
    assert running.task_id == task_id
    assert running.session_id == session_id
    assert running.parent_run_id == started.parent_run_id == "execution-run"
    assert running.parent_tool_call_id == started.parent_tool_call_id == "delegate"
    assert running.parent_run_id != "spoof-task-run"
    assert running.parent_run_id != "spoof-parent-run"
    assert running.parent_tool_call_id != "spoof-parent-tool"

    llm.release.set()
    completed = manager.wait(task_id, timeout=3)
    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.COMPLETED
    pair = _sub_events(events)
    assert [event.type for event in pair] == ["sub_run_started", "sub_run_completed"]
    assert all(event.task_id == task_id and event.session_id == session_id for event in pair)
    assert all(event.parent_run_id == "execution-run" for event in pair)


def test_async_configured_sub_agent_may_complete_after_parent(tmp_path: Path) -> None:
    llm, runtime, task, context, _token, manager, events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "task_description": "Finish after parent",
            "wait_for_completion": False,
        },
    )

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    task_id = json.loads(parent_result.cycles[0].tool_results[0].content)["task_id"]
    llm.child_started.get(timeout=2)
    running = manager.get(task_id)

    assert parent_result.status == AgentStatus.COMPLETED
    assert running is not None and running.is_running() and running.outcome is None
    assert _contract()["lifecycle"]["async_may_finish_after_parent"] is True

    llm.release.set()
    completed = manager.wait(task_id, timeout=3)

    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.COMPLETED
    assert _lifecycle_is_fully_paired(events) is _contract()["lifecycle"]["completed_after_every_started"]
    pair = _sub_events(events)
    assert all(event.parent_run_id == "parent-run" for event in pair)
    assert all(event.parent_tool_call_id == "delegate" for event in pair)


def test_parent_cancellation_reaches_batch_configured_sub_agent_workers(tmp_path: Path) -> None:
    cancellation = _assert_cancellation_contract("batch")
    llm, runtime, task, context, token, manager, events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "tasks": [
                {"task_description": "Inspect batch cancellation A"},
                {"task_description": "Inspect batch cancellation B"},
            ],
        },
        execution_backend=ThreadBackend(max_workers=2),
    )
    holder: dict[str, Any] = {}

    def run_parent() -> None:
        try:
            holder["result"] = runtime.run(task, ctx=context, sub_task_manager=manager)
        except BaseException as exc:
            holder["error"] = exc

    run_thread = threading.Thread(target=run_parent)
    run_thread.start()
    llm.child_started.get(timeout=2)
    llm.child_started.get(timeout=2)

    token.cancel()
    llm.release.set()
    run_thread.join(timeout=3)

    assert not run_thread.is_alive()
    assert "error" not in holder
    parent_result = holder.get("result")
    assert parent_result is not None
    assert parent_result.status == AgentStatus.FAILED
    assert parent_result.error == "Operation was cancelled"
    create_payload = json.loads(parent_result.cycles[0].tool_results[0].content)
    assert all(
        item["status"] == AgentStatus.FAILED.value for item in create_payload["details"]["results"]
    )
    grouped: dict[str, list[str]] = {}
    for event in _sub_events(events):
        grouped.setdefault(event.run_id, []).append(event.type)
        assert event.parent_run_id == "parent-run"
        assert event.parent_tool_call_id == "delegate"
    assert sorted(grouped.values()) == [["sub_run_started", "sub_run_completed"]] * 2
    for event in _sub_events(events):
        if not isinstance(event, SubRunStartedEvent):
            continue
        record = manager.get(event.task_id or "")
        assert record is not None and record.outcome is not None
        assert record.outcome.status.value == cancellation["terminal_status"]
        assert record.outcome.error is not None and "cancel" in record.outcome.error.lower()


def test_child_cancellation_does_not_cancel_parent(tmp_path: Path) -> None:
    cancellation = _contract()["cancellation"]
    llm, runtime, task, context, token, manager, _events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "task_description": "Inspect child cancellation",
            "wait_for_completion": False,
        },
    )

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    task_id = json.loads(parent_result.cycles[0].tool_results[0].content)["task_id"]
    llm.child_started.get(timeout=2)
    record = manager.get(task_id)

    assert record is not None and record.session is not None
    assert cast(Any, record.session).cancel() is True
    assert (not token.cancelled) is cancellation["child_does_not_cancel_parent"]
    llm.release.set()
    completed = manager.wait(task_id, timeout=3)

    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.FAILED
    assert (not token.cancelled) is cancellation["child_does_not_cancel_parent"]


def test_sub_task_manager_continuation_errors_match_shared_contract() -> None:
    errors = _contract()["manager"]["continuation_errors"]

    def assert_error(
        key: str,
        error_type: type[Exception],
        manager: SubTaskManager,
        *,
        task_id: str,
        prompt: str,
    ) -> None:
        with pytest.raises(error_type) as raised:
            manager.continue_task(task_id=task_id, prompt=prompt)
        assert raised.value.args[0] == errors[key].format(task_id=task_id)

    assert_error("empty_prompt", ValueError, _manager(), task_id="empty-prompt", prompt=" \t\n ")
    assert_error("not_found", KeyError, _manager(), task_id="missing-task", prompt="continue")

    detached_manager = _manager()
    detached_manager.record_outcome(
        "detached-task",
        SubTaskOutcome(
            task_id="detached-task",
            session_id="detached-session",
            agent_name="researcher",
            status=AgentStatus.COMPLETED,
        ),
    )
    assert_error(
        "session_not_attached",
        RuntimeError,
        detached_manager,
        task_id="detached-task",
        prompt="continue",
    )

    max_cycles_manager = _manager()
    _attach_continuable_manager_task(
        max_cycles_manager,
        task_id="max-cycles-task",
        session=_ManagerSession(lambda _prompt: _manager_result()),
        status=AgentStatus.MAX_CYCLES,
    )
    assert_error(
        "max_cycles",
        RuntimeError,
        max_cycles_manager,
        task_id="max-cycles-task",
        prompt="continue",
    )

    started = threading.Event()
    release = threading.Event()

    def block_continuation(_prompt: str) -> AgentResult:
        started.set()
        if not release.wait(timeout=3):
            raise TimeoutError("continuation error contract test was not released")
        return _manager_result()

    running_manager = _manager()
    _attach_continuable_manager_task(
        running_manager,
        task_id="running-task",
        session=_ManagerSession(block_continuation),
    )
    running_manager.continue_task(task_id="running-task", prompt="first continuation")
    assert started.wait(timeout=2)
    try:
        assert_error(
            "already_running",
            RuntimeError,
            running_manager,
            task_id="running-task",
            prompt="second continuation",
        )
    finally:
        release.set()
        running_manager.wait("running-task", timeout=3)


def test_sub_task_manager_continuation_admission_is_atomic() -> None:
    manager_contract = _contract()["manager"]
    errors = manager_contract["continuation_errors"]
    started = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    continuation_calls = 0

    def block_continuation(_prompt: str) -> AgentResult:
        nonlocal continuation_calls
        with call_lock:
            continuation_calls += 1
        started.set()
        if not release.wait(timeout=3):
            raise TimeoutError("atomic continuation test was not released")
        return _manager_result()

    manager = _manager()
    task_id = "atomic-continuation"
    _attach_continuable_manager_task(
        manager,
        task_id=task_id,
        session=_ManagerSession(block_continuation),
    )
    gate = threading.Barrier(3)
    results: queue.Queue[tuple[str, BaseException | None]] = queue.Queue()

    def continue_concurrently() -> None:
        gate.wait(timeout=2)
        try:
            manager.continue_task(task_id=task_id, prompt="continue once")
        except BaseException as exc:
            results.put(("error", exc))
        else:
            results.put(("accepted", None))

    callers = [threading.Thread(target=continue_concurrently) for _ in range(2)]
    for caller in callers:
        caller.start()
    gate.wait(timeout=2)
    for caller in callers:
        caller.join(timeout=2)

    try:
        outcomes = [results.get(timeout=2) for _ in callers]
        accepted = [outcome for outcome in outcomes if outcome[0] == "accepted"]
        rejected = [outcome[1] for outcome in outcomes if outcome[0] == "error"]
        admission_is_atomic = len(accepted) == 1 and len(rejected) == 1 and continuation_calls == 1
        assert admission_is_atomic is manager_contract["continuation_admission_atomic"]
        assert started.is_set()
        assert isinstance(rejected[0], RuntimeError)
        assert rejected[0].args[0] == errors["already_running"].format(task_id=task_id)
    finally:
        release.set()
        manager.wait(task_id, timeout=3)


@pytest.mark.parametrize("raises", [False, True], ids=["failed-outcome", "runner-exception"])
def test_sub_task_manager_failed_runs_use_shared_error_code(raises: bool) -> None:
    contract = _contract()
    expected_error_code = contract["continuation"]["failure_error_code"]
    assert expected_error_code == contract["lifecycle"]["failure_error_code_fallback"]
    manager = _manager()
    task_id = f"manager-failure-{raises}"

    def fail() -> SubTaskOutcome:
        if raises:
            raise RuntimeError("manager runner failed")
        return SubTaskOutcome(
            task_id=task_id,
            session_id=f"{task_id}-session",
            agent_name="researcher",
            status=AgentStatus.FAILED,
            error="manager run failed",
        )

    manager.submit(
        task_id=task_id,
        session_id=f"{task_id}-session",
        agent_name="researcher",
        task_title="fail directly",
        workspace_backend=MemoryWorkspaceBackend(),
        runner=fail,
    )
    record = manager.wait(task_id, timeout=3)

    assert record is not None and record.outcome is not None
    assert record.outcome.status == AgentStatus.FAILED
    assert record.outcome.error_code == expected_error_code


@pytest.mark.parametrize("raises", [False, True], ids=["failed-result", "continuation-exception"])
def test_sub_task_manager_failed_continuations_use_shared_error_code(raises: bool) -> None:
    expected_error_code = _contract()["continuation"]["failure_error_code"]
    task_id = f"continuation-failure-{raises}"

    def fail_continuation(_prompt: str) -> AgentResult:
        if raises:
            raise RuntimeError("manager continuation failed")
        return _manager_result(status=AgentStatus.FAILED, error="manager continuation failed")

    manager = _manager()
    _attach_continuable_manager_task(
        manager,
        task_id=task_id,
        session=_ManagerSession(fail_continuation),
    )
    manager.continue_task(task_id=task_id, prompt="fail the continuation")
    record = manager.wait(task_id, timeout=3)

    assert record is not None and record.outcome is not None
    assert record.outcome.status == AgentStatus.FAILED
    assert record.outcome.error == "manager continuation failed"
    assert record.outcome.error_code == expected_error_code


def test_manager_timeout_preserves_lineage_and_filtered_snapshot(tmp_path: Path) -> None:
    backend = MemoryWorkspaceBackend()
    backend.write_text("notes/readme.md", "notes")
    backend.write_text("src/main.py", "main")
    backend.write_text("generated/cache.bin", "cache")
    backend.write_text("logs/run.log", "log")
    llm, runtime, task, context, _token, manager, events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "task_description": "Inspect workspace",
            "exclude_files_pattern": _contract()["workspace_filter"]["pattern"],
            "wait_for_completion": False,
        },
        workspace_backend=backend,
    )

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    task_id = json.loads(parent_result.cycles[0].tool_results[0].content)["task_id"]
    llm.child_started.get(timeout=2)
    timed_out_record = manager.wait(task_id, timeout=0.01)

    assert timed_out_record is not None
    assert timed_out_record.is_running() is True
    manager_contract = _contract()["manager"]
    assert (timed_out_record.outcome is not None) is manager_contract["timeout_fabricates_terminal_state"]
    assert timed_out_record.parent_run_id == "parent-run"
    assert timed_out_record.parent_tool_call_id == "delegate"
    assert timed_out_record.workspace_backend is not None
    assert timed_out_record.workspace_backend.list_files(".", "**/*") == _contract()["workspace_filter"][
        "visible_paths"
    ]
    running_rejected = False
    with pytest.raises(RuntimeError, match="already running"):
        manager.continue_task(task_id=task_id, prompt="continue too early")
    running_rejected = True
    assert running_rejected is manager_contract["continue_running_rejected"]

    status_result = sub_task_status(
        ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=backend,
            sub_task_manager=manager,
            task_id="parent-task",
        ),
        {"task_ids": [task_id], "detail_level": "snapshot"},
    )
    snapshot = status_result.metadata["tasks"][0]["snapshot"]
    assert status_result.status_code == ToolResultStatus.SUCCESS
    assert ("snapshot" in status_result.metadata["tasks"][0]) is manager_contract[
        "snapshot_remains_queryable_after_timeout"
    ]
    assert snapshot["workspace_files"] == _contract()["workspace_filter"]["visible_paths"]
    assert status_result.metadata["tasks"][0]["status"] == AgentStatus.RUNNING.value

    llm.release.set()
    completed = manager.wait(task_id, timeout=3)

    assert completed is manager.get(task_id)
    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.COMPLETED
    pair = _sub_events(events)
    assert [event.type for event in pair] == ["sub_run_started", "sub_run_completed"]
    assert all(event.parent_run_id == "parent-run" for event in pair)


def test_manager_reports_session_not_ready_for_unattached_async_task(tmp_path: Path) -> None:
    manager = _manager()
    started = threading.Event()
    release = threading.Event()

    def run_unattached() -> SubTaskOutcome:
        started.set()
        release.wait(timeout=3)
        return SubTaskOutcome(
            task_id="unattached-task",
            session_id="unattached-session",
            agent_name="researcher",
            status=AgentStatus.COMPLETED,
        )

    manager.submit(
        task_id="unattached-task",
        session_id="unattached-session",
        agent_name="researcher",
        task_title="Waiting for session attachment",
        workspace_backend=MemoryWorkspaceBackend(),
        runner=run_unattached,
    )
    assert started.wait(timeout=2)
    try:
        result = sub_task_status(
            ToolContext(
                workspace=tmp_path,
                shared_state={},
                cycle_index=1,
                workspace_backend=MemoryWorkspaceBackend(),
                sub_task_manager=manager,
                task_id="parent-task",
            ),
            {"task_ids": ["unattached-task"], "message": "Focus on lineage"},
        )
    finally:
        release.set()
        manager.wait("unattached-task", timeout=3)

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == _contract()["manager"]["session_not_ready_error_code"]


@pytest.mark.parametrize("pattern", ["", "  \t\n  "])
def test_blank_workspace_filter_is_treated_as_absent(tmp_path: Path, pattern: str) -> None:
    backend = MemoryWorkspaceBackend()
    backend.write_text("notes/readme.md", "notes")
    backend.write_text("generated/cache.bin", "cache")
    llm, runtime, task, context, _token, manager, _events = _blocked_runtime(
        tmp_path,
        {
            "agent_id": "researcher",
            "task_description": "Inspect workspace",
            "exclude_files_pattern": pattern,
            "wait_for_completion": False,
        },
        workspace_backend=backend,
    )

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    task_id = json.loads(parent_result.cycles[0].tool_results[0].content)["task_id"]
    llm.child_started.get(timeout=2)
    record = manager.get(task_id)

    try:
        assert record is not None
        assert record.workspace_backend is backend
        assert set(record.workspace_backend.list_files(".", "**/*")) == {
            "generated/cache.bin",
            "notes/readme.md",
        }
    finally:
        llm.release.set()

    completed = manager.wait(task_id, timeout=3)
    assert completed is not None and completed.outcome is not None
    assert completed.outcome.status == AgentStatus.COMPLETED


def test_child_workspace_filter_hides_discovery_but_allows_known_path_reads(tmp_path: Path) -> None:
    backend = MemoryWorkspaceBackend()
    backend.write_text("notes/readme.md", "notes")
    backend.write_text("src/main.py", "main")
    backend.write_text("generated/cache.bin", "known cache content")
    backend.write_text("logs/run.log", "log")
    discovery_requests: list[LlmRequest] = []
    read_requests: list[LlmRequest] = []

    def read_hidden_file(request: LlmRequest) -> LLMResponse:
        discovery_requests.append(request)
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="read-hidden", name=READ_FILE_TOOL_NAME, arguments={"path": "generated/cache.bin"})],
        )

    def finish_after_read(request: LlmRequest) -> LLMResponse:
        read_requests.append(request)
        return _finish("child done", tool_call_id="child-finish")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                _delegate(
                    {
                        "agent_id": "researcher",
                        "task_description": "Inspect workspace",
                        "exclude_files_pattern": r"^(?:generated|logs)/",
                    }
                ),
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="discover",
                            name=FIND_FILES_TOOL_NAME,
                            arguments={"path": ".", "glob": "**/*", "sort": "path_asc"},
                        )
                    ],
                ),
                read_hidden_file,
                finish_after_read,
                _finish("parent done", tool_call_id="parent-finish"),
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        workspace_backend=backend,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=4,
            )
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    find_message = next(
        message
        for message in reversed(discovery_requests[0].messages)
        if message.role == "tool" and message.tool_call_id == "discover"
    )
    assert json.loads(find_message.content)["files"] == _contract()["workspace_filter"]["visible_paths"]
    read_message = next(
        message
        for message in reversed(read_requests[0].messages)
        if message.role == "tool" and message.tool_call_id == "read-hidden"
    )
    known_path_accessible = "known cache content" in read_message.content
    assert known_path_accessible is _contract()["workspace_filter"]["known_path_accessible"]
    assert (not known_path_accessible) is _contract()["workspace_filter"]["security_boundary"]


def test_configured_sub_agent_continuation_replays_complete_prior_turn(tmp_path: Path) -> None:
    continuation_requests: list[LlmRequest] = []
    continued_shared_state: list[dict[str, Any]] = []
    continued_run_ids: list[str] = []
    events: list[Any] = []

    def continue_child(request: LlmRequest) -> LLMResponse:
        continuation_requests.append(request)
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="inspect-state", name="inspect_state", arguments={})],
        )

    def finish_continuation(request: LlmRequest) -> LLMResponse:
        continuation_requests.append(request)
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="second-finish",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "second answer"},
                )
            ],
        )

    def fail_continuation(request: LlmRequest) -> LLMResponse:
        continuation_requests.append(request)
        raise RuntimeError("continuation failed")

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="delegate",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={"agent_id": "researcher", "task_description": "first prompt"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="remember", name="remember_state", arguments={})],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="first-finish",
                        name=TASK_FINISH_TOOL_NAME,
                        arguments={"message": "first answer"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="parent-finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
            continue_child,
            finish_continuation,
            fail_continuation,
        ]
    )
    manager = _manager()
    registry = build_default_registry()

    def remember_state(context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        context.shared_state["remembered"] = "from first run"
        return ToolExecutionResult(tool_call_id="", content="remembered")

    def inspect_state(context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        continued_shared_state.append(dict(context.shared_state))
        assert isinstance(context.run_context, RunContext)
        continued_run_ids.append(context.run_context.run_id)
        return ToolExecutionResult(tool_call_id="", content="inspected")

    registry.register_tool("remember_state", remember_state, "Remember state")
    registry.register_tool("inspect_state", inspect_state, "Inspect state")
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=registry,
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent",
        model="shared-model",
        system_prompt="Parent",
        user_prompt="Delegate",
        max_cycles=4,
        extra_tool_names=["remember_state", "inspect_state"],
        sub_agents={"researcher": SubAgentConfig(model="shared-model", description="Research")},
    )

    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )

    result = runtime.run(task, ctx=context, sub_task_manager=manager)
    payload = json.loads(result.cycles[0].tool_results[0].content)
    task_id = payload["task_id"]
    first_record = manager.get(task_id)
    assert first_record is not None and first_record.session is not None
    assert cast(Any, first_record.session).shared_state["remembered"] == "from first run"
    manager.continue_task(task_id=task_id, prompt="second prompt")
    record = manager.wait(task_id, timeout=2)

    assert record is not None
    assert record.outcome is not None
    assert record.outcome.status == AgentStatus.COMPLETED
    assert continuation_requests
    messages = continuation_requests[0].messages
    contents = [message.content for message in messages]
    required_history = _contract()["continuation"]["required_history"]
    positions = [next(index for index, content in enumerate(contents) if required in content) for required in required_history]
    assert positions == sorted(positions)
    assistant_tool_call_ids = {
        tool_call["id"]
        for message in messages
        if message.role == "assistant"
        for tool_call in (message.tool_calls or [])
    }
    tool_result_ids = {message.tool_call_id for message in messages if message.role == "tool"}
    complete_tool_turn = bool(assistant_tool_call_ids) and assistant_tool_call_ids <= tool_result_ids
    assert complete_tool_turn is _contract()["continuation"]["include_complete_tool_turn"]
    state_preserved = bool(continued_shared_state) and continued_shared_state[0].get("remembered") == "from first run"
    assert state_preserved is _contract()["continuation"]["preserve_shared_state"]
    assert record.session is not None
    assert cast(Any, record.session).shared_state["remembered"] == "from first run"
    manager.continue_task(task_id=task_id, prompt="third prompt")
    failed_record = manager.wait(task_id, timeout=2)

    assert failed_record is not None and failed_record.outcome is not None
    assert failed_record.outcome.status == AgentStatus.FAILED
    assert failed_record.outcome.error is not None
    assert "continuation failed" in failed_record.outcome.error
    lifecycle = _sub_events(events)
    assert [event.type for event in lifecycle] == [
        "sub_run_started",
        "sub_run_completed",
        "sub_run_started",
        "sub_run_completed",
        "sub_run_started",
        "sub_run_completed",
    ]
    assert lifecycle[0].run_id == lifecycle[1].run_id
    assert lifecycle[2].run_id == lifecycle[3].run_id
    assert lifecycle[4].run_id == lifecycle[5].run_id
    assert len({lifecycle[0].run_id, lifecycle[2].run_id, lifecycle[4].run_id}) == 3
    assert continuation_requests[0].messages[0].metadata["_vv_agent_run_id"] == lifecycle[2].run_id
    assert continuation_requests[1].messages[0].metadata["_vv_agent_run_id"] == lifecycle[2].run_id
    assert continuation_requests[2].messages[0].metadata["_vv_agent_run_id"] == lifecycle[4].run_id
    assert continuation_requests[0].messages[0].metadata["_vv_agent_run_id"] != lifecycle[0].run_id
    assert continuation_requests[2].messages[0].metadata["_vv_agent_run_id"] != lifecycle[2].run_id
    assert continued_run_ids == [lifecycle[2].run_id]
    failed_completed = lifecycle[5]
    assert isinstance(failed_completed, SubRunCompletedEvent)
    assert failed_completed.status == AgentStatus.FAILED.value
    assert failed_completed.error is not None and "continuation failed" in failed_completed.error
    assert all(event.trace_id == "trace-parity" for event in lifecycle)
    assert all(event.parent_run_id == "parent-run" for event in lifecycle)
    assert all(event.parent_tool_call_id == "delegate" for event in lifecycle)
    assert all(event.session_id == payload["session_id"] for event in lifecycle)
    assert all(event.task_id == task_id for event in lifecycle)


class _ContinuationCancellationLLM:
    def __init__(self) -> None:
        self.continuation_started: queue.Queue[None] = queue.Queue()
        self.release = threading.Event()
        self._child_calls = 0
        self._parent_calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Any = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, tools, stream_callback, model_settings, request_metadata
        is_child = bool(
            messages
            and messages[0].role == "system"
            and messages[0].content == "Child prompt"
        )
        with self._lock:
            if is_child:
                self._child_calls += 1
                call = self._child_calls
            else:
                self._parent_calls += 1
                call = self._parent_calls

        if is_child and call == 1:
            return _finish("initial child done", tool_call_id="initial-child-finish")
        if is_child:
            self.continuation_started.put(None)
            if not self.release.wait(timeout=5):
                raise TimeoutError("continuation cancellation test was not released")
            return _finish("continuation should be cancelled", tool_call_id="continued-child-finish")
        if call == 1:
            return _delegate({"agent_id": "researcher", "task_description": "initial prompt"})
        return _finish("parent done", tool_call_id="parent-finish")


def test_parent_cancellation_reaches_configured_sub_agent_continuation(tmp_path: Path) -> None:
    cancellation = _assert_cancellation_contract("continuation")
    llm = _ContinuationCancellationLLM()
    token = CancellationToken()
    events: list[Any] = []
    manager = _manager()
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
            )
        },
    )
    context = ExecutionContext(
        cancellation_token=token,
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        },
    )

    result = runtime.run(task, ctx=context, sub_task_manager=manager)
    payload = json.loads(result.cycles[0].tool_results[0].content)
    task_id = payload["task_id"]
    manager.continue_task(task_id=task_id, prompt="continue and block")
    llm.continuation_started.get(timeout=2)

    token.cancel()
    llm.release.set()
    record = manager.wait(task_id, timeout=3)

    assert record is not None and record.outcome is not None
    assert record.outcome.status.value == cancellation["terminal_status"]
    assert record.outcome.error is not None and "cancel" in record.outcome.error.lower()
    lifecycle = _sub_events(events)
    assert [event.type for event in lifecycle] == [
        "sub_run_started",
        "sub_run_completed",
        "sub_run_started",
        "sub_run_completed",
    ]
    assert lifecycle[0].run_id == lifecycle[1].run_id
    assert lifecycle[2].run_id == lifecycle[3].run_id
    assert lifecycle[0].run_id != lifecycle[2].run_id
    assert lifecycle[3].status == AgentStatus.FAILED.value
    assert _lifecycle_is_fully_paired(events) is _contract()["lifecycle"]["completed_after_every_started"]


class _ConfiguredChildPanic(BaseException):
    pass


class _PanicThenRecoverConfiguredLLM:
    def __init__(self) -> None:
        self._child_calls = 0
        self._parent_calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Any = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, tools, stream_callback, model_settings, request_metadata
        is_child = bool(
            messages
            and messages[0].role == "system"
            and messages[0].content == "Child prompt"
        )
        with self._lock:
            if is_child:
                self._child_calls += 1
                call = self._child_calls
            else:
                self._parent_calls += 1
                call = self._parent_calls

        if is_child and call == 1:
            raise _ConfiguredChildPanic("configured child panicked")
        if is_child:
            return _finish("child recovered", tool_call_id="recovered-child-finish")
        if call == 1:
            return _delegate(
                {
                    "agent_id": "researcher",
                    "task_description": "panic once",
                    "wait_for_completion": False,
                }
            )
        return _finish("parent done", tool_call_id="parent-finish")


def test_configured_async_child_panic_cleans_up_and_retained_session_can_continue(
    tmp_path: Path,
) -> None:
    cleanup_contract = _contract()["lifecycle"]["panic_or_exception_cleanup"]
    events: list[Any] = []
    manager = SubTaskManager(
        register_session=register_sub_agent_session,
        unregister_session=unregister_sub_agent_session,
    )
    runtime = AgentRuntime(
        llm_client=_PanicThenRecoverConfiguredLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent-task",
        model="shared-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        sub_agents={
            "researcher": SubAgentConfig(
                model="shared-model",
                description="Research",
                system_prompt="Child prompt",
            )
        },
    )
    context = ExecutionContext(
        metadata={
            "_vv_agent_run_id": "parent-run",
            "_vv_agent_trace_id": "trace-parity",
            "_vv_agent_emit_event": events.append,
        }
    )

    parent_result = runtime.run(task, ctx=context, sub_task_manager=manager)
    payload = json.loads(parent_result.cycles[0].tool_results[0].content)
    task_id = payload["task_id"]
    session_id = payload["session_id"]
    failed_record = manager.wait(task_id, timeout=3)

    assert failed_record is not None and failed_record.outcome is not None
    assert failed_record.outcome.status == AgentStatus.FAILED
    assert failed_record.outcome.error_code == "sub_task_failed"
    assert failed_record.outcome.error == "configured child panicked"
    assert (not failed_record.is_running()) is cleanup_contract["active_state_cleared"]
    assert (get_sub_agent_session(session_id=session_id) is None) is cleanup_contract["global_session_unregistered"]
    failed_lifecycle = _sub_events(events)
    assert [event.type for event in failed_lifecycle] == ["sub_run_started", "sub_run_completed"]
    assert (len(failed_lifecycle) == 2) is cleanup_contract["completed_event_emitted_once"]
    assert failed_lifecycle[0].run_id == failed_lifecycle[1].run_id
    assert failed_lifecycle[1].status == AgentStatus.FAILED.value

    manager.continue_task(task_id=task_id, prompt="recover now")
    recovered_record = manager.wait(task_id, timeout=3)

    assert recovered_record is not None and recovered_record.outcome is not None
    assert (recovered_record.outcome.status == AgentStatus.COMPLETED) is cleanup_contract[
        "retained_session_can_continue"
    ]
    assert recovered_record.outcome.final_answer == "child recovered"
    assert recovered_record.task_id == task_id
    assert recovered_record.session_id == session_id
    assert recovered_record.agent_name == "researcher"
    assert not recovered_record.is_running()
    assert get_sub_agent_session(session_id=session_id) is None
    lifecycle = _sub_events(events)
    assert [event.type for event in lifecycle] == [
        "sub_run_started",
        "sub_run_completed",
        "sub_run_started",
        "sub_run_completed",
    ]
    assert lifecycle[2].run_id == lifecycle[3].run_id
    assert lifecycle[0].run_id != lifecycle[2].run_id
    assert all(event.task_id == task_id for event in lifecycle)
    assert all(event.session_id == session_id for event in lifecycle)
    assert all(event.agent_name == "researcher" for event in lifecycle)
    assert lifecycle[3].status == AgentStatus.COMPLETED.value
