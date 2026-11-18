"""SDK definitions and handles over child records, without another execution ledger."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from vv_agent.agent import Agent
from vv_agent.background_task import BackgroundAgentTask, BackgroundAgentTaskSnapshot
from vv_agent.budget import RunBudgetLimits
from vv_agent.config import project_resolved_model_limits
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, SUB_TASK_STATUS_TOOL_NAME
from vv_agent.kernel._child_delegation import error as error
from vv_agent.kernel._child_delegation import success as success
from vv_agent.kernel._child_store import (
    _store_tenant_id,
    ordinary_result_for_children,
)
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings
from vv_agent.prompt.builder import build_raw_system_prompt_bundle, build_system_prompt_bundle
from vv_agent.run_config import RunConfig, ToolPolicy, merge_tool_policies
from vv_agent.tools.base import ToolContext, ToolSpec
from vv_agent.tools.executor import RegistryToolExecutor
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.tools.handlers.sub_task_status import (
    SubTaskStatusRequest,
    parse_sub_task_status,
    snapshot_sub_task_status,
    sub_task_status,
)
from vv_agent.tools.handlers.sub_task_status import _add_wait_metadata as add_wait_metadata
from vv_agent.tools.handlers.sub_task_status import _error as status_error
from vv_agent.tools.handlers.sub_task_status import _running_task_ids as running_task_ids
from vv_agent.tools.metadata import normalize_denied_side_effects, normalize_metadata_labels
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import (
    AgentStatus,
    AgentTask,
    CompletionReason,
    SubAgentConfig,
    SubTaskOutcome,
    SubTaskRequest,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)
from vv_agent.workspace import DiscoveryFilteredWorkspaceBackend
from vv_agent.workspace.local import LocalWorkspaceBackend

from .children import ChildSession
from .content_runtime import (
    INPUT_ACTOR_ATTRIBUTES,
    INPUT_CONTENT,
    SessionContentWriter,
    durable_content_reader,
)
from .projection import SessionHydration, hydrate_records
from .providers import Accepted, Definitive, Outcome, Unknown
from .records import CHILD_ADMISSION, InboxItem, Record, copy_json, validate
from .reducer import ExecutionState
from .result import project_result
from .store import Conflict, SessionStore, StoredRecord

if TYPE_CHECKING:
    from vv_agent.result import RunResult

    from .runtime import Runtime


def _hydrated_records(store: SessionStore, sid: str, records):
    hydration = SessionHydration.from_store(store, tenant_id=_store_tenant_id(store), session_id=sid)
    return hydrate_records(records, hydration)


def _created_record(store: SessionStore, sid: str) -> Record:
    stored = store.read(sid, limit=1).records[0]
    return _hydrated_records(store, sid, (stored,))[0].record


def register_adapters(runtime: Runtime, registry: ToolRegistry) -> None:
    if registry.has_executor("sub_task_status"):
        executor = registry.get_executor("sub_task_status")
        if isinstance(executor, RegistryToolExecutor) and executor.handler is sub_task_status:
            executor.handler = child_status
    if SUB_TASK_STATUS_TOOL_NAME not in runtime.providers:
        runtime.providers[SUB_TASK_STATUS_TOOL_NAME] = SubTaskStatusProvider(runtime)
    register_handoffs(runtime, registry)


def child_status(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    manager = cast(ChildTasks, context.sub_task_manager)
    # Tool providers run on a separate thread from the parent's driver.
    with manager.runtime.heartbeat_store() as store:
        tasks = ChildTasks(store, manager.parent_id, manager.runtime)
        return status_with_store(replace(context, sub_task_manager=tasks), arguments)


def status_with_store(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    message = arguments.get("message")
    if not isinstance(message, str) or not message.strip():
        return sub_task_status(context, arguments)
    validation = arguments | {"message": ""}
    for name in ("wait_for_response", "wait_for_completion"):
        if isinstance(validation.get(name, False), bool):
            validation[name] = False
    result = sub_task_status(context, validation)
    if result.status_code == ToolResultStatus.ERROR:
        return result
    manager = cast(ChildTasks, context.sub_task_manager)
    target = result.metadata["tasks"][0]["task_id"]
    record = manager.get(target)
    if record is None:
        return status_error(f"Sub-task {target} not found.", error_code="sub_task_not_found", details={"task_id": target})
    if record.outcome.status == AgentStatus.MAX_CYCLES:
        return error("sub_task_max_cycles_reached", f"Sub-task {target} reached max cycles and cannot continue.")
    action = manager.message(target, f"message/{context.idempotency_key}", message.strip())
    if arguments.get("wait_for_response", False):
        manager.wait(target)
    result = sub_task_status(context, arguments | {"message": "", "wait_for_response": False})
    result.metadata["interaction"] = {"task_id": target, "action": action, "previous_status": record.outcome.status.value}
    result.content = json.dumps(result.metadata, ensure_ascii=False)
    return result


def register_handoffs(runtime: Runtime, registry: ToolRegistry) -> None:
    for transfer in runtime.agent.handoffs:
        assert transfer.tool_name is not None
        registry.register_schema(
            transfer.tool_name,
            {
                "type": "function",
                "function": {
                    "name": transfer.tool_name,
                    "description": transfer.description or f"Transfer to {transfer.agent.name}.",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        registry.register(
            ToolSpec(
                transfer.tool_name,
                lambda _ctx, _args: error("session_handoff_adapter_required", "Handoff requires child admission."),
            )
        )


def configured_agent(runtime: Runtime, name: str, sub: SubAgentConfig) -> Agent:
    return _configured_agent_for(runtime.agent, name, sub)


def _configured_agent_for(parent: Agent, name: str, sub: SubAgentConfig) -> Agent:
    # Handlers remain host bindings; policy and prompt come from the admitted definition.
    return Agent(
        name,
        sub.system_prompt or sub.description or "Complete the delegated task.",
        model=sub.model,
        metadata=copy_json(parent.metadata),
        tools=[
            tool
            for tool in parent.tools
            if not isinstance(tool, FunctionTool) or tool.metadata.get("mode") not in {"agent_as_tool", "background_task"}
        ],
        max_cycles=max(1, sub.max_cycles),
    )


def target_agent(runtime: Runtime, selector: str, mode: str, sub_config=None) -> Agent:
    if mode == "configured":
        assert sub_config is not None
        return configured_agent(runtime, selector, SubAgentConfig.from_dict(sub_config))
    if mode == "handoff":
        target = next((t.agent for t in runtime.agent.handoffs if t.tool_name == selector), None)
    else:
        executor = runtime.registry.get_executor(selector) if runtime.registry.has_executor(selector) else None
        target = executor.metadata.get("agent") if executor else None
    if not isinstance(target, Agent):
        raise ValueError(f"Missing child agent host binding: {selector}")
    return target


def child_config(runtime: Runtime, context: ToolContext, agent: Agent, mode: str, sub: SubAgentConfig | None = None):
    return _child_config(
        runtime.config,
        mode,
        sub,
        shared_state=runtime.durable_state(context.shared_state),
        workspace=context.workspace,
        workspace_backend=context.workspace_backend,
    )


def _child_config(
    config: RunConfig,
    mode: str,
    sub: SubAgentConfig | None,
    *,
    shared_state: dict[str, Any] | None,
    workspace: str | Path | None,
    workspace_backend: Any | None,
) -> RunConfig:
    config = replace(
        config,
        model=None,
        model_settings=None,
        metadata={**copy_json(config.metadata), **(copy_json(sub.metadata) if sub else {})},
        stream=None,
        shared_state=shared_state,
        initial_messages=None,
        before_cycle_messages=None,
        interruption_messages=None,
        cancellation_token=None,
        host_cost_meter=None,
        workspace=workspace,
        workspace_backend=workspace_backend,
    )
    if sub:
        policy = merge_tool_policies(
            config.tool_policy,
            ToolPolicy(
                disallowed_tools=[*sub.exclude_tools, "create_sub_task", "sub_task_status"],
                denied_side_effects=sub.denied_side_effects,
                denied_capability_tags=sub.denied_capability_tags,
                denied_cost_dimensions=sub.denied_cost_dimensions,
                deny_terminal_tools=sub.deny_terminal_tools,
            ),
        )
        config = replace(
            config,
            model=ModelRef.backend(sub.backend, sub.model) if sub.backend else ModelRef.named(sub.model),
            tool_policy=policy,
            max_cycles=max(sub.max_cycles, 1),
            no_tool_policy="finish",
            session_memory_enabled=sub.session_memory_enabled,
            shared_state={},
        )
    elif mode == "handoff":
        config = replace(config, max_cycles=None, no_tool_policy=None)
    return config


@dataclass(frozen=True, slots=True)
class _StaticChildDeclaration:
    agent: Agent
    config: RunConfig


def _static_child_declarations(parent: Agent, config: RunConfig) -> Iterator[_StaticChildDeclaration]:
    """Return child agents and their effective trusted configuration without a Runtime."""

    for name, sub in parent.sub_agents.items():
        agent = _configured_agent_for(parent, name, sub)
        yield _StaticChildDeclaration(
            agent,
            _child_config(
                config,
                "configured",
                sub,
                shared_state=None,
                workspace=None,
                workspace_backend=None,
            ),
        )
    for transfer in parent.handoffs:
        yield _StaticChildDeclaration(
            transfer.agent,
            _child_config(
                config,
                "handoff",
                None,
                shared_state=None,
                workspace=None,
                workspace_backend=None,
            ),
        )
    for tool in parent.tools:
        metadata = tool.metadata if isinstance(tool.metadata, Mapping) else {}
        agent = metadata.get("agent")
        if isinstance(agent, Agent):
            mode = metadata.get("mode")
            yield _StaticChildDeclaration(
                agent,
                _child_config(
                    config,
                    mode if isinstance(mode, str) else "agent_as_tool",
                    None,
                    shared_state=None,
                    workspace=None,
                    workspace_backend=None,
                ),
            )


def assemble(
    runtime: Runtime, plan: Record, context: ToolContext, task: AgentTask, store: SessionStore
) -> list[ChildSession] | ToolExecutionResult | None:
    from vv_agent.kernel._child_delegation import assemble as assemble_steps

    from ._dispatch_delegation import ChildCompilationHost

    attributes = _created_record(store, plan.session_id).payload["attributes"]
    return ChildCompilationHost(runtime, context, session_attributes=attributes).run(assemble_steps(store, plan, task))


def reconstruct(runtime: Runtime, store: SessionStore, sid: str) -> Runtime:
    created = _created_record(store, sid)
    attributes = created.payload["attributes"]
    admission = attributes["child_admission"]
    validate(admission, CHILD_ADMISSION)
    if admission["handler_version"] != runtime.handler_version:
        raise Conflict("child admission handler version mismatch")
    agent = target_agent(runtime, admission["selector"], admission["mode"], admission["sub_config"])
    task = AgentTask.from_dict(admission["definition"]["task"])
    # Reapply current host restrictions; the task keeps the frozen restrictions as well.
    context = SimpleNamespace(
        shared_state=runtime.bind_state(task.initial_shared_state, task),
        workspace=Path(created.payload["workspace"]),
        workspace_backend=runtime.config.workspace_backend,
    )
    if context.workspace_backend is None:
        context.workspace_backend = LocalWorkspaceBackend(context.workspace)
    sub = SubAgentConfig.from_dict(admission["sub_config"]) if admission["sub_config"] else None
    config = child_config(runtime, cast(ToolContext, context), agent, admission["mode"], sub)
    if admission["exclude_files_pattern"]:
        config = replace(
            config,
            workspace_backend=DiscoveryFilteredWorkspaceBackend(context.workspace_backend, admission["exclude_files_pattern"]),
        )
    config = replace(config, budget_limits=RunBudgetLimits.from_dict(admission["budget"]))
    child_rt = runtime.for_agent(agent, config)
    child_rt.content_reader = durable_content_reader(
        store,
        tenant_id=_store_tenant_id(store),
        session_id=sid,
    )
    child_rt.frozen_task = task
    return child_rt


def outcome(
    store: SessionStore,
    sid: str,
    runtime: Runtime,
    turn_id: str | None = None,
    *,
    snapshot: tuple[ExecutionState, tuple[StoredRecord, ...], int] | None = None,
) -> SubTaskOutcome:
    snapshot = snapshot if snapshot is not None else store.read_state(sid)
    state, rows, _ = snapshot
    created = _hydrated_records(store, sid, rows[:1])[0].record
    attributes = created.payload["attributes"]
    turn_id = (
        turn_id
        or state.active_turn_id
        or next(
            reversed(state.turns),
            attributes["child_handle"]["turn_id"],
        )
    )
    admission = attributes.get("child_admission")
    name = admission["definition"]["agent_name"] if admission else runtime.agent.name
    if turn_id not in state.turns:
        return SubTaskOutcome(sid, name, AgentStatus.RUNNING, session_id=sid)
    result = project_result(store, sid, turn_id, runtime=runtime, snapshot=snapshot)
    return SubTaskOutcome(
        sid,
        name,
        result.status,
        session_id=sid,
        final_answer=result.raw_result.final_answer,
        wait_reason=result.wait_reason,
        error=result.raw_result.error.get("message") if result.raw_result.error else None,
        error_code=("sub_task_failed" if result.status == AgentStatus.FAILED else None)
        if admission and admission["mode"] == "configured"
        else result.raw_result.error_code,
        cycles=len(result.raw_result.cycles),
        todo_list=result.raw_result.shared_state.get("todo_list", []),
        resolved={
            "backend": runtime.resolved.backend,
            "selected_model": runtime.resolved.selected_model,
            "model_id": runtime.resolved.model_id,
        },
        completion_reason=result.completion_reason,
        completion_tool_name=result.completion_tool_name,
        partial_output=result.partial_output,
    )


def result_for_children(
    store: SessionStore, plan: Record, handles: list[dict[str, Any]], runtime: Runtime
) -> ToolExecutionResult:
    first = store.read_state(handles[0]["session_id"])
    admission = (
        _hydrated_records(store, handles[0]["session_id"], first[1][:1])[0].record.payload["attributes"].get("child_admission")
    )
    if admission is None:
        return ordinary_result_for_children(store, plan, handles, first_snapshot=first)
    mode = admission["mode"]
    if mode == "configured":
        outcomes = []
        for i, h in enumerate(handles):
            snapshot = first if i == 0 else store.read_state(h["session_id"])
            rt = reconstruct(runtime, store, h["session_id"])
            outcomes.append(outcome(store, h["session_id"], rt, h["turn_id"], snapshot=snapshot))
        values = iter(outcomes)
        ctx = SimpleNamespace(
            sub_task_runner=lambda _request: next(values), run_context=None, ctx=None, tool_call_id="", workspace_backend=None
        )
        result = create_sub_task(
            cast(ToolContext, ctx), copy_json(plan._payload["request"]["arguments"]) | {"wait_for_completion": True}
        )
    else:
        h = handles[0]
        rt = reconstruct(runtime, store, h["session_id"])
        projected = project_result(store, h["session_id"], h["turn_id"], runtime=rt, snapshot=first)
        result = ToolExecutionResult(
            "",
            projected.final_output or "",
            metadata={
                "agent": admission["definition"]["agent_name"],
                "mode": mode,
                "child_status": projected.status.value,
                "child_run_id": h["session_id"],
            },
        )
        if mode == "handoff":
            result.directive = ToolDirective.FINISH
            if projected.status != AgentStatus.COMPLETED:
                result.status_code = ToolResultStatus.ERROR
                result.error_code = projected.raw_result.error_code or "handoff_failed"
                result.content = (
                    (projected.raw_result.error.get("message") if projected.raw_result.error else None)
                    or projected.wait_reason
                    or "handoff failed"
                )
    result.tool_call_id = plan._payload["request"]["id"]
    return result


def admitted_result(store: SessionStore, plan: Record, handles: list[dict[str, Any]]) -> ToolExecutionResult:
    created = _created_record(store, handles[0]["session_id"])
    admission = created.payload["attributes"].get("child_admission")
    if not admission:
        if len(handles) > 1:
            return ToolExecutionResult(plan._payload["request"]["id"], json.dumps(handles), metadata={"children": handles})
        return ToolExecutionResult(plan._payload["request"]["id"], json.dumps(handles[0]), metadata={"child": handles[0]})
    name = admission["definition"]["agent_name"]
    if admission["mode"] == "background_task":
        payload = {
            "task_id": handles[0]["session_id"],
            "agent_name": name,
            "status": "running",
            "final_output": None,
            "error": None,
        }
    elif "tasks" in plan._payload["request"]["arguments"]:
        entries = [
            {
                "index": i,
                "task_id": h["session_id"],
                "session_id": h["session_id"],
                "agent_name": name,
                "status": "running",
                "task_description": plan._payload["request"]["arguments"]["tasks"][i]["task_description"].strip(),
            }
            for i, h in enumerate(handles)
        ]
        payload = {
            "summary": {"total": len(handles), "accepted": len(handles), "failed": 0},
            "task_ids": [h["session_id"] for h in handles],
            "results": entries,
            "wait_for_completion": False,
        }
    else:
        sid = handles[0]["session_id"]
        payload = {
            "task_id": sid,
            "session_id": sid,
            "agent_name": name,
            "status": "running",
            "task_description": plan._payload["request"]["arguments"]["task_description"].strip(),
            "wait_for_completion": False,
        }
    result = success(payload)
    result.tool_call_id = plan._payload["request"]["id"]
    return result


@dataclass
class ChildTasks:
    store: SessionStore
    parent_id: str
    runtime: Runtime

    def get(self, task_id: str):
        try:
            created = self.store.read(task_id, limit=1).records[0].record
        except (KeyError, IndexError, Conflict):
            return None
        if created._payload["parent_session_id"] != self.parent_id:
            return None
        rt = reconstruct(self.runtime, self.store, task_id)
        # Read pending inputs before the log so admission cannot look like an old terminal.
        pending = self.store.peek_inbox(task_id)
        snapshot = self.store.read_state(task_id)
        state, _, _ = snapshot
        value = outcome(self.store, task_id, rt, snapshot=snapshot)
        if state.active_turn_id is not None and state.phase == "active":
            value = replace(value, status=AgentStatus.RUNNING)
        if state.active_turn_id is None and (
            any(item.item.kind == "follow_up" and item.item.input_id not in state.admitted_inputs for item in pending)
            or any(
                r._payload["disposition"] == "queued" and i not in state.admitted_inputs for i, r in state.applied_inputs.items()
            )
        ):
            value = replace(value, status=AgentStatus.RUNNING)
        attributes = _created_record(self.store, task_id).payload["attributes"]
        task = AgentTask.from_dict(attributes["child_admission"]["definition"]["task"])
        latest_turn = state.turns[next(reversed(state.turns))] if state.turns else None
        task_title = task.user_prompt
        if latest_turn is not None:
            stored_start = next(row for row in snapshot[1] if row.record.record_id == latest_turn.start.record_id)
            hydrated_start = _hydrated_records(self.store, task_id, (stored_start,))[0].record
            task_title = AgentTask.from_dict(hydrated_start.payload["definition"]["task"]).user_prompt
        return SimpleNamespace(
            task_id=task_id,
            session_id=task_id,
            agent_name=value.agent_name,
            task_title=task_title,
            parent_run_id=task.metadata.get("parent_run_id"),
            parent_tool_call_id=task.metadata.get("parent_tool_call_id"),
            outcome=value,
            is_running=lambda: value.status in {AgentStatus.PENDING, AgentStatus.RUNNING},
            current_cycle_index=value.cycles or None,
            latest_cycle={"cycle_index": value.cycles, "status": value.status.value} if value.cycles else None,
            workspace_backend=rt.config.workspace_backend,
        )

    def wait(self, task_id: str, timeout: float | None = None):
        deadline = None if timeout is None else self.runtime.clock.monotonic() + timeout
        while True:
            record = self.get(task_id)
            if record is None or not record.is_running() or (deadline is not None and self.runtime.clock.monotonic() >= deadline):
                return record
            self.runtime.clock.sleep(0.01)

    def handle(self, task_id: str) -> ChildTaskHandle:
        if self.get(task_id) is None:
            raise KeyError(f"Unknown background agent task: {task_id}")
        return ChildTaskHandle(self, task_id)

    def control(self, task_id: str, input_id: str, kind: str, payload: dict[str, Any]) -> None:
        if self.get(task_id) is None:
            raise KeyError(task_id)
        attributes = _created_record(self.store, task_id).payload["attributes"]
        h = attributes["child_handle"]
        state, _, _ = self.store.read_state(task_id)
        turn_id = state.active_turn_id or h["turn_id"]
        generation = state.turns[turn_id].start._payload["generation"] if turn_id in state.turns else h["generation"]
        with self.store.atomic() as tx:
            writer = SessionContentWriter.bind(
                tx,
                tenant_id=_store_tenant_id(self.store),
                session_id=task_id,
            )
            actor, actor_attributes = writer.service_principal("child-tasks")
            tx.push(
                task_id,
                InboxItem(
                    input_id,
                    kind,
                    payload,
                    turn_id,
                    generation,
                    actor=actor,
                ),
            )
            writer.link_input(input_id, actor_attributes, INPUT_ACTOR_ATTRIBUTES)

    def message(self, task_id: str, input_id: str, text: str) -> str:
        if self.get(task_id) is None:
            raise KeyError(task_id)
        state, _, _ = self.store.read_state(task_id)
        applied = state.applied_inputs.get(input_id)
        pending = next((i.item for i in self.store.peek_inbox(task_id) if i.item.input_id == input_id), None)
        previous = InboxItem(**applied._payload["input"]) if applied else pending
        target = state.active_turn_id
        generation = state.turns[target].start._payload["generation"] if target else None
        action = "message_queued" if target else "continued"
        kind = ("steer", "follow_up")[target is None]
        with self.store.atomic() as tx:
            writer = SessionContentWriter.bind(
                tx,
                tenant_id=_store_tenant_id(self.store),
                session_id=task_id,
            )
            content = writer.put_json(text)
            actor, actor_attributes = writer.service_principal("child-tasks")
            tx.push(
                task_id,
                InboxItem(
                    input_id,
                    kind,
                    {"content": content.to_dict()},
                    target,
                    generation,
                    actor=actor,
                ),
            )
            writer.link_input(input_id, content, INPUT_CONTENT)
            writer.link_input(input_id, actor_attributes, INPUT_ACTOR_ATTRIBUTES)
        if previous is not None:
            return "continued" if previous.kind == "follow_up" else "message_queued"
        self.runtime.wake(task_id)
        self.runtime.hook("after_input", None)
        return action


class SubTaskStatusProvider:
    """Accepted provider whose query observes child SQL without owning a second ledger."""

    def __init__(self, runtime: Runtime, *, confirmed_poll: Callable[[], int] | None = None):
        self.runtime = runtime
        self.confirmed_poll = confirmed_poll

    def bind_confirmed_poll(self, confirmed_poll: Callable[[], int]) -> SubTaskStatusProvider:
        return SubTaskStatusProvider(self.runtime, confirmed_poll=confirmed_poll)

    def _now_ms(self) -> int:
        if self.confirmed_poll is None:
            raise RuntimeError("sub_task_status provider requires a confirmed lease snapshot")
        now_ms = self.confirmed_poll()
        if isinstance(now_ms, bool) or not isinstance(now_ms, int):
            raise TypeError("confirmed lease snapshot must expose integer database time")
        return now_ms

    @staticmethod
    def _result(plan: Record, result: ToolExecutionResult, evidence: str) -> Definitive:
        result.tool_call_id = plan._payload["request"]["id"]
        return Definitive(result.to_dict(), (evidence,))

    def _handle(self, plan: Record, state: dict[str, Any]) -> dict[str, Any]:
        operation_id = plan.operation_id
        attempt = plan.attempt
        assert operation_id is not None and attempt is not None
        evidence = f"sub-task-status/{operation_id}/{attempt}"
        return {
            "kind": "provider",
            "provider": plan._payload["provider_binding"],
            "job_id": evidence,
            "operation_id": operation_id,
            "attempt": attempt,
            "request_digest": plan._payload["request_digest"],
            "evidence": evidence,
            "query_ref": state,
            "cancel_ref": state,
        }

    @staticmethod
    def _update_message_result(
        result: ToolExecutionResult,
        *,
        target: str,
        action: str,
        previous_status: str,
        wait_for_completion: bool,
        check_interval_seconds: int,
        max_wait_seconds: int,
    ) -> None:
        tasks = result.metadata["tasks"]
        for entry in tasks:
            if entry["task_id"] == target:
                entry["status"] = AgentStatus.RUNNING.value
                for key in ("final_answer", "wait_reason", "error", "error_code", "cycles", "todo_list", "resolved"):
                    entry.pop(key, None)
                break
        currently_running = running_task_ids(tasks)
        add_wait_metadata(
            result.metadata,
            wait_for_completion=wait_for_completion,
            check_interval_seconds=check_interval_seconds,
            max_wait_seconds=max_wait_seconds,
            running_task_ids=currently_running,
            wait_exceeded=False,
        )
        result.metadata["interaction"] = {"task_id": target, "action": action, "previous_status": previous_status}
        result.content = json.dumps(result.metadata, ensure_ascii=False)

    def submit(self, plan: Record, *, context: ToolContext) -> Outcome:
        manager = context.sub_task_manager
        if manager is None:
            error = status_error("Sub-task manager is not available for this task", error_code="sub_task_manager_unavailable")
            return self._result(plan, error, f"sub-task-status/{plan.operation_id}/{plan.attempt}")
        started_ms = self._now_ms()
        with self.runtime.heartbeat_store() as store:
            tasks = ChildTasks(store, manager.parent_id, self.runtime)
            request = parse_sub_task_status(replace(context, sub_task_manager=tasks), plan._payload["request"]["arguments"])
            if isinstance(request, ToolExecutionResult):
                return self._result(plan, request, f"sub-task-status/{plan.operation_id}/{plan.attempt}")

            snapshot = snapshot_sub_task_status(request)
            if snapshot.status_code == ToolResultStatus.ERROR:
                return self._result(plan, snapshot, f"sub-task-status/{plan.operation_id}/{plan.attempt}")

            waiting_ids = running_task_ids(snapshot.metadata["tasks"])
            interaction: dict[str, Any] | None = None
            if request.message:
                target = request.task_ids[0]
                target_status = next(entry["status"] for entry in snapshot.metadata["tasks"] if entry["task_id"] == target)
                if target_status == AgentStatus.MAX_CYCLES.value:
                    error = status_error(
                        f"Sub-task {target} reached max cycles and cannot continue.",
                        error_code="sub_task_max_cycles_reached",
                    )
                    return self._result(plan, error, f"sub-task-status/{plan.operation_id}/{plan.attempt}")
                action = tasks.message(target, f"message/{context.idempotency_key}", request.message)
                interaction = {"task_id": target, "action": action, "previous_status": target_status}
                waiting = set(waiting_ids)
                if request.wait_for_response or action == "continued":
                    waiting.add(target)
                order = {task_id: index for index, task_id in enumerate(request.task_ids)}
                waiting_ids = sorted(waiting, key=order.__getitem__)
                if not request.wait_for_response and not request.wait_for_completion:
                    self._update_message_result(
                        snapshot,
                        target=target,
                        action=action,
                        previous_status=target_status,
                        wait_for_completion=False,
                        check_interval_seconds=request.check_interval_seconds,
                        max_wait_seconds=request.max_wait_seconds,
                    )
                    snapshot.metadata["interaction"] = interaction
                    snapshot.content = json.dumps(snapshot.metadata, ensure_ascii=False)

            should_wait = request.wait_for_completion or request.wait_for_response
            if not should_wait or not waiting_ids:
                return self._result(plan, snapshot, f"sub-task-status/{plan.operation_id}/{plan.attempt}")

            state = {
                "parent_id": manager.parent_id,
                "tool_call_id": plan._payload["request"]["id"],
                "task_ids": request.task_ids,
                "detail_level": request.detail_level,
                "workspace_file_limit": request.workspace_file_limit,
                "wait_for_response": request.wait_for_response,
                "wait_for_completion": request.wait_for_completion,
                "waiting_task_ids": waiting_ids,
                "check_interval_seconds": request.check_interval_seconds,
                "max_wait_seconds": request.max_wait_seconds,
                "deadline_ms": started_ms + request.max_wait_seconds * 1000,
                "interaction": interaction,
            }
            return Accepted(self._handle(plan, state))

    def query(self, handle: dict[str, Any]) -> Outcome:
        state = handle.get("query_ref")
        if type(state) is not dict:
            return Unknown("sub_task_status handle has no query state")
        try:
            parent_id = state["parent_id"]
            tool_call_id = state["tool_call_id"]
            task_ids = state["task_ids"]
            detail_level = state["detail_level"]
            workspace_file_limit = state["workspace_file_limit"]
            wait_for_response = state["wait_for_response"]
            wait_for_completion = state["wait_for_completion"]
            waiting_task_ids = state["waiting_task_ids"]
            check_interval_seconds = state["check_interval_seconds"]
            max_wait_seconds = state["max_wait_seconds"]
            deadline_ms = state["deadline_ms"]
            interaction = state["interaction"]
        except KeyError:
            return Unknown("sub_task_status handle query state is incomplete")
        if (
            not isinstance(parent_id, str)
            or not isinstance(tool_call_id, str)
            or type(task_ids) is not list
            or not task_ids
            or not all(isinstance(task_id, str) for task_id in task_ids)
            or detail_level not in {"basic", "snapshot"}
            or isinstance(workspace_file_limit, bool)
            or not isinstance(workspace_file_limit, int)
            or type(wait_for_response) is not bool
            or type(wait_for_completion) is not bool
            or type(waiting_task_ids) is not list
            or not all(isinstance(task_id, str) for task_id in waiting_task_ids)
            or isinstance(check_interval_seconds, bool)
            or not isinstance(check_interval_seconds, int)
            or isinstance(max_wait_seconds, bool)
            or not isinstance(max_wait_seconds, int)
            or isinstance(deadline_ms, bool)
            or not isinstance(deadline_ms, int)
            or (interaction is not None and type(interaction) is not dict)
        ):
            return Unknown("sub_task_status handle query state is invalid")

        with self.runtime.heartbeat_store() as store:
            tasks = ChildTasks(store, parent_id, self.runtime)
            request = SubTaskStatusRequest(
                manager=tasks,
                task_ids=task_ids,
                detail_level=detail_level,
                workspace_file_limit=workspace_file_limit,
                message="",
                wait_for_response=wait_for_response,
                wait_for_completion=wait_for_completion,
                check_interval_seconds=check_interval_seconds,
                max_wait_seconds=max_wait_seconds,
            )
            snapshot = snapshot_sub_task_status(request)
            current_running = set(running_task_ids(snapshot.metadata["tasks"]))
            still_waiting = any(task_id in current_running for task_id in waiting_task_ids)
            timed_out = still_waiting and self._now_ms() >= deadline_ms
            if timed_out and wait_for_completion:
                add_wait_metadata(
                    snapshot.metadata,
                    wait_for_completion=True,
                    check_interval_seconds=check_interval_seconds,
                    max_wait_seconds=max_wait_seconds,
                    running_task_ids=[task_id for task_id in waiting_task_ids if task_id in current_running],
                    wait_exceeded=True,
                )
                snapshot.content = json.dumps(snapshot.metadata, ensure_ascii=False)
            if interaction is not None:
                snapshot.metadata["interaction"] = interaction
                snapshot.content = json.dumps(snapshot.metadata, ensure_ascii=False)
            if still_waiting and not timed_out:
                return Accepted(handle)
            if timed_out and wait_for_response and not wait_for_completion:
                snapshot.metadata["wait_exceeded"] = True
                snapshot.metadata["message"] = (
                    "Sub-task(s) are still running after the maximum wait. "
                    "Call sub_task_status again later instead of tight polling."
                )
                snapshot.content = json.dumps(snapshot.metadata, ensure_ascii=False)
            snapshot.tool_call_id = tool_call_id
            return Definitive(snapshot.to_dict(), (handle["evidence"],))

    def cancel(self, handle: dict[str, Any]) -> Outcome:
        return self.query(handle)

    def authenticate(self, item: InboxItem, plan: Record) -> bool:
        return False


@dataclass
class ChildTaskHandle:
    tasks: ChildTasks
    task_id: str

    @property
    def agent_name(self) -> str:
        return self.tasks.get(self.task_id).agent_name

    @property
    def status(self) -> AgentStatus:
        return self.poll().status

    def poll(self) -> BackgroundAgentTaskSnapshot:
        record = self.tasks.get(self.task_id)
        value = record.outcome
        cancelled = value.completion_reason == CompletionReason.CANCELLED
        output = None
        if not record.is_running():
            state, _, _ = self.tasks.store.read_state(self.task_id)
            output = project_result(
                self.tasks.store,
                self.task_id,
                next(reversed(state.turns)),
                runtime=reconstruct(self.tasks.runtime, self.tasks.store, self.task_id),
            ).final_output
        return BackgroundAgentTaskSnapshot(
            self.task_id,
            value.agent_name,
            value.status,
            "Run was cancelled." if cancelled else output,
            "Run was cancelled." if cancelled else value.error,
        )

    def snapshot(self) -> BackgroundAgentTaskSnapshot:
        return self.poll()

    def wait(self, timeout: float | None = None) -> BackgroundAgentTaskSnapshot:
        record = self.tasks.wait(self.task_id, timeout)
        if record.is_running():
            raise TimeoutError(f"Background agent task {self.task_id} was not ready before timeout.")
        return self.poll()

    def cancel(self) -> None:
        self.tasks.control(self.task_id, f"handle-cancel/{self.task_id}", "control", {"action": "cancel"})


_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_tool_policy_approval",
    "_vv_agent_tool_policy_can_use_tool",
    "_vv_agent_denied_side_effects",
    "_vv_agent_denied_capability_tags",
    "_vv_agent_deny_terminal_tools",
    "_vv_agent_denied_cost_dimensions",
)
_INITIAL_BUDGET_USAGE_METADATA_KEY = "_vv_agent_initial_budget_usage"


def tool_run_config_from_context(*, context: ToolContext, fallback: RunConfig) -> RunConfig:
    runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
    task_metadata = context.task_metadata if isinstance(context.task_metadata, dict) else {}
    is_sub_task = task_metadata.get("is_sub_task") is True
    policy_keys_present = any(key in runtime_metadata for key in _TOOL_POLICY_METADATA_KEYS)

    if is_sub_task or policy_keys_present:
        allowed = runtime_metadata.get("_vv_agent_allowed_tools")
        disallowed = runtime_metadata.get("_vv_agent_disallowed_tools")
        can_use_tool = runtime_metadata.get("_vv_agent_tool_policy_can_use_tool")
        approval = runtime_metadata.get("_vv_agent_tool_policy_approval")
        denied_side_effects = runtime_metadata.get("_vv_agent_denied_side_effects")
        denied_capability_tags = runtime_metadata.get("_vv_agent_denied_capability_tags")
        deny_terminal_tools = runtime_metadata.get("_vv_agent_deny_terminal_tools")
        denied_cost_dimensions = runtime_metadata.get("_vv_agent_denied_cost_dimensions")
        tool_policy = None
        if policy_keys_present:
            tool_policy = ToolPolicy(
                allowed_tools=([name for name in allowed if isinstance(name, str)] if isinstance(allowed, list) else None),
                disallowed_tools=([name for name in disallowed if isinstance(name, str)] if isinstance(disallowed, list) else []),
                approval=(approval if approval in {"always", "never", "on_request"} else "default"),
                can_use_tool=can_use_tool if callable(can_use_tool) else None,
                denied_side_effects=(denied_side_effects if isinstance(denied_side_effects, list) else []),
                denied_capability_tags=(denied_capability_tags if isinstance(denied_capability_tags, list) else []),
                deny_terminal_tools=(deny_terminal_tools if isinstance(deny_terminal_tools, bool) else False),
                denied_cost_dimensions=(denied_cost_dimensions if isinstance(denied_cost_dimensions, list) else []),
            )
    else:
        tool_policy = fallback.tool_policy

    def runtime_capability(key: str, fallback_value: Any) -> Any:
        return runtime_metadata.get(key, fallback_value)

    return replace(
        fallback,
        tool_policy=tool_policy,
        approval_provider=runtime_capability(
            "_vv_agent_approval_provider",
            fallback.approval_provider,
        ),
        approval_broker=runtime_capability(
            "_vv_agent_approval_broker",
            fallback.approval_broker,
        ),
        approval_timeout_seconds=runtime_capability(
            "_vv_agent_approval_timeout_seconds",
            fallback.approval_timeout_seconds,
        ),
    )


def agent_tool_parent_config(context: ToolContext | None) -> RunConfig | None:
    if context is None or context.ctx is None:
        return None
    runtime_metadata = context.ctx.metadata
    parent_config = runtime_metadata.get("_vv_agent_run_config")
    if not isinstance(parent_config, RunConfig):
        parent_config = BackgroundAgentTask._inherited_run_config(context, None)
    parent_config = tool_run_config_from_context(context=context, fallback=parent_config)
    provider = runtime_metadata.get("_vv_agent_model_provider", parent_config.model_provider)
    if provider is None:
        return None
    return replace(
        parent_config,
        model_provider=provider,
        cancellation_token=context.ctx.cancellation_token,
        workspace=context.workspace,
        workspace_backend=context.workspace_backend,
        budget_limits=runtime_metadata.get("_vv_agent_budget_limits", parent_config.budget_limits),
        memory_providers=runtime_metadata.get("_vv_agent_memory_providers", parent_config.memory_providers),
        context=getattr(context.run_context, "context", parent_config.context),
    )


def run_child_agent(
    child_agent: Agent,
    *,
    arguments: dict[str, Any],
    parent_config: RunConfig,
    context: ToolContext | None = None,
) -> RunResult:
    child_input = child_agent_prompt(arguments=arguments, context=context)
    from vv_agent.runner import Runner

    return Runner.run_sync(
        child_agent,
        child_input,
        run_config=child_run_config(parent_config, context=context),
    )


def child_run_config(parent_config: RunConfig, *, context: ToolContext | None = None) -> RunConfig:
    cancellation_token = parent_config.cancellation_token
    if cancellation_token is not None:
        cancellation_token = cancellation_token.child()
    return replace(
        parent_config,
        model=None,
        model_settings=None,
        stream=None,
        shared_state=deepcopy(context.shared_state) if context is not None else None,
        initial_messages=None,
        before_cycle_messages=None,
        interruption_messages=None,
        cancellation_token=cancellation_token,
        host_cost_meter=None,
        metadata={key: value for key, value in parent_config.metadata.items() if key != _INITIAL_BUDGET_USAGE_METADATA_KEY},
    )


def child_agent_prompt(*, arguments: dict[str, Any], context: ToolContext | None) -> str:
    raw_task_description = arguments.get("task_description")
    task_description = raw_task_description.strip() if isinstance(raw_task_description, str) else ""
    if not task_description:
        raise ValueError("agent tool requires task_description")
    output_requirements = arguments.get("output_requirements")
    if isinstance(output_requirements, str) and output_requirements.strip():
        task_description += f"\n\n<Output Requirements>\n{output_requirements.strip()}\n</Output Requirements>"
    if arguments.get("include_main_summary") is True and context is not None:
        runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
        parent_summary = context.shared_state.get("main_task_summary")
        if not isinstance(parent_summary, str) or not parent_summary.strip():
            parent_summary = runtime_metadata.get("_vv_agent_input")
        if isinstance(parent_summary, str) and parent_summary.strip():
            task_description += f"\n\n<Main Task Summary>\n{parent_summary.strip()}\n</Main Task Summary>"
    return task_description


_RESERVED_SUB_AGENT_METADATA_KEYS = (
    "browser_scope_key",
    "is_sub_task",
    "parent_task_id",
    "session_id",
    "session_memory_enabled",
    "sub_agent_name",
    "task_id",
    "workspace",
    "run_id",
    "trace_id",
    "parent_run_id",
    "parent_tool_call_id",
    "_vv_agent_run_id",
    "_vv_agent_trace_id",
    "_vv_agent_agent_name",
    "_vv_agent_session_id",
    "_vv_agent_parent_run_id",
    "_vv_agent_parent_tool_call_id",
    *_TOOL_POLICY_METADATA_KEYS,
)


def build_sub_agent_task(
    *,
    parent_task: AgentTask,
    sub_task_id: str,
    sub_session_id: str,
    sub_agent_name: str,
    sub_agent: SubAgentConfig,
    resolved_model_id: str,
    child_run_id: str,
    trace_id: str,
    parent_run_id: str,
    parent_tool_call_id: str,
    request: SubTaskRequest,
    parent_shared_state: dict[str, Any],
    workspace_path: Path,
    resolved_native_multimodal: bool | None = None,
    resolved_context_length: int | None = None,
    resolved_max_output_tokens: int | None = None,
    effective_model_settings: ModelSettings | None = None,
    parent_tool_policy_metadata: dict[str, Any] | None = None,
) -> AgentTask:
    language = str(parent_task.metadata.get("language", "zh-CN"))
    available_skills = parent_task.metadata.get("available_skills")
    if not isinstance(available_skills, list):
        available_skills = None

    if sub_agent.system_prompt:
        prompt_bundle = build_raw_system_prompt_bundle(sub_agent.system_prompt)
    else:
        prompt_bundle = build_system_prompt_bundle(
            sub_agent.description,
            language=language,
            allow_interruption=False,
            use_workspace=parent_task.use_workspace,
            enable_todo_management=True,
            agent_type=parent_task.agent_type,
            available_skills=available_skills,
            workspace=workspace_path,
        )

    user_prompt = request.task_description
    if request.output_requirements:
        user_prompt = f"{user_prompt}\n\n<Output Requirements>\n{request.output_requirements}\n</Output Requirements>"
    if request.include_main_summary:
        parent_summary = build_parent_summary(parent_task=parent_task, parent_shared_state=parent_shared_state)
        if parent_summary:
            user_prompt = f"{user_prompt}\n\n<Main Task Summary>\n{parent_summary}\n</Main Task Summary>"

    excluded_tools = set(parent_task.exclude_tools)
    excluded_tools.update(sub_agent.exclude_tools)
    excluded_tools.update({CREATE_SUB_TASK_TOOL_NAME, SUB_TASK_STATUS_TOOL_NAME})
    metadata: dict[str, Any] = {
        "is_sub_task": True,
        "parent_task_id": parent_task.task_id,
        "sub_agent_name": sub_agent_name,
        "session_memory_enabled": sub_agent.session_memory_enabled,
        "workspace": str(workspace_path),
    }
    for key in (
        "bash_shell",
        "windows_shell_priority",
        "bash_env",
        "allow_outside_workspace_paths",
        "language",
        "available_skills",
        "active_skills",
    ):
        value = parent_task.metadata.get(key)
        if value is not None:
            metadata[key] = value
    if sub_agent.metadata:
        metadata.update(sub_agent.metadata)
    if request.metadata:
        metadata.update(request.metadata)
    for key in _RESERVED_SUB_AGENT_METADATA_KEYS:
        metadata.pop(key, None)
    metadata.update(parent_tool_policy_metadata or {})
    metadata.update(
        configured_child_metadata_denials(
            parent_tool_policy_metadata or {},
            sub_agent,
        )
    )
    project_resolved_model_limits(
        metadata,
        context_length=resolved_context_length,
        max_output_tokens=resolved_max_output_tokens,
    )
    metadata.update(
        {
            "is_sub_task": True,
            "parent_task_id": parent_task.task_id,
            "sub_agent_name": sub_agent_name,
            "session_memory_enabled": sub_agent.session_memory_enabled,
            "workspace": str(workspace_path),
        }
    )
    metadata = canonical_sub_run_metadata(
        metadata,
        sub_task_id=sub_task_id,
        sub_session_id=sub_session_id,
        sub_agent_name=sub_agent_name,
        child_run_id=child_run_id,
        trace_id=trace_id,
        parent_run_id=parent_run_id,
        parent_tool_call_id=parent_tool_call_id,
    )

    return AgentTask(
        task_id=sub_task_id,
        model=resolved_model_id,
        prompt_bundle=prompt_bundle,
        user_prompt=user_prompt,
        max_cycles=max(sub_agent.max_cycles, 1),
        memory_compact_threshold=parent_task.memory_compact_threshold,
        memory_threshold_percentage=parent_task.memory_threshold_percentage,
        microcompaction_policy=parent_task.microcompaction_policy,
        no_tool_policy="finish",
        allow_interruption=False,
        use_workspace=parent_task.use_workspace,
        sub_agents={},
        agent_type=parent_task.agent_type,
        native_multimodal=(parent_task.native_multimodal if resolved_native_multimodal is None else resolved_native_multimodal),
        extra_tool_names=list(parent_task.extra_tool_names),
        exclude_tools=sorted(excluded_tools),
        model_settings=(parent_task.model_settings if effective_model_settings is None else effective_model_settings),
        metadata=metadata,
    )


def configured_child_metadata_denials(
    parent_policy_metadata: dict[str, Any],
    sub_agent: SubAgentConfig,
) -> dict[str, Any]:
    denied_side_effects = normalize_denied_side_effects(
        [
            *parent_policy_metadata.get("_vv_agent_denied_side_effects", []),
            *sub_agent.denied_side_effects,
        ]
    )
    denied_capability_tags = normalize_metadata_labels(
        [
            *parent_policy_metadata.get("_vv_agent_denied_capability_tags", []),
            *sub_agent.denied_capability_tags,
        ],
        field_name="denied_capability_tags",
    )
    denied_cost_dimensions = normalize_metadata_labels(
        [
            *parent_policy_metadata.get("_vv_agent_denied_cost_dimensions", []),
            *sub_agent.denied_cost_dimensions,
        ],
        field_name="denied_cost_dimensions",
    )
    projected: dict[str, Any] = {}
    if denied_side_effects:
        projected["_vv_agent_denied_side_effects"] = [item.value for item in denied_side_effects]
    if denied_capability_tags:
        projected["_vv_agent_denied_capability_tags"] = denied_capability_tags
    if parent_policy_metadata.get("_vv_agent_deny_terminal_tools") is True or sub_agent.deny_terminal_tools:
        projected["_vv_agent_deny_terminal_tools"] = True
    if denied_cost_dimensions:
        projected["_vv_agent_denied_cost_dimensions"] = denied_cost_dimensions
    return projected


def canonical_sub_run_metadata(
    metadata: dict[str, Any],
    *,
    sub_task_id: str,
    sub_session_id: str,
    sub_agent_name: str,
    child_run_id: str,
    trace_id: str,
    parent_run_id: str,
    parent_tool_call_id: str,
) -> dict[str, Any]:
    canonical = dict(metadata)
    for key in (
        "browser_scope_key",
        "session_id",
        "sub_agent_name",
        "task_id",
        "run_id",
        "trace_id",
        "parent_run_id",
        "parent_tool_call_id",
        "_vv_agent_run_id",
        "_vv_agent_trace_id",
        "_vv_agent_agent_name",
        "_vv_agent_session_id",
        "_vv_agent_parent_run_id",
        "_vv_agent_parent_tool_call_id",
    ):
        canonical.pop(key, None)
    canonical.update(
        {
            "task_id": sub_task_id,
            "session_id": sub_session_id,
            "browser_scope_key": sub_session_id,
            "sub_agent_name": sub_agent_name,
            "run_id": child_run_id,
            "trace_id": trace_id,
            "_vv_agent_run_id": child_run_id,
            "_vv_agent_trace_id": trace_id,
            "_vv_agent_agent_name": sub_agent_name,
            "_vv_agent_session_id": sub_session_id,
        }
    )
    if parent_run_id:
        canonical["parent_run_id"] = parent_run_id
        canonical["_vv_agent_parent_run_id"] = parent_run_id
    if parent_tool_call_id:
        canonical["parent_tool_call_id"] = parent_tool_call_id
        canonical["_vv_agent_parent_tool_call_id"] = parent_tool_call_id
    return canonical


def build_parent_summary(*, parent_task: AgentTask, parent_shared_state: dict[str, Any]) -> str:
    lines = [f"Parent task goal: {parent_task.user_prompt}"]
    todo_list = parent_shared_state.get("todo_list")
    if isinstance(todo_list, list) and todo_list:
        lines.append("Parent TODO status:")
        for item in todo_list:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "Untitled"))
            status = str(item.get("status", "pending"))
            lines.append(f"- [{status}] {title}")
    return "\n".join(lines)
