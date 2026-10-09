"""SDK definitions and handles over child records, without another execution ledger."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from vv_agent.agent import Agent
from vv_agent.background_task import BackgroundAgentTaskSnapshot
from vv_agent.budget import RunBudgetLimits
from vv_agent.model import ModelRef
from vv_agent.prompt.builder import build_raw_system_prompt_bundle, build_system_prompt_bundle
from vv_agent.run_config import ToolPolicy, merge_tool_policies
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.tools.base import ToolContext, ToolSpec
from vv_agent.tools.executor import RegistryToolExecutor
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.tools.handlers.sub_task_status import _error as status_error
from vv_agent.tools.handlers.sub_task_status import sub_task_status
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
from .records import CHILD_ADMISSION, InboxItem, Record, SessionSpec, copy_json, digest, validate
from .reducer import ExecutionState
from .result import project_result
from .store import Conflict, SessionStore, StoredRecord

if TYPE_CHECKING:
    from .runtime import Runtime


def error(code: str, message: str) -> ToolExecutionResult:
    payload = {"ok": False, "error": message, "error_code": code}
    return ToolExecutionResult(
        "", json.dumps(payload, ensure_ascii=False), ToolResultStatus.ERROR, error_code=code, metadata=payload
    )


def success(payload: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult("", json.dumps(payload, ensure_ascii=False), metadata=payload)


def register_adapters(runtime: Runtime, registry: ToolRegistry) -> None:
    if registry.has_executor("sub_task_status"):
        executor = registry.get_executor("sub_task_status")
        if isinstance(executor, RegistryToolExecutor) and executor.handler is sub_task_status:
            executor.handler = child_status
    register_handoffs(runtime, registry)


def child_status(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    manager = cast(ChildTasks, context.sub_task_manager)
    # Tool providers run on a separate thread from the parent's driver.
    with manager.runtime.heartbeat_store() as store:
        tasks = ChildTasks(store, manager.parent_id, manager.runtime)
        return status_with_store(replace(context, sub_task_manager=tasks.tool_manager()), arguments)


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
    # Handlers remain host bindings; policy and prompt come from the admitted definition.
    return Agent(
        name,
        sub.system_prompt or sub.description or "Complete the delegated task.",
        model=sub.model,
        tools=[
            tool
            for tool in runtime.agent.tools
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
    config = replace(
        runtime.config,
        model=None,
        model_settings=None,
        session=None,
        stream=None,
        shared_state=runtime.durable_state(context.shared_state),
        initial_messages=None,
        before_cycle_messages=None,
        interruption_messages=None,
        sub_task_manager=None,
        cancellation_token=None,
        host_cost_meter=None,
        checkpoint_config=None,
        checkpoint_extensions=[],
        reconciliation_provider=None,
        workspace=context.workspace,
        workspace_backend=context.workspace_backend,
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


def assemble(
    runtime: Runtime, plan: Record, context: ToolContext, task: AgentTask, store: SessionStore
) -> list[ChildSession] | ToolExecutionResult | None:
    name, args = plan._payload["request"]["name"], copy_json(plan._payload["request"]["arguments"])
    executor = runtime.registry.get_executor(name) if runtime.registry.has_executor(name) else None
    mode = executor.metadata.get("mode") if executor else None
    transfer = next((t for t in runtime.agent.handoffs if t.tool_name == name), None)
    if transfer:
        mode = "handoff"
    requests = []
    sub = None
    background = mode == "background_task"
    if name == "create_sub_task" and task.sub_agents:
        mode = "configured"
        wait = args.get("wait_for_completion", True)
        if not isinstance(wait, bool):
            return error("invalid_tool_arguments", "`wait_for_completion` must be a boolean")
        background = not wait

        def collect(request):
            requests.append(request)
            if request.agent_name not in task.sub_agents:
                return SubTaskOutcome(
                    "pending",
                    request.agent_name,
                    AgentStatus.FAILED,
                    completion_reason=CompletionReason.FAILED,
                    error=f"Unknown sub-agent {request.agent_name!r}. Available: {', '.join(sorted(task.sub_agents))}",
                    error_code="sub_task_failed",
                )
            return SubTaskOutcome("pending", request.agent_name, AgentStatus.COMPLETED)

        parsed = create_sub_task(replace(context, sub_task_runner=collect), args | {"wait_for_completion": True})
        if parsed.status_code == ToolResultStatus.ERROR:
            return parsed
        selector = requests[0].agent_name
        sub = task.sub_agents.get(selector)
        if sub is None:
            return error("sub_task_failed", f"Unknown sub-agent {selector!r}. Available: {', '.join(sorted(task.sub_agents))}")
    elif mode in {"agent_as_tool", "background_task"}:
        selector = name
        from vv_agent.runner import Runner

        if context.ctx is not None:
            context.ctx.metadata.setdefault("_vv_agent_input", task.user_prompt)
        try:
            prompt = Runner._child_agent_prompt(arguments=args, context=context)
        except ValueError as exc:
            return error("invalid_tool_arguments", str(exc))
        requests = [SubTaskRequest(name, prompt)]
    elif mode == "handoff":
        selector = name
        value = args.get("input")
        if set(args) != {"input"} or not isinstance(value, str) or not value.strip():
            return error("invalid_handoff_arguments", "handoff requires a non-empty input string and no additional arguments")
        requests = [SubTaskRequest(name, value.strip())]
    else:
        return None
    assert isinstance(mode, str)
    created = store.read(plan.session_id, limit=1).records[0].record
    parent_admission = created._payload["attributes"].get("child_admission", {})
    count = parent_admission.get("handoff_count", 0)
    maximum = parent_admission.get(
        "max_handoffs", task.metadata.get("vv_session", {}).get("max_handoffs", runtime.config.max_handoffs)
    )
    assert maximum is not None
    if mode == "handoff" and count >= maximum:
        result = error("maximum_handoffs_exceeded", "maximum handoff depth exceeded")
        result.directive = ToolDirective.FINISH
        result.metadata["mode"] = "handoff"
        return result
    agent = target_agent(runtime, selector, mode, sub.to_dict() if sub else None)
    if mode == "handoff" and agent.name != task.metadata.setdefault("vv_session", {})["handoff_targets"][selector]:
        return error("handler_schema_or_capability_mismatch", "handoff target differs from frozen definition")
    result = []
    for index, request in enumerate(requests):
        sid = (
            f"{plan.session_id}/child/{digest({'operation_id': plan.operation_id, 'attempt': plan.attempt, 'index': index})[:24]}"
        )
        tid = f"{sid}/turn/start"
        config = child_config(runtime, context, agent, mode, sub)
        if request.exclude_files_pattern:
            config = replace(
                config,
                workspace_backend=DiscoveryFilteredWorkspaceBackend(context.workspace_backend, request.exclude_files_pattern),
            )
        child_rt = runtime.for_agent(agent, config)
        child_task = child_rt.compile(request.task_description, tid)
        child_task.task_id = tid
        if sub:
            metadata = (
                copy_json(child_task.metadata)
                | {
                    key: copy_json(task.metadata[key])
                    for key in (
                        "bash_shell",
                        "windows_shell_priority",
                        "bash_env",
                        "allow_outside_workspace_paths",
                        "language",
                        "available_skills",
                        "active_skills",
                    )
                    if task.metadata.get(key) is not None
                }
                | sub.metadata
                | request.metadata
                | {
                    "is_sub_task": True,
                    "parent_task_id": task.task_id,
                    "sub_agent_name": selector,
                    "parent_run_id": plan.turn_id,
                    "parent_tool_call_id": plan._payload["request"]["id"],
                    "session_id": sid,
                    "workspace": str(context.workspace),
                    "session_memory_enabled": sub.session_memory_enabled,
                }
            )
            # Do not allow configured metadata to erase the compiled policy.
            metadata.update({k: v for k, v in child_task.metadata.items() if k.startswith("_vv_agent_")})
            if sub.system_prompt:
                child_task.prompt_bundle = build_raw_system_prompt_bundle(sub.system_prompt)
            else:
                child_task.prompt_bundle = build_system_prompt_bundle(
                    sub.description,
                    language=str(task.metadata.get("language", "zh-CN")),
                    allow_interruption=False,
                    use_workspace=task.use_workspace,
                    enable_todo_management=True,
                    agent_type=task.agent_type,
                    available_skills=task.metadata.get("available_skills"),
                    workspace=context.workspace,
                )
            prompt = request.task_description
            if request.output_requirements:
                prompt += f"\n\n<Output Requirements>\n{request.output_requirements}\n</Output Requirements>"
            if request.include_main_summary:
                summary = [f"Parent task goal: {task.user_prompt}"]
                todos = context.shared_state.get("todo_list")
                if isinstance(todos, list) and todos:
                    summary.append("Parent TODO status:")
                    summary.extend(
                        f"- [{item.get('status', 'pending')}] {item.get('title', 'Untitled')}"
                        for item in todos
                        if isinstance(item, dict)
                    )
                parent_summary = "\n".join(summary)
                prompt += f"\n\n<Main Task Summary>\n{parent_summary}\n</Main Task Summary>"
            child_task.user_prompt = prompt
            child_task.metadata = metadata
            child_task.exclude_tools = sorted(
                set(task.exclude_tools) | set(sub.exclude_tools) | {"create_sub_task", "sub_task_status"}
            )
            child_task.sub_agents = {}
            child_task.allow_interruption = False
            child_task.memory_compact_threshold = task.memory_compact_threshold
            child_task.memory_threshold_percentage = task.memory_threshold_percentage
            child_task.microcompaction_policy = task.microcompaction_policy
            child_task.extra_tool_names = list(task.extra_tool_names)
            child_task.use_workspace = task.use_workspace
            child_task.agent_type = task.agent_type
        admission = {
            "mode": mode,
            "selector": selector,
            "definition": child_rt.definition(child_task),
            "definition_digest": child_rt.definition_digest(child_task),
            "budget": (config.budget_limits or RunBudgetLimits()).to_dict(),
            "handler_version": runtime.handler_version,
            "sub_config": sub.to_dict() if sub else None,
            "exclude_files_pattern": request.exclude_files_pattern,
            "handoff_count": count + (mode == "handoff"),
            "max_handoffs": maximum,
            "handoff_metadata": copy_json(transfer.metadata) if transfer else {},
        }
        validate(admission, CHILD_ADMISSION)
        result.append(
            ChildSession(
                SessionSpec(
                    sid, created._payload["principal"], str(context.workspace), attributes={"child_admission": admission}
                ),
                child_task.user_prompt,
                background=background,
            )
        )
    return result


def reconstruct(runtime: Runtime, store: SessionStore, sid: str, *, created: Record | None = None) -> Runtime:
    created = created if created is not None else store.read(sid, limit=1).records[0].record
    admission = created.payload["attributes"]["child_admission"]
    validate(admission, CHILD_ADMISSION)
    if admission["handler_version"] != runtime.handler_version:
        raise Conflict("child admission handler version mismatch")
    agent = target_agent(runtime, admission["selector"], admission["mode"], admission["sub_config"])
    task = AgentTask.from_dict(admission["definition"]["task"])
    # Reapply current host restrictions; the task keeps the frozen restrictions as well.
    context = SimpleNamespace(
        shared_state=runtime.bind_state(task.initial_shared_state, task),
        workspace=Path(created._payload["workspace"]),
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
    turn_id = (
        turn_id
        or state.active_turn_id
        or next(reversed(state.turns), rows[0].record._payload["attributes"]["child_handle"]["turn_id"])
    )
    admission = rows[0].record._payload["attributes"].get("child_admission")
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
    admission = first[1][0].record._payload["attributes"].get("child_admission")
    if admission is None:
        raise ValueError("SDK child projection needs its admitted definition")
    mode = admission["mode"]
    if mode == "configured":
        outcomes = []
        for i, h in enumerate(handles):
            snapshot = first if i == 0 else store.read_state(h["session_id"])
            rt = reconstruct(runtime, store, h["session_id"], created=snapshot[1][0].record)
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
        rt = reconstruct(runtime, store, h["session_id"], created=first[1][0].record)
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
    admission = store.read(handles[0]["session_id"], limit=1).records[0].record._payload["attributes"].get("child_admission")
    if not admission:
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
        value = outcome(self.store, task_id, rt)
        state, _, _ = self.store.read_state(task_id)
        latest_turn = state.turns[next(reversed(state.turns))] if state.turns else None
        if state.active_turn_id is not None and state.phase == "active":
            value = replace(value, status=AgentStatus.RUNNING)
        if state.active_turn_id is None and (
            any(item.item.kind == "follow_up" for item in self.store.peek_inbox(task_id))
            or any(
                r._payload["disposition"] == "queued" and i not in state.admitted_inputs for i, r in state.applied_inputs.items()
            )
        ):
            value = replace(value, status=AgentStatus.RUNNING)
        task = AgentTask.from_dict(created.payload["attributes"]["child_admission"]["definition"]["task"])
        return SimpleNamespace(
            task_id=task_id,
            session_id=task_id,
            agent_name=value.agent_name,
            task_title=latest_turn.start._payload["definition"]["task"]["user_prompt"] if latest_turn else task.user_prompt,
            parent_run_id=task.metadata.get("parent_run_id"),
            parent_tool_call_id=task.metadata.get("parent_tool_call_id"),
            outcome=value,
            is_running=lambda: value.status in {AgentStatus.PENDING, AgentStatus.RUNNING},
            current_cycle_index=value.cycles or None,
            latest_cycle={"cycle_index": value.cycles, "status": value.status.value} if value.cycles else None,
            workspace_backend=rt.config.workspace_backend,
        )

    def wait(self, task_id: str, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            record = self.get(task_id)
            if record is None or not record.is_running() or (deadline is not None and time.monotonic() >= deadline):
                return record
            Event().wait(0.01)

    def handle(self, task_id: str) -> ChildTaskHandle:
        if self.get(task_id) is None:
            raise KeyError(f"Unknown background agent task: {task_id}")
        return ChildTaskHandle(self, task_id)

    def control(self, task_id: str, input_id: str, kind: str, payload: dict[str, Any]) -> None:
        if self.get(task_id) is None:
            raise KeyError(task_id)
        created = self.store.read(task_id, limit=1).records[0].record
        h = created._payload["attributes"]["child_handle"]
        state, _, _ = self.store.read_state(task_id)
        turn_id = state.active_turn_id or h["turn_id"]
        generation = state.turns[turn_id].start._payload["generation"] if turn_id in state.turns else h["generation"]
        with self.store.atomic() as tx:
            tx.push(task_id, InboxItem(input_id, kind, payload, turn_id, generation))

    def message(self, task_id: str, input_id: str, text: str) -> str:
        if self.get(task_id) is None:
            raise KeyError(task_id)
        state, _, _ = self.store.read_state(task_id)
        applied = state.applied_inputs.get(input_id)
        pending = next((i.item for i in self.store.peek_inbox(task_id) if i.item.input_id == input_id), None)
        previous = InboxItem(**applied._payload["input"]) if applied else pending
        if previous is not None:
            content = previous.payload.get("content")
            previous_text = content.get("text") if isinstance(content, dict) else content
            if previous_text != text:
                raise Conflict("child message id reused with different content")
            with self.store.atomic() as tx:
                tx.push(task_id, previous)
            return "continued" if previous.kind == "follow_up" else "message_queued"
        target = state.active_turn_id
        generation = state.turns[target].start._payload["generation"] if target else None
        action = "message_queued" if target else "continued"
        kind, payload = ("steer", {"content": text}) if target else ("follow_up", {"content": text})
        turn_wait = state.turns[target].wait if target else None
        if turn_wait:
            kind = "user"
            payload = {"content": {"interaction_id": turn_wait._payload["interaction_id"], "text": text}}
        elif target:
            waits = [
                a.wait
                for op in state.operations.values()
                for a in op.attempts.values()
                if a.wait and a.wait["handle"]["kind"] == "user"
            ]
            if waits:
                wait = waits[0]
                kind = "user"
                payload = {
                    "content": {
                        "interaction_id": wait["handle"]["interaction_id"],
                        "operation_id": next(
                            oid for oid, op in state.operations.items() if any(a.wait is wait for a in op.attempts.values())
                        ),
                        "text": text,
                    }
                }
        with self.store.atomic() as tx:
            tx.push(task_id, InboxItem(input_id, kind, payload, target, generation))
        return action

    def tool_manager(self) -> SubTaskManager:
        return cast(SubTaskManager, self)


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
