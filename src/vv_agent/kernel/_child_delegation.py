"""Actual delegated-child admission rules, with SDK compilation as explicit leaves."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ._budget_values import RunBudgetLimits
from ._child_store import ChildSession
from ._execution_values import AgentStatus, AgentTask, CompletionReason, SubAgentConfig, SubTaskOutcome, SubTaskRequest
from ._records import CHILD_ADMISSION, Record, SessionSpec, copy_json, digest, validate
from ._store import SessionStore
from ._token_count import _StopIterationTransport
from ._tool_values import ToolDirective, ToolExecutionResult, ToolResultStatus


class ChildRead(Enum):
    MAX_HANDOFFS = auto()
    AGENT_NAME = auto()
    WORKSPACE = auto()
    TODO_LIST = auto()
    BUDGET_LIMITS = auto()
    HANDLER_VERSION = auto()
    TRANSFER_METADATA = auto()


@dataclass(frozen=True, slots=True)
class ChildBindingMode:
    name: str


@dataclass(frozen=True, slots=True)
class ParseConfiguredChildren:
    arguments: dict[str, Any]
    collect: Callable[[SubTaskRequest], SubTaskOutcome]


@dataclass(frozen=True, slots=True)
class ParentInputMetadata:
    user_prompt: str


@dataclass(frozen=True, slots=True)
class ChildPrompt:
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SelectChildAgent:
    selector: str
    mode: str
    sub: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class CompileChild:
    request: SubTaskRequest
    turn_id: str
    mode: str
    sub: SubAgentConfig | None


@dataclass(frozen=True, slots=True)
class ChildPromptBundle:
    task: AgentTask
    sub: SubAgentConfig


@dataclass(frozen=True, slots=True)
class ChildDefinition:
    task: AgentTask


ChildRequest = (
    ChildRead
    | ChildBindingMode
    | ParseConfiguredChildren
    | ParentInputMetadata
    | ChildPrompt
    | SelectChildAgent
    | CompileChild
    | ChildPromptBundle
    | ChildDefinition
)


def error(code: str, message: str) -> ToolExecutionResult:
    payload = {"ok": False, "error": message, "error_code": code}
    return ToolExecutionResult(
        "", json.dumps(payload, ensure_ascii=False), ToolResultStatus.ERROR, error_code=code, metadata=payload
    )


def success(payload: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult("", json.dumps(payload, ensure_ascii=False), metadata=payload)


def assemble(
    store: SessionStore, plan: Record, task: AgentTask
) -> Generator[ChildRequest, Any, list[ChildSession] | ToolExecutionResult | None]:
    try:
        name, args = plan._payload["request"]["name"], copy_json(plan._payload["request"]["arguments"])
        mode, transfer = yield ChildBindingMode(name)
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

            parsed = yield ParseConfiguredChildren(args | {"wait_for_completion": True}, collect)
            if parsed.status_code == ToolResultStatus.ERROR:
                return parsed
            selector = requests[0].agent_name
            sub = task.sub_agents.get(selector)
            if sub is None:
                return error(
                    "sub_task_failed", f"Unknown sub-agent {selector!r}. Available: {', '.join(sorted(task.sub_agents))}"
                )
        elif mode in {"agent_as_tool", "background_task"}:
            selector = name

            yield ParentInputMetadata(task.user_prompt)
            try:
                prompt = yield ChildPrompt(args)
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
            "max_handoffs", task.metadata.get("vv_session", {}).get("max_handoffs", (yield ChildRead.MAX_HANDOFFS))
        )
        assert maximum is not None
        if mode == "handoff" and count >= maximum:
            result = error("maximum_handoffs_exceeded", "maximum handoff depth exceeded")
            result.directive = ToolDirective.FINISH
            result.metadata["mode"] = "handoff"
            return result
        yield SelectChildAgent(selector, mode, sub.to_dict() if sub else None)
        if (
            mode == "handoff"
            and (yield ChildRead.AGENT_NAME) != task.metadata.setdefault("vv_session", {})["handoff_targets"][selector]
        ):
            return error("handler_schema_or_capability_mismatch", "handoff target differs from frozen definition")
        result = []
        for index, request in enumerate(requests):
            child_key = digest({"operation_id": plan.operation_id, "attempt": plan.attempt, "index": index})
            sid = f"{plan.session_id}/child/{child_key[:24]}"
            tid = f"{sid}/turn/start"
            child_task = yield CompileChild(request, tid, mode, sub)
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
                        "workspace": (yield ChildRead.WORKSPACE),
                        "session_memory_enabled": sub.session_memory_enabled,
                    }
                )
                # Do not allow configured metadata to erase the compiled policy.
                metadata.update({k: v for k, v in child_task.metadata.items() if k.startswith("_vv_agent_")})
                child_task.prompt_bundle = yield ChildPromptBundle(task, sub)
                prompt = request.task_description
                if request.output_requirements:
                    prompt += f"\n\n<Output Requirements>\n{request.output_requirements}\n</Output Requirements>"
                if request.include_main_summary:
                    summary = [f"Parent task goal: {task.user_prompt}"]
                    todos = yield ChildRead.TODO_LIST
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
            facts = yield ChildDefinition(child_task)
            definition = facts.definitions.definition(child_task, facts.bindings, facts.catalog)
            facts = yield ChildDefinition(child_task)
            definition_digest = facts.definitions.definition_digest(child_task, facts.bindings, facts.catalog)
            admission = {
                "mode": mode,
                "selector": selector,
                "definition": definition,
                "definition_digest": definition_digest,
                "budget": ((yield ChildRead.BUDGET_LIMITS) or RunBudgetLimits()).to_dict(),
                "handler_version": (yield ChildRead.HANDLER_VERSION),
                "sub_config": sub.to_dict() if sub else None,
                "exclude_files_pattern": request.exclude_files_pattern,
                "handoff_count": count + (mode == "handoff"),
                "max_handoffs": maximum,
                "handoff_metadata": copy_json((yield ChildRead.TRANSFER_METADATA)) if transfer else {},
            }
            validate(admission, CHILD_ADMISSION)
            result.append(
                ChildSession(
                    SessionSpec(
                        sid, created._payload["principal"], (yield ChildRead.WORKSPACE), attributes={"child_admission": admission}
                    ),
                    child_task.user_prompt,
                    background=background,
                )
            )
        return result
    except StopIteration as exc:
        raise _StopIterationTransport(exc) from exc
