from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME
from vv_agent.result import _PendingToolApproval
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.runtime.hooks import RuntimeHookManager
from vv_agent.runtime.tool_planner import plan_tool_names
from vv_agent.tools import ToolContext, ToolRegistry
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import AgentTask, CycleRecord, Message, ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext


@dataclass(slots=True)
class ToolRunOutcome:
    directive_result: ToolExecutionResult | None = None
    interruption_messages: list[Message] = field(default_factory=list)


class _ConfiguredSubTaskCancelledError(CancelledError):
    pass


class ToolCallRunner:
    def __init__(self, *, tool_registry: ToolRegistry, hook_manager: RuntimeHookManager | None = None) -> None:
        self.tool_registry = tool_registry
        self.hook_manager = hook_manager or RuntimeHookManager()
        self.tool_orchestrator = ToolOrchestrator.from_registry(tool_registry)

    def run(
        self,
        *,
        task: AgentTask,
        tool_calls: list[ToolCall],
        context: ToolContext,
        messages: list[Message],
        cycle_record: CycleRecord,
        interruption_provider: Callable[[], list[Message]] | None = None,
        on_tool_start: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolCall, ToolExecutionResult], None] | None = None,
        ctx: ExecutionContext | None = None,
    ) -> ToolRunOutcome:
        latest_directive_result: ToolExecutionResult | None = None
        interruption_messages: list[Message] = []
        image_notifications: list[Message] = []
        planned_tool_names = cycle_record._planned_tool_names
        allowed_tool_names = set(plan_tool_names(task) if planned_tool_names is None else planned_tool_names)

        for index, call in enumerate(tool_calls):
            if ctx is not None:
                ctx.check_cancelled()
            patched_call, short_circuit_result = self.hook_manager.apply_before_tool_call(
                task=task,
                cycle_index=context.cycle_index,
                call=call,
                context=context,
            )
            if short_circuit_result is not None:
                result = short_circuit_result
                if not result.tool_call_id:
                    result.tool_call_id = call.id
            else:
                call_context = replace(
                    context,
                    tool_call_id=patched_call.id,
                    tool_name=patched_call.name,
                    arguments=dict(patched_call.arguments),
                )
                if on_tool_start is not None:
                    on_tool_start(patched_call)
                result = self.tool_orchestrator.run_one(
                    patched_call,
                    context=call_context,
                    allowed_tool_names=allowed_tool_names,
                )
                if result.error_code == "tool_approval_required" and ctx is not None and isinstance(result.metadata, dict):
                    interruption_id = result.metadata.get("approval_interruption_id")
                    if isinstance(interruption_id, str) and interruption_id.strip():
                        ctx._pending_tool_approval = _PendingToolApproval(
                            interruption_id=interruption_id,
                            call=deepcopy(patched_call),
                            cycle_index=context.cycle_index,
                            context=replace(
                                call_context,
                                arguments=deepcopy(call_context.arguments),
                                shared_state=deepcopy(call_context.shared_state),
                            ),
                            allowed_tool_names=frozenset(allowed_tool_names),
                            orchestrator=self.tool_orchestrator,
                            task=task,
                            hook_manager=self.hook_manager,
                        )
                if ctx is not None and not self._defer_cancellation_check(task=task, call=patched_call):
                    ctx.check_cancelled()
            result = self.hook_manager.apply_after_tool_call(
                task=task,
                cycle_index=context.cycle_index,
                call=patched_call,
                context=replace(
                    context,
                    tool_call_id=patched_call.id,
                    tool_name=patched_call.name,
                    arguments=dict(patched_call.arguments),
                ),
                result=result,
            )
            if self._needs_tool_call_id(result.tool_call_id):
                result.tool_call_id = patched_call.id
            self._apply_tool_use_behavior(task=task, call=patched_call, result=result)

            cycle_record.tool_results.append(result)
            messages.append(result.to_tool_message())
            image_notification = self._build_image_notification(
                result=result,
                include_image=task.native_multimodal,
            )
            if image_notification is not None:
                image_notifications.append(image_notification)
            if on_tool_result is not None:
                on_tool_result(call, result)
            if ctx is not None and self._defer_cancellation_check(task=task, call=patched_call):
                try:
                    ctx.check_cancelled()
                except CancelledError as exc:
                    raise _ConfiguredSubTaskCancelledError(str(exc)) from exc

            if result.directive in (ToolDirective.WAIT_USER, ToolDirective.FINISH):
                latest_directive_result = result
                skip_code = "skipped_due_to_wait_user" if result.directive == ToolDirective.WAIT_USER else "skipped_due_to_finish"
                skip_message = (
                    "Tool skipped because a previous tool requested user input."
                    if result.directive == ToolDirective.WAIT_USER
                    else "Tool skipped because a previous tool finished the task."
                )
                for skipped_call in tool_calls[index + 1 :]:
                    skipped = self._build_skipped_result(
                        skipped_call,
                        error_code=skip_code,
                        message=skip_message,
                    )
                    cycle_record.tool_results.append(skipped)
                    messages.append(skipped.to_tool_message())
                    if on_tool_result is not None:
                        on_tool_result(skipped_call, skipped)
                break

            if interruption_provider is not None:
                pending_messages = interruption_provider()
                if pending_messages:
                    interruption_messages.extend(pending_messages)
                    for skipped_call in tool_calls[index + 1 :]:
                        skipped = self._build_skipped_result(
                            skipped_call,
                            error_code="skipped_due_to_steering",
                            message="Tool skipped due to queued steering message.",
                        )
                        cycle_record.tool_results.append(skipped)
                        messages.append(skipped.to_tool_message())
                        if on_tool_result is not None:
                            on_tool_result(skipped_call, skipped)
                    break

        if image_notifications:
            messages.extend(image_notifications)

        return ToolRunOutcome(
            directive_result=latest_directive_result,
            interruption_messages=interruption_messages,
        )

    @staticmethod
    def _defer_cancellation_check(*, task: AgentTask, call: ToolCall) -> bool:
        return bool(task.sub_agents) and call.name == CREATE_SUB_TASK_TOOL_NAME

    @staticmethod
    def _apply_tool_use_behavior(
        *,
        task: AgentTask,
        call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        if result.directive != ToolDirective.CONTINUE or result.status_code != ToolResultStatus.SUCCESS:
            return
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        behavior = str(metadata.get("_vv_agent_tool_use_behavior") or "run_llm_again")
        should_stop = behavior == "stop_on_first_tool"
        if behavior == "stop_at_tool_names":
            raw_names = metadata.get("_vv_agent_stop_at_tool_names")
            stop_names = {str(name) for name in raw_names} if isinstance(raw_names, list) else set()
            should_stop = call.name in stop_names
        if should_stop:
            result.directive = ToolDirective.FINISH

    @staticmethod
    def _needs_tool_call_id(value: str | None) -> bool:
        if value is None:
            return True
        stripped = value.strip()
        return stripped == "" or stripped == "pending"

    @staticmethod
    def _build_image_notification(*, result: ToolExecutionResult, include_image: bool) -> Message | None:
        if not include_image:
            return None
        if result.image_url:
            # For data URLs keep text empty to avoid duplicating base64 payload as plain text.
            content = f"[Image loaded] {result.image_path}" if result.image_path else ""
            return Message(
                role="user",
                content=content,
                image_url=result.image_url,
            )
        if result.image_path:
            return Message(role="user", content=f"[Image loaded] {result.image_path}")
        return None

    @staticmethod
    def _build_skipped_result(
        call: ToolCall,
        *,
        error_code: str,
        message: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=call.id,
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code=error_code,
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": error_code,
                },
                ensure_ascii=False,
            ),
        )
