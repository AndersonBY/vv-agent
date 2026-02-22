from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vv_agent.runtime.hooks import RuntimeHookManager
from vv_agent.tools import ToolContext, ToolRegistry
from vv_agent.tools.dispatcher import dispatch_tool_call
from vv_agent.types import AgentTask, CycleRecord, Message, ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext


@dataclass(slots=True)
class ToolRunOutcome:
    directive_result: ToolExecutionResult | None = None
    interruption_messages: list[Message] = field(default_factory=list)


class ToolCallRunner:
    def __init__(self, *, tool_registry: ToolRegistry, hook_manager: RuntimeHookManager | None = None) -> None:
        self.tool_registry = tool_registry
        self.hook_manager = hook_manager or RuntimeHookManager()

    def run(
        self,
        *,
        task: AgentTask,
        tool_calls: list[ToolCall],
        context: ToolContext,
        messages: list[Message],
        cycle_record: CycleRecord,
        interruption_provider: Callable[[], list[Message]] | None = None,
        on_tool_result: Callable[[ToolCall, ToolExecutionResult], None] | None = None,
        ctx: ExecutionContext | None = None,
    ) -> ToolRunOutcome:
        latest_directive_result: ToolExecutionResult | None = None
        interruption_messages: list[Message] = []

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
                result = dispatch_tool_call(
                    registry=self.tool_registry,
                    context=context,
                    call=patched_call,
                )
            result = self.hook_manager.apply_after_tool_call(
                task=task,
                cycle_index=context.cycle_index,
                call=patched_call,
                context=context,
                result=result,
            )
            if self._needs_tool_call_id(result.tool_call_id):
                result.tool_call_id = patched_call.id

            cycle_record.tool_results.append(result)
            messages.append(result.to_tool_message())
            self._append_image_notification(result=result, messages=messages)
            if on_tool_result is not None:
                on_tool_result(call, result)

            if result.directive in (ToolDirective.WAIT_USER, ToolDirective.FINISH):
                latest_directive_result = result
                skip_code = (
                    "skipped_due_to_wait_user"
                    if result.directive == ToolDirective.WAIT_USER
                    else "skipped_due_to_finish"
                )
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

        return ToolRunOutcome(
            directive_result=latest_directive_result,
            interruption_messages=interruption_messages,
        )

    @staticmethod
    def _needs_tool_call_id(value: str | None) -> bool:
        if value is None:
            return True
        stripped = value.strip()
        return stripped == "" or stripped == "pending"

    @staticmethod
    def _append_image_notification(*, result: ToolExecutionResult, messages: list[Message]) -> None:
        if result.image_url:
            image_ref = result.image_path or result.image_url
            messages.append(
                Message(
                    role="user",
                    content=f"[Image loaded] {image_ref}",
                    image_url=result.image_url,
                )
            )
        elif result.image_path:
            messages.append(Message(role="user", content=f"[Image loaded] {result.image_path}"))

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
