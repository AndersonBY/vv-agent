from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from v_agent.tools import ToolContext
from v_agent.types import AgentTask, LLMResponse, Message, ToolCall, ToolExecutionResult


@dataclass(slots=True)
class BeforeMemoryCompactEvent:
    task: AgentTask
    cycle_index: int
    messages: list[Message]
    shared_state: dict[str, Any]


@dataclass(slots=True)
class BeforeLLMEvent:
    task: AgentTask
    cycle_index: int
    messages: list[Message]
    tool_schemas: list[dict[str, Any]]
    shared_state: dict[str, Any]


@dataclass(slots=True)
class AfterLLMEvent:
    task: AgentTask
    cycle_index: int
    messages: list[Message]
    tool_schemas: list[dict[str, Any]]
    response: LLMResponse
    shared_state: dict[str, Any]


@dataclass(slots=True)
class BeforeToolCallEvent:
    task: AgentTask
    cycle_index: int
    call: ToolCall
    context: ToolContext


@dataclass(slots=True)
class AfterToolCallEvent:
    task: AgentTask
    cycle_index: int
    call: ToolCall
    context: ToolContext
    result: ToolExecutionResult


@dataclass(slots=True)
class BeforeLLMPatch:
    messages: list[Message] | None = None
    tool_schemas: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class BeforeToolCallPatch:
    call: ToolCall | None = None
    result: ToolExecutionResult | None = None


class RuntimeHook(Protocol):
    """Marker protocol for runtime hooks.

    Hook objects are duck-typed at runtime; any subset of hook methods is accepted.
    """


class BaseRuntimeHook:
    """Typed base class for runtime hooks.

    Subclass this and override only the methods you need.
    """

    def before_memory_compact(self, event: BeforeMemoryCompactEvent) -> list[Message] | None:
        del event
        return None

    def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch | None:
        del event
        return None

    def after_llm(self, event: AfterLLMEvent) -> LLMResponse | None:
        del event
        return None

    def before_tool_call(
        self,
        event: BeforeToolCallEvent,
    ) -> BeforeToolCallPatch | ToolCall | ToolExecutionResult | None:
        del event
        return None

    def after_tool_call(self, event: AfterToolCallEvent) -> ToolExecutionResult | None:
        del event
        return None


@dataclass(slots=True)
class RuntimeHookManager:
    hooks: list[RuntimeHook] = field(default_factory=list)

    def has_hooks(self) -> bool:
        return bool(self.hooks)

    def apply_before_memory_compact(
        self,
        *,
        task: AgentTask,
        cycle_index: int,
        messages: list[Message],
        shared_state: dict[str, Any],
    ) -> list[Message]:
        current = list(messages)
        if not self.hooks:
            return current

        for hook in self.hooks:
            handler = getattr(hook, "before_memory_compact", None)
            if not callable(handler):
                continue
            replacement = handler(
                BeforeMemoryCompactEvent(
                    task=task,
                    cycle_index=cycle_index,
                    messages=list(current),
                    shared_state=shared_state,
                )
            )
            if replacement is not None:
                current = list(replacement)
        return current

    def apply_before_llm(
        self,
        *,
        task: AgentTask,
        cycle_index: int,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        shared_state: dict[str, Any],
    ) -> tuple[list[Message], list[dict[str, Any]]]:
        current_messages = list(messages)
        current_tool_schemas = list(tool_schemas)
        if not self.hooks:
            return current_messages, current_tool_schemas

        for hook in self.hooks:
            handler = getattr(hook, "before_llm", None)
            if not callable(handler):
                continue
            patch = handler(
                BeforeLLMEvent(
                    task=task,
                    cycle_index=cycle_index,
                    messages=list(current_messages),
                    tool_schemas=list(current_tool_schemas),
                    shared_state=shared_state,
                )
            )
            if patch is None:
                continue
            if patch.messages is not None:
                current_messages = list(patch.messages)
            if patch.tool_schemas is not None:
                current_tool_schemas = list(patch.tool_schemas)
        return current_messages, current_tool_schemas

    def apply_after_llm(
        self,
        *,
        task: AgentTask,
        cycle_index: int,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        response: LLMResponse,
        shared_state: dict[str, Any],
    ) -> LLMResponse:
        current = response
        if not self.hooks:
            return current

        for hook in self.hooks:
            handler = getattr(hook, "after_llm", None)
            if not callable(handler):
                continue
            patched = handler(
                AfterLLMEvent(
                    task=task,
                    cycle_index=cycle_index,
                    messages=list(messages),
                    tool_schemas=list(tool_schemas),
                    response=current,
                    shared_state=shared_state,
                )
            )
            if patched is not None:
                current = patched
        return current

    def apply_before_tool_call(
        self,
        *,
        task: AgentTask,
        cycle_index: int,
        call: ToolCall,
        context: ToolContext,
    ) -> tuple[ToolCall, ToolExecutionResult | None]:
        current_call = call
        short_circuit: ToolExecutionResult | None = None
        if not self.hooks:
            return current_call, short_circuit

        for hook in self.hooks:
            handler = getattr(hook, "before_tool_call", None)
            if not callable(handler):
                continue
            patch = handler(
                BeforeToolCallEvent(
                    task=task,
                    cycle_index=cycle_index,
                    call=current_call,
                    context=context,
                )
            )
            if patch is None:
                continue
            if isinstance(patch, ToolExecutionResult):
                short_circuit = patch
                break
            if isinstance(patch, ToolCall):
                current_call = patch
                continue
            if isinstance(patch, BeforeToolCallPatch):
                if patch.call is not None:
                    current_call = patch.call
                if patch.result is not None:
                    short_circuit = patch.result
                    break
        return current_call, short_circuit

    def apply_after_tool_call(
        self,
        *,
        task: AgentTask,
        cycle_index: int,
        call: ToolCall,
        context: ToolContext,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        current = result
        if not self.hooks:
            return current

        for hook in self.hooks:
            handler = getattr(hook, "after_tool_call", None)
            if not callable(handler):
                continue
            patched = handler(
                AfterToolCallEvent(
                    task=task,
                    cycle_index=cycle_index,
                    call=call,
                    context=context,
                    result=current,
                )
            )
            if patched is not None:
                current = patched
        return current
