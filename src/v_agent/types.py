from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
NoToolPolicy = Literal["continue", "wait_user", "finish"]


class AgentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAIT_USER = "wait_user"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_CYCLES = "max_cycles"


class ToolDirective(StrEnum):
    CONTINUE = "continue"
    WAIT_USER = "wait_user"
    FINISH = "finish"


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_openai_message(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolExecutionResult:
    tool_call_id: str
    status: Literal["success", "error"]
    content: str
    directive: ToolDirective = ToolDirective.CONTINUE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_message(self) -> Message:
        return Message(role="tool", content=self.content, tool_call_id=self.tool_call_id)


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CycleRecord:
    index: int
    assistant_message: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    memory_compacted: bool = False


@dataclass(slots=True)
class AgentTask:
    task_id: str
    model: str
    system_prompt: str
    user_prompt: str
    max_cycles: int = 8
    memory_threshold_chars: int = 24_000
    no_tool_policy: NoToolPolicy = "continue"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    status: AgentStatus
    messages: list[Message]
    cycles: list[CycleRecord]
    final_answer: str | None = None
    wait_reason: str | None = None
    error: str | None = None
    shared_state: dict[str, Any] = field(default_factory=dict)

    @property
    def todo_list(self) -> list[dict[str, Any]]:
        todo = self.shared_state.get("todo_list")
        if isinstance(todo, list):
            return todo
        return []
