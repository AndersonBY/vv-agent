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


class ToolResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    WAIT_RESPONSE = "WAIT_RESPONSE"
    RUNNING = "RUNNING"
    BATCH_RUNNING = "BATCH_RUNNING"
    PENDING_COMPRESS = "PENDING_COMPRESS"


class CycleStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    WAIT_RESPONSE = "wait_response"
    FAILED = "failed"


_LEGACY_STATUS_TO_CODE: dict[str, ToolResultStatus] = {
    "success": ToolResultStatus.SUCCESS,
    "error": ToolResultStatus.ERROR,
}

_STATUS_CODE_TO_LEGACY: dict[ToolResultStatus, Literal["success", "error"]] = {
    ToolResultStatus.SUCCESS: "success",
    ToolResultStatus.ERROR: "error",
    ToolResultStatus.WAIT_RESPONSE: "success",
    ToolResultStatus.RUNNING: "success",
    ToolResultStatus.BATCH_RUNNING: "success",
    ToolResultStatus.PENDING_COMPRESS: "success",
}


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None
    image_url: str | None = None

    def to_openai_message(self, *, include_reasoning_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.role == "assistant" and self.tool_calls:
            payload["tool_calls"] = self.tool_calls
            if not self.content:
                payload["content"] = None
        if include_reasoning_content and self.role == "assistant" and self.reasoning_content:
            payload["reasoning_content"] = self.reasoning_content
        if self.role == "user" and self.image_url:
            content_blocks: list[dict[str, Any]] = []
            if self.content:
                content_blocks.append({"type": "text", "text": self.content})
            content_blocks.append({"type": "image_url", "image_url": {"url": self.image_url}})
            payload["content"] = content_blocks
        return payload

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.image_url is not None:
            d["image_url"] = self.image_url
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls"),
            reasoning_content=data.get("reasoning_content"),
            image_url=data.get("image_url"),
        )


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(id=data["id"], name=data["name"], arguments=dict(data.get("arguments", {})))


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def has_usage(self) -> bool:
        return any(
            (
                self.prompt_tokens,
                self.completion_tokens,
                self.total_tokens,
                self.cached_tokens,
                self.reasoning_tokens,
                self.input_tokens,
                self.output_tokens,
                self.cache_creation_tokens,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "raw": dict(self.raw),
        }


@dataclass(slots=True)
class CycleTokenUsage:
    cycle_index: int
    usage: TokenUsage

    def to_dict(self) -> dict[str, Any]:
        payload = self.usage.to_dict()
        payload["cycle_index"] = self.cycle_index
        return payload


@dataclass(slots=True)
class TaskTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cycles: list[CycleTokenUsage] = field(default_factory=list)

    def add_cycle(self, cycle_index: int, usage: TokenUsage) -> None:
        if not usage.has_usage():
            return
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.cached_tokens += usage.cached_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_creation_tokens += usage.cache_creation_tokens
        self.cycles.append(CycleTokenUsage(cycle_index=cycle_index, usage=usage))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cycles": [item.to_dict() for item in self.cycles],
        }


@dataclass(slots=True)
class ToolExecutionResult:
    tool_call_id: str
    content: str
    status: Literal["success", "error"] | None = None
    status_code: ToolResultStatus | None = None
    directive: ToolDirective = ToolDirective.CONTINUE
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    image_url: str | None = None
    image_path: str | None = None

    def __post_init__(self) -> None:
        if self.status is None and self.status_code is None:
            self.status = "success"
            self.status_code = ToolResultStatus.SUCCESS
            return

        if self.status_code is None:
            self.status_code = _LEGACY_STATUS_TO_CODE.get(self.status or "success", ToolResultStatus.SUCCESS)

        if self.status is None:
            self.status = _STATUS_CODE_TO_LEGACY[self.status_code]

    def to_tool_message(self) -> Message:
        return Message(role="tool", content=self.content, tool_call_id=self.tool_call_id)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "directive": self.directive.value,
        }
        if self.status is not None:
            d["status"] = self.status
        if self.status_code is not None:
            d["status_code"] = self.status_code.value
        if self.error_code is not None:
            d["error_code"] = self.error_code
        if self.metadata:
            d["metadata"] = self.metadata
        if self.image_url is not None:
            d["image_url"] = self.image_url
        if self.image_path is not None:
            d["image_path"] = self.image_path
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExecutionResult:
        status_code_raw = data.get("status_code")
        status_code = ToolResultStatus(status_code_raw) if status_code_raw else None
        directive_raw = data.get("directive", "continue")
        return cls(
            tool_call_id=data.get("tool_call_id", ""),
            content=data.get("content", ""),
            status=data.get("status"),
            status_code=status_code,
            directive=ToolDirective(directive_raw),
            error_code=data.get("error_code"),
            metadata=dict(data.get("metadata", {})),
            image_url=data.get("image_url"),
            image_path=data.get("image_path"),
        )


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
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "assistant_message": self.assistant_message,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "memory_compacted": self.memory_compacted,
            "token_usage": self.token_usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleRecord:
        return cls(
            index=data["index"],
            assistant_message=data.get("assistant_message", ""),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_results=[ToolExecutionResult.from_dict(tr) for tr in data.get("tool_results", [])],
            memory_compacted=data.get("memory_compacted", False),
        )


@dataclass(slots=True)
class SubAgentConfig:
    model: str
    description: str
    backend: str | None = None
    system_prompt: str | None = None
    max_cycles: int = 8
    exclude_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentTask:
    task_id: str
    model: str
    system_prompt: str
    user_prompt: str
    max_cycles: int = 8
    memory_compact_threshold: int = 128_000
    memory_threshold_percentage: int = 90
    no_tool_policy: NoToolPolicy = "continue"
    allow_interruption: bool = True
    use_workspace: bool = True
    # Legacy switch; prefer configuring `sub_agents` with concrete entries.
    has_sub_agents: bool = False
    sub_agents: dict[str, SubAgentConfig] = field(default_factory=dict)
    agent_type: str | None = None
    native_multimodal: bool = False
    extra_tool_names: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sub_agents_enabled(self) -> bool:
        return self.has_sub_agents or bool(self.sub_agents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "max_cycles": self.max_cycles,
            "memory_compact_threshold": self.memory_compact_threshold,
            "memory_threshold_percentage": self.memory_threshold_percentage,
            "no_tool_policy": self.no_tool_policy,
            "allow_interruption": self.allow_interruption,
            "use_workspace": self.use_workspace,
            "has_sub_agents": self.has_sub_agents,
            "agent_type": self.agent_type,
            "native_multimodal": self.native_multimodal,
            "extra_tool_names": list(self.extra_tool_names),
            "exclude_tools": list(self.exclude_tools),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTask:
        return cls(
            task_id=data["task_id"],
            model=data["model"],
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"],
            max_cycles=data.get("max_cycles", 8),
            memory_compact_threshold=data.get("memory_compact_threshold", 128_000),
            memory_threshold_percentage=data.get("memory_threshold_percentage", 90),
            no_tool_policy=data.get("no_tool_policy", "continue"),
            allow_interruption=data.get("allow_interruption", True),
            use_workspace=data.get("use_workspace", True),
            has_sub_agents=data.get("has_sub_agents", False),
            agent_type=data.get("agent_type"),
            native_multimodal=data.get("native_multimodal", False),
            extra_tool_names=list(data.get("extra_tool_names", [])),
            exclude_tools=list(data.get("exclude_tools", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SubTaskRequest:
    agent_name: str
    task_description: str
    output_requirements: str = ""
    include_main_summary: bool = False
    exclude_files_pattern: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubTaskOutcome:
    task_id: str
    agent_name: str
    status: AgentStatus
    session_id: str | None = None
    final_answer: str | None = None
    wait_reason: str | None = None
    error: str | None = None
    cycles: int = 0
    todo_list: list[dict[str, Any]] = field(default_factory=list)
    resolved: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "session_id": self.session_id,
            "final_answer": self.final_answer,
            "wait_reason": self.wait_reason,
            "error": self.error,
            "cycles": self.cycles,
            "todo_list": self.todo_list,
            "resolved": self.resolved,
        }


@dataclass(slots=True)
class AgentResult:
    status: AgentStatus
    messages: list[Message]
    cycles: list[CycleRecord]
    final_answer: str | None = None
    wait_reason: str | None = None
    error: str | None = None
    shared_state: dict[str, Any] = field(default_factory=dict)
    token_usage: TaskTokenUsage = field(default_factory=TaskTokenUsage)

    @property
    def todo_list(self) -> list[dict[str, Any]]:
        todo = self.shared_state.get("todo_list")
        if isinstance(todo, list):
            return todo
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "messages": [m.to_dict() for m in self.messages],
            "cycles": [c.to_dict() for c in self.cycles],
            "final_answer": self.final_answer,
            "wait_reason": self.wait_reason,
            "error": self.error,
            "shared_state": self.shared_state,
            "token_usage": self.token_usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResult:
        token_usage_raw = data.get("token_usage")
        token_usage = TaskTokenUsage()
        if isinstance(token_usage_raw, dict):
            token_usage.prompt_tokens = token_usage_raw.get("prompt_tokens", 0)
            token_usage.completion_tokens = token_usage_raw.get("completion_tokens", 0)
            token_usage.total_tokens = token_usage_raw.get("total_tokens", 0)
            token_usage.cached_tokens = token_usage_raw.get("cached_tokens", 0)
            token_usage.reasoning_tokens = token_usage_raw.get("reasoning_tokens", 0)
            token_usage.input_tokens = token_usage_raw.get("input_tokens", 0)
            token_usage.output_tokens = token_usage_raw.get("output_tokens", 0)
            token_usage.cache_creation_tokens = token_usage_raw.get("cache_creation_tokens", 0)
        return cls(
            status=AgentStatus(data["status"]),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            cycles=[CycleRecord.from_dict(c) for c in data.get("cycles", [])],
            final_answer=data.get("final_answer"),
            wait_reason=data.get("wait_reason"),
            error=data.get("error"),
            shared_state=data.get("shared_state", {}),
            token_usage=token_usage,
        )
