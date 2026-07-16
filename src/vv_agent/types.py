from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, cast

from vv_agent.budget import BudgetExhaustion, BudgetUsageSnapshot
from vv_agent.model_settings import ModelSettings

Role = Literal["system", "user", "assistant", "tool"]
NoToolPolicy = Literal["continue", "wait_user", "finish"]
_NO_TOOL_POLICIES = frozenset({"continue", "wait_user", "finish"})
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_U8 = (1 << 8) - 1


def _trim_portable_whitespace(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and (value[start].isspace() or "\x1c" <= value[start] <= "\x1f"):
        start += 1
    while end > start and (value[end - 1].isspace() or "\x1c" <= value[end - 1] <= "\x1f"):
        end -= 1
    return value[start:end]


class AgentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAIT_USER = "wait_user"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_CYCLES = "max_cycles"


class CompletionReason(StrEnum):
    TOOL_FINISH = "tool_finish"
    NO_TOOL_FINISH = "no_tool_finish"
    STOP_ON_FIRST_TOOL = "stop_on_first_tool"
    STOP_AT_TOOL_NAME = "stop_at_tool_name"
    WAIT_USER = "wait_user"
    MAX_CYCLES = "max_cycles"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


def _validate_no_tool_policy(value: object, field_name: str) -> NoToolPolicy | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _NO_TOOL_POLICIES:
        supported = ", ".join(sorted(_NO_TOOL_POLICIES))
        raise ValueError(f"{field_name} must be one of: {supported}")
    return cast(NoToolPolicy, value)


class ToolDirective(StrEnum):
    CONTINUE = "continue"
    WAIT_USER = "wait_user"
    FINISH = "finish"


class ToolResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    WAIT_RESPONSE = "WAIT_RESPONSE"
    RUNNING = "RUNNING"
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
    metadata: dict[str, Any] = field(default_factory=dict)

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
        if self.metadata:
            d["metadata"] = dict(self.metadata)
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
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.extra_content is not None:
            payload["extra_content"] = deepcopy(self.extra_content)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        raw_extra_content = data.get("extra_content")
        extra_content = deepcopy(raw_extra_content) if isinstance(raw_extra_content, dict) else None
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=dict(data.get("arguments", {})),
            extra_content=extra_content,
        )

class UsageSource(StrEnum):
    PROVIDER_REPORTED = "provider_reported"
    ESTIMATED = "estimated"
    ACCOUNTING_MISSING = "accounting_missing"


class CacheUsageStatus(StrEnum):
    PROVIDER_REPORTED = "provider_reported"
    ACCOUNTING_MISSING = "accounting_missing"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class CacheUsage:
    status: CacheUsageStatus = CacheUsageStatus.ACCOUNTING_MISSING
    read_tokens: int | None = None
    write_tokens: int | None = None
    uncached_input_tokens: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CacheUsageStatus):
            self.status = CacheUsageStatus(self.status)
        for name in ("read_tokens", "write_tokens", "uncached_input_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"cache usage {name} must be a non-negative integer or None")
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError("cache usage source must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "read_tokens": self.read_tokens,
            "write_tokens": self.write_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheUsage:
        return cls(
            status=CacheUsageStatus(data.get("status", CacheUsageStatus.ACCOUNTING_MISSING.value)),
            read_tokens=_optional_non_negative_int(data.get("read_tokens")),
            write_tokens=_optional_non_negative_int(data.get("write_tokens")),
            uncached_input_tokens=_optional_non_negative_int(data.get("uncached_input_tokens")),
            source=data.get("source") if isinstance(data.get("source"), str) else None,
        )

def _aggregate_cache_usage(observations: list[CacheUsage]) -> CacheUsage:
    if not observations:
        return CacheUsage()
    statuses = {observation.status for observation in observations}
    if statuses == {CacheUsageStatus.PROVIDER_REPORTED}:
        status = CacheUsageStatus.PROVIDER_REPORTED
    elif statuses == {CacheUsageStatus.UNSUPPORTED}:
        status = CacheUsageStatus.UNSUPPORTED
    else:
        status = CacheUsageStatus.ACCOUNTING_MISSING

    def complete_sum(name: str) -> int | None:
        values = [getattr(observation, name) for observation in observations]
        if status is not CacheUsageStatus.PROVIDER_REPORTED or any(value is None for value in values):
            return None
        return sum(cast(int, value) for value in values)

    return CacheUsage(
        status=status,
        read_tokens=complete_sum("read_tokens"),
        write_tokens=complete_sum("write_tokens"),
        uncached_input_tokens=complete_sum("uncached_input_tokens"),
        source="aggregate",
    )


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cache usage readings must be non-negative integers or None")
    return value


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
    usage_source: UsageSource = UsageSource.ACCOUNTING_MISSING
    cache_usage: CacheUsage = field(default_factory=CacheUsage)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.usage_source, UsageSource):
            self.usage_source = UsageSource(self.usage_source)
        if not isinstance(self.cache_usage, CacheUsage):
            raise TypeError("cache_usage must be a CacheUsage instance")

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
                self.usage_source is not UsageSource.ACCOUNTING_MISSING,
                self.cache_usage.status is not CacheUsageStatus.ACCOUNTING_MISSING,
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
            "usage_source": self.usage_source.value,
            "cache_usage": self.cache_usage.to_dict(),
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        cache_usage = data.get("cache_usage")
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            cached_tokens=int(data.get("cached_tokens", 0) or 0),
            reasoning_tokens=int(data.get("reasoning_tokens", 0) or 0),
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cache_creation_tokens=int(data.get("cache_creation_tokens", 0) or 0),
            usage_source=UsageSource(data.get("usage_source", UsageSource.ACCOUNTING_MISSING.value)),
            cache_usage=CacheUsage.from_dict(cache_usage if isinstance(cache_usage, dict) else {}),
            raw=dict(data.get("raw", {})) if isinstance(data.get("raw", {}), dict) else {},
        )


@dataclass(slots=True)
class CycleTokenUsage:
    cycle_index: int
    usage: TokenUsage

    def to_dict(self) -> dict[str, Any]:
        payload = self.usage.to_dict()
        payload["cycle_index"] = self.cycle_index
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleTokenUsage:
        return cls(
            cycle_index=int(data.get("cycle_index", 0) or 0),
            usage=TokenUsage.from_dict(data),
        )


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
    cache_usage: CacheUsage = field(default_factory=CacheUsage)
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
        self.cache_usage = _aggregate_cache_usage([item.usage.cache_usage for item in self.cycles])

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
            "cache_usage": self.cache_usage.to_dict(),
            "cycles": [item.to_dict() for item in self.cycles],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTokenUsage:
        cache_usage = data.get("cache_usage")
        usage = cls(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            cached_tokens=int(data.get("cached_tokens", 0) or 0),
            reasoning_tokens=int(data.get("reasoning_tokens", 0) or 0),
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cache_creation_tokens=int(data.get("cache_creation_tokens", 0) or 0),
            cache_usage=CacheUsage.from_dict(cache_usage if isinstance(cache_usage, dict) else {}),
        )
        cycles = data.get("cycles", [])
        if isinstance(cycles, list):
            usage.cycles = [CycleTokenUsage.from_dict(item) for item in cycles if isinstance(item, dict)]
        return usage


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
    _planned_tool_names: tuple[str, ...] | None = field(default=None, repr=False, compare=False)

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
            token_usage=TokenUsage.from_dict(data.get("token_usage", {})),
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

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("sub-agent model must be a string")
        normalized_model = _trim_portable_whitespace(self.model)
        if not normalized_model:
            raise ValueError("sub-agent model cannot be empty")
        if not isinstance(self.description, str):
            raise TypeError("sub-agent description must be a string")
        if self.backend is not None and not isinstance(self.backend, str):
            raise TypeError("sub-agent backend must be a string or None")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TypeError("sub-agent system_prompt must be a string or None")
        if self.system_prompt is not None and not _trim_portable_whitespace(self.system_prompt):
            raise ValueError("sub-agent system_prompt cannot be empty when provided")
        if isinstance(self.max_cycles, bool) or not isinstance(self.max_cycles, int):
            raise TypeError("sub-agent max_cycles must be an integer")
        if not 0 <= self.max_cycles <= _MAX_U32:
            raise ValueError("sub-agent max_cycles must be in the u32 range")
        if not isinstance(self.exclude_tools, list) or not all(isinstance(tool_name, str) for tool_name in self.exclude_tools):
            raise TypeError("sub-agent exclude_tools must be a list of strings")
        if not isinstance(self.metadata, dict) or not all(isinstance(key, str) for key in self.metadata):
            raise TypeError("sub-agent metadata must be a dict with string keys")
        self.model = normalized_model
        self.exclude_tools = list(self.exclude_tools)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "description": self.description,
            "backend": self.backend,
            "system_prompt": self.system_prompt,
            "max_cycles": self.max_cycles,
            "exclude_tools": list(self.exclude_tools),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubAgentConfig:
        if not isinstance(data, dict):
            raise TypeError("SubAgentConfig payload must be a dict")
        return cls(
            model=data["model"],
            description=data.get("description", ""),
            backend=data.get("backend"),
            system_prompt=data.get("system_prompt"),
            max_cycles=data.get("max_cycles", 8),
            exclude_tools=data.get("exclude_tools", []),
            metadata=data.get("metadata", {}),
        )


def _agent_task_required_string(data: dict[str, Any], field_name: str) -> str:
    value = data[field_name]
    if not isinstance(value, str):
        raise TypeError(f"AgentTask field {field_name!r} must be a string")
    return value


def _agent_task_integer(
    data: dict[str, Any],
    field_name: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = data.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"AgentTask field {field_name!r} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"AgentTask field {field_name!r} is outside the supported range")
    return value


def _agent_task_bool(data: dict[str, Any], field_name: str, *, default: bool) -> bool:
    value = data.get(field_name, default)
    if not isinstance(value, bool):
        raise TypeError(f"AgentTask field {field_name!r} must be a boolean")
    return value


def _agent_task_no_tool_policy(data: dict[str, Any]) -> NoToolPolicy:
    value = data.get("no_tool_policy", "continue")
    if not isinstance(value, str):
        raise TypeError("AgentTask field 'no_tool_policy' must be a string")
    if value not in {"continue", "wait_user", "finish"}:
        raise ValueError(f"Unknown AgentTask no_tool_policy: {value}")
    return cast(NoToolPolicy, value)


def _agent_task_optional_string(data: dict[str, Any], field_name: str) -> str | None:
    value = data.get(field_name)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"AgentTask field {field_name!r} must be a string or None")
    return value


def _agent_task_string_list(data: dict[str, Any], field_name: str) -> list[str]:
    value = data.get(field_name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"AgentTask field {field_name!r} must be a list of strings")
    return list(cast(list[str], value))


def _agent_task_metadata(data: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = data.get(field_name, {})
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"AgentTask field {field_name!r} must be a dict with string keys")
    return dict(value)


def _agent_task_sub_agents(data: dict[str, Any]) -> dict[str, SubAgentConfig]:
    value = data.get("sub_agents", {})
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise TypeError("AgentTask field 'sub_agents' must be a dict with string keys")

    sub_agents: dict[str, SubAgentConfig] = {}
    for name, payload in value.items():
        if not isinstance(payload, dict):
            raise TypeError(f"AgentTask sub-agent {name!r} must be a dict")
        sub_agents[name] = SubAgentConfig.from_dict(payload)
    return sub_agents


def _agent_task_model_settings(data: dict[str, Any]) -> ModelSettings | None:
    value = data.get("model_settings")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("AgentTask field 'model_settings' must be a dict or None")
    return ModelSettings.from_dict(value)


def _agent_task_messages(data: dict[str, Any]) -> list[Message]:
    value = data.get("initial_messages", [])
    if not isinstance(value, list):
        raise TypeError("AgentTask field 'initial_messages' must be a list")

    messages: list[Message] = []
    for index, payload in enumerate(value):
        if not isinstance(payload, dict):
            raise TypeError(f"AgentTask initial_messages[{index}] must be a dict")
        typed_payload = cast(dict[str, Any], payload)
        role = typed_payload.get("role")
        if not isinstance(role, str):
            raise TypeError(f"AgentTask initial_messages[{index}].role must be a string")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unknown AgentTask initial_messages[{index}].role: {role}")
        if "content" in typed_payload and not isinstance(typed_payload["content"], str):
            raise TypeError(f"AgentTask initial_messages[{index}].content must be a string")
        for field_name in ("name", "tool_call_id", "reasoning_content", "image_url"):
            field_value = typed_payload.get(field_name)
            if field_value is not None and not isinstance(field_value, str):
                raise TypeError(f"AgentTask initial_messages[{index}].{field_name} must be a string or None")
        if "tool_calls" in typed_payload:
            tool_calls = typed_payload["tool_calls"]
            if not isinstance(tool_calls, list) or not all(isinstance(tool_call, dict) for tool_call in tool_calls):
                raise TypeError(f"AgentTask initial_messages[{index}].tool_calls must be a list of dicts")
        if "metadata" in typed_payload:
            metadata = typed_payload["metadata"]
            if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
                raise TypeError(
                    f"AgentTask initial_messages[{index}].metadata must be a dict with string keys"
                )
        messages.append(Message.from_dict(typed_payload))
    return messages


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
    model_settings: ModelSettings | None = None
    initial_messages: list[Message] = field(default_factory=list)
    initial_shared_state: dict[str, Any] = field(default_factory=dict)
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
            "sub_agents": {name: config.to_dict() for name, config in self.sub_agents.items()},
            "extra_tool_names": list(self.extra_tool_names),
            "exclude_tools": list(self.exclude_tools),
            "model_settings": self.model_settings.to_dict() if self.model_settings is not None else None,
            "initial_messages": [message.to_dict() for message in self.initial_messages],
            "initial_shared_state": deepcopy(self.initial_shared_state),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTask:
        """Restore the public dict form; only the four constructor fields are required."""
        if not isinstance(data, dict):
            raise TypeError("AgentTask payload must be a dict")
        return cls(
            task_id=_agent_task_required_string(data, "task_id"),
            model=_agent_task_required_string(data, "model"),
            system_prompt=_agent_task_required_string(data, "system_prompt"),
            user_prompt=_agent_task_required_string(data, "user_prompt"),
            max_cycles=_agent_task_integer(data, "max_cycles", default=8, maximum=_MAX_U32),
            memory_compact_threshold=_agent_task_integer(
                data,
                "memory_compact_threshold",
                default=128_000,
                maximum=_MAX_U64,
            ),
            memory_threshold_percentage=_agent_task_integer(
                data,
                "memory_threshold_percentage",
                default=90,
                maximum=_MAX_U8,
            ),
            no_tool_policy=_agent_task_no_tool_policy(data),
            allow_interruption=_agent_task_bool(data, "allow_interruption", default=True),
            use_workspace=_agent_task_bool(data, "use_workspace", default=True),
            has_sub_agents=_agent_task_bool(data, "has_sub_agents", default=False),
            sub_agents=_agent_task_sub_agents(data),
            agent_type=_agent_task_optional_string(data, "agent_type"),
            native_multimodal=_agent_task_bool(data, "native_multimodal", default=False),
            extra_tool_names=_agent_task_string_list(data, "extra_tool_names"),
            exclude_tools=_agent_task_string_list(data, "exclude_tools"),
            model_settings=_agent_task_model_settings(data),
            initial_messages=_agent_task_messages(data),
            initial_shared_state=deepcopy(_agent_task_metadata(data, "initial_shared_state")),
            metadata=_agent_task_metadata(data, "metadata"),
        )


@dataclass(slots=True)
class SubTaskRequest:
    agent_name: str
    task_description: str
    output_requirements: str = ""
    include_main_summary: bool = False
    exclude_files_pattern: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _last_assistant_output(cycles: list[CycleRecord]) -> str | None:
    for cycle in reversed(cycles):
        if cycle.assistant_message.strip():
            return cycle.assistant_message
    return None


@dataclass(slots=True)
class SubTaskOutcome:
    task_id: str
    agent_name: str
    status: AgentStatus
    session_id: str | None = None
    final_answer: str | None = None
    wait_reason: str | None = None
    error: str | None = None
    error_code: str | None = None
    cycles: int = 0
    todo_list: list[dict[str, Any]] = field(default_factory=list)
    resolved: dict[str, str] = field(default_factory=dict)
    completion_reason: CompletionReason | None = None
    completion_tool_name: str | None = None
    partial_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.completion_reason is not None:
            payload["completion_reason"] = self.completion_reason.value
        if self.completion_tool_name is not None:
            payload["completion_tool_name"] = self.completion_tool_name
        if self.partial_output is not None:
            payload["partial_output"] = self.partial_output
        return payload


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
    completion_reason: CompletionReason | None = None
    completion_tool_name: str | None = None
    partial_output: str | None = None
    budget_usage: BudgetUsageSnapshot | None = None
    budget_exhaustion: BudgetExhaustion | None = None

    @property
    def todo_list(self) -> list[dict[str, Any]]:
        todo = self.shared_state.get("todo_list")
        if isinstance(todo, list):
            return todo
        return []

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status.value,
            "completion_reason": self.completion_reason.value if self.completion_reason is not None else None,
            "completion_tool_name": self.completion_tool_name,
            "partial_output": self.partial_output,
            "messages": [m.to_dict() for m in self.messages],
            "cycles": [c.to_dict() for c in self.cycles],
            "final_answer": self.final_answer,
            "wait_reason": self.wait_reason,
            "error": self.error,
            "shared_state": self.shared_state,
            "token_usage": self.token_usage.to_dict(),
        }
        if self.budget_usage is not None:
            payload["budget_usage"] = self.budget_usage.to_dict()
        if self.budget_exhaustion is not None:
            payload["budget_exhaustion"] = self.budget_exhaustion.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResult:
        token_usage_raw = data.get("token_usage")
        token_usage = TaskTokenUsage()
        if isinstance(token_usage_raw, dict):
            token_usage = TaskTokenUsage.from_dict(token_usage_raw)
        completion_reason_raw = data.get("completion_reason")
        if completion_reason_raw is not None and not isinstance(completion_reason_raw, str):
            raise TypeError("AgentResult field 'completion_reason' must be a string or None")
        completion_tool_name = data.get("completion_tool_name")
        if completion_tool_name is not None and not isinstance(completion_tool_name, str):
            raise TypeError("AgentResult field 'completion_tool_name' must be a string or None")
        partial_output = data.get("partial_output")
        if partial_output is not None and not isinstance(partial_output, str):
            raise TypeError("AgentResult field 'partial_output' must be a string or None")
        budget_usage_raw = data.get("budget_usage")
        if budget_usage_raw is not None and not isinstance(budget_usage_raw, dict):
            raise TypeError("AgentResult field 'budget_usage' must be an object or None")
        budget_exhaustion_raw = data.get("budget_exhaustion")
        if budget_exhaustion_raw is not None and not isinstance(budget_exhaustion_raw, dict):
            raise TypeError("AgentResult field 'budget_exhaustion' must be an object or None")
        return cls(
            status=AgentStatus(data["status"]),
            completion_reason=(CompletionReason(completion_reason_raw) if completion_reason_raw is not None else None),
            completion_tool_name=completion_tool_name,
            partial_output=partial_output,
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            cycles=[CycleRecord.from_dict(c) for c in data.get("cycles", [])],
            final_answer=data.get("final_answer"),
            wait_reason=data.get("wait_reason"),
            error=data.get("error"),
            shared_state=data.get("shared_state", {}),
            token_usage=token_usage,
            budget_usage=(BudgetUsageSnapshot.from_dict(budget_usage_raw) if budget_usage_raw is not None else None),
            budget_exhaustion=(
                BudgetExhaustion.from_dict(budget_exhaustion_raw) if budget_exhaustion_raw is not None else None
            ),
        )
