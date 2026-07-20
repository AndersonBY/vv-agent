from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from vv_agent.checkpoint import utf16_sort_key
from vv_agent.types import CompletionReason, CycleRecord, Message, TaskTokenUsage

AFTER_CYCLE_CONTROL_STATE_KEY = "_vv_agent_after_cycle_control"
AFTER_CYCLE_CONTROL_SCHEMA = "vv-agent.after-cycle-control.v1"
MAX_STEERING_MESSAGES = 32
MAX_STEERING_MESSAGE_UTF8_BYTES = 16_384
MAX_TOTAL_STEERING_UTF8_BYTES = 65_536
MAX_DISALLOW_TOOLS = 1_024
MAX_TOOL_NAME_UTF8_BYTES = 256
MAX_STOP_CODE_ASCII_BYTES = 128
MAX_STOP_MESSAGE_UTF8_BYTES = 4_096
_STOP_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


class AfterCycleAction(StrEnum):
    CONTINUE = "continue"
    STEER = "steer"
    STOP_NON_SUCCESS = "stop_non_success"


class NativeCycleOutcomeKind(StrEnum):
    CONTINUE = "continue"
    COMPLETED = "completed"
    WAIT_USER = "wait_user"
    MAX_CYCLES = "max_cycles"


@dataclass(frozen=True, slots=True)
class NativeCycleOutcome:
    kind: NativeCycleOutcomeKind
    completion_reason: CompletionReason | None = None
    completion_tool_name: str | None = None
    steer_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NativeCycleOutcomeKind):
            object.__setattr__(self, "kind", NativeCycleOutcomeKind(self.kind))
        if self.completion_reason is not None and not isinstance(
            self.completion_reason,
            CompletionReason,
        ):
            object.__setattr__(
                self,
                "completion_reason",
                CompletionReason(self.completion_reason),
            )
        if self.completion_tool_name is not None and (
            not isinstance(self.completion_tool_name, str)
            or not self.completion_tool_name.strip()
        ):
            raise ValueError("completion_tool_name must be a non-empty string or None")
        if not isinstance(self.steer_allowed, bool):
            raise TypeError("steer_allowed must be a boolean")


@dataclass(frozen=True, slots=True)
class AfterCycleSnapshot:
    task_id: str
    cycle_index: int
    max_cycles: int
    remaining_cycles: int
    cycle: CycleRecord
    messages: tuple[Message, ...]
    shared_state: Mapping[str, Any]
    cumulative_token_usage: TaskTokenUsage
    available_tool_names: tuple[str, ...]
    disallowed_tool_names: tuple[str, ...]
    native_outcome: NativeCycleOutcome

    @classmethod
    def capture(
        cls,
        *,
        task_id: str,
        cycle_index: int,
        max_cycles: int,
        cycle: CycleRecord,
        messages: list[Message],
        shared_state: dict[str, Any],
        cumulative_token_usage: TaskTokenUsage,
        available_tool_names: list[str],
        disallowed_tool_names: list[str],
        native_outcome: NativeCycleOutcome,
    ) -> AfterCycleSnapshot:
        return cls(
            task_id=task_id,
            cycle_index=cycle_index,
            max_cycles=max_cycles,
            remaining_cycles=max(0, max_cycles - cycle_index),
            cycle=deepcopy(cycle),
            messages=tuple(deepcopy(messages)),
            shared_state=MappingProxyType(deepcopy(shared_state)),
            cumulative_token_usage=deepcopy(cumulative_token_usage),
            available_tool_names=tuple(available_tool_names),
            disallowed_tool_names=tuple(disallowed_tool_names),
            native_outcome=native_outcome,
        )


@dataclass(frozen=True, slots=True)
class AfterCycleStop:
    code: str
    message: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not self.code.isascii()
            or _STOP_CODE_RE.fullmatch(self.code) is None
            or len(self.code.encode("ascii")) > MAX_STOP_CODE_ASCII_BYTES
        ):
            raise ValueError("after-cycle stop code is invalid")
        _validate_bounded_text(
            self.message,
            "after-cycle stop message",
            MAX_STOP_MESSAGE_UTF8_BYTES,
        )


@dataclass(frozen=True, slots=True)
class AfterCycleDecision:
    action: AfterCycleAction = AfterCycleAction.CONTINUE
    steering_messages: tuple[str, ...] = ()
    disallow_tools: tuple[str, ...] = ()
    stop: AfterCycleStop | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, AfterCycleAction):
            object.__setattr__(self, "action", AfterCycleAction(self.action))
        messages = tuple(self.steering_messages)
        tools = tuple(self.disallow_tools)
        object.__setattr__(self, "steering_messages", messages)
        object.__setattr__(self, "disallow_tools", tools)
        if len(messages) > MAX_STEERING_MESSAGES:
            raise ValueError("after-cycle steering message count exceeds the limit")
        total_bytes = 0
        for message in messages:
            total_bytes += len(
                _validate_bounded_text(
                    message,
                    "after-cycle steering message",
                    MAX_STEERING_MESSAGE_UTF8_BYTES,
                ).encode("utf-8")
            )
        if total_bytes > MAX_TOTAL_STEERING_UTF8_BYTES:
            raise ValueError("after-cycle steering messages exceed the total byte limit")
        if len(tools) > MAX_DISALLOW_TOOLS:
            raise ValueError("after-cycle disallowed tool count exceeds the limit")
        if len(set(tools)) != len(tools):
            raise ValueError("after-cycle disallowed tools must be unique")
        for tool_name in tools:
            _validate_bounded_text(
                tool_name,
                "after-cycle disallowed tool name",
                MAX_TOOL_NAME_UTF8_BYTES,
            )
        if self.action is AfterCycleAction.CONTINUE:
            if messages or self.stop is not None:
                raise ValueError("continue cannot include steering messages or a stop payload")
        elif self.action is AfterCycleAction.STEER:
            if not messages or self.stop is not None:
                raise ValueError("steer requires messages and cannot include a stop payload")
        elif self.action is AfterCycleAction.STOP_NON_SUCCESS and (
            messages or tools or not isinstance(self.stop, AfterCycleStop)
        ):
            raise ValueError("stop_non_success requires only a typed stop payload")

    @classmethod
    def continue_run(
        cls,
        *,
        disallow_tools: tuple[str, ...] | list[str] = (),
    ) -> AfterCycleDecision:
        return cls(disallow_tools=tuple(disallow_tools))

    @classmethod
    def steer(
        cls,
        messages: tuple[str, ...] | list[str],
        *,
        disallow_tools: tuple[str, ...] | list[str] = (),
    ) -> AfterCycleDecision:
        return cls(
            action=AfterCycleAction.STEER,
            steering_messages=tuple(messages),
            disallow_tools=tuple(disallow_tools),
        )

    @classmethod
    def stop_non_success(cls, *, code: str, message: str) -> AfterCycleDecision:
        return cls(
            action=AfterCycleAction.STOP_NON_SUCCESS,
            stop=AfterCycleStop(code=code, message=message),
        )


@runtime_checkable
class AfterCycleHook(Protocol):
    def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision | None: ...


class AfterCycleHookError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class AfterCycleHookManager:
    hooks: list[AfterCycleHook] = field(default_factory=list)

    def has_hooks(self) -> bool:
        return bool(self.hooks)

    def apply(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
        steering_messages: list[str] = []
        disallow_tools: list[str] = []
        seen_tools: set[str] = set()
        for hook in self.hooks:
            try:
                decision = hook.after_cycle(snapshot)
            except Exception as exc:
                raise AfterCycleHookError(
                    f"after-cycle hook failed: {exc}",
                    code="after_cycle_hook_failed",
                ) from exc
            if decision is None:
                continue
            if not isinstance(decision, AfterCycleDecision):
                raise AfterCycleHookError(
                    "after-cycle hook returned an invalid decision",
                    code="after_cycle_decision_invalid",
                )
            for tool_name in decision.disallow_tools:
                if tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    disallow_tools.append(tool_name)
            if decision.action is AfterCycleAction.STOP_NON_SUCCESS:
                return decision
            if decision.action is AfterCycleAction.STEER:
                steering_messages.extend(decision.steering_messages)
        if steering_messages:
            try:
                return AfterCycleDecision.steer(
                    steering_messages,
                    disallow_tools=disallow_tools,
                )
            except (TypeError, ValueError) as exc:
                raise AfterCycleHookError(
                    f"composed after-cycle decision is invalid: {exc}",
                    code="after_cycle_decision_invalid",
                ) from exc
        try:
            return AfterCycleDecision.continue_run(disallow_tools=disallow_tools)
        except (TypeError, ValueError) as exc:
            raise AfterCycleHookError(
                f"composed after-cycle decision is invalid: {exc}",
                code="after_cycle_decision_invalid",
            ) from exc


def read_after_cycle_disallowed_tools(shared_state: Mapping[str, Any]) -> tuple[str, ...]:
    raw = shared_state.get(AFTER_CYCLE_CONTROL_STATE_KEY)
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "disallowed_tools"}:
        raise AfterCycleHookError(
            "after-cycle control state has missing or unknown fields",
            code="after_cycle_control_state_invalid",
        )
    if raw.get("schema_version") != AFTER_CYCLE_CONTROL_SCHEMA:
        raise AfterCycleHookError(
            "after-cycle control state schema is unsupported",
            code="after_cycle_control_state_invalid",
        )
    values = raw.get("disallowed_tools")
    if not isinstance(values, list):
        raise AfterCycleHookError(
            "after-cycle control disallowed_tools must be an array",
            code="after_cycle_control_state_invalid",
        )
    try:
        decision = AfterCycleDecision.continue_run(disallow_tools=values)
    except (TypeError, ValueError) as exc:
        raise AfterCycleHookError(
            str(exc),
            code="after_cycle_control_state_invalid",
        ) from exc
    sorted_values = tuple(sorted(decision.disallow_tools, key=utf16_sort_key))
    if tuple(values) != sorted_values:
        raise AfterCycleHookError(
            "after-cycle control disallowed_tools must be sorted and unique",
            code="after_cycle_control_state_invalid",
        )
    return sorted_values


def persist_after_cycle_disallowed_tools(
    shared_state: dict[str, Any],
    additional_tools: tuple[str, ...],
) -> tuple[str, ...]:
    existing = set(read_after_cycle_disallowed_tools(shared_state))
    existing.update(additional_tools)
    if not existing:
        return ()
    ordered = tuple(sorted(existing, key=utf16_sort_key))
    shared_state[AFTER_CYCLE_CONTROL_STATE_KEY] = {
        "schema_version": AFTER_CYCLE_CONTROL_SCHEMA,
        "disallowed_tools": list(ordered),
    }
    return ordered


def _validate_bounded_text(value: Any, field_name: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} exceeds {max_bytes} UTF-8 bytes")
    return value
