from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: Literal["auto", "none", "required", "tool"]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "none", "required", "tool"}:
            raise ValueError(f"Unknown tool_choice mode: {self.mode}")
        if self.mode == "tool":
            if not isinstance(self.name, str) or not self.name.strip():
                raise ValueError("named tool_choice requires a non-empty function name")
        elif self.name is not None:
            raise ValueError(f"tool_choice mode {self.mode!r} cannot include a function name")

    @classmethod
    def auto(cls) -> ToolChoice:
        return cls("auto")

    @classmethod
    def none(cls) -> ToolChoice:
        return cls("none")

    @classmethod
    def required(cls) -> ToolChoice:
        return cls("required")

    @classmethod
    def tool(cls, name: str) -> ToolChoice:
        return cls("tool", name=name)

    @classmethod
    def from_wire(cls, value: Any) -> ToolChoice:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if value not in {"auto", "none", "required"}:
                raise ValueError(f"Unknown tool_choice mode: {value}")
            return cls(value)
        if not isinstance(value, dict) or set(value) != {"type", "function"} or value.get("type") != "function":
            raise ValueError("named tool_choice must use the standard function object")
        function = value.get("function")
        if not isinstance(function, dict) or set(function) != {"name"}:
            raise ValueError("named tool_choice function must contain only name")
        return cls.tool(function.get("name"))

    def to_wire(self) -> str | dict[str, Any]:
        if self.mode != "tool":
            return self.mode
        return {"type": "function", "function": {"name": self.name}}


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type not in {"text", "json_object", "json_schema"}:
            raise ValueError(f"Unknown response_format type: {self.type}")
        if self.type == "json_schema":
            if not isinstance(self.json_schema, dict):
                raise TypeError("json_schema response_format requires an object")
        elif self.json_schema is not None:
            raise ValueError(f"response_format type {self.type!r} cannot include json_schema")

    @classmethod
    def text(cls) -> ResponseFormat:
        return cls("text")

    @classmethod
    def json_object(cls) -> ResponseFormat:
        return cls("json_object")

    @classmethod
    def json_schema_format(cls, json_schema: dict[str, Any]) -> ResponseFormat:
        if not isinstance(json_schema, dict):
            raise TypeError("json_schema response_format requires an object")
        return cls("json_schema", json_schema=dict(json_schema))

    @classmethod
    def from_wire(cls, value: Any) -> ResponseFormat:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or not isinstance(value.get("type"), str):
            raise TypeError("response_format must be an object with a type")
        format_type = value["type"]
        if format_type in {"text", "json_object"}:
            if set(value) != {"type"}:
                raise ValueError(f"response_format type {format_type!r} cannot include extra fields")
            return cls(format_type)
        if format_type == "json_schema" and set(value) == {"type", "json_schema"}:
            return cls.json_schema_format(value["json_schema"])
        raise ValueError("Invalid response_format wire shape")

    def to_wire(self) -> dict[str, Any]:
        if self.type != "json_schema":
            return {"type": self.type}
        return {"type": "json_schema", "json_schema": dict(self.json_schema or {})}


@dataclass(frozen=True, slots=True)
class RetrySettings:
    max_attempts: int = 3
    backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("retry.max_attempts must be an integer greater than zero")
        if (
            isinstance(self.backoff_seconds, bool)
            or not isinstance(self.backoff_seconds, int | float)
            or not math.isfinite(self.backoff_seconds)
        ):
            raise ValueError("retry.backoff_seconds must be a finite non-negative number")
        if self.backoff_seconds < 0:
            raise ValueError("retry.backoff_seconds must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: ToolChoice | Literal["auto", "required", "none"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    response_format: ResponseFormat | dict[str, Any] | None = None
    timeout_seconds: float | None = None
    retry: RetrySettings | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    extra_args: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._validate_optional_number("temperature", self.temperature, minimum=0.0)
        self._validate_optional_number("top_p", self.top_p, minimum=0.0, maximum=1.0)
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be an integer greater than zero")
        self._validate_optional_number("timeout_seconds", self.timeout_seconds, minimum=0.0, exclusive_minimum=True)
        if self.tool_choice is not None and not isinstance(self.tool_choice, ToolChoice):
            object.__setattr__(self, "tool_choice", ToolChoice.from_wire(self.tool_choice))
        if self.parallel_tool_calls is not None and not isinstance(self.parallel_tool_calls, bool):
            raise TypeError("parallel_tool_calls must be a boolean")
        if self.reasoning == {}:
            object.__setattr__(self, "reasoning", None)
        elif self.reasoning is not None and not isinstance(self.reasoning, dict):
            raise TypeError("reasoning must be an object")
        if self.response_format is not None and not isinstance(self.response_format, ResponseFormat):
            object.__setattr__(self, "response_format", ResponseFormat.from_wire(self.response_format))
        self._validate_extra_maps()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None or value == {}:
                continue
            if isinstance(value, RetrySettings):
                payload[item.name] = {
                    "max_attempts": value.max_attempts,
                    "backoff_seconds": value.backoff_seconds,
                }
            elif isinstance(value, ToolChoice | ResponseFormat):
                payload[item.name] = value.to_wire()
            else:
                payload[item.name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSettings:
        values = dict(payload)
        for name in ("extra_headers", "extra_body", "extra_args"):
            if name in values and not isinstance(values[name], dict):
                raise TypeError(f"{name} must be an object")
        if "max_tokens" not in values and "max_output_tokens" in values:
            values["max_tokens"] = values.pop("max_output_tokens")
        retry = values.get("retry")
        if isinstance(retry, dict):
            unknown_retry = sorted(set(retry) - {"max_attempts", "backoff_seconds"})
            if unknown_retry:
                raise ValueError(f"Unknown RetrySettings fields: {', '.join(unknown_retry)}")
            values["retry"] = RetrySettings(
                max_attempts=retry.get("max_attempts", 3),
                backoff_seconds=retry.get("backoff_seconds", 2.0),
            )
        elif retry is not None and not isinstance(retry, RetrySettings):
            raise TypeError("retry must be an object")
        known_fields = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known_fields)
        if unknown:
            raise ValueError(f"Unknown ModelSettings fields: {', '.join(unknown)}")
        return cls(**values)

    def _validate_extra_maps(self) -> None:
        for name in ("extra_headers", "extra_body", "extra_args"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be an object")
        if self.extra_headers is not None and any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in self.extra_headers.items()
        ):
            raise TypeError("extra_headers keys and values must be strings")

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        if override is None:
            return self

        values: dict[str, Any] = {}
        for item in fields(self):
            current = getattr(self, item.name)
            incoming = getattr(override, item.name)
            if item.name in {"extra_headers", "extra_body", "extra_args"}:
                values[item.name] = self._merge_dicts(current, incoming)
            else:
                values[item.name] = incoming if incoming is not None else current
        return ModelSettings(**values)

    @staticmethod
    def _merge_dicts(base: dict[Any, Any] | None, override: dict[Any, Any] | None) -> dict[Any, Any] | None:
        if base is None and override is None:
            return None
        merged: dict[Any, Any] = {}
        if base:
            merged.update(base)
        if override:
            merged.update(override)
        return merged

    @staticmethod
    def _validate_optional_number(
        name: str,
        value: float | None,
        *,
        minimum: float,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
    ) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        below_minimum = value <= minimum if exclusive_minimum else value < minimum
        if below_minimum or (maximum is not None and value > maximum):
            interval = f"({minimum}, {maximum}]" if exclusive_minimum else f"[{minimum}, {maximum}]"
            if maximum is None:
                interval = f"greater than {minimum}" if exclusive_minimum else f"at least {minimum}"
            raise ValueError(f"{name} must be {interval}")
