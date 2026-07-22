from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vv_agent.checkpoint import ToolIdempotency

_PORTABLE_WHITESPACE = frozenset({"\t", "\n", "\r", " "})
_MAX_COLLECTION_ITEMS = 32
_MAX_LABEL_CODE_POINTS = 128
_TOOL_METADATA_FIELDS = frozenset({"side_effect", "idempotency", "terminal", "capability_tags", "cost_dimensions"})


class ToolSideEffect(StrEnum):
    UNKNOWN = "unknown"
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    EXTERNAL = "external"


def _trim_portable_whitespace(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and value[start] in _PORTABLE_WHITESPACE:
        start += 1
    while end > start and value[end - 1] in _PORTABLE_WHITESPACE:
        end -= 1
    return value[start:end]


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be", errors="surrogatepass")


def normalize_metadata_labels(value: Iterable[str], *, field_name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must contain only strings")
        label = _trim_portable_whitespace(item)
        if not label:
            raise ValueError(f"{field_name} cannot contain a blank label")
        if len(label) > _MAX_LABEL_CODE_POINTS:
            raise ValueError(f"{field_name} labels cannot exceed {_MAX_LABEL_CODE_POINTS} Unicode code points")
        normalized.add(label)
    if len(normalized) > _MAX_COLLECTION_ITEMS:
        raise ValueError(f"{field_name} cannot contain more than {_MAX_COLLECTION_ITEMS} labels")
    return sorted(normalized, key=_utf16_sort_key)


def normalize_denied_side_effects(value: Iterable[ToolSideEffect | str]) -> list[ToolSideEffect]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("denied_side_effects must be a collection of side-effect values")
    normalized: set[ToolSideEffect] = set()
    for item in value:
        try:
            normalized.add(item if isinstance(item, ToolSideEffect) else ToolSideEffect(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported denied side effect: {item!r}") from exc
    if len(normalized) > _MAX_COLLECTION_ITEMS:
        raise ValueError(f"denied_side_effects cannot contain more than {_MAX_COLLECTION_ITEMS} values")
    return sorted(normalized, key=lambda item: _utf16_sort_key(item.value))


@dataclass(slots=True)
class ToolMetadata:
    side_effect: ToolSideEffect = ToolSideEffect.UNKNOWN
    idempotency: ToolIdempotency = ToolIdempotency.UNKNOWN
    terminal: bool = False
    capability_tags: list[str] = field(default_factory=list)
    cost_dimensions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            self.side_effect = (
                self.side_effect if isinstance(self.side_effect, ToolSideEffect) else ToolSideEffect(self.side_effect)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported tool side effect: {self.side_effect!r}") from exc
        try:
            self.idempotency = (
                self.idempotency if isinstance(self.idempotency, ToolIdempotency) else ToolIdempotency(self.idempotency)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported tool idempotency: {self.idempotency!r}") from exc
        if not isinstance(self.terminal, bool):
            raise TypeError("tool metadata terminal must be a boolean")
        self.capability_tags = normalize_metadata_labels(
            self.capability_tags,
            field_name="capability_tags",
        )
        self.cost_dimensions = normalize_metadata_labels(
            self.cost_dimensions,
            field_name="cost_dimensions",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolMetadata:
        if not isinstance(value, Mapping):
            raise TypeError("tool_metadata must be an object")
        unknown_fields = set(value) - _TOOL_METADATA_FIELDS
        if unknown_fields:
            names = ", ".join(sorted(str(item) for item in unknown_fields))
            raise ValueError(f"tool_metadata contains unknown fields: {names}")
        return cls(
            side_effect=value.get("side_effect", ToolSideEffect.UNKNOWN),
            idempotency=value.get("idempotency", ToolIdempotency.UNKNOWN),
            terminal=value.get("terminal", False),
            capability_tags=value.get("capability_tags", []),
            cost_dimensions=value.get("cost_dimensions", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_effect": self.side_effect.value,
            "idempotency": self.idempotency.value,
            "terminal": self.terminal,
            "capability_tags": list(self.capability_tags),
            "cost_dimensions": list(self.cost_dimensions),
        }


def normalize_tool_metadata(
    value: ToolMetadata | Mapping[str, Any] | None,
) -> ToolMetadata | None:
    if value is None:
        return None
    metadata = value if isinstance(value, ToolMetadata) else ToolMetadata.from_dict(value)
    return ToolMetadata.from_dict(metadata.to_dict())


def metadata_policy_denial_source(
    metadata: ToolMetadata | None,
    *,
    denied_side_effects: Iterable[ToolSideEffect | str] = (),
    denied_capability_tags: Iterable[str] = (),
    deny_terminal_tools: bool = False,
    denied_cost_dimensions: Iterable[str] = (),
) -> str | None:
    if metadata is None:
        return None
    denied_effects = set(normalize_denied_side_effects(denied_side_effects))
    denied_tags = set(normalize_metadata_labels(denied_capability_tags, field_name="denied_capability_tags"))
    denied_costs = set(normalize_metadata_labels(denied_cost_dimensions, field_name="denied_cost_dimensions"))
    if metadata.side_effect in denied_effects:
        return "metadata.side_effect"
    if deny_terminal_tools and metadata.terminal:
        return "metadata.terminal"
    if any(tag in denied_tags for tag in metadata.capability_tags):
        return "metadata.capability_tag"
    if any(dimension in denied_costs for dimension in metadata.cost_dimensions):
        return "metadata.cost_dimension"
    return None


__all__ = [
    "ToolMetadata",
    "ToolSideEffect",
    "metadata_policy_denial_source",
    "normalize_denied_side_effects",
    "normalize_metadata_labels",
    "normalize_tool_metadata",
]
