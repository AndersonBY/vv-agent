"""Pure contract-41 requirement values and stable worker capability tags."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._tool_metadata import ToolMetadata

_REQUIREMENT_FIELDS = frozenset({"level", "handler_version", "capabilities"})
_MODEL_IDENTITY_FIELDS = ("backend", "model", "context_length", "max_output_tokens")


@dataclass(frozen=True, slots=True)
class SessionRequirements:
    """The closed typed requirement value used by L2 creation and inbox inputs."""

    level: int
    handler_version: str | None
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.level) is not int or self.level < 1:
            raise ValueError("requirements.level must be a positive integer")
        if self.handler_version is not None and (not isinstance(self.handler_version, str) or not self.handler_version):
            raise ValueError("requirements.handler_version must be non-empty text or null")
        if isinstance(self.capabilities, (str, bytes)) or not isinstance(self.capabilities, Sequence):
            raise ValueError("requirements.capabilities must be an array of text")
        capabilities = tuple(self.capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("requirements.capabilities must contain non-empty text")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("requirements.capabilities contains duplicates")
        if list(capabilities) != sorted(capabilities):
            raise ValueError("requirements.capabilities is not sorted")
        object.__setattr__(self, "capabilities", capabilities)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SessionRequirements:
        if not isinstance(value, Mapping) or set(value) != _REQUIREMENT_FIELDS:
            raise ValueError("requirements must contain exactly level, handler_version and capabilities")
        return cls(value["level"], value["handler_version"], value["capabilities"])

    @classmethod
    def normalized(
        cls,
        *,
        level: int,
        handler_version: str | None,
        capabilities: Iterable[str],
    ) -> SessionRequirements:
        values = list(capabilities)
        if any(not isinstance(item, str) or not item for item in values):
            raise ValueError("requirements.capabilities must contain non-empty text")
        return cls(level, handler_version, tuple(sorted(set(values))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "handler_version": self.handler_version,
            "capabilities": list(self.capabilities),
        }


def normalize_requirements(
    value: SessionRequirements | Mapping[str, Any],
) -> SessionRequirements:
    if isinstance(value, SessionRequirements):
        return value
    if isinstance(value, Mapping):
        return SessionRequirements.from_dict(value)
    raise ValueError("requirements must be a typed SessionRequirements object or object value")


def _normalized_tool_metadata(
    metadata: ToolMetadata | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if metadata is None or (isinstance(metadata, Mapping) and not metadata):
        return {}
    if isinstance(metadata, ToolMetadata):
        return metadata.to_dict()
    if isinstance(metadata, Mapping):
        return ToolMetadata.from_dict(metadata).to_dict()
    raise ValueError("tool metadata must be normalized ToolMetadata, an object or absent")


def _tool_schema_fact(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ValueError("tool name must be non-empty text")
    if not isinstance(schema, Mapping):
        raise ValueError("tool schema must be the complete OpenAI function object")
    function = schema.get("function")
    if schema.get("type") != "function" or not isinstance(function, Mapping):
        raise ValueError("tool schema must include outer type=function and function object")
    if function.get("name") != name:
        raise ValueError("tool schema function.name must equal the tagged name")
    if not isinstance(schema, dict) or not isinstance(function, dict):
        raise ValueError("tool schema must be a plain JSON object")
    return dict(schema)


def tool_capability_tag(
    name: str,
    schema: Mapping[str, Any],
    metadata: ToolMetadata | Mapping[str, Any] | None = None,
) -> str:
    """Hash one full registry schema plus its separately normalized metadata."""

    fact = {
        "name": name,
        "schema": _tool_schema_fact(name, schema),
        "metadata": _normalized_tool_metadata(metadata),
    }
    return "tool:" + sha256(canonical_json_bytes(fact)).hexdigest()


def model_capability_tag(binding: Mapping[str, Any]) -> str:
    """Hash one atomic resolved binding without endpoint or route state."""

    if not isinstance(binding, Mapping):
        raise ValueError("model binding must be an object")
    missing = set(_MODEL_IDENTITY_FIELDS) - set(binding)
    if missing:
        raise ValueError("model binding is missing " + ", ".join(sorted(missing)))
    identity = {key: binding[key] for key in _MODEL_IDENTITY_FIELDS}
    if not isinstance(identity["backend"], str) or not identity["backend"]:
        raise ValueError("model binding backend must be non-empty text")
    if not isinstance(identity["model"], str) or not identity["model"]:
        raise ValueError("model binding model must be non-empty text")
    for field in ("context_length", "max_output_tokens"):
        value = identity[field]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"model binding {field} must be a positive integer or null")
    return "model:" + sha256(canonical_json_bytes(identity)).hexdigest()


def model_capability_tags(binding: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unique tags for a primary binding and each internal purpose binding."""

    tags = {model_capability_tag(binding)}
    internal = binding.get("internal")
    if internal is not None:
        if not isinstance(internal, Mapping):
            raise ValueError("model binding internal must be an object")
        for purpose, item in internal.items():
            if not isinstance(purpose, str) or not purpose:
                raise ValueError("model binding internal purpose must be non-empty text")
            tags.add(model_capability_tag(item))
    return tuple(sorted(tags))


def capability_supports(
    *,
    tools: Iterable[tuple[Mapping[str, Any], ToolMetadata | Mapping[str, Any] | None]] = (),
    models: Iterable[Mapping[str, Any]] = (),
    extra_capabilities: Iterable[str] = (),
) -> tuple[str, ...]:
    """Build supports from trusted complete declarations without enumerating subsets."""

    if isinstance(extra_capabilities, (str, bytes)):
        raise ValueError("extra capabilities must be a collection of text")
    capabilities: set[str] = set(extra_capabilities)
    if any(not isinstance(item, str) or not item for item in capabilities):
        raise ValueError("extra capabilities must contain non-empty text")
    for schema, metadata in tools:
        function = schema.get("function") if isinstance(schema, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str):
            raise ValueError("tool declaration must expose function.name")
        capabilities.add(tool_capability_tag(name, schema, metadata))
    for binding in models:
        capabilities.update(model_capability_tags(binding))
    return tuple(sorted(capabilities))


def static_capability_requirements(
    *,
    level: int,
    handler_version: str | None,
    tools: Iterable[tuple[Mapping[str, Any], ToolMetadata | Mapping[str, Any] | None]] = (),
    models: Iterable[Mapping[str, Any]] = (),
    extra_capabilities: Iterable[str] = (),
) -> SessionRequirements:
    """Construct a pre-admission bound from trusted declarations only."""

    return SessionRequirements.normalized(
        level=level,
        handler_version=handler_version,
        capabilities=capability_supports(
            tools=tools,
            models=models,
            extra_capabilities=extra_capabilities,
        ),
    )


def definition_requirements(
    definition: Mapping[str, Any],
    *,
    handler_version: str,
    level: int = 2,
) -> SessionRequirements:
    """Derive exact tags from one already frozen runtime definition."""

    if not isinstance(definition, Mapping):
        raise ValueError("compiled definition must be an object")
    tools = definition.get("tools")
    metadata_by_name = definition.get("capabilities", {})
    model_binding = definition.get("model_binding")
    if not isinstance(tools, list) or not all(isinstance(item, Mapping) for item in tools):
        raise ValueError("compiled definition tools must be a complete schema array")
    if not isinstance(metadata_by_name, Mapping):
        raise ValueError("compiled definition capabilities must be a metadata lookup")
    if not isinstance(model_binding, Mapping):
        raise ValueError("compiled definition model_binding must be an object")

    capabilities: set[str] = set()
    for schema in tools:
        function = schema.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError("compiled definition tool must expose function.name")
        capabilities.add(
            tool_capability_tag(
                name,
                schema,
                metadata_by_name.get(name, {}),
            )
        )
    capabilities.update(model_capability_tags(model_binding))
    return SessionRequirements.normalized(
        level=level,
        handler_version=handler_version,
        capabilities=capabilities,
    )


__all__ = [
    "SessionRequirements",
    "capability_supports",
    "definition_requirements",
    "model_capability_tag",
    "model_capability_tags",
    "normalize_requirements",
    "static_capability_requirements",
    "tool_capability_tag",
]
