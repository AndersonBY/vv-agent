from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from vv_agent.budget import RunBudgetLimits
from vv_agent.checkpoint import (
    MAX_CHECKPOINT_KEY_BYTES,
    MAX_WIRE_INTEGER,
    RUN_DEFINITION_V1_SCHEMA,
    AmbiguousModelPolicy,
    AmbiguousToolPolicy,
    CheckpointConfig,
    ResumePolicy,
    utf16_sort_key,
    validate_extension_namespace,
)
from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.state import StateStoreSpec
from vv_agent.tools import ToolRegistry, build_default_registry
from vv_agent.tools.metadata import (
    ToolSideEffect,
    normalize_denied_side_effects,
    normalize_metadata_labels,
)
from vv_agent.types import AgentTask

DISTRIBUTED_RUN_SCHEMA_VERSION_V1 = "vv-agent.distributed-run.v1"
DISTRIBUTED_RUN_SCHEMA_VERSION_V2 = "vv-agent.distributed-run.v2"
# Keep the historical public constant and default pinned to v1.
DISTRIBUTED_RUN_SCHEMA_VERSION = DISTRIBUTED_RUN_SCHEMA_VERSION_V1
DEFAULT_TOOLSET_ID = "vv-agent.builtin-tools"
DEFAULT_TOOLSET_VERSION = "1"
DEFAULT_TOOLSET_SCHEMA_DIGEST = "f85422117d41d28ffa3cdfcfd9a42892854de624808fadc2124f4ebe7a452b61"
DEFAULT_CYCLE_NAME = "vv_agent.distributed.run_single_cycle"
DEFAULT_LEASE_DURATION_MS = 5 * 60 * 1000
_MAX_U64 = (1 << 64) - 1

CapabilityKind = Literal[
    "llm_client",
    "workspace_backend",
    "approval_provider",
    "approval_broker",
    "cancellation",
    "event_sink",
    "host_cost_meter",
    "app_state",
    "memory_provider",
    "hook",
    "after_cycle_hook",
    "observer",
    "sub_task_manager",
    "tool_predicate",
    "checkpoint_store",
    "checkpoint_event_store",
    "checkpoint_extension",
    "reconciliation_provider",
]
ClaimMode = Literal["continue", "recovery"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JSON_POINTER_ESCAPE_RE = re.compile(r"~(?:0|1)")


class DistributedContractError(ValueError):
    """A distributed envelope or capability contract is invalid."""


class DistributedCapabilityError(RuntimeError):
    """A worker cannot resolve a declared distributed capability."""


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DistributedContractError(f"{key} must be a non-empty string")
    return value


def _optional_ref(payload: Mapping[str, Any], key: str) -> CapabilityRef | None:
    value = payload.get(key)
    if value is None:
        return None
    return CapabilityRef.from_dict(value, field_name=key)


def _validate_json_pointer(pointer: str) -> None:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise DistributedContractError("credential slot must be a non-empty RFC 6901 JSON pointer")
    for token in pointer[1:].split("/"):
        if "~" in _JSON_POINTER_ESCAPE_RE.sub("", token):
            raise DistributedContractError("credential slot contains an invalid RFC 6901 escape")


def _validate_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DistributedContractError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    id: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise DistributedContractError("capability id must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise DistributedContractError("capability version must be a non-empty string")

    @property
    def key(self) -> tuple[str, str]:
        return self.id, self.version

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version}

    @classmethod
    def from_dict(cls, payload: Any, *, field_name: str = "capability_ref") -> CapabilityRef:
        if not isinstance(payload, Mapping):
            raise DistributedContractError(f"{field_name} must be an object")
        identifier = payload.get("id")
        version = payload.get("version")
        if not isinstance(identifier, str) or not identifier.strip():
            raise DistributedContractError(f"{field_name}.id must be a non-empty string")
        if not isinstance(version, str) or not version.strip():
            raise DistributedContractError(f"{field_name}.version must be a non-empty string")
        return cls(
            id=identifier,
            version=version,
        )


@dataclass(frozen=True, slots=True)
class CheckpointExtensionRef:
    namespace: str
    reference: CapabilityRef
    required: bool

    def __post_init__(self) -> None:
        try:
            validate_extension_namespace(self.namespace)
        except (TypeError, ValueError) as exc:
            raise DistributedContractError(str(exc)) from exc
        if not isinstance(self.required, bool):
            raise DistributedContractError("checkpoint extension required must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "reference": self.reference.to_dict(),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Any, *, index: int) -> CheckpointExtensionRef:
        field_name = f"checkpoint_extension_refs[{index}]"
        if not isinstance(payload, Mapping):
            raise DistributedContractError(f"{field_name} must be an object")
        namespace = payload.get("namespace")
        if not isinstance(namespace, str):
            raise DistributedContractError(f"{field_name}.namespace must be a string")
        required = payload.get("required")
        if not isinstance(required, bool):
            raise DistributedContractError(f"{field_name}.required must be a boolean")
        return cls(
            namespace=namespace,
            reference=CapabilityRef.from_dict(
                payload.get("reference"),
                field_name=f"{field_name}.reference",
            ),
            required=required,
        )


@dataclass(frozen=True, slots=True)
class DistributedCheckpointConfig:
    key: str
    resume_policy: ResumePolicy
    ambiguous_model_policy: AmbiguousModelPolicy
    ambiguous_tool_policy: AmbiguousToolPolicy
    required_extension_namespaces: tuple[str, ...] = ()
    max_extension_state_bytes: int = 262_144
    credential_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise DistributedContractError("checkpoint_config.key must be a non-empty string")
        if len(self.key.encode("utf-8")) > MAX_CHECKPOINT_KEY_BYTES:
            raise DistributedContractError(f"checkpoint_config.key must be at most {MAX_CHECKPOINT_KEY_BYTES} UTF-8 bytes")
        try:
            object.__setattr__(self, "resume_policy", ResumePolicy(self.resume_policy))
            object.__setattr__(
                self,
                "ambiguous_model_policy",
                AmbiguousModelPolicy(self.ambiguous_model_policy),
            )
            object.__setattr__(
                self,
                "ambiguous_tool_policy",
                AmbiguousToolPolicy(self.ambiguous_tool_policy),
            )
        except (TypeError, ValueError) as exc:
            raise DistributedContractError("checkpoint_config contains an unsupported policy") from exc
        if (
            isinstance(self.max_extension_state_bytes, bool)
            or not isinstance(self.max_extension_state_bytes, int)
            or not 0 <= self.max_extension_state_bytes <= MAX_WIRE_INTEGER
        ):
            raise DistributedContractError(f"max_extension_state_bytes must be between 0 and {MAX_WIRE_INTEGER}")
        namespaces = tuple(self.required_extension_namespaces)
        if namespaces != tuple(sorted(set(namespaces), key=utf16_sort_key)):
            raise DistributedContractError("checkpoint_config.required_extension_namespaces must be sorted and unique")
        for namespace in namespaces:
            try:
                validate_extension_namespace(namespace)
            except (TypeError, ValueError) as exc:
                raise DistributedContractError(str(exc)) from exc
        slots = tuple(self.credential_slots)
        if slots != tuple(sorted(set(slots), key=utf16_sort_key)):
            raise DistributedContractError("checkpoint_config.credential_slots must be sorted and unique")
        for pointer in slots:
            _validate_json_pointer(pointer)
        object.__setattr__(self, "required_extension_namespaces", namespaces)
        object.__setattr__(self, "credential_slots", slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "resume_policy": self.resume_policy.value,
            "ambiguous_model_policy": self.ambiguous_model_policy.value,
            "ambiguous_tool_policy": self.ambiguous_tool_policy.value,
            "required_extension_namespaces": list(self.required_extension_namespaces),
            "max_extension_state_bytes": self.max_extension_state_bytes,
            "credential_slots": list(self.credential_slots),
        }

    def to_runtime_config(
        self,
        *,
        store: Any,
        capability_refs: Mapping[str, Mapping[str, str]] | None = None,
    ) -> CheckpointConfig:
        return CheckpointConfig(
            store=store,
            key=self.key,
            resume_policy=self.resume_policy,
            ambiguous_model_policy=self.ambiguous_model_policy,
            ambiguous_tool_policy=self.ambiguous_tool_policy,
            required_extension_namespaces=list(self.required_extension_namespaces),
            max_extension_state_bytes=self.max_extension_state_bytes,
            credential_slots=list(self.credential_slots),
            capability_refs={key: dict(value) for key, value in (capability_refs or {}).items()},
        )

    @classmethod
    def from_checkpoint_config(
        cls,
        config: CheckpointConfig,
        *,
        require_existing: bool = True,
    ) -> DistributedCheckpointConfig:
        if config.key is None:
            raise DistributedContractError("distributed v2 requires an explicit checkpoint key")
        return cls(
            key=config.key,
            resume_policy=(ResumePolicy.REQUIRE_EXISTING if require_existing else config.resume_policy),
            ambiguous_model_policy=config.ambiguous_model_policy,
            ambiguous_tool_policy=config.ambiguous_tool_policy,
            required_extension_namespaces=tuple(config.required_extension_namespaces),
            max_extension_state_bytes=config.max_extension_state_bytes,
            credential_slots=tuple(config.credential_slots),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedCheckpointConfig:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("distributed v2 requires checkpoint_config")

        namespaces = payload.get("required_extension_namespaces", [])
        credential_slots = payload.get("credential_slots", [])
        if not isinstance(namespaces, list):
            raise DistributedContractError("checkpoint_config.required_extension_namespaces must be an array")
        if not isinstance(credential_slots, list):
            raise DistributedContractError("checkpoint_config.credential_slots must be an array")
        return cls(
            key=_required_string(payload, "key"),
            resume_policy=payload.get("resume_policy"),
            ambiguous_model_policy=payload.get("ambiguous_model_policy"),
            ambiguous_tool_policy=payload.get("ambiguous_tool_policy"),
            required_extension_namespaces=tuple(namespaces),
            max_extension_state_bytes=payload.get("max_extension_state_bytes", 262_144),
            credential_slots=tuple(credential_slots),
        )


@dataclass(frozen=True, slots=True)
class ToolsetRef:
    id: str = DEFAULT_TOOLSET_ID
    version: str = DEFAULT_TOOLSET_VERSION
    schema_digest: str = DEFAULT_TOOLSET_SCHEMA_DIGEST

    def __post_init__(self) -> None:
        CapabilityRef(self.id, self.version)
        if not isinstance(self.schema_digest, str) or len(self.schema_digest) != 64:
            raise DistributedContractError("toolset_ref.schema_digest must be a lowercase SHA-256 hex digest")
        if self.schema_digest != self.schema_digest.lower() or any(
            character not in "0123456789abcdef" for character in self.schema_digest
        ):
            raise DistributedContractError("toolset_ref.schema_digest must be a lowercase SHA-256 hex digest")

    @property
    def key(self) -> tuple[str, str]:
        return self.id, self.version

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version, "schema_digest": self.schema_digest}

    @classmethod
    def from_dict(cls, payload: Any) -> ToolsetRef:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("toolset_ref must be an object")
        reference = CapabilityRef.from_dict(payload, field_name="toolset_ref")
        schema_digest = payload.get("schema_digest")
        if not isinstance(schema_digest, str) or not schema_digest.strip():
            raise DistributedContractError("toolset_ref.schema_digest must be a non-empty string")
        return cls(
            id=reference.id,
            version=reference.version,
            schema_digest=schema_digest,
        )


@dataclass(frozen=True, slots=True)
class DistributedToolPolicy:
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    approval: Literal["default", "always", "never", "on_request"] = "default"
    predicate_ref: CapabilityRef | None = None
    denied_side_effects: tuple[ToolSideEffect | str, ...] = ()
    denied_capability_tags: tuple[str, ...] = ()
    deny_terminal_tools: bool = False
    denied_cost_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.approval not in {"default", "always", "never", "on_request"}:
            raise DistributedContractError("tool_policy.approval is unsupported")
        for field_name, values in (
            ("tool_policy.allowed_tools", self.allowed_tools),
            ("tool_policy.disallowed_tools", self.disallowed_tools),
        ):
            if values is not None and any(not isinstance(value, str) or not value.strip() for value in values):
                raise DistributedContractError(f"{field_name} must contain non-empty strings")
        try:
            denied_side_effects = tuple(value.value for value in normalize_denied_side_effects(self.denied_side_effects))
            denied_capability_tags = tuple(
                normalize_metadata_labels(
                    self.denied_capability_tags,
                    field_name="denied_capability_tags",
                )
            )
            denied_cost_dimensions = tuple(
                normalize_metadata_labels(
                    self.denied_cost_dimensions,
                    field_name="denied_cost_dimensions",
                )
            )
        except (TypeError, ValueError) as exc:
            raise DistributedContractError(str(exc)) from exc
        if not isinstance(self.deny_terminal_tools, bool):
            raise DistributedContractError("tool_policy.deny_terminal_tools must be a boolean")
        object.__setattr__(self, "denied_side_effects", denied_side_effects)
        object.__setattr__(
            self,
            "denied_capability_tags",
            denied_capability_tags,
        )
        object.__setattr__(
            self,
            "denied_cost_dimensions",
            denied_cost_dimensions,
        )

    def to_dict(self, *, include_metadata_denials: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
            "disallowed_tools": list(self.disallowed_tools),
            "approval": self.approval,
            "predicate_ref": self.predicate_ref.to_dict() if self.predicate_ref is not None else None,
        }
        if include_metadata_denials:
            payload.update(
                {
                    "denied_side_effects": list(self.denied_side_effects),
                    "denied_capability_tags": list(self.denied_capability_tags),
                    "deny_terminal_tools": self.deny_terminal_tools,
                    "denied_cost_dimensions": list(self.denied_cost_dimensions),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedToolPolicy:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("tool_policy must be an object")
        allowed = payload.get("allowed_tools")
        disallowed = payload.get("disallowed_tools", [])
        if allowed is not None and not isinstance(allowed, list):
            raise DistributedContractError("tool_policy.allowed_tools must be an array or null")
        if not isinstance(disallowed, list):
            raise DistributedContractError("tool_policy.disallowed_tools must be an array")
        denied_side_effects = payload.get("denied_side_effects", [])
        denied_capability_tags = payload.get("denied_capability_tags", [])
        denied_cost_dimensions = payload.get("denied_cost_dimensions", [])
        for field_name, values in (
            ("denied_side_effects", denied_side_effects),
            ("denied_capability_tags", denied_capability_tags),
            ("denied_cost_dimensions", denied_cost_dimensions),
        ):
            if not isinstance(values, list):
                raise DistributedContractError(f"tool_policy.{field_name} must be an array")
        approval = payload.get("approval", "default")
        if not isinstance(approval, str):
            raise DistributedContractError("tool_policy.approval must be a string")
        predicate = payload.get("predicate_ref")
        return cls(
            allowed_tools=tuple(allowed) if allowed is not None else None,
            disallowed_tools=tuple(disallowed),
            approval=approval,  # type: ignore[arg-type]
            predicate_ref=(
                CapabilityRef.from_dict(predicate, field_name="tool_policy.predicate_ref") if predicate is not None else None
            ),
            denied_side_effects=tuple(denied_side_effects),
            denied_capability_tags=tuple(denied_capability_tags),
            deny_terminal_tools=payload.get("deny_terminal_tools", False),
            denied_cost_dimensions=tuple(denied_cost_dimensions),
        )

    def resolve(self, registry: DistributedCapabilityRegistry) -> ToolPolicy:
        predicate = None
        if self.predicate_ref is not None:
            predicate = registry.resolve("tool_predicate", self.predicate_ref)
            if not callable(predicate):
                raise DistributedCapabilityError(
                    f"distributed capability tool_predicate {self.predicate_ref.id}@{self.predicate_ref.version} "
                    "did not resolve to a callable"
                )
        return ToolPolicy(
            allowed_tools=list(self.allowed_tools) if self.allowed_tools is not None else None,
            disallowed_tools=list(self.disallowed_tools),
            approval=self.approval,
            can_use_tool=predicate,
            denied_side_effects=list(self.denied_side_effects),
            denied_capability_tags=list(self.denied_capability_tags),
            deny_terminal_tools=self.deny_terminal_tools,
            denied_cost_dimensions=list(self.denied_cost_dimensions),
        )


def _policy_with_task_metadata_denials(
    policy: DistributedToolPolicy,
    task: AgentTask,
) -> DistributedToolPolicy:
    metadata = getattr(task, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise DistributedContractError("task.metadata must be an object")

    def values(key: str) -> tuple[Any, ...]:
        if key not in metadata:
            return ()
        value = metadata[key]
        if not isinstance(value, list):
            raise DistributedContractError(f"task.metadata.{key} must be an array")
        return tuple(value)

    deny_terminal_tools = policy.deny_terminal_tools
    terminal_key = "_vv_agent_deny_terminal_tools"
    if terminal_key in metadata:
        terminal_value = metadata[terminal_key]
        if not isinstance(terminal_value, bool):
            raise DistributedContractError(f"task.metadata.{terminal_key} must be a boolean")
        deny_terminal_tools = deny_terminal_tools or terminal_value

    return replace(
        policy,
        denied_side_effects=(*policy.denied_side_effects, *values("_vv_agent_denied_side_effects")),
        denied_capability_tags=(
            *policy.denied_capability_tags,
            *values("_vv_agent_denied_capability_tags"),
        ),
        deny_terminal_tools=deny_terminal_tools,
        denied_cost_dimensions=(
            *policy.denied_cost_dimensions,
            *values("_vv_agent_denied_cost_dimensions"),
        ),
    )


def _recipe_with_task_metadata_denials(
    recipe: RuntimeRecipe,
    task: AgentTask,
) -> RuntimeRecipe:
    capabilities = replace(
        recipe.capabilities,
        tool_policy=_policy_with_task_metadata_denials(recipe.capabilities.tool_policy, task),
    )
    return replace(recipe, capabilities=capabilities)


def _task_with_policy_metadata_denials(
    task: AgentTask,
    policy: DistributedToolPolicy,
) -> AgentTask:
    effective_policy = _policy_with_task_metadata_denials(policy, task)
    metadata = dict(task.metadata)
    if effective_policy.denied_side_effects:
        metadata["_vv_agent_denied_side_effects"] = list(effective_policy.denied_side_effects)
    if effective_policy.denied_capability_tags:
        metadata["_vv_agent_denied_capability_tags"] = list(effective_policy.denied_capability_tags)
    if effective_policy.deny_terminal_tools:
        metadata["_vv_agent_deny_terminal_tools"] = True
    if effective_policy.denied_cost_dimensions:
        metadata["_vv_agent_denied_cost_dimensions"] = list(effective_policy.denied_cost_dimensions)
    return replace(task, metadata=metadata)


@dataclass(frozen=True, slots=True)
class DistributedCapabilities:
    toolset_ref: ToolsetRef = field(default_factory=ToolsetRef)
    tool_policy: DistributedToolPolicy = field(default_factory=DistributedToolPolicy)
    llm_client_ref: CapabilityRef | None = None
    workspace_backend_ref: CapabilityRef | None = None
    approval_provider_ref: CapabilityRef | None = None
    approval_broker_ref: CapabilityRef | None = None
    approval_timeout_seconds: float | None = None
    cancellation_ref: CapabilityRef | None = None
    event_sink_ref: CapabilityRef | None = None
    host_cost_meter_ref: CapabilityRef | None = None
    app_state_ref: CapabilityRef | None = None
    sub_task_manager_ref: CapabilityRef | None = None
    memory_provider_refs: tuple[CapabilityRef, ...] = ()
    hook_refs: tuple[CapabilityRef, ...] = ()
    after_cycle_hook_refs: tuple[CapabilityRef, ...] = ()
    observer_refs: tuple[CapabilityRef, ...] = ()
    checkpoint_store_ref: CapabilityRef | None = None
    checkpoint_event_store_ref: CapabilityRef | None = None
    checkpoint_extension_refs: tuple[CheckpointExtensionRef, ...] = ()
    reconciliation_provider_ref: CapabilityRef | None = None

    def __post_init__(self) -> None:
        if (self.approval_provider_ref is None) != (self.approval_broker_ref is None):
            raise DistributedContractError("approval_provider_ref and approval_broker_ref must be declared together")
        if self.approval_timeout_seconds is not None and (
            isinstance(self.approval_timeout_seconds, bool)
            or not isinstance(self.approval_timeout_seconds, (int, float))
            or not math.isfinite(float(self.approval_timeout_seconds))
            or self.approval_timeout_seconds <= 0
        ):
            raise DistributedContractError("approval_timeout_seconds must be a finite positive number or null")
        namespaces = tuple(reference.namespace for reference in self.checkpoint_extension_refs)
        if namespaces != tuple(sorted(set(namespaces), key=utf16_sort_key)):
            raise DistributedContractError("checkpoint_extension_refs must be sorted by unique namespace")

    def to_dict(self, *, include_checkpoint: bool | None = None) -> dict[str, Any]:
        if include_checkpoint is None:
            include_checkpoint = bool(
                self.checkpoint_store_ref is not None
                or self.checkpoint_event_store_ref is not None
                or self.checkpoint_extension_refs
                or self.reconciliation_provider_ref is not None
                or self.after_cycle_hook_refs
            )
        payload = {
            "toolset_ref": self.toolset_ref.to_dict(),
            "tool_policy": self.tool_policy.to_dict(include_metadata_denials=include_checkpoint),
            "llm_client_ref": self.llm_client_ref.to_dict() if self.llm_client_ref else None,
            "workspace_backend_ref": self.workspace_backend_ref.to_dict() if self.workspace_backend_ref else None,
            "approval_provider_ref": self.approval_provider_ref.to_dict() if self.approval_provider_ref else None,
            "approval_broker_ref": self.approval_broker_ref.to_dict() if self.approval_broker_ref else None,
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "cancellation_ref": self.cancellation_ref.to_dict() if self.cancellation_ref else None,
            "event_sink_ref": self.event_sink_ref.to_dict() if self.event_sink_ref else None,
            "host_cost_meter_ref": self.host_cost_meter_ref.to_dict() if self.host_cost_meter_ref else None,
            "app_state_ref": self.app_state_ref.to_dict() if self.app_state_ref else None,
            "sub_task_manager_ref": self.sub_task_manager_ref.to_dict() if self.sub_task_manager_ref else None,
            "memory_provider_refs": [reference.to_dict() for reference in self.memory_provider_refs],
            "hook_refs": [reference.to_dict() for reference in self.hook_refs],
            "observer_refs": [reference.to_dict() for reference in self.observer_refs],
        }
        if include_checkpoint:
            payload.update(
                {
                    "checkpoint_store_ref": (
                        self.checkpoint_store_ref.to_dict() if self.checkpoint_store_ref is not None else None
                    ),
                    "checkpoint_event_store_ref": (
                        self.checkpoint_event_store_ref.to_dict() if self.checkpoint_event_store_ref is not None else None
                    ),
                    "checkpoint_extension_refs": [reference.to_dict() for reference in self.checkpoint_extension_refs],
                    "after_cycle_hook_refs": [reference.to_dict() for reference in self.after_cycle_hook_refs],
                    "reconciliation_provider_ref": (
                        self.reconciliation_provider_ref.to_dict() if self.reconciliation_provider_ref is not None else None
                    ),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedCapabilities:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("capabilities must be an object")

        def refs(key: str) -> tuple[CapabilityRef, ...]:
            values = payload.get(key, [])
            if not isinstance(values, list):
                raise DistributedContractError(f"capabilities.{key} must be an array")
            return tuple(
                CapabilityRef.from_dict(value, field_name=f"capabilities.{key}[{index}]") for index, value in enumerate(values)
            )

        checkpoint_extension_values = payload.get("checkpoint_extension_refs", [])
        if not isinstance(checkpoint_extension_values, list):
            raise DistributedContractError("capabilities.checkpoint_extension_refs must be an array")

        return cls(
            toolset_ref=ToolsetRef.from_dict(payload.get("toolset_ref")),
            tool_policy=DistributedToolPolicy.from_dict(payload.get("tool_policy")),
            llm_client_ref=_optional_ref(payload, "llm_client_ref"),
            workspace_backend_ref=_optional_ref(payload, "workspace_backend_ref"),
            approval_provider_ref=_optional_ref(payload, "approval_provider_ref"),
            approval_broker_ref=_optional_ref(payload, "approval_broker_ref"),
            approval_timeout_seconds=payload.get("approval_timeout_seconds"),
            cancellation_ref=_optional_ref(payload, "cancellation_ref"),
            event_sink_ref=_optional_ref(payload, "event_sink_ref"),
            host_cost_meter_ref=_optional_ref(payload, "host_cost_meter_ref"),
            app_state_ref=_optional_ref(payload, "app_state_ref"),
            sub_task_manager_ref=_optional_ref(payload, "sub_task_manager_ref"),
            memory_provider_refs=refs("memory_provider_refs"),
            hook_refs=refs("hook_refs"),
            after_cycle_hook_refs=refs("after_cycle_hook_refs"),
            observer_refs=refs("observer_refs"),
            checkpoint_store_ref=_optional_ref(payload, "checkpoint_store_ref"),
            checkpoint_event_store_ref=_optional_ref(payload, "checkpoint_event_store_ref"),
            checkpoint_extension_refs=tuple(
                CheckpointExtensionRef.from_dict(value, index=index) for index, value in enumerate(checkpoint_extension_values)
            ),
            reconciliation_provider_ref=_optional_ref(payload, "reconciliation_provider_ref"),
        )


@dataclass(slots=True)
class RuntimeRecipe:
    settings_file: str
    backend: str
    model: str
    workspace: str
    timeout_seconds: float = 90.0
    log_preview_chars: int | None = None
    state_store: StateStoreSpec | None = None
    capabilities: DistributedCapabilities = field(default_factory=DistributedCapabilities)

    def __post_init__(self) -> None:
        for field_name in ("settings_file", "backend", "model", "workspace"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistributedContractError(f"runtime_recipe.{field_name} must be a non-empty string")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise DistributedContractError("runtime_recipe.timeout_seconds must be a finite positive number")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise DistributedContractError("runtime_recipe.timeout_seconds must be a finite positive number")
        if self.log_preview_chars is not None and (
            isinstance(self.log_preview_chars, bool) or not isinstance(self.log_preview_chars, int) or self.log_preview_chars < 0
        ):
            raise DistributedContractError("runtime_recipe.log_preview_chars must be a non-negative integer or null")

    def to_dict(self, *, include_checkpoint_capabilities: bool | None = None) -> dict[str, Any]:
        return {
            "settings_file": self.settings_file,
            "backend": self.backend,
            "model": self.model,
            "workspace": self.workspace,
            "timeout_seconds": float(self.timeout_seconds),
            "log_preview_chars": self.log_preview_chars,
            "state_store": self.state_store.to_dict() if self.state_store is not None else None,
            "capabilities": self.capabilities.to_dict(include_checkpoint=include_checkpoint_capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RuntimeRecipe:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("runtime_recipe must be an object")
        timeout = payload.get("timeout_seconds", 90.0)
        log_preview_chars = payload.get("log_preview_chars")
        return cls(
            settings_file=_required_string(payload, "settings_file"),
            backend=_required_string(payload, "backend"),
            model=_required_string(payload, "model"),
            workspace=_required_string(payload, "workspace"),
            timeout_seconds=timeout,
            log_preview_chars=log_preview_chars,
            state_store=(StateStoreSpec.from_dict(payload["state_store"]) if payload.get("state_store") is not None else None),
            capabilities=DistributedCapabilities.from_dict(payload.get("capabilities")),
        )


@dataclass(frozen=True, slots=True)
class DistributedRunEnvelope:
    job_id: str
    run_id: str
    task: AgentTask
    recipe: RuntimeRecipe
    cycle_name: str
    cycle_index: int
    idempotency_key: str
    deadline_unix_ms: int | None
    budget_limits: RunBudgetLimits | None = None
    lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS
    schema_version: str = DISTRIBUTED_RUN_SCHEMA_VERSION
    root_run_id: str | None = None
    trace_id: str | None = None
    run_definition_schema: str | None = None
    run_definition_digest: str | None = None
    claim_mode: ClaimMode | None = None
    resume_attempt: int | None = None
    checkpoint_config: DistributedCheckpointConfig | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            DISTRIBUTED_RUN_SCHEMA_VERSION_V1,
            DISTRIBUTED_RUN_SCHEMA_VERSION_V2,
        }:
            raise DistributedContractError(f"unsupported distributed schema_version: {self.schema_version}")
        for field_name in ("job_id", "run_id", "cycle_name", "idempotency_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistributedContractError(f"distributed envelope {field_name} must be a non-empty string")
        if isinstance(self.cycle_index, bool) or not isinstance(self.cycle_index, int) or not 1 <= self.cycle_index < 1 << 32:
            raise DistributedContractError("distributed envelope cycle_index must be between 1 and 4294967295")
        if self.deadline_unix_ms is not None and (
            isinstance(self.deadline_unix_ms, bool)
            or not isinstance(self.deadline_unix_ms, int)
            or self.deadline_unix_ms < 0
            or self.deadline_unix_ms > _MAX_U64
        ):
            raise DistributedContractError("distributed envelope deadline_unix_ms must be a non-negative integer or null")
        if (
            isinstance(self.lease_duration_ms, bool)
            or not isinstance(self.lease_duration_ms, int)
            or self.lease_duration_ms <= 0
            or self.lease_duration_ms > _MAX_U64
        ):
            raise DistributedContractError("distributed envelope lease_duration_ms must be a positive integer")
        if self.budget_limits is not None and not isinstance(self.budget_limits, RunBudgetLimits):
            raise DistributedContractError("distributed envelope budget_limits must be an object or null")
        if self.schema_version == DISTRIBUTED_RUN_SCHEMA_VERSION_V1:
            capabilities = self.recipe.capabilities
            if (
                capabilities.checkpoint_store_ref is not None
                or capabilities.checkpoint_event_store_ref is not None
                or capabilities.checkpoint_extension_refs
                or capabilities.reconciliation_provider_ref is not None
                or capabilities.after_cycle_hook_refs
            ):
                raise DistributedContractError("distributed v1 envelope cannot contain checkpoint v2 capability refs")
            if any(
                value is not None
                for value in (
                    self.root_run_id,
                    self.trace_id,
                    self.run_definition_schema,
                    self.run_definition_digest,
                    self.claim_mode,
                    self.resume_attempt,
                    self.checkpoint_config,
                )
            ):
                raise DistributedContractError("distributed v1 envelope cannot contain checkpoint v2 fields")
            return

        if self.run_definition_schema != RUN_DEFINITION_V1_SCHEMA:
            raise DistributedContractError("checkpoint_definition_schema_unsupported")
        for field_name in ("root_run_id", "trace_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistributedContractError(f"distributed v2 {field_name} must be a non-empty string")
        _validate_sha256(self.run_definition_digest, "run_definition_digest")
        if self.claim_mode not in {"continue", "recovery"}:
            raise DistributedContractError("checkpoint_claim_mode_invalid")
        if (
            isinstance(self.resume_attempt, bool)
            or not isinstance(self.resume_attempt, int)
            or not 1 <= self.resume_attempt <= MAX_WIRE_INTEGER
        ):
            raise DistributedContractError(f"resume_attempt must be between 1 and {MAX_WIRE_INTEGER}")
        if not isinstance(self.checkpoint_config, DistributedCheckpointConfig):
            raise DistributedContractError("distributed v2 requires checkpoint_config")
        capabilities = self.recipe.capabilities
        if capabilities.checkpoint_store_ref is None:
            raise DistributedContractError("distributed v2 requires checkpoint_store_ref")
        extensions_by_namespace = {reference.namespace: reference for reference in capabilities.checkpoint_extension_refs}
        for namespace in self.checkpoint_config.required_extension_namespaces:
            reference = extensions_by_namespace.get(namespace)
            if reference is None or not reference.required:
                raise DistributedContractError(f"required checkpoint extension {namespace} is unavailable")

    @classmethod
    def for_cycle(
        cls,
        *,
        task: AgentTask,
        recipe: RuntimeRecipe,
        cycle_index: int,
        cycle_name: str = DEFAULT_CYCLE_NAME,
        run_id: str | None = None,
        deadline_unix_ms: int | None = None,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
        budget_limits: RunBudgetLimits | None = None,
    ) -> DistributedRunEnvelope:
        effective_run_id = run_id or str(task.metadata.get("_vv_agent_run_id") or task.task_id)
        idempotency_key = f"{effective_run_id}:cycle:{cycle_index}"
        return cls(
            job_id=idempotency_key,
            run_id=effective_run_id,
            task=task,
            budget_limits=budget_limits,
            recipe=recipe,
            cycle_name=cycle_name,
            cycle_index=cycle_index,
            idempotency_key=idempotency_key,
            deadline_unix_ms=deadline_unix_ms,
            lease_duration_ms=lease_duration_ms,
        )

    @classmethod
    def for_checkpoint_cycle(
        cls,
        *,
        task: AgentTask,
        recipe: RuntimeRecipe,
        cycle_index: int,
        root_run_id: str,
        trace_id: str,
        run_definition_digest: str,
        claim_mode: ClaimMode,
        resume_attempt: int,
        checkpoint_config: DistributedCheckpointConfig,
        cycle_name: str = DEFAULT_CYCLE_NAME,
        run_id: str | None = None,
        deadline_unix_ms: int | None = None,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
        budget_limits: RunBudgetLimits | None = None,
    ) -> DistributedRunEnvelope:
        recipe = _recipe_with_task_metadata_denials(recipe, task)
        effective_run_id = run_id or str(task.metadata.get("_vv_agent_run_id") or root_run_id)
        idempotency_key = f"{effective_run_id}:cycle:{cycle_index}"
        return cls(
            schema_version=DISTRIBUTED_RUN_SCHEMA_VERSION_V2,
            job_id=idempotency_key,
            run_id=effective_run_id,
            task=task,
            budget_limits=budget_limits,
            recipe=recipe,
            cycle_name=cycle_name,
            cycle_index=cycle_index,
            idempotency_key=idempotency_key,
            deadline_unix_ms=deadline_unix_ms,
            lease_duration_ms=lease_duration_ms,
            root_run_id=root_run_id,
            trace_id=trace_id,
            run_definition_schema=RUN_DEFINITION_V1_SCHEMA,
            run_definition_digest=run_definition_digest,
            claim_mode=claim_mode,
            resume_attempt=resume_attempt,
            checkpoint_config=checkpoint_config,
        )

    @property
    def is_checkpoint_v2(self) -> bool:
        return self.schema_version == DISTRIBUTED_RUN_SCHEMA_VERSION_V2

    def remaining_seconds(self, *, now_ms: int | None = None) -> float | None:
        if self.deadline_unix_ms is None:
            return None
        current = time.time_ns() // 1_000_000 if now_ms is None else now_ms
        return max(0.0, (self.deadline_unix_ms - current) / 1000)

    def ensure_not_expired(self, *, now_ms: int | None = None) -> None:
        remaining = self.remaining_seconds(now_ms=now_ms)
        if remaining is not None and remaining <= 0:
            raise DistributedContractError(f"distributed job {self.job_id} deadline has expired")

    def to_dict(self) -> dict[str, Any]:
        serialized_task = self.task
        if not self.is_checkpoint_v2:
            serialized_task = _task_with_policy_metadata_denials(
                self.task,
                self.recipe.capabilities.tool_policy,
            )
        payload = {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "task": serialized_task.to_dict(),
            "budget_limits": self.budget_limits.to_dict() if self.budget_limits is not None else None,
            "recipe": self.recipe.to_dict(include_checkpoint_capabilities=self.is_checkpoint_v2),
            "cycle_name": self.cycle_name,
            "cycle_index": self.cycle_index,
            "idempotency_key": self.idempotency_key,
            "deadline_unix_ms": self.deadline_unix_ms,
            "lease_duration_ms": self.lease_duration_ms,
        }
        if self.is_checkpoint_v2:
            assert self.checkpoint_config is not None
            payload.update(
                {
                    "root_run_id": self.root_run_id,
                    "trace_id": self.trace_id,
                    "run_definition_schema": self.run_definition_schema,
                    "run_definition_digest": self.run_definition_digest,
                    "claim_mode": self.claim_mode,
                    "resume_attempt": self.resume_attempt,
                    "checkpoint_config": self.checkpoint_config.to_dict(),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedRunEnvelope:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("distributed envelope must be an object")
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, str):
            raise DistributedContractError("distributed envelope schema_version must be a string")
        if schema_version not in {
            DISTRIBUTED_RUN_SCHEMA_VERSION_V1,
            DISTRIBUTED_RUN_SCHEMA_VERSION_V2,
        }:
            raise DistributedContractError(f"unsupported distributed schema_version: {schema_version}")
        task_payload = payload.get("task")
        if not isinstance(task_payload, Mapping):
            raise DistributedContractError("distributed envelope task must be an object")
        budget_limits = None
        if payload.get("budget_limits") is not None:
            try:
                budget_limits = RunBudgetLimits.from_dict(payload["budget_limits"])
            except (TypeError, ValueError) as exc:
                raise DistributedContractError(
                    f"distributed envelope budget limit must be between 0 and 9007199254740991: {exc}"
                ) from exc
        return cls(
            schema_version=schema_version,
            job_id=_required_string(payload, "job_id"),
            run_id=_required_string(payload, "run_id"),
            task=AgentTask.from_dict(dict(task_payload)),
            budget_limits=budget_limits,
            recipe=RuntimeRecipe.from_dict(payload.get("recipe")),
            cycle_name=_required_string(payload, "cycle_name"),
            cycle_index=payload.get("cycle_index"),
            idempotency_key=_required_string(payload, "idempotency_key"),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            lease_duration_ms=payload.get("lease_duration_ms", DEFAULT_LEASE_DURATION_MS),
            root_run_id=payload.get("root_run_id"),
            trace_id=payload.get("trace_id"),
            run_definition_schema=payload.get("run_definition_schema"),
            run_definition_digest=payload.get("run_definition_digest"),
            claim_mode=payload.get("claim_mode"),
            resume_attempt=payload.get("resume_attempt"),
            checkpoint_config=(
                DistributedCheckpointConfig.from_dict(payload.get("checkpoint_config"))
                if schema_version == DISTRIBUTED_RUN_SCHEMA_VERSION_V2
                else None
            ),
        )

    @classmethod
    def from_v1_dict(cls, payload: Any) -> DistributedRunEnvelope:
        """Decode only the immutable v1 wire surface."""
        if not isinstance(payload, Mapping):
            raise DistributedContractError("distributed envelope must be an object")
        if payload.get("schema_version") != DISTRIBUTED_RUN_SCHEMA_VERSION_V1:
            raise DistributedContractError(f"unsupported distributed schema_version: {payload.get('schema_version')}")
        return cls.from_dict(payload)


def toolset_schema_digest(registry: ToolRegistry) -> str:
    canonical = json.dumps(
        registry.list_openai_schemas(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class DistributedCapabilityRegistry:
    """Worker-local registry for resolving language-neutral capability references."""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._capabilities: dict[tuple[str, str, str], Any] = {}
        self._toolsets: dict[tuple[str, str], ToolRegistry] = {}
        if include_defaults:
            self.register_toolset(ToolsetRef(), build_default_registry())

    def register_toolset(self, reference: ToolsetRef, registry: ToolRegistry) -> None:
        actual = toolset_schema_digest(registry)
        if actual != reference.schema_digest:
            raise DistributedCapabilityError(
                f"toolset {reference.id}@{reference.version} schema digest mismatch: "
                f"expected {reference.schema_digest}, got {actual}"
            )
        self._toolsets[reference.key] = registry

    def register(self, kind: CapabilityKind, reference: CapabilityRef, capability: Any) -> None:
        self._capabilities[(kind, *reference.key)] = capability

    def resolve(self, kind: CapabilityKind, reference: CapabilityRef) -> Any:
        key = (kind, *reference.key)
        if key not in self._capabilities:
            raise DistributedCapabilityError(f"unknown distributed capability {kind} {reference.id}@{reference.version}")
        return self._capabilities[key]

    def resolve_toolset(self, reference: ToolsetRef) -> ToolRegistry:
        registry = self._toolsets.get(reference.key)
        if registry is None:
            raise DistributedCapabilityError(f"unknown distributed toolset {reference.id}@{reference.version}")
        actual = toolset_schema_digest(registry)
        if actual != reference.schema_digest:
            raise DistributedCapabilityError(
                f"toolset {reference.id}@{reference.version} schema digest mismatch: "
                f"expected {reference.schema_digest}, got {actual}"
            )
        return registry

    def validate(
        self,
        capabilities: DistributedCapabilities,
        *,
        require_checkpoint_v2: bool = False,
    ) -> None:
        self.resolve_toolset(capabilities.toolset_ref)
        capabilities.tool_policy.resolve(self)
        for kind, reference in (
            ("llm_client", capabilities.llm_client_ref),
            ("workspace_backend", capabilities.workspace_backend_ref),
            ("approval_provider", capabilities.approval_provider_ref),
            ("approval_broker", capabilities.approval_broker_ref),
            ("cancellation", capabilities.cancellation_ref),
            ("event_sink", capabilities.event_sink_ref),
            ("host_cost_meter", capabilities.host_cost_meter_ref),
            ("app_state", capabilities.app_state_ref),
            ("sub_task_manager", capabilities.sub_task_manager_ref),
            ("checkpoint_store", capabilities.checkpoint_store_ref),
            ("checkpoint_event_store", capabilities.checkpoint_event_store_ref),
            ("reconciliation_provider", capabilities.reconciliation_provider_ref),
        ):
            if reference is not None:
                self.resolve(kind, reference)  # type: ignore[arg-type]
        for kind, references in (
            ("memory_provider", capabilities.memory_provider_refs),
            ("hook", capabilities.hook_refs),
            ("after_cycle_hook", capabilities.after_cycle_hook_refs),
            ("observer", capabilities.observer_refs),
        ):
            for reference in references:
                self.resolve(kind, reference)  # type: ignore[arg-type]
        for extension in capabilities.checkpoint_extension_refs:
            self.resolve("checkpoint_extension", extension.reference)
        if require_checkpoint_v2 and capabilities.checkpoint_store_ref is None:
            raise DistributedCapabilityError("distributed v2 requires checkpoint_store_ref")
