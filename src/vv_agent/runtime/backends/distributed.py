from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.state import StateStoreSpec
from vv_agent.tools import ToolRegistry, build_default_registry
from vv_agent.types import AgentTask

DISTRIBUTED_RUN_SCHEMA_VERSION = "vv-agent.distributed-run.v1"
DEFAULT_TOOLSET_ID = "vv-agent.builtin-tools"
DEFAULT_TOOLSET_VERSION = "1"
DEFAULT_TOOLSET_SCHEMA_DIGEST = "f85422117d41d28ffa3cdfcfd9a42892854de624808fadc2124f4ebe7a452b61"
DEFAULT_CYCLE_NAME = "vv_agent.distributed.run_single_cycle"
DEFAULT_LEASE_DURATION_MS = 5 * 60 * 1000

CapabilityKind = Literal[
    "llm_client",
    "workspace_backend",
    "approval_provider",
    "approval_broker",
    "cancellation",
    "event_sink",
    "app_state",
    "memory_provider",
    "hook",
    "observer",
    "sub_task_manager",
    "tool_predicate",
]


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

    def __post_init__(self) -> None:
        if self.approval not in {"default", "always", "never", "on_request"}:
            raise DistributedContractError("tool_policy.approval is unsupported")
        for field_name, values in (
            ("tool_policy.allowed_tools", self.allowed_tools),
            ("tool_policy.disallowed_tools", self.disallowed_tools),
        ):
            if values is not None and any(not isinstance(value, str) or not value.strip() for value in values):
                raise DistributedContractError(f"{field_name} must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
            "disallowed_tools": list(self.disallowed_tools),
            "approval": self.approval,
            "predicate_ref": self.predicate_ref.to_dict() if self.predicate_ref is not None else None,
        }

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
        approval = payload.get("approval", "default")
        if not isinstance(approval, str):
            raise DistributedContractError("tool_policy.approval must be a string")
        predicate = payload.get("predicate_ref")
        return cls(
            allowed_tools=tuple(allowed) if allowed is not None else None,
            disallowed_tools=tuple(disallowed),
            approval=approval,  # type: ignore[arg-type]
            predicate_ref=(
                CapabilityRef.from_dict(predicate, field_name="tool_policy.predicate_ref")
                if predicate is not None
                else None
            ),
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
        )


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
    app_state_ref: CapabilityRef | None = None
    sub_task_manager_ref: CapabilityRef | None = None
    memory_provider_refs: tuple[CapabilityRef, ...] = ()
    hook_refs: tuple[CapabilityRef, ...] = ()
    observer_refs: tuple[CapabilityRef, ...] = ()

    def __post_init__(self) -> None:
        if (self.approval_provider_ref is None) != (self.approval_broker_ref is None):
            raise DistributedContractError(
                "approval_provider_ref and approval_broker_ref must be declared together"
            )
        if self.approval_timeout_seconds is not None and (
            isinstance(self.approval_timeout_seconds, bool)
            or not isinstance(self.approval_timeout_seconds, (int, float))
            or not math.isfinite(float(self.approval_timeout_seconds))
            or self.approval_timeout_seconds <= 0
        ):
            raise DistributedContractError("approval_timeout_seconds must be a finite positive number or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "toolset_ref": self.toolset_ref.to_dict(),
            "tool_policy": self.tool_policy.to_dict(),
            "llm_client_ref": self.llm_client_ref.to_dict() if self.llm_client_ref else None,
            "workspace_backend_ref": self.workspace_backend_ref.to_dict() if self.workspace_backend_ref else None,
            "approval_provider_ref": self.approval_provider_ref.to_dict() if self.approval_provider_ref else None,
            "approval_broker_ref": self.approval_broker_ref.to_dict() if self.approval_broker_ref else None,
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "cancellation_ref": self.cancellation_ref.to_dict() if self.cancellation_ref else None,
            "event_sink_ref": self.event_sink_ref.to_dict() if self.event_sink_ref else None,
            "app_state_ref": self.app_state_ref.to_dict() if self.app_state_ref else None,
            "sub_task_manager_ref": self.sub_task_manager_ref.to_dict() if self.sub_task_manager_ref else None,
            "memory_provider_refs": [reference.to_dict() for reference in self.memory_provider_refs],
            "hook_refs": [reference.to_dict() for reference in self.hook_refs],
            "observer_refs": [reference.to_dict() for reference in self.observer_refs],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedCapabilities:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("capabilities must be an object")

        def refs(key: str) -> tuple[CapabilityRef, ...]:
            values = payload.get(key, [])
            if not isinstance(values, list):
                raise DistributedContractError(f"capabilities.{key} must be an array")
            return tuple(
                CapabilityRef.from_dict(value, field_name=f"capabilities.{key}[{index}]")
                for index, value in enumerate(values)
            )

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
            app_state_ref=_optional_ref(payload, "app_state_ref"),
            sub_task_manager_ref=_optional_ref(payload, "sub_task_manager_ref"),
            memory_provider_refs=refs("memory_provider_refs"),
            hook_refs=refs("hook_refs"),
            observer_refs=refs("observer_refs"),
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
            isinstance(self.log_preview_chars, bool)
            or not isinstance(self.log_preview_chars, int)
            or self.log_preview_chars < 0
        ):
            raise DistributedContractError("runtime_recipe.log_preview_chars must be a non-negative integer or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_file": self.settings_file,
            "backend": self.backend,
            "model": self.model,
            "workspace": self.workspace,
            "timeout_seconds": float(self.timeout_seconds),
            "log_preview_chars": self.log_preview_chars,
            "state_store": self.state_store.to_dict() if self.state_store is not None else None,
            "capabilities": self.capabilities.to_dict(),
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
            state_store=(
                StateStoreSpec.from_dict(payload["state_store"])
                if payload.get("state_store") is not None
                else None
            ),
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
    lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS
    schema_version: str = DISTRIBUTED_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DISTRIBUTED_RUN_SCHEMA_VERSION:
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
        ):
            raise DistributedContractError("distributed envelope deadline_unix_ms must be a non-negative integer or null")
        if (
            isinstance(self.lease_duration_ms, bool)
            or not isinstance(self.lease_duration_ms, int)
            or self.lease_duration_ms <= 0
        ):
            raise DistributedContractError("distributed envelope lease_duration_ms must be a positive integer")

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
    ) -> DistributedRunEnvelope:
        effective_run_id = run_id or str(task.metadata.get("_vv_agent_run_id") or task.task_id)
        idempotency_key = f"{effective_run_id}:cycle:{cycle_index}"
        return cls(
            job_id=idempotency_key,
            run_id=effective_run_id,
            task=task,
            recipe=recipe,
            cycle_name=cycle_name,
            cycle_index=cycle_index,
            idempotency_key=idempotency_key,
            deadline_unix_ms=deadline_unix_ms,
            lease_duration_ms=lease_duration_ms,
        )

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
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "task": self.task.to_dict(),
            "recipe": self.recipe.to_dict(),
            "cycle_name": self.cycle_name,
            "cycle_index": self.cycle_index,
            "idempotency_key": self.idempotency_key,
            "deadline_unix_ms": self.deadline_unix_ms,
            "lease_duration_ms": self.lease_duration_ms,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedRunEnvelope:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("distributed envelope must be an object")
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, str):
            raise DistributedContractError("distributed envelope schema_version must be a string")
        task_payload = payload.get("task")
        if not isinstance(task_payload, Mapping):
            raise DistributedContractError("distributed envelope task must be an object")
        return cls(
            schema_version=schema_version,
            job_id=_required_string(payload, "job_id"),
            run_id=_required_string(payload, "run_id"),
            task=AgentTask.from_dict(dict(task_payload)),
            recipe=RuntimeRecipe.from_dict(payload.get("recipe")),
            cycle_name=_required_string(payload, "cycle_name"),
            cycle_index=payload.get("cycle_index"),
            idempotency_key=_required_string(payload, "idempotency_key"),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            lease_duration_ms=payload.get("lease_duration_ms", DEFAULT_LEASE_DURATION_MS),
        )


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
            raise DistributedCapabilityError(
                f"unknown distributed capability {kind} {reference.id}@{reference.version}"
            )
        return self._capabilities[key]

    def resolve_toolset(self, reference: ToolsetRef) -> ToolRegistry:
        registry = self._toolsets.get(reference.key)
        if registry is None:
            raise DistributedCapabilityError(
                f"unknown distributed toolset {reference.id}@{reference.version}"
            )
        actual = toolset_schema_digest(registry)
        if actual != reference.schema_digest:
            raise DistributedCapabilityError(
                f"toolset {reference.id}@{reference.version} schema digest mismatch: "
                f"expected {reference.schema_digest}, got {actual}"
            )
        return registry

    def validate(self, capabilities: DistributedCapabilities) -> None:
        self.resolve_toolset(capabilities.toolset_ref)
        capabilities.tool_policy.resolve(self)
        for kind, reference in (
            ("llm_client", capabilities.llm_client_ref),
            ("workspace_backend", capabilities.workspace_backend_ref),
            ("approval_provider", capabilities.approval_provider_ref),
            ("approval_broker", capabilities.approval_broker_ref),
            ("cancellation", capabilities.cancellation_ref),
            ("event_sink", capabilities.event_sink_ref),
            ("app_state", capabilities.app_state_ref),
            ("sub_task_manager", capabilities.sub_task_manager_ref),
        ):
            if reference is not None:
                self.resolve(kind, reference)  # type: ignore[arg-type]
        for kind, references in (
            ("memory_provider", capabilities.memory_provider_refs),
            ("hook", capabilities.hook_refs),
            ("observer", capabilities.observer_refs),
        ):
            for reference in references:
                self.resolve(kind, reference)  # type: ignore[arg-type]
