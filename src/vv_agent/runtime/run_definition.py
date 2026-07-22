from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
from typing import Any

from vv_agent.agent import Agent
from vv_agent.checkpoint import (
    CREDENTIAL_REDACTION_VALUE,
    CheckpointError,
    compute_run_definition_digest,
    utf16_sort_key,
    validate_checkpoint_extension,
    validate_run_definition,
)
from vv_agent.config import ResolvedModelConfig
from vv_agent.model_settings import ModelSettings
from vv_agent.run_config import RunConfig, ToolPolicy
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, Message

_SPECIAL_CAPABILITY_SLOTS = {
    "context": "context_ref",
    "workspace": "workspace_ref",
    "session": "session_ref",
    "tool_policy.predicate": "tool_policy_predicate_ref",
}
_RUNTIME_METADATA_KEYS = {
    "_vv_agent_initial_budget_usage",
    "_vv_agent_active_cycle_index",
    "_vv_agent_checkpoint_controller",
    "_vv_agent_checkpoint_budget_snapshot",
    "_vv_agent_run_id",
    "_vv_agent_trace_id",
}
_KNOWN_CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "api-key"})


def build_run_definition(
    *,
    agent: Agent,
    root_input: str,
    run_config: RunConfig,
    resolved: ResolvedModelConfig,
    model_settings: ModelSettings,
    task: AgentTask,
    registry: ToolRegistry,
    initial_messages: list[Message],
    credential_slots: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build the immutable, credential-redacted definition for one root run."""

    checkpoint_config = run_config.checkpoint_config
    if checkpoint_config is None:
        raise CheckpointError(
            "checkpoint_config is required to build a run definition",
            code="checkpoint_config_invalid",
        )
    refs = deepcopy(checkpoint_config.capability_refs)
    _validate_behavior_capability_refs(agent=agent, run_config=run_config, refs=refs)

    model_settings_payload = model_settings.to_dict()
    transport_timeout = model_settings_payload.pop("timeout_seconds", None)

    _normalize_extra_headers(model_settings_payload)
    provider_slots = list(credential_slots or [])
    normalized_provider_slots = sorted(set(provider_slots), key=utf16_sort_key)
    if normalized_provider_slots != provider_slots:
        raise CheckpointError(
            "provider credential_slots must be sorted and unique",
            code="checkpoint_credential_slots_invalid",
        )
    normalized_slots = sorted(
        set([*checkpoint_config.credential_slots, *normalized_provider_slots]),
        key=utf16_sort_key,
    )

    context_ref = _take_ref(refs, "context", required=run_config.context is not None)
    workspace_ref = _take_ref(
        refs,
        "workspace",
        required=run_config.workspace is not None or run_config.workspace_backend is not None,
    )
    session_ref = _take_ref(refs, "session", required=run_config.session is not None)
    predicate_ref = _take_ref(
        refs,
        "tool_policy.predicate",
        required=bool(run_config.tool_policy and run_config.tool_policy.can_use_tool is not None),
    )

    definition: dict[str, Any] = {
        "schema_version": "vv-agent.run-definition.v1",
        "agent": {"name": agent.name, "type": task.agent_type},
        "root_input": root_input,
        "compiled_prompt": task.system_prompt,
        "initial_messages": [message.to_dict() for message in initial_messages],
        "initial_shared_state": deepcopy(run_config.shared_state or {}),
        "run_metadata": _behavior_metadata(agent=agent, run_config=run_config),
        "context_ref": context_ref,
        "model": {
            "backend": resolved.backend,
            "model_id": resolved.model_id,
            "settings": model_settings_payload,
            "transport_timeout_seconds": transport_timeout,
        },
        "credential_slots": normalized_slots,
        "runtime_controls": {
            "max_cycles": task.max_cycles,
            "max_handoffs": run_config.max_handoffs,
            "no_tool_policy": task.no_tool_policy,
            "memory_compact_threshold": task.memory_compact_threshold,
            "memory_threshold_percentage": task.memory_threshold_percentage,
            "allow_interruption": task.allow_interruption,
            "native_multimodal": task.native_multimodal,
            "tool_use_behavior": agent.tool_use_behavior,
            "stop_at_tool_names": list(agent.stop_at_tool_names),
        },
        "tools": _tool_definitions(registry=registry, task=task, refs=refs),
        "tool_policy": _tool_policy_definition(
            run_config.tool_policy,
            predicate_ref=predicate_ref,
            approval_timeout_seconds=run_config.approval_timeout_seconds,
        ),
        "checkpoint_policy": {
            "ambiguous_model_policy": checkpoint_config.ambiguous_model_policy.value,
            "ambiguous_tool_policy": checkpoint_config.ambiguous_tool_policy.value,
            "max_extension_state_bytes": checkpoint_config.max_extension_state_bytes,
        },
        "budget_limits": (
            run_config.budget_limits.to_dict()
            if run_config.budget_limits is not None and run_config.budget_limits.has_limits
            else None
        ),
        "output_schema": _output_schema(agent.output_type),
        "workspace_ref": workspace_ref,
        "session_ref": session_ref,
        "extensions": _extension_definitions(run_config),
        "capability_refs": dict(sorted(refs.items(), key=lambda item: utf16_sort_key(item[0]))),
    }
    _require_declared_credential_headers(definition, normalized_slots)
    _redact_credential_slots(definition, normalized_slots)
    validated = validate_run_definition(definition)
    return validated, compute_run_definition_digest(validated)


def _tool_definitions(
    *,
    registry: ToolRegistry,
    task: AgentTask,
    refs: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for schema in plan_tool_schemas(registry=registry, task=task):
        function = schema.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not registry.has_executor(name):
            continue
        executor = registry.get_executor(name)
        tool_metadata = executor.tool_metadata
        if tool_metadata is None:
            tool_metadata_payload = None
        else:
            to_dict = getattr(tool_metadata, "to_dict", None)
            if not callable(to_dict):
                raise CheckpointError(
                    f"tool {name!r} has invalid typed metadata",
                    code="checkpoint_definition_invalid",
                )
            tool_metadata_payload = to_dict()
        needs_approval = getattr(executor, "needs_approval", False)
        if callable(needs_approval):
            approval_ref = _take_ref(refs, f"tool_approval:{name}", required=True)
            approval = {"mode": "referenced", "ref": approval_ref}
        else:
            approval = {"mode": "static", "required": bool(needs_approval)}
        definitions.append(
            {
                "schema": schema,
                "tool_metadata": tool_metadata_payload,
                "timeout_seconds": getattr(executor, "timeout_seconds", None),
                "approval": approval,
            }
        )
    return definitions


def _tool_policy_definition(
    policy: ToolPolicy | None,
    *,
    predicate_ref: dict[str, str] | None,
    approval_timeout_seconds: float | None,
) -> dict[str, Any]:
    policy = policy or ToolPolicy()
    return {
        "allowed_tools": _normalized_name_set(policy.allowed_tools),
        "disallowed_tools": _normalized_name_set(policy.disallowed_tools) or [],
        "approval": policy.approval,
        "predicate_ref": predicate_ref,
        "approval_timeout_seconds": approval_timeout_seconds,
        "denied_side_effects": [getattr(value, "value", value) for value in getattr(policy, "denied_side_effects", [])],
        "denied_capability_tags": list(getattr(policy, "denied_capability_tags", [])),
        "deny_terminal_tools": getattr(policy, "deny_terminal_tools", False),
        "denied_cost_dimensions": list(getattr(policy, "denied_cost_dimensions", [])),
    }


def _extension_definitions(run_config: RunConfig) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    checkpoint_config = run_config.checkpoint_config
    if checkpoint_config is None:
        raise CheckpointError(
            "checkpoint_config is required to define checkpoint extensions",
            code="checkpoint_config_invalid",
        )
    required_namespaces = set(checkpoint_config.required_extension_namespaces)
    for extension in run_config.checkpoint_extensions:
        validate_checkpoint_extension(extension)
        if extension.namespace in seen:
            raise CheckpointError(
                f"duplicate checkpoint extension {extension.namespace}",
                code="checkpoint_extension_namespace_duplicate",
            )
        seen.add(extension.namespace)
        required = bool(extension.required or extension.namespace in required_namespaces)
        definitions.append(
            {
                "namespace": extension.namespace,
                "version": extension.version,
                "required": required,
            }
        )
    missing = required_namespaces - seen
    if missing:
        raise CheckpointError(
            f"missing required checkpoint extensions: {', '.join(sorted(missing))}",
            code="checkpoint_extension_missing",
        )
    return sorted(definitions, key=lambda item: utf16_sort_key(item["namespace"]))


def _validate_behavior_capability_refs(
    *,
    agent: Agent,
    run_config: RunConfig,
    refs: dict[str, dict[str, str]],
) -> None:
    required_slots: list[str] = []
    if callable(agent.instructions):
        required_slots.append("agent.instructions")
    required_slots.extend(f"input_guardrail:{index}" for index, _ in enumerate(agent.input_guardrails))
    required_slots.extend(f"output_guardrail:{index}" for index, _ in enumerate(agent.output_guardrails))
    required_slots.extend(f"runtime_hook:{index}" for index, _ in enumerate([*agent.hooks, *run_config.hooks]))
    required_slots.extend(f"after_cycle_hook:{index}" for index, _ in enumerate(run_config.after_cycle_hooks))
    required_slots.extend(f"context_provider:{index}" for index, _ in enumerate(run_config.context_providers))
    required_slots.extend(f"memory_provider:{index}" for index, _ in enumerate(run_config.memory_providers))
    for present, slot in (
        (bool(_behavior_metadata(agent=agent, run_config=run_config)), "behavior_affecting_run_metadata"),
        (run_config.before_cycle_messages is not None, "before_cycle_messages"),
        (run_config.interruption_messages is not None, "interruption_messages"),
        (run_config.approval_provider is not None, "approval_provider"),
        (run_config.host_cost_meter is not None, "host_cost_meter"),
        (run_config.reconciliation_provider is not None, "reconciliation_provider"),
        (run_config.tool_registry_factory is not None, "tool_registry_factory"),
        (run_config.sub_task_manager is not None, "sub_task_manager"),
        (
            (agent.output_validation_enabled and agent.output_validator is not None)
            or agent.output_type not in {None, str, dict, list},
            "output_validator",
        ),
        (
            agent.output_validation_enabled and agent.output_repair is not None and agent.output_validation_max_repairs == 1,
            "output_repair",
        ),
    ):
        if present:
            required_slots.append(slot)
    missing = [slot for slot in required_slots if slot not in refs]
    if missing:
        raise CheckpointError(
            f"checkpoint v2 requires stable capability refs for: {', '.join(missing)}",
            code="checkpoint_definition_unstable",
        )


def _take_ref(
    refs: dict[str, dict[str, str]],
    slot: str,
    *,
    required: bool,
) -> dict[str, str] | None:
    value = refs.pop(slot, None)
    if required and value is None:
        raise CheckpointError(
            f"checkpoint v2 requires stable capability ref {slot!r}",
            code="checkpoint_definition_unstable",
        )
    return value


def _normalized_name_set(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    return sorted(set(values), key=utf16_sort_key)


def _output_schema(output_type: Any) -> dict[str, Any] | None:
    if output_type is None:
        return None
    if output_type is str:
        return {"type": "string"}
    if output_type is dict:
        return {"type": "object"}
    if output_type is list:
        return {"type": "array"}
    model_json_schema = getattr(output_type, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
        if isinstance(schema, dict):
            return schema
    if isinstance(output_type, type) and is_dataclass(output_type):
        properties: dict[str, Any] = {}
        required: list[str] = []
        for item in fields(output_type):
            properties[item.name] = _annotation_schema(item.type)
            if item.default is MISSING and item.default_factory is MISSING:
                required.append(item.name)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema
    raise CheckpointError(
        "output_type requires a stable JSON schema for checkpoint v2",
        code="checkpoint_definition_unstable",
    )


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
    }.get(annotation, {})


def _redact_credential_slots(definition: dict[str, Any], slots: list[str]) -> None:
    for pointer in slots:
        if not pointer.startswith("/"):
            raise CheckpointError(
                "credential slot must be a non-empty RFC 6901 JSON pointer",
                code="checkpoint_credential_slots_invalid",
            )
        current: Any = definition
        tokens = pointer[1:].split("/")
        for raw_token in tokens[:-1]:
            token = _decode_pointer_token(raw_token)
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                current = current[int(token)]
            else:
                raise CheckpointError(
                    f"credential slot {pointer!r} does not resolve",
                    code="checkpoint_credential_slot_unresolved",
                )
        final = _decode_pointer_token(tokens[-1])
        if isinstance(current, dict) and final in current:
            current[final] = CREDENTIAL_REDACTION_VALUE
        elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
            current[int(final)] = CREDENTIAL_REDACTION_VALUE
        else:
            raise CheckpointError(
                f"credential slot {pointer!r} does not resolve",
                code="checkpoint_credential_slot_unresolved",
            )


def _decode_pointer_token(token: str) -> str:
    index = 0
    while index < len(token):
        if token[index] == "~":
            if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                raise CheckpointError(
                    "credential slot contains an invalid RFC 6901 escape",
                    code="checkpoint_credential_slots_invalid",
                )
            index += 2
            continue
        index += 1
    return token.replace("~1", "/").replace("~0", "~")


def _behavior_metadata(*, agent: Agent, run_config: RunConfig) -> dict[str, Any]:
    metadata = {**agent.metadata, **run_config.metadata}
    return {key: deepcopy(value) for key, value in metadata.items() if key not in _RUNTIME_METADATA_KEYS}


def _normalize_extra_headers(model_settings_payload: dict[str, Any]) -> None:
    raw_headers = model_settings_payload.get("extra_headers")
    if raw_headers is None:
        return
    if not isinstance(raw_headers, dict):
        raise CheckpointError(
            "model extra_headers must be an object",
            code="checkpoint_definition_invalid",
        )
    normalized: dict[str, Any] = {}
    for raw_name, value in raw_headers.items():
        if not isinstance(raw_name, str) or not raw_name.isascii():
            raise CheckpointError(
                "model extra header names must be ASCII strings",
                code="checkpoint_definition_invalid",
            )
        name = raw_name.lower()
        if name in normalized:
            raise CheckpointError(
                f"model extra header name collides after ASCII lowercasing: {raw_name!r}",
                code="checkpoint_definition_header_collision",
            )
        normalized[name] = value
    model_settings_payload["extra_headers"] = normalized


def _require_declared_credential_headers(
    definition: dict[str, Any],
    credential_slots: list[str],
) -> None:
    settings = definition["model"]["settings"]
    headers = settings.get("extra_headers") if isinstance(settings, dict) else None
    if not isinstance(headers, dict):
        return
    declared = set(credential_slots)
    missing = [
        name
        for name in sorted(headers)
        if name in _KNOWN_CREDENTIAL_HEADERS and f"/model/settings/extra_headers/{name}" not in declared
    ]
    if missing:
        raise CheckpointError(
            f"model credential headers require explicit credential_slots: {', '.join(missing)}",
            code="checkpoint_definition_unstable",
        )
