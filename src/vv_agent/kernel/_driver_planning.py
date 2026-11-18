"""The actual operation/model planning rules, suspended only for host observations."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Protocol

from ._compaction_text import is_microcompacted_tool_content
from ._content_publication import declared_content_ref, make_record_before_publication
from ._context import protected_start
from ._context_messages import retained_start
from ._cycle_values import read_after_cycle_disallowed_tools
from ._execution_values import AgentTask, execution_id_for_values
from ._hydration import DefinitionContent, KeyUnavailable
from ._message_values import Message
from ._model_settings import ModelSettings, RetrySettings
from ._record_codec import RecordError
from ._records import Record, _RetainedMessages, copy_json, digest, record_identity
from ._reducer import Fold
from ._state import ExecutionState
from ._store import SessionTx, StoredRecord

W3_COMMAND_BINDING = "_vv_command_binding"
W3_COMMAND_TOOL = "bash"
W3_DEFAULT_OUTPUT_BYTES = 65_536
W3_UNSUPPORTED_CONTEXT_KEYS = frozenset({"bash_env", "bash_shell", "windows_shell_priority"})
W3_BASH_ARGUMENT_KEYS = frozenset({"auto_confirm", "command", "exec_dir", "stdin", "timeout_seconds", "yield_time_ms"})


class InvalidW3CommandProvider(ValueError):
    """A level-3 command cannot use the selected host provider."""

    code = "invalid_w3_command_provider"


class InvalidW3Command(ValueError):
    """A level-3 bash request cannot be bound to a stable command."""

    code = "invalid_w3_command"


class PlanningView(Protocol):
    @property
    def state(self) -> ExecutionState: ...
    @property
    def records(self) -> Sequence[StoredRecord]: ...
    @property
    def fold(self) -> Fold: ...
    @property
    def sid(self) -> str: ...
    def hydrated_definition(self, record: Record) -> DefinitionContent: ...
    def hydrate_record_field(self, record: Record, slot: str, ref: Any) -> Any: ...
    def hydrate_plan(self, record: Record) -> Record: ...
    def session_attributes(self) -> dict[str, Any]: ...


class PlanningRead(Enum):
    MODEL_BINDING = auto()
    ENDPOINT_SUPPORT = auto()
    HAS_HOOKS = auto()
    SHARED_STATE = auto()
    INTERRUPTION_MESSAGES = auto()
    PREPARE_BEFORE_LLM = auto()
    DEFAULT_MODEL_BINDING = auto()


@dataclass(frozen=True, slots=True)
class ResolveModel:
    request: dict[str, Any]
    purpose: str


@dataclass(frozen=True, slots=True)
class OrderedEndpoints:
    preferred: str | None


@dataclass(frozen=True, slots=True)
class ProviderAvailable:
    name: str


@dataclass(frozen=True, slots=True)
class ResolveW3Command:
    request: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BeforeCycle:
    cycle: int
    messages: list[Message]
    shared: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BeforeLLM:
    task: AgentTask
    cycle: int
    messages: list[Message]
    schemas: list[dict[str, Any]]
    shared: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DurableState:
    shared: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClosePlanning:
    status: str
    reason: str


class _PlanningStopIteration(Exception):
    def __init__(self, original: StopIteration):
        self.original = original


PlanningRequest = (
    PlanningRead
    | ResolveModel
    | OrderedEndpoints
    | ProviderAvailable
    | ResolveW3Command
    | BeforeCycle
    | BeforeLLM
    | DurableState
    | ClosePlanning
)


def plan(
    view: PlanningView,
    oid: str,
    request: dict[str, Any],
    kind: str,
    *,
    number: int = 1,
    dependencies: list[str] | None = None,
    capability: dict[str, Any] | None = None,
    purpose: str = "primary",
) -> Generator[PlanningRequest, Any, Record]:
    try:
        if kind == "model":
            yield ResolveModel(request, purpose)
            session = request["metadata"].get("vv_session", {})
            if purpose != "output_repair":
                session = session | {"model_binding": (yield PlanningRead.MODEL_BINDING)}
            if (yield PlanningRead.ENDPOINT_SUPPORT):
                preferred = view.state.preferred_endpoint_id
                order = yield OrderedEndpoints(preferred)
                if order:
                    session = session | {"endpoint_order": order}
            request = request | {"metadata": request["metadata"] | {"vv_session": session}}
        if kind == "model" and view.state.active_turn_id is not None:
            source = view.state.turns[view.state.active_turn_id].start
            definition = view.hydrated_definition(source)
            tools = request.get("tools")
            frozen = list(definition.tools)
            if isinstance(tools, list) and isinstance(frozen, list) and len(tools) == len(frozen) and tools == frozen:
                request = request | {"tools": frozen}
        if kind == "model" and isinstance(request.get("messages"), list):
            retained = view.fold._messages.retain_array(request["messages"])
            request = request | {"messages": retained.value if isinstance(retained, _RetainedMessages) else retained}
        provider_available = None
        if kind != "model":
            provider_available = yield ProviderAvailable(request["name"])
        binding = "model" if kind == "model" else request["name"] if provider_available else "function"
        feature_level = 1
        command_fields: dict[str, Any] = {}
        if _is_w3_bash_request(view, request):
            if not provider_available:
                raise InvalidW3CommandProvider("level-3 bash requires a registered trusted command provider")
            command_spec = yield ResolveW3Command(request)
            if type(command_spec) is not dict:
                raise InvalidW3Command("level-3 bash command normalization must be an object")
            request, command_fields = _w3_command_plan_fields(view, oid, request, command_spec)
            feature_level = 3
            binding = W3_COMMAND_TOOL
        request_digest = digest(_tool_call_request(request)) if kind != "model" else digest(request)
        return make_record_before_publication(
            "op_planned",
            session_id=view.sid,
            turn_id=view.state.active_turn_id,
            operation_id=oid,
            attempt=number,
            payload={
                "op_kind": kind,
                "purpose": purpose if kind == "model" else None,
                "request": request,
                "request_digest": request_digest,
                "context_version": str(view.records[-1].seq),
                "dependencies": dependencies or [],
                "tool": capability,
                "idempotency_key": (
                    f"{view.sid}/{oid}"
                    if feature_level == 3
                    else None
                    if capability and capability.get("idempotency") == "unsupported"
                    else f"{view.sid}/{oid}"
                ),
                "budget_admission": {},
                "not_before_ms": None,
                "provider_binding": binding,
                "consumed_unknowns": [
                    {"operation_id": key, "attempt": n}
                    for key, op in view.state.operations.items()
                    for n, a in op.attempts.items()
                    if kind == "model"
                    and request["metadata"].get("mode") != "child_summary"
                    and op.turn_id == view.state.active_turn_id
                    and a.state == "unknown"
                    and op.kind != "model"
                ],
                **command_fields,
            },
            feature_level=feature_level,
        )
    except StopIteration as error:
        raise _PlanningStopIteration(error) from error


def is_w3_command_plan(view: PlanningView, plan: Record) -> bool:
    """Return whether this record belongs to the level-3 bash command slice."""

    tid = plan.turn_id or view.state.active_turn_id
    if tid is None or tid not in view.state.turns:
        return False
    start = view.state.turns[tid].start
    request = plan._payload.get("request")
    return (
        start.feature_level == 3
        and type(start._payload.get("environment")) is dict
        and type(request) is dict
        and request.get("name") == W3_COMMAND_TOOL
    )


def assert_prepared_command_identity(view: PlanningView, plan: Record, call: Any) -> None:
    """Keep execution identity, provider and environment immovable across BeforeTool."""

    request = _tool_call_request(plan._payload["request"])
    patched = _tool_call_request(call.to_dict())
    if patched["id"] != request["id"]:
        raise InvalidW3Command("level-3 BeforeTool cannot change tool call identity")
    if patched["name"] != request["name"]:
        raise InvalidW3Command("level-3 BeforeTool cannot switch provider or tool name")
    envelope = _request_envelope(plan)
    _, environment, _ = _frozen_environment(view, plan.turn_id)
    if envelope.get("execution_identity", {}).get("environment_id") != environment["environment_id"]:
        raise InvalidW3Command("level-3 command cannot switch environment")


def committed_command_plan(view: PlanningView, started: Record) -> Record:
    """Return a detached execution plan from the retained start transaction."""

    stored = view.stored_record(started)
    if stored.record.kind != "op_started":
        raise InvalidW3Command("level-3 command submission requires op_started")
    operation = view.state.operations.get(started.operation_id or "")
    attempt = operation.attempts.get(started.attempt) if operation is not None else None
    if attempt is None or attempt.prepared is None or attempt.dispatch is None or attempt.dispatch.record_id != started.record_id:
        raise InvalidW3Command("level-3 command submission requires its committed prepared request")
    plan = view.hydrate_plan(attempt.execution_plan)
    return replace(plan, payload=copy_json(plan._payload))


def validate_command_submit(
    view: PlanningView,
    plan: Record,
    started: Record,
    command_spec: dict[str, Any],
) -> dict[str, Any]:
    """Verify the exact retained command against the post-start execution plan."""

    if not is_w3_command_plan(view, plan):
        raise InvalidW3Command("level-3 command binding is incomplete")
    committed = committed_command_plan(view, started)
    for field in (
        "request",
        "request_digest",
        "command_digest",
        "execution_identity",
        "provider_binding",
        "idempotency_key",
    ):
        if plan._payload.get(field) != committed._payload.get(field):
            raise InvalidW3Command(f"level-3 command changed after preparation: {field}")
    envelope = _request_envelope(committed)
    expected_identity = envelope.get("execution_identity")
    if committed._payload.get("execution_identity") != expected_identity:
        raise InvalidW3Command("level-3 command changed execution identity")
    if expected_identity is None or expected_identity.get("tenant_id") != _tenant_id(view):
        raise InvalidW3Command("level-3 command changed session tenant")
    descriptor, environment, operational_policy = _frozen_environment(view, plan.turn_id)
    if envelope.get("descriptor") != descriptor:
        raise InvalidW3Command("level-3 command changed environment descriptor")
    if envelope.get("environment") != environment:
        raise InvalidW3Command("level-3 command changed environment capabilities or generation")
    if envelope.get("operational_policy") != operational_policy:
        raise InvalidW3Command("level-3 command changed operational policy")
    call = _tool_call_request(committed._payload["request"])
    if envelope.get("command_spec") != command_spec:
        raise InvalidW3Command("level-3 command changed its normalized CommandSpec")
    if committed._payload.get("command_digest") != envelope.get("command_digest"):
        raise InvalidW3Command("level-3 command digest does not match its committed binding")
    if committed._payload.get("provider_binding") != W3_COMMAND_TOOL or call["name"] != W3_COMMAND_TOOL:
        raise InvalidW3Command("level-3 command changed its trusted provider")
    return envelope


def resolve_command_records(driver: Any, tx: SessionTx, records: list[Record]) -> list[Record]:
    """Resolve one deferred command HMAC before records are encoded or appended."""

    writer = None
    resolved: list[Record] = []
    for record in records:
        if (
            record.feature_level != 3
            or record.kind not in {"op_planned", "op_prepared"}
            or record._payload.get("command_digest") is not None
        ):
            resolved.append(record)
            continue
        plan = record
        if record.kind == "op_prepared":
            operation = driver.state.operations.get(record.operation_id or "")
            attempt = operation.attempts.get(record.attempt) if operation is not None else None
            if attempt is None:
                raise InvalidW3Command("prepared command has no planned operation")
            plan = attempt.plan
        envelope = _request_envelope(record)
        if envelope.get("execution_identity") != plan._payload.get("execution_identity"):
            raise InvalidW3Command("command preparation changed execution identity")
        if record.kind == "op_prepared" and record._payload.get("provider_binding") != plan._payload.get("provider_binding"):
            raise InvalidW3Command("command preparation changed provider")
        descriptor, environment, operational_policy = _frozen_environment(driver, record.turn_id)
        if envelope.get("descriptor") != descriptor or envelope.get("environment") != environment:
            raise InvalidW3Command("command preparation changed its frozen environment")
        if envelope.get("operational_policy") != operational_policy:
            raise InvalidW3Command("command preparation changed its frozen operational policy")
        call = _tool_call_request(record._payload["request"])
        if call["name"] != W3_COMMAND_TOOL or record._payload.get("provider_binding") != W3_COMMAND_TOOL:
            raise InvalidW3Command("command preparation changed its trusted provider")
        if record._payload.get("request_digest") != digest(call):
            raise InvalidW3Command("command request digest does not match the actual ToolCall")
        command_spec = envelope.get("command_spec")
        if type(command_spec) is not dict:
            raise InvalidW3Command("command proposal is missing its normalized CommandSpec")
        if writer is None:
            writer = driver.content_writer(tx)
        command_digest = writer.command_digest(
            command_spec=command_spec,
            environment=environment,
            operational_policy=operational_policy,
        )
        binding = copy_json(envelope) | {"command_digest": command_digest}
        payload = copy_json(record._payload)
        payload["command_digest"] = command_digest
        payload["request"] = call | {W3_COMMAND_BINDING: binding}
        resolved.append(
            Record(
                record_identity(
                    record.kind,
                    record.session_id,
                    record.turn_id,
                    record.operation_id,
                    record.attempt,
                    payload,
                ),
                record.kind,
                record.session_id,
                record.turn_id,
                record.operation_id,
                record.attempt,
                payload,
                feature_level=record.feature_level,
            )
        )
    return resolved


def _is_w3_bash_request(view: PlanningView, request: dict[str, Any]) -> bool:
    tid = view.state.active_turn_id
    if tid is None or tid not in view.state.turns or request.get("name") != W3_COMMAND_TOOL:
        return False
    start = view.state.turns[tid].start
    return start.feature_level == 3 and type(start._payload.get("environment")) is dict


def _w3_command_plan_fields(
    view: PlanningView,
    oid: str,
    request: dict[str, Any],
    command_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    call = _tool_call_request(request)
    descriptor, environment, operational_policy = _frozen_environment(view, view.state.active_turn_id)
    tenant_id = _tenant_id(view)
    operation_key = f"{view.sid}/{oid}"
    identity = {
        "operation_key": operation_key,
        "tenant_id": tenant_id,
        "environment_id": environment["environment_id"],
        "session_id": view.sid,
        "operation_id": oid,
        "execution_id": execution_id_for_values(tenant_id, environment["environment_id"], operation_key),
    }
    envelope = {
        "execution_identity": identity,
        "descriptor": descriptor,
        "environment": environment,
        "operational_policy": operational_policy,
        "command_spec": copy_json(command_spec),
        "command_digest": None,
    }
    return call | {W3_COMMAND_BINDING: envelope}, {"execution_identity": identity, "command_digest": None}


def _frozen_environment(view: PlanningView, tid: str | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if tid is None or tid not in view.state.turns:
        raise InvalidW3Command("level-3 command has no frozen turn environment")
    start = view.state.turns[tid].start
    raw = start._payload.get("environment")
    if start.feature_level != 3 or type(raw) is not dict:
        raise InvalidW3Command("level-3 command has no frozen turn environment")
    descriptor = view.hydrate_record_field(start, "environment_descriptor", raw.get("descriptor"))
    operational_policy = view.hydrate_record_field(start, "environment_operational_policy", raw.get("operational_policy"))
    capability_payload = {key: value for key, value in raw.items() if key not in {"descriptor", "operational_policy"}}
    if type(descriptor) is not dict or type(operational_policy) is not dict:
        raise InvalidW3Command("frozen environment content must be objects")
    return copy_json(descriptor), copy_json(capability_payload), copy_json(operational_policy)


def _tool_call_request(value: dict[str, Any]) -> dict[str, Any]:
    from ._tool_values import ToolCall

    return ToolCall.from_dict({key: item for key, item in value.items() if key != W3_COMMAND_BINDING}).to_dict()


def _request_envelope(record: Record) -> dict[str, Any]:
    request = record._payload.get("request")
    if type(request) is not dict or type(request.get(W3_COMMAND_BINDING)) is not dict:
        raise InvalidW3Command("level-3 command record is missing its encrypted binding")
    return request[W3_COMMAND_BINDING]


def _tenant_id(view: PlanningView) -> str:
    tenant_id = getattr(getattr(view, "store", None), "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise InvalidW3Command("level-3 command requires an authenticated tenant")
    return tenant_id


def effective_model(view: PlanningView) -> Generator[PlanningRequest, Any, dict[str, Any]]:
    try:
        selected = view.state.model_selection
        if selected is not None:
            payload = selected._payload
            input_id = payload["input"]["input_id"]
            applied = view.state.applied_inputs.get(input_id)
            if applied is not None:
                payload = applied._payload
            selection = payload["input"]["payload"]["selection"]
            binding = payload["model_binding"]
            try:
                if declared_content_ref(selection) is not None or declared_content_ref(binding) is not None:
                    raise KeyUnavailable("model selection requires hydrated logical state")
            except ValueError as exc:
                raise RecordError("model selection contains a malformed ContentRef") from exc
            if type(selection) is not dict or type(binding) is not dict:
                raise RecordError("model selection and binding must hydrate to objects")
            return selection | {"binding": binding}
        fork = view.session_attributes().get("app_server_fork")
        if fork is not None:
            return copy_json(fork["effective_model"])
        assert view.state.active_turn_id is not None
        task = view.hydrated_definition(view.state.turns[view.state.active_turn_id].start).task
        return {
            "model": task.model,
            "model_settings": task.model_settings.to_dict() if task.model_settings else {},
            "route": {},
            "binding": (yield PlanningRead.DEFAULT_MODEL_BINDING),
        }
    except StopIteration as error:
        raise _PlanningStopIteration(error) from error


def model_plan(view: PlanningView, messages: list[Message]) -> Generator[PlanningRequest, Any, Record | None]:
    try:
        protection = protected_start(view.records, view.state)
        protected_source = [Message.from_dict(copy_json(m.to_dict())) for m in messages] if protection is not None else []
        tid = view.state.active_turn_id
        assert tid is not None
        start = view.state.turns[tid].start
        copy_task = yield PlanningRead.HAS_HOOKS
        definition = view.hydrated_definition(start)
        task = deepcopy(definition.task) if copy_task else definition.task

        def prompt_too_long(attempt: Any) -> bool:
            receipt = attempt.result
            if receipt is None:
                return False
            result = view.hydrate_record_field(receipt, "result", receipt._payload["result"])
            if type(result) is not dict:
                raise RecordError("completed result hydration must return an object")
            return result.get("error_code") == "prompt_too_long"

        cycle = 1 + sum(
            op.turn_id == tid
            and op.kind == "model"
            and op.attempts[1].plan._payload["purpose"] == "primary"
            and not (op.selected_attempt and prompt_too_long(op.attempts[op.selected_attempt]))
            for op in view.state.operations.values()
        )
        shared = yield PlanningRead.SHARED_STATE
        enabled, additions = yield PlanningRead.INTERRUPTION_MESSAGES
        if enabled:
            messages.extend(additions)
        enabled, additions = yield BeforeCycle(cycle, messages, shared)
        if enabled:
            messages.extend(additions)
        has_hooks = yield PlanningRead.PREPARE_BEFORE_LLM
        messages, schemas = yield BeforeLLM(
            task,
            cycle,
            messages,
            copy_json(list(definition.tools)) if has_hooks else list(definition.tools),
            shared,
        )
        retained_start(protected_source, messages, protection)
        denied = read_after_cycle_disallowed_tools(shared)
        if denied:
            schemas = [schema for schema in schemas if schema["function"]["name"] not in denied]
        if not any(schema["function"]["name"] == "read_file" for schema in schemas) and any(
            (message.role == "tool" and is_microcompacted_tool_content(message.content))
            or any(message.metadata.get("_vv_agent_compaction", {}).get(key) for key in ("artifacts", "cursors"))
            for message in messages
        ):
            yield ClosePlanning("failed", "microcompaction_recovery_unavailable")
            return None
        effective = yield from effective_model(view)
        request = {
            "model": effective["model"],
            "route": effective["route"],
            "messages": [m.to_dict() for m in messages],
            "tools": schemas,
            "metadata": dict(task.metadata)
            | {
                "vv_session": {
                    "shared_state": (yield DurableState(shared)),
                    "cycle_index": cycle,
                    "model_binding": effective["binding"],
                }
            },
            "prompt_bundle": task.prompt_bundle.to_dict(),
            "model_settings": replace(
                ModelSettings.from_dict(effective.get("model_settings", {})),
                retry=RetrySettings(max_attempts=1, backoff_seconds=0),
            ).to_dict(),
        }
        number = (
            sum(
                op.kind == "model" and op.turn_id == tid and op.attempts[1].plan._payload["purpose"] == "primary"
                for op in view.state.operations.values()
            )
            + 1
        )
        return (yield from plan(view, f"{tid}/model/primary/{number}", request, "model"))
    except StopIteration as error:
        raise _PlanningStopIteration(error) from error
