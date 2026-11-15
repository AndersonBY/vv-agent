"""The actual operation/model planning rules, suspended only for host observations."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Protocol

from ._compaction_text import is_microcompacted_tool_content
from ._content_publication import make_record_before_publication
from ._context import protected_start
from ._context_messages import retained_start
from ._cycle_values import read_after_cycle_disallowed_tools
from ._execution_values import AgentTask
from ._hydration import DefinitionContent
from ._message_values import Message
from ._model_settings import ModelSettings, RetrySettings
from ._record_codec import RecordError
from ._records import Record, _RetainedMessages, copy_json, digest
from ._reducer import Fold
from ._state import ExecutionState
from ._store import StoredRecord


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
    PlanningRead | ResolveModel | OrderedEndpoints | ProviderAvailable | BeforeCycle | BeforeLLM | DurableState | ClosePlanning
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
        binding = "model" if kind == "model" else request["name"] if (yield ProviderAvailable(request["name"])) else "function"
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
                "request_digest": digest(request),
                "context_version": str(view.records[-1].seq),
                "dependencies": dependencies or [],
                "tool": capability,
                "idempotency_key": None if capability and capability.get("idempotency") == "unsupported" else f"{view.sid}/{oid}",
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
            },
        )
    except StopIteration as error:
        raise _PlanningStopIteration(error) from error


def effective_model(view: PlanningView) -> Generator[PlanningRequest, Any, dict[str, Any]]:
    try:
        selected = view.state.model_selection
        if selected is not None:
            selection = selected._payload["input"]["payload"]["selection"]
            return selection | {"binding": selected._payload["model_binding"]}
        fork = view.records[0].record._payload["attributes"].get("app_server_fork")
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
