from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from vv_agent.budget import BudgetExhaustion
from vv_agent.events import (
    ModelCallCompletedEvent,
    ModelCallFailedEvent,
    ModelCallStartedEvent,
    RunEvent,
)
from vv_agent.runtime.token_usage import normalize_token_usage, summarize_task_token_usage
from vv_agent.types import (
    LLMResponse,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    TaskTokenUsage,
    TokenUsage,
)

if TYPE_CHECKING:
    from vv_agent.llm.base import LlmRequest


_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DEFINITIVE_ERROR_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "prompt_too_long",
    "request too large",
    "too many tokens",
)


@dataclass(frozen=True, slots=True)
class ModelCallIdentity:
    call_id: str
    operation_id: str
    attempt: int
    operation: ModelCallOperation
    cycle_index: int
    backend: str
    model: str

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        attempt: int,
        operation: ModelCallOperation,
        cycle_index: int,
        backend: str,
        model: str,
    ) -> ModelCallIdentity:
        probe = ModelCallRecord(
            call_id=f"{operation_id}:attempt:{attempt}",
            operation_id=operation_id,
            attempt=attempt,
            operation=operation,
            cycle_index=cycle_index,
            backend=backend,
            model=model,
            status=ModelCallStatus.COMPLETED,
            usage=TokenUsage(),
        )
        return cls(
            call_id=probe.call_id,
            operation_id=probe.operation_id,
            attempt=probe.attempt,
            operation=probe.operation,
            cycle_index=probe.cycle_index,
            backend=probe.backend,
            model=probe.model,
        )


@dataclass(slots=True)
class ModelCallLedger:
    _records: list[ModelCallRecord] = field(default_factory=list, repr=False)

    def replace(self, records: list[ModelCallRecord]) -> None:
        candidate = summarize_task_token_usage(records)
        self._records = deepcopy(candidate.model_calls)

    def append(self, record: ModelCallRecord) -> None:
        candidate = [*self._records, deepcopy(record)]
        summarize_task_token_usage(candidate)
        self._records = candidate

    def records(self) -> list[ModelCallRecord]:
        return deepcopy(self._records)

    def usage(self) -> TaskTokenUsage:
        return summarize_task_token_usage(self._records)

    def previous_agent_input_tokens(self, cycle_index: int) -> int | None:
        for record in reversed(self._records):
            if (
                record.operation is ModelCallOperation.AGENT_CYCLE
                and record.cycle_index < cycle_index
                and record.status is ModelCallStatus.COMPLETED
            ):
                return record.usage.input_tokens
        return None


@dataclass(frozen=True, slots=True)
class ModelCallBudgetObservation:
    exhaustion: BudgetExhaustion | None = None
    event: RunEvent | None = None


class DurableModelDispatcher(Protocol):
    def complete_model(
        self,
        *,
        cycle_index: int,
        operation_slot: str,
        operation: ModelCallOperation,
        backend: str,
        model: str,
        request: LlmRequest,
        invoke: Callable[[], LLMResponse],
        accounting: ModelCallCoordinator,
    ) -> ModelCallDispatchResult: ...


class ModelCallBudgetExhausted(RuntimeError):
    vv_agent_control_flow = True

    def __init__(self, exhaustion: BudgetExhaustion) -> None:
        super().__init__("Run budget exhausted by a model call.")
        self.exhaustion = exhaustion


@dataclass(frozen=True, slots=True)
class ModelCallTerminal:
    record: ModelCallRecord
    event: RunEvent
    budget: ModelCallBudgetObservation


@dataclass(frozen=True, slots=True)
class ModelCallDispatchResult:
    response: LLMResponse
    usage: TokenUsage
    identity: ModelCallIdentity
    budget_exhaustion: BudgetExhaustion | None = None
    replayed: bool = False


class ModelCallCoordinator:
    def __init__(
        self,
        *,
        ledger: ModelCallLedger,
        run_id: str,
        trace_id: str,
        agent_name: str | None,
        session_id: str | None,
        parent_run_id: str | None,
        event_sink: Callable[[RunEvent], None] | None,
        budget_observer: Callable[[int, TokenUsage], ModelCallBudgetObservation] | None = None,
        durable_dispatcher: DurableModelDispatcher | None = None,
    ) -> None:
        self.ledger = ledger
        self.run_id = run_id
        self.trace_id = trace_id
        self.agent_name = agent_name
        self.session_id = session_id
        self.parent_run_id = parent_run_id
        self.event_sink = event_sink
        self.budget_observer = budget_observer
        self.durable_dispatcher = durable_dispatcher
        self._slot_counts: dict[tuple[int, str], int] = {}

    @property
    def usage(self) -> TaskTokenUsage:
        return self.ledger.usage()

    def dispatch(
        self,
        *,
        operation: ModelCallOperation,
        cycle_index: int,
        operation_slot: str,
        backend: str,
        model: str,
        request: LlmRequest,
        invoke: Callable[[], LLMResponse],
    ) -> ModelCallDispatchResult:
        if self.durable_dispatcher is not None:
            return self.durable_dispatcher.complete_model(
                cycle_index=cycle_index,
                operation_slot=operation_slot,
                operation=operation,
                backend=backend,
                model=model,
                request=request,
                invoke=invoke,
                accounting=self,
            )

        identity = self.new_identity(
            cycle_index=cycle_index,
            operation_slot=operation_slot,
            operation=operation,
            backend=backend,
            model=model,
        )
        self.emit(self.started_event(identity))
        try:
            response = invoke()
        except BaseException as error:
            terminal = self.failed_terminal(
                identity,
                error_code=model_error_code(error),
                ambiguous=not is_definitive_model_error(error),
            )
            self.commit_terminal(terminal)
            raise
        usage = normalize_token_usage(
            response.raw.get("usage"),
            usage_source=response.raw.get("usage_source"),
            cache_status=response.raw.get("cache_status"),
        )
        terminal = self.completed_terminal(identity, usage)
        self.commit_terminal(terminal)
        return ModelCallDispatchResult(
            response=response,
            usage=usage,
            identity=identity,
            budget_exhaustion=terminal.budget.exhaustion,
        )

    def new_identity(
        self,
        *,
        cycle_index: int,
        operation_slot: str,
        operation: ModelCallOperation,
        backend: str,
        model: str,
        attempt: int = 1,
        operation_id: str | None = None,
    ) -> ModelCallIdentity:
        normalized_slot = _normalize_operation_slot(operation_slot)
        if operation_id is None:
            key = (cycle_index, normalized_slot)
            count = self._slot_counts.get(key, 0) + 1
            self._slot_counts[key] = count
            slot = normalized_slot if count == 1 else f"{normalized_slot}_{count}"
            operation_id = f"op_model_cycle_{cycle_index}_{slot}"
        return ModelCallIdentity.create(
            operation_id=operation_id,
            attempt=attempt,
            operation=operation,
            cycle_index=cycle_index,
            backend=backend,
            model=model,
        )

    def started_event(
        self,
        identity: ModelCallIdentity,
        *,
        event_id: str | None = None,
        created_at: float | None = None,
    ) -> ModelCallStartedEvent:
        return ModelCallStartedEvent(
            **self._event_identity(identity),
            event_id=event_id,
            created_at=created_at,
        )

    def completed_terminal(
        self,
        identity: ModelCallIdentity,
        usage: TokenUsage,
        *,
        event_id: str | None = None,
        created_at: float | None = None,
    ) -> ModelCallTerminal:
        record = ModelCallRecord(
            **self._record_identity(identity),
            status=ModelCallStatus.COMPLETED,
            usage=deepcopy(usage),
        )
        event = ModelCallCompletedEvent(
            **self._event_identity(identity),
            usage=deepcopy(usage),
            event_id=event_id,
            created_at=created_at,
        )
        return ModelCallTerminal(record=record, event=event, budget=self.observe_budget(identity, usage))

    def failed_terminal(
        self,
        identity: ModelCallIdentity,
        *,
        error_code: str,
        ambiguous: bool,
        usage: TokenUsage | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
    ) -> ModelCallTerminal:
        normalized_usage = deepcopy(usage) if usage is not None else TokenUsage()
        status = ModelCallStatus.AMBIGUOUS if ambiguous else ModelCallStatus.FAILED
        record = ModelCallRecord(
            **self._record_identity(identity),
            status=status,
            usage=normalized_usage,
            error_code=error_code,
        )
        event = ModelCallFailedEvent(
            **self._event_identity(identity),
            outcome="ambiguous" if ambiguous else "definitive",
            usage=deepcopy(normalized_usage),
            error_code=error_code,
            event_id=event_id,
            created_at=created_at,
        )
        return ModelCallTerminal(
            record=record,
            event=event,
            budget=self.observe_budget(identity, normalized_usage),
        )

    def observe_budget(self, identity: ModelCallIdentity, usage: TokenUsage) -> ModelCallBudgetObservation:
        if self.budget_observer is None:
            return ModelCallBudgetObservation()
        return self.budget_observer(identity.cycle_index, usage)

    def commit_terminal(self, terminal: ModelCallTerminal) -> None:
        self.ledger.append(terminal.record)
        self.emit(terminal.event)
        if terminal.budget.event is not None:
            self.emit(terminal.budget.event)

    def emit(self, event: RunEvent) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    def _event_identity(self, identity: ModelCallIdentity) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            **self._record_identity(identity),
        }

    @staticmethod
    def _record_identity(identity: ModelCallIdentity) -> dict[str, Any]:
        return {
            "call_id": identity.call_id,
            "operation_id": identity.operation_id,
            "attempt": identity.attempt,
            "operation": identity.operation,
            "cycle_index": identity.cycle_index,
            "backend": identity.backend,
            "model": identity.model,
        }


def is_definitive_model_error(error: BaseException) -> bool:
    if getattr(error, "definitive_outcome", False) is True:
        return True
    text = str(error).lower()
    return any(marker in text for marker in _DEFINITIVE_ERROR_MARKERS)


def model_error_code(error: BaseException) -> str:
    raw = getattr(error, "code", None)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if _ERROR_CODE_RE.fullmatch(normalized):
            return normalized
    if is_definitive_model_error(error) and any(marker in str(error).lower() for marker in _DEFINITIVE_ERROR_MARKERS):
        return "prompt_too_long"
    return "model_request_failed"


def _normalize_operation_slot(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not normalized:
        raise ValueError("model operation slot must be non-empty")
    return normalized
