"""Child summary scheduling and acceptance over the parent's actual ledger."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from . import _driver_outcomes as outcomes
from . import _summary_compaction as summary
from ._boundary_hydration import _prepare_fold_records, active_task, hydrate_stored_records
from ._context import project_context
from ._driver import CoreDriver
from ._driver_planning import plan as plan_steps
from ._driver_requests import CommitRecords
from ._message_values import Message
from ._provider_outcomes import Definitive
from ._records import Record, digest
from ._reducer import CountMessages, logical_inputs
from ._summary_requests import summary_pending
from ._token_count import TokenCount, _StopIterationTransport
from ._tool_values import ToolExecutionResult, ToolResultStatus


class SummaryRead(Enum):
    PREPARE_MEMORY = auto()
    PLAN_SETTINGS = auto()
    ACCEPT_SETTINGS = auto()
    MODEL_CONTEXT_WINDOW = auto()
    MODEL_ROUTE = auto()


@dataclass(frozen=True, slots=True)
class RequiresSummary:
    plan: Record


@dataclass(frozen=True, slots=True)
class CountSummaryPrompt:
    messages: list[Message]


@dataclass(frozen=True, slots=True)
class CountPrefix:
    request: TokenCount
    count_messages: CountMessages


def source_at(driver: CoreDriver, version: str) -> Generator[CountPrefix, int, list[Message]]:
    prefix = tuple(row for row in driver.records if row.seq <= int(version))
    hydrated = hydrate_stored_records(driver, prefix)
    fold_records, logical_inputs_by_id, _ = _prepare_fold_records(prefix, hydrated)
    folded = driver._fold_factory()
    count = None
    with logical_inputs(logical_inputs_by_id):
        steps = folded.extend_steps(
            fold_records,
            consumed_inputs=logical_inputs_by_id.values(),
            bodies=(row.record.encode() for row in prefix),
        )
        try:
            while True:
                try:
                    request = next(steps) if count is None else steps.send(count)
                except StopIteration as done:
                    state = done.value
                    break
                count = yield CountPrefix(request, folded._count_messages)
        except StopIteration as error:
            raise _StopIterationTransport(error) from error
        finally:
            steps.close()
    return project_context(hydrated, state)


def child_summary(driver: CoreDriver, plan: Record) -> Generator[object, Any, tuple[bool, str | None]]:
    try:
        oid = f"{plan.operation_id}/summary"
        op = driver.state.operations.get(oid)
        yield SummaryRead.PREPARE_MEMORY
        if op is None and not (yield RequiresSummary(plan)):
            return True, None
        if op is None:
            parent = driver.state.operations[plan._payload["dependencies"][0]]
            receipt = parent.attempts[parent.selected_attempt or 1].result
            assert receipt is not None
            version = str(driver.stored_record(receipt).seq - 1)
            source = yield from source_at(driver, version)
            summary_plan = summary.plan_full_summary(source, **(yield SummaryRead.PLAN_SETTINGS))
            if summary_plan is not None and (yield CountSummaryPrompt([Message("user", summary_plan.prompt)])) < (
                yield SummaryRead.MODEL_CONTEXT_WINDOW
            ):
                model_id = yield SummaryRead.MODEL_ROUTE
                assert driver.state.active_turn_id is not None
                task = active_task(driver)
                parent_plan = driver.hydrate_plan(driver.state.operations[plan._payload["dependencies"][0]].attempts[1].plan)
                request = {
                    "model": model_id,
                    "messages": [Message("user", summary_plan.prompt).to_dict()],
                    "tools": [],
                    "metadata": {
                        "purpose": "compaction",
                        "mode": "child_summary",
                        "source_digest": digest([m.to_dict() for m in source]),
                        "source_context_version": version,
                        "child_operation_id": plan.operation_id,
                        "vv_session": {"cycle_index": parent_plan._payload["request"]["metadata"]["vv_session"]["cycle_index"]},
                    },
                    "prompt_bundle": None,
                    "model_settings": task.model_settings.to_dict() if task.model_settings else None,
                }
                yield CommitRecords([(yield from plan_steps(driver, oid, request, "model", purpose="compaction"))])
                return False, None
        elif op.selected_attempt is not None:
            attempt = op.attempts[op.selected_attempt]
            plan_record = driver.hydrate_plan(attempt.plan)
            meta = plan_record._payload["request"]["metadata"]
            source = yield from source_at(driver, meta["source_context_version"])
            if digest([m.to_dict() for m in source]) != meta["source_digest"]:
                raise ValueError("child summary source mismatch")
            summary_plan = summary.plan_full_summary(source, **(yield SummaryRead.PLAN_SETTINGS))
            result = (
                driver.hydrate_record_field(attempt.result, "result", attempt.result._payload["result"])
                if attempt.result is not None
                else None
            )
            if summary_plan is not None and result is not None and not result.get("error_code") and attempt.context == "normal":
                accepted = yield from summary.accept_summary(
                    summary_plan,
                    result["content"],
                    **(yield SummaryRead.ACCEPT_SETTINGS),
                    require_reduction=False,
                )
                if accepted.accepted:
                    return True, next(m.content for m in accepted.messages if m.name == "memory_summary")
        elif summary_pending(driver, plan.operation_id or ""):
            return False, None
        plan_record = driver.hydrate_plan(plan)
        result = ToolExecutionResult(
            plan_record._payload["request"]["id"],
            "A fresh parent summary is unavailable; no child was admitted.",
            status_code=ToolResultStatus.ERROR,
            error_code="child_summary_unavailable",
        )
        yield CommitRecords((yield from outcomes.completed_steps(driver, plan, Definitive(result.to_dict()))))
        return False, None
    except StopIteration as error:
        raise _StopIterationTransport(error) from error
