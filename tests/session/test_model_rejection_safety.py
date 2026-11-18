"""Kernel persistence and rollback boundaries for mixed provider evidence."""

import json
from contextlib import suppress
from dataclasses import replace

import pytest
from test_model_rejection_safety import LEAKS, ambiguous

from vv_agent.llm.errors import ModelRequestRejected
from vv_agent.output_validation import OutputValidationResult
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.types import LLMResponse

from .test_model_rejection import fail, http_error
from .test_recovery_matrix import records_of, runtime, start


@pytest.mark.parametrize("kind", ["timeout", "transport", "408", "500", "stream"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_exception_kernel_keeps_unknown(store, database, kind, explicit, reverse):
    rejected = ModelRequestRejected(400, "Rejected") if explicit else http_error(400)
    uncertain = ambiguous(kind)
    outer, inner = (uncertain, rejected) if reverse else (rejected, uncertain)
    outer.__cause__ = inner
    calls = []
    step = fail(outer, calls)
    start(store)
    drive(store, "s", runtime=runtime(database, [step, step]))
    assert len(calls) == 2
    unknowns = records_of(store, "op_unknown")
    assert len(unknowns) == 2 and all(r.payload["duplicate_cost_risk"] for r in unknowns)
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["reason"] == "model_outcome_unknown"
    assert terminal["unconfirmed_operations"]


@pytest.mark.parametrize("message", LEAKS)
@pytest.mark.parametrize("repair", [False, True])
def test_diagnostics_are_safe_in_receipts_terminal_and_events(store, database, message, repair):
    if repair:
        rt = runtime(database, [LLMResponse("invalid")])
        rt.agent.output_validation_enabled = True
        rt.agent.output_validator = lambda value, context: OutputValidationResult.reject("invalid")
        # Use a raw exception to prove the repair callback does not copy its text.
        rt.agent.output_repair = fail(http_error(400, message), [])
    else:
        raw_error = http_error(400, message)
        raw_error.body["error"].update(code="api_key:SYNTHETIC_IDENTIFIER_SECRET", type="secret:private")
        rt = runtime(database, [fail(raw_error, [])])
    start(store)
    drive(store, "s", runtime=rt)
    terminal = records_of(store, "turn_ended")[-1].payload
    rows = read_state(store, "s")[1]
    for value in ([r.record.payload for r in rows], terminal, [e.to_dict() for e in project_records(rows)]):
        saved = json.dumps(value)
        assert "SYNTHETIC" not in saved and "provided: SY" not in saved
    assert terminal["reason"] == ("output_validation_failed" if repair else "model_request_rejected")


@pytest.mark.parametrize(
    "case", ["clean", "ordinary_result", "unknown", "permanent_window", "retry_wait", "retry_finished", "permanent_finished"]
)
def test_rollback_precheck_reads_all_history_without_writes(store, database, monkeypatch, case):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "rollback_precheck", Path(__file__).resolve().parents[2] / "scripts/check_model_rejection_rollback.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_store = module.check_store

    class WorkerDeath(BaseException):
        pass

    def cut(point, record):
        if point == "after_commit" and (
            (case == "permanent_window" and record.kind == "op_completed")
            or (case == "retry_wait" and record.kind == "op_planned" and record.attempt == 2)
        ):
            raise WorkerDeath

    start(store)
    if case == "ordinary_result":
        drive(store, "s", runtime=runtime(database, [LLMResponse("done")]))
    elif case != "clean":
        error = (
            TimeoutError("lost")
            if case == "unknown"
            else ModelRequestRejected(429 if case.startswith("retry") else 400, "Rejected")
        )
        step = fail(error, [])
        monkeypatch.setattr("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0)
        with suppress(WorkerDeath):
            drive(store, "s", runtime=runtime(database, [step, step], hook=cut))

    class ReadOnly:
        def list_sessions(self, **kwargs):
            return store.list_sessions(**kwargs)

        def read(self, session_id, **kwargs):
            page = store.read(session_id, **kwargs)
            if case == "ordinary_result":
                page = replace(
                    page,
                    records=tuple(
                        replace(row, record=replace(row.record, payload=row.record.payload | {"result": "ordinary JSON result"}))
                        if row.record.kind == "op_completed"
                        else row
                        for row in page.records
                    ),
                )
            return page

    before = store.read("s").head_seq
    findings = check_store(ReadOnly(), page_size=1)
    assert bool(findings) is (case not in {"clean", "ordinary_result", "unknown"})
    if findings:
        assert findings[0]["session_id"] == "s"
        reason = {
            "permanent_window": "rejection_without_turn_ended",
            "retry_wait": "rejection_retry_history",
            "retry_finished": "rejection_retry_history",
            "permanent_finished": "model_rejection_history",
        }[case]
        assert reason in findings[0]["reasons"]
    assert store.read("s").head_seq == before


def test_unobservable_http_context_stays_unknown_in_kernel(store, database):
    error = http_error(400, "maximum context length exceeded", observed=False)
    step = fail(error, [])
    start(store)
    drive(store, "s", runtime=runtime(database, [step, step]))
    assert len(records_of(store, "op_unknown")) == 2
    assert not records_of(store, "context_compacted")
    assert not records_of(store, "op_completed")
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["reason"] == "model_outcome_unknown" and terminal["unconfirmed_operations"]
