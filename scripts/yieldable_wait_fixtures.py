"""Produce canonical yield evidence from real kernel operations and SQLite commits."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from scripts.s1_fixture_values import (
    create_fixture_session,
    fixture_child_spec,
    fixture_content_reader,
    push_fixture_input,
    wrap_fixture_store,
)
from tests.session.test_batch_children import complete, deliver, rows
from tests.session.test_recovery_matrix import runtime
from tests.session.test_yieldable_waits import YieldProvider, provider_runtime, waiting_runtime
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.session import children
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store, project_records
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, ToolExecutionResult


def _digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _historical_input(value):
    return {
        "input_id": value["input_id"],
        "kind": value["kind"],
        "payload": value["payload"],
        "target_turn_id": value["target_turn_id"],
        "generation": value["generation"],
        "available_ms": value["available_ms"],
        "schema_version": 3,
    }


def _historical_evidence(value, terminals):
    if not isinstance(value, str):
        return value
    parts = value.split("/", 2)
    if len(parts) != 3 or parts[0] != "child" or not parts[1].isdigit():
        return value
    terminal_seq = int(parts[1])
    for digest, seq in terminals.values():
        if seq == terminal_seq:
            return f"child/{terminal_seq}/{digest}"
    return value


def _historical_nested(value, terminals):
    if isinstance(value, dict):
        result = {key: _historical_nested(item, terminals) for key, item in value.items()}
        session_id = result.get("session_id")
        if session_id in terminals and "terminal_digest" in result:
            result["terminal_digest"], result["terminal_seq"] = terminals[session_id]
        return result
    if isinstance(value, list):
        return [_historical_nested(item, terminals) for item in value]
    if isinstance(value, tuple):
        return tuple(_historical_nested(item, terminals) for item in value)
    return _historical_evidence(value, terminals)


def _historical_recovery_input(value, plans):
    item = _historical_input(value.to_dict())
    payload = item["payload"]
    key = (payload.get("operation_id"), payload.get("attempt"))
    if key in plans:
        request_digest = _digest(plans[key])
        payload["request_digest"] = request_digest
        handle = payload.get("handle")
        if isinstance(handle, dict) and "request_digest" in handle:
            handle["request_digest"] = request_digest
    return item


def _historical_record(row, store, plans, terminals):
    payload = hydrate_record_payload(row, hydration_for_session_store(store, row.record.session_id))
    if row.record.kind == "input_applied":
        payload.pop("input_proof", None)
        item = _historical_input(payload["input"])
        child_payload = item["payload"]
        child_session_id = child_payload.get("session_id")
        if child_session_id in terminals and "terminal_digest" in child_payload:
            child_payload["terminal_digest"], child_payload["terminal_seq"] = terminals[child_session_id]
        payload["input"] = item
        payload["input_digest"] = _digest(item)
    elif row.record.kind == "op_completed":
        payload.pop("policy_decision", None)
        key = (row.record.operation_id, row.record.attempt)
        if key in plans:
            payload["request_digest"] = _digest(plans[key])
        payload["result_digest"] = _digest(payload["result"])
        payload["evidence"] = [_historical_evidence(value, terminals) for value in payload["evidence"]]
    elif row.record.kind == "op_parked":
        key = (row.record.operation_id, row.record.attempt)
        handle = payload.get("handle")
        if key in plans and isinstance(handle, dict) and "request_digest" in handle:
            handle["request_digest"] = _digest(plans[key])
    return {
        key: value
        for key, value in (row.record.to_dict() | {"schema_version": 1, "payload": payload}).items()
        if key != "feature_level"
    }


def produce():
    cases = []
    with (
        patch("vv_agent.session.timing.time.monotonic_ns", return_value=100_000_000),
        patch("vv_agent.session.timing.time.time_ns", return_value=1_800_000_000_000_000_000),
    ):
        for name, count in (
            ("single_child", 1),
            ("child_batch", 3),
            ("provider", 0),
            ("terminal_before_steer", 1),
            ("provider_recovered", 0),
            ("completed_and_waiting", 2),
        ):
            with (
                TemporaryDirectory(prefix="yieldable-wait-") as directory,
                SQLiteStore.standalone(f"{directory}/session.sqlite") as store,
            ):
                store.connection.create_function("session_now_ms", 0, lambda: 1_800_000_000_000)
                store.install_schema()
                database = Path(store.path)
                store = wrap_fixture_store(store)
                create_fixture_session(
                    store,
                    "s",
                    attributes={
                        "seed": {
                            "messages": [],
                            "shared_state": {},
                            "unseen_input_protection": True,
                            "response_rejection": True,
                        }
                    },
                )
                push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
                provider = YieldProvider(callback=True) if count == 0 else None
                captured = []

                def observer(plan, handles, reason, captured=captured):
                    captured.append(handles)
                    return ToolExecutionResult(plan.payload["request"]["id"], "Interrupted with completed child result")

                rt, observed = (
                    provider_runtime(store, database, provider)
                    if provider
                    else waiting_runtime(
                        store,
                        database,
                        count=count,
                        callback=observer if name == "completed_and_waiting" else None,
                    )
                )
                if provider is None:

                    def admit_children(_plan, store=store, count=count):
                        return [
                            children.ChildSession(
                                fixture_child_spec(store, "s", f"yield-child-{index}"),
                                "work",
                                yield_on_steer=True,
                            )
                            for index in range(count)
                        ]

                    rt.children["child_task"] = (
                        children.ChildTool(admit_children, on_yield=observer)
                        if name == "completed_and_waiting"
                        else admit_children
                    )
                rt.content_reader = fixture_content_reader(store, "s")
                recovery_input = None
                if name == "provider_recovered":

                    class Cut(BaseException):
                        pass

                    def cut(point, record):
                        if point == "before_commit" and record and record.kind == "op_parked":
                            raise Cut()

                    rt.hook = cut
                    try:
                        drive(store, "s", runtime=rt)
                    except Cut:
                        pass
                    else:
                        raise AssertionError("acceptance cut was not reached")
                    rt.hook = lambda *_: None
                    assert provider is not None
                    h = provider.handle
                    payload = {k: h[k] for k in ("operation_id", "attempt", "request_digest")}
                    payload.update(handle=h, yield_on_steer=True)
                    recovery_input = InboxItem("recovered-acceptance", "provider_evidence", payload, "s/turn/initial", 0)
                    push_fixture_input(store, "s", recovery_input)
                drive(store, "s", runtime=rt)
                park = next(r for r in rows(store, kind="op_parked"))
                if name == "completed_and_waiting":
                    complete(store, Path(store.path), "yield-child-0")
                    deliver(store, "yield-child-0")
                    child_rt = runtime(database, [LLMResponse("Need a human answer")])
                    child_rt.content_reader = fixture_content_reader(store, "yield-child-1")
                    child_rt.config.no_tool_policy = "wait_user"
                    drive(store, "yield-child-1", runtime=child_rt)
                turn_id = read_state(store, "s")[0].active_turn_id
                push_fixture_input(
                    store,
                    "s",
                    InboxItem("steer", "steer", {"content": "Change direction now"}, turn_id, 0),
                )
                if name == "terminal_before_steer":
                    complete(store, Path(store.path), "yield-child-0")
                    deliver(store, "yield-child-0")
                drive(store, "s", runtime=rt)
                assert len(observed) == 1
                retained = rows(store)
                receipt = next(
                    r for r in retained if r.record.kind == "op_completed" and r.record.operation_id == park.record.operation_id
                )
                applied = next(
                    r for r in retained if r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == "steer"
                )
                yielded = name != "terminal_before_steer"
                assert ("yielded_by" in receipt.record.payload) == yielded
                if yielded:
                    assert receipt.commit_id == applied.commit_id
                living = [f"yield-child-{i}" for i in range(count) if not rows(store, f"yield-child-{i}", "turn_ended")]
                assert len(living) == (1 if name == "completed_and_waiting" else count if yielded else 0)
                before = receipt.record.encode()
                if count and yielded and name != "completed_and_waiting":
                    for sid in living:
                        complete(store, Path(store.path), sid)
                        deliver(store, sid)
                    drive(store, "s", runtime=rt)
                    retained = rows(store)
                    assert sum(
                        hydrate_record_payload(r, hydration_for_session_store(store, "s")).get("reason")
                        == "background child notification"
                        for r in retained
                        if r.record.kind == "input_applied"
                    ) == len(living)
                    assert next(r.record.encode() for r in retained if r.seq == receipt.seq) == before
                assert "Change direction now" in str(observed[0].messages)
                assert read_state(store, "s")[0].active_turn_id is not None
                if recovery_input is not None:
                    assert park.record.payload["yield_on_steer"] is True
                if captured:
                    child_terminal = rows(store, "yield-child-0", "turn_ended")[0]
                    assert (
                        captured[0][0]["result"]
                        == hydrate_record_payload(
                            child_terminal,
                            hydration_for_session_store(store, "yield-child-0"),
                        )["result"]
                    )
                hydrated_plans = {
                    (row.record.operation_id, row.record.attempt): hydrate_record_payload(
                        row,
                        hydration_for_session_store(store, "s"),
                    )["request"]
                    for row in retained
                    if row.record.kind == "op_planned"
                }
                terminals = {}
                for session_id in store.list_sessions():
                    if session_id == "s":
                        continue
                    child_terminal_rows = rows(store, session_id, "turn_ended")
                    if child_terminal_rows:
                        terminal = child_terminal_rows[0]
                        terminals[session_id] = (
                            _digest(_historical_record(terminal, store, {}, {})),
                            terminal.seq,
                        )
                cases.append(
                    {
                        "name": name,
                        **(
                            {"recovery_input": _historical_recovery_input(recovery_input, hydrated_plans)}
                            if recovery_input
                            else {}
                        ),
                        **({"callback_handles": _historical_nested(captured[0], terminals)} if captured else {}),
                        "records": [
                            _historical_record(r, store, hydrated_plans, terminals)
                            for r in retained
                            if r.record.kind == "op_parked"
                            or (r.record.kind == "op_completed" and r.record.operation_id == park.record.operation_id)
                            or (
                                r.record.kind == "input_applied"
                                and r.record.payload["input"]["kind"] in {"steer", "child_result"}
                            )
                        ],
                        "next_messages": [m.to_dict() for m in observed[0].messages if m.role != "system"],
                        "child_running_at_yield": living,
                        "atomic_yield": yielded and receipt.commit_id == applied.commit_id,
                        "completed_events": [
                            e.to_dict()["type"]
                            for e in project_records(retained, hydration=hydration_for_session_store(store, "s"))
                            if type(e).__name__ == "ToolCallCompletedEvent"
                        ],
                    }
                )
    return {
        "historical_contract_version": "39.0.0",
        "historical_record_schema_version": 1,
        "historical_inbox_schema_version": 3,
        "schema_version": 1,
        "cases": cases,
    }


def generate(path):
    path.write_text(json.dumps(produce(), ensure_ascii=True, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
