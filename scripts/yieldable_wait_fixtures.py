"""Produce canonical yield evidence from real kernel operations and SQLite commits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from tests.session.test_batch_children import complete, deliver, rows
from tests.session.test_recovery_matrix import runtime, start
from tests.session.test_yieldable_waits import YieldProvider, provider_runtime, steer, waiting_runtime
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, ToolExecutionResult


def produce():
    cases = []
    with patch("vv_agent.session.kernel.time.monotonic_ns", return_value=100_000_000):
        for name, count in (
            ("single_child", 1),
            ("child_batch", 3),
            ("provider", 0),
            ("terminal_before_steer", 1),
            ("provider_recovered", 0),
            ("completed_and_waiting", 2),
        ):
            with SQLiteStore.standalone(":memory:") as store:
                store.connection.create_function("session_now_ms", 0, lambda: 1_800_000_000_000)
                store.install_schema()
                start(store)
                provider = YieldProvider(callback=True) if count == 0 else None
                captured = []

                def observer(plan, handles, reason, captured=captured):
                    captured.append(handles)
                    return ToolExecutionResult(plan.payload["request"]["id"], "Interrupted with completed child result")

                rt, observed = (
                    provider_runtime(store, provider)
                    if provider
                    else waiting_runtime(store, count=count, callback=observer if name == "completed_and_waiting" else None)
                )
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
                    h = provider.handle
                    payload = {k: h[k] for k in ("operation_id", "attempt", "request_digest")}
                    payload.update(handle=h, yield_on_steer=True)
                    recovery_input = InboxItem("recovered-acceptance", "provider_evidence", payload, "s/turn/initial", 0)
                    with store.atomic() as tx:
                        tx.push("s", recovery_input)
                drive(store, "s", runtime=rt)
                park = next(r for r in rows(store, kind="op_parked"))
                if name == "completed_and_waiting":
                    complete(store, store, "yield-child-0")
                    deliver(store, "yield-child-0")
                    child_rt = runtime(store, [LLMResponse("Need a human answer")])
                    child_rt.config.no_tool_policy = "wait_user"
                    drive(store, "yield-child-1", runtime=child_rt)
                steer(store)
                if name == "terminal_before_steer":
                    complete(store, store, "yield-child-0")
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
                        complete(store, store, sid)
                        deliver(store, sid)
                    drive(store, "s", runtime=rt)
                    retained = rows(store)
                    assert sum(r.record.payload.get("reason") == "background child notification" for r in retained) == len(living)
                    assert next(r.record.encode() for r in retained if r.seq == receipt.seq) == before
                assert "Change direction now" in str(observed[0].messages)
                assert read_state(store, "s")[0].active_turn_id is not None
                if recovery_input is not None:
                    assert park.record.payload["yield_on_steer"] is True
                if captured:
                    assert captured[0][0]["result"] == rows(store, "yield-child-0", "turn_ended")[0].record.payload["result"]
                cases.append(
                    {
                        "name": name,
                        **({"recovery_input": recovery_input.to_dict()} if recovery_input else {}),
                        **({"callback_handles": captured[0]} if captured else {}),
                        "records": [
                            r.record.to_dict()
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
                            e.to_dict()["type"] for e in project_records(retained) if type(e).__name__ == "ToolCallCompletedEvent"
                        ],
                    }
                )
    return {"schema_version": 1, "cases": cases}


def generate(path):
    path.write_text(json.dumps(produce(), ensure_ascii=True, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
