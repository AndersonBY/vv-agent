"""Produce contract facts from real kernel scenarios; no synthetic kernel receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.session import test_unseen_inputs as cases
from vv_agent.session.context import project_context, protected_start
from vv_agent.session.sqlite import SQLiteStore


def generate():
    with (
        patch("vv_agent.session.kernel.time.monotonic_ns", return_value=100_000_000),
        patch.object(cases, "HTTP_PORT", 18737),
    ):
        return _generate()


def _generate():
    result = []
    scenarios = [
        ("steer", cases.test_steer_survives_ptl_in_real_primary_request, (1,)),
        ("twelve_steers", cases.test_steer_survives_ptl_in_real_primary_request, (12,)),
        ("user", cases.test_user_reply_survives_ptl, ()),
        ("tool_user_reply", cases.test_tool_user_reply_keeps_whole_batch, ()),
        ("child", cases.test_native_children_survive_ptl, (1,)),
        ("twelve_children", cases.test_native_children_survive_ptl, (12,)),
        ("capacity", cases.test_unseen_capacity_ends_without_summary_or_retry, ()),
        *[
            ("restart_" + cut, cases.test_restart_preserves_unseen_input_once, (cut,))
            for cut in ("planned", "receipt", "accepted")
        ],
    ]
    scenarios += [("provider_callback", cases.test_provider_callback_and_steer_keep_whole_group_after_ptl, (True,))]
    scenarios += [
        ("provider_" + delivery, cases.test_provider_callback_only_protects_projected_content, (delivery,))
        for delivery in ("correction", "audit")
    ]
    scenarios += [
        ("cancelled_" + reply, cases.test_cancelled_interaction_does_not_protect_audit_content, (reply,))
        for reply in ("user", "deny", "timeout")
    ]
    scenarios += [
        ("http_" + mode, cases.http_adapter_primary_coverage, (mode,)) for mode in ("normal", "ptl", "retry", "unknown")
    ]
    scenarios += [
        ("approval_" + decision, cases.test_approval_reason_keeps_original_group_after_ptl, (decision,))
        for decision in ("deny", "timeout")
    ]
    scenarios += [
        (f"snapshot_{reply}_{prepend}", cases.test_normal_callback_after_snapshot_reaches_next_primary, (reply, prepend))
        for reply in ("provider", "user", "deny", "timeout")
        for prepend in (False, True)
    ]
    for name, run, args in scenarios:
        with SQLiteStore.standalone(":memory:") as store:
            store.install_schema()
            store.connection.create_function("session_now_ms", 0, lambda: 1_000_000)
            extra = run(store, store, *args)
            state, records, _ = store.read_state("s")
            plans = [r.record for r in records if r.record.kind == "op_planned" and r.record.payload["purpose"] == "primary"]
            result.append(
                {
                    **(extra or {}),
                    "case": name,
                    "creation": records[0].record.to_dict(),
                    "record_digests": [r.record.digest for r in records],
                    "last_primary_request": plans[-1].payload["request"] if plans else None,
                    "compactions": [r.payload for r in state.compactions],
                    "input_ids": list(state.applied_inputs),
                    "terminal_reasons": [r.record.payload["reason"] for r in records if r.record.kind == "turn_ended"],
                    "protected_start": protected_start(records, state),
                    "final_context": [m.to_dict() for m in project_context(records, state)],
                }
            )
    return {"producer": "scripts/unseen_input_fixtures.py", "cases": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + "\n")
