"""Capture host-summary facts from real S1 kernel producers and fake transport."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import json
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from scripts.s1_fixture_values import deterministic_content_producer, fixture_key_provider
from tests.session.helpers import create_fixture_session, push_fixture_input, wrap_fixture_store
from tests.session.test_compaction import SUMMARY
from tests.session.test_recovery_matrix import MODEL_USAGE
from tests.session.test_summary_requests import (
    call,
    configured,
    drive_session,
    hydrated_input_payload,
    records_of,
    summary_plans,
)
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._hydration import reference_digest
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, Message


def _vector(store, database, *, child):
    create_fixture_session(
        store,
        "s",
        attributes={
            "seed": {
                "messages": [
                    Message("user", "original request").to_dict(),
                    Message("assistant", "old facts " * 200).to_dict(),
                ],
                "shared_state": {},
                "unseen_input_protection": False,
                "response_rejection": True,
            }
        },
    )
    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
    rt = configured(
        store,
        database,
        [
            call("child_task" if child else "compact"),
            LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE}),
            LLMResponse("done"),
        ],
        child=child,
        initial=False,
    )
    drive_session(store, rt)
    (plan,) = summary_plans(store)
    retained_plan = next(row.record for row in store.read("s").records if row.record.record_id == plan.record_id)
    request_ref = ContentRef.from_dict(retained_plan.payload["request"])
    assert plan.payload["request_digest"] == retained_plan.payload["request_digest"] == reference_digest(request_ref)
    receipts = records_of(store, "op_completed")
    receipt = next(r for r in receipts if r.operation_id == plan.operation_id)
    assert receipt.payload["request_digest"] == retained_plan.payload["request_digest"]
    compactions = records_of(store, "context_compacted")
    assert len(compactions) == (0 if child else 1)
    if child:
        child_item = store.peek_inbox("child")[0].item
        assert child_item.schema_version == "S1" and child_item.feature_level == 1
        ContentRef.from_dict(child_item.payload["content"])

    wire = {
        "id": "child_summary" if child else "tool_compaction",
        "summary_plan": plan.to_dict(),
        "summary_result": receipt.payload["result"],
        "summary_usage": {k: v for k, v in receipt.payload["usage"].items() if not k.startswith("session_")},
        "context_replacements": [r.to_dict() for r in compactions],
        "child_input": hydrated_input_payload(store, "child")["content"] if child else None,
        "tool_result": next(r.payload["result"] for r in receipts if r.payload["result"].get("tool_call_id") == "request"),
    }
    encoded = canonical_json_bytes(wire)
    return {"wire": wire, "bytes_base64": base64.b64encode(encoded).decode(), "sha256": sha256(encoded).hexdigest()}


@deterministic_content_producer
def produce() -> dict:
    """Produce a current S1 projection alongside the immutable Contract 39 capture."""

    vectors = []
    for child in (False, True):
        with TemporaryDirectory(prefix="host-summary-") as tmp:
            database = Path(tmp) / "session.sqlite"
            with (
                SQLiteStore.standalone(str(database)) as raw_store,
                closing(fixture_key_provider(Path(tmp) / "content-keys.sqlite")) as keys,
            ):
                raw_store.install_schema()
                vectors.append(_vector(wrap_fixture_store(raw_store, keys), database, child=child))
    return {
        "schema_version": 1,
        "record_schema_version": "S1",
        "inbox_schema_version": "S1",
        "parent_replaced_for_child": False,
        "terminal_precedence": ["finish", "wait_user"],
        "batch_requests_coalesce": True,
        "vectors": vectors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(produce(), indent=2, ensure_ascii=False) + "\n")
