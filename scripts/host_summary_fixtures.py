"""Generate host-summary facts from real kernel producers and fake transport."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.session.test_compaction import SUMMARY
from tests.session.test_recovery_matrix import MODEL_USAGE, records_of, start
from tests.session.test_summary_requests import call, configured, summary_plans
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, Message


def produce() -> dict:
    scenarios = []
    for child in (False, True):
        with TemporaryDirectory(prefix="host-summary-") as tmp:
            db = Path(tmp) / "session.sqlite"
            with SQLiteStore.standalone(str(db)) as store:
                store.install_schema()
                start(store)
                rt = configured(
                    db,
                    [
                        call("child_task" if child else "compact"),
                        LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE}),
                        LLMResponse("done"),
                    ],
                    child=child,
                )
                rt.config = replace(
                    rt.config,
                    initial_messages=[Message("user", "original request"), Message("assistant", "old facts " * 200)],
                )
                drive(store, "s", runtime=rt)
                plans = summary_plans(store)
                assert len(plans) == 1
                receipt = next(r for r in records_of(store, "op_completed") if r.operation_id == plans[0].operation_id)
                compactions = records_of(store, "context_compacted")
                assert len(compactions) == (0 if child else 1)
                wire = {
                    "id": "child_summary" if child else "tool_compaction",
                    "summary_plan": plans[0].to_dict(),
                    "summary_result": receipt.payload["result"],
                    "summary_usage": {k: v for k, v in receipt.payload["usage"].items() if not k.startswith("session_")},
                    "context_replacements": [r.to_dict() for r in compactions],
                    "child_input": store.peek_inbox("child")[0].item.payload["content"] if child else None,
                    "tool_result": next(
                        r.payload["result"]
                        for r in records_of(store, "op_completed")
                        if r.payload["result"].get("tool_call_id") == "request"
                    ),
                }
                encoded = canonical_json_bytes(wire)
                scenarios.append(
                    {"wire": wire, "bytes_base64": base64.b64encode(encoded).decode(), "sha256": sha256(encoded).hexdigest()}
                )
    return {
        "schema_version": 1,
        "record_schema_version": 1,
        "inbox_schema_version": 3,
        "parent_replaced_for_child": False,
        "terminal_precedence": ["finish", "wait_user"],
        "batch_requests_coalesce": True,
        "vectors": scenarios,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(produce(), indent=2, ensure_ascii=False) + "\n")
