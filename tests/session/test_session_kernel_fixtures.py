"""Authoring fixtures stay outside the v23 snapshot and prove real producer bytes."""

import base64
import json
import re
import runpy
import subprocess
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vv_agent.events import event_from_dict
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem, Record, RecordError
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.store import StoredRecord
from vv_agent.session.tracing import project_spans

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "session_kernel_fixtures.py"
AUTHOR = runpy.run_path(str(SCRIPT))
FILES = {
    "session_record.schema.json",
    "session_inbox.schema.json",
    "session_records.jsonl",
    "session_inbox.jsonl",
    "session_codec_vectors.json",
    "session_invalid.json",
    "session_semantics.json",
    "session_recovery.json",
    "session_projection.json",
    "session_compaction.json",
    "app_server_protocol.json",
}


def checked_bytes(vector):
    raw = base64.b64decode(vector["bytes_base64"], validate=True)
    assert sha256(raw).hexdigest() == vector["sha256"]
    return raw


def test_session_kernel_fixtures_deterministic_and_revalidated(tmp_path):
    directories = [tmp_path / "first", tmp_path / "second"]
    for output in directories:
        subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)], check=True, capture_output=True, timeout=120)
        assert {p.name for p in output.iterdir()} == FILES
    first, second = directories
    for name in FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def read(name):
        return json.loads((first / name).read_text())

    corpus = read("session_codec_vectors.json")
    coverage = corpus["coverage"]
    assert len(coverage["record_kinds"]) == 14
    assert len(coverage["inbox_kinds"]) == 8
    assert len(coverage["boundary_stages"]) == 7
    assert len(coverage["handle_variants"]) == 4
    assert all(coverage["optional_fields"].values())
    assert coverage["optional_fields"]["boundary.output_checked.partial_output"] > 0
    validators = {kind: Draft202012Validator(read(f"session_{kind}.schema.json")) for kind in ("record", "inbox")}
    sessions, records, inputs = defaultdict(list), {}, {}
    independent = AUTHOR["independent_bytes"]([v["wire"] for v in corpus["vectors"]])
    for vector, expected in zip(corpus["vectors"], independent, strict=True):
        raw = checked_bytes(vector)
        assert raw == expected
        validators[vector["type"]].validate(vector["wire"])
        parsed = (Record if vector["type"] == "record" else InboxItem).parse(raw)
        assert parsed.encode() == raw
        if isinstance(parsed, Record):
            assert vector["record_id"] == parsed.record_id == AUTHOR["expected_id"](vector["wire"])
            records[(parsed.session_id, parsed.record_id)] = parsed
            sessions[parsed.session_id].append(
                StoredRecord(parsed, vector["seq"], vector["commit_id"], vector["writer_epoch"], vector["created_ms"])
            )
        else:
            inputs[(vector["session_id"], parsed.input_id)] = parsed
    for sid, rows in sessions.items():
        logical = [r.record for r in rows]
        consumed = [InboxItem(**r.payload["input"]) for r in logical if r.kind == "input_applied"]
        fold(logical, consumed_inputs=consumed)
        assert all(inputs[(sid, item.input_id)].encode() == item.encode() for item in consumed)
    for kind, name in (("record", "session_records.jsonl"), ("inbox", "session_inbox.jsonl")):
        assert (first / name).read_bytes() == b"".join(checked_bytes(v) + b"\n" for v in corpus["vectors"] if v["type"] == kind)

    invalid = read("session_invalid.json")["vectors"]
    classes = {v["rejection_class"] for v in invalid}
    assert {
        "missing_version",
        "stale_version",
        "unknown_version",
        "malformed_version",
        "unknown_kind",
        "unknown_field",
        "unknown_enum",
        "incorrect_nullability",
        "wrong_identity",
        "embedded_digest",
        "lone_surrogate",
        "non_string_key",
        "nonfinite_number",
        "unsafe_integer",
        "non_json_host_object",
        "duplicate_member",
        "nonobject_envelope",
        "cross_record_order",
        "dependencies",
        "request_drift",
        "authorization",
        "stale_generation",
        "terminal_revival",
        "provider_binding_drift",
        "provider_authentication",
        "child_authentication",
        "trailing_newline_hash",
    } <= classes
    for vector in invalid:
        raw = checked_bytes(vector)
        layer = vector["layer"]
        if layer in {"codec", "inbox_codec"}:
            with pytest.raises((RecordError, ValueError, TypeError)):
                (Record if layer == "codec" else InboxItem).parse(raw)
        elif layer == "constructor":
            wire = json.loads(raw)
            key, value = (1, "value") if vector["mutation"] == "non_string_key" else ("host", object())
            wire["payload"]["request"][key] = value
            with pytest.raises((RecordError, ValueError, TypeError)):
                Record(**wire).encode()
        elif layer == "fold":
            sid = vector["session_id"]
            log = [records[(sid, identity)] for identity in vector["prefix_ids"]] + [Record.parse(raw)]
            consumed = [InboxItem(**r.payload["input"]) for r in log if r.kind == "input_applied"]
            with pytest.raises(TransitionError, match=re.escape(vector["reason"])):
                fold(log, consumed_inputs=consumed)
        else:
            assert layer == "admission"
            item = InboxItem.parse(raw)
            audit = records[(vector["session_id"], vector["audit_record_id"])]
            assert audit.payload["disposition"] == "rejected" and audit.payload["reason"] == vector["reason"]
            assert InboxItem(**audit.payload["input"]).encode() == item.encode()

    for name, key in (
        ("session_semantics.json", "cases"),
        ("session_recovery.json", "cases"),
        ("session_projection.json", "sessions"),
        ("app_server_protocol.json", "transcripts"),
    ):
        for vector in read(name)[key]:
            value = {k: v for k, v in vector.items() if k not in {"bytes_base64", "sha256", "record_id"}}
            assert checked_bytes(vector) == AUTHOR["independent_bytes"]([value])[0]
    for projection in read("session_projection.json")["sessions"]:
        rows = tuple(sessions[projection["session_id"]])
        assert [e.to_dict() for e in project_records(rows)] == projection["events"]
        assert [
            {"seq": seq, "method": method, "span": span.to_dict()} for seq, method, span in project_spans(rows)
        ] == projection["spans"]
        for event in projection["events"]:
            assert event["version"] == "v6"
            assert event_from_dict(event, _kernel=True).to_dict() == event
        for prefix in projection["prefix_states"]:
            log = [r.record for r in rows[: prefix["seq"]]]
            consumed = [InboxItem(**r.payload["input"]) for r in log if r.kind == "input_applied"]
            state = fold(log, consumed_inputs=consumed)
            assert (state.phase, state.active_turn_id, state.closed, state.terminal_seq) == (
                prefix["phase"],
                prefix["active_turn_id"],
                prefix["closed"],
                prefix["terminal_seq"],
            )
    for recovery in read("session_recovery.json")["cases"]:
        if "receipt_id" in recovery:
            receipt = records[(recovery["session_id"], recovery["receipt_id"])]
            assert receipt.kind == "op_completed"
    for vector in read("session_compaction.json")["vectors"]:
        parsed = Record.parse(checked_bytes(vector))
        assert parsed.record_id == vector["record_id"] == AUTHOR["expected_id"](vector["wire"])
        assert parsed.to_dict() == vector["wire"]

    app = read("app_server_protocol.json")
    schemas = {name: json.loads(source) for name, source in app["schemas"]["jsonSchema"].items()}
    envelope = Draft202012Validator(schemas["JsonRpcMessage"])
    results = {
        "initialize": "InitializeResponse",
        "thread/start": "ThreadStartResponse",
        "thread/read": "ThreadReadResponse",
        "thread/resume": "ThreadResumeResponse",
        "turn/start": "TurnStartResponse",
        "turn/resume": "TurnResumeResponse",
    }
    for entry in app["transcripts"]:
        request = entry.get("request")
        if request is not None:
            errors = list(Draft202012Validator(schemas["ClientRequest"]).iter_errors(request))
            if errors:
                assert entry["responses"][0]["error"]["code"] == -32602
                assert "checkpointKey" in request["params"]
            for response in entry["responses"]:
                envelope.validate(response)
                if "result" in response and request["method"] in results:
                    Draft202012Validator(schemas[results[request["method"]]]).validate(response["result"])
        for notification in entry.get("notifications", []):
            envelope.validate(notification)
    assert app["facts"]["observer_cannot_approve"] and app["facts"]["timeout_at_absolute_deadline"]
