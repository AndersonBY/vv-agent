"""Coverage reads task metadata from hydrated records without changing raw vectors."""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from vv_agent.kernel._content import ContentRef

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = runpy.run_path(str(ROOT / "scripts" / "session_kernel_fixtures.py"))
PARITY = ROOT / "tests" / "fixtures" / "parity"
TASK_METADATA = {
    "host_binding_names": ["host"],
    "max_handoffs": 0,
    "handoff_targets": {},
    "input_messages": [],
    "memory_initial_state": {},
    "input_blocked": "blocked",
}
REQUEST_METADATA = {
    "endpoint_order": ["endpoint"],
    "endpoint_id": "endpoint",
    "shared_state": {},
    "cycle_index": 1,
    "model_binding": {
        "backend": "scripted",
        "context_length": None,
        "endpoints": ["endpoint"],
        "max_output_tokens": None,
        "model": "m",
    },
}


def _logical_records(vectors, task_metadata, request_metadata):
    logical = []
    for vector in vectors:
        if vector["type"] != "record":
            continue
        wire = deepcopy(vector["wire"])
        if wire["kind"] == "turn_started":
            wire["payload"]["definition"] = {"task": {"metadata": {"vv_session": task_metadata}}}
        elif wire["kind"] == "op_planned":
            wire["payload"]["request"] = {"metadata": {"vv_session": request_metadata}}
        elif wire["kind"] == "op_completed":
            wire["payload"]["usage"] = {"tokens": {}, "budget": {}}
        elif wire["kind"] == "boundary_recorded" and wire["payload"]["stage"] == "output_checked":
            wire["payload"]["data"]["partial_output"] = "partial"
        logical.append(wire)
    return logical


def test_coverage_reads_hydrated_task_metadata_and_preserves_raw_content_ref():
    vectors = json.loads((PARITY / "session_codec_vectors.json").read_text())["vectors"]
    semantics = json.loads((PARITY / "session_semantics.json").read_text())["cases"]
    raw_starts = [v["wire"] for v in vectors if v["type"] == "record" and v["wire"]["kind"] == "turn_started"]
    assert raw_starts
    assert all(ContentRef.from_dict(r["payload"]["definition"]) is not None for r in raw_starts)

    logical = _logical_records(vectors, TASK_METADATA, REQUEST_METADATA)
    result = AUTHOR["coverage"](vectors, semantics, logical)
    for key in TASK_METADATA:
        assert result["optional_fields"][f"task.metadata.vv_session.{key}"] == len(raw_starts)
    assert all(ContentRef.from_dict(r["payload"]["definition"]) is not None for r in raw_starts)

    missing_metadata = dict(TASK_METADATA)
    missing_metadata.pop("input_blocked")
    with pytest.raises(AssertionError, match=r"task\.metadata\.vv_session\.input_blocked"):
        AUTHOR["coverage"](vectors, semantics, _logical_records(vectors, missing_metadata, REQUEST_METADATA))


def test_coverage_reads_content_backed_optionals_from_logical_records():
    vectors = json.loads((PARITY / "session_codec_vectors.json").read_text())["vectors"]
    semantics = json.loads((PARITY / "session_semantics.json").read_text())["cases"]
    raw_plans = [v["wire"] for v in vectors if v["type"] == "record" and v["wire"]["kind"] == "op_planned"]
    raw_completions = [v["wire"] for v in vectors if v["type"] == "record" and v["wire"]["kind"] == "op_completed"]
    output_checks = [
        v["wire"]
        for v in vectors
        if v["type"] == "record"
        and v["wire"]["kind"] == "boundary_recorded"
        and v["wire"]["payload"]["stage"] == "output_checked"
    ]
    assert raw_plans and raw_completions
    assert all(ContentRef.from_dict(r["payload"]["request"]) is not None for r in raw_plans)
    assert all(ContentRef.from_dict(r["payload"]["usage"]) is not None for r in raw_completions)
    assert all("usage_counts" not in r["payload"] for r in raw_completions)

    result = AUTHOR["coverage"](vectors, semantics, _logical_records(vectors, TASK_METADATA, REQUEST_METADATA))
    assert result["optional_fields"]["op_completed.usage_counts"] == len(raw_completions)
    assert result["optional_fields"]["boundary.output_checked.partial_output"] == len(output_checks)
    for key in REQUEST_METADATA:
        assert result["optional_fields"][f"request.metadata.vv_session.{key}"] > 0
    assert all(ContentRef.from_dict(r["payload"]["request"]) is not None for r in raw_plans)
    assert all(ContentRef.from_dict(r["payload"]["usage"]) is not None for r in raw_completions)


def test_coverage_aligns_records_by_session_and_record_id():
    vectors = json.loads((PARITY / "session_codec_vectors.json").read_text())["vectors"]
    semantics = json.loads((PARITY / "session_semantics.json").read_text())["cases"]
    records = [v["wire"] for v in vectors if v["type"] == "record"]
    first_session = records[0]["session_id"]
    first = records[0]
    second = next(r for r in records if r["session_id"] != first_session)
    shared_id = "coverage/shared-record-id"
    assert all(r["record_id"] != shared_id for r in records)
    first["record_id"] = shared_id
    second["record_id"] = shared_id

    result = AUTHOR["coverage"](vectors, semantics, _logical_records(vectors, TASK_METADATA, REQUEST_METADATA))
    assert result["record_kinds"]


def test_fixture_publishes_approval_answer_policy_optionals():
    fixtures = AUTHOR["Fixtures"]()
    try:
        fixtures.approval_answer_optionals()
        item = fixtures.inbox[("approval_optional", "answer")]
        assert {"policy_digest", "rule_id", "expires_ms"} <= set(item.payload)
        assert {key: item.payload[key] for key in ("policy_digest", "rule_id", "expires_ms")} == {
            "policy_digest": None,
            "rule_id": None,
            "expires_ms": None,
        }
    finally:
        fixtures.kernel.close()
