"""Receipts produced by released 0.34.1 retain their selected archive identity."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, Record, SessionSpec
from vv_agent.session.tool_results import prepare_tool_resources, project_tool_result
from vv_agent.types import ToolArtifactRef, ToolExecutionResult
from vv_agent.workspace import LocalWorkspaceBackend
from vv_agent.workspace.artifacts import verify_text_artifact

from .test_recovery_matrix import effect_tool, runtime

FIXTURE = Path(__file__).with_name("fixtures") / "tool_result_0341.json"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "body",
        "length",
        "metadata",
        "tool_call_id",
        "directive",
        "status_code",
        "error_code",
        "image_url",
        "image_path",
        "usage",
    ],
)
def test_0341_original_receipt_uses_saved_artifact_without_rewriting(store, database, tmp_path, mutation):
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["producer"] == "0.34.1"
    records = tuple(Record.parse(canonical_json_bytes(r)) for r in fixture["records"])
    created = records[0]
    with store.atomic() as tx:
        tx.create(SessionSpec("s", **created.payload), consumers=("events",))
        for record in records:
            if record.kind == "input_applied":
                tx.push("s", InboxItem.parse(canonical_json_bytes(record.payload["input"])))
    lease = store.acquire("s", owner="restore-0341-fixture", ttl_ms=60000)
    with store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=1,
            commit_id="restore-0341-fixture",
            records=records[1:],
            consume_input_ids=tuple(r.payload["input"]["input_id"] for r in records if r.kind == "input_applied"),
        )
    store.release(lease)
    state, retained, _ = read_state(store, "s")
    assert [r.record.encode() for r in retained] == [r.encode() for r in records]
    completion = next(
        r for r in retained if r.record.kind == "op_completed" and r.record.payload["result"].get("tool_call_id") == "a"
    )
    stored = completion.record.payload["result"]
    artifact = ToolArtifactRef.from_dict(stored["artifact"])
    callback = fixture["callback"]
    original_body = callback["payload"]["result"]["content"]
    backend = LocalWorkspaceBackend(tmp_path / "workspace")
    backend.write_text_exclusive(artifact.path, original_body)
    verify_text_artifact(backend, artifact)
    files = {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in backend._artifact_root.rglob("*.txt")}
    assert completion.record.operation_id is not None
    plan = state.operations[completion.record.operation_id].attempts[1].execution_plan
    assert prepare_tool_resources(plan, stored, backend)["artifact"] == stored["artifact"]
    assert project_tool_result(completion)["resources"][0]["status"] == "retained"
    payload = callback["payload"]
    if mutation == "body":
        payload["result"]["content"] = original_body[:15000] + "Q" + original_body[15001:]
    elif mutation == "length":
        payload["result"]["content"] += "P"
    elif mutation == "metadata":
        payload["result"][mutation] = {"different": True}
    elif mutation == "tool_call_id":
        payload["result"]["tool_call_id"] = "other"
    elif mutation == "usage":
        payload["usage"] = {"prompt_tokens": 1}
    elif mutation == "directive":
        payload["result"][mutation] = "finish"
    elif mutation == "status_code":
        payload["result"][mutation] = "ERROR"
    elif mutation == "error_code":
        payload["result"].update(status_code="ERROR", error_code="different")
    elif mutation in {"image_url", "image_path"}:
        payload["result"][mutation] = "different"
    ToolExecutionResult.from_dict(payload["result"])

    class ReceiptProvider:
        def authenticate(self, item, plan):
            return canonical_json_bytes(item.payload) == canonical_json_bytes(payload)

        def submit(self, *args, **kwargs):
            raise AssertionError("completed effect must not execute again")

        def query(self, *args, **kwargs):
            raise AssertionError("completed effect must not be queried")

    rt = runtime(database, [], [effect_tool()], providers={"effect": ReceiptProvider()})
    rt.config = replace(rt.config, workspace_backend=backend)
    rt.workspace_backend = backend
    with store.atomic() as tx:
        tx.push("s", InboxItem.parse(canonical_json_bytes(callback | {"input_id": "repeat-original"})))
    drive(store, "s", runtime=rt)
    state, after, _ = read_state(store, "s")
    applied = state.applied_inputs["repeat-original"].payload
    print(
        "LEGACY_DUPLICATE",
        {"producer": "0.34.1", "mutation": mutation, "disposition": applied["disposition"], "reason": applied["reason"]},
    )
    assert (applied["disposition"], applied["reason"]) == (
        ("noop", "retained result") if mutation is None else ("rejected", "result conflict")
    )
    assert [r.record.encode() for r in after[: len(records)]] == [r.encode() for r in records]
    assert len([r for r in after if r.record.kind == "op_completed"]) == len(
        [r for r in retained if r.record.kind == "op_completed"]
    )
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in backend._artifact_root.rglob("*.txt")} == files
