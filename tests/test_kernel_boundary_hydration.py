from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent.kernel._boundary_hydration import (
    hydrate_boundary_data,
    hydrate_context_compacted,
    hydrate_turn_ended,
)
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, SourceKind
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._hydration import MissingContentLink, ReferenceDigestMismatch, reference_digest
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import Record, record_identity
from vv_agent.kernel._store import StoredRecord


class _Reader:
    binding = ("tenant", "session")

    def __init__(self) -> None:
        self.links: dict[tuple[str, str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []

    def add(self, owner: str, slot: str, ref: ContentRef, value: Any) -> None:
        self.links[(owner, slot, ref.content_id)] = canonical_json_bytes(value)

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        assert (tenant_id, session_id) == self.binding
        self.calls.append((owner, slot, ref.content_id))
        try:
            return self.links[(owner, slot, ref.content_id)]
        except KeyError as exc:
            raise MissingContentLink("owner/slot/ref link is absent") from exc


def _ref(label: str, value: Any) -> ContentRef:
    body = canonical_json_bytes(value)
    return ContentRef(
        content_id=f"content/{label}",
        mac=sha256(body).hexdigest(),
        size=len(body),
        key_id="key/session",
        labels=ContentLabels((SourceKind.INTERNAL,), Classification.INTERNAL),
    )


def _stored(seq: int, kind: str, payload: dict[str, Any]) -> StoredRecord:
    turn_id = None if kind == "session_created" else "turn"
    operation_id = "operation" if kind.startswith("op_") else None
    attempt = 1 if operation_id is not None else None
    record = Record(
        record_identity(kind, "session", turn_id, operation_id, attempt, payload),
        kind,
        "session",
        turn_id,
        operation_id,
        attempt,
        payload,
    )
    return StoredRecord(record, seq, f"commit/{seq}", 1, 1)


def _driver(rows: tuple[StoredRecord, ...], reader: _Reader | None) -> CoreDriver:
    driver = object.__new__(CoreDriver)
    driver.sid = "session"
    driver.store = SimpleNamespace(tenant_id="tenant")
    driver.content_reader = reader
    driver._hydrators = {}
    driver.records = rows
    driver._stored_by_id = {row.record.record_id: row for row in rows}
    driver.state = SimpleNamespace(operations={}, active_turn_id="turn")
    return driver


def test_context_compaction_keeps_array_order_and_source_digest() -> None:
    first = _ref("first", {"role": "system", "content": "first"})
    second = _ref("second", {"role": "user", "content": "second"})
    evidence = _ref("evidence", {"artifacts": [], "cursors": []})
    source_digest = sha256(b"clear-source-digest-must-survive").hexdigest()
    stored = _stored(
        3,
        "context_compacted",
        {
            "source_digest": source_digest,
            "prefix_ids": ["m1"],
            "tail_ids": ["m2"],
            "mode": "summary",
            "summary_operation_id": "summary",
            "replacement": [second.to_dict(), first.to_dict()],
            "evidence_manifest": evidence.to_dict(),
        },
    )
    reader = _Reader()
    reader.add("record:3", "replacement[0]", second, {"role": "user", "content": "second"})
    reader.add("record:3", "replacement[1]", first, {"role": "system", "content": "first"})
    reader.add("record:3", "evidence_manifest", evidence, {"artifacts": [], "cursors": []})

    hydrated = hydrate_context_compacted(_driver((stored,), reader), stored.record)

    assert hydrated.payload["replacement"] == [
        {"role": "user", "content": "second"},
        {"role": "system", "content": "first"},
    ]
    assert hydrated.payload["evidence_manifest"] == {"artifacts": [], "cursors": []}
    assert hydrated.payload["source_digest"] == source_digest
    assert hydrated.payload["prefix_ids"] == ["m1"]
    assert reader.calls == [
        ("record:3", "replacement[0]", second.content_id),
        ("record:3", "replacement[1]", first.content_id),
        ("record:3", "evidence_manifest", evidence.content_id),
    ]


def test_boundary_output_and_memory_entries_hydrate_only_content() -> None:
    reason = _ref("reason", "invalid")
    value = _ref("value", {"answer": 1})
    partial = _ref("partial", {"answer": 0})
    memory_one = _ref("memory-one", "keep user constraint")
    memory_two = _ref("memory-two", "keep file decision")
    output = _stored(
        4,
        "boundary_recorded",
        {
            "stage": "output_checked",
            "boundary_id": "final",
            "source_operation_id": None,
            "source_digest": None,
            "data": {
                "status": "failed",
                "reason": reason.to_dict(),
                "value": value.to_dict(),
                "partial_output": partial.to_dict(),
            },
        },
    )
    memory = _stored(
        5,
        "boundary_recorded",
        {
            "stage": "session_memory_saved",
            "boundary_id": "m",
            "source_operation_id": None,
            "source_digest": None,
            "data": {
                "state": {
                    "entries": [
                        {"category": "decision", "content": memory_one.to_dict(), "source_cycle": 4, "importance": 8},
                        {"category": "key_fact", "content": memory_two.to_dict(), "source_cycle": 5, "importance": 7},
                    ],
                    "last_extracted_message_index": 9,
                    "tokens_at_last_extraction": 100,
                    "initialized": True,
                }
            },
        },
    )
    reader = _Reader()
    for slot, ref, body in (
        ("output_reason", reason, "invalid"),
        ("output_value", value, {"answer": 1}),
        ("partial_output", partial, {"answer": 0}),
        ("session_memory_content[0]", memory_one, "keep user constraint"),
        ("session_memory_content[1]", memory_two, "keep file decision"),
    ):
        owner = "record:4" if slot.startswith(("output", "partial")) else "record:5"
        reader.add(owner, slot, ref, body)
    driver = _driver((output, memory), reader)

    assert hydrate_boundary_data(driver, output.record) == {
        "status": "failed",
        "reason": "invalid",
        "value": {"answer": 1},
        "partial_output": {"answer": 0},
    }
    state = hydrate_boundary_data(driver, memory.record)["state"]
    assert [entry["content"] for entry in state["entries"]] == ["keep user constraint", "keep file decision"]
    assert [(entry["category"], entry["importance"]) for entry in state["entries"]] == [
        ("decision", 8),
        ("key_fact", 7),
    ]
    assert state["last_extracted_message_index"] == 9


def test_turn_ended_hydrates_reason_result_and_budget() -> None:
    reason = _ref("terminal-reason", "tool_finish")
    result = _ref("terminal-result", {"content": "done"})
    budget = _ref("terminal-budget", {"cycles": 1})
    stored = _stored(
        6,
        "turn_ended",
        {
            "status": "completed",
            "reason": reason.to_dict(),
            "result": result.to_dict(),
            "adopted_results": [],
            "budget": budget.to_dict(),
            "unconfirmed_operations": [],
        },
    )
    reader = _Reader()
    reader.add("record:6", "turn_ended_reason", reason, "tool_finish")
    reader.add("record:6", "turn_ended_result", result, {"content": "done"})
    reader.add("record:6", "turn_ended_budget", budget, {"cycles": 1})

    hydrated = hydrate_turn_ended(_driver((stored,), reader), stored.record)

    assert hydrated.payload["reason"] == "tool_finish"
    assert hydrated.payload["result"] == {"content": "done"}
    assert hydrated.payload["budget"] == {"cycles": 1}


def test_declared_slots_fail_closed_for_missing_reader_plaintext_and_ref_digest() -> None:
    message = _ref("replacement", {"role": "user", "content": "go"})
    empty_evidence = _ref("empty-evidence", {"artifacts": [], "cursors": []})
    compacted = _stored(
        7,
        "context_compacted",
        {
            "source_digest": sha256(b"source").hexdigest(),
            "prefix_ids": [],
            "tail_ids": [],
            "mode": "micro",
            "summary_operation_id": None,
            "replacement": [message.to_dict()],
            "evidence_manifest": empty_evidence.to_dict(),
        },
    )
    with pytest.raises(RecordError, match="hydration is unavailable"):
        hydrate_context_compacted(_driver((compacted,), None), compacted.record)

    plaintext = _stored(
        8,
        "context_compacted",
        compacted.record.payload | {"replacement": [{"role": "user", "content": "plaintext"}]},
    )
    with pytest.raises(RecordError, match="invalid session schema"):
        hydrate_context_compacted(_driver((plaintext,), _Reader()), plaintext.record)

    definition = _ref("definition", {"agent_name": "agent"})
    started = _stored(
        9,
        "turn_started",
        {
            "definition": definition.to_dict(),
            "definition_digest": "0" * 64,
            "budget": None,
        },
    )
    reader = _Reader()
    reader.add("record:9", "definition", definition, {"agent_name": "agent"})
    with pytest.raises(ReferenceDigestMismatch):
        starter = _driver((started,), reader)
        starter.hydrated_definition(started.record)
    assert reference_digest(definition) != started.record.payload["definition_digest"]
