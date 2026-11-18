from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._records import digest
from vv_agent.session.content_runtime import (
    SessionContentWriter,
    durable_content_store,
    read_input_json,
    read_record_json,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"r" * 32
SESSION = "record-slots"
RECORD_SLOT_CASES = (
    ("definition", {"kind": "definition"}),
    ("budget", {"max_tokens": 100}),
    ("result", {"answer": "record-result"}),
    ("usage", {"input_tokens": 7}),
    ("replacement[0]", {"role": "assistant", "content": "replacement"}),
    ("definition.task.initial_messages[1]", {"role": "user", "content": "initial"}),
    ("request.messages[2]", {"role": "system", "content": "request"}),
    ("hook_result.metadata.children[0]", {"child": "result"}),
    ("input.payload.content", {"text": "input"}),
    ("input.payload.result", {"answer": "provider"}),
    ("input.payload.handle.job_id", {"job": "job-id"}),
    ("message[0]", {"role": "user", "content": "boundary"}),
    ("stop_message", {"reason": "stop"}),
    ("session_memory_content[0]", {"memory": "entry"}),
    ("boundary_error", {"kind": "error"}),
    ("budget_usage", {"input_tokens": 5}),
)
PROVIDER_INPUT_CASES = (
    (
        "provider-result",
        "provider_result",
        {
            "operation_id": "operation/result",
            "attempt": 1,
            "request_digest": digest({"request": "result"}),
            "provider_binding": None,
            "result": {"answer": "provider-result"},
            "usage": {"input_tokens": 3},
            "evidence": {"request_id": "request-1"},
        },
        (
            ("provider.result", "result"),
            ("provider.usage", "usage"),
            ("provider.evidence", "evidence"),
        ),
    ),
    (
        "provider-evidence",
        "provider_evidence",
        {
            "operation_id": "operation/evidence",
            "attempt": 1,
            "request_digest": digest({"request": "evidence"}),
            "handle": {
                "kind": "provider",
                "provider": "test",
                "operation_id": "operation/evidence",
                "attempt": 1,
                "request_digest": digest({"request": "evidence"}),
                "job_id": {"job": "job-1"},
                "evidence": {"attempt": 1},
                "query_ref": {"cursor": "query-1"},
                "cancel_ref": {"token": "cancel-1"},
            },
        },
        (
            ("provider.handle.job_id", "handle.job_id"),
            ("provider.handle.evidence", "handle.evidence"),
            ("provider.handle.query_ref", "handle.query_ref"),
            ("provider.handle.cancel_ref", "handle.cancel_ref"),
        ),
    ),
)


@contextmanager
def _driver(tmp_path: Path, name: str) -> Iterator[tuple[SQLiteStore, SessionDriver]]:
    with SQLiteStore.standalone(str(tmp_path / f"{name}.sqlite")) as store:
        store.install_schema()
        yield store, SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))


def _value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return value


def test_record_inventory_slots_link_assert_and_read(tmp_path: Path) -> None:
    with _driver(tmp_path, "record") as (store, driver):
        driver.create(SESSION, "/workspace", {})
        refs: dict[str, ContentRef] = {}

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            for slot, value in RECORD_SLOT_CASES:
                ref = writer.put_json({"slot": slot, "value": value})
                writer.link_record(1, ref, slot)
                assert writer.assert_record(1, ref, slot) == ref
                assert writer.get_json(ref) == {"slot": slot, "value": value}
                refs[slot] = ref

        for slot, value in RECORD_SLOT_CASES:
            assert read_record_json(
                store,
                tenant_id="default",
                session_id=SESSION,
                record_seq=1,
                ref=refs[slot],
                slot=slot,
            ) == {"slot": slot, "value": value}


def test_provider_input_slots_use_real_input_owner_writer(tmp_path: Path) -> None:
    with _driver(tmp_path, "provider-input") as (store, driver):
        driver.create(SESSION, "/workspace", {})
        for input_id, kind, payload, _slots in PROVIDER_INPUT_CASES:
            driver.push(SESSION, InboxItem(input_id, kind, payload))

        for input_id, _kind, payload, slots in PROVIDER_INPUT_CASES:
            stored = store.read_input(SESSION, input_id)
            assert stored is not None

            with store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
                for slot, path in slots:
                    ref = ContentRef.from_dict(_value(stored.item.payload, path))
                    assert writer.assert_input(input_id, ref, slot) == ref
                    assert writer.get_json(ref) == _value(payload, path)

            for slot, path in slots:
                ref = ContentRef.from_dict(_value(stored.item.payload, path))
                assert read_input_json(
                    store,
                    tenant_id="default",
                    session_id=SESSION,
                    input_id=input_id,
                    ref=ref,
                    slot=slot,
                ) == _value(payload, path)


def test_owner_specific_slots_are_not_interchangeable(tmp_path: Path) -> None:
    with _driver(tmp_path, "owner-isolation") as (store, driver):
        driver.create(SESSION, "/workspace", {})
        driver.push(SESSION, InboxItem("user-input", "user", {"content": {"text": "hello"}}))

        with pytest.raises(ValueError, match="stable declared slot"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            writer.link_input("user-input", writer.put_json("record-only"), "definition")

        with pytest.raises(ValueError, match="stable declared slot"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            writer.link_record(1, writer.put_json("input-only"), "input.content")

        with pytest.raises(ValueError, match="stable declared slot"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            writer.durable.link_record(
                "default",
                SESSION,
                1,
                writer.put_json("durable-bypass"),
                "input.content",
            )

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            with pytest.raises(ValueError, match="stable declared slot"):
                durable_content_store(store).link_record(
                    "default",
                    SESSION,
                    1,
                    writer.put_json("durable-helper-bypass"),
                    "input.content",
                )


@pytest.mark.parametrize(
    "slot",
    (
        "undeclared",
        "replacement",
        "replacement[*]",
        "replacement[**]",
        "replacement[01]",
        "replacement[-1]",
        "replacement[0][0]",
        "replacement[0].content",
        "message[*]",
        "message[0].content",
        "definition.task.initial_messages",
        "definition.task.initial_messages[00]",
        "input.payload.content.more",
        "input.payload.unknown",
        "boundary.data.reason",
    ),
)
def test_record_writer_rejects_invalid_or_out_of_inventory_slots(tmp_path: Path, slot: str) -> None:
    with _driver(tmp_path, "invalid-record-slot") as (store, driver):
        driver.create(SESSION, "/workspace", {})

        with pytest.raises(ValueError, match="stable declared slot"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            writer.link_record(1, writer.put_json({"slot": slot}), slot)
