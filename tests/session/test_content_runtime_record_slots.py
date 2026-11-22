from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._driver_recovery import PushProviderEvidence
from vv_agent.session._recovery_phase import _Host
from vv_agent.session.content_runtime import (
    SessionContentWriter,
    durable_content_store,
    read_input_json,
    read_record_json,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, _Scope, drive
from vv_agent.session.providers import Accepted, Unknown
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

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
PROVIDER_RESULT_SLOTS = (
    ("provider.result", "result"),
    ("provider.usage", "usage"),
    ("provider.evidence", "evidence"),
)
PROVIDER_EVIDENCE_SLOTS = (
    ("provider.handle.job_id", "handle.job_id"),
    ("provider.handle.evidence", "handle.evidence"),
    ("provider.handle.query_ref", "handle.query_ref"),
    ("provider.handle.cancel_ref", "handle.cancel_ref"),
)


@function_tool
def provider_slots() -> str:
    raise AssertionError("the provider owns execution")


class _SlotProvider:
    def submit(self, plan, *, context):
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "job_id": "job-1",
            "evidence": "attempt-1",
            "query_ref": "cursor-1",
            "cancel_ref": "token-1",
        }
        self.result = ToolExecutionResult(
            tool_call_id=plan.payload["request"]["id"],
            content="provider-result",
        ).to_dict()
        return Accepted(self.handle)

    def query(self, handle):
        return Accepted(handle)

    def cancel(self, handle):
        return Unknown("provider slots do not test cancellation")

    def authenticate(self, item, plan):
        if any(item.payload[field] != self.handle[field] for field in ("operation_id", "attempt", "request_digest")):
            return False
        if item.kind == "provider_evidence":
            return item.payload["handle"] == self.handle
        return item.payload["result"] == self.result and item.payload["evidence"] == [self.handle["evidence"]]


@contextmanager
def _driver(tmp_path: Path, name: str) -> Iterator[tuple[SQLiteStore, SessionDriver]]:
    with SQLiteStore.standalone(str(tmp_path / f"{name}.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{name}-keys.sqlite3")
        yield store, SessionDriver(store=store, content_key_provider=provider)


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


@pytest.mark.parametrize(
    ("kind", "slots"),
    [("provider_result", PROVIDER_RESULT_SLOTS), ("provider_evidence", PROVIDER_EVIDENCE_SLOTS)],
)
def test_provider_input_slots_use_real_input_owner_writer(
    tmp_path: Path,
    kind: str,
    slots: tuple[tuple[str, str], ...],
) -> None:
    with _driver(tmp_path, "provider-input") as (store, driver):
        driver.create(SESSION, str(tmp_path), {})
        driver.push(SESSION, InboxItem("initial", "user", {"content": "start"}))
        provider = _SlotProvider()
        model = ScriptedModelProvider.new(
            "record-slots",
            "record-slots",
            [LLMResponse("", [ToolCall("call", "provider_slots", {})]), LLMResponse("done")],
        )
        runtime = driver.runtime(
            Agent("record-slots", "Exercise provider content slots.", tools=[provider_slots]),
            RunConfig(model_provider=model, workspace=tmp_path, tool_providers={"provider_slots": provider}),
            session_id=SESSION,
        )
        drive(store, SESSION, runtime=runtime)

        handle = provider.handle
        state = store.read_state(SESSION)[0]
        operation_id, attempt = handle["operation_id"], handle["attempt"]
        assert isinstance(operation_id, str) and isinstance(attempt, int)
        assert state.operations[operation_id].attempts[attempt].state == "parked"
        turn_id = state.active_turn_id
        assert turn_id is not None
        generation = state.turns[turn_id].start._payload["generation"]
        identity = {key: handle[key] for key in ("operation_id", "attempt", "request_digest")}
        if kind == "provider_result":
            payload = identity | {
                "provider_binding": handle["provider"],
                "result": provider.result,
                "usage": {"input_tokens": 3},
                "evidence": [handle["evidence"]],
            }
            driver.push(SESSION, InboxItem("callback", kind, payload, turn_id, generation))
        else:
            payload = identity | {"handle": handle}
            item = InboxItem("callback", kind, payload, turn_id, generation)
            lease = store.acquire(SESSION, owner="slot-evidence", ttl_ms=15_000)
            assert lease is not None
            try:
                scope = _Scope(lease, runtime, now_ms=store._now())
                recovery = _Host(_Driver(driver.store, SESSION, runtime, scope))
                assert recovery.handle(PushProviderEvidence(item)).status == "admitted"
            finally:
                store.release(lease)

        drive(store, SESSION, runtime=runtime)
        stored = store.read_input(SESSION, "callback")
        assert stored is not None

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            for slot, path in slots:
                ref = ContentRef.from_dict(_value(stored.item.payload, path))
                assert writer.assert_input("callback", ref, slot) == ref
                assert writer.get_json(ref) == _value(payload, path)

        for slot, path in slots:
            ref = ContentRef.from_dict(_value(stored.item.payload, path))
            assert read_input_json(
                store,
                tenant_id="default",
                session_id=SESSION,
                input_id="callback",
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
