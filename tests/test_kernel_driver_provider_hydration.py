"""Provider inbox authentication keeps retained references separate from logical values."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent.kernel._boundary_hydration import hydrate_input
from vv_agent.kernel._content_publication import (
    declared_content_ref,
    link_declared_record_content,
    publish_record_content,
    resolve_record_content_links,
)
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_inputs import (
    AuthenticateProvider,
    CommitInput,
    CompleteResult,
    InputSetting,
    apply_input,
)
from vv_agent.kernel._provider_outcomes import (
    Accepted,
    ProviderResultInvalid,
    validate_acceptance,
    validate_reference_provider_handle,
)
from vv_agent.kernel._records import InboxItem, Record, digest, record_identity
from vv_agent.kernel._sql import stored_input_proof
from vv_agent.kernel._state import Attempt, ExecutionState, Operation, Turn
from vv_agent.session.content_runtime import SessionContentWriter, durable_content_reader
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"k" * 32
TURN_ID = "s/turn/provider"
OPERATION_ID = "operation/provider-result"
REQUEST_DIGEST = digest({"request": "provider"})


@dataclass(frozen=True)
class _Session:
    store: SQLiteStore
    sessions: SessionDriver


class _Signer:
    key_id = "provider-hydration-key"

    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def _session(tmp_path) -> _Session:
    store = SQLiteStore(str(tmp_path / "provider-hydration.sqlite"))
    store.install_schema()
    store.configure_input_admission(_Signer())
    sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
    sessions.create("s", "/workspace", {})
    return _Session(store, sessions)


def _provider_result() -> tuple[InboxItem, dict, dict]:
    result = {
        "content": "provider-result-secret",
        "opaque_metadata": {"content_id": "host-value", "label": "opaque-result-secret"},
    }
    usage = {"input_tokens": 7, "opaque": {"usage": "usage-secret"}}
    return (
        InboxItem(
            "provider-result",
            "provider_result",
            {
                "operation_id": OPERATION_ID,
                "attempt": 1,
                "request_digest": REQUEST_DIGEST,
                "provider_binding": "echo",
                "result": result,
                "usage": usage,
                "evidence": ["receipt/result"],
            },
            target_turn_id=TURN_ID,
            generation=0,
        ),
        result,
        usage,
    )


def _provider_evidence() -> tuple[InboxItem, dict]:
    handle = {
        "kind": "provider",
        "provider": "echo",
        "job_id": "job-secret",
        "operation_id": OPERATION_ID,
        "attempt": 1,
        "request_digest": REQUEST_DIGEST,
        "evidence": ["handle-evidence-secret"],
        "query_ref": "query-secret",
        "cancel_ref": "cancel-secret",
    }
    return (
        InboxItem(
            "provider-evidence",
            "provider_evidence",
            {
                "operation_id": OPERATION_ID,
                "attempt": 1,
                "request_digest": REQUEST_DIGEST,
                "handle": handle,
            },
            target_turn_id=TURN_ID,
            generation=0,
        ),
        handle,
    )


def _plan() -> Record:
    payload = {
        "op_kind": "model",
        "purpose": "primary",
        "request": {"id": "call", "name": "echo", "arguments": {}},
        "request_digest": REQUEST_DIGEST,
        "context_version": "1",
        "dependencies": [],
        "tool": None,
        "idempotency_key": None,
        "budget_admission": {},
        "not_before_ms": None,
        "provider_binding": "echo",
        "consumed_unknowns": [],
        "policy_decision": None,
    }
    return Record(
        record_identity("op_planned", "s", TURN_ID, OPERATION_ID, 1, payload),
        "op_planned",
        "s",
        TURN_ID,
        OPERATION_ID,
        1,
        payload,
    )


def _driver(session: _Session, *, handle: dict, attempt_state: str, input_seq: int) -> CoreDriver:
    plan = _plan()
    attempt = Attempt(
        plan,
        state=attempt_state,
        dispatch=plan,
        wait=(
            {
                "handle": handle,
                "poll_at_ms": 0,
                "deadline_ms": None,
            }
            if attempt_state == "parked"
            else None
        ),
    )
    start = Record(
        f"turn/{TURN_ID}/start",
        "turn_started",
        "s",
        TURN_ID,
        None,
        None,
        {"generation": 0},
    )
    state = ExecutionState(
        session_id="s",
        operations={OPERATION_ID: Operation(TURN_ID, "model", {1: attempt})},
        turns={TURN_ID: Turn(start)},
        active_turn_id=TURN_ID,
    )
    driver = object.__new__(CoreDriver)
    driver.sid = "s"
    driver.store = session.store
    driver.content_reader = durable_content_reader(session.store, tenant_id="default", session_id="s")
    driver._hydrators = {}
    driver.state = state
    driver.watermark = input_seq
    return driver


def _pending(session: _Session, input_id: str):
    stored = session.store.read_input("s", input_id)
    assert stored is not None
    return stored


def test_logical_and_reference_provider_handles_have_separate_validators() -> None:
    plan = _plan()
    handle = {
        "kind": "provider",
        "provider": "echo",
        "job_id": "job/options",
        "operation_id": OPERATION_ID,
        "attempt": 1,
        "request_digest": REQUEST_DIGEST,
        "evidence": ["receipt/result"],
        "query_ref": "query/options",
        "cancel_ref": None,
    }

    validate_acceptance(plan, Accepted(handle))
    with pytest.raises(ProviderResultInvalid):
        validate_reference_provider_handle(plan, handle)


def test_provider_result_round_trip_authenticates_logical_and_retains_reference_bytes(tmp_path) -> None:
    session = _session(tmp_path)
    item, expected_result, expected_usage = _provider_result()
    evidence_item, _ = _provider_evidence()
    try:
        session.sessions.push("s", item)
        session.sessions.push("s", evidence_item)
        raw = _pending(session, item.input_id)
        raw_evidence = _pending(session, evidence_item.input_id)

        assert declared_content_ref(raw.item.payload["result"]) is not None
        assert declared_content_ref(raw.item.payload["usage"]) is not None
        assert declared_content_ref(raw.item.payload["evidence"]) is not None
        assert "provider-result-secret" not in raw.item.encode().decode()
        handle = {
            "kind": "provider",
            "provider": "echo",
            "job_id": "job/result",
            "operation_id": OPERATION_ID,
            "attempt": 1,
            "request_digest": REQUEST_DIGEST,
            "evidence": "receipt/result",
            "query_ref": None,
            "cancel_ref": None,
        }
        driver = _driver(session, handle=handle, attempt_state="parked", input_seq=raw.input_seq)

        logical = hydrate_input(driver, raw.item)
        logical_evidence = hydrate_input(driver, raw_evidence.item)
        assert logical.payload["result"] == expected_result
        assert logical.payload["usage"] == expected_usage
        assert logical_evidence.payload["handle"]["job_id"] == "job-secret"
        assert logical_evidence.payload["handle"]["evidence"] == ["handle-evidence-secret"]

        pending = apply_input(driver, one_turn=True)
        authenticated = next(pending)
        assert isinstance(authenticated, AuthenticateProvider)
        assert authenticated.retained_item.to_dict() == raw.item.to_dict()
        assert declared_content_ref(authenticated.retained_item.payload["result"]) is not None
        assert authenticated.item.payload["result"] == expected_result

        completed = pending.send(True)
        assert isinstance(completed, CompleteResult)
        assert completed.outcome.result == expected_result
        assert completed.outcome.usage == expected_usage
        assert completed.outcome.evidence == ("receipt/result",)

        committed = pending.send([])
        assert isinstance(committed, CommitInput)
        applied = committed.records[0]
        proof = stored_input_proof(raw)
        assert proof is not None
        assert applied._payload["input"] == raw.item.to_dict()
        assert InboxItem(**applied._payload["input"]).encode() == raw.item.encode()
        assert applied._payload["input_digest"] == raw.item.digest
        assert applied._payload["input_proof"] == proof
        assert declared_content_ref(applied._payload["input"]["payload"]["result"]) is not None
        try:
            next(pending)
        except StopIteration as done:
            assert done.value is True
        else:
            raise AssertionError("apply_input yielded after CommitInput")

        retained_payload = applied._payload | {
            "disposition": "rejected",
            "reason": "retained reference-form audit",
        }
        retained_record = Record(
            record_identity(
                "input_applied",
                "s",
                applied.turn_id,
                None,
                None,
                retained_payload,
            ),
            "input_applied",
            "s",
            applied.turn_id,
            None,
            None,
            retained_payload,
        )
        lease = session.store.acquire("s", owner="provider-hydration", ttl_ms=15_000)
        assert lease is not None
        with session.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            publication = publish_record_content(writer, (retained_record,))
            receipt = tx.append(
                "s",
                lease=lease,
                expected_seq=session.store.read("s").head_seq,
                commit_id="provider-result-hydration",
                records=publication.records,
                consume_input_ids=(item.input_id,),
            )
            links = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, links)

        retained = next(row.record for row in session.store.read("s").records if row.record.kind == "input_applied")
        assert retained._payload["input"] == raw.item.to_dict()
        assert retained._payload["input_digest"] == raw.item.digest
        assert retained._payload["input_proof"] == proof
        persisted = b"\n".join(
            row[0]
            for table in ("sk_inbox", "sk_record")
            for row in session.store._rows(
                f"SELECT body FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )
        )
        for plaintext in (
            b"provider-result-secret",
            b"opaque-result-secret",
            b"usage-secret",
            b"job-secret",
            b"handle-evidence-secret",
        ):
            assert plaintext not in persisted
    finally:
        session.sessions.close()
        session.store.connection.close()


def test_provider_evidence_authenticates_raw_and_accepts_logical_handle(tmp_path) -> None:
    session = _session(tmp_path)
    item, expected_handle = _provider_evidence()
    try:
        session.sessions.push("s", item)
        raw = _pending(session, item.input_id)
        assert declared_content_ref(raw.item.payload["handle"]["job_id"]) is not None
        assert "job-secret" not in raw.item.encode().decode()
        driver = _driver(session, handle=expected_handle, attempt_state="started", input_seq=raw.input_seq)

        pending = apply_input(driver, one_turn=True)
        authenticated = next(pending)
        assert isinstance(authenticated, AuthenticateProvider)
        assert authenticated.retained_item.payload["handle"]["job_id"] != expected_handle["job_id"]
        assert declared_content_ref(authenticated.retained_item.payload["handle"]["job_id"]) is not None
        assert authenticated.item.payload["handle"]["job_id"] == expected_handle["job_id"]

        assert pending.send(True) is InputSetting.POLL_NOW
        assert pending.send(1_000) is InputSetting.POLL_MS
        committed = pending.send(250)
        assert isinstance(committed, CommitInput)
        applied, parked = committed.records
        proof = stored_input_proof(raw)
        assert proof is not None
        assert applied.kind == "input_applied"
        assert applied._payload["input"] == raw.item.to_dict()
        assert InboxItem(**applied._payload["input"]).encode() == raw.item.encode()
        assert applied._payload["input_digest"] == raw.item.digest
        assert applied._payload["input_proof"] == proof
        assert parked.kind == "op_parked"
        assert parked._payload["phase"] == "after_dispatch"
        assert parked._payload["handle"] == expected_handle
        assert parked._payload["poll_at_ms"] == 1_250
        assert parked._payload["deadline_ms"] is None
        try:
            next(pending)
        except StopIteration as done:
            assert done.value is True
        else:
            raise AssertionError("apply_input yielded after CommitInput")
    finally:
        session.sessions.close()
        session.store.connection.close()
