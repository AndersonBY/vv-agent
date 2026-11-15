"""S1 input proof and cross-session content boundary regressions."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent._memory_settings import MemoryManagerSettings
from vv_agent.kernel._content import ContentLabels, ContentRef, ContentUnavailable, Principal
from vv_agent.kernel._driver_inputs import CommitInput
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
from vv_agent.kernel._records import InboxItem
from vv_agent.kernel._sql import InputAdmissionProofError
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import InMemoryEncryptedContentStore, LocalKeyProvider
from vv_agent.session.fork import fork_thread
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"k" * 32

ACTOR = Principal(
    tenant_id="default",
    subject_id="input-user",
    subject_kind="user",
    attributes=ContentRef("actor", "0" * 64, 0, "key", ContentLabels()),
).to_dict()


class _Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "ledger-key-a"

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


def _signed_driver(tmp_path, mutation: str):
    store = SQLiteStore.standalone(str(tmp_path / f"{mutation}.sqlite"))
    return store


def _start_signed(tmp_path, mutation: str = "input"):
    stack = _signed_driver(tmp_path, mutation)
    store = stack.__enter__()
    store.install_schema()
    signer = _Signer()
    driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
    driver.store.configure_input_admission(signer)
    driver.create("session", "/workspace", {"seed": {}})
    return stack, store, driver, signer


def _input(mutation: str) -> InboxItem:
    if mutation == "control":
        return InboxItem("input", "control", {"action": "archive"})
    return InboxItem("input", "user", {"content": "trusted input"})


def test_signed_admission_is_stored_deduped_and_copied_exactly(tmp_path):
    stack, store, driver, signer = _start_signed(tmp_path)
    try:
        first = driver.push("session", _input("user"))
        stored = store.read_input("session", "input")
        assert stored is not None
        proof = stored.item.admission_proof
        assert proof is not None
        assert proof["admission_key_id"] == signer.key_id
        assert proof["admission_signed_ms"] == stored.received_ms
        assert store._rows("SELECT admission_key_id,admission_signature IS NOT NULL FROM sk_inbox") == [(signer.key_id, 1)]

        replay = driver.push("session", _input("user"))
        assert replay.replayed is True
        assert replay.input_seq == first.input_seq
        assert replay.received_ms == first.received_ms
        assert store.read_input("session", "input") == stored

        expected = {
            "input_seq": stored.input_seq,
            "received_ms": stored.received_ms,
            "key_id": proof["admission_key_id"],
            "signature": proof["admission_signature"],
            "signed_ms": proof["admission_signed_ms"],
        }
        from vv_agent.kernel._sql import stored_input_proof

        assert stored_input_proof(stored) == expected
    finally:
        stack.__exit__(None, None, None)


def test_explicit_unsigned_local_admission_keeps_null_proof(tmp_path):
    with SQLiteStore.standalone(str(tmp_path / "unsigned.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        driver.create("session", "/workspace", {"seed": {}})
        driver.push("session", _input("user"))

        stored = store.read_input("session", "input")
        assert stored is not None
        assert stored.item.admission_proof is None
        assert store._rows("SELECT admission_key_id,admission_signature,admission_signed_ms FROM sk_inbox") == [
            (None, None, None)
        ]


@pytest.mark.parametrize(
    "mutation",
    ["actor", "control", "target", "generation", "input_seq", "received_ms", "proof", "missing_proof"],
)
def test_signed_admission_tampering_fails_before_content_read(tmp_path, monkeypatch, mutation):
    stack, store, driver, _signer = _start_signed(tmp_path, mutation)
    try:
        driver.push("session", _input(mutation))
        body, _seq, _received, _digest, _key_id, signature, _signed_ms = store._rows(
            "SELECT body,input_seq,received_ms,digest,admission_key_id,admission_signature,admission_signed_ms FROM sk_inbox"
        )[0]
        item = InboxItem.parse(body)
        if mutation == "actor":
            item = replace(item, actor=item.actor | {"subject_id": "forged"})
        elif mutation == "control":
            item = replace(item, payload={"action": "cancel"})
        elif mutation == "target":
            item = replace(item, target_turn_id="forged-turn")
        elif mutation == "generation":
            item = replace(item, generation=99)
        if mutation in {"actor", "control", "target", "generation"}:
            changed = item.encode()
            store.connection.execute(
                "UPDATE sk_inbox SET body=?,digest=? WHERE input_id='input'",
                (changed, sha256(changed).hexdigest()),
            )
        elif mutation == "input_seq":
            store.connection.execute("UPDATE sk_inbox SET input_seq=2 WHERE input_id='input'")
        elif mutation == "received_ms":
            store.connection.execute("UPDATE sk_inbox SET received_ms=received_ms+1 WHERE input_id='input'")
        elif mutation == "proof":
            store.connection.execute(
                "UPDATE sk_inbox SET admission_signature=? WHERE input_id='input'",
                (bytes([signature[0] ^ 1]) + signature[1:],),
            )
        elif mutation == "missing_proof":
            store.connection.execute(
                "UPDATE sk_inbox SET admission_key_id=NULL,admission_signature=NULL,admission_signed_ms=NULL "
                "WHERE input_id='input'"
            )
        store.connection.commit()

        def fail_get(*_args, **_kwargs):
            raise AssertionError("content decryption must not start for tampered admission")

        monkeypatch.setattr(SQLiteContentStore, "get", fail_get)
        ledger_before = store._rows("SELECT count(*) FROM sk_record")[0][0]
        with pytest.raises(InputAdmissionProofError):
            store.peek_inbox("session", ready_only=False)
        assert store._rows("SELECT count(*) FROM sk_record")[0][0] == ledger_before
    finally:
        stack.__exit__(None, None, None)


def test_input_applied_uses_the_stored_proof_not_a_new_one(tmp_path):
    item = replace(
        _input("control"),
        actor=ACTOR,
        admission_proof={
            "admission_key_id": "ledger-key-a",
            "admission_signature": "AAAA",
            "admission_signed_ms": 11,
        },
    )
    record = driver_record(item, input_seq=7, received_ms=5)
    payload = record.payload
    assert payload["input_proof"] == {
        "input_seq": 7,
        "received_ms": 5,
        "key_id": "ledger-key-a",
        "signature": "AAAA",
        "signed_ms": 11,
    }
    record.encode()


def driver_record(item: InboxItem, *, input_seq: int, received_ms: int):
    from vv_agent.kernel._driver import CoreDriver
    from vv_agent.kernel._driver_inputs import apply_input
    from vv_agent.kernel._state import ExecutionState

    driver = object.__new__(CoreDriver)
    driver.sid = "session"
    driver.watermark = input_seq
    driver.state = ExecutionState()
    stored = SimpleNamespace(item=item, input_seq=input_seq, received_ms=received_ms)
    driver.store = SimpleNamespace(peek_inbox=lambda *_args, **_kwargs: [stored])
    pending = apply_input(driver, one_turn=True)
    request = next(pending)
    assert isinstance(request, CommitInput)
    with suppress(StopIteration):
        next(pending)
    return request.records[0]


def _fork_attributes() -> dict:
    task = AgentTask(
        task_id="source/context",
        model="test-model",
        prompt_bundle=PromptBundle((PromptSection("base", "system prompt"),)),
        user_prompt="",
    ).to_dict()
    return {
        "app_server": {"agent_key": "default", "cwd": None, "metadata": {}},
        "app_server_context": {
            "agent_name": "default",
            "memory_settings": MemoryManagerSettings()._definition_settings(),
            "child_tools": [],
            "model_binding": {},
            "capabilities": {},
            "task": task,
            "tools": [],
        },
        "seed": {"messages": [], "shared_state": {}},
    }


@pytest.mark.parametrize("forgotten", ["source", "destination"])
def test_fork_reencrypts_session_identity_and_survives_one_side_forget(tmp_path, forgotten):
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / f"fork-{forgotten}.sqlite")) as store:
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=provider)
        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="source")

            def source_spec(content):
                principal, _ = content.service_principal("source-host", attributes={"role": "source"})
                return SessionSpec(
                    "source",
                    principal,
                    content.put_json("/workspace").to_dict(),
                    attributes=content.put_json(_fork_attributes()).to_dict(),
                )

            writer.create_session(source_spec, consumers=("events", "app_server", "traces"))

        result = fork_thread(
            driver,
            host=SimpleNamespace(),
            params={"threadId": "source", "forkId": "copy"},
        )
        destination_id = result["threadId"]
        source_created = store.read("source", limit=1).records[0].record.payload
        destination_created = store.read(destination_id, limit=1).records[0].record.payload
        for field in ("attributes", "workspace"):
            assert destination_created[field]["key_id"] != source_created[field]["key_id"]
            assert destination_created[field]["content_id"] != source_created[field]["content_id"]
        destination_principal = destination_created["principal"]["attributes"]
        source_principal = source_created["principal"]["attributes"]
        assert destination_principal["key_id"] != source_principal["key_id"]
        assert destination_principal["content_id"] != source_principal["content_id"]

        survivor = destination_id if forgotten == "source" else "source"
        with store.atomic() as tx:
            SQLiteContentStore(tx, provider).forget_session("default", forgotten)

        survivor_created = store.read(survivor, limit=1).records[0]
        with driver.store.atomic() as tx:
            survivor_writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=survivor)
            attributes = survivor_writer.get_json(survivor_created.record.payload["attributes"])
            workspace = survivor_writer.get_json(survivor_created.record.payload["workspace"])
            principal_attributes = survivor_writer.get_json(survivor_created.record.payload["principal"]["attributes"])
        assert attributes["seed"] == {"messages": [], "shared_state": {}}
        assert workspace == "/workspace"
        assert principal_attributes == {"role": "source"}


@pytest.mark.parametrize("forgotten", ["parent", "child"])
def test_child_result_copy_uses_destination_key_and_outlives_each_side(tmp_path, forgotten):
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "child-copy.sqlite")) as store:
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=provider)

        def create_session(session_id: str) -> None:
            with driver.store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=session_id)

                def spec(content):
                    principal, _ = content.service_principal(f"{session_id}-host")
                    return SessionSpec(
                        session_id,
                        principal,
                        content.put_json("/workspace").to_dict(),
                        attributes=content.put_json({"seed": {}}).to_dict(),
                    )

                writer.create_session(spec, consumers=())

        create_session("parent")
        create_session("child")
        with driver.store.atomic() as tx:
            parent = SessionContentWriter.bind(tx, tenant_id="default", session_id="parent")
            child = SessionContentWriter.bind(tx, tenant_id="default", session_id="child")
            parent_result = parent.put_json({"text": "same result"})
            child_copy = child.copy_json_from(parent, parent_result)
            child_result = child.put_json({"text": "same result"})
            parent_copy = parent.copy_json_from(child, child_result)

        assert child_copy.key_id != parent_result.key_id
        assert child_copy.content_id != parent_result.content_id
        assert child_copy.mac != parent_result.mac
        assert parent_copy.key_id != child_result.key_id
        assert parent_copy.content_id != child_result.content_id

        with store.atomic() as tx:
            SQLiteContentStore(tx, provider).forget_session("default", forgotten)
        survivor = "child" if forgotten == "parent" else "parent"
        survivor_ref = child_result if survivor == "child" else parent_result
        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=survivor)
            assert writer.get_json(survivor_ref) == {"text": "same result"}
        with pytest.raises(ContentUnavailable), driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=forgotten)
            writer.get_json(child_result if forgotten == "child" else parent_result)


def test_same_plaintext_uses_unrelated_tenant_and_session_keys():
    provider = LocalKeyProvider(MASTER_KEY)
    store = InMemoryEncryptedContentStore(provider)
    labels = ContentLabels()
    tenant_a = store.put("tenant-a", "session-shared", b"same", labels)
    tenant_b = store.put("tenant-b", "session-shared", b"same", labels)

    assert tenant_a.key_id != tenant_b.key_id
    assert tenant_a.content_id != tenant_b.content_id
    assert tenant_a.mac != tenant_b.mac
    with pytest.raises(ContentUnavailable):
        store.get("tenant-a", "session-shared", tenant_b)
