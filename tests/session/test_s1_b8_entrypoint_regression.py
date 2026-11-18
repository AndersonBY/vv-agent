"""B8 regressions for disposable ownership and unconfigured fail-closed reads."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent import Agent, RunConfig, Runner
from vv_agent.kernel._sql import LedgerIntegrityError
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

from .helpers import create_fixture_session, item, push_fixture_input, wrap_fixture_store

MESSAGE = "explicit signed signer or allow_unsigned hash-only"


class Signer:
    key_id = "b8-ledger-key"

    def __init__(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()

    def bind_key(self):
        private_key = self.private_key

        class Bound:
            key_id = self.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return private_key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.private_key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def test_signed_store_reopen_without_verifier_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "signed.sqlite"
    signer = Signer()
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create_fixture_session(wrap_fixture_store(store), "s")

    with SQLiteStore.standalone(str(path)) as reopened:
        with pytest.raises(LedgerIntegrityError, match=MESSAGE):
            reopened.read("s")

        reopened.configure_input_admission(signer)
        assert reopened.read("s").head_seq == 1


def test_unconfigured_strong_reads_and_idle_deferral_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "unconfigured.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        wrapped = wrap_fixture_store(store)
        create_fixture_session(wrapped, "s", consumers=("host",))
        push_fixture_input(wrapped, "s", item("input"))
        lease = wrapped.acquire("s", owner="writer", ttl_ms=60_000)
        assert lease is not None

    with SQLiteStore.standalone(str(path)) as unconfigured:
        for read in (
            lambda: unconfigured.read("s"),
            lambda: unconfigured.read_state("s"),
            lambda: unconfigured.read_input("s", "input"),
            lambda: unconfigured.peek_inbox("s"),
        ):
            with pytest.raises(LedgerIntegrityError, match=MESSAGE):
                read()

        with pytest.raises(LedgerIntegrityError, match=MESSAGE), unconfigured.atomic() as tx:
            tx.consumer_batch("s", "host")
        with pytest.raises(LedgerIntegrityError, match=MESSAGE):
            unconfigured._defer_idle_drive(lease, poll_ms=1_000)
        with pytest.raises(LedgerIntegrityError, match=MESSAGE):
            unconfigured.defer_idle_drive(lease, poll_ms=1_000)


def test_default_driver_and_runner_configure_only_their_temporary_store(tmp_path: Path) -> None:
    driver = SessionDriver()
    try:
        assert driver._root_store._ledger_integrity_mode() == "hash-only"
        driver.create("driver-session", ".")
        assert driver.store.read_state("driver-session")[0].session_id == "driver-session"
    finally:
        driver.close()

    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    result = Runner.run_sync(Agent("runner", "Answer.", model="m"), "go", run_config=RunConfig(model_provider=provider))
    assert result.final_output == "done"

    key_provider = LocalKeyProvider(b"k" * 32, state_path=tmp_path / "keys.sqlite3")
    durable = SessionDriver(tmp_path / "durable.sqlite", content_key_provider=key_provider)
    try:
        assert durable._root_store._ledger_integrity_mode() == "unconfigured"
        with pytest.raises(LedgerIntegrityError, match=MESSAGE):
            durable.create("durable-session", ".")
    finally:
        durable.close()
        key_provider.close()
