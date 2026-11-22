"""Core-owned S1 content runtime over caller-owned SQL storage."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from vv_agent.kernel import _content_runtime as core_runtime
from vv_agent.kernel._content import ContentLabels, ContentRef
from vv_agent.kernel._content_publication import (
    declare_input_content,
    link_declared_input_content,
    link_declared_record_content,
    publish_record_content,
    resolve_record_content_links,
)
from vv_agent.kernel._content_reader import DurableContentReader as CoreDurableContentReader
from vv_agent.kernel._content_reader import input_owner, record_owner
from vv_agent.kernel._records import InboxItem, SessionSpec, digest, make_record
from vv_agent.kernel._sql import LedgerIntegrityError
from vv_agent.session import content_runtime as sdk_runtime
from vv_agent.session.content_reader import DurableContentReader as SDKDurableContentReader
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

from .conftest import open_store
from .helpers import content_ref

MASTER_KEY = b"c" * 32
TENANT = "default"
SESSION = "core-content"
TURN = "core-content/turn/1"
INPUT = "core-content-input"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"


def _binding(tmp_path: Path) -> tuple[LocalKeyProvider, AESGCMCipher, core_runtime.ContentRuntimeBinding]:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "core-content-keys.sqlite3")
    cipher = AESGCMCipher()
    return provider, cipher, core_runtime.ContentRuntimeBinding(provider, cipher)


def _create_session(store: Any, binding: core_runtime.ContentRuntimeBinding) -> Any:
    wrapped = core_runtime.wrap_session_content_store(store, binding)
    with wrapped.atomic() as tx:
        writer = core_runtime.SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)

        def spec(content):
            actor, _ = content.service_principal("core-runtime-host")
            return SessionSpec(
                SESSION,
                actor,
                content.put_json({"workspace": "core-runtime"}).to_dict(),
                attributes={"purpose": "core-runtime-probe"},
            )

        writer.create_session(spec, consumers=("events",))
    return wrapped


def test_unconfigured_ledger_rejects_creation_before_content_factory(tmp_path: Path) -> None:
    _, _, binding = _binding(tmp_path)
    with open_store(tmp_path / "unconfigured.sqlite", allow_unsigned=False) as raw_store:
        raw_store.install_schema()
        wrapped = core_runtime.wrap_session_content_store(raw_store, binding)
        with wrapped.atomic() as tx:
            writer = core_runtime.SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)

            def spec_factory(_writer: core_runtime.SessionContentWriter) -> SessionSpec:
                pytest.fail("unconfigured ledger must reject before invoking the content factory")

            with pytest.raises(LedgerIntegrityError, match="session creation requires an explicit signed signer"):
                writer.create_session(spec_factory, consumers=("events",))


def _publish_input_and_record(store: Any) -> tuple[ContentRef, ContentRef, int]:
    with store.atomic() as tx:
        writer = core_runtime.SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
        actor, _ = writer.service_principal(f"user/{INPUT}")
        publication = declare_input_content(
            writer,
            InboxItem(INPUT, "user", {"content": {"text": "published input"}}, actor=actor),
        )
        tx.push(SESSION, publication.item)
        link_declared_input_content(writer, publication.item)
        input_ref = ContentRef.from_dict(publication.item.payload["content"])

    lease = store.acquire(SESSION, owner="core-content-append", ttl_ms=60_000)
    assert lease is not None
    try:
        with store.atomic() as tx:
            writer = core_runtime.SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
            definition = writer.put_json({"purpose": "core runtime", "messages": []})
            budget = writer.put_json({"max_tokens": 100})
            producer = make_record(
                "turn_started",
                session_id=SESSION,
                turn_id=TURN,
                payload={
                    "input_ids": [],
                    "definition": definition.to_dict(),
                    "definition_digest": digest(definition.to_dict()),
                    "handler_version": "1",
                    "budget": budget.to_dict(),
                    "binding": "task",
                    "generation": 1,
                    "policy_digest": None,
                    "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
                    "tokenizer_binding": None,
                    "environment": None,
                },
            )
            publication = publish_record_content(writer, (producer,))
            receipt = tx.append(
                SESSION,
                lease=lease,
                expected_seq=1,
                commit_id="core-content-turn-start",
                records=publication.records,
            )
            links = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, links)
            definition_link = next(link for link in links if link.slot == "definition")
            record_ref, record_seq = definition_link.ref, definition_link.record_seq
    finally:
        store.release(lease)
    return input_ref, record_ref, record_seq


@pytest.mark.persistent_store
def test_core_runtime_real_store_roundtrip_owner_links_rollback_and_cold_read(
    store_backend: str,
    database: str | Path,
    tmp_path: Path,
) -> None:
    provider, cipher, binding = _binding(tmp_path)
    immutable_field = "cipher"
    with pytest.raises(FrozenInstanceError):
        setattr(binding, immutable_field, cipher)

    with open_store(database) as raw_store:
        raw_store.install_schema()
        store = _create_session(raw_store, binding)
        with store.heartbeat_store() as heartbeat:
            assert isinstance(heartbeat, core_runtime._ContentSessionStoreAdapter)
            assert core_runtime.content_runtime_binding_for(heartbeat) is binding
            assert heartbeat.connection is not raw_store.connection

        input_ref, record_ref, record_seq = _publish_input_and_record(store)
        durable = core_runtime.durable_content_store(store, binding=binding)
        reader = CoreDurableContentReader(durable, tenant_id=TENANT, session_id=SESSION)
        assert reader.read(TENANT, SESSION, input_owner(INPUT), "input.content", input_ref) == b'{"text":"published input"}'
        assert reader.read(TENANT, SESSION, record_owner(record_seq), "definition", record_ref)

        with store.atomic() as tx:
            writer = core_runtime.SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
            expected = core_runtime.CoreSQLiteContentStore if store_backend == "sqlite" else core_runtime.CorePostgresContentStore
            assert type(writer._durable) is expected
            assert isinstance(writer._durable._cipher, AESGCMCipher)
            assert writer._durable._key_provider is provider

        before = raw_store._one("SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s", (TENANT, SESSION))[0]

        class _Rollback(BaseException):
            pass

        with pytest.raises(_Rollback), store.atomic() as tx:
            core_runtime.SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION).put_json({"rolled": "back"})
            raise _Rollback
        after = raw_store._one("SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s", (TENANT, SESSION))[0]
        assert after == before

    with open_store(database) as recovered:
        durable = core_runtime.durable_content_store(recovered, binding=binding)
        reader = CoreDurableContentReader(durable, tenant_id=TENANT, session_id=SESSION)
        assert reader.read(TENANT, SESSION, input_owner(INPUT), "input.content", input_ref) == b'{"text":"published input"}'
        assert reader.read(TENANT, SESSION, record_owner(record_seq), "definition", record_ref)
        assert durable.assert_link(TENANT, SESSION, record_ref, "definition", record_seq=record_seq) == record_ref


@pytest.mark.persistent_store
def test_drive_store_factory_opens_configured_content_runtime_connections(
    store_backend: str,
    database: str | Path,
    tmp_path: Path,
) -> None:
    _provider, _cipher, binding = _binding(tmp_path)
    with open_store(database) as raw_store:
        raw_store.install_schema()
        raw_store.configure_worker(level=3, handler_versions=("1", "2", "3"), capabilities=("child-admission",))
        raw_store.configure_input_admission(allow_unsigned=True)
        store = _create_session(raw_store, binding)
        input_ref, _record_ref, _record_seq = _publish_input_and_record(store)
        factory = store.drive_store_factory()

        with factory() as drive, store.heartbeat_store() as heartbeat:
            assert isinstance(drive, core_runtime._ContentSessionStoreAdapter)
            assert core_runtime.content_runtime_binding_for(drive) is binding
            assert drive.tenant_id == TENANT
            assert drive.worker_supports == raw_store.worker_supports
            assert drive.fold_factory is raw_store.fold_factory
            assert drive._ledger_integrity_mode() == "hash-only"
            assert drive._root.connection is not raw_store.connection
            assert drive._root.connection is not heartbeat._root.connection

            if store_backend == "sqlite":
                assert isinstance(raw_store, SQLiteStore)
                assert drive._root.clock is raw_store.clock
            else:
                drive_timeouts = drive._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')")
                heartbeat_timeouts = heartbeat._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')")
                assert heartbeat_timeouts == ("250ms", "1s")
                assert drive_timeouts != heartbeat_timeouts

            reader = CoreDurableContentReader(
                core_runtime.durable_content_store(drive),
                tenant_id=TENANT,
                session_id=SESSION,
            )
            assert reader.read(TENANT, SESSION, input_owner(INPUT), "input.content", input_ref) == b'{"text":"published input"}'

            with factory() as second_drive:
                assert core_runtime.content_runtime_binding_for(second_drive) is binding
                assert second_drive._root.connection is not drive._root.connection
                if store_backend == "postgres":
                    first_pid = drive._one("SELECT pg_backend_pid()")[0]
                    second_pid = second_drive._one("SELECT pg_backend_pid()")[0]
                    assert second_pid != first_pid


def test_sdk_constructor_defaults_to_real_aes_and_reexports_core_identities(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "sdk-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "sdk-content.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        with store.atomic() as tx:
            writer = sdk_runtime.SessionContentWriter(tx=tx, tenant_id=TENANT, session_id=SESSION, key_provider=provider)
            writer.create_session(
                lambda content: SessionSpec(
                    SESSION,
                    content.service_principal("sdk-host")[0],
                    content.put_json("/workspace").to_dict(),
                    attributes=content_ref("attributes"),
                ),
                consumers=(),
            )
            adapter = SQLiteContentStore(tx, provider)
            ref = adapter.put(TENANT, SESSION, b"sdk-default-aes", ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, ref, "workspace")
        assert adapter.get(TENANT, SESSION, ref) == b"sdk-default-aes"

    assert sdk_runtime._ContentSessionStoreAdapter is core_runtime._ContentSessionStoreAdapter
    assert sdk_runtime._ContentSessionTxAdapter is core_runtime._ContentSessionTxAdapter
    assert SDKDurableContentReader is CoreDurableContentReader
    assert issubclass(sdk_runtime.SessionContentWriter, core_runtime.SessionContentWriter)


def test_core_runtime_executes_with_real_aes_under_isolated_dependency_guard(tmp_path: Path) -> None:
    probe = r"""
import importlib.abc
import sys
import sysconfig

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

class HostCipher:
    def encrypt(self, key, nonce, plaintext, aad):
        return AESGCM(key).encrypt(nonce, plaintext, aad)
    def decrypt(self, key, nonce, ciphertext, aad):
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    def random_bytes(self, n):
        return secrets.token_bytes(n)

class HostProvider:
    provider_id = "host-aesgcm-v1"
    def __init__(self):
        self.master = AESGCM(b"m" * 32)
    def generate_data_key(self, tenant_id, session_id):
        from vv_agent.kernel._content import WrappedDataKey
        key_id = "key/" + secrets.token_hex(8)
        key = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        aad = (tenant_id + "\0" + session_id + "\0" + key_id).encode()
        return WrappedDataKey(tenant_id, session_id, key_id, nonce + self.master.encrypt(nonce, key, aad))
    def unwrap(self, tenant_id, session_id, wrapped):
        aad = (tenant_id + "\0" + session_id + "\0" + wrapped.key_id).encode()
        return self.master.decrypt(wrapped.wrapped_key[:12], wrapped.wrapped_key[12:], aad)
    def destroy(self, tenant_id, key_id):
        raise RuntimeError("not used")

class HostClock:
    def __init__(self): self.value = 1
    def now_ms(self):
        self.value += 1
        return self.value
    def monotonic(self): return float(self.value) / 1000
    def monotonic_ns(self): return self.value * 1_000_000
    def sleep(self, seconds): self.value += int(seconds * 1000)

class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cryptography") or fullname == "vv_agent.session" or fullname.startswith("vv_agent.session."):
            raise AssertionError("forbidden kernel dependency: " + fullname)

crypto_before = {name for name in sys.modules if name.startswith("cryptography")}
sys.meta_path.insert(0, Forbidden())

from vv_agent.kernel._content import ContentLabels
from vv_agent.kernel._content_reader import DurableContentReader, record_owner
from vv_agent.kernel._content_runtime import (
    ContentRuntimeBinding,
    SessionContentWriter,
    SESSION_WORKSPACE,
    content_runtime_binding_for,
    durable_content_store,
    wrap_session_content_store,
)
from vv_agent.kernel._records import SessionSpec
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sqlite import CoreSQLiteContentStore, SQLiteStore

assert not any(name == "vv_agent.session" or name.startswith("vv_agent.session.") for name in sys.modules)
assert {name for name in sys.modules if name.startswith("cryptography")} == crypto_before
assert CoreSQLiteContentStore.__module__ == "vv_agent.kernel._sqlite"
assert SessionContentWriter.__module__ == "vv_agent.kernel._content_runtime"

provider, cipher = HostProvider(), HostCipher()
binding = ContentRuntimeBinding(provider, cipher)
def fold_factory():
    return Fold(normalize_memory_settings=lambda settings, _level: settings, count_messages=lambda messages, *, model: 0)

with SQLiteStore.standalone(sys.argv[3], fold_factory=fold_factory, clock=HostClock()) as raw:
    raw.install_schema()
    raw.configure_input_admission(allow_unsigned=True)
    store = wrap_session_content_store(raw, binding)
    workspace = []
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="isolated")
        def spec(content):
            workspace.append(content.put_json({"workspace": "isolated"}))
            return SessionSpec(
                "isolated",
                content.service_principal("isolated-host")[0],
                workspace[-1].to_dict(),
                attributes={"purpose": "isolated"},
            )
        writer.create_session(spec, consumers=())
    with store.drive_store_factory()() as drive:
        assert content_runtime_binding_for(drive) is binding
        drive_reader = DurableContentReader(durable_content_store(drive), tenant_id="default", session_id="isolated")
        assert (
            drive_reader.read("default", "isolated", record_owner(1), SESSION_WORKSPACE, workspace[0])
            == b'{"workspace":"isolated"}'
        )
    content = durable_content_store(raw, binding=binding)
    reader = DurableContentReader(content, tenant_id="default", session_id="isolated")
    assert reader.read("default", "isolated", record_owner(1), SESSION_WORKSPACE, workspace[0]) == b'{"workspace":"isolated"}'
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            textwrap.dedent(probe),
            str(SOURCE),
            sysconfig.get_paths()["purelib"],
            str(tmp_path / "isolated.sqlite"),
        ],
        cwd="/",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
