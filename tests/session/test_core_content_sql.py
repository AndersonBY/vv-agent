"""Core-owned SQL content rules, exercised through the production SDK adapter."""

from __future__ import annotations

import ast
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vv_agent.kernel import _content as content
from vv_agent.kernel._content_sql import ContentExportRef as CoreContentExportRef
from vv_agent.kernel._content_sql import (
    CoreSQLContentStore,
    _content_aad,
    _content_mac,
    _key_from_row,
    _random_token,
)
from vv_agent.kernel._content_sql import (
    _SQLContentBackend as CoreSQLContentBackend,
)
from vv_agent.kernel._records import SessionSpec
from vv_agent.session import content_store as sdk_content_store
from vv_agent.session.content_store import AESGCMCipher, ContentExportRef, LocalKeyProvider
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
CORE_SOURCE = SOURCE / "vv_agent" / "kernel" / "_content_sql.py"
MASTER_KEY = b"k" * 32
TENANT = "tenant-a"
SESSION = "session-a"


def _content_ref(label: str) -> dict[str, Any]:
    payload = label.encode()
    return {
        "content_id": f"content/contract/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": "key/contract",
        "labels": {"source": [], "classification": "internal"},
    }


def _principal() -> dict[str, Any]:
    return {
        "tenant_id": TENANT,
        "subject_id": "user/session-a",
        "subject_kind": "user",
        "attributes": _content_ref("principal"),
    }


def _create_session(store: SQLiteStore) -> None:
    with store.for_tenant(TENANT).atomic() as tx:
        tx.create(
            SessionSpec(
                SESSION,
                _principal(),
                _content_ref("workspace"),
                attributes=_content_ref("attributes"),
            ),
            consumers=(),
        )


def test_core_content_sql_imports_without_session_or_cryptography() -> None:
    tree = ast.parse(CORE_SOURCE.read_text(encoding="utf-8"), filename=str(CORE_SOURCE))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not any(name == "cryptography" or name.startswith("cryptography.") for name in imported)
    assert not any(name == "vv_agent.session" or name.startswith("vv_agent.session.") for name in imported)

    probe = r"""
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])

class ForbiddenDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cryptography") or fullname == "vv_agent.session" or fullname.startswith("vv_agent.session."):
            raise AssertionError(f"forbidden core dependency: {fullname}")

blocker = ForbiddenDependency()
sys.meta_path.insert(0, blocker)
from vv_agent.kernel._content_sql import CoreSQLContentStore, _content_aad, _content_mac

assert CoreSQLContentStore.__module__ == "vv_agent.kernel._content_sql"
assert _content_aad.__module__ == "vv_agent.kernel._content_sql"
assert _content_mac.__module__ == "vv_agent.kernel._content_sql"
assert not any(name == "cryptography" or name.startswith("cryptography.") for name in sys.modules)
assert not any(name == "vv_agent.session" or name.startswith("vv_agent.session.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", probe, str(SOURCE)],
        cwd="/",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_framing_bytes_and_sdk_identity_are_stable() -> None:
    labels = content.ContentLabels()
    aad = _content_aad(
        "tenant-a",
        "session-a",
        "content_fixed",
        "key_fixed",
        "ab" * 32,
        7,
        labels,
    )
    assert sha256(aad).hexdigest() == "e470938b9a741b867276031f499221faee8318e3afa401fcbc68f6ba487e3117"
    assert _content_mac(b"k" * 32, "tenant-a", "session-a", b"payload") == (
        "e88cdfe37a56fba5a3cae61003fd709329d472bd3d0b77af9c6479619ffdeb22"
    )
    assert ContentExportRef is CoreContentExportRef
    assert sdk_content_store._SQLContentBackend is CoreSQLContentBackend
    assert sdk_content_store._content_aad is _content_aad
    assert sdk_content_store._content_mac is _content_mac
    assert sdk_content_store._key_from_row is _key_from_row
    assert issubclass(SQLiteContentStore, CoreSQLContentStore)
    with patch.object(sdk_content_store.secrets, "token_bytes", return_value=b"r" * 32):
        assert _random_token(AESGCMCipher(), 32) == sdk_content_store.secrets.token_urlsafe(32)


def test_sqlite_sdk_round_trip_linking_and_tenant_isolation(tmp_path: Path) -> None:
    path = tmp_path / "core-content.sqlite"
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "key-state.sqlite3")
    plaintext = b"core-owned encrypted content"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        _create_session(store)
        with store.for_tenant(TENANT).atomic():
            adapter = SQLiteContentStore(store, provider)
            ref = adapter.put(TENANT, SESSION, plaintext, content.ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, ref, "probe")

        assert adapter.get(TENANT, SESSION, ref) == plaintext
        assert adapter.assert_link(TENANT, SESSION, ref, "probe", record_seq=1) is ref
        assert adapter.read_links(TENANT, SESSION, record_seq=1)[0].ref == ref
        with pytest.raises(content.ContentUnavailable):
            adapter.get("tenant-b", SESSION, ref)
        with pytest.raises(content.ContentUnavailable):
            adapter.get(TENANT, "session-b", ref)


def test_real_aes_adapter_preserves_transport_errors_and_classifies_bad_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "core-content-integrity.sqlite"
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "key-state-integrity.sqlite3")
    plaintext = b"authenticated by the real SDK cipher"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        _create_session(store)
        with store.for_tenant(TENANT).atomic():
            adapter = SQLiteContentStore(store, provider)
            ref = adapter.put(TENANT, SESSION, plaintext, content.ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, ref, "probe")

        def timeout_decrypt(_key: bytes, _nonce: bytes, _ciphertext: bytes, _aad: bytes) -> bytes:
            raise TimeoutError("cipher backend timed out")

        with monkeypatch.context() as scoped:
            scoped.setattr(adapter._cipher, "decrypt", timeout_decrypt)
            with pytest.raises(TimeoutError, match="cipher backend timed out"):
                adapter.get(TENANT, SESSION, ref)

        nonce, ciphertext = store._one(
            "SELECT nonce,ciphertext FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (TENANT, SESSION, ref.content_id),
        )
        tampered = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
        store._rows(
            "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (tampered, TENANT, SESSION, ref.content_id),
        )
        with pytest.raises(content.ContentIntegrityError, match="AEAD") as tampered_error:
            adapter.get(TENANT, SESSION, ref)
        assert isinstance(tampered_error.value.__cause__, InvalidTag)

        key_id, _provider_id, wrapped_key, _state = store._one(
            "SELECT key_id,provider_id,wrapped_key,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        )
        data_key = provider.unwrap(TENANT, SESSION, content.WrappedDataKey(TENANT, SESSION, key_id, wrapped_key))
        wrong_aad = _content_aad(
            "tenant-b",
            SESSION,
            ref.content_id,
            ref.key_id,
            ref.mac,
            ref.size,
            ref.labels,
        )
        wrong_aad_ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, wrong_aad)
        store._rows(
            "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (wrong_aad_ciphertext, TENANT, SESSION, ref.content_id),
        )
        with pytest.raises(content.ContentIntegrityError, match="AEAD") as aad_error:
            adapter.get(TENANT, SESSION, ref)
        assert isinstance(aad_error.value.__cause__, InvalidTag)
