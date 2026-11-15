"""Focused tests for the encrypted S1 content-store foundation."""

from __future__ import annotations

from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vv_agent.kernel import _content as content
from vv_agent.kernel import _content_store as content_store_contract
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.session import content_store

MASTER_KEY = b"m" * 32
TENANT = "tenant-a"
SESSION = "session-a"


def _store() -> tuple[content_store.LocalKeyProvider, content_store.InMemoryEncryptedContentStore]:
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    return provider, content_store.InMemoryEncryptedContentStore(provider)


def _stored(store: content_store.InMemoryEncryptedContentStore, ref: content.ContentRef):
    return store._contents[(TENANT, SESSION, ref.content_id)]


def test_store_protocol_and_round_trip_keep_plaintext_out_of_backend() -> None:
    provider, store = _store()
    plaintext = b"secret user content\x00with bytes"
    ref = store.put(TENANT, SESSION, plaintext, content.ContentLabels())

    assert isinstance(provider, content.KeyProvider)
    assert isinstance(store, content_store_contract.ContentStore)
    assert store.get(TENANT, SESSION, ref) == plaintext
    stored = _stored(store, ref)
    assert stored.ciphertext != plaintext
    assert plaintext not in stored.ciphertext
    assert plaintext not in stored.nonce
    assert ref.mac.encode() not in stored.ciphertext
    assert plaintext not in repr(store._contents).encode()


def test_session_local_dedup_and_cross_tenant_session_isolation() -> None:
    _provider, store = _store()
    labels = content.ContentLabels()
    first = store.put(TENANT, SESSION, b"same plaintext", labels)
    repeated = store.put(TENANT, SESSION, b"same plaintext", labels)
    other_tenant = store.put("tenant-b", SESSION, b"same plaintext", labels)
    other_session = store.put(TENANT, "session-b", b"same plaintext", labels)

    assert repeated is first
    assert len({first.content_id, other_tenant.content_id, other_session.content_id}) == 3
    assert len({first.key_id, other_tenant.key_id, other_session.key_id}) == 3
    assert other_tenant.mac != first.mac
    assert other_session.mac != first.mac


def test_labels_participate_in_identity_and_each_object_gets_fresh_randomness() -> None:
    _provider, store = _store()
    internal = store.put(TENANT, SESSION, b"same plaintext", content.ContentLabels())
    user = store.put(
        TENANT,
        SESSION,
        b"same plaintext",
        content.ContentLabels((content.SourceKind.USER,), content.Classification.CONFIDENTIAL),
    )
    internal_stored = _stored(store, internal)
    user_stored = _stored(store, user)

    assert internal.content_id != user.content_id
    assert internal.mac == user.mac
    assert internal_stored.nonce != user_stored.nonce
    assert internal_stored.ciphertext != user_stored.ciphertext


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "content_id", "size", "mac", "labels"])
def test_tampered_backend_or_reference_is_rejected(field: str) -> None:
    _provider, store = _store()
    plaintext = b"authenticated content"
    ref = store.put(TENANT, SESSION, plaintext, content.ContentLabels())
    key = (TENANT, SESSION, ref.content_id)
    stored = store._contents[key]

    if field == "ciphertext":
        stored = replace(stored, ciphertext=bytes([stored.ciphertext[0] ^ 1]) + stored.ciphertext[1:])
        store._contents[key] = stored
    elif field == "nonce":
        stored = replace(stored, nonce=bytes([stored.nonce[0] ^ 1]) + stored.nonce[1:])
        store._contents[key] = stored
    elif field == "content_id":
        stored = replace(stored, ref=replace(stored.ref, content_id="content_other"))
        store._contents[key] = stored
    elif field == "size":
        ref = replace(ref, size=ref.size + 1)
    elif field == "mac":
        ref = replace(ref, mac="0" * 64)
    else:
        ref = replace(ref, labels=content.ContentLabels(classification=content.Classification.CONFIDENTIAL))

    with pytest.raises(content.ContentIntegrityError):
        store.get(TENANT, SESSION, ref)


def test_wrong_content_id_and_key_binding_are_classified() -> None:
    _provider, store = _store()
    ref = store.put(TENANT, SESSION, b"bound content", content.ContentLabels())

    with pytest.raises(content.ContentUnavailable):
        store.get(TENANT, SESSION, replace(ref, content_id="content_missing"))
    with pytest.raises(content.ContentBindingMismatch):
        store.get(TENANT, SESSION, replace(ref, key_id="key_other"))


def test_wrong_aad_is_rejected_even_with_new_authentication_tag() -> None:
    provider, store = _store()
    plaintext = b"aad-bound content"
    ref = store.put(TENANT, SESSION, plaintext, content.ContentLabels())
    stored = _stored(store, ref)
    wrapped_key = store._session_keys[(TENANT, SESSION)]
    data_key = provider.unwrap(TENANT, SESSION, wrapped_key)
    wrong_aad = canonical_json_bytes(
        {
            "domain": "vv-agent/S1/content",
            "tenant_id": "tenant-other",
            "session_id": SESSION,
            "content_id": ref.content_id,
            "key_id": ref.key_id,
            "size": ref.size,
            "labels": ref.labels.to_dict(),
        }
    )
    tampered = AESGCM(data_key).encrypt(stored.nonce, plaintext, wrong_aad)
    store._contents[(TENANT, SESSION, ref.content_id)] = replace(stored, ciphertext=tampered)

    with pytest.raises(content.ContentIntegrityError):
        store.get(TENANT, SESSION, ref)


def test_unknown_tenant_or_session_cannot_read_content() -> None:
    _provider, store = _store()
    ref = store.put(TENANT, SESSION, b"isolated", content.ContentLabels())

    with pytest.raises(content.ContentUnavailable):
        store.get("tenant-b", SESSION, ref)
    with pytest.raises(content.ContentUnavailable):
        store.get(TENANT, "session-b", ref)


def test_destroyed_key_and_forgotten_session_fail_closed() -> None:
    provider, store = _store()
    ref = store.put(TENANT, SESSION, b"erase me", content.ContentLabels())
    wrapped_key = store._session_keys[(TENANT, SESSION)]

    assert provider.destroy(TENANT, wrapped_key.key_id).status is content.KeyDestroyStatus.DESTROYED
    with pytest.raises(content.KeyUnavailable):
        store.get(TENANT, SESSION, ref)
    assert provider.destroy(TENANT, wrapped_key.key_id).status is content.KeyDestroyStatus.ALREADY_DESTROYED


def test_forget_destroys_key_before_purging_content() -> None:
    provider, store = _store()
    ref = store.put(TENANT, SESSION, b"forget me", content.ContentLabels())
    wrapped_key = store._session_keys[(TENANT, SESSION)]

    store.forget_session(TENANT, SESSION)

    assert provider.destroy(TENANT, wrapped_key.key_id).status is content.KeyDestroyStatus.ALREADY_DESTROYED
    assert not store._contents
    assert not store._session_keys
    with pytest.raises(content.ContentUnavailable):
        store.get(TENANT, SESSION, ref)


def test_forget_does_not_purge_when_key_destroy_is_unconfirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, store = _store()
    plaintext = b"keep until key revocation"
    ref = store.put(TENANT, SESSION, plaintext, content.ContentLabels())

    def reject_destroy(tenant_id: str, key_id: str) -> content.KeyDestroyOutcome:
        return content.KeyDestroyOutcome(
            False,
            content.KeyDestroyStatus.KEY_UNAVAILABLE,
            None,
            "provider unavailable",
        )

    monkeypatch.setattr(provider, "destroy", reject_destroy)
    with pytest.raises(content.KeyUnavailable):
        store.forget_session(TENANT, SESSION)
    assert store.get(TENANT, SESSION, ref) == plaintext
