"""Current lease ownership and content transaction entry contracts."""

from dataclasses import replace
from typing import cast

import pytest

from vv_agent.kernel._content import KeyDestroyOutcome, KeyDestroyStatus, WrappedDataKey
from vv_agent.kernel._content_runtime import bind_content_runtime, wrap_session_content_store
from vv_agent.kernel._lease_scope import HeartbeatStopped, LeaseOwnership
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._store import ContentSessionTx, Lease, LeaseLost, LeasePoll, SessionStore
from vv_agent.session.reducer import Fold


class _Clock:
    def __init__(self, monotonic_value: float = 10.0) -> None:
        self.monotonic_value = monotonic_value

    def now_ms(self) -> int:
        return 1_000

    def monotonic(self) -> float:
        return self.monotonic_value

    def monotonic_ns(self) -> int:
        return int(self.monotonic_value * 1_000_000_000)

    def sleep(self, seconds: float) -> None:
        raise AssertionError(f"unexpected sleep: {seconds}")


class _Signal:
    def __init__(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        return self.flag


class _Store:
    def __init__(self, receipt: LeasePoll | None = None, *, owner: str | None = None) -> None:
        self.receipt = receipt
        self.owner = owner
        self.renewals = 0

    def renew(self, lease: Lease, *, ttl_ms: int) -> LeasePoll:
        del lease, ttl_ms
        self.renewals += 1
        if self.receipt is None:
            raise AssertionError("renewal was not configured")
        if self.owner is None:
            return self.receipt
        return replace(self.receipt, lease=replace(self.receipt.lease, owner=self.owner))


def _ownership(clock: _Clock | None = None) -> tuple[LeaseOwnership, _Clock, Lease]:
    clock = clock or _Clock()
    lease = Lease("s", "owner", 1, 5_000)
    state = LeaseOwnership(
        lease,
        4_000,
        0.04,
        clock=clock,
        stop=_Signal(),
        ready=_Signal(),
        now_ms=clock.now_ms(),
    )
    return state, clock, lease


def test_lease_ownership_has_no_deleted_latch_api_and_anchors_commits():
    state, _clock, lease = _ownership()
    assert not any(hasattr(state, name) for name in ("lock", "poll_lock", "commit_lock", "_latch_entry"))
    assert state.latest_poll is None
    with state.committing() as leased:
        assert leased is lease


def test_renewal_snapshot_is_the_commit_authority():
    state, _clock, lease = _ownership()
    receipt = LeasePoll(replace(lease, expires_at_ms=6_000), 1_100, 3, ())
    store = _Store(receipt)

    assert state.poll(cast(SessionStore, store)) is receipt
    assert state.latest_poll is receipt
    with state.committing() as committed:
        assert committed is receipt.lease
    assert store.renewals == 1


def test_renewal_identity_change_loses_authority_without_a_latch_release_path():
    state, _clock, lease = _ownership()
    receipt = LeasePoll(replace(lease, expires_at_ms=6_000), 1_100, 3, ())
    store = _Store(receipt, owner="replacement")

    with pytest.raises(LeaseLost, match="changed lease identity"):
        state.poll(cast(SessionStore, store))
    assert state.lost
    with pytest.raises(LeaseLost, match="lost ownership"), state.committing():
        pass


def test_stopped_and_expired_scopes_fail_closed():
    stopped, _clock, _lease = _ownership()
    stopped.request_stop()
    with pytest.raises(HeartbeatStopped, match="heartbeat stopped"), stopped.committing():
        pass

    expired, clock, _lease = _ownership()
    clock.monotonic_value = expired.expires_deadline
    with pytest.raises(LeaseLost, match="expired"), expired.committing():
        pass


class _KeyProvider:
    provider_id = "lease-entry-test"

    def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
        return WrappedDataKey(tenant_id, session_id, f"key/{session_id}", b"wrapped-key")

    def unwrap(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
        del tenant_id, session_id
        return wrapped_key.wrapped_key

    def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome:
        return KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, f"destroy/{tenant_id}/{key_id}")


class _Cipher:
    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        del key, nonce, aad
        return plaintext

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        del key, nonce, aad
        return ciphertext

    def random_bytes(self, n: int) -> bytes:
        return b"r" * n


def test_wrapped_store_atomic_yields_content_capable_transaction(tmp_path):
    with SQLiteStore.standalone(
        str(tmp_path / "content-tx.sqlite"),
        fold_factory=Fold,
        clock=_Clock(),
    ) as raw_store:
        raw_store.install_schema()
        binding = bind_content_runtime(raw_store, key_provider=_KeyProvider(), cipher=_Cipher())
        store = wrap_session_content_store(raw_store, binding)
        with store.atomic() as tx:
            assert isinstance(tx, ContentSessionTx)
            writer = tx.content_writer(tenant_id=raw_store.tenant_id, session_id="s")
            assert writer.tenant_id == raw_store.tenant_id
            assert writer.session_id == "s"
