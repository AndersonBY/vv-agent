"""H6 regressions for outer commit fences and shared SQLite transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.kernel._content_runtime import bind_content_runtime, durable_content_store
from vv_agent.kernel._store import LeaseLost
from vv_agent.session.content_store import AESGCMCipher

from .conftest import open_store
from .helpers import (
    acquire_fixture_lease,
    create_fixture_session,
    fixture_key_provider,
    runtime_turn_record,
    wrap_fixture_store,
)

TENANT = "tenant-a"


@pytest.mark.persistent_store
@pytest.mark.parametrize("fence", ["epoch", "owner", "expiry"])
def test_outer_atomic_rechecks_lease_before_commit(store, fence: str) -> None:
    wrapped = wrap_fixture_store(store)
    create_fixture_session(wrapped, "s", consumers=())
    lease = acquire_fixture_lease(store, "s", owner="active-worker", ttl_ms=60_000)
    assert lease is not None
    record = runtime_turn_record()
    mutations = {
        "epoch": "UPDATE sk_session SET lease_epoch=lease_epoch+1 WHERE tenant_id=%s AND session_id=%s",
        "owner": "UPDATE sk_session SET lease_owner='replacement-worker' WHERE tenant_id=%s AND session_id=%s",
        "expiry": "UPDATE sk_session SET lease_until_ms=0 WHERE tenant_id=%s AND session_id=%s",
    }

    with pytest.raises(LeaseLost, match="expired or fenced"), store.atomic() as tx:
        tx.bind_lease_guard(lambda: lease)
        tx.append(
            "s",
            lease=lease,
            expected_seq=1,
            commit_id=f"stale-{fence}",
            records=(record,),
        )
        store._rows(mutations[fence], (store.tenant_id, "s"))

    assert store.read("s").records[0].record.kind == "session_created"


def test_sqlite_tenant_facade_nested_atomic_uses_outer_savepoint(tmp_path: Path) -> None:
    with open_store(tmp_path / "nested.sqlite") as store:
        store.install_schema()
        facade = store.for_tenant(TENANT)

        with store.atomic():
            with facade.atomic():
                store._rows("UPDATE sk_settings SET write_level=2")
            assert store._one("SELECT write_level FROM sk_settings") == (2,)

            with pytest.raises(RuntimeError, match="rollback inner"), facade.atomic():
                store._rows("UPDATE sk_settings SET write_level=3")
                raise RuntimeError("rollback inner")
            assert store._one("SELECT write_level FROM sk_settings") == (2,)


def test_sqlite_independent_connections_keep_separate_transaction_state(tmp_path: Path) -> None:
    path = tmp_path / "independent.sqlite"
    with open_store(path) as schema:
        schema.install_schema()

    with open_store(path) as first, open_store(path) as second:
        with first.atomic():
            assert first._transaction_in_progress()
            assert not second._transaction_in_progress()
        with second.atomic():
            assert second._transaction_in_progress()


def test_sqlite_durable_content_store_retains_tenant_facade(tmp_path: Path) -> None:
    with open_store(tmp_path / "content-tenant.sqlite") as store:
        store.install_schema()
        provider = fixture_key_provider()
        bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher())

        durable = durable_content_store(store.for_tenant(TENANT))
        backend = cast(Any, durable)._durable._backend

        assert backend.tenant_context_id == TENANT
        assert backend.runtime_mode == "tenant"
