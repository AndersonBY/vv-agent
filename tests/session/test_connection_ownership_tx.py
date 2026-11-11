"""Lexical and thread guards execute before waiting on a real connection lock."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from vv_agent.session.ownership import OwnershipError
from vv_agent.session.postgres import join_transaction


@pytest.mark.parametrize("inner_rollback", [False, True])
def test_lexical_thread(store, inner_rollback):
    class Rollback(Exception):
        pass

    with store.atomic() as outer:
        with ThreadPoolExecutor(1) as pool:

            def crossed():
                with pytest.raises(OwnershipError) as caught:
                    _ = outer.connection
                assert caught.value.code == "transaction_thread"

            pool.submit(crossed).result(2)
        try:
            with store.atomic() as inner:
                assert inner.connection is outer.connection
                if inner_rollback:
                    raise Rollback()
        except Rollback:
            pass
        for operation in [lambda: inner.connection, lambda: inner.lock_session("absent")]:
            with pytest.raises(OwnershipError) as caught:
                operation()
            assert caught.value.code == "transaction_scope"
        assert outer.connection is store.connection
    with pytest.raises(OwnershipError, match="transaction_scope"):
        _ = outer.connection


@pytest.mark.postgres
def test_native_join(store):
    with store.connection.transaction():
        tx = join_transaction(store.connection)
        assert tx.connection is store.connection
        tx.connection.execute("SELECT 1")
        assert tx.store._target is None
    with store.connection.transaction(), pytest.raises(OwnershipError, match="transaction_scope"):
        _ = tx.connection


def test_fixed_connection_identity_precedes_native_sql(store, monkeypatch):
    calls = []
    with store.atomic() as tx:
        connection = store.connection
        try:
            store.connection = object()
            monkeypatch.setattr(store, "_transaction_id", lambda: calls.append(1))
            with pytest.raises(OwnershipError, match="transaction_scope"):
                _ = tx.connection
            with pytest.raises(OwnershipError, match="transaction_scope"):
                tx.lock_session("never-query")
            assert calls == []
        finally:
            store.connection = connection
