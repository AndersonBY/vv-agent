"""Real native ownership lifetimes across a factory's open interval."""

import json
import sys
from contextlib import contextmanager, nullcontext
from weakref import ref

import pytest

from vv_agent.session.ownership import OwnershipError, checkout, native_keys
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sqlite import SQLiteStore

from .test_connection_ownership_factories import alias


@pytest.mark.sqlite_file
def test_retired_host_during_factory_open(store, database, tmp_path):
    retired = SQLiteStore.standalone(str(database))
    old = retired.__enter__()
    old_keys = native_keys(old)
    dead = ref(old)
    del old
    factory = store.execution_store
    exited = False
    observations = {"old_keys": old_keys}

    @contextmanager
    def replacement():
        nonlocal exited
        # The normal owner closes and forgets H after checkout took its snapshot.
        retired.__exit__(None, None, None)
        exited = True
        assert dead() is None
        with factory() as candidate:
            observations.update(
                old_wrapper_dead=True,
                candidate_keys=native_keys(candidate),
                source_keys=native_keys(store),
                target_equal=candidate._target == store._target,
                raw_is_source=candidate.connection is store.connection,
            )
            (tmp_path / "identity-lifetime.json").write_text(json.dumps(observations))
            print("REAL-NATIVE-LIFETIME", observations, flush=True)
            yield candidate

    try:
        with checkout(store, factory=replacement) as execution:
            assert execution._one("SELECT 1")[0] == 1
        assert store._one("SELECT 1")[0] == 1
    finally:
        if not exited:
            retired.__exit__(None, None, None)


@pytest.mark.persistent_store
def test_snapshot_raw_pins_end_even_with_retained_error(store):
    raw = store.connection
    # Match the caller frame's lifetime after a traceback already exists.
    try:
        raise ValueError("baseline traceback")
    except ValueError:
        baseline = sys.getrefcount(raw)
    failure = ValueError("retained execution error")
    with pytest.raises(ValueError) as caught, checkout(store):
        assert sys.getrefcount(raw) > baseline
        raise failure
    assert caught.value is failure and caught.value.__traceback__ is not None
    assert sys.getrefcount(raw) == baseline


@pytest.mark.persistent_store
def test_live_raw_alias_after_host_wrapper_dies(store, database, monkeypatch):
    if isinstance(store, SQLiteStore):
        old = SQLiteStore(str(database))
    else:
        import psycopg

        old = PostgresStore(psycopg.connect(database, autocommit=True))
    raw = old.connection
    dead = ref(old)
    holder = [old]
    del old
    candidates, validations = [], []
    validate = type(store)._validate_checkout

    def checked(candidate, source):
        validations.append(id(candidate.connection))
        return validate(candidate, source)

    @contextmanager
    def borrowed():
        candidate = alias(holder.pop())
        candidates.append(candidate)
        assert dead() is None
        assert candidate.connection is raw
        yield candidate

    monkeypatch.setattr(type(store), "_validate_checkout", checked)
    try:
        with pytest.raises(OwnershipError, match="execution"), checkout(store, factory=borrowed):
            pytest.fail("a pre-open known H was borrowed")
        assert dead() is None and validations == []
        assert candidates[0]._one("SELECT 1")[0] == 1
        assert store._one("SELECT 1")[0] == 1
    finally:
        raw.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("role", ["execution", "heartbeat"])
def test_active_raw_alias_before_validation(store, monkeypatch, role):
    validations = []
    validate = type(store)._validate_checkout

    def checked(candidate, source):
        validations.append(id(candidate.connection))
        return validate(candidate, source)

    with checkout(store) as owner:
        monkeypatch.setattr(type(store), "_validate_checkout", checked)
        with pytest.raises(OwnershipError, match=role), checkout(store, factory=lambda: nullcontext(alias(owner)), role=role):
            pytest.fail("an active connection was rented twice")
        assert validations == []
        assert owner._one("SELECT 1")[0] == store._one("SELECT 1")[0] == 1


@pytest.mark.persistent_store
@pytest.mark.parametrize("primary", [False, True])
def test_closed_exit_error_keeps_original_and_cleans_reservation(store, primary):
    from vv_agent.session.ownership import _active, _lock

    failure, cleanup = ValueError("original execution"), OSError("closed checkout exit")
    factory = store.execution_store
    keys = []

    @contextmanager
    def closing():
        try:
            with factory() as candidate:
                keys.extend(native_keys(candidate))
                yield candidate
        finally:
            raise cleanup

    with pytest.raises(ValueError if primary else OSError) as caught, checkout(store, factory=closing):
        if primary:
            raise failure
    assert caught.value is (failure if primary else cleanup)
    if primary:
        assert caught.value.__notes__ == ["checkout exit failed: OSError"]
    with _lock:
        assert all(key not in _active for key in keys)
    with checkout(store) as fresh:
        assert fresh._one("SELECT 1")[0] == 1
    assert store._one("SELECT 1")[0] == 1
