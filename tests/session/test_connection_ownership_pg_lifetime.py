"""Simulated PID reuse, with real PostgreSQL connections and unchanged native pgraw keys.

These tests do not claim or attempt real server PID reuse. Only the candidate's
reported backend PID is replaced with a real earlier PID when that identity
element exists. With raw-only identities this substitution has no key to change.
"""

import ast
import inspect
import json
import sqlite3
from contextlib import contextmanager, nullcontext
from weakref import ref

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from vv_agent.session import ownership
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sql import SQLStore
from vv_agent.session.sqlite import SQLiteStore

from .test_connection_ownership_factories import alias

pytestmark = pytest.mark.postgres


def backend_key(store):
    info = store.connection.info
    return ("pgbackend", info.host, info.port, info.dbname, info.backend_pid)


def simulate_pid_reuse(monkeypatch, source, backend, facts, *, excluded=()):
    original = ownership.native_keys

    def reported(candidate):
        actual = original(candidate)
        if candidate is source or any(candidate is owner for owner in excluded):
            return actual
        keys = tuple(backend if key[0] == "pgbackend" else key for key in actual)
        facts.setdefault("candidates", []).append(
            {"actual": actual, "reported": keys, "actual_backend": backend_key(candidate), "simulated_backend": backend}
        )
        assert tuple(key for key in keys if key[0] == "pgraw") == tuple(key for key in actual if key[0] == "pgraw")
        return keys

    monkeypatch.setattr(ownership, "native_keys", reported)


def observe_validation(monkeypatch, facts):
    original = PostgresStore._validate_checkout

    def validated(candidate, source):
        result = original(candidate, source)
        facts.setdefault("validated", []).append(
            {"raw": id(candidate.connection.pgconn), "target_equal": candidate._target == source._target}
        )
        return result

    monkeypatch.setattr(PostgresStore, "_validate_checkout", validated)


def save(tmp_path, facts):
    (tmp_path / "simulated-pid-reuse.json").write_text(json.dumps(facts, indent=2))
    print("SIMULATED-PID-REUSE", json.dumps(facts), flush=True)


def test_simulated_pid_reuse_after_retired_host_during_factory_open(store, database, tmp_path, monkeypatch):
    """Simulated PID reuse after a real other H closes/forgets during default factory open."""
    retired = PostgresStore.standalone(database)
    owner = retired.__enter__()
    old_keys = ownership.native_keys(owner)
    backend = backend_key(owner)
    holders = [owner]
    del owner
    facts = {"simulated_pid_reuse": True, "old_keys": old_keys, "validated": []}
    factory = store.execution_store
    simulate_pid_reuse(monkeypatch, store, backend, facts)
    observe_validation(monkeypatch, facts)

    @contextmanager
    def replacement():
        old = holders.pop()
        retired.__exit__(None, None, None)
        ownership.forget_host(old)
        facts["retired_closed"] = old.connection.closed
        assert old.connection.closed
        del old
        with factory() as candidate:
            yield candidate

    try:
        with ownership.checkout(store, factory=replacement) as execution:
            assert execution._one("SELECT 1")[0] == 1
            facts["executed"] = True
        assert len(facts["validated"]) >= 2 and all(v["target_equal"] for v in facts["validated"])
        assert store._one("SELECT 1")[0] == 1
    finally:
        if holders:
            old = holders.pop()
            retired.__exit__(None, None, None)
            ownership.forget_host(old)
        save(tmp_path, facts)


def test_simulated_pid_reuse_after_source_closed_default_factory(store, tmp_path, monkeypatch):
    """Simulated PID reuse must not compare a closed source's cached PID in factory-alias."""
    old_keys = ownership.native_keys(store)
    backend = backend_key(store)
    store.connection.close()
    ownership.forget_host(store)
    facts = {"simulated_pid_reuse": True, "old_keys": old_keys, "source_closed": True, "validated": []}
    simulate_pid_reuse(monkeypatch, store, backend, facts)
    observe_validation(monkeypatch, facts)
    try:
        with ownership.checkout(store) as execution:
            assert execution.connection is not store.connection
            assert execution._one("SELECT 1")[0] == 1
            facts["executed"] = True
        assert len(facts["validated"]) >= 2 and all(v["target_equal"] for v in facts["validated"])
        assert store.connection.closed
    finally:
        save(tmp_path, facts)


@pytest.mark.parametrize("host", ["source", "other"])
def test_simulated_pid_collision_with_live_host_uses_distinct_raw(store, database, tmp_path, monkeypatch, host):
    """Simulated PID reuse with a live H does not alias a distinct real PGconn."""
    context = nullcontext(store) if host == "source" else PostgresStore.standalone(database)
    facts = {"simulated_pid_reuse": True, "host": host, "validated": []}
    with context as owner:
        simulate_pid_reuse(monkeypatch, store, backend_key(owner), facts, excluded=(owner,))
        observe_validation(monkeypatch, facts)
        with ownership.checkout(store) as candidate:
            assert candidate.connection.pgconn is not owner.connection.pgconn
            assert ownership.native_keys(candidate) == (("pgraw", id(candidate.connection.pgconn)),)
            assert candidate._one("SELECT 1")[0] == 1
            facts["executed"] = True
        assert len(facts["validated"]) >= 2 and all(v["target_equal"] for v in facts["validated"])
        assert not owner.connection.closed
        assert owner._one("SELECT 1")[0] == store._one("SELECT 1")[0] == 1
    save(tmp_path, facts)


def test_live_pgraw_alias_is_rejected(store, monkeypatch):
    """A real shared pgconn still rejects; omit the backend tag to isolate the raw guard."""
    candidate = alias(store)
    original = ownership.native_keys
    validations = []
    assert candidate.connection.pgconn is store.connection.pgconn

    def raw_only(value):
        keys = original(value)
        return tuple(key for key in keys if key[0] == "pgraw") if value is candidate else keys

    monkeypatch.setattr(ownership, "native_keys", raw_only)
    monkeypatch.setattr(PostgresStore, "_validate_checkout", lambda *args: validations.append(args))
    with (
        pytest.raises(ownership.OwnershipError, match="execution"),
        ownership.checkout(store, factory=lambda: nullcontext(candidate)),
    ):
        pytest.fail("an actual pgraw alias was borrowed")
    assert validations == [] and store._one("SELECT 1")[0] == 1


def test_simulated_pid_reuse_cleans_candidate_registration(store, database, tmp_path, monkeypatch):
    """Simulated PID reuse also cleans a fresh candidate despite the stale snapshot entry."""
    retired = PostgresStore.standalone(database)
    owner = retired.__enter__()
    backend = backend_key(owner)
    holders = [owner]
    del owner
    facts = {"simulated_pid_reuse": True}
    candidates = []
    simulate_pid_reuse(monkeypatch, store, backend, facts)

    @contextmanager
    def replacement():
        old = holders.pop()
        retired.__exit__(None, None, None)
        ownership.forget_host(old)
        with PostgresStore.standalone(database) as candidate:
            candidates.append(candidate)
            yield candidate
        # standalone closes its own connection; checkout owns the temporary ledger cleanup.

    try:
        with ownership.checkout(store, factory=replacement) as execution:
            keys = execution._ownership_keys
            assert execution._one("SELECT 1")[0] == 1
        assert candidates[0].connection.closed
        with ownership._lock:
            assert all(key not in ownership._active for key in keys)
            assert all(key not in ownership._hosts or ownership._hosts[key][0]() is not candidates[0] for key in keys)
        facts["registration_cleaned"] = True
    finally:
        if holders:
            old = holders.pop()
            retired.__exit__(None, None, None)
            ownership.forget_host(old)
        save(tmp_path, facts)


def independent_factory(source, database, registered):
    if registered:
        return source.execution_store

    @contextmanager
    def unregistered():
        conninfo = make_conninfo(database, passfile="/dev/null", sslmode="disable", gssencmode="disable")
        with psycopg.connect(conninfo, autocommit=True) as connection:
            candidate = PostgresStore.__new__(PostgresStore)
            candidate._initialize(connection, source=source)
            yield candidate

    return unregistered


def confirm_backend_exit(observer, backend, victim, facts):
    # One independent observation; no I/O on the victim and no retry/polling.
    count = observer._one("SELECT count(*) FROM pg_stat_activity WHERE pid=%s", (backend[-1],))[0]
    facts.update(old_backend_count=count, old_closed=victim.connection.closed, old_pgraw=id(victim.connection.pgconn))
    assert count == 0


def accept_fresh(store, factory, victim, facts):
    with ownership.checkout(store, factory=factory) as candidate:
        assert candidate.connection.pgconn is not victim.connection.pgconn
        assert ownership.native_keys(candidate) == (("pgraw", id(candidate.connection.pgconn)),)
        assert candidate._one("SELECT 1")[0] == 1
        facts["executed"] = True


@pytest.mark.parametrize("registered", [True, False], ids=["default", "unregistered"])
def test_simulated_pid_reuse_after_closed_active(store, database, tmp_path, monkeypatch, registered):
    """Simulated PID reuse cannot borrow the identity of a normally closed active E."""
    facts = {"case": "closed-active", "simulated_pid_reuse": True, "registered": registered, "validated": []}
    with PostgresStore.standalone(database) as observer:
        try:
            with ownership.checkout(store) as old:
                backend = backend_key(old)
                old.connection.close()
                confirm_backend_exit(observer, backend, old, facts)
                simulate_pid_reuse(monkeypatch, store, backend, facts, excluded=(old, observer))
                observe_validation(monkeypatch, facts)
                accept_fresh(store, independent_factory(store, database, registered), old, facts)
                assert facts["validated"] and all(v["target_equal"] for v in facts["validated"])
        finally:
            save(tmp_path, facts)


@pytest.mark.parametrize("registered", [True, False], ids=["default", "unregistered"])
def test_simulated_pid_reuse_after_closed_quarantine(store, database, tmp_path, monkeypatch, registered):
    """Simulated PID reuse after owner close does not inherit an old quarantine's PID."""
    facts = {"case": "closed-quarantine", "simulated_pid_reuse": True, "registered": registered, "validated": []}
    owners, entries = [], {}
    cleanup_error = OSError("factory left its own connection open")

    @contextmanager
    def failed_exit():
        conninfo = make_conninfo(database, passfile="/dev/null", sslmode="disable", gssencmode="disable")
        connection = psycopg.connect(conninfo, autocommit=True)
        owner = PostgresStore(connection)
        owners.append(owner)
        yield owner
        raise cleanup_error

    with PostgresStore.standalone(database) as observer:
        try:
            with pytest.raises(OSError) as caught, ownership.checkout(store, factory=failed_exit):
                pass
            assert caught.value is cleanup_error
            old = owners[0]
            backend = backend_key(old)
            with ownership._lock:
                entries = {key: ownership._active[key] for key in old._ownership_keys}
            with pytest.raises(ownership.OwnershipError), ownership.checkout(store, factory=lambda: nullcontext(alias(old))):
                pytest.fail("live quarantined raw was leased again")
            facts["live_quarantine_alias_rejected"] = True
            old.connection.close()
            ownership.forget_host(old)
            confirm_backend_exit(observer, backend, old, facts)
            simulate_pid_reuse(monkeypatch, store, backend, facts, excluded=(old, observer))
            observe_validation(monkeypatch, facts)
            accept_fresh(store, independent_factory(store, database, registered), old, facts)
            assert facts["validated"] and all(v["target_equal"] for v in facts["validated"])
        finally:
            # Isolate only our deliberate quarantine, after the acceptance result.
            for old in owners:
                old.connection.close()
                ownership.forget_host(old)
            with ownership._lock:
                for key, entry in entries.items():
                    if ownership._active.get(key) is entry:
                        del ownership._active[key]
            save(tmp_path, facts)


@pytest.mark.parametrize("victim_kind", ["source", "other"])
def test_simulated_pid_reuse_after_server_exit_without_client_io(store, database, tmp_path, monkeypatch, victim_kind):
    """Simulated PID reuse after server exit; cached client closed=False is not liveness."""
    facts = {"case": "server-exit-without-client-io", "simulated_pid_reuse": True, "victim": victim_kind, "validated": []}
    with PostgresStore.standalone(database) as observer:
        context = nullcontext(store) if victim_kind == "source" else PostgresStore.standalone(database)
        with context as victim:
            try:
                backend = backend_key(victim)
                assert observer._one("SELECT pg_terminate_backend(%s, 5000)", (backend[-1],))[0]
                confirm_backend_exit(observer, backend, victim, facts)
                assert not victim.connection.closed
                simulate_pid_reuse(monkeypatch, store, backend, facts, excluded=(victim, observer))
                observe_validation(monkeypatch, facts)
                accept_fresh(store, store.execution_store, victim, facts)
                assert len(facts["validated"]) >= 2 and all(v["target_equal"] for v in facts["validated"])
            finally:
                victim.connection.close()
                save(tmp_path, facts)


def test_quarantine_strongly_pins_raw_after_wrapper_is_discarded(tmp_path, monkeypatch):
    """Real SQLite checkout writes the pin; only weak observations escape the failed factory."""
    path = str(tmp_path / "quarantine.sqlite")
    source = SQLiteStore(path)
    source.install_schema()
    original_connect = sqlite3.connect

    class ObservableConnection(sqlite3.Connection):
        def __del__(self):
            self.close()

    def connect(*args, **kwargs):
        return original_connect(*args, factory=ObservableConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)

    def quarantine():
        witness = {}

        @contextmanager
        def failed_exit():
            candidate = SQLiteStore(path)
            witness["wrapper"] = ref(candidate)
            witness["raw"] = ref(candidate.connection)
            yield candidate
            raise OSError("factory failed without releasing its own connection")

        context = ownership.checkout(source, factory=failed_exit)
        try:
            with context as candidate:
                key = ownership.native_keys(candidate)[0]
                witness["key"] = key
                with ownership._lock:
                    witness["token"] = ownership._active[key][0]
                assert candidate._one("SELECT 1")[0] == 1
        except OSError as error:
            error.__traceback__ = error.__context__ = error.__cause__ = None
        else:
            pytest.fail("factory exit did not fail")
        return witness

    witness = quarantine()
    key, token = witness["key"], witness["token"]
    raw = None
    try:
        assert witness["wrapper"]() is None
        with ownership._lock:
            entry = ownership._active[key]
            assert entry[0] is token
            assert entry[1] is not None, "production quarantine lost its strong raw pin"
            assert entry[1] is witness["raw"]()
            raw = entry[1]
        borrower = alias(source)
        borrower.connection = raw
        with pytest.raises(ownership.OwnershipError), ownership.checkout(source, factory=lambda: nullcontext(borrower)):
            pytest.fail("production quarantine lent its raw again")
    finally:
        if raw is None:
            raw = witness["raw"]()
        if raw is not None:
            raw.close()
        with ownership._lock:
            entry = ownership._active.get(key)
            if entry is not None and entry[0] is token:
                del ownership._active[key]
        source.connection.close()
        ownership.forget_host(source)


def test_normal_checkout_removes_its_raw_reservation(store):
    with ownership.checkout(store) as candidate:
        keys = ownership.native_keys(candidate)
        raw = candidate.connection.pgconn
        with ownership._lock:
            assert all(ownership._active[key][1] is raw for key in keys)
    with ownership._lock:
        assert all(key not in ownership._active for key in keys)


def test_late_old_token_cleanup_keeps_new_reservation(store):
    new_token = object()
    keys = ()
    try:
        with ownership.checkout(store) as candidate:
            keys = ownership.native_keys(candidate)
            raw = candidate.connection.pgconn
            with ownership._lock:
                assert all(ownership._active[key][0] is not new_token for key in keys)
                for key in keys:
                    ownership._active[key] = (new_token, raw)
        with ownership._lock:
            assert all(ownership._active[key][0] is new_token and ownership._active[key][1] is raw for key in keys)
    finally:
        with ownership._lock:
            for key in keys:
                entry = ownership._active.get(key)
                if entry is not None and entry[0] is new_token:
                    del ownership._active[key]


def test_ownership_identity_and_reservation_do_not_read_backend_identity():
    tree = ast.parse(inspect.getsource(ownership))
    names = {"_native_identity", "native_keys", "register_host", "checkout"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    forbidden = {"pgbackend", "backend_pid", "info", "host", "port", "dbname"}
    for function in functions:
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, (function.name, node.attr)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in forbidden, (function.name, node.value)


def cleanup_owned_quarantine(connection, entries):
    if connection is not None:
        connection.close()
    with ownership._lock:
        for key, token in entries.items():
            entry = ownership._active.get(key)
            if entry is not None and entry[0] is token:
                del ownership._active[key]


def test_default_pg_factory_closes_residual_transaction_and_releases_lock(store, database, tmp_path):
    """The SDK's own factory closes even when a leftover transaction makes commit fail."""
    connection, entries = None, {}
    facts = {}
    with PostgresStore.standalone(database) as observer:
        try:
            with pytest.raises(psycopg.ProgrammingError) as caught, ownership.checkout(store) as execution:
                connection = execution.connection
                pending = connection.transaction()
                pending.__enter__()
                execution._one("SELECT pg_advisory_xact_lock(803001)")
                with ownership._lock:
                    entries = {key: ownership._active[key][0] for key in ownership.native_keys(execution)}
            facts["primary_error"] = type(caught.value).__name__
            facts["closed"] = connection.closed
            facts["lock_available"] = observer._one("SELECT pg_try_advisory_xact_lock(803001)")[0]
            with ownership._lock:
                facts["active_removed"] = all(key not in ownership._active for key in entries)
            assert connection.closed
            assert facts["lock_available"] and facts["active_removed"]
        finally:
            cleanup_owned_quarantine(connection, entries)
            (tmp_path / "default-pg-close.json").write_text(json.dumps(facts, indent=2))
            print("DEFAULT-PG-CLOSE", json.dumps(facts), flush=True)


def test_default_pg_factory_alias_rejection_does_not_close_source(store, monkeypatch):
    original = PostgresStore._initialize
    validations = []

    def initialize(candidate, connection, **kwargs):
        original(candidate, connection, **kwargs)
        if kwargs.get("source") is store:
            candidate.connection = store.connection

    monkeypatch.setattr(PostgresStore, "_initialize", initialize)
    monkeypatch.setattr(PostgresStore, "_validate_checkout", lambda *args: validations.append(args))
    with pytest.raises(ownership.OwnershipError) as caught, ownership.checkout(store):
        pytest.fail("source alias accepted")
    assert caught.value.stage == "factory-alias"
    assert validations == [] and not store.connection.closed
    assert store._one("SELECT 1")[0] == 1


def test_default_pg_close_failure_keeps_original_error_and_quarantine(store, monkeypatch):
    original_close = psycopg.Connection.close
    owned, entries = [], {}
    close_error = OSError("owned connection close failed")

    def close(connection):
        assert not ownership._lock.locked()
        if owned and connection is owned[0]:
            raise close_error
        return original_close(connection)

    monkeypatch.setattr(psycopg.Connection, "close", close)
    try:
        with pytest.raises(psycopg.ProgrammingError) as caught, ownership.checkout(store) as execution:
            connection = execution.connection
            owned.append(connection)
            pending = connection.transaction()
            pending.__enter__()
            with ownership._lock:
                entries = {key: ownership._active[key][0] for key in ownership.native_keys(execution)}
        assert "checkout close failed: OSError" in caught.value.__notes__
        assert not connection.closed
        with ownership._lock:
            assert all(
                ownership._active[key][0] is token and ownership._active[key][1] is connection.pgconn
                for key, token in entries.items()
            )
            assert all(ownership._hosts[key][0]() is execution for key in entries)
        assert not store.connection.closed and store._one("SELECT 1")[0] == 1
    finally:
        monkeypatch.setattr(psycopg.Connection, "close", original_close)
        cleanup_owned_quarantine(owned[0] if owned else None, entries)


def test_default_pg_factory_closes_after_rollback_baseexception(store, monkeypatch):
    class Interrupted(BaseException):
        pass

    original_rollback = psycopg.Connection.rollback
    owned, entries = [], {}
    primary = ValueError("original execution error")

    def rollback(connection):
        if owned and connection is owned[0]:
            raise Interrupted("rollback interrupted")
        return original_rollback(connection)

    monkeypatch.setattr(psycopg.Connection, "rollback", rollback)
    try:
        with pytest.raises(ValueError) as caught, ownership.checkout(store) as execution:
            connection = execution.connection
            owned.append(connection)
            with ownership._lock:
                entries = {key: ownership._active[key][0] for key in ownership.native_keys(execution)}
            raise primary
        assert caught.value is primary
        assert "checkout exit failed: Interrupted" in caught.value.__notes__
        assert connection.closed
        with ownership._lock:
            assert all(key not in ownership._active for key in entries)
    finally:
        cleanup_owned_quarantine(owned[0] if owned else None, entries)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError], ids=["interrupt", "exception"])
def test_default_pg_initialization_failure_preserves_primary(store, monkeypatch, tmp_path, error_type):
    """A connected but unbound candidate must not replace its fixed initialization error."""
    primary = error_type("SQLStore initialization interrupted")
    connect = psycopg.connect
    connections, candidates, facts = [], [], {}

    def recording_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def fail_initialization(candidate):
        candidates.append(candidate)
        assert not hasattr(candidate, "connection")
        raise primary

    monkeypatch.setattr(psycopg, "connect", recording_connect)
    monkeypatch.setattr(SQLStore, "__init__", fail_initialization)
    try:
        try:
            with pytest.raises(error_type) as caught:
                try:
                    with ownership.checkout(store):
                        pytest.fail("partly initialized store was yielded")
                except BaseException as observed:
                    facts["top_error"] = type(observed).__name__
                    facts["same_primary"] = observed is primary
                    raise
            assert caught.value is primary
        finally:
            facts["owned_count"] = len(connections)
            facts["closed"] = len(connections) == 1 and connections[0].closed
            facts["source_open"] = not store.connection.closed
            facts["source_select"] = store._one("SELECT 1")[0]
            keys = [("pgraw", id(c.pgconn)) for c in connections]
            with ownership._lock:
                facts["active_absent"] = all(key not in ownership._active for key in keys)
                facts["host_absent"] = all(
                    host[0]() is not candidate for candidate in candidates for host in ownership._hosts.values()
                )
            (tmp_path / "pg-initialize-primary.json").write_text(json.dumps(facts, indent=2))
            print("PG-INITIALIZE-PRIMARY", json.dumps(facts), flush=True)
            assert len(connections) == len(candidates) == 1
            assert facts["closed"] and facts["source_open"] and facts["source_select"] == 1
            assert facts["active_absent"] and facts["host_absent"]
    finally:
        for connection in connections:
            if not connection.closed:
                connection.close()


@pytest.mark.parametrize("point", ["initialize", "register", "validate"])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, GeneratorExit], ids=["exception", "interrupt", "generator_exit"]
)
@pytest.mark.parametrize("close_fault", [False, True], ids=["close_ok", "close_fault"])
def test_default_pg_entry_preserves_primary_without_psycopg_exit(store, monkeypatch, tmp_path, point, error_type, close_fault):
    """Entry errors keep identity; the factory makes one close attempt, without retry."""
    primary = error_type("fixed PG entry failure")
    secondary = OSError("fixed first close failure")
    real_connect, real_close = psycopg.connect, psycopg.Connection.close
    initialize, register = PostgresStore._initialize, ownership.register_host
    validate = PostgresStore._validate_checkout
    connections, candidates, close_calls, observed = [], [], [], []

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def initialize_candidate(candidate, connection, **kwargs):
        candidates.append(candidate)
        if point == "initialize":
            assert not hasattr(candidate, "connection")
            raise primary
        return initialize(candidate, connection, **kwargs)

    def register_candidate(candidate, **kwargs):
        result = register(candidate, **kwargs)
        if point == "register":
            raise primary
        return result

    def validate_candidate(candidate, source):
        if point == "validate":
            raise primary
        return validate(candidate, source)

    def close(connection):
        assert not ownership._lock.locked()
        if any(connection is owned for owned in connections):
            close_calls.append(connection)
            if close_fault and len(close_calls) == 1:
                raise secondary
        return real_close(connection)

    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setattr(PostgresStore, "_initialize", initialize_candidate)
    monkeypatch.setattr(ownership, "register_host", register_candidate)
    monkeypatch.setattr(PostgresStore, "_validate_checkout", validate_candidate)
    monkeypatch.setattr(psycopg.Connection, "close", close)
    try:
        try:
            with ownership.checkout(store):
                pytest.fail("entry failure yielded a store")
        except BaseException as error:
            observed.append(error)
        keys = [("pgraw", id(connection.pgconn)) for connection in connections]
        with ownership._lock:
            active_absent = all(key not in ownership._active for key in keys)
            host_absent = all(host[0]() is not candidate for candidate in candidates for host in ownership._hosts.values())
        facts = {
            "point": point,
            "error_type": error_type.__name__,
            "close_fault": close_fault,
            "same_primary": len(observed) == 1 and observed[0] is primary,
            "top_type": type(observed[0]).__name__ if observed else None,
            "notes": list(getattr(primary, "__notes__", ())),
            "owned_count": len(connections),
            "close_calls": len(close_calls),
            "closed": len(connections) == 1 and connections[0].closed,
            "source_open": not store.connection.closed,
            "source_select": store._one("SELECT 1")[0],
            "active_absent": active_absent,
            "host_absent": host_absent,
        }
        (tmp_path / "pg-entry-primary.json").write_text(json.dumps(facts, indent=2))
        print("PG-ENTRY-PRIMARY", json.dumps(facts), flush=True)
        assert len(connections) == len(candidates) == 1
        assert facts["same_primary"]
        assert facts["close_calls"] == 1
        if close_fault:
            assert "checkout close failed: OSError" in facts["notes"]
        else:
            assert facts["closed"] and facts["notes"] == []
        assert facts["source_open"] and facts["source_select"] == 1
        assert facts["active_absent"] and facts["host_absent"]
    finally:
        # Test-owned cleanup uses the real close, outside the measured factory attempt.
        import sys

        failure = sys.exc_info()[1]
        try:
            for connection in connections:
                real_close(connection)
            for candidate in candidates:
                ownership.forget_host(candidate)
        except BaseException as cleanup:
            if failure is None:
                raise
            failure.add_note(f"test cleanup failed: {type(cleanup).__name__}")


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, GeneratorExit], ids=["exception", "interrupt", "generator_exit"]
)
@pytest.mark.parametrize("close_mode", ["first", "always"])
def test_default_pg_body_error_preserves_exit_quarantine(store, monkeypatch, tmp_path, error_type, close_mode):
    """Ownership keeps primary identity and independently tracks an unconfirmed release."""
    import sys

    primary = error_type("fixed ownership body error")
    cleanup_error = OSError("fixed owned connection close error")
    real_close = psycopg.Connection.close
    connection = execution = raw = None
    entries, calls, observed = {}, [], []

    def close(candidate):
        assert not ownership._lock.locked()
        if candidate is connection:
            calls.append(candidate)
            if close_mode == "always" or len(calls) == 1:
                raise cleanup_error
        return real_close(candidate)

    monkeypatch.setattr(psycopg.Connection, "close", close)
    try:
        try:
            with ownership.checkout(store) as execution:
                connection = execution.connection
                raw = connection.pgconn
                with ownership._lock:
                    entries = {key: ownership._active[key][0] for key in ownership.native_keys(execution)}
                    assert all(ownership._active[key][1] is raw for key in entries)
                raise primary
        except BaseException as error:
            observed.append(error)
        with ownership._lock:
            retained_before = all(
                ownership._active.get(key) is not None and ownership._active[key][0] is token and ownership._active[key][1] is raw
                for key, token in entries.items()
            )
            active_absent = all(key not in ownership._active for key in entries)
            host_retained = all(
                ownership._hosts.get(key) is not None and ownership._hosts[key][0]() is execution for key in entries
            )
        reborrow_rejected, reborrow_accepted, reborrow_select = False, False, None
        if close_mode == "always":
            try:
                with ownership.checkout(store, factory=lambda: nullcontext(execution)) as candidate:
                    reborrow_accepted = True
                    assert candidate.connection.pgconn is raw
                    reborrow_select = candidate._one("SELECT 1")[0]
            except ownership.OwnershipError as rejected:
                assert rejected.stage == "execution"
                reborrow_rejected = True
        with ownership._lock:
            retained_after = all(
                ownership._active.get(key) is not None and ownership._active[key][0] is token and ownership._active[key][1] is raw
                for key, token in entries.items()
            )
        facts = {
            "error_type": error_type.__name__,
            "close_mode": close_mode,
            "same_primary": len(observed) == 1 and observed[0] is primary,
            "top_type": type(observed[0]).__name__ if observed else None,
            "primary_notes": list(getattr(primary, "__notes__", ())),
            "cleanup_notes": list(getattr(cleanup_error, "__notes__", ())),
            "close_calls": len(calls),
            "closed": connection is not None and connection.closed,
            "active_retained_before": retained_before,
            "active_retained_after": retained_after,
            "active_absent": active_absent,
            "host_retained": host_retained,
            "reborrow_rejected": reborrow_rejected,
            "reborrow_accepted": reborrow_accepted,
            "reborrow_select": reborrow_select,
            "source_open": not store.connection.closed,
            "source_select": store._one("SELECT 1")[0],
            "reservation_count": len(entries),
        }
        (tmp_path / "pg-owned-exit-quarantine.json").write_text(json.dumps(facts, indent=2))
        print("PG-OWNED-EXIT-QUARANTINE", json.dumps(facts), flush=True)
        assert facts["same_primary"] and facts["close_calls"] == 2
        assert facts["reservation_count"] == 1 and facts["source_open"] and facts["source_select"] == 1
        assert "checkout exit failed: OSError" in facts["primary_notes"]
        if close_mode == "always":
            assert not facts["closed"]
            assert retained_before and retained_after and host_retained
            assert reborrow_rejected and not reborrow_accepted
            # The finally close note belongs to the cleanup error tracked by ownership.
            assert "checkout close failed: OSError" in facts["cleanup_notes"]
        else:
            assert facts["closed"] and active_absent and not host_retained
    finally:
        failure = sys.exc_info()[1]
        try:
            if connection is not None:
                real_close(connection)
            with ownership._lock:
                for key, token in entries.items():
                    entry = ownership._active.get(key)
                    if entry is not None and entry[0] is token:
                        del ownership._active[key]
            if execution is not None:
                ownership.forget_host(execution)
        except BaseException as cleanup:
            if failure is None:
                raise
            failure.add_note(f"test cleanup failed: {type(cleanup).__name__}")


def test_default_pg_interrupted_catalog_transaction_ends_with_statement(store, database, monkeypatch, tmp_path):
    """An unclosed autocommit query can retain a transaction only while in flight."""
    import sys
    import time

    from psycopg import waiting

    primary = RuntimeError("fixed interruption during catalog lock wait")
    secondary = OSError("fixed entry close failure")
    real_connect, real_close = psycopg.connect, psycopg.Connection.close
    real_execute, real_wait = psycopg.Cursor.execute, waiting.wait
    initialize = PostgresStore._initialize
    owned, candidates, calls, observed, hits = [], [], [], [], []
    armed = False
    lock_query = "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND relation=1259 AND NOT granted)"
    state_query = "SELECT state, xact_start IS NULL, wait_event_type FROM pg_stat_activity WHERE pid=%s"
    with real_connect(database, autocommit=True) as observer, real_connect(database, autocommit=True) as blocker:
        observer.execute(lock_query, (0,)).fetchone()
        observer.execute(state_query, (0,)).fetchone()
        blocker.execute("BEGIN")
        blocker_transaction = True

        def connect(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            owned.append(connection)
            return connection

        def initialize_candidate(candidate, connection, **kwargs):
            result = initialize(candidate, connection, **kwargs)
            candidates.append(candidate)
            return result

        def execute(cursor, query, *args, **kwargs):
            nonlocal armed
            if owned and cursor.connection is owned[0] and "unnest(ARRAY" in str(query):
                blocker.execute("LOCK TABLE pg_catalog.pg_class IN ACCESS EXCLUSIVE MODE")
                armed = True
            return real_execute(cursor, query, *args, **kwargs)

        def wait(generator, fileno, *args, **kwargs):
            nonlocal armed
            if armed and owned and fileno == owned[0].pgconn.socket:
                armed = False
                next(generator)  # Send the SDK's original query before interrupting the wait.
                assert owned[0].info.transaction_status.name == "ACTIVE"
                backend = owned[0].info.backend_pid
                deadline = time.monotonic() + 5
                while not observer.execute(lock_query, (backend,)).fetchone()[0]:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                hits.append({"backend": backend, "query_sent": True, "catalog_lock_wait": True})
                raise primary
            return real_wait(generator, fileno, *args, **kwargs)

        def close(connection):
            assert not ownership._lock.locked()
            if any(connection is item for item in owned):
                calls.append(connection)
                if len(calls) == 1:
                    raise secondary
            return real_close(connection)

        monkeypatch.setattr(psycopg, "connect", connect)
        monkeypatch.setattr(PostgresStore, "_initialize", initialize_candidate)
        monkeypatch.setattr(psycopg.Cursor, "execute", execute)
        monkeypatch.setattr(waiting, "wait", wait)
        monkeypatch.setattr(psycopg.Connection, "close", close)
        facts = {}
        try:
            try:
                with ownership.checkout(store):
                    pytest.fail("interrupted entry yielded")
            except BaseException as error:
                observed.append(error)
            backend = owned[0].info.backend_pid
            before = observer.execute(state_query, (backend,)).fetchone()
            keys = [("pgraw", id(connection.pgconn)) for connection in owned]
            with ownership._lock:
                active_absent = all(key not in ownership._active for key in keys)
                candidate_absent = all(
                    host[0]() is not candidate for candidate in candidates for host in ownership._hosts.values()
                )
            facts.update(
                {
                    "same_primary": len(observed) == 1 and observed[0] is primary,
                    "notes": list(getattr(primary, "__notes__", ())),
                    "close_calls": len(calls),
                    "closed": owned[0].closed,
                    "hits": hits,
                    "backend": backend,
                    "before_release": list(before) if before else None,
                    "active_absent": active_absent,
                    "candidate_absent": candidate_absent,
                    "source_open": not store.connection.closed,
                    "source_select": store._one("SELECT 1")[0],
                }
            )
            assert len(owned) == len(candidates) == len(hits) == 1
            assert facts["same_primary"] and "checkout close failed: OSError" in facts["notes"]
            assert facts["close_calls"] == 1 and not facts["closed"]
            assert active_absent and candidate_absent and facts["source_open"] and facts["source_select"] == 1
            assert before is not None and before[0] == "active" and before[1] is False and before[2] == "Lock"
            released = time.monotonic()
            blocker.execute("ROLLBACK")
            blocker_transaction = False
            samples = []
            deadline = released + 5
            while True:
                row = observer.execute(state_query, (backend,)).fetchone()
                now = time.monotonic()
                samples.append({"elapsed_seconds": now - released, "backend_state": list(row) if row else None})
                if row is not None and row[1] is True:
                    break
                assert now < deadline, samples
                time.sleep(0.01)
            facts.update(after_release_samples=samples, transaction_ended=True, observation_limit_seconds=5)
            assert not owned[0].closed
        finally:
            failure = sys.exc_info()[1]
            (tmp_path / "pg-inflight-statement.json").write_text(json.dumps(facts, indent=2))
            print("PG-INFLIGHT-STATEMENT", json.dumps(facts), flush=True)
            cleanup_errors = []
            if blocker_transaction:
                try:
                    blocker.execute("ROLLBACK")
                except BaseException as cleanup:
                    cleanup_errors.append(cleanup)
            for connection in owned:
                try:
                    real_close(connection)
                except BaseException as cleanup:
                    cleanup_errors.append(cleanup)
            for candidate in candidates:
                ownership.forget_host(candidate)
            if cleanup_errors:
                if failure is None:
                    raise cleanup_errors[0]
                for cleanup in cleanup_errors:
                    failure.add_note(f"test cleanup failed: {type(cleanup).__name__}")
