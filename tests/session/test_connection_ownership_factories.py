"""Factories fail at the real stage before acquire, timeout SQL, callbacks or dispatch."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Event

import pytest

from vv_agent.session.kernel import drive
from vv_agent.session.ownership import OwnershipError, checkout
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .conftest import open_store
from .test_recovery_matrix import runtime, start


def alias(store):
    wrapper = object.__new__(type(store))
    wrapper.__dict__.update(store.__dict__)
    return wrapper


@pytest.mark.parametrize("damage", ["missing", "enter", "alias"])
def test_execution_stage(store, database, monkeypatch, damage):
    start(store)
    counts = {"acquire": 0, "build": 0, "hook": 0, "dispatch": 0}
    real = type(store).acquire

    def acquire(*a, **kw):
        counts["acquire"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(type(store), "acquire", acquire)
    error = OSError("factory enter")

    @contextmanager
    def failed():
        raise error
        yield

    def build(sid):
        counts["build"] += 1
        return runtime(database, [])

    # memory C is intentionally borrowed, not an invalid persistent alias.
    if damage == "alias" and getattr(store, "path", None) == ":memory:":
        candidate = SQLiteStore(":memory:")
        candidate.install_schema()
    else:
        candidate = alias(store)
    monkeypatch.setattr(
        store, "execution_store", None if damage == "missing" else failed if damage == "enter" else lambda: nullcontext(candidate)
    )
    expected = OSError if damage == "enter" else OwnershipError
    try:
        with pytest.raises(expected) as caught:
            drive(store, "s", runtime=build)
        if damage == "enter":
            assert caught.value is error
        else:
            assert caught.value.code == "connection_ownership"
        assert counts == {"acquire": 0, "build": 0, "hook": 0, "dispatch": 0}
        assert store._one("SELECT 1")[0] == 1
    finally:
        if damage == "alias" and getattr(store, "path", None) == ":memory:":
            candidate.connection.close()


def ready_wait_interval(wait, timeout):
    """Sample inside the actual Python wait call, excluding all drive/SQL cleanup."""
    import sys
    import time

    code = getattr(wait, "__func__", wait).__code__
    interval = {}
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if frame.f_code is code and event in {"call", "return"}:
            # Latest entry / earliest return gives an interval inside this call.
            first = time.time_ns()
            monotonic = time.monotonic_ns()
            raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            last = time.time_ns()
            interval[event] = {
                "realtime_first_ns": first,
                "monotonic_ns": monotonic,
                "monotonic_raw_ns": raw,
                "realtime_last_ns": last,
            }

    sys.setprofile(observe)
    try:
        result = wait(timeout)
    finally:
        sys.setprofile(previous)
    assert set(interval) == {"call", "return"}
    interval["target"] = {"filename": code.co_filename, "function": code.co_name}
    interval["realtime_resolution_ns"] = time.get_clock_info("time").resolution * 1_000_000_000
    return result, interval


@pytest.mark.persistent_store
@pytest.mark.parametrize("stage", ["bootstrap", "runtime_direct", "runtime_factory"])
def test_heartbeat_alias_stage(store, database, monkeypatch, stage):
    start(store)
    counts = {"acquire": 0, "build": 0, "release": 0, "host_sql": 0, "callbacks": 0, "dispatch": 0}
    cls = type(store)
    acquire = cls.acquire
    release = cls.release
    rows = cls._rows

    def counted_acquire(*a, **kw):
        counts["acquire"] += 1
        return acquire(*a, **kw)

    def counted_release(*a, **kw):
        counts["release"] += 1
        return release(*a, **kw)

    def counted_rows(candidate, *a, **kw):
        if candidate.connection is store.connection:
            counts["host_sql"] += 1
        return rows(candidate, *a, **kw)

    def hook(*args, tx):
        counts["callbacks"] += 1

    def model(request):
        counts["dispatch"] += 1
        return LLMResponse("done")

    rt = runtime(database, [model], hook=hook)
    rt.heartbeat_store = lambda: nullcontext(alias(store))
    if stage == "bootstrap":
        monkeypatch.setattr(cls, "heartbeat_store", lambda _: nullcontext(alias(store)))
    monkeypatch.setattr(cls, "acquire", counted_acquire)
    monkeypatch.setattr(cls, "release", counted_release)
    monkeypatch.setattr(cls, "_rows", counted_rows)

    def build(sid):
        counts["build"] += 1
        return rt

    with pytest.raises(OwnershipError) as caught:
        drive(store, "s", runtime=rt if stage == "runtime_direct" else build)
    assert caught.value.code == "connection_ownership"
    assert counts["acquire"] == counts["release"] == 1
    assert counts["build"] == int(stage == "runtime_factory")
    assert counts["callbacks"] == counts["dispatch"] == counts["host_sql"] == 0
    print("CO2-F02/F03", type(store).__name__, stage, counts, flush=True)


@pytest.mark.persistent_store
def test_duplicate_checkout_reserved_before_validation(store, monkeypatch):
    entered = Event()
    release = Event()
    validations = []
    holder = []
    factory = store.execution_store
    original = type(store)._validate_checkout

    def gate(candidate, source):
        if not holder:
            holder.append(candidate)
            validations.append(1)
            entered.set()
            assert release.wait(5)
        return original(candidate, source)

    @contextmanager
    def shared():
        if holder:
            yield alias(holder[0])
        else:
            with factory() as candidate:
                yield candidate

    monkeypatch.setattr(type(store), "_validate_checkout", gate)
    monkeypatch.setattr(store, "execution_store", shared)

    def first():
        with checkout(store):
            return 1

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(first)
        assert entered.wait(5)
        try:
            with pytest.raises(OwnershipError, match="connection_ownership"), checkout(store):
                pass
            assert validations == [1]
        finally:
            release.set()
        assert future.result(5) == 1


def test_setup_only_and_borrow(tmp_path):
    with SQLiteStore.standalone(str(tmp_path / "setup.db")) as source:
        with pytest.raises(OwnershipError, match="setup-only"):
            source.execution_store()
        source.install_schema()
        with checkout(source) as execution:
            assert execution.connection is not source.connection
    with SQLiteStore.standalone(":memory:") as source:
        source.install_schema()
        with checkout(source) as execution:
            assert execution is source
        assert source._one("SELECT 1")[0] == 1


@pytest.mark.postgres
def test_postgres_schema_frozen_and_quoted(store, database):
    from psycopg import sql

    from vv_agent.session.postgres import DDL

    name = 'tenant " quoted'
    store.connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    store.connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(name)))
    store.connection.execute(DDL)
    source = PostgresStore(store.connection)
    # The new wrapper is the authoritative H after idle initialization.
    target = source._target
    store.connection.execute("SET search_path TO public")
    with checkout(source) as execution:
        assert execution._target == target
        assert execution._one("SELECT current_schema()")[0] == name
    with store.connection.transaction():
        from vv_agent.session.postgres import join_transaction

        tx = join_transaction(store.connection)
        assert tx.store._target is None
        with pytest.raises(OwnershipError, match="setup-only"):
            tx.store.execution_store()


@pytest.mark.persistent_store
@pytest.mark.parametrize(
    "damage", ["transaction", "non_autocommit", "closed", "wrong_target", "temp_shadow", "isolation_or_namespace"]
)
def test_candidate_state_before_acquire(store, database, tmp_path, monkeypatch, damage):
    start(store)
    acquires = []
    real = type(store).acquire
    factory = store.execution_store

    def acquire(*a, **kw):
        acquires.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(type(store), "acquire", acquire)

    @contextmanager
    def damaged():
        with factory() as candidate:
            if damage == "wrong_target":
                if isinstance(candidate, SQLiteStore):
                    with SQLiteStore.standalone(str(tmp_path / "other.db")) as other:
                        other.install_schema()
                        yield other
                    return
                from vv_agent.session.postgres import DDL

                candidate.connection.execute("CREATE SCHEMA other")
                candidate.connection.execute("SET search_path TO other")
                candidate.connection.execute(DDL)
            elif damage == "temp_shadow":
                candidate.connection.execute("CREATE TEMP TABLE sk_record (shadow integer)")
            elif damage == "isolation_or_namespace":
                if isinstance(candidate, SQLiteStore):
                    candidate.connection.execute("PRAGMA read_uncommitted=1")
                else:
                    candidate.connection.execute("SET search_path TO pg_catalog")
            elif damage == "non_autocommit":
                if isinstance(candidate, SQLiteStore):
                    candidate.connection.isolation_level = ""
                else:
                    candidate.connection.autocommit = False
            elif damage == "closed":
                candidate.connection.close()
            if damage == "transaction":
                with candidate.atomic():
                    yield candidate
            else:
                yield candidate

    monkeypatch.setattr(store, "execution_store", damaged)
    with pytest.raises(OwnershipError):
        drive(store, "s", runtime=runtime(database, []))
    assert acquires == []


@pytest.mark.postgres
def test_native_caller_execution_rejected(store, database, monkeypatch):
    start(store)
    called = []
    monkeypatch.setattr(store, "execution_store", lambda: called.append(1))
    with store.connection.transaction(), pytest.raises(OwnershipError, match="caller-transaction"):
        drive(store, "s", runtime=runtime(database, []))
    assert called == []


@pytest.mark.parametrize("scheduled", [False, True])
def test_phase_and_native_identity_witness(store, database, monkeypatch, scheduled):
    import json
    from threading import current_thread, get_ident

    start(store)
    events = []
    cls = type(store)
    validate_checkout = cls._validate_checkout
    acquire = cls.acquire
    renew = cls.renew

    def identity(candidate):
        return {
            "store": id(candidate),
            "raw": id(candidate.connection),
            "thread": get_ident(),
            "name": current_thread().name,
            "backend_pid": candidate.connection.info.backend_pid if isinstance(candidate, PostgresStore) else None,
        }

    def validated(candidate, source):
        result = validate_checkout(candidate, source)
        events.append({"phase": "validated", **identity(candidate)})
        return result

    def acquired(candidate, *a, **kw):
        events.append({"phase": "acquire", **identity(candidate)})
        return acquire(candidate, *a, **kw)

    def renewed(candidate, *a, **kw):
        events.append({"phase": "renew", **identity(candidate)})
        return renew(candidate, *a, **kw)

    monkeypatch.setattr(cls, "_validate_checkout", validated)
    monkeypatch.setattr(cls, "acquire", acquired)
    monkeypatch.setattr(cls, "renew", renewed)

    def hook(point, record, *, tx):
        if point == "before_commit":
            events.append({"phase": "callback", **identity(tx.store)})

    def model(request):
        events.append({"phase": "model"})
        return LLMResponse("done")

    rt = runtime(database, [model], hook=hook)

    def build(sid):
        events.append({"phase": "build"})
        return rt

    assert drive(store, "s", runtime=build if scheduled else rt)
    assert sum(e["phase"] == "build" for e in events) == int(scheduled)
    assert sum(e["phase"] == "model" for e in events) == 1
    execution = next(e["raw"] for e in events if e["phase"] == "acquire")
    if getattr(store, "path", None) != ":memory:":
        assert execution != id(store.connection)
        for index, event in enumerate(events):
            if event["phase"] == "renew":
                if event["name"] == "session-heartbeat":
                    assert event["raw"] not in {execution, id(store.connection)}
                assert any(e["phase"] == "validated" and e["raw"] == event["raw"] for e in events[:index])
    assert all(e["raw"] == execution for e in events if e["phase"] == "callback")
    before = len(events)
    with checkout(store, role="projection") as projection:
        projection.read_state("s")
    assert not any(e["phase"] in {"acquire", "renew", "model"} for e in events[before:])
    print(
        "CO2-F04-NATIVE-WITNESS",
        json.dumps(
            {
                "backend": type(store).__name__,
                "memory": getattr(store, "path", None) == ":memory:",
                "scheduled": scheduled,
                "H": identity(store),
                "events": events,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@pytest.mark.parametrize("stage", ["bootstrap", "runtime_direct", "runtime_factory"])
def test_first_poll_failure_is_not_successful_ready(store, database, monkeypatch, stage):
    from threading import current_thread

    from vv_agent.session import kernel
    from vv_agent.session.store import LeaseLost

    start(store)
    error = LeaseLost("first renewal rejected")
    heartbeats = []
    counts = {"build": 0, "load": 0, "hook": 0, "model": 0, "release": 0}
    initialize = kernel._LeaseHeartbeat.__init__
    renew = type(store).renew
    release = type(store).release

    def captured(heartbeat, *a, **kw):
        initialize(heartbeat, *a, **kw)
        heartbeats.append(heartbeat)

    def failed(candidate, *a, **kw):
        if current_thread().name == "session-heartbeat" and (stage != "runtime_factory" or counts["build"]):
            raise error
        return renew(candidate, *a, **kw)

    def released(candidate, *a, **kw):
        counts["release"] += 1
        return release(candidate, *a, **kw)

    def load(driver):
        counts["load"] += 1
        raise AssertionError("unready Driver loaded")

    def hook(*a, tx):
        counts["hook"] += 1

    def model(request):
        counts["model"] += 1
        return LLMResponse("unused")

    rt = runtime(database, [model], hook=hook)

    def build(sid):
        counts["build"] += 1
        return rt

    monkeypatch.setattr(kernel._LeaseHeartbeat, "__init__", captured)
    monkeypatch.setattr(type(store), "renew", failed)
    monkeypatch.setattr(type(store), "release", released)
    monkeypatch.setattr(kernel._Driver, "load_state", load)
    with pytest.raises(LeaseLost) as caught:
        drive(store, "s", runtime=rt if stage == "runtime_direct" else build)
    assert caught.value is error
    assert counts == {"build": int(stage == "runtime_factory"), "load": 0, "hook": 0, "model": 0, "release": 1}
    assert heartbeats[-1].ready.is_set() and not heartbeats[-1].successful_ready
    assert all(not heartbeat.thread.is_alive() for heartbeat in heartbeats)
    print("CO2-F02/F03-FIRST-POLL", stage, type(store).__name__, counts, flush=True)


@pytest.mark.postgres
def test_wrong_database_with_valid_same_named_tables(store, database, monkeypatch):
    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = "co2_other_" + uuid4().hex
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:

            @contextmanager
            def wrong():
                with PostgresStore.standalone(make_conninfo(database, dbname=name)) as candidate:
                    candidate.install_schema()
                    yield candidate

            acquired = []
            monkeypatch.setattr(store, "execution_store", wrong)
            monkeypatch.setattr(PostgresStore, "acquire", lambda *a, **kw: acquired.append(1))
            with pytest.raises(OwnershipError, match="target"):
                drive(store, "s", runtime=runtime(database, []))
            assert acquired == []
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


@pytest.mark.persistent_store
@pytest.mark.parametrize("stage", ["bootstrap", "runtime_direct", "runtime_factory"])
@pytest.mark.parametrize("damage", ["transaction", "non_autocommit", "closed", "wrong_target"])
def test_heartbeat_state_before_timeout_and_renew(store, database, tmp_path, monkeypatch, stage, damage):

    start(store)
    factory = store.execution_store
    bad = []
    sdk_config = []
    bad_renew = []
    counts = {"build": 0, "callback": 0, "model": 0, "release": 0}
    cls = type(store)
    rows = cls._rows
    renew = cls.renew
    release = cls.release

    @contextmanager
    def faulty():
        with factory() as candidate:
            if damage == "wrong_target":
                if isinstance(candidate, SQLiteStore):
                    with SQLiteStore.standalone(str(tmp_path / "wrong-hb.db")) as wrong:
                        wrong.install_schema()
                        bad.append(wrong.connection)
                        yield wrong
                    return
                from vv_agent.session.postgres import DDL

                candidate.connection.execute("CREATE SCHEMA bad_hb")
                candidate.connection.execute("SET search_path TO bad_hb")
                candidate.connection.execute(DDL)
            elif damage == "non_autocommit":
                if isinstance(candidate, SQLiteStore):
                    candidate.connection.isolation_level = ""
                else:
                    candidate.connection.autocommit = False
            elif damage == "closed":
                candidate.connection.close()
            bad.append(candidate.connection)
            if damage == "transaction":
                with candidate.atomic():
                    yield candidate
            else:
                yield candidate

    def counted_rows(candidate, query, *a, **kw):
        if candidate.connection in bad and "set_config('lock_timeout','250ms'" in query:
            sdk_config.append(1)
        return rows(candidate, query, *a, **kw)

    def counted_renew(candidate, *a, **kw):
        if candidate.connection in bad:
            bad_renew.append(1)
        return renew(candidate, *a, **kw)

    def counted_release(candidate, *a, **kw):
        counts["release"] += 1
        return release(candidate, *a, **kw)

    def hook(*a, tx):
        counts["callback"] += 1

    def model(request):
        counts["model"] += 1
        return LLMResponse("unused")

    rt = runtime(database, [model], hook=hook)
    rt.heartbeat_store = faulty

    def build(sid):
        counts["build"] += 1
        return rt

    if stage == "bootstrap":
        monkeypatch.setattr(cls, "heartbeat_store", lambda _: faulty())
    monkeypatch.setattr(cls, "_rows", counted_rows)
    monkeypatch.setattr(cls, "renew", counted_renew)
    monkeypatch.setattr(cls, "release", counted_release)
    with pytest.raises(OwnershipError):
        drive(store, "s", runtime=rt if stage == "runtime_direct" else build)
    assert counts == {"build": int(stage == "runtime_factory"), "callback": 0, "model": 0, "release": 1}
    assert len(bad) == 1 and sdk_config == bad_renew == []
    print("CO2-F02/F03-STATE", type(store).__name__, stage, damage, counts, "bad_timeout_sql", 0, "bad_renew", 0, flush=True)


@pytest.mark.postgres
def test_setup_install_native_outer_requires_reconstruction(database):
    with PostgresStore.standalone(database) as source:
        assert source._target is None
        with pytest.raises(OwnershipError, match="setup-only"):
            source.execution_store()
        with source.connection.transaction():
            source.install_schema()
            assert source._target is None
        assert source._target is None
        rebuilt = PostgresStore(source.connection)
        with rebuilt.execution_store() as execution:
            assert execution._target == rebuilt._target
            assert execution.connection is not source.connection


@pytest.mark.postgres
def test_failed_rewrap_does_not_forget_existing_host(store):
    from vv_agent.session.ownership import _hosts, native_keys

    keys = native_keys(store)
    store.connection.execute("CREATE TEMP TABLE sk_record (shadow integer)")
    with pytest.raises(OwnershipError):
        PostgresStore(store.connection)
    assert all(_hosts[key][0]() is store for key in keys)
    store.connection.execute("DROP TABLE pg_temp.sk_record")
    assert store._one("SELECT 1")[0] == 1


@pytest.mark.persistent_store
@pytest.mark.parametrize("wrap", [False, True])
def test_other_known_host_is_not_a_checkout(store, database, monkeypatch, wrap):
    with open_store(database) as other_host:
        candidate = alias(other_host) if wrap else other_host
        monkeypatch.setattr(store, "execution_store", lambda: nullcontext(candidate))
        with pytest.raises(OwnershipError), checkout(store):
            pass
        assert other_host._one("SELECT 1")[0] == 1


@pytest.mark.persistent_store
def test_known_host_rewrap_during_factory_is_rejected_before_sql(store, database, monkeypatch):
    if isinstance(store, SQLiteStore):
        # The public SQLite constructor creates a new raw handle. The alias is
        # represented by the actual wrapper a custom pool returned.
        candidate = alias(store)
        monkeypatch.setattr(store, "execution_store", lambda: nullcontext(candidate))
    else:

        @contextmanager
        def invalid():
            yield PostgresStore(store.connection)

        monkeypatch.setattr(store, "execution_store", invalid)
    sql = []
    original = type(store)._rows

    def counted(candidate, *a, **kw):
        if candidate.connection is store.connection:
            sql.append(a[0])
        return original(candidate, *a, **kw)

    monkeypatch.setattr(type(store), "_rows", counted)
    with pytest.raises(OwnershipError), checkout(store):
        pass
    assert sql == []


@pytest.mark.parametrize("stage", ["bootstrap", "runtime_direct", "runtime_factory"])
@pytest.mark.parametrize("fault", ["open", "ready_timeout"])
def test_open_and_real_ready_timeout_wait_interval(store, database, monkeypatch, stage, fault):
    from time import monotonic

    from vv_agent.session import kernel
    from vv_agent.session.store import LeaseLost

    start(store)
    unblock = Event()
    heartbeats = []
    waits = []
    counts = {"build": 0, "load": 0, "callback": 0, "model": 0, "release": 0}
    original_init = kernel._LeaseHeartbeat.__init__
    original_close = kernel._LeaseHeartbeat.close
    release = type(store).release
    factory = store.execution_store
    error = OSError("heartbeat factory refused")

    def capture(heartbeat, *a, **kw):
        original_init(heartbeat, *a, **kw)
        heartbeat._co2_blocked = stage != "runtime_factory" or bool(heartbeats)
        heartbeats.append(heartbeat)
        original_wait = heartbeat.ready.wait

        def observed_wait(timeout):
            result, interval = ready_wait_interval(original_wait, timeout)
            waits.append((timeout, result, interval))
            return result

        heartbeat.ready.wait = observed_wait

    @contextmanager
    def blocked():
        if fault == "open":
            raise error
        with factory() as candidate:
            assert unblock.wait(kernel.FACTORY_LEASE_TTL_MS / 1000 + kernel.HEARTBEAT_CLOSE_SECONDS)
            yield candidate

    def close(heartbeat):
        # This is reached only after the real 15-second ready wait expires.
        # Stop first, then let the factory leave, exactly as bounded shutdown requires.
        heartbeat.stop.set()
        unblock.set()
        return original_close(heartbeat)

    def released(candidate, *a, **kw):
        counts["release"] += 1
        return release(candidate, *a, **kw)

    def load(driver):
        counts["load"] += 1
        raise AssertionError("unready Driver loaded")

    def hook(*a, tx):
        counts["callback"] += 1

    def model(request):
        counts["model"] += 1
        return LLMResponse("unused")

    rt = runtime(database, [model], hook=hook)
    rt.heartbeat_store = blocked

    def build(sid):
        counts["build"] += 1
        return rt

    if stage == "bootstrap":
        monkeypatch.setattr(type(store), "heartbeat_store", lambda _: blocked())
    monkeypatch.setattr(kernel._LeaseHeartbeat, "__init__", capture)

    # Bootstrap's successful close must not release the runtime timeout gate early.
    def closing(heartbeat):
        is_blocked = heartbeat._co2_blocked
        return close(heartbeat) if is_blocked else original_close(heartbeat)

    monkeypatch.setattr(kernel._LeaseHeartbeat, "close", closing)
    monkeypatch.setattr(type(store), "release", released)
    monkeypatch.setattr(kernel._Driver, "load_state", load)
    begin = monotonic()
    try:
        with pytest.raises(OSError if fault == "open" else LeaseLost) as caught:
            drive(store, "s", runtime=rt if stage == "runtime_direct" else build)
        print(
            "READY_DIAGNOSTIC",
            stage,
            fault,
            "runtime_ttl",
            rt.ttl_ms,
            "waits",
            waits,
            "error",
            type(caught.value).__name__,
            str(caught.value),
            "wall",
            monotonic() - begin,
            flush=True,
        )
        if fault == "open":
            assert caught.value is error
        else:
            assert rt.ttl_ms == kernel.FACTORY_LEASE_TTL_MS == 15000
            assert waits[-1][0] == 15.0 and waits[-1][1] is False
        assert counts == {"build": int(stage == "runtime_factory"), "load": 0, "callback": 0, "model": 0, "release": 1}
        assert not heartbeats[-1].successful_ready
        assert all(not heartbeat.thread.is_alive() for heartbeat in heartbeats)
        print(
            "CO2-F02/F03-READY-WAIT-INTERVAL",
            type(store).__name__,
            stage,
            fault,
            counts,
            "wall_seconds",
            monotonic() - begin,
            flush=True,
        )
        if fault == "ready_timeout":
            interval = waits[-1][2]
            assert interval["return"]["realtime_first_ns"] - interval["call"]["realtime_last_ns"] >= 15_000_000_000, (
                "ready.wait interval shorter than 15s",
                interval,
            )
    finally:
        unblock.set()


@pytest.mark.postgres
def test_temporary_input_wrapper_does_not_forget_live_host(store):
    import gc

    from vv_agent.session.ownership import _hosts, native_keys

    keys = native_keys(store)
    with store.connection.transaction():
        temporary = PostgresStore(store.connection)
        assert temporary._target is None
        del temporary
        gc.collect()
        assert all(_hosts[key][0]() is store for key in keys)
    assert store._one("SELECT 1")[0] == 1


@pytest.mark.parametrize("invalid", [None, object()])
def test_invalid_factory_context_is_ownership_error_before_acquire(store, database, monkeypatch, invalid):
    acquired = []
    monkeypatch.setattr(store, "execution_store", lambda: invalid)
    monkeypatch.setattr(type(store), "acquire", lambda *a, **kw: acquired.append(1))
    with pytest.raises(OwnershipError, match="factory-context"):
        drive(store, "s", runtime=runtime(database, []))
    assert acquired == []


@pytest.mark.parametrize("value", [None, object()])
def test_factory_yield_requires_a_store_before_acquire(store, database, monkeypatch, value):
    acquired = []
    monkeypatch.setattr(store, "execution_store", lambda: nullcontext(value))
    monkeypatch.setattr(type(store), "acquire", lambda *a, **kw: acquired.append(1))
    with pytest.raises(OwnershipError, match="factory-store"):
        drive(store, "s", runtime=runtime(database, []))
    assert acquired == []
