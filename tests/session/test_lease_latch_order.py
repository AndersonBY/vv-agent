"""Lease latch ordering: real stores, event schedules, and bounded child processes."""

import json
import subprocess
import sys
import threading
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.session import kernel
from vv_agent.session.children import ChildSession
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.providers import Accepted, Unknown
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Lease, LeaseLost, LeasePoll, SessionStore, SessionTx
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .test_children import parent_runtime
from .test_recovery_matrix import effect_tool, runtime, start


class PlainTx(SessionTx):
    def __init__(self, tx):
        self._tx = tx

    def lock_session(self, *args, **kwargs):
        return self._tx.lock_session(*args, **kwargs)

    def create(self, *args, **kwargs):
        return self._tx.create(*args, **kwargs)

    def push(self, *args, **kwargs):
        return self._tx.push(*args, **kwargs)

    def append(self, *args, **kwargs):
        return self._tx.append(*args, **kwargs)


class Opaque:
    def __init__(self, store, *, plain=False):
        self._store, self._plain = store, plain

    def __getattr__(self, name):
        return getattr(self._store, name)

    @contextmanager
    def atomic(self):
        with self._store.atomic() as tx:
            yield PlainTx(tx) if self._plain else tx


class WaitingProvider:
    def __init__(self):
        self.submits = 0
        self.queries = 0
        self.handle = None

    def submit(self, plan, *, context):
        self.submits += 1
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "l1",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "l1-receipt",
            "query_ref": "l1",
            "cancel_ref": None,
        }
        return Accepted(self.handle)

    def query(self, handle):
        self.queries += 1
        assert handle == self.handle
        return Accepted(handle)

    def cancel(self, handle):
        return Unknown("not cancelled")

    def authenticate(self, item, plan):
        return False


def _child(params):
    """A failing old memory schedule cannot keep the pytest process alive."""
    result = {"ok": False, "case": params["case"]}
    unhandled = []

    def unraisable(args):
        unhandled.append(type(args.exc_value).__name__)
        print("L1_UNHANDLED " + type(args.exc_value).__name__, flush=True)

    def thread_error(args):
        unhandled.append(type(args.exc_value).__name__)
        print("L1_UNHANDLED " + type(args.exc_value).__name__, flush=True)

    sys.unraisablehook = unraisable
    threading.excepthook = thread_error
    try:
        with pytest.MonkeyPatch.context() as patch:
            if params["case"] == "guard":
                _guard_case(params["kind"])
            else:
                value = params["database"]
                database = value if value == ":memory:" or params.get("postgres") else Path(value)
                if params.get("postgres"):
                    _record_pg_connects(database, patch)
                context = SQLiteStore.standalone(":memory:") if value == ":memory:" else open_store(database)
                with context as host:
                    host.install_schema()
                    start(host)
                    if params["case"] == "admission":
                        result["facts"] = _admission_case(host, host if value == ":memory:" else database, params, patch)
                    else:
                        result["facts"] = _reverse_case(host, host if value == ":memory:" else database, params["gate"], patch)
        result["ok"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    if unhandled:
        result.update(ok=False, unhandled=unhandled)
    print("L1_RESULT " + json.dumps(result), flush=True)


def _record_pg_connects(dsn, patch):
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    expected = conninfo_to_dict(dsn)
    original = psycopg.Connection.connect.__func__

    def connect(cls, conninfo="", **kwargs):
        values = conninfo_to_dict(conninfo)
        assert values["host"] == expected["host"] and values["port"] == expected["port"]
        kwargs.update(passfile="/dev/null", sslmode="disable", gssencmode="disable")
        connection = original(cls, conninfo, **kwargs)
        print(
            "L1_CONNECT "
            + json.dumps(
                {
                    "dsn": make_conninfo(
                        "",
                        **(
                            {k: v for k, v in values.items() if k != "password"}
                            | {k: kwargs[k] for k in ("passfile", "sslmode", "gssencmode")}
                        ),
                    ),
                    "thread": threading.get_ident(),
                    "raw": id(connection.pgconn),
                }
            ),
            flush=True,
        )
        return connection

    patch.setattr(psycopg.Connection, "connect", classmethod(connect))
    patch.setattr(psycopg, "connect", psycopg.Connection.connect)


def _run_child(tmp_path, **params):
    source = Path(__file__).resolve().parents[2]
    childenv = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": ":".join(map(str, (source / "src", source / "tests", source))),
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    code = "import importlib,json,sys;importlib.import_module(sys.argv[1])._child(json.loads(sys.argv[2]))"
    argv = [sys.executable, "-B", "-W", "error", "-c", code, __name__, json.dumps(params)]
    with subprocess.Popen(
        argv, cwd=source, env=childenv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as child:
        try:
            fields = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
            identity = {"pid": child.pid, "start_ticks": fields[19]}
        except FileNotFoundError:
            identity = None
        timed_out = False
        try:
            output, _ = child.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            output, _ = child.communicate(timeout=5)
        print(
            "L1_CHILD_RECEIPT "
            + json.dumps(
                {
                    "argv": argv,
                    "cwd": str(source),
                    "env": childenv,
                    "identity": identity,
                    "actual_exit": child.returncode,
                    "wall_limit_seconds": 25,
                    "timed_out": timed_out,
                }
            ),
            flush=True,
        )
        print(output, end="")
        if timed_out:
            pytest.fail("lease latch child did not finish within 25s; old lock cycle isolated\n" + output)
        results = [json.loads(line.split(" ", 1)[1]) for line in output.splitlines() if line.startswith("L1_RESULT ")]
        assert "L1_UNHANDLED " not in output, output
        assert not any(
            line.startswith(("Exception ignored", "Exception in thread", "Error in sys.unraisablehook"))
            for line in output.splitlines()
        ), output
        assert child.returncode == 0 and len(results) == 1, output
        assert results[0]["ok"], results[0]


def _admission_case(host, database, params, patch):
    entered = threading.Event()
    armed = threading.Event()
    state: dict[str, Any] = {"scope": None, "C_free": None, "calls": 0, "renewed": 0, "raws": [], "execution_raw": None}
    initialize = kernel._Scope.__init__
    backend = type(host)
    renew = backend.renew

    def init(scope, *args, **kwargs):
        initialize(scope, *args, **kwargs)
        state["scope"] = scope

    def controlled(store, lease, *, ttl_ms):
        if armed.is_set() and threading.current_thread().name == "session-heartbeat":
            free = state["scope"].commit_lock.acquire(blocking=False)
            state["C_free"] = free
            if free:
                state["scope"].commit_lock.release()
            entered.set()
        result = renew(store, lease, ttl_ms=ttl_ms)
        if armed.is_set():
            state["renewed"] += 1
        return result

    @contextmanager
    def heartbeat():
        with host.heartbeat_store() as store:
            state["raws"].append(id(store.connection))
            yield Opaque(store)

    def admit(plan):
        state["calls"] += 1
        state["execution_raw"] = id(host.connection)
        with host.atomic() as tx:
            if params.get("postgres"):
                tx.lock_session("s")
            tx.create(SessionSpec("business", "test", "/tmp"), consumers=())
        armed.set()
        assert entered.wait(5), "HB never entered renew while admission held its writer"
        assert state["C_free"], "heartbeat held commit latch across renew"
        return ChildSession(SessionSpec("l1-child", "test", "/tmp"), "go")

    patch.setattr(kernel._Scope, "__init__", init)
    patch.setattr(backend, "renew", controlled)
    rt = parent_runtime(database)
    rt.heartbeat_store = heartbeat
    rt.children["child_task"] = admit
    source = host if params.get("native") else cast(SessionStore, Opaque(host, plain=True))
    assert kernel.drive(source, "s", runtime=rt)
    scope = cast(kernel._Scope, state["scope"])
    assert not scope.lost and not scope.thread.is_alive()
    assert state["calls"] == 1 and state["renewed"] >= 1
    assert host.list_sessions() == ("business", "l1-child", "s")
    assert host._rows("SELECT count(*) FROM sk_session WHERE session_id='business'")[0][0] == 1
    assert (state["raws"][0] == state["execution_raw"]) == (params["database"] == ":memory:")
    return {
        "admission_calls": state["calls"],
        "renewed": state["renewed"],
        "commit_free_in_renew": state["C_free"],
        "execution_raw": state["execution_raw"],
        "heartbeat_raws": state["raws"],
        "drive_thread": threading.get_ident(),
        "heartbeat_thread": scope.thread.ident,
        "heartbeat_lost": scope.lost,
        "heartbeat_alive": scope.thread.is_alive(),
    }


def _reverse_case(host, database, gate, patch):
    armed, writer, at_c = threading.Event(), threading.Event(), threading.Event()
    state: dict[str, Any] = {"scope": None, "gated": 0, "renewed": 0, "store_gate": 0}
    initialize, committing = kernel._Scope.__init__, kernel._LeaseHeartbeat.committing
    transaction, renew = SQLiteStore._transaction, SQLiteStore.renew

    def init(scope, *args, **kwargs):
        initialize(scope, *args, **kwargs)
        state["scope"] = scope

    @contextmanager
    def tx(store):
        with transaction(store):
            if armed.is_set() and not writer.is_set() and threading.current_thread().name == "session-heartbeat":
                writer.set()
                assert at_c.wait(5), "execution did not reach its commit latch"
            yield

    @contextmanager
    def latch(scope):
        frame = sys._getframe(1)
        while frame and frame.f_code.co_filename != kernel.__file__:
            frame = frame.f_back
        target = "step" if gate == "provider" else "drive"
        selected = frame is not None and frame.f_code.co_name == target and not armed.is_set()
        if selected:
            armed.set()
            assert writer.wait(5), "native HB did not acquire the real writer"
        with committing(scope) as lease:
            if selected:
                state["gated"] += 1
                at_c.set()
            yield lease

    def renewed(store, value, *, ttl_ms):
        result = renew(store, value, ttl_ms=ttl_ms)
        if armed.is_set():
            state["renewed"] += 1
        return result

    original_gate = getattr(SQLiteStore, "next_drive_delay_ms" if gate == "provider" else "defer_idle_drive")

    def store_gate(store, *args, **kwargs):
        if armed.is_set() and threading.current_thread().name != "session-heartbeat":
            state["store_gate"] += 1
        return original_gate(store, *args, **kwargs)

    provider = WaitingProvider()
    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("l1", "effect", {})])] if gate == "provider" else [],
        [effect_tool()] if gate == "provider" else (),
        providers={"effect": provider} if gate == "provider" else {},
    )
    rt.heartbeat_store = host.heartbeat_store
    if gate == "provider":
        # A new driver queries an already parked provider; submission alone bypasses this gate.
        assert kernel.drive(cast(SessionStore, Opaque(host)), "s", runtime=rt)
        assert provider.submits == 1, ("provider bootstrap", provider.submits)
    patch.setattr(kernel._Scope, "__init__", init)
    patch.setattr(kernel._LeaseHeartbeat, "committing", latch)
    patch.setattr(SQLiteStore, "_transaction", tx)
    patch.setattr(SQLiteStore, "renew", renewed)
    patch.setattr(SQLiteStore, "next_drive_delay_ms" if gate == "provider" else "defer_idle_drive", store_gate)
    if gate == "idle":
        patch.setattr(kernel._Driver, "step", lambda driver: False)
    assert kernel.drive(cast(SessionStore, Opaque(host)), "s", runtime=rt)
    assert state["gated"] == 1 and state["store_gate"] >= 1 and state["renewed"] >= 1, state
    assert not state["scope"].lost and not state["scope"].thread.is_alive()
    if gate == "provider":
        assert provider.submits == 1
    return {
        "gate": gate,
        "gate_entries": state["gated"],
        "store_gate_calls": state["store_gate"],
        "renewed": state["renewed"],
        "heartbeat_lost": state["scope"].lost,
        "heartbeat_alive": state["scope"].thread.is_alive(),
    }


@pytest.mark.parametrize("layout", ["opaque_sqlite", "native_sqlite", "opaque_memory"])
def test_admission_writer_does_not_wait_for_heartbeat_latch(tmp_path, layout):
    database = ":memory:" if layout == "opaque_memory" else str(tmp_path / "session.sqlite")
    _run_child(tmp_path, case="admission", database=database, native=layout == "native_sqlite")


@pytest.mark.postgres
def test_pg_admission_parent_lock_does_not_wait_for_heartbeat_latch(database, tmp_path):
    _run_child(tmp_path, case="admission", database=database, postgres=True)
    with open_store(database) as observer:
        assert observer.list_sessions() == ("business", "l1-child", "s")


@pytest.mark.parametrize("gate", ["provider", "idle"])
@pytest.mark.parametrize("memory", [False, True])
def test_sqlite_reverse_writer_order_finishes(tmp_path, gate, memory):
    database = ":memory:" if memory else str(tmp_path / "session.sqlite")
    _run_child(tmp_path, case="reverse", database=database, gate=gate)


def test_private_memory_renew_boundary_holds_no_commit_or_state_latch():
    """Private boundary; does not represent public drive support for this pairing."""
    entered, finish = threading.Event(), threading.Event()
    errors, poll_threads = [], []
    with SQLiteStore.standalone(":memory:") as memory:
        memory.install_schema()
        start(memory)
        lease = memory.acquire("s", owner="private-boundary", ttl_ms=15000)
        assert lease is not None
        heartbeat = kernel._LeaseHeartbeat(
            lease, lambda: nullcontext(cast(SessionStore, PollStore())), 15000, 0.25, now_ms=memory._now()
        )

        class Controlled:
            def renew(self, value, *, ttl_ms):
                with memory._transaction():
                    poll_threads.append(threading.get_ident())
                    entered.set()
                    assert finish.wait(5)
                    result = memory.renew(value, ttl_ms=ttl_ms)
                return result

        def poll():
            try:
                heartbeat.poll(cast(SessionStore, Controlled()))
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=poll, daemon=True)
        worker.start()
        try:
            assert entered.wait(5)
            assert poll_threads == [worker.ident] and worker.ident != threading.get_ident()
            commit_free = heartbeat.commit_lock.acquire(blocking=False)
            if commit_free:
                heartbeat.commit_lock.release()
            state_free = heartbeat.lock.acquire(blocking=False)
            if state_free:
                heartbeat.lock.release()
            assert commit_free, "private boundary poll held the commit latch across renew"
            assert state_free, "private boundary poll held the state lock across renew"
        finally:
            finish.set()
            worker.join(3)
        assert not worker.is_alive() and not errors and not heartbeat.lost


def _heartbeat(factory=None):
    lease = Lease("s", "owner", 1, 15000)
    return kernel._LeaseHeartbeat(lease, factory or (lambda: nullcontext(cast(SessionStore, PollStore()))), 15000, 0.25, now_ms=0)


class PollStore:
    @contextmanager
    def atomic(self):
        raise AssertionError("unit poll should not open atomic")
        yield

    def read_state(self, *args, **kwargs):
        raise AssertionError("unit poll should not read state")

    def acquire(self, *args, **kwargs):
        raise AssertionError("unit poll should not acquire")

    def release(self, *args, **kwargs):
        raise AssertionError("unit poll should not release")

    def renew(self, lease, *, ttl_ms):
        now = lease.expires_at_ms - ttl_ms + 1
        return LeasePoll(replace(lease, expires_at_ms=now + ttl_ms), now, 0, ())


def _guard_case(kind):
    heartbeat = _heartbeat()
    if kind == "commit":
        with heartbeat.committing(), pytest.raises(RuntimeError, match="poll while holding commit/poll latch"):
            heartbeat.poll(PollStore())
    else:

        class Recursive:
            def renew(self, lease, *, ttl_ms):
                return heartbeat.poll(self)

        with pytest.raises(RuntimeError, match="poll while holding commit/poll latch"):
            heartbeat.poll(Recursive())
    heartbeat.poll(PollStore())


@pytest.mark.parametrize("kind", ["commit", "poll"])
def test_poll_reentry_fails_before_waiting(tmp_path, kind):
    _run_child(tmp_path, case="guard", kind=kind)


@pytest.mark.postgres
def test_renew_publishes_while_commit_latch_is_held(store, database):
    start(store)
    lease = store.acquire("s", owner="owner", ttl_ms=15000)
    heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, 15000, 0.25, now_ms=store._now())
    try:
        with heartbeat.committing() as captured:
            heartbeat.thread.start()
            assert heartbeat.ready.wait(5), "renew/publication waited for commit latch"
            assert heartbeat.successful_ready and heartbeat.lease is not captured and captured is lease
    finally:
        if heartbeat.thread.ident is not None:
            heartbeat.close()
    assert not heartbeat.lost and not heartbeat.thread.is_alive()


def test_poll_keeps_renewal_and_snapshot_serial(monkeypatch):
    heartbeat = _heartbeat()
    first, finish, second_attempt = threading.Event(), threading.Event(), threading.Event()
    incoming, outputs, errors = [], {}, []
    active = maximum = 0
    actual = heartbeat.poll_lock

    class SerialLock:
        def __enter__(self):
            if threading.current_thread().name == "second-poll":
                second_attempt.set()
            actual.acquire()

        def __exit__(self, *_):
            actual.release()

    monkeypatch.setattr(heartbeat, "poll_lock", SerialLock())

    class Store:
        def renew(self, value, *, ttl_ms):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            incoming.append(value)
            if len(incoming) == 1:
                first.set()
                assert finish.wait(5)
            result = PollStore().renew(value, ttl_ms=ttl_ms)
            active -= 1
            return result

    def poll(which):
        try:
            outputs[which] = heartbeat.poll(Store()).lease
        except BaseException as error:
            errors.append(error)

    a = threading.Thread(target=poll, args=("first",), daemon=True)
    b = threading.Thread(target=poll, args=("second",), name="second-poll", daemon=True)
    a.start()
    try:
        assert first.wait(5)
        b.start()
        assert second_attempt.wait(5), "poll bypassed its serialization latch"
        assert maximum == 1
    finally:
        finish.set()
        a.join(3)
        if b.ident is not None:
            b.join(3)
    assert not a.is_alive() and not b.is_alive() and not errors
    assert maximum == 1 and len(incoming) == 2 and incoming[1] is outputs["first"]
    assert heartbeat.lease is outputs["second"]


def test_renew_does_not_hold_state_lock():
    heartbeat = _heartbeat()

    class Store:
        def renew(self, value, *, ttl_ms):
            assert heartbeat.lock.acquire(blocking=False), "state lock held across store call"
            heartbeat.lock.release()
            return PollStore().renew(value, ttl_ms=ttl_ms)

    heartbeat.poll(Store())


@pytest.mark.sqlite_file
def test_native_poll_does_not_open_outer_transaction(store, database, monkeypatch):
    start(store)
    lease = store.acquire("s", owner="owner", ttl_ms=15000)
    heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, 15000, 0.25, now_ms=store._now())
    renew = SQLiteStore.renew
    depths = []

    def observed(value, lease, *, ttl_ms):
        depths.append(value._depth)
        return renew(value, lease, ttl_ms=ttl_ms)

    monkeypatch.setattr(SQLiteStore, "renew", observed)
    with store.heartbeat_store() as other:
        heartbeat.poll(other)
    assert depths == [0], "kernel opened an extra writer transaction around renewal"


def test_stop_before_publication_never_marks_success_ready():
    entered, finish = threading.Event(), threading.Event()

    class Store(PollStore):
        def renew(self, lease, *, ttl_ms):
            result = PollStore().renew(lease, ttl_ms=ttl_ms)
            entered.set()
            assert finish.wait(5)
            return result

    @contextmanager
    def factory():
        yield Store()

    heartbeat = _heartbeat(factory)
    old = heartbeat.lease
    heartbeat.thread.start()
    try:
        assert entered.wait(5)
        assert heartbeat.lock.acquire(blocking=False), "state lock held across renew"
        heartbeat.stop.set()
        heartbeat.lock.release()
    finally:
        finish.set()
        heartbeat.thread.join(3)
    assert not heartbeat.thread.is_alive() and not heartbeat.lost
    assert heartbeat.ready.is_set() and not heartbeat.successful_ready and heartbeat.lease is old


def test_deadline_uses_call_start_and_identity_change_is_terminal(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(kernel, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    heartbeat = _heartbeat()

    class Store:
        def renew(self, lease, *, ttl_ms):
            clock[0] = 110.0
            return LeasePoll(replace(lease, expires_at_ms=20000), 5000, 0, ())

    heartbeat.poll(Store())
    assert heartbeat.expires_deadline == 115.0
    retained = heartbeat.lease

    class Changed:
        def renew(self, lease, *, ttl_ms):
            return LeasePoll(replace(lease, owner="other"), 5000, 0, ())

    with pytest.raises(LeaseLost, match="identity"):
        heartbeat.poll(Changed())
    assert heartbeat.lost and heartbeat.lease is retained


@pytest.mark.postgres
def test_captured_expired_lease_rolls_back_business_after_row_renewal(store, database, monkeypatch):
    start(store)
    clock = [store._now()]
    monkeypatch.setattr(PostgresStore, "_now", lambda self: clock[0])
    original = kernel._LeaseHeartbeat.committing
    admissions = []
    with nullcontext(store) as execution:
        lease = execution.acquire("s", owner="owner", ttl_ms=15000)
        scope = kernel._Scope(lease, runtime(database, []), now_ms=clock[0])
        driver = kernel._Driver(execution, "s", scope.runtime, scope)

        @contextmanager
        def captured(current):
            with original(current) as frozen:
                clock[0] = lease.expires_at_ms - 14000
                scope.thread.start()
                assert scope.ready.wait(5), "HB did not publish under the captured commit lease"
                assert scope.successful_ready and scope.lease is not frozen
                clock[0] = lease.expires_at_ms + 1
                assert scope.lease.expires_at_ms > clock[0] >= frozen.expires_at_ms
                yield frozen

        def business(tx):
            admissions.append(1)
            tx.create(SessionSpec("business", "test", "/tmp"), consumers=())
            return []

        monkeypatch.setattr(scope, "committing", lambda: captured(scope))
        try:
            with pytest.raises(LeaseLost, match=r"expired|fenced"):
                driver.commit([], prepare=business)
        finally:
            if scope.thread.ident is not None:
                scope.close()
            execution.release(scope.lease)
        assert not scope.lost and not scope.thread.is_alive()
    assert admissions == [1] and store.list_sessions() == ("s",)


@pytest.mark.sqlite_file
def test_native_provider_gate_does_not_open_outer_transaction(store, database, monkeypatch):
    start(store)
    provider = WaitingProvider()
    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("l1", "effect", {})])],
        [effect_tool()],
        providers={"effect": provider},
    )
    original = store.next_drive_delay_ms
    depths = []

    def observed(*args, **kwargs):
        depths.append(store._depth)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "next_drive_delay_ms", observed)
    assert kernel.drive(store, "s", runtime=rt)
    assert provider.submits == 1 and depths
    assert set(depths) == {0}, "kernel provider gate opened a writer transaction before taking C"
