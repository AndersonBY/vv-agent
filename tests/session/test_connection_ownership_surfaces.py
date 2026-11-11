"""CO2-S01/S02/S03: real non-AppServer producers and their actual connection owners."""

import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from threading import Event, Thread, get_ident

import pytest

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, RunConfig, Runner, ScriptedModelProvider
from vv_agent.model import ModelRef
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.session import kernel, ownership, supervisor, surfaces
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.records import InboxItem
from vv_agent.session.sql import SQLStore
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, SubAgentConfig, ToolCall

from .test_recovery_matrix import runtime, start


def raw(store):
    return getattr(store.connection, "pgconn", store.connection)


@pytest.fixture
def owners(store, monkeypatch, tmp_path):
    import vv_agent.run_handle as handles

    records, held, active, loads, maintenance = [], [], {}, [], []
    real_checkout, real_load, real_acquire = ownership.checkout, _Driver.__init__, SQLStore.acquire
    real_event_store = SessionRunEventStore.__init__
    h = raw(store)
    tracked = [store]
    ignored = []

    @contextmanager
    def checkout(source, *, factory=None, role="execution"):
        with real_checkout(source, factory=factory, role=role) as current:
            if not any(source is owner for owner in tracked):
                ignored.append({"role": role, "source": id(source), "raw": id(raw(current)), "thread": get_ident()})
                yield current
                return
            tracked.append(current)
            identity = raw(current)
            held.append(identity)
            key = get_ident(), id(identity)
            previous = active.get(key)
            active[key] = role
            records.append({"role": role, "raw": id(identity), "thread": get_ident()})
            if getattr(store, "path", None) != ":memory:":
                assert identity is not h
            try:
                yield current
            finally:
                if previous is None:
                    del active[key]
                else:
                    active[key] = previous

    def load(driver, *args, **kwargs):
        owner = args[0] if args else kwargs["store"]
        if not any(owner is current for current in tracked):
            return real_load(driver, *args, **kwargs)
        identity = raw(owner)
        assert active.get((get_ident(), id(identity))) == "execution"
        real_load(driver, *args, **kwargs)
        assert driver.scope.successful_ready and not driver.scope.lost
        loads.append({"sid": driver.sid, "raw": id(identity), "thread": get_ident()})

    def acquire(current, *args, **kwargs):
        if not any(current is owner for owner in tracked):
            return real_acquire(current, *args, **kwargs)
        role = active.get((get_ident(), id(raw(current))))
        assert role != "projection", "P cannot acquire an execution lease"
        if role == "maintenance":
            maintenance.append(("acquire", id(raw(current)), get_ident()))
        return real_acquire(current, *args, **kwargs)

    def event_store(current, owner, *args, **kwargs):
        if not any(owner is current_owner for current_owner in tracked):
            return real_event_store(current, owner, *args, **kwargs)
        assert active.get((get_ident(), id(raw(owner)))) == "projection", "publication must own P"
        return real_event_store(current, owner, *args, **kwargs)

    for name in ["read", "read_state", "peek_inbox", "is_runnable", "list_runnable", "lease_status"]:
        original = getattr(store, name)

        def guarded(*args, _name=name, _original=original, **kwargs):
            if getattr(store, "path", None) != ":memory:":
                assert store._atomic_stack, f"implicit H read: {_name}"
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, guarded)
    monkeypatch.setattr(ownership, "checkout", checkout)
    monkeypatch.setattr(kernel, "checkout", checkout)
    monkeypatch.setattr(handles, "checkout", checkout)
    monkeypatch.setattr(surfaces, "checkout", checkout, raising=False)
    monkeypatch.setattr(supervisor, "checkout", checkout, raising=False)
    monkeypatch.setattr(_Driver, "__init__", load)
    monkeypatch.setattr(SQLStore, "acquire", acquire)
    monkeypatch.setattr(SessionRunEventStore, "__init__", event_store)
    facts = {"records": records, "loads": loads, "maintenance": maintenance, "H": id(h), "ignored": ignored}
    yield facts
    (tmp_path / "surface-owners.json").write_text(json.dumps(facts, indent=2))
    print("CO2-SURFACE-OWNERS", json.dumps(facts), flush=True)


@pytest.mark.parametrize("entry", ["runner", "configured", "compiled", "resume", "interactive", "direct", "inline"])
def test_non_appserver_entry_owns_execution_and_projection(store, database, tmp_path, monkeypatch, entry, owners):
    """CO2-S01: run the actual entry, observe Driver E and consumer P, and retain outputs."""
    driver = SessionDriver(store=store)
    monkeypatch.setattr(surfaces, "SessionDriver", lambda **kwargs: driver)
    agent = Agent("ownership", "Answer precisely.", model="m")
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    config = RunConfig(workspace=tmp_path, model_provider=provider, session_memory_enabled=False)
    try:
        if entry in {"runner", "configured"}:
            handle = (
                Runner.start(agent, "go", run_config=config)
                if entry == "runner"
                else Runner.configured(config).start(agent, "go")
            )
            assert handle.result(10).final_output == "done"
        elif entry == "compiled":
            task = AgentCompiler().compile(
                agent=agent, input="go", run_config=config, resolved=provider.resolve(ModelRef.named("m")), trace_id="compiled"
            )
            handle = Runner._start_compiled(agent, "go", task=task, run_config=config)
            assert handle.result(10).final_output == "done"
        elif entry == "resume":
            provider = ScriptedModelProvider.new(
                "test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Color?"})]), LLMResponse("blue")]
            )
            handle = Runner.start(agent, "go", run_config=replace(config, model_provider=provider))
            parked = handle.result(10)
            wait = parked.metadata["session_waits"][0]
            driver.push(
                handle.session_id,
                InboxItem(
                    "reply",
                    "user",
                    {"content": {"interaction_id": wait["interaction_id"], "operation_id": wait["operation_id"], "text": "blue"}},
                    handle.run_id,
                    0,
                ),
            )
            assert Runner.resume(handle.session_id, handle.run_id).final_output == "blue"
        elif entry == "interactive":
            client = InteractiveAgentClient(
                options=AgentSessionOptions(model_provider=provider, workspace=tmp_path, session_store=store)
            )
            session = client.create_session(agent=agent, session_id="interactive")
            assert session.prompt("go").result.final_answer == "done"
            assert session.messages
        else:
            start(store)
            rt = runtime(database, [LLMResponse("done")])
            if entry == "direct":
                drive(store, "s", runtime=rt)
            else:
                supervisor.tick(store, runtime=lambda _sid: rt, project=lambda _sid, _consumer: None)
            from vv_agent.session.result import project_result

            with ownership.checkout(store, role="projection") as projection:
                state, _, _ = projection.read_state("s")
                assert project_result(projection, "s", next(reversed(state.turns))).final_output == "done"
        assert owners["loads"]
        executions = [r for r in owners["records"] if r["role"] == "execution"]
        assert executions
        if getattr(store, "path", None) == ":memory:":
            assert all(r["raw"] == owners["H"] for r in owners["records"])
        else:
            assert all(r["raw"] != owners["H"] for r in owners["records"])
            for e in executions:
                assert e["raw"] not in {r["raw"] for r in owners["records"] if r["role"] in {"projection", "heartbeat"}}
    finally:
        driver.close()


@pytest.mark.parametrize("background", [False, True])
def test_child_reconstruction_and_delivery_keep_explicit_owners(store, tmp_path, monkeypatch, owners, background):
    """CO2-S02: real configured child, default child hook and committed delivery."""
    driver = SessionDriver(store=store)
    monkeypatch.setattr(surfaces, "SessionDriver", lambda **kwargs: driver)
    args = {"agent_id": "worker", "task_description": "child", "wait_for_completion": not background}
    from support import ModelMapProvider

    from vv_agent.config import ResolvedModelConfig
    from vv_agent.llm.scripted import ScriptedLLM

    provider = ModelMapProvider(
        {
            name: (ScriptedLLM(steps), ResolvedModelConfig("test", name, name, name, []))
            for name, steps in [
                (
                    "parent",
                    [
                        LLMResponse("", [ToolCall("delegate", "create_sub_task", args)]),
                        LLMResponse("parent done"),
                        LLMResponse("after delivery"),
                    ],
                ),
                ("m", [LLMResponse("child done")]),
            ]
        },
        default_model="parent",
    )
    agent = Agent("parent", "Delegate.", model="parent", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")})
    original_child = surfaces.Runtime.child_runtime
    child_scopes = []

    def child_runtime(rt, current, sid):
        child = original_child(rt, current, sid)
        assert child.hook is not rt.hook
        assert not hasattr(child.hook, "_transaction_original")
        child_scopes.append({"sid": sid, "raw": id(raw(current)), "thread": get_ident()})
        return child

    monkeypatch.setattr(surfaces.Runtime, "child_runtime", child_runtime)
    import vv_agent.run_handle as handle_module

    original_delivery, original_wake = handle_module.child_delivery, driver._wake
    deliveries, wakes = [], []

    def delivery(current, tx, sid, *, hook):
        pushed = False

        def point(name):
            nonlocal pushed
            pushed |= name == "after_parent_inbox"
            hook(name)

        through = original_delivery(current, tx, sid, hook=point)
        if pushed:
            assert tx.connection is current.connection
            parent = current.read(sid, limit=1).records[0].record.payload["parent_session_id"]
            deliveries.append((current, tx, sid, parent))
        return through

    def wake(sid):
        for current, tx, child_id, parent in reversed(deliveries):
            if parent == sid:
                with current._connection_lock if isinstance(current, SQLiteStore) else nullcontext():
                    assert not tx._scope_active and not current._atomic_stack
                with ownership.checkout(store, role="projection") as projection, projection.atomic() as read_tx:
                    assert read_tx.consumer_batch(child_id, "child_delivery") is None
                wakes.append((sid, child_id, get_ident()))
                break
        return original_wake(sid)

    monkeypatch.setattr(handle_module, "child_delivery", delivery)
    monkeypatch.setattr(driver, "_wake", wake)
    try:
        handle = Runner.start(
            agent, "go", run_config=RunConfig(workspace=tmp_path, model_provider=provider, session_memory_enabled=False)
        )
        assert handle.result(10).final_output in {"parent done", "after delivery"}
        for worker in driver.handles:
            worker.join(10)
            assert worker.done() and worker._error is None
        assert child_scopes and deliveries and wakes
        roots = [r for r in owners["loads"] if r["sid"] == handle.session_id]
        children = [r for r in owners["loads"] if r["sid"] != handle.session_id]
        assert roots and children
        if background:
            assert all(r["thread"] != roots[0]["thread"] for r in children)
            if getattr(store, "path", None) != ":memory:":
                assert all(r["raw"] != roots[0]["raw"] for r in children)
        else:
            assert all((r["raw"], r["thread"]) == (roots[0]["raw"], roots[0]["thread"]) for r in children)
    finally:
        driver.close()


def test_dispatch_only_scan_has_no_execution_or_runtime_build(store, database, owners):
    """CO2-S03: real scan may own P; dispatch alone cannot build or acquire E."""
    start(store)
    built, dispatched = [], []

    def build(sid):
        built.append(sid)
        return runtime(database, [])

    assert supervisor.tick(store, project=lambda *_: None, dispatch=dispatched.append, runtime=build) >= 1
    assert dispatched == ["s"] and built == []
    assert owners["loads"] == [] and owners["maintenance"] == []
    assert {r["role"] for r in owners["records"]} == {"projection"}


@pytest.mark.parametrize("failure", ["project", "dispatch"])
def test_tick_errors_keep_maintenance_separate_from_projection(store, database, owners, failure):
    start(store)
    error = ValueError("fixed scheduling error")

    def fail(*args):
        raise error

    if failure == "project":
        drive(store, "s", runtime=runtime(database, [LLMResponse("done")]))
        # Count only this tick; the completed setup leaves durable projection work.
        for key in ["records", "loads", "maintenance"]:
            owners[key].clear()
        project = fail

        def dispatch(_sid):
            return None
    else:

        def project(*_args):
            return None

        dispatch = fail
    with pytest.raises(ExceptionGroup) as caught:
        supervisor.tick(store, project=project, dispatch=dispatch)
    assert any(e is error for e in caught.value.exceptions)
    assert not owners["loads"]
    assert not any(r["role"] == "execution" for r in owners["records"])
    if failure == "dispatch":
        assert len(owners["maintenance"]) == 1


def test_memory_projection_waits_for_outer_commit():
    """CO2-S01 memory retains one C and waits on S instead of seeing provisional input."""
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        entered, completed = Event(), Event()
        seen, errors = [], []

        def reader():
            entered.set()
            try:
                with ownership.checkout(store, role="projection") as projection:
                    seen.append((projection is store, len(projection.peek_inbox("s"))))
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        with store.atomic() as tx:
            from vv_agent.session.records import SessionSpec

            tx.create(SessionSpec("s", "test", "/tmp"), consumers=())
            tx.push("s", InboxItem("initial", "user", {"content": "go"}))
            thread = Thread(target=reader)
            thread.start()
            assert entered.wait(5)
            assert not completed.wait(0.05)
        thread.join(5)
        assert not thread.is_alive() and not errors
        assert seen == [(True, 1)]
        assert store.connection.execute("SELECT 1").fetchone() == (1,)


def test_result_trace_events_and_cancel_use_projection_owner(store, tmp_path, monkeypatch, owners):
    """CO2-S02: retained result/events/trace/cancel and targeted successor isolation."""
    from vv_agent.runtime import CancellationToken
    from vv_agent.session import tracing

    driver = SessionDriver(store=store)
    monkeypatch.setattr(surfaces, "SessionDriver", lambda **kwargs: driver)
    token = CancellationToken()
    projected = []
    real_deliver = tracing.deliver_spans

    class Processor:
        def on_span_start(self, span):
            projected.append(("start", span.span_id))

        def on_span_end(self, span):
            projected.append(("end", span.span_id))

        def flush(self):
            pass

    def trace(current, *args, **kwargs):
        assert owners["records"][-1]["role"] == "projection"
        assert owners["records"][-1]["raw"] == id(raw(current))
        return real_deliver(current, *args, **kwargs)

    monkeypatch.setattr(tracing, "deliver_spans", trace)
    provider = ScriptedModelProvider.new(
        "test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Color?"})]), LLMResponse("successor")]
    )
    try:
        handle = Runner.start(
            Agent("parent", "Ask.", model="m"),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=provider,
                cancellation_token=token,
                session_memory_enabled=False,
                tracing={"processors": [Processor()]},
            ),
        )
        parked = handle.result(10)
        assert parked.metadata["session_waits"] and list(handle.events()) and projected
        assert handle._cancel_bound
        assert handle.cancel()
        cancelled = Runner.resume(handle.session_id, handle.run_id)
        assert cancelled.completion_reason.value == "cancelled"
        assert not handle.cancel()
    finally:
        driver.close()


def test_control_marker_waits_for_outer_commit_and_drops_rollback(store, database, monkeypatch):
    """CO2-S02: an inner admission cannot publish its marker before the owner commits."""
    start(store)
    drive(store, "s", runtime=runtime(database, [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Color?"})])]))
    driver = SessionDriver(store=store)
    marks = []
    monkeypatch.setattr(driver, "_wake", lambda sid: marks.append((sid, bool(store._atomic_stack))))
    with store.atomic():
        driver.control("s", "cancel", "control-commit")
        assert marks == []
    assert marks == [("s", False)]
    marks.clear()
    with pytest.raises(ValueError, match="rollback"), store.atomic():
        driver.control("s", "cancel", "control-rollback")
        assert marks == []
        raise ValueError("rollback")
    assert marks == []
    assert all(item.item.input_id != "control-rollback" for item in store.peek_inbox("s"))
    driver.close()


def test_default_driver_closes_its_connection_when_owner_is_released():
    """Discarding an internally owned driver releases C without a GC sweep."""
    import sqlite3

    driver = SessionDriver()
    connection = driver.store.connection
    del driver
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    finally:
        connection.close()


def test_releasing_driver_keeps_host_supplied_connection_open(store):
    """An external H remains the host's resource after its facade disappears."""
    driver = SessionDriver(store=store)
    del driver
    assert store.connection.execute("SELECT 1").fetchone()[0] == 1


def test_default_driver_fallback_closes_on_another_thread():
    """The owning driver may die away from the thread that opened its SQLite C."""
    import sqlite3

    driver = SessionDriver()
    connection = driver.store.connection
    owners, errors, released_by = [driver], [], []
    del driver

    def release():
        try:
            owners.clear()
            released_by.append(get_ident())
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=release)
    try:
        worker.start()
        worker.join(5)
        assert not worker.is_alive() and not errors
        assert released_by == [worker.ident] and worker.ident != get_ident()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    finally:
        worker.join(5)
        connection.close()


def test_sync_result_retains_driver_for_explicit_identity_resume(tmp_path):
    """Returning a result is not the lifetime end of its retained session."""
    result = Runner.run_sync(
        Agent("retained", "Answer."),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
        ),
    )
    driver = result._session_driver
    try:
        assert driver.store.connection.execute("SELECT 1").fetchone() == (1,)
        assert Runner.resume(result.raw_result.session_id, result.raw_result.turn_id).to_dict() == result.to_dict()
    finally:
        driver.close()


def test_closing_one_interactive_session_preserves_shared_driver(tmp_path):
    """A closed-session observer and another session still need their client owner."""
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            workspace=tmp_path,
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
        )
    )
    first = client.create_session(agent=Agent("first", "Answer."))
    second = client.create_session(agent=Agent("second", "Answer."))
    try:
        assert first.close()
        assert first.state().closed
        assert second.prompt("go").final_output == "done"
        assert client.driver.store.connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        client.driver.close()


def test_ownership_reentry_from_destructor_is_bounded():
    """An actual checkout under its snapshot lock must reject destructor reentry."""
    import subprocess
    import sys
    from pathlib import Path

    source = str(Path(ownership.__file__).parents[2])
    script = """
import json, sys, threading
sys.path.insert(0, SOURCE)
from vv_agent.session import ownership
from vv_agent.session.sqlite import SQLiteStore
stages, errors = [], []
with SQLiteStore.standalone(":memory:") as store:
    store.install_schema()
    original = ownership._host_alive
    fired = False
    class Reenter:
        def __del__(self):
            try:
                with ownership.checkout(store):
                    stages.append("unexpected_checkout")
            except ownership.OwnershipError as error:
                stages.append(error.stage)
    def snapshot(host):
        global fired
        if not fired:
            fired = True
            value = Reenter()
            del value
        return original(host)
    ownership._host_alive = snapshot
    def run():
        try:
            with ownership.checkout(store):
                pass
        except BaseException as error:
            errors.append(type(error).__name__)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(1)
    print("INTERNAL_JOIN", worker.is_alive(), flush=True)
    worker.join()
    assert not errors, errors
    assert stages == ["lock-reentry"], stages
    print(json.dumps({"stages": stages, "errors": errors}), flush=True)
""".replace("SOURCE", repr(source))
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-c", script],
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"destructor reentry hung; external child timeout: {error.stdout!r}")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert b'"lock-reentry"' in result.stdout
    print("FIX2_LOCK_REENTRY", result.stdout.decode(), flush=True)


def test_dropping_session_has_no_storage_or_lock_work(tmp_path, monkeypatch):
    import weakref

    from vv_agent.session import surfaces

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            workspace=tmp_path, model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("unused")])
        )
    )
    session = client.create_session(agent=Agent("drop", "Answer."))
    listeners = session._listeners
    session.subscribe(lambda *_: None)
    pulled = session.subscribe()
    cancelled, calls = [], []
    session._background_command_unsubscribers["local"] = lambda: cancelled.append("local")
    real_checkout, real_rows = surfaces.checkout, SQLiteStore._rows

    def checkout(*args, **kwargs):
        calls.append("checkout")
        return real_checkout(*args, **kwargs)

    def rows(current, *args, **kwargs):
        if current is client.driver.store:
            calls.append("SQL")
        return real_rows(current, *args, **kwargs)

    class NoLock:
        def __enter__(self):
            raise AssertionError("session finalizer acquired its lock")

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(surfaces, "checkout", checkout)
    monkeypatch.setattr(SQLiteStore, "_rows", rows)
    session._lock = NoLock()
    reference = weakref.ref(session)
    try:
        del session
        assert reference() is None
        assert calls == []
        assert cancelled == ["local"]
        assert listeners == [] and pulled.closed
    finally:
        client.driver.close()


def test_background_unsubscribe_is_local_under_session_lock(monkeypatch):
    from threading import Lock
    from types import SimpleNamespace

    from vv_agent.runtime.background_sessions import BackgroundSessionManager

    manager = BackgroundSessionManager()
    state = SimpleNamespace(lock=Lock(), listeners=[], status="running")
    monkeypatch.setattr(manager, "_get", lambda _sid: state)
    received = []
    unsubscribe = manager.subscribe("owned", received.append)
    notify = state.listeners[0]
    worker = Thread(target=unsubscribe)
    try:
        with state.lock:
            worker.start()
            worker.join(1)
            immediate = not worker.is_alive()
            if immediate:
                notify({"status": "completed"})
    finally:
        worker.join(5)
    assert not worker.is_alive()
    assert immediate, "local cancellation waited for the background session lock"
    assert received == []


def test_memory_callbacks_belong_to_calling_thread():
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        driver = SessionDriver(store=store)
        entered, release = Event(), Event()
        callbacks, errors, owners = [], [], []

        def transaction():
            try:
                with store.atomic() as tx:
                    owners.append((tx._thread, store._connection_lock._is_owned()))
                    entered.set()
                    assert release.wait(5)
            except BaseException as error:
                errors.append(error)

        worker = Thread(target=transaction)
        worker.start()
        try:
            assert entered.wait(5)
            thread = get_ident()
            assert owners[0][0] != thread and owners[0][1]
            assert not store._connection_lock._is_owned()
            driver._after_admission_commit("owning-thread", lambda: callbacks.append(get_ident()))
            assert not store._defer_memory_wake("owning-thread", lambda: None)
            assert callbacks == [thread], "callback was attached to a foreign transaction"
        finally:
            release.set()
            worker.join(5)
            driver.close()
        assert not worker.is_alive() and errors == []
        with store.atomic():
            driver._after_admission_commit("outer", lambda: callbacks.append(get_ident()))
            assert callbacks == [thread]
        assert callbacks == [thread, thread]
