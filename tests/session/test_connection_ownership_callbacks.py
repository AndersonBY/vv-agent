"""Observable strict callback signatures and synchronous return boundaries."""

import gc
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest

from vv_agent.run_handle import RunHandle
from vv_agent.session.callbacks import CallbackContractError, validate
from vv_agent.session.children import ChildSession, SummaryChildTool
from vv_agent.session.kernel import drive
from vv_agent.session.ownership import OwnershipError
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import SessionTx

from .test_children import parent_runtime
from .test_recovery_matrix import runtime, start


def test_signatures():
    def direct(plan, *, tx):
        return plan

    def summary(plan, summary, *, tx):
        return plan

    def hook(point, record, *, tx):
        return None

    class Instance:
        def __call__(self, plan, *, tx):
            return plan

    for fn, n in [(direct, 1), (summary, 2), (hook, 2), (partial(direct), 1), (Instance(), 1)]:
        assert validate(fn, n) is fn

    async def async_fn(plan, *, tx):
        pass

    class AsyncInstance:
        async def __call__(self, plan, *, tx):
            pass

    class Opaque:
        @property
        def __signature__(self):
            raise ValueError("opaque")

        def __call__(self, *args, **kwargs):
            pass

    for fn in [
        lambda plan: None,
        lambda plan, **kwargs: None,
        lambda plan, *, tx=None: None,
        partial(direct, tx=None),
        async_fn,
        AsyncInstance(),
        Opaque(),
    ]:
        with pytest.raises(CallbackContractError):
            validate(fn, 1)

    def wrapper(plan, *, tx):
        return Opaque()(plan, tx=tx)

    assert validate(wrapper, 1) is wrapper


@pytest.mark.parametrize("point", ["admit", "before_commit", "after_commit", "after_drive", "typeerror"])
def test_synchronous_boundary(store, database, point):
    start(store)
    called = []
    after = []
    retained = []
    error = TypeError("business TypeError")

    async def coroutine():
        return None

    def admit(plan, *, tx):
        called.append(tx)
        if point == "typeerror":
            raise error
        if point == "admit":
            return coroutine()
        return [ChildSession(SessionSpec("a", "test", "/tmp"), "a"), ChildSession(SessionSpec("b", "test", "/tmp"), "b")]

    def hook(where, record, *, tx):
        admission = record is not None and record.kind == "op_parked"
        if where == "before_commit" and admission:
            assert tx is called[0]
            retained.append(tx)
        if where == "after_commit" and admission:
            assert tx is None
            after.append(1)
        if (where == point and admission) or (point == "after_drive" and where == "after_drive"):
            return coroutine()

    rt = parent_runtime(database)
    rt.children["child_task"] = admit
    rt.hook = hook
    with pytest.raises(TypeError if point == "typeerror" else CallbackContractError) as caught:
        drive(store, "s", runtime=rt)
    if point == "typeerror":
        assert caught.value is error
    assert len(called) == 1
    durable = point in {"after_commit", "after_drive"}
    assert store._one("SELECT count(*) FROM sk_session WHERE parent_session_id='s'")[0] == (2 if durable else 0)
    assert after == ([1] if durable else [])
    gc.collect()


def test_original_hook_and_summary_rejected_early(store, database):
    start(store)
    observed = []
    rt = parent_runtime(database)
    rt.children["child_task"] = SummaryChildTool(lambda plan: observed.append("summary"), lambda plan, summary: None)
    with pytest.raises(CallbackContractError):
        drive(store, "s", runtime=rt)
    assert observed == [] and len(rt.llm.steps) == 2
    rt = runtime(database, [])
    rt.hook = lambda point, record: observed.append("old")
    from vv_agent.session.surfaces import SessionDriver

    kernel = SessionDriver(store=store)
    with pytest.raises(CallbackContractError):
        RunHandle(kernel, "s", "tid", rt, None)
    assert observed == []


def test_non_sql_protocol_handle_access_only():
    class OrdinaryTx(SessionTx):
        def push(self, sid, item):
            return sid

    tx = OrdinaryTx()

    def callback(plan, *, tx):
        return tx.push("s", InboxItem("i", "steer", {"content": "hello"}))

    assert validate(callback, 1)(None, tx=tx) == "s"
    with pytest.raises(OwnershipError, match="unsupported_transaction_handle"):
        _ = tx.connection


def test_import_without_optional_psycopg(tmp_path):
    src = Path(__file__).resolve().parents[2] / "src"
    code = f"""import sys
sys.path.insert(0, {str(src)!r})
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'psycopg' or fullname.startswith('psycopg.'):
            raise ImportError('optional dependency absent')
sys.meta_path.insert(0, Block())
import vv_agent
from vv_agent.session.store import SessionTx
print('SDK_IMPORT_WITHOUT_PSYCOPG_OK')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "SDK_IMPORT_WITHOUT_PSYCOPG_OK" in result.stdout


@pytest.mark.parametrize("raw_access", [False, True])
def test_non_sql_adapter_commits_plain_operations(store, database, raw_access):
    start(store)
    admitted = []

    class PlainTx(SessionTx):
        def __init__(self, actual):
            self.actual = actual

        def lock_session(self, *a, **kw):
            return self.actual.lock_session(*a, **kw)

        def create(self, *a, **kw):
            return self.actual.create(*a, **kw)

        def push(self, *a, **kw):
            return self.actual.push(*a, **kw)

        def append(self, *a, **kw):
            return self.actual.append(*a, **kw)

    class Adapter:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        @contextmanager
        def atomic(self):
            with self.actual.atomic() as actual:
                yield PlainTx(actual)

        @contextmanager
        def heartbeat_store(self):
            with store.heartbeat_store() as actual:
                yield Adapter(actual)

        @contextmanager
        def execution_store(self):
            with store.execution_store() as actual:
                yield Adapter(actual)

    def admit(plan, *, tx):
        admitted.append(tx)
        tx.create(SessionSpec("custom-side", "test", "/tmp"), consumers=())
        tx.push("custom-side", InboxItem("custom-tx", "steer", {"content": "custom"}))
        if raw_access:
            _ = tx.connection
        return [ChildSession(SessionSpec("custom-child", "test", "/tmp"), "go")]

    rt = parent_runtime(database)
    rt.children["child_task"] = admit
    rt.heartbeat_store = Adapter(store).heartbeat_store
    if raw_access:
        with pytest.raises(OwnershipError, match="unsupported_transaction_handle"):
            drive(Adapter(store), "s", runtime=rt)
        assert store.list_sessions() == ("s",)
    else:
        drive(Adapter(store), "s", runtime=rt)
        assert store.read_input("custom-side", "custom-tx") is not None
        assert store.list_sessions() == ("custom-child", "custom-side", "s")
    assert len(admitted) == 1


@pytest.mark.parametrize("form", ["direct", "partial", "instance", "wrapper"])
def test_valid_forms_run_real_admission(store, database, form):
    start(store)
    store._rows("CREATE TABLE callback_host (id integer PRIMARY KEY)")
    calls = []

    def base(plan, *, tx, tag):
        calls.append((tx, tag))
        tx.connection.execute("INSERT INTO callback_host VALUES (1)")
        return ChildSession(SessionSpec("form-child", "test", "/tmp"), "go")

    def direct(plan, *, tx):
        return base(plan, tx=tx, tag=form)

    class Instance:
        def __call__(self, plan, *, tx):
            return base(plan, tx=tx, tag=form)

    class Opaque:
        @property
        def __signature__(self):
            raise ValueError("opaque")

        def __call__(self, plan, *, tx):
            return base(plan, tx=tx, tag=form)

    opaque = Opaque()

    def wrapper(plan, *, tx):
        return opaque(plan, tx=tx)

    choices = {"direct": direct, "partial": partial(base, tag=form), "instance": Instance(), "wrapper": wrapper}
    rt = parent_runtime(database)
    rt.children["child_task"] = choices[form]
    assert drive(store, "s", runtime=rt)
    assert len(calls) == 1 and calls[0][1] == form
    assert store._one("SELECT count(*) FROM callback_host")[0] == 1
    assert store.list_sessions() == ("form-child", "s")
    assert len(rt.llm.steps) == 1


def test_replaced_summary_binding_rejected_before_preparation(store, database):
    start(store)
    observed = []
    rt = parent_runtime(database)
    response = rt.llm.steps[0]

    def model(request):
        observed.append("model")
        rt.children["child_task"] = SummaryChildTool(
            lambda plan: observed.append("predicate"), lambda plan, summary: observed.append("old")
        )
        return response

    rt.llm.steps[0] = model
    with pytest.raises(CallbackContractError):
        drive(store, "s", runtime=rt)
    assert observed == ["model"]
    assert any(r.record.kind == "op_completed" for r in store.read_state("s")[1])


@pytest.mark.persistent_store
def test_opaque_store_can_supply_validated_builtin_heartbeat(store, database):
    start(store)

    class PlainTx(SessionTx):
        def __init__(self, tx):
            self.actual = tx

        def create(self, *a, **kw):
            return self.actual.create(*a, **kw)

        def push(self, *a, **kw):
            return self.actual.push(*a, **kw)

        def append(self, *a, **kw):
            return self.actual.append(*a, **kw)

    class Opaque:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        @contextmanager
        def execution_store(self):
            with store.execution_store() as actual:
                yield Opaque(actual)

        @contextmanager
        def atomic(self):
            with self.actual.atomic() as tx:
                yield PlainTx(tx)

    rt = parent_runtime(database)
    rt.heartbeat_store = store.heartbeat_store
    assert drive(Opaque(store), "s", runtime=rt)
    assert store.list_sessions() == ("child", "s")


def test_summary_selection_still_matches_captured_binding(store, database):
    start(store)
    called = []
    rt = parent_runtime(database)

    def replacement(plan, *, tx):
        called.append("replacement")
        return ChildSession(SessionSpec("unexpected", "test", "/tmp"), "go")

    def predicate(plan):
        rt.children["child_task"] = replacement
        return False

    def original(plan, summary, *, tx):
        called.append("original")
        return ChildSession(SessionSpec("unexpected", "test", "/tmp"), "go")

    rt.children["child_task"] = SummaryChildTool(predicate, original)
    with pytest.raises(CallbackContractError):
        drive(store, "s", runtime=rt)
    assert called == [] and store.list_sessions() == ("s",)
