"""One core fold and explicit ports across SQL, driver and historical projection."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from vv_agent.kernel import _context, _reducer, _sql, _state, _store
from vv_agent.memory import MemoryManager
from vv_agent.session import context, reducer, sql, store
from vv_agent.session.context import project_context
from vv_agent.session.kernel import _Driver, _Scope, drive
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .helpers import base, record
from .test_compaction import SUMMARY
from .test_recovery_matrix import runtime, start


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("first", ["_store", "_state", "_context", "_reducer", "_sql"])
def test_coherent_fold_imports_and_executes_without_outer_sdk(tmp_path, site_flags, first):
    source = Path(__file__).resolve().parents[2] / "src"
    script = """
import importlib
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'}
                or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
importlib.import_module('vv_agent.kernel.' + sys.argv[2])
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._records import SessionSpec
from vv_agent.kernel._context import project_context
from vv_agent.kernel._sql import SQLStore

def unexpected(*args, **kwargs):
    raise AssertionError('creation must not normalize settings or count')
fold = Fold(normalize_memory_settings=unexpected, count_messages=unexpected)
state = fold.extend([SessionSpec('s', 'p', 'w').record()])
assert state.session_id == 's' and state._protection is not None
assert project_context(fold.history, state) == []
assert fold.fork()._count_messages is unexpected
store = SQLStore(fold_factory=lambda: fold.fork())
assert store.fold_factory().state.session_id == 's'
assert all(name in {'vv_agent', 'vv_agent.kernel'} or name.startswith('vv_agent.kernel.')
           for name in sys.modules if name.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(source), first],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_value_exports_and_single_transition_implementation():
    for name in (
        "Conflict",
        "SequenceConflict",
        "LeaseLost",
        "Lease",
        "StoredRecord",
        "StoredInput",
        "ReadPage",
        "CommitReceipt",
        "CreateReceipt",
        "InputReceipt",
        "ConsumerBatch",
        "WorkCursor",
        "WorkItem",
        "LeasePoll",
        "SessionTx",
        "SessionStore",
        "LeaseRetryExhausted",
    ):
        assert getattr(store, name) is getattr(_store, name)
    for name in ("Attempt", "Operation", "Turn", "ExecutionState"):
        assert getattr(reducer, name) is getattr(_state, name)
    assert reducer.TransitionError is _reducer.TransitionError
    assert context.InputProtection is _context.InputProtection
    assert sql.SQLSessionTx is _sql.SQLSessionTx
    assert reducer.Fold.extend is _reducer.Fold.extend
    assert reducer.Fold.fork is _reducer.Fold.fork
    receipt = store.CommitReceipt("c", (), 1)
    assert receipt == store.CommitReceipt(
        "c", (), 1, records=(store.StoredRecord(SessionSpec("s", "p", "w").record(), 1, "c", 1, 0),), inbox_seq=9
    )
    assert "records=" not in repr(receipt)
    with pytest.raises(FrozenInstanceError):
        receipt.__setattr__("head_seq", 2)
    assert store.WorkCursor("a", "drive") < store.WorkCursor("b", "drive")
    error = store.LeaseRetryExhausted("s", "t", 2)
    assert isinstance(error, store.LeaseLost) and isinstance(error, ValueError)
    assert str(error) == "Execution lease recovery exhausted after 2 attempts for s/t"


def test_core_fork_preserves_ports_without_eager_work():
    def unused(*args, **kwargs):
        raise AssertionError("non-compaction transitions must not use ports")

    fold = _reducer.Fold(normalize_memory_settings=unused, count_messages=unused)
    fold.extend(base())
    fork = fold.fork()
    assert fork._normalize_memory_settings is unused and fork._count_messages is unused
    fork.extend([record("op_completed")])
    assert fork.state.operations["o"].selected_attempt == 1
    assert fold.state.operations["o"].selected_attempt is None
    with pytest.raises(reducer.TransitionError, match="result request/provider mismatch"):
        fold.fork().extend([record("op_completed", provider_binding="wrong")])


def _binding():
    calls, folds = [], []

    def normalize(settings, protected):
        calls.append(("normalize", protected))
        return reducer._normalize_memory_settings(settings, protected)

    def count(messages, *, model):
        calls.append(("count", model))
        return sum(len(str(message.get("content", ""))) for message in messages)

    def factory():
        fold = _reducer.Fold(normalize_memory_settings=normalize, count_messages=count)
        folds.append(fold)
        return fold

    return factory, calls, folds, normalize, count


def test_sql_driver_and_historical_rebuild_share_nondefault_ports():
    factory, calls, folds, normalize, count = _binding()
    with SQLiteStore.standalone(":memory:") as database:
        assert database.fold_factory is reducer.Fold  # A class default must not bind self.
        database.fold_factory = factory
        database.install_schema()
        start(database)

        def too_long(request):
            raise RuntimeError("maximum context length exceeded")

        rt = runtime(database, [LLMResponse("seen facts " * 4000), too_long, LLMResponse(SUMMARY), LLMResponse("done")])
        rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=2)
        rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
        drive(database, "s", runtime=rt)
        state, rows, _ = database.read_state("s")
        assert state._protection is not None
        assert calls and any(call[0] == "count" for call in calls)
        assert all(f._normalize_memory_settings is normalize and f._count_messages is count for f in folds)
        lease = database.acquire("s", owner="probe", ttl_ms=15000)
        assert lease is not None
        driver = _Driver(database, "s", rt, _Scope(lease, rt))
        assert database.release(lease)
        assert driver.fold._count_messages is count
        assert driver.fold._normalize_memory_settings is normalize
        cached = database._fold_cache
        assert cached is not None and cached.reducer._count_messages is count
        assert cached.reducer.fork()._count_messages is count
        end = next(i + 1 for i, row in enumerate(rows) if row.record.kind == "context_compacted")
        prefix = rows[:end]
        assert prefix[-1].seq < state._protection.prefix_seq
        before = len(calls)
        historical = project_context(prefix, state, _fold_factory=factory)
        assert any(message.name == "memory_summary" for message in historical)
        assert any(call[0] == "count" for call in calls[before:]), "historical compaction must revalidate with the same counter"
        assert calls[before][0] == "normalize"
        assert folds[-1]._count_messages is count
        # The historical core projection accepts only an aligned state.
        with pytest.raises(AssertionError, match="context prefix is not aligned"):
            _context.project_context(prefix, state)
        assert all(row.record.encode() == rows[i].record.encode() for i, row in enumerate(prefix))


def test_replacing_sql_factory_discards_both_cached_prefixes():
    first, _, first_folds, _, _ = _binding()
    second, _, second_folds, normalize, count = _binding()
    with SQLiteStore.standalone(":memory:") as database:
        database.install_schema()
        database.fold_factory = first
        with database.atomic() as tx:
            for sid in ("parent", "child"):
                tx.create(SessionSpec(sid, "p", "w"), consumers=())
        for sid in ("parent", "child"):
            database.read_state(sid)
        assert database._fold_cache is not None and database._previous_prefix is not None
        assert first_folds
        database.fold_factory = second
        assert database._fold_cache is database._previous_prefix is None
        for sid in ("parent", "child"):
            database.read_state(sid)
        assert len(second_folds) == 2
        assert all(f._count_messages is count and f._normalize_memory_settings is normalize for f in second_folds)


def test_factory_selection_recognizes_core_sql_and_keeps_other_store_defaults():
    factory, _, _, _, _ = _binding()
    core_store = _sql.SQLStore(fold_factory=factory)
    assert reducer._fold_factory_for(core_store) is factory
    with SQLiteStore.standalone(":memory:") as sdk_store:
        sdk_store.fold_factory = factory
        assert reducer._fold_factory_for(sdk_store) is factory
    assert reducer._fold_factory_for(object()) is reducer.Fold


def test_real_app_server_fork_preserves_explicit_store_factory(monkeypatch):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.app_server import DefaultAppServerHost

    from .test_app_server_fork import run_turn, send, server_for

    factory, _, folds, _, _ = _binding()
    with SQLiteStore.standalone(":memory:") as database:
        database.install_schema()
        database.fold_factory = factory
        host = DefaultAppServerHost(
            agent=Agent("test", "Answer."),
            run_config=RunConfig(model_provider=ScriptedModelProvider.from_callback("test", "m", lambda _: LLMResponse("done"))),
        )
        server = server_for(database, host)
        try:
            send(server, "create", "thread/start", {"inputId": "source"})
            first = run_turn(server, "source", "first")
            before = len(folds)

            def lost_binding(*args, **kwargs):
                raise AssertionError("fork lost explicit CoreFold factory")

            monkeypatch.setattr(reducer.Fold, "__init__", lost_binding)
            result = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy", "throughTurnId": first})
            assert result["accepted"] is True
            assert len(folds) > before
            assert database.read_state(result["threadId"])[0].session_id == result["threadId"]
        finally:
            server.kernel.close()
