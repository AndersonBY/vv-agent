"""Shared ordinary child projection on real SQL stores."""

import inspect
import json

import pytest

from vv_agent.kernel import _child_store as core
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.memory import token_utils
from vv_agent.session import delegation, reducer
from vv_agent.session.kernel import drive
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .test_children import parent_runtime
from .test_compaction import SUMMARY, configured, start
from .test_recovery_matrix import runtime


def admitted(store, database, names):
    start(store)
    rt = parent_runtime(database)
    rt.children["child_task"] = lambda _plan: tuple(
        core.ChildSession(SessionSpec(name, "test", "/tmp"), "work") for name in names
    )
    drive(store, "s", runtime=rt)
    state, rows, _ = store.read_state("s")
    park = next(row.record for row in rows if row.record.kind == "op_parked")
    plan = state.operations[park.operation_id].attempts[park.attempt].execution_plan
    return rt, plan, core.child_handles(park.payload["handle"])


@pytest.mark.parametrize("names", [("child-a",), ("child-b", "child-a")])
def test_sdk_and_core_reuse_first_snapshot_once_and_preserve_result_bytes(store, database, monkeypatch, names):
    rt, plan, handles = admitted(store, database, names)
    for name in names:
        drive(store, name, runtime=runtime(database, [LLMResponse("雪-" + name)]))
    reads, snapshots, forwarded = [], {}, []
    actual_read = store.read_state

    def read(sid):
        reads.append(sid)
        snapshots[sid] = actual_read(sid)
        return snapshots[sid]

    def ordinary(*args, **kwargs):
        forwarded.append(kwargs["first_snapshot"])
        return core.ordinary_result_for_children(*args, **kwargs)

    monkeypatch.setattr(store, "read_state", read)
    monkeypatch.setattr(delegation, "ordinary_result_for_children", ordinary)
    sdk = delegation.result_for_children(store, plan, handles, rt)
    assert reads == list(names)
    assert forwarded == [snapshots[names[0]]] and forwarded[0] is snapshots[names[0]]
    reads.clear()
    direct = core.ordinary_result_for_children(store, plan, handles)
    assert reads == list(names)
    assert canonical_json_bytes(sdk.to_dict()) == canonical_json_bytes(direct.to_dict())
    expected = [
        {
            "tool_call_id": "child-call",
            "content": "雪-" + name,
            "status_code": "SUCCESS",
            "directive": "continue",
            "metadata": {"child_session_id": name, "child_turn_id": name + "/turn/start", "status": "completed"},
        }
        for name in names
    ]
    if len(names) == 1:
        assert direct.to_dict() == expected[0]
    else:
        assert direct.content == json.dumps(expected)
        assert direct.metadata == {"children": [item["metadata"] for item in expected]}


@pytest.mark.parametrize("empty", [False, True])
def test_ordinary_projection_preserves_first_failure_and_read_order(store, database, monkeypatch, empty):
    rt, plan, handles = admitted(store, database, ("first", "second"))
    reads = []
    actual = store.read_state

    def read(sid):
        reads.append(sid)
        return actual(sid)

    monkeypatch.setattr(store, "read_state", read)
    handles = [] if empty else handles
    error = IndexError if empty else StopIteration
    with pytest.raises(error):
        delegation.result_for_children(store, plan, handles, rt)
    assert reads == ([] if empty else ["first"])
    reads.clear()
    with pytest.raises(error):
        core.ordinary_result_for_children(store, plan, handles)
    assert reads == ([] if empty else ["first"])


@pytest.mark.parametrize("entry", ["ordinary_result_for_children", "cancel_children"])
def test_cold_compacted_child_read_keeps_pure_count_steps_suspended(tmp_path, monkeypatch, entry):
    from test_token_count_suspension import assert_core_suspended

    path = tmp_path / "child.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        rt, plan, handles = admitted(store, path, ("child",))
        drive(store, "child", runtime=configured(path, [LLMResponse(SUMMARY), LLMResponse("child done")]))
        snapshot = store.read_state("child")
        assert any(row.record.kind == "context_compacted" for row in snapshot[1])
        expected = delegation.result_for_children(store, plan, handles, rt)
        parent = store.read_state("s")[0]
        assert parent.active_turn_id is not None
        baseline, calls, chains, storage_frames = [], [], set(), set()

        def baseline_count(messages, *, model):
            assert_core_suspended()
            baseline.append((model, messages))
            return token_utils.count_messages_tokens(messages, model=model)

        monkeypatch.setattr(reducer, "count_messages_tokens", baseline_count)
        store._fold_cache = store._previous_prefix = None
        store.read_state("child")
        assert len(baseline) == 2

        def count(messages, *, model):
            frame = inspect.currentframe()
            assert frame is not None
            frame = frame.f_back
            while frame is not None:
                module = frame.f_globals.get("__name__", "")
                if module.startswith("vv_agent.kernel.") and module != "vv_agent.kernel._sql":
                    # Only this real storage caller may remain active; pure core
                    # validation/planning frames are still forbidden.
                    assert (module, frame.f_code.co_name) == ("vv_agent.kernel._child_store", entry)
                    storage_frames.add(frame.f_code.co_name)
                steps = frame.f_locals.get("steps")
                while inspect.isgenerator(steps):
                    assert inspect.getgeneratorstate(steps) == inspect.GEN_SUSPENDED
                    chains.add(steps.gi_code.co_name)
                    steps = steps.gi_yieldfrom
                frame = frame.f_back
            calls.append((model, messages))
            return token_utils.count_messages_tokens(messages, model=model)

        monkeypatch.setattr(reducer, "count_messages_tokens", count)
        store._fold_cache = store._previous_prefix = None
        if entry == "ordinary_result_for_children":
            actual = core.ordinary_result_for_children(store, plan, handles)
            assert actual.to_dict() == expected.to_dict()
        else:
            with store.atomic() as tx:
                assert core.cancel_children(store, tx, parent, "s", parent.active_turn_id) == []
            assert not store.peek_inbox("child"), "completed child must not receive a new cancellation"
        assert calls == baseline
        assert storage_frames == {entry}
        assert {"extend_steps", "_compacted", "accept_summary"} <= chains
