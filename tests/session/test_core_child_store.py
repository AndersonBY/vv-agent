"""Shared ordinary child projection on real SQL stores."""

import json

import pytest

from vv_agent.kernel import _child_store as core
from vv_agent.kernel._boundary_hydration import fold_hydrated_records
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.memory import token_utils
from vv_agent.session import delegation, reducer
from vv_agent.session.kernel import drive
from vv_agent.session.projection import hydrate_records, hydrate_stored_record, hydration_for_session_store
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .helpers import create_fixture_session, fixture_child_spec, fixture_content_reader, push_fixture_input, wrap_fixture_store
from .test_children import fixture_runtime, parent_runtime
from .test_compaction import SUMMARY, configured


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def start(store):
    create_fixture_session(store, "s", workspace="/tmp/s", consumers=("events",))
    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))


def admitted(store, database, names):
    start(store)
    rt = parent_runtime(store, database)
    rt.children["child_task"] = lambda _plan: tuple(
        core.ChildSession(
            fixture_child_spec(store, _plan.session_id, name),
            "work",
        )
        for name in names
    )
    drive(store, "s", runtime=rt)
    _state, rows, _ = store.read_state("s")
    park = next(row.record for row in rows if row.record.kind == "op_parked")
    stored_plan = next(
        row
        for row in rows
        if row.record.kind == "op_planned" and row.record.operation_id == park.operation_id and row.record.attempt == park.attempt
    )
    plan = hydrate_stored_record(stored_plan, hydration_for_session_store(store, "s")).record
    return rt, plan, core.child_handles(park.payload["handle"])


@pytest.mark.parametrize("names", [("child-a",), ("child-b", "child-a")])
def test_sdk_and_core_reuse_first_snapshot_once_and_preserve_result_bytes(store, database, monkeypatch, names):
    rt, plan, handles = admitted(store, database, names)
    for name in names:
        drive(store, name, runtime=fixture_runtime(store, database, name, [LLMResponse("雪-" + name)]))
    reads, snapshots, forwarded = [], {}, []
    actual_read = store.read_state

    def read(sid):
        reads.append(sid)
        snapshots[sid] = actual_read(sid)
        return snapshots[sid]

    def ordinary(selected_store, selected_plan, selected_handles, *, first_snapshot):
        forwarded.append(first_snapshot)
        return core.ordinary_result_for_children(
            selected_store,
            selected_plan,
            selected_handles,
            first_snapshot=first_snapshot,
        )

    monkeypatch.setattr(store, "read_state", read)
    first = store.read_state(names[0])
    reads.clear()
    direct = core.ordinary_result_for_children(store, plan, handles, first_snapshot=first)
    assert reads == list(names[1:])

    reads.clear()
    direct_without_first = core.ordinary_result_for_children(store, plan, handles)
    assert reads == list(names)

    reads.clear()
    monkeypatch.setattr(delegation, "ordinary_result_for_children", ordinary)
    sdk = delegation.result_for_children(store, plan, handles, rt)
    assert reads == list(names)
    assert forwarded == [snapshots[names[0]]]

    assert canonical_json_bytes(sdk.to_dict()) == canonical_json_bytes(direct.to_dict())
    assert canonical_json_bytes(direct_without_first.to_dict()) == canonical_json_bytes(direct.to_dict())
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
    with SQLiteStore.standalone(str(path)) as raw_store:
        store = wrap_fixture_store(raw_store)
        store.install_schema()
        rt, plan, handles = admitted(store, path, ("child",))
        drive(
            store,
            "child",
            runtime=configured(
                path,
                [LLMResponse(SUMMARY), LLMResponse("child done")],
                content_reader=fixture_content_reader(store, "child"),
            ),
        )
        snapshot = store.read_state("child")
        assert any(row.record.kind == "context_compacted" for row in snapshot[1])
        expected = delegation.result_for_children(store, plan, handles, rt)
        parent = store.read_state("s")[0]
        assert parent.active_turn_id is not None
        raw_calls, logical_calls, chains = [], [], set()

        def raw_count(messages, *, model):
            assert_core_suspended()
            raw_calls.append((model, messages))
            return token_utils.count_messages_tokens(messages, model=model)

        monkeypatch.setattr(reducer, "count_messages_tokens", raw_count)
        store._fold_cache = store._previous_prefix = None
        store.read_state("child")
        assert raw_calls == [], "raw SQL folding must not invoke logical token counting"

        def count(messages, *, model):
            chains.update(assert_core_suspended())
            logical_calls.append((model, messages))
            return token_utils.count_messages_tokens(messages, model=model)

        monkeypatch.setattr(reducer, "count_messages_tokens", count)
        hydration = hydration_for_session_store(store, "child")
        source_rows = snapshot[1]
        hydrated_rows = hydrate_records(source_rows, hydration)
        fold_hydrated_records(hydrated_rows, source_rows, reducer.Fold, hydration)
        assert len(logical_calls) == 2
        assert {"extend_steps", "_compacted", "accept_summary"} <= chains

        logical_baseline = len(logical_calls)
        store._fold_cache = store._previous_prefix = None
        if entry == "ordinary_result_for_children":
            actual = core.ordinary_result_for_children(store, plan, handles)
            assert actual.to_dict() == expected.to_dict()
        else:
            with store.atomic() as tx:
                assert core.cancel_children(store, tx, parent, "s", parent.active_turn_id) == []
            assert not store.peek_inbox("child"), "completed child must not receive a new cancellation"
        assert len(logical_calls) == logical_baseline
