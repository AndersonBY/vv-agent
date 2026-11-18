"""A8 child delivery remains consumable after the child is forgotten."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.kernel._child_store import (
    PENDING_CHILD_COMPLETION,
    ChildCompletion,
    verify_completion,
)
from vv_agent.kernel._content import ContentRef, Principal, TombstoneState
from vv_agent.kernel._driver_inputs import result_for_parent_copies
from vv_agent.kernel._records import Record
from vv_agent.kernel._sql import LedgerIntegrityError
from vv_agent.kernel._tool_values import ToolDirective, ToolResultStatus
from vv_agent.session.content_runtime import SessionContentWriter, durable_content_store
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.store import Conflict

from .helpers import create_fixture_session, wrap_fixture_store
from .test_children import child_result, complete_child, delivered, parked


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


@pytest.mark.parametrize("operation", ["forget", "purge"])
@pytest.mark.parametrize("child_level", [1, 2])
def test_parent_consumes_copied_child_result_once_after_child_deletion(store, database, operation, child_level):
    parent_runtime, _, handle = parked(store, database)
    complete_child(store, database)
    if child_level == 2:
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        store._rows(
            "UPDATE sk_session SET level=2 WHERE tenant_id=%s AND session_id='child'",
            (store.tenant_id,),
        )
        store.configure_worker(level=2)
    delivered(store)

    queued = child_result(store)
    copied_result = ContentRef.from_dict(queued.payload["result"])
    parent_creation = store.read("s", limit=1).records[0].record.payload
    child_creation = store.read("child", limit=1).records[0].record.payload
    parent_principal = Principal.from_dict(parent_creation["principal"])
    child_principal = Principal.from_dict(child_creation["principal"])
    assert copied_result.key_id == parent_principal.attributes.key_id

    deletion = SessionDeletionStore(store, durable_content_store(store), allow_unsigned=True)
    receipt = getattr(deletion, f"{operation}_session")(
        store.tenant_id,
        "child",
        actor=child_principal,
    )
    assert receipt.session_ids == ("child",)
    if operation == "forget":
        assert store._rows(
            "SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, "child"),
        ) == [("forgotten",)]
    else:
        assert store._rows(
            "SELECT count(*) FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, "child"),
        ) == [(0,)]
    assert store._tombstone("child").state is TombstoneState.COMPLETE
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="s")
        assert writer.get_json(copied_result) == "child done"

    drive(store, "s", runtime=parent_runtime)
    state, records, _ = read_state(store, "s")
    assert state.active_turn_id is None
    applied = [
        stored
        for stored in records
        if stored.record.kind == "input_applied"
        and stored.record.payload["input"]["kind"] == "child_result"
        and stored.record.payload["input"]["payload"]["session_id"] == "child"
    ]
    assert len(applied) == 1
    assert applied[0].record.payload["disposition"] == "applied"
    assert applied[0].record.payload["input"]["payload"]["terminal_seq"] == queued.payload["terminal_seq"]
    assert applied[0].record.payload["input"]["payload"]["terminal_digest"] == queued.payload["terminal_digest"]
    completed = [
        stored
        for stored in records
        if stored.record.kind == "op_completed" and stored.record.operation_id == handle["delivery_target"]["operation_id"]
    ]
    assert len(completed) == 1

    drive(store, "s", runtime=parent_runtime)
    _, records_after_replay, _ = read_state(store, "s")
    assert (
        len(
            [
                stored
                for stored in records_after_replay
                if stored.record.kind == "input_applied"
                and stored.record.payload["input"]["kind"] == "child_result"
                and stored.record.payload["input"]["payload"]["session_id"] == "child"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("operation", ["forget", "purge"])
def test_parent_rejects_unlinked_child_result_after_child_deletion(store, database, operation):
    parent_runtime, _, handle = parked(store, database)
    complete_child(store, database)
    delivered(store)
    valid = child_result(store)
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="s")
        unlinked_result = writer.put_json("forged child result")
    forged = replace(
        valid,
        input_id="forged-child-result",
        payload=valid.payload | {"result": unlinked_result.to_dict()},
    )
    child_creation = store.read("child", limit=1).records[0].record.payload
    child_principal = Principal.from_dict(child_creation["principal"])
    deletion = SessionDeletionStore(store, durable_content_store(store), allow_unsigned=True)
    getattr(deletion, f"{operation}_session")(store.tenant_id, "child", actor=child_principal)

    assert verify_completion(store, "s", handle, forged) is None
    drive(store, "s", runtime=parent_runtime)
    _, records, _ = read_state(store, "s")
    child_inputs = [
        stored.record.payload
        for stored in records
        if stored.record.kind == "input_applied" and stored.record.payload["input"]["kind"] == "child_result"
    ]
    assert len(child_inputs) == 1
    assert child_inputs[0]["input"]["input_id"] == valid.input_id
    assert child_inputs[0]["disposition"] == "applied"
    assert (
        sum(
            stored.record.kind == "op_completed" and stored.record.operation_id == handle["delivery_target"]["operation_id"]
            for stored in records
        )
        == 1
    )


def test_child_result_admission_rejects_foreign_or_unrelated_child(store, database):
    _, _, _ = parked(store, database)
    complete_child(store, database)
    delivered(store)
    valid = child_result(store)

    other = wrap_fixture_store(store.for_tenant("other"))
    create_fixture_session(other, "other-child", workspace="/tmp/other-child")
    foreign = replace(valid, input_id="foreign-child-result", payload=valid.payload | {"session_id": "other-child"})
    with pytest.raises(Conflict, match="missing child session"), store.atomic() as tx:
        tx.push("s", foreign)

    create_fixture_session(store, "unrelated", workspace="/tmp/unrelated")
    unrelated = replace(valid, input_id="unrelated-child-result", payload=valid.payload | {"session_id": "unrelated"})
    with pytest.raises(Conflict, match="does not belong to this parent session"), store.atomic() as tx:
        tx.push("s", unrelated)


@pytest.mark.parametrize("operation", ["forget", "purge"])
def test_parent_waits_for_complete_tombstone_while_child_copy_is_pending(store, database, operation):
    parent_runtime, _, handle = parked(store, database)
    complete_child(store, database)
    delivered(store)
    queued = child_result(store)
    child_creation = store.read("child", limit=1).records[0].record.payload
    child_principal = Principal.from_dict(child_creation["principal"])

    if operation == "forget":
        store.begin_forget(
            "child",
            child_principal,
            bound_signer=None,
            verifier=None,
            allow_unsigned=True,
        )
    else:
        store.begin_purge(
            "child",
            child_principal,
            bound_signer=None,
            verifier=None,
            allow_unsigned=True,
        )
    tombstone = store._tombstone("child")
    assert tombstone is not None and tombstone.state is TombstoneState.PENDING
    assert verify_completion(store, "s", handle, queued) is PENDING_CHILD_COMPLETION

    drive(store, "s", runtime=parent_runtime)
    state, records, _ = read_state(store, "s")
    assert state.active_turn_id is not None
    assert not any(record.record.kind == "turn_ended" for record in records)
    assert child_result(store) == queued

    if operation == "forget":
        store.complete_forget("child")
    else:
        store.complete_purge("child")
    assert store.child_is_deleted("child")

    drive(store, "s", runtime=parent_runtime)
    state, records, _ = read_state(store, "s")
    assert state.active_turn_id is None
    assert any(record.record.kind == "turn_ended" for record in records)


def test_tampered_parent_routing_level_cannot_certify_itself(store, database):
    _, _, _ = parked(store, database)
    complete_child(store, database)
    delivered(store)
    store._rows(
        "UPDATE sk_session SET level=999 WHERE tenant_id=%s AND session_id='s'",
        (store.tenant_id,),
    )

    row = store._lock("s")
    with pytest.raises(LedgerIntegrityError, match="does not match the verified requirements"):
        store._require_routing_cache("s", row)


def _projection_plan(arguments: dict[str, Any]) -> Record:
    return cast(
        Record,
        SimpleNamespace(_payload={"request": {"id": "delegation-call", "arguments": arguments}}),
    )


def test_configured_parent_copy_preserves_create_sub_task_envelope():
    completion = ChildCompletion("child", "child/turn/start", "completed", "child done", 7, "a" * 64, True)
    result = result_for_parent_copies(
        _projection_plan({"agent_id": "worker", "task_description": "work"}),
        [{"session_id": "child", "turn_id": "child/turn/start"}],
        [completion],
        {"mode": "configured", "agent_name": "worker", "metadata": {}},
    )

    payload = json.loads(result.content)
    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.tool_call_id == "delegation-call"
    assert payload["agent_name"] == "worker"
    assert payload["status"] == "completed"
    assert payload["final_answer"] == "child done"
    assert result.metadata == payload


def test_handoff_parent_copy_preserves_finish_directive():
    completion = ChildCompletion("child", "child/turn/start", "completed", "handoff done", 8, "b" * 64, True)
    result = result_for_parent_copies(
        _projection_plan({"input": "delegate"}),
        [{"session_id": "child", "turn_id": "child/turn/start"}],
        [completion],
        {"mode": "handoff", "agent_name": "target", "metadata": {"source": "test"}},
    )

    assert result.directive is ToolDirective.FINISH
    assert result.content == "handoff done"
    assert result.metadata == {
        "agent": "target",
        "mode": "handoff",
        "child_status": "completed",
        "child_run_id": "child",
    }
