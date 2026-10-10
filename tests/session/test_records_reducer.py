import json
from dataclasses import replace

import pytest

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.records import InboxItem, Record, RecordError, digest
from vv_agent.session.reducer import TransitionError, fold

from .helpers import base, item, record


@pytest.mark.parametrize(
    "kind",
    [
        "session_created",
        "turn_started",
        "input_applied",
        "op_planned",
        "op_started",
        "op_parked",
        "op_completed",
        "op_unknown",
        "context_compacted",
        "usage_observed",
        "turn_ended",
    ],
)
def test_record_roundtrip(kind):
    rec = record(kind)
    assert Record.parse(rec.encode()) == rec
    assert rec.encode() == canonical_json_bytes(rec.to_dict())
    assert rec.digest == digest(rec.to_dict())


@pytest.mark.parametrize(
    "mutation", ["kind", "field", "payload_field", "version", "missing_version", "bool_version", "id", "digest"]
)
def test_strict_records(mutation):
    value = record("op_planned").to_dict()
    if mutation == "kind":
        value["kind"] = "future"
    elif mutation == "field":
        value["extra"] = 1
    elif mutation == "payload_field":
        value["payload"]["extra"] = 1
    elif mutation == "missing_version":
        del value["schema_version"]
    elif mutation == "id":
        value["record_id"] = "random-result-hash"
    elif mutation == "digest":
        value["payload"]["request_digest"] = "0" * 64
    else:
        value["schema_version"] = True if mutation == "bool_version" else 2
    with pytest.raises(RecordError):
        Record.parse(json.dumps(value).encode())


def test_inbox_closed_and_duplicate_json_keys():
    assert InboxItem.parse(item().encode()) == item()
    for key, val in [("kind", "other"), ("schema_version", 0), ("extra", True)]:
        with pytest.raises(RecordError):
            InboxItem.parse(canonical_json_bytes(item().to_dict() | {key: val}))
    with pytest.raises(RecordError):
        InboxItem.parse(item().encode().replace(b'"schema_version":2', b'"schema_version":2,"schema_version":2'))
    with pytest.raises(RecordError):
        replace(item(), payload={"content": "x", "extra": 1}).encode()


@pytest.mark.parametrize(
    "tail",
    [
        [record("op_started", dispatch_id="another")],
        [record("op_unknown"), record("op_started", dispatch_id="another")],
        [record("op_parked"), record("op_unknown")],
        [record("op_completed"), record("op_unknown")],
        [record("op_planned", attempt=2)],
    ],
)
def test_illegal_transitions(tail):
    with pytest.raises(TransitionError):
        fold(base() + tail)


@pytest.mark.parametrize("prefix", [[], [record("op_unknown")], [record("op_parked")]])
def test_late_result_closes_original_attempt(prefix):
    state = fold(base() + prefix + [record("op_completed")])
    assert state.operations["o"].attempts[1].state == "completed"
    assert state.operations["o"].selected_attempt == 1
    assert state.operations["o"].attempts[1].context == "normal"


def test_unknown_consumed_requires_correction():
    log = [
        *base(),
        record("op_unknown"),
        record("op_planned", oid="next", consumed_unknowns=[{"operation_id": "o", "attempt": 1}]),
    ]
    with pytest.raises(TransitionError):
        fold([*log, record("op_completed")])
    state = fold([*log, record("op_completed", context="correction")])
    assert state.operations["o"].attempts[1].context == "correction"


@pytest.mark.parametrize("started", [False, True])
def test_model_attempt_race(started):
    log = [*base(), record("op_unknown"), record("op_planned", attempt=2)]
    if started:
        log += [record("op_started", attempt=2)]
    state = fold([*log, record("op_completed", context="audit" if started else "normal")])
    op = state.operations["o"]
    assert op.selected_attempt == (None if started else 1)
    assert op.attempts[2].state == ("started" if started else "completed")
    if not started:
        with pytest.raises(TransitionError):
            fold([*log, record("op_completed"), record("op_started", attempt=2)])


@pytest.mark.parametrize("ending", ["cancel", "ended", "generation"])
def test_terminal_late_result_is_audit(ending):
    log = base()
    consumed = ()
    if ending == "cancel":
        control = item(kind="control", target_turn_id="t", generation=1)
        log += [record("input_applied", input=control.to_dict(), input_digest=digest(control.to_dict()))]
        consumed = (control,)
    else:
        log += [record("turn_ended", unconfirmed_operations=["o"])]
        if ending == "generation":
            log += [record("turn_started", tid="t2", generation=2)]
    state = fold([*log, record("op_completed", context="audit")], consumed_inputs=consumed)
    assert state.operations["o"].selected_attempt is None
    assert state.active_turn_id == ("t2" if ending == "generation" else "t" if ending == "cancel" else None)


@pytest.mark.parametrize("kw", [{"provider_binding": "wrong"}, {"request_digest": "0" * 64}, {"attempt": 2}, {"oid": "wrong"}])
def test_mismatched_result_rejected(kw):
    with pytest.raises(TransitionError):
        fold([*base(), record("op_completed", **kw)])


def test_parked_provider_evidence_and_duplicate_result():
    log = [*base(), record("op_parked")]
    with pytest.raises(TransitionError):
        fold([*log, record("op_completed", evidence=[])])
    done = record("op_completed")
    assert fold([*log, done, done]).operations["o"].selected_attempt == 1
    with pytest.raises(TransitionError):
        fold([*log, done, record("op_completed", result={"ok": False}, result_digest=digest({"ok": False}))])


def test_controls_require_consumed_input_and_matching_generation():
    control = item(kind="control", target_turn_id="t", generation=9)
    applied = record("input_applied", input=control.to_dict(), input_digest=digest(control.to_dict()))
    with pytest.raises(TransitionError):
        fold([*base(), applied])
    with pytest.raises(TransitionError):
        fold([*base(), applied], consumed_inputs=(control,))


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("user", {"content": "hello"}),
        ("steer", {"content": "continue"}),
        ("follow_up", {"content": "later"}),
        ("control", {"action": "archive"}),
        (
            "approval_answer",
            {
                "operation_id": "o",
                "attempt": 1,
                "request_id": "q",
                "request_digest": digest({}),
                "decision": "approve",
                "scope": [],
            },
        ),
        (
            "provider_result",
            {
                "operation_id": "o",
                "attempt": 1,
                "request_digest": digest({}),
                "provider_binding": "p",
                "result": {},
                "usage": {},
                "evidence": ["receipt"],
            },
        ),
        (
            "child_result",
            {
                "session_id": "child",
                "turn_id": "ct",
                "operation_id": "o",
                "attempt": 1,
                "result": {},
                "terminal_seq": 1,
                "terminal_digest": digest({}),
                "status": "completed",
            },
        ),
        (
            "provider_evidence",
            {
                "operation_id": "o",
                "attempt": 1,
                "request_digest": digest({}),
                "handle": {
                    "kind": "provider",
                    "provider": "p",
                    "job_id": "j",
                    "operation_id": "o",
                    "attempt": 1,
                    "request_digest": digest({}),
                    "evidence": "e",
                    "query_ref": None,
                    "cancel_ref": None,
                },
            },
        ),
    ],
)
def test_all_inbox_variants_closed(kind, payload):
    incoming = InboxItem("stable-source-id", kind, payload)
    assert InboxItem.parse(incoming.encode()) == incoming
    with pytest.raises(RecordError):
        replace(incoming, payload=payload | {"unexpected": None}).encode()
    with pytest.raises(RecordError):
        replace(incoming, payload={}).encode()


@pytest.mark.parametrize("codec,current,stale", [(Record, 1, 2), (InboxItem, 2, 1)])
@pytest.mark.parametrize("value", [1.0, True, "1", None, -1, 0, "stale"])
def test_version_requires_exact_integer(codec, current, stale, value):
    wire = record("session_created").to_dict() if codec is Record else item().to_dict()
    assert wire["schema_version"] == current
    with pytest.raises(RecordError):
        codec.parse(json.dumps(wire | {"schema_version": stale if value == "stale" else value}).encode())


@pytest.mark.parametrize("action", ["close", "archive"])
def test_session_control_while_idle(action):
    incoming = item(kind="control", action=action)
    applied = replace(record("input_applied", input=incoming.to_dict(), input_digest=digest(incoming.to_dict())), turn_id=None)
    state = fold([record("session_created"), applied], consumed_inputs=(incoming,))
    assert state.closed == (action == "close")
    assert state.archived == (action == "archive")
    assert state.phase == ("closed" if action == "close" else "idle")


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_approval_answer_schedules_resolution_and_enforces_permission(decision):
    planned = record("op_planned", op_kind="tool", purpose=None)
    handle = {
        "kind": "approval",
        "request_id": "approval",
        "request_digest": planned.payload["request_digest"],
        "scope": ["write"],
    }
    parked = record("op_parked", phase="before_dispatch", handle=handle)
    incoming = InboxItem(
        "answer",
        "approval_answer",
        {
            "operation_id": "o",
            "attempt": 1,
            "request_id": "approval",
            "request_digest": planned.payload["request_digest"],
            "decision": decision,
            "scope": ["write"],
        },
        target_turn_id="t",
        generation=1,
    )
    applied = record(
        "input_applied",
        input=incoming.to_dict(),
        input_digest=digest(incoming.to_dict()),
        target_operation_id="o",
        target_wait_id="approval",
    )
    log = [record("session_created"), record("turn_started"), planned, parked, applied]
    state = fold(log, consumed_inputs=(incoming,))
    assert state.next_drive_ms == 0
    if decision == "deny":
        with pytest.raises(TransitionError):
            fold([*log, record("op_started")], consumed_inputs=(incoming,))
        assert (
            fold([*log, record("op_completed", execution_started=False)], consumed_inputs=(incoming,)).operations["o"].state
            == "completed"
        )
    else:
        assert fold([*log, record("op_started")], consumed_inputs=(incoming,)).operations["o"].state == "started"


def test_waiting_dependency_does_not_spin():
    log = [
        record("session_created"),
        record("turn_started"),
        record("op_planned", op_kind="interaction", purpose=None),
        record("op_parked", phase="before_dispatch", handle={"kind": "user", "interaction_id": "q", "question": "?"}),
        record("op_planned", oid="dependent", dependencies=["o"]),
    ]
    state = fold(log)
    assert state.phase == "parked" and state.next_drive_ms is None
    with pytest.raises(TransitionError):
        fold([*log, record("op_started", oid="dependent")])


def test_cancel_cannot_end_successfully_or_dispatch():
    incoming = item(kind="control", action="cancel", target_turn_id="t", generation=1)
    applied = record("input_applied", input=incoming.to_dict(), input_digest=digest(incoming.to_dict()))
    log = [*base(), applied]
    with pytest.raises(TransitionError):
        fold([*log, record("turn_ended", unconfirmed_operations=["o"])], consumed_inputs=(incoming,))
    with pytest.raises(TransitionError):
        fold([*log, record("op_planned", oid="new")], consumed_inputs=(incoming,))
    state = fold([*log, record("turn_ended", status="cancelled", unconfirmed_operations=["o"])], consumed_inputs=(incoming,))
    assert state.active_turn_id is None and state.terminal_seq == len(log) + 1


def test_queued_follow_up_remains_runnable_after_turn_ends():
    incoming = item(kind="follow_up", target_turn_id=None)
    applied = record("input_applied", input=incoming.to_dict(), input_digest=digest(incoming.to_dict()), disposition="queued")
    log = [record("session_created"), record("turn_started"), applied, record("turn_ended")]
    state = fold(log, consumed_inputs=(incoming,))
    assert state.active_turn_id is None and state.next_drive_ms == 0
    state = fold([*log, record("turn_started", tid="t2", input_ids=["i"])], consumed_inputs=(incoming,))
    assert state.active_turn_id == "t2"
    with pytest.raises(TransitionError):
        fold(
            [
                *log,
                record("turn_started", tid="t2", input_ids=["i"]),
                record("turn_ended", tid="t2"),
                record("turn_started", tid="t3", input_ids=["i"]),
            ],
            consumed_inputs=(incoming,),
        )


def test_unstarted_operation_requires_explicit_closure_before_terminal():
    with pytest.raises(TransitionError):
        fold(
            [
                record("session_created"),
                record("turn_started"),
                record("op_planned"),
                record("turn_ended", unconfirmed_operations=["o"]),
            ]
        )


def test_duplicate_consumed_input_identity_cannot_overwrite():
    incoming = item()
    with pytest.raises(TransitionError):
        fold(
            [record("session_created"), record("input_applied")],
            consumed_inputs=(replace(incoming, payload={"content": "other"}), incoming),
        )


def test_started_attempt_retains_dispatch_identity_for_repair():
    attempt = fold(base()).operations["o"].attempts[1]
    assert attempt.dispatch == record("op_started")


@pytest.mark.parametrize("field", ["request_digest", "result_digest", "source_digest", "terminal_digest"])
def test_hash_rejects_trailing_newline(field):
    from vv_agent.session.records import INPUT_PAYLOADS, PAYLOADS, validate

    if field == "terminal_digest":
        schema = INPUT_PAYLOADS["child_result"]
    else:
        schema = PAYLOADS[
            {"request_digest": "op_started", "result_digest": "op_completed", "source_digest": "context_compacted"}[field]
        ]
        if field == "request_digest":
            schema = PAYLOADS["op_planned"]
    hash_schema = schema["properties"][field]
    from vv_agent.session.records import RecordError, _compile_check

    assert not _compile_check(hash_schema)("a" * 64 + "\n")
    # Exercise both fast and diagnostic validator paths on real envelopes.
    if field == "request_digest":
        original = record("op_planned")
    elif field == "result_digest":
        original = record("op_completed")
    elif field == "source_digest":
        original = record("context_compacted")
    else:
        from vv_agent.session.records import InboxItem

        payload = {
            "session_id": "child",
            "turn_id": "turn",
            "operation_id": "op",
            "attempt": 1,
            "result": {},
            "status": "completed",
            "terminal_seq": 1,
            "terminal_digest": "a" * 64 + "\n",
        }
        with pytest.raises(RecordError):
            InboxItem("result", "child_result", payload).encode()
        return
    value = original.to_dict()
    value["payload"][field] += "\n"
    with pytest.raises(RecordError):
        Record(**value).encode()
    with pytest.raises(RecordError):
        validate(value["payload"], schema)


def test_compaction_identity_omits_absent_summary_segment():
    compact = record("context_compacted")
    assert compact.record_id == f"compact/{compact.payload['source_digest']}/micro"
    summary = record("context_compacted", mode="summary", summary_operation_id="summary/op")
    assert summary.record_id == f"compact/{summary.payload['source_digest']}/summary/summary/op"


@pytest.mark.parametrize("seed", [{"messages": []}, {"messages": [], "shared_state": {}, "extra": True}])
def test_creation_seed_is_closed(seed):
    from vv_agent.session.records import RecordError, SessionSpec

    with pytest.raises(RecordError):
        SessionSpec("seed", "test", ".", attributes={"seed": seed}).record()
