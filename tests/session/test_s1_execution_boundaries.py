"""Cold S1 execution boundaries hydrate only at the call site."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.events import ApprovalResolvedEvent, RunStateChangedEvent
from vv_agent.kernel._after_cycle import ReadCycleProjection
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, SourceKind
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._cycle_values import AfterCycleDecision
from vv_agent.kernel._record_schema import AFTER_CYCLE_GENERIC_STOP_CODE
from vv_agent.kernel._records import make_record
from vv_agent.kernel._state import Attempt, ExecutionState, Operation
from vv_agent.kernel._store import StoredRecord
from vv_agent.kernel._summary_requests import pending_compaction_requests
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store, project_records
from vv_agent.session.records import InboxItem
from vv_agent.session.reducer import Fold, fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, NoToolPolicy, ToolCall

from .helpers import content_ref as fixture_content_ref
from .helpers import record as fixture_record

MASTER_KEY = b"p2-session-boundary-key-00000000"
WORKSPACE = Path("/workspace")


def _runtime(
    sessions: SessionDriver,
    provider: ScriptedModelProvider,
    *,
    tools=(),
    no_tool_policy: NoToolPolicy = "finish",
    max_cycles: int = 8,
    after_cycle_hooks=(),
):
    return sessions.runtime(
        Agent("p2-boundary", "Exercise durable execution boundaries.", model="scripted", tools=list(tools)),
        RunConfig(
            model_provider=provider,
            workspace=WORKSPACE,
            no_tool_policy=no_tool_policy,
            max_cycles=max_cycles,
            after_cycle_hooks=list(after_cycle_hooks),
        ),
        session_id="s",
    )


def _raw_records(store: SQLiteStore):
    return store.read("s").records


def _event_payload(store: SQLiteStore, event_type: type):
    hydration = hydration_for_session_store(store, "s")
    return [event for event in project_records(_raw_records(store), hydration=hydration) if isinstance(event, event_type)]


def _assert_persisted_references_only(store: SQLiteStore, *plaintext: str) -> None:
    bodies = b"".join(row.record.encode() for row in _raw_records(store))
    for value in plaintext:
        assert value.encode() not in bodies


def test_after_cycle_host_stop_code_stays_in_encrypted_content(tmp_path: Path) -> None:
    path = tmp_path / "after-cycle-host-stop.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)
    host_code = "host.private_policy"
    detail = "Private host policy details remain diagnosable."

    class Hook:
        def after_cycle(self, _snapshot):
            return AfterCycleDecision.stop_non_success(code=host_code, message=detail)

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "start"}))
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("candidate")]),
            after_cycle_hooks=(Hook(),),
        )

        assert drive(sessions.store, "s", runtime=runtime)
        boundary_row = next(
            row
            for row in _raw_records(store)
            if row.record.kind == "boundary_recorded" and row.record.payload["stage"] == "after_cycle"
        )
        boundary = boundary_row.record
        data = boundary.payload["data"]
        assert data["stop"]["code"] == AFTER_CYCLE_GENERIC_STOP_CODE
        assert declared_content_ref(data["stop"]["message"]) is not None
        assert declared_content_ref(data["error"]) is not None
        _assert_persisted_references_only(store, host_code, detail)

        hydrated = hydrate_record_payload(boundary_row, hydration_for_session_store(store, "s"))
        hydrated_data = hydrated["data"]
        assert hydrated_data["stop"]["code"] == AFTER_CYCLE_GENERIC_STOP_CODE
        assert hydrated_data["stop"]["message"] == detail
        assert hydrated_data["error"] == f"{host_code}: {detail}"


def test_cold_s1_wait_user_projects_and_redrives_logical_question(tmp_path: Path) -> None:
    path = tmp_path / "wait-user.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)
    question = "Which deployment should run?"
    reply = "deploy the approved build"

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "start"}))
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse(question)]),
            no_tool_policy="wait_user",
        )

        assert drive(sessions.store, "s", runtime=runtime)
        parked = next(row for row in _raw_records(store) if row.record.kind == "turn_parked").record
        assert declared_content_ref(parked.payload["question"]) is not None
        turn_id = parked.turn_id
        assert turn_id is not None
        _assert_persisted_references_only(store, question)

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=reopened, content_key_provider=key_provider)
        waits = [event for event in _event_payload(reopened, RunStateChangedEvent) if event.state == "wait_user"]
        assert len(waits) == 1
        assert waits[0].metadata["interaction"]["prompt"] == question

        sessions.push(
            "s",
            InboxItem(
                "reply",
                "user",
                {"content": {"interaction_id": parked.payload["interaction_id"], "text": reply}},
                target_turn_id=turn_id,
                generation=0,
            ),
        )
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("finished")]),
            no_tool_policy="finish",
        )
        drive(sessions.store, "s", runtime=runtime)

        applied = next(
            row.record
            for row in _raw_records(reopened)
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "reply"
        )
        assert applied.payload["disposition"] == "applied"
        waits = [event for event in _event_payload(reopened, RunStateChangedEvent) if event.state == "wait_user"]
        assert [event.metadata["interaction"]["prompt"] for event in waits] == [question, "finished"]
        _assert_persisted_references_only(reopened, question, reply, "finished")


def test_cold_s1_approval_reply_hydrates_scope_reason_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "approval.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)
    effects: list[str] = []

    @function_tool(needs_approval=True)
    def guarded() -> str:
        effects.append("ran")
        return "allowed"

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "run guarded tool"}))
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new(
                "scripted",
                "scripted",
                [LLMResponse("", [ToolCall("guarded-call", "guarded", {})])],
            ),
            tools=(guarded,),
        )
        drive(sessions.store, "s", runtime=runtime)

        parked_row = next(row for row in _raw_records(store) if row.record.kind == "op_parked")
        parked = parked_row.record
        assert declared_content_ref(parked.payload["handle"]["scope"]) is not None
        turn_id = parked.turn_id
        assert turn_id is not None

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=reopened, content_key_provider=key_provider)
        hydration = hydration_for_session_store(reopened, "s")
        logical_wait = hydrate_record_payload(
            next(row for row in _raw_records(reopened) if row.record.kind == "op_parked"), hydration
        )
        handle = logical_wait["handle"]
        assert declared_content_ref(handle["scope"]) is None

        reason = "operator approved this exact request"
        metadata = {"ticket": "OPS-42", "origin": "cold-worker"}
        sessions.push(
            "s",
            InboxItem(
                "approval-reply",
                "approval_answer",
                {
                    "operation_id": parked.operation_id,
                    "attempt": parked.attempt,
                    "request_id": handle["request_id"],
                    "request_digest": handle["request_digest"],
                    "scope": handle["scope"],
                    "decision": "approve",
                    "reason": reason,
                    "metadata": metadata,
                },
                target_turn_id=turn_id,
                generation=0,
            ),
        )
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("approval completed output")]),
            tools=(guarded,),
        )
        drive(sessions.store, "s", runtime=runtime)

        assert effects == ["ran"]
        applied = next(
            row.record
            for row in _raw_records(reopened)
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "approval-reply"
        )
        assert declared_content_ref(applied.payload["input"]["payload"]["scope"]) is not None
        assert declared_content_ref(applied.payload["input"]["payload"]["reason"]) is not None
        assert declared_content_ref(applied.payload["input"]["payload"]["metadata"]) is not None
        approval = _event_payload(reopened, ApprovalResolvedEvent)[0]
        assert approval.action == "allow"
        assert approval.metadata["reason"] == reason
        assert approval.metadata["decision_metadata"] == metadata
        _assert_persisted_references_only(reopened, reason, "OPS-42", "cold-worker", "approval completed output")


class _Crash(Exception):
    pass


def test_external_model_exception_records_durable_unknown(tmp_path: Path) -> None:
    path = tmp_path / "after-cycle-unknown.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)
    calls = 0

    def crash_second(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse("first cycle")
        raise _Crash

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "continue cycles"}))
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.from_callback("scripted", "scripted", crash_second),
            no_tool_policy="continue",
            max_cycles=2,
        )
        drive(sessions.store, "s", runtime=runtime)
        unknown = [row.record for row in _raw_records(store) if row.record.kind == "op_unknown"]
        assert [record.payload["retry"] for record in unknown] == ["retry", "stop"]
        assert calls == 3


@pytest.mark.parametrize(
    ("raw_data", "logical_data"),
    [
        pytest.param(
            {
                "action": "steer",
                "steering_messages": [fixture_content_ref("steering")],
                "disallow_tools": ["write_file"],
                "stop": None,
                "shared_state": fixture_content_ref("shared"),
                "error": None,
            },
            {
                "action": "steer",
                "steering_messages": ["Keep the durable boundary reference intact."],
                "disallow_tools": ["write_file"],
                "stop": None,
                "shared_state": {"phase": "after_cycle"},
                "error": None,
            },
            id="steering",
        ),
        pytest.param(
            {
                "action": "stop_non_success",
                "steering_messages": [],
                "disallow_tools": [],
                "stop": {"code": "stopped", "message": fixture_content_ref("stop")},
                "shared_state": fixture_content_ref("shared"),
                "error": fixture_content_ref("stop-error"),
            },
            {
                "action": "stop_non_success",
                "steering_messages": [],
                "disallow_tools": [],
                "stop": {"code": "stopped", "message": "Stop requested."},
                "shared_state": {"phase": "after_cycle"},
                "error": "stopped: Stop requested.",
            },
            id="stop",
        ),
        pytest.param(
            {
                "action": "continue",
                "steering_messages": [],
                "disallow_tools": [],
                "stop": None,
                "shared_state": fixture_content_ref("shared"),
                "error": fixture_content_ref("hook-error"),
            },
            {
                "action": "continue",
                "steering_messages": [],
                "disallow_tools": [],
                "stop": None,
                "shared_state": {"phase": "after_cycle"},
                "error": "after_cycle_hook_failed: boom",
            },
            id="error",
        ),
    ],
)
def test_raw_after_cycle_boundary_defers_plaintext_validation_to_hydrated_fold(
    raw_data: dict,
    logical_data: dict,
) -> None:
    records = [
        fixture_record("session_created"),
        fixture_record("turn_started"),
        fixture_record("op_planned"),
        fixture_record("op_started"),
        fixture_record("op_completed"),
    ]
    boundary = make_record(
        "boundary_recorded",
        session_id="s",
        turn_id="t",
        payload={
            "stage": "after_cycle",
            "boundary_id": "o",
            "source_operation_id": "o",
            "source_digest": None,
            "data": raw_data,
        },
    )
    bodies = [*(record.encode() for record in records), boundary.encode()]

    raw_state = fold([*records, boundary])
    assert raw_state.boundaries[("t", "after_cycle", "o")]._payload["data"] == raw_data

    hydrated_boundary = replace(boundary, payload=boundary._payload | {"data": logical_data})
    hydrated_state = Fold().extend([*records, hydrated_boundary], bodies=bodies)
    assert hydrated_state.boundaries[("t", "after_cycle", "o")]._payload["data"] == logical_data


def test_hydrated_after_cycle_boundary_keeps_steering_bounded_text_validation() -> None:
    records = [
        fixture_record("session_created"),
        fixture_record("turn_started"),
        fixture_record("op_planned"),
        fixture_record("op_started"),
        fixture_record("op_completed"),
    ]
    raw_data = {
        "action": "steer",
        "steering_messages": [fixture_content_ref("steering")],
        "disallow_tools": [],
        "stop": None,
        "shared_state": fixture_content_ref("shared"),
        "error": None,
    }
    boundary = make_record(
        "boundary_recorded",
        session_id="s",
        turn_id="t",
        payload={
            "stage": "after_cycle",
            "boundary_id": "o",
            "source_operation_id": "o",
            "source_digest": None,
            "data": raw_data,
        },
    )
    bodies = [*(record.encode() for record in records), boundary.encode()]
    hydrated_boundary = replace(
        boundary,
        payload=boundary._payload
        | {
            "data": raw_data
            | {
                "steering_messages": [""],
                "shared_state": {"phase": "after_cycle"},
            }
        },
    )

    with pytest.raises(ValueError, match="after-cycle steering message must be a non-empty string"):
        Fold().extend([*records, hydrated_boundary], bodies=bodies)


def test_cold_after_cycle_redrive_commits_retained_steering_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cold-after-cycle-redrive.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)
    steering = "Keep the durable boundary reference intact."

    class _Pause(Exception):
        pass

    class ConfiguredHook:
        def after_cycle(self, snapshot):
            return AfterCycleDecision.continue_run()

    original_handle = _recovery_phase._Host.handle

    def pause_projection(host, request):
        if isinstance(request, ReadCycleProjection):
            raise _Pause
        return original_handle(host, request)

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "continue cycles"}))
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("first cycle")]),
            no_tool_policy="continue",
            max_cycles=2,
            after_cycle_hooks=(ConfiguredHook(),),
        )
        with monkeypatch.context() as patcher:
            patcher.setattr(_recovery_phase._Host, "handle", pause_projection)
            with pytest.raises(_Pause):
                drive(sessions.store, "s", runtime=runtime)
        assert not any(row.record.kind == "boundary_recorded" for row in _raw_records(store))

    boundary_key = ("", "after_cycle", "")
    with SQLiteStore.standalone(str(path)) as cold_store:
        cold_store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=cold_store, content_key_provider=key_provider)
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", []),
            no_tool_policy="continue",
            max_cycles=2,
        )
        lease = sessions.store.acquire("s", owner="cold-after-cycle", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=cold_store._now()))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        model_oid = next(
            oid
            for oid, operation in driver.state.operations.items()
            if operation.kind == "model"
            and operation.attempts[1].plan._payload["purpose"] == "primary"
            and operation.selected_attempt is not None
        )
        boundary = driver.boundary_record(
            "after_cycle",
            model_oid,
            {
                "action": "steer",
                "steering_messages": [steering],
                "disallow_tools": ["write_file"],
                "stop": None,
                "shared_state": {},
                "error": None,
            },
            source_operation_id=model_oid,
        )
        driver.commit([boundary])

        boundary_key = (turn_id, "after_cycle", model_oid)
        retained = driver.state.boundaries[boundary_key]
        retained_data = retained._payload["data"]
        assert declared_content_ref(retained_data["steering_messages"][0]) is not None
        assert declared_content_ref(retained_data["shared_state"]) is not None
        assert sessions.store.release(lease)

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(allow_unsigned=True)
        raw_state, raw_records, _ = reopened.read_state("s")
        retained = raw_state.boundaries[boundary_key]
        retained_data = retained._payload["data"]
        assert raw_state.active_turn_id == turn_id
        assert declared_content_ref(retained_data["steering_messages"][0]) is not None
        assert declared_content_ref(retained_data["shared_state"]) is not None
        assert any(row.record.record_id == retained.record_id for row in raw_records)
        bodies = b"".join(row.record.encode() for row in raw_records)
        assert steering.encode() not in bodies


def test_cold_s1_after_cycle_projection_hydrates_primary_values_and_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "after-cycle.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)

    class _Pause(Exception):
        pass

    class ConfiguredHook:
        def after_cycle(self, snapshot):
            return AfterCycleDecision.continue_run()

    original_handle = _recovery_phase._Host.handle

    def pause_projection(host, request):
        if isinstance(request, ReadCycleProjection):
            raise _Pause
        return original_handle(host, request)

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "continue cycles"}))
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("first cycle")]),
            no_tool_policy="continue",
            max_cycles=2,
            after_cycle_hooks=(ConfiguredHook(),),
        )
        with monkeypatch.context() as patcher:
            patcher.setattr(_recovery_phase._Host, "handle", pause_projection)
            with pytest.raises(_Pause):
                drive(sessions.store, "s", runtime=runtime)
        assert not any(row.record.kind == "boundary_recorded" for row in _raw_records(store))

    snapshots = []

    class Hook:
        def after_cycle(self, snapshot):
            snapshots.append(snapshot)
            return AfterCycleDecision.continue_run(disallow_tools=("write_file",))

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=reopened, content_key_provider=key_provider)
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("final cycle")]),
            no_tool_policy="continue",
            max_cycles=2,
            after_cycle_hooks=(Hook(),),
        )
        drive(sessions.store, "s", runtime=runtime)

        assert snapshots[0].cycle.assistant_message == "first cycle"
        assert any(message.content == "continue cycles" for message in snapshots[0].messages)
        boundaries = [row.record for row in _raw_records(reopened) if row.record.kind == "boundary_recorded"]
        after_cycle = next(record for record in boundaries if record.payload["stage"] == "after_cycle")
        data = after_cycle.payload["data"]
        assert data["steering_messages"] == []
        assert data["disallow_tools"] == ["write_file"]
        assert declared_content_ref(data["shared_state"]) is not None
        _assert_persisted_references_only(reopened, "first cycle", "final cycle")


class _RecordReader:
    def __init__(self, tenant_id: str = "tenant", session_id: str = "s") -> None:
        self.binding = (tenant_id, session_id)
        self.links: dict[tuple[str, str, str], bytes] = {}
        self.calls: list[tuple[str, str]] = []

    def add(self, owner: str, slot: str, ref: ContentRef, value) -> None:
        self.links[(owner, slot, ref.content_id)] = canonical_json_bytes(value)

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        assert (tenant_id, session_id) == self.binding
        self.calls.append((owner, slot))
        return self.links[(owner, slot, ref.content_id)]


@pytest.fixture
def narrow_summary_driver(tmp_path: Path) -> Iterator[tuple[_Driver, str]]:
    with SQLiteStore.standalone(str(tmp_path / "summary-intent.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        sessions.create("s", str(WORKSPACE), {"seed": {"messages": [], "shared_state": {}}})
        runtime = _runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", []),
        )
        lease = sessions.store.acquire("s", owner="summary-intent", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(
            sessions.store,
            "s",
            runtime,
            _Scope(lease, runtime, now_ms=store._now()),
        )
        try:
            yield driver, store.tenant_id
        finally:
            assert sessions.store.release(lease)


def test_s1_summary_intent_hydrates_only_the_completed_result_owner_slot(
    narrow_summary_driver: tuple[_Driver, str],
) -> None:
    result = {"status_code": "SUCCESS", "metadata": {"request_compaction": True}}
    body = canonical_json_bytes(result)
    result_ref = ContentRef(
        content_id="content/summary-result",
        mac=sha256(body).hexdigest(),
        size=len(body),
        key_id="key/test",
        labels=ContentLabels((SourceKind.INTERNAL,), Classification.INTERNAL),
    )
    raw = fixture_record(
        "op_completed",
        result=result_ref.to_dict(),
        result_digest=sha256(canonical_json_bytes(result_ref.to_dict())).hexdigest(),
    )
    stored = StoredRecord(raw, 1, "commit/1", 1, 1)
    logical = replace(raw, payload=raw._payload | {"result": result})
    attempt = Attempt(plan=raw, context="normal", result=logical)
    operation = Operation(
        turn_id="t",
        kind="tool",
        selected_attempt=1,
        attempts={1: attempt},
    )

    driver, tenant_id = narrow_summary_driver
    reader = _RecordReader(tenant_id, driver.sid)
    reader.add("record:1", "result", result_ref, result)
    driver.content_reader = reader
    driver._hydrators = {}
    driver.records = (stored,)
    driver._stored_by_id = {raw.record_id: stored}
    driver.state = ExecutionState(
        active_turn_id="t",
        operations={operation_id: operation for operation_id in ("o",)},
    )

    assert pending_compaction_requests(driver) == ["o"]
    assert reader.calls == [("record:1", "result")]
