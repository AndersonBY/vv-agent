"""Provider callbacks survive an S1 content-runtime restart."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._sql import stored_input_proof
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Accepted, Unknown
from vv_agent.session.records import InboxItem, Record
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

MASTER_KEY = b"p" * 32
SESSION_ID = "provider-lifecycle"
RESULT_SECRET = "s1-provider-result-secret"


@function_tool
def provider_effect() -> str:
    raise AssertionError("the provider owns execution")


class _Provider:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result
        self.handle: dict | None = None
        self.authentications: list[tuple[InboxItem, InboxItem | None, Record]] = []

    def submit(self, plan, *, context):
        self.result = ToolExecutionResult(tool_call_id=plan.payload["request"]["id"], content=RESULT_SECRET).to_dict()
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "job/options",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "accepted/receipt",
            "query_ref": "query/options",
            "cancel_ref": None,
        }
        return Accepted(self.handle)

    def query(self, handle):
        return Accepted(handle)

    def cancel(self, handle):
        return Unknown("not cancelled")

    def authenticate(self, item, plan):
        self.authentications.append((item, getattr(item, "retained_item", None), plan))
        if item.kind == "provider_evidence":
            return item.payload["handle"] == self.handle
        return item.payload["result"] == self.result and item.payload["evidence"] == ["accepted/receipt"]


def _runtime(sessions: SessionDriver, *, workspace: Path, provider: _Provider):
    def model(request):
        if any(message.role == "tool" for message in request.messages):
            return LLMResponse("done")
        return LLMResponse("", [ToolCall("effect-call", "provider_effect", {})])

    config = RunConfig(
        model_provider=ScriptedModelProvider.from_callback("test", "provider-lifecycle-model", model),
        workspace=workspace,
        tool_providers={"provider_effect": provider},
    )
    return sessions.runtime(
        Agent("provider-lifecycle", "Wait for an external effect.", tools=[provider_effect]),
        config,
        session_id=SESSION_ID,
    )


def _park(
    path: Path,
    *,
    workspace: Path,
    provider: _Provider,
) -> tuple[Record, dict]:
    sessions = SessionDriver(path, content_key_provider=_key_provider())
    try:
        sessions.create(SESSION_ID, str(workspace), {})
        sessions.push(SESSION_ID, InboxItem("initial", "user", {"content": "start"}))
        drive(sessions.store, SESSION_ID, runtime=_runtime(sessions, workspace=workspace, provider=provider))

        state = sessions.store.read_state(SESSION_ID)[0]
        operation = next(operation for operation in state.operations.values() if operation.kind == "tool")
        attempt = operation.attempts[1]
        assert attempt.state == "parked"
        assert attempt.wait is not None
        assert declared_content_ref(attempt.wait["handle"]["job_id"]) is not None
        assert declared_content_ref(attempt.wait["handle"]["evidence"]) is not None
        parked = next(row.record for row in sessions.store.read(SESSION_ID).records if row.record.kind == "op_parked")
        assert "job/options" not in parked.encode().decode()
        assert "accepted/receipt" not in parked.encode().decode()
        return attempt.execution_plan, attempt.wait["handle"]
    finally:
        sessions.close()


def _key_provider():
    from vv_agent.session.content_store import LocalKeyProvider

    return LocalKeyProvider(MASTER_KEY)


def test_s1_logical_accepted_is_parked_with_reference_form_handle(tmp_path: Path) -> None:
    workspace = tmp_path / "accepted-workspace"
    workspace.mkdir()
    provider = _Provider()

    _, retained_handle = _park(tmp_path / "accepted.sqlite", workspace=workspace, provider=provider)

    assert provider.handle is not None
    assert declared_content_ref(retained_handle["job_id"]) is not None
    assert declared_content_ref(retained_handle["evidence"]) is not None
    assert declared_content_ref(retained_handle["query_ref"]) is not None


@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_s1_parked_provider_callbacks_rehydrate_after_restart(tmp_path: Path, kind: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = tmp_path / f"{kind}.sqlite"
    first = _Provider()
    plan, _raw_handle = _park(path, workspace=workspace, provider=first)
    assert first.handle is not None
    logical_handle = {
        **first.handle,
        "job_id": "job/options",
        "evidence": "accepted/receipt",
        "query_ref": "query/options",
    }

    sessions = SessionDriver(path, content_key_provider=_key_provider())
    provider = _Provider(result=first.result)
    provider.handle = logical_handle
    try:
        assert provider.result is not None
        payload = {
            "operation_id": first.handle["operation_id"],
            "attempt": first.handle["attempt"],
            "request_digest": first.handle["request_digest"],
        }
        if kind == "provider_evidence":
            payload["handle"] = logical_handle
        else:
            payload.update(
                provider_binding=first.handle["provider"],
                result=provider.result,
                usage={},
                evidence=["accepted/receipt"],
            )
        callback = InboxItem(
            kind,
            kind,
            payload,
            target_turn_id=plan.turn_id,
            generation=_state_generation(sessions.store, plan.turn_id),
        )
        sessions.push(SESSION_ID, callback)
        stored = sessions.store.read_input(SESSION_ID, kind)
        assert stored is not None
        drive(sessions.store, SESSION_ID, runtime=_runtime(sessions, workspace=workspace, provider=provider))

        state, records, _ = sessions.store.read_state(SESSION_ID)
        operation = next(operation for operation in state.operations.values() if operation.kind == "tool")
        attempt = operation.attempts[1]
        assert provider.authentications
        logical, retained, authenticated_plan = provider.authentications[0]
        assert retained is not None
        if kind == "provider_evidence":
            assert logical.payload["handle"] == logical_handle
            assert declared_content_ref(retained.payload["handle"]["job_id"]) is not None
        else:
            assert logical.payload["result"] == provider.result
            assert declared_content_ref(retained.payload["result"]) is not None
        assert authenticated_plan == attempt.execution_plan

        applied = next(
            row.record
            for row in records
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == kind
        )
        assert applied.payload["input_digest"] == stored.item.digest
        assert applied.payload["input_proof"] == stored_input_proof(stored)
        if kind == "provider_evidence":
            assert attempt.state == "parked"
            assert applied.payload["disposition"] == "noop"
            assert state.active_turn_id == SESSION_ID + "/turn/initial"
        else:
            assert attempt.state == "completed"
            assert applied.payload["disposition"] == "applied"
            assert state.active_turn_id is None

        replay = replace(callback, input_id=f"{kind}-replay")
        sessions.push(SESSION_ID, replay)
        drive(sessions.store, SESSION_ID, runtime=_runtime(sessions, workspace=workspace, provider=provider))
        state, records, _ = sessions.store.read_state(SESSION_ID)
        replay_receipt = next(
            row.record
            for row in records
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == replay.input_id
        )
        assert replay_receipt.payload["disposition"] == "noop"
        replay_attempt = next(operation for operation in state.operations.values() if operation.kind == "tool").attempts[1]
        assert replay_attempt.state == ("parked" if kind == "provider_evidence" else "completed")
    finally:
        sessions.close()


def _state_generation(store: SQLiteStore, turn_id: str | None) -> int:
    assert turn_id is not None
    return store.read_state(SESSION_ID)[0].turns[turn_id].start._payload["generation"]
