"""S1 provider and child yields expose logical values to host callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.session import children
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.providers import Accepted, Unknown
from vv_agent.session.records import InboxItem, SessionSpec, record_identity
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_children import child_task

MASTER_KEY = b"y" * 32
PROVIDER_SESSION = "provider-yield"
CHILD_SESSION = "parent-yield"
JOB_ID = "provider/job-secret"
PROVIDER_EVIDENCE = "provider/evidence-secret"
PROVIDER_EVIDENCE_2 = "provider/evidence-second-secret"
QUERY_REF = "provider/query-secret"
PROVIDER_RENDERED = "provider callback rendered"
CHILD_RESULT = "child logical result"
CHILD_RENDERED = "child callback rendered"


@function_tool
def provider_effect() -> str:
    raise AssertionError("the provider owns execution")


class _YieldProvider:
    def __init__(self, evidence: str | list[str] = PROVIDER_EVIDENCE) -> None:
        self.handle: dict[str, Any] | None = None
        self.yield_handles: list[dict[str, Any]] = []
        self.on_yield = self._render
        self.cancels = 0
        self.evidence = evidence

    def submit(self, plan, *, context):
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": JOB_ID,
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": self.evidence,
            "query_ref": QUERY_REF,
            "cancel_ref": None,
        }
        return Accepted(self.handle, yield_on_steer=True)

    def query(self, handle):
        return Accepted(handle, yield_on_steer=True)

    def cancel(self, handle):
        self.cancels += 1
        return Unknown("not cancelled")

    def authenticate(self, item, plan):
        return item.kind in {"provider_result", "provider_evidence"}

    def _render(self, plan, handle, reason):
        assert reason == "steer"
        self.yield_handles.append(handle)
        assert handle == {
            **(self.handle or {}),
            "job_id": JOB_ID,
            "evidence": self.evidence,
            "query_ref": QUERY_REF,
        }
        assert declared_content_ref(handle["job_id"]) is None
        assert declared_content_ref(handle["evidence"]) is None
        assert declared_content_ref(handle["query_ref"]) is None
        return ToolExecutionResult(plan.payload["request"]["id"], PROVIDER_RENDERED)


def _key_provider() -> LocalKeyProvider:
    return LocalKeyProvider(MASTER_KEY)


def _provider_runtime(sessions: SessionDriver, workspace: Path, provider: _YieldProvider):
    def model(request):
        if any(message.role == "tool" for message in request.messages):
            return LLMResponse("done")
        return LLMResponse("", [ToolCall("effect-call", "provider_effect", {})])

    return sessions.runtime(
        Agent("provider-yield", "Wait for an external effect.", tools=[provider_effect]),
        RunConfig(
            model_provider=ScriptedModelProvider.from_callback("test", "provider-yield-model", model),
            workspace=workspace,
            tool_providers={"provider_effect": provider},
        ),
        session_id=PROVIDER_SESSION,
    )


def _logical_payload(store, session_id: str, stored) -> dict[str, Any]:
    return hydrate_record_payload(stored, hydration_for_session_store(store, session_id))


def _persisted_skeletons(store, session_ids: tuple[str, ...]) -> bytes:
    placeholders = ",".join("%s" for _ in session_ids)
    bodies = [
        row[0]
        for row in store._rows(
            f"SELECT body FROM sk_record WHERE tenant_id=%s AND session_id IN ({placeholders})",
            ("default", *session_ids),
        )
    ]
    bodies.extend(
        row[0]
        for row in store._rows(
            f"SELECT body FROM sk_inbox WHERE tenant_id=%s AND session_id IN ({placeholders})",
            ("default", *session_ids),
        )
    )
    return b"\n".join(bodies)


def _yield_from_raw_driver(store, session_id: str, runtime, item: InboxItem):
    lease = store.acquire(session_id, owner="raw-yield-regression", ttl_ms=15_000)
    assert lease is not None
    try:
        driver = _Driver(
            store,
            session_id,
            runtime,
            _Scope(lease, runtime, now_ms=store._now()),
        )
        # The public state view is hydrated, while yield resolution checks the retained raw fold.
        raw_fold = driver._fold_factory()
        driver.state = raw_fold.extend(
            (row.record for row in driver.records),
            consumed_inputs=(
                InboxItem(**row.record.payload["input"]) for row in driver.records if row.record.kind == "input_applied"
            ),
        )
        driver.fold = raw_fold
        operation = next(operation for operation in driver.state.operations.values() if operation.kind == "tool")
        return driver.yield_results([operation.attempts[1]], item)
    finally:
        assert store.release(lease)


@pytest.mark.parametrize(
    "provider_evidence",
    [PROVIDER_EVIDENCE, [PROVIDER_EVIDENCE, PROVIDER_EVIDENCE_2]],
    ids=["string-evidence", "list-evidence"],
)
def test_provider_yield_callback_and_receipt_use_owner_bound_values(tmp_path: Path, provider_evidence: str | list[str]) -> None:
    workspace = tmp_path / "provider-workspace"
    workspace.mkdir()
    sessions = SessionDriver(content_key_provider=_key_provider())
    provider = _YieldProvider(provider_evidence)
    try:
        sessions.create(PROVIDER_SESSION, str(workspace), {})
        sessions.push(PROVIDER_SESSION, InboxItem("initial", "user", {"content": "start"}))
        drive(sessions.store, PROVIDER_SESSION, runtime=_provider_runtime(sessions, workspace, provider))

        raw_parked = next(row.record for row in sessions.store.read(PROVIDER_SESSION).records if row.record.kind == "op_parked")
        raw_handle = raw_parked.payload["handle"]
        assert declared_content_ref(raw_handle["job_id"]) is not None
        assert declared_content_ref(raw_handle["evidence"]) is not None
        assert declared_content_ref(raw_handle["query_ref"]) is not None

        state = sessions._read_state(PROVIDER_SESSION)[0]
        turn_id = state.active_turn_id
        assert turn_id is not None
        attempt = next(operation for operation in state.operations.values() if operation.kind == "tool").attempts[1]
        assert attempt.wait is not None
        retained_handle = attempt.wait["handle"]
        assert retained_handle["job_id"] == JOB_ID
        assert retained_handle["evidence"] == provider_evidence
        assert retained_handle["query_ref"] == QUERY_REF
        assert declared_content_ref(retained_handle["job_id"]) is None
        assert declared_content_ref(retained_handle["evidence"]) is None
        assert declared_content_ref(retained_handle["query_ref"]) is None

        generation = state.turns[turn_id].start._payload["generation"]
        steer_item = InboxItem("steer", "steer", {"content": "change course"}, turn_id, generation)
        raw_records = _yield_from_raw_driver(
            sessions.store, PROVIDER_SESSION, _provider_runtime(sessions, workspace, provider), steer_item
        )
        raw_completion = next(record for record in raw_records if record.kind == "op_completed")
        assert raw_completion.payload["yielded_by"] == "steer"
        assert raw_completion.record_id == record_identity(
            raw_completion.kind,
            raw_completion.session_id,
            raw_completion.turn_id,
            raw_completion.operation_id,
            raw_completion.attempt,
            raw_completion.payload,
        )
        assert len(provider.yield_handles) == 1
        assert provider.yield_handles[0]["job_id"] == JOB_ID

        sessions.push(PROVIDER_SESSION, steer_item)
        drive(sessions.store, PROVIDER_SESSION, runtime=_provider_runtime(sessions, workspace, provider))

        receipt = next(
            row
            for row in sessions.store.read(PROVIDER_SESSION).records
            if row.record.kind == "op_completed" and "yielded_by" in row.record.payload
        )
        assert declared_content_ref(receipt.record.payload["result"]) is not None
        assert declared_content_ref(receipt.record.payload["evidence"]) is not None
        assert attempt.wait["handle"] == retained_handle

        logical = _logical_payload(sessions.store, PROVIDER_SESSION, receipt)
        assert logical["result"]["content"] == PROVIDER_RENDERED
        expected_evidence = [provider_evidence] if isinstance(provider_evidence, str) else provider_evidence
        assert logical["evidence"] == ["steer/steer", *expected_evidence]

        persisted = _persisted_skeletons(sessions.store, (PROVIDER_SESSION,))
        for value in (JOB_ID, *expected_evidence, QUERY_REF, PROVIDER_RENDERED, "change course"):
            assert value.encode() not in persisted
        assert provider.cancels == 0
    finally:
        sessions.close()


def _created_principal(store, session_id: str) -> dict[str, Any]:
    created = store.read(session_id, limit=1).records[0]
    return created.record.payload["principal"]


def _parent_runtime(
    sessions: SessionDriver,
    workspace: Path,
    principal: dict[str, Any],
    *,
    callback,
):
    requests = []

    def model(request):
        requests.append(request)
        if any(message.role == "tool" for message in request.messages):
            return LLMResponse("parent done")
        return LLMResponse("", [ToolCall("child-call", "child_task", {})])

    def admit(_plan):
        workspace_ref = sessions.store.read(CHILD_SESSION, limit=1).records[0].record.payload["workspace"]
        return [
            children.ChildSession(
                SessionSpec(f"{CHILD_SESSION}/child-{index}", principal, workspace_ref, attributes={}),
                f"work {index}",
                yield_on_steer=True,
            )
            for index in range(2)
        ]

    runtime = sessions.runtime(
        Agent("parent-yield", "Wait for children.", tools=[child_task]),
        RunConfig(
            model_provider=ScriptedModelProvider.from_callback("test", "parent-yield-model", model),
            workspace=workspace,
        ),
        session_id=CHILD_SESSION,
    )
    runtime.children["child_task"] = children.ChildTool(admit, on_yield=callback)
    return runtime, requests


def _child_runtime(sessions: SessionDriver, session_id: str, workspace: Path):
    def model(_request):
        return LLMResponse(CHILD_RESULT)

    return sessions.runtime(
        Agent("child-yield", "Finish the child.", model="child"),
        RunConfig(
            model_provider=ScriptedModelProvider.from_callback("test", "child-yield-model", model),
            workspace=workspace,
        ),
        session_id=session_id,
    )


def test_child_yield_callback_and_receipt_use_owner_bound_completion(tmp_path: Path) -> None:
    workspace = tmp_path / "child-workspace"
    workspace.mkdir()
    sessions = SessionDriver(content_key_provider=_key_provider())
    callback_handles: list[list[dict[str, Any]]] = []

    def render(plan, handles, reason):
        assert reason == "steer"
        callback_handles.append(handles)
        assert handles[0]["status"] == "completed"
        assert handles[0]["result"] == CHILD_RESULT
        assert declared_content_ref(handles[0]["result"]) is None
        assert handles[1]["status"] == "running"
        return ToolExecutionResult(plan.payload["request"]["id"], CHILD_RENDERED)

    try:
        sessions.create(CHILD_SESSION, str(workspace), {})
        sessions.push(CHILD_SESSION, InboxItem("initial", "user", {"content": "start"}))
        principal = _created_principal(sessions.store, CHILD_SESSION)
        runtime, requests = _parent_runtime(sessions, workspace, principal, callback=render)
        drive(sessions.store, CHILD_SESSION, runtime=runtime)

        first_child = f"{CHILD_SESSION}/child-0"
        drive(sessions.store, first_child, runtime=_child_runtime(sessions, first_child, workspace))
        with sessions.store.atomic() as tx:
            children.child_delivery(sessions.store, tx, first_child)

        state = sessions.store.read_state(CHILD_SESSION)[0]
        turn_id = state.active_turn_id
        assert turn_id is not None
        generation = state.turns[turn_id].start._payload["generation"]
        steer_item = InboxItem("steer", "steer", {"content": "change course"}, turn_id, generation)
        _yield_from_raw_driver(sessions.store, CHILD_SESSION, runtime, steer_item)
        assert len(callback_handles) == 1

        sessions.push(CHILD_SESSION, steer_item)
        drive(sessions.store, CHILD_SESSION, runtime=runtime)

        assert len(callback_handles) == 2
        assert any(CHILD_RENDERED in str(request.messages) for request in requests)
        receipt = next(
            row
            for row in sessions.store.read(CHILD_SESSION).records
            if row.record.kind == "op_completed" and "yielded_by" in row.record.payload
        )
        assert declared_content_ref(receipt.record.payload["result"]) is not None
        assert _logical_payload(sessions.store, CHILD_SESSION, receipt)["result"]["content"] == CHILD_RENDERED

        child_result = next(
            row
            for row in sessions.store.read(CHILD_SESSION).records
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["kind"] == "child_result"
            and row.record.payload["input"]["payload"]["session_id"] == first_child
        )
        assert declared_content_ref(child_result.record.payload["input"]["payload"]["result"]) is not None

        persisted = _persisted_skeletons(sessions.store, (CHILD_SESSION, first_child))
        for value in (CHILD_RESULT, CHILD_RENDERED, "change course"):
            assert value.encode() not in persisted
    finally:
        sessions.close()
