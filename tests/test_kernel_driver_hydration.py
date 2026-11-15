"""Driver reads hydrate declared C fields from their stored record owners."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, ContentSessionWriter, SourceKind
from vv_agent.kernel._content_publication import publish_record_content
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import IntegrityFailure, MissingContentLink, ReferenceDigestMismatch, reference_digest
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import Record, record_identity
from vv_agent.kernel._store import StoredRecord
from vv_agent.session.kernel import _Driver


class _Reader:
    binding = ("tenant", "s")

    def __init__(self) -> None:
        self.links: dict[tuple[str, str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []

    def add(self, owner: str, slot: str, ref: ContentRef, value: Any) -> ContentRef:
        self.links[(owner, slot, ref.content_id)] = canonical_json_bytes(value)
        return ref

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        assert (tenant_id, session_id) == self.binding
        self.calls.append((owner, slot, ref.content_id))
        try:
            return self.links[(owner, slot, ref.content_id)]
        except KeyError:
            raise MissingContentLink("missing test link") from None


def _ref(label: str, value: Any) -> ContentRef:
    body = canonical_json_bytes(value)
    return ContentRef(
        content_id=f"content/{label}",
        mac=sha256(body).hexdigest(),
        size=len(body),
        key_id="key/session",
        labels=ContentLabels((SourceKind.INTERNAL,), Classification.INTERNAL),
    )


def _stored(seq: int, kind: str, payload: dict[str, Any], *, attempt: int = 1) -> StoredRecord:
    turn_id = "t"
    operation_id = "o" if kind.startswith("op_") else None
    attempt_value = attempt if kind.startswith("op_") else None
    return StoredRecord(
        Record(
            record_identity(kind, "s", turn_id, operation_id, attempt_value, payload),
            kind,
            "s",
            turn_id,
            operation_id,
            attempt_value,
            payload,
        ),
        seq,
        f"commit/{seq}",
        1,
        1,
    )


def _driver(rows: tuple[StoredRecord, ...], state: Any | None = None) -> tuple[CoreDriver, _Reader]:
    reader = _Reader()
    driver = object.__new__(CoreDriver)
    driver.sid = "s"
    driver.store = SimpleNamespace(tenant_id="tenant")
    driver.content_reader = reader
    driver._hydrators = {}
    driver.records = rows
    driver._stored_by_id = {row.record.record_id: row for row in rows}
    driver.state = state or SimpleNamespace(operations={})
    return driver, reader


def test_definition_and_session_attributes_hydrate_from_creation_owners() -> None:
    attributes = _ref("attributes-value", {"seed": {"unseen_input_protection": False}})
    created = _stored(
        1,
        "session_created",
        {
            "principal": {},
            "workspace": None,
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": attributes.to_dict(),
        },
    )
    task = AgentTask(
        task_id="t",
        model="scripted",
        prompt_bundle=PromptBundle((PromptSection("system", "Work."),)),
        user_prompt="go",
    )
    definition = {
        "agent_name": "agent",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": [],
            "context_length": None,
            "max_output_tokens": None,
        },
        "capabilities": {},
        "task": task.to_dict(),
        "tools": [],
    }
    definition_ref = _ref("definition-value", definition)
    started = _stored(
        2,
        "turn_started",
        {
            "input_ids": [],
            "definition": definition_ref.to_dict(),
            "definition_digest": reference_digest(definition_ref),
            "handler_version": "1",
            "budget": None,
            "binding": "task",
            "generation": 1,
        },
    )
    driver, reader = _driver((created, started))
    reader.add("record:1", "session.attributes", attributes, {"seed": {"unseen_input_protection": False}})
    reader.add("record:2", "definition", definition_ref, definition)

    assert driver.session_attributes() == {"seed": {"unseen_input_protection": False}}
    assert driver.hydrated_definition(started.record).task.user_prompt == "go"
    assert reader.calls == [
        ("record:1", "session.attributes", attributes.content_id),
        ("record:2", "definition", definition_ref.content_id),
    ]


def test_definition_and_session_attribute_nested_messages_hydrate_explicitly() -> None:
    initial = Message("user", "initial").to_dict()
    input_message = Message("user", "input").to_dict()
    seed = Message("user", "seed").to_dict()
    initial_ref = _ref("definition-initial", initial)
    input_ref = _ref("definition-input", input_message)
    seed_ref = _ref("seed-message", seed)
    task = AgentTask(
        task_id="t",
        model="scripted",
        prompt_bundle=PromptBundle((PromptSection("system", "Work."),)),
        user_prompt="go",
        initial_messages=[Message.from_dict(initial)],
        metadata={"vv_session": {"input_messages": [input_message]}},
    ).to_dict()
    task["initial_messages"] = [initial_ref.to_dict()]
    task["metadata"]["vv_session"]["input_messages"] = [input_ref.to_dict()]
    definition = {
        "agent_name": "agent",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": [],
            "context_length": None,
            "max_output_tokens": None,
        },
        "capabilities": {},
        "task": task,
        "tools": [],
    }
    definition_ref = _ref("nested-definition", definition)
    attributes_ref = _ref("nested-attributes", {"seed": {"messages": [seed_ref.to_dict()]}})
    created = _stored(
        1,
        "session_created",
        {
            "principal": {},
            "workspace": None,
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": attributes_ref.to_dict(),
        },
    )
    started = _stored(
        2,
        "turn_started",
        {
            "definition": definition_ref.to_dict(),
            "definition_digest": reference_digest(definition_ref),
            "handler_version": "1",
            "budget": None,
            "binding": "task",
            "generation": 1,
        },
    )
    driver, reader = _driver((created, started))
    reader.add("record:1", "session.attributes", attributes_ref, {"seed": {"messages": [seed_ref.to_dict()]}})
    reader.add("record:1", "session.attributes.seed.messages[0]", seed_ref, seed)
    reader.add("record:2", "definition", definition_ref, definition)
    reader.add("record:2", "definition.task.initial_messages[0]", initial_ref, initial)
    reader.add(
        "record:2",
        "definition.task.metadata.vv_session.input_messages[0]",
        input_ref,
        input_message,
    )

    attributes = driver.session_attributes()
    hydrated = driver.hydrated_definition(started.record)
    session_driver = object.__new__(_Driver)
    session_driver.sid = "s"
    session_driver.store = SimpleNamespace(tenant_id="tenant")
    session_driver.content_reader = reader
    session_driver._hydrators = {}
    session_driver.records = (created, started)
    session_driver._stored_by_id = {row.record.record_id: row for row in session_driver.records}
    session_driver.state = SimpleNamespace(
        active_turn_id="t",
        operations={},
        turns={"t": SimpleNamespace(start=started.record)},
    )
    session_hydrated = session_driver._hydrated_definition()

    assert attributes["seed"]["messages"] == [seed]
    assert [message.to_dict() for message in hydrated.task.initial_messages] == [initial]
    assert hydrated.task.metadata["vv_session"]["input_messages"] == [input_message]
    assert [message.to_dict() for message in session_hydrated.task.initial_messages] == [initial]
    assert session_hydrated.task.metadata["vv_session"]["input_messages"] == [input_message]
    assert created.record._payload["attributes"] == attributes_ref.to_dict()
    assert started.record._payload["definition"] == definition_ref.to_dict()

    before = list(reader.calls)
    mismatched = replace(started.record, payload=started.record._payload | {"definition_digest": "0" * 64})
    with pytest.raises(ReferenceDigestMismatch):
        driver.hydrated_definition(mismatched)
    assert reader.calls == before


class _PutWriter:
    def __init__(self) -> None:
        self.puts: list[Any] = []

    def put_json(self, value: Any, *, labels: Any = ...) -> ContentRef:
        self.puts.append(value)
        return _ref(f"put-{len(self.puts)}", value)


def test_plan_uses_prepared_record_as_owner_for_every_prepared_slot() -> None:
    request = _ref("plan-request", {"id": "call", "name": "tool", "source": "planned"})
    tool = _ref("plan-tool", {"idempotency": "unsupported", "source": "planned"})
    budget = _ref("plan-budget", {})
    prepared_request = _ref("prepared-request", {"id": "call", "name": "tool", "source": "prepared"})
    prepared_tool = _ref("prepared-tool", {"idempotency": "supported", "source": "prepared"})
    hook_result = _ref("hook-result", {"content": "short"})
    shared_state = _ref("prepared-shared", {"step": 2})
    plan = _stored(
        10,
        "op_planned",
        {
            "op_kind": "tool",
            "purpose": None,
            "request": request.to_dict(),
            "request_digest": "0" * 64,
            "context_version": "1",
            "dependencies": [],
            "tool": tool.to_dict(),
            "idempotency_key": None,
            "budget_admission": budget.to_dict(),
            "not_before_ms": None,
            "provider_binding": "function",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )
    prepared = _stored(
        11,
        "op_prepared",
        {
            "request": prepared_request.to_dict(),
            "request_digest": "1" * 64,
            "tool": prepared_tool.to_dict(),
            "op_kind": "tool",
            "provider_binding": "function",
            "idempotency_key": None,
            "hook_result": hook_result.to_dict(),
            "shared_state": shared_state.to_dict(),
            "policy_decision": None,
        },
    )
    synthetic = replace(
        plan.record,
        payload=plan.record._payload
        | {
            "request": prepared_request.to_dict(),
            "tool": prepared_tool.to_dict(),
            "budget_admission": {
                "hook_result": hook_result.to_dict(),
                "shared_state": shared_state.to_dict(),
            },
        },
    )
    attempt = SimpleNamespace(plan=plan.record, prepared=prepared.record)
    state = SimpleNamespace(operations={"o": SimpleNamespace(attempts={1: attempt})})
    driver, reader = _driver((plan, prepared), state)
    for owner, slot, ref, value in (
        ("record:11", "request", prepared_request, {"id": "call", "name": "tool", "source": "prepared"}),
        ("record:11", "tool", prepared_tool, {"idempotency": "supported", "source": "prepared"}),
        ("record:11", "hook_result", hook_result, {"content": "short"}),
        ("record:11", "shared_state", shared_state, {"step": 2}),
    ):
        reader.add(owner, slot, ref, value)

    hydrated = driver.hydrate_plan(synthetic)

    assert hydrated._payload["request"] == {"id": "call", "name": "tool", "source": "prepared"}
    assert hydrated._payload["tool"] == {"idempotency": "supported", "source": "prepared"}
    assert hydrated._payload["budget_admission"] == {
        "hook_result": {"content": "short"},
        "shared_state": {"step": 2},
    }
    assert {owner for owner, _slot, _content_id in reader.calls} == {"record:11"}
    assert {slot for _owner, slot, _content_id in reader.calls} == {"request", "tool", "hook_result", "shared_state"}


def test_canonical_plan_hydrates_prepared_hook_nested_children_overlay() -> None:
    request = _ref("planned-request", {"id": "call", "name": "tool"})
    tool = _ref("planned-tool", {"idempotency": "supported"})
    budget = _ref("planned-budget", {})
    prepared_request = _ref("prepared-request", {"id": "call", "name": "tool"})
    prepared_tool = _ref("prepared-tool", {"idempotency": "supported"})
    child = {"session_id": "child-a", "status": "running"}
    running = "child-a"
    child_ref = _ref("hook-child", child)
    running_ref = _ref("hook-running-child", running)
    opaque = {"content_id": "provider-selected", "detail": {"content_id": "opaque"}}
    hook_result = {
        "tool_call_id": "call",
        "content": "handled",
        "metadata": {
            "children": [child_ref.to_dict()],
            "running_children": [running_ref.to_dict()],
            "provider_observation": opaque,
        },
    }
    hook_ref = _ref("hook-result", hook_result)
    shared = _ref("prepared-shared", {"step": 2})
    plan = _stored(
        20,
        "op_planned",
        {
            "op_kind": "tool",
            "purpose": None,
            "request": request.to_dict(),
            "request_digest": "0" * 64,
            "context_version": "1",
            "dependencies": [],
            "tool": tool.to_dict(),
            "idempotency_key": None,
            "budget_admission": budget.to_dict(),
            "not_before_ms": None,
            "provider_binding": "function",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )
    prepared = _stored(
        21,
        "op_prepared",
        {
            "request": prepared_request.to_dict(),
            "request_digest": "1" * 64,
            "tool": prepared_tool.to_dict(),
            "op_kind": "tool",
            "provider_binding": "function",
            "idempotency_key": None,
            "hook_result": hook_ref.to_dict(),
            "shared_state": shared.to_dict(),
            "policy_decision": None,
        },
    )
    attempt = SimpleNamespace(plan=plan.record, prepared=prepared.record)
    state = SimpleNamespace(operations={"o": SimpleNamespace(attempts={1: attempt})})
    driver, reader = _driver((plan, prepared), state)
    for owner, slot, ref, value in (
        ("record:21", "request", prepared_request, {"id": "call", "name": "tool"}),
        ("record:21", "tool", prepared_tool, {"idempotency": "supported"}),
        ("record:21", "hook_result", hook_ref, hook_result),
        ("record:21", "hook_result.metadata.children[0]", child_ref, child),
        ("record:21", "hook_result.metadata.running_children[0]", running_ref, running),
        ("record:21", "shared_state", shared, {"step": 2}),
    ):
        reader.add(owner, slot, ref, value)

    hydrated = driver.hydrate_plan(plan.record)

    metadata = hydrated._payload["budget_admission"]["hook_result"]["metadata"]
    assert metadata["children"] == [child]
    assert metadata["running_children"] == [running]
    assert metadata["provider_observation"] == opaque
    assert plan.record._payload["request"] == request.to_dict()
    assert plan.record._payload["budget_admission"] == budget.to_dict()


def test_plan_hydrates_declared_request_messages_from_the_outer_record_owner() -> None:
    message = Message(role="user", content="nested plaintext").to_dict()
    message_ref = _ref("request-message", message)
    request_value = {
        "id": "call",
        "name": "model",
        "arguments": {"opaque": {"messages": [{"content_id": "not-a-declared-ref"}]}},
        "messages": [message_ref.to_dict()],
    }
    request = _ref("request-with-messages", request_value)
    tool = _ref("request-tool", {})
    budget = _ref("request-budget", {})
    plan = _stored(
        12,
        "op_planned",
        {
            "op_kind": "model",
            "purpose": None,
            "request": request.to_dict(),
            "request_digest": "0" * 64,
            "context_version": "1",
            "dependencies": [],
            "tool": tool.to_dict(),
            "idempotency_key": None,
            "budget_admission": budget.to_dict(),
            "not_before_ms": None,
            "provider_binding": "model",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )
    driver, reader = _driver((plan,))
    reader.add("record:12", "request", request, request_value)
    reader.add("record:12", "request.messages[0]", message_ref, message)
    reader.add("record:12", "tool", tool, {})
    reader.add("record:12", "budget_admission", budget, {})

    hydrated = driver.hydrate_plan(plan.record)

    request_value = hydrated._payload["request"]
    assert request_value["messages"] == [message]
    assert request_value["arguments"]["opaque"]["messages"] == [{"content_id": "not-a-declared-ref"}]
    assert Message.from_dict(request_value["messages"][0]) == Message.from_dict(message)
    assert plan.record._payload["request"] == request.to_dict()
    assert {owner for owner, _slot, _content_id in reader.calls} == {"record:12"}
    assert {slot for _owner, slot, _content_id in reader.calls} == {
        "request",
        "request.messages[0]",
        "tool",
        "budget_admission",
    }


def test_completed_unknown_and_terminal_fields_hydrate_with_nullable_none() -> None:
    result = _ref("result", {"content": "done", "tool_calls": []})
    usage = _ref("usage", {"prompt_tokens": 3})
    shared = _ref("shared", {"answer": 42})
    evidence = _ref("evidence", ["provider/receipt"])
    completed = _stored(
        12,
        "op_completed",
        {
            "result": result.to_dict(),
            "result_digest": "0" * 64,
            "usage": usage.to_dict(),
            "shared_state": shared.to_dict(),
            "evidence": evidence.to_dict(),
            "execution_started": True,
            "context": "normal",
            "request_digest": "1" * 64,
            "provider_binding": "function",
            "policy_decision": None,
        },
    )
    reason = _ref("unknown-reason", "worker lost")
    dispatch_evidence = _ref("dispatch-evidence", ["dispatch/1"])
    observation = _ref("observation", {"retryable": True})
    unknown = _stored(
        13,
        "op_unknown",
        {
            "reason": reason.to_dict(),
            "dispatch_evidence": dispatch_evidence.to_dict(),
            "observation": observation.to_dict(),
            "retry": "retry",
            "retry_at_ms": 0,
            "duplicate_cost_risk": True,
            "measurement_missing": True,
        },
    )
    terminal_reason = _ref("terminal-reason", "tool_finish")
    terminal_result = _ref("terminal-result", {"content": "done"})
    terminal_budget = _ref("terminal-budget", {"elapsed_ms": 1})
    terminal = _stored(
        14,
        "turn_ended",
        {
            "status": "completed",
            "reason": terminal_reason.to_dict(),
            "result": terminal_result.to_dict(),
            "adopted_results": [],
            "budget": terminal_budget.to_dict(),
            "unconfirmed_operations": [],
        },
    )
    driver, reader = _driver((completed, unknown, terminal))
    for owner, slot, ref, value in (
        ("record:12", "result", result, {"content": "done", "tool_calls": []}),
        ("record:12", "usage", usage, {"prompt_tokens": 3}),
        ("record:12", "shared_state", shared, {"answer": 42}),
        ("record:12", "evidence", evidence, ["provider/receipt"]),
        ("record:13", "op_unknown_reason", reason, "worker lost"),
        ("record:13", "op_unknown_dispatch_evidence", dispatch_evidence, ["dispatch/1"]),
        ("record:13", "op_unknown_observation", observation, {"retryable": True}),
        ("record:14", "turn_ended_reason", terminal_reason, "tool_finish"),
        ("record:14", "turn_ended_result", terminal_result, {"content": "done"}),
        ("record:14", "turn_ended_budget", terminal_budget, {"elapsed_ms": 1}),
    ):
        reader.add(owner, slot, ref, value)

    assert driver.hydrate_record_field(completed.record, "result", result.to_dict()) == {
        "content": "done",
        "tool_calls": [],
    }
    assert driver.hydrate_record_field(completed.record, "usage", usage.to_dict()) == {"prompt_tokens": 3}
    assert driver.hydrate_record_field(completed.record, "shared_state", shared.to_dict()) == {"answer": 42}
    assert driver.hydrate_record_field(completed.record, "evidence", evidence.to_dict()) == ["provider/receipt"]
    assert driver.hydrate_record_field(unknown.record, "op_unknown_reason", reason.to_dict()) == "worker lost"
    assert driver.hydrate_record_field(terminal.record, "turn_ended_result", terminal_result.to_dict()) == {"content": "done"}

    null_completed = _stored(
        15,
        "op_completed",
        completed.record._payload | {"usage": None, "shared_state": None},
    )
    assert driver.hydrate_record_field(null_completed.record, "usage", None) is None
    assert driver.hydrate_record_field(null_completed.record, "shared_state", None) is None


def test_parked_provider_user_approval_and_delegation_fields_use_declared_slots() -> None:
    job = _ref("job", {"id": "job/1"})
    evidence = _ref("parked-evidence", ["receipt/1"])
    query = _ref("query", {"cursor": 2})
    cancel = _ref("cancel", {"reason": "stop"})
    interaction = _ref("interaction", {"content": "waiting", "tool_call_id": "call"})
    agent_name = _ref("agent-name", "researcher")
    metadata = _ref("delegation-metadata", {"mode": "handoff"})
    provider_payload = {
        "phase": "after_dispatch",
        "handle": {
            "kind": "provider",
            "operation_id": "o",
            "attempt": 1,
            "job_id": job.to_dict(),
            "evidence": evidence.to_dict(),
            "query_ref": query.to_dict(),
            "cancel_ref": cancel.to_dict(),
        },
        "poll_at_ms": 10,
        "deadline_ms": None,
        "interaction_result": interaction.to_dict(),
        "delegation": {"agent_name": agent_name.to_dict(), "metadata": metadata.to_dict()},
    }
    provider = _stored(20, "op_parked", provider_payload)
    scope = _ref("scope", ["tool/name"])
    question = _ref("question", "Proceed?")
    approval_payload = {
        "phase": "before_dispatch",
        "handle": {
            "kind": "approval",
            "operation_id": "o",
            "attempt": 1,
            "scope": scope.to_dict(),
            "question": question.to_dict(),
        },
        "poll_at_ms": None,
        "deadline_ms": 20,
    }
    approval = _stored(21, "op_parked", approval_payload)
    driver, reader = _driver((provider, approval))
    for owner, slot, ref, value in (
        ("record:20", "provider_handle_job_id", job, {"id": "job/1"}),
        ("record:20", "provider_handle_evidence", evidence, ["receipt/1"]),
        ("record:20", "provider_handle_query_ref", query, {"cursor": 2}),
        ("record:20", "provider_handle_cancel_ref", cancel, {"reason": "stop"}),
        ("record:20", "interaction_result", interaction, {"content": "waiting", "tool_call_id": "call"}),
        ("record:20", "delegation_agent_name", agent_name, "researcher"),
        ("record:20", "delegation_metadata", metadata, {"mode": "handoff"}),
        ("record:21", "approval_scope", scope, ["tool/name"]),
        ("record:21", "user_question", question, "Proceed?"),
    ):
        reader.add(owner, slot, ref, value)

    hydrated = driver.hydrate_parked_wait("o", 1, provider_payload)
    assert hydrated["handle"]["job_id"] == {"id": "job/1"}
    assert hydrated["handle"]["evidence"] == ["receipt/1"]
    assert hydrated["handle"]["query_ref"] == {"cursor": 2}
    assert hydrated["handle"]["cancel_ref"] == {"reason": "stop"}
    assert hydrated["interaction_result"] == {"content": "waiting", "tool_call_id": "call"}
    assert hydrated["delegation"] == {"agent_name": "researcher", "metadata": {"mode": "handoff"}}

    approval_wait = driver.hydrate_parked_wait("o", 1, approval_payload)
    assert approval_wait["handle"]["scope"] == ["tool/name"]
    assert approval_wait["handle"]["question"] == "Proceed?"


def test_retry_publication_reuses_original_plan_refs_without_put() -> None:
    request = _ref("retry-request", {"id": "call", "name": "effect"})
    tool = _ref("retry-tool", {"idempotency": "supported"})
    budget = _ref("retry-budget", {})
    plan = _stored(
        25,
        "op_planned",
        {
            "op_kind": "tool",
            "purpose": None,
            "request": request.to_dict(),
            "request_digest": reference_digest(request),
            "context_version": "1",
            "dependencies": [],
            "tool": tool.to_dict(),
            "idempotency_key": None,
            "budget_admission": budget.to_dict(),
            "not_before_ms": 0,
            "provider_binding": "function",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )
    retry = replace(plan.record, attempt=2, payload=plan.record._payload | {"not_before_ms": 10})
    writer = _PutWriter()

    published = publish_record_content(cast(ContentSessionWriter, writer), (retry,))

    assert writer.puts == []
    assert published.records[0]._payload["request"] == request.to_dict()
    assert published.records[0]._payload["tool"] == tool.to_dict()
    assert published.records[0]._payload["budget_admission"] == budget.to_dict()


def test_missing_link_missing_key_plaintext_and_wrong_owner_fail_closed() -> None:
    ref = _ref("missing", {"value": 1})
    completed = _stored(
        30,
        "op_completed",
        {
            "result": ref.to_dict(),
            "result_digest": "0" * 64,
            "usage": None,
            "shared_state": None,
            "evidence": None,
            "execution_started": True,
            "context": "normal",
            "request_digest": "1" * 64,
            "provider_binding": "function",
            "policy_decision": None,
        },
    )
    driver, reader = _driver((completed,))

    with pytest.raises(MissingContentLink):
        driver.hydrate_record_field(completed.record, "result", ref.to_dict())
    reader.add("record:999", "result", ref, {"value": 1})
    with pytest.raises(MissingContentLink):
        driver.hydrate_record_field(completed.record, "result", ref.to_dict())
    reader.links[("record:30", "result", ref.content_id)] = b"short"
    with pytest.raises(IntegrityFailure):
        driver.hydrate_record_field(completed.record, "result", ref.to_dict())
    driver.content_reader = None
    with pytest.raises(RecordError, match="content hydration is unavailable"):
        driver.hydrate_record_field(completed.record, "result", ref.to_dict())

    missing = _stored(31, "turn_ended", {"status": "failed"})
    with pytest.raises(RecordError, match="missing content field"):
        driver.hydrate_record_field(missing.record, "turn_ended_reason", ref.to_dict())
    with pytest.raises(RecordError, match="requires a valid ContentRef"):
        driver.hydrate_record_field(completed.record, "result", {"plaintext": True})
