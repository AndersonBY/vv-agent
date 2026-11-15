from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from vv_agent.kernel._boundary_hydration import hydrate_input, hydrate_stored_payload
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentBindingMismatch, ContentRef, ContentUnavailable
from vv_agent.kernel._content_publication import (
    _BOUNDARY_CONTENT_SLOTS,
    _CONTENT_SLOTS,
    _INPUT_SLOTS,
    declared_content_ref,
    link_declared_record_content,
    publish_record_content,
    publish_turn_started,
    resolve_record_content_links,
)
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._record_schema import BOUNDARY_DATA, INPUT_PAYLOADS, PAYLOADS
from vv_agent.kernel._records import InboxItem, Record, digest, record_identity
from vv_agent.kernel._store import StoredRecord
from vv_agent.prompt.builder import build_raw_system_prompt_bundle
from vv_agent.session.content_runtime import SessionContentWriter, durable_content_reader
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import Message, ToolExecutionResult

MASTER_KEY = b"k" * 32


class _OwnerMapReader:
    binding = ("default", "s")

    def __init__(self, values: dict[tuple[str, str, str], bytes]) -> None:
        self.values = values

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        assert (tenant_id, session_id) == self.binding
        try:
            return self.values[(owner, slot, ref.content_id)]
        except KeyError as exc:
            raise AssertionError(f"missing owner link: {owner}/{slot}/{ref.content_id}") from exc


@contextmanager
def _sessions(tmp_path, *, attributes: dict[str, Any] | None = None) -> Iterator[tuple[SQLiteStore, SessionDriver]]:
    with SQLiteStore.standalone(str(tmp_path / "inventory.sqlite")) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        sessions.create("s", "/workspace", attributes or {})
        yield store, sessions


def _boundary_driver(store: SQLiteStore, session_id: str = "s") -> CoreDriver:
    tenant_id = getattr(store, "tenant_id", "default")
    driver = object.__new__(CoreDriver)
    driver.sid = session_id
    driver.store = store
    driver.content_reader = durable_content_reader(store, tenant_id=tenant_id, session_id=session_id)
    driver._hydrators = {}
    return driver


def _links(store: SQLiteStore, *, record_seq: int | None = None, input_id: str | None = None):
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
        return writer.durable.read_links("default", "s", record_seq=record_seq, input_id=input_id)


def _content_count(store: SQLiteStore) -> int:
    return store._one(
        "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
        ("default", "s"),
    )[0]


def _link_count(store: SQLiteStore) -> int:
    return store._one(
        "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
        ("default", "s"),
    )[0]


def _definition(message: dict[str, Any], *, input_message: dict[str, Any] | None = None) -> dict[str, Any]:
    task = AgentTask(
        task_id="task",
        model="scripted",
        prompt_bundle=build_raw_system_prompt_bundle("system"),
        user_prompt="work",
        initial_messages=[Message.from_dict(message)],
        metadata={"vv_session": {"input_messages": [input_message]}} if input_message is not None else {},
    ).to_dict()
    return {
        "agent_name": "kernel",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": ["scripted"],
            "context_length": None,
            "max_output_tokens": None,
        },
        "capabilities": {},
        "task": task,
        "tools": [],
    }


def _provider_items() -> tuple[InboxItem, InboxItem, dict[str, Any], dict[str, Any]]:
    request_digest = digest({"request": "provider"})
    result = {
        "answer": "provider-result-secret",
        "opaque_metadata": {"content_id": "host-value", "note": "not a ContentRef"},
    }
    usage = {"input_tokens": 7, "provider_usage": {"opaque": "usage-secret"}}
    evidence = {"content_id": "opaque-evidence", "note": "host-defined"}
    provider_result = InboxItem(
        "provider-result",
        "provider_result",
        {
            "operation_id": "operation/result",
            "attempt": 1,
            "request_digest": request_digest,
            "provider_binding": None,
            "result": result,
            "usage": usage,
            "evidence": evidence,
        },
        target_turn_id="s/turn/provider",
        generation=0,
    )
    provider_evidence = InboxItem(
        "provider-evidence",
        "provider_evidence",
        {
            "operation_id": "operation/evidence",
            "attempt": 1,
            "request_digest": request_digest,
            "handle": {
                "kind": "provider",
                "provider": "test",
                "job_id": {"job": "job-secret"},
                "operation_id": "operation/evidence",
                "attempt": 1,
                "request_digest": request_digest,
                "evidence": {"opaque": "handle-evidence-secret"},
                "query_ref": {"query": "query-secret"},
                "cancel_ref": {"cancel": "cancel-secret"},
            },
        },
        target_turn_id="s/turn/provider",
        generation=0,
    )
    return provider_result, provider_evidence, result, usage


def test_content_inventory_covers_all_records_inputs_and_boundaries() -> None:
    assert set(_CONTENT_SLOTS) == set(PAYLOADS)
    assert len(PAYLOADS) == 14
    assert set(_INPUT_SLOTS) == set(INPUT_PAYLOADS)
    assert len(INPUT_PAYLOADS) == 9
    assert set(_BOUNDARY_CONTENT_SLOTS) == set(BOUNDARY_DATA)
    assert len(BOUNDARY_DATA) == 7

    assert {slot.slot for slot in _INPUT_SLOTS["provider_result"]} == {
        "provider.result",
        "provider.usage",
        "provider.evidence",
    }
    assert {slot.slot for slot in _INPUT_SLOTS["provider_evidence"]} == {
        "provider.handle.job_id",
        "provider.handle.evidence",
        "provider.handle.query_ref",
        "provider.handle.cancel_ref",
    }
    session_slots = {slot.slot: slot for slot in _CONTENT_SLOTS["session_created"]}
    assert session_slots["session.attributes"].nested[0].path == ("seed", "messages", "*")
    turn_slots = {slot.slot: slot for slot in _CONTENT_SLOTS["turn_started"]}
    assert turn_slots["definition"].nested[0].path == ("task", "initial_messages", "*")
    planned_slots = {slot.slot: slot for slot in _CONTENT_SLOTS["op_planned"]}
    assert planned_slots["request"].nested[0].path == ("messages", "*")


def test_provider_inbox_round_trip_idempotence_and_opaque_content(tmp_path) -> None:
    provider_result, provider_evidence, expected_result, expected_usage = _provider_items()
    with _sessions(tmp_path) as (store, sessions):
        baseline_content = _content_count(store)
        assert not sessions.push("s", provider_result).replayed
        assert not sessions.push("s", provider_evidence).replayed
        first_content = _content_count(store)
        first_links = _link_count(store)
        assert first_content > baseline_content

        assert sessions.push("s", provider_result).replayed
        assert sessions.push("s", provider_evidence).replayed
        assert _content_count(store) == first_content
        assert _link_count(store) == first_links

        driver = _boundary_driver(store)
        pending = {stored.item.input_id: stored.item for stored in store.peek_inbox("s")}
        hydrated_result = hydrate_input(driver, pending["provider-result"])
        hydrated_evidence = hydrate_input(driver, pending["provider-evidence"])

        assert hydrated_result.payload["result"] == expected_result
        assert hydrated_result.payload["usage"] == expected_usage
        assert hydrated_result.payload["evidence"] == {"content_id": "opaque-evidence", "note": "host-defined"}
        assert hydrated_evidence.payload["handle"]["job_id"] == {"job": "job-secret"}
        assert hydrated_evidence.payload["handle"]["evidence"] == {"opaque": "handle-evidence-secret"}
        assert hydrated_evidence.payload["handle"]["query_ref"] == {"query": "query-secret"}
        assert hydrated_evidence.payload["handle"]["cancel_ref"] == {"cancel": "cancel-secret"}

        result_links = {link.slot for link in _links(store, input_id="provider-result")}
        evidence_links = {link.slot for link in _links(store, input_id="provider-evidence")}
        assert result_links == {"input.actor.attributes", "provider.result", "provider.usage", "provider.evidence"}
        assert evidence_links == {
            "input.actor.attributes",
            "provider.handle.job_id",
            "provider.handle.evidence",
            "provider.handle.query_ref",
            "provider.handle.cancel_ref",
        }
        bodies = "\n".join(
            row[0].decode() if isinstance(row[0], bytes) else row[0]
            for row in store._rows("SELECT body FROM sk_inbox WHERE tenant_id=%s AND session_id=%s", ("default", "s"))
        )
        assert "provider-result-secret" not in bodies
        assert "job-secret" not in bodies


def test_nested_creation_definition_and_request_round_trip_and_replay(tmp_path) -> None:
    message = Message(role="user", content="shared-message-secret").to_dict()
    input_message = Message(role="user", content="turn-input-secret").to_dict()
    with _sessions(tmp_path, attributes={"seed": {"messages": [message], "shared_state": {"seed": "state"}}}) as (
        store,
        _sessions_driver,
    ):
        driver = _boundary_driver(store)
        creation = store.read("s").records[0]
        hydrated_creation = hydrate_stored_payload(driver, creation)
        assert hydrated_creation["attributes"]["seed"]["messages"] == [message]
        assert "session.attributes.seed.messages[0]" in {link.slot for link in _links(store, record_seq=1)}

        definition = _definition(message, input_message=input_message)
        request = {
            "model": "scripted",
            "messages": [message, message, Message(role="assistant", content="second-message-secret").to_dict()],
        }
        definition_digest = digest(definition)
        turn_id = "s/turn/nested"
        operation_id = "operation/nested"
        budget = {
            "max_total_tokens": 100,
            "max_uncached_input_tokens": None,
            "max_tool_calls": 2,
            "max_tool_calls_by_name": {},
            "max_wall_time_ms": None,
            "max_host_cost": None,
            "unavailable_metric_policy": "continue_and_mark",
        }
        lease = store.acquire("s", owner="inventory", ttl_ms=15_000)
        assert lease is not None
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            started_content = publish_turn_started(
                writer,
                definition,
                budget,
                admission_definition_digest=definition_digest,
            )
            started = Record(
                record_identity("turn_started", "s", turn_id, None, None, {}),
                "turn_started",
                "s",
                turn_id,
                None,
                None,
                {
                    "input_ids": [],
                    "definition": started_content.definition.to_dict(),
                    "definition_digest": started_content.definition_digest,
                    "handler_version": "1",
                    "budget": started_content.budget.to_dict(),
                    "binding": None,
                    "generation": 0,
                    "policy_digest": None,
                    "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
                    "tokenizer_binding": None,
                    "environment": None,
                },
            )
            planned = Record(
                record_identity("op_planned", "s", turn_id, operation_id, 1, {}),
                "op_planned",
                "s",
                turn_id,
                operation_id,
                1,
                {
                    "op_kind": "model",
                    "purpose": "primary",
                    "request": request,
                    "request_digest": digest(request),
                    "context_version": "1",
                    "dependencies": [],
                    "tool": None,
                    "idempotency_key": None,
                    "budget_admission": {"max_total_tokens": 100},
                    "not_before_ms": None,
                    "provider_binding": "model",
                    "consumed_unknowns": [],
                    "policy_decision": None,
                },
            )
            publication = publish_record_content(writer, (started, planned))
            receipt = tx.append(
                "s",
                lease=lease,
                expected_seq=store.read("s").head_seq,
                commit_id="nested-content",
                records=publication.records,
            )
            resolved = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, resolved)

        records = {stored.record.kind: stored for stored in store.read("s").records}
        hydrated_started = hydrate_stored_payload(driver, records["turn_started"])
        hydrated_planned = hydrate_stored_payload(driver, records["op_planned"])
        assert hydrated_started["definition"]["task"]["initial_messages"] == [message]
        assert hydrated_started["definition"]["task"]["metadata"]["vv_session"]["input_messages"] == [input_message]
        assert hydrated_planned["request"]["messages"] == request["messages"]
        assert "shared-message-secret" not in records["turn_started"].record.encode().decode()
        assert "turn-input-secret" not in records["turn_started"].record.encode().decode()
        assert "second-message-secret" not in records["op_planned"].record.encode().decode()

        started_links = {link.slot: link for link in _links(store, record_seq=records["turn_started"].seq)}
        planned_slots = {link.slot for link in _links(store, record_seq=records["op_planned"].seq)}
        assert "definition.task.initial_messages[0]" in started_links
        assert "definition.task.metadata.vv_session.input_messages[0]" in started_links
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            published_definition = writer.get_json(ContentRef.from_dict(records["turn_started"].record.payload["definition"]))
        published_task = published_definition["task"]
        initial_ref = declared_content_ref(published_task["initial_messages"][0])
        input_ref = declared_content_ref(published_task["metadata"]["vv_session"]["input_messages"][0])
        assert initial_ref is not None
        assert input_ref is not None
        assert started_links["definition.task.initial_messages[0]"].ref == initial_ref
        assert started_links["definition.task.metadata.vv_session.input_messages[0]"].ref == input_ref
        assert {"request", "request.messages[0]", "request.messages[1]", "request.messages[2]"} <= planned_slots

        before_content, before_links = _content_count(store), _link_count(store)
        committed_records = store.read("s").records
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            replay = publish_record_content(writer, tuple(stored.record for stored in committed_records))
            replay_links = resolve_record_content_links(
                replay.links,
                tuple((stored.record.record_id, stored.seq) for stored in committed_records),
            )
            link_declared_record_content(writer, replay_links)
        assert _content_count(store) == before_content
        assert _link_count(store) == before_links


def test_child_admission_definition_messages_publish_and_hydrate_in_place(tmp_path) -> None:
    initial_message = Message(role="user", content="child-initial-secret").to_dict()
    input_message = Message(role="user", content="child-input-secret").to_dict()
    definition = _definition(initial_message, input_message=input_message)
    admission = {
        "mode": "configured",
        "selector": "kernel",
        "definition": definition,
        "definition_digest": digest(definition),
        "budget": {
            "max_total_tokens": 100,
            "max_uncached_input_tokens": None,
            "max_tool_calls": 2,
            "max_tool_calls_by_name": {},
            "max_wall_time_ms": None,
            "max_host_cost": None,
            "unavailable_metric_policy": "continue_and_mark",
        },
        "handler_version": "1",
        "sub_config": None,
        "exclude_files_pattern": None,
        "handoff_count": 0,
        "max_handoffs": 100,
        "handoff_metadata": {},
    }
    with _sessions(tmp_path, attributes={"child_admission": admission}) as (store, _sessions_driver):
        creation = store.read("s").records[0]
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            attributes_ref = ContentRef.from_dict(creation.record.payload["attributes"])
            attributes = writer.get_json(attributes_ref)

        task = attributes["child_admission"]["definition"]["task"]
        initial_ref = declared_content_ref(task["initial_messages"][0])
        input_ref = declared_content_ref(task["metadata"]["vv_session"]["input_messages"][0])
        assert initial_ref is not None
        assert input_ref is not None

        links = {link.slot: link for link in _links(store, record_seq=creation.seq)}
        initial_slot = "session.attributes.child_admission.definition.task.initial_messages[0]"
        input_slot = "session.attributes.child_admission.definition.task.metadata.vv_session.input_messages[0]"
        assert links[initial_slot].ref == initial_ref
        assert links[input_slot].ref == input_ref

        hydrated = hydrate_stored_payload(_boundary_driver(store), creation)
        hydrated_task = hydrated["attributes"]["child_admission"]["definition"]["task"]
        assert hydrated_task["initial_messages"] == [initial_message]
        assert hydrated_task["metadata"]["vv_session"]["input_messages"] == [input_message]


def test_prepared_hook_result_publishes_producer_arrays_without_recursing_opaque_provider_data(tmp_path) -> None:
    opaque_provider_data = {"content_id": "provider-selected-id", "detail": {"content_id": "nested-id"}}
    assert declared_content_ref(opaque_provider_data) is None
    hook_result = ToolExecutionResult(
        tool_call_id="call",
        content="hook-short-circuit",
        metadata={
            "children": [{"session_id": "child-a", "status": "running"}],
            "running_children": ["child-a"],
            "provider_observation": opaque_provider_data,
        },
    ).to_dict()
    request = {"id": "call", "name": "tool", "arguments": {}}
    turn_id = "s/turn/hook"
    operation_id = "operation/hook"
    record = Record(
        record_identity("op_prepared", "s", turn_id, operation_id, 1, {}),
        "op_prepared",
        "s",
        turn_id,
        operation_id,
        1,
        {
            "request": request,
            "request_digest": digest(request),
            "op_kind": "tool",
            "tool": {},
            "provider_binding": "function",
            "idempotency_key": None,
            "hook_result": hook_result,
            "shared_state": {},
            "policy_decision": None,
        },
    )
    with _sessions(tmp_path) as (store, _sessions_driver):
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            publication = publish_record_content(writer, (record,))
            links = {link.slot: link for link in publication.links}
            assert {"request", "tool", "hook_result", "shared_state"} <= set(links)
            published_content = {slot: writer.get_json(link.ref) for slot, link in links.items()}
            published_hook = published_content["hook_result"]
            owner = "record:2"
            owner_values = {
                (owner, slot, link.ref.content_id): canonical_json_bytes(writer.get_json(link.ref))
                for slot, link in links.items()
            }

        children = published_hook["metadata"]["children"]
        running_children = published_hook["metadata"]["running_children"]
        assert declared_content_ref(children[0]) is not None
        assert declared_content_ref(running_children[0]) is not None
        assert published_hook["metadata"]["provider_observation"] == opaque_provider_data

        stored = StoredRecord(publication.records[0], 2, "prepared-hook-content", 1, 1)
        hook_ref = declared_content_ref(stored.record.payload["hook_result"])
        assert hook_ref is not None
        assert links["hook_result"].ref == hook_ref
        assert links["hook_result.metadata.children[0]"].ref == declared_content_ref(children[0])
        assert links["hook_result.metadata.running_children[0]"].ref == declared_content_ref(running_children[0])

        driver = _boundary_driver(store)
        driver.content_reader = _OwnerMapReader(owner_values)
        driver._hydrators = {}
        hydrated = hydrate_stored_payload(driver, stored)
        assert hydrated["hook_result"] == hook_result
        assert hydrated["hook_result"]["metadata"]["provider_observation"] == opaque_provider_data


def test_input_applied_provider_copy_uses_record_owner_slots(tmp_path) -> None:
    provider_result, _, expected_result, expected_usage = _provider_items()
    with _sessions(tmp_path) as (store, sessions):
        sessions.push("s", provider_result)
        stored_input = store.read_input("s", provider_result.input_id)
        assert stored_input is not None
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            turn_id = "s/turn/provider"
            record = Record(
                record_identity("input_applied", "s", turn_id, None, None, {"input": {"input_id": provider_result.input_id}}),
                "input_applied",
                "s",
                turn_id,
                None,
                None,
                {
                    "input": stored_input.item.to_dict(),
                    "input_digest": stored_input.item.digest,
                    "input_proof": None,
                    "disposition": "applied",
                    "reason": None,
                    "target_operation_id": None,
                    "target_wait_id": None,
                    "position": "between_actions",
                },
            )
            publication = publish_record_content(writer, (record,))
            links = resolve_record_content_links(publication.links, ((record.record_id, 1),))
            link_declared_record_content(writer, links)

        published = publication.records[0]
        stored = StoredRecord(published, 1, "input-applied-content", 0, 0)
        skeleton = published.payload["input"]
        assert set(skeleton["actor"]["attributes"]) == {"content_id", "mac", "size", "key_id", "labels"}
        assert set(skeleton["payload"]["result"]) == {"content_id", "mac", "size", "key_id", "labels"}
        hydrated = hydrate_stored_payload(_boundary_driver(store), stored)
        assert hydrated["input"]["payload"]["result"] == expected_result
        assert hydrated["input"]["payload"]["usage"] == expected_usage
        assert hydrated["input"]["actor"]["attributes"] == {"role": "local"}
        assert {
            "input.actor.attributes",
            "input.payload.result",
            "input.payload.usage",
            "input.payload.evidence",
        } <= {link.slot for link in _links(store, record_seq=stored.seq)}


def test_cross_session_and_cross_tenant_content_links_are_rejected(tmp_path) -> None:
    with _sessions(tmp_path) as (store, sessions):
        sessions.create("same-tenant", "/workspace", {})
        foreign_same_tenant = store.read("same-tenant").records[0].record.payload["attributes"]
        foreign_tenant_store = store.for_tenant("tenant-b")
        foreign_sessions = SessionDriver(
            store=foreign_tenant_store,
            tenant_id="tenant-b",
            content_key_provider=LocalKeyProvider(MASTER_KEY),
        )
        foreign_sessions.create("foreign", "/workspace", {})
        foreign_tenant = foreign_tenant_store.read("foreign").records[0].record.payload["attributes"]

        for candidate in (foreign_same_tenant, foreign_tenant):
            with pytest.raises((ContentUnavailable, ContentBindingMismatch)), store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
                writer.link_record(1, ContentRef.from_dict(candidate), "provider.result")
