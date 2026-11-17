from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import pytest
from session.helpers import fixture_child_spec, fixture_key_provider, wrap_fixture_store
from session.helpers import record as fixture_record
from session.test_app_server_fork import run_turn, send, server_for
from support.kernel_runtime import start_runner

from vv_agent import Agent, RunConfig, ScriptedModelProvider, SubAgentConfig
from vv_agent.app_server import DefaultAppServerHost
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._child_store import _bound_actor as bind_child_actor
from vv_agent.kernel._content import (
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
    Classification,
    ContentLabels,
    ContentRef,
    ContentUnavailable,
    Principal,
    SourceKind,
)
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import MissingContentLink
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.prompt.builder import build_raw_system_prompt_bundle
from vv_agent.session import delegation
from vv_agent.session.children import ChildSession, child_delivery, child_handles, create_child
from vv_agent.session.content_reader import record_owner
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.fork import fork_thread
from vv_agent.session.kernel import Runtime, drive
from vv_agent.session.projection import SessionHydration, hydrate_records
from vv_agent.session.records import InboxItem, SessionSpec, digest
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Conflict
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, Message, ToolCall


def _definition(initial_message: Message, input_message: Message) -> dict[str, Any]:
    task = AgentTask(
        task_id="child/task",
        model="scripted",
        prompt_bundle=build_raw_system_prompt_bundle("Work."),
        user_prompt="Do the child work.",
        initial_messages=[initial_message],
        metadata={"vv_session": {"input_messages": [input_message.to_dict()]}},
    ).to_dict()
    return {
        "agent_name": "worker",
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


def test_reconstruct_hydrates_child_admission_nested_messages(tmp_path, monkeypatch) -> None:
    initial = Message("user", "child initial")
    input_message = Message("user", "child input")
    definition = _definition(initial, input_message)
    admission = {
        "mode": "configured",
        "selector": "worker",
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

    with SQLiteStore.standalone(str(tmp_path / "child.sqlite")) as raw_store:
        raw_store.install_schema()
        provider = fixture_key_provider()
        store = wrap_fixture_store(raw_store, provider)
        sessions = SessionDriver(
            store=store,
            tenant_id="child-tenant",
            content_key_provider=provider,
        )
        sessions.create("child", "/workspace", {"child_admission": admission})
        created = sessions.store.read("child", limit=1).records[0].record

        with sessions.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="child-tenant", session_id="child")
            attributes_ref = ContentRef.from_dict(created.payload["attributes"])
            published = writer.get_json(attributes_ref)
        task = published["child_admission"]["definition"]["task"]
        assert declared_content_ref(task["initial_messages"][0]) is not None
        assert declared_content_ref(task["metadata"]["vv_session"]["input_messages"][0]) is not None

        resolved = ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", [])

        def heartbeat_store():
            return nullcontext(sessions.store)

        child_runtime = Runtime(
            agent=Agent(
                "worker",
                "Work.",
                model="scripted",
                sub_agents={"worker": SubAgentConfig(model="scripted", description="Work.")},
            ),
            config=RunConfig(workspace="/workspace"),
            resolved=resolved,
            llm=ScriptedLLM([]),
            heartbeat_store=heartbeat_store,
        )
        runtime = Runtime(
            agent=Agent(
                "worker",
                "Work.",
                model="scripted",
                sub_agents={"worker": SubAgentConfig(model="scripted", description="Work.")},
            ),
            config=RunConfig(workspace="/workspace"),
            resolved=resolved,
            llm=ScriptedLLM([]),
            heartbeat_store=heartbeat_store,
        )
        monkeypatch.setattr(delegation, "target_agent", lambda *args, **kwargs: Agent("worker", "Work.", model="scripted"))
        monkeypatch.setattr(runtime, "for_agent", lambda agent, config: child_runtime)
        monkeypatch.setattr(
            delegation,
            "child_config",
            lambda *args, **kwargs: RunConfig(workspace="/workspace"),
        )

        reconstructed = delegation.reconstruct(runtime, sessions.store, "child")
        hydration = SessionHydration.from_store(
            sessions.store,
            tenant_id="child-tenant",
            session_id="child",
        )
        with pytest.raises(MissingContentLink):
            hydration.read(record_owner(2), SESSION_WORKSPACE, ContentRef.from_dict(created.payload["workspace"]))
        with pytest.raises(Conflict):
            sessions.store.for_tenant("other-tenant").read("child", limit=1)

    assert reconstructed is child_runtime
    assert created.payload["principal"]["tenant_id"] == "child-tenant"
    assert reconstructed.frozen_task is not None
    assert reconstructed.frozen_task.initial_messages == [initial]
    assert reconstructed.frozen_task.metadata["vv_session"]["input_messages"] == [input_message.to_dict()]
    assert created.payload["attributes"] == attributes_ref.to_dict()


def test_child_admission_requires_owner_linked_parent_principal_content():
    driver = SessionDriver()
    principal = {
        "tenant_id": "default",
        "subject_id": "parent-user",
        "subject_kind": "user",
        "attributes": {"role": "owner"},
    }
    try:
        driver.create("parent", "/workspace", principal=principal)
        parent_created = driver.store.read("parent", limit=1).records[0]
        plan = fixture_record("op_planned", sid="parent", tid="parent/turn", oid="parent-operation")

        valid_spec = fixture_child_spec(driver.store, "parent", "valid-child")
        with driver.store.atomic() as tx:
            create_child(tx, ChildSession(valid_spec, "work"), plan, 0)
        child_principal = driver.store.read("valid-child", limit=1).records[0].record.payload["principal"]
        assert child_principal["tenant_id"] == "default"
        assert child_principal["subject_id"] == "parent-user"
        assert child_principal["subject_kind"] == "user"

        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id="parent")
            unlinked = writer.put_json({"role": "forged"})
        forged_spec = replace(
            fixture_child_spec(driver.store, "parent", "forged-child"),
            principal=parent_created.record.payload["principal"] | {"attributes": unlinked.to_dict()},
        )
        with pytest.raises(ContentUnavailable), driver.store.atomic() as tx:
            create_child(tx, ChildSession(forged_spec, "work"), plan, 0)
        assert "forged-child" not in driver.store.list_sessions()
    finally:
        driver.close()


def test_child_admission_preserves_principal_and_workspace_labels():
    driver = SessionDriver()
    principal_labels = ContentLabels((SourceKind.USER,), Classification.CONFIDENTIAL)
    workspace_labels = ContentLabels((SourceKind.SYSTEM,), Classification.RESTRICTED)
    try:
        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id="parent")

            def parent_spec(content):
                principal_attributes = content.put_json({"role": "owner"}, labels=principal_labels)
                workspace = content.put_json("/workspace", labels=workspace_labels)
                return SessionSpec(
                    "parent",
                    Principal("default", "parent-agent", "agent", principal_attributes).to_dict(),
                    workspace.to_dict(),
                    attributes={},
                )

            writer.create_session(
                parent_spec,
                consumers=(),
            )

        plan = fixture_record("op_planned", sid="parent", tid="parent/turn", oid="parent-operation")
        with driver.store.atomic() as tx:
            create_child(tx, ChildSession(fixture_child_spec(driver.store, "parent", "child"), "work"), plan, 0)

        child_created = driver.store.read("child", limit=1).records[0].record.payload
        assert ContentRef.from_dict(child_created["principal"]["attributes"]).labels == principal_labels
        assert ContentRef.from_dict(child_created["workspace"]).labels == workspace_labels
        assert child_created["principal"]["subject_id"] == "parent-agent"
        assert child_created["principal"]["subject_kind"] == "agent"
    finally:
        driver.close()


def test_child_actor_content_ref_requires_principal_slot_and_plain_mapping_links():
    driver = SessionDriver()
    principal = {
        "tenant_id": "default",
        "subject_id": "parent-user",
        "subject_kind": "user",
        "attributes": {"role": "owner", "labels": "business", "size": 17},
    }
    try:
        driver.create("parent", "/workspace", principal=principal)
        created = driver.store.read("parent", limit=1).records[0].record.payload
        persisted = Principal.from_dict(created["principal"])
        workspace = ContentRef.from_dict(created["workspace"])

        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id="parent")
            with pytest.raises(ValueError, match="plaintext mapping"):
                bind_child_actor(
                    writer,
                    principal | {"attributes": workspace},
                    subject_id="child-delivery",
                )

            actor, actor_ref = bind_child_actor(writer, persisted, subject_id="child-delivery")
            assert writer.get_json(actor_ref) == principal["attributes"]
            assert writer.get_json(actor["attributes"]) == principal["attributes"]

            ingress_actor, ingress_ref = bind_child_actor(writer, principal, subject_id="child-delivery")
            assert writer.get_json(ingress_ref) == principal["attributes"]
            assert writer.get_json(ingress_actor["attributes"]) == principal["attributes"]
    finally:
        driver.close()


@pytest.mark.parametrize("slot", [SESSION_PRINCIPAL_ATTRIBUTES, SESSION_WORKSPACE])
def test_fork_rejects_tampered_creation_owner_links(tmp_path, slot):
    with SQLiteStore.standalone(str(tmp_path / f"fork-{slot.replace('.', '-')}.sqlite")) as store:
        store.install_schema()
        host = DefaultAppServerHost(
            agent=Agent("test", "Answer."),
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            ),
        )
        server = server_for(store, host, tmp_path / "app-server-key-state.sqlite")
        try:
            send(server, "create", "thread/start", {"inputId": "source"})
            run_turn(server, "source", "turn")
            with store.atomic() as tx:
                tx.store._rows(
                    "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id='source' AND record_seq=1 AND slot=%s",
                    (store.tenant_id, slot),
                )
            with pytest.raises((ContentUnavailable, MissingContentLink)):
                fork_thread(server.kernel, server.host, {"threadId": "source", "forkId": slot})
        finally:
            server.kernel.close()


def test_configured_child_delivery_hydrates_result_owners_and_session_bindings(tmp_path) -> None:
    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "parent",
            Agent(
                "parent",
                "Delegate.",
                sub_agents={"worker": SubAgentConfig(model="m", description="Work.")},
            ),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.from_steps(
                    "scripted",
                    "m",
                    [
                        LLMResponse(
                            "",
                            [
                                ToolCall(
                                    "delegate",
                                    "create_sub_task",
                                    {"agent_id": "worker", "task_description": "work"},
                                )
                            ],
                        ),
                        LLMResponse("child done"),
                        LLMResponse("parent done"),
                    ],
                ),
            ),
        )
        assert handle.result().final_output == "parent done"
        for running in driver.handles:
            running.join(2)

        parent_rows = driver.store.read_state("parent")[1]
        parked = next(row for row in parent_rows if row.record.kind == "op_parked")
        child_id = child_handles(parked.record.payload["handle"])[0]["session_id"]
        child_rows = driver.store.read_state(child_id)[1]

        parent_hydration = SessionHydration.from_store(driver.store, tenant_id=driver.tenant_id, session_id="parent")
        child_hydration = SessionHydration.from_store(driver.store, tenant_id=driver.tenant_id, session_id=child_id)
        hydrated_parent_created = hydrate_records(parent_rows[:1], parent_hydration)[0].record
        hydrated_child_created = hydrate_records(child_rows[:1], child_hydration)[0].record
        assert type(hydrated_parent_created.payload["attributes"]) is dict
        assert type(hydrated_child_created.payload["attributes"]) is dict
        assert hydrated_parent_created.payload["principal"]["tenant_id"] == driver.tenant_id
        assert hydrated_child_created.payload["principal"]["tenant_id"] == driver.tenant_id
        assert hydrated_child_created.payload["attributes"]["child_admission"]["mode"] == "configured"

        raw_terminal = next(row for row in child_rows if row.record.kind == "turn_ended")
        hydrated_terminal = hydrate_records((raw_terminal,), child_hydration)[0]
        terminal_result = declared_content_ref(raw_terminal.record._payload["result"])
        assert terminal_result is not None
        assert raw_terminal.seq > 1
        assert hydrated_terminal.record.payload["result"] == "child done"
        with pytest.raises(MissingContentLink):
            child_hydration.read(record_owner(raw_terminal.seq + 1), "turn_ended_result", terminal_result)

        raw_child_input = next(
            row
            for row in parent_rows
            if row.record.kind == "input_applied" and row.record._payload["input"]["kind"] == "child_result"
        )
        hydrated_child_input = hydrate_records((raw_child_input,), parent_hydration)[0]
        assert declared_content_ref(raw_child_input.record._payload["input"]["payload"]["result"]) is not None
        assert hydrated_child_input.record.payload["input"]["payload"]["result"] == "child done"

        raw_parent_result = next(
            row
            for row in parent_rows
            if row.record.kind == "op_completed"
            and row.record.operation_id == parked.record.operation_id
            and row.record._payload["result"] is not None
        )
        hydrated_parent_result = hydrate_records((raw_parent_result,), parent_hydration)[0]
        assert declared_content_ref(raw_parent_result.record._payload["result"]) is not None
        assert hydrated_parent_result.record.payload["result"]["metadata"]["final_answer"] == "child done"
    finally:
        driver.close()


def test_ordinary_child_delivery_hydrates_turn_result_without_admission(tmp_path) -> None:
    @function_tool
    def child_task() -> str:
        raise AssertionError("custom child must run through the session kernel")

    driver = SessionDriver()
    try:
        driver.create("parent", str(tmp_path), {})
        driver.push("parent", InboxItem("initial", "user", {"content": "go"}))
        parent_runtime = driver.runtime(
            Agent("parent", "Delegate.", tools=[child_task], model="parent"),
            RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.from_steps(
                    "scripted",
                    "parent",
                    [
                        LLMResponse("", [ToolCall("child-call", "child_task", {})]),
                        LLMResponse("parent done"),
                    ],
                ),
            ),
            session_id="parent",
        )
        parent_runtime.children["child_task"] = lambda _plan: ChildSession(
            replace(
                fixture_child_spec(driver.store, "parent", "child"),
                workspace=str(tmp_path / "child"),
                attributes={"ordinary": True},
            ),
            "do child",
        )
        drive(driver.store, "parent", runtime=parent_runtime)

        child_runtime = driver.runtime(
            Agent("worker", "Work.", model="worker"),
            RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.from_steps("scripted", "worker", [LLMResponse("child done")]),
            ),
            session_id="child",
        )
        drive(driver.store, "child", runtime=child_runtime)
        with driver.store.atomic() as tx:
            child_delivery(driver.store, tx, "child")
        drive(driver.store, "parent", runtime=parent_runtime)

        raw_terminal = next(row for row in driver.store.read_state("child")[1] if row.record.kind == "turn_ended")
        child_hydration = SessionHydration.from_store(driver.store, tenant_id=driver.tenant_id, session_id="child")
        hydrated_terminal = hydrate_records((raw_terminal,), child_hydration)[0]
        assert declared_content_ref(raw_terminal.record._payload["result"]) is not None
        assert hydrated_terminal.record.payload["result"] == "child done"

        parent_rows = driver.store.read_state("parent")[1]
        child_operation = next(row.record.operation_id for row in parent_rows if row.record.kind == "op_parked")
        raw_parent_result = next(
            row for row in parent_rows if row.record.kind == "op_completed" and row.record.operation_id == child_operation
        )
        parent_hydration = SessionHydration.from_store(driver.store, tenant_id=driver.tenant_id, session_id="parent")
        hydrated_parent_result = hydrate_records((raw_parent_result,), parent_hydration)[0]
        assert declared_content_ref(raw_parent_result.record._payload["result"]) is not None
        assert hydrated_parent_result.record.payload["result"]["content"] == "child done"

        ordinary = delegation.outcome(driver.store, "child", child_runtime)
        assert ordinary.agent_name == "worker"
        assert ordinary.final_answer == "child done"
    finally:
        driver.close()


def test_session_principal_ref_shaped_ingress_attributes_hydrate_once() -> None:
    plaintext_attributes = ContentRef(
        content_id="content_plaintext_principal_attributes",
        mac="0" * 64,
        size=17,
        key_id="key_plaintext_principal_attributes",
        labels=ContentLabels((SourceKind.USER,), Classification.CONFIDENTIAL),
    ).to_dict()
    principal = {
        "tenant_id": "default",
        "subject_id": "ingress-user",
        "subject_kind": "user",
        "attributes": plaintext_attributes,
    }
    driver = SessionDriver()
    try:
        driver.create("session", "/workspace", principal=principal)
        stored = driver.store.read("session", limit=1).records[0]
        assert stored.record.payload["principal"]["attributes"] != plaintext_attributes

        hydration = SessionHydration.from_store(driver.store, tenant_id=driver.tenant_id, session_id="session")
        hydrated = hydrate_records((stored,), hydration)[0].record
        assert hydrated.payload["principal"]["attributes"] == plaintext_attributes
    finally:
        driver.close()
