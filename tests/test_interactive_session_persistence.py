"""Creation seed and durable transcript projections replace mutable backing sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from session.fold_history import logical_record
from session.helpers import fixture_key_provider, wrap_fixture_store
from support.s1_fixture_values import append_fixture_records, clear_fixture_state_cache

from vv_agent import (
    Agent,
    AgentSessionOptions,
    AgentStatus,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    Message,
    ScriptedModelProvider,
    SQLiteStore,
    handoff,
)
from vv_agent.kernel._budget_values import RunBudgetLimits
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.records import digest
from vv_agent.session.store import Conflict
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult


def client(tmp_path, steps, store=None, *, model_provider=None):
    return InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=model_provider or ScriptedModelProvider.from_steps("test", "m", steps),
            workspace=tmp_path,
        ),
        store=store,
    )


def file_store(database, key_provider):
    store = wrap_fixture_store(SQLiteStore(str(database)), key_provider)
    store.install_schema()
    store.configure_input_admission(allow_unsigned=True)
    return store


@pytest.mark.capability("interactive")
def test_creation_seed_hydrates_and_reuses_durable_session_without_duplicates(tmp_path: Path):
    requests = []

    def respond(request):
        requests.append(request.messages)
        return LLMResponse(f"done {len(requests)}")

    seed = {
        "messages": [Message("user", "earlier question").to_dict(), Message("assistant", "earlier answer").to_dict()],
        "shared_state": {"host": "seeded"},
    }
    provider = fixture_key_provider()
    store = file_store(tmp_path / "sessions.sqlite", provider)
    first_client = client(tmp_path, [respond], store)
    first = first_client.create_session(agent=Agent("assistant", "Remember context."), session_id="thread", session=seed)
    assert [m.content for m in first.messages] == ["earlier question", "earlier answer"]
    result = first.prompt("first", auto_follow_up=False)
    assert result.raw_result.session_id == "thread"
    first_client.driver.close()
    store.connection.close()
    reopened = file_store(tmp_path / "sessions.sqlite", provider)
    second_client = client(tmp_path, [respond], reopened)
    second = second_client.create_session(agent=Agent("assistant", "Remember context."), session_id="thread")
    assert second.shared_state["host"] == "seeded"
    second.prompt("second", auto_follow_up=False)
    assert [m.content for m in second.messages if m.role == "user"] == ["earlier question", "first", "second"]
    assert [m.content for m in requests[-1] if m.role == "user"] == ["earlier question", "first", "second"]
    second_client.driver.close()
    reopened.connection.close()

    archive_store = file_store(tmp_path / "sessions.sqlite", provider)
    archiver = client(tmp_path, [], archive_store)
    try:
        cold = archiver.create_session(agent=Agent("assistant", "Remember context."), session_id="thread")
        assert not archiver.driver._runtimes
        cold.archive()
        assert archiver.driver.store.read_state("thread")[0].archived
        assert len(requests) == 2
    finally:
        archiver.driver.close()
        archive_store.connection.close()


@pytest.mark.parametrize("inline", [False, True])
def test_cold_reopen_is_read_only_and_answers_the_retained_turn(tmp_path, monkeypatch, inline):
    database = tmp_path / "parked.sqlite"
    key_provider = fixture_key_provider()
    store = file_store(database, key_provider)
    requests = []

    def ask(request):
        requests.append(request)
        return LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose a value."})])

    def finish(request):
        requests.append(request)
        return LLMResponse("resumed")

    agent = InteractiveAgentDefinition(description="Ask.", model="m") if inline else Agent("desktop", "Ask.", model="m")
    first_client = client(tmp_path, [ask], store)
    first = first_client.create_session(agent=agent, session_id="parked")
    try:
        waiting = first.prompt("go", auto_follow_up=False)
        assert waiting.status == AgentStatus.WAIT_USER
        messages = first.messages
        form = waiting.metadata["session_waits"][0]["interaction"]
        assert len(requests) == 1
    finally:
        first_client.driver.close()
        store.connection.close()

    store = file_store(database, key_provider)
    second_client = client(tmp_path, [finish], store)
    driver = second_client.driver
    sid = "parked"
    try:
        baseline = driver.store.read_state(sid), driver.store.peek_inbox(sid), store.connection.total_changes

        def forbidden(*_args, **_kwargs):
            pytest.fail("projection tried to use an execution callback or a write transaction")

        with monkeypatch.context() as readonly:
            for callback in ("resolve", "client", "default_settings", "default_model_ref"):
                readonly.setattr(ScriptedModelProvider, callback, forbidden)
            readonly.setattr(driver._worker, "start", forbidden)
            readonly.setattr(type(driver.store), "atomic", forbidden)
            second = second_client.create_session(agent=agent, session_id=sid)
            restored = second.latest_run
            assert restored is not None
            assert restored.status == AgentStatus.WAIT_USER
            assert restored.run_id == waiting.run_id
            assert restored.metadata["session_waits"][0]["interaction"] == form
            assert restored.resolved_model is not None
            assert (restored.resolved_model.backend, restored.resolved_model.model_id) == ("test", "m")
            assert restored.resolved_model.context_length == waiting.resolved_model.context_length
            assert restored.resolved_model.max_output_tokens == waiting.resolved_model.max_output_tokens
            assert second.messages == messages
            del second
            second = second_client.create_session(agent=agent, session_id=sid)
        assert not driver._runtimes
        assert (driver.store.read_state(sid), driver.store.peek_inbox(sid), store.connection.total_changes) == baseline

        resumed = second.continue_run("chosen value")
        assert resumed.final_output == "resumed"
        assert resumed.run_id == waiting.run_id
        assert len(requests) == 2
        state, rows, _ = driver.store.read_state(sid)
        assert len(state.turns) == 1
        assert sum(row.record.kind == "turn_started" for row in rows) == 1
        assert sum(row.record.kind == "turn_ended" for row in rows) == 1
        replies = [
            row.record.payload["input"]
            for row in rows
            if row.record.kind == "input_applied" and row.record.payload["input"]["target_turn_id"] == form["turnId"]
        ]
        assert len(replies) == 1
        assert replies[0]["generation"] == form["generation"]
        assert not driver.store.peek_inbox(sid)
    finally:
        driver.close()
        store.connection.close()


def test_cold_reopen_projects_typed_handoff_result(tmp_path, monkeypatch):
    database = tmp_path / "handoff.sqlite"
    key_provider = fixture_key_provider()
    store = file_store(database, key_provider)
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    writer = Agent("writer", "Write a JSON object.", model="writer", output_type=dict)
    triage = Agent("triage", "Transfer.", model="triage", handoffs=[handoff(agent=writer, description="Write.")])
    owner = client(
        tmp_path,
        [
            LLMResponse("", [ToolCall("transfer", "transfer_to_writer", {"input": "Write it."})]),
            LLMResponse('{"answer":"done"}'),
        ],
        store,
    )
    try:
        first = owner.create_session(agent=triage, session_id="handoff")
        result = first.prompt("go")
        assert result.final_output == {"answer": "done"}
        assert result.agent_name == "writer" and result.status == AgentStatus.COMPLETED
        _, records, _ = owner.driver.store.read_state(first.session_id)
        (receipt,) = (
            row
            for row in records
            if row.record.kind == "op_completed" and row.record.operation_id is not None and "/tool/" in row.record.operation_id
        )
        payload = hydrate_record_payload(receipt, hydration_for_session_store(owner.driver.store, first.session_id))
        tool_result = ToolExecutionResult.from_dict(payload["result"])
        assert tool_result.tool_call_id == "transfer"
        assert tool_result.directive.value == "finish"
        assert tool_result.content == result.raw_result.final_answer
        assert json.loads(tool_result.content) == result.final_output
    finally:
        owner.driver.close()
        store.connection.close()

    store = file_store(database, key_provider)
    owner = client(tmp_path, [], store)
    try:

        def forbidden(*_args, **_kwargs):
            pytest.fail("opening a completed handoff tried to call its provider")

        monkeypatch.setattr(ScriptedModelProvider, "resolve", forbidden)
        monkeypatch.setattr(ScriptedModelProvider, "client", forbidden)
        restored = owner.create_session(agent=triage, session_id="handoff").latest_run
        assert restored is not None
        assert (restored.run_id, restored.agent_name, restored.final_output) == (
            result.run_id,
            "writer",
            {"answer": "done"},
        )
        assert restored.resolved_model is not None and restored.resolved_model.model_id == "writer"
        assert not owner.driver._runtimes
    finally:
        owner.driver.close()
        store.connection.close()


def test_create_session_retries_after_missing_session_lookup(tmp_path: Path, monkeypatch):
    owner = client(tmp_path, [])

    def missing(_store, session_id):
        raise KeyError(session_id)

    monkeypatch.setattr(owner.driver, "_created_payload", missing)
    session = owner.create_session(
        agent=InteractiveAgentDefinition(description="desktop agent", model="m"),
        session_id="new-id",
    )
    assert session.session_id == "new-id"
    owner.driver.close()


def test_interactive_definition_restores_and_continues_hydrated_active_turn(tmp_path: Path, monkeypatch):
    class StopBeforeDispatch(Exception):
        pass

    provider = ScriptedModelProvider.from_steps("test", "m", [LLMResponse("done")])
    key_provider = fixture_key_provider()
    database = str(tmp_path / "sessions.sqlite")
    store = file_store(database, key_provider)
    first_client = client(tmp_path, [], store, model_provider=provider)
    first_client.driver.create("active", str(tmp_path), {"seed": {"messages": [], "shared_state": {}}})
    agent = InteractiveAgentDefinition(description="desktop agent", model="m")

    prepared = {}

    def capture_current_definition(session_id, sdk_agent, config, _content, *, task, **_kwargs):
        prepared["task"] = task
        runtime = first_client.driver.runtime(sdk_agent, config, task=task, session_id=session_id)
        prepared["definition"] = runtime.definition(task)
        raise StopBeforeDispatch

    monkeypatch.setattr(first_client.driver, "start", capture_current_definition)
    with pytest.raises(StopBeforeDispatch):
        first_client._execute(prompt="go", agent=agent, workspace=tmp_path, session_id="active")
    first_task = prepared["task"]
    definition = prepared["definition"]
    assert first_task.user_prompt == "go"
    append_fixture_records(
        store,
        "active",
        [
            logical_record(
                "turn_started",
                sid="active",
                tid=first_task.task_id,
                definition=definition,
                definition_digest=digest(definition),
                budget=RunBudgetLimits().to_dict(),
            )
        ],
    )
    first_client.driver.close()
    clear_fixture_state_cache(store)
    store.connection.close()

    store = file_store(database, key_provider)
    owner = client(tmp_path, [], store, model_provider=provider)

    class StopExecute(Exception):
        pass

    retained = {}

    def stop_execute(_session_id, _agent, _config, _content, *, task, **_kwargs):
        retained["task"] = task
        raise StopExecute

    monkeypatch.setattr(owner.driver, "start", stop_execute)
    with pytest.raises(StopExecute):
        owner._execute(prompt="continue", agent=agent, workspace=tmp_path, session_id="active")
    assert retained["task"] == first_task
    assert retained["task"].user_prompt == "go"
    assert retained["task"].prompt_bundle == first_task.prompt_bundle
    owner.driver.close()
    store.connection.close()


def test_creation_seed_is_immutable_and_transcript_is_detached(tmp_path):
    owner = client(tmp_path, [])
    agent = Agent("assistant", "Answer.")
    seed = {"messages": [Message("user", "stored").to_dict()], "shared_state": {"nested": {"value": 1}}}
    session = owner.create_session(agent=agent, session_id="immutable", session=seed)
    seed["messages"][0]["content"] = "host mutation"
    session.messages[0].content = "projection mutation"
    session.shared_state["nested"]["value"] = 9
    assert session.messages[0].content == "stored" and session.shared_state["nested"]["value"] == 1
    with pytest.raises(Conflict):
        owner.create_session(agent=agent, session_id="immutable", session=seed)
    owner.driver.close()


@pytest.mark.parametrize("seed", [{}, {"messages": []}, {"messages": [], "shared_state": {}, "extra": True}])
def test_seed_is_closed(tmp_path, seed):
    owner = client(tmp_path, [])
    with pytest.raises(ValueError, match="exactly"):
        owner.create_session(agent=Agent("assistant", "Answer."), session=seed)
    owner.driver.close()


def test_store_sessions_are_isolated(tmp_path):
    owner = client(tmp_path, [LLMResponse("one"), LLMResponse("two")])
    a = owner.create_session(agent=Agent("assistant", "Answer."), session_id="a")
    b = owner.create_session(agent=Agent("assistant", "Answer."), session_id="b")
    a.prompt("first", auto_follow_up=False)
    b.prompt("second", auto_follow_up=False)
    assert [m.content for m in a.messages if m.role == "user"] == ["first"]
    assert [m.content for m in b.messages if m.role == "user"] == ["second"]
    owner.driver.close()
