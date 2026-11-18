"""Creation seed and durable transcript projections replace mutable backing sessions."""

from __future__ import annotations

from pathlib import Path

import pytest
from session.fold_history import logical_record
from session.helpers import fixture_key_provider, wrap_fixture_store
from support.s1_fixture_values import append_fixture_records, clear_fixture_state_cache

from vv_agent import (
    Agent,
    AgentSessionOptions,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    Message,
    ScriptedModelProvider,
    SQLiteStore,
)
from vv_agent.kernel._budget_values import RunBudgetLimits
from vv_agent.session.records import digest
from vv_agent.session.store import Conflict
from vv_agent.types import LLMResponse


def client(tmp_path, steps, store=None, *, model_provider=None):
    return InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=model_provider or ScriptedModelProvider.from_steps("test", "m", steps),
            workspace=tmp_path,
        ),
        store=store,
    )


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
    store = wrap_fixture_store(SQLiteStore(str(tmp_path / "sessions.sqlite")), provider)
    store.install_schema()
    first_client = client(tmp_path, [respond], store)
    first = first_client.create_session(agent=Agent("assistant", "Remember context."), session_id="thread", session=seed)
    assert [m.content for m in first.messages] == ["earlier question", "earlier answer"]
    result = first.prompt("first", auto_follow_up=False)
    assert result.raw_result.session_id == "thread"
    first_client.driver.close()
    store.connection.close()
    reopened = wrap_fixture_store(SQLiteStore(str(tmp_path / "sessions.sqlite")), provider)
    reopened.install_schema()
    second_client = client(tmp_path, [respond], reopened)
    second = second_client.create_session(agent=Agent("assistant", "Remember context."), session_id="thread")
    assert second.shared_state["host"] == "seeded"
    second.prompt("second", auto_follow_up=False)
    assert [m.content for m in second.messages if m.role == "user"] == ["earlier question", "first", "second"]
    assert [m.content for m in requests[-1] if m.role == "user"] == ["earlier question", "first", "second"]
    second_client.driver.close()
    reopened.connection.close()


def test_create_session_retries_after_missing_session_lookup(tmp_path: Path, monkeypatch):
    owner = client(tmp_path, [])

    def missing(_session_id):
        raise KeyError(_session_id)

    monkeypatch.setattr(owner.driver, "session_attributes", missing)
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
    raw_store = SQLiteStore(database)
    raw_store.install_schema()
    store = wrap_fixture_store(raw_store, key_provider)
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
    raw_store.connection.close()

    raw_store = SQLiteStore(database)
    raw_store.install_schema()
    store = wrap_fixture_store(raw_store, key_provider)
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
    raw_store.connection.close()


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
