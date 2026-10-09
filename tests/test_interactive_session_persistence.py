"""Creation seed and durable transcript projections replace mutable backing sessions."""

from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, Message, ScriptedModelProvider, SQLiteStore
from vv_agent.session.store import Conflict
from vv_agent.types import LLMResponse


def client(tmp_path, steps, store=None):
    return InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=ScriptedModelProvider.from_steps("test", "m", steps),
            workspace=tmp_path,
        ),
        store=store,
    )


def test_creation_seed_hydrates_and_reuses_durable_session_without_duplicates(tmp_path: Path):
    requests = []

    def respond(request):
        requests.append(request.messages)
        return LLMResponse(f"done {len(requests)}")

    seed = {
        "messages": [Message("user", "earlier question").to_dict(), Message("assistant", "earlier answer").to_dict()],
        "shared_state": {"host": "seeded"},
    }
    store = SQLiteStore(str(tmp_path / "sessions.sqlite"))
    first_client = client(tmp_path, [respond], store)
    first = first_client.create_session(agent=Agent("assistant", "Remember context."), session_id="thread", session=seed)
    assert [m.content for m in first.messages] == ["earlier question", "earlier answer"]
    result = first.prompt("first", auto_follow_up=False)
    assert result.raw_result.session_id == "thread"
    first_client.driver.close()
    store.connection.close()
    reopened = SQLiteStore(str(tmp_path / "sessions.sqlite"))
    second_client = client(tmp_path, [respond], reopened)
    second = second_client.create_session(agent=Agent("assistant", "Remember context."), session_id="thread")
    assert second.shared_state["host"] == "seeded"
    second.prompt("second", auto_follow_up=False)
    assert [m.content for m in second.messages if m.role == "user"] == ["earlier question", "first", "second"]
    assert [m.content for m in requests[-1] if m.role == "user"] == ["earlier question", "first", "second"]
    second_client.driver.close()
    reopened.connection.close()


def test_creation_seed_is_immutable_and_transcript_is_detached(tmp_path):
    owner = client(tmp_path, [])
    agent = Agent("assistant", "Answer.")
    seed = {"messages": [Message("user", "stored").to_dict()], "shared_state": {"nested": {"value": 1}}}
    session = owner.create_session(agent=agent, session_id="immutable", session=seed)
    seed["messages"][0]["content"] = "host mutation"
    session.messages[0].content = "projection mutation"
    session.shared_state["nested"]["value"] = 9
    assert session.messages[0].content == "stored" and session.shared_state["nested"]["value"] == 1
    for name in ("replace_messages", "replace_shared_state", "clear_queues", "session"):
        assert not hasattr(session, name)
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
