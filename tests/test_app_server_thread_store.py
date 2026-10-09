"""Thread, turn and item snapshots are projections of the sole session ledger."""

from dataclasses import replace

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider, SQLiteStore
from vv_agent.app_server import AppServer, ChannelTransport
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.types import LLMResponse, ToolCall


@pytest.fixture(params=["memory", "file"])
def server(request, tmp_path):
    with SQLiteStore.standalone(":memory:" if request.param == "memory" else str(tmp_path / "threads.sqlite")) as store:
        store.install_schema()
        value = AppServer(
            transport=ChannelTransport(connection_id="owner"),
            store=store,
            host=DefaultAppServerHost(
                agent=Agent("assistant", "Answer."),
                run_config=RunConfig(
                    model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("first"), LLMResponse("second")])
                ),
            ),
        )
        yield value
        for handle in value.kernel.handles:
            handle.cancel("test teardown")
        value.run_adapter.join()
        value.kernel.close()


def start(server, thread, text="hello"):
    started = server.run_adapter.start_turn(
        connection_id="owner", thread_id=thread.thread_id, input=[{"type": "text", "text": text}]
    )
    started.handle.result(timeout=2)
    server.run_adapter.join()
    return started.turn


def test_create_thread_returns_stable_id_and_stores_metadata(server):
    thread = server.store.create_thread(agent_key="default", cwd="/tmp/project", metadata={"source": "test"})
    assert thread.thread_id == "thread_1"
    assert thread.agent_key == "default"
    assert thread.cwd == "/tmp/project"
    assert thread.metadata == {"source": "test"}
    rows = server.kernel.store.read_state(thread.thread_id)[1]
    assert rows[0].record.payload["attributes"]["app_server"]["metadata"] == thread.metadata


def test_real_turn_links_to_thread_and_retains_wire_input(server):
    thread = server.store.create_thread(agent_key="default")
    turn = start(server, thread)
    assert turn.turn_id == turn.run_id == "thread_1/turn/turn_1"
    assert turn.thread_id == thread.thread_id
    assert turn.input == [{"type": "text", "text": "hello"}]
    assert server.store.read_thread(thread.thread_id).turns[0].status == "completed"


def test_read_thread_preserves_turn_and_item_order(server):
    thread = server.store.create_thread(agent_key="default")
    first, second = start(server, thread, "first"), start(server, thread, "second")
    snapshot = server.store.read_thread(thread.thread_id)
    assert [turn.turn_id for turn in snapshot.turns] == [first.turn_id, second.turn_id]
    messages = [item for item in snapshot.items if item.item_type == "agentMessage"]
    assert [item.payload["text"] for item in messages] == ["first", "second"]
    assert [item.turn_id for item in messages] == [first.turn_id, second.turn_id]
    assert len({item.item_id for item in snapshot.items}) == len(snapshot.items)


def test_archive_hides_thread_from_active_list(server):
    thread = server.store.create_thread(agent_key="default")
    assert [record.thread_id for record in server.store.list_threads()] == [thread.thread_id]
    server.store.archive_thread(thread.thread_id)
    assert server.store.list_threads() == []
    assert [record.thread_id for record in server.store.list_threads(include_archived=True)] == [thread.thread_id]
    assert server.store.read_thread(thread.thread_id).thread.status == "archived"


def test_projection_reopen_preserves_metadata_turns_items_and_status(server):
    from vv_agent.app_server.thread_store import ThreadStore

    thread = server.store.create_thread(agent_key="default", metadata={"source": "persist"})
    start(server, thread)
    reopened = ThreadStore(server.kernel.store)
    assert reopened.read_thread(thread.thread_id) == server.store.read_thread(thread.thread_id)
    assert reopened.read_thread(thread.thread_id).thread.status == "idle"


def test_reopening_waiting_thread_retains_original_active_turn(server):
    server.host._run_config.model_provider = ScriptedModelProvider.new(
        "test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which?"})])]
    )
    thread = server.store.create_thread(agent_key="default")
    turn = start(server, thread)
    from vv_agent.app_server.thread_store import ThreadStore

    reopened = ThreadStore(server.kernel.store).read_thread(thread.thread_id)
    assert reopened.thread.status == "interrupted"
    assert reopened.thread.active_turn_id == turn.turn_id
    assert reopened.turns[0].result["waitReason"] == "Which?"


def test_item_validation_has_no_second_ledger_and_rejects_fabrication(server):
    thread = server.store.create_thread(agent_key="default")
    start(server, thread)
    before = server.kernel.store.read_state(thread.thread_id)[1]
    item = server.store.read_thread(thread.thread_id).items[-1]
    assert server.store.append_item(item, run_event_id="projection") is True
    assert server.store.append_item(item, run_event_id="projection") is True
    with pytest.raises(ValueError, match="project kernel records"):
        server.store.append_item(replace(item, payload={"text": "replacement"}))
    assert server.kernel.store.read_state(thread.thread_id)[1] == before
    assert server.store.read_thread(thread.thread_id).items[-1] == item
