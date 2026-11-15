import pytest

from vv_agent.app_server.protocol.thread import ThreadModelSelectParams
from vv_agent.app_server.server import AppServer
from vv_agent.session.sqlite import SQLiteStore


def test_model_select_parser_returns_request_dto_without_actor():
    params = ThreadModelSelectParams.from_dict(
        {
            "threadId": "thread",
            "inputId": "select",
            "selection": {"model": "test-model", "route": {}},
        }
    )

    assert params.to_dict() == {
        "threadId": "thread",
        "inputId": "select",
        "selection": {"model": "test-model", "route": {}},
    }
    assert ThreadModelSelectParams.from_dict({"threadId": "thread", "inputId": "clear", "selection": None}).selection is None

    with pytest.raises(ValueError, match="invalid session schema"):
        ThreadModelSelectParams.from_dict(
            {
                "threadId": "thread",
                "inputId": "select",
                "selection": {"model": "test-model", "route": []},
            }
        )


def test_app_server_writers_bind_selection_and_thread_input_to_tenant_actor(monkeypatch):
    root_store = SQLiteStore(":memory:")
    root_store.install_schema()
    store = root_store.for_tenant("tenant-a")
    server = AppServer(store=store)
    try:
        server.kernel.create("thread", ".")
        monkeypatch.setattr(server.run_adapter._store, "read_thread", lambda thread_id: None)

        server.run_adapter.select_model(
            thread_id="thread",
            input_id="select",
            selection={"model": "test-model", "route": {}},
        )
        server.run_adapter.submit_input(
            connection_id=None,
            thread_id="thread",
            input_id="prompt",
            expected_turn_id=None,
            kind="start",
            input=[{"type": "text", "text": "go"}],
        )

        items = {item.item.kind: item.item for item in server.kernel.store.peek_inbox("thread")}
        selection = items["model_selection"]
        prompt = items["user"]
        selection_actor = selection.actor

        assert selection_actor is not None
        assert selection_actor == prompt.actor
        assert selection_actor["tenant_id"] == "tenant-a"
        assert selection_actor["subject_kind"] == "service"
        assert selection_actor["subject_id"] == "app-server"
        assert "content_id" in selection.payload["selection"]
        assert "content_id" in prompt.payload["content"]
    finally:
        server.kernel.close()
