import pytest
from session.app_server_s1_helpers import bind_development_content

from vv_agent.app_server.protocol.thread import ThreadModelSelectParams
from vv_agent.app_server.server import AppServer
from vv_agent.kernel._content import INPUT_ACTOR_ATTRIBUTES, SESSION_PRINCIPAL_ATTRIBUTES
from vv_agent.kernel._requirements import SessionRequirements
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


def test_app_server_writers_use_trusted_ingress_principal(tmp_path):
    root_store = SQLiteStore(str(tmp_path / "app-server.sqlite"))
    root_store.install_schema()
    root_store.configure_input_admission(allow_unsigned=True)
    root_store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    bind_development_content(root_store)
    store = root_store.for_tenant("tenant-a")
    server = AppServer(store=store)
    principal = {
        "tenant_id": "tenant-a",
        "subject_id": "ingress-user",
        "subject_kind": "user",
        "attributes": {"role": "operator"},
    }
    try:
        server.store.create_thread_receipt(
            input_id="thread",
            agent_key="agent",
            cwd=".",
            principal=principal,
            bootstrap=SessionRequirements(2, server.kernel.default_handler_version, ()),
        )

        server.run_adapter.select_model(
            thread_id="thread",
            input_id="select",
            selection={"model": "test-model", "route": {}},
            actor=principal,
        )
        server.run_adapter.submit_input(
            connection_id=None,
            thread_id="thread",
            input_id="prompt",
            expected_turn_id=None,
            kind="start",
            input=[{"type": "text", "text": "go"}],
            actor=principal,
        )
        server.run_adapter.select_model(
            thread_id="thread",
            input_id="owner-select",
            selection={"model": "owner-model", "route": {}},
        )

        items = {item.item.input_id: item.item for item in server.kernel.store.peek_inbox("thread")}
        selection = items["select"]
        prompt = items["prompt"]
        owner_selection = items["owner-select"]
        selection_actor = selection.actor
        prompt_actor = prompt.actor
        owner_selection_actor = owner_selection.actor

        assert selection_actor is not None
        assert prompt_actor is not None
        assert owner_selection_actor is not None
        assert {key: selection_actor[key] for key in ("tenant_id", "subject_kind", "subject_id")} == {
            "tenant_id": "tenant-a",
            "subject_kind": "user",
            "subject_id": "ingress-user",
        }
        assert {key: prompt_actor[key] for key in ("tenant_id", "subject_kind", "subject_id")} == {
            "tenant_id": "tenant-a",
            "subject_kind": "user",
            "subject_id": "ingress-user",
        }
        assert owner_selection_actor["subject_id"] == "ingress-user"
        assert owner_selection_actor["subject_kind"] == "user"
        assert "content_id" in selection.payload["selection"]
        assert "content_id" in prompt.payload["content"]
        assert server.kernel.read_input_content(
            "thread", "select", selection_actor["attributes"], slot=INPUT_ACTOR_ATTRIBUTES
        ) == {"role": "operator"}
        assert server.kernel.read_input_content("thread", "prompt", prompt_actor["attributes"], slot=INPUT_ACTOR_ATTRIBUTES) == {
            "role": "operator"
        }

        created = server.kernel.store.read("thread", limit=1).records[0]
        created_principal = created.record.payload["principal"]
        assert created_principal["subject_kind"] == "user"
        assert created_principal["subject_id"] == "ingress-user"
        assert server.kernel.read_record_content(
            "thread",
            created.seq,
            created_principal["attributes"],
            slot=SESSION_PRINCIPAL_ATTRIBUTES,
        ) == {"role": "operator"}
        assert "ingress-user" not in str(prompt.payload["content"])

        with pytest.raises(ValueError, match="session tenant"):
            server.run_adapter.select_model(
                thread_id="thread",
                input_id="foreign-select",
                selection={"model": "test-model", "route": {}},
                actor=principal | {"tenant_id": "tenant-b"},
            )
        assert all(item.item.input_id != "foreign-select" for item in server.kernel.store.peek_inbox("thread"))
    finally:
        server.kernel.close()
