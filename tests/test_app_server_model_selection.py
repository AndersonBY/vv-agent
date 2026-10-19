"""App Server admits strict selection commands into the canonical session inbox."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost, ThreadModelSelectParams
from vv_agent.app_server.schema import json_schema_bundle
from vv_agent.model import ScriptedModelProvider


def test_model_selection_method_admission_replay_conflict_and_schema():
    transport = ChannelTransport(connection_id="test")
    server = AppServer(
        transport=transport,
        host=DefaultAppServerHost(
            agent=Agent("test", "Answer."), run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []))
        ),
    )

    def request(identity, method, params):
        server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (response := transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return response

    try:
        request(0, "initialize", {"clientInfo": {"name": "test"}})
        thread = request(1, "thread/start", {})["result"]["threadId"]
        params = ThreadModelSelectParams(thread, "command", {"model": "m", "route": {}}).to_dict()
        expected = {"threadId": thread, "inputId": "command", "accepted": True}
        assert request(2, "thread/model/select", params)["result"] == expected
        assert request(3, "thread/model/select", params)["result"] == expected
        assert request(4, "thread/model/select", params | {"selection": None})["error"]["code"] == -32602
        for identity, bad in enumerate(
            [
                params | {"extra": 1},
                {"threadId": thread},
                params | {"inputId": ""},
                params | {"selection": {"model": "m", "route": []}},
                params | {"selection": {"model": "m", "route": {}, "model_settings": {"max_tokens": False}}},
            ],
            5,
        ):
            assert request(identity, "thread/model/select", bad)["error"]["code"] == -32602
        stored = server.kernel.store.peek_inbox(thread)
        assert len(stored) == 1 and stored[0].item.kind == "model_selection"
        assert stored[0].item.target_turn_id is None and stored[0].item.generation is None
        schema = json.loads(json_schema_bundle()["ClientRequest"])
        validator = Draft202012Validator(schema)
        wire = {"jsonrpc": "2.0", "id": 1, "method": "thread/model/select", "params": params}
        validator.validate(wire)
        with pytest.raises(ValidationError):
            validator.validate(wire | {"params": params | {"extra": True}})
    finally:
        server.kernel.close()
