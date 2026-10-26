"""Author A2 vectors from real App Server producers, using only local doubles."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts.session_kernel_fixtures import FixedDatetime, Fixtures, facts, independent_bytes, write_json
from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.app_server.host import model_summary
from vv_agent.config import resolve_model_endpoint
from vv_agent.session.records import SessionSpec
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import LLMResponse, Message


def produce() -> dict:
    fixtures = Fixtures()
    calls = []

    def complete(request):
        calls.append(request.metadata.get("purpose"))
        return LLMResponse(
            json.dumps({"original_user_messages": ["request"], "current_work_state": "Continue"})
            if request.metadata.get("purpose") == "compaction"
            else "done"
        )

    catalog = {
        "VERSION": "2",
        "backends": {
            "test": {
                "models": {
                    "known": {
                        "id": "known-id",
                        "endpoints": ["local"],
                        "context_length": 100000,
                        "max_output_tokens": 9000,
                        "function_call_available": True,
                        "native_multimodal": True,
                        "reasoning_available": True,
                        "supported_reasoning_efforts": ["low", "high"],
                        "default_reasoning_effort": "high",
                        "prompt_cache_available": False,
                        "continuation_available": True,
                    },
                    "unknown": {"id": "unknown-id", "endpoints": ["local"]},
                }
            }
        },
        "endpoints": [{"id": "local", "api_key": "placeholder", "api_base": "https://example.invalid/v1"}],
    }
    host = DefaultAppServerHost(
        agent=Agent("fixture", "Be precise."),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.from_callback("test", "m", complete),
            workspace="/fixture",
            tool_registry_factory=ToolRegistry,
            metadata={"memory_keep_recent_messages": 1},
        ),
        models=[model_summary(resolve_model_endpoint(catalog, "test", key)) for key in ("known", "unknown")],
    )
    server = AppServer(
        transport=ChannelTransport(connection_id="a2", outbound_capacity=10000), store=fixtures.kernel.store, host=host
    )

    def request(identity, method, params):
        server.processor.process_message("a2", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (wire := server.transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return wire

    with ExitStack() as stack:
        stack.callback(server.kernel.close)
        stack.callback(fixtures.kernel.close)
        stack.enter_context(patch("vv_agent.session.kernel.time.monotonic_ns", return_value=100_000_000))
        stack.enter_context(patch("vv_agent.prompt.builder.datetime", FixedDatetime))
        assert request(0, "initialize", {"clientInfo": {"name": "fixture"}})["result"]["protocolVersion"] == "v4"
        with fixtures.kernel.store.atomic() as tx:
            tx.create(
                SessionSpec(
                    "a2",
                    "test",
                    "/fixture",
                    attributes={
                        "app_server": {"agent_key": "default", "cwd": "/fixture", "metadata": {}},
                        "seed": {
                            "messages": [
                                Message("user", "request").to_dict(),
                                Message("assistant", "old facts " * 1000).to_dict(),
                            ],
                            "shared_state": {},
                        },
                    },
                ),
                consumers=("events", "app_server", "traces"),
            )
        assert "result" in request(
            1,
            "turn/start",
            {"threadId": "a2", "inputId": "prompt", "expectedTurnId": None, "input": [{"type": "text", "text": "go"}]},
        )
        server.run_adapter.join()
        before = request(2, "thread/context/read", {"threadId": "a2"})["result"]
        params = {"threadId": "a2", "actionId": "compact", "expectedContextVersion": before["contextVersion"]}
        receipt = request(3, "thread/compact/start", params)["result"]
        server.run_adapter.join()
        after = request(4, "thread/context/read", {"threadId": "a2"})["result"]
        replay = request(5, "thread/compact/start", params)["result"]
        conflict = request(6, "thread/compact/start", {"threadId": "a2", "actionId": "compact"})["error"]
        stale = request(7, "thread/compact/start", params | {"actionId": "stale"})["error"]
        assert calls == ["primary", "compaction"]
        assert replay == receipt | {"replayed": True}
        assert after["contextVersion"] != before["contextVersion"]
        assert after["lastCompaction"] and "Continue" in after["lastCompaction"]["summary"]
        from vv_agent.app_server.item_mapper import map_run_event
        from vv_agent.session.projection import project_records

        lifecycle = [
            {"method": method, "params": params}
            for event in project_records(fixtures.kernel.store.read_state("a2")[1])
            for method, params in map_run_event(event, thread_id="a2", turn_id=event.run_id).additional_notifications
            if method.startswith("contextCompaction/")
        ]
        fork_params = {"threadId": "a2", "forkId": "branch", "workspace": {"snapshot": "prepared"}}
        fork = request("fork", "thread/fork", fork_params)["result"]
        fork_replay = request("fork-replay", "thread/fork", fork_params)["result"]
        fork_conflict = request("fork-conflict", "thread/fork", fork_params | {"workspace": {}})["error"]
        fork_context = request("fork-context", "thread/context/read", {"threadId": fork["threadId"]})["result"]
        fork_seed = fixtures.kernel.store.read(fork["threadId"]).records[0].record._payload["attributes"]["seed"]
        return {
            "fork": {
                "params": fork_params,
                "receipt": fork,
                "replay": fork_replay,
                "conflict": fork_conflict,
                "context": fork_context,
                "seed": fork_seed,
            },
            "lifecycle": lifecycle,
            "schema_version": 1,
            "protocol_version": "v4",
            "before": before,
            "after": after,
            "receipt": receipt,
            "replay": replay,
            "conflict": conflict,
            "stale": stale,
            "modelCalls": calls,
            "models": request(8, "model/list", {})["result"]["models"],
        }


def exports(output):
    from scripts._session_kernel_replacements import Author

    with (
        patch("vv_agent.session.kernel.time.monotonic_ns", return_value=100_000_000),
        patch("vv_agent.prompt.builder.datetime", FixedDatetime),
    ):
        fixtures = Fixtures()
        try:
            fixtures.produce()
            app = fixtures.app_server()
            author = Author(fixtures, independent_bytes, facts)
            author.app_observable(app)
            author.public_api()
            write_json(output, "public_api.json", author.outputs["public_api.json"])
            from scripts._session_kernel_curation import app_server_source, curate_app_server

            write_json(output, "app_server_observable.json", author.outputs["app_server_observable.json"])
            for entry, raw in zip(app["transcripts"], independent_bytes(app["transcripts"]), strict=True):
                entry.update(facts(raw))
            curate_app_server(app)
            (output / "app_server_protocol.json").write_text(app_server_source(app))
        finally:
            fixtures.kernel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exports", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out, "app_server_a2.json", produce())
    if args.exports:
        exports(args.out)
