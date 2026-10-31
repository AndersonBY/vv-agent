"""Typed interactions from real AppServer, ChannelTransport and JSON-RPC producers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vv_agent import Agent, AppServerUserInteraction, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.canonical_json import canonical_json_bytes, canonical_json_sha256
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools import build_default_registry
from vv_agent.tools.registry import RegistryToolExecutor
from vv_agent.types import LLMResponse, ToolCall
from vv_agent.workspace import MemoryWorkspaceBackend

THREAD = "interaction-thread"
TURN = THREAD + "/turn/question"


def target_of(form):
    return AppServerUserInteraction.from_dict(form).target.to_dict()


class InteractionHost(DefaultAppServerHost):
    def __init__(self, form=None, *, no_tool=False, questions=1, agent=None, model=None):
        self.form = {"question": "Choose", **(form or {})}
        self.calls, self.asks, self.requests, self.denied, self.authorizations = 0, 0, [], set(), []
        self.questions = questions
        self.backend = MemoryWorkspaceBackend()

        def respond(request):
            self.calls += 1
            self.requests.append(request)
            if model:
                return model(request)
            answered = sum(m.role == "tool" and m.tool_call_id.startswith("ask-") for m in request.messages)
            # The frozen plan supplies the original arguments even when this host is rebuilt.
            if no_tool:
                return LLMResponse("Question without tool")
            if answered < self.questions:
                return LLMResponse("", [ToolCall("ask-" + str(answered), "ask_user", dict(self.form))])
            return LLMResponse("done")

        super().__init__(
            agent=agent or Agent("interactions", "Ask and accept the answer.", model="root"),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "root", respond),
                workspace_backend=self.backend,
                approval_timeout_seconds=10,
                no_tool_policy="wait_user" if no_tool else "finish",
                tool_registry_factory=self.registry,
            ),
        )

    def registry(self):
        registry = build_default_registry()
        executor = registry.get_executor("ask_user")
        assert isinstance(executor, RegistryToolExecutor)
        original = executor.handler

        def counted_ask(context, arguments):
            self.asks += 1
            return original(context, arguments)

        executor.handler = counted_ask
        return registry

    def authorize(self, connection_id, method, params):
        self.authorizations.append((method, params.get("threadId")))
        return params.get("threadId") not in self.denied


class InteractionServer:
    def __init__(self, store, host=None, *, connection="interaction-owner"):
        self.host = host or InteractionHost()
        self.transport = ChannelTransport(connection_id=connection, outbound_capacity=10000)
        self.server = AppServer(transport=self.transport, host=self.host, store=store)
        self.messages, self.identity = [], 0
        self.rpc("initialize", {"clientInfo": {"name": "interactions"}})
        self.server.processor.process_message(connection, {"jsonrpc": "2.0", "method": "initialized"})

    def rpc(self, method, params):
        self.identity += 1
        self.server.processor.process_message(
            self.transport.connection_id,
            {"jsonrpc": "2.0", "id": self.identity, "method": method, "params": params},
        )
        while True:
            wire = self.transport.receive_outbound(timeout=20)
            self.messages.append(wire)
            if wire.get("id") == self.identity:
                return wire

    def join(self):
        self.server.run_adapter.join()

    def start(self, thread=THREAD, *, settle=True):
        assert "result" in self.rpc("thread/start", {"inputId": thread})
        receipt = self.rpc(
            "turn/start",
            {
                "threadId": thread,
                "inputId": "question",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "ask"}],
            },
        )
        assert "result" in receipt, receipt
        if settle:
            self.join()
        return receipt["result"]

    def status(self, thread=THREAD):
        wire = self.rpc("thread/status", {"threadId": thread})
        assert "result" in wire, wire
        return wire["result"]

    def form(self, thread=THREAD):
        return self.status(thread)["pendingInteraction"]

    def answer(self, form, answer, *, action_id="answer", thread=THREAD, turn=TURN):
        wire = self.rpc(
            "turn/action",
            {
                "threadId": thread,
                "turnId": turn,
                "actionId": action_id,
                "action": {"kind": "respond", "target": target_of(form), "answer": answer},
            },
        )
        self.join()
        return wire

    def events(self, thread=THREAD):
        after, events = None, []
        while True:
            params = {"threadId": thread, "limit": 256}
            if after:
                params["afterCursor"] = after
            page = self.rpc("thread/events/list", params)["result"]
            events.extend(e["event"] for e in page["events"])
            if not page["more"]:
                return events
            after = page["nextCursor"]

    def close(self):
        self.server.router.cancel_matching_server_requests()
        self.join()
        self.server.run_adapter.close()
        self.server.kernel.close()
        self.join()


def produce():
    cases = []
    for kind, args, answer in (
        ("text", {}, {"kind": "text", "text": "A free reply"}),
        (
            "single",
            {"options": [" A ", "B", "A"], "allow_custom_options": False},
            {"kind": "single", "selected": ["A"], "custom": []},
        ),
        ("multi", {"options": ["A", "B"], "selection_type": "multi"}, {"kind": "multi", "selected": ["A", "B"], "custom": []}),
        ("custom", {"options": ["A"], "allow_custom_options": True}, {"kind": "single", "selected": [], "custom": ["  Other  "]}),
        ("choice-free-text", {"options": ["A", "B"], "allow_custom_options": False}, {"kind": "text", "text": "NOT_AN_OPTION"}),
    ):
        with SQLiteStore.standalone(":memory:") as store:
            store.install_schema()
            server = InteractionServer(store, InteractionHost(args))
            try:
                server.start()
                form = server.form()
                expected = {k: v for k, v in form.items() if k not in {"kind", "requestDigest"}}
                assert form["requestDigest"] == canonical_json_sha256(expected)
                read = server.rpc("thread/read", {"threadId": THREAD})["result"]
                resumed = server.rpc("thread/resume", {"threadId": THREAD, "subscribe": False})["result"]
                requested = next(e for e in server.events() if e["type"] == "host_interaction_requested")
                assert (
                    read["state"]["pendingInteraction"]
                    == resumed["state"]["pendingInteraction"]
                    == requested["metadata"]["interaction"]
                    == form
                )
                rejected = server.answer(
                    form, {"kind": "single", "selected": ["NOT_AN_OPTION"], "custom": []}, action_id="invalid"
                )
                assert rejected["error"]["code"] == -32602 and server.form() == form
                response = server.answer(form, answer)
                assert response["result"]["accepted"]
                tool = next(m for m in server.host.requests[-1].messages if m.role == "tool" and m.tool_call_id == "ask-0")
                assert json.loads(tool.content)["reply_mode"] == ("free_text" if answer["kind"] == "text" else "selection")
                assert server.answer(form, answer)["result"]["accepted"]
                conflict = server.answer(form, {"kind": "text", "text": "different"})
                assert conflict["error"]["code"] == -32024
                assert server.host.asks == 1
                cases.append(
                    {
                        "name": kind,
                        "form": form,
                        "answer": answer,
                        "modelResult": json.loads(tool.content),
                        "invalidAnswer": rejected["error"],
                        "conflict": conflict["error"],
                        "modelCalls": server.host.calls,
                    }
                )
            finally:
                server.close()
    return {
        "contract": "app_server_interactions",
        "protocolVersion": "v5",
        "limits": {
            "promptBytes": 16384,
            "options": 64,
            "optionBytes": 1024,
            "formBytes": 65536,
            "answerBytes": 65536,
        },
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(produce()) + b"\n")
