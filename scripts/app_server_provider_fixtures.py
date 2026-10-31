"""Local App Server producer with an independent, durable business database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from vv_agent import Accepted, Agent, Definitive, RunConfig, ScriptedModelProvider, Unknown, function_tool
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.records import InboxItem, Record
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.metadata import ToolMetadata
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult, ToolResultStatus

THREAD = "provider-thread"
TURN = f"{THREAD}/turn/work"


class BusinessDB:
    def __init__(self, path: Path):
        self.path = path
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, plan TEXT NOT NULL, done INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (kind TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def setting(self, key, value=None):
        with closing(self.connect()) as db, db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def count(self, kind):
        with closing(self.connect()) as db:
            return db.execute("SELECT count(*) FROM calls WHERE kind=?", (kind,)).fetchone()[0]

    def call(self, kind):
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO calls VALUES (?)", (kind,))

    def finish(self):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE jobs SET done=1")

    def job(self, key):
        with closing(self.connect()) as db:
            row = db.execute("SELECT plan, done FROM jobs WHERE key=?", (key,)).fetchone()
        assert row is not None
        return Record.parse(row[0].encode()), row[1]


class JobsProvider:
    def __init__(self, business: BusinessDB):
        self.business = business

    def handle(self, plan: Record, key: str):
        return {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": key,
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": f"business-accept/{key}",
            "query_ref": key,
            "cancel_ref": key,
        }

    def submit(self, plan, *, context):
        self.business.call("submit")
        key = context.idempotency_key or f"{plan.session_id}/{plan.operation_id}"
        with closing(self.business.connect()) as db, db:
            if db.execute("SELECT 1 FROM jobs WHERE key=?", (key,)).fetchone() is None:
                db.execute("INSERT INTO jobs VALUES (?, ?, 0)", (key, plan.encode().decode()))
                db.execute("INSERT INTO calls VALUES ('effect')")
            else:
                db.execute("UPDATE jobs SET plan=? WHERE key=?", (plan.encode().decode(), key))
        if self.business.setting("pause_after_accept"):
            self.business.setting("accepted", True)
            Event().wait(60)
        if self.business.setting("submit_result"):
            return self.result(self.handle(plan, key))
        handle = self.handle(plan, key)
        bad = self.business.setting("invalid_handle")
        if bad:
            handle[bad] = 2 if bad == "attempt" else "0" * 64 if bad == "request_digest" else "wrong"
        return Accepted(handle)

    def result(self, handle, *, cancelled=False):
        plan, state = self.business.job(handle["job_id"])
        cancelled = cancelled or state == 2
        value = ToolExecutionResult(
            tool_call_id=plan.payload["request"]["id"],
            content="stopped" if cancelled else "business complete",
            status_code=ToolResultStatus.ERROR if cancelled else ToolResultStatus.SUCCESS,
            error_code="tool_cancelled" if cancelled else None,
            metadata={"definitive_outcome": True} if cancelled else {},
        ).to_dict()
        evidence = (handle["evidence"], f"business-result/{handle['job_id']}")
        outcome = Definitive(value, evidence, usage={"business_units": 1})
        invalid = self.business.setting("invalid")
        if invalid == "call":
            outcome = replace(outcome, result=value | {"tool_call_id": "WRONG"})
        elif invalid == "status":
            outcome = replace(outcome, result=value | {"status_code": "RUNNING"})
        elif invalid == "evidence":
            outcome = replace(outcome, evidence=("unrelated-evidence",))
        elif invalid == "empty_evidence":
            outcome = replace(outcome, evidence=())
        elif invalid == "ambiguous":
            outcome = replace(outcome, result=value | {"status_code": "ERROR", "error_code": "tool_cancelled", "metadata": {}})
        elif invalid == "fields":
            outcome = replace(outcome, result=value | {"unexpected": True})
        return outcome

    def query(self, handle):
        self.business.call("query")
        if self.business.setting("changed_query_handle"):
            return Accepted(handle | {"job_id": "another-job"})
        plan, done = self.business.job(handle["job_id"])
        if self.handle(plan, handle["job_id"]) != handle:
            return Unknown("business binding differs")
        if self.business.setting("query_paused"):
            return Accepted(handle)
        return self.result(handle) if done else Accepted(handle)

    def cancel(self, handle):
        self.business.call("cancel")
        mode = self.business.setting("cancel") or "unknown"
        if mode == "confirmed":
            with closing(self.business.connect()) as db, db:
                db.execute("UPDATE jobs SET done=2 WHERE key=?", (handle["job_id"],))
            return self.result(handle, cancelled=True)
        if mode == "accepted":
            return Accepted(handle)
        return Unknown("business stop cannot be confirmed")

    def callback(self, handle, *, input_id="callback"):
        plan, state = self.business.job(handle["job_id"])
        assert state in {1, 2}, "callback requires a durable business terminal fact"
        outcome = self.result(handle)
        return InboxItem(
            input_id,
            "provider_result",
            {
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "request_digest": plan.payload["request_digest"],
                "provider_binding": plan.payload["provider_binding"],
                "result": outcome.result,
                "evidence": list(outcome.evidence),
                "usage": outcome.usage,
            },
            target_turn_id=plan.turn_id,
            generation=0,
        )

    def authenticate(self, item, plan):
        if item.kind != "provider_result":
            return False
        key = plan.payload["idempotency_key"]
        expected = self.callback(self.handle(plan, key), input_id=item.input_id)
        return canonical_json_bytes(expected.to_dict()) == canonical_json_bytes(item.to_dict())


def business_tool(business, *, supported=True, needs_approval=False):
    @function_tool(
        needs_approval=needs_approval,
        tool_metadata=ToolMetadata(side_effect="external", idempotency="supported" if supported else "unsupported"),
    )
    def work(value: int = 1):
        business.call("sync")
        return str(value)

    return work


class JobsHost(DefaultAppServerHost):
    def __init__(self, business, *, config=None, agent=None, bound=True, supported=True, needs_approval=False):
        self.business, self.bound = business, bound
        self.builds = 0
        agent = agent or Agent(
            "jobs",
            "Complete the business job.",
            tools=[business_tool(business, supported=supported, needs_approval=needs_approval)],
        )

        def model(request):
            return (
                LLMResponse("done")
                if any(m.role == "tool" for m in request.messages)
                else LLMResponse("work", [ToolCall("business-call", "work", {"value": 1})])
            )

        super().__init__(
            agent=agent,
            run_config=config
            or RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "m", model),
                workspace="/fixture",
                tool_registry_factory=ToolRegistry,
            ),
        )

    def build_run_config(self, request):
        self.builds += 1
        return replace(
            super().build_run_config(request), tool_providers={"work": JobsProvider(self.business)} if self.bound else {}
        )


class Server:
    def __init__(self, store, business, **host_options):
        self.host = JobsHost(business, **host_options)
        self.transport = ChannelTransport(connection_id="owner", outbound_capacity=10000)
        self.server = AppServer(transport=self.transport, store=store, host=self.host)
        self.messages = []
        self.identity = 0
        self.connect("owner")

    def connect(self, connection_id):
        transport = (
            self.transport if connection_id == "owner" else ChannelTransport(connection_id=connection_id, outbound_capacity=10000)
        )
        if connection_id != "owner":
            self.server.router.register_transport(transport)
        self.rpc("initialize", {"clientInfo": {"name": "provider-test"}}, transport=transport)
        self.server.processor.process_message(connection_id, {"jsonrpc": "2.0", "method": "initialized"})
        return transport

    def rpc(self, method, params, *, transport=None):
        transport = transport or self.transport
        self.identity += 1
        self.server.processor.process_message(
            transport.connection_id, {"jsonrpc": "2.0", "id": self.identity, "method": method, "params": params}
        )
        while True:
            wire = transport.receive_outbound(timeout=10)
            self.messages.append(wire)
            if wire.get("id") == self.identity:
                return wire

    def start(self, *, wait=True):
        assert "result" in self.rpc("thread/start", {"inputId": THREAD})
        result = self.rpc(
            "turn/start",
            {"threadId": THREAD, "inputId": "work", "expectedTurnId": None, "input": [{"type": "text", "text": "go"}]},
        )
        assert "result" in result, result
        if wait:
            self.join()
        return result["result"]

    def resume(self, *, transport=None):
        result = self.rpc("turn/resume", {"threadId": THREAD, "turnId": TURN}, transport=transport)
        self.join()
        return result

    def due(self):
        delay = self.server.kernel.store.next_drive_delay_ms(THREAD)
        if delay:
            assert delay < 60_000
            Event().wait(delay / 1000 + 0.02)

    def join(self):
        self.server.run_adapter.join()

    def state(self):
        return self.server.kernel.store.read_state(THREAD)[0]

    def attempt(self):
        op = next(op for op in self.state().operations.values() if op.kind == "tool")
        return op.attempts[max(op.attempts)]

    def terminal(self):
        return next(
            r.record.payload
            for r in reversed(self.server.kernel.store.read_state(THREAD)[1])
            if r.record.kind == "turn_ended" and r.record.turn_id == TURN
        )

    def close(self):
        self.join()
        self.server.run_adapter.close()
        self.server.kernel.close()


def produce():
    with TemporaryDirectory(prefix="gap2-producer-") as tmp:
        root = Path(tmp)
        business = BusinessDB(root / "business.sqlite")
        path = root / "sessions.sqlite"
        with SQLiteStore.standalone(str(path)) as store:
            store.install_schema()
            first = Server(store, business)
            receipt = first.start()
            attempt = first.attempt()
            assert attempt.wait is not None
            handle = attempt.wait["handle"]
            before = {
                "phase": first.state().phase,
                "status": first.rpc("thread/read", {"threadId": THREAD})["result"]["thread"]["status"],
                "handle": handle,
                "effect_count": business.count("effect"),
            }
            assert before["status"] == "provider_running"
            first.close()
        business.finish()
        with SQLiteStore.standalone(str(path)) as store:
            second = Server(store, business)
            second.due()
            second.resume()
            attempt = second.attempt()
            assert attempt.result is not None
            completion = attempt.result.payload
            assert completion["result"]["tool_call_id"] == "business-call"
            assert business.count("effect") == 1 and business.count("submit") == 1 and business.count("sync") == 0
            value = {
                "schema_version": 1,
                "protocol_version": "v5",
                "receipt": {k: v for k, v in receipt.items() if k != "status"},
                "accepted": before,
                "recovered": {
                    "phase": second.state().phase,
                    "status": second.rpc("thread/read", {"threadId": THREAD})["result"]["thread"]["status"],
                    "result": completion["result"],
                    "evidence": completion["evidence"],
                    "usage": completion["usage"],
                    "context": completion["context"],
                    "effect_count": business.count("effect"),
                    "submit_count": business.count("submit"),
                },
            }
            second.close()
        value["strict_results"] = []
        for invalid in ("call", "status", "evidence", "empty_evidence", "ambiguous", "fields"):
            business = BusinessDB(root / f"invalid-{invalid}.sqlite")
            with SQLiteStore.standalone(":memory:") as store:
                store.install_schema()
                server = Server(store, business)
                try:
                    server.start()
                    retained = server.attempt().wait["handle"]
                    business.finish()
                    business.setting("invalid", invalid)
                    server.due()
                    server.resume()
                    server.rpc("thread/read", {"threadId": THREAD})
                    warning = next(
                        w["params"]["code"]
                        for w in server.messages
                        if w.get("method") == "error/warning" and w["params"]["code"] == "provider_result_invalid"
                    )
                    value["strict_results"].append(
                        {
                            "case": invalid,
                            "state": server.attempt().state,
                            "handle_retained": server.attempt().wait["handle"] == retained,
                            "turn_ended": server.state().turns[TURN].ended,
                            "warning": warning,
                            "effect_count": business.count("effect"),
                            "submit_count": business.count("submit"),
                        }
                    )
                finally:
                    server.close()
        value["cancellation"] = []
        for mode in ("confirmed", "accepted", "unknown", "invalid"):
            business = BusinessDB(root / f"cancel-{mode}.sqlite")
            business.setting("cancel", "confirmed" if mode == "invalid" else mode)
            if mode == "invalid":
                business.setting("invalid", "evidence")
            with SQLiteStore.standalone(":memory:") as store:
                store.install_schema()
                server = Server(store, business)
                try:
                    server.start()
                    reply = server.rpc("turn/interrupt", {"threadId": THREAD, "expectedTurnId": TURN, "actionId": "cancel"})
                    server.join()
                    terminal = server.terminal()
                    value["cancellation"].append(
                        {
                            "provider_return": mode,
                            "requested": reply["result"]["accepted"],
                            "turn_status": terminal["status"],
                            "unconfirmed_count": len(terminal["unconfirmed_operations"]),
                            "completion_context": server.attempt().context,
                            "business_state": "stopped"
                            if business.job(server.attempt().plan.payload["idempotency_key"])[1] == 2
                            else "running",
                        }
                    )
                finally:
                    server.close()
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-store")
    parser.add_argument("--business", type=Path)
    parser.add_argument("--cut", choices=("accepted", "parked"))
    parser.add_argument("--supported", type=int, default=1)
    args = parser.parse_args()
    if args.worker_store:
        from vv_agent.session.postgres import PostgresStore

        business = BusinessDB(args.business)
        business.setting("pause_after_accept", args.cut == "accepted")
        opener = PostgresStore if "dbname=" in args.worker_store else SQLiteStore
        with opener.standalone(args.worker_store) as store:
            server = Server(store, business, supported=bool(args.supported))
            server.start()
            assert server.attempt().state == "parked"
            business.setting("parked", True)
            Event().wait(60)
            raise AssertionError("worker was not killed")
    else:
        assert args.output is not None
        args.output.write_text(json.dumps(produce(), indent=2, sort_keys=True) + "\n")
