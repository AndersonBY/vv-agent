"""Local HTTP through vv-llm and the production adapter/kernel completion boundary."""

import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from unittest.mock import patch

import pytest

from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.context import project_context, protected_start
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec, make_record

from .test_recovery_matrix import records_of, runtime

CASES = (
    "filter_stream",
    "filter_nonstream",
    "refusal_stream",
    "refusal_nonstream",
    "anthropic_stream",
    "anthropic_nonstream",
    "eof",
    "done_only",
    "finish_only",
    "normal",
    "degraded",
    "length",
    "other",
    "missing_nonstream",
    "filter_with_refusal",
)
PARTIAL = "temporary filtered body"
REFUSAL = "I cannot help with that request."


class DecoderV2:
    """Keep native SSE parsing but trigger vv-llm's unsupported-decoder diagnostic."""

    def __init__(self, decoder):
        self.decoder = decoder

    def iter_bytes(self, source):
        yield from self.decoder.iter_bytes(source)


def response_bytes(case, streaming):
    anthropic = case.startswith("anthropic")
    if anthropic:
        message = {
            "id": "msg_local",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        if not streaming:
            return json.dumps(message).encode()
        events = [
            ("message_start", {"type": "message_start", "message": message | {"stop_reason": None}}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "refusal", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        return "".join(f"event: {name}\ndata: {json.dumps(value)}\n\n" for name, value in events).encode()
    reason = (
        "content_filter"
        if case.startswith("filter")
        else "length"
        if case == "length"
        else "custom_end"
        if case == "other"
        else "stop"
    )
    refusal = REFUSAL if case.startswith("refusal") or case == "filter_with_refusal" else None
    content = None if refusal else PARTIAL if case.startswith("filter") else "ordinary answer"
    base: dict[str, Any] = {"id": "local", "object": "chat.completion", "created": 0, "model": "test"}
    if not streaming:
        return json.dumps(
            base
            | {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, "refusal": refusal},
                        "finish_reason": None if case == "missing_nonstream" else reason,
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        ).encode()
    base["object"] = "chat.completion.chunk"
    chunks = [base | {"choices": [{"index": 0, "delta": {"content": content, "refusal": refusal}, "finish_reason": None}]}]
    if case not in {"eof", "done_only", "degraded"}:
        chunks.append(base | {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if case not in {"eof", "finish_only"}:
        body += "data: [DONE]\n\n"
    return body.encode()


@contextmanager
def local_completion(case):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "request": request})
            streaming = request.get("stream", False)
            body = response_bytes(case, streaming)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    original = __import__("openai").OpenAI._make_sse_decoder

    def decoder(client):
        return DecoderV2(original(client))

    try:
        with patch("openai.OpenAI._make_sse_decoder", decoder) if case == "degraded" else nullcontext():
            yield server, calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def run_completion_case(store, database, case, declared, *, create=True):
    if create:
        # Unseen protection is independent: test both response policies with the same protected input.
        attributes = {"seed": {"unseen_input_protection": True}}
        if declared is not None:
            attributes["seed"]["response_rejection"] = declared
        historical = make_record(
            "session_created",
            session_id="s",
            payload={
                "principal": "test",
                "workspace": "/tmp",
                "parent_session_id": None,
                "parent_operation_id": None,
                "attributes": attributes,
            },
        )
        with patch.object(SessionSpec, "record", return_value=historical), store.atomic() as tx:
            tx.create(SessionSpec("s", "test", "/tmp"), consumers=("events",))
            tx.push("s", InboxItem("initial", "user", {"content": "go"}))
    deltas = []
    with local_completion(case) as (server, calls):
        rt = runtime(database, [])
        rt.config = replace(rt.config, stream=deltas.append)
        rt.llm = VvLlmClient(
            [EndpointTarget(name, "synthetic-unused", f"http://127.0.0.1:{server.server_port}/{name}") for name in ("a", "b")],
            backend="anthropic" if case.startswith("anthropic") else "openai",
            randomize_endpoints=False,
        )
        with patch.object(VvLlmClient, "_should_use_stream", return_value=not case.endswith("nonstream")):
            drive(store, "s", runtime=rt)
    state, rows, _ = store.read_state("s")
    results = [r.payload["result"] for r in records_of(store, "op_completed")]
    unknowns = records_of(store, "op_unknown")
    terminal = records_of(store, "turn_ended")[-1].payload
    from vv_agent.events import event_from_dict
    from vv_agent.session.projection import project_records

    events = [
        event.to_dict() for event in project_records(rows) if event.to_dict()["type"] in {"model_call_failed", "run_failed"}
    ]
    for event in events:
        assert event_from_dict(event).to_dict() == event
    is_rejection = declared is True and (case.startswith("filter") or case.startswith("anthropic"))
    if case == "eof":
        assert len(unknowns) == 2 and not results
        assert terminal["reason"] == "model_outcome_unknown"
        assert all("ModelStreamInterrupted" in r.payload["reason"] for r in unknowns)
        assert len(calls) == 2
        if state._protection is not None:
            assert protected_start(rows, state) == 1
    elif is_rejection:
        assert len(results) == 1 and not unknowns and len(calls) == 1
        assert terminal["reason"] == "model_request_rejected"
        failure = next(event for event in events if event["type"] == "model_call_failed")
        assert failure["outcome"] == "definitive" and failure["metadata"]["model_rejection"]["status_code"] == 200
        assert "HTTP 200" in next(event for event in events if event["type"] == "run_failed")["error"]
        result = results[0]
        assert result["rejection"] == {
            "status_code": 200,
            "provider_code": "refusal" if case.startswith("anthropic") else "content_filter",
            "provider_type": "refusal" if case.startswith("anthropic") else "content_filter",
            "message": "Model provider response was rejected.",
            "retryable": False,
        }
        assert PARTIAL not in json.dumps([row.record.to_dict() for row in rows])
        assert not any(m.role == "assistant" and m.content == PARTIAL for m in project_context(rows, state))
        if state._protection is not None:
            assert protected_start(rows, state) == 1
        if case == "filter_stream":
            assert any(getattr(d, "delta", None) == PARTIAL for d in deltas)
    else:
        assert len(results) == 1 and not unknowns and len(calls) == 1
        assert terminal["reason"] != "model_request_rejected"
        result = results[0]
        evidence = result["raw"]["completion_metadata"]
        if case.startswith("refusal"):
            assert result["content"] == REFUSAL and evidence["refusal"] == REFUSAL
            assert any(m.role == "assistant" and m.content == REFUSAL for m in project_context(rows, state))
        if case == "degraded":
            assert evidence["stream_terminated"] is None
            assert evidence["attributes"]["openai_done_observer"] == "unsupported_decoder"
        elif case.endswith("nonstream"):
            assert evidence["stream_terminated"] is None and evidence["stream_exhausted"] is None
        else:
            assert evidence["stream_terminated"] is True and evidence["stream_exhausted"] is True
        assert protected_start(rows, state) is None
    return {
        "case": case,
        "declared": declared,
        "results": results,
        "unknown_reasons": [r.payload["reason"] for r in unknowns],
        "terminal_reason": terminal["reason"],
        "failure_events": events,
        "protected_start": protected_start(rows, state),
        "context": [m.to_dict() for m in project_context(rows, state)],
        "http_calls": calls,
    }


@pytest.mark.parametrize("declared", [None, False, True])
@pytest.mark.parametrize("case", CASES)
def test_real_completion_mapping(store, database, case, declared):
    run_completion_case(store, database, case, declared)


def test_completion_fixture():
    import runpy

    from support.contract import fixture_root

    generate = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/model_completion_fixtures.py"))["generate"]
    assert generate() == json.loads((fixture_root() / "model_completion_evidence.json").read_text())


@pytest.mark.parametrize("policy", ["unseen_input_protection", "response_rejection"])
@pytest.mark.parametrize("value", [None, False, True])
def test_seed_policy_creation_replay(store, policy, value):
    from vv_agent.session.store import Conflict

    attributes = {"seed": {"messages": [], "shared_state": {}}}
    if value is not None:
        attributes["seed"][policy] = value
    creation = make_record(
        "session_created",
        session_id="policy",
        payload={
            "principal": "test",
            "workspace": "/tmp",
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": attributes,
        },
    )
    with patch.object(SessionSpec, "record", return_value=creation), store.atomic() as tx:
        tx.create(SessionSpec("policy", "test", "/tmp"), consumers=("events",))
    before = store.read("policy").records[0].record.encode()
    with store.atomic() as tx:
        assert tx.create(
            SessionSpec("policy", "test", "/tmp", attributes={"seed": {"messages": [], "shared_state": {}}}),
            consumers=("events",),
        ).replayed
    assert store.read("policy").records[0].record.encode() == before
    with pytest.raises(Conflict), store.atomic() as tx:
        tx.create(
            SessionSpec(
                "policy", "test", "/tmp", attributes={"seed": {"messages": [], "shared_state": {}, policy: not bool(value)}}
            ),
            consumers=("events",),
        )


@pytest.mark.parametrize("value", ["true", 1, None])
def test_response_policy_is_boolean(value):
    from vv_agent.session.records import RecordError

    with pytest.raises(RecordError):
        SessionSpec("invalid", "test", "/tmp", attributes={"seed": {"response_rejection": value}}).record()


@pytest.mark.persistent_store
@pytest.mark.parametrize("case", ["eof", "refusal_stream", "filter_stream"])
def test_0333_rollback_real_store(database, case):
    import os
    import subprocess

    from .conftest import open_store

    python = os.environ.get("VV_AGENT_0333_PYTHON")
    if not python:
        pytest.skip("Set VV_AGENT_0333_PYTHON to an isolated PyPI 0.33.3 interpreter")
    script = r"""
import sys
from pathlib import Path
from importlib.metadata import version
from vv_agent import Agent, RunConfig
from vv_agent.types import LLMResponse
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.kernel import Runtime, drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.postgres import PostgresStore
assert version('vv-agent') == '0.33.3'
kind, database, phase = sys.argv[1:]
factory = SQLiteStore if kind == 'sqlite' else PostgresStore
with factory.standalone(database) as store:
    if phase == 'create':
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec('s', 'test', '/tmp'), consumers=('events',))
            tx.push('s', InboxItem('initial', 'user', {'content': 'go'}))
    else:
        state, rows, _ = store.read_state('s')
        assert state.active_turn_id is None
        before = [r.record.encode() for r in rows]
        with store.atomic() as tx:
            tx.push('s', InboxItem('continue', 'user', {'content': 'continue'}))
        rt = Runtime(
            agent=Agent(name='kernel', instructions='Be precise.'),
            config=RunConfig(workspace=Path('/tmp'), session_memory_enabled=False),
            resolved=ResolvedModelConfig('scripted','scripted','scripted','scripted',[]),
            llm=ScriptedLLM([LLMResponse('old runtime continued')]),
            heartbeat_store=lambda: factory.standalone(database),
        )
        drive(store, 's', runtime=rt)
        state, rows, _ = store.read_state('s')
        assert [r.record.encode() for r in rows[:len(before)]] == before
        assert rows[-1].record.payload['status'] == 'completed'
        assert rows[-1].record.payload['result'] == 'old runtime continued'
print('0.33.3 ' + kind + ' ' + phase + ': PASS')
"""
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VV_AGENT_CONTRACT_SOURCE"}}
    kind = "sqlite" if isinstance(database, Path) else "postgres"

    def old(phase):
        result = subprocess.run(
            [python, "-c", script, kind, str(database), phase], cwd="/tmp", env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout.strip())

    old("create")
    with open_store(database) as store:
        before = store.read("s").records[0].record.encode()
        run_completion_case(store, database, case, None, create=False)
        assert store.read("s").records[0].record.encode() == before
    old("continue")
    print(f"ROLLBACK {kind} {case}: 0.33.3 create -> 0.34.0 write -> 0.33.3 read/continue PASS")


@pytest.mark.parametrize("enabled", [False, True])
def test_0333_rejects_response_declaration_alone(enabled):
    import os
    import subprocess

    python = os.environ.get("VV_AGENT_0333_PYTHON")
    if not python:
        pytest.skip("Set VV_AGENT_0333_PYTHON to isolated 0.33.3")
    record = make_record(
        "session_created",
        session_id="new",
        payload={
            "principal": "test",
            "workspace": "/tmp",
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": {"seed": {"response_rejection": enabled}},
        },
    )
    script = """
import sys
from importlib.metadata import version
from vv_agent.session.records import Record, RecordError
assert version('vv-agent') == '0.33.3'
try:
    Record.parse(sys.stdin.buffer.read())
except RecordError as error:
    assert 'response_rejection' in str(error)
    print('0.33.3 rejects response_rejection declaration: PASS')
else:
    raise AssertionError('old runtime accepted new policy')
"""
    result = subprocess.run(
        [python, "-c", script],
        input=record.encode(),
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VV_AGENT_CONTRACT_SOURCE"}},
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.decode().strip())


def test_children_from_undeclared_source_freeze_false(store, database):
    from vv_agent.session.children import ChildSession

    from .test_children import parent_runtime

    creation = make_record(
        "session_created",
        session_id="s",
        payload={
            "principal": "test",
            "workspace": "/tmp",
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": {},
        },
    )
    with patch.object(SessionSpec, "record", return_value=creation), store.atomic() as tx:
        tx.create(SessionSpec("s", "test", "/tmp"), consumers=("events",))
        tx.push("s", InboxItem("initial", "user", {"content": "go"}))
    rt = parent_runtime(database)
    rt.children["child_task"] = lambda _: ChildSession(SessionSpec("child", "test", "/tmp"), "work")
    drive(store, "s", runtime=rt)
    assert store.read("child").records[0].record.payload["attributes"]["seed"] == {
        "unseen_input_protection": False,
        "response_rejection": False,
    }


def test_fork_from_undeclared_source_freezes_false(store):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.app_server import DefaultAppServerHost
    from vv_agent.types import LLMResponse

    from .test_app_server_fork import run_turn, send, server_for

    host = DefaultAppServerHost(
        agent=Agent("test", "test", model="test"),
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "test", [LLMResponse("done")])),
    )
    server = server_for(store, host)
    original = SessionSpec.record

    def undeclared(spec, **kwargs):
        record = original(spec, **kwargs)
        payload = record.payload
        payload["attributes"].pop("seed", None)
        return make_record("session_created", session_id=spec.session_id, payload=payload)

    try:
        with patch.object(SessionSpec, "record", undeclared):
            server.kernel.create("source", "/tmp", {"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}})
        turn = run_turn(server, "source", "go")
        result = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy", "throughTurnId": turn})
        seed = store.read(result["threadId"]).records[0].record.payload["attributes"]["seed"]
        assert seed["response_rejection"] is False and seed["unseen_input_protection"] is False
    finally:
        server.kernel.close()


@pytest.mark.parametrize("declared", [False, True])
def test_response_rejection_in_output_repair_preserves_repair_failure(store, database, declared):
    from vv_agent.llm.base import LlmRequest
    from vv_agent.output_validation import OutputValidationResult
    from vv_agent.types import LLMResponse, Message

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "test", "/tmp", attributes={"seed": {"response_rejection": declared}}), consumers=("events",))
        tx.push("s", InboxItem("initial", "user", {"content": "go"}))
    rt = runtime(database, [LLMResponse("invalid")])
    rt.agent.output_validation_enabled = True
    rt.agent.output_validator = lambda value, _: (
        OutputValidationResult.accept() if value == PARTIAL else OutputValidationResult.reject("invalid")
    )
    with local_completion("filter_stream") as (server, _):
        client = VvLlmClient(
            [EndpointTarget("local", "synthetic-unused", f"http://127.0.0.1:{server.server_port}/v1")], randomize_endpoints=False
        )
        rt.agent.output_repair = lambda _: client.complete(
            LlmRequest(model="test", messages=[Message("user", "repair")], tools=[])
        )
        drive(store, "s", runtime=rt)
    result = records_of(store, "op_completed")[-1].payload["result"]
    if declared:
        assert result["error_code"] == "repair_provider_error"
        assert PARTIAL not in json.dumps(result)
        assert records_of(store, "turn_ended")[-1].payload["reason"] == "output_validation_failed"
    else:
        assert result["content"] == PARTIAL


@pytest.mark.persistent_store
@pytest.mark.parametrize("enabled", [False, True])
def test_0333_store_read_rejects_response_policy(store, database, enabled):
    import os
    import subprocess

    python = os.environ.get("VV_AGENT_0333_PYTHON")
    if not python:
        pytest.skip("Set VV_AGENT_0333_PYTHON to isolated 0.33.3")
    creation = make_record(
        "session_created",
        session_id="declared",
        payload={
            "principal": "test",
            "workspace": "/tmp",
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": {"seed": {"response_rejection": enabled}},
        },
    )
    with patch.object(SessionSpec, "record", return_value=creation), store.atomic() as tx:
        tx.create(SessionSpec("declared", "test", "/tmp"), consumers=("events",))
    script = """
import sys
from importlib.metadata import version
from vv_agent.session.records import RecordError
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.postgres import PostgresStore
assert version('vv-agent') == '0.33.3'
kind, database = sys.argv[1:]
factory = SQLiteStore if kind == 'sqlite' else PostgresStore
with factory.standalone(database) as store:
    try:
        store.read_state('declared')
    except RecordError as error:
        assert 'response_rejection' in str(error)
    else:
        raise AssertionError('0.33.3 read a response-policy session')
print('0.33.3 ' + kind + ' store rejects response_rejection: PASS')
"""
    kind = "sqlite" if isinstance(database, Path) else "postgres"
    result = subprocess.run(
        [python, "-c", script, kind, str(database)],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VV_AGENT_CONTRACT_SOURCE"}},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
