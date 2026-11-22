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

from vv_agent.kernel._content import ContentRef
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.context import protected_start
from vv_agent.session.kernel import drive
from vv_agent.session.projection import (
    fold_hydrated_records,
    hydrate_record_payload,
    hydrate_records,
    hydration_for_session_store,
    project_records,
)
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.reducer import _fold_factory_for
from vv_agent.session.result import project_result

from .helpers import (
    content_ref,
    create_fixture_session,
    fixture_child_spec,
    fixture_content_reader,
    principal,
    push_fixture_input,
    wrap_fixture_store,
)
from .test_recovery_matrix import runtime

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


def run_completion_case(store, database, case, declared):
    store = wrap_fixture_store(store)
    # Unseen protection is independent: test both response policies with the same protected input.
    attributes = {"seed": {"unseen_input_protection": True}}
    if declared is not None:
        attributes["seed"]["response_rejection"] = declared
    create_fixture_session(store, "s", attributes=attributes)
    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
    creation = store.read("s").records[0]
    ContentRef.from_dict(creation.record._payload["attributes"])
    hydration = hydration_for_session_store(store, "s")
    hydrated_attributes = hydrate_record_payload(creation, hydration)["attributes"]
    assert hydrated_attributes["seed"]["unseen_input_protection"] is True
    if declared is None:
        assert "response_rejection" not in hydrated_attributes["seed"]
    else:
        assert hydrated_attributes["seed"]["response_rejection"] is declared
    content_reader = fixture_content_reader(store, "s")
    deltas = []
    with local_completion(case) as (server, calls):
        rt = runtime(database, [], content_reader=content_reader)
        rt.config = replace(rt.config, stream=deltas.append)
        rt.llm = VvLlmClient(
            [EndpointTarget(name, "synthetic-unused", f"http://127.0.0.1:{server.server_port}/{name}") for name in ("a", "b")],
            backend="anthropic" if case.startswith("anthropic") else "openai",
            randomize_endpoints=False,
        )
        with patch.object(VvLlmClient, "_should_use_stream", return_value=not case.endswith("nonstream")):
            drive(store, "s", runtime=rt)
    _, rows, _ = store.read_state("s")
    retained_bodies = tuple(row.record.encode() for row in rows)
    hydration = hydration_for_session_store(store, "s")
    hydrated_rows = hydrate_records(rows, hydration)
    logical_state = fold_hydrated_records(hydrated_rows, rows, _fold_factory_for(store), hydration)
    completed = [row for row in rows if row.record.kind == "op_completed"]
    results = [hydrate_record_payload(row, hydration).get("result") for row in completed]
    unknown_rows = [row for row in rows if row.record.kind == "op_unknown"]
    unknown_reasons = [hydrate_record_payload(row, hydration).get("reason") for row in unknown_rows]
    terminal = hydrate_record_payload(next(row for row in reversed(rows) if row.record.kind == "turn_ended"), hydration)
    projection = project_result(store, "s", "s/turn/initial")
    projected_context = projection.raw_result.messages
    from vv_agent.events import event_from_dict

    events = [
        event.to_dict()
        for event in project_records(rows, hydration=hydration)
        if event.to_dict()["type"] in {"model_call_failed", "run_failed"}
    ]
    for event in events:
        assert event_from_dict(event).to_dict() == event
    is_rejection = declared is True and (case.startswith("filter") or case.startswith("anthropic"))
    if case == "eof":
        assert len(unknown_rows) == 2 and not results
        assert terminal["reason"] == "model_outcome_unknown"
        assert all(isinstance(reason, str) and "ModelStreamInterrupted" in reason for reason in unknown_reasons)
        assert len(calls) == 2
        if logical_state._protection is not None:
            assert protected_start(rows, logical_state) == 1
    elif is_rejection:
        assert len(results) == 1 and not unknown_rows and len(calls) == 1
        assert hydrated_attributes["seed"]["response_rejection"] is True
        assert terminal["reason"] == "model_request_rejected"
        failure = next(event for event in events if event["type"] == "model_call_failed")
        assert failure["outcome"] == "definitive" and failure["metadata"]["model_rejection"]["status_code"] == 200
        assert "HTTP 200" in next(event for event in events if event["type"] == "run_failed")["error"]
        result = results[0]
        assert isinstance(result, dict)
        assert result["rejection"] == {
            "status_code": 200,
            "provider_code": "refusal" if case.startswith("anthropic") else "content_filter",
            "provider_type": "refusal" if case.startswith("anthropic") else "content_filter",
            "message": "Model provider response was rejected.",
            "retryable": False,
        }
        assert projection.error_code == "model_request_rejected"
        assert PARTIAL not in json.dumps([row.record.to_dict() for row in rows])
        assert not any(m.role == "assistant" and m.content == PARTIAL for m in projected_context)
        if logical_state._protection is not None:
            assert protected_start(rows, logical_state) == 1
        if case == "filter_stream":
            assert any(getattr(d, "delta", None) == PARTIAL for d in deltas)
        retained_rows = store.read_state("s")[1]
        assert tuple(row.record.encode() for row in retained_rows) == retained_bodies
        replayed_events = [
            event.to_dict()
            for event in project_records(retained_rows, hydration=hydration_for_session_store(store, "s"))
            if event.to_dict()["type"] in {"model_call_failed", "run_failed"}
        ]
        assert replayed_events == events
        replayed_projection = project_result(store, "s", "s/turn/initial")
        assert replayed_projection.error_code == projection.error_code
    else:
        assert len(results) == 1 and not unknown_rows and len(calls) == 1
        assert terminal["reason"] != "model_request_rejected"
        result = results[0]
        assert isinstance(result, dict)
        evidence = result["raw"]["completion_metadata"]
        assert isinstance(evidence, dict)
        if case.startswith("refusal"):
            assert result["content"] == REFUSAL and evidence["refusal"] == REFUSAL
            assert any(m.role == "assistant" and m.content == REFUSAL for m in projected_context)
        if case == "degraded":
            assert evidence["stream_terminated"] is None
            assert evidence["attributes"]["openai_done_observer"] == "unsupported_decoder"
        elif case.endswith("nonstream"):
            assert evidence["stream_terminated"] is None and evidence["stream_exhausted"] is None
        else:
            assert evidence["stream_terminated"] is True and evidence["stream_exhausted"] is True
        assert protected_start(rows, logical_state) is None
    return {
        "case": case,
        "declared": declared,
        "results": results,
        "unknown_reasons": unknown_reasons,
        "terminal_reason": terminal["reason"],
        "failure_events": events,
        "protected_start": protected_start(rows, logical_state),
        "context": [m.to_dict() for m in projected_context],
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

    store = wrap_fixture_store(store)
    attributes = {"seed": {"messages": [], "shared_state": {}}}
    if value is not None:
        attributes["seed"][policy] = value
    receipt = create_fixture_session(store, "policy", attributes=attributes)
    assert receipt.replayed is False
    created = store.read("policy").records[0]
    ContentRef.from_dict(created.record._payload["attributes"])
    hydrated = hydrate_record_payload(created, hydration_for_session_store(store, "policy"))["attributes"]["seed"]
    if value is None:
        assert policy not in hydrated
    else:
        assert hydrated[policy] is value
    before = created.record.encode()
    with store.atomic() as tx:
        creation = created.record._payload
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="policy")
        replayed = writer.create_session(
            lambda _content: SessionSpec(
                "policy",
                creation["principal"],
                creation["workspace"],
                attributes=creation["attributes"],
            ),
            consumers=("events",),
        )
        assert replayed.replayed is True
    assert store.read("policy").records[0].record.encode() == before
    with pytest.raises(Conflict), store.atomic() as tx:
        changed_attributes = {"seed": {"messages": [], "shared_state": {}, policy: not bool(value)}}
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="policy")
        changed_ref = writer.put_json(changed_attributes)
        creation = created.record._payload
        writer.create_session(
            lambda _content: SessionSpec(
                "policy",
                creation["principal"],
                creation["workspace"],
                attributes=changed_ref.to_dict(),
            ),
            consumers=("events",),
        )


@pytest.mark.parametrize("value", ["true", 1, None])
def test_response_policy_is_boolean(value):
    from vv_agent.session.records import RecordError

    with pytest.raises(RecordError):
        SessionSpec(
            "invalid",
            principal(),
            content_ref("workspace"),
            attributes={"seed": {"response_rejection": value}},
        ).record()


def test_children_from_undeclared_source_freeze_false(store, database):
    from vv_agent.session.children import ChildSession

    from .test_children import parent_runtime

    store = wrap_fixture_store(store)
    create_fixture_session(store, "s", attributes={})
    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
    rt = parent_runtime(store, database)
    rt.content_reader = fixture_content_reader(store, "s")
    rt.children["child_task"] = lambda plan: ChildSession(fixture_child_spec(store, plan.session_id, "child"), "work")
    drive(store, "s", runtime=rt)
    creation = store.read("child").records[0]
    ContentRef.from_dict(creation.record._payload["attributes"])
    seed = hydrate_record_payload(creation, hydration_for_session_store(store, "child"))["attributes"]["seed"]
    assert seed == {
        "unseen_input_protection": False,
        "response_rejection": False,
    }


def test_fork_from_undeclared_source_freezes_false(store, tmp_path):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.app_server import DefaultAppServerHost
    from vv_agent.types import LLMResponse

    from .test_app_server_fork import run_turn, send, server_for

    host = DefaultAppServerHost(
        agent=Agent("test", "test", model="test"),
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "test", [LLMResponse("done")])),
    )
    server = server_for(store, host, tmp_path / "fork-keys.sqlite3")
    store = server.kernel.store

    try:
        create_fixture_session(
            store,
            "source",
            attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}},
            consumers=("events", "app_server", "traces"),
        )
        turn = run_turn(server, "source", "go")
        result = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy", "throughTurnId": turn})
        creation = store.read(result["threadId"]).records[0]
        ContentRef.from_dict(creation.record._payload["attributes"])
        seed = hydrate_record_payload(creation, hydration_for_session_store(store, result["threadId"]))["attributes"]["seed"]
        assert seed["response_rejection"] is False and seed["unseen_input_protection"] is False
    finally:
        server.kernel.close()


@pytest.mark.parametrize("declared", [False, True])
def test_response_rejection_in_output_repair_preserves_repair_failure(store, database, declared):
    from vv_agent.llm.base import LlmRequest
    from vv_agent.output_validation import OutputValidationResult
    from vv_agent.types import LLMResponse, Message

    store = wrap_fixture_store(store)
    create_fixture_session(store, "s", attributes={"seed": {"response_rejection": declared}})
    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
    creation = store.read("s").records[0]
    ContentRef.from_dict(creation.record._payload["attributes"])
    hydrated_seed = hydrate_record_payload(creation, hydration_for_session_store(store, "s"))["attributes"]["seed"]
    assert hydrated_seed["response_rejection"] is declared
    rt = runtime(database, [LLMResponse("invalid")], content_reader=fixture_content_reader(store, "s"))
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
    _, rows, _ = store.read_state("s")
    hydration = hydration_for_session_store(store, "s")
    result = hydrate_record_payload(next(row for row in reversed(rows) if row.record.kind == "op_completed"), hydration)["result"]
    if declared:
        assert result["error_code"] == "repair_provider_error"
        assert result["content"] == "HTTP 200: Model provider response was rejected."
        assert PARTIAL not in json.dumps(result)
        terminal = hydrate_record_payload(next(row for row in reversed(rows) if row.record.kind == "turn_ended"), hydration)
        assert terminal["reason"] == "output_validation_failed"
    else:
        assert result["content"] == PARTIAL
