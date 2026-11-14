"""Only complete, successfully decoded vendor error bodies can settle the kernel."""

import gzip
import json
from contextlib import contextmanager
from unittest.mock import patch

import httpx2
import pytest
from httpx2._decoders import GZipDecoder

from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore

from .test_recovery_matrix import records_of, runtime, start


class ResponseBody(httpx2.SyncByteStream):
    def __init__(self, case, body):
        self.case, self.body = case, body
        self.exhausted = self.closed = False

    def __iter__(self):
        if self.case == "mid_body":
            yield gzip.compress(self.body)[:12]
            yield b"invalid compressed continuation" * 4
        elif self.case in {"flush", "gzip_complete"}:
            yield gzip.compress(self.body)
        else:
            yield self.body
        self.exhausted = True

    def close(self):
        self.closed = True


def response_for(case, status=413):
    wire = json.dumps({"error": {"message": "Synthetic rejected"}}).encode()
    if case == "success_openai":
        wire = (
            b'data: {"id":"test","object":"chat.completion.chunk","created":0,"model":"test",'
            b'"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":null}]}\n\n'
            b"data: [DONE]\n\n"
        )
    elif case == "success_anthropic":
        wire = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message",'
            b'"role":"assistant","model":"test","content":[],"stop_reason":null,"stop_sequence":null,'
            b'"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"done"}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
    body = ResponseBody(case, wire)
    headers = (
        {"content-encoding": ", ".join(["gzip"] * 6)}
        if case == "pre_raw"
        else ({"content-encoding": "gzip"} if case in {"flush", "mid_body", "gzip_complete"} else {})
    )
    if case.startswith("success_"):
        headers["content-type"] = "text/event-stream"
    response = httpx2.Response(status, headers=headers, stream=body)
    return response, body


def failed_flush(self):
    # Exercise failure after iter_raw() has reached EOF and closed the stream.
    raise httpx2.DecodingError("synthetic decoder finalization failure")


@contextmanager
def local_response_transport(case, status=413):
    responses, bodies, clients = [], [], []
    original = httpx2.Client.__init__

    def handle(request):
        response, body = response_for(case, status)
        responses.append(response)
        bodies.append(body)
        return response

    def local_client(self, **kwargs):
        kwargs.update(transport=httpx2.MockTransport(handle), trust_env=False)
        original(self, **kwargs)
        if case == "closed_unread":
            self.event_hooks["response"].insert(0, lambda response: response.close())
        clients.append(self)

    with (
        patch.object(httpx2.Client, "__init__", local_client),
        patch("openai._base_client.SyncAPIClient._sleep_for_retry", lambda *a, **k: None),
        patch("anthropic._base_client.SyncAPIClient._sleep_for_retry", lambda *a, **k: None),
        patch("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0),
    ):
        if case == "flush":
            with patch.object(GZipDecoder, "flush", failed_flush):
                yield responses, bodies, clients
        else:
            yield responses, bodies, clients


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("case", ["pre_raw", "flush", "mid_body", "closed_unread"])
def test_incomplete_vendor_413_stays_unknown(provider, case):
    with local_response_transport(case) as (responses, bodies, clients), SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        rt = runtime(store, [])
        rt.llm = VvLlmClient([EndpointTarget("a", "unused", "https://provider.invalid")], backend=provider)
        drive(store, "s", runtime=rt)
        assert len(responses) == 2
        assert len(records_of(store, "op_unknown")) == 2
        assert all(row.payload["duplicate_cost_risk"] for row in records_of(store, "op_unknown"))
        assert not records_of(store, "op_completed")
        assert not records_of(store, "context_compacted")
        terminal = records_of(store, "turn_ended")[-1].payload
        assert terminal["reason"] == "model_outcome_unknown"
        assert terminal["unconfirmed_operations"]
        assert all(not response.extensions["vv_agent_http_observation"].all_rejected for response in responses)
    assert all(client.is_closed for client in clients)
    assert all(body.closed for body in bodies)
    assert all(body.exhausted == (case == "flush") for body in bodies)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("status", [400, 429])
def test_complete_streamed_vendor_errors_still_settle(provider, status):
    with local_response_transport("gzip_complete", status) as (responses, bodies, _), SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        rt = runtime(store, [])
        rt.llm = VvLlmClient([EndpointTarget("a", "unused", "https://provider.invalid")], backend=provider)
        drive(store, "s", runtime=rt)
        assert len(responses) == (6 if status == 429 else 1)
        assert all(response.extensions["vv_agent_http_observation"].all_rejected for response in responses)
        assert not records_of(store, "op_unknown")
        terminal = records_of(store, "turn_ended")[-1].payload
        assert terminal["reason"] == "model_request_rejected"
        assert not terminal["unconfirmed_operations"]
    assert all(body.closed and body.exhausted for body in bodies)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_successful_vendor_stream_still_completes(provider):
    with (
        local_response_transport(f"success_{provider}", 200) as (responses, bodies, _),
        SQLiteStore.standalone(":memory:") as store,
    ):
        store.install_schema()
        start(store)
        rt = runtime(store, [])
        rt.llm = VvLlmClient([EndpointTarget("a", "unused", "https://provider.invalid")], backend=provider)
        drive(store, "s", runtime=rt)
        assert len(responses) == 1
        assert not records_of(store, "op_unknown")
        completed = records_of(store, "op_completed")
        assert completed and completed[0].payload["result"]["content"] == "done"
        terminal = records_of(store, "turn_ended")[-1].payload
        assert terminal["reason"] != "model_outcome_unknown"
        assert not terminal["unconfirmed_operations"]
        assert all(not response.extensions["vv_agent_http_observation"].all_rejected for response in responses)
    assert all(body.closed for body in bodies)
