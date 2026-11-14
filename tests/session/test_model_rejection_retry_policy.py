"""Recovered provider imports and transparent, bounded HTTP rejection observation."""

from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from threading import Thread
from time import sleep
from unittest.mock import patch

import httpx2
import pytest

from vv_agent.llm import errors
from vv_agent.llm.base import LlmRequest
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import Message

from .test_model_rejection_native import assert_unknown, native_error
from .test_model_response_completion import ResponseBody, response_for
from .test_recovery_matrix import records_of, runtime, start


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("name", ["APIConnectionError", "APITimeoutError"])
@pytest.mark.parametrize("status", [400, 413])
def test_recovered_import_without_cache_clear_still_vetoes(monkeypatch, provider, name, status):
    def load(module):
        if module == provider:
            raise RuntimeError("synthetic first import failure")
        return import_module(module)

    with monkeypatch.context() as first:
        first.setattr(errors, "import_module", load)
        assert errors.model_request_rejection(errors.ModelRequestRejected(400, "Rejected"))
    error = errors.ModelRequestRejected(status, "Rejected")
    error.__cause__ = native_error(provider, name)
    assert_unknown(error)


class RetryBody(ResponseBody):
    def __iter__(self):
        if self.case == "timeout":
            raise httpx2.ReadTimeout("synthetic body timeout")
        if self.case == "transport":
            raise httpx2.RemoteProtocolError("synthetic broken body")
        yield from super().__iter__()


@contextmanager
def retry_transport(provider, case="complete", status=429, headers=None, observed=True, wire_url=None):
    """Use real vv-llm/provider SDKs, substituting only HTTP and sleeping."""
    native = import_module(provider)
    responses, bodies, sleeps, clients, exceptions = [], [], [], [], []
    original = httpx2.Client.__init__

    def handle(request):
        if status == 200:
            response, body = response_for(f"success_{provider}", 200)
        else:
            body = RetryBody(case, b'{"error":{"message":"Synthetic rejected"}}' + (b" " * (1048577) if case == "large" else b""))
            response = httpx2.Response(status, headers=headers, stream=body)
        responses.append(response)
        bodies.append(body)
        return response

    class WireTransport(httpx2.HTTPTransport):
        def handle_request(self, request):
            request.url = httpx2.URL(wire_url + request.url.path)
            response = super().handle_request(request)
            responses.append(response)
            return response

    def init(self, **kwargs):
        kwargs.update(transport=WireTransport() if wire_url else httpx2.MockTransport(handle), trust_env=False)
        original(self, **kwargs)
        clients.append(self)

    def sleep(self, **kwargs):
        response = kwargs["response"]
        delay = self._calculate_retry_timeout(
            kwargs["max_retries"] - kwargs["retries_taken"], kwargs["options"], response.headers if response is not None else None
        )
        sleeps.append((response.status_code if response is not None else None, delay))

    def native_client(_observation, **kwargs):
        return native.DefaultHttpxClient(**kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(httpx2.Client, "__init__", init))
        stack.enter_context(patch.object(native._base_client.SyncAPIClient, "_sleep_for_retry", sleep))
        stack.enter_context(patch.object(native._base_client, "random", lambda: 0.5))
        stack.enter_context(patch("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0))
        stack.enter_context(patch.object(VvLlmClient, "_sleep_backoff", lambda *args: None))
        if not observed:
            symbol = "ObservedHttpClient" if provider == "openai" else "ObservedAnthropicHttpClient"
            stack.enter_context(patch("vv_agent.llm.vv_llm_client." + symbol, native_client))
        yield responses, bodies, sleeps, clients, exceptions


def invoke(provider, evidence, timeout=90.0):
    client = VvLlmClient(
        [EndpointTarget("a", "synthetic-unused", "https://provider.invalid")],
        backend=provider,
        max_retries_per_endpoint=1,
        timeout_seconds=timeout,
    )
    try:
        client.complete(LlmRequest(model="test", messages=[Message(role="user", content="Synthetic")], tools=[]))
    except Exception as error:
        evidence[-1].extend(type(item).__name__ for item in errors._exception_graph(error))
        return error
    return None


RETRY_CASES = [
    ("complete", 429, {"retry-after": "7"}),
    ("timeout", 429, {"retry-after": "7"}),
    ("timeout", 429, {"retry-after": "7", "x-should-retry": "false"}),
    ("complete", 429, {"x-should-retry": "false"}),
    ("complete", 409, {}),
    ("complete", 400, {"x-should-retry": "true"}),
    ("complete", 400, {}),
    ("complete", 413, {}),
    ("complete", 200, {}),
    ("large", 400, {}),
    ("transport", 429, {"retry-after": "7"}),
]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("case,status,headers", RETRY_CASES)
def test_native_retry_policy_is_unchanged(provider, case, status, headers):
    results = []
    for observed in (False, True):
        with retry_transport(provider, case, status, headers, observed) as evidence:
            error = invoke(provider, evidence)
            responses, bodies, sleeps, clients, exceptions = evidence
            results.append((len(responses), sleeps, type(error), sorted(exceptions)))
            assert all(client.is_closed for client in clients)
            if observed:
                observations = [r.extensions["vv_agent_http_observation"] for r in responses]
                assert all(o.all_rejected for o in observations) is (case == "complete" and status != 200)
                if case == "large":
                    assert len(responses[-1].content) > 1048576
                    assert responses[-1].content == bodies[-1].body
                if case in {"timeout", "transport", "large"}:
                    assert error is not None and errors.model_request_rejection(error) is None
    assert results[0] == results[1]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("case,status,headers", RETRY_CASES)
def test_retry_observation_real_kernel(provider, case, status, headers):
    with retry_transport(provider, case, status, headers) as evidence, SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        rt = runtime(store, [])
        rt.llm = VvLlmClient([EndpointTarget("a", "synthetic-unused", "https://provider.invalid")], backend=provider)
        drive(store, "s", runtime=rt)
        terminal = records_of(store, "turn_ended")[-1].payload
        if case in {"timeout", "transport", "large"}:
            assert len(records_of(store, "op_unknown")) == 2
            assert not records_of(store, "op_completed")
            assert terminal["reason"] == "model_outcome_unknown"
            assert terminal["unconfirmed_operations"]
        else:
            assert not records_of(store, "op_unknown")
            assert not terminal["unconfirmed_operations"]
            if status in {400, 409, 429}:
                assert terminal["reason"] == "model_request_rejected"
            if status == 409:
                assert len(evidence[0]) == 3
            if status == 413:
                assert all(r.payload["result"]["error_code"] == "prompt_too_long" for r in records_of(store, "op_completed"))


@contextmanager
def wire_server(case, status, headers):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", "0")))
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                if case == "timeout":
                    sleep(0.2)
                    self.close_connection = True
                    return
                payload = b'{"error":{"message":"Synthetic rejected"}}'
                if case == "trickle":
                    chunks = [payload[:1]] * 16
                elif case == "large":
                    chunks = [payload, b" " * 1048577]
                else:
                    chunks = [payload]
                for chunk in chunks:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    if case == "trickle":
                        sleep(0.015)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


WIRE_CASES = [
    ("complete", 429, {"retry-after": "7"}),
    ("timeout", 429, {"retry-after": "7"}),
    ("timeout", 429, {"retry-after": "7", "x-should-retry": "false"}),
    ("complete", 409, {}),
    ("trickle", 400, {}),
    ("large", 400, {}),
]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("case,status,headers", WIRE_CASES)
def test_loopback_native_policy_and_real_kernel(provider, case, status, headers):
    results = []
    with wire_server(case, status, headers) as url:
        for observed in (False, True):
            with retry_transport(provider, observed=observed, wire_url=url) as evidence:
                error = invoke(provider, evidence, timeout=0.06)
                responses, _, sleeps, _, exceptions = evidence
                results.append((len(responses), sleeps, type(error), sorted(exceptions)))
                if observed:
                    assert error is not None
                    assert (errors.model_request_rejection(error) is not None) is (case == "complete")
        assert results[0] == results[1]
        with retry_transport(provider, wire_url=url), SQLiteStore.standalone(":memory:") as store:
            store.install_schema()
            start(store)
            rt = runtime(store, [])
            rt.llm = VvLlmClient(
                [EndpointTarget("a", "synthetic-unused", "https://provider.invalid")], backend=provider, timeout_seconds=0.06
            )
            drive(store, "s", runtime=rt)
            terminal = records_of(store, "turn_ended")[-1].payload
            if case == "complete":
                assert not records_of(store, "op_unknown")
                assert terminal["reason"] == "model_request_rejected"
            else:
                unknowns = records_of(store, "op_unknown")
                assert len(unknowns) == 2 and all(row.payload["duplicate_cost_risk"] for row in unknowns)
                assert not records_of(store, "op_completed")
                assert terminal["reason"] == "model_outcome_unknown"
                assert terminal["unconfirmed_operations"]
