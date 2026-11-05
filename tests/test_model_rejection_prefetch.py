"""The observer stops waiting at its wall-clock budget without losing body bytes."""

import gzip
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest

from vv_agent.llm import http_observation
from vv_agent.llm.http_observation import HttpObservation, ObservedAnthropicHttpClient, ObservedHttpClient


@pytest.fixture
def trickle_url():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.send_response(400)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Retry-After", "7")
            self.end_headers()
            try:
                for _ in range(16):
                    self.wfile.write(b"1\r\nx\r\n")
                    self.wfile.flush()
                    time.sleep(0.015)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
@pytest.mark.parametrize("read_timeout", [0.06, None])
def test_trickled_body_has_total_wall_deadline_and_is_replayed(client_type, trickle_url, read_timeout, monkeypatch):
    if read_timeout is None:
        assert http_observation._PREFETCH_SECONDS == 10.0
        monkeypatch.setattr(http_observation, "_PREFETCH_SECONDS", 0.06)
    observation = HttpObservation()
    with client_type(observation, trust_env=False, timeout=read_timeout) as client:
        began = time.monotonic()
        response = client.send(client.build_request("GET", trickle_url), stream=True)
        elapsed = time.monotonic() - began
        assert elapsed < 0.19
        assert not observation.all_rejected
        assert observation.outcomes == ["incomplete"]
        assert response.status_code == 400
        assert response.headers["retry-after"] == "7"
        assert response.read() == b"x" * 16
        assert not observation.all_rejected


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
@pytest.mark.parametrize("mode", ["timeout", "decoding", "large", "complete"])
def test_prefetch_preserves_headers_and_saved_error_or_raw_bytes(client_type, mode):
    original_error = (
        httpx2.ReadTimeout("original timeout") if mode == "timeout" else httpx2.DecodingError("original decoding failure")
    )

    class Body(httpx2.SyncByteStream):
        def __iter__(self):
            yield b"x" * (1048577 if mode == "large" else 1)
            if mode == "timeout":
                raise original_error
            if mode == "decoding":
                raise original_error

    observation = HttpObservation()
    with client_type(
        observation,
        transport=httpx2.MockTransport(
            lambda req: httpx2.Response(409, headers={"retry-after": "7", "x-should-retry": "false"}, stream=Body())
        ),
        trust_env=False,
    ) as client:
        response = client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
        assert response.status_code == 409
        assert response.headers["retry-after"] == "7"
        assert response.headers["x-should-retry"] == "false"
        assert observation.all_rejected is (mode == "complete")
        if mode in {"timeout", "decoding"}:
            error_type = httpx2.ReadTimeout if mode == "timeout" else httpx2.DecodingError
            with pytest.raises(error_type, match="original") as caught:
                response.read()
            assert caught.value is original_error
            assert caught.value.request is response.request
        else:
            assert response.read() == b"x" * (1048577 if mode == "large" else 1)
        assert observation.all_rejected is (mode == "complete")


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
def test_abandoned_prefetch_closes_original_stream(client_type):
    class Body(httpx2.SyncByteStream):
        closed = False

        def __iter__(self):
            raise httpx2.ReadTimeout("synthetic native recv timeout")
            yield b"body"

        def close(self):
            self.closed = True

    body = Body()
    observation = HttpObservation()
    with client_type(
        observation,
        transport=httpx2.MockTransport(lambda req: httpx2.Response(429, stream=body)),
        trust_env=False,
        timeout=0.03,
    ) as client:
        response = client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
        assert not body.closed
        assert not observation.all_rejected
        response.close()
        assert body.closed
        assert not observation.all_rejected


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
@pytest.mark.parametrize("read", [None, 0.03, 90])
@pytest.mark.parametrize("status", [200, 307, 400, 408, 429, 500])
def test_read_timeout_copy_is_installed_before_raw_iteration(client_type, read, status):
    original = dict(connect=3, read=read, write=4, pool=5)
    seen = []

    def handle(request):
        class Body(httpx2.SyncByteStream):
            def __iter__(self):
                seen.append(request.extensions["timeout"])
                yield b"{}"

        return httpx2.Response(status, stream=Body())

    observation = HttpObservation()
    with client_type(observation, transport=httpx2.MockTransport(handle), trust_env=False) as client:
        response = client.send(
            client.build_request("GET", "https://provider.invalid", extensions={"timeout": original}), stream=True
        )
        eligible = 400 <= status < 500 and status != 408
        assert len(seen) == int(eligible)
        assert response.read() == b"{}"
        expected = min(read, 10) if read is not None else 10
        assert seen == [{**original, "read": expected} if eligible else original]
        assert (seen[0] is original) is not eligible
        assert original == dict(connect=3, read=read, write=4, pool=5)


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
def test_eof_after_deadline_is_not_complete_evidence(client_type, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(http_observation, "monotonic", lambda: clock[0])

    class Body(httpx2.SyncByteStream):
        def __iter__(self):
            yield b"{}"
            clock[0] = 11.0

    observation = HttpObservation()
    with client_type(
        observation,
        transport=httpx2.MockTransport(lambda request: httpx2.Response(400, stream=Body())),
        trust_env=False,
    ) as client:
        response = client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
        assert observation.outcomes == ["incomplete"]
        assert response.read() == b"{}"
        assert not observation.all_rejected


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
def test_decoded_size_bound_replays_original_compressed_stream(client_type):
    payload = b"x" * 1048577
    wire = gzip.compress(payload)
    assert len(wire) < 1048576
    observation = HttpObservation()
    with client_type(
        observation,
        transport=httpx2.MockTransport(
            lambda req: httpx2.Response(400, headers={"content-encoding": "gzip"}, stream=httpx2.ByteStream(wire))
        ),
        trust_env=False,
    ) as client:
        response = client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
        assert observation.outcomes == ["incomplete"]
        assert response.read() == payload
        assert not observation.all_rejected
