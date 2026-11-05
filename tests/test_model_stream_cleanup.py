"""Keep callback tracebacks alive while checking real loopback response lifetimes."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import httpx2
import pytest

from vv_agent.llm.base import LlmRequest
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.types import Message


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_callback_error_closes_loopback_streams(monkeypatch, provider):
    streams, clients, retained = [], [], []
    wire = (
        b'data: {"id":"test","object":"chat.completion.chunk","created":0,"model":"test",'
        b'"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )
    if provider == "anthropic":
        wire = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message",'
            b'"role":"assistant","model":"test","content":[],"stop_reason":null,"stop_sequence":null,'
            b'"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"hello"}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, format, *args):
            pass

    class TrackedStream(httpx2.SyncByteStream):
        def __init__(self, stream):
            self.stream, self.closed = stream, False

        def __iter__(self):
            yield from self.stream

        def close(self):
            self.closed = True
            self.stream.close()

    original_handle = httpx2.HTTPTransport.handle_request
    original_init = httpx2.Client.__init__

    def handle(self, request):
        response = original_handle(self, request)
        response.stream = TrackedStream(response.stream)
        streams.append(response.stream)
        return response

    def init(self, **kwargs):
        kwargs["trust_env"] = False
        original_init(self, **kwargs)
        clients.append(self)

    def callback(_):
        raise RuntimeError("synthetic callback exit")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(httpx2.Client, "__init__", init)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for _ in range(3):
                client = VvLlmClient(
                    [EndpointTarget("a", "unused", f"http://127.0.0.1:{server.server_port}/v1")], backend=provider
                )
                with pytest.raises(RuntimeError) as error:
                    client.complete_with_stream(LlmRequest("test", [Message("user", "synthetic")]), callback)
                retained.append(error.value)
            assert len(retained) == len(streams) == 3
            assert all(c.is_closed for c in clients)
            all_streams_closed = all(s.closed for s in streams)
            print(f"all_streams_closed={str(all_streams_closed).lower()}")
            assert all_streams_closed
        finally:
            server.shutdown()
            thread.join(5)
