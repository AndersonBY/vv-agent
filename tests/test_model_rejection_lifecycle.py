"""Real HTTP/1.1 stalled bodies must not allocate observer threads or leak sockets."""

import gc
import importlib
import json
import os
import selectors
import socket
import sys
import threading
import time
from contextlib import contextmanager, suppress

import httpx2
import pytest
from session.test_recovery_matrix import records_of, runtime, start

from vv_agent.llm import http_observation as h
from vv_agent.llm.base import LlmRequest
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import Message

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux procfs FD and socket counts")


class Loopback:
    """One fixed server thread multiplexes sockets; stalled peers never send a body."""

    def __init__(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.listener, selectors.EVENT_READ)
        self.peers = {}
        self.requests = 0
        self.retry = False
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.serve, name="loopback-http-server")
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.listener.getsockname()[1]}"

    def close_peer(self, peer):
        self.selector.unregister(peer)
        self.peers.pop(peer)
        peer.close()

    def serve(self):
        while not self.stopping.is_set():
            for key, _ in self.selector.select(0.01):
                peer = key.fileobj
                assert isinstance(peer, socket.socket)
                if peer is self.listener:
                    peer, _ = self.listener.accept()
                    self.peers[peer] = b""
                    self.selector.register(peer, selectors.EVENT_READ)
                    continue
                try:
                    data = peer.recv(65536)
                except ConnectionResetError:
                    data = b""
                if not data:
                    self.close_peer(peer)
                    continue
                self.peers[peer] += data
                head, separator, body = self.peers[peer].partition(b"\r\n\r\n")
                if not separator:
                    continue
                length = next(
                    (int(line.split(b":", 1)[1]) for line in head.lower().split(b"\r\n") if line.startswith(b"content-length:")),
                    0,
                )
                if len(body) < length:
                    continue
                self.requests += 1
                stalled = self.requests % 3 != 0 if self.retry else b"/stall" in head.split(b"\r\n", 1)[0]
                status = b"429 Too Many Requests" if stalled or self.retry else b"200 OK"
                peer.sendall(b"HTTP/1.1 " + status + b"\r\nContent-Length: 2\r\nRetry-After: 7\r\nConnection: close\r\n\r\n")
                self.peers[peer] = b""
                if not stalled:
                    peer.sendall(b"{}")
                    self.close_peer(peer)
        for peer in list(self.peers):
            self.close_peer(peer)

    def close(self):
        self.stopping.set()
        self.thread.join(2)
        assert not self.thread.is_alive()
        self.selector.close()
        self.listener.close()


@pytest.fixture
def loopback():
    server = Loopback()
    try:
        yield server
    finally:
        server.close()


def resources():
    gc.collect()
    threads = [(thread.name, thread.ident) for thread in threading.enumerate()]
    fds = os.listdir("/proc/self/fd")
    sockets = 0
    for fd in fds:
        with suppress(FileNotFoundError):
            sockets += os.readlink(f"/proc/self/fd/{fd}").startswith("socket:")
    return dict(active_count=threading.active_count(), threads=threads, fds=len(fds), sockets=sockets)


@contextmanager
def unchanged_resources(case, server, **details):
    before = resources()
    yield
    until = time.monotonic() + 0.5
    while server.peers and time.monotonic() < until:
        time.sleep(0.005)
    after = resources()
    print(json.dumps(dict(case=case, before=before, after=after, **details)), flush=True)
    assert not [t for t in threading.enumerate() if t.name == "model-http-prefetch"]
    assert after == before


@pytest.mark.parametrize("client_type", [h.ObservedHttpClient, h.ObservedAnthropicHttpClient])
@pytest.mark.parametrize("timeout", [None, 2.0])
def test_eight_closed_stalls_then_success_with_one_connection(loopback, monkeypatch, client_type, timeout):
    monkeypatch.setattr(h, "_PREFETCH_SECONDS", 0.025)
    observation = h.HttpObservation()
    with unchanged_resources("eight_closed_stalls", loopback, client=client_type.__name__, timeout=timeout):
        with client_type(
            observation, trust_env=False, timeout=timeout, limits=httpx2.Limits(max_connections=1, max_keepalive_connections=1)
        ) as client:
            for _ in range(8):
                response = client.send(client.build_request("GET", loopback.url + "/stall"), stream=True)
                assert not observation.all_rejected
                response.close()
            normal = client.get(loopback.url + "/normal")
            assert normal.status_code == 200 and normal.content == b"{}"
        assert loopback.requests == 9
        assert not observation.all_rejected


def test_default_ten_second_deadline_with_ninety_second_read_timeout(loopback):
    assert h._PREFETCH_SECONDS == 10.0
    observation = h.HttpObservation()
    with (
        unchanged_resources("default_ten_second_deadline", loopback),
        h.ObservedHttpClient(observation, trust_env=False, timeout=90) as client,
    ):
        began = time.monotonic()
        response = client.send(client.build_request("GET", loopback.url + "/stall"), stream=True)
        elapsed = time.monotonic() - began
        print(
            json.dumps(
                dict(default_cap=h._PREFETCH_SECONDS, native_read_timeout=90, elapsed=elapsed, outcomes=observation.outcomes)
            ),
            flush=True,
        )
        assert 9.9 <= elapsed < 20.5
        assert not observation.all_rejected
        response.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("kernel", [False, True])
def test_real_vv_llm_stalled_intermediate_429_releases_resources(loopback, monkeypatch, provider, kernel):
    native = importlib.import_module(provider)
    loopback.retry = True
    monkeypatch.setattr(h, "_PREFETCH_SECONDS", 0.025)
    monkeypatch.setattr(native._base_client.SyncAPIClient, "_sleep_for_retry", lambda self, **kwargs: None)
    monkeypatch.setattr(VvLlmClient, "_sleep_backoff", lambda *args: None)
    original = httpx2.Client.__init__
    clients = []

    def init(self, **kwargs):
        kwargs["trust_env"] = False
        original(self, **kwargs)
        clients.append(self)

    monkeypatch.setattr(httpx2.Client, "__init__", init)
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        rt = runtime(store, [])
        rt.llm = VvLlmClient(
            [EndpointTarget("a", "synthetic-unused", loopback.url)],
            backend=provider,
            max_retries_per_endpoint=1,
            timeout_seconds=5,
        )
        with unchanged_resources("real_vv_llm_stalled_429", loopback, provider=provider, kernel=kernel):
            if kernel:
                drive(store, "s", runtime=rt)
                terminal = records_of(store, "turn_ended")[-1].payload
                assert terminal["reason"] == "model_outcome_unknown"
                assert terminal["unconfirmed_operations"]
                assert len(records_of(store, "op_unknown")) == 2
                assert not records_of(store, "op_completed")
                assert loopback.requests == 6
            else:
                with pytest.raises(RuntimeError):
                    rt.llm.complete(LlmRequest(model="test", messages=[Message(role="user", content="synthetic")], tools=[]))
                assert loopback.requests == 3
            assert all(client.is_closed for client in clients)
