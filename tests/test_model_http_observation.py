"""Physical HTTP evidence also vetoes hidden response/read failures."""

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from vv_llm.types import APIStatusError

from vv_agent.llm.base import LlmRequest
from vv_agent.llm.errors import ModelRequestRejected, is_prompt_too_long_error, model_request_rejection
from vv_agent.llm.http_observation import HttpObservation, ObservedAnthropicHttpClient, ObservedHttpClient
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.model_settings import ModelSettings
from vv_agent.types import Message


@pytest.mark.parametrize(
    "outcomes", [[], [None], [400, None], [500, 400], [408, 400], ["timeout", 400], ["transport_error", 400], [200, 400]]
)
def test_incomplete_or_ambiguous_observation_never_confirms_rejection(outcomes):
    error = ModelRequestRejected(400, "prompt is too long")
    cast(Any, error)._vv_http_observation = HttpObservation(outcomes)
    if outcomes:
        assert model_request_rejection(error) is None
        assert not is_prompt_too_long_error(error)
    else:
        # No managed transport: only the explicit assertion can confirm rejection.
        assert model_request_rejection(error) is error


@pytest.mark.parametrize("failure", ["transport", "body_timeout", "streamed_body_timeout", "incomplete", "redirect"])
def test_observer_tracks_every_physical_request_and_read_failure(failure):
    observation = HttpObservation()

    class BrokenBody(httpx2.SyncByteStream):
        def __iter__(self):
            raise httpx2.ReadTimeout("synthetic response body timeout")
            yield b""  # pragma: no cover

    calls = []

    def handle(request):
        calls.append(1)
        if failure == "transport":
            raise httpx2.ReadError("lost")
        if failure == "incomplete":
            raise RuntimeError("interrupted")
        if failure == "redirect" and len(calls) == 1:
            return httpx2.Response(307, headers={"location": "/next"})
        return httpx2.Response(400, stream=BrokenBody() if "body_timeout" in failure else httpx2.ByteStream(b"rejected"))

    with ObservedHttpClient(observation, transport=httpx2.MockTransport(handle), follow_redirects=True) as client:
        if failure == "redirect":
            client.get("https://provider.invalid")
        elif failure == "streamed_body_timeout":
            response = client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
            with pytest.raises(httpx2.ReadTimeout):
                response.read()
        else:
            with pytest.raises((httpx2.TransportError, RuntimeError)):
                client.get("https://provider.invalid")
    assert len(observation.outcomes) == len(calls)
    assert not observation.all_rejected


def test_concurrent_send_failure_belongs_to_its_request():
    entered, release = Event(), Event()
    observation = HttpObservation()

    def handle(request):
        if request.url.path == "/a":
            entered.set()
            assert release.wait(5)
            raise httpx2.ReadTimeout("synthetic timeout", request=request)
        return httpx2.Response(400)

    with ObservedHttpClient(observation, transport=httpx2.MockTransport(handle)) as client, ThreadPoolExecutor(2) as pool:
        future = pool.submit(client.get, "https://provider.invalid/a")
        try:
            assert entered.wait(5)
            client.get("https://provider.invalid/b")
        finally:
            release.set()
        with pytest.raises(httpx2.ReadTimeout):
            future.result()
    assert observation.outcomes == ["timeout", 400]
    assert not observation.all_rejected


def test_concurrent_decode_failure_keeps_its_own_index():
    reading, second_entered, release_first, release_second = Event(), Event(), Event(), Event()
    observation = HttpObservation()

    class InvalidGzip(httpx2.SyncByteStream):
        def __iter__(self):
            reading.set()
            assert release_first.wait(5)
            yield b"not a gzip stream"

    def handle(request):
        if request.url.path == "/a":
            return httpx2.Response(400, headers={"content-encoding": "gzip"}, stream=InvalidGzip())
        second_entered.set()
        assert release_second.wait(5)
        return httpx2.Response(400)

    with ObservedHttpClient(observation, transport=httpx2.MockTransport(handle)) as client, ThreadPoolExecutor(2) as pool:
        first = pool.submit(client.get, "https://provider.invalid/a")
        try:
            assert reading.wait(5)
            second = pool.submit(client.get, "https://provider.invalid/b")
            assert second_entered.wait(5)
            release_first.set()
            with pytest.raises(httpx2.DecodingError):
                first.result()
            assert observation.outcomes == ["incomplete", None]
        finally:
            release_first.set()
            release_second.set()
        second.result()
    assert observation.outcomes == ["incomplete", 400]
    assert not observation.all_rejected


def test_failure_without_request_index_permanently_vetoes_observation():
    observation = HttpObservation()

    class BrokenAuth(httpx2.Auth):
        def auth_flow(self, request):
            raise RuntimeError("synthetic failure before request hook")
            yield request  # pragma: no cover

    with ObservedHttpClient(observation, transport=httpx2.MockTransport(lambda r: httpx2.Response(400))) as client:
        with pytest.raises(RuntimeError):
            client.get("https://provider.invalid", auth=BrokenAuth())
        client.get("https://provider.invalid")
        client.get("https://provider.invalid")
    assert observation.outcomes == [400, 400]
    assert not observation.all_rejected
    error = ModelRequestRejected(400, "Rejected")
    cast(Any, error)._vv_http_observation = observation
    assert model_request_rejection(error) is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_managed_client_preserves_vendor_network_defaults(monkeypatch, provider):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    captured = []

    def create(**kwargs):
        captured.append(kwargs["http_client"])
        return SimpleNamespace(create_completion=lambda **kw: [SimpleNamespace(content="done", tool_calls=[], usage={})])

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", create)
    client = VvLlmClient([EndpointTarget("a", "unused", "https://provider.invalid")], backend=provider)
    client.complete(LlmRequest("test", [Message("user", "synthetic")]))
    with import_module(provider).DefaultHttpxClient() as default:
        expected = default._transport._pool
        actual = captured[0]._transport._pool
        assert actual._max_connections == expected._max_connections
        assert actual._max_keepalive_connections == expected._max_keepalive_connections
        assert actual._keepalive_expiry == expected._keepalive_expiry
        assert actual._socket_options == expected._socket_options
        assert actual._http2 == expected._http2
        assert captured[0].trust_env == default.trust_env
        assert captured[0].follow_redirects == default.follow_redirects
    assert captured[0].is_closed


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
@pytest.mark.parametrize("mode", ["environment", "explicit", "no_environment"])
def test_proxy_and_trust_env_priority(monkeypatch, client_type, mode):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:33331")
    proxy = "http://127.0.0.1:33332" if mode == "explicit" else None
    with client_type(HttpObservation(), proxy=proxy, trust_env=mode != "no_environment") as client:
        pool = client._transport_for_url(httpx2.URL("https://provider.invalid"))._pool
        if mode == "no_environment":
            assert not hasattr(pool, "_proxy_url")
        else:
            assert pool._proxy_url.port == (33332 if mode == "explicit" else 33331)
        with import_module(
            "anthropic" if client_type is ObservedAnthropicHttpClient else "openai"
        ).DefaultHttpxClient() as default:
            assert pool._max_connections == default._transport._pool._max_connections
            assert pool._max_keepalive_connections == default._transport._pool._max_keepalive_connections
            assert pool._keepalive_expiry == default._transport._pool._keepalive_expiry
            assert pool._socket_options == default._transport._pool._socket_options


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("timeout", [None, 7.0])
def test_vendor_per_request_timeout_overrides_client_default(monkeypatch, provider, timeout):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    timeouts = []

    def handle(request):
        timeouts.append(request.extensions["timeout"])
        return httpx2.Response(400, json={"error": {"message": "Synthetic rejected", "type": "invalid_request_error"}})

    original = httpx2.Client.__init__

    def init(self, **kwargs):
        kwargs["transport"] = httpx2.MockTransport(handle)
        original(self, **kwargs)

    monkeypatch.setattr(httpx2.Client, "__init__", init)
    client = VvLlmClient(
        [EndpointTarget("a", "unused", "https://provider.invalid")], backend=provider, max_retries_per_endpoint=1
    )
    with pytest.raises(APIStatusError):
        client.complete(LlmRequest("test", [Message("user", "synthetic")], model_settings=ModelSettings(timeout_seconds=timeout)))
    assert timeouts == [dict.fromkeys(("connect", "read", "write", "pool"), timeout or 90.0)]
