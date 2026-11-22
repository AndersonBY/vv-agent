from __future__ import annotations

from contextlib import closing, suppress
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from time import monotonic_ns, sleep
from typing import Any

import pytest

from vv_agent.kernel._content import ContentCleanupUnavailable
from vv_agent.kernel._content_runtime import _provision_sqlite_object_content
from vv_agent.kernel._external_content import _SQLiteObjectContent
from vv_agent.session._s3_transport import _S3ProcessTransport, _S3Request, _S3TransportError, _S3TransportTimeout
from vv_agent.session.content_runtime import bind_content_runtime
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.s3_content import CiphertextTemporarilyUnavailable, S3CiphertextStore
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .test_s3_reservation_adapter import _OWNER, _pin
from .test_sqlite_object_session_driver import _provision_journal

_PAYLOAD = b"0123456789abcdefghijklmnopqrstuv"


def _client(endpoint_url: str) -> Any:
    boto3 = pytest.importorskip("boto3")
    config = pytest.importorskip("botocore.config")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name="us-east-1",
        aws_access_key_id="fixture-access",
        aws_secret_access_key="fixture-secret",
        config=config.Config(
            connect_timeout=1,
            read_timeout=1,
            proxies={},
            retries={"total_max_attempts": 1, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )


class _Server:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[str] = []
        self.release = Event()

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                owner.calls.append("GET")
                if self.path.endswith("?versioning"):
                    self._reply(200, b"<VersioningConfiguration/>")
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(_PAYLOAD)))
                self.end_headers()
                with suppress(BrokenPipeError, ConnectionResetError):
                    if owner.mode == "trickle-get":
                        for offset in range(0, len(_PAYLOAD), 4):
                            self.wfile.write(_PAYLOAD[offset : offset + 4])
                            sleep(0.08)
                    else:
                        if owner.mode == "delayed-get":
                            owner.release.wait(2)
                        self.wfile.write(_PAYLOAD)

            def do_PUT(self) -> None:
                owner.calls.append("PUT")
                if owner.mode == "stop-reading":
                    owner.release.wait(2)
                    self.close_connection = True
                    return
                self.rfile.read(int(self.headers["Content-Length"]))
                if owner.mode == "slow-publication":
                    with suppress(BrokenPipeError, ConnectionResetError):
                        sleep(0.12)
                        if self.headers.get("If-Match") is not None:
                            owner.release.wait(2)
                        self._reply(200, b"", headers={"ETag": '"fixture"'})
                    return
                if owner.mode == "trickle-put":
                    with suppress(BrokenPipeError, ConnectionResetError):
                        for chunk in (b"HTTP/1.1 200 OK\r\n", b'ETag: "fixture"\r\n', b"X-Test: trace\r\n"):
                            self.wfile.write(chunk)
                            sleep(0.2)
                        self.wfile.write(b"Content-Length: 0\r\n\r\n")
                    return
                self._reply(200, b"", headers={"ETag": '"fixture"'})

            def do_DELETE(self) -> None:
                owner.calls.append("DELETE")
                with suppress(BrokenPipeError, ConnectionResetError):
                    if owner.mode == "delayed-delete":
                        owner.release.wait(2)
                    self._reply(204, b"")

            def _reply(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    @property
    def endpoint_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> _Server:
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release.set()
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()


def _transport(server: _Server) -> _S3ProcessTransport:
    pytest.importorskip("boto3")
    return _S3ProcessTransport(partial(_client, server.endpoint_url), endpoint_url=server.endpoint_url)


def _put(body: bytes = b"marker") -> _S3Request:
    return _S3Request("put_object", "fixture", "ciphertext", body=body, if_none_match="*")


def _get() -> _S3Request:
    return _S3Request("get_object", "fixture", "ciphertext", ciphertext_size=len(_PAYLOAD))


def _deadline(seconds: float = 0.2) -> int:
    return monotonic_ns() + int(seconds * 1_000_000_000)


def test_authorizer_rejects_before_dispatch() -> None:
    def reject() -> int:
        raise RuntimeError("fence expired")

    with _Server("normal") as server, _transport(server) as transport:
        with pytest.raises(RuntimeError, match="expired"):
            transport.execute(_put(), deadline_ns=_deadline(), authorize=reject)
        assert server.calls == []
        assert transport.execute(_get(), deadline_ns=_deadline()).body == _PAYLOAD
    assert server.calls == ["GET"]


def test_authorizer_can_tighten_deadline_before_dispatch() -> None:
    with _Server("trickle-put") as server, _transport(server) as transport, pytest.raises(_S3TransportTimeout) as raised:
        transport.execute(_put(), deadline_ns=_deadline(), authorize=monotonic_ns)
    assert raised.value.may_have_dispatched is False
    assert server.calls == []


@pytest.mark.parametrize("mode,method", [("trickle-put", "PUT"), ("trickle-get", "GET"), ("delayed-get", "GET")])
def test_cumulative_deadline_rejects_stalled_or_trickled_response(mode: str, method: str) -> None:
    with _Server(mode) as server, _transport(server) as transport:
        deadline_ns = _deadline()
        with pytest.raises(_S3TransportTimeout) as raised:
            transport.execute(_put() if method == "PUT" else _get(), deadline_ns=deadline_ns)
        assert raised.value.may_have_dispatched is True
        assert monotonic_ns() < deadline_ns + 200_000_000
        assert server.calls == [method]


def test_upload_deadline_rejects_server_that_never_reads_and_invalidates_slot() -> None:
    request = _put(b"x" * (16 * 1024 * 1024))
    with _Server("stop-reading") as server, _transport(server) as transport:
        deadline_ns = _deadline(0.35)
        with pytest.raises(_S3TransportTimeout) as raised:
            transport.execute(request, deadline_ns=deadline_ns)
        assert raised.value.may_have_dispatched is True
        assert monotonic_ns() < deadline_ns + 250_000_000
        with pytest.raises(_S3TransportError, match="slot is unavailable"):
            transport.execute(_put(), deadline_ns=_deadline())
        assert server.calls == ["PUT"]


def test_one_send_per_request_after_success() -> None:
    with _Server("normal") as server, _transport(server) as transport:
        deadline_ns = _deadline(1)
        versioning = transport.execute(_S3Request("get_bucket_versioning", "fixture"), deadline_ns=deadline_ns)
        assert versioning.status == 200 and versioning.versioning is None
        marker = transport.execute(_put(), deadline_ns=deadline_ns)
        assert marker.status == 200 and marker.etag == '"fixture"'
        body = transport.execute(_get(), deadline_ns=deadline_ns)
        assert body.status == 200 and body.body == _PAYLOAD
        deleted = transport.execute(_S3Request("delete_object", "fixture", "ciphertext"), deadline_ns=deadline_ns)
        assert deleted.status == 204 and not deleted.delete_marker
    assert server.calls == ["GET", "PUT", "GET", "DELETE"]


def test_real_transport_timeout_rolls_back_sql_create_and_keeps_bound_orphan(tmp_path) -> None:
    class ShortBudgetStore(S3CiphertextStore):
        def put(self, *args, **kwargs):
            kwargs.update(deadline_ns=_deadline(0.3), clock_ns=monotonic_ns)
            return super().put(*args, **kwargs)

    with (
        _Server("slow-publication") as server,
        _transport(server) as transport,
        SQLiteStore.standalone(str(tmp_path / "primary.sqlite")) as raw,
        closing(LocalKeyProvider(b"x" * 32, state_path=tmp_path / "keys.sqlite")) as provider,
    ):
        raw.configure_input_admission(allow_unsigned=True)
        journal = _provision_journal(raw, tmp_path / "allocation.sqlite")
        objects = ShortBudgetStore(transport, bucket="fixture", journal=journal)
        binding = _SQLiteObjectContent(journal, objects, provider._provision_exact_key_lifecycle())
        _provision_sqlite_object_content(raw, key_provider=provider, objects=binding)
        bind_content_runtime(raw, key_provider=provider, cipher=AESGCMCipher(), _object_content=binding)
        with closing(SessionDriver(store=raw)) as driver, pytest.raises(CiphertextTemporarilyUnavailable) as raised:
            driver.create("root", "/external")
        assert raised.value.may_have_dispatched is True
        assert raw._rows("SELECT session_id FROM sk_session") == []
        assert raw._rows("SELECT object_locator FROM sk_content") == []
        (allocation,) = journal.scoped_allocations("default", ("root",))
        assert allocation.state == "bound" and allocation.owner is not None
    assert server.calls == ["GET", "PUT", "PUT"]


def test_real_transport_delete_timeout_keeps_unconfirmed_claim(tmp_path) -> None:
    with (
        _Server("delayed-delete") as server,
        _transport(server) as transport,
        SQLiteStore.standalone(str(tmp_path / "primary.sqlite")) as raw,
    ):
        journal = _provision_journal(raw, tmp_path / "allocation.sqlite")
        store = S3CiphertextStore(transport, bucket="fixture", journal=journal)
        locator = store.put(_OWNER, _PAYLOAD, pin_until_ms=_pin(), check_fence=lambda: None)
        allocation = journal.get(locator[1:])
        assert allocation is not None
        claim = journal.claim_bound(allocation)
        assert claim is not None
        with pytest.raises(ContentCleanupUnavailable) as raised:
            store.delete_claimed(claim, check_fence=lambda: None, deadline_ns=_deadline(), clock_ns=monotonic_ns)
        cause = raised.value.__cause__
        assert isinstance(cause, _S3TransportTimeout) and cause.may_have_dispatched
        assert journal.get(locator[1:]) == claim and not claim.delete_confirmed
    assert server.calls == ["GET", "PUT", "PUT", "GET", "DELETE"]
