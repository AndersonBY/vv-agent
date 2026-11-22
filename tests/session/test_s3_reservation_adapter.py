from __future__ import annotations

import time
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from vv_agent.kernel import _allocation_journal
from vv_agent.kernel._allocation_journal import AllocationOwner, SQLiteAllocationJournal
from vv_agent.kernel._content import ContentBindingMismatch, ContentCleanupUnavailable, ContentIntegrityError, ContentUnavailable
from vv_agent.session.s3_content import CiphertextTemporarilyUnavailable, S3CiphertextStore

_OWNER = AllocationOwner("tenant", "child", "content", "parent", "admission", "provider", "key")
_MARKER_TAG = '"marker-ETag"'
_CIPHERTEXT = b"encrypted content" * 2


def _pin() -> int:
    return time.time_ns() // 1_000_000 + 10_000


class _Client:
    def __init__(self, journal: SQLiteAllocationJournal) -> None:
        self.journal = journal
        self.calls: list[dict[str, Any]] = []
        self.objects: dict[str, bytes] = {}
        self.marker_etag = _MARKER_TAG
        self.fail = ""
        self.versioned = False
        self.fence_held = True
        self.reply_body: bytes | None = None
        self.meta = self
        self.endpoint_url = "http://fixture.invalid"
        self.config = self
        self.events = self
        self.retries = {"total_max_attempts": 1}

    def register_first(self, *_: Any, **kwargs: Any) -> None:
        assert kwargs["unique_id"] == "vv-agent-s1-first-put-only"

    def fence(self) -> None:
        if not self.fence_held:
            raise RuntimeError("fence is lost")

    def put_object(self, **request: Any) -> dict[str, Any]:
        self.fence()
        self.calls.append(request)
        key = request["Key"]
        row = self.journal.get(bytes.fromhex(key.rsplit("/", maxsplit=1)[1]))
        assert row is not None and request["ContentLength"] == len(request["Body"])
        if "IfNoneMatch" in request:
            assert request["IfNoneMatch"] == "*" and request.keys() == {"Bucket", "Key", "Body", "ContentLength", "IfNoneMatch"}
            assert row.owner is None and row.reservation_etag is None and row.state == "unbound"
            self.objects[key] = request["Body"]
            if self.fail == "marker":
                raise TimeoutError
            return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ETag": self.marker_etag}
        assert request["IfMatch"] == _MARKER_TAG and row.reservation_etag == _MARKER_TAG
        assert row.owner == _OWNER and row.state == "bound"
        self.objects[key] = request["Body"]
        if self.fail == "ciphertext":
            raise TimeoutError
        if self.fail == "fence-after-put":
            self.fence_held = False
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        assert Bucket == "fixture"
        body = self.objects[Key]
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "ContentLength": len(body),
            "Body": BytesIO(body if self.reply_body is None else self.reply_body),
        }

    def get_bucket_versioning(self, *, Bucket: str) -> dict[str, Any]:
        assert Bucket == "fixture"
        return {"ResponseMetadata": {"HTTPStatusCode": 200}, "Status": "Suspended" if self.versioned else None}

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.fence()
        assert Bucket == "fixture"
        self.objects.pop(Key, None)
        if self.fail == "delete":
            raise TimeoutError
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}


@pytest.fixture
def adapter(tmp_path: Path) -> tuple[SQLiteAllocationJournal, _Client, S3CiphertextStore]:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="database", namespace="fixture")
    journal = SQLiteAllocationJournal(path, identity)
    client = _Client(journal)
    return journal, client, S3CiphertextStore(client, bucket="fixture", journal=journal)


def test_s3_marker_then_bound_upload_and_exact_read(adapter) -> None:
    _, client, store = adapter
    locator = store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=client.fence)
    assert len(locator) == 33 and locator[0] == 1
    marker, ciphertext = client.calls
    assert marker["Key"] == ciphertext["Key"]
    assert marker["Key"].endswith("/alloc/" + locator[1:].hex())
    assert len(marker["Body"]) < len(ciphertext["Body"])
    assert store.get(_OWNER, locator, ciphertext_size=len(_CIPHERTEXT)) == _CIPHERTEXT
    with pytest.raises(ContentBindingMismatch):
        store.get(replace(_OWNER, admission_id="other"), locator, ciphertext_size=len(_CIPHERTEXT))
    with pytest.raises(ContentBindingMismatch):
        store.get(_OWNER, b"\x02" + locator[1:], ciphertext_size=len(_CIPHERTEXT))
    client.reply_body = _CIPHERTEXT + b"x"
    with pytest.raises(ContentIntegrityError):
        store.get(_OWNER, locator, ciphertext_size=len(_CIPHERTEXT))


@pytest.mark.parametrize("mode", ["marker", "ciphertext", "fence-after-put"])
def test_unknown_result_retains_exact_recovery_state(adapter, mode: str) -> None:
    journal, client, store = adapter
    client.fail = mode
    with pytest.raises(RuntimeError if mode == "fence-after-put" else CiphertextTemporarilyUnavailable):
        store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=client.fence)
    key = client.calls[0]["Key"]
    row = journal.get(bytes.fromhex(key.rsplit("/", maxsplit=1)[1]))
    assert row is not None
    if mode == "marker":
        assert len(client.calls) == 1 and row.state == "unbound" and row.owner is None
    else:
        assert len(client.calls) == 2 and row.state == "bound" and row.owner == _OWNER


def test_invalid_marker_etag_cannot_bind(adapter) -> None:
    journal, client, store = adapter
    client.marker_etag = 'W/"weak"'
    with pytest.raises(CiphertextTemporarilyUnavailable):
        store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=client.fence)
    assert len(client.calls) == 1
    row = journal.get(bytes.fromhex(client.calls[0]["Key"].rsplit("/", maxsplit=1)[1]))
    assert row is not None and row.state == "unbound"


@pytest.mark.parametrize("operation", ["reserve", "bind"])
@pytest.mark.parametrize("commit_matches", [True, False])
def test_lost_journal_ack_requires_exact_durable_readback(
    adapter, monkeypatch: pytest.MonkeyPatch, operation: str, commit_matches: bool
) -> None:
    journal, client, store = adapter
    original = getattr(journal, operation)

    def lost_ack(allocation_id: bytes, *args: Any, **kwargs: Any) -> None:
        if not commit_matches:
            if operation == "reserve":
                kwargs["pin_until_ms"] += 1
            else:
                kwargs["reservation_etag"] = '"wrong-etag"'
        assert original(allocation_id, *args, **kwargs)
        raise ContentUnavailable("journal reply lost")

    monkeypatch.setattr(journal, operation, lost_ack)
    if commit_matches:
        locator = store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=client.fence)
        assert store.get(_OWNER, locator, ciphertext_size=len(_CIPHERTEXT)) == _CIPHERTEXT
        assert len(client.calls) == 2
    else:
        with pytest.raises(CiphertextTemporarilyUnavailable):
            store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=client.fence)
        assert len(client.calls) == (0 if operation == "reserve" else 1)


def test_anonymous_delete_ack_keeps_claim_for_late_marker(adapter, monkeypatch: pytest.MonkeyPatch) -> None:
    journal, client, store = adapter
    deadline = _pin()
    client.fail = "marker"
    with pytest.raises(CiphertextTemporarilyUnavailable):
        store.put(_OWNER, _CIPHERTEXT, pin_until_ms=deadline, check_fence=client.fence)
    request = client.calls[0]
    allocation_id = bytes.fromhex(request["Key"].rsplit("/", maxsplit=1)[1])
    monkeypatch.setattr(_allocation_journal, "_NOW_MS", str(deadline))
    claim = journal.claim_anonymous(allocation_id)
    assert claim is not None and claim.owner is None
    store.delete_claimed(claim, check_fence=client.fence)
    assert request["Key"] not in client.objects
    # A late completion after the DELETE must still be discoverable.
    client.objects[request["Key"]] = request["Body"]
    assert journal.anonymous_candidates(limit=1) == (claim,)
    store.delete_claimed(claim, check_fence=client.fence)
    assert request["Key"] not in client.objects
    assert journal.get(allocation_id) == claim


def test_delete_requires_exact_durable_claim_and_preserves_unknown_ack(adapter) -> None:
    journal, client, store = adapter
    locator = store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=client.fence)
    bound = journal.get(locator[1:])
    assert bound is not None
    with pytest.raises(ContentCleanupUnavailable):
        store.delete_claimed(replace(bound, state="reaping"), check_fence=client.fence)
    assert client.objects
    claim = journal.claim_bound(bound)
    assert claim is not None
    with pytest.raises(ContentUnavailable):
        store.get(_OWNER, locator, ciphertext_size=len(_CIPHERTEXT))
    with pytest.raises(ContentCleanupUnavailable):
        store.delete_claimed(replace(claim, reservation_etag='"wrong"'), check_fence=client.fence)
    client.versioned = True
    with pytest.raises(ContentCleanupUnavailable, match="unversioned"):
        store.delete_claimed(claim, check_fence=client.fence)
    assert client.objects
    client.versioned = False
    client.fail = "delete"
    with pytest.raises(ContentCleanupUnavailable):
        store.delete_claimed(claim, check_fence=client.fence)
    assert journal.get(claim.allocation_id) == claim
    client.fail = ""
    store.delete_claimed(claim, check_fence=client.fence)
    assert journal.get(claim.allocation_id) == claim
    assert journal.finish_reap(claim, remote_terminal=True)


def test_rejects_client_retry_policy_and_versioned_bucket(adapter) -> None:
    journal, client, _ = adapter
    client.retries["total_max_attempts"] = 2
    with pytest.raises(ValueError, match="total_max_attempts"):
        S3CiphertextStore(client, bucket="fixture", journal=journal)
    client.retries["total_max_attempts"] = 1
    client.versioned = True
    with pytest.raises(ContentCleanupUnavailable, match="unversioned"):
        S3CiphertextStore(client, bucket="fixture", journal=journal)


def test_botocore_sends_one_marker_even_after_region_redirect(adapter) -> None:
    boto3 = pytest.importorskip("boto3")
    config = pytest.importorskip("botocore.config")
    journal, _, _ = adapter
    marker_keys: list[str] = []

    class RedirectServer(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-amz-bucket-region", "us-west-2")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._reply(200, b"<VersioningConfiguration/>")

        def do_PUT(self) -> None:
            marker_keys.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self._reply(301, b"<Error><Code>PermanentRedirect</Code></Error>")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), RedirectServer) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            with closing(
                boto3.client(
                    "s3",
                    endpoint_url=f"http://127.0.0.1:{server.server_port}",
                    region_name="us-east-1",
                    aws_access_key_id="fixture-access-key",
                    aws_secret_access_key="fixture-secret-key",
                    config=config.Config(
                        connect_timeout=1,
                        read_timeout=1,
                        proxies={},
                        retries={"total_max_attempts": 1, "mode": "standard"},
                        s3={"addressing_style": "path"},
                    ),
                ),
            ) as client:
                store = S3CiphertextStore(client, bucket="fixture", journal=journal)
                with pytest.raises(CiphertextTemporarilyUnavailable):
                    store.put(_OWNER, _CIPHERTEXT, pin_until_ms=_pin(), check_fence=lambda: None)
        finally:
            server.shutdown()
            thread.join(timeout=2)
    assert len(marker_keys) == 1
