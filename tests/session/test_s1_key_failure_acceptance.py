"""S1-A5 key and signer outage acceptance through real storage paths."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Any, cast

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from support.s1_fixture_values import append_fixture_records, create_fixture_session, push_fixture_input, wrap_fixture_store

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.kernel._content import (
    ContentRef,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyUnavailable,
    Principal,
    TombstoneState,
)
from vv_agent.kernel._records import InboxItem
from vv_agent.kernel._sql import SessionNotLive
from vv_agent.session.content_runtime import (
    INPUT_CONTENT,
    SessionContentWriter,
    durable_content_reader,
    durable_content_store,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.supervisor import _tick
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall

from .helpers import record

TENANT = "default"
MASTER_KEY = b"a" * 32
SESSION_TABLES = (
    "sk_session",
    "sk_session_key",
    "sk_content",
    "sk_content_link",
    "sk_record",
    "sk_inbox",
    "sk_commit",
    "sk_checkpoint",
    "sk_tombstone",
)


class _Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "s1-a5-ledger-key"
        self.fail = False

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                if signer.fail:
                    raise RuntimeError("signer unavailable")
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


class _ToggleContentReader:
    def __init__(self) -> None:
        self.delegate = None
        self.outage = False
        self.calls = 0
        self.failed = Event()

    def bind(self, store: Any, session_id: str) -> None:
        self.delegate = durable_content_reader(store, tenant_id=store.tenant_id, session_id=session_id)

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        self.calls += 1
        if self.outage:
            self.failed.set()
            raise KeyUnavailable("temporary key-provider outage")
        assert self.delegate is not None
        return self.delegate.read(tenant_id, session_id, owner, slot, ref)


def _wait_until(predicate, *, timeout: float = 8.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("condition did not become true")


def _set_default_retry_window(monkeypatch: pytest.MonkeyPatch, seconds: float = 0.2) -> None:
    from vv_agent.session.runtime import Runtime

    monkeypatch.setattr(Runtime.__dataclass_fields__["lease_retry_base_seconds"], "default", seconds)
    monkeypatch.setattr(Runtime.__dataclass_fields__["lease_retry_cap_seconds"], "default", seconds)
    monkeypatch.setattr(
        Runtime.__dataclass_fields__["lease_retry_jitter"],
        "default",
        lambda _low, _high: seconds,
    )


def _drive_retry_at(driver: SessionDriver, session_id: str) -> int:
    store = cast(SQLiteStore, driver._root_store)
    return store._one(
        "SELECT drive_retry_at_ms FROM sk_session WHERE tenant_id=? AND session_id=?",
        (TENANT, session_id),
    )[0]


def _record_kinds(driver: SessionDriver, session_id: str) -> list[str]:
    return [stored.record.kind for stored in driver.store.read(session_id, limit=(1 << 31) - 1).records]


def _counts(store: Any, session_id: str) -> dict[str, int]:
    return {
        table: store._one(
            f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
            (TENANT, session_id),
        )[0]
        for table in SESSION_TABLES
    }


def _head_and_inbox(store: Any, session_id: str) -> tuple[tuple[Any, ...], str]:
    head = store._one(
        "SELECT head_seq,inbox_seq,active_turn_id FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (TENANT, session_id),
    )
    return head, store._chain_anchor(session_id, head[0])


def _assert_plaintext_absent(store: Any, marker: str) -> None:
    marker_bytes = marker.encode()
    for table in SESSION_TABLES:
        for row in store._rows(f"SELECT * FROM {table} WHERE tenant_id=%s", (TENANT,)):
            for value in row:
                if isinstance(value, bytes | bytearray | memoryview):
                    assert marker_bytes not in bytes(value)
                elif isinstance(value, str):
                    assert marker not in value


def _provider(tmp_path: Path, name: str) -> LocalKeyProvider:
    return LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{name}.sqlite3")


@pytest.mark.parametrize("operation", ["create", "push", "append"])
def test_signer_failure_rolls_back_writes_without_partial_admission(
    store,
    tmp_path: Path,
    operation: str,
) -> None:
    signer = _Signer()
    store.configure_input_admission(signer)
    key_provider = _provider(tmp_path, operation)
    wrapped = wrap_fixture_store(store, key_provider)
    session_id = f"s1-a5-{operation}"
    marker = f"s1-a5-{operation}-plaintext"
    try:
        if operation == "create":
            signer.fail = True
            with pytest.raises(RuntimeError, match="signer unavailable"):
                create_fixture_session(
                    wrapped,
                    session_id,
                    workspace=marker,
                    attributes={"seed": {"messages": [], "shared_state": {}}, "marker": marker},
                )

            assert set(_counts(store, session_id).values()) == {0}
            _assert_plaintext_absent(store, marker)
            return

        create_fixture_session(wrapped, session_id)
        before_counts = _counts(store, session_id)
        before_head = _head_and_inbox(store, session_id)

        signer.fail = True
        with pytest.raises(RuntimeError, match="signer unavailable"):
            if operation == "push":
                push_fixture_input(
                    wrapped,
                    session_id,
                    InboxItem("s1-a5-input", "user", {"content": marker}),
                )
            else:
                append_fixture_records(
                    wrapped,
                    session_id,
                    (
                        record(
                            "turn_started",
                            sid=session_id,
                            tid=f"{session_id}/turn",
                            handler_version=marker,
                        ),
                    ),
                    commit_id="s1-a5-append",
                )

        signer.fail = False
        assert _counts(store, session_id) == before_counts
        assert _head_and_inbox(store, session_id) == before_head
        assert store.read_input(session_id, "s1-a5-input") is None
        assert store.read(session_id).head_seq == before_head[0][0]
        _assert_plaintext_absent(store, marker)
    finally:
        key_provider.close()


def test_key_unavailable_has_one_kernel_exception_type() -> None:
    from vv_agent.kernel._hydration import KeyUnavailable as hydration_key_unavailable

    assert hydration_key_unavailable is KeyUnavailable


def test_runtime_key_unavailable_tracking_is_internal() -> None:
    from vv_agent.session.runtime import Runtime

    tracked = Runtime.__dataclass_fields__["_key_unavailable"]
    assert tracked.default is False
    assert tracked.init is False
    assert tracked.repr is False


@pytest.mark.parametrize("scenario", ["missing", "key", "reset"])
def test_core_key_tracking_preserves_idle_and_retry_paths(scenario: str) -> None:
    from vv_agent.kernel._drive import DriveBinding, DriveLeaseScope, DriveStep, drive_core
    from vv_agent.kernel._store import Lease, StoredRecord
    from vv_agent.session.timing import SystemClock

    class RuntimeWithoutTracking:
        __slots__ = ("poll_ms", "ttl_ms")

        def __init__(self) -> None:
            self.ttl_ms = 15_000
            self.poll_ms = 37

    class RuntimeWithTracking(RuntimeWithoutTracking):
        __slots__ = ("_key_unavailable",)

        def __init__(self) -> None:
            super().__init__()
            self._key_unavailable = False

    session_id = f"key-tracking-{scenario}"
    lease = Lease(session_id, "owner", 1, 30_000)
    runtime = RuntimeWithoutTracking() if scenario == "missing" else RuntimeWithTracking()
    events: list[tuple[str, int] | str] = []
    seen: list[bool] = []

    class Scope:
        clock = SystemClock()
        ready = Event()
        ready.set()
        lost = False
        successful_ready = True
        heartbeat_error: Exception | None = None
        latest_poll = None
        expires_deadline = 30.0

        def __init__(self) -> None:
            self.lease = lease

        def start(self) -> None:
            pass

        def close(self) -> None:
            events.append("close_scope")

        def adopt_confirmed(self, poll, *, expires_deadline: float) -> None:
            del poll
            raise AssertionError(expires_deadline)

        def committing(self):
            return nullcontext(lease)

    class Step:
        def __init__(self) -> None:
            self.records: tuple[StoredRecord, ...] = ()
            self.calls = 0

        def step(self) -> bool:
            self.calls += 1
            seen.append(bool(getattr(runtime, "_key_unavailable", False)))
            if scenario == "key":
                cast(Any, runtime)._key_unavailable = True
                return False
            if scenario == "reset" and self.calls == 1:
                cast(Any, runtime)._key_unavailable = True
                return True
            return False

    class Store:
        tenant_id = TENANT

        def defer_drive(self, _lease: Lease, *, retry_after_ms: int) -> None:
            events.append(("defer_drive", retry_after_ms))

        def defer_idle_drive(self, _lease: Lease, *, poll_ms: int) -> bool:
            events.append(("defer_idle_drive", poll_ms))
            return False

        @staticmethod
        def release(_lease: Lease) -> None:
            events.append("release")

        @staticmethod
        def is_runnable(_session_id: str) -> bool:
            return False

    scope = Scope()
    step = Step()
    binding = DriveBinding(
        clock=cast(Any, scope.clock),
        ttl_ms=runtime.ttl_ms,
        runtime=cast(Any, runtime),
        make_factory_heartbeat=None,
        make_runtime=None,
        factory_retry_after_ms=lambda _error: None,
        make_scope=lambda _lease, _runtime: cast(DriveLeaseScope, scope),
        make_driver=lambda _store, _session_id, _runtime, _scope: cast(DriveStep, step),
        after_drive=lambda _runtime: events.append("after_drive"),
        wake_if_runnable=lambda _runtime: events.append("wake"),
    )

    assert drive_core(cast(Any, Store()), session_id, lease=lease, failure_backoff_ms=123, make_binding=lambda: binding)
    retry_events = [event for event in events if isinstance(event, tuple)]
    assert retry_events == [("defer_drive", 123)] if scenario == "key" else [("defer_idle_drive", 37)]
    assert events[-2:] == ["release", "after_drive"]
    assert seen == ([False, False] if scenario == "reset" else [False])
    if scenario == "missing":
        assert not hasattr(runtime, "_key_unavailable")
    else:
        assert cast(Any, runtime)._key_unavailable is False


def test_session_worker_defers_preturn_key_outage_then_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_default_retry_window(monkeypatch)
    key_provider = _provider(tmp_path, "worker-preturn")
    reader = _ToggleContentReader()
    model_calls: list[Any] = []
    holder: dict[str, SessionDriver] = {}

    def runtime_factory(store, agent, config, task, session_id):
        reader.bind(store, session_id)
        runtime = holder["driver"].runtime(agent, config, task, session_id=session_id, store=store)
        runtime.content_reader = reader
        return runtime

    provider = ScriptedModelProvider.from_callback(
        "scripted",
        "m",
        lambda request: model_calls.append(request) or LLMResponse("recovered"),
    )
    driver = SessionDriver(
        content_key_provider=key_provider,
        runtime_factory=runtime_factory,
    )
    holder["driver"] = driver
    session_id = "worker-preturn"
    try:
        driver.create(session_id, ".")
        agent = Agent("worker-preturn", "Answer.", model="m")
        handle = driver.start(
            session_id,
            agent,
            RunConfig(model_provider=provider),
            "go",
            autostart=False,
        )
        reader.outage = True
        handle.start()

        _wait_until(
            lambda: reader.failed.is_set() and _drive_retry_at(driver, session_id) > cast(SQLiteStore, driver._root_store)._now()
        )
        assert "turn_failed" not in _record_kinds(driver, session_id)
        assert model_calls == []

        reader.outage = False
        driver._wake_session(session_id)
        assert handle.result(timeout=8).final_output == "recovered"
        state, _, _ = driver.store.read_state(session_id)
        assert state.active_turn_id is None
        assert len(model_calls) == 1
        assert "turn_failed" not in _record_kinds(driver, session_id)
        assert "turn_ended" in _record_kinds(driver, session_id)
    finally:
        driver.close()
        key_provider.close()


def test_normal_idle_completion_does_not_set_failure_retry() -> None:
    driver = SessionDriver()
    session_id = "normal-idle"
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which?"})])],
    )
    try:
        driver.create(session_id, ".")
        handle = driver.start(
            session_id,
            Agent("normal-idle", "Ask once.", model="m"),
            RunConfig(model_provider=provider),
            "go",
        )
        assert handle.result(timeout=8).status.value == "wait_user"
        assert _drive_retry_at(driver, session_id) == 0
        assert "turn_failed" not in _record_kinds(driver, session_id)
    finally:
        driver.close()


def test_session_worker_recovers_active_turn_without_replaying_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vv_agent.interaction import InteractionAnswer, InteractionReplyTarget

    _set_default_retry_window(monkeypatch)
    key_provider = _provider(tmp_path, "worker-active")
    reader = _ToggleContentReader()
    model_calls: list[Any] = []
    tool_calls: list[str] = []
    holder: dict[str, SessionDriver] = {}

    @function_tool
    def effect(value: str) -> str:
        tool_calls.append(value)
        return value

    def respond(request):
        model_calls.append(request)
        if len(model_calls) == 1:
            return LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which?"})])
        if len(model_calls) == 2:
            return LLMResponse("", [ToolCall("tool-1", "effect", {"value": "once"})])
        return LLMResponse("done")

    def runtime_factory(store, agent, config, task, session_id):
        reader.bind(store, session_id)
        runtime = holder["driver"].runtime(agent, config, task, session_id=session_id, store=store)
        runtime.content_reader = reader
        return runtime

    provider = ScriptedModelProvider.from_callback("scripted", "m", respond)
    driver = SessionDriver(content_key_provider=key_provider, runtime_factory=runtime_factory)
    holder["driver"] = driver
    session_id = "worker-active"
    agent = Agent("worker-active", "Ask, use the tool once, then answer.", model="m", tools=[effect])
    config = RunConfig(model_provider=provider)
    try:
        driver.create(session_id, ".")
        handle = driver.start(
            session_id,
            agent,
            config,
            "go",
            autostart=False,
        )
        handle.start()
        assert handle.result(timeout=8).status.value == "wait_user"

        runtime = driver.runtime(agent, config, session_id=session_id)
        form = next(wait["interaction"] for _sid, _runtime, wait in driver.waits(session_id, runtime) if "interaction" in wait)
        target = InteractionReplyTarget.from_dict(
            {
                name: form[name]
                for name in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
            }
        )
        reader.outage = True
        reader.failed.set()
        driver.answer(
            session_id,
            runtime,
            target,
            InteractionAnswer("text", text="blue"),
            "reply",
        )

        _wait_until(
            lambda: reader.failed.is_set() and _drive_retry_at(driver, session_id) > cast(SQLiteStore, driver._root_store)._now()
        )
        state, _, _ = driver.store.read_state(session_id)
        assert state.active_turn_id is not None
        assert "turn_failed" not in _record_kinds(driver, session_id)
        assert len(model_calls) == 1
        assert tool_calls == []

        reader.outage = False
        driver._wake_session(session_id)
        assert handle.result(timeout=8).final_output == "done"
        state, _, _ = driver.store.read_state(session_id)
        assert state.active_turn_id is None
        assert len(model_calls) == 3
        assert tool_calls == ["once"]
        assert "turn_failed" not in _record_kinds(driver, session_id)
        assert "turn_ended" in _record_kinds(driver, session_id)
    finally:
        driver.close()
        key_provider.close()


def test_supervisor_retry_gate_recovers_preturn_key_outage(tmp_path: Path) -> None:
    key_provider = _provider(tmp_path, "supervisor-preturn")
    reader = _ToggleContentReader()
    model_calls: list[dict[str, Any]] = []
    driver = SessionDriver(content_key_provider=key_provider)
    session_id = "supervisor-preturn"

    def respond(request):
        model_calls.append(request)
        return LLMResponse("supervisor-recovered")

    provider = ScriptedModelProvider.from_callback("scripted", "m", respond)
    try:
        driver.create(session_id, ".")
        driver.push(session_id, InboxItem("go", "user", {"content": "go"}))
        runtime = driver.runtime(
            Agent("supervisor-preturn", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            session_id=session_id,
        )
        reader.bind(driver.store, session_id)
        runtime.content_reader = reader
        reader.outage = True

        assert (
            _tick(
                driver.store,
                runtime=lambda _sid: runtime,
                project=lambda _sid, _consumer: None,
                failure_backoff_ms=100,
                stop_after_claimed_drive=True,
            )
            == 1
        )
        _wait_until(lambda: _drive_retry_at(driver, session_id) > cast(SQLiteStore, driver._root_store)._now())
        retry_at = _drive_retry_at(driver, session_id)
        assert retry_at <= cast(SQLiteStore, driver._root_store)._now() + 150
        assert "turn_failed" not in _record_kinds(driver, session_id)
        assert model_calls == []

        reader.outage = False
        _wait_until(lambda: driver.store.is_runnable(session_id))
        assert (
            _tick(
                driver.store,
                runtime=lambda _sid: runtime,
                project=lambda _sid, _consumer: None,
                failure_backoff_ms=100,
                stop_after_claimed_drive=True,
            )
            == 1
        )
        state, _, _ = driver.store.read_state(session_id)
        assert state.active_turn_id is None
        assert len(model_calls) == 1
        assert "turn_failed" not in _record_kinds(driver, session_id)
        assert "turn_ended" in _record_kinds(driver, session_id)
    finally:
        driver.close()
        key_provider.close()


@pytest.mark.parametrize("operation", ["put", "get"])
def test_content_key_unavailable_has_no_plaintext_or_dangling_commit(
    store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    key_provider = _provider(tmp_path, f"content-{operation}")
    wrapped = wrap_fixture_store(store, key_provider)
    session_id = f"s1-a5-content-{operation}"
    marker = f"s1-a5-content-{operation}-plaintext"
    try:
        create_fixture_session(wrapped, session_id)
        ref: ContentRef | None = None
        if operation == "get":
            push_fixture_input(
                wrapped,
                session_id,
                InboxItem("s1-a5-content-input", "user", {"content": marker}),
            )
            stored = store.read_input(session_id, "s1-a5-content-input")
            assert stored is not None
            ref = ContentRef.from_dict(stored.item.payload["content"])

        before_counts = _counts(store, session_id)
        before_head = _head_and_inbox(store, session_id)

        def unavailable(_tenant_id: str, _session_id: str, _wrapped_key: Any) -> bytes:
            raise KeyUnavailable("temporary key-provider outage")

        monkeypatch.setattr(key_provider, "unwrap", unavailable)
        with pytest.raises(KeyUnavailable, match="temporary key-provider outage"), wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)
            if operation == "put":
                writer.put_json({"secret": marker})
            else:
                assert ref is not None
                verified = writer.assert_input("s1-a5-content-input", ref, INPUT_CONTENT)
                writer.get_json(verified)

        assert _counts(store, session_id) == before_counts
        assert _head_and_inbox(store, session_id) == before_head
        _assert_plaintext_absent(store, marker)

        monkeypatch.undo()
        if operation == "get":
            assert ref is not None
            with wrapped.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)
                verified = writer.assert_input("s1-a5-content-input", ref, INPUT_CONTENT)
                assert writer.get_json(verified) == marker
    finally:
        key_provider.close()


def test_destroy_failure_stays_fenced_and_retry_erases_content(
    store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer = _Signer()
    store.configure_input_admission(signer)
    key_provider = _provider(tmp_path, "destroy")
    wrapped = wrap_fixture_store(store, key_provider)
    session_id = "s1-a5-destroy"
    marker = "s1-a5-destroy-plaintext"
    original_destroy = key_provider.destroy
    try:
        create_fixture_session(wrapped, session_id)
        push_fixture_input(
            wrapped,
            session_id,
            InboxItem("s1-a5-destroy-input", "user", {"content": marker}),
        )
        stored = store.read_input(session_id, "s1-a5-destroy-input")
        assert stored is not None
        ref = ContentRef.from_dict(stored.item.payload["content"])
        created = store.read(session_id, limit=1).records[0].record
        actor = Principal.from_dict(created._payload["principal"])
        adapter = durable_content_store(wrapped)
        deletion = SessionDeletionStore(store, adapter, signer=signer)

        monkeypatch.setattr(
            key_provider,
            "destroy",
            lambda _tenant_id, _key_id: KeyDestroyOutcome(
                False,
                KeyDestroyStatus.KEY_UNAVAILABLE,
                None,
                "temporary key-provider outage",
            ),
        )
        with pytest.raises(KeyUnavailable, match="temporary key-provider outage"):
            deletion.forget_session(TENANT, session_id, actor=actor)

        state, tombstone_state = store._one(
            "SELECT s.deletion_state,t.state FROM sk_session s "
            "JOIN sk_tombstone t ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id "
            "WHERE s.tenant_id=%s AND s.session_id=%s",
            (TENANT, session_id),
        )
        assert (state, tombstone_state) == ("forgetting", "pending")
        with pytest.raises(SessionNotLive):
            store.read(session_id)
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, session_id, ref)
        _assert_plaintext_absent(store, marker)

        monkeypatch.setattr(key_provider, "destroy", original_destroy)
        receipt = deletion.forget_session(TENANT, session_id, actor=actor)
        assert receipt.session_ids == (session_id,)
        skeleton = deletion.read_skeleton(TENANT, session_id)
        assert skeleton.tombstone is not None
        assert skeleton.tombstone.state is TombstoneState.COMPLETE
        assert _counts(store, session_id)["sk_content"] == 0
        assert _counts(store, session_id)["sk_content_link"] == 0
        assert _counts(store, session_id)["sk_session_key"] == 0
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, session_id, ref)
        _assert_plaintext_absent(store, marker)
    finally:
        key_provider.close()
