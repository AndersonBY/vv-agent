"""W4 local worker lifecycle, transaction binding and owner cleanup."""

from __future__ import annotations

from contextlib import ExitStack
from gc import collect
from pathlib import Path
from threading import Barrier, Event, Thread, current_thread
from threading import enumerate as thread_enumerate
from time import monotonic, sleep
from typing import Any, cast

import pytest

import vv_agent.session.surfaces as surface_module
from vv_agent import Agent, RunConfig, Runner
from vv_agent.events import AssistantDeltaEvent
from vv_agent.kernel._content import ContentRef, Principal
from vv_agent.kernel._content import KeyUnavailable as BackendKeyUnavailable
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.kernel._sql import SessionNotLive, UnsupportedFeatureLevel, UnsupportedWorkerCapabilities
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
from vv_agent.model import ScriptedModelProvider, resolve_model_selection
from vv_agent.run_handle import RunHandle
from vv_agent.session.bindings import MissingHostBinding
from vv_agent.session.children import ChildSession, InvalidChildBatch
from vv_agent.session.content_runtime import bind_content_key_provider
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.runtime import (
    static_model_selection_requirements,
    static_session_requirements,
    static_worker_supports,
)
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Conflict
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import AgentStatus, LLMResponse, ToolCall

from .conftest import open_store as configured_open_store
from .helpers import (
    content_ref,
    create_fixture_session,
    fixture_key_provider,
    record,
    wrap_fixture_store,
)


@function_tool
def child_task() -> str:
    raise AssertionError("child_task must use child admission")


class _HostAdapter:
    """A backend-style adapter over the connection supplied to the runtime factory."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.store._rows(sql, params)


def _wait_for(predicate, timeout: float = 8.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


def test_bootstrap_requirements_never_authorize_worker_capabilities(tmp_path: Path) -> None:
    model_calls: list[object] = []
    tool_calls: list[object] = []
    factory_calls: list[object] = []

    @function_tool
    def forbidden_tool() -> str:
        tool_calls.append(object())
        return "unexpected"

    agent = Agent("capability-bound", "Answer.", model="m", tools=[forbidden_tool])
    provider = ScriptedModelProvider.from_callback(
        "scripted",
        "m",
        lambda request: model_calls.append(request) or LLMResponse("unexpected"),
    )
    config = RunConfig(model_provider=provider)
    trusted_supports = static_worker_supports(agent, config, resolved=None)
    base = static_session_requirements(
        agent,
        config,
        resolved=None,
        handler_version="1",
    )
    bootstrap = SessionRequirements(
        base.level,
        base.handler_version,
        tuple(sorted({*base.capabilities, "deployment:unavailable"})),
    )
    assert "deployment:unavailable" not in trusted_supports

    key_provider = LocalKeyProvider(b"c" * 32)
    database = tmp_path / "capability-bound.sqlite3"
    with ExitStack() as stack:
        producer_store = stack.enter_context(SQLiteStore.standalone(str(database)))
        producer_store.configure_input_admission(allow_unsigned=True)
        producer_store.install_schema()
        producer_store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        consumer_store = stack.enter_context(SQLiteStore.standalone(str(database)))
        consumer_store.configure_input_admission(allow_unsigned=True)
        compatible_store = stack.enter_context(SQLiteStore.standalone(str(database)))
        compatible_store.configure_input_admission(allow_unsigned=True)

        def unexpected_factory(*_args):
            factory_calls.append(object())
            raise AssertionError("incompatible registration must not construct a Runtime")

        producer = SessionDriver(
            store=producer_store,
            content_key_provider=key_provider,
        )
        producer._configure_static_worker(
            level=bootstrap.level,
            capabilities=tuple(sorted({*trusted_supports, "deployment:unavailable"})),
            requirements=bootstrap,
        )
        consumer = SessionDriver(
            store=consumer_store,
            content_key_provider=key_provider,
            runtime_factory=unexpected_factory,
        )
        compatible = SessionDriver(
            store=compatible_store,
            content_key_provider=key_provider,
            capabilities=("deployment:unavailable",),
            runtime_factory=unexpected_factory,
        )
        try:
            producer.create("capability-bound", str(tmp_path))
            producer.push(
                "capability-bound",
                InboxItem("input", "user", {"content": "go"}),
            )

            assert "deployment:unavailable" not in consumer._worker.support.capabilities
            with pytest.raises(UnsupportedWorkerCapabilities, match="deployment:unavailable"):
                consumer._static_registration(
                    "capability-bound",
                    agent,
                    config,
                    None,
                    handler_version="1",
                )
            assert "deployment:unavailable" not in consumer._worker.support.capabilities
            assert (
                consumer.store.claim_runnable(
                    owner="incompatible-worker",
                    ttl_ms=10_000,
                    limit=1,
                    work_kinds=("drive",),
                    projection_consumers=(),
                )
                == ()
            )
            _, epoch, remaining_ms = consumer.store.lease_status("capability-bound")
            assert epoch == 0 and remaining_ms == 0

            compatible._static_registration(
                "capability-bound",
                agent,
                config,
                None,
                handler_version="1",
            )
            assert "deployment:unavailable" in compatible._worker.support.capabilities
            claimed = compatible.store.claim_runnable(
                owner="compatible-worker",
                ttl_ms=10_000,
                limit=1,
                work_kinds=("drive",),
                projection_consumers=(),
            )
            assert len(claimed) == 1 and claimed[0].lease is not None
            assert claimed[0].lease.session_id == "capability-bound"
            assert compatible.store.release(claimed[0].lease)

            assert factory_calls == []
            assert model_calls == []
            assert tool_calls == []
        finally:
            producer.close()
            consumer.close()
            compatible.close()


def test_persisted_bootstrap_level_never_raises_worker_feature_level(tmp_path: Path) -> None:
    model_calls: list[object] = []
    tool_calls: list[object] = []
    factory_calls: list[object] = []

    @function_tool
    def forbidden_tool() -> str:
        tool_calls.append(object())
        return "unexpected"

    agent = Agent("format-bound", "Answer.", model="m", tools=[forbidden_tool])
    provider = ScriptedModelProvider.from_callback(
        "scripted",
        "m",
        lambda request: model_calls.append(request) or LLMResponse("unexpected"),
    )
    config = RunConfig(model_provider=provider)
    trusted = static_session_requirements(
        agent,
        config,
        resolved=None,
        handler_version="1",
    )
    raised_bootstrap = SessionRequirements(
        3,
        trusted.handler_version,
        trusted.capabilities,
    )

    key_provider = LocalKeyProvider(b"l" * 32)
    database = tmp_path / "format-bound.sqlite3"
    with ExitStack() as stack:
        producer_store = stack.enter_context(SQLiteStore.standalone(str(database)))
        producer_store.configure_input_admission(allow_unsigned=True)
        producer_store.install_schema()
        producer_store._rows("UPDATE sk_settings SET write_level=3 WHERE singleton=1")
        consumer_store = stack.enter_context(SQLiteStore.standalone(str(database)))
        consumer_store.configure_input_admission(allow_unsigned=True)
        explicit_store = stack.enter_context(SQLiteStore.standalone(str(database)))
        explicit_store.configure_input_admission(allow_unsigned=True)

        def unexpected_factory(*_args):
            factory_calls.append(object())
            raise AssertionError("incompatible format must not construct a Runtime")

        producer = SessionDriver(
            store=producer_store,
            content_key_provider=key_provider,
            worker_level=3,
        )
        producer._configure_static_worker(
            level=3,
            capabilities=static_worker_supports(agent, config, resolved=None),
            requirements=raised_bootstrap,
        )
        consumer = SessionDriver(
            store=consumer_store,
            content_key_provider=key_provider,
            worker_level=2,
            runtime_factory=unexpected_factory,
        )
        explicit = SessionDriver(
            store=explicit_store,
            content_key_provider=key_provider,
            worker_level=3,
            runtime_factory=unexpected_factory,
        )
        try:
            producer.create("format-bound", str(tmp_path))
            producer_store.configure_reader(level=3)
            created = producer.store.read("format-bound", limit=1).records[0].record
            assert created.feature_level == 2
            assert created._payload["bootstrap"]["level"] == 3
            producer.push(
                "format-bound",
                InboxItem("format-3-input", "user", {"content": "go"}),
            )

            assert consumer._worker.support.level == 2
            assert consumer._worker.support.handler_versions == ("1",)
            consumer_store.configure_reader(level=3)
            with pytest.raises(UnsupportedFeatureLevel, match="level 3"):
                consumer._static_registration(
                    "format-bound",
                    agent,
                    config,
                    None,
                    handler_version="1",
                )
            assert consumer._worker.support.level == 2
            assert consumer._worker.support.handler_versions == ("1",)
            assert (
                consumer.store.claim_runnable(
                    owner="unsupported-format-worker",
                    ttl_ms=10_000,
                    limit=1,
                    work_kinds=("drive",),
                    projection_consumers=(),
                )
                == ()
            )
            _, epoch, remaining_ms = consumer.store.lease_status("format-bound")
            assert epoch == 0 and remaining_ms == 0

            explicit_store.configure_reader(level=3)
            explicit._static_registration(
                "format-bound",
                agent,
                config,
                None,
                handler_version="1",
            )
            assert explicit._worker.support.level == 3
            claimed = explicit.store.claim_runnable(
                owner="explicit-level-three-worker",
                ttl_ms=10_000,
                limit=1,
                work_kinds=("drive",),
                projection_consumers=(),
            )
            assert len(claimed) == 1 and claimed[0].lease is not None
            assert explicit.store.release(claimed[0].lease)

            assert factory_calls == []
            assert model_calls == []
            assert tool_calls == []
        finally:
            producer.close()
            consumer.close()
            explicit.close()


def test_static_requirements_resolve_model_without_constructing_client() -> None:
    class ResolveOnlyProvider:
        def __init__(self) -> None:
            self._delegate = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
            self.allow_client = False
            self.client_calls = 0

        def resolve(self, model):
            return self._delegate.resolve(model)

        def client(self, resolved):
            self.client_calls += 1
            if not self.allow_client:
                raise AssertionError("static support must not construct an execution client")
            return self._delegate.client(resolved)

        def default_settings(self, resolved):
            return self._delegate.default_settings(resolved)

        def default_model_ref(self):
            return self._delegate.default_model_ref()

    provider = ResolveOnlyProvider()
    agent = Agent("static-model", "Answer.", model="m")
    config = RunConfig(model_provider=cast(Any, provider))
    requirements = static_session_requirements(
        agent,
        config,
        resolved=None,
        handler_version="1",
    )
    supports = static_worker_supports(agent, config, resolved=None)
    assert requirements.level == 2
    assert supports
    assert provider.client_calls == 0

    driver = SessionDriver()
    try:
        driver.create("static-model", ".")
        handle = driver.start(
            "static-model",
            agent,
            config,
            "go",
            autostart=False,
        )
        assert provider.client_calls == 0
        provider.allow_client = True
        handle.start()
        assert handle.result(timeout=8).final_output == "done"
        assert provider.client_calls > 0
    finally:
        driver.close()


@pytest.mark.parametrize("valid", [False, True])
def test_runtime_factory_binds_host_admission_to_worker_transaction(tmp_path: Path, valid: bool) -> None:
    key_provider = LocalKeyProvider(b"w" * 32)
    callback_threads: list[str] = []
    holder: dict[str, SessionDriver] = {}

    with SQLiteStore.standalone(str(tmp_path / "admission.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        store._rows("CREATE TABLE host_admission (marker TEXT NOT NULL)")

        def runtime_factory(drive_store, agent, config, task, session_id):
            adapter = _HostAdapter(drive_store)

            def admit(plan):
                callback_threads.append(current_thread().name)
                adapter.execute("INSERT INTO host_admission(marker) VALUES ('admitted')")
                child = ChildSession(
                    SessionSpec(
                        "child",
                        holder["driver"]._session_principal(plan.session_id).to_dict(),
                        cast(Any, str(tmp_path / "child")),
                        attributes={},
                    ),
                    "child work",
                )
                return [child] if valid else [child, object()]

            runtime = holder["driver"].runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="handler/v2",
            )
            runtime.children["child_task"] = admit
            return runtime

        driver = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=2,
            handler_versions=("handler/v2",),
            capabilities=("toolset:test",),
        )
        holder["driver"] = driver
        try:
            provider = ScriptedModelProvider.new(
                "scripted",
                "m",
                [
                    LLMResponse("", [ToolCall("child-call", "child_task", {})]),
                    LLMResponse("child done"),
                    LLMResponse("parent done"),
                ],
            )
            agent = Agent("parent", "Use the child tool.", model="m", tools=[child_task])
            config = RunConfig(model_provider=provider)
            expected_supports = static_worker_supports(
                agent,
                config,
                extra_capabilities=("toolset:test",),
            )
            driver.create("parent", str(tmp_path), {"seed": {"messages": [], "shared_state": {}}})
            handle = driver.start(
                "parent",
                agent,
                config,
                "go",
                handler_version="handler/v2",
            )
            assert driver._worker.support.handler_versions == ("handler/v2",)
            assert driver._worker.support.capabilities == expected_supports
            assert "toolset:test" in expected_supports
            if valid:
                assert handle.result().final_output == "parent done"
            else:
                with pytest.raises(InvalidChildBatch):
                    handle.result()

            records = store.read("parent", limit=100).records
            business = store._rows("SELECT marker FROM host_admission")
            child_rows = store._rows("SELECT session_id FROM sk_session WHERE session_id='child'")
            parent_parked = [row for row in records if row.record.kind == "op_parked"]
            assert len(callback_threads) == 1 and callback_threads[0].startswith("session-local-worker-drive")
            if valid:
                assert business == [("admitted",)]
                assert child_rows == [("child",)]
                assert parent_parked
            else:
                assert business == []
                assert child_rows == []
                assert parent_parked == []
        finally:
            driver.close()


def test_host_bound_runtime_is_rejected_without_runtime_factory() -> None:
    driver = SessionDriver()
    try:
        driver.create("session", ".")
        provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
        runtime = driver.runtime(
            Agent("agent", "Answer.", model="m"),
            RunConfig(model_provider=provider),
        )
        runtime.children["child_task"] = cast(Any, lambda _plan: None)

        with pytest.raises(MissingHostBinding, match="runtime_factory"):
            driver._register_runtime("session", runtime)

        assert driver.store.peek_inbox("session") == ()
    finally:
        driver.close()


def test_public_session_driver_passes_preclaimed_lease_to_model_drive(monkeypatch, tmp_path: Path) -> None:
    provider_threads: list[str] = []
    invoked = Event()
    acquire_calls: list[str] = []

    def model_call(_request):
        provider_threads.append(current_thread().name)
        invoked.set()
        return LLMResponse("done")

    def unexpected_acquire(self, session_id, **_kwargs):
        acquire_calls.append(session_id)
        raise AssertionError("preclaimed worker lease was acquired a second time")

    monkeypatch.setattr(CoreSQLiteStore, "acquire", unexpected_acquire)
    with SQLiteStore.standalone(str(tmp_path / "lease.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"w" * 32))
        try:
            driver.create("session", str(tmp_path))
            provider = ScriptedModelProvider.new("scripted", "m", cast(Any, [model_call]))
            result = driver.start(
                "session",
                Agent("agent", "Answer.", model="m"),
                RunConfig(model_provider=provider),
                "go",
            ).result()
            assert result.final_output == "done"
            assert invoked.is_set()
            assert provider_threads == ["session-external-call"]
            assert acquire_calls == []
        finally:
            driver.close()


def test_worker_support_includes_statically_reachable_agent_tools(tmp_path: Path) -> None:
    @function_tool
    def child_only_tool() -> str:
        return "child"

    child = Agent("child", "Run child work.", model="m", tools=[child_only_tool])
    parent = Agent("parent", "Delegate child work.", model="m", tools=[child.as_tool()])
    config = RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", []))

    with SQLiteStore.standalone(str(tmp_path / "support.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"s" * 32))
        try:
            driver.create("parent", str(tmp_path))
            driver.start("parent", parent, config, "go", autostart=False)
            parent_supports = set(static_worker_supports(parent, config))
            child_supports = set(static_worker_supports(child, config))
            assert child_supports - parent_supports
            assert driver._worker.support.level == 2
            assert set(driver._worker.support.capabilities) == parent_supports | child_supports
        finally:
            driver.close()


def test_temporary_driver_keeps_wal_and_full_synchronous(tmp_path: Path) -> None:
    del tmp_path
    driver = SessionDriver()
    try:
        with driver._open_drive_store() as store:
            synchronous = store.connection.execute("PRAGMA synchronous").fetchone()[0]
            journal_mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert synchronous == 2
        assert journal_mode.lower() == "wal"
    finally:
        driver.close()


def test_drive_connection_does_not_reuse_heartbeat_connection_factory(monkeypatch) -> None:
    driver = SessionDriver()
    try:
        root = driver._root_store

        def unexpected_heartbeat_store():
            raise AssertionError("the drive connection must not come from heartbeat_store()")

        monkeypatch.setattr(root, "heartbeat_store", unexpected_heartbeat_store)
        with driver._open_drive_store() as drive_store:
            assert drive_store is not root
            assert cast(Any, drive_store).connection is not cast(Any, root).connection
    finally:
        driver.close()


def test_postgres_uses_normal_drive_store_factory(monkeypatch) -> None:
    class DriveManager:
        def __enter__(self):
            return "drive-store"

        def __exit__(self, *_exc_info) -> None:
            return None

    class FakePostgresStore:
        def drive_store_factory(self):
            return DriveManager

        def _heartbeat_connection(self, _conninfo):
            raise AssertionError("heartbeat connections are keeper-only")

    monkeypatch.setattr(surface_module, "CorePostgresStore", FakePostgresStore)
    driver = object.__new__(SessionDriver)
    driver._root_store = FakePostgresStore()
    driver._drive_store_factory = driver._root_store.drive_store_factory()
    with driver._open_default_drive_store() as store:
        assert store == "drive-store"


def test_after_commit_watermark_uses_batch_head(tmp_path: Path) -> None:
    provider = fixture_key_provider()
    database = tmp_path / "watermark.sqlite"
    with SQLiteStore.standalone(str(database)) as root:
        root.configure_input_admission(allow_unsigned=True)
        root.install_schema()
        store = wrap_fixture_store(root, provider)
        create_fixture_session(store, "watermark", consumers=())

        driver = SessionDriver(store=root, content_key_provider=provider)
        try:
            scripted = ScriptedModelProvider.new("scripted", "m", [])
            registration_runtime = driver.runtime(
                Agent("watermark", "Answer.", model="m"),
                RunConfig(model_provider=scripted),
                session_id="watermark",
            )
            driver._register_runtime("watermark", registration_runtime, armed=True)

            watermark_cursor = RunHandle(driver, "watermark", "t", None, "events")
            watermarks: list[int] = []
            cast(Any, watermark_cursor)._add_stream_event = lambda _event, watermark: watermarks.append(watermark)
            driver.handles.append(watermark_cursor)
            runtime = driver._build_runtime_for_claim(driver.store, "watermark")

            lease = store.acquire("watermark", owner="fixture/watermark", ttl_ms=10_000)
            assert lease is not None
            batch = (
                record("turn_started", sid="watermark"),
                record("usage_observed", sid="watermark", observation=1, meter_id="watermark-1"),
                record("usage_observed", sid="watermark", observation=1, meter_id="watermark-2"),
            )
            with store.atomic() as tx:
                receipt = tx.append(
                    "watermark",
                    lease=lease,
                    expected_seq=store.read("watermark").head_seq,
                    commit_id="watermark-batch",
                    records=batch,
                )
            assert store.release(lease)
            runtime.hook("after_commit", None)
            assert runtime.config.stream is not None
            runtime.config.stream(
                AssistantDeltaEvent(
                    run_id="t",
                    trace_id="t",
                    session_id="watermark",
                    delta="live",
                )
            )
            assert watermarks == [receipt.head_seq]
        finally:
            driver.close()


def test_driver_does_not_mutate_unowned_projection_consumers(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "consumers.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"c" * 32),
        )
        try:
            driver.create("consumers", str(tmp_path))
            before = store._rows(
                "SELECT last_seq,project_retry_at_ms FROM sk_consumer WHERE session_id='consumers' AND consumer='app_server'"
            )[0]
            driver._project_worker_session(driver.store, "consumers", "app_server")
            after = store._rows(
                "SELECT last_seq,project_retry_at_ms FROM sk_consumer WHERE session_id='consumers' AND consumer='app_server'"
            )[0]
            assert after == before
        finally:
            driver.close()


def test_runner_start_handle_close_releases_worker_and_temporary_store() -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which color?"})])],
    )
    handle = Runner.start(
        Agent("owned", "Ask once.", model="m"),
        "choose",
        run_config=RunConfig(model_provider=provider),
    )
    owner = handle.kernel
    assert owner.path is not None
    database = Path(owner.path)
    worker_thread = owner._worker._thread
    try:
        assert handle.result().status is AgentStatus.WAIT_USER
        assert worker_thread is not None and worker_thread.is_alive()
        assert database.exists()
    finally:
        handle.close()
    _wait_for(lambda: not owner._worker.running and not database.parent.exists())
    assert worker_thread is not None and not worker_thread.is_alive()


def test_explicit_open_store_factory_is_used_for_worker_drive(tmp_path: Path) -> None:
    database = tmp_path / "custom-drive.sqlite"
    opened = 0
    with SQLiteStore.standalone(str(database)) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()

        def open_store():
            nonlocal opened
            opened += 1
            return configured_open_store(database)

        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"w" * 32),
            open_store=open_store,
        )
        try:
            driver.create("session", str(tmp_path))
            provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
            result = driver.start(
                "session",
                Agent("agent", "Answer.", model="m"),
                RunConfig(model_provider=provider),
                "go",
            ).result()
            assert result.final_output == "done"
            assert opened > 0
        finally:
            driver.close()


def test_session_driver_and_handle_have_one_worker_driven_structure() -> None:
    from vv_agent.run_handle import RunHandle

    assert not hasattr(surface_module, "ContextVar")
    assert "_HydratedSessionStore" not in surface_module.__dict__
    for name in (
        "_drive_tree",
        "_run_on_worker",
        "_schedule_child",
        "_schedule_background",
        "_deliver_child",
        "_stop_for_close",
    ):
        assert not hasattr(RunHandle, name)
    for name in ("_condition", "_thread", "_worker_managed", "_drive_tree"):
        assert name not in RunHandle.__annotations__
    driver = SessionDriver()
    try:
        driver.create("structure", ".")
        handle = driver.start(
            "structure",
            Agent("agent", "Answer.", model="m"),
            RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", [])),
            "queued",
            autostart=False,
        )
        for name in (
            "_condition",
            "_thread",
            "_worker_managed",
            "_stream_lock",
            "_worker_active",
            "_worker_done",
            "_worker_condition",
            "_drive_tree",
            "_seen",
        ):
            assert not hasattr(handle, name)
        for name in ("_worker_handles", "_worker_handles_lock", "_background", "_background_lock"):
            assert not hasattr(driver, name)
        assert not hasattr(driver, "_runtime_lock")
        assert surface_module._RuntimeRegistration.__dataclass_params__.frozen
        assert not hasattr(driver, "_use_drive_store")
        assert not hasattr(driver, "_hydrate_drive_store")
    finally:
        driver.close()


def test_second_start_is_rejected_while_first_submission_is_unarmed() -> None:
    driver = SessionDriver()
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    try:
        driver.create("session", ".")
        first = driver.start(
            "session",
            Agent("first", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "first",
            input_id="first",
            autostart=False,
        )
        with pytest.raises(Conflict, match="active RunHandle"):
            driver.start(
                "session",
                Agent("second", "Answer.", model="m"),
                RunConfig(model_provider=provider),
                "second",
                input_id="second",
                autostart=False,
            )
        assert driver.store.peek_inbox("session") == ()
        assert not first.done()
        first.start()
        _wait_for(lambda: first.done())
        assert first.done()
    finally:
        driver.close()
    assert first.done()


def test_start_publishes_static_registration_without_caller_runtime(monkeypatch) -> None:
    driver = SessionDriver()
    runtime_threads: list[str] = []
    original_runtime = driver.runtime

    def runtime(*args, **kwargs):
        runtime_threads.append(current_thread().name)
        return original_runtime(*args, **kwargs)

    monkeypatch.setattr(driver, "runtime", runtime)
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    try:
        driver.create("publish", ".")
        handle = driver.start(
            "publish",
            Agent("publisher", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "go",
            autostart=False,
        )
        assert runtime_threads == []
        handle.start()
        assert handle.result(timeout=8).final_output == "done"
        assert len(runtime_threads) == 1 and runtime_threads[0].startswith("session-local-worker-drive")
    finally:
        driver.close()


def test_runner_publishes_bootstrap_and_follow_up_requirements() -> None:
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("first"), LLMResponse("second")])
    agent = Agent("routed", "Answer.", model="m")
    config = RunConfig(model_provider=provider)
    handle = Runner.start(agent, "one", run_config=config)
    driver = handle.kernel
    try:
        assert handle.result(timeout=8).final_output == "first"
        created = driver.store.read(handle.session_id, limit=1).records[0].record
        assert created.feature_level == 2
        bootstrap = created.payload["bootstrap"]
        expected_bootstrap = static_session_requirements(
            agent,
            config,
            resolved=None,
            handler_version="1",
        )
        expected_supports = static_worker_supports(agent, config, resolved=None)
        assert bootstrap == expected_bootstrap.to_dict()
        assert driver._worker.support.capabilities == expected_supports

        handle.follow_up("two")
        second = driver.start(handle.session_id, agent, config, None)
        assert second.result(timeout=8).final_output == "second"
        driver.push(
            handle.session_id,
            InboxItem(
                "select-model",
                "model_selection",
                {"selection": {"model": "m", "route": {}}},
            ),
        )
        _wait_for(
            lambda: any(
                stored.record.kind == "input_applied" and stored.record.payload["input"]["input_id"] == "select-model"
                for stored in driver.store.read(handle.session_id, limit=100).records
            )
        )
        applied = [
            stored.record.payload["input"]
            for stored in driver.store.read(handle.session_id, limit=100).records
            if stored.record.kind == "input_applied"
        ]
        assert [item["kind"] for item in applied] == ["user", "follow_up", "model_selection"]
        assert all(item["feature_level"] == 2 for item in applied)
        assert [item["requirements"] for item in applied[:2]] == [bootstrap, bootstrap]
        expected_selection = static_model_selection_requirements(
            {"model": "m", "route": {}},
            resolve_model=lambda selection: resolve_model_selection(provider, selection),
            handler_version="1",
        )
        assert applied[2]["requirements"] == expected_selection.to_dict()
        assert all(capability.startswith("model:") for capability in expected_selection.capabilities)
        assert set(bootstrap["capabilities"]) <= set(driver._worker.support.capabilities)
        assert driver._worker.support.level == 2
        assert "1" in driver._worker.support.handler_versions
    finally:
        driver.close()


def test_custom_factory_extra_capabilities_feed_l2_bootstrap_and_worker(
    tmp_path: Path,
) -> None:
    key_provider = LocalKeyProvider(b"x" * 32)
    factory_threads: list[str] = []
    holder: dict[str, SessionDriver] = {}

    with SQLiteStore.standalone(str(tmp_path / "custom-factory.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")

        def runtime_factory(drive_store, agent, config, task, session_id):
            factory_threads.append(current_thread().name)
            return holder["driver"].runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="handler/v2",
            )

        driver = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=2,
            handler_versions=("handler/v2",),
            capabilities=("family:custom",),
        )
        holder["driver"] = driver
        try:
            provider = ScriptedModelProvider.new(
                "scripted",
                "m",
                [LLMResponse("done")],
            )
            agent = Agent("custom", "Answer.", model="m")
            config = RunConfig(model_provider=provider)
            expected_bootstrap = static_session_requirements(
                agent,
                config,
                resolved=None,
                handler_version="handler/v2",
                extra_capabilities=("family:custom",),
            )
            expected_supports = static_worker_supports(
                agent,
                config,
                resolved=None,
                extra_capabilities=("family:custom",),
            )
            driver._configure_static_worker(
                level=expected_bootstrap.level,
                capabilities=expected_supports,
                requirements=expected_bootstrap,
            )
            driver.create("custom", str(tmp_path))
            handle = driver.start(
                "custom",
                agent,
                config,
                "go",
                handler_version="handler/v2",
            )
            assert handle.result(timeout=8).final_output == "done"

            created = driver.store.read("custom", limit=1).records[0].record
            applied = next(
                stored.record.payload["input"]
                for stored in driver.store.read("custom", limit=100).records
                if stored.record.kind == "input_applied"
            )
            assert created.payload["bootstrap"] == expected_bootstrap.to_dict()
            assert applied["requirements"] == expected_bootstrap.to_dict()
            assert "family:custom" in expected_bootstrap.capabilities
            assert driver._worker.support.capabilities == expected_supports
            assert len(factory_threads) == 1 and factory_threads[0].startswith("session-local-worker-drive")
        finally:
            driver.close()


def test_closing_unstarted_handle_releases_runtime_reservation() -> None:
    driver = SessionDriver()
    try:
        driver.create("session", ".")
        provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
        abandoned = driver.start(
            "session",
            Agent("abandoned", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "unused",
            input_id="unused",
            autostart=False,
        )
        abandoned.close()
        assert driver.store.peek_inbox("session") == ()

        handle = driver.start(
            "session",
            Agent("replacement", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "go",
            input_id="replacement",
        )
        assert handle.result(timeout=8).final_output == "done"
    finally:
        driver.close()


def test_concurrent_starts_use_one_sql_admission(store, database, tmp_path: Path) -> None:
    del database
    driver = SessionDriver(
        store=store,
        content_key_provider=LocalKeyProvider(b"r" * 32),
    )
    driver.create("race", str(tmp_path))
    barrier = Barrier(3)
    results: dict[str, str] = {}
    errors: dict[str, BaseException] = {}

    def launch(label: str, input_id: str) -> None:
        provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse(label)])
        barrier.wait()
        try:
            handle = driver.start(
                "race",
                Agent(label, "Answer.", model="m"),
                RunConfig(model_provider=provider),
                label,
                input_id=input_id,
            )
            results[label] = handle.result(timeout=8).final_output or ""
        except BaseException as exc:
            errors[label] = exc

    threads = [
        Thread(target=launch, args=("first", "first-input"), name="start-first"),
        Thread(target=launch, args=("second", "second-input"), name="start-second"),
    ]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(next(iter(errors.values())), Conflict)
        winner = next(iter(results))
        state, _, _ = driver.store.read_state("race")
        applied = [
            record._payload["input"] for record in state.applied_inputs.values() if record._payload["input"]["kind"] == "user"
        ]
        assert [item["input_id"] for item in applied] == [f"{winner}-input"]
        registration = driver._runtime_registration("race")
        assert registration.armed
        assert registration.agent.name == winner
    finally:
        driver.close()


def test_observer_projection_uses_incremental_record_cursor(monkeypatch) -> None:
    monkeypatch.setattr(surface_module, "_EVENT_PAGE_SIZE", 2)
    observer_events = []
    observed_pages: list[int] = []
    original_project_page = surface_module._project_record_page

    def project_page(store, sid, *, after_seq, through_seq=None, run_id=None):
        observed_pages.append(after_seq)
        return original_project_page(
            store,
            sid,
            after_seq=after_seq,
            through_seq=through_seq,
            run_id=run_id,
        )

    monkeypatch.setattr(surface_module, "_project_record_page", project_page)
    driver = SessionDriver()
    try:
        provider = ScriptedModelProvider.new(
            "scripted",
            "m",
            [LLMResponse("first"), LLMResponse("second")],
        )
        driver.create("observer", ".")
        handle = driver.start(
            "observer",
            Agent("observer", "Answer.", model="m"),
            RunConfig(model_provider=provider, stream=observer_events.append),
            "one",
        )
        assert handle.result().final_output == "first"
        first_head = driver._read_state("observer")[1][-1].seq
        assert driver._observer_cursors["observer"] == first_head

        observed_pages.clear()
        observer_events.clear()
        handle.follow_up("two")
        _wait_for(
            lambda: (
                len((state := driver._read_state("observer")[0]).turns) == 2 and all(turn.ended for turn in state.turns.values())
            )
        )
        _wait_for(lambda: observer_events)
        second_turn_id = next(reversed(driver._read_state("observer")[0].turns))
        assert len(observed_pages) > 1
        assert all(after_seq >= first_head for after_seq in observed_pages)
        assert {event.run_id for event in observer_events} == {second_turn_id}
        event_ids = [event.event_id for event in observer_events]
        assert len(event_ids) == len(set(event_ids))
    finally:
        driver.close()


def test_completed_handle_wakes_worker_for_follow_up() -> None:
    provider_threads: list[str] = []

    def response(value: str):
        def call(_request):
            provider_threads.append(current_thread().name)
            return LLMResponse(value)

        return call

    driver = SessionDriver()
    try:
        provider = ScriptedModelProvider.new("scripted", "m", [response("first"), response("second")])
        driver.create("session", ".")
        handle = driver.start(
            "session",
            Agent("agent", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "one",
        )
        assert handle.result().final_output == "first"
        handle.follow_up("two")
        _wait_for(
            lambda: (
                len(driver._read_state("session")[0].turns) == 2
                and all(turn.ended for turn in driver._read_state("session")[0].turns.values())
            )
        )
        assert driver.handles == [handle]
        assert provider_threads == ["session-external-call", "session-external-call"]
    finally:
        driver.close()


def test_level_one_start_and_follow_up_keep_level_one_and_lock_settings_first(store, monkeypatch) -> None:
    driver = SessionDriver(
        store=store,
        content_key_provider=fixture_key_provider(),
        worker_level=1,
        handler_versions=("1",),
    )
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [LLMResponse("first"), LLMResponse("second")],
    )
    lock_events: list[str] = []
    original_ceiling = store._write_ceiling
    original_lock = store._lock

    def write_ceiling():
        lock_events.append("write_ceiling")
        return original_ceiling()

    def lock(session_id):
        lock_events.append("lock")
        return original_lock(session_id)

    monkeypatch.setattr(store, "_write_ceiling", write_ceiling)
    monkeypatch.setattr(store, "_lock", lock)
    try:
        driver.create("level-one", ".")
        store._rows("UPDATE sk_settings SET write_level=1 WHERE singleton=1")
        handle = driver.start(
            "level-one",
            Agent("level-one", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "one",
            input_id="start",
        )

        assert lock_events[0] == "write_ceiling", lock_events
        assert handle.result(timeout=8).final_output == "first"
        handle.follow_up("two")
        _wait_for(
            lambda: (
                len((state := driver._read_state("level-one")[0]).turns) == 2 and all(turn.ended for turn in state.turns.values())
            )
        )

        applied = [
            stored.record._payload["input"]
            for stored in driver.store.read("level-one", limit=100).records
            if stored.record.kind == "input_applied"
        ]
        assert [(item["kind"], item["feature_level"]) for item in applied] == [
            ("user", 1),
            ("follow_up", 1),
        ]
        assert all("requirements" not in item for item in applied)
    finally:
        driver.close()


def test_read_state_fails_closed_through_entire_deletion_fence(store) -> None:
    key_provider = fixture_key_provider()
    try:
        wrapped = wrap_fixture_store(store, key_provider)
        create_fixture_session(wrapped, "deletion-fence")
        assert store.read_state("deletion-fence")[1]

        actor = Principal(
            store.tenant_id,
            "fixture/deleter",
            "user",
            ContentRef.from_dict(content_ref("deletion-actor")),
        )
        deletion = SessionDeletionStore(store, cast(Any, None), allow_unsigned=True)

        store.begin_forget(
            "deletion-fence",
            actor,
            bound_signer=None,
            verifier=None,
            allow_unsigned=True,
        )
        with pytest.raises(SessionNotLive, match="fenced for deletion"):
            store.read_state("deletion-fence")
        assert deletion.read_skeleton(store.tenant_id, "deletion-fence").tombstone is not None

        store.complete_forget("deletion-fence")
        with pytest.raises(SessionNotLive, match="fenced for deletion"):
            store.read_state("deletion-fence")
        skeleton = deletion.read_skeleton(store.tenant_id, "deletion-fence")
        assert skeleton.tombstone is not None
        assert skeleton.tombstone.state.value == "complete"

        store.begin_purge(
            "deletion-fence",
            actor,
            bound_signer=None,
            verifier=None,
            allow_unsigned=True,
        )
        with pytest.raises(SessionNotLive, match="fenced for deletion"):
            store.read_state("deletion-fence")
    finally:
        key_provider.close()


def test_recovery_defers_transient_key_unavailable_then_completes(store, monkeypatch) -> None:
    key_provider = fixture_key_provider()
    driver = SessionDriver(store=store, content_key_provider=key_provider, worker_level=1, handler_versions=("1",))
    model_calls: list[object] = []
    unwrap_threads: list[str] = []
    fail_next_worker_unwrap = True
    original_unwrap = key_provider.unwrap
    provider = ScriptedModelProvider.from_callback(
        "scripted",
        "m",
        lambda request: model_calls.append(request) or LLMResponse("recovered"),
    )

    def transient_unwrap(tenant_id, session_id, wrapped_key):
        nonlocal fail_next_worker_unwrap
        if fail_next_worker_unwrap and current_thread().name.startswith("session-local-worker"):
            fail_next_worker_unwrap = False
            unwrap_threads.append(current_thread().name)
            raise BackendKeyUnavailable("temporary key-provider outage")
        return original_unwrap(tenant_id, session_id, wrapped_key)

    monkeypatch.setattr(key_provider, "unwrap", transient_unwrap)
    try:
        driver.create("transient-key", ".")
        handle = driver.start(
            "transient-key",
            Agent("transient-key", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "go",
            autostart=False,
        )
        handle.start()
        assert handle.result(timeout=8).final_output == "recovered"

        state, records, _ = driver._read_state("transient-key")
        assert state.active_turn_id is None
        assert len(model_calls) == 1
        assert unwrap_threads and unwrap_threads[0].startswith("session-local-worker")
        assert "transient-key" not in driver._errors
        assert [row.record.payload["status"] for row in records if row.record.kind == "turn_ended"] == ["completed"]
        assert not [row.record for row in records if row.record.kind == "turn_failed"]
    finally:
        driver.close()
        key_provider.close()


def test_two_handles_are_independent_cursor_subscribers() -> None:
    observer_threads: list[str] = []

    def observe(_event) -> None:
        observer_threads.append(current_thread().name)

    driver = SessionDriver()
    try:
        provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
        agent = Agent("agent", "Answer.", model="m")
        config = RunConfig(model_provider=provider, stream=observe)
        driver.create("session", ".")
        first_handle = driver.start("session", agent, config, "go")
        second_handle = driver.start("session", agent, config, None)
        first_events = first_handle.events()
        second_events = second_handle.events()
        first = first_handle.result()
        second = second_handle.result()
        assert first.final_output == second.final_output == "done"
        assert [event.to_dict() for event in first_events] == [event.to_dict() for event in second_events]
        assert observer_threads and all(name.startswith("session-local-worker") for name in observer_threads)
        assert current_thread().name not in observer_threads
    finally:
        driver.close()


def test_handle_runtime_is_a_connection_free_projection_view() -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which color?"})])],
    )
    handle = Runner.start(
        Agent("waiting", "Ask once.", model="m"),
        "choose",
        run_config=RunConfig(model_provider=provider),
    )
    try:
        first = handle.result()
        view = handle.runtime
        wakes = []
        view.wake = wakes.append
        view.wake(handle.session_id)
        second = handle.result()
        assert wakes == [handle.session_id]
        registration = handle.kernel._runtime_registration(handle.session_id)
        assert view.agent is registration.agent
        assert view.config is registration.config
        assert registration.resolved is None
        assert view.resolved.model_id == "m"
        assert view.handler_version == registration.handler_version
        assert second.to_dict() == first.to_dict()
        assert not hasattr(view, "heartbeat_store")
    finally:
        handle.kernel.close()


def test_close_releases_driver_owned_temporary_key_provider() -> None:
    driver = SessionDriver()
    provider = driver.content_key_provider
    assert driver._temporary_content_key_state is not None
    assert driver._temporary_store_directory is not None
    key_state = Path(driver._temporary_content_key_state.name)
    store_directory = Path(driver._temporary_store_directory.name)
    assert key_state.is_dir()
    assert store_directory.is_dir()

    driver.close()

    assert cast(LocalKeyProvider, provider)._connection is None
    assert not key_state.exists()
    assert not store_directory.exists()


@pytest.mark.parametrize(
    "bound_by_root_store",
    [False, True],
    ids=["injected-provider", "root-bound-provider"],
)
def test_close_does_not_release_borrowed_key_provider(tmp_path: Path, bound_by_root_store: bool) -> None:
    provider = LocalKeyProvider(b"i" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "borrowed-provider.sqlite")) as store:
            store.install_schema()
            if bound_by_root_store:
                bind_content_key_provider(store, provider)
            driver = SessionDriver(
                store=store,
                content_key_provider=None if bound_by_root_store else provider,
            )
            try:
                assert driver.content_key_provider is provider
                assert driver._temporary_content_key_state is None
            finally:
                driver.close()
            assert provider._connection is not None
    finally:
        provider.close()


def test_dropping_unconsumed_stream_closes_owned_driver(monkeypatch) -> None:
    created: list[Path] = []
    original_temporary_directory = surface_module.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        value = original_temporary_directory(*args, **kwargs)
        created.append(Path(value.name))
        return value

    monkeypatch.setattr(surface_module, "TemporaryDirectory", temporary_directory)
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    stream = Runner.stream_sync(
        Agent("agent", "Answer.", model="m"),
        "go",
        run_config=RunConfig(model_provider=provider),
    )
    assert created
    del stream
    collect()
    _wait_for(lambda: all(not path.exists() for path in created))


def test_control_runs_kernel_on_worker_thread(monkeypatch) -> None:
    from vv_agent.session import kernel as kernel_module

    original_drive = kernel_module.drive
    drive_threads: list[str] = []

    def drive(*args, **kwargs):
        drive_threads.append(current_thread().name)
        return original_drive(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "drive", drive)
    driver = SessionDriver()
    try:
        driver.create("session", ".")
        assert not current_thread().name.startswith("session-local-worker-drive")
        driver.control("session", "archive", "archive")
        assert len(drive_threads) == 1 and drive_threads[0].startswith("session-local-worker-drive")
        assert driver._read_state("session")[0].archived
    finally:
        driver.close()


def test_sync_runner_closes_worker_and_temporary_directories(monkeypatch) -> None:
    created: list[Path] = []
    original_temporary_directory = surface_module.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        value = original_temporary_directory(*args, **kwargs)
        created.append(Path(value.name))
        return value

    monkeypatch.setattr(surface_module, "TemporaryDirectory", temporary_directory)
    before = {
        thread.ident
        for thread in thread_enumerate()
        if thread.name == "session-local-worker" or thread.name.startswith("session-local-worker-drive")
    }
    agent = Agent("runner", "Answer.", model="m")
    for value in ("one", "two", "three"):
        provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse(value)])
        assert Runner.run_sync(agent, value, run_config=RunConfig(model_provider=provider)).final_output == value
        stream_provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse(value)])
        assert list(Runner.stream_sync(agent, value, run_config=RunConfig(model_provider=stream_provider)))

    after = {
        thread.ident
        for thread in thread_enumerate()
        if thread.name == "session-local-worker" or thread.name.startswith("session-local-worker-drive")
    }
    assert after == before
    assert created
    assert all(not path.exists() for path in created)


def test_sync_runner_retains_waiting_owner_until_resume(tmp_path: Path) -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which color?"})]),
        ],
    )
    runner = Runner.configured(
        RunConfig(
            workspace=tmp_path,
            model_provider=provider,
        )
    )
    waiting = runner.run_sync(Agent("agent", "Ask once, then answer.", model="m"), "choose")

    assert waiting.status is AgentStatus.WAIT_USER
    session_id = waiting.raw_result.session_id
    turn_id = waiting.raw_result.turn_id
    assert session_id is not None and turn_id is not None
    owner = waiting._session_driver
    assert owner is not None
    database = Path(owner.path)
    assert database.exists()
    resumed = runner.resume(session_id, turn_id)

    assert resumed.status is AgentStatus.WAIT_USER
    assert resumed._session_driver is owner
    assert database.exists()
    owner.close()
    assert not database.parent.exists()
