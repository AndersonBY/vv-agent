"""The real turn-start producer publishes definition and budget as S1 refs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_session_stored_record
from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.kernel._content import ContentRef, ContentSessionWriter
from vv_agent.kernel._content_publication import make_record_before_publication, publish_turn_started
from vv_agent.kernel._hydration import reference_digest
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.environments import EnvironmentCapabilities, IsolationMode, NetworkMode
from vv_agent.session.kernel import _Driver
from vv_agent.session.records import InboxItem, SessionSpec, digest
from vv_agent.session.runtime import (
    InvalidTrustedEnvironmentBinding,
    TrustedEnvironmentBinding,
    static_session_requirements,
    static_worker_supports,
)
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

from .conftest import open_store

MASTER_KEY = b"k" * 32


def _restricted_capabilities() -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id="environment/turn-producer",
        generation=7,
        isolation=IsolationMode.CONTAINER,
        network=NetworkMode.ALLOWLIST,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=("command.execute", "filesystem.workspace", "network.allowlist"),
        network_policy_ref_digest="1" * 64,
        isolation_policy_ref_digest="2" * 64,
    )


def _operational_policy() -> dict[str, str]:
    return {
        "network_policy_digest": canonical_json_sha256({"egress": ["api.example"]}),
        "isolation_policy_digest": canonical_json_sha256({"cpu_millis": 1000, "memory_mib": 256}),
    }


def _trusted_binding(
    capabilities: EnvironmentCapabilities | object | None = None,
    *,
    operational_policy: dict | None = None,
) -> TrustedEnvironmentBinding:
    return TrustedEnvironmentBinding(
        capabilities=_restricted_capabilities() if capabilities is None else capabilities,
        descriptor={"kind": "test-environment", "route": "sandbox-plaintext-marker"},
        operational_policy=_operational_policy() if operational_policy is None else operational_policy,
    )


def _producer_agent() -> Agent:
    return Agent("kernel", "Work.", model="scripted")


def _producer_config(steps: list[LLMResponse]) -> RunConfig:
    return RunConfig(
        model_provider=ScriptedModelProvider.new("scripted", "scripted", steps),
        workspace="/workspace",
    )


def _bootstrap_requirements(level: int) -> SessionRequirements:
    requirements = static_session_requirements(
        _producer_agent(),
        _producer_config([]),
        resolved=None,
        handler_version="1",
    )
    return SessionRequirements(level, requirements.handler_version, requirements.capabilities)


def _worker_supports() -> tuple[str, ...]:
    return static_worker_supports(
        _producer_agent(),
        _producer_config([]),
        resolved=None,
    )


def _create_bootstrapped_session(
    database: str | Path,
    provider: LocalKeyProvider,
    *,
    level: int,
    input_text: str | None = None,
) -> None:
    with open_store(database) as store:
        store.install_schema()
        store._rows("UPDATE sk_settings SET write_level=%s WHERE singleton=1", (level,))
        store.configure_reader(level=level)
        sessions = SessionDriver(store=store, content_key_provider=provider, worker_level=level)
        try:
            sessions._configure_static_worker(
                level=level,
                capabilities=_worker_supports(),
                requirements=_bootstrap_requirements(level),
            )
            sessions.create("s", "/workspace")
            if input_text is not None:
                sessions.push("s", InboxItem("i", "user", {"content": input_text}))
        finally:
            sessions.close()


@contextmanager
def _driver(
    path: Path,
    provider: LocalKeyProvider,
    *,
    attributes: dict | None = None,
    level: int = 1,
    bootstrap_level: int | None = None,
) -> Iterator[tuple]:
    agent = _producer_agent()
    config = _producer_config([])
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        if level == 3:
            store._rows("UPDATE sk_settings SET write_level=3 WHERE singleton=1")
            store.configure_reader(level=3)
            store.configure_worker(level=3, handler_versions=("1",), capabilities=())
        sessions = SessionDriver(store=store, content_key_provider=provider, worker_level=level)
        if bootstrap_level is not None:
            sessions._configure_static_worker(
                level=bootstrap_level,
                capabilities=_worker_supports(),
                requirements=_bootstrap_requirements(bootstrap_level),
            )
        sessions.create("s", "/workspace", attributes or {})
        sessions.push("s", InboxItem("i", "user", {"content": "do work"}))
        runtime = sessions.runtime(
            agent,
            config,
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="producer-test", ttl_ms=15_000)
        assert lease is not None
        scope = _Scope(lease, runtime, now_ms=store._now())
        yield store, sessions, runtime, _Driver(sessions.store, "s", runtime, scope)


def _start_turn(driver: _Driver) -> None:
    assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
    assert _recovery_phase.run(driver, inputs.start_turn(driver))


def _assert_turn_content(
    driver: _Driver,
    *,
    definition: ContentRef,
    budget: ContentRef,
    definition_digest: str,
) -> None:
    assert driver.hydrator_for(3).definition(definition).task.user_prompt == "do work"
    assert driver.hydrator_for(3).budget(budget) == RunBudgetLimits()
    assert definition_digest == reference_digest(definition)


def test_turn_started_is_ref_only_links_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "producer.sqlite"
    provider = LocalKeyProvider(MASTER_KEY)
    with _driver(path, provider) as opened:
        store, sessions, _, driver = opened
        _start_turn(driver)

        stored = driver.records[-1]
        payload = stored.record.payload
        definition = ContentRef.from_dict(payload["definition"])
        budget = ContentRef.from_dict(payload["budget"])
        assert stored.seq == 3
        assert stored.record.feature_level == 1
        assert store._one("SELECT write_level FROM sk_settings WHERE singleton=1")[0] == 1
        assert payload["input_ids"] == ["i"]
        assert payload["handler_version"] == "1"
        assert payload["binding"] is None
        assert payload["generation"] == 0
        assert payload["requirements"] == {"level": 1, "handler_version": "1", "capabilities": []}
        assert payload["definition_digest"] == reference_digest(definition)
        assert payload["environment"] is None
        assert driver.record_content_ref(stored.record, "environment_descriptor") is None
        assert driver.record_content_ref(stored.record, "environment_operational_policy") is None
        assert "do work" not in stored.record.encode().decode()

        body = json.loads(
            store._one(
                "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=3",
                ("default", "s"),
            )[0]
        )
        assert set(body["payload"]["definition"]) == {"content_id", "mac", "size", "key_id", "labels"}
        assert set(body["payload"]["budget"]) == {"content_id", "mac", "size", "key_id", "labels"}
        with sessions.store.atomic() as tx:
            writer = driver.content_writer(tx)
            links = writer.durable.read_links("default", "s", record_seq=3)
        assert {link.slot: link.ref for link in links} == {"definition": definition, "budget": budget}
        _assert_turn_content(
            driver,
            definition=definition,
            budget=budget,
            definition_digest=payload["definition_digest"],
        )
        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        assert hydrate_session_stored_record(stored, hydration).record.payload["environment"] is None
        before_replay = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in ("sk_record", "sk_commit", "sk_content", "sk_content_link")
        }
        active_turn_id = driver.state.active_turn_id
        driver.state.active_turn_id = None

        def replay(tx):
            replayed = publish_turn_started(driver.content_writer(tx), definition, budget)
            return [
                driver.record(
                    "turn_started",
                    payload
                    | {
                        "definition": replayed.definition.to_dict(),
                        "budget": replayed.budget.to_dict(),
                        "definition_digest": replayed.definition_digest,
                    },
                    tid="s/turn/i",
                )
            ]

        driver.commit([], guarded=True, prepare=replay)
        after_replay = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in before_replay
        }
        assert after_replay == before_replay
        assert driver.records[-1].record.payload["definition"] == definition.to_dict()
        assert driver.records[-1].record.payload["budget"] == budget.to_dict()
        driver.state.active_turn_id = active_turn_id
        assert sessions.store.release(driver.scope.lease)

    reopened_provider = provider
    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=reopened, content_key_provider=reopened_provider)
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="reopen-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=reopened._now()))
        stored = driver.records[-1]
        payload = stored.record.payload
        _assert_turn_content(
            driver,
            definition=ContentRef.from_dict(payload["definition"]),
            budget=ContentRef.from_dict(payload["budget"]),
            definition_digest=payload["definition_digest"],
        )


def test_trusted_environment_producer_publishes_level_3_content_refs(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / "trusted-environment-producer.sqlite"
    binding = _trusted_binding()
    capabilities = binding.capabilities
    assert isinstance(capabilities, EnvironmentCapabilities)
    with _driver(path, provider, level=3, bootstrap_level=3) as opened:
        store, _, runtime, driver = opened
        assert store._one("SELECT write_level FROM sk_settings WHERE singleton=1")[0] == 3
        assert store.worker_level == 3
        runtime.trusted_environment_binding = binding
        _start_turn(driver)

        stored = driver.records[-1]
        payload = stored.record.payload
        environment = payload["environment"]
        assert stored.record.feature_level == 3
        assert environment | {} == capabilities.to_dict() | {
            "descriptor": environment["descriptor"],
            "operational_policy": environment["operational_policy"],
        }
        assert environment["environment_id"] == "environment/turn-producer"
        assert environment["generation"] == 7
        assert environment["isolation"] == "container"
        assert environment["network"] == "allowlist"
        assert environment["environment_capabilities"] == [
            "command.execute",
            "filesystem.workspace",
            "network.allowlist",
        ]
        assert environment["capabilities_digest"] == capabilities.capabilities_digest
        assert environment["network_policy_ref_digest"] == "1" * 64
        assert environment["isolation_policy_ref_digest"] == "2" * 64

        descriptor = ContentRef.from_dict(environment["descriptor"])
        policy = ContentRef.from_dict(environment["operational_policy"])
        assert driver.record_content_ref(stored.record, "environment_descriptor") == descriptor
        assert driver.record_content_ref(stored.record, "environment_operational_policy") == policy
        assert driver.hydrate_record_field(stored.record, "environment_descriptor", descriptor) == binding.descriptor
        assert driver.hydrate_record_field(stored.record, "environment_operational_policy", policy) == binding.operational_policy
        encoded = stored.record.encode().decode()
        assert "sandbox-plaintext-marker" not in encoded
        assert "api.example" not in encoded

        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        hydrated = hydrate_session_stored_record(stored, hydration).record
        assert hydrated.feature_level == 3
        assert hydrated.payload["environment"] == environment | {
            "descriptor": binding.descriptor,
            "operational_policy": binding.operational_policy,
        }


@pytest.mark.persistent_store
def test_level_2_worker_cannot_claim_trusted_environment_turn(store_backend: str, database: str | Path) -> None:
    del store_backend
    provider = LocalKeyProvider(MASTER_KEY)
    _create_bootstrapped_session(database, provider, level=3, input_text="do work")
    with open_store(database) as store:
        store.configure_reader(level=3)
        factory_calls = []

        def runtime_factory(*_args):
            factory_calls.append(True)
            raise AssertionError("a level-2 claim must not construct a runtime")

        sessions = SessionDriver(
            store=store,
            content_key_provider=provider,
            runtime_factory=runtime_factory,
            worker_level=2,
        )
        try:
            created = store.read("s", limit=1).records[0].record
            assert created.feature_level == 2
            assert created.payload["bootstrap"]["level"] == 3
            assert (
                store.claim_runnable(
                    owner="level-2-worker",
                    ttl_ms=15_000,
                    limit=1,
                    work_kinds=("drive",),
                )
                == ()
            )
            assert factory_calls == []
            records = store.read("s", limit=100).records
            assert not any(stored.record.kind == "turn_started" for stored in records)
            assert store._one("SELECT lease_owner FROM sk_session WHERE tenant_id='default' AND session_id='s'")[0] is None
        finally:
            sessions.close()


@pytest.mark.persistent_store
def test_level_3_bootstrap_drives_first_turn_with_trusted_environment(
    store_backend: str,
    database: str | Path,
) -> None:
    del store_backend
    provider = LocalKeyProvider(MASTER_KEY)
    _create_bootstrapped_session(database, provider, level=3)
    holder: dict[str, SessionDriver] = {}
    factory_calls = []

    with open_store(database) as store:
        store.configure_reader(level=3)

        def runtime_factory(drive_store, agent, config, task, session_id):
            factory_calls.append(session_id)
            runtime = holder["driver"].runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            runtime.trusted_environment_binding = _trusted_binding()
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["driver"] = sessions
        try:
            handle = sessions.start(
                "s",
                _producer_agent(),
                _producer_config([LLMResponse("done")]),
                "do work",
            )
            assert handle.result(timeout=8).final_output == "done"
            turn = next(stored.record for stored in store.read("s", limit=100).records if stored.record.kind == "turn_started")
            assert turn.feature_level == 3
            assert turn.payload["requirements"]["level"] == 3
            assert turn.payload["environment"]["capabilities_digest"] == _restricted_capabilities().capabilities_digest
            assert factory_calls == ["s"]
        finally:
            sessions.close()


@pytest.mark.parametrize(
    ("bootstrap_level", "with_binding"),
    [(3, False), (2, True)],
)
def test_bootstrap_and_runtime_binding_must_agree(
    tmp_path: Path,
    bootstrap_level: int,
    with_binding: bool,
) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / f"binding-mismatch-{bootstrap_level}-{with_binding}.sqlite"
    _create_bootstrapped_session(path, provider, level=bootstrap_level)
    holder: dict[str, SessionDriver] = {}

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.configure_reader(level=3)

        def runtime_factory(drive_store, agent, config, task, session_id):
            runtime = holder["driver"].runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            if with_binding:
                runtime.trusted_environment_binding = _trusted_binding()
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["driver"] = sessions
        try:
            handle = sessions.start(
                "s",
                _producer_agent(),
                _producer_config([LLMResponse("unexpected")]),
                "do work",
            )
            with pytest.raises(InvalidTrustedEnvironmentBinding):
                handle.result(timeout=8)
            records = store.read("s", limit=100).records
            assert not any(stored.record.kind == "turn_started" for stored in records)
        finally:
            sessions.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize(
    ("reader_level", "worker_level"),
    [(3, 2), (3, 3)],
)
def test_default_store_factories_preserve_reader_and_worker_levels(
    store_backend: str,
    database: str | Path,
    reader_level: int,
    worker_level: int,
) -> None:
    del store_backend
    with open_store(database) as source:
        source.configure_reader(level=reader_level)
        source.configure_worker(level=worker_level, handler_versions=("1",), capabilities=())

        with source.drive_store_factory()() as drive:
            assert drive.reader_level == reader_level
            assert drive.worker_level == worker_level

        with source.heartbeat_store_factory()() as heartbeat:
            assert heartbeat.reader_level == reader_level
            assert heartbeat.worker_level == worker_level


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not_capabilities", "must be EnvironmentCapabilities"),
        ("missing_digest", "capabilities_digest is required"),
        ("malformed_operational_policy", "closed object"),
        ("invalid_operational_digest", "operational_policy.network_policy_digest"),
    ],
)
def test_invalid_trusted_environment_fails_closed_without_level_1_fallback(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    capabilities: EnvironmentCapabilities | object = _restricted_capabilities()
    operational_policy = _operational_policy()
    if case == "not_capabilities":
        capabilities = capabilities.to_dict()
    elif case == "missing_digest":
        capabilities = EnvironmentCapabilities.from_dict(capabilities.to_dict() | {"capabilities_digest": None})
    elif case == "malformed_operational_policy":
        operational_policy["extra"] = "rejected"
    elif case == "invalid_operational_digest":
        operational_policy["network_policy_digest"] = "not-a-digest"

    provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / f"invalid-environment-{case}.sqlite"
    with _driver(path, provider, level=3, bootstrap_level=3) as opened:
        _, _, runtime, driver = opened
        runtime.trusted_environment_binding = _trusted_binding(
            capabilities,
            operational_policy=operational_policy,
        )
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        with pytest.raises(InvalidTrustedEnvironmentBinding, match=message):
            _recovery_phase.run(driver, inputs.start_turn(driver))
        assert not any(stored.record.kind == "turn_started" for stored in driver.records)


def test_turn_environment_content_hydrates_through_nested_paths(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / "environment-producer.sqlite"
    with _driver(path, provider) as opened:
        store, _, runtime, driver = opened
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))

        task = runtime.compile("do work", "s/turn/i", seed=None, control=False)
        definition = runtime.definition(task)
        environment = {
            "descriptor": {"kind": "local", "route": "sandbox"},
            "environment_capabilities": [],
            "capabilities_digest": None,
            "network_policy_ref_digest": None,
            "isolation_policy_ref_digest": None,
            "operational_policy": {"network": "allow"},
        }
        turn = make_record_before_publication(
            "turn_started",
            session_id="s",
            turn_id="s/turn/i",
            payload={
                "input_ids": ["i"],
                "definition": definition,
                "definition_digest": digest(definition),
                "handler_version": "1",
                "budget": RunBudgetLimits().to_dict(),
                "binding": None,
                "generation": 0,
                "environment": environment,
            },
        )
        driver.commit([turn])

        stored = driver.records[-1]
        payload = stored.record.payload
        descriptor = ContentRef.from_dict(payload["environment"]["descriptor"])
        policy = ContentRef.from_dict(payload["environment"]["operational_policy"])
        assert driver.record_content_ref(stored.record, "environment_descriptor") == descriptor
        assert driver.record_content_ref(stored.record, "environment_operational_policy") == policy
        assert driver.hydrate_record_field(stored.record, "environment_descriptor", descriptor) == environment["descriptor"]
        assert (
            driver.hydrate_record_field(stored.record, "environment_operational_policy", policy)
            == environment["operational_policy"]
        )

        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        hydrated = hydrate_session_stored_record(stored, hydration).record
        assert hydrated.payload["environment"] == environment
        encoded = stored.record.encode().decode()
        assert '"route":"sandbox"' not in encoded
        assert '"network":"allow"' not in encoded


def test_plain_child_admission_is_published_once_in_the_turn_transaction(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / "child.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=provider)
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        task = runtime.compile("child", "s/turn/i", seed=None, control=False)
        definition = runtime.definition(task)
        admission = {
            "mode": "configured",
            "selector": "kernel",
            "definition": definition,
            "definition_digest": digest(definition),
            "budget": RunBudgetLimits(max_tool_calls=3).to_dict(),
            "handler_version": "1",
            "sub_config": None,
            "exclude_files_pattern": None,
            "handoff_count": 0,
            "max_handoffs": 100,
            "handoff_metadata": {},
        }
        with sessions.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")

            def session_spec(content: ContentSessionWriter) -> SessionSpec:
                principal, _ = content.service_principal("local")
                return SessionSpec(
                    "s",
                    principal,
                    content.put_json("/workspace").to_dict(),
                    attributes=content.put_json({"child_admission": admission}).to_dict(),
                )

            writer.create_session(session_spec, consumers=())
        sessions.push("s", InboxItem("start", "user", {"content": "child"}))
        lease = sessions.store.acquire("s", owner="child-producer-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        before = store._one("SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s", ("default", "s"))[0]
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        after = store._one("SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s", ("default", "s"))[0]
        assert after == before + 2

        payload = driver.records[-1].record.payload
        definition_ref = ContentRef.from_dict(payload["definition"])
        budget_ref = ContentRef.from_dict(payload["budget"])
        assert payload["definition_digest"] == reference_digest(definition_ref)
        assert driver.hydrator_for(driver.records[-1].seq).definition(definition_ref).task == task
        assert driver.hydrator_for(driver.records[-1].seq).budget(budget_ref) == RunBudgetLimits(max_tool_calls=3)
