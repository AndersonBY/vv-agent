import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from scripts.s1_fixture_values import create_fixture_session
from tests.sim.assembly import SIM_HANDLER_VERSION
from tests.sim.scenarios import Provider
from tests.sim.world import SimWorld
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._postgres import RuntimeBinding
from vv_agent.kernel._requirements import SessionRequirements, capability_supports
from vv_agent.kernel._sql import UnsupportedFeatureLevel
from vv_agent.kernel._store import WorkerSupports
from vv_agent.session.records import InboxItem, SessionSpec

from .conftest import open_store
from .helpers import content_ref, item, principal, record


def create(
    store,
    session_id: str,
    *,
    consumers: tuple[str, ...] = (),
    bootstrap: SessionRequirements | None = None,
) -> None:
    spec = SessionSpec(
        session_id,
        principal(),
        content_ref("workspace"),
        attributes=content_ref(f"attributes/{session_id}"),
        feature_level=2 if bootstrap is not None else 1,
        bootstrap=bootstrap,
    )
    with store.atomic() as tx:
        tx.create(spec, consumers=consumers)


def append_turn(store, session_id: str, *, level: int, handler_version: str, capabilities: tuple[str, ...]) -> None:
    lease = store.acquire(session_id, owner=f"upgrade/{session_id}", ttl_ms=10_000)
    assert lease is not None
    requirements = {
        "level": level,
        "handler_version": handler_version,
        "capabilities": list(capabilities),
    }
    turn = record(
        "turn_started",
        sid=session_id,
        handler_version=handler_version,
        requirements=requirements,
    )
    with store.atomic() as tx:
        tx.append(
            session_id,
            lease=lease,
            expected_seq=store.read(session_id).head_seq,
            commit_id=f"turn/{session_id}",
            records=(turn,),
        )
    assert store.release(lease)


def required_session(
    store,
    session_id: str,
    *,
    level: int,
    handler_version: str,
    capabilities: tuple[str, ...],
) -> None:
    store._rows("UPDATE sk_settings SET write_level=%s WHERE singleton=1", (max(2, level),))
    requirements = SessionRequirements.normalized(
        level=level,
        handler_version=handler_version,
        capabilities=capabilities,
    )
    store.configure_worker(
        level=max(2, level),
        handler_versions=(handler_version,),
        capabilities=capabilities,
    )
    create(store, session_id, consumers=("host",), bootstrap=requirements)
    append_turn(
        store,
        session_id,
        level=level,
        handler_version=handler_version,
        capabilities=capabilities,
    )


def configure(store, *, level: int, handler_version: str, capabilities: tuple[str, ...] = ()) -> None:
    store.configure_worker(
        level=level,
        handler_versions=(handler_version,),
        capabilities=capabilities,
    )


def assert_skipped_but_claimable(store, session_id: str, *, visible: bool = True) -> None:
    assert store.acquire(session_id, owner="incompatible", ttl_ms=10_000) is None
    assert (session_id in {work.session_id for work in store.list_runnable()}) is visible
    claimed = store.claim_runnable(owner="incompatible", ttl_ms=10_000)
    assert session_id not in {entry.work.session_id for entry in claimed if entry.work.kind == "drive"}
    assert store._one(
        "SELECT deletion_state,lease_epoch,lease_owner,lease_until_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    ) == ("live", 1, None, None)


def test_worker_support_serialization_is_stable() -> None:
    supports = WorkerSupports(
        max_level=3,
        handler_versions=("handler/v2", "handler/v1", "handler/v1"),
        capabilities=("toolset:b", "env:remote", "toolset:b"),
    )

    assert supports.to_dict() == {
        "storage_major": "S1",
        "max_level": 3,
        "handler_versions": ["handler/v1", "handler/v2"],
        "capabilities": ["env:remote", "toolset:b"],
    }
    assert json.dumps(supports.to_dict(), sort_keys=True) == (
        '{"capabilities": ["env:remote", "toolset:b"], "handler_versions": '
        '["handler/v1", "handler/v2"], "max_level": 3, "storage_major": "S1"}'
    )


def test_old_worker_skips_higher_level_and_upgraded_worker_claims(store) -> None:
    required_session(store, "high", level=2, handler_version="1", capabilities=())
    configure(store, level=1, handler_version="1")

    assert_skipped_but_claimable(store, "high")

    configure(store, level=2, handler_version="1")
    lease = store.acquire("high", owner="upgraded", ttl_ms=10_000)
    assert lease is not None
    store.release(lease)


def test_two_node_s1_rolling_upgrade_continues_admission_and_driving(database) -> None:
    with SimWorld(database, seed=17) as world:
        world.scenario = "accepted"
        world.install_write_observer()
        host = world.connection("host")
        observer = world.connection("observer")
        provider = Provider(world)

        def assert_completed(session_id: str, input_id: str) -> None:
            state, rows, _ = world.read_state(session_id)
            terminals = [row.record for row in rows if row.record.kind == "turn_ended"]
            assert [(result.turn_id, result.payload["status"]) for result in terminals] == [
                (f"{session_id}/turn/{input_id}", "completed")
            ]
            assert state.active_turn_id is None

        world.connection("node-a/v1").configure_reader(level=1)
        world.connection("node-b/v1").configure_reader(level=1)
        world.create()
        assert world.drive(participant="node-a/v1", provider=provider)
        assert world.read_state("s")[0].active_turn_id == "s/turn/initial"

        runtime = world.runtime("higher-level", "node-a/v2")
        capabilities = capability_supports(
            tools=((runtime.registry.get_schema("effect"), runtime.registry.tool_metadata("effect")),),
            models=(runtime.model_binding,),
        )
        requirements = SessionRequirements(2, SIM_HANDLER_VERSION, capabilities)
        level_two_input = InboxItem("level-two", "user", {"content": "go"}, feature_level=2, requirements=requirements)

        for node, survivor in (("node-a", "node-b/v1"), ("node-b", "node-a/v2")):
            retired = f"{node}/v1"
            world.connections.pop(retired).connection.close()
            if retired_heartbeat := world.connections.pop(f"{retired}/heartbeat", None):
                retired_heartbeat.connection.close()
            world.create(f"while-upgrading-{node}")
            assert world.drive(f"while-upgrading-{node}", participant=survivor)
            assert_completed(f"while-upgrading-{node}", "initial")

            upgraded = f"{node}/v2"
            configure(world.connection(upgraded), level=2, handler_version=SIM_HANDLER_VERSION, capabilities=capabilities)
            session_id = f"after-upgrading-{node}"
            world.create(session_id)
            assert world.drive(session_id, participant=upgraded)
            assert_completed(session_id, "initial")
            assert {row.record.feature_level for row in world.read_state(session_id)[1]} == {1}

            with pytest.raises(UnsupportedFeatureLevel, match="write level 1"):
                world.push(session_id, level_two_input)
            assert observer.read_input(session_id, "level-two") is None
            assert world.read_state("s")[0].active_turn_id == "s/turn/initial"

        with pytest.raises(UnsupportedFeatureLevel, match="write level 1"):
            create_fixture_session(host, "higher-level", consumers=(), feature_level=2, bootstrap=requirements)
        assert "higher-level" not in observer.list_sessions()

        with host.atomic():
            host._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")

        provider.ready = True
        world.push("s", provider.callback())
        assert world.drive(participant="node-b/v2", provider=provider)
        assert_completed("s", "initial")
        assert sum(call[3] == "tool" for call in world.calls) == 1

        create_fixture_session(host, "higher-level", consumers=(), feature_level=2, bootstrap=requirements)
        world.push("higher-level", level_two_input)
        assert world.drive("higher-level", participant="node-a/v2")
        assert_completed("higher-level", "level-two")
        records = {row.record.kind: row.record for row in world.read_state("higher-level")[1]}
        assert records["session_created"].feature_level == 2
        assert records["input_applied"].payload["input"]["feature_level"] == 2
        assert records["turn_started"].payload["requirements"] == requirements.to_dict()
        world.check(quiescent=True, every_prefix=True)
        world.check_accounting()


def test_handler_version_mismatch_skips_session(store) -> None:
    required_session(store, "handler", level=1, handler_version="2", capabilities=())
    configure(store, level=2, handler_version="1")

    assert_skipped_but_claimable(store, "handler")

    configure(store, level=2, handler_version="2")
    lease = store.acquire("handler", owner="matching", ttl_ms=10_000)
    assert lease is not None
    store.release(lease)


def test_missing_required_capability_skips_session(store) -> None:
    required_session(
        store,
        "capability",
        level=1,
        handler_version="1",
        capabilities=("env:remote", "toolset:digest"),
    )
    configure(store, level=2, handler_version="1", capabilities=("env:remote",))

    assert_skipped_but_claimable(store, "capability")

    configure(store, level=2, handler_version="1", capabilities=("env:remote", "toolset:digest"))
    lease = store.acquire("capability", owner="matching", ttl_ms=10_000)
    assert lease is not None
    store.release(lease)


def test_fully_matching_worker_claims_and_scans_session(store) -> None:
    required_session(
        store,
        "match",
        level=2,
        handler_version="handler/v2",
        capabilities=("env:remote", "toolset:digest"),
    )
    configure(
        store,
        level=2,
        handler_version="handler/v2",
        capabilities=("env:remote", "toolset:digest"),
    )

    assert "match" in {work.session_id for work in store.list_runnable()}
    lease = store.acquire("match", owner="matching", ttl_ms=10_000)
    assert lease is not None
    store.release(lease)


def test_unmatched_workers_leave_session_for_upgraded_worker(store) -> None:
    required_session(
        store,
        "later",
        level=2,
        handler_version="handler/v2",
        capabilities=("env:remote", "toolset:digest"),
    )

    configure(store, level=1, handler_version="handler/v2", capabilities=("env:remote", "toolset:digest"))
    assert_skipped_but_claimable(store, "later")
    configure(store, level=2, handler_version="handler/v1", capabilities=("env:remote", "toolset:digest"))
    assert_skipped_but_claimable(store, "later")
    configure(store, level=2, handler_version="handler/v2", capabilities=("env:remote",))
    assert_skipped_but_claimable(store, "later")

    configure(
        store,
        level=2,
        handler_version="handler/v2",
        capabilities=("env:remote", "toolset:digest"),
    )
    lease = store.acquire("later", owner="upgraded", ttl_ms=10_000)
    assert lease is not None
    store.release(lease)


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        (RuntimeBinding("worker"), " FOR NO KEY UPDATE OF s SKIP LOCKED"),
        (RuntimeBinding("tenant", "tenant-a"), ""),
        (RuntimeBinding("shared-gateway"), ""),
    ],
)
def test_runnable_candidates_are_unlocked_and_claim_lock_is_worker_only(binding: RuntimeBinding, expected: str) -> None:
    store = object.__new__(CorePostgresStore)
    store._runtime_binding = binding
    store._root_store = None
    store._all_tenants = False
    store._worker_supports = WorkerSupports()
    store._reader_level = 2
    store._tenant_id = "default"
    store.configure_input_admission(allow_unsigned=True)
    queries: list[str] = []
    store._rows = lambda query, params=(): queries.append(query) or []  # type: ignore[method-assign]

    assert (
        store._runnable_work(limit=1, after=None, apply_worker_filter=True, work_kinds=("drive",), projection_consumers=()) == ()
    )
    assert store._runnable_lock_clause() == expected
    assert " FOR NO KEY UPDATE OF s SKIP LOCKED" not in queries[0]
    assert store.lock_clause == " FOR NO KEY UPDATE"
    if binding.mode != "worker":
        with pytest.raises(PermissionError, match="worker binding"):
            store.claim_runnable(owner="forbidden", ttl_ms=10_000)


def test_claim_drive_locks_only_the_selected_session() -> None:
    store = object.__new__(CorePostgresStore)
    store._runtime_binding = RuntimeBinding("worker")
    store._root_store = None
    store._all_tenants = False
    store._worker_supports = WorkerSupports()
    store._reader_level = 2
    store._tenant_id = "default"
    queries: list[str] = []

    def rows(query: str, params=()):
        queries.append(query)
        return (
            [(0, 0, 0, None, None, 0, 1, "live", "1", b"[]", None, 1_000)]
            if query.lstrip().startswith("WITH")
            else [("selected",)]
        )

    store._rows = rows  # type: ignore[method-assign]
    store._require_routing_cache = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        requirements=SessionRequirements(1, "1", ())
    )
    lease = store._claim_drive("selected", owner="worker", ttl_ms=10_000)

    assert lease is not None and lease.epoch == 1
    assert queries[0].count(" FOR NO KEY UPDATE OF s SKIP LOCKED") == 1
    assert "session_id=%s" in queries[0]


@pytest.mark.postgres
def test_postgres_worker_claim_skips_locked_session(store, database: str) -> None:
    create(store, "a-locked")
    create(store, "b-available")
    with store.atomic() as tx:
        tx.push("a-locked", item("locked-input"))
        tx.push("b-available", item("available-input"))

    with store._transaction():
        store._rows(
            "SELECT session_id FROM sk_session WHERE tenant_id=%s AND session_id='a-locked' FOR UPDATE",
            (store.tenant_id,),
        )
        with open_store(database) as scanner:
            assert isinstance(scanner, CorePostgresStore)
            scanner._runtime_binding = RuntimeBinding("worker")
            scanner.configure_worker(level=1, handler_versions=("1",), capabilities=())
            claimed = scanner.claim_runnable(owner="scanner", ttl_ms=10_000, limit=1)

    assert [(entry.work.kind, entry.work.session_id) for entry in claimed if entry.work.kind == "drive"] == [
        ("drive", "b-available")
    ]
    assert claimed[0].lease is not None
    with open_store(database) as releaser:
        assert isinstance(releaser, CorePostgresStore)
        releaser._runtime_binding = RuntimeBinding("worker")
        assert releaser.release(claimed[0].lease)


@pytest.mark.postgres
def test_parallel_postgres_workers_never_hold_the_same_lease(store, database: str) -> None:
    create(store, "shared")
    barrier = Barrier(2)

    def claim(owner: str) -> str | None:
        with open_store(database) as worker:
            worker.configure_worker(level=1, handler_versions=("1",), capabilities=())
            barrier.wait(timeout=10)
            lease = worker.acquire("shared", owner=owner, ttl_ms=10_000)
            return None if lease is None else lease.owner

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = list(executor.map(claim, ("worker-a", "worker-b")))

    assert sum(owner is not None for owner in owners) == 1
    owner = next(owner for owner in owners if owner is not None)
    assert store._one(
        "SELECT lease_epoch,lease_owner FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, "shared"),
    ) == (1, owner)


def test_thin_reader_can_read_session_state_without_execution_supports(store) -> None:
    required_session(store, "read-blocked", level=1, handler_version="handler/v2", capabilities=("env:remote",))
    configure(store, level=1, handler_version="handler/v1", capabilities=())
    store.configure_reader(level=2)

    assert store.read_state("read-blocked")[0].session_id == "read-blocked"
    assert store.peek_inbox("read-blocked") == ()
    assert store.acquire("read-blocked", owner="incompatible", ttl_ms=10_000) is None
