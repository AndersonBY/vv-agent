"""W4 host callbacks join the local worker's real PostgreSQL drive transaction."""

from __future__ import annotations

from pathlib import Path
from threading import current_thread
from typing import Any

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.kernel import _driver_dispatch
from vv_agent.model import ScriptedModelProvider
from vv_agent.session import kernel
from vv_agent.session.children import ChildSession, InvalidChildBatch
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import SessionSpec
from vv_agent.session.sql import SQLSessionTx
from vv_agent.session.store import LeaseLost, SequenceConflict
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .helpers import wrap_fixture_store

PARENT = "parent"
CHILDREN = ("child-0", "child-1")


class AdmissionCut(BaseException):
    """Stop recovery retries after the injected SQL fence failure."""


@function_tool
def child_task() -> str:
    raise AssertionError("child_task must use child admission")


class HostAdapter:
    """A host-style adapter over the connection selected by the runtime factory."""

    def __init__(self, store) -> None:
        self.store = store

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.store._rows(sql, params)


def _store_connection(store) -> Any:
    current = store
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        connection = getattr(current, "connection", None)
        if connection is not None:
            return connection
        parent = getattr(current, "_store", None) or getattr(current, "_root", None)
        if parent is None or parent is current:
            break
        current = parent
    raise AssertionError("store does not expose a PostgreSQL connection")


def _sql_tx(tx) -> SQLSessionTx:
    current = tx
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SQLSessionTx):
            return current
        parent = getattr(current, "_tx", None) or getattr(current, "tx", None) or getattr(current, "store", None)
        if parent is None or parent is current:
            break
        current = parent
    raise AssertionError("transaction does not expose its SQLSessionTx")


def _tx_evidence(tx) -> tuple[int, str]:
    sql_tx = _sql_tx(tx)
    pid = _store_connection(sql_tx.store).info.backend_pid
    assert isinstance(pid, int)
    return pid, str(sql_tx.transaction_id)


def _callback_evidence(store) -> tuple[int, str]:
    pid, transaction_id = store._one("SELECT pg_backend_pid(), pg_current_xact_id()::text")
    return pid, transaction_id


def _setup_driver(store, tmp_path: Path, runtime_factory=None) -> tuple[SessionDriver, LocalKeyProvider]:
    key_provider = LocalKeyProvider(b"w" * 32, state_path=tmp_path / "content-keys.sqlite3")
    driver = SessionDriver(
        store=store,
        content_key_provider=key_provider,
        runtime_factory=runtime_factory,
    )
    return driver, key_provider


def _install_host_tables(store) -> None:
    store._rows("CREATE TABLE allowed (id integer PRIMARY KEY)")
    store._rows("INSERT INTO allowed VALUES (1)")
    store._rows(
        "CREATE TABLE business_child (id integer PRIMARY KEY, marker integer REFERENCES allowed(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )


def _start_handle(driver: SessionDriver, tmp_path: Path):
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("child-call", "child_task", {})]),
            LLMResponse("child zero done"),
            LLMResponse("child one done"),
            LLMResponse("parent done"),
        ],
    )
    agent = Agent("parent", "Use the child tool.", model="m", tools=[child_task])
    return driver.start(
        PARENT,
        agent,
        RunConfig(model_provider=provider),
        "go",
        input_id="go",
        autostart=False,
    )


def _observe(database, key_provider: LocalKeyProvider) -> dict[str, Any]:
    with open_store(database) as raw_reader:
        reader = wrap_fixture_store(raw_reader, key_provider)
        parent_records = reader.read(PARENT, limit=100).records
        content_counts = {
            sid: reader._one(
                "SELECT count(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
                (reader.tenant_id, sid),
            )[0]
            for sid in (PARENT, *CHILDREN)
        }
        child_state = {
            sid: {
                "session": reader._one(
                    "SELECT count(*) FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                    (reader.tenant_id, sid),
                )[0],
                "records": reader._one(
                    "SELECT count(*) FROM sk_record WHERE tenant_id=%s AND session_id=%s",
                    (reader.tenant_id, sid),
                )[0],
                "inbox": reader._one(
                    "SELECT count(*) FROM sk_inbox WHERE tenant_id=%s AND session_id=%s",
                    (reader.tenant_id, sid),
                )[0],
                "turn_ended": reader._one(
                    "SELECT count(*) FROM sk_record WHERE tenant_id=%s AND session_id=%s AND kind='turn_ended'",
                    (reader.tenant_id, sid),
                )[0],
                "content": content_counts[sid],
            }
            for sid in CHILDREN
        }
        return {
            "business": reader._rows("SELECT id, marker FROM business_child ORDER BY id"),
            "sessions": reader.list_sessions(),
            "parent_records": parent_records,
            "parent_parked": [row.record for row in parent_records if row.record.kind == "op_parked"],
            "parent_input": reader._rows(
                "SELECT input_id, consumed_seq FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id='go'",
                (reader.tenant_id, PARENT),
            ),
            "parent_pending_input": reader._one(
                "SELECT count(*) FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id='go' AND consumed_seq IS NULL",
                (reader.tenant_id, PARENT),
            )[0],
            "content_counts": content_counts,
            "children": child_state,
        }


@pytest.mark.postgres
def test_public_local_worker_host_admission_commits_business_children_and_parent_atomically(
    store,
    database,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_host_tables(store)
    host_pid = _store_connection(store).info.backend_pid
    holder: dict[str, SessionDriver] = {}
    factory_calls: list[tuple[str, str, int]] = []
    callbacks: list[tuple[str, str, int, str]] = []
    child_creates: list[tuple[str, int, str]] = []
    append_evidence: list[tuple[int, str]] = []
    drive_calls: list[tuple[str, str, int]] = []
    child_specs: list[ChildSession] = []

    def runtime_factory(drive_store, agent, config, task, session_id):
        factory_calls.append((session_id, current_thread().name, _store_connection(drive_store).info.backend_pid))
        adapter = HostAdapter(drive_store)

        def admit(plan):
            pid, transaction_id = _callback_evidence(drive_store)
            callbacks.append((plan.session_id, current_thread().name, pid, transaction_id))
            for index in range(2):
                adapter.execute("INSERT INTO business_child VALUES (%s,1)", (index,))
            return child_specs

        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=drive_store,
        )
        runtime.children["child_task"] = admit
        return runtime

    driver, key_provider = _setup_driver(store, tmp_path, runtime_factory)
    holder["driver"] = driver
    original_create = _driver_dispatch.create_child
    original_append = SQLSessionTx.append
    original_drive = kernel.drive

    def create_child(tx, child, plan, generation):
        pid, transaction_id = _tx_evidence(tx)
        child_creates.append((child.spec.session_id, pid, transaction_id))
        return original_create(tx, child, plan, generation)

    def append(tx, sid, **kwargs):
        if any(record.kind == "op_parked" for record in kwargs["records"]):
            append_evidence.append(_tx_evidence(tx))
        return original_append(tx, sid, **kwargs)

    def drive(tracked_store, sid, **kwargs):
        drive_calls.append((current_thread().name, sid, _store_connection(tracked_store).info.backend_pid))
        return original_drive(tracked_store, sid, **kwargs)

    monkeypatch.setattr(_driver_dispatch, "create_child", create_child)
    monkeypatch.setattr(SQLSessionTx, "append", append)
    monkeypatch.setattr(kernel, "drive", drive)

    try:
        driver.create(PARENT, str(tmp_path / "parent"), {"seed": {"messages": [], "shared_state": {}}})
        principal = driver._session_principal(PARENT).to_dict()
        parent_created = driver.store.read(PARENT, limit=1).records[0].record
        workspace = parent_created.payload["workspace"]
        attributes = parent_created.payload["attributes"]
        child_specs.extend(
            ChildSession(
                SessionSpec(child_id, principal, workspace, attributes=attributes),
                f"work for {child_id}",
            )
            for child_id in CHILDREN
        )
        handle = _start_handle(driver, tmp_path)
        handle.start()
        assert handle.result().final_output == "parent done"
    finally:
        driver.close()

    observed = _observe(database, key_provider)
    worker_pid = callbacks[0][2]
    worker_transaction = callbacks[0][3]
    assert observed["business"] == [(0, 1), (1, 1)]
    assert observed["sessions"] == (*CHILDREN, PARENT)
    assert len(observed["parent_parked"]) == 1
    assert observed["parent_input"][0][1] is not None
    assert observed["parent_pending_input"] == 0
    assert all(observed["children"][sid]["session"] == 1 for sid in CHILDREN)
    assert all(observed["children"][sid]["records"] > 1 for sid in CHILDREN)
    assert all(observed["children"][sid]["inbox"] == 1 for sid in CHILDREN)
    assert all(observed["children"][sid]["turn_ended"] == 1 for sid in CHILDREN)
    assert all(observed["children"][sid]["content"] > 0 for sid in CHILDREN)
    assert observed["content_counts"][PARENT] > 0
    assert len(callbacks) == 1
    assert callbacks[0][0] == PARENT
    assert callbacks[0][2:] == (worker_pid, worker_transaction)
    assert worker_pid != host_pid
    assert factory_calls[0][0] == PARENT
    assert factory_calls[0][2] == worker_pid
    assert (PARENT, callbacks[0][1], worker_pid) in factory_calls
    assert all(pid == worker_pid and transaction_id == worker_transaction for _sid, pid, transaction_id in child_creates)
    assert append_evidence == [(worker_pid, worker_transaction)]
    assert drive_calls[0][1] == PARENT
    assert drive_calls[0][2] == worker_pid
    # Each later claim may use a different pooled connection. Tie every drive to
    # its own runtime-factory admission instead of requiring one PID across claims.
    assert all((sid, thread_name, pid) in factory_calls for thread_name, sid, pid in drive_calls)


@pytest.mark.postgres
@pytest.mark.parametrize(
    "fault",
    ["invalid_second", "create_second", "cas", "lease_expired", "lease_epoch", "commit"],
)
def test_public_local_worker_host_admission_fault_rolls_back_every_part(
    store,
    database,
    monkeypatch,
    tmp_path: Path,
    fault: str,
) -> None:
    _install_host_tables(store)
    host_pid = _store_connection(store).info.backend_pid
    holder: dict[str, SessionDriver] = {}
    factory_calls: list[tuple[str, str, int]] = []
    callbacks: list[tuple[str, str, int, str]] = []
    child_creates: list[tuple[str, int, str]] = []
    append_attempts: list[tuple[int, str]] = []
    append_failures: list[type[BaseException]] = []
    append_returned: list[bool] = []
    drive_calls: list[tuple[str, str, int]] = []
    admission_snapshots: list[dict[str, Any]] = []
    child_specs: list[ChildSession] = []

    def runtime_factory(drive_store, agent, config, task, session_id):
        factory_calls.append((session_id, current_thread().name, _store_connection(drive_store).info.backend_pid))
        adapter = HostAdapter(drive_store)

        def admit(plan):
            observed = _observe(database, key_provider)
            admission_snapshots.append(
                {
                    "parent_records": [row.record.encode() for row in observed["parent_records"]],
                    "parent_input": observed["parent_input"],
                    "parent_pending_input": observed["parent_pending_input"],
                }
            )
            pid, transaction_id = _callback_evidence(drive_store)
            callbacks.append((plan.session_id, current_thread().name, pid, transaction_id))
            marker = 2 if fault == "commit" else 1
            for index in range(2):
                adapter.execute("INSERT INTO business_child VALUES (%s,%s)", (index, marker))
            return [child_specs[0], object()] if fault == "invalid_second" else child_specs

        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=drive_store,
        )
        runtime.children["child_task"] = admit
        return runtime

    driver, key_provider = _setup_driver(store, tmp_path, runtime_factory)
    holder["driver"] = driver
    original_create = _driver_dispatch.create_child
    original_append = SQLSessionTx.append
    original_drive = kernel.drive

    def create_child(tx, child, plan, generation):
        pid, transaction_id = _tx_evidence(tx)
        child_creates.append((child.spec.session_id, pid, transaction_id))
        result = original_create(tx, child, plan, generation)
        if fault == "create_second" and child.spec.session_id == CHILDREN[1]:
            raise RuntimeError("second child failed")
        return result

    def append(tx, sid, **kwargs):
        parent_admission = sid == PARENT and any(record.kind == "op_parked" for record in kwargs["records"])
        if parent_admission:
            append_attempts.append(_tx_evidence(tx))
            if fault == "cas":
                kwargs["expected_seq"] -= 1
            elif fault.startswith("lease_"):
                # The public append samples the scope's late guard after encoding.
                # Move the real leased row after that sample so SQL owns the fence.
                guard = tx._lease_guard
                assert guard is not None
                kwargs["lease"] = guard()
                sql_store = _sql_tx(tx).store
                if fault == "lease_expired":
                    sql_store._rows(
                        "UPDATE sk_session SET lease_until_ms=0 WHERE tenant_id=%s AND session_id=%s",
                        (sql_store.tenant_id, sid),
                    )
                else:
                    sql_store._rows(
                        "UPDATE sk_session SET lease_epoch=lease_epoch+1 WHERE tenant_id=%s AND session_id=%s",
                        (sql_store.tenant_id, sid),
                    )
        try:
            receipt = original_append(tx, sid, **kwargs)
        except (SequenceConflict, LeaseLost) as exc:
            if parent_admission and fault in {"cas", "lease_expired", "lease_epoch"}:
                append_failures.append(type(exc))
                raise AdmissionCut(str(exc)) from exc
            raise
        if parent_admission:
            append_returned.append(True)
        return receipt

    def drive(tracked_store, sid, **kwargs):
        drive_calls.append((current_thread().name, sid, _store_connection(tracked_store).info.backend_pid))
        return original_drive(tracked_store, sid, **kwargs)

    monkeypatch.setattr(_driver_dispatch, "create_child", create_child)
    monkeypatch.setattr(SQLSessionTx, "append", append)
    monkeypatch.setattr(kernel, "drive", drive)

    try:
        driver.create(PARENT, str(tmp_path / "parent"), {"seed": {"messages": [], "shared_state": {}}})
        principal = driver._session_principal(PARENT).to_dict()
        parent_created = driver.store.read(PARENT, limit=1).records[0].record
        workspace = parent_created.payload["workspace"]
        attributes = parent_created.payload["attributes"]
        child_specs.extend(
            ChildSession(
                SessionSpec(child_id, principal, workspace, attributes=attributes),
                f"work for {child_id}",
            )
            for child_id in CHILDREN
        )
        handle = _start_handle(driver, tmp_path)
        handle.start()
        if fault == "commit":
            import psycopg

            expected = psycopg.errors.ForeignKeyViolation
        elif fault == "invalid_second":
            expected = InvalidChildBatch
        elif fault == "create_second":
            expected = RuntimeError
        elif fault == "cas":
            expected_cause = SequenceConflict
        elif fault.startswith("lease_"):
            expected_cause = LeaseLost
        else:
            raise AssertionError(f"unexpected fault: {fault}")
        if fault in {"cas", "lease_expired", "lease_epoch"}:
            with pytest.raises(AdmissionCut) as caught:
                handle.result()
            assert isinstance(caught.value.__cause__, expected_cause)
        else:
            with pytest.raises(expected):
                handle.result()
    finally:
        driver.close()

    observed = _observe(database, key_provider)
    worker_pid = callbacks[0][2]
    callback_transactions = {callback[3] for callback in callbacks}
    admission_baseline = admission_snapshots[0]
    assert observed["business"] == []
    assert observed["sessions"] == (PARENT,)
    assert all(snapshot == admission_baseline for snapshot in admission_snapshots)
    assert [row.record.encode() for row in observed["parent_records"]] == admission_baseline["parent_records"]
    assert observed["parent_parked"] == []
    assert admission_baseline["parent_input"][0][1] is not None
    assert observed["parent_input"] == admission_baseline["parent_input"]
    assert observed["parent_pending_input"] == admission_baseline["parent_pending_input"] == 0
    assert all(
        observed["children"][sid] == {"session": 0, "records": 0, "inbox": 0, "turn_ended": 0, "content": 0} for sid in CHILDREN
    )
    assert callbacks
    assert all(pid == worker_pid for _sid, _thread, pid, _transaction_id in callbacks)
    assert worker_pid != host_pid
    assert factory_calls[0][0] == PARENT
    assert factory_calls[0][2] == worker_pid
    assert all(pid == worker_pid for _sid, _thread, pid in factory_calls)
    assert all(pid == worker_pid and transaction_id in callback_transactions for _sid, pid, transaction_id in child_creates)
    assert all(pid == worker_pid and transaction_id in callback_transactions for pid, transaction_id in append_attempts)
    if fault == "invalid_second":
        assert child_creates == []
    else:
        assert {sid for sid, _pid, _transaction_id in child_creates} == set(CHILDREN)
    assert drive_calls[0][1] == PARENT
    assert drive_calls[0][2] == worker_pid
    assert all(pid == worker_pid for _thread, _sid, pid in drive_calls)
    if fault == "cas":
        assert append_failures == [SequenceConflict]
    elif fault.startswith("lease_"):
        assert append_failures == [LeaseLost]
    if fault == "commit":
        assert append_returned == [True]
    else:
        assert append_returned == []


@pytest.mark.postgres
def test_public_local_worker_exposes_runtime_factory_failure(
    store,
    database,
    tmp_path: Path,
) -> None:
    _install_host_tables(store)
    host_pid = _store_connection(store).info.backend_pid
    factory_calls: list[tuple[str, str, int]] = []

    def runtime_factory(drive_store, agent, config, task, session_id):
        del agent, config, task
        factory_calls.append(
            (
                session_id,
                current_thread().name,
                _store_connection(drive_store).info.backend_pid,
            )
        )
        raise RuntimeError("worker runtime factory failed")

    driver, key_provider = _setup_driver(store, tmp_path, runtime_factory)
    try:
        driver.create(PARENT, str(tmp_path / "parent"), {"seed": {"messages": [], "shared_state": {}}})
        handle = _start_handle(driver, tmp_path)
        assert factory_calls == []
        handle.start()
        with pytest.raises(RuntimeError, match="worker runtime factory failed"):
            handle.result(timeout=8)
    finally:
        driver.close()

    observed = _observe(database, key_provider)
    assert observed["business"] == []
    assert observed["sessions"] == (PARENT,)
    assert observed["parent_parked"] == []
    assert observed["parent_pending_input"] == 1
    assert len(observed["parent_input"]) == 1
    assert observed["parent_input"][0][0] == "go"
    assert observed["parent_input"][0][1] is None
    assert all(observed["children"][sid]["session"] == 0 for sid in CHILDREN)
    assert len(factory_calls) == 1
    sid, thread_name, worker_pid = factory_calls[0]
    assert sid == PARENT
    assert thread_name.startswith("session-local-worker-drive")
    assert worker_pid != host_pid
