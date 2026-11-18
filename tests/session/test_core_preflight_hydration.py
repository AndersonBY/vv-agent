from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_and_fold_records
from vv_agent.kernel._content_runtime import durable_content_reader
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import InboxItem, Record, copy_json, digest
from vv_agent.kernel._reducer import TransitionError
from vv_agent.kernel._store import Lease
from vv_agent.runtime.hooks import BaseRuntimeHook
from vv_agent.session import kernel
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall

SESSION_ID = "core-preflight"
MASTER_KEY = b"p" * 32


class _Cut(BaseException):
    pass


class _CoreScope(LeaseOwnership):
    def bind_turn(self, turn_id, generation, applied_inputs) -> None:
        del turn_id, generation, applied_inputs


def _sessions(provider: LocalKeyProvider, store: SQLiteStore) -> SessionDriver:
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=provider)
    sessions.create(SESSION_ID, "/workspace", {"seed": {"messages": [], "shared_state": {}}})
    sessions.push(SESSION_ID, InboxItem("input", "user", {"content": "go"}))
    return sessions


def _capture_producer_records(
    monkeypatch: pytest.MonkeyPatch,
    sessions: SessionDriver,
    runtime,
    target_kind: str,
) -> list[Record]:
    captured: list[Record] = []
    original = kernel._Driver.commit

    def cut_target_commit(self, records, consumed=(), *, guarded=False, prepare=None, _logical_base=None):
        if any(record.kind == target_kind for record in records):
            captured.extend(records)
            raise _Cut
        return original(
            self,
            records,
            consumed,
            guarded=guarded,
            prepare=prepare,
            _logical_base=_logical_base,
        )

    monkeypatch.setattr(kernel._Driver, "commit", cut_target_commit)
    with pytest.raises(_Cut):
        drive(sessions.store, SESSION_ID, runtime=runtime)
    assert captured and any(record.kind == target_kind for record in captured)
    return captured


def _direct_driver(store: SQLiteStore, sessions: SessionDriver) -> tuple[CoreDriver, Lease]:
    lease = sessions.store.acquire(SESSION_ID, owner="core-preflight", ttl_ms=15_000)
    assert lease is not None
    scope = _CoreScope(
        lease,
        15_000,
        0.25,
        clock=store.clock,
        stop=Event(),
        ready=Event(),
        now_ms=store._now(),
    )
    driver = CoreDriver(
        sessions.store,
        SESSION_ID,
        scope,
        fold_factory=store.fold_factory,
        clock=store.clock,
        observe_cost=lambda: None,
        hook=lambda _point, _record: None,
        emit_phase=lambda _duration, _fields: None,
        content_reader=durable_content_reader(
            sessions.store,
            tenant_id=store.tenant_id,
            session_id=SESSION_ID,
        ),
    )
    return driver, lease


def _commit_from_core(
    store: SQLiteStore,
    sessions: SessionDriver,
    records: list[Record],
    *,
    logical_base: bool = False,
) -> None:
    driver, lease = _direct_driver(store, sessions)
    try:
        assert not hasattr(driver, "_logical_state_scope")
        base = None
        if logical_base:
            assert driver.content_reader is not None
            base = hydrate_and_fold_records(
                driver.records,
                store.fold_factory,
                SessionHydration(
                    driver.content_reader,
                    tenant_id=store.tenant_id,
                    session_id=SESSION_ID,
                ),
            ).fold
        driver.commit(records, _logical_base=base)
    finally:
        sessions.store.release(lease)


def _table_counts(store: SQLiteStore) -> dict[str, int]:
    return {
        table: int(
            store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=? AND session_id=?",
                (store.tenant_id, SESSION_ID),
            )[0]
        )
        for table in ("sk_record", "sk_commit", "sk_session_key", "sk_content", "sk_content_link")
    }


def _kind_count(store: SQLiteStore, kind: str) -> int:
    return int(
        store._one(
            "SELECT COUNT(*) FROM sk_record WHERE tenant_id=? AND session_id=? AND kind=?",
            (store.tenant_id, SESSION_ID, kind),
        )[0]
    )


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_kernel_load_keeps_stdlib_import_boundary(flags) -> None:
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import sys

        sys.path.insert(0, sys.argv[1])

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                assert (
                    fullname in {"vv_agent", "vv_agent.kernel"}
                    or fullname.startswith("vv_agent.kernel.")
                    or fullname.split(".")[0] in sys.stdlib_module_names
                ), fullname

        sys.meta_path.insert(0, CoreOnly())

        from vv_agent.kernel._driver import CoreDriver
        from vv_agent.kernel._boundary_hydration import validate_logical_preflight
        from vv_agent.kernel._content_reader import DurableContentReader

        assert CoreDriver and validate_logical_preflight and DurableContentReader
        assert all(
            name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
            for name in sys.modules
            if name.startswith("vv_agent")
        )
        assert not any(name.startswith(("vv_agent.session", "cryptography", "psycopg")) for name in sys.modules)
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            *flags,
            "-B",
            "-c",
            script,
            str(Path(__file__).parents[2] / "src"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.sqlite_file
def test_core_commits_wait_user_turn_parked_from_real_raw_prefix(tmp_path: Path, monkeypatch) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    runtime_agent = Agent("wait", "wait for the user", model="scripted")
    try:
        with SQLiteStore.standalone(str(tmp_path / "wait.sqlite3")) as store:
            sessions = _sessions(provider, store)
            runtime = sessions.runtime(
                runtime_agent,
                RunConfig(
                    model_provider=ScriptedModelProvider.new(
                        "scripted",
                        "scripted",
                        [LLMResponse("Which deployment should run?")],
                    ),
                    workspace="/workspace",
                    no_tool_policy="wait_user",
                ),
                session_id=SESSION_ID,
            )
            records = _capture_producer_records(monkeypatch, sessions, runtime, "turn_parked")
            _commit_from_core(store, sessions, records)

            assert any(record.record.kind == "turn_parked" for record in store.read(SESSION_ID).records)
    finally:
        provider.close()


@pytest.mark.sqlite_file
@pytest.mark.parametrize("logical_base", [False, True], ids=["fallback", "explicit"])
def test_core_commits_hook_prepared_from_real_raw_prefix(tmp_path: Path, monkeypatch, logical_base) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "prepared-keys.sqlite3")

    @function_tool
    def echo(text: str) -> str:
        return text

    runtime_agent = Agent(
        "prepared",
        "prepare a tool",
        model="scripted",
        tools=[echo],
        hooks=[BaseRuntimeHook()],
    )
    try:
        with SQLiteStore.standalone(str(tmp_path / "prepared.sqlite3")) as store:
            sessions = _sessions(provider, store)
            runtime = sessions.runtime(
                runtime_agent,
                RunConfig(
                    model_provider=ScriptedModelProvider.new(
                        "scripted",
                        "scripted",
                        [LLMResponse("", [ToolCall("call", "echo", {"text": "hello"})])],
                    ),
                    workspace="/workspace",
                ),
                session_id=SESSION_ID,
            )
            records = _capture_producer_records(monkeypatch, sessions, runtime, "op_prepared")
            _commit_from_core(store, sessions, records, logical_base=logical_base)

            assert any(record.record.kind == "op_prepared" for record in store.read(SESSION_ID).records)
    finally:
        provider.close()


@pytest.mark.sqlite_file
def test_core_preflight_rejects_changed_producer_prepared_identity_and_rolls_back(tmp_path: Path, monkeypatch) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "wrong-prepared-keys.sqlite3")

    @function_tool
    def echo(text: str) -> str:
        return text

    runtime_agent = Agent(
        "prepared",
        "prepare a tool",
        model="scripted",
        tools=[echo],
        hooks=[BaseRuntimeHook()],
    )
    try:
        with SQLiteStore.standalone(str(tmp_path / "wrong-prepared.sqlite3")) as store:
            sessions = _sessions(provider, store)
            runtime = sessions.runtime(
                runtime_agent,
                RunConfig(
                    model_provider=ScriptedModelProvider.new(
                        "scripted",
                        "scripted",
                        [LLMResponse("", [ToolCall("call", "echo", {"text": "hello"})])],
                    ),
                    workspace="/workspace",
                ),
                session_id=SESSION_ID,
            )
            records = _capture_producer_records(monkeypatch, sessions, runtime, "op_prepared")
            prepared = next(record for record in records if record.kind == "op_prepared")
            request = copy_json(prepared._payload["request"])
            request["id"] = "changed-call"
            changed = replace(
                prepared,
                payload=copy_json(prepared._payload)
                | {
                    "request": request,
                    "request_digest": digest(request),
                },
            )
            before_counts = _table_counts(store)
            before_usage = _kind_count(store, "usage_observed")

            driver, lease = _direct_driver(store, sessions)
            try:
                with pytest.raises(TransitionError, match="hook changed tool call identity"):
                    driver.commit([changed])
            finally:
                sessions.store.release(lease)

            assert _table_counts(store) == before_counts
            assert _kind_count(store, "usage_observed") == before_usage
            assert not any(record.record.kind == "op_prepared" for record in store.read(SESSION_ID).records)
    finally:
        provider.close()
