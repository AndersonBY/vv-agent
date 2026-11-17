"""The real turn-start producer publishes definition and budget as S1 refs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content import ContentRef, ContentSessionWriter
from vv_agent.kernel._content_publication import publish_turn_started
from vv_agent.kernel._hydration import reference_digest
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.records import InboxItem, SessionSpec, digest
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"k" * 32


@contextmanager
def _driver(path: Path, provider: LocalKeyProvider, *, attributes: dict | None = None) -> Iterator[tuple]:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", attributes or {})
        sessions.push("s", InboxItem("i", "user", {"content": "do work"}))
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
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
        assert payload["input_ids"] == ["i"]
        assert payload["handler_version"] == "1"
        assert payload["binding"] is None
        assert payload["generation"] == 0
        assert payload["requirements"] == {"level": 1, "handler_version": "1", "capabilities": []}
        assert payload["definition_digest"] == reference_digest(definition)
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


def test_plain_child_admission_is_published_once_in_the_turn_transaction(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / "child.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
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
