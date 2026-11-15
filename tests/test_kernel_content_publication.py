"""Kernel record publication uses declared refs and transaction-owned links."""

from __future__ import annotations

from copy import deepcopy

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, SourceKind
from vv_agent.kernel._content_publication import (
    declared_record_content,
    link_declared_record_content,
    publish_turn_started,
)
from vv_agent.kernel._hydration import reference_digest
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import SessionSpec, digest, make_record
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider, SQLContentStore
from vv_agent.session.kernel import _Driver
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"k" * 32


def _ref(name: str) -> ContentRef:
    return ContentRef(
        content_id=f"content/{name}",
        mac=digest({"mac": name}),
        size=0,
        key_id="key/session",
        labels=ContentLabels((SourceKind.INTERNAL,), Classification.INTERNAL),
    )


class _LinkStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, ContentRef, str]] = []

    def link_record(
        self,
        tenant_id: str,
        session_id: str,
        record_seq: int,
        ref: ContentRef,
        slot: str,
    ) -> None:
        self.calls.append((tenant_id, session_id, record_seq, ref, slot))


class _Writer:
    tenant_id = "tenant-a"
    session_id = "session-a"

    def __init__(self, refs: list[ContentRef]) -> None:
        self.refs = iter(refs)
        self.puts: list[object] = []
        self.durable = _LinkStore()

    def put_json(self, value, *, labels=...):
        self.puts.append(deepcopy(value))
        return next(self.refs)

    def get_json(self, ref):
        raise AssertionError("unexpected get_json")

    def copy_json_from(self, source, ref):
        raise AssertionError("unexpected copy_json_from")

    def service_principal(self, subject_id, *, attributes=None):
        raise AssertionError("unexpected service_principal")

    def link_input_actor(self, input_id, ref):
        raise AssertionError("unexpected link_input_actor")

    def link_input_content(self, input_id, ref):
        raise AssertionError("unexpected link_input_content")

    def link_child_result(self, input_id, ref):
        raise AssertionError("unexpected link_child_result")

    def assert_input(self, input_id, ref, slot):
        raise AssertionError("unexpected assert_input")

    def assert_record(self, record_seq, ref, slot):
        raise AssertionError("unexpected assert_record")

    def create_session(self, spec_factory, *, consumers):
        raise AssertionError("unexpected create_session")


def test_legal_refs_are_preserved_and_only_declared_slots_are_linked() -> None:
    definition_ref = _ref("definition")
    budget_ref = _ref("budget")
    writer = _Writer([])

    published = publish_turn_started(writer, definition_ref.to_dict(), budget_ref)
    assert published.definition == definition_ref
    assert published.budget == budget_ref
    assert published.definition_digest == reference_digest(definition_ref)
    assert writer.puts == []

    record = make_record(
        "turn_started",
        session_id="session-a",
        turn_id="turn-a",
        payload={
            "input_ids": [],
            "definition": definition_ref.to_dict(),
            "definition_digest": published.definition_digest,
            "handler_version": "1",
            "budget": budget_ref.to_dict(),
            "binding": None,
            "generation": 0,
        },
    )
    links = declared_record_content((record,), ((record.record_id, 7),))
    link_declared_record_content(writer, links)

    assert [(link.record_seq, link.ref, link.slot) for link in links] == [
        (7, definition_ref, "definition"),
        (7, budget_ref, "budget"),
    ]
    assert writer.durable.calls == [
        ("tenant-a", "session-a", 7, definition_ref, "definition"),
        ("tenant-a", "session-a", 7, budget_ref, "budget"),
    ]


def test_plain_values_are_put_once_and_child_digest_is_checked() -> None:
    definition = {"agent_name": "kernel", "task": {"user_prompt": "work"}}
    budget = RunBudgetLimits(max_tool_calls=2).to_dict()
    definition_ref = _ref("definition")
    budget_ref = _ref("budget")
    writer = _Writer([definition_ref, budget_ref])

    published = publish_turn_started(
        writer,
        definition,
        budget,
        admission_definition_digest=digest(definition),
    )
    assert published.definition == definition_ref
    assert published.budget == budget_ref
    assert writer.puts == [definition, budget]

    rejected = _Writer([])
    with pytest.raises(ValueError, match="definition digest mismatch"):
        publish_turn_started(
            rejected,
            definition,
            budget,
            admission_definition_digest=digest({"different": True}),
        )
    assert rejected.puts == []


def _turn_payload(definition: ContentRef, budget: ContentRef) -> dict:
    return {
        "input_ids": [],
        "definition": definition.to_dict(),
        "definition_digest": reference_digest(definition),
        "handler_version": "1",
        "budget": budget.to_dict(),
        "binding": None,
        "generation": 0,
    }


def test_link_failure_rolls_back_record_key_content_and_links(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rollback.sqlite"
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {})
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="rollback-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        before = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in ("sk_record", "sk_commit", "sk_session_key", "sk_content", "sk_content_link")
        }

        def fail_link(self, tenant_id, session_id, record_seq, ref, slot):
            raise RuntimeError("forced link failure")

        monkeypatch.setattr(SQLContentStore, "link_record", fail_link)

        def prepare(tx):
            content = publish_turn_started(
                driver.content_writer(tx),
                {"agent_name": "kernel", "model_binding": {"endpoints": []}},
                RunBudgetLimits().to_dict(),
            )
            return [driver.record("turn_started", _turn_payload(content.definition, content.budget), tid="s/turn/rollback")]

        with pytest.raises(RuntimeError, match="forced link failure"):
            driver.commit([], guarded=True, prepare=prepare)

        after = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in before
        }
        assert after == before
        assert [row.record.kind for row in driver.records] == ["session_created"]


def test_missing_transaction_capability_rejects_record_after_append(tmp_path) -> None:
    path = tmp_path / "no-capability.sqlite"
    definition = _ref("definition")
    budget = _ref("budget")
    principal = {
        "tenant_id": "default",
        "subject_id": "local",
        "subject_kind": "service",
        "attributes": _ref("principal").to_dict(),
    }
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(
                SessionSpec(
                    "s",
                    principal,
                    _ref("workspace").to_dict(),
                    attributes=_ref("attributes").to_dict(),
                ),
                consumers=(),
            )
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = store.acquire("s", owner="no-capability", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))

        def prepare(tx):
            return [driver.record("turn_started", _turn_payload(definition, budget), tid="s/turn/no-capability")]

        with pytest.raises(ValueError, match="ContentSessionTx capability"):
            driver.commit([], guarded=True, prepare=prepare)
        assert [row.record.kind for row in store.read("s").records] == ["session_created"]


def test_missing_reader_rejects_turn_start_before_any_record(tmp_path) -> None:
    path = tmp_path / "no-reader.sqlite"
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {})
        sessions.push("s", InboxItem("i", "user", {"content": "work"}))
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="no-reader", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        before = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in ("sk_record", "sk_commit", "sk_content", "sk_content_link")
        }
        driver.content_reader = None

        with pytest.raises(RecordError, match="content hydration is unavailable"):
            _recovery_phase.run(driver, inputs.start_turn(driver))

        after = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in before
        }
        assert after == before
        assert [row.record.kind for row in driver.records] == ["session_created", "input_applied"]
