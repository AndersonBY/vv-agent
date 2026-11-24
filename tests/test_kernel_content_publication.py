"""Kernel record publication uses declared refs and transaction-owned links."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, SourceKind
from vv_agent.kernel._content_publication import (
    declared_record_content,
    link_declared_record_content,
    publish_content_container,
    publish_json,
    publish_record_content,
    publish_turn_started,
)
from vv_agent.kernel._content_runtime import SessionContentWriter as CoreSessionContentWriter
from vv_agent.kernel._content_sql import CoreSQLContentStore
from vv_agent.kernel._hydration import reference_digest
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import Record, SessionSpec, digest, make_record, record_identity
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import Conflict
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
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
        self.values: dict[ContentRef, object] = {}
        self.durable = _LinkStore()

    def put_json(self, value, *, labels=...):
        self.puts.append(deepcopy(value))
        ref = next(self.refs)
        self.values[ref] = deepcopy(value)
        return ref

    def get_json(self, ref):
        return deepcopy(self.values[ref])

    def remember(self, ref: ContentRef, value: object) -> None:
        self.values[ref] = deepcopy(value)

    def copy_json_from(self, source, ref):
        raise AssertionError("unexpected copy_json_from")

    def service_principal(self, subject_id, *, attributes=None):
        raise AssertionError("unexpected service_principal")

    def link_input(self, input_id, ref, slot):
        raise AssertionError("unexpected link_input")

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
    writer.remember(definition_ref, {"agent_name": "kernel", "task": {"user_prompt": "work"}})

    published = publish_turn_started(writer, definition_ref, budget_ref)
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


def test_producer_entries_preserve_only_typed_refs() -> None:
    typed = _ref("typed")
    typed_writer = _Writer([])
    assert publish_json(typed_writer, typed) == typed
    assert publish_content_container(typed_writer, typed, slot="attributes").ref == typed
    assert typed_writer.puts == []

    reference_shaped_json = typed.to_dict()
    json_ref = _ref("json")
    json_writer = _Writer([json_ref])
    published = publish_json(json_writer, reference_shaped_json)
    assert published == json_ref
    assert json_writer.puts == [reference_shaped_json]


def _plain_record(kind: str, payload: dict, *, suffix: str = "1", session_id: str = "session-a") -> Record:
    operation_id = f"operation/{suffix}"
    turn_id = "turn-a"
    return Record(
        record_id=record_identity(kind, session_id, turn_id, operation_id, 1, payload),
        kind=kind,
        session_id=session_id,
        turn_id=turn_id,
        operation_id=operation_id,
        attempt=1,
        payload=payload,
    )


def test_publisher_handles_planned_and_completed_refs_and_digests() -> None:
    request = {"messages": [{"role": "user", "content": "work"}]}
    budget_admission = {"tool": "search"}
    result = {"content": "done", "tool_calls": []}
    evidence = ["synchronous-return"]
    request_ref = _ref("request")
    message_ref = _ref("request-message")
    tool_ref = _ref("tool")
    budget_ref = _ref("budget-admission")
    result_ref = _ref("result")
    evidence_ref = _ref("evidence")
    usage_ref = _ref("usage")
    writer = _Writer([message_ref, request_ref, budget_ref, result_ref, evidence_ref])

    planned = _plain_record(
        "op_planned",
        {
            "op_kind": "tool",
            "purpose": None,
            "request": request,
            "request_digest": digest(request),
            "context_version": "7",
            "dependencies": [],
            "tool": tool_ref,
            "idempotency_key": "operation/1",
            "budget_admission": budget_admission,
            "not_before_ms": None,
            "provider_binding": "function",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
        suffix="planned",
    )
    completed = _plain_record(
        "op_completed",
        {
            "result": result,
            "result_digest": digest(result),
            "usage": usage_ref,
            "shared_state": None,
            "evidence": evidence,
            "execution_started": True,
            "context": "normal",
            "request_digest": digest(request),
            "provider_binding": "function",
            "policy_decision": None,
        },
        suffix="completed",
    )

    publication = publish_record_content(writer, (planned, completed))

    planned_payload, completed_payload = (record.payload for record in publication.records)
    assert planned_payload["request"] == request_ref.to_dict()
    assert planned_payload["tool"] == tool_ref.to_dict()
    assert planned_payload["budget_admission"] == budget_ref.to_dict()
    assert planned_payload["request_digest"] == reference_digest(request_ref)
    assert completed_payload["result"] == result_ref.to_dict()
    assert completed_payload["result_digest"] == reference_digest(result_ref)
    assert completed_payload["usage"] == usage_ref.to_dict()
    assert completed_payload["evidence"] == evidence_ref.to_dict()
    assert completed_payload["shared_state"] is None
    assert writer.puts == [
        request["messages"][0],
        request | {"messages": [message_ref.to_dict()]},
        budget_admission,
        result,
        evidence,
    ]
    assert [(link.record_id, link.slot, link.ref) for link in publication.links] == [
        (planned.record_id, "request", request_ref),
        (planned.record_id, "request.messages[0]", message_ref),
        (planned.record_id, "tool", tool_ref),
        (planned.record_id, "budget_admission", budget_ref),
        (completed.record_id, "result", result_ref),
        (completed.record_id, "usage", usage_ref),
        (completed.record_id, "evidence", evidence_ref),
    ]
    assert planned.payload["request"] == request
    assert completed.payload["result"] == result
    assert all(len(record.encode()) for record in publication.records)


def test_publisher_does_not_recursively_guess_unknown_slots() -> None:
    request_ref = _ref("request")
    tool_ref = _ref("tool")
    budget_ref = _ref("admission")
    record = _plain_record(
        "op_planned",
        {
            "op_kind": "model",
            "purpose": "primary",
            "request": request_ref,
            "request_digest": reference_digest(request_ref),
            "context_version": "1",
            "dependencies": [],
            "tool": tool_ref,
            "idempotency_key": None,
            "budget_admission": budget_ref,
            "not_before_ms": None,
            "provider_binding": "model",
            "consumed_unknowns": [],
            "policy_decision": None,
            "replacement": {"not": "a declared op_planned slot"},
        },
    )

    writer = _Writer([])
    writer.remember(request_ref, {"messages": []})
    publication = publish_record_content(writer, (record,))

    assert publication.records[0].payload["replacement"] == {"not": "a declared op_planned slot"}
    assert {link.slot for link in publication.links} == {"request", "tool", "budget_admission"}


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
        store.configure_input_admission(allow_unsigned=True)
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

        def fail_link(self, tenant_id, session_id, links):
            raise RuntimeError("forced link failure")

        monkeypatch.setattr(CoreSQLContentStore, "link_records", fail_link)

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


def test_op_publication_append_failure_rolls_back_refs_and_records(tmp_path, monkeypatch) -> None:
    path = tmp_path / "op-rollback.sqlite"
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {})
        with store.atomic() as tx:
            sessions._push_in_transaction(tx, "s", InboxItem("initial", "user", {"content": "work"}))
        agent = Agent("kernel", "Work.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="op-rollback-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None

        result = {"content": "done", "tool_calls": []}
        request = {"id": "call", "name": "tool", "arguments": {}}
        tool = {"name": "tool"}
        budget_admission = {}
        operation_id = "operation/transaction"
        plan_payload = {
            "op_kind": "tool",
            "purpose": None,
            "request": request,
            "request_digest": digest(request),
            "context_version": "1",
            "dependencies": [],
            "tool": tool,
            "idempotency_key": None,
            "budget_admission": budget_admission,
            "not_before_ms": None,
            "provider_binding": None,
            "consumed_unknowns": [],
            "policy_decision": None,
        }
        plan = Record(
            record_id=record_identity("op_planned", "s", turn_id, operation_id, 1, plan_payload),
            kind="op_planned",
            session_id="s",
            turn_id=turn_id,
            operation_id=operation_id,
            attempt=1,
            payload=plan_payload,
        )
        driver.commit([plan], guarded=True)
        published_plan = next(row.record for row in driver.records if row.record.kind == "op_planned")
        request_digest = published_plan._payload["request_digest"]
        started = driver.record(
            "op_started",
            {
                "dispatch_id": "dispatch",
                "authorization_version": "v1",
                "epoch": 1,
                "mode": "sync",
            },
            published_plan,
        )
        driver.commit([started], guarded=True)

        def fail_append(self, *args, **kwargs):
            raise RuntimeError("forced op append failure")

        before = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in ("sk_record", "sk_commit", "sk_session_key", "sk_content", "sk_content_link")
        }
        before_kinds = [row.record.kind for row in driver.records]
        monkeypatch.setattr(SQLSessionTx, "append", fail_append)
        completed = driver.record(
            "op_completed",
            {
                "result": result,
                "result_digest": digest(result),
                "usage": {},
                "shared_state": {},
                "evidence": ["synchronous-return"],
                "execution_started": True,
                "context": "normal",
                "request_digest": request_digest,
                "provider_binding": None,
                "policy_decision": None,
            },
            published_plan,
        )

        with pytest.raises(RuntimeError, match="forced op append failure"):
            driver.commit([completed], guarded=True)

        after = {
            table: store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "s"),
            )[0]
            for table in before
        }
        assert after == before
        assert [row.record.kind for row in driver.records] == before_kinds


def test_replay_reuses_committed_refs_and_repairs_a_missing_link(tmp_path, monkeypatch) -> None:
    path = tmp_path / "replay.sqlite"
    provider = LocalKeyProvider(MASTER_KEY)
    puts: list[object] = []
    put_json = CoreSessionContentWriter.put_json

    def record_put(self, value, *, labels=...):
        puts.append(deepcopy(value))
        return put_json(self, value) if labels is ... else put_json(self, value, labels=labels)

    monkeypatch.setattr(CoreSessionContentWriter, "put_json", record_put)
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
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
        lease = sessions.store.acquire("s", owner="replay-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        definition = {"agent_name": "kernel", "task": {"user_prompt": "work"}}
        budget = RunBudgetLimits(max_tool_calls=2).to_dict()
        turn_id = "s/turn/replay"
        record = Record(
            record_id=f"turn/{turn_id}/started",
            kind="turn_started",
            session_id="s",
            turn_id=turn_id,
            operation_id=None,
            attempt=None,
            payload={
                "input_ids": [],
                "definition": definition,
                "definition_digest": digest(definition),
                "handler_version": "1",
                "budget": budget,
                "binding": None,
                "generation": 0,
                "policy_digest": None,
                "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
                "tokenizer_binding": None,
                "environment": None,
            },
        )

        puts.clear()
        driver.commit([record], guarded=True)
        assert puts == [definition, budget]
        turn = driver.records[-1]
        commit_id = digest({"record_ids": [record.record_id], "consumed": []})

        def publication_counts() -> dict[str, int]:
            return {
                table: store._one(
                    f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                    ("default", "s"),
                )[0]
                for table in ("sk_record", "sk_commit", "sk_content")
            }

        counts = publication_counts()
        store._rows(
            "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND record_seq=%s AND slot=%s",
            ("default", "s", turn.seq, "definition"),
        )

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            replay = declared_record_content((turn.record,), ((turn.record.record_id, turn.seq),))
            link_declared_record_content(writer, replay)

        assert puts == [definition, budget]
        links = store._rows(
            "SELECT slot FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND record_seq=%s ORDER BY slot",
            ("default", "s", turn.seq),
        )
        assert [row[0] for row in links] == ["budget", "definition"]
        assert publication_counts() == counts

        changed_payload = deepcopy(record.payload)
        changed_payload["definition"] = {"agent_name": "kernel", "task": {"user_prompt": "different"}}
        with pytest.raises(Conflict, match="commit identity has different bytes"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
            publication = publish_record_content(writer, (replace(record, payload=changed_payload),))
            tx.append(
                "s",
                lease=lease,
                expected_seq=driver.records[-1].seq,
                commit_id=commit_id,
                records=publication.records,
            )
        assert publication_counts() == counts


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
        store.configure_input_admission(allow_unsigned=True)
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
        store.configure_input_admission(allow_unsigned=True)
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
