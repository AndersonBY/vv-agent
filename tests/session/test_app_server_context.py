"""A2 intents share the existing SQL inbox and summary receipt ledger."""

from dataclasses import replace

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, DefaultAppServerHost
from vv_agent.session.context_view import context_view
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.types import LLMResponse, Message

from .test_compaction import SUMMARY
from .test_recovery_matrix import records_of
from .test_summary_requests import configured as summary_runtime
from .test_summary_requests import summary_plans


def configured(database, steps, **kwargs):
    rt = summary_runtime(database, steps, **kwargs)
    rt.config = replace(rt.config, initial_messages=[*rt.config.initial_messages, Message("user", "latest request")])
    return rt


def server_for(store, database, monkeypatch):
    with store.atomic() as tx:
        tx.create(
            SessionSpec("s", "test", "/tmp", attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}}),
            consumers=("events", "app_server", "traces"),
        )
    rt = configured(database, [LLMResponse("done")])
    server = AppServer(
        store=store,
        host=DefaultAppServerHost(
            agent=Agent("test", "Be precise."), run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []))
        ),
    )
    monkeypatch.setattr(server.run_adapter, "recover", lambda *a, **k: None)
    return server, rt


def test_compaction_receipt_cut_reuses_summary_after_restart(store, database, monkeypatch):
    server, rt = server_for(store, database, monkeypatch)
    receipt = server.run_adapter.compact_thread(thread_id="s", action_id="compact")

    class Cut(BaseException):
        pass

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_completed" and record.payload["result"].get("content") == SUMMARY:
            raise Cut()

    rt = configured(database, [LLMResponse(SUMMARY)], hook=cut)
    with pytest.raises(Cut):
        drive(store, "s", runtime=rt)
    assert len(summary_plans(store)) == 1
    # No responses remain: a second summary request would fail.
    drive(store, "s", runtime=configured(database, []))
    assert len(records_of(store, "context_compacted")) == 1
    assert len(summary_plans(store)) == 1
    replay = server.run_adapter.compact_thread(thread_id="s", action_id="compact")
    assert replay == receipt | {"replayed": True}
    assert context_view(store, "s")["lastCompaction"]["summary"]
    server.kernel.close()


def test_compaction_no_history_is_noop_and_does_not_call_primary(store, database, monkeypatch):
    server, rt = server_for(store, database, monkeypatch)
    server.run_adapter.compact_thread(thread_id="s", action_id="empty")
    rt.config = replace(rt.config, initial_messages=[])
    drive(store, "s", runtime=rt)
    assert not summary_plans(store)
    assert not records_of(store, "context_compacted")
    completed = [r for r in records_of(store, "boundary_recorded") if r.payload["stage"] == "memory_completed"]
    assert len(completed) == 1 and completed[0].payload["data"]["event"]["changed"] is False
    server.kernel.close()


def test_compaction_failure_keeps_history_and_reports_failed(store, database, monkeypatch):
    from vv_agent.app_server.item_mapper import map_run_event
    from vv_agent.session.projection import project_records

    server, _ = server_for(store, database, monkeypatch)
    server.run_adapter.compact_thread(thread_id="s", action_id="reject")
    # An invalid summary is a logged, non-replacing result.
    drive(store, "s", runtime=configured(database, [LLMResponse("not a structured summary")]))
    assert not records_of(store, "context_compacted")
    assert context_view(store, "s")["counts"]["modelVisibleHistoryMessages"] == 3
    events = project_records(store.read_state("s")[1])
    notices = [m for e in events for m, _ in map_run_event(e, thread_id="s", turn_id=e.run_id).additional_notifications]
    assert notices == ["contextCompaction/started", "contextCompaction/failed"]
    server.kernel.close()


def test_compaction_does_not_answer_a_parked_user_wait(store, database, monkeypatch):
    server, rt = server_for(store, database, monkeypatch)
    rt.config = replace(rt.config, no_tool_policy="wait_user")
    with store.atomic() as tx:
        tx.push("s", InboxItem("prompt", "user", {"content": "go"}))
    drive(store, "s", runtime=rt)
    waiting = store.read_state("s")[0]
    tid = waiting.active_turn_id
    assert waiting.turns[tid].wait is not None
    server.run_adapter.compact_thread(thread_id="s", action_id="wait")
    drive(store, "s", runtime=rt)
    state = store.read_state("s")[0]
    assert state.turns[tid].wait == waiting.turns[tid].wait
    assert not summary_plans(store)
    server.kernel.close()


def test_context_limits_match_selected_runtime_policy(store, database, monkeypatch):
    from vv_agent.model import ModelRef

    server, rt = server_for(store, database, monkeypatch)
    selected = ScriptedModelProvider.new("scripted", "small", [LLMResponse("selected")]).with_token_limits(64000, 8000)
    binding = selected.resolve(ModelRef.named("small"))
    rt.model_resolver = lambda request: (selected.llm, binding)
    with store.atomic() as tx:
        tx.push("s", InboxItem("prompt", "user", {"content": "go"}))
        tx.push(
            "s",
            InboxItem(
                "select",
                "model_selection",
                {
                    "selection": {
                        "model": "small",
                        "model_settings": {"max_tokens": 1000},
                        "route": {},
                    }
                },
            ),
        )
    drive(store, "s", runtime=rt)
    view = context_view(store, "s")
    assert view["limits"] == {
        "contextWindow": 64000,
        "maxOutputTokens": 8000,
        "runtimeContextWindow": 64000,
        "outputReserve": 1000,
        "outputReserveSource": "model_settings",
        "compactionThreshold": 50000,
    }
    server.kernel.close()


def test_future_pending_input_version_does_not_depend_on_database_clock(store, database, monkeypatch):
    server, _ = server_for(store, database, monkeypatch)
    future = 9_999_999_999_999
    with store.atomic() as tx:
        tx.push("s", InboxItem("future", "user", {"content": "accepted but not due"}, available_ms=future))
    assert not store.peek_inbox("s")
    before = context_view(store, "s")
    assert before["counts"]["pendingInputs"] == 1
    assert before["estimates"]["parts"]["pendingInput"] > 0
    assert before["estimates"]["total"] is None
    monkeypatch.setattr(store, "clock_sql", str(future + 1))
    assert store.peek_inbox("s")
    assert context_view(store, "s") == before
    server.kernel.close()
