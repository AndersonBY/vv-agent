"""A2 intents share the existing SQL inbox and summary receipt ledger."""

from dataclasses import replace

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, DefaultAppServerHost
from vv_agent.session.content_runtime import wrap_session_content_store
from vv_agent.session.context_view import context_view
from vv_agent.session.kernel import drive
from vv_agent.types import LLMResponse, Message

from .app_server_s1_helpers import bind_development_content, create_s1_session, push_s1_input
from .helpers import fixture_content_reader, read_fixture_state
from .test_compaction import SUMMARY
from .test_recovery_matrix import records_of
from .test_summary_requests import configured as summary_runtime
from .test_summary_requests import summary_plans


def configured(store, database, steps, **kwargs):
    rt = summary_runtime(store, database, steps, **kwargs)
    rt = replace(rt, content_reader=fixture_content_reader(store, "s"))
    rt.config = replace(rt.config, initial_messages=[*rt.config.initial_messages, Message("user", "latest request")])
    return rt


def server_for(store, database, monkeypatch):
    provider = bind_development_content(store)
    trusted_store = wrap_session_content_store(store, provider)
    create_s1_session(
        trusted_store,
        "s",
        {"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}},
        workspace="/tmp",
        consumers=("events", "app_server", "traces"),
    )
    server = AppServer(
        store=store,
        content_key_provider=provider,
        host=DefaultAppServerHost(
            agent=Agent("test", "Be precise."), run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []))
        ),
    )
    server.kernel._on_close(provider.close)
    rt = configured(trusted_store, database, [LLMResponse("done")])
    monkeypatch.setattr(server.run_adapter, "recover", lambda *a, **k: None)
    return server, rt, trusted_store, store


def test_compaction_receipt_cut_reuses_summary_after_restart(store, database, monkeypatch):
    server, _, trusted_store, raw_store = server_for(store, database, monkeypatch)
    receipt = server.run_adapter.compact_thread(thread_id="s", action_id="compact")

    class Cut(BaseException):
        pass

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_completed":
            raise Cut()

    rt = configured(trusted_store, database, [LLMResponse(SUMMARY)], hook=cut)
    with pytest.raises(Cut):
        drive(trusted_store, "s", runtime=rt)
    assert len(summary_plans(raw_store)) == 1
    # No responses remain: a second summary request would fail.
    drive(trusted_store, "s", runtime=configured(trusted_store, database, []))
    assert len(records_of(raw_store, "context_compacted")) == 1
    assert len(summary_plans(raw_store)) == 1
    replay = server.run_adapter.compact_thread(thread_id="s", action_id="compact")
    assert replay == receipt | {"replayed": True}
    assert context_view(trusted_store, "s")["lastCompaction"]["summary"]
    server.kernel.close()


def test_compaction_no_history_is_noop_and_does_not_call_primary(store, database, monkeypatch):
    server, rt, trusted_store, raw_store = server_for(store, database, monkeypatch)
    server.run_adapter.compact_thread(thread_id="s", action_id="empty")
    rt.config = replace(rt.config, initial_messages=[])
    drive(trusted_store, "s", runtime=rt)
    assert not summary_plans(raw_store)
    assert not records_of(raw_store, "context_compacted")
    _, hydrated, _ = read_fixture_state(trusted_store, "s")
    completed = [
        row.record
        for row in hydrated
        if row.record.kind == "boundary_recorded" and row.record.payload["stage"] == "memory_completed"
    ]
    assert len(completed) == 1 and completed[0].payload["data"]["event"]["changed"] is False
    server.kernel.close()


def test_compaction_failure_keeps_history_and_reports_failed(store, database, monkeypatch):
    from vv_agent.app_server.item_mapper import map_run_event
    from vv_agent.session.projection import hydration_for_session_store, project_records

    server, _, trusted_store, raw_store = server_for(store, database, monkeypatch)
    server.run_adapter.compact_thread(thread_id="s", action_id="reject")
    # An invalid summary is a logged, non-replacing result.
    drive(
        trusted_store,
        "s",
        runtime=configured(trusted_store, database, [LLMResponse("not a structured summary")]),
    )
    assert not records_of(raw_store, "context_compacted")
    assert context_view(trusted_store, "s")["counts"]["modelVisibleHistoryMessages"] == 3
    events = project_records(trusted_store.read_state("s")[1], hydration=hydration_for_session_store(trusted_store, "s"))
    notices = [m for e in events for m, _ in map_run_event(e, thread_id="s", turn_id=e.run_id).additional_notifications]
    assert notices == ["contextCompaction/started", "contextCompaction/failed"]
    server.kernel.close()


def test_compaction_does_not_answer_a_parked_user_wait(store, database, monkeypatch):
    server, rt, trusted_store, raw_store = server_for(store, database, monkeypatch)
    rt.config = replace(rt.config, no_tool_policy="wait_user")
    push_s1_input(trusted_store, "s", "prompt", "go")
    drive(trusted_store, "s", runtime=rt)
    waiting = trusted_store.read_state("s")[0]
    tid = waiting.active_turn_id
    assert waiting.turns[tid].wait is not None
    server.run_adapter.compact_thread(thread_id="s", action_id="wait")
    drive(trusted_store, "s", runtime=rt)
    state = trusted_store.read_state("s")[0]
    assert state.turns[tid].wait == waiting.turns[tid].wait
    assert not summary_plans(raw_store)
    server.kernel.close()


def test_context_limits_match_selected_runtime_policy(store, database, monkeypatch):
    from vv_agent.model import ModelRef

    server, rt, trusted_store, _raw_store = server_for(store, database, monkeypatch)
    selected = ScriptedModelProvider.new("scripted", "small", [LLMResponse("selected")]).with_token_limits(64000, 8000)
    binding = selected.resolve(ModelRef.named("small"))
    rt.model_resolver = lambda request: (selected.llm, binding)
    push_s1_input(trusted_store, "s", "prompt", "go")
    push_s1_input(
        trusted_store,
        "s",
        "select",
        {"selection": {"model": "small", "model_settings": {"max_tokens": 1000}, "route": {}}},
        kind="model_selection",
    )
    drive(trusted_store, "s", runtime=rt)
    view = context_view(trusted_store, "s")
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
    server, _, trusted_store, raw_store = server_for(store, database, monkeypatch)
    future = 9_999_999_999_999
    push_s1_input(trusted_store, "s", "future", "accepted but not due", available_ms=future)
    assert not raw_store.peek_inbox("s")
    before = context_view(trusted_store, "s")
    assert before["counts"]["pendingInputs"] == 1
    assert before["estimates"]["parts"]["pendingInput"] > 0
    assert before["estimates"]["total"] is None
    monkeypatch.setattr(raw_store, "clock_sql", str(future + 1))
    assert raw_store.peek_inbox("s")
    assert context_view(trusted_store, "s") == before
    server.kernel.close()
