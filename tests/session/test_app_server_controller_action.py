from __future__ import annotations

from typing import Any

import pytest
from support.interactions import displayed_target

from vv_agent.app_server import AppServer
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.run_adapter import TurnResumeError
from vv_agent.session.app_server import SubmissionConflict

from .app_server_s1_helpers import bind_development_content


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_id", "😀" * 200),
        ("thread_id", "😀" * 200),
        ("turn_id", "😀" * 200),
    ],
)
def test_turn_action_rejects_identity_over_utf8_limit_without_store_access(
    field: str,
    value: str,
    surface,
) -> None:
    adapter = AppServer(store=surface.store).run_adapter
    kwargs: dict[str, Any] = {
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "action_id": "action-1",
        "action": {"kind": "cancel"},
    }
    kwargs[field] = value
    with pytest.raises(TurnResumeError, match="UTF-8 byte limit"):
        adapter.controller_action(connection_id="conn_1", **kwargs)


def test_turn_action_rejects_respond_unknown_fields_before_durable_lookup(surface) -> None:
    adapter = AppServer(store=surface.store).run_adapter
    with pytest.raises(TurnResumeError, match="respond action fields"):
        adapter.controller_action(
            connection_id="conn_1",
            thread_id="thread-1",
            turn_id="turn-1",
            action_id="action-1",
            action={"kind": "respond", "target": {}, "answer": {}, "secret": "x"},
        )


@pytest.mark.capability("app-server-turns")
@pytest.mark.parametrize("terminal_action", ["cancel", "abort"])
def test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items(tmp_path, store, terminal_action):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.types import LLMResponse, ToolCall

    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    content_key_provider = bind_development_content(store)
    agent = Agent("assistant", "Ask.", model="m")
    provider = ScriptedModelProvider.new(
        "test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose one"})]), LLMResponse("done")]
    )
    server = AppServer(
        store=store,
        content_key_provider=content_key_provider,
        host=DefaultAppServerHost(agent=agent, run_config=RunConfig(model_provider=provider, workspace=tmp_path)),
    )
    adapter = server.run_adapter
    try:
        thread_id = server.store.create_thread_receipt(input_id="thread_1", agent_key="default")["threadId"]
        started = adapter.start_turn(
            connection_id=None,
            thread_id=thread_id,
            input=[{"type": "text", "text": "go"}],
            input_id="prompt",
            expected_turn_id=None,
        )
        tid = started["turnId"]
        adapter.join()
        adapter.controller_action(
            connection_id="conn_1", thread_id=thread_id, turn_id=tid, action_id="suspend", action={"kind": "suspend"}
        )
        adapter.join()
        pending = server.kernel.store.peek_inbox(thread_id)
        before = server.kernel.store.read(thread_id).head_seq
        adapter.controller_action(
            connection_id="conn_1", thread_id=thread_id, turn_id=tid, action_id="suspend", action={"kind": "suspend"}
        )
        assert server.kernel.store.peek_inbox(thread_id) == pending
        with pytest.raises(SubmissionConflict, match="submission_conflict"):
            adapter.controller_action(
                connection_id="conn_1", thread_id=thread_id, turn_id=tid, action_id="suspend", action={"kind": "cancel"}
            )
        adapter.join()
        assert server.kernel.store.read(thread_id).head_seq == before
        assert server.kernel.store.peek_inbox(thread_id) == pending
        assert adapter.public_thread_status(thread_id)["status"] == "suspended"
        assert server.store.read_thread(thread_id).turns[0].result["waitReason"] == "suspended"
        reply = {
            "kind": "respond",
            "target": displayed_target(server.kernel, thread_id).to_dict(),
            "answer": {"kind": "text", "text": "value"},
        }
        adapter.controller_action(connection_id="conn_1", thread_id=thread_id, turn_id=tid, action_id="reply", action=reply)
        adapter.join()
        assert server.kernel.store.read_state(thread_id)[0].phase == "suspended"
        with server.kernel.store.atomic():
            adapter.controller_action(
                connection_id="conn_1", thread_id=thread_id, turn_id=tid, action_id="resume", action={"kind": "resume"}
            )
            adapter.controller_action(
                connection_id="conn_1",
                thread_id=thread_id,
                turn_id=tid,
                action_id="terminal",
                action={"kind": terminal_action},
            )
        adapter.join()
        state, records, _ = server.kernel.store.read_state(thread_id)
        assert state.turns[tid].ended
        assert [r.record._payload["status"] for r in records if r.record.kind == "turn_ended"] == [
            "cancelled" if terminal_action == "cancel" else "aborted"
        ]
        assert sum(r.record.kind == "op_started" and r.record._payload["mode"] == "sync" for r in records) == 1
        assert {r.record._payload["input"]["kind"] for r in records if r.record.kind == "input_applied"} == {"user", "control"}
    finally:
        adapter.join()
        server.kernel.close()
        try:
            assert store.connection.execute("SELECT 1").fetchone() == (1,)
        finally:
            content_key_provider.close()
