from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from support.interactions import displayed_target

from vv_agent.app_server import AppServer
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.run_adapter import TurnResumeError
from vv_agent.session.app_server import SubmissionConflict

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"


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


@pytest.mark.parametrize("terminal_action", ["cancel", "abort"])
def test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items(tmp_path, terminal_action):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.session.kernel import drive
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse, ToolCall

    kernel = SessionDriver()
    try:
        agent = Agent("assistant", "Ask.", model="m")
        provider = ScriptedModelProvider.new(
            "test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose one"})]), LLMResponse("done")]
        )
        server = AppServer(
            host=DefaultAppServerHost(agent=agent, run_config=RunConfig(model_provider=provider, workspace=tmp_path)),
            store=kernel.store,
        )
        thread = server.store.read_thread(
            server.store.create_thread_receipt(input_id="thread_1", agent_key="default")["threadId"]
        ).thread
        handle = kernel.start(
            thread.thread_id,
            agent,
            RunConfig(model_provider=provider, workspace=tmp_path),
            {"text": "go", "app_server": {"owner": "disconnected", "input": [], "metadata": {}}},
        )
        handle.result()
        cast(Any, server.store).runtimes[thread.thread_id] = handle.runtime
        adapter = server.run_adapter
        tid = handle.run_id
        adapter.controller_action(
            connection_id="conn_1", thread_id=thread.thread_id, turn_id=tid, action_id="suspend", action={"kind": "suspend"}
        )
        server.run_adapter.join()
        pending = kernel.store.peek_inbox(thread.thread_id)
        before = kernel.store.read(thread.thread_id).head_seq
        adapter.controller_action(
            connection_id="conn_1", thread_id=thread.thread_id, turn_id=tid, action_id="suspend", action={"kind": "suspend"}
        )
        assert kernel.store.peek_inbox(thread.thread_id) == pending
        with pytest.raises(SubmissionConflict, match="submission_conflict"):
            adapter.controller_action(
                connection_id="conn_1", thread_id=thread.thread_id, turn_id=tid, action_id="suspend", action={"kind": "cancel"}
            )
        server.run_adapter.join()
        assert kernel.store.read(thread.thread_id).head_seq == before
        assert kernel.store.peek_inbox(thread.thread_id) == pending
        drive(kernel.store, thread.thread_id, runtime=handle.runtime, _one_turn=True)
        assert adapter.public_thread_status(thread.thread_id)["status"] == "suspended"
        assert server.store.read_thread(thread.thread_id).turns[0].result["waitReason"] == "suspended"
        reply = {
            "kind": "respond",
            "target": displayed_target(kernel, thread.thread_id).to_dict(),
            "answer": {"kind": "text", "text": "value"},
        }
        adapter.controller_action(
            connection_id="conn_1", thread_id=thread.thread_id, turn_id=tid, action_id="reply", action=reply
        )
        drive(kernel.store, thread.thread_id, runtime=handle.runtime, _one_turn=True)
        assert kernel.store.read_state(thread.thread_id)[0].phase == "suspended"
        with kernel.store.atomic():
            adapter.controller_action(
                connection_id="conn_1", thread_id=thread.thread_id, turn_id=tid, action_id="resume", action={"kind": "resume"}
            )
            adapter.controller_action(
                connection_id="conn_1",
                thread_id=thread.thread_id,
                turn_id=tid,
                action_id="terminal",
                action={"kind": terminal_action},
            )
        server.run_adapter.join()
        state, records, _ = kernel.store.read_state(thread.thread_id)
        assert state.turns[tid].ended
        assert [r.record._payload["status"] for r in records if r.record.kind == "turn_ended"] == [
            "cancelled" if terminal_action == "cancel" else "aborted"
        ]
        assert sum(r.record.kind == "op_started" and r.record._payload["mode"] == "sync" for r in records) == 1
        assert {r.record._payload["input"]["kind"] for r in records if r.record.kind == "input_applied"} == {"user", "control"}
    finally:
        server.run_adapter.join()
        kernel.close()
