from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest

import vv_agent.interactive as interactive_module
from vv_agent import AgentSessionRun, AgentStatus, InteractiveAgentDefinition, Message, create_agent_session
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.interactive import AgentSessionEventGapError, AgentSessionEventStreamClosed
from vv_agent.types import AgentResult


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="key", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
    )


def _completed_run(prompt: str) -> AgentSessionRun:
    return AgentSessionRun(
        agent_name="inline",
        result=AgentResult(
            status=AgentStatus.COMPLETED,
            messages=[Message(role="user", content=prompt), Message(role="assistant", content="done")],
            cycles=[],
            final_answer="done",
            shared_state={"todo_list": []},
        ),
        resolved=_resolved(),
    )


def _session(tmp_path: Path, execute_run=None):
    return create_agent_session(
        execute_run=execute_run or (lambda **kwargs: _completed_run(kwargs["prompt"])),
        session_id="interactive-lifecycle",
        agent_name="inline",
        definition=InteractiveAgentDefinition(description="assistant", model="test-model"),
        workspace=tmp_path,
    )


def test_pull_subscribers_are_independent_and_report_bounded_gaps(tmp_path: Path) -> None:
    session = _session(tmp_path)
    fast = session.subscribe(capacity=2)
    slow = session.subscribe(capacity=2)

    session.steer("one")
    assert fast.recv(timeout=0).payload["prompt"] == "one"
    session.steer("two")
    assert fast.recv(timeout=0).payload["prompt"] == "two"
    session.steer("three")
    assert fast.recv(timeout=0).payload["prompt"] == "three"

    with pytest.raises(AgentSessionEventGapError) as lagged:
        slow.recv(timeout=0)
    assert lagged.value.missed == 1
    assert [slow.recv(timeout=0).payload["prompt"], slow.recv(timeout=0).payload["prompt"]] == [
        "two",
        "three",
    ]


def test_callback_failure_isolated_from_other_listeners_and_pull_subscribers(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session(tmp_path)
    pulled = session.subscribe(capacity=4)
    received: list[str] = []

    def fail(_event: str, _payload: dict[str, Any]) -> None:
        raise RuntimeError("listener failed")

    session.subscribe(fail)
    session.subscribe(lambda event, _payload: received.append(event))

    session.follow_up("next")

    assert received == ["session_follow_up_queued"]
    assert pulled.recv(timeout=0).event == "session_follow_up_queued"
    assert "Agent session listener failed" in caplog.text


class _BackgroundManager:
    def __init__(self) -> None:
        self.listeners: dict[str, Any] = {}
        self.unsubscribe_count = 0

    def subscribe(self, session_id: str, listener):
        self.listeners[session_id] = listener

        def unsubscribe() -> None:
            self.unsubscribe_count += 1
            self.listeners.pop(session_id, None)

        return unsubscribe

    def finish(self, session_id: str) -> None:
        self.listeners[session_id](
            {
                "status": "completed",
                "session_id": session_id,
                "command": "printf bridge-ready",
                "output": "bridge-ready",
                "exit_code": 0,
            }
        )


def test_background_completion_emits_idle_notification_for_host_resume(monkeypatch, tmp_path: Path) -> None:
    manager = _BackgroundManager()
    monkeypatch.setattr(interactive_module, "background_session_manager", manager)
    prompts: list[str] = []

    def execute_run(**kwargs):
        prompts.append(kwargs["prompt"])
        return _completed_run(kwargs["prompt"])

    session = _session(tmp_path, execute_run)
    events: list[tuple[str, dict[str, Any]]] = []
    session.subscribe(lambda event, payload: events.append((event, payload)))
    session._session_log_handler(
        "tool_result",
        {
            "tool_name": "bash",
            "status": "running",
            "metadata": {"status": "running", "session_id": "bg_contract"},
        },
    )

    manager.finish("bg_contract")

    assert session.state().pending_steering == 0
    terminal = next(payload for event, payload in events if event == "background_command_terminal")
    assert terminal["background_session_id"] == "bg_contract"
    assert terminal["queued_to_session"] is False
    assert terminal["queued_to_running_session"] is False
    assert manager.unsubscribe_count == 1

    session.prompt(terminal["notification_message"], auto_follow_up=False)

    assert prompts == [
        "System notification: background command bg_contract completed.\n"
        "Command: printf bridge-ready\n"
        "Summary: bridge-ready"
    ]
    assert session.state().pending_steering == 0


class _ActiveHandle:
    def __init__(self) -> None:
        self.cancel_reasons: list[str] = []
        self.attached = False

    def attach_controller(self, _controller) -> None:
        self.attached = True

    def detach_controller(self, _controller) -> None:
        self.attached = False

    def cancel(self, reason: str = "") -> bool:
        self.cancel_reasons.append(reason)
        return True


def test_close_aborts_active_handle_closes_stream_and_rejects_new_work(tmp_path: Path) -> None:
    active = threading.Event()
    handle = _ActiveHandle()

    def execute_run(**kwargs):
        kwargs["active_handle_callback"](handle)
        active.set()
        token = kwargs["cancellation_token"]
        cancelled = threading.Event()
        token.on_cancel(cancelled.set)
        assert cancelled.wait(timeout=3)
        return _completed_run(kwargs["prompt"])

    session = _session(tmp_path, execute_run)
    stream = session.subscribe(capacity=16)
    worker = threading.Thread(target=lambda: session.prompt("run", auto_follow_up=False), daemon=True)
    worker.start()
    assert active.wait(timeout=3)

    assert session.close() is True
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert session.closed is True
    assert session.active_run_handle is None
    assert handle.cancel_reasons == ["interactive session closed"]
    assert handle.attached is False
    assert session.close() is False
    with pytest.raises(RuntimeError, match="closed"):
        session.prompt("after close")

    observed = []
    while True:
        try:
            observed.append(stream.recv(timeout=0).event)
        except AgentSessionEventStreamClosed:
            break
    assert "session_active_run_handle_changed" in observed
    assert "session_closed" in observed


def test_dropping_session_unsubscribes_background_completion_listener(monkeypatch, tmp_path: Path) -> None:
    manager = _BackgroundManager()
    monkeypatch.setattr(interactive_module, "background_session_manager", manager)
    session = _session(tmp_path)
    session._session_log_handler(
        "tool_result",
        {
            "tool_name": "bash",
            "status": "running",
            "metadata": {"status": "running", "session_id": "bg_drop"},
        },
    )
    reference = weakref.ref(session)

    del session
    gc.collect()

    assert reference() is None
    assert manager.unsubscribe_count == 1
    assert "bg_drop" not in manager.listeners
