from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest

import vv_agent.interactive as interactive_module
from vv_agent import Agent, AgentSessionOptions, ScriptedModelProvider, create_agent_session
from vv_agent.events import DiagnosticEvent
from vv_agent.interactive import AgentSessionEventGapError, AgentSessionEventStreamClosed
from vv_agent.types import LLMResponse


def _session(tmp_path: Path, provider=None):
    return create_agent_session(
        agent=Agent("inline", "Answer."),
        options=AgentSessionOptions(
            model_provider=provider or ScriptedModelProvider.from_callback("test", "m", lambda request: LLMResponse("done")),
            workspace=tmp_path,
        ),
        session_id="interactive-lifecycle",
    )


def _background_started_event(session_id: str) -> DiagnosticEvent:
    return DiagnosticEvent(
        run_id="run-background",
        trace_id="trace-background",
        level="debug",
        code="tool_result",
        details={
            "tool_name": "bash",
            "status": "running",
            "metadata": {"status": "running", "session_id": session_id},
        },
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

    def respond(request):
        prompts.append(next(m.content for m in reversed(request.messages) if m.role == "user"))
        return LLMResponse("done")

    session = _session(tmp_path, ScriptedModelProvider.from_callback("test", "m", respond))
    events: list[tuple[str, dict[str, Any]]] = []
    session.subscribe(lambda event, payload: events.append((event, payload)))
    session._session_event_handler(_background_started_event("bg_contract"))

    manager.finish("bg_contract")

    assert session.state().pending_steering == 0
    terminal = next(payload for event, payload in events if event == "background_command_terminal")
    assert terminal["background_session_id"] == "bg_contract"
    assert terminal["queued_to_session"] is False
    assert terminal["queued_to_running_session"] is False
    assert manager.unsubscribe_count == 1

    session.prompt(terminal["notification_message"], auto_follow_up=False)

    assert prompts == [
        "System notification: background command bg_contract completed.\nCommand: printf bridge-ready\nSummary: bridge-ready"
    ]
    assert session.state().pending_steering == 0


def test_close_aborts_active_handle_closes_stream_and_rejects_new_work(tmp_path):
    active, release = threading.Event(), threading.Event()

    def respond(request):
        active.set()
        release.wait(2)
        return LLMResponse("late answer")

    session = _session(tmp_path, ScriptedModelProvider.from_callback("test", "m", respond))
    stream = session.subscribe(capacity=128)
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(session.prompt("run", auto_follow_up=False)))
    try:
        worker.start()
        assert active.wait(2)
        handle = session.active_run_handle
        assert handle is not None
        assert session.close() is True
        worker.join(2)
        assert not worker.is_alive()
        assert session.closed and session.active_run_handle is None
        assert handle.done()
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
    finally:
        release.set()
        worker.join(2)
        session.driver.close()


def test_dropping_session_unsubscribes_background_completion_listener(monkeypatch, tmp_path: Path) -> None:
    manager = _BackgroundManager()
    monkeypatch.setattr(interactive_module, "background_session_manager", manager)
    session = _session(tmp_path)
    session._session_event_handler(_background_started_event("bg_drop"))
    reference = weakref.ref(session)

    del session
    gc.collect()

    assert reference() is None
    assert manager.unsubscribe_count == 1
    assert "bg_drop" not in manager.listeners
