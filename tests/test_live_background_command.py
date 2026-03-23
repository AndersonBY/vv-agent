from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.constants import ASK_USER_TOOL_NAME, BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME
from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.types import AgentStatus, Message

pytestmark = pytest.mark.live


def _resolve_live_settings_file() -> Path:
    return Path(
        os.getenv(
            "V_AGENT_LOCAL_SETTINGS",
            Path(__file__).resolve().parents[1] / "local_settings.py",
        )
    )


def _summarize_events(events: list[tuple[str, dict[str, Any]]]) -> str:
    if not events:
        return "no session events captured"

    lines: list[str] = []
    for event, payload in events[-20:]:
        lines.append(
            str(
                {
                    "event": event,
                    "tool_name": payload.get("tool_name"),
                    "status": payload.get("status"),
                    "session_id": payload.get("session_id"),
                    "queued_to_running_session": payload.get("queued_to_running_session"),
                    "final_answer": payload.get("final_answer"),
                    "wait_reason": payload.get("wait_reason"),
                    "error": payload.get("error"),
                }
            )
        )
    return "\n".join(lines)


def _select_live_backend_model(settings_file: Path) -> tuple[str, str]:
    explicit_backend = str(os.getenv("V_AGENT_LIVE_BACKEND") or "").strip()
    explicit_model = str(os.getenv("V_AGENT_LIVE_MODEL") or "").strip()
    if explicit_backend or explicit_model:
        backend = explicit_backend or "openai"
        model = explicit_model or "gpt-4o-mini"
        return backend, model

    candidates = [
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-4o"),
        ("gemini", "gemini-2.5-flash"),
        ("deepseek", "deepseek-chat"),
        ("qwen", "qwen2.5-72b-instruct"),
        ("moonshot", "kimi-k2.5"),
    ]
    failures: list[str] = []

    for backend, model in candidates:
        try:
            llm, resolved = build_openai_llm_from_local_settings(
                settings_file,
                backend=backend,
                model=model,
                timeout_seconds=30,
            )
            response = llm.complete(
                model=resolved.model_id,
                messages=[Message(role="user", content="Reply with exactly one word: pong")],
                tools=[],
            )
            if str(response.content or "").strip():
                return backend, model
            failures.append(f"{backend}/{model}: empty response")
        except Exception as exc:
            failures.append(f"{backend}/{model}: {type(exc).__name__}: {exc}")

    raise AssertionError(
        "No usable live backend/model pair found in local_settings.py.\n"
        + "\n".join(failures)
    )


def test_live_background_timeout_handoff_auto_notifies_session(tmp_path: Path) -> None:
    if os.getenv("V_AGENT_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set V_AGENT_RUN_LIVE_TESTS=1 to run live integration tests")

    settings_file = _resolve_live_settings_file()
    if not settings_file.exists():
        pytest.skip(f"Live settings file not found: {settings_file}")

    backend, model = _select_live_backend_model(settings_file)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=tmp_path,
            auto_discover_resources=False,
        ),
        agent=AgentDefinition(
            description="Deterministic live integration test agent.",
            system_prompt=(
                "You are running a deterministic integration test.\n"
                "Follow this protocol exactly.\n"
                "1. On your first action, call `bash` exactly once with "
                "`command=\"sleep 1.2 && echo BG_DONE\"` and `timeout=1`.\n"
                "2. Do not set `run_in_background`.\n"
                "3. Never call `check_background_command`.\n"
                "4. Do not call `task_finish` until you receive a system notification "
                "that the background command completed.\n"
                "5. Before that notification arrives, reply with exactly `WAITING` and no tool calls.\n"
                "6. After that notification arrives, call `task_finish` with exactly "
                "`background observed`.\n"
                "Do not deviate from this protocol."
            ),
            model=model,
            backend=backend,
            language="en-US",
            max_cycles=10,
            no_tool_policy="continue",
            allow_interruption=True,
            use_workspace=False,
            enable_todo_management=False,
            agent_type="computer",
            exclude_tools=[ASK_USER_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME],
        ),
    )

    session = client.create_session(workspace=tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []
    session.subscribe(lambda event, payload: events.append((event, dict(payload))))

    run = session.prompt("Run the timeout-handoff background notification test.", auto_follow_up=False)
    event_summary = f"backend={backend} model={model}\n{_summarize_events(events)}"

    assert run.result.status == AgentStatus.COMPLETED, event_summary
    assert "background observed" in str(run.result.final_answer or "").lower(), event_summary

    bash_background_events = [
        payload
        for event, payload in events
        if event == "tool_result"
        and payload.get("tool_name") == BASH_TOOL_NAME
        and isinstance(payload.get("metadata"), dict)
        and payload["metadata"].get("transitioned_to_background") is True
    ]
    assert bash_background_events, event_summary

    background_events = [
        payload
        for event, payload in events
        if event == "background_command_completed"
    ]
    assert background_events, event_summary
    assert background_events[0].get("queued_to_running_session") is True, event_summary
    assert "BG_DONE" in str(background_events[0].get("output", "")), event_summary

    steer_events = [
        payload
        for event, payload in events
        if event == "session_steer_queued"
    ]
    assert steer_events, event_summary

    check_events = [
        payload
        for event, payload in events
        if event == "tool_result" and payload.get("tool_name") == CHECK_BACKGROUND_COMMAND_TOOL_NAME
    ]
    assert not check_events, event_summary
