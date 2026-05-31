#!/usr/bin/env python3
"""Public SDK usage with Agent, Runner, RunConfig, sessions, and typed events."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from vv_agent import Agent, MemorySession, ModelSettings, RunConfig, Runner, function_tool
from vv_agent.events import RunEvent


@function_tool
def save_note(path: str, content: str) -> str:
    """Save a short note into the current working directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"saved {target}"


def print_event(event: RunEvent) -> None:
    if event.type == "assistant_delta":
        print(event.to_dict().get("delta", ""), end="", flush=True)
    elif event.type in {"tool_finished", "run_completed", "run_failed"}:
        print(f"\n[{event.type}] {event.to_dict()}", flush=True)


agents = {
    "default": Agent(
        name="planner",
        instructions="你是任务规划 Agent, 先拆任务, 再逐步执行并维护 todo.",
        model="kimi-k2.6",
        model_settings=ModelSettings(temperature=0.2),
        tools=[save_note],
    ),
    "translator": Agent(
        name="translator",
        instructions="你是专业翻译 Agent, 按段翻译并持续写入目标文件.",
        model="MiniMax-M2.5",
        model_settings=ModelSettings(temperature=0.1),
        tools=[save_note],
    ),
}


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}
    agent_name = os.getenv("V_AGENT_EXAMPLE_AGENT", "default")
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "先拆分任务, 再逐步完成并汇报")
    max_cycles = int(os.getenv("V_AGENT_EXAMPLE_MAX_CYCLES", "10"))
    session_id = os.getenv("V_AGENT_EXAMPLE_SESSION_ID", "").strip()

    workspace.mkdir(parents=True, exist_ok=True)

    agent = agents.get(agent_name, agents["default"])
    if agent_name == "translator" and os.getenv("V_AGENT_EXAMPLE_BACKEND") is None:
        backend = "minimax"

    config = RunConfig(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        max_cycles=max(max_cycles, 1),
        stream=print_event if verbose else None,
        session=MemorySession(session_id) if session_id else None,
    )

    try:
        result = Runner.run_sync(agent, prompt, run_config=config)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Error running agent: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
