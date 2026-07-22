#!/usr/bin/env python3
"""Advanced public SDK run with streaming and ThreadBackend."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner
from vv_agent.runtime.backends.thread import ThreadBackend


def print_event(event) -> None:
    if event.type == "assistant_delta":
        print(event.delta, end="", flush=True)
    elif event.type in {"tool_call_started", "tool_call_completed", "run_completed"}:
        print(f"\n[{event.type}] {event.to_dict()}", flush=True)


def main() -> None:
    agent = Agent(
        name="advanced",
        instructions="Stream progress when possible and finish with task_finish.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        execution_backend=ThreadBackend(max_workers=2),
        stream=print_event,
    )
    result = Runner.run_sync(agent, os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Explain ThreadBackend."), run_config=config)
    print("\nfinal:", result.final_output)


if __name__ == "__main__":
    main()
