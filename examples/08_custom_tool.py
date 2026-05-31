#!/usr/bin/env python3
"""Create a typed custom tool with @function_tool."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, function_tool


@function_tool
def create_ticket(title: str, priority: str = "normal") -> dict[str, str]:
    """Create a support ticket."""
    return {"ticket_id": "TCK-1001", "title": title, "priority": priority}


def main() -> None:
    agent = Agent(
        name="support",
        instructions="Create tickets when the user reports work. Finish with task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
        tools=[create_ticket],
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Create a high priority ticket for login failures.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
