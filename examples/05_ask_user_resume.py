#!/usr/bin/env python3
"""Demonstrate the public tool approval WAIT_USER path."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, ToolPolicy, function_tool


@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """Delete a workspace file after host approval."""
    target = Path(path)
    target.unlink(missing_ok=True)
    return f"deleted {path}"


def main() -> None:
    agent = Agent(
        name="operator",
        instructions="Use tools when needed. Finish with task_finish.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
        tools=[delete_file],
    )
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        tool_policy=ToolPolicy(approval=os.getenv("VV_AGENT_EXAMPLE_APPROVAL", "default")),
    )
    result = Runner.run_sync(
        agent,
        os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Delete obsolete.txt if it exists."),
        run_config=config,
    )
    print(result.status.value, result.final_output)
    for event in result.events:
        if event.type == "approval_requested":
            print(event.to_dict())


if __name__ == "__main__":
    main()
