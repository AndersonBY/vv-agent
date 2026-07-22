#!/usr/bin/env python3
"""Enable a tool only for a specific run through the public FunctionTool API."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, RunContext, Runner, function_tool


def temp_tool_enabled(ctx: RunContext, _agent: Agent) -> bool:
    return bool((ctx.context or {}).get("enable_temp_tool"))


@function_tool(is_enabled=temp_tool_enabled)
def temporary_lookup(key: str) -> str:
    """Lookup a temporary value available only for selected runs."""
    return {"alpha": "temporary answer", "beta": "second value"}.get(key, "missing")


def main() -> None:
    agent = Agent(
        name="temporary-tool-demo",
        instructions="Use temporary_lookup when it is available, then call task_finish.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
        tools=[temporary_lookup],
    )
    enabled = os.getenv("VV_AGENT_TEMP_TOOL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        context={"enable_temp_tool": enabled},
    )
    prompt = os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Use temporary_lookup for key alpha if the tool exists.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
