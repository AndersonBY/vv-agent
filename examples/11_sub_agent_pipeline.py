#!/usr/bin/env python3
"""Compose agents with agent.as_tool() and handoff()."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, handoff

researcher = Agent(
    name="researcher",
    instructions="Collect the relevant facts and finish with task_finish.",
    model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
)

writer = Agent(
    name="writer",
    instructions="Use research when useful, then write the final answer with task_finish.",
    model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    tools=[researcher.as_tool(name="research", description="Ask the researcher for facts.")],
)

triage = Agent(
    name="triage",
    instructions="Transfer writing requests to writer.",
    model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    handoffs=[handoff(agent=writer, description="Use for writing tasks.")],
)


def main() -> None:
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        max_cycles=8,
    )
    prompt = os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Write a short note about vv-agent sessions.")
    result = Runner.run_sync(triage, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
