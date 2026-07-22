#!/usr/bin/env python3
"""Retry a public Runner call when the runtime does not complete."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, AgentStatus, RunConfig, Runner


def main() -> None:
    agent = Agent(
        name="recoverable",
        instructions="If possible, finish with task_finish.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        max_cycles=int(os.getenv("VV_AGENT_EXAMPLE_MAX_CYCLES", "4")),
    )
    prompt = os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Finish a concise answer.")
    max_retries = int(os.getenv("VV_AGENT_EXAMPLE_MAX_RETRIES", "2"))

    result = None
    for attempt in range(1, max_retries + 2):
        result = Runner.run_sync(agent, prompt, run_config=config)
        print(f"[attempt {attempt}] {result.status.value}")
        if result.status == AgentStatus.COMPLETED:
            break
        prompt = f"Previous run status was {result.status.value}. Finish explicitly with task_finish."

    print(result.final_output if result else "no result")


if __name__ == "__main__":
    main()
