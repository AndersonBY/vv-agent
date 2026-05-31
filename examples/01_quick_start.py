#!/usr/bin/env python3
"""Minimal public SDK run with Agent and Runner."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner


def main() -> None:
    agent = Agent(
        name="assistant",
        instructions="Answer concisely. Use task_finish when the answer is ready.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    result = Runner.run_sync(
        agent,
        os.getenv("V_AGENT_EXAMPLE_PROMPT", "Summarize vv-agent in one paragraph."),
        run_config=config,
    )
    print(result.final_output)


if __name__ == "__main__":
    main()
