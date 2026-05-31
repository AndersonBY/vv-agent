#!/usr/bin/env python3
"""Expose available skills through Agent metadata."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner


def main() -> None:
    available_skills = [
        {"name": "code-review", "description": "Review code and list concrete findings."},
        {"name": "release-notes", "description": "Summarize user-facing changes."},
    ]
    agent = Agent(
        name="skill-router",
        instructions="Choose an available skill when it helps, then call task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
        metadata={"available_skills": available_skills},
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Pick the best skill for reviewing a patch.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
