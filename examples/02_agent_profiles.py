#!/usr/bin/env python3
"""Select between reusable Agent profiles."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, ModelSettings, RunConfig, Runner

profiles = {
    "researcher": Agent(
        name="researcher",
        instructions="Collect facts first, then call task_finish with a sourced summary.",
        model="kimi-k2.6",
        model_settings=ModelSettings(temperature=0.2),
    ),
    "translator": Agent(
        name="translator",
        instructions="Translate the user input into natural Chinese and call task_finish.",
        model="MiniMax-M2.5",
        model_settings=ModelSettings(temperature=0.1),
    ),
}


def main() -> None:
    profile = os.getenv("V_AGENT_EXAMPLE_PROFILE", "researcher")
    agent = profiles.get(profile, profiles["researcher"])
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "minimax" if profile == "translator" else "moonshot")
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=backend,
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Explain what this package is for.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
