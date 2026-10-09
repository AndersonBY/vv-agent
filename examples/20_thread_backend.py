#!/usr/bin/env python3
"""Non-blocking execution through the current public RunHandle."""

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, VvLlmModelProvider


def main() -> None:
    agent = Agent("background-demo", "Answer briefly.", model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"))
    config = RunConfig(
        model_provider=VvLlmModelProvider(
            settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
            default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        ),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    handle = Runner.start(agent, os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Explain Python's GIL in one sentence."), run_config=config)
    try:
        print("The host can process other work while the handle runs.")
        print(handle.result().final_output)
    finally:
        handle.kernel.close()


if __name__ == "__main__":
    main()
