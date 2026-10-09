#!/usr/bin/env python3
"""Cancel a live kernel RunHandle after a bounded wait."""

import os
import threading
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, VvLlmModelProvider


def main() -> None:
    agent = Agent("cancel-demo", "Complete the task carefully.", model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"))
    config = RunConfig(
        model_provider=VvLlmModelProvider(
            settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
            default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        ),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    handle = Runner.start(
        agent, os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Write a detailed history of artificial intelligence."), run_config=config
    )
    timer = threading.Timer(float(os.getenv("VV_AGENT_EXAMPLE_TIMEOUT", "10")), handle.cancel)
    timer.start()
    try:
        result = handle.result()
        print(result.status.value, result.raw_result.completion_reason, result.final_output)
    finally:
        timer.cancel()
        handle.kernel.close()


if __name__ == "__main__":
    main()
