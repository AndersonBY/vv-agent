#!/usr/bin/env python3
"""Use a typed tool to pass image metadata to an agent."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, function_tool


@function_tool
def image_info(path: str) -> dict[str, object]:
    """Return basic metadata for an image file in the workspace."""
    target = Path(path)
    return {
        "path": path,
        "exists": target.exists(),
        "size_bytes": target.stat().st_size if target.exists() else 0,
        "suffix": target.suffix,
    }


def main() -> None:
    agent = Agent(
        name="image-reporter",
        instructions="Use image_info, then produce a Markdown report with task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
        tools=[image_info],
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Inspect ./example.png and report what metadata is available.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
