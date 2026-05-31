#!/usr/bin/env python3
"""Run a batch-style task by exposing a child agent as a tool."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, function_tool


@function_tool
def list_documents() -> list[str]:
    """Return document ids to summarize."""
    return ["doc-a", "doc-b", "doc-c"]


summarizer = Agent(
    name="summarizer",
    instructions="Summarize the requested document id in one sentence and finish with task_finish.",
    model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
)


def main() -> None:
    coordinator = Agent(
        name="batch-coordinator",
        instructions="Use list_documents, call summarize_doc for each item, then combine the results.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
        tools=[
            list_documents,
            summarizer.as_tool(name="summarize_doc", description="Summarize one document id."),
        ],
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        max_cycles=10,
    )
    result = Runner.run_sync(coordinator, "Summarize all available documents.", run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
