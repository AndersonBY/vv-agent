#!/usr/bin/env python3
"""A small paper-search pipeline built with function tools."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, function_tool


@function_tool
def search_arxiv(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Return mock arXiv search results for a query."""
    return [
        {
            "title": f"{query} survey #{index}",
            "url": f"https://arxiv.org/abs/0000.0000{index}",
            "summary": "Placeholder result for offline example execution.",
        }
        for index in range(1, max_results + 1)
    ]


@function_tool
def save_report(path: str, content: str) -> str:
    """Save a Markdown report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"saved {target}"


def main() -> None:
    agent = Agent(
        name="paper-analyst",
        instructions="Search papers, write a short Chinese summary, save it, then call task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
        tools=[search_arxiv, save_report],
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        max_cycles=8,
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Find three papers about agent memory and save a report.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
