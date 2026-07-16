#!/usr/bin/env python3
"""Limit a run with the public multi-dimensional budget API."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunBudgetLimits, RunConfig, Runner

TOKEN_BUDGET = int(os.getenv("V_AGENT_EXAMPLE_TOKEN_BUDGET", "4000"))


def main() -> None:
    agent = Agent(
        name="budgeted",
        instructions="Keep the answer concise and call task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        budget_limits=RunBudgetLimits(
            max_total_tokens=TOKEN_BUDGET,
            max_tool_calls=int(os.getenv("V_AGENT_EXAMPLE_TOOL_BUDGET", "12")),
        ),
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Summarize how Agent run budgets work.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.status.value, result.final_output)
    if result.budget_usage is not None:
        print(result.budget_usage.to_dict())
    if result.budget_exhaustion is not None:
        print(result.budget_exhaustion.to_dict())


if __name__ == "__main__":
    main()
