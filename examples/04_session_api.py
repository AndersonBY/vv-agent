#!/usr/bin/env python3
"""Persist conversation history with a public Session implementation."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, MemorySession, RunConfig, Runner


def main() -> None:
    session = MemorySession(os.getenv("V_AGENT_EXAMPLE_SESSION_ID", "demo-thread"))
    agent = Agent(
        name="assistant",
        instructions="Use prior turns from the session when answering. Finish with task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        session=session,
    )

    first = Runner.run_sync(agent, "Remember that the project codename is River.", run_config=config)
    print("first:", first.final_output)
    second = Runner.run_sync(agent, "What codename did I give you?", run_config=config)
    print("second:", second.final_output)


if __name__ == "__main__":
    main()
