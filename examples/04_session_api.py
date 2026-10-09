#!/usr/bin/env python3
"""Retain conversation history in one kernel session."""

import os
from pathlib import Path

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, VvLlmModelProvider


def main() -> None:
    agent = Agent(
        name="assistant",
        instructions="Use prior turns from the session when answering.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    )
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=VvLlmModelProvider(
                settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
                default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
            ),
            workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        )
    )
    try:
        session = client.create_session(agent=agent, session_id=os.getenv("VV_AGENT_EXAMPLE_SESSION_ID", "demo-thread"))
        print("first:", session.prompt("Remember that the project codename is River.").final_output)
        print("second:", session.prompt("What codename did I give you?").final_output)
    finally:
        client.driver.close()


if __name__ == "__main__":
    main()
