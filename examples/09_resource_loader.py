#!/usr/bin/env python3
"""Resource loader example: auto-discover profiles/prompts/skills/hooks from .vv-agent/."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from vv_agent.sdk import AgentDefinition, AgentResourceLoader, AgentSDKClient, AgentSDKOptions


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "请概述这个项目的关键能力。")

    workspace.mkdir(parents=True, exist_ok=True)

    loader = AgentResourceLoader(workspace=workspace)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            resource_loader=loader,
        ),
        agent=AgentDefinition(
            description="你是默认 Agent, 当未提供 profile 时用于兜底.",
            model=model,
            backend=backend,
            max_cycles=16,
            enable_todo_management=True,
        ),
    )

    print("[discovered agents]", client.list_agents(), flush=True)
    if client.resource_diagnostics:
        print("[resource diagnostics]")
        for item in client.resource_diagnostics:
            print("-", item)

    try:
        selected_agent = os.getenv("V_AGENT_EXAMPLE_AGENT", "")
        if selected_agent and selected_agent in client.list_agents():
            run = client.run(agent=selected_agent, prompt=prompt)
        else:
            run = client.run(prompt=prompt)

        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

