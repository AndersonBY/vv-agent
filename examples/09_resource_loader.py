#!/usr/bin/env python3
"""Load agent profiles from a small JSON resource file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vv_agent import Agent, RunConfig, Runner


def load_agent(path: Path, profile: str) -> Agent:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "researcher": {
                        "instructions": "Collect facts and finish with task_finish.",
                        "model": "kimi-k2.6",
                    },
                    "writer": {
                        "instructions": "Write a concise final answer and finish with task_finish.",
                        "model": "kimi-k2.6",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    data: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get(profile, data["researcher"])
    return Agent(name=profile, instructions=str(raw["instructions"]), model=str(raw["model"]))


def main() -> None:
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace"))
    profile = os.getenv("V_AGENT_EXAMPLE_PROFILE", "researcher")
    agent = load_agent(workspace / "agent_profiles.json", profile)
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=workspace,
    )
    result = Runner.run_sync(
        agent,
        os.getenv("V_AGENT_EXAMPLE_PROMPT", "Explain why resource-backed profiles are useful."),
        run_config=config,
    )
    print(result.final_output)


if __name__ == "__main__":
    main()
