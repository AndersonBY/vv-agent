#!/usr/bin/env python3
"""Observe memory compaction with a runtime hook."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeMemoryCompactEvent


class MemoryAuditHook(BaseRuntimeHook):
    def before_memory_compact(self, event: BeforeMemoryCompactEvent):
        print(f"[memory] cycle={event.cycle_index} messages={len(event.messages)}")
        return None


def main() -> None:
    agent = Agent(
        name="memory-auditor",
        instructions="Answer normally and call task_finish.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        hooks=[MemoryAuditHook()],
    )
    prompt = os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Explain what memory compaction hooks can observe.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
