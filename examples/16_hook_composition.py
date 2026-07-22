#!/usr/bin/env python3
"""Compose multiple runtime hooks in one RunConfig."""

from __future__ import annotations

import os
import time
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner
from vv_agent.runtime.hooks import AfterLLMEvent, BaseRuntimeHook, BeforeLLMEvent


class TimingHook(BaseRuntimeHook):
    def before_llm(self, event: BeforeLLMEvent):
        event.shared_state["llm_started_at"] = time.monotonic()
        return None

    def after_llm(self, event: AfterLLMEvent):
        started = float(event.shared_state.get("llm_started_at", time.monotonic()))
        print(f"[timing] cycle={event.cycle_index} llm_seconds={time.monotonic() - started:.2f}")
        return None


class AuditHook(BaseRuntimeHook):
    def after_llm(self, event: AfterLLMEvent):
        print(f"[audit] cycle={event.cycle_index} tool_calls={len(event.response.tool_calls)}")
        return None


def main() -> None:
    agent = Agent(
        name="hooked",
        instructions="Answer and call task_finish.",
        model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        hooks=[TimingHook(), AuditHook()],
    )
    result = Runner.run_sync(agent, "Describe composed hooks.", run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
