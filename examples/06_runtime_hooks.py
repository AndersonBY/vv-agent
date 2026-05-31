#!/usr/bin/env python3
"""Attach low-level runtime hooks through RunConfig."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeLLMEvent, BeforeLLMPatch
from vv_agent.types import Message


class ContextHintHook(BaseRuntimeHook):
    def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch:
        messages = list(event.messages)
        messages.append(Message(role="user", content="Runtime hint: keep the answer under 120 words."))
        return BeforeLLMPatch(messages=messages)


def main() -> None:
    agent = Agent(
        name="assistant",
        instructions="Answer and call task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
        runtime_hooks=[ContextHintHook()],
    )
    result = Runner.run_sync(agent, "Describe the hook mechanism.", run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    main()
