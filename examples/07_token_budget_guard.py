#!/usr/bin/env python3
"""Guard a run with a simple public input guardrail."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, GuardrailResult, RunConfig, Runner, input_guardrail

TOKEN_BUDGET = int(os.getenv("V_AGENT_EXAMPLE_TOKEN_BUDGET", "4000"))


@input_guardrail
def reject_oversized_prompt(_ctx, input_text: str) -> GuardrailResult:
    estimated_tokens = max(1, len(input_text) // 4)
    if estimated_tokens > TOKEN_BUDGET:
        print(f"Guardrail blocked: prompt is about {estimated_tokens} tokens; budget is {TOKEN_BUDGET}")
        return GuardrailResult.block(f"prompt is about {estimated_tokens} tokens; budget is {TOKEN_BUDGET}")
    return GuardrailResult.allow()


def main() -> None:
    agent = Agent(
        name="budgeted",
        instructions="Keep the answer concise and call task_finish.",
        model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6"),
        input_guardrails=[reject_oversized_prompt],
    )
    config = RunConfig(
        settings_file=Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py")),
        default_backend=os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot"),
        workspace=Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")),
    )
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "Summarize how token budget guardrails work.")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(result.status.value, result.final_output)


if __name__ == "__main__":
    main()
