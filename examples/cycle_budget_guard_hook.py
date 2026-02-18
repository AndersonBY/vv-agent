#!/usr/bin/env python3
"""Budget guard example: force finish when cycle budget is reached."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.constants import TASK_FINISH_TOOL_NAME
from v_agent.runtime import AfterLLMEvent, BaseRuntimeHook
from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import LLMResponse, ToolCall

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
max_cycle_budget = int(os.getenv("V_AGENT_EXAMPLE_CYCLE_BUDGET", "6"))
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)


class CycleBudgetHook(BaseRuntimeHook):
    def __init__(self, cycle_budget: int) -> None:
        self.cycle_budget = max(cycle_budget, 1)

    def after_llm(self, event: AfterLLMEvent) -> LLMResponse | None:
        if event.cycle_index < self.cycle_budget:
            return None

        has_finish = any(call.name == TASK_FINISH_TOOL_NAME for call in event.response.tool_calls)
        if has_finish:
            return None

        if verbose:
            print(
                f"[hook.cycle_budget] force finish at cycle={event.cycle_index}",
                flush=True,
            )

        forced_calls = list(event.response.tool_calls)
        forced_calls.append(
            ToolCall(
                id=f"budget_finish_{event.cycle_index}",
                name=TASK_FINISH_TOOL_NAME,
                arguments={
                    "message": (
                        "Cycle budget reached. Return concise summary and propose next actions for a follow-up run."
                    )
                },
            )
        )
        return LLMResponse(
            content=event.response.content,
            tool_calls=forced_calls,
            raw=dict(event.response.raw),
        )


def runtime_log(event: str, payload: dict[str, Any]) -> None:
    if not verbose:
        return
    if event in {
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_max_cycles",
        "cycle_failed",
    }:
        print(f"[{event}] {payload}", flush=True)


client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        runtime_hooks=[CycleBudgetHook(max_cycle_budget)],
        log_handler=runtime_log,
    ),
    agent=AgentDefinition(
        description=(
            "你是迭代式执行 Agent. 先探索问题, 再给出可执行方案."
            "如果信息不足, 优先给出下一轮需要补充的数据."
        ),
        model=model,
        backend=backend,
        max_cycles=24,
        enable_todo_management=True,
    ),
)

run = client.run(
    prompt=(
        "请梳理 workspace 下的任务上下文, 形成一个可执行计划."
        "如果无法一次完成, 请给出明确的后续输入需求."
    ),
)
print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
