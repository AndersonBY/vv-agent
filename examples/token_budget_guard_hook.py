#!/usr/bin/env python3
"""Budget guard example: force finish when token budget is reached."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from v_agent.constants import TASK_FINISH_TOOL_NAME, TASK_FINISH_TOOL_SCHEMA
from v_agent.runtime import AfterLLMEvent, BaseRuntimeHook, BeforeLLMEvent, BeforeLLMPatch
from v_agent.runtime.token_usage import normalize_token_usage
from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import LLMResponse, Message, ToolCall

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
max_token_budget = int(os.getenv("V_AGENT_EXAMPLE_TOKEN_BUDGET", "6000"))
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)


class TokenBudgetHook(BaseRuntimeHook):
    def __init__(self, token_budget: int) -> None:
        self.token_budget = max(token_budget, 1)
        self.consumed_tokens = 0
        self.finalize_mode = False
        self.finalize_prompt_injected = False

    def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch | None:
        if not self.finalize_mode:
            return None

        restricted_schemas = [
            schema
            for schema in event.tool_schemas
            if isinstance(schema.get("function"), dict) and schema["function"].get("name") == TASK_FINISH_TOOL_NAME
        ]
        if not restricted_schemas:
            restricted_schemas = [deepcopy(TASK_FINISH_TOOL_SCHEMA)]

        patched_messages = list(event.messages)
        if not self.finalize_prompt_injected:
            patched_messages.append(
                Message(
                    role="user",
                    content=(
                        "Token budget 已达上限. 请基于现有信息给出最终简洁总结, "
                        "并调用 _task_finish, 把总结写入 message 字段."
                    ),
                )
            )
            self.finalize_prompt_injected = True
            if verbose:
                print("[hook.token_budget] enter finalization cycle", flush=True)

        return BeforeLLMPatch(messages=patched_messages, tool_schemas=restricted_schemas)

    def after_llm(self, event: AfterLLMEvent) -> LLMResponse | None:
        usage = normalize_token_usage(event.response.raw.get("usage"))
        cycle_tokens = usage.total_tokens
        if cycle_tokens <= 0:
            cycle_tokens = usage.prompt_tokens + usage.completion_tokens
        self.consumed_tokens += max(cycle_tokens, 0)

        if verbose:
            print(
                f"[hook.token_budget] cycle={event.cycle_index} "
                f"cycle_tokens={cycle_tokens} total_tokens={self.consumed_tokens}/{self.token_budget}",
                flush=True,
            )

        has_finish = any(call.name == TASK_FINISH_TOOL_NAME for call in event.response.tool_calls)

        if self.finalize_mode:
            if has_finish:
                return None
            summary_message = event.response.content.strip() or (
                "Token budget reached. Please run another task if you need deeper analysis."
            )
            forced_calls = [
                ToolCall(
                    id=f"budget_finish_{event.cycle_index}",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": summary_message},
                )
            ]
            return LLMResponse(
                content=event.response.content,
                tool_calls=forced_calls,
                raw=dict(event.response.raw),
            )

        if self.consumed_tokens < self.token_budget:
            return None

        if has_finish:
            return None

        self.finalize_mode = True
        if verbose:
            print(
                f"[hook.token_budget] budget reached at cycle={event.cycle_index}, switch to finalization mode",
                flush=True,
            )

        # Stop current cycle tool execution and run one extra LLM finalization cycle.
        return LLMResponse(
            content=event.response.content,
            tool_calls=[],
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
        runtime_hooks=[TokenBudgetHook(max_token_budget)],
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
