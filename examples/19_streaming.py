#!/usr/bin/env python3
"""Streaming: 实时接收 LLM 输出 token, 适合 UI 逐字显示."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime, ExecutionContext
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {"cycle_started", "run_completed"}:
        print(f"\n[{event}] {payload}", flush=True)


# 收集所有 token 用于统计
collected_tokens: list[str] = []


def stream_callback(text: str) -> None:
    """每收到一个 token 就立即输出, 不换行."""
    collected_tokens.append(text)
    print(text, end="", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent. Answer concisely.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    task = AgentTask(
        task_id=f"stream_demo_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt=os.getenv("V_AGENT_EXAMPLE_PROMPT", "用三句话介绍 Python 语言"),
        max_cycles=5,
    )

    # 通过 ExecutionContext 传入 stream_callback
    ctx = ExecutionContext(stream_callback=stream_callback)

    print("[demo] 流式输出开始:\n")
    try:
        result = runtime.run(task, ctx=ctx)
        print(f"\n\n[demo] 状态: {result.status.value}")
        print(f"[demo] 共收到 {len(collected_tokens)} 个 token 片段")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
