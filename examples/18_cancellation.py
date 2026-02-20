#!/usr/bin/env python3
"""Cancellation: 在后台线程中运行 agent, 超时后自动取消."""

from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime, CancellationToken, ExecutionContext
from vv_agent.runtime.backends.thread import ThreadBackend
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {"cycle_started", "cycle_llm_response", "run_completed", "cycle_failed"}:
        print(f"  [{event}] {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    timeout = float(os.getenv("V_AGENT_EXAMPLE_TIMEOUT", "10"))
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)

    # 使用 ThreadBackend 以便在后台线程执行
    thread_backend = ThreadBackend(max_workers=2)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
        execution_backend=thread_backend,
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent. Complete the task step by step.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    task = AgentTask(
        task_id=f"cancel_demo_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt="写一篇关于人工智能发展历史的长文, 至少 2000 字",
        max_cycles=20,
    )

    # 创建 CancellationToken + ExecutionContext
    token = CancellationToken()
    ctx = ExecutionContext(cancellation_token=token)

    # 设置超时取消: timeout 秒后自动 cancel
    timer = threading.Timer(timeout, token.cancel)
    timer.start()
    print(f"[demo] 任务已启动, {timeout}s 后将自动取消...")

    try:
        result = runtime.run(task, ctx=ctx)
        print(f"\n[demo] 最终状态: {result.status.value}")
        print(f"[demo] 完成 cycles: {len(result.cycles)}")
        if result.error:
            print(f"[demo] 错误信息: {result.error}")
        if result.final_answer:
            print(f"[demo] 最终回答: {result.final_answer[:200]}...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
