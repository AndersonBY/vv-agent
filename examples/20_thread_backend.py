#!/usr/bin/env python3
"""ThreadBackend: 非阻塞提交 + Future 模式, 适合桌面端/Web 后端嵌入."""

from __future__ import annotations

import os
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
    if event in {"cycle_started", "run_completed", "cycle_failed"}:
        print(f"  [{event}] {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend_name = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(
        settings_file, backend=backend_name, model=model,
    )

    # ThreadBackend: 内部使用线程池, 支持 submit() 非阻塞提交
    thread_backend = ThreadBackend(max_workers=4)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
        execution_backend=thread_backend,
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    # --- 方式 1: 同步调用 (行为与 InlineBackend 一致) ---
    print("[demo] 方式 1: 同步调用 runtime.run()")
    task1 = AgentTask(
        task_id=f"thread_sync_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt="1+1 等于几?",
        max_cycles=3,
    )
    result1 = runtime.run(task1)
    print(f"  状态: {result1.status.value}, 回答: {result1.final_answer}\n")

    # --- 方式 2: 非阻塞 submit + Future ---
    print("[demo] 方式 2: 非阻塞 submit, 主线程可做其他事")
    task2 = AgentTask(
        task_id=f"thread_async_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt="Python 的 GIL 是什么? 一句话回答",
        max_cycles=3,
    )

    token = CancellationToken()
    ctx = ExecutionContext(cancellation_token=token)
    future = thread_backend.submit(lambda: runtime.run(task2, ctx=ctx))

    print("  [主线程] Future 已提交, 正在做其他事...")
    # 这里主线程可以做任何事, 比如更新 UI、处理其他请求
    import time
    time.sleep(0.5)
    print("  [主线程] 等待结果...")

    result2 = future.result(timeout=120)
    print(f"  状态: {result2.status.value}, 回答: {result2.final_answer}\n")

    print("[demo] 完成!")


if __name__ == "__main__":
    main()
