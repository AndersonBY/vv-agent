#!/usr/bin/env python3
"""State Checkpoint: 使用 SqliteStateStore 持久化 agent 执行状态."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime, ExecutionContext
from vv_agent.runtime.state import Checkpoint, InMemoryStateStore
from vv_agent.runtime.stores.sqlite import SqliteStateStore
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {"cycle_started", "run_completed", "cycle_failed"}:
        print(f"  [{event}] {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}
    db_path = os.getenv("V_AGENT_EXAMPLE_DB", ":memory:")

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    task_id = f"ckpt_demo_{uuid.uuid4().hex[:8]}"
    task = AgentTask(
        task_id=task_id,
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt="2+3 等于几?",
        max_cycles=5,
    )

    # --- 使用 SqliteStateStore 持久化 checkpoint ---
    store = SqliteStateStore(db_path=db_path)
    ctx = ExecutionContext(state_store=store)

    print(f"[demo] 运行任务 {task_id}...")
    result = runtime.run(task, ctx=ctx)
    print(f"[demo] 状态: {result.status.value}")
    if result.final_answer:
        print(f"[demo] 回答: {result.final_answer}")

    # 手动保存一个 checkpoint 演示 save/load 往返
    checkpoint = Checkpoint(
        task_id=task_id,
        cycle_index=len(result.cycles),
        status=result.status,
        messages=result.messages,
        cycles=result.cycles,
        shared_state=result.shared_state,
    )
    store.save_checkpoint(checkpoint)
    print(f"\n[demo] Checkpoint 已保存, task_id={task_id}")

    # 列出所有 checkpoint
    all_ids = store.list_checkpoints()
    print(f"[demo] 当前 checkpoint 列表: {all_ids}")

    # 加载并验证
    loaded = store.load_checkpoint(task_id)
    if loaded is not None:
        print(f"[demo] 加载成功: cycle_index={loaded.cycle_index}, status={loaded.status.value}")
        print(f"[demo] messages 数量: {len(loaded.messages)}, cycles 数量: {len(loaded.cycles)}")
    else:
        print("[demo] 加载失败!")

    # 清理
    store.delete_checkpoint(task_id)
    print(f"[demo] Checkpoint 已删除, 剩余: {store.list_checkpoints()}")
    store.close()

    # --- InMemoryStateStore 对比 ---
    print("\n[demo] InMemoryStateStore 同样实现 StateStore 协议:")
    mem_store = InMemoryStateStore()
    mem_store.save_checkpoint(checkpoint)
    print(f"  save → list: {mem_store.list_checkpoints()}")
    mem_store.delete_checkpoint(task_id)
    print(f"  delete → list: {mem_store.list_checkpoints()}")

    print("\n[demo] 完成!")


if __name__ == "__main__":
    main()
