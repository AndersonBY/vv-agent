#!/usr/bin/env python3
"""CeleryBackend: 分布式 Celery 后端, 支持 cycle 级分发 + celery.group 并行执行.

默认使用 eager 模式 (本地同进程执行, 无需 Redis/worker), 方便快速体验.

分布式模式 (需要 Redis + worker):
  1. 安装 celery 可选依赖:  uv sync --extra celery  (或 uv sync --dev)
  2. 启动 Redis:            docker run -d -p 6379:6379 redis:7
  3. 启动 Celery worker:    cd vv-agent && uv run celery -A examples.23_celery_backend worker -l info
  4. 运行本示例:            V_AGENT_EXAMPLE_CELERY_DISTRIBUTED=1 uv run python examples/23_celery_backend.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from celery import Celery

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.backends.celery import (
    CeleryBackend,
    RuntimeRecipe,
    register_cycle_task,
)
from vv_agent.runtime.stores.sqlite import SqliteStateStore
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask

# ---------------------------------------------------------------------------
# 1. Celery app (Redis 作为 broker + result backend)
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("V_AGENT_EXAMPLE_REDIS_URL", "redis://localhost:6379/3")
DISTRIBUTED = os.getenv(
    "V_AGENT_EXAMPLE_CELERY_DISTRIBUTED", "",
).strip().lower() in {"1", "true", "yes", "on"}

app = Celery("vv_agent_example", broker=REDIS_URL, backend=REDIS_URL)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]

if not DISTRIBUTED:
    # Eager 模式: task 在当前进程同步执行, 无需 Redis/worker
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True

# ---------------------------------------------------------------------------
# 2. 注册 worker 端 cycle task + 独立 agent task
# ---------------------------------------------------------------------------
register_cycle_task(app)


@app.task(name="examples.23_celery_backend.run_agent_task")
def run_agent_task(prompt: str) -> dict[str, Any]:
    """在 worker 进程中执行一次 agent 调用, 返回结果摘要."""
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend_name = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(
        settings_file, backend=backend_name, model=model,
    )
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
    )
    system_prompt = build_system_prompt(
        "You are a helpful assistant. Answer concisely.",
        language="zh-CN",
        use_workspace=True,
    )
    task = AgentTask(
        task_id=f"celery_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_cycles=3,
    )
    result = runtime.run(task)
    return {"status": result.status.value, "answer": result.final_answer}


# ---------------------------------------------------------------------------
# 3. 主入口
# ---------------------------------------------------------------------------


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {"cycle_started", "run_completed", "cycle_failed"}:
        print(f"  [{event}] {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend_name = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv(
        "V_AGENT_EXAMPLE_VERBOSE", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(
        settings_file, backend=backend_name, model=model,
    )

    # Build a RuntimeRecipe so CeleryBackend uses cycle-level dispatch.
    recipe = RuntimeRecipe(
        settings_file=str(settings_file),
        backend=backend_name,
        model=model,
        workspace=str(workspace),
    )

    # Use SqliteStateStore for eager mode (shared within the same process).
    db_path = workspace / ".vv-agent-state" / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(db_path=db_path)

    celery_backend = CeleryBackend(
        celery_app=app,
        state_store=store,
        runtime_recipe=recipe,
    )

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
        execution_backend=celery_backend,
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    # --- 场景 1: Cycle 级分布式执行 ---
    print("[demo] 场景 1: CeleryBackend cycle 级分布式执行")
    task = AgentTask(
        task_id=f"celery_dist_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt="1+1 等于几?",
        max_cycles=3,
    )
    result = runtime.run(task)
    print(f"  状态: {result.status.value}, 回答: {result.final_answer}\n")

    # --- 场景 2: parallel_map 通过 celery.group 并行 ---
    mode = "distributed (celery worker)" if DISTRIBUTED else "eager (本地同进程)"
    print(f"[demo] 场景 2: parallel_map 通过 celery.group 并行分发 [{mode}]")
    prompts = [
        "Python 的 GIL 是什么? 一句话回答",
        "什么是 REST API? 一句话回答",
        "Docker 和虚拟机的区别? 一句话回答",
    ]
    results = celery_backend.parallel_map(run_agent_task, prompts)
    for prompt, res in zip(prompts, results, strict=True):
        print(f"  Q: {prompt}")
        print(f"  A: {res}\n")

    print("[demo] 完成!")


if __name__ == "__main__":
    main()
