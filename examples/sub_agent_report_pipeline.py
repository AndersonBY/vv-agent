#!/usr/bin/env python3
"""Sub-agent pipeline example: research -> write -> final report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import SubAgentConfig

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)
source_dir = workspace / "inputs"
source_dir.mkdir(parents=True, exist_ok=True)

(source_dir / "product_brief.md").write_text(
    """# Product Brief

- Product: VectorVein Agent Platform
- Goal: Build reliable multi-agent runtime with strong tool protocol.
- KPI: Reduce failed runs by 35%, cut debugging time by 40%.
""",
    encoding="utf-8",
)
(source_dir / "ops_notes.md").write_text(
    """# Ops Notes

- Main risk: models may loop without clear finish signal.
- Priority: improve observability and guardrails.
- Suggestion: enforce structured todo + runtime hook safety checks.
""",
    encoding="utf-8",
)


def runtime_log(event: str, payload: dict[str, Any]) -> None:
    if not verbose:
        return
    if event in {
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_steered",
        "cycle_failed",
    }:
        print(f"[{event}] {payload}", flush=True)


client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        log_handler=runtime_log,
    ),
    agent=AgentDefinition(
        description=(
            "你是项目总控 Agent. 你必须优先将任务委派给子 Agent, 再整合产出最终结果."
            "最终请把报告写入 artifacts/final_report.md。"
        ),
        model=model,
        backend=backend,
        max_cycles=20,
        enable_sub_agents=True,
        enable_todo_management=True,
        sub_agents={
            "research-sub": SubAgentConfig(
                model=model,
                backend=backend,
                description="负责阅读输入文档并提取事实要点。",
                max_cycles=8,
            ),
            "writer-sub": SubAgentConfig(
                model=model,
                backend=backend,
                description="负责把要点写成中文可执行报告。",
                max_cycles=10,
            ),
        },
    ),
)

run = client.run(
    prompt=(
        "请先调用 `_create_sub_task` 给 `research-sub`, 读取 inputs/ 下文档并产出结构化要点."
        "然后调用 `_create_sub_task` 给 `writer-sub`, 基于要点输出 `artifacts/final_report.md` 的正文草稿."
        "最后由你整合并确认结果。"
    ),
)
print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
