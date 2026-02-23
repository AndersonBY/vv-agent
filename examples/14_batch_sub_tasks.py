#!/usr/bin/env python3
"""Batch sub-tasks example: parallel multi-document processing with sub-agents."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.types import SubAgentConfig


def runtime_log(event: str, payload: dict[str, Any]) -> None:
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not verbose:
        return
    if event in {
        "run_started",
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_wait_user",
        "run_max_cycles",
        "cycle_failed",
    }:
        print(f"[{event}] {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace/batch_demo")).resolve()
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")

    workspace.mkdir(parents=True, exist_ok=True)
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    # Seed sample documents for parallel processing.
    for i, (title, body) in enumerate(
        [
            ("API Design", "RESTful API should use resource-oriented URLs and proper HTTP verbs."),
            ("Testing Strategy", "Unit tests cover logic; integration tests cover boundaries."),
            ("Deployment", "Blue-green deployment minimizes downtime during releases."),
        ],
        start=1,
    ):
        (docs_dir / f"doc_{i}.md").write_text(f"# {title}\n\n{body}\n", encoding="utf-8")

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description=(
                "你是文档处理总控 Agent. 使用 `batch_sub_tasks` 将多个文档的摘要任务"
                "并行分派给子 Agent, 最后汇总结果写入 artifacts/summary.md。"
            ),
            model=model,
            backend=backend,
            max_cycles=20,
            enable_sub_agents=True,
            enable_todo_management=True,
            sub_agents={
                "summarizer-a": SubAgentConfig(
                    model=model,
                    backend=backend,
                    description="负责阅读单篇文档并输出中文摘要。",
                    max_cycles=6,
                ),
                "summarizer-b": SubAgentConfig(
                    model=model,
                    backend=backend,
                    description="负责阅读单篇文档并输出中文摘要。",
                    max_cycles=6,
                ),
            },
        ),
    )

    try:
        run = client.run(
            prompt=(
                "docs/ 目录下有 3 篇文档. 请使用 `batch_sub_tasks` 并行分派给子 Agent, "
                "每个子任务负责读取一篇文档并输出中文摘要. "
                "所有子任务完成后, 将摘要汇总写入 artifacts/summary.md, "
                "然后调用 `task_finish` 输出结论。"
            ),
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
