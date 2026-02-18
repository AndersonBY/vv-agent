#!/usr/bin/env python3
"""End-to-end example: arXiv search + download + image read + Chinese translation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace/arxiv_memory_demo")).resolve()
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {
        "run_started",
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_wait_user",
        "cycle_failed",
        "run_max_cycles",
    }:
        print(f"[{event}] {payload}", flush=True)


prompt = (
    "请完成一个端到端任务, 主题是 AI Agent Memory.\n\n"
    "目标要求(必须按顺序执行):\n"
    "1) 在 arXiv 搜索最近 30 天内发布且与 AI Agent Memory 高度相关的论文, 选择 1 篇最匹配的.\n"
    "2) 将该论文 PDF 下载到 `artifacts/paper.pdf`, 并把元数据保存到 `artifacts/paper_meta.json`.\n"
    "   元数据需包含: 标题, 作者, 发布日期, arXiv 链接, 选择理由.\n"
    "3) 从论文中提取第一张图片并保存为 `artifacts/figure1.png`.\n"
    "   优先从 PDF 抽取, 若失败可从论文页面获取首图.\n"
    "4) 必须调用 `_read_image` 读取 `artifacts/figure1.png`, 并解释图片主要内容与其在论文中的作用.\n"
    "5) 将论文内容翻译为中文并输出到 `artifacts/paper_zh.md`:\n"
    "   - 按段落逐步翻译并持续写入(不要一次性整篇输出);\n"
    "   - 保留公式, 引用编号, 图表编号;\n"
    "   - 术语翻译前后一致.\n"
    "6) 完成后调用 `_task_finish`, 最终汇报中必须包含:\n"
    "   - 论文标题, arXiv URL, 发布日期(UTC);\n"
    "   - 下载文件路径, 图片路径, 翻译文件路径;\n"
    "   - 图片解释摘要;\n"
    "   - 翻译完成度(已翻译段落数量, 是否有失败段落).\n\n"
    "执行约束:\n"
    "- 不要伪造文件或结论, 如果某一步失败, 要先修复后再继续.\n"
    "- 尽量使用工具链完成, 不要只给计划.\n"
)

client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        log_handler=log_handler if verbose else None,
    ),
    agent=AgentDefinition(
        description="你是资深 AI 研究助理, 擅长检索论文、处理 PDF、解释图片并做中文学术翻译.",
        backend=backend,
        model=model,
        language="zh-CN",
        max_cycles=200,
        enable_todo_management=True,
        use_workspace=True,
        agent_type="computer",
        native_multimodal=True,
    ),
)

run = client.run(prompt=prompt)
print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
