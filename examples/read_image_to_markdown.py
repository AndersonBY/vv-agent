#!/usr/bin/env python3
"""Example: call _read_image with kimi-k2.5 and write Markdown report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
image_path = os.getenv("V_AGENT_EXAMPLE_IMAGE_PATH", "test_image.png")
output_path = os.getenv("V_AGENT_EXAMPLE_OUTPUT_PATH", "artifacts/image_read_report.md")
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)

image_file = (workspace / image_path).resolve()
if not image_file.is_file():
    available = sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    raise FileNotFoundError(f"Image not found: {image_path}. Available images: {available}")


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
    "请完成以下任务并严格执行:\n"
    f"1) 调用 `_read_image` 读取 `{image_path}`.\n"
    "2) 基于图片内容生成中文 Markdown, 内容至少包含:\n"
    "   - 标题\n"
    "   - 场景概述\n"
    "   - 关键元素(分点)\n"
    "   - 可见文字识别(若无则写明)\n"
    "   - 你对图片用途或上下文的推断(标注不确定性)\n"
    "3) 调用 `_write_file` 将 Markdown 写入目标文件(覆盖写入): "
    f"`{output_path}`.\n"
    "4) 调用 `_task_finish`, 并在最终 message 中包含输出文件路径.\n"
    "要求:\n"
    "- 不要假装读图, 必须先调用 `_read_image`.\n"
    "- 输出必须是 Markdown 格式.\n"
)

client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        log_handler=log_handler if verbose else None,
    ),
    agents={
        "image_markdown_agent": AgentDefinition(
            description="你是视觉理解助手, 你会读取图片并输出结构化 Markdown 分析.",
            backend=backend,
            model=model,
            language="zh-CN",
            max_cycles=12,
            use_workspace=True,
            native_multimodal=True,
            enable_todo_management=True,
        )
    },
)

run = client.run_agent(agent_name="image_markdown_agent", prompt=prompt)
print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))

output_file = (workspace / output_path).resolve()
if output_file.is_file():
    print("\n[Generated Markdown]")
    print(f"path: {output_file.relative_to(workspace)}")
    print(output_file.read_text(encoding="utf-8", errors="replace"))
