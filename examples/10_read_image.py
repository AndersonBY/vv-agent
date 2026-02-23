#!/usr/bin/env python3
"""Example: call read_image with kimi-k2.5 and write Markdown report."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def log_handler(event: str, payload: dict[str, Any]) -> None:
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
            for suffix in IMAGE_SUFFIXES
            for path in workspace.rglob(f"*{suffix}")
            if path.is_file()
        )
        raise FileNotFoundError(f"Image not found: {image_path}. Available images: {available}")

    prompt = (
        "请完成以下任务并严格执行:\n"
        f"1) 调用 `read_image` 读取 `{image_path}`.\n"
        "2) 基于图片内容生成中文 Markdown, 内容至少包含:\n"
        "   - 标题\n"
        "   - 场景概述\n"
        "   - 关键元素(分点)\n"
        "   - 可见文字识别(若无则写明)\n"
        "   - 你对图片用途或上下文的推断(标注不确定性)\n"
        "3) 调用 `write_file` 将 Markdown 写入目标文件(覆盖写入): "
        f"`{output_path}`.\n"
        "4) 调用 `task_finish`, 并在最终 message 中包含输出文件路径.\n"
        "要求:\n"
        "- 不要假装读图, 必须先调用 `read_image`.\n"
        "- 输出必须是 Markdown 格式.\n"
    )

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            log_handler=log_handler if verbose else None,
        ),
        agent=AgentDefinition(
            description="你是视觉理解助手, 你会读取图片并输出结构化 Markdown 分析.",
            backend=backend,
            model=model,
            language="zh-CN",
            max_cycles=12,
            use_workspace=True,
            native_multimodal=True,
            enable_todo_management=True,
        ),
    )

    try:
        run = client.run(prompt=prompt)
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))

        output_file = (workspace / output_path).resolve()
        if output_file.is_file():
            print("\n[Generated Markdown]")
            print(f"path: {output_file.relative_to(workspace)}")
            print(output_file.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
