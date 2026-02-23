#!/usr/bin/env python3
"""Example: auto-load skills directory and let agent choose skill(s) to activate."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.skills import prepare_skill_bundle


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
    skills_dir = os.getenv("V_AGENT_EXAMPLE_SKILLS_DIR", "skills")
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    max_cycles = int(os.getenv("V_AGENT_EXAMPLE_MAX_CYCLES", "80"))
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    source_root = Path(skills_dir).expanduser()
    if not source_root.is_absolute():
        source_root = (workspace / source_root).resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {source_root}")

    prepared = prepare_skill_bundle(source_root, workspace)
    runtime_skills_root = prepared[0].runtime.parent  # the bundle root
    # Convert to workspace-relative path
    try:
        runtime_skills_root_str = runtime_skills_root.relative_to(workspace).as_posix()
    except ValueError:
        runtime_skills_root_str = str(runtime_skills_root)

    print("[skills] prepared runtime bundle:", flush=True)
    for item in prepared:
        print(f"- {item.name}: source={item.source} -> runtime={item.runtime}", flush=True)

    prompt = (
        "请执行一个 Remotion 视频工程任务, 并尽量利用已提供的技能元数据:\n"
        "1) 请先查看系统提示词中的 `<available_skills>` 列表, 若有匹配技能, "
        "由你自主选择并调用 `activate_skill` (不要等待我指定技能名).\n"
        "2) 激活后, 读取必要规则文件(至少 3 个), 并记录你读取了哪些文件.\n"
        "3) 在 `artifacts/remotion_demo/` 下生成最小 Remotion 工程骨架, 至少包含:\n"
        "   - `package.json`\n"
        "   - `src/index.ts`\n"
        "   - `src/Root.tsx`\n"
        "   - `src/compositions/IntroCard.tsx`\n"
        "4) 示例视频规格: 1280x720, 30fps, 150 帧, 标题+副标题+入场动画, props 可配置.\n"
        "5) 写出 `artifacts/remotion_demo/README_zh.md`, 说明如何安装依赖、预览、渲染视频.\n"
        "6) 最后调用 `task_finish`, 汇报中必须包含: 激活技能名、读取规则文件、生成路径、下一步命令.\n"
        "约束:\n"
        f"- 运行时技能目录: `{runtime_skills_root_str}`\n"
        "- 不要只输出计划, 必须实际写文件.\n"
    )

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            log_handler=log_handler if verbose else None,
        ),
        agent=AgentDefinition(
            description="你是 Remotion 视频工程助手, 会自主匹配并激活合适技能后落地代码.",
            backend=backend,
            model=model,
            language="zh-CN",
            max_cycles=max(max_cycles, 1),
            enable_todo_management=True,
            use_workspace=True,
            agent_type="computer",
            skill_directories=[runtime_skills_root_str],
        ),
    )

    try:
        run = client.run(prompt=prompt)
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
