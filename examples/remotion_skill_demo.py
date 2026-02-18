#!/usr/bin/env python3
"""Example: run v-agent with the remotion skill at workspace/skills/remotion/SKILL.md."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.skills import read_properties, validate


def _log_handler(event: str, payload: dict[str, Any]) -> None:
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


def _resolve_skill_path(*, workspace: Path, skill_path: str) -> Path:
    path = Path(skill_path).expanduser()
    if not path.is_absolute():
        path = (workspace / path).resolve()
    return path


def _to_workspace_location(*, workspace: Path, path: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_skill_directory(*, workspace: Path, source_dir: Path, skill_name: str) -> Path:
    cache_dir = (workspace / ".v_agent_skill_cache" / skill_name).resolve()
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, cache_dir, dirs_exist_ok=True)
    return cache_dir


def ensure_skill_ready(*, workspace: Path, skill_path: str) -> tuple[str, str, str, str]:
    resolved = _resolve_skill_path(workspace=workspace, skill_path=skill_path)
    source_skill_md = resolved
    if resolved.is_dir():
        source_skill_md = (resolved / "SKILL.md").resolve()

    if not source_skill_md.is_file():
        raise FileNotFoundError(f"Skill file not found: {source_skill_md}")

    source_skill_dir = source_skill_md.parent
    props = read_properties(source_skill_dir)

    errors = validate(source_skill_dir)
    runtime_skill_dir = source_skill_dir
    if errors:
        if any("must match skill name" in message for message in errors):
            runtime_skill_dir = _normalize_skill_directory(
                workspace=workspace,
                source_dir=source_skill_dir,
                skill_name=props.name,
            )
            normalized_errors = validate(runtime_skill_dir)
            if normalized_errors:
                joined = "\n".join(f"- {item}" for item in normalized_errors)
                raise ValueError(f"Normalized skill validation failed:\n{joined}")
            print(
                "[skill] source directory name does not match skill name; "
                f"using normalized copy: {_to_workspace_location(workspace=workspace, path=runtime_skill_dir / 'SKILL.md')}",
                flush=True,
            )
        else:
            joined = "\n".join(f"- {item}" for item in errors)
            raise ValueError(f"Skill validation failed:\n{joined}")

    runtime_skill_md = (runtime_skill_dir / "SKILL.md").resolve()
    runtime_location = _to_workspace_location(workspace=workspace, path=runtime_skill_md)
    source_location = _to_workspace_location(workspace=workspace, path=source_skill_md)
    return props.name, props.description, runtime_location, source_location


def build_prompt(*, skill_name: str, runtime_skill_location: str, source_skill_location: str) -> str:
    return (
        "请执行一个 Remotion 技能实测任务, 并严格按顺序完成:\n"
        f"1) 第一步必须调用 `_activate_skill`, 参数 `skill_name` 必须是 `{skill_name}`.\n"
        "2) 激活后阅读技能文档中至少 3 个规则文件(其中必须包含 compositions 与 animations), "
        "并简要记录你读取了哪些文件.\n"
        "3) 在 `artifacts/remotion_demo/` 下生成一个最小 Remotion 工程骨架, 至少包含:\n"
        "   - `package.json`\n"
        "   - `src/index.ts`\n"
        "   - `src/Root.tsx`\n"
        "   - `src/compositions/IntroCard.tsx`\n"
        "4) 这个示例视频模板要求:\n"
        "   - 1280x720, 30fps, 150 帧\n"
        "   - 包含标题、副标题和一个简单入场动画\n"
        "   - props 参数可配置 title/subtitle\n"
        "5) 生成 `artifacts/remotion_demo/README_zh.md`, 说明如何安装依赖、预览、渲染视频.\n"
        "6) 最后调用 `_task_finish`, 最终消息必须包含:\n"
        "   - 激活的 skill 名称\n"
        "   - 读取过的规则文件列表\n"
        "   - 生成的文件路径列表\n"
        "   - 下一步用户可执行的命令\n"
        "约束:\n"
        f"- 技能来源文件: `{source_skill_location}`\n"
        f"- 运行时技能文件: `{runtime_skill_location}`\n"
        "- 不要只输出计划, 必须实际写文件.\n"
    )


def build_client(
    *,
    settings_file: Path,
    workspace: Path,
    backend: str,
    model: str,
    skill_name: str,
    skill_description: str,
    runtime_skill_location: str,
    max_cycles: int,
    verbose: bool,
) -> AgentSDKClient:
    options = AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        log_handler=_log_handler if verbose else None,
    )
    definition = AgentDefinition(
        description=(
            "你是 Remotion 视频工程助手。"
            "你会先激活指定技能, 再按技能建议生成结构化视频工程文件。"
        ),
        backend=backend,
        model=model,
        language="zh-CN",
        max_cycles=max(max_cycles, 1),
        enable_todo_management=True,
        use_workspace=True,
        agent_type="computer",
        metadata={
            "available_skills": [
                {
                    "name": skill_name,
                    "description": skill_description,
                    "location": runtime_skill_location,
                }
            ]
        },
    )
    return AgentSDKClient(options=options, agents={"remotion_skill_agent": definition})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remotion skill demo with v-agent SDK")
    parser.add_argument("--settings-file", default="local_settings.py", help="Path to local_settings.py")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument(
        "--skill-path",
        default="skills/remotion/SKILL.md",
        help="Skill path relative to workspace (or absolute path)",
    )
    parser.add_argument("--backend", default="moonshot", help="Backend key in local settings")
    parser.add_argument("--model", default="kimi-k2.5", help="Model key in backend config")
    parser.add_argument("--max-cycles", type=int, default=80, help="Max runtime cycles")
    parser.add_argument("--verbose", action="store_true", help="Print per-cycle logs")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    skill_name, skill_description, runtime_skill_location, source_skill_location = ensure_skill_ready(
        workspace=workspace,
        skill_path=args.skill_path,
    )

    client = build_client(
        settings_file=Path(args.settings_file),
        workspace=workspace,
        backend=args.backend,
        model=args.model,
        skill_name=skill_name,
        skill_description=skill_description,
        runtime_skill_location=runtime_skill_location,
        max_cycles=args.max_cycles,
        verbose=args.verbose,
    )

    run = client.run_agent(
        agent_name="remotion_skill_agent",
        prompt=build_prompt(
            skill_name=skill_name,
            runtime_skill_location=runtime_skill_location,
            source_skill_location=source_skill_location,
        ),
    )
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
