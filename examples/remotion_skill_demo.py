#!/usr/bin/env python3
"""Example: auto-load skills directory and let agent choose skill(s) to activate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.skills import discover_skill_dirs, read_properties, validate


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


def _resolve_path(*, workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (workspace / path).resolve()
    return path


def _to_workspace_path(*, workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def prepare_runtime_skill_bundle(*, workspace: Path, skills_dir: str) -> tuple[str, list[dict[str, str]]]:
    source_root = _resolve_path(workspace=workspace, value=skills_dir)
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {source_root}")

    discovered = discover_skill_dirs(source_root)
    if not discovered:
        raise ValueError(f"No SKILL.md discovered under: {source_root}")

    runtime_root = (workspace / ".v_agent_skill_cache" / "bundle").resolve()
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, str]] = []
    used_names: set[str] = set()

    for skill_dir in discovered:
        props = read_properties(skill_dir)
        if props.name in used_names:
            continue

        target_dir = (runtime_root / props.name).resolve()
        shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)

        errors = validate(target_dir)
        if errors:
            joined = "\n".join(f"- {item}" for item in errors)
            raise ValueError(
                f"Skill '{props.name}' is invalid after normalization copy:\n{joined}"
            )

        used_names.add(props.name)
        prepared.append(
            {
                "name": props.name,
                "description": props.description,
                "source": _to_workspace_path(workspace=workspace, path=skill_dir),
                "runtime": _to_workspace_path(workspace=workspace, path=target_dir),
            }
        )

    if not prepared:
        raise ValueError("No valid skills available after preparation.")

    runtime_root_rel = _to_workspace_path(workspace=workspace, path=runtime_root)
    return runtime_root_rel, prepared


def build_prompt(*, runtime_skills_root: str) -> str:
    return (
        "请执行一个 Remotion 视频工程任务, 并尽量利用已提供的技能元数据:\n"
        "1) 请先查看系统提示词中的 `<available_skills>` 列表, "
        "若有匹配技能, 由你自主选择并调用 `_activate_skill` (不要等待我指定技能名).\n"
        "2) 激活后, 读取必要规则文件(至少 3 个), 并记录你读取了哪些文件.\n"
        "3) 在 `artifacts/remotion_demo/` 下生成最小 Remotion 工程骨架, 至少包含:\n"
        "   - `package.json`\n"
        "   - `src/index.ts`\n"
        "   - `src/Root.tsx`\n"
        "   - `src/compositions/IntroCard.tsx`\n"
        "4) 示例视频规格:\n"
        "   - 1280x720, 30fps, 150 帧\n"
        "   - 标题 + 副标题 + 简单入场动画\n"
        "   - props 可配置 title/subtitle\n"
        "5) 写出 `artifacts/remotion_demo/README_zh.md`, 说明如何安装依赖、预览、渲染视频.\n"
        "6) 最后调用 `_task_finish`, 汇报中必须包含:\n"
        "   - 你激活的技能名\n"
        "   - 读取的规则文件\n"
        "   - 生成文件路径\n"
        "   - 下一步命令\n"
        "约束:\n"
        f"- 运行时技能目录: `{runtime_skills_root}`\n"
        "- 不要只输出计划, 必须实际写文件.\n"
    )


def build_client(
    *,
    settings_file: Path,
    workspace: Path,
    backend: str,
    model: str,
    runtime_skills_root: str,
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
        description="你是 Remotion 视频工程助手, 会自主匹配并激活合适技能后落地代码。",
        backend=backend,
        model=model,
        language="zh-CN",
        max_cycles=max(max_cycles, 1),
        enable_todo_management=True,
        use_workspace=True,
        agent_type="computer",
        skill_directories=[runtime_skills_root],
    )
    return AgentSDKClient(options=options, agents={"remotion_skill_agent": definition})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remotion skill demo with auto skill discovery")
    parser.add_argument("--settings-file", default="local_settings.py", help="Path to local_settings.py")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument(
        "--skills-dir",
        default="skills",
        help="Skills directory relative to workspace (or absolute path)",
    )
    parser.add_argument("--backend", default="moonshot", help="Backend key in local settings")
    parser.add_argument("--model", default="kimi-k2.5", help="Model key in backend config")
    parser.add_argument("--max-cycles", type=int, default=80, help="Max runtime cycles")
    parser.add_argument("--verbose", action="store_true", help="Print per-cycle logs")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    runtime_skills_root, prepared_skills = prepare_runtime_skill_bundle(
        workspace=workspace,
        skills_dir=args.skills_dir,
    )
    print("[skills] prepared runtime bundle:", flush=True)
    for item in prepared_skills:
        print(
            f"- {item['name']}: source={item['source']} -> runtime={item['runtime']}",
            flush=True,
        )

    client = build_client(
        settings_file=Path(args.settings_file),
        workspace=workspace,
        backend=args.backend,
        model=args.model,
        runtime_skills_root=runtime_skills_root,
        max_cycles=args.max_cycles,
        verbose=args.verbose,
    )

    run = client.run_agent(
        agent_name="remotion_skill_agent",
        prompt=build_prompt(runtime_skills_root=runtime_skills_root),
    )
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
