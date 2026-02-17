#!/usr/bin/env python3
"""Programmatic examples for running different agent profiles.

Inspired by SDK-style "agent definitions": each profile captures a reusable
set of model, prompt, runtime policy, and tool capability knobs.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask, NoToolPolicy

Language = Literal["zh-CN", "en-US"]


@dataclass(slots=True)
class AgentProfile:
    description: str
    backend: str
    model: str
    language: Language = "zh-CN"
    max_cycles: int = 10
    no_tool_policy: NoToolPolicy = "continue"
    allow_interruption: bool = True
    use_workspace: bool = True
    enable_todo_management: bool = True
    agent_type: str | None = None
    enable_document_tools: bool = False
    enable_document_write_tools: bool = False
    enable_workflow_tools: bool = False
    exclude_tools: list[str] = field(default_factory=list)


PROFILES: dict[str, AgentProfile] = {
    "researcher": AgentProfile(
        description="你是研究助理, 先检索材料再输出结构化结论。",
        backend="moonshot",
        model="kimi-k2.5",
        max_cycles=12,
        enable_todo_management=True,
    ),
    "translator": AgentProfile(
        description="你是专业翻译助理, 按段翻译并持续写入目标文件。",
        backend="minimax",
        model="MiniMax-M2.5",
        max_cycles=20,
        enable_todo_management=True,
    ),
    "computer": AgentProfile(
        description="你是桌面执行代理, 优先使用工具完成任务。",
        backend="moonshot",
        model="kimi-k2.5",
        max_cycles=16,
        agent_type="computer",
        enable_document_tools=True,
        enable_workflow_tools=True,
    ),
}


def run_profile(
    *,
    profile_name: str,
    prompt: str,
    settings_file: Path,
    workspace: Path,
) -> dict[str, object]:
    if profile_name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile: {profile_name}. Available: {available}")

    profile = PROFILES[profile_name]
    llm, resolved = build_openai_llm_from_local_settings(
        settings_file,
        backend=profile.backend,
        model=profile.model,
    )

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
    )

    system_prompt = build_system_prompt(
        profile.description,
        language=profile.language,
        allow_interruption=profile.allow_interruption,
        use_workspace=profile.use_workspace,
        enable_todo_management=profile.enable_todo_management,
        agent_type=profile.agent_type,
    )

    task = AgentTask(
        task_id=f"profile_{profile_name}_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_cycles=profile.max_cycles,
        no_tool_policy=profile.no_tool_policy,
        allow_interruption=profile.allow_interruption,
        use_workspace=profile.use_workspace,
        agent_type=profile.agent_type,
        enable_document_tools=profile.enable_document_tools,
        enable_document_write_tools=profile.enable_document_write_tools,
        enable_workflow_tools=profile.enable_workflow_tools,
        exclude_tools=list(profile.exclude_tools),
    )

    result = runtime.run(task)
    return {
        "profile": profile_name,
        "status": result.status.value,
        "final_answer": result.final_answer,
        "wait_reason": result.wait_reason,
        "error": result.error,
        "cycles": len(result.cycles),
        "todo_list": result.todo_list,
        "resolved": {
            "backend": resolved.backend,
            "selected_model": resolved.selected_model,
            "model_id": resolved.model_id,
            "endpoint": resolved.endpoint.endpoint_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v-agent with reusable code-level profiles")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES.keys()))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--settings-file", default="local_settings.py")
    parser.add_argument("--workspace", default="./workspace")
    args = parser.parse_args()

    output = run_profile(
        profile_name=args.profile,
        prompt=args.prompt,
        settings_file=Path(args.settings_file),
        workspace=Path(args.workspace),
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
