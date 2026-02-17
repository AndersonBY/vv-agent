#!/usr/bin/env python3
"""SDK-style programmatic wrapper around v-agent runtime.

This example borrows the design spirit from claude-agent-sdk:
- reusable AgentDefinition registry
- options object for shared runtime settings
- client object that executes named agents
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentResult, AgentTask, NoToolPolicy

RuntimeLogHandler = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class AgentDefinition:
    description: str
    model: str
    backend: str | None = None
    language: str = "zh-CN"
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


@dataclass(slots=True)
class VAgentSDKOptions:
    settings_file: Path
    backend: str
    workspace: Path
    agents: dict[str, AgentDefinition]
    log_handler: RuntimeLogHandler | None = None


class VAgentClient:
    def __init__(self, options: VAgentSDKOptions) -> None:
        self.options = options

    def run_agent(
        self,
        *,
        agent_name: str,
        prompt: str,
        shared_state: dict[str, Any] | None = None,
    ) -> AgentResult:
        if agent_name not in self.options.agents:
            available = ", ".join(sorted(self.options.agents))
            raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")

        definition = self.options.agents[agent_name]
        backend = definition.backend or self.options.backend
        llm, resolved = build_openai_llm_from_local_settings(
            self.options.settings_file,
            backend=backend,
            model=definition.model,
        )

        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
            default_workspace=self.options.workspace,
            log_handler=self.options.log_handler,
        )

        system_prompt = build_system_prompt(
            definition.description,
            language=definition.language,
            allow_interruption=definition.allow_interruption,
            use_workspace=definition.use_workspace,
            enable_todo_management=definition.enable_todo_management,
            agent_type=definition.agent_type,
        )

        task = AgentTask(
            task_id=f"{agent_name}_{uuid.uuid4().hex[:8]}",
            model=resolved.model_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_cycles=max(definition.max_cycles, 1),
            no_tool_policy=definition.no_tool_policy,
            allow_interruption=definition.allow_interruption,
            use_workspace=definition.use_workspace,
            agent_type=definition.agent_type,
            enable_document_tools=definition.enable_document_tools,
            enable_document_write_tools=definition.enable_document_write_tools,
            enable_workflow_tools=definition.enable_workflow_tools,
            exclude_tools=list(definition.exclude_tools),
        )

        return runtime.run(task, shared_state=shared_state)


def build_example_client(*, settings_file: Path, backend: str, workspace: Path, verbose: bool) -> VAgentClient:
    def log_handler(event: str, payload: dict[str, Any]) -> None:
        if event in {"cycle_started", "cycle_llm_response", "tool_result", "run_completed", "run_wait_user", "cycle_failed"}:
            print(f"[{event}] {payload}")

    options = VAgentSDKOptions(
        settings_file=settings_file,
        backend=backend,
        workspace=workspace,
        agents={
            "planner": AgentDefinition(
                description="你是任务规划 Agent, 先拆任务, 再逐步执行并维护 todo。",
                model="kimi-k2.5",
                max_cycles=10,
                enable_todo_management=True,
            ),
            "translator": AgentDefinition(
                description="你是专业翻译 Agent, 按段翻译并持续写入目标文件。",
                model="MiniMax-M2.5",
                backend="minimax",
                max_cycles=20,
                enable_todo_management=True,
            ),
            "document-worker": AgentDefinition(
                description="你是文档处理 Agent, 重点依赖文档相关工具。",
                model="kimi-k2.5",
                max_cycles=12,
                enable_document_tools=True,
                enable_document_write_tools=True,
            ),
        },
        log_handler=log_handler if verbose else None,
    )
    return VAgentClient(options)


def main() -> None:
    parser = argparse.ArgumentParser(description="SDK-style v-agent example with named agents")
    parser.add_argument("--agent", required=True, choices=["planner", "translator", "document-worker"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--settings-file", default="local_settings.py")
    parser.add_argument("--backend", default="moonshot")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    client = build_example_client(
        settings_file=Path(args.settings_file),
        backend=args.backend,
        workspace=Path(args.workspace),
        verbose=args.verbose,
    )
    result = client.run_agent(agent_name=args.agent, prompt=args.prompt)

    print(
        json.dumps(
            {
                "agent": args.agent,
                "status": result.status.value,
                "final_answer": result.final_answer,
                "wait_reason": result.wait_reason,
                "error": result.error,
                "cycles": len(result.cycles),
                "todo_list": result.todo_list,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
