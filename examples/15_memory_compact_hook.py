#!/usr/bin/env python3
"""Memory compact hook example: audit and optionally pin messages before compaction."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from v_agent.runtime import BaseRuntimeHook, BeforeMemoryCompactEvent
from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import Message


class MemoryAuditHook(BaseRuntimeHook):
    """Log compaction stats and optionally preserve pinned messages."""

    def __init__(self, pin_keywords: list[str] | None = None) -> None:
        self.pin_keywords = [kw.lower() for kw in (pin_keywords or [])]
        self.compact_count = 0

    def before_memory_compact(self, event: BeforeMemoryCompactEvent) -> list[Message] | None:
        self.compact_count += 1
        total_msgs = len(event.messages)
        total_chars = sum(len(m.content or "") for m in event.messages)
        print(
            f"[MemoryAuditHook] compact #{self.compact_count}: "
            f"{total_msgs} messages, {total_chars} chars, cycle={event.cycle_index}",
            flush=True,
        )

        if not self.pin_keywords:
            return None  # Let default compaction proceed.

        pinned = [
            m for m in event.messages
            if m.content and any(kw in m.content.lower() for kw in self.pin_keywords)
        ]
        if pinned:
            print(f"[MemoryAuditHook] pinning {len(pinned)} messages matching {self.pin_keywords}", flush=True)
            # Return pinned messages; runtime replaces the full list.
            return pinned
        return None


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
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    pin_keywords = os.getenv("V_AGENT_EXAMPLE_PIN_KEYWORDS", "priority,critical").split(",")

    workspace.mkdir(parents=True, exist_ok=True)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            runtime_hooks=[MemoryAuditHook(pin_keywords=pin_keywords)],
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description="你是迭代执行 Agent. 每轮产出大量中间文本以触发 memory compaction.",
            model=model,
            backend=backend,
            max_cycles=30,
            enable_todo_management=True,
        ),
    )

    try:
        run = client.run(
            prompt=(
                "请逐步生成一份详细的技术方案文档, 包含背景、目标、方案设计、风险评估、"
                "实施计划等章节. 每个章节至少 200 字. 标记 priority 和 critical 的内容"
                "会在 memory compaction 时被保留. 完成后调用 `_task_finish`。"
            ),
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()