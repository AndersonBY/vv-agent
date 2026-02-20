#!/usr/bin/env python3
"""SDK 集成: 通过 AgentSDKOptions 使用 ThreadBackend + 流式输出 + 取消."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from vv_agent.runtime.backends.thread import ThreadBackend
from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {"cycle_started", "run_completed"}:
        print(f"\n  [{event}]", flush=True)


token_count = 0


def on_token(text: str) -> None:
    """流式回调: 逐 token 输出."""
    global token_count
    token_count += 1
    print(text, end="", flush=True)


agent = AgentDefinition(
    description="你是一个简洁的助手, 用最少的话回答问题.",
    model=os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5"),
    max_cycles=5,
)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()

    workspace.mkdir(parents=True, exist_ok=True)

    # SDK 层面配置 execution_backend 和 stream_callback
    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            log_handler=log_handler,
            execution_backend=ThreadBackend(max_workers=2),
            stream_callback=on_token,
        ),
        agent=agent,
    )

    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "什么是量子计算? 三句话回答")
    print(f"[demo] 提问: {prompt}\n")
    print("[demo] 流式输出:\n")

    try:
        run = client.run(prompt=prompt)
        print(f"\n\n[demo] 状态: {run.result.status.value}")
        print(f"[demo] 共收到 {token_count} 个 token 片段")
        print(f"[demo] cycles: {len(run.result.cycles)}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
