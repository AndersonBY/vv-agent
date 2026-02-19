#!/usr/bin/env python3
"""Error recovery example: detect stuck loops and retry with adjusted parameters."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import AgentStatus


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


def run_with_recovery(
    client: AgentSDKClient,
    prompt: str,
    *,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run agent with automatic retry on recoverable failures."""
    attempt = 0
    last_error: str | None = None

    while attempt <= max_retries:
        attempt += 1
        effective_prompt = prompt
        if last_error:
            effective_prompt = (
                f"[重试 #{attempt - 1}] 上次执行失败: {last_error}\n"
                f"请调整策略后重新执行:\n{prompt}"
            )

        print(f"\n--- Attempt {attempt}/{max_retries + 1} ---", flush=True)
        run = client.run(prompt=effective_prompt)
        status = run.result.status

        if status == AgentStatus.COMPLETED:
            print(f"[OK] Completed in {len(run.result.cycles)} cycles.", flush=True)
            return run.to_dict()

        if status == AgentStatus.WAIT_USER:
            print(f"[WAIT] Agent needs input: {run.result.wait_reason}", flush=True)
            return run.to_dict()

        if status == AgentStatus.MAX_CYCLES:
            last_error = f"Reached max cycles ({len(run.result.cycles)})"
            print(f"[RETRY] {last_error}", flush=True)
            continue

        if status == AgentStatus.FAILED:
            last_error = run.result.error or "Unknown failure"
            print(f"[RETRY] Failed: {last_error}", flush=True)
            continue

        # Unexpected status — don't retry.
        print(f"[UNEXPECTED] status={status.value}", flush=True)
        return run.to_dict()

    print(f"[EXHAUSTED] All {max_retries + 1} attempts failed.", flush=True)
    return {"status": "exhausted", "last_error": last_error}


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    max_retries = int(os.getenv("V_AGENT_EXAMPLE_MAX_RETRIES", "2"))

    workspace.mkdir(parents=True, exist_ok=True)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description="你是可靠执行 Agent. 完成任务后必须调用 `_task_finish`.",
            model=model,
            backend=backend,
            max_cycles=8,
            enable_todo_management=True,
        ),
    )

    try:
        result = run_with_recovery(
            client,
            "请列出 workspace 下的文件并输出简要说明, 然后调用 `_task_finish`。",
            max_retries=max_retries,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
