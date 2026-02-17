#!/usr/bin/env python3
"""Quick start for using v-agent in pure Python (no CLI wrapper)."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask


def _print_runtime_log(event: str, payload: dict[str, Any]) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    if event == "cycle_llm_response":
        print(
            f"[{now}] cycle={payload.get('cycle')} tool_calls={payload.get('tool_calls')} "
            f"assistant={payload.get('assistant_preview')}",
            flush=True,
        )
        return
    if event == "tool_result":
        print(
            f"[{now}] cycle={payload.get('cycle')} tool={payload.get('tool_name')} "
            f"status={payload.get('status')} directive={payload.get('directive')}",
            flush=True,
        )
        return
    if event in {"run_started", "run_completed", "run_wait_user", "run_max_cycles", "cycle_failed"}:
        print(f"[{now}] {event}: {payload}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Programmatic v-agent quick start")
    parser.add_argument("--prompt", required=True, help="Task prompt")
    parser.add_argument("--backend", default="moonshot", help="Backend key in local_settings.py")
    parser.add_argument("--model", default="kimi-k2.5", help="Model key in backend models")
    parser.add_argument("--settings-file", default="local_settings.py", help="Path to local_settings.py")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Print per-cycle runtime logs")
    args = parser.parse_args()

    llm, resolved = build_openai_llm_from_local_settings(
        Path(args.settings_file),
        backend=args.backend,
        model=args.model,
    )

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=Path(args.workspace),
        log_handler=_print_runtime_log if args.verbose else None,
    )

    system_prompt = build_system_prompt(
        "You are a reliable execution agent. Use tools explicitly and give clear final outputs.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
        enable_todo_management=True,
    )

    task = AgentTask(
        task_id=f"quickstart_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt=args.prompt,
        max_cycles=max(args.max_cycles, 1),
    )

    result = runtime.run(task)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "final_answer": result.final_answer,
                "wait_reason": result.wait_reason,
                "error": result.error,
                "cycles": len(result.cycles),
                "resolved": {
                    "backend": resolved.backend,
                    "selected_model": resolved.selected_model,
                    "model_id": resolved.model_id,
                    "endpoint": resolved.endpoint.endpoint_id,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
