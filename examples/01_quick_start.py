#!/usr/bin/env python3
"""Quick start for embedding v-agent into a Python project."""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask


def log_handler(event: str, payload: dict[str, Any]) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    if event == "cycle_llm_response":
        print(
            f"[{now}] cycle={payload.get('cycle')} tool_calls={payload.get('tool_calls')} "
            f"assistant={payload.get('assistant_preview')}",
            flush=True,
        )
    elif event == "tool_result":
        print(
            f"[{now}] cycle={payload.get('cycle')} tool={payload.get('tool_name')} "
            f"status={payload.get('status')} directive={payload.get('directive')}",
            flush=True,
        )
    elif event in {"run_started", "run_completed", "run_wait_user", "run_max_cycles", "cycle_failed"}:
        print(f"[{now}] {event}: {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "请概述一下这个框架的特点")
    max_cycles = int(os.getenv("V_AGENT_EXAMPLE_MAX_CYCLES", "10"))
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
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
        user_prompt=prompt,
        max_cycles=max(max_cycles, 1),
    )

    try:
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
    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
