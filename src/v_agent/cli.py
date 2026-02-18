from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask


def _build_cli_log_handler(*, enabled: bool):
    if not enabled:
        return None

    def handler(event: str, payload: dict[str, Any]) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        if event == "run_started":
            print(
                f"[{now}] [run] start task={payload.get('task_id')} model={payload.get('model')} "
                f"max_cycles={payload.get('max_cycles')}",
                flush=True,
            )
            return
        if event == "cycle_started":
            print(
                f"[{now}] [cycle {payload.get('cycle')}] start "
                f"messages={payload.get('message_count')}",
                flush=True,
            )
            return
        if event == "cycle_llm_response":
            print(
                f"[{now}] [cycle {payload.get('cycle')}] llm "
                f"tool_calls={payload.get('tool_calls')} "
                f"assistant={payload.get('assistant_preview')}",
                flush=True,
            )
            return
        if event == "tool_result":
            print(
                f"[{now}] [cycle {payload.get('cycle')}] tool={payload.get('tool_name')} "
                f"status={payload.get('status')} directive={payload.get('directive')} "
                f"preview={payload.get('content_preview')}",
                flush=True,
            )
            return
        if event == "run_completed":
            print(f"[{now}] [run] completed: {payload.get('final_answer')}", flush=True)
            return
        if event == "run_wait_user":
            print(f"[{now}] [run] wait_user: {payload.get('wait_reason')}", flush=True)
            return
        if event == "run_max_cycles":
            print(f"[{now}] [run] max_cycles reached", flush=True)
            return
        if event == "cycle_failed":
            print(f"[{now}] [cycle {payload.get('cycle')}] failed: {payload.get('error')}", flush=True)
            return
        print(f"[{now}] [{event}] {payload}", flush=True)

    return handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a v-agent task against configured LLM endpoint")
    parser.add_argument("--prompt", required=True, help="Task prompt")
    parser.add_argument("--backend", default="moonshot", help="Provider backend key in LLM_SETTINGS")
    parser.add_argument("--model", default="kimi-k2.5", help="Model key in provider models")
    parser.add_argument(
        "--settings-file",
        default=os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"),
        help="Path to local_settings.py",
    )
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--max-cycles", type=int, default=80, help="Max runtime cycles")
    parser.add_argument("--language", default="zh-CN", help="System prompt language (en-US / zh-CN)")
    parser.add_argument("--agent-type", default=None, help="Agent type, e.g. computer")
    parser.add_argument("--verbose", action="store_true", help="Show per-cycle runtime logs")
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
        log_handler=_build_cli_log_handler(enabled=args.verbose),
        settings_file=Path(args.settings_file),
        default_backend=args.backend,
        tool_registry_factory=build_default_registry,
    )

    system_prompt = build_system_prompt(
        "You are Vector Vein agent runtime demo. Execute tasks with reliable tool usage and clear final outputs.",
        language=args.language,
        allow_interruption=True,
        use_workspace=True,
        enable_todo_management=True,
        agent_type=args.agent_type,
        workspace=Path(args.workspace),
    )

    task = AgentTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt=args.prompt,
        max_cycles=max(args.max_cycles, 1),
    )

    result = runtime.run(task)
    payload = {
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
