from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a v-agent task against configured LLM endpoint")
    parser.add_argument("--prompt", required=True, help="Task prompt")
    parser.add_argument("--backend", default="moonshot", help="Provider backend key in LLM_SETTINGS")
    parser.add_argument("--model", default="kimi-k2-thinking", help="Model key in provider models")
    parser.add_argument(
        "--settings-file",
        default=os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"),
        help="Path to local_settings.py",
    )
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--max-cycles", type=int, default=6, help="Max runtime cycles")
    parser.add_argument("--language", default="en-US", help="System prompt language (en-US / zh-CN)")
    parser.add_argument("--agent-type", default=None, help="Agent type, e.g. computer")
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
    )

    system_prompt = build_system_prompt(
        "You are Vector Vein agent runtime demo. Execute tasks with reliable tool usage and clear final outputs.",
        language=args.language,
        allow_interruption=True,
        use_workspace=True,
        enable_todo_management=True,
        agent_type=args.agent_type,
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
