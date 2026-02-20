#!/usr/bin/env python3
"""Custom tool example: turn runtime logs into structured local tickets."""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import ToolExecutionResult, ToolResultStatus

TICKET_STORE_TOOL_NAME = "_ticket_store"


def _load_tickets(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        data = json.loads(db_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _save_tickets(db_path: Path, tickets: list[dict[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")


def ticket_store(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    action = str(arguments.get("action", "")).strip().lower()
    db_path = context.resolve_workspace_path("artifacts/tickets.json")
    tickets = _load_tickets(db_path)

    if action == "create":
        title = str(arguments.get("title", "")).strip()
        if not title:
            return ToolExecutionResult(
                tool_call_id="",
                status="error",
                status_code=ToolResultStatus.ERROR,
                error_code="missing_title",
                content=json.dumps({"ok": False, "error": "`title` is required"}, ensure_ascii=False),
            )
        severity = str(arguments.get("severity", "medium")).strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        ticket = {
            "id": f"T-{uuid.uuid4().hex[:8]}",
            "title": title,
            "description": str(arguments.get("description", "")).strip(),
            "severity": severity,
            "status": "open",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        tickets.append(ticket)
        _save_tickets(db_path, tickets)
        return ToolExecutionResult(
            tool_call_id="",
            content=json.dumps({"ok": True, "action": action, "ticket": ticket, "total": len(tickets)}, ensure_ascii=False),
        )

    if action == "list":
        return ToolExecutionResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "ok": True,
                    "action": action,
                    "count": len(tickets),
                    "tickets": tickets,
                    "db_path": db_path.relative_to(context.workspace).as_posix(),
                },
                ensure_ascii=False,
            ),
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code="invalid_action",
        content=json.dumps(
            {"ok": False, "error": "Invalid action. Use create/list."},
            ensure_ascii=False,
        ),
    )


def build_registry_with_ticket_tool() -> ToolRegistry:
    registry = build_default_registry()
    registry.register_tool(
        name=TICKET_STORE_TOOL_NAME,
        handler=ticket_store,
        description="Manage local support tickets. Actions: create, list.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "One of: create, list."},
                "title": {"type": "string", "description": "Ticket title (required for create)."},
                "description": {"type": "string", "description": "Ticket description (for create)."},
                "severity": {"type": "string", "description": "low/medium/high (for create)."},
            },
            "required": ["action"],
        },
    )
    return registry


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

    workspace.mkdir(parents=True, exist_ok=True)
    sample_log = workspace / "logs" / "app.log"
    if not sample_log.exists():
        sample_log.parent.mkdir(parents=True, exist_ok=True)
        sample_log.write_text(
            "\n".join(
                [
                    "2026-02-18T01:21:03Z ERROR payment retry timeout order=ORD-1001",
                    "2026-02-18T01:21:12Z WARN redis connection unstable",
                    "2026-02-18T01:22:09Z ERROR email provider 503 campaign=SPRING",
                    "2026-02-18T01:22:44Z ERROR search index lag exceeds threshold",
                ]
            ),
            encoding="utf-8",
        )

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            tool_registry_factory=build_registry_with_ticket_tool,
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description=(
                "你是 SRE 值班助理. 先从日志中提炼问题, 再调用自定义工单工具落盘,"
                "最后给出处理优先级建议。"
            ),
            model=model,
            backend=backend,
            max_cycles=18,
            enable_todo_management=True,
            extra_tool_names=[TICKET_STORE_TOOL_NAME],
        ),
    )

    try:
        run = client.run(
            prompt=(
                "请读取 logs/app.log, 提炼至少 3 个问题并调用 `_ticket_store` action=create 写入工单。"
                "然后调用 `_ticket_store` action=list 检查当前工单。"
                f"最后调用 `{TASK_FINISH_TOOL_NAME}` 输出简洁结论。"
            ),
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

