#!/usr/bin/env python3
"""Temporarily inject tool schemas mid-run (demonstrates prompt-cache break risk).

WARNING:
- This example intentionally changes tool schemas between cycles.
- Doing this will change LLM request payloads and can break prompt-cache reuse.
- Use this pattern only when you truly need temporary tool exposure.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.runtime import BaseRuntimeHook, BeforeLLMEvent, BeforeLLMPatch, BeforeToolCallEvent
from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import Message, ToolExecutionResult, ToolResultStatus

EPHEMERAL_NOTE_TOOL_NAME = "_ephemeral_note"
EPHEMERAL_NOTE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EPHEMERAL_NOTE_TOOL_NAME,
        "description": (
            "Write one demo note to artifacts/ephemeral_notes.log. "
            "This tool is only used for temporary tool-injection demonstration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "One short note line to append.",
                }
            },
            "required": ["note"],
        },
    },
}


def ephemeral_note(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    note = str(arguments.get("note", "")).strip()
    if not note:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="note_required",
            content=json.dumps({"ok": False, "error": "`note` is required"}, ensure_ascii=False),
        )

    output_path = context.resolve_workspace_path("artifacts/ephemeral_notes.log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(note + "\n")

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=json.dumps(
            {
                "ok": True,
                "tool": EPHEMERAL_NOTE_TOOL_NAME,
                "note": note,
                "path": output_path.relative_to(context.workspace).as_posix(),
            },
            ensure_ascii=False,
        ),
    )


def build_registry_with_ephemeral_tool() -> ToolRegistry:
    registry = build_default_registry()
    registry.register_tool(
        name=EPHEMERAL_NOTE_TOOL_NAME,
        handler=ephemeral_note,
        description=EPHEMERAL_NOTE_TOOL_SCHEMA["function"]["description"],
        parameters=EPHEMERAL_NOTE_TOOL_SCHEMA["function"]["parameters"],
    )
    return registry


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


class TemporaryToolWindowHook(BaseRuntimeHook):
    """Inject one temporary tool for selected cycles.

    start_cycle <= cycle_index < end_cycle:
      - append ephemeral tool schema to tool_schemas
    other cycles:
      - keep default tool_schemas
    """

    def __init__(self, *, start_cycle: int, end_cycle: int, min_finish_cycle: int) -> None:
        self.start_cycle = max(1, start_cycle)
        self.end_cycle = max(self.start_cycle + 1, end_cycle)
        self.min_finish_cycle = max(min_finish_cycle, self.end_cycle)
        self.last_signature: str | None = None
        self.verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _schema_signature(tool_schemas: list[dict[str, Any]]) -> str:
        payload: list[dict[str, Any]] = []
        for schema in tool_schemas:
            fn = schema.get("function")
            if not isinstance(fn, dict):
                continue
            payload.append(
                {
                    "name": str(fn.get("name", "")),
                    "description": str(fn.get("description", "")),
                    "parameters": fn.get("parameters"),
                }
            )
        payload.sort(key=lambda item: item["name"])
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _tool_names(tool_schemas: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for schema in tool_schemas:
            fn = schema.get("function")
            if isinstance(fn, dict):
                name = str(fn.get("name", "")).strip()
                if name:
                    names.append(name)
        return names

    def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch | None:
        in_window = self.start_cycle <= event.cycle_index < self.end_cycle
        tool_schemas = deepcopy(event.tool_schemas)

        if in_window:
            names = self._tool_names(tool_schemas)
            if EPHEMERAL_NOTE_TOOL_NAME not in names:
                tool_schemas.append(deepcopy(EPHEMERAL_NOTE_TOOL_SCHEMA))

        signature = self._schema_signature(tool_schemas)
        if self.last_signature is not None and self.last_signature != signature:
            print(
                (
                    "[hook.temp_tool] WARNING: tool schema signature changed "
                    f"{self.last_signature} -> {signature}. "
                    "This can break LLM prompt cache and increase token cost."
                ),
                flush=True,
            )
        self.last_signature = signature

        if self.verbose:
            names = self._tool_names(tool_schemas)
            window_flag = "on" if in_window else "off"
            print(
                (
                    f"[hook.temp_tool] cycle={event.cycle_index} "
                    f"window={window_flag} tools={len(names)} signature={signature}"
                ),
                flush=True,
            )

        if not in_window:
            return None

        patched_messages = list(event.messages)
        patched_messages.append(
            Message(
                role="user",
                content=(
                    f"系统提示: 本轮临时开放 `{EPHEMERAL_NOTE_TOOL_NAME}` 工具用于演示。"
                    "注意: 这种动态增删 tools 会影响 prompt cache。"
                    "你可以按需调用一次, note 用简短文本即可。"
                ),
            )
        )
        return BeforeLLMPatch(messages=patched_messages, tool_schemas=tool_schemas)

    def before_tool_call(self, event: BeforeToolCallEvent) -> ToolExecutionResult | None:
        if event.call.name != TASK_FINISH_TOOL_NAME:
            return None
        if event.cycle_index >= self.min_finish_cycle:
            return None

        return ToolExecutionResult(
            tool_call_id=event.call.id,
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="demo_force_more_cycles",
            content=json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Demo guard: task_finish is temporarily blocked so you can observe "
                        "tool schema changes across cycles."
                    ),
                    "min_finish_cycle": self.min_finish_cycle,
                },
                ensure_ascii=False,
            ),
        )


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
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace/temp_tool_demo")).resolve()
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")

    start_cycle = _env_int("V_AGENT_TEMP_TOOL_START_CYCLE", 2)
    end_cycle = _env_int("V_AGENT_TEMP_TOOL_END_CYCLE", 4)
    min_finish_cycle = _env_int("V_AGENT_TEMP_TOOL_MIN_FINISH_CYCLE", 5)
    max_cycles = _env_int("V_AGENT_EXAMPLE_MAX_CYCLES", 7)

    workspace.mkdir(parents=True, exist_ok=True)
    context_file = workspace / "input" / "context.md"
    context_file.parent.mkdir(parents=True, exist_ok=True)
    if not context_file.exists():
        context_file.write_text(
            "\n".join(
                [
                    "# Demo Context",
                    "",
                    "- The goal is to demonstrate temporary tool injection.",
                    "- Keep final answer concise.",
                    "- Mention prompt-cache impact explicitly.",
                ]
            ),
            encoding="utf-8",
        )

    print(
        (
            "[example.warning] This demo intentionally mutates tool schemas between cycles. "
            "This may break prompt-cache reuse and increase cost."
        ),
        flush=True,
    )
    print(
        (
            f"[example.config] temp_tool_window=[{start_cycle}, {end_cycle}) "
            f"min_finish_cycle={min_finish_cycle} max_cycles={max_cycles}"
        ),
        flush=True,
    )

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            tool_registry_factory=build_registry_with_ephemeral_tool,
            runtime_hooks=[
                TemporaryToolWindowHook(
                    start_cycle=start_cycle,
                    end_cycle=end_cycle,
                    min_finish_cycle=min_finish_cycle,
                )
            ],
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description=(
                "你是运行时策略演示 Agent。你会按步骤读取上下文、在可用时使用临时工具、"
                "并最终总结风险与建议。"
            ),
            model=model,
            backend=backend,
            max_cycles=max_cycles,
            allow_interruption=False,
            enable_todo_management=True,
            use_workspace=True,
            no_tool_policy="continue",
        ),
    )

    try:
        run = client.run(
            prompt=(
                "请先读取 input/context.md 并给出执行计划。"
                f"当你看到 `{EPHEMERAL_NOTE_TOOL_NAME}` 可用时, 可调用一次写入简短 note。"
                f"最终请调用 `{TASK_FINISH_TOOL_NAME}`, 并明确说明动态增删 tools 对 prompt cache 的影响。"
            )
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Error during execution: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
