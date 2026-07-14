from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from vv_agent import RunConfig

FIXTURE = Path("tests/fixtures/parity/run_config_controls_v1.json")


def test_run_config_control_manifest_is_closed_and_matches_the_public_surface() -> None:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    controls = {entry["capability"]: entry for entry in contract["per_run_controls"]}

    assert contract["version"] == 1
    assert contract["framework_defaults"] == {
        "max_cycles": 10,
        "max_handoffs": 10,
        "no_tool_policy": "continue",
    }
    assert contract["app_server_defaults"]["max_cycles"] == 80
    assert all(entry["status"] == "equivalent" for entry in controls.values())
    assert set(controls) == {
        "model_selection",
        "model_settings",
        "workspace",
        "session_history",
        "cycle_and_handoff_limits",
        "no_tool_policy",
        "tool_policy",
        "per_run_tool_registry",
        "execution_backend",
        "cancellation",
        "approval",
        "event_store",
        "typed_run_events",
        "runtime_hooks_and_tracing",
        "application_context",
        "memory_providers",
        "initial_state",
        "cycle_injection",
        "sub_task_manager",
        "raw_runtime_observers",
        "diagnostics",
    }

    public_fields = {field.name for field in fields(RunConfig)}
    assert {
        "model",
        "model_provider",
        "model_settings",
        "workspace",
        "workspace_backend",
        "session",
        "max_cycles",
        "max_handoffs",
        "no_tool_policy",
        "tool_policy",
        "tool_registry_factory",
        "execution_backend",
        "cancellation_token",
        "approval_provider",
        "approval_broker",
        "approval_timeout_seconds",
        "event_store",
        "event_store_fail_closed",
        "stream",
        "hooks",
        "tracing",
        "context",
        "context_providers",
        "memory_providers",
        "shared_state",
        "initial_messages",
        "before_cycle_messages",
        "interruption_messages",
        "sub_task_manager",
        "runtime_log_handler",
        "runtime_stream_callback",
        "log_preview_chars",
        "debug_dump_dir",
    } <= public_fields
