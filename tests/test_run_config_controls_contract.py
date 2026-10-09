from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from vv_agent import RunConfig

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parity" / "run_config_controls.json"


def test_run_config_control_manifest_is_closed_and_matches_the_public_surface() -> None:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    controls = {entry["capability"]: entry for entry in contract["per_run_controls"]}

    assert contract["version"] == 2
    assert contract["framework_defaults"] == {
        "max_cycles": 10,
        "max_handoffs": 10,
        "no_tool_policy": "finish",
        "session_memory_enabled": False,
        "microcompaction_policy": {
            "trigger_ratio": 0.75,
            "target_ratio": 0.60,
            "keep_recent_cycles": 3,
            "min_result_chars": 500,
        },
        "budget_limits": None,
        "host_cost_meter": None,
    }
    assert contract["app_server_defaults"]["max_cycles"] == 80
    assert all(entry["status"] == "equivalent" for entry in controls.values())
    assert set(controls) == {
        "model_selection",
        "model_settings",
        "workspace",
        "session_history",
        "session_memory",
        "microcompaction_policy",
        "cycle_and_handoff_limits",
        "run_budget",
        "no_tool_policy",
        "tool_policy",
        "per_run_tool_registry",
        "cancellation",
        "approval",
        "event_store",
        "typed_run_events",
        "runtime_hooks_and_tracing",
        "application_context",
        "memory_providers",
        "initial_state",
        "cycle_injection",
        "diagnostics",
    }

    public_fields = {field.name for field in fields(RunConfig)}
    assert {
        "model",
        "model_provider",
        "model_settings",
        "workspace",
        "workspace_backend",
        "session_memory_enabled",
        "microcompaction_policy",
        "max_cycles",
        "max_handoffs",
        "no_tool_policy",
        "budget_limits",
        "host_cost_meter",
        "tool_policy",
        "tool_registry_factory",
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
        "log_preview_chars",
        "debug_dump_dir",
    } <= public_fields
