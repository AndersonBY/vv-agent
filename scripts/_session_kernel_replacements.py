"""Private v23 case migration and real-producer authoring for the v24 inventory."""

from __future__ import annotations

import base64
import importlib
import inspect
import json
import re
from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vv_agent import Agent, RunConfig, Runner, ToolPolicy
from vv_agent.budget import HostCost, RunBudgetLimits
from vv_agent.context_providers import ContextFragment
from vv_agent.events import event_from_dict
from vv_agent.model import ScriptedModelProvider
from vv_agent.output_validation import OutputValidationResult
from vv_agent.prompt import PromptBundle, PromptSection, SystemPromptBuilder, build_system_prompt_bundle
from vv_agent.run_config import effective_run_config
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.runtime.lifecycle import AfterCycleDecision, AfterCycleStop
from vv_agent.runtime.token_usage import normalize_token_usage, summarize_task_token_usage
from vv_agent.session.context import project_context
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.projection import project_records
from vv_agent.session.tracing import deliver_spans, project_spans
from vv_agent.tools.function import FunctionTool, function_tool
from vv_agent.tools.metadata import ToolMetadata
from vv_agent.tools.outputs import ToolOutputText
from vv_agent.types import (
    AgentResult,
    LLMResponse,
    Message,
    ModelCallRecord,
    SubAgentConfig,
    TaskTokenUsage,
    TokenUsage,
    ToolCall,
    ToolExecutionResult,
)
from vv_agent.workspace.memory import MemoryWorkspaceBackend

BASE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "parity"
REPLACE = tuple(
    [
        "after_cycle_hook.json",
        "app_server_observable.json",
        "approval_tool_policy.json",
        "bounded_tool_result.json",
        "budget_events.jsonl",
        "completion_policy.json",
        "configured_sub_agent.json",
        "configured_sub_agent_events.jsonl",
        "event_store_replay.jsonl",
        "handoff_contract.json",
        "llm_stream_projection.json",
        "manager_tool_envelope.json",
        "memory_lifecycle.json",
        "memory_local.json",
        "output_validation.json",
        "prompt_bundle.json",
        "public_api.json",
        "public_configured_sub_agent.json",
        "result_public.json",
        "run_budget.json",
        "run_config_controls.json",
        "run_definition.json",
        "run_events.jsonl",
        "run_events_invalid.json",
        "run_handle.json",
        "runner_events.jsonl",
        "runner_session_messages.jsonl",
        "runner_terminal.json",
        "runner_trace.jsonl",
        "runner_trace_spans.json",
        "session_codec.json",
        "session_items.jsonl",
        "token_usage.json",
        "tool_metadata.json",
    ]
)
KEEP = tuple(
    [
        "builtin_tools.json",
        "builtin_tool_behavior.json",
        "cli_contract.json",
        "model_ref.json",
        "model_settings.json",
        "assistant_reasoning_history.json",
        "bash_process_management.json",
    ]
)
# Proposal sections 4/6 and binding Q1/Q4/Q10. Negative vectors retain these as rejected inputs only.
RETIRED_FIELDS = frozenset(
    [
        "checkpoint_key",
        "checkpointKey",
        "resume_attempt",
        "consumed_revision",
        "checkpoint_config",
        "checkpoint_extensions",
        "reconciliation_provider",
        "require_reconciliation",
        "execution_backend",
        "resume_observations",
        "approval_snapshot",
        "session_host_binding_names",
        "session_max_handoffs",
        "session_handoff_targets",
        "session_input_messages",
        "session_input_blocked",
        "session_endpoint_order",
        "session_endpoint_id",
        "session_shared_state",
        "_vv_agent_session_memory_initial_state",
    ]
)
RETIRED_KINDS = frozenset(
    [
        "checkpoint_created",
        "checkpoint_resumed",
        "reconciliation_required",
        "reconciliation_resolved",
        "tool_call_deferred",
        "operation_replayed",
        "session_persisted",
        "deferred_result",
    ]
)
REASONS = {
    "after_cycle_hook.json": "section 4 F2d-1 hooks / F2d-2 boundary wire: committed boundary decisions",
    "app_server_observable.json": "section 5 and Q5-Q8: one status projection, retained owner, v2 wire",
    "approval_tool_policy.json": "section 4 F2d-1 approval deadline and Q7: absolute deadline and retained owner",
    "bounded_tool_result.json": "section 4 execution authority: record receipts replace old carriers; strict recovery retained",
    "budget_events.jsonl": "section 4 F2d-2 tool budget/internal tokens: record-derived accounting",
    "completion_policy.json": "section 4 F2d-1 waits/cancel: same-turn continuation and durable terminals",
    "configured_sub_agent.json": "section 4 F2d-3 delegation/added fields: one frozen child admission",
    "configured_sub_agent_events.jsonl": "section 4 F2d-2 child event ownership: parent-owned lifecycle",
    "event_store_replay.jsonl": "section 4 F2d-2 events and Q4: projection cursors, stable IDs, no invalid line",
    "handoff_contract.json": "section 4 F2d-3 handoff: admitted limits and terminal child adoption",
    "llm_stream_projection.json": "Q1: v6 scope; section 1.2 keeps provider delta bytes",
    "manager_tool_envelope.json": "section 4 F2d-3 child wait/background: running admission, terminal-only adoption",
    "memory_lifecycle.json": "section 4 F2d-2 rejected summary: logged internal calls and receipt reuse",
    "memory_local.json": "section 1.2 preserves summary/evidence vectors; section 4 replaces failure authority",
    "output_validation.json": "section 4 F2d-2 coercion/repair and Q1: logged output_repair and durable failures",
    "prompt_bundle.json": "section 1.2 rendered bytes retained; section 4 execution authority: session resume",
    "public_api.json": "section 6 and Q4/Q8/Q10: v8 planned exports, names and behavior for stores",
    "public_configured_sub_agent.json": "section 4 F2d-3 delegation: public child uses same admission",
    "result_public.json": "section 4 F2d-3 shared_state and Q1/Q4: session/turn references and current accounting",
    "run_budget.json": "section 4 F2d-2 batch/internal accounting and unavailable active wall interval",
    "run_config_controls.json": "section 6 removes execution selectors; Q10 creation seed",
    "run_definition.json": "section 4 F2d-3 admitted definition and Q3: frozen bindings, closed vv_session",
    "run_events.jsonl": "Q1: RunEvent v6 with required session identity; section 6 retired kinds removed",
    "run_events_invalid.json": "Q1/Q9: strict current versions/fields/kinds and full-match hashes",
    "run_handle.json": "section 4 execution authority and Q4: thin log/inbox controls",
    "runner_events.jsonl": "section 4 F2d-2 events: real producer ordering and stable identities",
    "runner_session_messages.jsonl": "section 4 F2d-1 ask_user/F2d-4 history: same-turn full transcript",
    "runner_terminal.json": "section 4 F2d-1 cancel/F2d-2 output: durable terminal before projections",
    "runner_trace.jsonl": "section 4 F2d-2 tracing: record-derived trace projection",
    "runner_trace_spans.json": "section 4 F2d-2 tracing: ACK-first at-most-once delivery",
    "session_codec.json": "section 1.2 strict Message content retained; Q9 strict hash; Q10 seed",
    "session_items.jsonl": "section 4 execution authority: transcript is a projection",
    "token_usage.json": "Q1: model-call v2/task-token-usage v3, TokenUsage v1 and output_repair",
    "tool_metadata.json": "section 4 execution authority: metadata/policy retained, retry governed by records",
}


def load(name):
    path = BASE / name
    if path.suffix == ".json":
        return json.loads(path.read_text())
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"invalid_jsonl": line})
    return rows


def resolve(path):
    pieces = path.split(".")
    for end in range(len(pieces), 0, -1):
        try:
            value = importlib.import_module(".".join(pieces[:end]))
        except ModuleNotFoundError:
            continue
        for name in pieces[end:]:
            value = getattr(value, name)
        return value
    raise AssertionError(path)


def reject(callback, value):
    try:
        callback(value)
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError(f"accepted invalid vector: {value!r}")


def signature(value):
    return {
        "async": inspect.iscoroutinefunction(value),
        "parameters": [
            {
                "kind": p.kind.name.lower(),
                "name": p.name,
                "required": p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD),
            }
            for p in inspect.signature(value).parameters.values()
            if not p.name.startswith("_")
        ],
    }


def result_facts(result):
    raw = result.raw_result
    return {
        "status": result.status.value,
        "completion_reason": result.completion_reason.value if result.completion_reason else None,
        "completion_tool_name": result.completion_tool_name,
        "final_output": result.final_output,
        "final_answer": raw.final_answer,
        "partial_output": result.partial_output,
        "wait_reason": result.wait_reason,
        "error": raw.error["message"] if raw.error else None,
        "error_code": result.error_code,
        "cycles": len(raw.cycles),
        "model_calls": len(result.token_usage.model_calls),
        "tool_execution_count": sum(e.type == "tool_call_started" for e in result.events),
        "session_id": raw.session_id,
        "turn_id": raw.turn_id,
    }


def start_runner(driver, session_id, agent, content, *, run_config=None):
    with (
        patch("vv_agent.session.surfaces.SessionDriver", return_value=driver),
        patch("vv_agent.runner.uuid.uuid4", return_value=SimpleNamespace(hex=session_id)),
    ):
        return Runner.start(agent, content, run_config=run_config)


class Author:
    def __init__(self, fixtures, independent, facts):
        self.f = fixtures
        self.independent = independent
        self.facts = facts
        self.outputs = {name: load(name) for name in REPLACE}
        self.evidence = {}
        self.keep = {}
        self.public_delta = {}
        self.handles = {}

    def execute(self, name, steps, *, agent=None, config=None, content="go"):
        sid = "c1c/" + name
        agent = agent or Agent("fixture", "Be precise.", model="m")
        from vv_agent.llm.scripted import ScriptedLLM

        provider = ScriptedModelProvider("test", "m", ScriptedLLM(steps), context_length=None, max_output_tokens=None)
        config = replace(self.f.config(config, steps), model_provider=provider, workspace="/fixture")
        handle = start_runner(self.f.kernel, sid, agent, content, run_config=config)
        self.f.runtimes[sid] = handle.runtime
        result = handle.result()
        assert AgentResult.from_dict(result.raw_result.to_dict()).to_dict() == result.raw_result.to_dict()
        self.handles[sid] = handle
        return result

    def evidence_for(self, name, sid):
        _, rows, _ = self.f.kernel.store.read_state(sid)
        return {"session_id": sid, "records": [r.record.record_id for r in rows]}

    def completion(self):
        d = self.outputs["completion_policy.json"]
        for case in d["cases"]:
            results = {r["tool_call_id"]: r for step in case["steps"] for r in step.get("tool_results", [])}
            tools = []
            for name in dict.fromkeys(c["name"] for step in case["steps"] for c in step["tool_calls"]):
                if name == "ask_user":
                    continue

                def handler(ctx, arguments, name=name, results=results):
                    del arguments
                    source = (
                        next(v for k, v in results.items() if k == ctx.metadata.get("tool_call_id"))
                        if "tool_call_id" in ctx.metadata
                        else next(v for v in results.values())
                    )
                    return ToolExecutionResult.from_dict({"status_code": "SUCCESS", **source})

                tools.append(
                    FunctionTool(name, name, {"type": "object", "properties": {}, "additionalProperties": False}, handler)
                )
            agent = Agent(
                "completion",
                "Follow the policy.",
                tools=tools,
                no_tool_policy=case["agent_policy"],
                tool_use_behavior=case["tool_use_behavior"],
                stop_at_tool_names=case.get("stop_at_tool_names", []),
            )
            configured = Runner.configured(RunConfig(no_tool_policy=case["runner_default_policy"]))
            config = effective_run_config(
                agent,
                RunConfig(no_tool_policy=case["run_policy"], max_cycles=case["max_cycles"]),
                runner_defaults=configured.default_run_config,
            )
            result = self.execute(
                case["name"],
                [LLMResponse(s["assistant_output"], [ToolCall.from_dict(c) for c in s["tool_calls"]]) for s in case["steps"]],
                agent=agent,
                config=config,
            )
            observed = result_facts(result)
            expected = {k: observed[k] for k in case["expected"] if k in observed}
            expected["effective_policy"] = self.f.runtimes["c1c/" + case["name"]].config.no_tool_policy
            expected["continuation_hint_emitted"] = any(
                m.role == "user" and m.content == "Continue working on the task." for m in result.raw_result.messages
            )
            case["expected"] = expected
            case["producer"] = self.evidence_for("completion", observed["session_id"])
        d.pop("approval_resume", None)
        d["same_turn"] = [self.evidence_for("wait", sid) for sid in ("user_wait", "turn_wait", "approval")]
        for key in (
            "approval_resume_uses_fresh_cycle_budget",
            "approved_resume_rejects_input_before_claim",
            "pre_cancelled_approval_resume_skips_side_effects",
        ):
            d["rules"].pop(key, None)
        d["rules"]["approval_resume_preserves_resource_budget"] = True

    def output_validation(self):
        d = self.outputs["output_validation.json"]
        for case in d["runner_cases"]:
            counts = {"validator_calls": 0, "repair_calls": 0}
            kind = case["validator"]

            def validator(value, context, counts=counts, kind=kind):
                del value, context
                counts["validator_calls"] += 1
                return (
                    OutputValidationResult.accept()
                    if kind == "pass" or (kind == "fail_then_pass" and counts["validator_calls"] == 2)
                    else OutputValidationResult.reject("invalid", "Invalid output.")
                )

            def repair(request, counts=counts, case=case):
                assert request.model is None
                counts["repair_calls"] += 1
                if case.get("repair") == "raises_provider_error":
                    raise RuntimeError("provider error")
                return LLMResponse("repaired" if case.get("repair") == "returns_valid" else "invalid")

            agent = Agent(
                "validation",
                "Return output.",
                output_validation_enabled=case["config"]["enabled"],
                output_validator=validator,
                output_repair=repair if case.get("repair") else None,
            )
            result = self.execute(case["name"], [LLMResponse("valid" if kind == "pass" else "invalid")], agent=agent)
            observed = counts | result_facts(result)
            observed["trace_unchanged"] = counts["validator_calls"] == counts["repair_calls"] == 0
            observed["second_repair_attempted"] = counts["repair_calls"] > 1
            _, records, _ = self.f.kernel.store.read_state(result.raw_result.session_id)
            observed["provider_error_is_observed"] = any(
                r.record.kind == "op_completed" and r.record.payload["result"].get("error_code") == "repair_provider_error"
                for r in records
            )
            case["expected"] = {k: observed[k] for k in case["expected"]}
            case["model_calls"] = [c.to_dict() for c in result.token_usage.model_calls]
            case["producer"] = self.evidence_for("validation", result.raw_result.session_id)
        d["repair"]["operation"] = "output_repair"
        d["durable_output_checked"] = self.evidence_for("repair", "repair")

    def budgets(self):
        d = self.outputs["run_budget.json"]

        class Meter:
            def __init__(self, readings):
                self.readings, self.last = iter(readings), None

            def read(self):
                self.last = next(self.readings, self.last)
                return HostCost.from_dict(self.last) if self.last else None

        for case in d["runner_cases"]:

            def response(s):
                usage = s.get("usage")
                raw = (
                    {"usage": {"total_tokens": usage["total_tokens"], "uncached_input_tokens": usage["uncached_input_tokens"]}}
                    if usage
                    else {}
                )
                return LLMResponse(s["assistant_output"], [ToolCall.from_dict(c) for c in s["tool_calls"]], raw=raw)

            token = CancellationToken()
            if case["pre_cancelled"]:
                token.cancel("cancelled by fixture")
            result = self.execute(
                case["name"],
                [response(s) for s in case["steps"]],
                config=RunConfig(
                    budget_limits=RunBudgetLimits.from_dict(case["limits"]) if case["limits"] is not None else None,
                    no_tool_policy=case["no_tool_policy"],
                    cancellation_token=token,
                    host_cost_meter=Meter(case["host_cost_readings"]) if case["host_cost_readings"] else None,
                ),
            )
            observed = result_facts(result)
            usage = result.budget_usage.to_dict() if result.budget_usage else None
            observed |= {
                "budget_usage": usage,
                "usage": usage,
                "budget_exhaustion": result.budget_exhaustion.to_dict() if result.budget_exhaustion else None,
                "budget_event_types": [e.type for e in result.events if e.type in {"budget_snapshot", "budget_exhausted"}],
                "unavailable_dimensions": usage["unavailable_dimensions"] if usage else [],
                "uncached_input_tokens": usage["uncached_input_tokens"] if usage else None,
                "tool_calls": usage["tool_calls"] if usage else 0,
            }
            case["expected"] = {k: observed[k] for k in case["expected"]}
            case["producer"] = self.evidence_for("budget", observed["session_id"])
        from vv_agent.budget import BudgetEvaluator, BudgetUsageSnapshot
        from vv_agent.types import CacheUsage

        for key, decoder in (("limits", RunBudgetLimits), ("snapshot", BudgetUsageSnapshot)):
            wire = d["wire_examples"][key]
            assert decoder.from_dict(wire).to_dict() == wire
            d["wire_examples"][key] = decoder.from_dict(wire).to_dict()
        for case in d["evaluator_cases"]:
            if case["name"].startswith("distributed_"):
                continue
            elapsed = [0]
            readings = case.get("observations", [])
            meter = Meter([r["host_cost"] for r in readings]) if readings else None
            if case.get("host_meter", {}).get("error") if case.get("host_meter") else False:

                class BrokenMeter:
                    def read(self):
                        raise RuntimeError("fixture meter failure")

                meter = BrokenMeter()
            initial = case.get("initial_usage", case.get("source_usage"))
            if "existing_tool_calls" in case:
                initial = BudgetUsageSnapshot(
                    tool_calls=case["existing_tool_calls"], tool_calls_by_name=case["existing_tool_calls_by_name"]
                ).to_dict()
            limits = RunBudgetLimits.from_dict(case.get("limits", {"max_wall_time_ms": 10000}))
            evaluator = BudgetEvaluator(
                limits,
                host_cost_meter=meter,
                initial_usage=BudgetUsageSnapshot.from_dict(initial) if initial else None,
                clock_ns=lambda elapsed=elapsed: elapsed[0] * 1_000_000,
            )
            exhaustion = None
            if readings:
                for observation in readings:
                    elapsed[0] = observation["elapsed_ms"]
                    method = observation["boundary"]
                    exhaustion = (
                        evaluator.model_call_complete(
                            TokenUsage(
                                total_tokens=0,
                                usage_source="provider_reported",
                                cache_usage=CacheUsage(status="provider_reported", uncached_input_tokens=0),
                            )
                        )
                        if method == "model_call_complete"
                        else getattr(evaluator, method)()
                    )
            elif "batch_tool_names" in case:
                evaluator.run_start()
                exhaustion = evaluator.preflight_tools(case["batch_tool_names"])
            elif "llm_usage" in case:
                usage = case["llm_usage"]
                exhaustion = evaluator.model_call_complete(
                    TokenUsage(
                        total_tokens=usage["total_tokens"],
                        usage_source="provider_reported",
                        cache_usage=CacheUsage(status="provider_reported", uncached_input_tokens=usage["uncached_input_tokens"]),
                    )
                )
            else:
                elapsed[0] = case.get("resumed_active_ms", 0)
                exhaustion = evaluator.run_start()
            snapshot = evaluator.snapshot().to_dict()
            observed = (exhaustion.to_dict() if exhaustion else {}) | {
                "terminal_exhaustion": exhaustion.to_dict() if exhaustion else None,
                "host_cost": snapshot["host_cost"],
                "total_tokens": snapshot["total_tokens"],
                "unavailable_reason": snapshot["unavailable_dimensions"][0]["reason"]
                if snapshot["unavailable_dimensions"]
                else None,
            }
            if "expected" in case:
                actual = {k: observed[k] for k in case["expected"]}
                assert actual == case["expected"], (case["name"], actual, case["expected"])
                case["expected"] = actual
            if "expected_usage" in case:
                assert snapshot == case["expected_usage"]
                case["expected_usage"] = snapshot
        for c in d["invalid_cases"]:
            reject(RunBudgetLimits.from_dict, c["limits"])
        d["evaluator_cases"] = [c for c in d["evaluator_cases"] if not c["name"].startswith("distributed_")]
        d["resume_semantics"] = {
            "active_interval": "durable_usage_observations",
            "lost_interval": "unavailable",
            "downtime_guessed": False,
        }
        d["model_call_accounting"].pop("checkpoint_terminal_record_and_budget_snapshot_are_atomic", None)
        d["model_call_accounting"]["operations"] = ["agent_cycle", "session_memory", "memory_compaction", "output_repair"]
        d["model_call_accounting"]["terminal_record_and_budget_snapshot_are_atomic"] = True
        d["recovery_cases"] = [c for c in self.f.recovery if c.get("case") == "lost_wall_interval"]

    def after_cycle(self):
        d = self.outputs["after_cycle_hook.json"]
        d.pop("distributed", None)
        d["durability"] = {
            "committed_boundary_reuses_decision": True,
            "precommit_may_rerun": True,
            "external_effects_must_be_idempotent": True,
        }
        d["default_behavior"].pop("checkpoint_extension_state_added", None)
        d["permission_state"].pop("survives_current_checkpoint_round_trip", None)
        d["permission_state"]["survives_session_resume"] = True
        d["runner_cases"] = [c for c in d["runner_cases"] if not c["name"].startswith("checkpoint_")]
        for case in d["runner_cases"]:
            seen = []
            decisions = [raw for h in case.get("hooks", []) for raw in h.get("decisions", [h.get("decision")]) if raw]

            class Hook:
                def after_cycle(self, snapshot, seen=seen, decisions=decisions):
                    seen.append(snapshot.cycle_index)
                    raw = decisions[min(len(seen) - 1, len(decisions) - 1)]
                    return AfterCycleDecision(**(raw | {"stop": AfterCycleStop(**raw["stop"]) if raw["stop"] else None}))

            cycles = case.get("cycles", [{"assistant": "done", "tool_calls": []}])
            steps = [LLMResponse(c["assistant"], [ToolCall.from_dict(t) for t in c["tool_calls"]]) for c in cycles]
            if case["name"] == "disabled_is_exact_noop":
                steps.append(LLMResponse("done"))
            if case["name"] == "permission_narrowing_is_enforced_and_durable":
                steps = [
                    LLMResponse("first"),
                    LLMResponse("", [ToolCall("denied", "bash", {"command": "true"})]),
                    LLMResponse("done"),
                ]

            def visible_native_tools():
                from vv_agent.tools.registry import ToolRegistry

                native = ToolRegistry()
                for name in ("bash", "read_file"):
                    native.register_executor(self.f.registry((name,)).get_executor(name), planner_extra=True)
                return native

            result = self.execute(
                case["name"],
                steps,
                config=RunConfig(
                    tool_registry_factory=visible_native_tools
                    if case["name"] == "permission_narrowing_is_enforced_and_durable"
                    else None,
                    after_cycle_hooks=[Hook()] if decisions else [],
                    max_cycles=1 if case["name"] == "steer_at_max_cycles_fails_closed" else case.get("max_cycles", 3),
                    no_tool_policy="continue"
                    if case["name"] == "permission_narrowing_is_enforced_and_durable"
                    else case.get("no_tool_policy", "finish"),
                ),
            )
            from vv_agent.runtime.lifecycle import AFTER_CYCLE_CONTROL_STATE_KEY

            messages = result.raw_result.messages
            control = result.raw_result.shared_state.get(AFTER_CYCLE_CONTROL_STATE_KEY)
            observed = result_facts(result) | {
                "hook_invocations": len(seen),
                "next_user_messages": [m.content for m in messages if m.role == "user" and m.content != "go"],
                "reserved_state_present": control is not None,
                "native_outcome": "continue" if case.get("no_tool_policy") == "continue" else "completed",
                "messages_added_by_hook": sum(len(raw["steering_messages"]) for raw in decisions),
                "injected_user_messages": [m.content for m in messages if m.role == "user" and m.content != "go"],
                "default_continuation_hint_count": sum(m.content == "Continue working on the task." for m in messages),
                "dispatch_of_bash": next((r.error_code for c in result.raw_result.cycles for r in c.tool_results), None),
                "durable_control_state": control,
                "next_available_tools": [
                    schema["function"]["name"]
                    for schema in next(
                        reversed(
                            [
                                op.attempts[1].plan.payload["request"]["tools"]
                                for op in self.f.state(result.raw_result.session_id).operations.values()
                                if op.kind == "model" and op.attempts[1].plan.payload["purpose"] == "primary"
                            ]
                        ),
                        [],
                    )
                ],
                "run_completed_event_count": sum(e.type == "run_completed" for e in result.events),
                "additional_model_calls": max(0, len(result.token_usage.model_calls) - 1),
            }
            case["expected"] = {k: observed[k] for k in case["expected"]}
            case["producer"] = self.evidence_for("hook", result.raw_result.session_id)
        for c in d["invalid_decisions"]:

            def parse(raw):
                return AfterCycleDecision(**(raw | {"stop": AfterCycleStop(**raw["stop"]) if raw["stop"] else None}))

            reject(parse, c["decision"])
        d["boundary_cases"] = [self.evidence_for("hook", sid) for sid in self.f.runtimes if "boundary" in sid]

    def prompts(self):
        d = self.outputs["prompt_bundle.json"]
        for case in d["scenarios"]:
            inputs, producer = case["input"], case["producer"]
            if producer == "build_system_prompt_bundle":
                bundle = build_system_prompt_bundle(
                    inputs["original_system_prompt"],
                    language=inputs["language"],
                    allow_interruption=inputs["allow_interruption"],
                    use_workspace=inputs["use_workspace"],
                    enable_todo_management=inputs["enable_todo_management"],
                    agent_type=inputs["agent_type"],
                    available_sub_agents=inputs["available_sub_agents"],
                    available_skills=inputs["available_skills"],
                    current_time_utc=datetime.fromisoformat(inputs["current_time_utc"].replace("Z", "+00:00")),
                    session_memory_enabled=inputs.get("session_memory_enabled", False),
                    session_memory_context=inputs["session_memory_context"],
                )
            elif producer == "SystemPromptBuilder":
                builder = SystemPromptBuilder()
                for raw in inputs["sections"]:
                    if raw["text"].strip():
                        builder.add_section(PromptSection.from_dict(raw))
                bundle = builder.build_result()
            elif producer == "PromptBundle":
                bundle = PromptBundle(tuple(PromptSection.from_dict(raw) for raw in inputs["sections"]))
            else:
                bundle, omitted = AgentCompiler._assemble_prompt_bundle(
                    instructions=PromptBundle(
                        tuple(PromptSection.from_dict(s) for s in inputs["instruction_bundle"]["sections"])
                    ),
                    compiler_sections=[PromptSection.from_dict(raw) for raw in inputs["compiler_owned_sections"]],
                    provider_fragments=[ContextFragment(**raw) for raw in inputs["provider_fragments"]],
                    max_prompt_chars=None,
                )
                assert not omitted
            sections = list(bundle.sections)
            if "computer_os" in case.get("normalizations", []):
                sections = [replace(s, text=re.sub(r"(?:Windows|macOS|Linux)", "<OS>", s.text)) for s in sections]
                bundle = PromptBundle(tuple(sections))
            raw = self.independent([[s.to_dict() for s in bundle.sections if s.stable]])[0]
            expected = {
                "sections": [s.to_dict() for s in bundle.sections],
                "flat_prompt": bundle.flatten(),
                "stable_hash": sha256(raw).hexdigest(),
            }
            if producer == "AgentCompiler":
                expected = {
                    "section_ids": [s.id for s in bundle.sections],
                    "flat_prompt": bundle.flatten(),
                    "stable_hash": sha256(raw).hexdigest(),
                }
            assert bundle.stable_hash == expected["stable_hash"]
            assert expected == case["output"], case["id"]
            case["output"] = expected
        scenarios = {c["id"]: c for c in d["scenarios"]}
        for vector in d["stable_hash_vectors"]:
            sections = scenarios[vector["scenario_ref"]]["output"]["sections"]
            raw = self.independent([[s for s in sections if s["stable"]]])[0]
            vector.update(
                canonical_json_base64=base64.b64encode(raw).decode(),
                canonical_json_utf8_bytes=len(raw),
                sha256=sha256(raw).hexdigest(),
            )
        d["run_scope"].pop("checkpoint_resume", None)
        d["run_scope"]["run_definition"] = {
            "carrier": "turn_started.definition",
            "field": "task.prompt_bundle",
            "validation_owner": "opaque J in turn_started.definition",
        }
        d["run_scope"]["session_resume"] = "reuse_frozen_definition_without_reinvoking_producers"
        d["run_scope"]["conformance_cases"] = [
            c for c in d["run_scope"]["conformance_cases"] if "checkpoint" not in json.dumps(c)
        ]
        d["invalid_cases"] = [c for c in d["invalid_cases"] if "run_definition" not in c["name"]]
        d["resume_producer"] = self.evidence_for("prompt", "tools")

    def public_api(self):
        d = self.outputs["public_api.json"]
        for domain in d["domains"]:
            for capability in domain["capabilities"]:
                resolve(capability["python"])
        for surface in d["surfaces"]:
            target = resolve(surface["python_target"])
            for group in ("members", "protocol_operations", "supporting_operations"):
                for member in surface.get(group, []):
                    py = member["python"]
                    owner = resolve(py["target"]) if "target" in py else target
                    if py["kind"] in {"method", "function", "property"}:
                        value = (
                            inspect.getattr_static(owner, py["name"]) if py["kind"] == "property" else getattr(owner, py["name"])
                        )
                        assert signature(value.fget if isinstance(value, property) else value) == py["signature"], (
                            surface["id"],
                            py["name"],
                        )
                    else:
                        assert py["name"] in {f.name for f in fields(owner)} or any(
                            py["name"] in getattr(b, "__annotations__", {}) for b in owner.__mro__
                        )
        self.public_delta = {"added": [], "removed": []}

    def usage(self):
        d = self.outputs["token_usage.json"]
        d["wire"].update(model_call_schema="vv-agent.model-call.v2", task_token_usage_schema="vv-agent.task-token-usage.v3")
        d["model_call_operations"] = ["agent_cycle", "session_memory", "memory_compaction", "output_repair"]
        d["task_usage_rules"]["schema_version"] = "vv-agent.task-token-usage.v3"
        for c in d["normalization_cases"]:
            i = c["input"]
            expected = normalize_token_usage(
                i["raw_usage"], usage_source=i["usage_source_hint"], cache_status=i["cache_status_hint"]
            ).to_dict()
            assert expected == c["expected"]
            c["expected"] = expected
        for c in d["invalid_wire_cases"]:
            if c["name"] == "unsupported_schema_version":
                c["replace"]["schema_version"] = "unsupported"
                reject(TokenUsage.from_dict, TokenUsage().to_dict() | c["replace"])
            if "input" in c:
                reject(TokenUsage.from_dict, c["input"])
        for c in d["invalid_task_wire_cases"]:
            if c["name"] == "unsupported_schema_version":
                c["mutation"]["replace"]["schema_version"] = "unsupported"
                reject(
                    lambda value: TaskTokenUsage.from_dict(value),
                    TaskTokenUsage().to_dict() | c["mutation"]["replace"],
                )
        from vv_agent.types import CacheUsage

        for c in d["aggregation_cases"]:
            summary = TaskTokenUsage()
            for index, observation in enumerate(c["model_calls"], 1):
                record = ModelCallRecord(
                    call_id=f"cache/{index}:1",
                    operation_id=f"cache/{index}",
                    attempt=1,
                    operation="agent_cycle",
                    cycle_index=index,
                    backend="test",
                    model="m",
                    status="completed",
                    usage=TokenUsage(
                        total_tokens=1, usage_source="provider_reported", cache_usage=CacheUsage.from_dict(observation)
                    ),
                )
                summary.add_model_call(record)
            actual = summary.cache_usage.to_dict()
            assert actual == c["expected"]
            c["expected"] = actual
        for c in d["task_aggregation_cases"]:
            calls = [
                ModelCallRecord.from_dict(raw | {"schema_version": "vv-agent.model-call.v2"}) for raw in c.get("model_calls", [])
            ]
            summary = summarize_task_token_usage(calls)
            c["model_calls"] = [call.to_dict() for call in calls]
            c["expected"] = {k: v for k, v in summary.to_dict().items() if k not in {"schema_version", "model_calls"}} | {
                "model_call_count": len(calls)
            }
        for case, sid in zip(d["compaction_cases"], ("micro", "summary", "second_summary", "summary_receipt"), strict=True):
            usage = self.f.result(sid).token_usage
            calls = [c.to_dict() for c in usage.model_calls if c.operation.value == "memory_compaction"]
            replay = case["name"] == "summary_receipt_replay"
            case["input"] = {
                "session_id": sid,
                "action": "reuse_summary_receipt" if replay else "drive",
                "transcript_ref": f"session_projection.json#/source_records/{sid}",
            }
            case["expected"] = {
                "model_calls": calls,
                "new_model_dispatches": 0 if replay else len(calls),
                "new_model_call_records": 0 if replay else len(calls),
                "new_budget_total_tokens": 0
                if replay or not calls
                else (
                    sum(c["usage"]["total_tokens"] for c in calls)
                    if all(c["usage"]["total_tokens"] is not None for c in calls)
                    else None
                ),
            }
            case["producer"] = self.evidence_for("compaction", sid)
        d["compaction_case_execution"] = {
            "transcript_ref": "session_projection session identity; raw record corpus supplies immutable source facts",
            "expected_ledger": "current logged compaction attempts, excluding primary calls from the compaction-only delta",
            "receipt_reuse": "summary_receipt kill cut resumes without a second summary dispatch",
            "unreported_usage": "null, never a fabricated zero measurement",
        }
        d["producer_cases"] = []
        for sid in ("repair", "memory", "summary", "model_unknown_retry"):
            if sid not in self.f.runtimes:
                continue
            result = self.f.result(sid)
            wire = result.token_usage.to_dict()
            assert TaskTokenUsage.from_dict(wire).to_dict() == wire
            d["producer_cases"].append({"name": sid, "usage": wire, "producer": self.evidence_for("usage", sid)})
        for collection in ("invalid_task_wire_cases", "invalid_model_call_cases"):
            for c in d[collection]:
                if "input" not in c:
                    continue
                value = deepcopy(c["input"])
                if collection == "invalid_task_wire_cases" and value.get("schema_version") == "vv-agent.task-token-usage.v2":
                    value["schema_version"] = "vv-agent.task-token-usage.v3"
                    for call in value.get("model_calls", []) if isinstance(value.get("model_calls"), list) else []:
                        call["schema_version"] = "vv-agent.model-call.v2"
                if collection == "invalid_model_call_cases":
                    value["schema_version"] = "vv-agent.model-call.v2"
                reject(
                    lambda x, collection=collection: (
                        TaskTokenUsage if collection == "invalid_task_wire_cases" else ModelCallRecord
                    ).from_dict(x),
                    value,
                )
                c["input"] = value

    def codecs(self):
        d = self.outputs["session_codec.json"]
        for case in d["canonical_cases"]:
            actual = Message.from_dict(case["input"]).to_dict()
            assert actual == case["canonical"]
            case["canonical"] = actual
        for case in d["invalid_cases"]:
            reject(lambda value: Message.from_dict(value), case["input"])
        d["transcript_authority"] = "read_only_record_projection"
        # The bytes are consumed as a creation-time seed by the real kernel.
        content = [load("session_items.jsonl"), load("runner_session_messages.jsonl")]
        for name, messages in zip(("session_items.jsonl", "runner_session_messages.jsonl"), content, strict=True):
            sid = "c1c/" + name
            parsed = [Message.from_dict(m) for m in messages]
            self.f.admit(sid, [], attributes={"seed": {"messages": [m.to_dict() for m in parsed], "shared_state": {}}})
            state, rows, _ = self.f.kernel.store.read_state(sid)
            actual = [m.to_dict() for m in project_context(rows, state)]
            assert actual == messages
            self.outputs[name] = actual
        d["same_turn_wait_producer"] = self.evidence_for("messages", "user_wait")

    def bounded(self):
        d = self.outputs["bounded_tool_result.json"]
        for name, raw in d["canonical_results"].items():
            backend = MemoryWorkspaceBackend()
            raw = deepcopy(raw)
            for pointer in ("artifact", "cursor"):
                if pointer in raw:
                    raw[pointer]["sha256"] = sha256(b"placeholder input").hexdigest()
            result = ToolExecutionResult.from_dict(raw)
            # v23 hash strings were placeholders. Bind v24 recovery pointers to independent bytes.
            if result.artifact:
                source = ("B" if name == "truncated_bash" else "E") * result.artifact.size_bytes
                backend.write_text_exclusive(result.artifact.path, source)
                result.artifact = replace(result.artifact, sha256=sha256(source.encode()).hexdigest())
            if result.cursor:
                source = "R" * result.original_bytes
                backend.write_text(result.cursor.path, source)
                result.cursor = replace(result.cursor, sha256=sha256(source.encode()).hexdigest())
            tool = FunctionTool(
                "bounded",
                "Return a retained bounded result.",
                {"type": "object", "properties": {}},
                lambda _ctx, _args, r=result: r,
            )
            actual = self.execute(
                "bounded_" + name,
                [LLMResponse("", [ToolCall(raw["tool_call_id"], "bounded", {})]), LLMResponse("done")],
                agent=Agent("bounded", "Use bounded.", tools=[tool]),
                config=RunConfig(workspace_backend=backend),
            )
            produced = actual.raw_result.cycles[0].tool_results[0]
            d["canonical_results"][name] = produced.to_dict()
            for case in d["tool_message_projection"]["cases"]:
                if case["result_ref"] == name:
                    expected = produced.content
                    if produced.truncated:
                        recovery = {
                            key: produced.to_dict()[key]
                            for key in d["tool_message_projection"]["recovery_fields"]
                            if key in produced.to_dict()
                        }
                        expected += "\n" + self.independent([{"vv_agent_recovery": recovery}])[0].decode()
                    assert next(m.content for m in actual.raw_result.messages if m.role == "tool") == expected
                    case["expected_message"] = expected
        d["artifact_contract"]["retention"] = "retained_session_artifact_policy"
        d["durability"]["preserve_sparse_fields_through"] = ["tool_message", "cycle_record", "agent_result", "op_completed"]
        for key in list(d["result_contract"]):
            if "deferred" in key:
                d["result_contract"].pop(key)

    def app_observable(self, app):
        d = self.outputs["app_server_observable.json"]
        notices = [m for row in app["transcripts"] for m in row.get("notifications", [])]
        responses = [r for row in app["transcripts"] for r in row.get("responses", []) if "result" in r]

        def named(method):
            return [m for m in notices if m.get("method") == method]

        d["version"] = 4
        d["protocolVersion"] = app["protocol_version"]
        d["restart"]["staleRunningThreadStatus"] = "running"
        d["approval"]["disconnectDecision"] = "retained_owner_until_absolute_deadline"
        d["approval"]["owner"] = app["facts"]["owner"]
        d["approval"]["deadline_ms"] = app["facts"]["deadline_ms"]
        d["approval"]["observerCannotApprove"] = app["facts"]["observer_cannot_approve"]
        d["terminal"]["tokenUsageProjection"].update(
            sourceSchemaVersion="vv-agent.task-token-usage.v3",
            value=next(m["params"]["tokenUsage"] for m in named("turn/completed") if "tokenUsage" in m["params"]),
        )
        d["terminal"]["agentStatusProjection"] = [
            c
            for c in d["terminal"]["agentStatusProjection"]
            if "deferred" not in json.dumps(c) and "reconciliation" not in json.dumps(c)
        ]
        terminal = json.loads(app["schemas"]["jsonSchema"]["ServerNotification"])["$defs"]["TurnCompletedParams"]
        assert terminal["additionalProperties"] is False
        d["terminal"]["optionalFieldsOmittedWhenAbsent"] = sorted(set(terminal["properties"]) - set(terminal["required"]))
        d["terminal"]["threadStatusValues"] = ["idle", "running", "interrupted", "archived", "closed"]
        d["liveReplay"]["item"] = next(m["params"] for m in named("item/completed"))
        started = [m for m in named("item/started") if m["params"]["type"] == "toolCall"]
        completed = [m for m in named("item/completed") if m["params"]["type"] == "toolCall"]
        d["toolLifecycle"]["executed"] = {"startedNotifications": started, "completedNotifications": completed}
        for stage, method in (("started", "item/started"), ("completed", "item/completed")):
            d["modelLifecycle"][stage + "Notifications"] = [m for m in named(method) if m["params"]["type"] == "modelCall"]
        d["modelLifecycle"]["failedNotifications"] = []
        d["durableResume"] = {
            "method": "turn/resume",
            "requestFields": ["threadId", "turnId"],
            "response": "TurnResumeResponse",
            "newInputAllowed": False,
            "responseFields": list(json.loads(app["schemas"]["jsonSchema"]["TurnResumeResponse"])["properties"]),
            "protocolCases": [
                row for row in app["transcripts"] if row.get("request", {}).get("method") in {"turn/resume", "thread/resume"}
            ],
            "projectionCases": [m["params"] for m in named("turn/completed")],
            "closedExecutionResumeError": {"code": -32602, "message": "Thread is closed"},
        }
        for row in d["durableResume"]["protocolCases"]:
            request = row.get("request", {})
            if "checkpointKey" in request.get("params", {}):
                raw = self.independent([request])[0]
                row["rejected_request"] = self.facts(raw)
                row.pop("request", None)
        action = d["actionAdmission"]
        for key in (
            "frameworkReceipt",
            "serverDerivedCommand",
            "internalDerivations",
            "publicReceiptFields",
            "receiptInternalFieldsNeverReturned",
            "internalFieldsRejected",
            "receiptInternalFieldsHidden",
        ):
            action.pop(key, None)
        action["responseFields"] = ["threadId", "turnId", "actionId", "accepted", "status"]
        action["askUserSameTurnContinuation"] = True
        action.pop("askUserTerminalSemanticsUnchanged", None)
        action["hostPromptProjection"] = {
            "status": "interrupted",
            "fields": ["threadId", "status", "waitReason", "prompt", "interactionId", "sessionId", "childTurnId", "interactions"],
            "readSurface": [r["result"] for r in responses if r["result"].get("status") == "interrupted"],
        }
        for vector in action["commandIdCases"]:
            self.action_id(vector)
        action["producerCases"] = [row for row in app["transcripts"] if row.get("request", {}).get("method") == "turn/action"]
        d["actionAdmission"] = action
        d["schema"]["json"] = list(app["schemas"]["jsonSchema"])
        d["schema"]["typescript"] = list(app["schemas"]["typescript"])
        d["statusProjection"] = [r["result"] for r in responses if "status" in r["result"] and "threadId" in r["result"]]

    def action_id(self, vector):
        value = {
            "schema_version": "vv-agent.controller-command-id.v1",
            "thread_id": vector.get("threadId", vector.get("thread_id")),
            "turn_id": vector.get("turnId", vector.get("turn_id")),
            "action_id": vector.get("actionId", vector.get("action_id")),
        }
        if any(v is None for v in value.values()):
            return
        raw = self.independent([value])[0]
        identity = sha256(b"vv-agent.controller-command-id.v1\0" + len(raw).to_bytes(8, "big") + raw).hexdigest()
        from vv_agent.interaction import derive_controller_command_id

        assert (
            derive_controller_command_id(thread_id=value["thread_id"], turn_id=value["turn_id"], action_id=value["action_id"])
            == identity
        )
        for key in ("commandId", "command_id", "expected", "expectedCommandId"):
            if key in vector and isinstance(vector[key], str):
                vector[key] = identity
        vector["bytes_base64"] = base64.b64encode(raw).decode()
        vector["sha256"] = sha256(raw).hexdigest()

    def children(self):
        d = self.outputs["configured_sub_agent.json"]
        parent = "children"
        _, rows, _ = self.f.kernel.store.read_state(parent)
        admitted = [r.record for r in rows if r.record.kind == "op_parked" and r.record.payload["handle"]["kind"] == "child"]
        child_sid = admitted[0].payload["handle"]["session_id"]
        child_state, _, _ = self.f.kernel.store.read_state(child_sid)
        child_turn = next(iter(child_state.turns.values())).start
        task = child_turn.payload["definition"]["task"]
        d["identity"] = {
            "session_id": child_sid,
            "turn_id": child_turn.turn_id,
            "parent_run_id": admitted[0].turn_id,
            "parent_operation_id": admitted[0].operation_id,
            "record_derived": True,
        }
        d["task_projection"] = task
        d["metadata_projection"] = task["metadata"]
        d["reserved_metadata_keys"] = ["vv_session"]
        d["capability_projection"] = {"frozen_admission": admitted[0].payload["handle"]}
        d["continuation"] = {"same_turn_reply": True, "terminal_only_parent_adoption": True, "full_history_from_records": True}
        d["lifecycle"] = {
            "event_sequence": [e.type for e in project_records(rows) if e.type.startswith("sub_run_")],
            "parent_owned": True,
        }
        d["manager"] = {
            "status": "record_projection",
            "admission": "atomic_child_session",
            "terminal_delivery": "authenticated_inbox_and_cursor_transaction",
        }
        d["producer_cases"] = [
            {"name": "configured_children", "admission": r.payload, "record_id": r.record_id} for r in admitted
        ]
        public = self.outputs["public_configured_sub_agent.json"]
        normalization = public["normalization"]
        configs = {c["id"]: SubAgentConfig.from_dict(c["config"]) for c in normalization["raw_entries"]}
        agent = Agent("coordinator", public["projection"]["sections"][0]["text"], sub_agents=configs)
        assert list(agent.sub_agents) == normalization["normalized_ids"]
        normalization["retained_researcher_config"] = agent.sub_agents["researcher"].to_dict()
        provider = ScriptedModelProvider.new("test", "shared-model", [])
        rt = self.f.kernel.runtime(agent, self.f.config(RunConfig(model_provider=provider, workspace="/fixture")))
        compiled = rt.compile("Delegate the research.", "public-child/turn/input")
        bundle = compiled.prompt_bundle
        projection = public["projection"]
        projection.update(
            prompt=bundle.flatten(),
            sections=[s.to_dict() for s in bundle.sections],
            sources={s.id: s.source for s in bundle.sections},
            stable_hash=sha256(self.independent([[s.to_dict() for s in bundle.sections if s.stable]])[0]).hexdigest(),
            total_chars=len(bundle.flatten()),
        )
        agent.model = "shared-model"
        result = self.execute(
            "public_configured_sub_agent",
            [
                LLMResponse(
                    "", [ToolCall("delegate", "create_sub_task", {"agent_id": "researcher", "task_description": "Research."})]
                ),
                LLMResponse("child done"),
                lambda request: LLMResponse("{}" if request.metadata.get("purpose") == "session_memory" else "parent done"),
                LLMResponse("parent done"),
            ],
            agent=agent,
        )
        public_sid = result.raw_result.session_id
        _, public_rows, _ = self.f.kernel.store.read_state(public_sid)
        parked = next(
            r.record for r in public_rows if r.record.kind == "op_parked" and r.record.payload["handle"]["kind"] == "child"
        )
        actual_child = parked.payload["handle"]["session_id"]
        public_rt = self.f.runtimes[public_sid]
        self.f.runtimes[actual_child] = public_rt.child_runtime(self.f.kernel.store, actual_child)
        child_result = self.f.result(actual_child)
        original = public["public_runner"]
        observed = {
            "constructs_agent_task": False,
            "parent_model": result.resolved_model.model_id,
            "child_model": child_result.resolved_model.model_id,
            "delegated_agent_id": "researcher",
            "child_final_output": child_result.final_output,
            "parent_final_output": result.final_output,
            "terminal_status": result.status.value,
        }
        assert observed == {key: original[key] for key in observed}, observed
        public["public_runner"] = observed | {"producer": self.evidence_for("child", public_sid)}
        manager = self.outputs["manager_tool_envelope.json"]
        for collection, tool_name in (
            ("create_error_cases", "create_sub_task"),
            ("status_success_cases", "sub_task_status"),
            ("status_error_cases", "sub_task_status"),
        ):
            for case in manager[collection]:
                result = self.execute(
                    "manager_" + case["name"],
                    [LLMResponse("", [ToolCall("manager", tool_name, case["arguments"])]), LLMResponse("done")],
                    agent=Agent(
                        "coordinator",
                        "Coordinate.",
                        sub_agents={"researcher": SubAgentConfig(model="m", description="Research.")},
                    ),
                )
                receipt = result.raw_result.cycles[0].tool_results[0]
                actual = json.loads(receipt.content)
                assert actual == case["expected"], (case["name"], actual)
                case["expected"] = actual
                case["producer"] = self.evidence_for("manager", result.raw_result.session_id)
        manager["sync_wait_outcome"] = {
            "parent_adopts_intermediate_wait": False,
            "same_turn_reply": True,
            "producer_session_id": "thread_2",
        }
        manager["worker_visibility"] = {
            "running_status_authoritative": True,
            "terminal_fields_hidden_until_terminal_delivery": True,
        }
        manager["admission"] = admitted[0].payload["handle"]
        manager["terminal_results"] = [
            r.record.payload["input"]["payload"]
            for r in rows
            if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "child_result"
        ]
        handoff = self.outputs["handoff_contract.json"]
        handoff_rows = self.f.kernel.store.read_state("handoff")[1]
        handoff["lifecycle_order"] = [
            e.type for e in project_records(handoff_rows) if e.type in {"handoff_started", "handoff_completed", "run_completed"}
        ]
        handoff["admitted_max_handoffs"] = (
            self.f.state("handoff")
            .turns["handoff/turn/initial"]
            .start.payload["definition"]["task"]["metadata"]["vv_session"]["max_handoffs"]
        )
        handoff["producer"] = self.evidence_for("handoff", "handoff")

    def definitions_results(self):
        from vv_agent.tools.registry import ToolRegistry

        baseline = load("run_definition.json")

        def compile_definition(source):
            from vv_agent.memory.manager import MemoryManager
            from vv_agent.microcompaction import MicrocompactionPolicy
            from vv_agent.types import AgentTask

            task = AgentTask.from_dict(deepcopy(source["task"]))
            tools = []
            for entry in source["tools"]:
                schema = entry["function"]
                tools.append(
                    FunctionTool(
                        schema["name"],
                        schema["description"],
                        schema["parameters"],
                        lambda _ctx, _args: ToolOutputText("ok"),
                        tool_metadata=ToolMetadata.from_dict(source["capabilities"][schema["name"]])
                        if schema["name"] in source["capabilities"]
                        else None,
                    )
                )
            binding = source["model_binding"]
            agent = Agent(source["agent_name"], task.prompt_bundle, model=binding["model"], tools=tools)
            memory = deepcopy(source["memory_settings"])
            memory["microcompaction_policy"] = MicrocompactionPolicy.from_dict(memory["microcompaction_policy"])
            rt = self.f.kernel.runtime(
                agent,
                RunConfig(
                    model_provider=ScriptedModelProvider.new(binding["backend"], binding["model"], []),
                    workspace="/fixture",
                    tool_registry_factory=ToolRegistry,
                ),
            )
            rt.memory_manager = MemoryManager(**memory)
            definition = rt.definition(task)
            raw = self.independent([definition])[0]
            assert rt.definition_digest(task) == sha256(raw).hexdigest()
            return definition, raw

        rows = []
        for case in baseline["golden_cases"][:3]:
            definition, raw = compile_definition(case["definition"])
            rows.append(
                {"name": case["name"], "definition": definition, "definition_digest": sha256(raw).hexdigest(), **self.facts(raw)}
            )
        for sid in ("tools", "multimodal", "bindings", "handoff", "children"):
            state, _, _ = self.f.kernel.store.read_state(sid)
            turn = next(iter(state.turns.values())).start
            definition = turn.payload["definition"]
            raw = self.independent([definition])[0]
            assert sha256(raw).hexdigest() == turn.payload["definition_digest"]
            rows.append(
                {
                    "name": "kernel_" + sid,
                    "definition": definition,
                    "definition_digest": sha256(raw).hexdigest(),
                    **self.facts(raw),
                }
            )
        producer_cases = []
        _, original_bytes = compile_definition(baseline["golden_cases"][1]["definition"])
        for case in baseline["producer_cases"]:
            definition, raw = compile_definition(case["definition"])
            relation = "equal" if original_bytes == raw else "different"
            producer_cases.append(
                {
                    "name": case["name"],
                    "base_golden_case": case["base_golden_case"],
                    "expected_digest_relation": relation,
                    "definition": definition,
                    "definition_digest": sha256(raw).hexdigest(),
                    **self.facts(raw),
                }
            )
        self.outputs["run_definition.json"] = {
            "contract": "run_definition",
            "version": 6,
            "canonicalization": load("run_definition.json")["canonicalization"],
            "required_fields": list(rows[0]["definition"]),
            "golden_cases": rows,
            "producer_cases": producer_cases,
            "top_level_field_policy": {"closed": False, "validation_owner": "opaque J in turn_started.definition"},
            "reserved_task_metadata": "vv_session",
        }
        d = self.outputs["result_public.json"]
        result = self.f.result("seed")
        raw = result.raw_result.to_dict()
        d["version"] = 7
        d["agent_result"] = raw
        d["agent_result_wire"]["required_fields"] = sorted(set(raw) - {"budget_usage", "budget_exhaustion", "error_code"})
        d["agent_result_wire"]["statuses"] = ["pending", "running", "suspended", "wait_user", "completed", "failed", "max_cycles"]
        for key in (
            "resume_observations",
            "checkpoint_only_statuses",
            "checkpoint_only_status_semantics",
            "deferred_status_semantics",
        ):
            d["agent_result_wire"].pop(key, None)
        for key in ("reconciliation_result", "deferred_pending_result", "resume_observation_vectors", "expected_approvals"):
            d.pop(key, None)
        projected = result.to_dict()
        d["projection_keys"] = list(projected)
        d["resolved_model_projection"] = projected["resolved_model"]
        d["run_result"] = projected
        d["producer_cases"] = []
        for sid in ("seed", "repair", "budget", "accepted", "unknown", "user_wait"):
            result = self.f.result(sid)
            d["producer_cases"].append(
                {"name": sid, "agent_result": result.raw_result.to_dict(), "public_result": result_facts(result)}
            )
        controls = self.outputs["run_config_controls.json"]
        defaults = effective_run_config(Agent("defaults", "Use framework defaults."), RunConfig())
        for key in list(controls["framework_defaults"]):
            if key in RETIRED_FIELDS:
                controls["framework_defaults"].pop(key)
                continue
            value = getattr(defaults, key)
            controls["framework_defaults"][key] = value.to_dict() if hasattr(value, "to_dict") else value
        controls["per_run_controls"] = [
            c
            for c in controls["per_run_controls"]
            if c["capability"] not in {"durable_checkpoint_resume", "execution_backend", "sub_task_manager"}
        ]
        for c in controls["per_run_controls"]:
            if c["capability"] == "session_history":
                c["python"] = "AgentSession read-only record projection"
            if c["capability"] == "initial_state":
                c["python"] = "creation-time seed.messages and seed.shared_state"
            if c["capability"] == "event_store":
                c["python"] = "SessionRunEventStore projection and JsonlRunEventStore sink"
        for key in ("checkpoint_key_utf8_bytes", "checkpoint_extension_entry_utf8_bytes"):
            controls["integer_bounds"].pop(key, None)
        controls["seed"] = {"messages": [], "shared_state": {}}

    def events_traces(self):
        all_events, identity_inputs, sources = [], [], []
        memory_ids = {}
        for sid, rows in self.f.collect():
            for index, stored in enumerate(rows):
                source = stored.record
                if source.kind != "boundary_recorded" or source.payload["stage"] != "memory_started":
                    continue
                from vv_agent.session.records import InboxItem
                from vv_agent.session.reducer import fold

                prefix = rows[:index]
                logical = [r.record for r in prefix]
                state = fold(
                    logical, consumed_inputs=[InboxItem(**r.payload["input"]) for r in logical if r.kind == "input_applied"]
                )
                event = source.payload["data"]["event"]
                messages = [m.to_dict() for m in project_context(prefix, state)]
                key = sha256(self.independent([[source.turn_id, messages, event["trigger"]]])[0]).hexdigest()
                assert key == source.payload["boundary_id"]
                memory_ids[(sid, key)] = key
            for e in project_records(rows):
                wire = e.to_dict()
                assert event_from_dict(wire).to_dict() == wire
                seq = e.metadata["session_seq"]
                source = next(r.record for r in rows if r.seq == seq)
                identity_inputs.append([sid, source.record_id])
                sources.append(source)
                all_events.append(wire)
        for wire, source, body in zip(all_events, sources, self.independent(identity_inputs), strict=True):
            identity = sha256(body).hexdigest()
            if source.kind == "boundary_recorded" and source.payload["stage"] in {"memory_started", "memory_completed"}:
                key = memory_ids[(source.session_id, source.payload["boundary_id"])]
                suffix = "started" if source.payload["stage"] == "memory_started" else "completed"
                assert wire["event_id"] == f"sk/memory/{key}/{suffix}"
            else:
                assert wire["event_id"].startswith("sk/" + identity + "/")
        self.outputs["run_events.jsonl"] = all_events
        self.outputs["budget_events.jsonl"] = [
            e.to_dict() for e in self.f.result("budget").events if e.type in {"budget_snapshot", "budget_exhausted", "run_failed"}
        ]
        child_events = [
            e.to_dict() for e in self.f.result("children").events if e.type in {"sub_run_started", "sub_run_completed"}
        ]
        assert all(e["run_id"] == "children/turn/initial" for e in child_events)
        self.outputs["configured_sub_agent_events.jsonl"] = child_events
        bridge = SessionRunEventStore(self.f.kernel.store, "children")
        replay = [e.to_dict() for e in bridge.replay(run_id="children/turn/initial")]
        self.outputs["event_store_replay.jsonl"] = replay
        self.outputs["runner_events.jsonl"] = [e.to_dict() for e in self.f.result("tools").events]
        rows = self.f.kernel.store.read_state("tools")[1]
        self.outputs["runner_trace.jsonl"] = [
            {"seq": seq, "method": method, "span": span.to_dict()} for seq, method, span in project_spans(rows)
        ]
        for row in self.outputs["runner_trace.jsonl"]:
            span = row["span"]
            source = next(r.record for r in rows if r.seq == row["seq"])
            identity_input = (
                ["tools", span["trace_id"], span["name"]]
                if span["name"] != "tool"
                else ["tools", source.operation_id, source.attempt]
            )
            assert span["span_id"] == "sk/span/" + sha256(self.independent([identity_input])[0]).hexdigest()
        delivered = []

        class Processor:
            def on_span_start(self, span):
                delivered.append(("start", span.name))

            def on_span_end(self, span):
                delivered.append(("end", span.name))

        processor = Processor()
        while deliver_spans(self.f.kernel.store, "tools", [processor]):
            pass
        assert deliver_spans(self.f.kernel.store, "tools", [processor]) == 0
        self.outputs["runner_trace_spans.json"] = {
            "version": "v2",
            "topology": {
                "start_order": [k for m, k in delivered if m == "start"],
                "end_order": [k for m, k in delivered if m == "end"],
                "parents": {
                    span.name: next(
                        (parent.name for _, _, parent in project_spans(rows) if parent.span_id == span.parent_id), None
                    )
                    for _, method, span in project_spans(rows)
                    if method == "on_span_start"
                },
            },
            "delivery": {"ack_first": True, "at_most_once": True, "replay_deliveries": 0, "telemetry_loss_possible": True},
        }
        failure = self.f.result("blocked")
        failed_spans = project_spans(self.f.kernel.store.read_state("blocked")[1])
        self.outputs["runner_trace_spans.json"]["failure_cleanup"] = {
            "started": [s.name for _, method, s in failed_spans if method == "on_span_start"],
            "ended": [s.name for _, method, s in failed_spans if method == "on_span_end"],
            "run_status": failure.status.value,
        }
        assert self.outputs["runner_trace_spans.json"]["failure_cleanup"] == load("runner_trace_spans.json")["failure_cleanup"]

        class BrokenProcessor:
            def on_span_start(self, _span):
                raise RuntimeError("fixture sink unavailable")

            on_span_end = on_span_start

            def flush(self):
                raise RuntimeError("fixture flush unavailable")

        delivered.clear()
        while deliver_spans(self.f.kernel.store, "repair", [BrokenProcessor(), processor]):
            pass
        repair_spans = project_spans(self.f.kernel.store.read_state("repair")[1])
        assert delivered == [("start" if m == "on_span_start" else "end", s.name) for _, m, s in repair_spans]
        self.outputs["runner_trace_spans.json"]["sink_failure"] = {
            "isolated": bool(delivered),
            "run_status": self.f.result("repair").status.value,
        }
        terminal = self.outputs["runner_terminal.json"]
        for key in ("reconciliation_required", "checkpoint_terminal_order", "event_store_fail_closed"):
            terminal.pop(key, None)
        terminal["version"] = "v2"
        terminal["producer_cases"] = [
            {
                "name": sid,
                **result_facts(self.f.result(sid)),
                "terminal_events": [
                    e.to_dict() for e in self.f.result(sid).events if e.type in {"run_completed", "run_failed", "run_cancelled"}
                ],
            }
            for sid in ("tools", "blocked", "budget", "repair")
        ]
        for key in ("success_with_session", "output_guardrail_block"):
            terminal[key]["tail"] = [t for t in terminal[key]["tail"] if t not in RETIRED_KINDS]
        self.outputs["run_handle.json"] = {
            "version": "v2",
            "authority": "session_log_and_inbox",
            "read_only_state": self.f.state("tools").phase,
            "result": result_facts(self.f.result("tools")),
            "event_replay": {
                "count": len(replay),
                "first_event_id": replay[0]["event_id"],
                "last_event_id": replay[-1]["event_id"],
                "ordered": all(a["metadata"]["session_seq"] <= b["metadata"]["session_seq"] for a, b in pairwise(replay)),
            },
            "controls": [self.evidence_for("controls", "controls")],
            "independent_replays": list(bridge.replay(run_id="children/turn/initial"))
            == list(bridge.replay(run_id="children/turn/initial")),
            **self.handle_observations(),
        }

    def terminal(self):
        from vv_agent.guardrails import GuardrailResult, output_guardrail

        calls = []

        @output_guardrail
        def block(_context, _output):
            calls.append("block")
            return GuardrailResult.block("blocked final output")

        @output_guardrail
        def later(_context, _output):
            calls.append("later")
            return GuardrailResult.allow()

        blocked = self.execute(
            "terminal_guardrail",
            [LLMResponse("blocked final output candidate")],
            agent=Agent("terminal", "Answer.", output_guardrails=[block, later]),
        )
        terminal = self.outputs["runner_terminal.json"]
        cases = {
            "success_with_session": self.handles["c1c/run_continue_overrides_runner_finish"].result(),
            "output_guardrail_block": blocked,
            "max_cycles": self.handles["c1c/max_cycles_preserves_last_assistant_output"].result(),
            "budget_exhausted": self.f.result("budget"),
        }
        token = CancellationToken()
        token.cancel("host requested cancellation")
        cases["cancellation"] = self.execute("terminal_cancel", [], config=RunConfig(cancellation_token=token))
        for key, result in cases.items():
            observed = result_facts(result)
            events = [e.to_dict() for e in result.events]
            terminals = [e for e in events if e["type"] in {"run_completed", "run_failed", "run_cancelled"}]
            assert len(terminals) == 1
            observed |= {
                "terminal": terminals[0]["type"],
                "terminal_count": len(terminals),
                "tail": [e["type"] for e in events[-2:]],
                "events_tail": [e["type"] for e in events[-2:]],
                "later_guardrails_run": "later" in calls,
                "budget_fixture": "run_budget.json",
            }
            terminal[key] = {name: observed[name] for name in terminal[key]}

    def handle_observations(self):
        from threading import Event

        from vv_agent.session.surfaces import SessionDriver

        kernel = SessionDriver()
        try:
            provider = ScriptedModelProvider.new("test", "m", [LLMResponse("draft") for _ in range(280)])
            handle = start_runner(
                kernel,
                "handle-burst",
                Agent("burst", "Continue."),
                "go",
                run_config=self.f.config(RunConfig(model_provider=provider, max_cycles=280, no_tool_policy="continue")),
            )
            first, second = handle.events(), handle.events()
            prefix_a, prefix_b = next(first), next(second)
            result = handle.result()
            a, b = [prefix_a, *first], [prefix_b, *second]
            count = len(a)
            assert count >= 1100 and a == b == list(handle.events()) == result.events
            subscribers = {
                "independent": a == b,
                "start_from_complete_backlog": a == list(handle.events()),
                "lossless_after_live_capacity": count >= 1100,
                "burst_event_count": count,
            }
            entered, released = Event(), Event()

            def blocking(_request):
                entered.set()
                assert released.wait(5)
                return LLMResponse("draft")

            cancelled = start_runner(
                kernel,
                "handle-cancel",
                Agent("cancel", "Answer."),
                "go",
                run_config=self.f.config(RunConfig(model_provider=ScriptedModelProvider.from_steps("test", "m", [blocking]))),
            )
            assert entered.wait(5)
            try:
                accepted = cancelled.cancel("host requested cancellation")
                repeated = cancelled.cancel("host requested cancellation")
            finally:
                released.set()
            outcome = cancelled.result()
            state, records, _ = kernel.store.read_state("handle-cancel")
            end = next(r.record for r in records if r.record.kind == "turn_ended")
            cancellation = {
                "accepted": accepted,
                "terminal_status": end.payload["status"],
                "repeated_request_accepted": repeated,
                "late_request_accepted": cancelled.cancel(),
                "reason": end.payload["reason"],
                "read_only_phase": state.phase,
            }
            assert accepted and not repeated and not cancellation["late_request_accepted"]
            assert outcome.completion_reason.value == "cancelled"
            return {
                "subscribers": subscribers,
                "cancellation": cancellation,
                "completion": {"parent_result_is_retained": True, "child_terminal_delivery_is_separate": True},
            }
        finally:
            kernel.close()

    def stream(self):
        d = self.outputs["llm_stream_projection.json"]
        baseline = d["synthetic_top_level"]
        emitted = []

        class Streaming:
            model_id = "stream-model"

            def __init__(self):
                self.calls = 0

            def complete(self, request):
                return self.complete_with_stream(request)

            def complete_with_stream(self, request, stream_callback=None):
                del request
                self.calls += 1
                if self.calls < 3:
                    return LLMResponse(f"draft {self.calls}")
                for raw in baseline["provider_payloads"]:
                    stream_callback(deepcopy(raw))
                return LLMResponse("done", [ToolCall("call_stream", "echo", {"message": "done"})])

        @function_tool
        def echo(message: str) -> str:
            return message

        sid = "c1c/stream"
        rt = self.f.admit(
            sid,
            [],
            agent=Agent(
                "stream-agent", "Return the third-cycle tool result.", tools=[echo], tool_use_behavior="stop_on_first_tool"
            ),
            config=RunConfig(max_cycles=3, no_tool_policy="continue", stream=emitted.append),
        )
        rt.llm = Streaming()
        from vv_agent import events as event_module

        with (
            patch.object(
                event_module.uuid,
                "uuid4",
                side_effect=(SimpleNamespace(hex=sha256(self.independent([[sid, i]])[0]).hexdigest()) for i in range(2000)),
            ),
            patch.object(event_module.time, "time", return_value=1000.0),
        ):
            self.f.run(sid)
        assert len(emitted) == 4
        wires = [e.to_dict() for e in emitted]
        assert [w.get("delta") for w in wires[:2]] == ["done", "plan"]
        baseline["expected_wire_events"] = wires
        baseline["context"] = {k: wires[0][k] for k in ("run_id", "trace_id", "session_id", "agent_name", "cycle_index")}
        d["wire_version"] = "v6"
        d["public_event_surface"]["durable_replay"] = "SessionRunEventStore"
        d["public_event_surface"]["typed_event_recorded_before_observer"] = False
        d["public_event_surface"]["provider_deltas_are_ephemeral"] = True

    def invalid_events(self):
        seed = self.outputs["runner_events.jsonl"][0]
        cases = []
        for identity, version in (
            ("missing_version", None),
            ("stale_version", "v5"),
            ("unknown_version", "v99"),
            ("malformed_version", 6),
        ):
            value = seed | {"version": version}
            if version is None:
                value.pop("version", None)
            cases.append((identity, value))
        for field in sorted(RETIRED_FIELDS & {"checkpoint_key", "resume_attempt", "consumed_revision"}):
            cases.append(("rejected_field_" + str(len(cases)), seed | {field: "old"}))
        for kind in sorted(RETIRED_KINDS):
            cases.append(("rejected_kind_" + str(len(cases)), seed | {"type": kind}))
        cases.append(("session_id_required", {k: v for k, v in seed.items() if k != "session_id"}))
        for c in load("run_events_invalid.json")["reject"]:
            value = deepcopy(c["input"]) if "input" in c else json.loads(base64.b64decode(c["bytes_base64"]))
            if (c["id"], value) in cases:
                continue
            if value.get("type") in RETIRED_KINDS or any(
                k in value for k in ("checkpoint_key", "resume_attempt", "consumed_revision")
            ):
                continue
            if "version" in value and value["version"] == "v5":
                value["version"] = "v6"
            if "session_id" not in value:
                value["session_id"] = "invalid-session"
            try:
                event_from_dict(value)
            except (TypeError, ValueError, KeyError):
                cases.append((c["id"], value))
        vectors = []
        for identity, value in cases:
            reject(lambda x: event_from_dict(x), value)
            raw = self.independent([value])[0]
            vectors.append({"id": identity, **self.facts(raw)})
        self.outputs["run_events_invalid.json"] = {"contract": "run_events_invalid", "wire_version": "v6", "reject": vectors}

    def memory(self):
        # Reuse the existing real MemoryManager producer harness to retain the v23 content bytes.
        import runpy
        import sys

        import pytest

        root = BASE.parents[2]
        with patch.object(sys, "path", [str(root / "tests"), *sys.path]):
            suite = runpy.run_path(str(root / "tests/test_memory_local_contract.py"))
            for name in (
                "test_memory_local_token_counts_match_fixture",
                "test_memory_local_microcompact_boundaries_match_fixture",
                "test_memory_local_session_prompt_truncation_matches_fixture",
                "test_memory_local_summary_and_excerpt_match_fixture",
                "test_session_memory_public_extract_matches_fixture",
                "test_standalone_summary_helpers_match_current_contract",
            ):
                suite[name]()
            for c in suite["_SUMMARY_CASES"]:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    suite["test_history_preserving_summary_contract"](c, monkeypatch)
            for c in suite["_SUMMARY"]["accepted_normalization_cases"]["variants"]:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    suite["test_summary_normalization_contract"](c, monkeypatch)
            for variant in suite["_FAILURE"]["variants"]:
                if "summary_input_fits" not in variant:
                    with pytest.MonkeyPatch.context() as monkeypatch:
                        suite["test_failed_or_empty_summary_preserves_prefix"](variant, monkeypatch)
            suite["test_session_memory_public_extract_handles_escaped_and_nested_json"]()
            for c in suite["_SUMMARY"]["invalid_block_cases"]:
                suite["test_invalid_atomic_blocks_preserve_history"](c)
            for language in ("zh-CN", "en-US"):
                with pytest.MonkeyPatch.context() as monkeypatch:
                    suite["test_localized_complete_prefix_prompt_bytes"](language, monkeypatch)
            for c in suite["_CONTRACT"]["microcompact"]["transcript_cases"]:
                suite["test_relative_transcript_microcompact_contract"](c)
            lifecycle_suite = runpy.run_path(str(root / "tests/test_memory_lifecycle_contract.py"))
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="c1c-memory-capacity-") as tmp:
                self.memory_capacity(lifecycle_suite, Path(tmp))
            for c in self.outputs["memory_lifecycle.json"]["emergency_cases"]:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    lifecycle_suite["test_emergency_requires_summary_before_removing_more_tail"](c, monkeypatch)
            for name in (
                "test_omitted_memory_compact_threshold_defaults_match_contract",
                "test_microcompaction_preserves_provider_usage_baseline_for_full_compaction",
                "test_public_memory_manager_reports_structural_as_stronger_than_microcompact",
                "test_public_memory_manager_keeps_warning_when_post_microcompact_usage_requires_it",
                "test_memory_provider_attempt_errors_are_fail_open",
                "test_session_memory_compaction_does_not_refresh_the_current_prompt",
            ):
                lifecycle_suite[name]()
            with pytest.MonkeyPatch.context() as monkeypatch:
                lifecycle_suite["test_prune_only_keeps_complete_history_contract"](monkeypatch)
            with patch.object(lifecycle_suite["SessionMemory"], "storage_path", return_value=None):
                self.memory_routes(lifecycle_suite)
        d = self.outputs["memory_local.json"]
        d["summary_compaction"]["control_failure_case"] = {
            k: ([c for c in v if "checkpoint" not in c["name"]] if k == "variants" else v)
            for k, v in d["summary_compaction"]["control_failure_case"].items()
        }
        d["session_kernel_producers"] = [
            self.evidence_for("memory", sid) for sid in ("summary", "rejected_summary", "micro", "memory")
        ]
        lifecycle = self.outputs["memory_lifecycle.json"]
        lifecycle["session_memory"].pop("checkpoint_receipt_replay", None)
        lifecycle["session_memory"]["receipt_replay"] = "reuse_recorded_extraction_without_a_second_provider_call"
        lifecycle["session_memory"]["control_outcomes_propagate"] = ["cancellation", "budget_exhaustion", "lost_ownership"]
        lifecycle["summary_pipeline"]["control_outcomes_propagate"] = ["cancellation", "budget_exhaustion", "lost_ownership"]
        lifecycle["logged_internal_calls"] = [
            {"name": sid, "usage": self.f.result(sid).token_usage.to_dict(), "producer": self.evidence_for("memory", sid)}
            for sid in ("summary", "rejected_summary", "memory")
        ]
        # Known v23 artifact preimage is retained; other evidence references are explicit input locators.
        assert (
            sha256(b"full archived tool output\n").hexdigest()
            == "73c3362aef1718f1c2d34f8691f66e6b5b4b73ba4fce4cdcb2bf27c3f00fea76"
        )

    def memory_capacity(self, suite, tmp):
        import pytest

        from vv_agent.session.compaction import manager_for
        from vv_agent.session.records import InboxItem
        from vv_agent.session.reducer import fold

        capacity = self.outputs["memory_lifecycle.json"]["capacity_contract"]
        builder = suite["build_memory_manager"]
        for cases, test, context_only in (
            (capacity["cases"], "test_runtime_resolves_memory_capacity_from_contract_cases", False),
            (capacity["context_window_resolution"]["cases"], "test_runtime_context_window_resolution_matches_contract", True),
        ):
            for case in cases:
                produced = []

                def capture(produced=produced, **kwargs):
                    manager = builder(**kwargs)
                    produced.append(manager)
                    return manager

                with (
                    pytest.MonkeyPatch.context() as monkeypatch,
                    patch.dict(suite[test].__globals__, {"build_memory_manager": capture}),
                ):
                    suite[test](case, tmp, monkeypatch)
                assert len(produced) == 1
                sid = "c1c/capacity/" + case["name"]
                rt = self.f.admit(sid, [LLMResponse("done")], memory=produced[0])
                self.f.run(sid)
                records = self.f.kernel.store.read_state(sid)[1]
                prefix = [
                    stored.record for stored in records[: next(i for i, r in enumerate(records) if r.record.kind == "turn_ended")]
                ]
                state = fold(
                    prefix, consumed_inputs=[InboxItem(**r._payload["input"]) for r in prefix if r.kind == "input_applied"]
                )
                admitted = manager_for(SimpleNamespace(state=state, sid=sid, runtime=rt))
                if context_only:
                    assert admitted.model_context_window == case["expected_model_context_window"]
                    case["expected_model_context_window"] = admitted.model_context_window
                else:
                    actual = {
                        "reserved_output_tokens": admitted.reserved_output_tokens,
                        "reserved_output_source": admitted.reserved_output_source,
                        "effective_threshold": admitted.autocompact_threshold,
                        "microcompact_threshold": admitted.microcompact_trigger_threshold,
                        "microcompact_target": admitted.microcompact_target_threshold,
                    }
                    assert actual == case["expected"]
                    case["expected"] = actual

    def memory_routes(self, suite):
        from vv_agent.config import ResolvedModelConfig
        from vv_agent.llm import ScriptedLLM
        from vv_agent.memory import MemoryManager
        from vv_agent.session.records import InboxItem

        lifecycle = self.outputs["memory_lifecycle.json"]
        for key, purpose, prefix in (
            ("summary_route", "compaction", "memory_summary"),
            ("session_extraction_route", "session_memory", "session_memory_extraction"),
        ):
            expected = lifecycle[key]
            requests = []

            def respond(request, purpose=purpose, requests=requests):
                requests.append(request)
                return LLMResponse(
                    suite["_summary_payload"]()
                    if purpose == "compaction"
                    else '[{"category":"decision","content":"route separately","importance":8}]'
                )

            primary = ResolvedModelConfig("main-backend", "main-model", "main-model", "main-model", [])
            internal = ResolvedModelConfig(expected["backend"], expected["model"], expected["model"], expected["model"], [])
            provider = suite["ModelMapProvider"](
                routes={
                    "main-model": (ScriptedLLM([LLMResponse("done")]), primary),
                    expected["model"]: (ScriptedLLM([respond]), internal),
                },
                default_model="main-model",
            )
            sid = "c1c/" + key
            config = RunConfig(
                model_provider=provider,
                initial_messages=[Message("user", "request"), Message("assistant", "old " * 1200)]
                if purpose == "compaction"
                else None,
                session_memory_enabled=purpose == "session_memory",
                metadata={
                    prefix + "_backend": expected["backend"],
                    prefix + "_model": expected["model"],
                    "session_memory_min_tokens": 1,
                    "session_memory_min_text_messages": 1,
                    "session_memory_storage_dir": "",
                },
            )
            rt = self.f.kernel.runtime(Agent("memory-route", "Be precise.", model="main-model"), self.f.config(config))
            if purpose == "compaction":
                rt.memory_manager = MemoryManager(compact_threshold=1000, keep_recent_messages=1)
            self.f.runtimes[sid] = rt
            self.f.kernel.create(sid, "/fixture")
            self.f.push(sid, InboxItem("initial", "user", {"content": "go"}))
            self.f.run(sid)
            result = self.f.result(sid)
            assert result.status == "completed"
            calls = [
                c
                for c in result.token_usage.model_calls
                if c.operation.value == ("memory_compaction" if purpose == "compaction" else "session_memory")
            ]
            assert len(calls) == len(requests) == 1
            actual = {
                "backend": calls[0].backend,
                "model": calls[0].model,
                "request_model": requests[0].model,
                "resolution_count": provider.resolved_models.count(expected["model"]),
            }
            assert actual == expected
            lifecycle[key] = actual
            # Retained projections must neither resolve nor call the internal provider again.
            self.f.run(sid)
            assert len(requests) == 1
            assert provider.resolved_models.count(expected["model"]) == 1

    def metadata_approval(self):
        from vv_agent.tools.base import ToolContext
        from vv_agent.tools.orchestrator import ToolOrchestrator

        d = self.outputs["tool_metadata.json"]
        import runpy
        import sys
        from tempfile import TemporaryDirectory

        import pytest

        root = BASE.parents[2]
        with patch.object(sys, "path", [str(root / "tests"), *sys.path]):
            suite = runpy.run_path(str(root / "tests/test_tool_metadata_contract.py"))
            for collection, test in (
                ("normalization_cases", "test_public_tool_metadata_normalization_cases"),
                ("invalid_cases", "test_public_function_tool_rejects_invalid_metadata_cases"),
                ("generated_invalid_cases", "test_public_producers_reject_generated_invalid_cases"),
            ):
                for case in d[collection]:
                    suite[test](case)
            with TemporaryDirectory(prefix="c1c-metadata-") as tmp:
                for case in d["producer_cases"]:
                    if "deferred" not in case["name"]:
                        with pytest.MonkeyPatch.context() as monkeypatch:
                            suite["test_real_orchestrator_consumes_canonical_producer_cases"](case, Path(tmp), monkeypatch)
                suite["test_generic_metadata_is_not_promoted_and_tool_metadata_is_not_model_visible"]()
                suite["test_parse_failure_boundary_is_driven_by_canonical_telemetry_contract"](Path(tmp))
        for c in d["normalization_cases"]:
            actual = ToolMetadata.from_dict(c["input"]).to_dict()
            assert actual == c["expected"]
            c["expected"] = actual
        for c in d["invalid_cases"]:
            reject(ToolMetadata.from_dict, c["input"])
        for c in d["policy_cases"]:
            tool = FunctionTool(
                "inspect",
                "Inspect.",
                {"type": "object", "properties": {}},
                lambda _ctx, _args: ToolOutputText("ok"),
                tool_metadata=ToolMetadata.from_dict(c["metadata"]) if c["metadata"] else None,
            )
            result = self.execute(
                "metadata_" + c["name"],
                [LLMResponse("", [ToolCall("inspect", "inspect", {})]), LLMResponse("done")],
                agent=Agent("metadata", "Inspect.", tools=[tool]),
                config=RunConfig(
                    tool_policy=ToolPolicy(
                        **(
                            c["policy"]
                            | ({"disallowed_tools": ["inspect"]} if c.get("existing_name_policy_allows") is False else {})
                        )
                    )
                ),
            )
            receipt = result.raw_result.cycles[0].tool_results[0]
            assert (receipt.error_code is None) == c["allowed"], (c["name"], receipt.to_dict())
            c["result"] = receipt.to_dict()
            c["producer"] = self.evidence_for("policy", result.raw_result.session_id)
        d.pop("checkpoint", None)
        d["definition_binding"] = {
            "tool_metadata": "capabilities",
            "policy": "definition.task.metadata",
            "retained_on_resume": True,
        }
        d["producer_cases"] = [c for c in d["producer_cases"] if "deferred" not in c["name"]]
        d["provider_cases"] = [self.evidence_for("provider", sid) for sid in ("accepted", "unknown")]
        d["public_construction"]["propagation"] = [p for p in d["public_construction"]["propagation"] if p != "checkpoint"]
        d["telemetry_contract"]["event_order"] = [
            p for p in d["telemetry_contract"]["event_order"] if p != "tool_call_deferred_when_admission_is_durable"
        ]
        for key in (
            "deferred_representation",
            "completed_never_accepts_deferred",
            "cross_process_deferred_duration_ms",
            "cross_process_deferred_execution_started",
        ):
            d["telemetry_contract"].pop(key, None)
        d["telemetry_contract"]["event_types"] = ["tool_call_planned", "tool_call_started", "tool_call_completed"]
        approval = self.outputs["approval_tool_policy.json"]
        approval["request_id"] = {
            "source": "retained_approval_handle.request_id",
            "same_value_at": ["op_parked.handle.request_id", "approval_requested.request_id", "approval_answer.request_id"],
        }
        approval["approval"]["absolute_deadline"] = True
        approval["approval"]["owner_retained"] = True
        approval["approval"]["producer_cases"] = [self.evidence_for("approval", sid) for sid in ("approval", "deadline")]
        from vv_agent.approval import ApprovalBroker, ApprovalDecision

        for case in approval["approval"]["decisions"]:
            effects = []

            gated = FunctionTool(
                "dangerous",
                "Use dangerous.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                lambda _ctx, args, effects=effects: effects.append(args["path"]) or ToolOutputText("ran"),
                needs_approval=True,
            )

            decision = ApprovalDecision(case["action"], case["reason"], case["metadata"])
            provider = SimpleNamespace(should_request=lambda _request: True, decide=lambda _request, decision=decision: decision)
            result = self.execute(
                "approval_" + case["action"],
                [LLMResponse("calling", [ToolCall.from_dict(approval["tool_call"])]), LLMResponse("done")],
                agent=Agent("approval", "Use dangerous.", tools=[gated]),
                config=RunConfig(approval_provider=provider),
            )
            receipt = result.raw_result.cycles[0].tool_results[0]
            assert not effects and receipt.error_code == case["error_code"]
            body = json.loads(receipt.content)
            assert body["error"] == case["message"]
            case["message"], case["error_code"], case["result"] = body["error"], receipt.error_code, receipt.to_dict()
            requested = next(e.to_dict() for e in result.events if e.type == "approval_requested")
            resolved = next(e.to_dict() for e in result.events if e.type == "approval_resolved")
            approval["approval"]["requested_event_metadata_keys"] = sorted(requested["metadata"])
            approval["approval"]["resolved_event_metadata_keys"] = sorted(resolved["metadata"])
            approval["approval"]["result_shape"]["content_keys"] = sorted(body)
            approval["approval"]["result_shape"]["metadata_keys"] = sorted(receipt.metadata)
            case["producer"] = self.evidence_for("approval", result.raw_result.session_id)

        failure = approval["approval"]["provider_failure"]
        effects = []

        @function_tool(name="dangerous", needs_approval=True)
        def gated_failure(path: str) -> str:
            effects.append(path)
            return "ran"

        def fail_decision(_request):
            raise RuntimeError(failure["message"])

        broker = ApprovalBroker()
        result = self.execute(
            "approval_provider_failure",
            [LLMResponse("calling", [ToolCall.from_dict(approval["tool_call"])])],
            agent=Agent("approval", "Use dangerous.", tools=[gated_failure]),
            config=RunConfig(
                approval_provider=SimpleNamespace(should_request=lambda _request: True, decide=fail_decision),
                approval_broker=broker,
            ),
        )
        request = next(e for e in result.events if e.type == "approval_requested")
        failure.update(
            status=result.status.value,
            events=[e.type for e in result.events if e.type in {"approval_requested", "approval_resolved", "run_failed"}],
            tool_executes=bool(effects),
            broker_retains_request=broker.pending_request(request.request_id) is not None,
        )
        assert result.raw_result.error["message"] == failure["message"] and not effects
        failure["producer"] = self.evidence_for("approval", result.raw_result.session_id)
        for c in approval["policy"]["cases"]:

            @function_tool
            def dangerous(path: str) -> str:
                return path

            rt = self.f.kernel.runtime(
                Agent("policy", "Deny.", tools=[dangerous]), RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []))
            )
            ctx = ToolContext(
                workspace=Path("/fixture"),
                shared_state={},
                cycle_index=1,
                workspace_backend=None,
                ctx=SimpleNamespace(
                    metadata={
                        "_vv_agent_allowed_tools": c["allowed_tools"],
                        "_vv_agent_disallowed_tools": c["disallowed_tools"],
                        "_vv_agent_tool_policy_can_use_tool": lambda *_, answer=c["can_use_tool"]: answer,
                    }
                ),
            )
            receipt = ToolOrchestrator.from_registry(rt.registry).run_one(
                ToolCall.from_dict(approval["tool_call"]), context=ctx, allowed_tool_names=c["planned_tools"]
            )
            assert isinstance(receipt, ToolExecutionResult)
            assert receipt.metadata["policy_source"] == c["policy_source"]
            c["result"] = receipt.to_dict()

    def verify_keep(self):
        from tempfile import TemporaryDirectory

        from vv_agent.memory.manager import filter_empty_assistant_messages
        from vv_agent.model import ModelRef

        registry = self.f.kernel.runtime(
            Agent("builtins", "Answer."), RunConfig(model_provider=self.f.runtimes["tools"].config.model_provider)
        ).registry
        golden = load("builtin_tools.json")
        builtin_names = [t["name"] for t in golden["tools"]]
        assert [name for name in registry.list_tool_names() if name in builtin_names] == builtin_names
        # Kernel registry adds the fixture's echo; canonical built-ins retain exactly their order/schema.
        for tool in golden["tools"]:
            executor = registry.get_executor(tool["name"])
            schema = executor.openai_schema(None)
            assert schema["function"]["description"] == tool["description"]
            assert schema["function"]["parameters"] == tool["parameters"]
            assert executor.exposure.value == tool["exposure"]
        self.keep["builtin_tools.json"] = (
            "PASS: kernel Runtime.registry reproduces all 15 descriptions, schemas, order and exposure"
        )
        for c in load("model_ref.json")["valid"]:
            assert ModelRef.from_dict(c).to_dict() == c
        for c in load("model_ref.json")["invalid"]:
            reject(ModelRef.from_dict, c)
        self.keep["model_ref.json"] = "PASS: exact current ModelRef codec used by kernel model resolution"
        import runpy
        import sys

        root = BASE.parents[2]
        with patch.object(sys, "path", [str(root / "tests"), *sys.path]):
            config = runpy.run_path(str(root / "tests/test_config.py"))
            config["test_model_settings_contract_fixture_is_enforced"]()
            self.keep["model_settings.json"] = (
                "PASS: real settings parser/normalizer and exact resolution; same producer used by Runtime"
            )
            behavior = runpy.run_path(str(root / "tests/test_builtin_tool_behavior_contract.py"))
            with TemporaryDirectory(prefix="c1c-keep-") as tmp:
                for name in (
                    "test_fixture_drives_schema_validation_before_handler_execution",
                    "test_fixture_drives_prompt_registry_dynamic_hint_and_projection",
                    "test_fixture_drives_builtin_handler_envelopes_and_metadata",
                ):
                    behavior[name](Path(tmp))
            self.keep["builtin_tool_behavior.json"] = (
                "PASS: shared registry, schema validator, tool executor and handlers; kernel uses the same producers"
            )
            cli_suite = runpy.run_path(str(root / "tests/test_cli_contract.py"))
            cli_suite["test_settings_file_precedence_uses_explicit_then_environment_then_default"]()
            cli_suite["test_multiword_prompt_model_settings_and_resolved_limits_project_to_task"]()
        reasoning = load("assistant_reasoning_history.json")
        for c in reasoning["cases"]:
            raw = c["message"]
            message = Message(
                raw["role"], raw["content"], reasoning_content=raw.get("reasoning_content"), tool_calls=raw.get("tool_calls")
            )
            retained = bool(filter_empty_assistant_messages([message]))
            assert retained == c["expected"]["retain_in_runtime_history"]
            sid = "c1c/reasoning/" + c["name"]
            self.f.admit(sid, [], attributes={"seed": {"messages": [message.to_dict()], "shared_state": {}}})
            state, rows, _ = self.f.kernel.store.read_state(sid)
            projected = project_context(rows, state)
            assert bool(projected) == c["expected"]["retain_in_resumable_history"] == retained
            if retained:
                assert projected[0].content == c["expected"]["visible_content"]
                assert projected[0].reasoning_content == c["expected"]["reasoning_content"]
                if "openai_compatible_projection" in c["expected"]:
                    assert projected[0].to_openai_message() == c["expected"]["openai_compatible_projection"]
        first = reasoning["runtime_case"]["first_response"]
        captured = []

        def second(request):
            captured.extend(request.messages)
            return LLMResponse("done")

        self.execute(
            "reasoning",
            [LLMResponse(first["content"], raw={"reasoning_content": first["reasoning_content"]}), second],
            config=RunConfig(no_tool_policy="continue", max_cycles=2),
        )
        assert any(m.content == "" and m.reasoning_content == "private reasoning chain" for m in captured)
        self.keep["assistant_reasoning_history.json"] = (
            "PASS: kernel retains reasoning-only turn in next model context, removes fully empty messages"
        )
        # Existing handler tests validate the shared management ingress; kernel launch uses the real provider path.
        bash = load("bash_process_management.json")
        from vv_agent.tools.base import ToolContext
        from vv_agent.tools.orchestrator import ToolOrchestrator

        with TemporaryDirectory(prefix="c1c-bash-") as tmp:
            ctx = ToolContext(workspace=Path(tmp), shared_state={}, cycle_index=1, workspace_backend=None)
            orchestrator = ToolOrchestrator.from_registry(registry)
            for c in bash["invalid_bash_arguments"]:
                arguments = c.get("arguments", c.get("input"))
                r = orchestrator.run_one(ToolCall("invalid", "bash", arguments), context=ctx)
                assert isinstance(r, ToolExecutionResult) and r.error_code == "invalid_tool_arguments"
            for name in ("check_background_command", "stop_background_command"):
                for case in bash["invalid_management_arguments"]:
                    r = orchestrator.run_one(ToolCall("invalid", name, case["arguments"]), context=ctx)
                    assert isinstance(r, ToolExecutionResult) and r.error_code == bash["invalid_arguments_error_code"]
            from vv_agent.session.surfaces import SessionDriver

            kernel = SessionDriver()
            from vv_agent.tools.registry import ToolRegistry

            def bash_registry():
                native = ToolRegistry()
                for name in ("bash", "check_background_command", "stop_background_command"):
                    native.register_executor(registry.get_executor(name), planner_extra=True)
                return native

            try:
                provider = ScriptedModelProvider.new(
                    "test",
                    "m",
                    [LLMResponse("", [ToolCall("bash", "bash", {"command": "printf c1c-bash"})]), LLMResponse("done")],
                )
                result = start_runner(
                    kernel,
                    "keep-bash",
                    Agent("bash", "Use bash.", model="m"),
                    "go",
                    run_config=RunConfig(model_provider=provider, workspace=tmp, tool_registry_factory=bash_registry),
                ).result()
                assert result.raw_result.cycles[0].tool_results[0].content == "c1c-bash", (
                    result.raw_result.cycles[0].tool_results[0].to_dict()
                )
                import shlex

                for case in bash["launch_cases"]:
                    captured = []
                    script = "import time; print('kernel live output', flush=True); time.sleep(30)"
                    command = shlex.quote(sys.executable) + " -u -c " + shlex.quote(script)

                    def query(request, captured=captured):
                        receipt = next(m for m in request.messages if m.tool_call_id == "launch")
                        session_id = json.loads(receipt.content)["session_id"]
                        captured.append(session_id)
                        return LLMResponse("", [ToolCall("query", "check_background_command", {"session_id": session_id})])

                    def stop(request, captured=captured):
                        receipt = next(m for m in request.messages if m.tool_call_id == "query")
                        assert json.loads(receipt.content)["status"] == "running"
                        return LLMResponse("", [ToolCall("stop", "stop_background_command", {"session_id": captured[0]})])

                    launch = LLMResponse(
                        "", [ToolCall("launch", "bash", {"command": command, "yield_time_ms": case["yield_time_ms"]})]
                    )
                    result = start_runner(
                        kernel,
                        "keep-bash-" + case["name"],
                        Agent("bash", "Launch, query and stop."),
                        "go",
                        run_config=RunConfig(
                            model_provider=ScriptedModelProvider.from_steps(
                                "test", "m", [launch, query, stop, LLMResponse("done")]
                            ),
                            workspace=tmp,
                            tool_registry_factory=bash_registry,
                        ),
                    ).result()
                    receipts = [r for cycle in result.raw_result.cycles for r in cycle.tool_results]
                    launch_result = receipts[0]
                    assert launch_result.status_code.value == bash["running_receipt"]["status_code"]
                    assert launch_result.directive.value == bash["running_receipt"]["directive"]
                    assert launch_result.metadata["status"] == bash["running_receipt"]["process_status"]
                    assert all(k in launch_result.metadata for k in bash["running_receipt"]["required_metadata"])
                    assert not set(bash["running_receipt"]["forbidden_metadata"]) & launch_result.metadata.keys()
                    assert receipts[-1].metadata["status"] == "stopped" and result.final_output == "done"
                import pytest

                from vv_agent.runtime.background_sessions import BackgroundSessionManager
                from vv_agent.tools.handlers import background as background_handler
                from vv_agent.tools.handlers import bash as bash_handler
                from vv_agent.workspace.local import LocalWorkspaceBackend

                with patch.object(sys, "path", [str(root / "tests"), *sys.path]):
                    suite = runpy.run_path(str(root / "tests/test_bash_process_management.py"))
                manager = BackgroundSessionManager()
                with (
                    patch.object(bash_handler, "background_session_manager", manager),
                    patch.object(background_handler, "background_session_manager", manager),
                ):
                    try:
                        for tool_name in ("check_background_command", "stop_background_command"):
                            for identity in ("task", "workspace"):
                                with TemporaryDirectory(dir=tmp) as scoped_tmp, pytest.MonkeyPatch.context() as monkeypatch:
                                    suite["test_foreign_owner_has_zero_process_output_artifact_or_stop_effects"](
                                        Path(scoped_tmp), manager, monkeypatch, tool_name, identity
                                    )
                        with TemporaryDirectory(dir=tmp) as scoped_tmp, pytest.MonkeyPatch.context() as monkeypatch:
                            suite["test_missing_and_unconfirmed_observations_do_not_invent_exit_codes"](
                                Path(scoped_tmp), manager, monkeypatch
                            )
                    finally:
                        for session in list(manager._sessions.values()):
                            manager.stop_for_tool(
                                session.session_id,
                                session.artifact_backend or LocalWorkspaceBackend(Path(session.owner_workspace)),
                                session.owner_task_id,
                                "cleanup",
                                workspace=Path(session.owner_workspace),
                            )
            finally:
                kernel.close()

        self.keep["bash_process_management.json"] = (
            "PASS: all 23 invalid ingress cases, both kernel launch/query/stop cases, running receipts; "
            "shared real handlers prove task/workspace ownership, missing/stopping/unknown observations; "
            "original-manager recovery limit remains unchanged"
        )
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from vv_agent import cli

        provider = ScriptedModelProvider.new("test", "m", [LLMResponse("cli done")])

        class ConfiguredProvider:
            @staticmethod
            def from_settings_file(path):
                del path
                return SimpleNamespace(with_default_backend=lambda _: provider)

        stdout, stderr = StringIO(), StringIO()
        kernel = SessionDriver()
        with (
            TemporaryDirectory(prefix="c1c-cli-") as tmp,
            patch.object(cli, "VvLlmModelProvider", ConfiguredProvider),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            assert (
                cli._run_task_cli(
                    ["--prompt", "go", "--model", "m", "--workspace", tmp], store=kernel.store, session_id="c1c-cli"
                )
                == 0
            )
        kernel.close()
        assert "cli done" in stdout.getvalue() and "Traceback" not in stderr.getvalue()
        from vv_agent.session.records import InboxItem

        for name in ("verbose_result", "failed_result", "cancelled_result"):
            kernel = SessionDriver()
            sid = "cli-" + name

            def response(_request, name=name, kernel=kernel, sid=sid):
                if name == "failed_result":
                    raise ValueError("fixture model failure")
                if name == "cancelled_result":
                    state, _, _ = kernel.store.read_state(sid)
                    kernel.push(sid, InboxItem("cancel", "control", {"action": "cancel"}, state.active_turn_id))
                return LLMResponse("cli done")

            provider = ScriptedModelProvider.from_steps("test", "m", [response, response])
            stdout, stderr = StringIO(), StringIO()
            with (
                TemporaryDirectory(prefix="c1c-cli-") as tmp,
                patch.object(cli, "VvLlmModelProvider", ConfiguredProvider),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                args = ["--prompt", "go", "--model", "m", "--workspace", tmp]
                if name == "verbose_result":
                    args.append("--verbose")
                code = cli._run_task_cli(args, store=kernel.store, session_id=sid)
            kernel.close()
            assert code == load("cli_contract.json")["process_outcomes"][name]["exit_code"]
            payload = json.loads(stdout.getvalue())
            assert payload["status"] == ("completed" if name == "verbose_result" else "failed")
            assert bool(stderr.getvalue()) == (name == "verbose_result")
            if name == "cancelled_result":
                assert payload["error"]["message"] == "cancel_requested", payload
        for args, outcome in ((["--help"], "help"), ([], "usage_error")):
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli._main(args)
            expected = load("cli_contract.json")["process_outcomes"][outcome]
            assert code == expected["exit_code"]
            assert bool(stdout.getvalue()) == (outcome == "help")
            assert bool(stderr.getvalue()) == (outcome != "help")

        class FailingProvider:
            @staticmethod
            def from_settings_file(_path):
                raise cli.ConfigError("fixture settings unavailable")

        stdout, stderr = StringIO(), StringIO()
        kernel = SessionDriver()
        with patch.object(cli, "VvLlmModelProvider", FailingProvider), redirect_stdout(stdout), redirect_stderr(stderr):
            assert cli._run_task_cli(["--prompt", "go"], store=kernel.store) == 1
        kernel.close()
        assert not stdout.getvalue() and stderr.getvalue().strip() == "fixture settings unavailable"
        self.keep["cli_contract.json"] = (
            "PASS: real private-selector CLI success/failure/cancellation/verbose and help/usage/configuration outcomes; "
            "shared parser/compiler prove settings precedence, argument and resolved-limit projections; "
            "stdout/stderr/exit codes unchanged"
        )

    def produce(self, app):
        self.completion()
        self.output_validation()
        self.budgets()
        self.after_cycle()
        self.prompts()
        self.public_api()
        self.usage()
        self.codecs()
        self.bounded()
        self.app_observable(app)
        self.children()
        self.definitions_results()
        self.memory()
        self.metadata_approval()
        self.stream()
        self.verify_keep()
        self.terminal()
        self.events_traces()
        self.invalid_events()
        self.outputs["session_codec.json"]["message_contract"]["reserved_metadata"]["_vv_agent_compaction"] = (
            "memory_local.json#/evidence_manifest"
        )
        self.outputs["memory_lifecycle.json"]["summary_pipeline"]["case_fixture"] = "memory_local.json#/summary_compaction"
        return self.outputs


def classify(before, after, path=""):
    """Classify named cases atomically and every other inventory entry by path."""
    if isinstance(before, list) and isinstance(after, list):

        def identity(value, index):
            return next(
                (str(value[k]) for k in ("id", "name", "case", "event_id") if isinstance(value, dict) and k in value), str(index)
            )

        old, new = ({identity(v, i): v for i, v in enumerate(items)} for items in (before, after))
        result = []
        for name in sorted(old.keys() | new.keys()):
            target = path + "[" + name + "]"
            if name not in new:
                result.append(("deleted", target))
            elif name not in old:
                result.append(("added", target))
            elif (isinstance(old[name], dict) and any(k in old[name] for k in ("name", "case", "event_id"))) or (
                isinstance(old[name], dict) and "id" in old[name] and "cases" in path
            ):
                result.append(("kept" if old[name] == new[name] else "changed", target))
            else:
                result.extend(classify(old[name], new[name], target))
        return result
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            target = path + "." + key if path else key
            if key not in after:
                result.append(("deleted", target))
            elif key not in before:
                result.append(("added", target))
            else:
                result.extend(classify(before[key], after[key], target))
        return result
    return [("kept" if before == after else "changed", path)]


def validate_replacements(values, independent):
    """Re-enter current value codecs; executable producers are checked during authoring."""
    from tempfile import TemporaryDirectory

    from vv_agent.budget import BudgetExhaustion, BudgetUsageSnapshot
    from vv_agent.event_store import JsonlRunEventStore

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            schema = value.get("schema_version")
            decoder = None
            if value.get("version") == "v6" and "event_id" in value:
                assert event_from_dict(value).to_dict() == value
            elif schema == "vv-agent.model-call.v2" and "call_id" in value:
                assert ModelCallRecord.from_dict(value).to_dict() == value
            elif schema == "vv-agent.task-token-usage.v3" and "model_calls" in value:
                assert TaskTokenUsage.from_dict(value).to_dict() == value
            elif schema == "vv-agent.token-usage.v1" and "usage_source" in value:
                decoder = TokenUsage
            elif {"cycles", "tool_calls", "elapsed_ms", "unavailable_dimensions"} <= value.keys():
                decoder = BudgetUsageSnapshot
            elif {"dimension", "reason", "enforcement_boundary", "overshoot"} <= value.keys():
                decoder = BudgetExhaustion
            elif {"role", "content"} <= value.keys():
                calls = value.get("tool_calls", [])
                if calls and any("name" in call for call in calls):
                    for call in calls:
                        assert ToolCall.from_dict(call).to_dict() == call
                else:
                    assert Message.from_dict(value).to_dict() == value
            elif {"tool_call_id", "content", "status_code", "directive"} <= value.keys():
                decoder = ToolExecutionResult
            elif {"task_id", "messages", "token_usage", "session_id", "turn_id"} <= value.keys():
                assert AgentResult.from_dict(value).to_dict() == value
            if decoder:
                assert decoder.from_dict(value).to_dict() == value
            for key, item in value.items():
                if key not in {
                    "input",
                    "invalid_cases",
                    "invalid_wire_cases",
                    "invalid_model_call_cases",
                    "invalid_task_wire_cases",
                    "agent_result_wire",
                    "task_projection_validation",
                    "wire_rejections",
                    "cases",
                }:
                    walk(item)

    for name, value in values.items():
        walk(value)
        if name.endswith(".jsonl") and name not in {"runner_trace.jsonl", "session_items.jsonl", "runner_session_messages.jsonl"}:
            for event in value:
                assert event_from_dict(event).to_dict() == event
    for case in values["session_codec.json"]["canonical_cases"]:
        assert Message.from_dict(case["input"]).to_dict() == case["canonical"]
    for case in values["session_codec.json"]["invalid_cases"]:
        reject(lambda item: Message.from_dict(item), case["input"])
    for case in values["run_events_invalid.json"]["reject"]:
        body = base64.b64decode(case["bytes_base64"], validate=True)
        assert sha256(body).hexdigest() == case["sha256"]
        reject(lambda item: event_from_dict(item), json.loads(body))
    for case in values["run_definition.json"]["golden_cases"] + values["run_definition.json"]["producer_cases"]:
        body = independent([case["definition"]])[0]
        assert body == base64.b64decode(case["bytes_base64"], validate=True)
        assert sha256(body).hexdigest() == case["definition_digest"] == case["sha256"]
    for case in values["prompt_bundle.json"]["scenarios"]:
        if "sections" in case["output"]:
            bundle = PromptBundle(tuple(PromptSection.from_dict(s) for s in case["output"]["sections"]))
            body = independent([[s.to_dict() for s in bundle.sections if s.stable]])[0]
            assert bundle.flatten() == case["output"]["flat_prompt"]
            assert bundle.stable_hash == sha256(body).hexdigest() == case["output"]["stable_hash"]
    with TemporaryDirectory(prefix="c1c1-sink-") as directory:
        sink = JsonlRunEventStore(Path(directory) / "events.jsonl")
        events = values["event_store_replay.jsonl"]
        for wire in events:
            sink.append(event_from_dict(wire))
        actual = [e.to_dict() for e in sink.replay(run_id=events[0]["run_id"])]
        assert actual == events
    return tuple(values)


def generate_replacements(fixtures, app, output, independent, facts, write_json):
    reference = BASE.parents[3] / "wt-c1-contract" / "fixtures"
    if reference.is_dir():
        for name in KEEP:
            assert (reference / name).read_bytes() == (BASE / name).read_bytes(), name
    keep_hashes = {name: sha256((BASE / name).read_bytes()).hexdigest() for name in KEEP}
    author = Author(fixtures, independent, facts)
    values = author.produce(app)
    validate_replacements(values, independent)
    changes = {}
    for name, value in values.items():
        changes[name] = classify(load(name), value)
        if name.endswith(".jsonl"):
            (output / name).write_bytes(b"".join(raw + b"\n" for raw in independent(value)))
        else:
            write_json(output, name, value)
    assert keep_hashes == {name: sha256((BASE / name).read_bytes()).hexdigest() for name in KEEP}
    return {
        "replaced": changes,
        "keep": author.keep,
        "public_api_delta": author.public_delta,
        "reclassifications": [],
        "intentional_invalid_jsonl_records": {},
        "keep_sha256": keep_hashes,
    }


def disposition_reason(name, action, entry):
    if action == "kept":
        return "retained observable content reproduced by current shared/kernel producers; " + REASONS[name]
    if action == "deleted":
        if name == "run_definition.json":
            return (
                "Q3 opaque definition replaces the deep v5 schema, credential slots and old capability registry; "
                "section 6 removes that authority"
            )
        if name.endswith(".jsonl"):
            return "old carriers/identities replaced by current real-producer log projections; " + REASONS[name]
        return "retired owner/carrier or superseded expectation removed, with no tombstone; " + REASONS[name]
    if name == "run_definition.json" and "tool_policy_set_reordering_is_normalized" in entry:
        return (
            "current opaque task metadata retains allowed-tools input order/duplicates, so digest now differs; "
            "Q3 freezes current descriptor facts"
        )
    return (
        ("current producer facts regenerated" if action == "changed" else "current producer coverage added")
        + "; "
        + REASONS[name]
    )


def write_report(report, path):
    from collections import Counter

    lines = [
        "# C1c-1b fixture production report",
        "",
        "v23 source: the unchanged vendored snapshot, byte-identical to the read-only contract fixtures at authoring start.",
        "All dispositions use proposal sections 1.2/4/5/6 and binding reviewer decisions; "
        "no contract or vendored files were written.",
        "The generator rechecks the nearby read-only contract source when present "
        "and verifies all Keep-byte hashes before/after.",
        "Only the private selector uses new Runner/ConfiguredRunner, Message/result and JSONL-sink paths. "
        "Source fixes preserve independent subscriptions, durable cancellation and idempotent cancellation admission, "
        "continuation hints, strict current values, and retained child error details. "
        "Configured memory routes now resolve once, freeze their own endpoint bindings, dispatch through the matching "
        "client and retain the matching backend in usage projections; the v23 route facts remain unchanged. "
        "Public exports/default selection remain v23.",
        "",
        "| File | Kept | Changed | Deleted | Added | Reason |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, entries in sorted(report["replaced"].items()):
        counts = Counter(action for action, _ in entries)
        lines.append(
            f"| {name} | {counts['kept']} | {counts['changed']} | {counts['deleted']} | {counts['added']} | {REASONS[name]} |"
        )
    lines += [
        "",
        "Counts classify named cases atomically; all other entries use their JSON paths.",
        "",
        "## Keep reproduction",
        "",
        *["- " + name + ": " + result for name, result in sorted(report["keep"].items())],
        "",
        "## Preservation",
        "",
        "Unchanged v23 named cases retain their identities and bytes. Rendered prompts, memory summaries, "
        "evidence and provider deltas are reproduced before emission. Retired inputs remain negative base64 vectors.",
        "The full producer corpus is validated before deterministic set cover. Event files contain representatives "
        "per type/variant; named semantic and recovery scenarios appear once. Built-in schemas are checked separately "
        "through Runtime.registry against builtin_tools.json and are not dumped into ordinary definitions.",
    ]
    path.write_text("\n".join(lines) + "\n")
