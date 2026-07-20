from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import platform
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vv_agent.prompt import PromptSection, SystemPromptBuilder, build_system_prompt_bundle
from vv_agent.tools import ToolExposure, build_default_registry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
CANONICAL_FIXTURES = ("public_api_v1.json", "prompt_bundle_v1.json", "builtin_tools_v1.json")
EXPECTED_DOMAINS = (
    "agent",
    "runner",
    "run_config",
    "result",
    "run_handle",
    "interactive",
    "app_server",
    "tools",
    "workspace",
    "memory",
    "skills",
    "tracing",
    "llm_bridge",
    "runtime_backend",
)
EXPECTED_RUNNER_OPERATIONS = ("run", "start", "stream", "resume", "configured")
EXPECTED_RUN_HANDLE_OPERATIONS = (
    "cancel",
    "events",
    "result",
    "state",
    "approve",
    "steer",
    "follow_up",
    "resume",
)
EXPECTED_APP_SERVER_PROTOCOL_OPERATIONS = (
    "initialize",
    "thread/start",
    "thread/resume",
    "thread/read",
    "thread/list",
    "thread/archive",
    "thread/unsubscribe",
    "turn/start",
    "turn/interrupt",
    "turn/resume",
    "turn/steer",
    "turn/followUp",
    "approval/resolve",
    "model/list",
    "schema/export",
    "initialized",
)


def _member(
    member_id: str,
    python_name: str,
    rust_name: str,
    *,
    kind: str,
    python_kind: str,
    rust_kind: str,
    adaptation: str | None = None,
    python_target: str | None = None,
    rust_target: str | None = None,
) -> dict[str, Any]:
    member: dict[str, Any] = {
        "id": member_id,
        "kind": kind,
        "python": {"kind": python_kind, "name": python_name},
        "rust": {"kind": rust_kind, "name": rust_name},
    }
    if adaptation:
        member["adaptation"] = adaptation
    if python_target:
        member["python"]["target"] = python_target
    if rust_target:
        member["rust"]["target"] = rust_target
    return member


def _operation(
    member_id: str,
    python_name: str,
    rust_name: str,
    *,
    adaptation: str | None = None,
    python_target: str | None = None,
    rust_target: str | None = None,
    python_kind: str = "method",
    rust_kind: str = "method",
) -> dict[str, Any]:
    return _member(
        member_id,
        python_name,
        rust_name,
        kind="operation",
        python_kind=python_kind,
        rust_kind=rust_kind,
        adaptation=adaptation,
        python_target=python_target,
        rust_target=rust_target,
    )


def _field(
    member_id: str,
    python_name: str,
    rust_name: str,
    *,
    adaptation: str | None = None,
    python_target: str | None = None,
    rust_target: str | None = None,
    python_kind: str = "field",
    rust_kind: str = "field",
) -> dict[str, Any]:
    return _member(
        member_id,
        python_name,
        rust_name,
        kind="field",
        python_kind=python_kind,
        rust_kind=rust_kind,
        adaptation=adaptation,
        python_target=python_target,
        rust_target=rust_target,
    )


PUBLIC_API_DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "id": "agent",
        "capabilities": [
            {"id": "agent.definition", "python": "vv_agent.Agent", "rust": "vv_agent::Agent"},
            {"id": "agent.run_context", "python": "vv_agent.RunContext", "rust": "vv_agent::RunContext"},
            {
                "id": "agent.tool_use_behavior",
                "python": "vv_agent.agent.ToolUseBehavior",
                "rust": "vv_agent::agent::ToolUseBehavior",
            },
            {"id": "agent.no_tool_policy", "python": "vv_agent.NoToolPolicy", "rust": "vv_agent::NoToolPolicy"},
            {"id": "agent.handoff", "python": "vv_agent.Handoff", "rust": "vv_agent::Handoff"},
            {
                "id": "agent.background_task",
                "python": "vv_agent.BackgroundAgentTask",
                "rust": "vv_agent::BackgroundAgentTask",
            },
            {
                "adaptation": "Python names the value GuardrailResult; Rust names it GuardrailOutcome.",
                "id": "agent.guardrail_outcome",
                "python": "vv_agent.GuardrailResult",
                "rust": "vv_agent::GuardrailOutcome",
            },
        ],
    },
    {
        "id": "runner",
        "capabilities": [
            {"id": "runner.facade", "python": "vv_agent.Runner", "rust": "vv_agent::Runner"},
            {
                "adaptation": "Python returns a configured facade; Rust uses a builder.",
                "id": "runner.configured",
                "python": "vv_agent.ConfiguredRunner",
                "rust": "vv_agent::runner::RunnerBuilder",
            },
        ],
    },
    {
        "id": "run_config",
        "capabilities": [
            {"id": "run_config.options", "python": "vv_agent.RunConfig", "rust": "vv_agent::RunConfig"},
            {
                "id": "run_config.no_tool_policy",
                "python": "vv_agent.NoToolPolicy",
                "rust": "vv_agent::NoToolPolicy",
            },
            {
                "id": "run_config.tool_policy",
                "python": "vv_agent.ToolPolicy",
                "rust": "vv_agent::ToolPolicy",
            },
            {
                "id": "run_config.approval_policy",
                "python": "vv_agent.ApprovalPolicy",
                "rust": "vv_agent::ApprovalPolicy",
            },
            {
                "id": "run_config.cancellation",
                "python": "vv_agent.runtime.CancellationToken",
                "rust": "vv_agent::CancellationToken",
            },
            {
                "id": "run_config.approval_broker",
                "python": "vv_agent.ApprovalBroker",
                "rust": "vv_agent::ApprovalBroker",
            },
            {
                "id": "run_config.context_provider",
                "python": "vv_agent.ContextProvider",
                "rust": "vv_agent::ContextProvider",
            },
            {
                "id": "run_config.after_cycle_hook",
                "python": "vv_agent.AfterCycleHook",
                "rust": "vv_agent::AfterCycleHook",
            },
            {
                "id": "run_config.after_cycle_snapshot",
                "python": "vv_agent.AfterCycleSnapshot",
                "rust": "vv_agent::AfterCycleSnapshot",
            },
            {
                "id": "run_config.after_cycle_decision",
                "python": "vv_agent.AfterCycleDecision",
                "rust": "vv_agent::AfterCycleDecision",
            },
            {
                "id": "run_config.after_cycle_action",
                "python": "vv_agent.AfterCycleAction",
                "rust": "vv_agent::AfterCycleAction",
            },
            {
                "id": "run_config.native_cycle_outcome",
                "python": "vv_agent.NativeCycleOutcome",
                "rust": "vv_agent::NativeCycleOutcome",
            },
            {
                "id": "run_config.native_cycle_outcome_kind",
                "python": "vv_agent.NativeCycleOutcomeKind",
                "rust": "vv_agent::NativeCycleOutcomeKind",
            },
            {
                "id": "run_config.run_budget_limits",
                "python": "vv_agent.RunBudgetLimits",
                "rust": "vv_agent::RunBudgetLimits",
            },
            {
                "id": "run_config.host_cost_meter",
                "python": "vv_agent.HostCostMeter",
                "rust": "vv_agent::HostCostMeter",
            },
            {
                "id": "run_config.unavailable_metric_policy",
                "python": "vv_agent.UnavailableMetricPolicy",
                "rust": "vv_agent::UnavailableMetricPolicy",
            },
            {
                "id": "run_config.checkpoint_config",
                "python": "vv_agent.CheckpointConfig",
                "rust": "vv_agent::CheckpointConfig",
            },
            {
                "id": "checkpoint_config.capability_refs",
                "python": "vv_agent.CheckpointConfig.capability_refs",
                "rust": "vv_agent::CheckpointConfig::capability_refs",
            },
            {
                "id": "checkpoint_config.credential_slots",
                "python": "vv_agent.CheckpointConfig.credential_slots",
                "rust": "vv_agent::CheckpointConfig::credential_slots",
            },
            {
                "id": "run_config.checkpoint_extension",
                "python": "vv_agent.CheckpointExtension",
                "rust": "vv_agent::CheckpointExtension",
            },
            {
                "id": "run_config.reconciliation_provider",
                "python": "vv_agent.ReconciliationProvider",
                "rust": "vv_agent::ReconciliationProvider",
            },
        ],
    },
    {
        "id": "result",
        "capabilities": [
            {"id": "result.public", "python": "vv_agent.RunResult", "rust": "vv_agent::RunResult"},
            {"id": "result.resume_state", "python": "vv_agent.RunState", "rust": "vv_agent::RunState"},
            {
                "id": "result.approval_snapshot",
                "python": "vv_agent.ApprovalSnapshot",
                "rust": "vv_agent::ApprovalSnapshot",
            },
            {
                "id": "result.runtime_result",
                "python": "vv_agent.types.AgentResult",
                "rust": "vv_agent::AgentResult",
            },
            {"id": "result.status", "python": "vv_agent.AgentStatus", "rust": "vv_agent::AgentStatus"},
            {
                "id": "result.completion_reason",
                "python": "vv_agent.CompletionReason",
                "rust": "vv_agent::CompletionReason",
            },
            {
                "id": "result.usage_source",
                "python": "vv_agent.UsageSource",
                "rust": "vv_agent::UsageSource",
            },
            {
                "id": "result.cache_usage_status",
                "python": "vv_agent.CacheUsageStatus",
                "rust": "vv_agent::CacheUsageStatus",
            },
            {
                "id": "result.cache_usage",
                "python": "vv_agent.CacheUsage",
                "rust": "vv_agent::CacheUsage",
            },
            {
                "id": "result.token_usage",
                "python": "vv_agent.TokenUsage",
                "rust": "vv_agent::TokenUsage",
            },
            {
                "id": "result.task_token_usage",
                "python": "vv_agent.TaskTokenUsage",
                "rust": "vv_agent::TaskTokenUsage",
            },
            {"id": "result.host_cost", "python": "vv_agent.HostCost", "rust": "vv_agent::HostCost"},
            {
                "id": "result.budget_dimension",
                "python": "vv_agent.BudgetDimension",
                "rust": "vv_agent::BudgetDimension",
            },
            {
                "id": "result.budget_enforcement_boundary",
                "python": "vv_agent.BudgetEnforcementBoundary",
                "rust": "vv_agent::BudgetEnforcementBoundary",
            },
            {
                "id": "result.budget_exhaustion_reason",
                "python": "vv_agent.BudgetExhaustionReason",
                "rust": "vv_agent::BudgetExhaustionReason",
            },
            {
                "id": "result.budget_unavailable_reason",
                "python": "vv_agent.BudgetUnavailableReason",
                "rust": "vv_agent::BudgetUnavailableReason",
            },
            {
                "id": "result.budget_unavailable_dimension",
                "python": "vv_agent.BudgetUnavailableDimension",
                "rust": "vv_agent::BudgetUnavailableDimension",
            },
            {
                "id": "result.budget_usage_snapshot",
                "python": "vv_agent.BudgetUsageSnapshot",
                "rust": "vv_agent::BudgetUsageSnapshot",
            },
            {
                "id": "result.budget_exhaustion",
                "python": "vv_agent.BudgetExhaustion",
                "rust": "vv_agent::BudgetExhaustion",
            },
            {
                "id": "result.resume_observation",
                "python": "vv_agent.ResumeObservation",
                "rust": "vv_agent::ResumeObservation",
            },
        ],
    },
    {
        "id": "run_handle",
        "capabilities": [
            {"id": "run_handle.live", "python": "vv_agent.RunHandle", "rust": "vv_agent::RunHandle"},
            {
                "id": "run_handle.snapshot",
                "python": "vv_agent.RunHandleState",
                "rust": "vv_agent::RunHandleState",
            },
            {
                "id": "run_handle.status",
                "python": "vv_agent.run_handle.RunHandleStatus",
                "rust": "vv_agent::RunHandleStatus",
            },
        ],
    },
    {
        "id": "interactive",
        "capabilities": [
            {
                "adaptation": "Python calls the embedded session AgentSession; Rust calls it InteractiveSession.",
                "id": "interactive.session",
                "python": "vv_agent.AgentSession",
                "rust": "vv_agent::InteractiveSession",
            },
            {
                "id": "interactive.options",
                "python": "vv_agent.AgentSessionOptions",
                "rust": "vv_agent::InteractiveSessionOptions",
            },
            {
                "id": "interactive.state",
                "python": "vv_agent.AgentSessionState",
                "rust": "vv_agent::InteractiveSessionState",
            },
            {
                "id": "interactive.client",
                "python": "vv_agent.InteractiveAgentClient",
                "rust": "vv_agent::InteractiveAgentClient",
            },
            {
                "id": "interactive.event",
                "python": "vv_agent.interactive.AgentSessionEvent",
                "rust": "vv_agent::InteractiveSessionEvent",
            },
            {
                "adaptation": "The languages expose their native session error hierarchy.",
                "id": "interactive.error",
                "python": "vv_agent.interactive.AgentSessionEventGapError",
                "rust": "vv_agent::InteractiveSessionError",
            },
        ],
    },
    {
        "id": "app_server",
        "capabilities": [
            {"id": "app_server.server", "python": "vv_agent.AppServer", "rust": "vv_agent::app_server::AppServer"},
            {
                "id": "app_server.host",
                "python": "vv_agent.AppServerHost",
                "rust": "vv_agent::AppServerHost",
            },
            {
                "id": "app_server.default_host",
                "python": "vv_agent.DefaultAppServerHost",
                "rust": "vv_agent::DefaultAppServerHost",
            },
            {
                "id": "app_server.client",
                "python": "vv_agent.app_server.AppServerClient",
                "rust": "vv_agent::app_server::client::AppServerClient",
            },
            {
                "id": "app_server.client_error",
                "python": "vv_agent.app_server.AppServerClientError",
                "rust": "vv_agent::app_server::client::AppServerClientError",
            },
            {
                "id": "app_server.processor",
                "python": "vv_agent.app_server.MessageProcessor",
                "rust": "vv_agent::app_server::processor::MessageProcessor",
            },
            {
                "adaptation": "Python names the sender facade OutgoingRouter; Rust names it OutgoingMessageSender.",
                "id": "app_server.outgoing_router",
                "python": "vv_agent.app_server.OutgoingRouter",
                "rust": "vv_agent::app_server::outgoing::OutgoingMessageSender",
            },
            {
                "id": "app_server.transport",
                "python": "vv_agent.app_server.transport.AppServerTransport",
                "rust": "vv_agent::app_server::transport::AppServerTransport",
            },
            {
                "id": "app_server.channel_transport",
                "python": "vv_agent.app_server.ChannelTransport",
                "rust": "vv_agent::app_server::transport::channel::ChannelTransport",
            },
            {
                "id": "app_server.stdio_transport",
                "python": "vv_agent.app_server.StdioJsonlTransport",
                "rust": "vv_agent::app_server::transport::stdio::StdioJsonlTransport",
            },
            {
                "id": "app_server.jsonrpc_message",
                "python": "vv_agent.app_server.JsonRpcMessage",
                "rust": "vv_agent::app_server::protocol::JsonRpcMessage",
            },
            {
                "id": "app_server.turn_resume_params",
                "python": "vv_agent.app_server.TurnResumeParams",
                "rust": "vv_agent::app_server::protocol::TurnResumeParams",
            },
            {
                "id": "app_server.turn_resume_response",
                "python": "vv_agent.app_server.TurnResumeResponse",
                "rust": "vv_agent::app_server::protocol::TurnResumeResponse",
            },
        ],
    },
    {
        "id": "tools",
        "capabilities": [
            {"id": "tools.public_tool", "python": "vv_agent.Tool", "rust": "vv_agent::Tool"},
            {
                "id": "tools.function_tool",
                "python": "vv_agent.FunctionTool",
                "rust": "vv_agent::FunctionTool",
            },
            {
                "id": "tools.registry",
                "python": "vv_agent.ToolRegistry",
                "rust": "vv_agent::ToolRegistry",
            },
            {
                "id": "tools.executor",
                "python": "vv_agent.tools.ToolExecutor",
                "rust": "vv_agent::ToolExecutor",
            },
            {"id": "tools.spec", "python": "vv_agent.tools.ToolSpec", "rust": "vv_agent::ToolSpec"},
            {
                "id": "tools.context",
                "python": "vv_agent.ToolContext",
                "rust": "vv_agent::ToolContext",
            },
            {
                "id": "tools.call_context",
                "python": "vv_agent.ToolCallContext",
                "rust": "vv_agent::ToolCallContext",
            },
            {
                "id": "tools.orchestrator",
                "python": "vv_agent.tools.ToolOrchestrator",
                "rust": "vv_agent::ToolOrchestrator",
            },
            {
                "id": "tools.exposure",
                "python": "vv_agent.ToolExposure",
                "rust": "vv_agent::ToolExposure",
            },
            {
                "id": "tools.output",
                "python": "vv_agent.ToolOutput",
                "rust": "vv_agent::ToolOutput",
            },
            {
                "adaptation": "Python adapts registry handlers; Rust adapts ToolSpec values.",
                "id": "tools.spec_executor",
                "python": "vv_agent.tools.RegistryToolExecutor",
                "rust": "vv_agent::ToolSpecExecutor",
            },
            {
                "id": "tools.not_found_error",
                "python": "vv_agent.tools.ToolNotFoundError",
                "rust": "vv_agent::ToolNotFoundError",
            },
            {
                "id": "tools.idempotency",
                "python": "vv_agent.ToolIdempotency",
                "rust": "vv_agent::ToolIdempotency",
            },
        ],
    },
    {
        "id": "workspace",
        "capabilities": [
            {
                "id": "workspace.backend",
                "python": "vv_agent.workspace.WorkspaceBackend",
                "rust": "vv_agent::WorkspaceBackend",
            },
            {
                "id": "workspace.local",
                "python": "vv_agent.workspace.LocalWorkspaceBackend",
                "rust": "vv_agent::LocalWorkspaceBackend",
            },
            {
                "id": "workspace.memory",
                "python": "vv_agent.workspace.MemoryWorkspaceBackend",
                "rust": "vv_agent::MemoryWorkspaceBackend",
            },
            {
                "id": "workspace.s3",
                "python": "vv_agent.workspace.S3WorkspaceBackend",
                "rust": "vv_agent::S3WorkspaceBackend",
            },
            {"id": "workspace.file_info", "python": "vv_agent.workspace.FileInfo", "rust": "vv_agent::FileInfo"},
            {
                "id": "workspace.discovery_filter",
                "python": "vv_agent.workspace.DiscoveryFilteredWorkspaceBackend",
                "rust": "vv_agent::DiscoveryFilteredWorkspaceBackend",
            },
            {
                "adaptation": "Both errors report the same portable-regex contract using native naming.",
                "id": "workspace.portable_regex_error",
                "python": "vv_agent.workspace.InvalidPortableRegexError",
                "rust": "vv_agent::PortableRegexError",
            },
        ],
    },
    {
        "id": "memory",
        "capabilities": [
            {"id": "memory.manager", "python": "vv_agent.memory.MemoryManager", "rust": "vv_agent::MemoryManager"},
            {
                "id": "memory.provider",
                "python": "vv_agent.memory.MemoryProvider",
                "rust": "vv_agent::MemoryProvider",
            },
            {
                "id": "memory.provider_result",
                "python": "vv_agent.memory.MemoryProviderResult",
                "rust": "vv_agent::MemoryProviderResult",
            },
            {
                "id": "memory.session",
                "python": "vv_agent.memory.SessionMemory",
                "rust": "vv_agent::SessionMemory",
            },
            {
                "id": "memory.session_config",
                "python": "vv_agent.memory.SessionMemoryConfig",
                "rust": "vv_agent::SessionMemoryConfig",
            },
            {
                "id": "memory.session_entry",
                "python": "vv_agent.memory.SessionMemoryEntry",
                "rust": "vv_agent::SessionMemoryEntry",
            },
            {
                "id": "memory.session_state",
                "python": "vv_agent.memory.SessionMemoryState",
                "rust": "vv_agent::SessionMemoryState",
            },
            {
                "id": "memory.search_request",
                "python": "vv_agent.memory.MemorySearchRequest",
                "rust": "vv_agent::MemorySearchRequest",
            },
            {
                "id": "memory.search_result",
                "python": "vv_agent.memory.MemorySearchResult",
                "rust": "vv_agent::MemorySearchResult",
            },
            {
                "id": "memory.save_request",
                "python": "vv_agent.memory.MemorySaveRequest",
                "rust": "vv_agent::MemorySaveRequest",
            },
            {
                "id": "memory.save_result",
                "python": "vv_agent.memory.MemorySaveResult",
                "rust": "vv_agent::MemorySaveResult",
            },
            {
                "id": "memory.compaction_exhausted",
                "python": "vv_agent.memory.CompactionExhaustedError",
                "rust": "vv_agent::CompactionExhaustedError",
            },
        ],
    },
    {
        "id": "skills",
        "capabilities": [
            {
                "id": "skills.properties",
                "python": "vv_agent.skills.SkillProperties",
                "rust": "vv_agent::skills::SkillProperties",
            },
            {
                "id": "skills.loaded",
                "python": "vv_agent.skills.LoadedSkill",
                "rust": "vv_agent::skills::LoadedSkill",
            },
            {"id": "skills.entry", "python": "vv_agent.skills.SkillEntry", "rust": "vv_agent::skills::SkillEntry"},
            {"id": "skills.error", "python": "vv_agent.skills.SkillError", "rust": "vv_agent::skills::SkillError"},
            {
                "id": "skills.parse_error",
                "python": "vv_agent.skills.SkillParseError",
                "rust": "vv_agent::skills::SkillParseError",
            },
            {
                "id": "skills.validation_error",
                "python": "vv_agent.skills.SkillValidationError",
                "rust": "vv_agent::skills::SkillValidationError",
            },
            {
                "id": "skills.validation_diagnostics",
                "python": "vv_agent.skills.ValidationDiagnostics",
                "rust": "vv_agent::skills::ValidationDiagnostics",
            },
            {
                "id": "skills.validation_mode",
                "python": "vv_agent.skills.ValidationMode",
                "rust": "vv_agent::skills::ValidationMode",
            },
        ],
    },
    {
        "id": "tracing",
        "capabilities": [
            {"id": "tracing.span", "python": "vv_agent.Span", "rust": "vv_agent::Span"},
            {"id": "tracing.sink", "python": "vv_agent.TraceSink", "rust": "vv_agent::TraceSink"},
            {
                "id": "tracing.jsonl_exporter",
                "python": "vv_agent.JsonlTraceExporter",
                "rust": "vv_agent::JsonlTraceExporter",
            },
        ],
    },
    {
        "id": "llm_bridge",
        "capabilities": [
            {
                "adaptation": "The acronyms follow each language's naming convention.",
                "id": "llm_bridge.client",
                "python": "vv_agent.llm.LLMClient",
                "rust": "vv_agent::LlmClient",
            },
            {"id": "llm_bridge.request", "python": "vv_agent.llm.LlmRequest", "rust": "vv_agent::LlmRequest"},
            {"id": "llm_bridge.error", "python": "vv_agent.llm.LlmError", "rust": "vv_agent::LlmError"},
            {
                "id": "llm_bridge.endpoint",
                "python": "vv_agent.llm.EndpointTarget",
                "rust": "vv_agent::EndpointTarget",
            },
            {
                "adaptation": "Python exposes ScriptedLLM; Rust exposes ScriptedLlmClient.",
                "id": "llm_bridge.scripted",
                "python": "vv_agent.llm.ScriptedLLM",
                "rust": "vv_agent::ScriptedLlmClient",
            },
            {
                "id": "llm_bridge.vv_llm_client",
                "python": "vv_agent.VvLlmClient",
                "rust": "vv_agent::VvLlmClient",
            },
            {
                "id": "llm_bridge.model_provider",
                "python": "vv_agent.ModelProvider",
                "rust": "vv_agent::ModelProvider",
            },
            {"id": "llm_bridge.model_ref", "python": "vv_agent.ModelRef", "rust": "vv_agent::ModelRef"},
            {
                "id": "llm_bridge.model_settings",
                "python": "vv_agent.ModelSettings",
                "rust": "vv_agent::ModelSettings",
            },
            {
                "id": "llm_bridge.response_format",
                "python": "vv_agent.ResponseFormat",
                "rust": "vv_agent::ResponseFormat",
            },
            {
                "id": "llm_bridge.retry_settings",
                "python": "vv_agent.RetrySettings",
                "rust": "vv_agent::RetrySettings",
            },
            {"id": "llm_bridge.tool_choice", "python": "vv_agent.ToolChoice", "rust": "vv_agent::ToolChoice"},
        ],
    },
    {
        "id": "runtime_backend",
        "capabilities": [
            {
                "adaptation": "Python defines the backend protocol; Rust uses a backend enum.",
                "id": "runtime_backend.execution",
                "python": "vv_agent.runtime.backends.ExecutionBackend",
                "rust": "vv_agent::RuntimeExecutionBackend",
            },
            {
                "id": "runtime_backend.inline",
                "python": "vv_agent.runtime.backends.InlineBackend",
                "rust": "vv_agent::InlineBackend",
            },
            {
                "id": "runtime_backend.thread",
                "python": "vv_agent.runtime.backends.ThreadBackend",
                "rust": "vv_agent::ThreadBackend",
            },
            {
                "adaptation": "Python integrates Celery; Rust exposes the distributed backend used by Apalis adapters.",
                "id": "runtime_backend.distributed",
                "python": "vv_agent.runtime.backends.celery.CeleryBackend",
                "rust": "vv_agent::DistributedBackend",
            },
            {
                "id": "runtime_backend.envelope",
                "python": "vv_agent.runtime.backends.distributed.DistributedRunEnvelope",
                "rust": "vv_agent::DistributedRunEnvelope",
            },
            {
                "id": "runtime_backend.capability_registry",
                "python": "vv_agent.runtime.backends.distributed.DistributedCapabilityRegistry",
                "rust": "vv_agent::DistributedCapabilityRegistry",
            },
            {
                "id": "runtime_backend.capability_ref",
                "python": "vv_agent.runtime.backends.distributed.CapabilityRef",
                "rust": "vv_agent::CapabilityRef",
            },
            {
                "id": "runtime_backend.recipe",
                "python": "vv_agent.runtime.backends.distributed.RuntimeRecipe",
                "rust": "vv_agent::RuntimeRecipe",
            },
            {
                "id": "runtime_backend.state_store",
                "python": "vv_agent.runtime.state.StateStore",
                "rust": "vv_agent::StateStore",
            },
            {
                "id": "runtime_backend.in_memory_state_store",
                "python": "vv_agent.runtime.state.InMemoryStateStore",
                "rust": "vv_agent::InMemoryStateStore",
            },
            {
                "id": "runtime_backend.sqlite_state_store",
                "python": "vv_agent.runtime.stores.sqlite.SqliteStateStore",
                "rust": "vv_agent::SqliteStateStore",
            },
            {
                "id": "runtime_backend.redis_state_store",
                "python": "vv_agent.runtime.stores.redis.RedisStateStore",
                "rust": "vv_agent::RedisStateStore",
            },
            {
                "id": "runtime_backend.checkpoint",
                "python": "vv_agent.runtime.Checkpoint",
                "rust": "vv_agent::Checkpoint",
            },
            {
                "id": "runtime_backend.agent_runtime",
                "python": "vv_agent.runtime.AgentRuntime",
                "rust": "vv_agent::AgentRuntime",
            },
            {
                "id": "runtime_backend.cycle_runner",
                "python": "vv_agent.runtime.CycleRunner",
                "rust": "vv_agent::CycleRunner",
            },
            {
                "id": "runtime_backend.tool_call_runner",
                "python": "vv_agent.runtime.ToolCallRunner",
                "rust": "vv_agent::ToolCallRunner",
            },
            {
                "id": "runtime_backend.checkpoint_v2",
                "python": "vv_agent.runtime.CheckpointV2",
                "rust": "vv_agent::CheckpointV2",
            },
            {
                "id": "runtime_backend.checkpoint_store_v2",
                "python": "vv_agent.runtime.CheckpointStoreV2",
                "rust": "vv_agent::CheckpointStoreV2",
            },
            {
                "id": "runtime_backend.idempotent_event_store",
                "python": "vv_agent.IdempotentRunEventStore",
                "rust": "vv_agent::IdempotentRunEventStore",
            },
            {
                "id": "runtime_backend.operation_journal",
                "python": "vv_agent.runtime.OperationJournalEntry",
                "rust": "vv_agent::OperationJournalEntry",
            },
        ],
    },
)


PUBLIC_API_SURFACES: tuple[dict[str, Any], ...] = (
    {
        "id": "agent",
        "python_target": "vv_agent.Agent",
        "rust_target": "vv_agent::Agent",
        "members": [
            _field("name", "name", "name", rust_kind="method"),
            _field("instructions", "instructions", "instructions", rust_kind="method"),
            _field("model", "model", "model", rust_kind="method"),
            _field("model_settings", "model_settings", "model_settings", rust_kind="method"),
            _field("tools", "tools", "tools", rust_kind="method"),
            _field("handoffs", "handoffs", "handoffs", rust_kind="method"),
            _field(
                "input_guardrails",
                "input_guardrails",
                "input_guardrail",
                adaptation="Python stores the public list directly; Rust configures it through AgentBuilder.",
                rust_kind="method",
                rust_target="vv_agent::agent::AgentBuilder",
            ),
            _field(
                "output_guardrails",
                "output_guardrails",
                "output_guardrail",
                adaptation="Python stores the public list directly; Rust configures it through AgentBuilder.",
                rust_kind="method",
                rust_target="vv_agent::agent::AgentBuilder",
            ),
            _field(
                "output_type",
                "output_type",
                "output_type_name",
                adaptation="Python stores the target type; Rust exposes its type name and validator.",
                rust_kind="method",
            ),
            _field("hooks", "hooks", "hooks", rust_kind="method"),
            _field("max_cycles", "max_cycles", "max_cycles", rust_kind="method"),
            _field("no_tool_policy", "no_tool_policy", "no_tool_policy", rust_kind="method"),
            _field("tool_policy", "tool_policy", "tool_policy", rust_kind="method"),
            _field(
                "tool_use_behavior",
                "tool_use_behavior",
                "tool_use_behavior",
                rust_kind="method",
            ),
            _field(
                "stop_at_tool_names",
                "stop_at_tool_names",
                "tool_use_behavior",
                adaptation="Rust stores stop names inside the ToolUseBehavior enum variant.",
                rust_kind="method",
            ),
            _field("metadata", "metadata", "metadata", rust_kind="method"),
            _field("sub_agents", "sub_agents", "sub_agents", rust_kind="method"),
            _operation("resolve_instructions", "resolve_instructions", "resolve_instructions"),
            _operation("as_tool", "as_tool", "as_tool"),
            _operation("as_background_task", "as_background_task", "as_background_task"),
        ],
    },
    {
        "id": "runner",
        "python_target": "vv_agent.Runner",
        "rust_target": "vv_agent::Runner",
        "members": [
            _operation(
                "run",
                "run_sync",
                "run",
                adaptation="Python exposes a synchronous class facade; Rust exposes an async instance method.",
            ),
            _operation("start", "start", "start"),
            _operation(
                "stream",
                "stream_sync",
                "stream",
                adaptation="Python returns a synchronous iterator; Rust returns an async-capable RunEventStream.",
            ),
            _operation(
                "resume",
                "resume",
                "resume",
                adaptation="Python takes optional input directly; Rust also exposes resume_with_input.",
            ),
            _operation(
                "configured",
                "configured",
                "builder",
                adaptation="Python returns ConfiguredRunner; Rust returns RunnerBuilder.",
            ),
        ],
    },
    {
        "id": "run_config",
        "python_target": "vv_agent.RunConfig",
        "rust_target": "vv_agent::RunConfig",
        "members": [
            _field("model", "model", "model"),
            _field("model_provider", "model_provider", "model_provider"),
            _field("model_settings", "model_settings", "model_settings"),
            _field("workspace", "workspace", "workspace"),
            _field("workspace_backend", "workspace_backend", "workspace_backend"),
            _field("session", "session", "session"),
            _field("initial_messages", "initial_messages", "initial_messages"),
            _field("max_cycles", "max_cycles", "max_cycles"),
            _field("no_tool_policy", "no_tool_policy", "no_tool_policy"),
            _field("max_handoffs", "max_handoffs", "max_handoffs"),
            _field("tool_policy", "tool_policy", "tool_policy"),
            _field("execution_backend", "execution_backend", "execution_backend"),
            _field("cancellation_token", "cancellation_token", "cancellation_token"),
            _field("hooks", "hooks", "hooks"),
            _field("after_cycle_hooks", "after_cycle_hooks", "after_cycle_hooks"),
            _field("event_store", "event_store", "event_store"),
            _field("event_store_fail_closed", "event_store_fail_closed", "event_store_fail_closed"),
            _field("approval_provider", "approval_provider", "approval_provider"),
            _field(
                "approval_timeout",
                "approval_timeout_seconds",
                "approval_timeout",
                adaptation="Python stores seconds as a float; Rust stores Duration.",
            ),
            _field("approval_broker", "approval_broker", "approval_broker"),
            _field("context_providers", "context_providers", "context_providers"),
            _field("max_context_chars", "max_context_chars", "max_context_chars"),
            _field("memory_providers", "memory_providers", "memory_providers"),
            _field(
                "app_state",
                "context",
                "app_state",
                adaptation="Python names arbitrary host state context; Rust names it app_state.",
            ),
            _field(
                "initial_shared_state",
                "shared_state",
                "initial_shared_state",
                adaptation="Both values seed per-run shared state using native naming.",
            ),
            _field("tool_registry_factory", "tool_registry_factory", "tool_registry_factory"),
            _field("log_preview_chars", "log_preview_chars", "log_preview_chars"),
            _field("debug_dump_dir", "debug_dump_dir", "debug_dump_dir"),
            _field("before_cycle_messages", "before_cycle_messages", "before_cycle_messages"),
            _field("interruption_messages", "interruption_messages", "interruption_messages"),
            _field("sub_task_manager", "sub_task_manager", "sub_task_manager"),
            _field("runtime_log_handler", "runtime_log_handler", "runtime_log_handler"),
            _field("runtime_stream_callback", "runtime_stream_callback", "runtime_stream_callback"),
            _field("metadata", "metadata", "metadata"),
            _field(
                "trace_sink",
                "tracing",
                "trace_sink",
                adaptation="Python groups trace controls in a mapping; Rust uses typed trace fields.",
            ),
            _field(
                "trace_id",
                "tracing",
                "trace_id",
                adaptation="Python reads trace_id from the tracing mapping; Rust exposes a field.",
            ),
            _field(
                "workflow_name",
                "tracing",
                "workflow_name",
                adaptation="Python reads workflow_name from the tracing mapping; Rust exposes a field.",
            ),
            _field(
                "event_observer",
                "stream",
                "events",
                adaptation="Python accepts a callback; Rust exposes independent typed RunHandle event streams.",
                rust_kind="method",
                rust_target="vv_agent::RunHandle",
            ),
            _field(
                "settings_file",
                "settings_file",
                "model_provider",
                adaptation="Python can construct a provider from a settings path at run time; Rust receives a ModelProvider.",
            ),
            _field(
                "default_backend",
                "default_backend",
                "model_provider",
                adaptation="Python's settings adapter carries the backend selector; Rust's ModelProvider owns it.",
            ),
            _field(
                "llm_builder",
                "llm_builder",
                "model_provider",
                adaptation="Python supports a legacy builder callback; Rust uses the ModelProvider interface.",
            ),
            _field(
                "timeout_seconds",
                "timeout_seconds",
                "model_provider",
                adaptation="Python's legacy settings adapter carries timeout; Rust's provider/client owns transport timeout.",
            ),
            _field("budget_limits", "budget_limits", "budget_limits"),
            _field("host_cost_meter", "host_cost_meter", "host_cost_meter"),
            _field("checkpoint_config", "checkpoint_config", "checkpoint_config"),
            _field("checkpoint_extensions", "checkpoint_extensions", "checkpoint_extensions"),
            _field("reconciliation_provider", "reconciliation_provider", "reconciliation_provider"),
        ],
    },
    {
        "id": "run_result",
        "python_target": "vv_agent.RunResult",
        "rust_target": "vv_agent::RunResult",
        "members": [
            _field("input", "input", "input", rust_kind="method"),
            _field("new_items", "new_items", "new_items", rust_kind="method"),
            _field("final_output", "final_output", "final_output", rust_kind="method"),
            _field("status", "status", "status", rust_kind="method"),
            _field(
                "completion_reason",
                "completion_reason",
                "completion_reason",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "completion_tool_name",
                "completion_tool_name",
                "completion_tool_name",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "partial_output",
                "partial_output",
                "partial_output",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "budget_usage",
                "budget_usage",
                "budget_usage",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "budget_exhaustion",
                "budget_exhaustion",
                "budget_exhaustion",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "checkpoint_key",
                "checkpoint_key",
                "checkpoint_key",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "resume_observation",
                "resume_observation",
                "resume_observation",
                python_kind="property",
                rust_kind="method",
            ),
            _field(
                "raw_result",
                "raw_result",
                "result",
                adaptation="Python names the inner runtime result raw_result; Rust names its getter result.",
                rust_kind="method",
            ),
            _field("events", "events", "events", rust_kind="method"),
            _field("token_usage", "token_usage", "token_usage", rust_kind="method"),
            _field("trace_id", "trace_id", "trace_id", rust_kind="method"),
            _field("run_id", "run_id", "run_id", rust_kind="method"),
            _field("metadata", "metadata", "metadata", rust_kind="method"),
            _field("agent_name", "agent_name", "agent_name", rust_kind="method"),
            _field("resolved_model", "resolved_model", "resolved_model", rust_kind="method"),
            _operation("approval_snapshot", "approval_snapshot", "approval_snapshot"),
            _operation("into_state", "into_state", "into_state"),
            _operation("to_dict", "to_dict", "to_dict"),
        ],
    },
    {
        "id": "run_state",
        "python_target": "vv_agent.RunState",
        "rust_target": "vv_agent::RunState",
        "members": [
            _field("result", "result", "result", rust_kind="method"),
            _operation("from_result", "from_result", "from_result"),
            _operation("approve", "approve", "approve"),
            _operation("pending_approval_ids", "pending_approval_ids", "pending_approval_ids"),
            _operation(
                "approved_interruption_ids",
                "approved_interruption_ids",
                "approved_interruption_ids",
            ),
            _operation("approval_snapshot", "approval_snapshot", "approval_snapshot"),
        ],
    },
    {
        "id": "run_handle",
        "python_target": "vv_agent.RunHandle",
        "rust_target": "vv_agent::RunHandle",
        "members": [
            _operation("cancel", "cancel", "cancel"),
            _operation("events", "events", "events"),
            _operation("result", "result", "result"),
            _operation("state", "state", "state"),
            _operation("approve", "approve", "approve"),
            _operation("steer", "steer", "steer"),
            _operation("follow_up", "follow_up", "follow_up"),
            _operation("resume", "resume", "resume"),
        ],
    },
    {
        "id": "interactive_session",
        "python_target": "vv_agent.AgentSession",
        "rust_target": "vv_agent::InteractiveSession",
        "members": [
            _field("messages", "messages", "messages", python_kind="property", rust_kind="method"),
            _field("session", "session", "session", python_kind="property", rust_kind="method"),
            _field(
                "shared_state",
                "shared_state",
                "shared_state",
                python_kind="property",
                rust_kind="method",
            ),
            _field("latest_run", "latest_run", "latest_run", python_kind="property", rust_kind="method"),
            _field("running", "running", "running", python_kind="property", rust_kind="method"),
            _field("closed", "closed", "closed", python_kind="property", rust_kind="method"),
            _field(
                "active_run_handle",
                "active_run_handle",
                "active_run_handle",
                python_kind="property",
                rust_kind="method",
            ),
            _operation("subscribe", "subscribe", "subscribe"),
            _operation("close", "close", "close"),
            _operation("steer", "steer", "steer"),
            _operation("follow_up", "follow_up", "follow_up"),
            _operation("clear_queues", "clear_queues", "clear_queues"),
            _operation("cancel", "cancel", "cancel"),
            _operation("approve", "approve", "approve"),
            _operation(
                "prompt",
                "prompt",
                "prompt",
                adaptation="Python executes synchronously; Rust awaits the session run.",
            ),
            _operation("continue_run", "continue_run", "continue_run"),
            _operation("query", "query", "query"),
            _operation("state", "state", "state"),
            _operation("replace_messages", "replace_messages", "replace_messages"),
            _operation("replace_shared_state", "replace_shared_state", "replace_shared_state"),
        ],
    },
    {
        "id": "interactive_client",
        "python_target": "vv_agent.InteractiveAgentClient",
        "rust_target": "vv_agent::InteractiveAgentClient",
        "members": [
            _operation(
                "create_session",
                "create_session",
                "create_session",
                adaptation="Python creates synchronously; Rust resolves the session asynchronously.",
            )
        ],
    },
    {
        "id": "app_server",
        "python_target": "vv_agent.AppServer",
        "rust_target": "vv_agent::app_server::AppServer",
        "members": [
            _operation(
                "run",
                "run_forever",
                "run",
                adaptation="Python owns a blocking loop; Rust awaits a generic transport server.",
            )
        ],
    },
    {
        "id": "app_server_client",
        "python_target": "vv_agent.app_server.AppServerClient",
        "rust_target": "vv_agent::app_server::client::AppServerClient",
        "protocol_operations": [
            _operation("initialize", "initialize", "initialize"),
            _operation("thread/start", "start_thread", "start_thread"),
            _operation("thread/resume", "resume_thread", "resume_thread"),
            _operation("thread/read", "read_thread", "read_thread"),
            _operation("thread/list", "list_threads", "list_threads"),
            _operation("thread/archive", "archive_thread", "archive_thread"),
            _operation("thread/unsubscribe", "unsubscribe_thread", "unsubscribe_thread"),
            _operation("turn/start", "start_turn", "start_turn"),
            _operation("turn/interrupt", "interrupt_turn", "interrupt_turn"),
            _operation("turn/resume", "resume_turn", "resume_turn"),
            _operation("turn/steer", "steer_turn", "steer_turn"),
            _operation("turn/followUp", "follow_up_turn", "follow_up_turn"),
            _operation("approval/resolve", "resolve_approval_request", "resolve_approval_request"),
            _operation("model/list", "list_models", "list_models"),
            _operation("schema/export", "export_schema", "export_schema"),
            _operation(
                "initialized",
                "initialize",
                "initialize",
                adaptation="Both initialize facades send the initialized notification after the initialize response.",
            ),
        ],
        "supporting_operations": [
            _operation("resolve_server_request", "resolve_approval", "resolve_approval"),
            _operation("send_response", "send_response", "send_response"),
            _operation(
                "next_message",
                "next_message",
                "next_message",
                adaptation="Python accepts a timeout; Rust uses its configured response timeout.",
            ),
            _operation("close", "close", "close"),
        ],
    },
    {
        "id": "tool",
        "python_target": "vv_agent.FunctionTool",
        "rust_target": "vv_agent::Tool",
        "members": [
            _field("name", "name", "name", rust_kind="method"),
            _field("description", "description", "description", rust_kind="method"),
            _field(
                "parameters_schema",
                "params_json_schema",
                "parameters_schema",
                adaptation="The field follows each ecosystem's schema naming.",
                rust_kind="method",
            ),
            _field(
                "strict_schema",
                "strict_json_schema",
                "strict_schema",
                rust_kind="method",
            ),
            _field("exposure", "exposure", "exposure", rust_kind="method"),
            _field(
                "timeout",
                "timeout_seconds",
                "timeout",
                adaptation="Python stores seconds; Rust stores Duration.",
                rust_kind="method",
            ),
            _field(
                "approval_rule",
                "needs_approval",
                "approval_rule",
                adaptation="Python accepts bool/callable; Rust exposes ToolApprovalRule.",
                rust_kind="method",
            ),
            _field("is_enabled", "is_enabled", "is_enabled", rust_kind="method"),
            _field(
                "metadata",
                "metadata",
                "metadata",
                adaptation="Rust exposes tool metadata through ToolExecutor.",
                rust_kind="method",
                rust_target="vv_agent::ToolExecutor",
            ),
            _field("idempotency", "idempotency", "idempotency", rust_kind="method"),
            _operation(
                "to_spec",
                "to_openai_schema",
                "as_tool_spec",
                adaptation="Python returns the OpenAI schema directly; Rust returns ToolSpec containing it.",
            ),
            _operation(
                "invoke",
                "invoke",
                "run",
                adaptation="Rust executes through ToolExecutor after adapting Tool to ToolSpec.",
                rust_target="vv_agent::ToolExecutor",
            ),
        ],
    },
    {
        "id": "tool_registry",
        "python_target": "vv_agent.ToolRegistry",
        "rust_target": "vv_agent::ToolRegistry",
        "members": [
            _operation("register", "register", "register"),
            _operation("register_many", "register_many", "register_many"),
            _operation("get", "get", "get"),
            _operation("has_tool", "has_tool", "has_tool"),
            _operation("register_schema", "register_schema", "register_schema"),
            _operation("register_schemas", "register_schemas", "register_schemas"),
            _operation("get_schema", "get_schema", "get_schema"),
            _operation("list_openai_schemas", "list_openai_schemas", "list_openai_schemas"),
            _operation("execute", "execute", "execute"),
            _operation(
                "executors",
                "list_tool_names",
                "executors",
                adaptation="Python exposes ordered names plus get_executor; Rust exposes ordered executor objects.",
            ),
            _operation(
                "register_executor",
                "register_executor",
                "register",
                adaptation="Python can register an executor directly; Rust registers its ToolSpec adaptation.",
            ),
        ],
    },
    {
        "id": "workspace_backend",
        "python_target": "vv_agent.workspace.WorkspaceBackend",
        "rust_target": "vv_agent::WorkspaceBackend",
        "members": [
            _operation("list_files", "list_files", "list_files"),
            _operation("read_text", "read_text", "read_text"),
            _operation("read_bytes", "read_bytes", "read_bytes"),
            _operation("write_text", "write_text", "write_text"),
            _operation("file_info", "file_info", "file_info"),
            _operation("exists", "exists", "exists"),
            _operation("is_file", "is_file", "is_file"),
            _operation("mkdir", "mkdir", "mkdir"),
        ],
    },
    {
        "id": "memory_provider",
        "python_target": "vv_agent.memory.MemoryProvider",
        "rust_target": "vv_agent::MemoryProvider",
        "members": [
            _operation("search", "search", "search"),
            _operation("save", "save", "save"),
            _operation("before_compact", "before_compact", "before_compact"),
            _operation("after_compact", "after_compact", "after_compact"),
        ],
    },
    {
        "id": "memory_manager",
        "python_target": "vv_agent.memory.MemoryManager",
        "rust_target": "vv_agent::MemoryManager",
        "members": [
            _operation("compact", "compact", "compact"),
            _operation("emergency_compact", "emergency_compact", "emergency_compact"),
            _field(
                "effective_context_window",
                "effective_context_window",
                "effective_context_window",
                python_kind="property",
                rust_kind="method",
            ),
            _operation(
                "estimate_memory_usage_percentage",
                "estimate_memory_usage_percentage",
                "estimate_memory_usage_percentage",
            ),
            _operation("microcompact_messages", "microcompact_messages", "microcompact_messages"),
            _operation(
                "apply_session_memory_context",
                "apply_session_memory_context",
                "apply_session_memory_context",
            ),
            _operation(
                "strip_session_memory_context",
                "strip_session_memory_context",
                "strip_session_memory_context",
            ),
        ],
    },
    {
        "id": "session_memory",
        "python_target": "vv_agent.memory.SessionMemory",
        "rust_target": "vv_agent::SessionMemory",
        "members": [
            _operation("should_extract", "should_extract", "should_extract"),
            _operation("extract", "extract", "extract"),
            _operation("render_as_system_context", "render_as_system_context", "render_as_system_context"),
            _operation("on_compaction", "on_compaction", "on_compaction"),
            _operation("load", "load", "load"),
        ],
    },
    {
        "id": "skills",
        "python_target": "vv_agent.skills",
        "rust_target": "vv_agent::skills",
        "members": [
            _operation("discover_skill_dirs", "discover_skill_dirs", "discover_skill_dirs", python_kind="function"),
            _operation("read_properties", "read_properties", "read_properties", python_kind="function"),
            _operation("read_skill", "read_skill", "read_skill", python_kind="function"),
            _operation("normalize_skill_list", "normalize_skill_list", "normalize_skill_list", python_kind="function"),
            _operation("render_skills_xml", "render_skills_xml", "render_skills_xml", python_kind="function"),
            _operation("validate", "validate", "validate", python_kind="function"),
            _operation(
                "validate_with_diagnostics",
                "validate_with_diagnostics",
                "validate_with_diagnostics",
                python_kind="function",
            ),
        ],
    },
    {
        "id": "tracing_span",
        "python_target": "vv_agent.Span",
        "rust_target": "vv_agent::Span",
        "members": [
            _field("name", "name", "name"),
            _field("trace_id", "trace_id", "trace_id"),
            _field("span_id", "span_id", "span_id"),
            _field("parent_id", "parent_id", "parent_id"),
            _field("started_at", "started_at", "started_at"),
            _field("ended_at", "ended_at", "ended_at"),
            _field("metadata", "metadata", "metadata"),
            _operation("finish", "finish", "finish"),
        ],
    },
    {
        "id": "tracing_sink",
        "python_target": "vv_agent.TraceSink",
        "rust_target": "vv_agent::TraceSink",
        "members": [
            _operation("on_span_start", "on_span_start", "on_span_start"),
            _operation("on_span_end", "on_span_end", "on_span_end"),
            _operation("flush", "flush", "flush"),
        ],
    },
    {
        "id": "llm_request",
        "python_target": "vv_agent.llm.LlmRequest",
        "rust_target": "vv_agent::LlmRequest",
        "members": [
            _field("model", "model", "model"),
            _field("messages", "messages", "messages"),
            _field("tools", "tools", "tools"),
            _field("metadata", "metadata", "metadata"),
            _field("model_settings", "model_settings", "model_settings"),
        ],
    },
    {
        "id": "llm_client",
        "python_target": "vv_agent.llm.LLMClient",
        "rust_target": "vv_agent::LlmClient",
        "members": [
            _operation("complete", "complete", "complete"),
            _operation(
                "complete_with_stream",
                "complete",
                "complete_with_stream",
                adaptation="Python passes stream_callback to complete; Rust exposes a separate defaulted method.",
            ),
        ],
    },
    {
        "id": "host_cost_meter",
        "python_target": "vv_agent.HostCostMeter",
        "rust_target": "vv_agent::HostCostMeter",
        "members": [_operation("read", "read", "read")],
    },
    {
        "id": "model_provider",
        "python_target": "vv_agent.ModelProvider",
        "rust_target": "vv_agent::ModelProvider",
        "members": [
            _operation("resolve", "resolve", "resolve"),
            _operation("client", "client", "client"),
            _operation("default_settings", "default_settings", "default_settings"),
            _operation("default_model_ref", "default_model_ref", "default_model_ref"),
        ],
    },
    {
        "id": "runtime_backend",
        "python_target": "vv_agent.runtime.backends.ExecutionBackend",
        "rust_target": "vv_agent::RuntimeExecutionBackend",
        "members": [
            _operation("execute", "execute", "execute"),
            _operation("parallel_map", "parallel_map", "parallel_map"),
        ],
    },
)


PROMPT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "en-US-full",
        "normalizations": ["computer_os"],
        "producer": "build_system_prompt_bundle",
        "input": {
            "agent_type": "computer",
            "allow_interruption": True,
            "available_skills": [
                {
                    "allowed_tools": "read_file search_files",
                    "compatibility": "vv-agent >= 1",
                    "description": "Review code against the requested contract.",
                    "location": "skills/review-code/SKILL.md",
                    "name": "review-code",
                }
            ],
            "available_sub_agents": {
                "researcher": "Finds source evidence.",
                "writer": "Writes the final report.",
            },
            "current_time_utc": "2026-07-10T08:30:00Z",
            "enable_todo_management": True,
            "language": "en-US",
            "original_system_prompt": "You are a careful coding agent.",
            "session_memory_context": "<Session Memory>\n- Keep source evidence.\n</Session Memory>",
            "use_workspace": True,
        },
    },
    {
        "id": "zh-CN-full",
        "normalizations": ["computer_os"],
        "producer": "build_system_prompt_bundle",
        "input": {
            "agent_type": "computer",
            "allow_interruption": True,
            "available_skills": [
                {
                    "allowed_tools": "read_file search_files",
                    "compatibility": "vv-agent >= 1",
                    "description": "核对代码与契约。",
                    "location": "skills/review-code/SKILL.md",
                    "name": "review-code",
                }
            ],
            "available_sub_agents": {
                "researcher": "查找源码证据。",
                "writer": "整理最终报告。",
            },
            "current_time_utc": "2026-07-10T08:30:00Z",
            "enable_todo_management": True,
            "language": "zh-CN",
            "original_system_prompt": "你是一个严谨的编码 Agent。",
            "session_memory_context": "<Session Memory>\n- 保留源码证据。\n</Session Memory>",
            "use_workspace": True,
        },
    },
    {
        "id": "en-US-minimal",
        "normalizations": [],
        "producer": "build_system_prompt_bundle",
        "input": {
            "agent_type": None,
            "allow_interruption": False,
            "available_skills": [],
            "available_sub_agents": {},
            "current_time_utc": "2026-07-10T08:30:00Z",
            "enable_todo_management": False,
            "language": "en-US",
            "original_system_prompt": "Return only verified facts.",
            "session_memory_context": "",
            "use_workspace": False,
        },
    },
    {
        "id": "custom-section-metadata",
        "normalizations": [],
        "producer": "SystemPromptBuilder",
        "input": {
            "sections": [
                {
                    "cache_hint": "ephemeral",
                    "id": "policy",
                    "metadata": {"owner": "parity", "priority": 1},
                    "source": "contract://policy",
                    "stable": True,
                    "text": "Use verified evidence.",
                },
                {
                    "id": "runtime",
                    "source": "runtime://request",
                    "stable": False,
                    "text": "request_id=fixture-1",
                },
                {
                    "cache_hint": "ignored-empty",
                    "id": "empty",
                    "stable": True,
                    "text": "   ",
                },
            ]
        },
    },
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _resolve_python_target(path: str) -> Any:
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as error:
        if error.name != path:
            raise
    return _resolve_python_export(path)


def _signature_projection(value: Any) -> dict[str, Any]:
    signature = inspect.signature(value)
    return {
        "async": inspect.iscoroutinefunction(value),
        "parameters": [
            {
                "kind": parameter.kind.name.lower(),
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
            }
            for parameter in signature.parameters.values()
        ],
    }


def _field_declaration(target: Any, name: str) -> str:
    if is_dataclass(target) and name in {field.name for field in fields(target)}:
        return "dataclass_field"
    if inspect.isclass(target):
        for base in target.__mro__:
            if name in getattr(base, "__annotations__", {}):
                return "annotation"
    raise AssertionError(f"{target!r}.{name} is not a declared public field")


def _project_python_member(default_target: Any, member: dict[str, Any]) -> None:
    python = member["python"]
    target = _resolve_python_target(python["target"]) if "target" in python else default_target
    name = str(python["name"])
    kind = str(python["kind"])
    if kind in {"method", "function"}:
        value = getattr(target, name)
        assert callable(value), f"{target!r}.{name} is not callable"
        python["signature"] = _signature_projection(value)
        return
    if kind == "property":
        value = inspect.getattr_static(target, name)
        assert isinstance(value, property), f"{target!r}.{name} is not a property"
        assert value.fget is not None
        python["declaration"] = "property"
        python["signature"] = _signature_projection(value.fget)
        return
    if kind == "field":
        python["declaration"] = _field_declaration(target, name)
        return
    raise AssertionError(f"unsupported Python member kind: {kind}")


def _project_public_api_surfaces() -> list[dict[str, Any]]:
    surfaces = list(deepcopy(PUBLIC_API_SURFACES))
    surface_ids: set[str] = set()
    for surface in surfaces:
        assert surface["id"] not in surface_ids
        surface_ids.add(surface["id"])
        target = _resolve_python_target(str(surface["python_target"]))
        member_ids: set[str] = set()
        for group in ("members", "protocol_operations", "supporting_operations"):
            for member in surface.get(group, []):
                assert member["id"] not in member_ids, f"duplicate {surface['id']} member: {member['id']}"
                member_ids.add(str(member["id"]))
                _project_python_member(target, member)
        assert member_ids, f"empty public API surface: {surface['id']}"
    return surfaces


def _build_public_api_manifest() -> dict[str, Any]:
    return {
        "contract": "vv-agent-public-api-v1",
        "domains": list(deepcopy(PUBLIC_API_DOMAINS)),
        "schema_version": 1,
        "surfaces": _project_public_api_surfaces(),
        "verification": {
            "python": "public export resolution plus dataclass field/property/getattr/callable/inspect.signature checks",
            "rust": "compiled exports and exhaustive method/function references plus typed public-field accessors",
        },
    }


def _stable_hash(sections: list[dict[str, Any]]) -> str:
    stable_text = "".join(str(section["text"]) for section in sections if section.get("stable") is True)
    return hashlib.sha256(stable_text.encode()).hexdigest()


def _normalize_computer_os(value: str) -> str:
    labels = {"Windows", "macOS", "Linux", platform.system() or "Unknown OS"}
    for label in labels:
        value = value.replace(label, "<OS>")
    return value


def _project_prompt_output(bundle: Any, normalizations: list[str]) -> dict[str, Any]:
    sections = deepcopy(bundle.sections)
    assert bundle.stable_hash == _stable_hash(sections)
    prompt = str(bundle.prompt)
    if "computer_os" in normalizations:
        prompt = _normalize_computer_os(prompt)
        for section in sections:
            section["text"] = _normalize_computer_os(str(section["text"]))
    return {"prompt": prompt, "sections": sections, "stable_hash": _stable_hash(sections)}


def _render_prompt_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    inputs = scenario["input"]
    producer = scenario["producer"]
    if producer == "build_system_prompt_bundle":
        current_time = datetime.fromisoformat(str(inputs["current_time_utc"]).replace("Z", "+00:00"))
        bundle = build_system_prompt_bundle(
            str(inputs["original_system_prompt"]),
            language=str(inputs["language"]),
            allow_interruption=bool(inputs["allow_interruption"]),
            use_workspace=bool(inputs["use_workspace"]),
            enable_todo_management=bool(inputs["enable_todo_management"]),
            agent_type=inputs["agent_type"],
            available_sub_agents=deepcopy(inputs["available_sub_agents"]),
            available_skills=deepcopy(inputs["available_skills"]),
            current_time_utc=current_time,
            session_memory_context=str(inputs["session_memory_context"]),
        )
    elif producer == "SystemPromptBuilder":
        builder = SystemPromptBuilder()
        for raw in inputs["sections"]:
            text = str(raw["text"])
            builder.add_section(
                PromptSection(
                    id=str(raw["id"]),
                    compute=lambda text=text: text,
                    stable=bool(raw["stable"]),
                    source=str(raw.get("source", "")),
                    cache_hint=raw.get("cache_hint"),
                    metadata=deepcopy(raw.get("metadata", {})),
                )
            )
        bundle = builder.build_result()
    else:
        raise AssertionError(f"unknown prompt producer: {producer}")
    return _project_prompt_output(bundle, list(scenario["normalizations"]))


def _build_prompt_bundle_manifest() -> dict[str, Any]:
    scenarios = []
    for source in PROMPT_SCENARIOS:
        scenario = deepcopy(source)
        scenario["output"] = _render_prompt_scenario(scenario)
        scenarios.append(scenario)
    return {
        "contract": "vv-agent-system-prompt-v1",
        "normalization_rules": {
            "computer_os": "Replace the host OS label in rendered environment text with <OS> and recompute the stable hash."
        },
        "scenarios": scenarios,
        "schema_version": 1,
    }


def _approval_name(needs_approval: Any) -> str:
    if callable(needs_approval):
        return "dynamic"
    return "required" if bool(needs_approval) else "not_required"


def _build_builtin_tools_manifest() -> dict[str, Any]:
    registry = build_default_registry()
    tools = []
    for name in registry.list_tool_names():
        executor = registry.get_executor(name)
        schema = executor.openai_schema(None)
        function = schema["function"]
        assert function["name"] == name
        assert function["description"] == executor.description
        tools.append(
            {
                "approval": _approval_name(executor.needs_approval),
                "description": executor.description,
                "exposure": executor.exposure.value,
                "kind": "function",
                "metadata": deepcopy(executor.metadata),
                "model_visible": executor.exposure != ToolExposure.HIDDEN,
                "name": name,
                "parameters": deepcopy(function["parameters"]),
                "strict": executor.strict_json_schema,
                "timeout_seconds": executor.timeout_seconds,
                "type": schema["type"],
            }
        )
    return {"contract": "vv-agent-builtin-tools-v1", "schema_version": 1, "tools": tools}


def _fixture_payloads() -> dict[str, dict[str, Any]]:
    return {
        "builtin_tools_v1.json": _build_builtin_tools_manifest(),
        "prompt_bundle_v1.json": _build_prompt_bundle_manifest(),
        "public_api_v1.json": _build_public_api_manifest(),
    }


def _load_fixture(name: str) -> Any:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _resolve_python_export(path: str) -> Any:
    parts = path.split(".")
    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            continue
        attributes = parts[split_index:]
        public_names = getattr(module, "__all__", None)
        if public_names is not None:
            assert attributes[0] in public_names, (
                f"{path} exists but is absent from {module_name}.__all__"
            )
        value: Any = module
        for attribute in attributes:
            value = getattr(value, attribute)
        return value
    raise ModuleNotFoundError(f"No importable module prefix for {path}")


def _verify_fixture_python_member(surface: dict[str, Any], member: dict[str, Any]) -> None:
    python = member["python"]
    target = _resolve_python_target(str(python.get("target", surface["python_target"])))
    name = str(python["name"])
    kind = str(python["kind"])
    if kind in {"method", "function"}:
        value = getattr(target, name)
        assert callable(value), f"{target!r}.{name} is not callable"
        assert python["signature"] == _signature_projection(value)
        return
    if kind == "property":
        value = inspect.getattr_static(target, name)
        assert isinstance(value, property), f"{target!r}.{name} is not a property"
        assert value.fget is not None
        assert python["declaration"] == "property"
        assert python["signature"] == _signature_projection(value.fget)
        return
    if kind == "field":
        assert python["declaration"] == _field_declaration(target, name)
        return
    raise AssertionError(f"unsupported fixture Python member kind: {kind}")


def test_public_api_manifest_resolves_real_python_exports() -> None:
    fixture = _load_fixture("public_api_v1.json")
    assert fixture == _build_public_api_manifest()
    assert tuple(domain["id"] for domain in fixture["domains"]) == EXPECTED_DOMAINS

    capability_ids: set[str] = set()
    for domain in fixture["domains"]:
        assert domain["capabilities"], domain["id"]
        for capability in domain["capabilities"]:
            assert capability["id"] not in capability_ids
            capability_ids.add(capability["id"])
            assert _resolve_python_export(capability["python"]) is not None
    assert len(capability_ids) == 147

    surfaces = {surface["id"]: surface for surface in fixture["surfaces"]}
    assert len(surfaces) == len(fixture["surfaces"])
    assert (
        sum(
            len(surface.get(group, []))
            for surface in fixture["surfaces"]
            for group in ("members", "protocol_operations", "supporting_operations")
        )
        == 229
    )
    assert tuple(member["id"] for member in surfaces["runner"]["members"]) == EXPECTED_RUNNER_OPERATIONS
    assert tuple(member["id"] for member in surfaces["run_handle"]["members"]) == EXPECTED_RUN_HANDLE_OPERATIONS
    assert (
        tuple(member["id"] for member in surfaces["app_server_client"]["protocol_operations"])
        == EXPECTED_APP_SERVER_PROTOCOL_OPERATIONS
    )

    for surface in surfaces.values():
        for group in ("members", "protocol_operations", "supporting_operations"):
            for member in surface.get(group, []):
                _verify_fixture_python_member(surface, member)


def test_prompt_bundle_manifest_uses_real_prompt_producers() -> None:
    fixture = _load_fixture("prompt_bundle_v1.json")
    assert fixture == _build_prompt_bundle_manifest()
    assert {scenario["producer"] for scenario in fixture["scenarios"]} == {
        "SystemPromptBuilder",
        "build_system_prompt_bundle",
    }


def test_builtin_tools_manifest_uses_real_default_registry() -> None:
    fixture = _load_fixture("builtin_tools_v1.json")
    assert fixture == _build_builtin_tools_manifest()
    assert len(fixture["tools"]) == 16
    assert all(tool["model_visible"] for tool in fixture["tools"])


def test_evidence_json_is_canonical_utf8() -> None:
    for name in CANONICAL_FIXTURES:
        path = FIXTURE_DIR / name
        raw = path.read_bytes()
        assert raw == _canonical_json_bytes(json.loads(raw.decode("utf-8"))), name


def test_sha256sums_covers_every_parity_fixture() -> None:
    checksum_path = FIXTURE_DIR / "SHA256SUMS"
    entries: dict[str, str] = {}
    listed_names: list[str] = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        assert name not in entries
        entries[name] = digest
        listed_names.append(name)

    fixture_names = sorted(path.name for path in FIXTURE_DIR.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    assert listed_names == fixture_names
    for name in fixture_names:
        assert hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest() == entries[name]
