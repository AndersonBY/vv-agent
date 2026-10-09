from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.background_task import BackgroundAgentTask
from vv_agent.config import ResolvedModelConfig
from vv_agent.events import (
    RunEvent,
)
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm.base import LLMClient
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings
from vv_agent.output_validation import (
    OUTPUT_VALIDATION_FAILED,
    OutputValidationContext,
    OutputValidationResult,
)
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig, ToolPolicy, _validate_bounded_int, merge_tool_policy_layers
from vv_agent.run_handle import RunHandle
from vv_agent.tools import ToolContext
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolSideEffect
from vv_agent.tracing import TraceProcessor
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
    Message,
    _last_assistant_output,
)

_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_tool_policy_approval",
    "_vv_agent_tool_policy_can_use_tool",
    "_vv_agent_denied_side_effects",
    "_vv_agent_denied_capability_tags",
    "_vv_agent_deny_terminal_tools",
    "_vv_agent_denied_cost_dimensions",
)
_TASK_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_denied_side_effects",
    "_vv_agent_denied_capability_tags",
    "_vv_agent_deny_terminal_tools",
    "_vv_agent_denied_cost_dimensions",
)
_PLANNED_TOOL_NAMES_METADATA_KEY = "_vv_agent_planned_tool_names"
_INITIAL_BUDGET_USAGE_METADATA_KEY = "_vv_agent_initial_budget_usage"


def _tool_policy_metadata(policy: ToolPolicy | None) -> dict[str, Any]:
    if policy is None:
        return {}
    metadata: dict[str, Any] = {}
    if policy.allowed_tools is not None:
        metadata["_vv_agent_allowed_tools"] = list(policy.allowed_tools)
    if policy.disallowed_tools:
        metadata["_vv_agent_disallowed_tools"] = list(policy.disallowed_tools)
    if policy.can_use_tool is not None:
        metadata["_vv_agent_tool_policy_can_use_tool"] = policy.can_use_tool
    if policy.approval != "default":
        metadata["_vv_agent_tool_policy_approval"] = policy.approval
    if policy.denied_side_effects:
        metadata["_vv_agent_denied_side_effects"] = [ToolSideEffect(item).value for item in policy.denied_side_effects]
    if policy.denied_capability_tags:
        metadata["_vv_agent_denied_capability_tags"] = list(policy.denied_capability_tags)
    if policy.deny_terminal_tools:
        metadata["_vv_agent_deny_terminal_tools"] = True
    if policy.denied_cost_dimensions:
        metadata["_vv_agent_denied_cost_dimensions"] = list(policy.denied_cost_dimensions)
    return metadata


class Runner:
    @classmethod
    def configured(cls, default_run_config: RunConfig | None = None) -> ConfiguredRunner:
        return ConfiguredRunner(default_run_config=default_run_config or RunConfig())

    @classmethod
    def run_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return cls.start(agent, input, run_config=run_config).result()

    @classmethod
    def stream_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        handle = cls.start(agent, input, run_config=run_config)
        yield from handle.events()
        handle.result()

    @classmethod
    def start(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        from vv_agent.session.surfaces import SessionDriver

        config = cls._effective_run_config(agent, run_config or RunConfig())
        driver = SessionDriver()
        sid = uuid.uuid4().hex
        seed = {"messages": [m.to_dict() for m in config.initial_messages or []], "shared_state": config.shared_state or {}}
        driver.create(sid, str(config.workspace or Path.cwd()), {"seed": seed})
        return driver.start(sid, agent, config, input, input_id="initial")

    @classmethod
    def _run_compiled_sync(cls, agent: Agent, input: str, *, task: AgentTask, run_config: RunConfig | None = None) -> RunResult:
        return cls._start_compiled(agent, input, task=task, run_config=run_config).result()

    @classmethod
    def _start_compiled(cls, agent: Agent, input: str, *, task: AgentTask, run_config: RunConfig | None = None) -> RunHandle:
        from vv_agent.session.surfaces import SessionDriver

        config = cls._effective_run_config(agent, run_config or RunConfig())
        driver = SessionDriver()
        sid = uuid.uuid4().hex
        driver.create(sid, str(config.workspace or Path.cwd()))
        return driver.start(sid, agent, config, input, task=task)

    @classmethod
    def resume(cls, session_id: str, turn_id: str) -> RunResult:
        from vv_agent.session.surfaces import resume_turn

        return resume_turn(session_id, turn_id)

    @staticmethod
    def _effective_run_config(
        agent: Agent,
        run_config: RunConfig | None,
        *,
        runner_defaults: RunConfig | None = None,
    ) -> RunConfig:
        defaults = runner_defaults or RunConfig()
        config = run_config or RunConfig()

        provider_overridden = config.model_provider is not None
        model = config.model
        if model is None:
            model = agent.model
        if model is None and not provider_overridden:
            model = defaults.model

        configured_max_cycles = next(
            (value for value in (config.max_cycles, defaults.max_cycles, agent.max_cycles) if value is not None),
            10,
        )
        configured_max_handoffs = next(
            (value for value in (config.max_handoffs, defaults.max_handoffs) if value is not None),
            10,
        )
        configured_no_tool_policy = next(
            (value for value in (config.no_tool_policy, defaults.no_tool_policy, agent.no_tool_policy) if value is not None),
            "finish",
        )
        effective_max_cycles = _validate_bounded_int(configured_max_cycles, "max_cycles", minimum=1)
        effective_max_handoffs = _validate_bounded_int(configured_max_handoffs, "max_handoffs", minimum=0)
        assert effective_max_cycles is not None
        assert effective_max_handoffs is not None
        model_settings = (
            ModelSettings().resolve(defaults.model_settings).resolve(agent.model_settings).resolve(config.model_settings)
        )
        shared_state = None
        if defaults.shared_state is not None or config.shared_state is not None:
            shared_state = {**(defaults.shared_state or {}), **(config.shared_state or {})}

        def prefer_run(name: str) -> Any:
            value = getattr(config, name)
            return value if value is not None else getattr(defaults, name)

        return replace(
            config,
            model=model,
            model_provider=config.model_provider or defaults.model_provider,
            model_settings=model_settings,
            workspace=prefer_run("workspace"),
            workspace_backend=prefer_run("workspace_backend"),
            max_cycles=effective_max_cycles,
            max_handoffs=effective_max_handoffs,
            no_tool_policy=configured_no_tool_policy,
            tool_policy=merge_tool_policy_layers(agent.tool_policy, defaults.tool_policy, config.tool_policy),
            cancellation_token=prefer_run("cancellation_token"),
            approval_provider=prefer_run("approval_provider"),
            approval_timeout_seconds=prefer_run("approval_timeout_seconds"),
            approval_broker=prefer_run("approval_broker"),
            event_store=prefer_run("event_store"),
            event_store_fail_closed=defaults.event_store_fail_closed or config.event_store_fail_closed,
            stream=prefer_run("stream"),
            hooks=[*defaults.hooks, *config.hooks],
            after_cycle_hooks=[
                *defaults.after_cycle_hooks,
                *config.after_cycle_hooks,
            ],
            tracing=prefer_run("tracing"),
            context=prefer_run("context"),
            context_providers=[*defaults.context_providers, *config.context_providers],
            max_context_chars=prefer_run("max_context_chars"),
            memory_providers=[*defaults.memory_providers, *config.memory_providers],
            metadata={**defaults.metadata, **config.metadata},
            tool_registry_factory=prefer_run("tool_registry_factory"),
            log_preview_chars=prefer_run("log_preview_chars"),
            debug_dump_dir=prefer_run("debug_dump_dir"),
            shared_state=shared_state,
            initial_messages=prefer_run("initial_messages"),
            before_cycle_messages=prefer_run("before_cycle_messages"),
            interruption_messages=prefer_run("interruption_messages"),
            budget_limits=prefer_run("budget_limits"),
            host_cost_meter=prefer_run("host_cost_meter"),
        )

    @staticmethod
    def _apply_input_guardrails(*, agent: Agent, run_context: RunContext[Any], user_input: str) -> GuardrailResult:
        current_input = user_input
        for guardrail in agent.input_guardrails:
            result = guardrail(run_context, current_input)
            if result.outcome == "rewrite":
                current_input = str(result.value)
                continue
            if result.outcome != "allow":
                return result
        if current_input != user_input:
            return GuardrailResult.rewrite(current_input)
        return GuardrailResult.allow()

    @staticmethod
    def _apply_output_guardrails(*, agent: Agent, run_context: RunContext[Any], final_output: Any) -> GuardrailResult:
        current_output = final_output
        for guardrail in agent.output_guardrails:
            result = guardrail(run_context, current_output)
            if result.outcome == "rewrite":
                current_output = result.value
                continue
            if result.outcome != "allow":
                return result
        if current_output != final_output:
            return GuardrailResult.rewrite(current_output)
        return GuardrailResult.allow()

    @classmethod
    def _postprocess_output(
        cls,
        *,
        agent: Agent,
        run_context: RunContext[Any],
        raw_result: AgentResult,
        final_output: Any,
        cancellation_token: Any | None,
    ) -> tuple[Any, Exception | None]:
        cls._normalize_completion_observation(raw_result, cancellation_token=cancellation_token)
        if raw_result.completion_reason in {CompletionReason.CANCELLED, CompletionReason.BUDGET_EXHAUSTED}:
            return final_output, None
        output_result = cls._apply_output_guardrails(
            agent=agent,
            run_context=run_context,
            final_output=final_output,
        )
        if output_result.outcome == "rewrite":
            final_output = output_result.value
            cls._replace_raw_result_output(raw_result, final_output)
        elif output_result.outcome in {"block", "require_approval"}:
            final_output = output_result.message or "Output blocked by guardrail."
            raw_result.status = AgentStatus.FAILED
            raw_result.completion_reason = CompletionReason.FAILED
            raw_result.completion_tool_name = None
            raw_result.partial_output = raw_result.partial_output or _last_assistant_output(raw_result.cycles)
            raw_result.final_answer = None
            raw_result.wait_reason = None
            raw_result.error = {
                "code": "agent_failed",
                "message": str(final_output),
                "retryable": False,
            }

        cls._normalize_completion_observation(raw_result, cancellation_token=cancellation_token)
        output_coercion_error: Exception | None = None
        if raw_result.status == AgentStatus.COMPLETED:
            try:
                final_output = cls._coerce_output_type(agent=agent, final_output=final_output)
            except Exception as exc:
                output_coercion_error = ValueError(f"failed to validate final output: {exc}")
        return final_output, output_coercion_error

    @classmethod
    def _run_output_validator(
        cls,
        *,
        agent: Agent,
        validator: Callable[[Any, OutputValidationContext], OutputValidationResult],
        validation_context: OutputValidationContext,
        candidate: Any,
        output_coercion_error: Exception | None,
        coerce: bool,
    ) -> tuple[Any, OutputValidationResult, bool]:
        if output_coercion_error is not None:
            return candidate, OutputValidationResult.reject("output_type_invalid", str(output_coercion_error)), True
        if coerce:
            try:
                candidate = cls._coerce_output_type(agent=agent, final_output=candidate)
            except Exception as exc:
                return candidate, OutputValidationResult.reject("output_type_invalid", str(exc)), True
        try:
            result = validator(candidate, validation_context)
        except Exception as exc:
            return candidate, OutputValidationResult.reject("output_validator_error", str(exc)), False
        if not isinstance(result, OutputValidationResult):
            return (
                candidate,
                OutputValidationResult.reject(
                    "output_validator_contract_invalid",
                    "output_validator must return OutputValidationResult",
                ),
                False,
            )
        return candidate, result, True

    @staticmethod
    def _output_validation_error(result: OutputValidationResult) -> str:
        detail = cast(str, result.code)
        if result.message:
            detail = f"{detail}: {result.message}"
        return f"{OUTPUT_VALIDATION_FAILED}: {detail}"

    @staticmethod
    def _replace_raw_result_output(result: AgentResult, output: Any) -> None:
        value = output if isinstance(output, str) or output is None else json.dumps(output, ensure_ascii=False)
        if result.status == AgentStatus.COMPLETED:
            result.final_answer = value
        elif result.status == AgentStatus.WAIT_USER:
            result.wait_reason = value
        else:
            result.error = {
                "code": result.error_code or "agent_failed",
                "message": value or "operation failed",
                "retryable": False,
            }

    @staticmethod
    def _normalize_completion_observation(
        result: AgentResult,
        *,
        cancellation_token: Any | None,
    ) -> None:
        cancelled = bool(result.status == AgentStatus.FAILED and cancellation_token is not None and cancellation_token.cancelled)
        if cancelled:
            result.completion_reason = CompletionReason.CANCELLED
            result.completion_tool_name = None
        elif result.status == AgentStatus.WAIT_USER:
            result.completion_reason = result.completion_reason or CompletionReason.WAIT_USER
        elif result.status == AgentStatus.MAX_CYCLES:
            result.completion_reason = CompletionReason.MAX_CYCLES
            result.completion_tool_name = None
        elif result.status == AgentStatus.FAILED:
            if result.completion_reason != CompletionReason.BUDGET_EXHAUSTED:
                result.completion_reason = CompletionReason.FAILED
            result.completion_tool_name = None

        if result.status == AgentStatus.COMPLETED:
            result.partial_output = None
        else:
            result.partial_output = result.partial_output or _last_assistant_output(result.cycles)

    @staticmethod
    def _coerce_output_type(*, agent: Agent, final_output: Any) -> Any:
        output_type = agent.output_type
        if output_type is None or final_output is None:
            return final_output
        if output_type is str:
            return str(final_output)

        payload = final_output
        if isinstance(final_output, str):
            payload = json.loads(final_output)

        if output_type is dict:
            if not isinstance(payload, dict):
                raise ValueError("Expected final output JSON object for output_type=dict.")
            return payload
        if output_type is list:
            if not isinstance(payload, list):
                raise ValueError("Expected final output JSON array for output_type=list.")
            return payload
        if isinstance(output_type, type) and is_dataclass(output_type):
            if not isinstance(payload, dict):
                raise ValueError("Expected final output JSON object for dataclass output_type.")
            field_names = {item.name for item in fields(output_type)}
            return output_type(**{key: value for key, value in payload.items() if key in field_names})

        model_validate = getattr(output_type, "model_validate", None)
        if callable(model_validate):
            return model_validate(payload)
        return final_output

    @classmethod
    def _resolve_model(cls, *, agent: Agent, run_config: RunConfig) -> tuple[LLMClient, ResolvedModelConfig]:
        provider = run_config.model_provider
        if provider is None:
            raise ValueError("RunConfig.model_provider is required.")
        model = run_config.model or agent.model
        if model is None:
            model = provider.default_model_ref()
        if model is None:
            raise ValueError("Agent.model, RunConfig.model, or ModelProvider.default_model_ref() is required.")
        resolved = provider.resolve(ModelRef.coerce(model))
        return provider.client(resolved), resolved

    @staticmethod
    def _tool_run_config_from_context(*, context: ToolContext, fallback: RunConfig) -> RunConfig:
        runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
        task_metadata = context.task_metadata if isinstance(context.task_metadata, dict) else {}
        is_sub_task = task_metadata.get("is_sub_task") is True
        policy_keys_present = any(key in runtime_metadata for key in _TOOL_POLICY_METADATA_KEYS)

        if is_sub_task or policy_keys_present:
            allowed = runtime_metadata.get("_vv_agent_allowed_tools")
            disallowed = runtime_metadata.get("_vv_agent_disallowed_tools")
            can_use_tool = runtime_metadata.get("_vv_agent_tool_policy_can_use_tool")
            approval = runtime_metadata.get("_vv_agent_tool_policy_approval")
            denied_side_effects = runtime_metadata.get("_vv_agent_denied_side_effects")
            denied_capability_tags = runtime_metadata.get("_vv_agent_denied_capability_tags")
            deny_terminal_tools = runtime_metadata.get("_vv_agent_deny_terminal_tools")
            denied_cost_dimensions = runtime_metadata.get("_vv_agent_denied_cost_dimensions")
            tool_policy = None
            if policy_keys_present:
                tool_policy = ToolPolicy(
                    allowed_tools=([name for name in allowed if isinstance(name, str)] if isinstance(allowed, list) else None),
                    disallowed_tools=(
                        [name for name in disallowed if isinstance(name, str)] if isinstance(disallowed, list) else []
                    ),
                    approval=(approval if approval in {"always", "never", "on_request"} else "default"),
                    can_use_tool=can_use_tool if callable(can_use_tool) else None,
                    denied_side_effects=(denied_side_effects if isinstance(denied_side_effects, list) else []),
                    denied_capability_tags=(denied_capability_tags if isinstance(denied_capability_tags, list) else []),
                    deny_terminal_tools=(deny_terminal_tools if isinstance(deny_terminal_tools, bool) else False),
                    denied_cost_dimensions=(denied_cost_dimensions if isinstance(denied_cost_dimensions, list) else []),
                )
        else:
            tool_policy = fallback.tool_policy

        def runtime_capability(key: str, fallback_value: Any) -> Any:
            return runtime_metadata.get(key, fallback_value)

        return replace(
            fallback,
            tool_policy=tool_policy,
            approval_provider=runtime_capability(
                "_vv_agent_approval_provider",
                fallback.approval_provider,
            ),
            approval_broker=runtime_capability(
                "_vv_agent_approval_broker",
                fallback.approval_broker,
            ),
            approval_timeout_seconds=runtime_capability(
                "_vv_agent_approval_timeout_seconds",
                fallback.approval_timeout_seconds,
            ),
        )

    @staticmethod
    def _tool_is_enabled(*, tool: FunctionTool, agent: Agent, run_config: RunConfig) -> bool:
        if callable(tool.is_enabled):
            run_context = RunContext(
                context=run_config.context,
                metadata={**agent.metadata, **run_config.metadata},
            )
            return bool(tool.is_enabled(run_context, agent))
        return bool(tool.is_enabled)

    @classmethod
    def _agent_tool_parent_config(cls, context: ToolContext | None) -> RunConfig | None:
        if context is None or context.ctx is None:
            return None
        runtime_metadata = context.ctx.metadata
        parent_config = runtime_metadata.get("_vv_agent_run_config")
        if not isinstance(parent_config, RunConfig):
            parent_config = BackgroundAgentTask._inherited_run_config(context, None)
        parent_config = cls._tool_run_config_from_context(context=context, fallback=parent_config)
        provider = runtime_metadata.get("_vv_agent_model_provider", parent_config.model_provider)
        if provider is None:
            return None
        return replace(
            parent_config,
            model_provider=provider,
            cancellation_token=context.ctx.cancellation_token,
            workspace=context.workspace,
            workspace_backend=context.workspace_backend,
            budget_limits=runtime_metadata.get("_vv_agent_budget_limits", parent_config.budget_limits),
            memory_providers=runtime_metadata.get("_vv_agent_memory_providers", parent_config.memory_providers),
            context=getattr(context.run_context, "context", parent_config.context),
        )

    @classmethod
    def _run_child_agent(
        cls,
        child_agent: Agent,
        *,
        arguments: dict[str, Any],
        parent_config: RunConfig,
        context: ToolContext | None = None,
    ) -> RunResult:
        child_input = cls._child_agent_prompt(arguments=arguments, context=context)
        return cls.run_sync(
            child_agent,
            child_input,
            run_config=cls._child_run_config(parent_config, context=context),
        )

    @staticmethod
    def _child_run_config(parent_config: RunConfig, *, context: ToolContext | None = None) -> RunConfig:
        cancellation_token = parent_config.cancellation_token
        if cancellation_token is not None:
            cancellation_token = cancellation_token.child()
        return replace(
            parent_config,
            model=None,
            model_settings=None,
            stream=None,
            shared_state=deepcopy(context.shared_state) if context is not None else None,
            initial_messages=None,
            before_cycle_messages=None,
            interruption_messages=None,
            cancellation_token=cancellation_token,
            host_cost_meter=None,
            metadata={key: value for key, value in parent_config.metadata.items() if key != _INITIAL_BUDGET_USAGE_METADATA_KEY},
        )

    @staticmethod
    def _child_agent_prompt(*, arguments: dict[str, Any], context: ToolContext | None) -> str:
        raw_task_description = arguments.get("task_description")
        task_description = raw_task_description.strip() if isinstance(raw_task_description, str) else ""
        if not task_description:
            raise ValueError("agent tool requires task_description")
        output_requirements = arguments.get("output_requirements")
        if isinstance(output_requirements, str) and output_requirements.strip():
            task_description += f"\n\n<Output Requirements>\n{output_requirements.strip()}\n</Output Requirements>"
        if arguments.get("include_main_summary") is True and context is not None:
            runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
            parent_summary = context.shared_state.get("main_task_summary")
            if not isinstance(parent_summary, str) or not parent_summary.strip():
                parent_summary = runtime_metadata.get("_vv_agent_input")
            if isinstance(parent_summary, str) and parent_summary.strip():
                task_description += f"\n\n<Main Task Summary>\n{parent_summary.strip()}\n</Main Task Summary>"
        return task_description

    @staticmethod
    def _new_session_items(*, initial_messages: list[Message] | None, result: AgentResult) -> list[Message]:
        history = list(initial_messages or [])
        result_messages = list(result.messages)
        prefix_length = len(history)
        if not history or history[0].role != "system":
            prefix_length += 1
        if prefix_length > len(result_messages):
            return []
        return deepcopy(result_messages[prefix_length:])

    @staticmethod
    def _trace_processors(run_config: RunConfig) -> list[TraceProcessor]:
        tracing = run_config.tracing or {}
        raw_processors = tracing.get("processors") if isinstance(tracing, dict) else None
        if not isinstance(raw_processors, list):
            return []
        processors: list[TraceProcessor] = []
        for processor in raw_processors:
            if callable(getattr(processor, "on_span_start", None)) and callable(getattr(processor, "on_span_end", None)):
                processors.append(processor)
        return processors


@dataclass(frozen=True, slots=True)
class ConfiguredRunner:
    default_run_config: RunConfig = field(default_factory=RunConfig)

    def run_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return self.start(agent, input, run_config=run_config).result()

    def stream_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        handle = self.start(agent, input, run_config=run_config)
        yield from handle.events()
        handle.result()

    def start(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        config = Runner._effective_run_config(agent, run_config or RunConfig(), runner_defaults=self.default_run_config)
        return Runner.start(agent, input, run_config=config)

    def resume(self, session_id: str, turn_id: str) -> RunResult:
        return Runner.resume(session_id, turn_id)
