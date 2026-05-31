from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.config import (
    EndpointConfig,
    EndpointOption,
    ResolvedModelConfig,
    build_openai_llm_from_local_settings,
    load_llm_settings_from_file,
)
from vv_agent.events import (
    HandoffEvent,
    RunEvent,
    RunFailedEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    event_from_runtime_log,
    event_from_stream_payload,
    new_trace_id,
)
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm.base import LLMClient
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.runtime.context import ExecutionContext
from vv_agent.tools import ToolContext, ToolSpec, build_default_registry
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.registry import ToolRegistry
from vv_agent.tracing import Span, TraceProcessor
from vv_agent.types import AgentResult, AgentStatus, Message, ToolDirective, ToolExecutionResult, ToolResultStatus


class Runner:
    @classmethod
    def run_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return cls._run(agent, input, run_config=run_config or RunConfig())

    @classmethod
    def stream_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        result = cls._run(agent, input, run_config=run_config or RunConfig())
        yield from result.events

    @classmethod
    def _run(cls, agent: Agent, input: str, *, run_config: RunConfig) -> RunResult:
        run_id = f"run_{uuid.uuid4().hex}"
        trace_id = cls._resolve_trace_id(run_config)
        collected_events: list[RunEvent] = []
        trace_processors = cls._trace_processors(run_config)
        run_span = cls._start_span(
            trace_processors,
            name="run",
            trace_id=trace_id,
            metadata={
                "agent_name": agent.name,
                "workflow_name": cls._workflow_name(run_config),
            },
        )
        tool_spans: dict[str, Span] = {}
        user_input = cls._normalize_input(input)

        def capture_event(event: RunEvent | None) -> None:
            if event is None:
                return
            collected_events.append(event)
            if isinstance(event, ToolStartedEvent):
                tool_spans[event.tool_call_id] = cls._start_span(
                    trace_processors,
                    name="tool",
                    trace_id=trace_id,
                    parent_id=run_span.span_id,
                    metadata={"tool_name": event.tool_name, "agent_name": agent.name},
                )
            elif isinstance(event, ToolFinishedEvent | HandoffEvent):
                span = tool_spans.pop(event.tool_call_id, None)
                if span is not None:
                    cls._end_span(trace_processors, span, metadata=event.to_dict())
            if run_config.stream is not None:
                run_config.stream(event)

        def log_handler(event: str, payload: dict[str, Any]) -> None:
            capture_event(
                event_from_runtime_log(
                    event,
                    payload,
                    run_id=run_id,
                    trace_id=trace_id,
                    agent_name=agent.name,
                    user_input=user_input,
                )
            )

        def stream_callback(payload: dict[str, Any]) -> None:
            capture_event(
                event_from_stream_payload(
                    payload,
                    run_id=run_id,
                    trace_id=trace_id,
                    agent_name=agent.name,
                )
            )

        guardrail_context = RunContext(context=run_config.context, metadata={**agent.metadata, **run_config.metadata})
        input_result = cls._apply_input_guardrails(agent=agent, run_context=guardrail_context, user_input=user_input)
        if input_result.outcome in {"block", "require_approval"}:
            message = input_result.message or "Input blocked by guardrail."
            failed_event = RunFailedEvent(run_id=run_id, trace_id=trace_id, error=message, agent_name=agent.name)
            collected_events.append(failed_event)
            raw_result = AgentResult(status=AgentStatus.FAILED, messages=[], cycles=[], error=message)
            ended_run_span = cls._end_span(trace_processors, run_span, metadata={"status": "failed", "error": message})
            return RunResult(
                input=user_input,
                new_items=[],
                final_output=message,
                status=AgentStatus.FAILED,
                raw_result=raw_result,
                events=collected_events,
                token_usage=raw_result.token_usage,
                trace_id=trace_id,
                run_id=run_id,
                metadata={"run_span": ended_run_span.to_dict()},
            )
        if input_result.outcome == "rewrite":
            user_input = str(input_result.value)

        llm_client, resolved = cls._resolve_model(agent=agent, run_config=run_config)
        resolved_model_settings = agent.model_settings.resolve(run_config.model_settings)
        if run_config.debug_dump_dir:
            cast(Any, llm_client).debug_dump_dir = run_config.debug_dump_dir

        registry = cls._build_tool_registry(agent=agent, run_config=run_config)
        runtime = AgentRuntime(
            llm_client=llm_client,
            tool_registry=registry,
            default_workspace=cls._resolve_workspace(run_config.workspace),
            log_handler=log_handler,
            log_preview_chars=run_config.log_preview_chars,
            execution_backend=run_config.execution_backend,
            hooks=list(run_config.runtime_hooks),
        )
        task = AgentCompiler().compile(
            agent=agent,
            input=user_input,
            run_config=run_config,
            resolved=resolved,
            trace_id=trace_id,
        )
        initial_messages = cls._session_initial_messages(run_config)
        ctx = ExecutionContext(
            cancellation_token=run_config.cancellation_token,
            stream_callback=stream_callback,
            metadata={
                "trace_id": trace_id,
                "_vv_agent_model_settings": resolved_model_settings,
                "_vv_agent_run_context": guardrail_context,
                "_vv_agent_session": run_config.session,
                **dict(run_config.metadata),
            },
        )

        raw_result = runtime.run(
            task,
            workspace=cls._resolve_workspace(run_config.workspace),
            initial_messages=initial_messages,
            user_message=user_input,
            ctx=ctx,
        )
        new_items = cls._new_session_items(user_input=user_input, result=raw_result)
        if run_config.session is not None and new_items:
            run_config.session.add_items(new_items)

        final_output = raw_result.final_answer or raw_result.wait_reason or raw_result.error
        output_result = cls._apply_output_guardrails(
            agent=agent,
            run_context=guardrail_context,
            final_output=final_output,
        )
        if output_result.outcome == "rewrite":
            final_output = str(output_result.value)
        elif output_result.outcome in {"block", "require_approval"}:
            final_output = output_result.message or "Output blocked by guardrail."
            raw_result.status = AgentStatus.FAILED
            raw_result.error = final_output
        final_output = cls._coerce_output_type(agent=agent, final_output=final_output)
        ended_run_span = cls._end_span(
            trace_processors,
            run_span,
            metadata={"status": raw_result.status.value, "final_output": final_output},
        )

        return RunResult(
            input=user_input,
            new_items=new_items,
            final_output=final_output,
            status=raw_result.status,
            raw_result=raw_result,
            events=collected_events,
            token_usage=raw_result.token_usage,
            trace_id=trace_id,
            run_id=run_id,
            metadata={"resolved_model": resolved.model_id, "backend": resolved.backend, "run_span": ended_run_span.to_dict()},
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
        if run_config.model_provider is not None:
            return run_config.model_provider(agent, run_config)
        model = run_config.model or agent.model
        if hasattr(model, "complete"):
            model_id = getattr(model, "model_id", "direct")
            return cast(LLMClient, model), cls._direct_resolved(str(model_id))
        if model is None:
            raise ValueError("Agent.model or RunConfig.model is required when no model_provider is configured.")
        backend = run_config.default_backend or cls._infer_backend(run_config.settings_file, str(model))
        return build_openai_llm_from_local_settings(
            run_config.settings_file,
            backend=backend,
            model=str(model),
            timeout_seconds=run_config.timeout_seconds,
        )

    @classmethod
    def _build_tool_registry(cls, *, agent: Agent, run_config: RunConfig) -> ToolRegistry:
        registry = (
            run_config.tool_registry_factory()
            if run_config.tool_registry_factory is not None
            else build_default_registry()
        )
        for tool in agent.tools:
            if not isinstance(tool, FunctionTool):
                continue
            if not cls._tool_is_enabled(tool=tool, agent=agent, run_config=run_config):
                continue
            registry.register_schema(tool.name, tool.to_openai_schema())

            def handler(context: ToolContext, arguments: dict[str, Any], *, _tool: FunctionTool = tool) -> ToolExecutionResult:
                result = cls._execute_function_tool(_tool, context=context, arguments=arguments, run_config=run_config)
                if agent.tool_use_behavior == "stop_on_first_tool" and result.directive == ToolDirective.CONTINUE:
                    result.directive = ToolDirective.FINISH
                return result

            registry.register(ToolSpec(name=tool.name, handler=handler))
        for transfer in agent.handoffs:
            if not transfer.tool_name:
                continue
            registry.register_schema(
                transfer.tool_name,
                {
                    "type": "function",
                    "function": {
                        "name": transfer.tool_name,
                        "description": transfer.description or f"Transfer control to {transfer.agent.name}.",
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                            "required": ["input"],
                            "additionalProperties": False,
                        },
                    },
                },
            )

            def handoff_handler(
                context: ToolContext,
                arguments: dict[str, Any],
                *,
                _target: Agent = transfer.agent,
            ) -> ToolExecutionResult:
                del context
                child = cls._run_child_agent(_target, arguments=arguments, parent_config=run_config)
                return ToolExecutionResult(
                    tool_call_id="",
                    content=child.final_output or "",
                    directive=ToolDirective.FINISH,
                    metadata={
                        "agent": _target.name,
                        "mode": "handoff",
                        "handoff_from": agent.name,
                        "handoff_to": _target.name,
                        "child_status": child.status.value,
                        "child_run_id": child.run_id,
                    },
                )

            registry.register(ToolSpec(name=transfer.tool_name, handler=handoff_handler))
        return registry

    @classmethod
    def _execute_function_tool(
        cls,
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult:
        approval_result = cls._approval_result(tool, context=context, arguments=arguments, run_config=run_config)
        if approval_result is not None:
            return approval_result
        denied_result = cls._policy_denial_result(tool, arguments=arguments, run_config=run_config)
        if denied_result is not None:
            return denied_result
        if tool.metadata.get("mode") in {"agent_as_tool", "background_task"} and isinstance(tool.metadata.get("agent"), Agent):
            child_agent = tool.metadata["agent"]
            child = cls._run_child_agent(child_agent, arguments=arguments, parent_config=run_config)
            return ToolExecutionResult(
                tool_call_id="",
                content=child.final_output or "",
                metadata={
                    "agent": child_agent.name,
                    "mode": tool.metadata.get("mode"),
                    "child_status": child.status.value,
                    "child_run_id": child.run_id,
                },
            )
        return tool.to_tool_execution_result(tool.on_invoke(context, arguments))

    @staticmethod
    def _tool_is_enabled(*, tool: FunctionTool, agent: Agent, run_config: RunConfig) -> bool:
        if callable(tool.is_enabled):
            run_context = RunContext(
                context=run_config.context,
                metadata={**agent.metadata, **run_config.metadata},
            )
            return bool(tool.is_enabled(run_context, agent))
        return bool(tool.is_enabled)

    @staticmethod
    def _policy_denial_result(
        tool: FunctionTool,
        *,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult | None:
        policy = run_config.tool_policy
        if policy is None:
            return None

        if tool.name in policy.disallowed_tools or (policy.allowed_tools is not None and tool.name not in policy.allowed_tools):
            allowed = False
        elif policy.can_use_tool is not None:
            allowed = bool(policy.can_use_tool(tool.name, dict(arguments)))
        else:
            allowed = True

        if allowed:
            return None

        message = f"Tool {tool.name} is not allowed for these arguments."
        return ToolExecutionResult(
            tool_call_id="",
            content=message,
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="tool_not_allowed",
            metadata={
                "mode": "permission_denied",
                "tool_name": tool.name,
                "arguments": dict(arguments),
                "message": message,
            },
        )

    @staticmethod
    def _approval_result(
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult | None:
        policy = run_config.tool_policy
        if policy is not None and policy.approval == "never":
            return None
        needs_approval = policy is not None and policy.approval == "always"
        if callable(tool.needs_approval):
            needs_approval = needs_approval or bool(tool.needs_approval(context, arguments))
        else:
            needs_approval = needs_approval or bool(tool.needs_approval)
        if not needs_approval:
            return None
        message = f"Approval required for tool {tool.name}."
        return ToolExecutionResult(
            tool_call_id="",
            content=message,
            status_code=ToolResultStatus.WAIT_RESPONSE,
            directive=ToolDirective.WAIT_USER,
            metadata={
                "mode": "approval_requested",
                "tool_name": tool.name,
                "arguments": dict(arguments),
                "message": message,
            },
        )

    @classmethod
    def _run_child_agent(cls, child_agent: Agent, *, arguments: dict[str, Any], parent_config: RunConfig) -> RunResult:
        child_input = str(arguments.get("input") or arguments.get("prompt") or "")
        child_config = replace(parent_config, session=None, stream=None)
        return cls.run_sync(child_agent, child_input, run_config=child_config)

    @staticmethod
    def _session_initial_messages(run_config: RunConfig) -> list[Message] | None:
        if run_config.session is None:
            return None
        items = run_config.session.get_items()
        return list(items) or None

    @staticmethod
    def _new_session_items(*, user_input: str, result: Any) -> list[Message]:
        items = [Message(role="user", content=user_input)]
        assistant_content = ""
        if getattr(result, "cycles", None):
            assistant_content = result.cycles[-1].assistant_message
        if assistant_content:
            items.append(Message(role="assistant", content=assistant_content))
        return items

    @staticmethod
    def _normalize_input(input: str) -> str:
        return str(input)

    @staticmethod
    def _resolve_workspace(workspace: Any | None) -> Path | None:
        if workspace is None or hasattr(workspace, "read_text"):
            return None
        return Path(workspace).expanduser().resolve()

    @staticmethod
    def _resolve_trace_id(run_config: RunConfig) -> str:
        tracing = run_config.tracing or {}
        candidate = tracing.get("trace_id") if isinstance(tracing, dict) else None
        return str(candidate) if candidate else new_trace_id()

    @staticmethod
    def _workflow_name(run_config: RunConfig) -> str | None:
        tracing = run_config.tracing or {}
        candidate = tracing.get("workflow_name") if isinstance(tracing, dict) else None
        return str(candidate) if candidate else None

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

    @staticmethod
    def _start_span(
        processors: list[TraceProcessor],
        *,
        name: str,
        trace_id: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        span = Span(name=name, trace_id=trace_id, parent_id=parent_id, metadata=dict(metadata or {}))
        for processor in processors:
            processor.on_span_start(span)
        return span

    @staticmethod
    def _end_span(processors: list[TraceProcessor], span: Span, *, metadata: dict[str, Any] | None = None) -> Span:
        ended = span.finish(metadata=metadata)
        for processor in processors:
            processor.on_span_end(ended)
        return ended

    @staticmethod
    def _infer_backend(settings_file: str | Path, model: str) -> str:
        settings = load_llm_settings_from_file(settings_file)
        providers = settings.get("providers")
        if not isinstance(providers, dict):
            providers = settings.get("backends")
        matches: list[str] = []
        if isinstance(providers, dict):
            for backend, config in providers.items():
                models = config.get("models") if isinstance(config, dict) else None
                if isinstance(models, dict) and model in models:
                    matches.append(str(backend))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Cannot infer backend for model {model!r}; set RunConfig.default_backend.")
        raise ValueError(f"Model {model!r} exists in multiple backends {matches}; set RunConfig.default_backend.")

    @staticmethod
    def _direct_resolved(model: str) -> ResolvedModelConfig:
        endpoint = EndpointConfig(endpoint_id="direct", api_key="", api_base="")
        return ResolvedModelConfig(
            backend="direct",
            requested_model=model,
            selected_model=model,
            model_id=model,
            endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
        )
