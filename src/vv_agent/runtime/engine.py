from __future__ import annotations

import ast
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from vv_agent.config import ResolvedModelConfig, build_openai_llm_from_local_settings
from vv_agent.constants import BATCH_SUB_TASKS_TOOL_NAME, CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.llm.base import LLMClient
from vv_agent.memory import MemoryManager
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime.backends.base import ExecutionBackend
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.runtime.hooks import RuntimeHook, RuntimeHookManager
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.runtime.tool_call_runner import ToolCallRunner
from vv_agent.tools import ToolContext, ToolRegistry
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CycleRecord,
    Message,
    SubAgentConfig,
    SubTaskOutcome,
    SubTaskRequest,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
)
from vv_agent.workspace import LocalWorkspaceBackend, WorkspaceBackend

RuntimeLogHandler = Callable[[str, dict[str, Any]], None]
BeforeCycleMessageProvider = Callable[[int, list[Message], dict[str, Any]], list[Message]]
InterruptionMessageProvider = Callable[[], list[Message]]


class LLMBuilder(Protocol):
    def __call__(
        self,
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[LLMClient, ResolvedModelConfig]:
        ...


class AgentRuntime:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        default_workspace: str | Path | None = None,
        log_handler: RuntimeLogHandler | None = None,
        log_preview_chars: int = 220,
        settings_file: str | Path | None = None,
        default_backend: str | None = None,
        llm_builder: LLMBuilder | None = None,
        tool_registry_factory: Callable[[], ToolRegistry] | None = None,
        sub_agent_timeout_seconds: float = 90.0,
        hooks: list[RuntimeHook] | None = None,
        execution_backend: ExecutionBackend | None = None,
        workspace_backend: WorkspaceBackend | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.default_workspace = Path(default_workspace).resolve() if default_workspace else None
        self.log_handler = log_handler
        self.log_preview_chars = max(log_preview_chars, 40)
        self.settings_file = Path(settings_file).resolve() if settings_file else None
        self.default_backend = default_backend
        self.llm_builder = llm_builder or build_openai_llm_from_local_settings
        self.tool_registry_factory = tool_registry_factory
        self.sub_agent_timeout_seconds = max(sub_agent_timeout_seconds, 1.0)
        self.hook_manager = RuntimeHookManager(hooks=list(hooks or []))
        self.execution_backend: ExecutionBackend = execution_backend or InlineBackend()
        self._workspace_backend = workspace_backend
        self._memory_summary_clients: dict[tuple[str, str], LLMClient] = {}
        self._memory_summary_defaults: tuple[str | None, str | None] | None = None
        self.cycle_runner = CycleRunner(
            llm_client=llm_client,
            tool_registry=tool_registry,
            hook_manager=self.hook_manager,
        )
        self.tool_call_runner = ToolCallRunner(
            tool_registry=tool_registry,
            hook_manager=self.hook_manager,
        )

    def run(
        self,
        task: AgentTask,
        *,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        initial_messages: list[Message] | None = None,
        user_message: str | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        ctx: ExecutionContext | None = None,
    ) -> AgentResult:
        workspace_path = self._prepare_workspace(workspace)
        shared = dict(shared_state or {})
        shared.setdefault("todo_list", [])
        if isinstance(task.metadata, dict):
            if "available_skills" not in shared and task.metadata.get("available_skills") is not None:
                shared["available_skills"] = task.metadata.get("available_skills")
            if "available_skills" not in shared and task.metadata.get("skill_directories") is not None:
                shared["available_skills"] = task.metadata.get("skill_directories")
            if "bound_skills" not in shared and task.metadata.get("bound_skills") is not None:
                shared["bound_skills"] = task.metadata.get("bound_skills")
            if "active_skills" not in shared and task.metadata.get("active_skills") is not None:
                shared["active_skills"] = list(task.metadata.get("active_skills") or [])

        messages = self._build_initial_messages(
            task=task,
            initial_messages=initial_messages,
            user_message=user_message,
        )
        self._emit_log(
            "run_started",
            task_id=task.task_id,
            model=task.model,
            workspace=str(workspace_path),
            max_cycles=task.max_cycles,
        )

        memory_manager = self._build_memory_manager(task=task, workspace_path=workspace_path)

        cycle_executor = self._build_cycle_executor(
            task=task,
            workspace_path=workspace_path,
            workspace_backend=self._workspace_backend or LocalWorkspaceBackend(workspace_path),
            memory_manager=memory_manager,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
        )

        return self.execution_backend.execute(
            task=task,
            initial_messages=messages,
            shared_state=shared,
            cycle_executor=cycle_executor,
            ctx=ctx,
            max_cycles=task.max_cycles,
        )

    def _build_cycle_executor(
        self,
        *,
        task: AgentTask,
        workspace_path: Path,
        workspace_backend: WorkspaceBackend,
        memory_manager: MemoryManager,
        before_cycle_messages: BeforeCycleMessageProvider | None,
        interruption_messages: InterruptionMessageProvider | None,
    ) -> Callable[[int, list[Message], list[CycleRecord], dict[str, Any], ExecutionContext | None], AgentResult | None]:
        def executor(
            cycle_index: int,
            messages: list[Message],
            cycles: list[CycleRecord],
            shared: dict[str, Any],
            ctx: ExecutionContext | None,
        ) -> AgentResult | None:
            if before_cycle_messages is not None:
                injected = before_cycle_messages(cycle_index, messages, shared)
                if injected:
                    messages.extend(injected)
                    self._emit_log(
                        "cycle_injected_messages",
                        cycle=cycle_index,
                        count=len(injected),
                    )
            self._emit_log(
                "cycle_started",
                cycle=cycle_index,
                max_cycles=task.max_cycles,
                message_count=len(messages),
            )
            previous_total_tokens: int | None = None
            recent_tool_call_ids: set[str] | None = None
            if cycles:
                last_cycle = cycles[-1]
                usage = last_cycle.token_usage
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                if total_tokens <= 0:
                    total_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) + int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                if total_tokens > 0:
                    previous_total_tokens = total_tokens

                candidate_ids = {call.id for call in last_cycle.tool_calls if getattr(call, "id", "")}
                if candidate_ids:
                    recent_tool_call_ids = candidate_ids
            try:
                updated_messages, cycle_record = self.cycle_runner.run_cycle(
                    task=task,
                    messages=messages,
                    cycle_index=cycle_index,
                    memory_manager=memory_manager,
                    previous_total_tokens=previous_total_tokens,
                    recent_tool_call_ids=recent_tool_call_ids,
                    shared_state=shared,
                    ctx=ctx,
                )
            except Exception as exc:
                self._emit_log("cycle_failed", cycle=cycle_index, error=str(exc))
                return AgentResult(
                    status=AgentStatus.FAILED,
                    messages=messages,
                    cycles=cycles,
                    error=f"LLM call failed in cycle {cycle_index}: {exc}",
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                )
            # Replace messages list contents in-place so caller sees updates
            messages.clear()
            messages.extend(updated_messages)

            self._emit_log(
                "cycle_llm_response",
                cycle=cycle_index,
                assistant_preview=self._preview_text(cycle_record.assistant_message),
                tool_calls=[call.name for call in cycle_record.tool_calls],
                tool_call_count=len(cycle_record.tool_calls),
                token_usage=cycle_record.token_usage.to_dict(),
            )

            if cycle_record.tool_calls:
                context = ToolContext(
                    workspace=workspace_path,
                    shared_state=shared,
                    cycle_index=cycle_index,
                    workspace_backend=workspace_backend,
                    sub_task_runner=self._build_sub_task_runner(
                        parent_task=task,
                        workspace_path=workspace_path,
                        parent_shared_state=shared,
                        ctx=ctx,
                    ),
                    ctx=ctx,
                )

                def _on_tool_result(call: ToolCall, result: ToolExecutionResult, *, _cycle: int = cycle_index) -> None:
                    self._emit_log(
                        "tool_result",
                        cycle=_cycle,
                        tool_name=call.name,
                        tool_call_id=result.tool_call_id,
                        status=result.status_code.value if result.status_code else result.status,
                        directive=result.directive.value,
                        error_code=result.error_code,
                        content_preview=self._preview_text(result.content),
                    )

                tool_outcome = self.tool_call_runner.run(
                    task=task,
                    tool_calls=cycle_record.tool_calls,
                    context=context,
                    messages=messages,
                    cycle_record=cycle_record,
                    interruption_provider=interruption_messages,
                    on_tool_result=_on_tool_result,
                    ctx=ctx,
                )
                tool_result = tool_outcome.directive_result
                cycles.append(cycle_record)
                if tool_outcome.interruption_messages:
                    messages.extend(tool_outcome.interruption_messages)
                    self._emit_log(
                        "run_steered",
                        cycle=cycle_index,
                        steering_count=len(tool_outcome.interruption_messages),
                    )

                if tool_result and tool_result.directive == ToolDirective.WAIT_USER:
                    wait_reason = tool_result.metadata.get("question") if isinstance(tool_result.metadata, dict) else None
                    if not wait_reason:
                        wait_reason = tool_result.content
                    self._emit_log(
                        "run_wait_user",
                        cycle=cycle_index,
                        wait_reason=self._preview_text(str(wait_reason)),
                    )
                    return AgentResult(
                        status=AgentStatus.WAIT_USER,
                        messages=messages,
                        cycles=cycles,
                        wait_reason=str(wait_reason),
                        shared_state=shared,
                        token_usage=summarize_task_token_usage(cycles),
                    )

                if tool_result and tool_result.directive == ToolDirective.FINISH:
                    final_answer = self._extract_final_message(tool_result)
                    self._emit_log(
                        "run_completed",
                        cycle=cycle_index,
                        final_answer=self._preview_text(final_answer),
                    )
                    return AgentResult(
                        status=AgentStatus.COMPLETED,
                        messages=messages,
                        cycles=cycles,
                        final_answer=final_answer,
                        shared_state=shared,
                        token_usage=summarize_task_token_usage(cycles),
                    )

                return None  # continue to next cycle

            cycles.append(cycle_record)
            if task.no_tool_policy == "finish":
                self._emit_log(
                    "run_completed",
                    cycle=cycle_index,
                    final_answer=self._preview_text(cycle_record.assistant_message),
                )
                return AgentResult(
                    status=AgentStatus.COMPLETED,
                    messages=messages,
                    cycles=cycles,
                    final_answer=cycle_record.assistant_message,
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                )

            if task.no_tool_policy == "wait_user":
                self._emit_log(
                    "run_wait_user",
                    cycle=cycle_index,
                    wait_reason=self._preview_text(cycle_record.assistant_message or "No tool call"),
                )
                return AgentResult(
                    status=AgentStatus.WAIT_USER,
                    messages=messages,
                    cycles=cycles,
                    wait_reason=cycle_record.assistant_message or "No tool call and runtime is waiting for user.",
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                )

            if cycle_index < task.max_cycles:
                messages.append(Message(role="user", content=self._build_continue_hint()))

            return None  # continue to next cycle

        return executor

    def _emit_cycle_tool_results(self, *, cycle_record: CycleRecord) -> None:
        for idx, result in enumerate(cycle_record.tool_results):
            tool_name = None
            if idx < len(cycle_record.tool_calls):
                tool_name = cycle_record.tool_calls[idx].name
            self._emit_log(
                "tool_result",
                cycle=cycle_record.index,
                tool_name=tool_name or "unknown",
                tool_call_id=result.tool_call_id,
                status=result.status_code.value if result.status_code else result.status,
                directive=result.directive.value,
                error_code=result.error_code,
                content_preview=self._preview_text(result.content),
            )

    def _emit_log(self, event: str, **payload: Any) -> None:
        if self.log_handler is None:
            return
        self.log_handler(event, payload)

    @staticmethod
    def _copy_message(message: Message) -> Message:
        return Message(
            role=message.role,
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
            tool_calls=list(message.tool_calls) if message.tool_calls else None,
            reasoning_content=message.reasoning_content,
            image_url=message.image_url,
        )

    def _build_initial_messages(
        self,
        *,
        task: AgentTask,
        initial_messages: list[Message] | None,
        user_message: str | None,
    ) -> list[Message]:
        if initial_messages:
            prepared = [self._copy_message(message) for message in initial_messages]
            if not prepared or prepared[0].role != "system":
                prepared.insert(0, Message(role="system", content=task.system_prompt))
            message_to_append = task.user_prompt if user_message is None else user_message
            if message_to_append:
                prepared.append(Message(role="user", content=message_to_append))
            return prepared

        first_user_message = task.user_prompt if user_message is None else user_message
        return [
            Message(role="system", content=task.system_prompt),
            Message(role="user", content=first_user_message),
        ]

    def _build_memory_manager(self, *, task: AgentTask, workspace_path: Path) -> MemoryManager:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}

        def read_int(key: str, default: int, *, minimum: int = 0) -> int:
            raw = metadata.get(key, default)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = default
            return max(value, minimum)

        warning_threshold = max(1, min(task.memory_threshold_percentage, 100))
        local_summary_backend, local_summary_model = self._load_local_memory_summary_defaults()
        metadata_summary_backend = self._read_optional_str(
            metadata,
            "memory_summary_backend",
            "compress_memory_summary_backend",
            "memory_compress_backend",
        )
        metadata_summary_model = self._read_optional_str(
            metadata,
            "memory_summary_model",
            "compress_memory_summary_model",
            "memory_compress_model",
        )
        summary_backend = metadata_summary_backend or local_summary_backend or self.default_backend
        summary_model = metadata_summary_model or local_summary_model or task.model
        return MemoryManager(
            compact_threshold=max(task.memory_compact_threshold, 0),
            keep_recent_messages=read_int("memory_keep_recent_messages", 10, minimum=1),
            language=str(metadata.get("language", "zh-CN")),
            warning_threshold_percentage=warning_threshold,
            include_memory_warning=bool(metadata.get("include_memory_warning", False)),
            tool_result_compact_threshold=read_int("tool_result_compact_threshold", 2000),
            tool_result_keep_last=read_int("tool_result_keep_last", 3),
            tool_result_excerpt_head=read_int("tool_result_excerpt_head", 200),
            tool_result_excerpt_tail=read_int("tool_result_excerpt_tail", 200),
            tool_calls_keep_last=read_int("tool_calls_keep_last", 3),
            assistant_no_tool_keep_last=read_int("assistant_no_tool_keep_last", 1),
            tool_result_artifact_dir=str(metadata.get("tool_result_artifact_dir", ".memory/tool_results")),
            workspace=workspace_path if task.use_workspace else None,
            summary_event_limit=read_int("summary_event_limit", 40, minimum=1),
            summary_backend=summary_backend,
            summary_model=summary_model,
            summary_callback=self._summarize_memory_prompt,
        )

    @staticmethod
    def _read_optional_str(metadata: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            raw = metadata.get(key)
            if isinstance(raw, str):
                value = raw.strip()
                if value:
                    return value
        return None

    def _load_local_memory_summary_defaults(self) -> tuple[str | None, str | None]:
        if self._memory_summary_defaults is not None:
            return self._memory_summary_defaults

        backend: str | None = None
        model: str | None = None
        settings_file = self.settings_file
        if settings_file is None or not settings_file.exists():
            self._memory_summary_defaults = (None, None)
            return self._memory_summary_defaults

        try:
            module = ast.parse(settings_file.read_text(encoding="utf-8"), filename=str(settings_file))
            backend = self._read_literal_setting(
                module,
                "DEFAULT_USER_MEMORY_SUMMARIZE_BACKEND",
                "DEFAULT_MEMORY_SUMMARIZE_BACKEND",
                "VV_AGENT_MEMORY_SUMMARY_BACKEND",
            )
            model = self._read_literal_setting(
                module,
                "DEFAULT_USER_MEMORY_SUMMARIZE_MODEL",
                "DEFAULT_MEMORY_SUMMARIZE_MODEL",
                "VV_AGENT_MEMORY_SUMMARY_MODEL",
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to load memory summary defaults from settings file",
                exc_info=True,
            )

        self._memory_summary_defaults = (backend, model)
        return self._memory_summary_defaults

    @staticmethod
    def _read_literal_setting(module: ast.Module, *names: str) -> str | None:
        if not names:
            return None
        name_set = set(names)

        for node in module.body:
            target_name: str | None = None
            value_node: ast.expr | None = None

            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                value_node = node.value
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_name = target.id
                        value_node = node.value
                        break

            if target_name not in name_set or value_node is None:
                continue

            try:
                literal = ast.literal_eval(value_node)
            except Exception:
                continue
            if isinstance(literal, str):
                value = literal.strip()
                if value:
                    return value
        return None

    def _summarize_memory_prompt(self, prompt: str, backend: str | None, model: str | None) -> str | None:
        backend_name = (backend or self.default_backend or "").strip()
        model_name = (model or "").strip()
        if not backend_name or not model_name:
            return None
        if self.settings_file is None:
            return None

        cache_key = (backend_name, model_name)
        client = self._memory_summary_clients.get(cache_key)
        if client is None:
            client, _ = self.llm_builder(
                self.settings_file,
                backend=backend_name,
                model=model_name,
                timeout_seconds=self.sub_agent_timeout_seconds,
            )
            self._memory_summary_clients[cache_key] = client

        response = client.complete(
            model=model_name,
            messages=[Message(role="user", content=prompt)],
            tools=[],
        )
        content = (response.content or "").strip()
        return content or None

    def _preview_text(self, text: str) -> str:
        cleaned = text.replace("\n", " ").strip()
        if len(cleaned) <= self.log_preview_chars:
            return cleaned
        return f"{cleaned[: self.log_preview_chars - 3]}..."

    def _prepare_workspace(self, workspace: str | Path | None) -> Path:
        target = Path(workspace) if workspace else self.default_workspace
        if target is None:
            target = Path.cwd() / ".vv-agent-workspace"
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _build_continue_hint() -> str:
        return (
            "No tool call was produced. "
            f"Continue the task and call `{TASK_FINISH_TOOL_NAME}` "
            "when all todo items are done."
        )

    @staticmethod
    def _extract_final_message(result: ToolExecutionResult) -> str:
        if isinstance(result.metadata, dict):
            final = result.metadata.get("final_message")
            if isinstance(final, str) and final:
                return final

        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, str) and message:
                return message

        return result.content

    def _build_sub_task_runner(
        self,
        *,
        parent_task: AgentTask,
        workspace_path: Path,
        parent_shared_state: dict[str, Any],
        ctx: ExecutionContext | None = None,
    ) -> Callable[[SubTaskRequest], SubTaskOutcome] | None:
        if not parent_task.sub_agents:
            return None

        def runner(request: SubTaskRequest) -> SubTaskOutcome:
            return self._run_sub_task(
                parent_task=parent_task,
                workspace_path=workspace_path,
                parent_shared_state=parent_shared_state,
                request=request,
                ctx=ctx,
            )

        return runner

    def _run_sub_task(
        self,
        *,
        parent_task: AgentTask,
        workspace_path: Path,
        parent_shared_state: dict[str, Any],
        request: SubTaskRequest,
        ctx: ExecutionContext | None = None,
    ) -> SubTaskOutcome:
        sub_task_id = f"{parent_task.task_id}_sub_{request.agent_name}_{uuid.uuid4().hex[:8]}"
        sub_agent = parent_task.sub_agents.get(request.agent_name)
        if sub_agent is None:
            available = ", ".join(sorted(parent_task.sub_agents))
            return SubTaskOutcome(
                task_id=sub_task_id,
                agent_name=request.agent_name,
                status=AgentStatus.FAILED,
                error=f"Unknown sub-agent {request.agent_name!r}. Available: {available}",
            )

        try:
            llm_client, model_id, resolved = self._resolve_sub_agent_client(
                parent_task=parent_task,
                sub_agent=sub_agent,
            )
            sub_task = self._build_sub_agent_task(
                parent_task=parent_task,
                sub_task_id=sub_task_id,
                sub_agent_name=request.agent_name,
                sub_agent=sub_agent,
                resolved_model_id=model_id,
                request=request,
                parent_shared_state=parent_shared_state,
                workspace_path=workspace_path,
            )
            sub_runtime = AgentRuntime(
                llm_client=llm_client,
                tool_registry=self._build_sub_agent_registry(),
                default_workspace=workspace_path,
                log_handler=self._build_sub_agent_log_handler(request.agent_name),
                log_preview_chars=self.log_preview_chars,
                settings_file=self.settings_file,
                default_backend=self.default_backend,
                llm_builder=self.llm_builder,
                tool_registry_factory=self.tool_registry_factory,
                sub_agent_timeout_seconds=self.sub_agent_timeout_seconds,
            )
            sub_result = sub_runtime.run(
                sub_task,
                workspace=workspace_path,
                shared_state={"todo_list": []},
                ctx=self._build_child_ctx(ctx),
            )
        except Exception as exc:
            return SubTaskOutcome(
                task_id=sub_task_id,
                agent_name=request.agent_name,
                status=AgentStatus.FAILED,
                error=str(exc),
            )

        return SubTaskOutcome(
            task_id=sub_task_id,
            agent_name=request.agent_name,
            status=sub_result.status,
            final_answer=sub_result.final_answer,
            wait_reason=sub_result.wait_reason,
            error=sub_result.error,
            cycles=len(sub_result.cycles),
            todo_list=sub_result.todo_list,
            resolved=resolved,
        )

    def _resolve_sub_agent_client(
        self,
        *,
        parent_task: AgentTask,
        sub_agent: SubAgentConfig,
    ) -> tuple[LLMClient, str, dict[str, str]]:
        if self.settings_file is None:
            if sub_agent.model != parent_task.model:
                raise ValueError(
                    "Sub-agent model resolution requires runtime settings_file when sub-agent model differs from parent model."
                )
            return self.llm_client, parent_task.model, {}

        backend = sub_agent.backend or self.default_backend
        if not backend:
            raise ValueError("Sub-agent backend is required when settings_file is configured.")

        llm_client, resolved = self.llm_builder(
            self.settings_file,
            backend=backend,
            model=sub_agent.model,
            timeout_seconds=self.sub_agent_timeout_seconds,
        )
        return llm_client, resolved.model_id, {
            "backend": resolved.backend,
            "selected_model": resolved.selected_model,
            "model_id": resolved.model_id,
            "endpoint": resolved.endpoint.endpoint_id,
        }

    def _build_sub_agent_task(
        self,
        *,
        parent_task: AgentTask,
        sub_task_id: str,
        sub_agent_name: str,
        sub_agent: SubAgentConfig,
        resolved_model_id: str,
        request: SubTaskRequest,
        parent_shared_state: dict[str, Any],
        workspace_path: Path,
    ) -> AgentTask:
        language = str(parent_task.metadata.get("language", "zh-CN"))
        available_skills = parent_task.metadata.get("available_skills")
        if not isinstance(available_skills, list):
            available_skills = parent_task.metadata.get("bound_skills")
        if not isinstance(available_skills, list):
            available_skills = None

        system_prompt = sub_agent.system_prompt or build_system_prompt(
            sub_agent.description,
            language=language,
            allow_interruption=False,
            use_workspace=parent_task.use_workspace,
            enable_todo_management=True,
            agent_type=parent_task.agent_type,
            available_skills=available_skills,
            workspace=workspace_path,
        )

        user_prompt = request.task_description
        if request.output_requirements:
            user_prompt = (
                f"{user_prompt}\n\n<Output Requirements>\n{request.output_requirements}\n</Output Requirements>"
            )
        if request.include_main_summary:
            parent_summary = self._build_parent_summary(parent_task=parent_task, parent_shared_state=parent_shared_state)
            if parent_summary:
                user_prompt = (
                    f"{user_prompt}\n\n<Main Task Summary>\n{parent_summary}\n</Main Task Summary>"
                )

        excluded_tools = set(parent_task.exclude_tools)
        excluded_tools.update(sub_agent.exclude_tools)
        excluded_tools.update({CREATE_SUB_TASK_TOOL_NAME, BATCH_SUB_TASKS_TOOL_NAME})
        metadata = {
            "is_sub_task": True,
            "parent_task_id": parent_task.task_id,
            "sub_agent_name": sub_agent_name,
        }
        if request.metadata:
            metadata.update(request.metadata)

        return AgentTask(
            task_id=sub_task_id,
            model=resolved_model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_cycles=max(sub_agent.max_cycles, 1),
            memory_compact_threshold=parent_task.memory_compact_threshold,
            memory_threshold_percentage=parent_task.memory_threshold_percentage,
            no_tool_policy="continue",
            allow_interruption=False,
            use_workspace=parent_task.use_workspace,
            has_sub_agents=False,
            sub_agents={},
            agent_type=parent_task.agent_type,
            native_multimodal=parent_task.native_multimodal,
            extra_tool_names=list(parent_task.extra_tool_names),
            exclude_tools=sorted(excluded_tools),
            metadata=metadata,
        )

    def _build_sub_agent_registry(self) -> ToolRegistry:
        if self.tool_registry_factory is not None:
            return self.tool_registry_factory()
        return self.tool_registry

    @staticmethod
    def _build_child_ctx(ctx: ExecutionContext | None) -> ExecutionContext | None:
        if ctx is None:
            return None
        child_token = ctx.cancellation_token.child() if ctx.cancellation_token is not None else None
        return ExecutionContext(
            cancellation_token=child_token,
            stream_callback=ctx.stream_callback,
            state_store=ctx.state_store,
            metadata=dict(ctx.metadata),
        )

    def _build_sub_agent_log_handler(self, sub_agent_name: str) -> RuntimeLogHandler | None:
        if self.log_handler is None:
            return None
        parent_handler = self.log_handler

        def handler(event: str, payload: dict[str, Any]) -> None:
            enriched = dict(payload)
            enriched["sub_agent_name"] = sub_agent_name
            parent_handler(f"sub_agent_{event}", enriched)

        return handler

    @staticmethod
    def _build_parent_summary(*, parent_task: AgentTask, parent_shared_state: dict[str, Any]) -> str:
        lines = [f"Parent task goal: {parent_task.user_prompt}"]
        todo_list = parent_shared_state.get("todo_list")
        if isinstance(todo_list, list) and todo_list:
            lines.append("Parent TODO status:")
            for item in todo_list:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "Untitled"))
                status = str(item.get("status", "pending"))
                lines.append(f"- [{status}] {title}")
        return "\n".join(lines)
