from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vv_agent.memory.session_memory import SessionMemoryConfig
from vv_agent.memory.token_utils import resolve_model_token_limits
from vv_agent.model_settings import ModelSettings
from vv_agent.types import AgentTask
from vv_agent.workspace.local import LocalWorkspaceBackend

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.tools.registry import ToolRegistry

import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

from vv_agent.kernel import _summary_compaction
from vv_agent.kernel._compaction_text import build_compacted_tool_content, has_recovery_envelope
from vv_agent.kernel._compaction_threshold import compute_compaction_threshold
from vv_agent.kernel._context_messages import atomic_start, filter_empty_assistant_messages
from vv_agent.kernel._context_messages import retained_start as retained_start
from vv_agent.kernel._message_values import Message, ToolArtifactRef
from vv_agent.kernel._microcompaction import (
    EXCERPT_METADATA_KEY,
    MicrocompactCandidate,
    MicrocompactPlan,
    estimate_tool_artifact_path,
    plan_memory_microcompact,
    replace_with_compacted_marker,
)
from vv_agent.kernel._summary_compaction import SummaryPlan as SummaryPlan
from vv_agent.kernel._token_count import TokenCount
from vv_agent.kernel._tool_values import ToolExecutionResult as ToolExecutionResult
from vv_agent.memory._counting import _drain_counts
from vv_agent.memory.session_memory import SessionMemory
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.workspace.artifacts import persist_text_artifact
from vv_agent.workspace.base import WorkspaceBackend

_MEMORY_WARNING_PROMPTS = {
    "zh-CN": (
        "当前记忆已使用容量超过 {memory_threshold_percentage}%,"
        "建议立即整理、记录对话中的关键信息、资料, 并储存至工作区, 避免记忆压缩后资料丢失。\n\n"
    ),
    "en-US": (
        "The current memory usage has exceeded {memory_threshold_percentage}%. "
        "It is recommended to immediately organize and record key information and materials "
        "from the conversation, and store them in the workspace to prevent data loss after "
        "memory compression.\n\n"
    ),
}


SummaryCallback = Callable[[str, str | None, str | None], str | None]
CompactionMode = Literal["none", "micro", "structural", "summary", "emergency"]
_COMPACTION_MODE_RANK: dict[CompactionMode, int] = {
    "none": 0,
    "micro": 1,
    "structural": 2,
    "summary": 3,
    "emergency": 4,
}


def _strongest_compaction_mode(*modes: CompactionMode) -> CompactionMode:
    return max(modes, key=_COMPACTION_MODE_RANK.__getitem__, default="none")


ReservedOutputSource = Literal[
    "model_settings",
    "task_metadata",
    "framework_fallback",
    "framework_fallback_capped_by_model_capability",
]


@dataclass(frozen=True, slots=True)
class MemoryCompactionResult:
    messages: list[Message]
    mode: CompactionMode
    changed: bool
    archived_count: int = 0
    reclaimed_tokens: int = 0
    artifact_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class MicrocompactionApplyResult:
    messages: list[Message]
    archived_count: int = 0
    reclaimed_tokens: int = 0
    artifact_failure_count: int = 0


@dataclass(slots=True)
class MemoryManager:
    compact_threshold: int = 250_000
    keep_recent_messages: int = 10
    model: str = ""
    model_context_window: int = 279_000
    model_max_output_tokens: int | None = None
    reserved_output_tokens: int = 16_000
    reserved_output_source: ReservedOutputSource = "framework_fallback"
    autocompact_buffer_tokens: int = 13_000
    language: str = "zh-CN"
    warning_threshold_percentage: int = 90
    include_memory_warning: bool = False
    tool_result_excerpt_head: int = 200
    tool_result_excerpt_tail: int = 200
    microcompaction_policy: MicrocompactionPolicy = field(default_factory=MicrocompactionPolicy)
    tool_result_retentions: dict[str, ToolResultRetention] = field(default_factory=dict)
    workspace_backend: WorkspaceBackend | None = None
    recovery_tool_available: bool = False
    artifact_scope: str | tuple[str, str] = "run"
    summary_event_limit: int = 40
    summary_backend: str | None = None
    summary_model: str | None = None
    summary_callback: SummaryCallback | None = None
    base_system_prompt: str = ""
    session_memory: SessionMemory | None = None
    protected_start: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.microcompaction_policy, MicrocompactionPolicy):
            self.microcompaction_policy = MicrocompactionPolicy.from_dict(self.microcompaction_policy)

    def _definition_settings(self) -> dict[str, Any]:
        memory_settings = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in {
                "protected_start",
                "workspace_backend",
                "summary_callback",
                "session_memory",
                "microcompaction_policy",
                "recovery_tool_available",
            }
        }
        memory_settings["microcompaction_policy"] = self.microcompaction_policy.to_dict()
        return memory_settings

    @property
    def effective_context_window(self) -> int:
        """Token budget available to prompt messages after reserving model output."""
        return max(self.model_context_window - self.reserved_output_tokens, 0)

    @property
    def autocompact_threshold(self) -> int:
        """Prompt token threshold that triggers automatic compaction."""
        return compute_compaction_threshold(
            configured_threshold=self.compact_threshold,
            model_context_window=self.model_context_window,
            reserved_output_tokens=self.reserved_output_tokens,
            autocompact_buffer_tokens=self.autocompact_buffer_tokens,
        )

    @property
    def warning_threshold(self) -> int:
        """Token threshold that emits a memory warning before compaction."""
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        return int(threshold * self.warning_threshold_percentage / 100)

    @property
    def microcompact_trigger_threshold(self) -> int:
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        return int(threshold * self.microcompaction_policy.trigger_ratio)

    @property
    def microcompact_target_threshold(self) -> int:
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        return int(threshold * self.microcompaction_policy.target_ratio)

    def compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
        force: bool = False,
    ) -> tuple[list[Message], bool]:
        result = self.compact_with_result(
            messages,
            cycle_index=cycle_index,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
            force=force,
        )
        return result.messages, result.changed

    def compact_with_result(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
        force: bool = False,
        microcompact_plan: MicrocompactPlan | None = None,
        microcompact_planned: bool = False,
    ) -> MemoryCompactionResult:
        original_messages = list(messages)
        strongest_mode: CompactionMode = "none"
        if not messages:
            return MemoryCompactionResult(messages=messages, mode="none", changed=False)

        # Apply a precomputed plan to exactly the view it was built against.
        working_messages = list(messages)
        if self._summary_parts(working_messages, self.keep_recent_messages) is None:
            return MemoryCompactionResult(messages=working_messages, mode="none", changed=False)

        message_length = self.effective_tokens(
            working_messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        self._maybe_extract_session_memory(
            working_messages,
            cycle_index=cycle_index,
            current_tokens=message_length,
        )
        microcompacted_messages = working_messages
        microcompact_result = MicrocompactionApplyResult(messages=working_messages)
        if not force:
            effective_plan = microcompact_plan
            if not microcompact_planned:
                effective_plan = self.plan_microcompaction(
                    working_messages,
                    cycle_index=cycle_index,
                    current_tokens=message_length,
                )
            if effective_plan is not None and effective_plan.candidates:
                microcompact_result = self.apply_microcompaction(
                    working_messages,
                    plan=effective_plan,
                )
                microcompacted_messages = microcompact_result.messages
            if microcompact_result.archived_count > 0:
                strongest_mode = _strongest_compaction_mode(strongest_mode, "micro")
                message_length = max(
                    message_length - microcompact_result.reclaimed_tokens,
                    0,
                )

        sanitized_messages = filter_empty_assistant_messages(microcompacted_messages)
        if sanitized_messages != microcompacted_messages:
            strongest_mode = "structural"
            microcompacted_messages = sanitized_messages

        if not force and message_length <= self.autocompact_threshold:
            maybe_warned_messages, warning_inserted = self._maybe_append_memory_warning(
                microcompacted_messages,
                message_length=message_length,
            )
            if warning_inserted:
                strongest_mode = _strongest_compaction_mode(strongest_mode, "structural")
            return self._compaction_result(
                original_messages,
                maybe_warned_messages,
                strongest_mode,
                archived_count=microcompact_result.archived_count,
                reclaimed_tokens=microcompact_result.reclaimed_tokens,
                artifact_failure_count=microcompact_result.artifact_failure_count,
            )

        summarized_messages, summarized = self.compress_memory(microcompacted_messages, cycle_index=cycle_index)
        if summarized:
            strongest_mode = _strongest_compaction_mode(strongest_mode, "summary")
        return self._compaction_result(
            original_messages,
            summarized_messages,
            strongest_mode,
            archived_count=microcompact_result.archived_count,
            reclaimed_tokens=microcompact_result.reclaimed_tokens,
            artifact_failure_count=microcompact_result.artifact_failure_count,
        )

    @staticmethod
    def _compaction_result(
        before_messages: list[Message],
        after_messages: list[Message],
        mode: CompactionMode,
        *,
        archived_count: int = 0,
        reclaimed_tokens: int = 0,
        artifact_failure_count: int = 0,
    ) -> MemoryCompactionResult:
        content_changed = before_messages != after_messages
        if not content_changed:
            mode = "none"
        elif mode == "none":
            mode = "structural"
        return MemoryCompactionResult(
            messages=after_messages,
            mode=mode,
            changed=content_changed,
            archived_count=archived_count,
            reclaimed_tokens=reclaimed_tokens,
            artifact_failure_count=artifact_failure_count,
        )

    def emergency_compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        drop_ratio: float = 0.2,
    ) -> list[Message]:
        return self._summarize_prefix(messages, keep_recent=self.keep_recent_messages, drop_ratio=drop_ratio)[0]

    def compress_memory(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
    ) -> tuple[list[Message], bool]:
        return self._summarize_prefix(messages, keep_recent=self.keep_recent_messages)

    def _summary_parts(
        self, messages: list[Message], keep_recent: int
    ) -> tuple[list[Message], list[Message], list[Message], list[Message]] | None:
        return _summary_compaction._summary_parts(messages, keep_recent, protected_start=self.protected_start)

    def plan_summary(
        self, messages: list[Message], *, keep_recent: int | None = None, drop_ratio: float = 0
    ) -> SummaryPlan | None:
        """Plan without inference, pruning, or callbacks; preserve complete block boundaries."""
        return _summary_compaction.plan_summary(
            messages,
            keep_recent=self.keep_recent_messages if keep_recent is None else keep_recent,
            drop_ratio=drop_ratio,
            protected_start=self.protected_start,
            language=self.language,
            event_limit=self.summary_event_limit,
        )

    def plan_full_summary(self, messages: list[Message]) -> SummaryPlan | None:
        """Summarize the full history without changing the caller's context."""
        return _summary_compaction.plan_full_summary(
            messages, protected_start=self.protected_start, language=self.language, event_limit=self.summary_event_limit
        )

    def _summarize_prefix(
        self, messages: list[Message], *, keep_recent: int, drop_ratio: float = 0
    ) -> tuple[list[Message], bool]:
        if self.summary_callback is None:
            return list(messages), False
        plan = self.plan_summary(messages, keep_recent=keep_recent, drop_ratio=drop_ratio)
        if plan is None:
            return list(messages), False
        return self.accept_summary(plan, self._generate_summary(plan.prompt))

    def accept_summary(
        self, plan: SummaryPlan, text: str, *, notify: bool = True, require_reduction: bool = True
    ) -> tuple[list[Message], bool]:
        """Validate a retained model result. Log owners defer notification until commit."""
        result = self._drain_counts(
            _summary_compaction.accept_summary(
                plan,
                text,
                protected_start=self.protected_start,
                recovery_tool_available=self.recovery_tool_available,
                effective_context_window=self.effective_context_window,
                require_reduction=require_reduction,
                model=self.model,
            )
        )
        if result.accepted and notify and self.session_memory is not None:
            self.session_memory.on_compaction(current_tokens=result.current_tokens)
        return result.messages, result.accepted

    def compaction_evidence(self, messages: list[Message]) -> dict[str, list[dict[str, Any]]]:
        """Collect validated recovery references without reading files or dropping evidence."""
        return _summary_compaction.compaction_evidence(messages, protected_start=self.protected_start)

    def apply_session_memory_context(self, messages: list[Message]) -> list[Message]:
        self._capture_base_system_prompt(messages)
        rendered_system_prompt = self._render_system_prompt()
        if not rendered_system_prompt:
            return list(messages)

        updated_messages = list(messages)
        if updated_messages and updated_messages[0].role == "system":
            if updated_messages[0].content == rendered_system_prompt:
                return updated_messages
            updated_messages[0] = replace(updated_messages[0], content=rendered_system_prompt)
            return updated_messages
        return [Message(role="system", content=rendered_system_prompt), *updated_messages]

    def strip_session_memory_context(self, messages: list[Message]) -> list[Message]:
        self._capture_base_system_prompt(messages)
        if not messages:
            return []
        if not self.base_system_prompt:
            return list(messages)

        updated_messages = list(messages)
        if updated_messages[0].role == "system" and updated_messages[0].content != self.base_system_prompt:
            updated_messages[0] = replace(updated_messages[0], content=self.base_system_prompt)
        return updated_messages

    def _drain_counts[T](self, steps: Generator[TokenCount, int, T]) -> T:
        def count(request: TokenCount) -> int:
            if isinstance(request.messages, Message):
                return self._estimate_message_tokens(request.messages)
            return self._calculate_message_length(request.messages)

        return _drain_counts(steps, count)

    def _calculate_message_length(self, messages: list[Message]) -> int:
        if not messages:
            return 0
        payload = [message.to_openai_message() for message in messages]
        return count_messages_tokens(payload, model=self.model)

    def _estimate_tool_message_length(self, messages: list[Message], recent_tool_call_ids: set[str] | None) -> int:
        if not recent_tool_call_ids:
            return 0
        tool_messages = [
            message.to_openai_message()
            for message in messages
            if message.role == "tool" and message.tool_call_id in recent_tool_call_ids
        ]
        if not tool_messages:
            return 0
        return count_messages_tokens(tool_messages, model=self.model)

    def effective_tokens(
        self,
        messages: list[Message],
        *,
        total_tokens: int | None,
        recent_tool_call_ids: set[str] | None,
    ) -> int:
        if total_tokens is not None and total_tokens > 0:
            return total_tokens + self._estimate_tool_message_length(messages, recent_tool_call_ids)
        return self._calculate_message_length(messages)

    def estimate_memory_usage_percentage(
        self,
        messages: list[Message],
        *,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
    ) -> int:
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        used_tokens = self.effective_tokens(
            messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        return int((used_tokens / threshold) * 100)

    def should_preemptive_microcompact(
        self,
        messages: list[Message],
        *,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
    ) -> bool:
        threshold = self.microcompact_trigger_threshold
        if threshold <= 0:
            return False
        effective_length = self.effective_tokens(
            messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        return effective_length > threshold

    def microcompact_messages(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
    ) -> tuple[list[Message], int]:
        plan = self.plan_microcompaction(
            messages,
            cycle_index=cycle_index,
            current_tokens=self._calculate_message_length(messages),
        )
        if plan is None:
            return messages, 0
        result = self.apply_microcompaction(messages, plan=plan)
        return result.messages, result.archived_count

    def plan_microcompaction(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None,
        current_tokens: int,
        recovery_tool_available: bool | None = None,
    ) -> MicrocompactPlan | None:
        return self._drain_counts(
            plan_memory_microcompact(
                messages,
                cycle_index=cycle_index,
                current_tokens=current_tokens,
                settings=self._definition_settings(),
                protected_start=self.protected_start,
                recovery_tool_available=self.recovery_tool_available
                if recovery_tool_available is None
                else recovery_tool_available,
                artifact_path_estimate_for=estimate_tool_artifact_path,
                model=self.model,
            )
        )

    def apply_microcompaction(
        self,
        messages: list[Message],
        *,
        plan: MicrocompactPlan,
    ) -> MicrocompactionApplyResult:
        updated_messages = list(messages)
        remaining_tokens = plan.current_tokens
        archived_count = 0
        reclaimed_tokens = 0
        artifact_failure_count = 0
        protected = atomic_start(messages, self.protected_start) if self.protected_start is not None else len(messages)
        for candidate in plan.candidates:
            if candidate.message_index >= protected:
                continue
            if remaining_tokens <= plan.target_tokens:
                break
            message = updated_messages[candidate.message_index]
            artifact = self._archive_tool_message(message, candidate)
            if artifact is None:
                artifact_failure_count += 1
                continue
            marker = build_compacted_tool_content(
                self._compaction_excerpt(message),
                artifact_path=artifact.path,
                tool_name=candidate.tool_name,
                excerpt_head_chars=self.tool_result_excerpt_head,
                excerpt_tail_chars=self.tool_result_excerpt_tail,
            )
            replacement = replace_with_compacted_marker(
                message,
                candidate,
                artifact=artifact,
                marker=marker,
            )
            actual_reclaimed = max(
                self._estimate_message_tokens(message) - self._estimate_message_tokens(replacement),
                0,
            )
            if actual_reclaimed == 0:
                continue
            updated_messages[candidate.message_index] = replacement
            archived_count += 1
            reclaimed_tokens += actual_reclaimed
            remaining_tokens = max(remaining_tokens - actual_reclaimed, 0)
        return MicrocompactionApplyResult(
            messages=updated_messages,
            archived_count=archived_count,
            reclaimed_tokens=reclaimed_tokens,
            artifact_failure_count=artifact_failure_count,
        )

    def _estimate_message_tokens(self, message: Message) -> int:
        return count_messages_tokens([message.to_openai_message()], model=self.model)

    def _render_system_prompt(self) -> str:
        session_context = self.session_memory.render_as_system_context() if self.session_memory is not None else ""
        if self.base_system_prompt and session_context:
            return f"{self.base_system_prompt}\n\n{session_context}"
        return session_context or self.base_system_prompt

    def _capture_base_system_prompt(self, messages: list[Message]) -> None:
        if self.base_system_prompt:
            return
        if not messages or messages[0].role != "system":
            return
        current_content = messages[0].content
        marker = "\n\n<Session Memory>"
        if marker in current_content:
            current_content = current_content.split(marker, 1)[0]
        if current_content:
            self.base_system_prompt = current_content

    def _maybe_extract_session_memory(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None,
        current_tokens: int,
    ) -> bool:
        if self.session_memory is None or current_tokens <= 0:
            return False

        text_message_count = sum(1 for message in messages if message.role in {"user", "assistant"})
        if not self.session_memory.should_extract(current_tokens, text_message_count):
            return False

        extracted = self.session_memory.extract(
            messages,
            current_cycle=cycle_index or 0,
            current_tokens=current_tokens,
        )
        return extracted > 0

    def _maybe_append_memory_warning(self, messages: list[Message], *, message_length: int) -> tuple[list[Message], bool]:
        if not self.include_memory_warning:
            return messages, False
        if self.autocompact_threshold <= 0:
            return messages, False

        if message_length < self.warning_threshold:
            return messages, False

        template = _MEMORY_WARNING_PROMPTS.get(self.language, _MEMORY_WARNING_PROMPTS["en-US"])
        warning_text = template.format(memory_threshold_percentage=self.warning_threshold_percentage)
        for message in reversed(messages[-10:]):
            if message.role == "user" and warning_text in message.content:
                return messages, False

        warned = list(messages)
        warned.append(Message(role="user", content=warning_text))
        return warned, True

    def _archive_tool_message(
        self,
        message: Message,
        candidate: MicrocompactCandidate,
    ) -> ToolArtifactRef | None:
        backend = self.workspace_backend
        if backend is None:
            return None
        try:
            if candidate.existing_artifact is not None:
                if self._artifact_is_intact(candidate.existing_artifact):
                    return candidate.existing_artifact
                return None
            if has_recovery_envelope(message.content):
                return None
            return persist_text_artifact(
                backend,
                self.artifact_scope,
                candidate.tool_call_id,
                message.content,
                reuse_existing=True,
            )
        except Exception as exc:
            from vv_agent.runtime.cancellation import CancelledError

            if isinstance(exc, CancelledError) or getattr(exc, "vv_agent_control_flow", False):
                raise
            return None

    @staticmethod
    def _compaction_excerpt(message: Message) -> str:
        excerpt = message.metadata.get(EXCERPT_METADATA_KEY)
        return excerpt if isinstance(excerpt, str) else message.content

    def _artifact_is_intact(self, artifact: ToolArtifactRef) -> bool:
        backend = self.workspace_backend
        if backend is None:
            return False
        try:
            from vv_agent.workspace.streaming import scan_text

            scanned = scan_text(backend, artifact.path, lambda _text: None)
        except (OSError, ValueError):
            return False
        return scanned.valid_utf8 and scanned.size_bytes == artifact.size_bytes and scanned.sha256 == artifact.sha256

    def _generate_summary(self, prompt: str) -> str:
        if self.summary_callback is not None:
            try:
                summarized = self.summary_callback(prompt, self.summary_backend, self.summary_model)
                if isinstance(summarized, str):
                    return summarized
            except Exception as exc:
                from vv_agent.runtime.cancellation import CancelledError

                if isinstance(exc, CancelledError) or getattr(exc, "vv_agent_control_flow", False):
                    raise
                logging.getLogger(__name__).debug("Memory summary callback failed", exc_info=True)
        return ""


def build_memory_manager(
    *,
    task: AgentTask,
    workspace_path: Path,
    workspace_backend: WorkspaceBackend | None = None,
    ctx: ExecutionContext | None = None,
    tool_registry: ToolRegistry,
) -> MemoryManager:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}

    def read_optional_int(key: str, *, minimum: int = 0) -> int | None:
        if key not in metadata:
            return None
        raw = metadata.get(key)
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return max(value, minimum)

    def read_int(key: str, default: int, *, minimum: int = 0) -> int:
        value = read_optional_int(key, minimum=minimum)
        return max(default, minimum) if value is None else value

    warning_threshold = max(1, min(task.memory_threshold_percentage, 100))
    summary_backend = read_optional_string(metadata, "memory_summary_backend")
    summary_model = read_optional_string(metadata, "memory_summary_model") or task.model
    session_memory_extraction_backend = read_optional_string(metadata, "session_memory_extraction_backend") or summary_backend
    session_memory_extraction_model = read_optional_string(metadata, "session_memory_extraction_model") or summary_model
    session_memory_enabled = read_session_memory_enabled(metadata)
    model_context_window = metadata_token_limit(
        metadata,
        "model_context_window",
        minimum=1,
    )
    model_max_output_tokens = metadata_token_limit(
        metadata,
        "model_max_output_tokens",
        minimum=0,
    )
    if model_context_window is None or model_max_output_tokens is None:
        fallback_context_window, fallback_max_output_tokens = resolve_model_token_limits(task.model)
        if model_context_window is None:
            model_context_window = fallback_context_window
        if model_max_output_tokens is None:
            model_max_output_tokens = fallback_max_output_tokens
    effective_model_settings = task.model_settings
    if ctx is not None:
        runtime_model_settings = ctx.metadata.get("_vv_agent_model_settings")
        if isinstance(runtime_model_settings, ModelSettings):
            effective_model_settings = runtime_model_settings
    request_max_tokens = effective_model_settings.max_tokens if effective_model_settings is not None else None
    task_reserved_output_tokens = metadata_token_limit(
        metadata,
        "reserved_output_tokens",
        minimum=0,
    )
    if request_max_tokens is not None:
        reserved_output_tokens = request_max_tokens
        reserved_output_source = "model_settings"
    elif task_reserved_output_tokens is not None:
        reserved_output_tokens = task_reserved_output_tokens
        reserved_output_source = "task_metadata"
    else:
        reserved_output_tokens = 16_000
        reserved_output_source = "framework_fallback"
        if model_max_output_tokens is not None and model_max_output_tokens < reserved_output_tokens:
            reserved_output_tokens = model_max_output_tokens
            reserved_output_source = "framework_fallback_capped_by_model_capability"
    autocompact_buffer_tokens = read_int("autocompact_buffer_tokens", 13_000, minimum=0)
    if model_context_window is None:
        planning_prompt_capacity = task.memory_compact_threshold if task.memory_compact_threshold > 0 else 250_000
        model_context_window = min(
            planning_prompt_capacity + reserved_output_tokens + autocompact_buffer_tokens,
            (1 << 64) - 1,
        )
    session_memory: SessionMemory | None = None
    if session_memory_enabled:
        session_memory_scope = read_optional_string(metadata, "session_id", "task_id") or str(task.task_id or "").strip()
        session_memory = SessionMemory(
            SessionMemoryConfig(
                min_tokens_before_extraction=read_int("session_memory_min_tokens", 10_000, minimum=1),
                max_tokens=read_int("session_memory_max_tokens", 40_000, minimum=1),
                min_text_messages=read_int("session_memory_min_text_messages", 5, minimum=1),
                storage_dir=str(metadata.get("session_memory_storage_dir", ".memory/session")),
                extraction_callback=None,
                extraction_backend=session_memory_extraction_backend,
                extraction_model=session_memory_extraction_model,
                token_model=task.model or "",
            ),
            workspace=workspace_path if task.use_workspace else None,
            storage_scope=session_memory_scope,
        )
        session_memory.load()
    tool_result_retentions: dict[str, ToolResultRetention] = {}
    for tool_name in tool_registry.list_tool_names():
        declared_metadata = tool_registry.tool_metadata(tool_name)
        tool_result_retentions[tool_name] = (
            declared_metadata.result_retention if declared_metadata is not None else ToolResultRetention.ARCHIVE
        )
    return MemoryManager(
        compact_threshold=max(task.memory_compact_threshold, 0),
        keep_recent_messages=read_int("memory_keep_recent_messages", 10, minimum=1),
        model=task.model or "",
        model_context_window=model_context_window,
        model_max_output_tokens=model_max_output_tokens,
        reserved_output_tokens=reserved_output_tokens,
        reserved_output_source=reserved_output_source,
        autocompact_buffer_tokens=autocompact_buffer_tokens,
        language=str(metadata.get("language", "zh-CN")),
        warning_threshold_percentage=warning_threshold,
        include_memory_warning=bool(metadata.get("include_memory_warning", False)),
        tool_result_excerpt_head=read_int("tool_result_excerpt_head", 200),
        tool_result_excerpt_tail=read_int("tool_result_excerpt_tail", 200),
        microcompaction_policy=task.microcompaction_policy,
        tool_result_retentions=tool_result_retentions,
        workspace_backend=workspace_backend or (LocalWorkspaceBackend(workspace_path) if task.use_workspace else None),
        artifact_scope=task.task_id,
        summary_event_limit=read_int("summary_event_limit", 40, minimum=1),
        summary_backend=summary_backend,
        summary_model=summary_model,
        summary_callback=None,
        base_system_prompt=task.prompt_bundle.flatten(),
        session_memory=session_memory,
    )


def read_optional_string(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, str):
            value = raw.strip()
            if value:
                return value
    return None


def read_session_memory_enabled(metadata: dict[str, Any]) -> bool:
    explicit = metadata.get("session_memory_enabled", False)
    if not isinstance(explicit, bool):
        raise ValueError("session_memory_enabled must be a boolean")
    return explicit


def metadata_token_limit(metadata: dict[str, Any], key: str, *, minimum: int) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value
