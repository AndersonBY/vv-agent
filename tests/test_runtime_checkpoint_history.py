from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from test_checkpoint import _minimal_checkpoint

from vv_agent.checkpoint import CheckpointConfig, ResumePolicy
from vv_agent.events import DiagnosticEvent, RunEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, CompletionReason, LLMResponse


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_public_runtime_returns_complete_archived_result_and_logical_diagnostic(
    tmp_path: Path, resume: bool, cancel: bool
) -> None:
    token = CancellationToken()
    model_calls = 0

    class ObservedStore(InMemoryCheckpointStore):
        history_reads = 0
        crash_once = resume

        def load_checkpoint_history(self, checkpoint_key: str) -> Any:
            assert model_calls == 4, "active execution must not load archived history"
            self.history_reads += 1
            return super().load_checkpoint_history(checkpoint_key)

        def commit_checkpoint(self, checkpoint: Any, **kwargs: Any) -> bool:
            written = super().commit_checkpoint(checkpoint, **kwargs)
            if written and checkpoint.cycle_index == 2 and self.crash_once:
                self.crash_once = False
                raise SystemExit("crash after cycle commit")
            if written and checkpoint.cycle_index == 4 and cancel:
                token.cancel()
            return written

    store = ObservedStore()
    seed = _minimal_checkpoint(key="public-runtime-history")
    events: list[RunEvent] = []

    def new_controller() -> CheckpointResumeController:
        return CheckpointResumeController(
            config=CheckpointConfig(store=store, key=seed.checkpoint_key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
            task_id=seed.task_id,
            run_id=seed.root_run_id,
            trace_id=seed.trace_id,
            run_definition=deepcopy(seed.run_definition),
            run_definition_digest=seed.run_definition_digest,
            initial_messages=[],
            initial_shared_state={},
            initial_budget_usage=None,
            extensions=[],
            reconciliation_provider=None,
            event_sink=events.append,
        )

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        assert store.history_reads == 0
        model_calls += 1
        return LLMResponse(
            content=f"cycle {model_calls}",
            raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[complete] * 4),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id=seed.task_id,
        model="test-model",
        prompt_bundle=build_raw_system_prompt_bundle("continue"),
        user_prompt="retain history",
        max_cycles=5 if cancel else 4,
        no_tool_policy="continue",
        use_workspace=False,
    )
    controller = new_controller()
    try:
        assert controller.admit() is None
        if resume:
            with pytest.raises(SystemExit, match="crash after cycle commit"):
                runtime.run(task, checkpoint_controller=controller)
            controller.close()
            assert model_calls == 2
            assert store.history_reads == 0
            controller = new_controller()
            assert controller.admit() is None

        result = runtime.run(
            task,
            checkpoint_controller=controller,
            ctx=ExecutionContext(cancellation_token=token, event_handler=events.append),
        )
        assert result.status is (AgentStatus.FAILED if cancel else AgentStatus.MAX_CYCLES)
        assert result.completion_reason is (CompletionReason.CANCELLED if cancel else CompletionReason.MAX_CYCLES)
        assert [cycle.index for cycle in result.cycles] == [1, 2, 3, 4]
        assert [call.cycle_index for call in result.token_usage.model_calls] == [1, 2, 3, 4]
        assert result.token_usage.total_tokens == 60
        assert store.history_reads == 1
        diagnostics = [
            event
            for event in events
            if isinstance(event, DiagnosticEvent) and event.code == ("run_cancelled" if cancel else "run_max_cycles")
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].cycle_index == 4

        terminal = controller.finalize(result)
        assert terminal.cycles == result.cycles
        assert terminal.token_usage == result.token_usage
        checkpoint = store.load_checkpoint(seed.checkpoint_key)
        assert checkpoint is not None
        assert checkpoint.history["cycle_count"] == 3
        assert [cycle.index for cycle in checkpoint.cycles] == [4]
        assert checkpoint.terminal_result is not None
        assert [cycle.index for cycle in checkpoint.terminal_result.cycles] == [4]

        controller.close()
        controller = new_controller()
        replay = controller.admit()
        assert replay is not None
        assert replay.cycles == result.cycles
        assert replay.token_usage == result.token_usage
        assert model_calls == 4
    finally:
        controller.close()
