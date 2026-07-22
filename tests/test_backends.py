from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack

import pytest

from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.backends.thread import ThreadBackend
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.context import ExecutionContext
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse


class _TaskOverrides(TypedDict, total=False):
    task_id: str
    model: str
    system_prompt: str
    user_prompt: str
    max_cycles: int
    no_tool_policy: Literal["continue", "wait_user", "finish"]
    metadata: dict[str, Any]


def _make_task(**overrides: Unpack[_TaskOverrides]) -> AgentTask:
    defaults: _TaskOverrides = {
        "task_id": "backend-test",
        "model": "test",
        "system_prompt": "sys",
        "user_prompt": "hi",
        "max_cycles": 2,
        "no_tool_policy": "finish",
    }
    defaults.update(overrides)
    return AgentTask(**defaults)


class TestInlineBackend:
    def test_basic_run(self):
        llm = ScriptedLLM(steps=[LLMResponse(content="done")])
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
            execution_backend=InlineBackend(),
        )
        result = runtime.run(_make_task())
        assert result.status == AgentStatus.COMPLETED
        assert result.final_answer == "done"


class TestThreadBackend:
    def test_basic_run(self):
        llm = ScriptedLLM(steps=[LLMResponse(content="threaded")])
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
            execution_backend=ThreadBackend(max_workers=2),
        )
        result = runtime.run(_make_task())
        assert result.status == AgentStatus.COMPLETED
        assert result.final_answer == "threaded"

    def test_cancel_with_thread_backend(self):
        llm = ScriptedLLM(
            steps=[
                LLMResponse(content="c1"),
                LLMResponse(content="c2"),
            ]
        )
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
            execution_backend=ThreadBackend(),
        )
        token = CancellationToken()
        token.cancel()
        ctx = ExecutionContext(cancellation_token=token)
        result = runtime.run(_make_task(no_tool_policy="continue"), ctx=ctx)
        assert result.status == AgentStatus.FAILED
        assert "cancelled" in (result.error or "").lower()

    def test_parallel_map(self):
        backend = ThreadBackend(max_workers=4)
        results = backend.parallel_map(lambda x: x * 2, [1, 2, 3, 4])
        assert results == [2, 4, 6, 8]

    def test_submit_future(self):
        backend = ThreadBackend(max_workers=2)
        future = backend.submit(lambda: 42)
        assert future.result(timeout=5) == 42


class TestCeleryBackend:
    def test_import_without_celery(self):
        """CeleryBackend can be imported even without celery installed."""
        from vv_agent.runtime.backends.celery import CeleryBackend

        # Just verify the class exists
        assert CeleryBackend is not None

    def test_distributed_backend_accepts_current_recipe(self, tmp_path: Path):
        """CeleryBackend accepts the current capability-based recipe."""
        from vv_agent.runtime.backends.celery import (
            _CELERY_AVAILABLE,
            CeleryBackend,
            RuntimeRecipe,
            register_cycle_task,
        )

        if not _CELERY_AVAILABLE:
            pytest.skip("celery not installed")

        from celery import Celery

        from vv_agent.runtime.backends.distributed import (
            CapabilityRef,
            DistributedCapabilities,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        settings_file = tmp_path / "local_settings.py"
        settings_file.write_text("BACKENDS = {}\nENDPOINTS = {}\nMODELS = {}\n")
        celery_app = Celery("test_distributed")
        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = True
        celery_app.conf.task_serializer = "json"
        celery_app.conf.result_serializer = "json"
        celery_app.conf.accept_content = ["json"]
        register_cycle_task(celery_app)

        checkpoint_ref = CapabilityRef("checkpoint.test", "1")
        recipe = RuntimeRecipe(
            settings_file=str(settings_file),
            backend="test",
            model="test",
            workspace=str(workspace),
            capabilities=DistributedCapabilities(
                checkpoint_store_ref=checkpoint_ref,
            ),
        )
        backend = CeleryBackend(
            celery_app=celery_app,
            runtime_recipe=recipe,
        )

        assert backend.runtime_recipe == recipe

    def test_runtime_recipe_serialization(self):
        """Test RuntimeRecipe round-trip serialization."""
        from vv_agent.runtime.backends.celery import _CELERY_AVAILABLE, RuntimeRecipe
        from vv_agent.runtime.backends.distributed import CapabilityRef, DistributedCapabilities

        if not _CELERY_AVAILABLE:
            pytest.skip("celery not installed")

        recipe = RuntimeRecipe(
            settings_file="/tmp/settings.py",
            backend="openai",
            model="gpt-4",
            workspace="/tmp/workspace",
            timeout_seconds=120.0,
            log_preview_chars=300,
            capabilities=DistributedCapabilities(
                hook_refs=(CapabilityRef("hook.logging", "1"),),
                checkpoint_store_ref=CapabilityRef("checkpoint.test", "1"),
            ),
        )
        d = recipe.to_dict()
        restored = RuntimeRecipe.from_dict(d)
        assert restored.settings_file == recipe.settings_file
        assert restored.backend == recipe.backend
        assert restored.model == recipe.model
        assert restored.workspace == recipe.workspace
        assert restored.timeout_seconds == recipe.timeout_seconds
        assert restored.capabilities.hook_refs == recipe.capabilities.hook_refs
        assert restored.capabilities.checkpoint_store_ref == recipe.capabilities.checkpoint_store_ref
        assert restored.log_preview_chars == recipe.log_preview_chars

    def test_agent_task_serialization(self):
        """Test AgentTask round-trip serialization."""
        task = _make_task(metadata={"language": "en"})
        d = task.to_dict()
        restored = AgentTask.from_dict(d)
        assert restored.task_id == task.task_id
        assert restored.model == task.model
        assert restored.system_prompt == task.system_prompt
        assert restored.user_prompt == task.user_prompt
        assert restored.max_cycles == task.max_cycles
        assert restored.no_tool_policy == task.no_tool_policy
        assert restored.metadata == task.metadata

    def test_agent_result_serialization(self):
        """Test AgentResult round-trip serialization."""
        from vv_agent.types import AgentResult, Message

        result = AgentResult(
            status=AgentStatus.COMPLETED,
            messages=[Message(role="user", content="hi")],
            cycles=[],
            final_answer="done",
            shared_state={"key": "value"},
        )
        d = result.to_dict()
        restored = AgentResult.from_dict(d)
        assert restored.status == result.status
        assert restored.final_answer == result.final_answer
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "hi"
        assert restored.shared_state == {"key": "value"}
