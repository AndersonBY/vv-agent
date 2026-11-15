from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vv_agent.agent import Agent
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.prompt.builder import build_raw_system_prompt_bundle
from vv_agent.run_config import RunConfig
from vv_agent.session import delegation
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import digest
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import Message


def _definition(initial_message: Message, input_message: Message) -> dict[str, Any]:
    task = AgentTask(
        task_id="child/task",
        model="scripted",
        prompt_bundle=build_raw_system_prompt_bundle("Work."),
        user_prompt="Do the child work.",
        initial_messages=[initial_message],
        metadata={"vv_session": {"input_messages": [input_message.to_dict()]}},
    ).to_dict()
    return {
        "agent_name": "worker",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": [],
            "context_length": None,
            "max_output_tokens": None,
        },
        "capabilities": {},
        "task": task,
        "tools": [],
    }


def test_reconstruct_hydrates_child_admission_nested_messages(tmp_path, monkeypatch) -> None:
    initial = Message("user", "child initial")
    input_message = Message("user", "child input")
    definition = _definition(initial, input_message)
    admission = {
        "mode": "configured",
        "selector": "worker",
        "definition": definition,
        "definition_digest": digest(definition),
        "budget": {
            "max_total_tokens": 100,
            "max_uncached_input_tokens": None,
            "max_tool_calls": 2,
            "max_tool_calls_by_name": {},
            "max_wall_time_ms": None,
            "max_host_cost": None,
            "unavailable_metric_policy": "continue_and_mark",
        },
        "handler_version": "1",
        "sub_config": None,
        "exclude_files_pattern": None,
        "handoff_count": 0,
        "max_handoffs": 100,
        "handoff_metadata": {},
    }

    with SQLiteStore.standalone(str(tmp_path / "child.sqlite")) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))
        sessions.create("child", "/workspace", {"child_admission": admission})
        created = store.read("child", limit=1).records[0].record

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="child")
            attributes_ref = ContentRef.from_dict(created.payload["attributes"])
            published = writer.get_json(attributes_ref)
        task = published["child_admission"]["definition"]["task"]
        assert declared_content_ref(task["initial_messages"][0]) is not None
        assert declared_content_ref(task["metadata"]["vv_session"]["input_messages"][0]) is not None

        child_runtime = SimpleNamespace()
        runtime = SimpleNamespace(
            handler_version="1",
            config=SimpleNamespace(workspace_backend=None),
            bind_state=lambda state, task: dict(state),
            for_agent=lambda agent, config: child_runtime,
        )
        monkeypatch.setattr(delegation, "target_agent", lambda *args, **kwargs: Agent("worker", "Work.", model="scripted"))
        monkeypatch.setattr(
            delegation,
            "child_config",
            lambda *args, **kwargs: RunConfig(workspace="/workspace"),
        )

        reconstructed = delegation.reconstruct(runtime, store, "child")

    assert reconstructed is child_runtime
    assert child_runtime.frozen_task.initial_messages == [initial]
    assert child_runtime.frozen_task.metadata["vv_session"]["input_messages"] == [input_message.to_dict()]
    assert created.payload["attributes"] == attributes_ref.to_dict()
