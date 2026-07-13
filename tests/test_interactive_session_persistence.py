from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    AgentSessionOptions,
    AgentSessionRun,
    AgentStatus,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    MemorySession,
    MemorySessionStore,
    Message,
    RunResult,
    create_agent_session,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.types import AgentResult, LLMResponse, ToolCall

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent_v1.json"


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="key", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
    )


def _definition() -> InteractiveAgentDefinition:
    return InteractiveAgentDefinition(
        description="Remember the full conversation and finish each turn.",
        model="test-model",
        max_cycles=1,
    )


def _standalone_run(*, prompt: str, messages: list[Message]) -> AgentSessionRun:
    return AgentSessionRun(
        agent_name="inline",
        result=AgentResult(
            status=AgentStatus.COMPLETED,
            messages=messages,
            cycles=[],
            final_answer=f"answer: {prompt}",
            shared_state={"todo_list": []},
        ),
        resolved=_resolved(),
    )


def test_agent_session_options_keeps_workspace_as_third_positional_argument(tmp_path: Path) -> None:
    options = AgentSessionOptions(tmp_path / "settings.py", "test", tmp_path)

    assert options.workspace == tmp_path
    assert options.session is None


def test_interactive_client_hydrates_and_reuses_backing_session_without_duplicates(tmp_path: Path) -> None:
    backing = MemorySession("persistent-thread")
    seeded = [
        Message(role="user", content="earlier question", metadata={"turn": 0}),
        Message(role="assistant", content="earlier answer", metadata={"turn": 0}),
    ]
    backing.add_items(seeded)
    requests: list[list[Message]] = []
    turn = 0

    def llm_builder(*_: Any, **__: Any):
        nonlocal turn
        turn += 1
        current_turn = turn

        def respond(request: LlmRequest) -> LLMResponse:
            requests.append(list(request.messages))
            return LLMResponse(
                content=f"answer {current_turn}",
                tool_calls=[
                    ToolCall(
                        id=f"finish-{current_turn}",
                        name=TASK_FINISH_TOOL_NAME,
                        arguments={"message": f"done {current_turn}"},
                    )
                ],
            )

        return ScriptedLLM(steps=[respond]), _resolved()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=llm_builder,
            session=backing,
        )
    )

    first_facade = client.create_session(agent=_definition())
    assert first_facade.session_id == "persistent-thread"
    assert first_facade.session is backing
    assert first_facade.messages == seeded

    first_run = first_facade.prompt("first", auto_follow_up=False)
    after_first = backing.get_items()
    assert isinstance(first_run, RunResult)
    assert first_run.agent_name == "inline"
    assert first_run.run_id.startswith("run_")
    assert first_run.trace_id.startswith("trace_")
    assert first_run.new_items == after_first[len(seeded) :]
    assert first_run.resolved is not None
    assert first_run.resolved.model_id == "test-model"
    assert first_facade.messages == after_first
    assert [message.content for message in after_first if message.role == "user"] == [
        "earlier question",
        "first",
    ]

    rebuilt_facade = client.create_session(agent=_definition())
    assert rebuilt_facade.session_id == first_facade.session_id
    assert rebuilt_facade.messages == after_first
    rebuilt_facade.prompt("second", auto_follow_up=False)

    persisted = backing.get_items()
    assert rebuilt_facade.messages == persisted
    assert [message.content for message in persisted if message.role == "user"] == [
        "earlier question",
        "first",
        "second",
    ]
    assert [message.content for message in requests[1] if message.role == "user"] == [
        "earlier question",
        "first",
        "second",
    ]
    assert sum(message.tool_call_id == "finish-1" for message in persisted) == 1
    assert sum(message.tool_call_id == "finish-2" for message in persisted) == 1


def test_create_agent_session_hydrates_replaces_and_validates_backing_session(tmp_path: Path) -> None:
    backing = MemorySession("stored-id")
    backing.add_items([Message(role="user", content="stored")])

    def execute_run(**_: Any) -> AgentSessionRun:
        raise AssertionError("replace_messages must not execute a run")

    facade = create_agent_session(
        execute_run=execute_run,
        agent_name="inline",
        definition=_definition(),
        workspace=tmp_path,
        session=backing,
    )
    replacement = [
        Message(role="user", content="replacement"),
        Message(role="assistant", content="replacement answer", metadata={"source": "host"}),
    ]

    assert facade.session_id == "stored-id"
    assert facade.messages == backing.get_items()
    facade.replace_messages(replacement)

    assert facade.messages == replacement
    assert backing.get_items() == replacement
    rebuilt = create_agent_session(
        execute_run=execute_run,
        agent_name="inline",
        definition=_definition(),
        workspace=tmp_path,
        session=backing,
    )
    assert rebuilt.messages == replacement

    with pytest.raises(ValueError, match="does not match backing Session"):
        create_agent_session(
            execute_run=execute_run,
            session_id="different-id",
            agent_name="inline",
            definition=_definition(),
            workspace=tmp_path,
            session=backing,
        )


def test_client_session_override_and_store_isolation(tmp_path: Path) -> None:
    store = MemorySessionStore()
    default_backing = store.session("default-thread")
    override_backing = store.session("override-thread")
    default_backing.add_items([Message(role="user", content="default history")])
    override_backing.add_items([Message(role="user", content="override history")])
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            session=default_backing,
        )
    )

    default_facade = client.create_session(agent=_definition())
    override_facade = client.create_session(agent=_definition(), session=override_backing)
    override_facade.replace_messages([Message(role="assistant", content="override replacement")])

    assert default_facade.session_id == "default-thread"
    assert default_facade.messages == [Message(role="user", content="default history")]
    assert override_facade.session_id == "override-thread"
    assert override_backing.get_items() == [Message(role="assistant", content="override replacement")]
    assert default_backing.get_items() == [Message(role="user", content="default history")]


def test_session_without_explicit_storage_uses_memory_session_as_source_of_truth(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def execute_run(**kwargs: Any) -> AgentSessionRun:
        calls.append(kwargs)
        history = list(kwargs["session"].get_items())
        prompt = kwargs["prompt"]
        messages = [
            *history,
            Message(role="user", content=prompt),
            Message(role="assistant", content=f"answer: {prompt}"),
        ]
        return _standalone_run(prompt=prompt, messages=messages)

    facade = create_agent_session(
        execute_run=execute_run,
        session_id="memory-only",
        agent_name="inline",
        definition=_definition(),
        workspace=tmp_path,
    )

    facade.prompt("first", auto_follow_up=False)
    facade.prompt("second", auto_follow_up=False)

    assert isinstance(facade.session, MemorySession)
    assert calls[0]["session"] is facade.session
    assert calls[1]["session"] is facade.session
    assert calls[0]["initial_messages"] is None
    assert calls[1]["initial_messages"] is None
    assert [message.content for message in facade.messages] == [
        "first",
        "answer: first",
        "second",
        "answer: second",
    ]


def test_custom_run_replaces_rewritten_history_without_duplicate_continuations(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["continuation"]
    assert contract["history_update"] == "replace_when_existing_prefix_is_rewritten"

    backing = MemorySession("rewritten-history")
    backing.add_items(
        [
            Message(role="user", content="seed question", metadata={"revision": 0}),
            Message(role="assistant", content="seed answer", metadata={"revision": 0}),
        ]
    )
    turn = 0

    def execute_run(**kwargs: Any) -> AgentSessionRun:
        nonlocal turn
        turn += 1
        rewritten = [
            replace(message, metadata={**message.metadata, "revision": turn})
            for message in kwargs["session"].get_items()
        ]
        prompt = kwargs["prompt"]
        return _standalone_run(
            prompt=prompt,
            messages=[
                *rewritten,
                Message(role="user", content=prompt),
                Message(role="assistant", content=f"answer: {prompt}"),
            ],
        )

    facade = create_agent_session(
        execute_run=execute_run,
        agent_name="inline",
        definition=_definition(),
        workspace=tmp_path,
        session=backing,
    )

    facade.prompt("first continuation", auto_follow_up=False)
    facade.prompt("second continuation", auto_follow_up=False)

    persisted = backing.get_items()
    contents = [message.content for message in persisted]
    assert contents == [
        "seed question",
        "seed answer",
        "first continuation",
        "answer: first continuation",
        "second continuation",
        "answer: second continuation",
    ]
    has_duplicate_history = len(contents) != len(set(contents))
    assert has_duplicate_history is contract["duplicate_history_allowed"]
    assert [message.metadata.get("revision") for message in persisted[:4]] == [2, 2, 2, 2]
