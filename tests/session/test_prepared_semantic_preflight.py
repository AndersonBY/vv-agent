from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.budget import RunBudgetLimits
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.kernel._reducer import TransitionError
from vv_agent.kernel._sql import SQLStore
from vv_agent.llm.base import LlmRequest
from vv_agent.llm.scripted import ScriptedLLM, ScriptStep
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeToolCallPatch
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import Runtime, drive
from vv_agent.session.records import Record, digest
from vv_agent.types import LLMResponse, ToolCall

from .conftest import Database, open_store
from .helpers import (
    append_fixture_records,
    create_fixture_session,
    fixture_content_reader,
    wrap_fixture_store,
)
from .helpers import record as fixture_record

SESSION_ID = "prepared-preflight"
MASTER_KEY = b"p" * 32
RESOLVED = ResolvedModelConfig("scripted", "m", "m", "m", [])


class _ObservedLLM(ScriptedLLM):
    def __init__(self, steps: list[ScriptStep]) -> None:
        super().__init__(steps)
        self.calls = 0

    def complete(self, request: LlmRequest) -> LLMResponse:
        self.calls += 1
        return super().complete(request)


class _IdentityHook(BaseRuntimeHook):
    def __init__(self, *, change_id: bool) -> None:
        self.change_id = change_id

    def before_tool_call(self, event):
        call_id = "tampered-id" if self.change_id else event.call.id
        return BeforeToolCallPatch(call=ToolCall(call_id, event.call.name, {"text": "patched"}))


def _runtime(
    store,
    database: Database,
    agent: Agent,
    llm: _ObservedLLM,
    *,
    hook: Callable[[str, Record | None], None] | None = None,
) -> Runtime:
    runtime = Runtime(
        agent,
        RunConfig(workspace=Path("/tmp"), session_memory_enabled=False),
        RESOLVED,
        llm,
        lambda: open_store(database),
        content_reader=fixture_content_reader(store, SESSION_ID),
    )
    if hook is not None:
        runtime.hook = hook
    return runtime


def _seed(store, agent: Agent) -> None:
    create_fixture_session(
        store,
        SESSION_ID,
        workspace="/tmp",
        attributes={"seed": {"messages": [], "shared_state": {}}},
    )
    seed_runtime = _runtime(store, Path("unused.sqlite"), agent, _ObservedLLM([]))
    turn_id = f"{SESSION_ID}/turn/initial"
    task = seed_runtime.compile("go", turn_id)
    definition = seed_runtime.definition(task)
    payload = fixture_record("turn_started", sid=SESSION_ID, tid=turn_id)._payload | {
        "definition": definition,
        "definition_digest": digest(definition),
        "budget": RunBudgetLimits().to_dict(),
    }
    turn = make_record_before_publication(
        "turn_started",
        session_id=SESSION_ID,
        turn_id=turn_id,
        payload=payload,
    )
    object.__setattr__(turn, "_task_value", task)
    append_fixture_records(store, SESSION_ID, (turn,), commit_id="seed-turn")


def _counts(store: SQLStore) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in ("sk_record", "sk_commit", "sk_session_key", "sk_content", "sk_content_link"):
        result[table] = int(
            store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                (store.tenant_id, SESSION_ID),
            )[0]
        )
    return result


@pytest.mark.persistent_store
def test_changed_prepared_id_fails_before_commit_without_effects(store, database, tmp_path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "key-state.sqlite")
    trusted_store = wrap_fixture_store(store, provider)
    effects: list[str] = []

    @function_tool
    def echo(text: str) -> str:
        effects.append(text)
        return text

    class Cut(BaseException):
        pass

    def cut_after_tool_plan(point: str, record: Record | None) -> None:
        if (
            point == "after_commit"
            and record is not None
            and record.kind == "op_planned"
            and record._payload["op_kind"] == "tool"
        ):
            raise Cut()

    try:
        agent = Agent("test", "test", tools=[echo], hooks=[_IdentityHook(change_id=True)])
        _seed(trusted_store, agent)
        first_llm = _ObservedLLM([LLMResponse("", [ToolCall("original-id", "echo", {"text": "original"})])])
        with pytest.raises(Cut):
            drive(
                trusted_store,
                SESSION_ID,
                runtime=_runtime(trusted_store, database, agent, first_llm, hook=cut_after_tool_plan),
            )

        before = _counts(store)
        resumed_llm = _ObservedLLM([])
        with pytest.raises(TransitionError, match="hook changed tool call identity"):
            drive(
                trusted_store,
                SESSION_ID,
                runtime=_runtime(trusted_store, database, agent, resumed_llm),
            )

        rows = store.read_state(SESSION_ID)[1]
        assert not any(row.record.kind == "op_prepared" for row in rows)
        assert _counts(store) == before
        assert effects == []
        assert resumed_llm.calls == 0
    finally:
        provider.close()


@pytest.mark.persistent_store
def test_same_prepared_id_with_patched_arguments_executes(store, database, tmp_path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "key-state.sqlite")
    trusted_store = wrap_fixture_store(store, provider)
    effects: list[str] = []

    @function_tool
    def echo(text: str) -> str:
        effects.append(text)
        return text

    try:
        agent = Agent("test", "test", tools=[echo], hooks=[_IdentityHook(change_id=False)])
        _seed(trusted_store, agent)
        llm = _ObservedLLM(
            [
                LLMResponse("", [ToolCall("original-id", "echo", {"text": "original"})]),
                LLMResponse("done"),
            ]
        )
        drive(trusted_store, SESSION_ID, runtime=_runtime(trusted_store, database, agent, llm))

        rows = store.read_state(SESSION_ID)[1]
        prepared = [row.record for row in rows if row.record.kind == "op_prepared"]
        assert len(prepared) == 1
        assert effects == ["patched"]
        assert llm.calls == 2
    finally:
        provider.close()
