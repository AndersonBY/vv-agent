from __future__ import annotations

import pytest

from v_agent.runtime.state import Checkpoint, InMemoryStateStore
from v_agent.runtime.stores.sqlite import SqliteStateStore
from v_agent.types import (
    AgentStatus,
    CycleRecord,
    Message,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)


def _make_checkpoint(task_id: str = "task-1", cycle_index: int = 3) -> Checkpoint:
    return Checkpoint(
        task_id=task_id,
        cycle_index=cycle_index,
        status=AgentStatus.RUNNING,
        messages=[
            Message(role="system", content="sys prompt"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ]),
        ],
        cycles=[
            CycleRecord(
                index=1,
                assistant_message="hi there",
                tool_calls=[ToolCall(id="c1", name="test", arguments={"key": "val"})],
                tool_results=[
                    ToolExecutionResult(
                        tool_call_id="c1",
                        content="result",
                        status="success",
                        status_code=ToolResultStatus.SUCCESS,
                    )
                ],
            )
        ],
        shared_state={"todo_list": [], "counter": 42},
    )


class TestSerializationRoundTrip:
    def test_message_roundtrip(self):
        msg = Message(role="assistant", content="hello", name="bot", reasoning_content="think")
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.name == msg.name
        assert restored.reasoning_content == msg.reasoning_content

    def test_tool_call_roundtrip(self):
        tc = ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        d = tc.to_dict()
        restored = ToolCall.from_dict(d)
        assert restored.id == tc.id
        assert restored.name == tc.name
        assert restored.arguments == tc.arguments

    def test_tool_execution_result_roundtrip(self):
        tr = ToolExecutionResult(
            tool_call_id="c1",
            content="ok",
            status="success",
            status_code=ToolResultStatus.SUCCESS,
            directive=ToolDirective.FINISH,
            error_code=None,
            metadata={"key": "val"},
        )
        d = tr.to_dict()
        restored = ToolExecutionResult.from_dict(d)
        assert restored.tool_call_id == tr.tool_call_id
        assert restored.content == tr.content
        assert restored.directive == tr.directive
        assert restored.metadata == tr.metadata

    def test_cycle_record_roundtrip(self):
        cr = CycleRecord(
            index=1,
            assistant_message="hi",
            tool_calls=[ToolCall(id="c1", name="test", arguments={})],
            tool_results=[ToolExecutionResult(tool_call_id="c1", content="ok")],
            memory_compacted=True,
        )
        d = cr.to_dict()
        restored = CycleRecord.from_dict(d)
        assert restored.index == cr.index
        assert restored.assistant_message == cr.assistant_message
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "test"
        assert len(restored.tool_results) == 1
        assert restored.memory_compacted is True


class TestInMemoryStateStore:
    def test_save_load(self):
        store = InMemoryStateStore()
        cp = _make_checkpoint()
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("task-1")
        assert loaded is not None
        assert loaded.task_id == "task-1"
        assert loaded.cycle_index == 3
        assert len(loaded.messages) == 3
        assert len(loaded.cycles) == 1

    def test_load_missing(self):
        store = InMemoryStateStore()
        assert store.load_checkpoint("nonexistent") is None

    def test_delete(self):
        store = InMemoryStateStore()
        store.save_checkpoint(_make_checkpoint())
        store.delete_checkpoint("task-1")
        assert store.load_checkpoint("task-1") is None

    def test_list_checkpoints(self):
        store = InMemoryStateStore()
        store.save_checkpoint(_make_checkpoint("a"))
        store.save_checkpoint(_make_checkpoint("b"))
        assert store.list_checkpoints() == ["a", "b"]

    def test_overwrite(self):
        store = InMemoryStateStore()
        store.save_checkpoint(_make_checkpoint("t", cycle_index=1))
        store.save_checkpoint(_make_checkpoint("t", cycle_index=5))
        loaded = store.load_checkpoint("t")
        assert loaded is not None
        assert loaded.cycle_index == 5


class TestSqliteStateStore:
    def test_save_load_roundtrip(self):
        store = SqliteStateStore(":memory:")
        cp = _make_checkpoint()
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("task-1")
        assert loaded is not None
        assert loaded.task_id == "task-1"
        assert loaded.cycle_index == 3
        assert loaded.status == AgentStatus.RUNNING
        assert len(loaded.messages) == 3
        assert loaded.messages[0].role == "system"
        assert loaded.messages[2].tool_calls is not None
        assert len(loaded.cycles) == 1
        assert loaded.cycles[0].tool_calls[0].name == "test"
        assert loaded.shared_state["counter"] == 42
        store.close()

    def test_load_missing(self):
        store = SqliteStateStore(":memory:")
        assert store.load_checkpoint("nope") is None
        store.close()

    def test_delete(self):
        store = SqliteStateStore(":memory:")
        store.save_checkpoint(_make_checkpoint())
        store.delete_checkpoint("task-1")
        assert store.load_checkpoint("task-1") is None
        store.close()

    def test_list_checkpoints(self):
        store = SqliteStateStore(":memory:")
        store.save_checkpoint(_make_checkpoint("x"))
        store.save_checkpoint(_make_checkpoint("y"))
        assert store.list_checkpoints() == ["x", "y"]
        store.close()

    def test_overwrite(self):
        store = SqliteStateStore(":memory:")
        store.save_checkpoint(_make_checkpoint("t", cycle_index=1))
        store.save_checkpoint(_make_checkpoint("t", cycle_index=7))
        loaded = store.load_checkpoint("t")
        assert loaded is not None
        assert loaded.cycle_index == 7
        store.close()
