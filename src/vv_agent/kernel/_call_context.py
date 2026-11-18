"""Task/plan call facts evaluated at the caller's existing construction points."""

from __future__ import annotations

from typing import Any

from ._execution_values import AgentTask
from ._records import Record, copy_json
from ._tool_values import ToolCall


def approved_tool_call(plan: Record) -> ToolCall:
    return ToolCall.from_dict(copy_json(plan._payload["request"]))


class CompiledCallFacts:
    """Retain the checked source; defer facts separated by host construction."""

    __slots__ = ("_plan", "_source", "_task")

    def __init__(self, task: AgentTask, plan: Record):
        source = plan.operation_id if plan._payload["op_kind"] == "model" else plan._payload["dependencies"][0]
        assert source is not None
        self._task, self._plan, self._source = task, plan, source

    @property
    def cycle_index(self) -> int:
        return (
            self._plan._payload["request"]["metadata"]["vv_session"]["cycle_index"]
            if self._plan._payload["op_kind"] == "model"
            and "cycle_index" in self._plan._payload["request"].get("metadata", {}).get("vv_session", {})
            else int(self._source.rsplit("/", 1)[1])
        )

    @property
    def task_id(self) -> str:
        return self._plan.turn_id or self._task.task_id

    @property
    def idempotency_key(self) -> str | None:
        return self._plan._payload["idempotency_key"]

    def metadata(self, session_id: str | None) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "operation_id": self._plan.operation_id,
            "attempt": self._plan.attempt,
            "session_tool_names": [s["function"]["name"] for s in self._plan._payload["request"].get("tools", [])],
        }


def matches_preapproved_call(call: Any, *, tool_call_id: str, tool_name: str, arguments: dict[str, Any]) -> bool:
    return bool(
        call is not None
        and getattr(call, "id", None) == tool_call_id
        and getattr(call, "name", None) == tool_name
        and getattr(call, "arguments", None) == arguments
    )
