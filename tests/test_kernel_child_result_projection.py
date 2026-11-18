from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from vv_agent.kernel._child_store import ChildCompletion
from vv_agent.kernel._driver_inputs import result_for_parent_copies
from vv_agent.kernel._execution_values import SubTaskOutcome
from vv_agent.kernel._records import Record
from vv_agent.kernel._tool_values import ToolExecutionResult
from vv_agent.session.kernel import _project_configured_child_results


def test_configured_parent_copy_uses_injected_result_projector() -> None:
    plan = cast(
        Record,
        SimpleNamespace(_payload={"request": {"id": "delegation-call", "arguments": {"agent_id": "worker"}}}),
    )
    completion = ChildCompletion("child", "child/turn/start", "completed", "child done", 7, "a" * 64, True)
    received: list[tuple[list[SubTaskOutcome], dict]] = []

    def project(
        received_plan: Record,
        outcomes: list[SubTaskOutcome],
        delegation: dict,
    ) -> ToolExecutionResult:
        assert received_plan is plan
        received.append((list(outcomes), delegation))
        return ToolExecutionResult("delegation-call", "projected")

    result = result_for_parent_copies(
        plan,
        [{"session_id": "child", "turn_id": "child/turn/start"}],
        [completion],
        {"mode": "configured", "agent_name": "worker", "metadata": {}},
        configured_projector=project,
    )

    assert result.content == "projected"
    assert len(received) == 1
    outcomes, delegation = received[0]
    assert outcomes[0].agent_name == "worker"
    assert outcomes[0].final_answer == "child done"
    assert delegation["mode"] == "configured"


@pytest.mark.parametrize(
    ("arguments", "handles", "completions"),
    [
        (
            {"agent_id": "worker", "task_description": "work"},
            [{"session_id": "child", "turn_id": "child/turn/start"}],
            [ChildCompletion("child", "child/turn/start", "completed", "child done", 7, "a" * 64, True)],
        ),
        (
            {"agent_id": "worker", "task_description": "work"},
            [{"session_id": "child", "turn_id": "child/turn/start"}],
            [ChildCompletion("child", "child/turn/start", "failed", "child failed", 7, "b" * 64, True)],
        ),
        (
            {"agent_id": "worker", "tasks": [{"task_description": "one"}, {"task_description": "two"}]},
            [
                {"session_id": "child-1", "turn_id": "child-1/turn/start"},
                {"session_id": "child-2", "turn_id": "child-2/turn/start"},
            ],
            [
                ChildCompletion("child-1", "child-1/turn/start", "completed", "one done", 7, "c" * 64, True),
                ChildCompletion("child-2", "child-2/turn/start", "failed", "two failed", 8, "d" * 64, True),
            ],
        ),
    ],
)
def test_direct_projection_matches_injected_create_sub_task_envelope(arguments, handles, completions) -> None:
    plan = cast(
        Record,
        SimpleNamespace(_payload={"request": {"id": "delegation-call", "arguments": arguments}}),
    )
    delegation = {"mode": "configured", "agent_name": "worker", "metadata": {}}

    direct = result_for_parent_copies(plan, handles, completions, delegation)
    injected = result_for_parent_copies(
        plan,
        handles,
        completions,
        delegation,
        configured_projector=_project_configured_child_results,
    )

    assert direct.to_dict() == injected.to_dict()
