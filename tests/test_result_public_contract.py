from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.result import RunResult
from vv_agent.types import AgentResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "result_public_v1.json"


def _contract() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _result() -> RunResult:
    contract = _contract()
    raw_result = AgentResult.from_dict(contract["agent_result"])
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="requested",
        selected_model="selected",
        model_id="model-id",
        endpoint_options=[
            EndpointOption(
                endpoint=EndpointConfig(
                    endpoint_id="endpoint-public",
                    api_key="secret-must-not-serialize",
                    api_base="https://example.invalid/v1",
                ),
                model_id="model-id",
            )
        ],
    )
    return RunResult(
        input="approve it",
        new_items=[],
        final_output=raw_result.wait_reason,
        status=raw_result.status,
        raw_result=raw_result,
        token_usage=raw_result.token_usage,
        trace_id="trace_1",
        run_id="run_1",
        metadata={"tenant": "acme"},
        agent_name="assistant",
        resolved_model=resolved,
    )


def test_approval_snapshot_and_state_match_shared_contract() -> None:
    contract = _contract()
    result = _result()

    assert [snapshot.to_dict() for snapshot in result.approvals] == contract["expected_approvals"]
    state = result.into_state()
    state.approve("approval_1")
    approved = [snapshot.to_dict() for snapshot in state.approvals]
    assert approved[0] == {**contract["expected_approvals"][0], "approved": True}
    assert state.pending_approval_ids() == ["approval_1"]
    assert state.approved_ids == ("approval_1",)


def test_run_result_public_projection_matches_shared_contract_without_credentials() -> None:
    contract = _contract()
    projection = _result().to_dict()

    assert sorted(projection) == contract["projection_keys"]
    assert projection["status"] == "wait_user"
    assert projection["final_output"] == "Approval is required."
    assert projection["resolved_model"] == contract["resolved_model_projection"]
    assert "secret-must-not-serialize" not in json.dumps(projection, sort_keys=True)

    result = _result()
    assert result.resolved_model is not None
    result.resolved_model.endpoint_options.clear()
    assert result.to_dict()["resolved_model"]["endpoint"] is None
