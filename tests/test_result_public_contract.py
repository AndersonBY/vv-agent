from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.result import RunResult
from vv_agent.types import AgentResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "result_public.json"


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
    assert projection["token_usage"] == contract["agent_result"]["token_usage"]
    assert projection["resolved_model"] == contract["resolved_model_projection"]
    assert "secret-must-not-serialize" not in json.dumps(projection, sort_keys=True)

    result = _result()
    assert result.resolved_model is not None
    result.resolved_model.endpoint_options.clear()
    assert result.to_dict()["resolved_model"]["endpoint"] is None


def test_agent_result_reader_enforces_the_closed_current_wire() -> None:
    contract = _contract()
    raw = contract["agent_result"]
    wire = contract["agent_result_wire"]

    assert AgentResult.from_dict(raw).to_dict() == raw
    for field_name in wire["required_fields"]:
        invalid = deepcopy(raw)
        del invalid[field_name]
        with pytest.raises((KeyError, TypeError, ValueError), match="AgentResult"):
            AgentResult.from_dict(invalid)

    for field_name in wire["optional_fields"]:
        invalid = {**deepcopy(raw), field_name: None}
        with pytest.raises(ValueError, match="must be omitted"):
            AgentResult.from_dict(invalid)

    with pytest.raises(ValueError, match="unknown"):
        AgentResult.from_dict({**deepcopy(raw), "legacy": True})
