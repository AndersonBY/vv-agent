from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import pytest

from vv_agent.budget import BudgetEnforcementBoundary, HostCost, RunBudgetLimits
from vv_agent.events import BudgetExhaustedEvent, BudgetSnapshotEvent, RunEvent
from vv_agent.llm import LlmRequest
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.runtime.backends.celery_tasks import (
    _lease_expiry_at,
    _lease_heartbeat_interval_seconds,
    _LeaseHeartbeat,
    run_single_cycle,
)
from vv_agent.runtime.backends.distributed import (
    DEFAULT_TOOLSET_SCHEMA_DIGEST,
    CapabilityRef,
    DistributedCapabilities,
    DistributedCapabilityError,
    DistributedCapabilityRegistry,
    DistributedContractError,
    DistributedRunEnvelope,
    DistributedToolPolicy,
    RuntimeRecipe,
    ToolsetRef,
    toolset_schema_digest,
)
from vv_agent.runtime.engine import AgentRuntime
from vv_agent.runtime.state import Checkpoint, CheckpointConflictError, InMemoryStateStore
from vv_agent.runtime.stores.sqlite import SqliteStateStore
from vv_agent.tools import build_default_registry
from vv_agent.types import (
    AgentStatus,
    AgentTask,
    LLMResponse,
    Message,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "distributed_run_envelope_v1.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _lease_lifecycle() -> dict[str, Any]:
    lifecycle = _fixture()["lease_lifecycle"]
    assert isinstance(lifecycle, dict)
    return lifecycle


def _worker_case(name: str) -> dict[str, Any]:
    cases = _lease_lifecycle()["worker_cases"]
    assert isinstance(cases, list)
    return next(case for case in cases if case["name"] == name)


def _set_path(payload: dict[str, Any], path: list[str], value: Any) -> None:
    target: Any = payload
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value


def test_distributed_envelope_fixture_round_trips_and_default_digest_matches() -> None:
    fixture = _fixture()
    canonical = fixture["canonical_envelope"]
    assert isinstance(canonical, dict)

    envelope = DistributedRunEnvelope.from_dict(canonical)

    assert envelope.to_dict() == canonical
    assert fixture["schema_version"] == envelope.schema_version
    assert fixture["default_toolset_schema_digest"] == DEFAULT_TOOLSET_SCHEMA_DIGEST
    assert toolset_schema_digest(build_default_registry()) == DEFAULT_TOOLSET_SCHEMA_DIGEST


def test_distributed_envelope_invalid_cases_match_shared_contract() -> None:
    fixture = _fixture()
    canonical = fixture["canonical_envelope"]
    assert isinstance(canonical, dict)
    invalid_cases = fixture["invalid_cases"]
    assert isinstance(invalid_cases, list)

    for case in invalid_cases:
        assert isinstance(case, dict)
        payload = copy.deepcopy(canonical)
        _set_path(payload, case["path"], case["value"])
        with pytest.raises(DistributedContractError, match=str(case["error"])):
            DistributedRunEnvelope.from_v1_dict(payload)


@pytest.mark.parametrize("field_name", ["deadline_unix_ms", "lease_duration_ms"])
def test_distributed_envelope_rejects_integers_beyond_u64(field_name: str) -> None:
    payload = copy.deepcopy(_fixture()["canonical_envelope"])
    payload[field_name] = 1 << 64

    with pytest.raises(DistributedContractError, match=field_name):
        DistributedRunEnvelope.from_dict(payload)


def test_distributed_capability_registry_fails_closed_for_unknown_reference() -> None:
    fixture = _fixture()["unknown_capability"]
    assert isinstance(fixture, dict)
    reference = CapabilityRef.from_dict(fixture["reference"])

    with pytest.raises(DistributedCapabilityError, match=str(fixture["error"])):
        DistributedCapabilityRegistry().resolve(fixture["kind"], reference)


def test_default_distributed_capabilities_resolve_without_hidden_fallbacks() -> None:
    registry = DistributedCapabilityRegistry()

    registry.validate(DistributedCapabilities())


def test_python_and_rust_distributed_fixture_copies_are_byte_identical() -> None:
    explicit_rust_root = os.environ.get("VV_AGENT_RS_REPO")
    rust_root = Path(explicit_rust_root or Path(__file__).resolve().parents[2] / "vv-agent-rs")
    rust_copy = rust_root / "crates" / "vv-agent" / "tests" / "fixtures" / "parity" / FIXTURE_PATH.name
    fixture_bytes = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == ("c1eb11591c93e8ac880fd4688cf06e0fe60a8b4522f7707ea13e1cccf40208e0")
    if explicit_rust_root is None:
        try:
            python_lock = json.loads((Path(__file__).resolve().parents[1] / "contract.lock.json").read_text())
            rust_lock = json.loads((rust_root / "contract.lock.json").read_text())
        except (OSError, json.JSONDecodeError):
            return
        locks_match = (
            python_lock.get("contract_version") == rust_lock.get("contract_version")
            and python_lock.get("contract_revision") == rust_lock.get("contract_revision")
        )
        if not rust_copy.exists() or not locks_match:
            return
    assert rust_copy.read_bytes() == fixture_bytes


def test_distributed_worker_reconstructs_custom_tool_policy_and_app_state(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "checkpoints.sqlite3")
    task = AgentTask(task_id="worker-custom", model="model-x", system_prompt="system", user_prompt="prompt")
    task.extra_tool_names.append("custom_probe")
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )

    tool_ran = Event()
    tools = build_default_registry()

    def custom_probe(context, _arguments):
        assert context.app_state == "tenant-7"
        tool_ran.set()
        return ToolExecutionResult(
            tool_call_id="",
            content="custom worker done",
            directive=ToolDirective.FINISH,
        )

    tools.register_tool("custom_probe", custom_probe, "Read worker app state and finish.")
    custom_toolset = ToolsetRef(
        id="toolset.project",
        version="7",
        schema_digest=toolset_schema_digest(tools),
    )
    llm_ref = CapabilityRef("llm.scripted", "1")
    app_state_ref = CapabilityRef("app.tenant", "7")
    predicate_ref = CapabilityRef("policy.project", "3")
    predicate_ran = Event()
    registry = DistributedCapabilityRegistry()
    registry.register_toolset(custom_toolset, tools)
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(
            steps=[
                LLMResponse(
                    content="run probe",
                    tool_calls=[ToolCall(id="probe-1", name="custom_probe", arguments={})],
                )
            ]
        ),
    )
    registry.register("app_state", app_state_ref, "tenant-7")

    def predicate(name: str, _arguments: dict[str, object]) -> bool:
        predicate_ran.set()
        return name == "custom_probe"

    registry.register("tool_predicate", predicate_ref, predicate)
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="backend-x",
        model="model-x",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(
            toolset_ref=custom_toolset,
            tool_policy=DistributedToolPolicy(
                allowed_tools=("custom_probe",),
                approval="never",
                predicate_ref=predicate_ref,
            ),
            llm_client_ref=llm_ref,
            app_state_ref=app_state_ref,
        ),
    )
    envelope = DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=recipe,
        cycle_index=1,
        run_id="run-worker-custom",
        deadline_unix_ms=2_000_000_000_000,
    )

    dispatch = run_single_cycle(envelope_dict=envelope.to_dict(), capability_registry=registry)

    assert dispatch["finished"] is True
    assert dispatch["result"]["status"] == "completed"
    assert dispatch["result"]["final_answer"] == "custom worker done"
    assert tool_ran.is_set()
    assert predicate_ran.is_set()
    terminal = store.load_checkpoint(task.task_id)
    assert terminal is not None and terminal.terminal_result is not None
    assert dispatch["checkpoint_revision"] == terminal.revision


def test_distributed_budget_usage_persists_and_blocks_the_next_worker_cycle(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "budget-checkpoints.sqlite3")
    task = AgentTask(
        task_id="worker-budget",
        model="model-x",
        system_prompt="system",
        user_prompt="prompt",
        no_tool_policy="continue",
    )
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )

    model_calls = 0

    def first_cycle(_request: LlmRequest) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(
            content="need another cycle",
            raw={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 0},
                }
            },
        )

    llm_ref = CapabilityRef("llm.budget", "1")
    event_ref = CapabilityRef("events.budget", "1")
    events: list[RunEvent] = []
    registry = DistributedCapabilityRegistry()
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[first_cycle]))
    registry.register("event_sink", event_ref, events.append)
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="backend-x",
        model="model-x",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            event_sink_ref=event_ref,
        ),
    )
    limits = RunBudgetLimits(max_total_tokens=10)

    first = run_single_cycle(
        envelope_dict=DistributedRunEnvelope.for_cycle(
            task=task,
            recipe=recipe,
            cycle_index=1,
            run_id="run-worker-budget",
            deadline_unix_ms=2_000_000_000_000,
            budget_limits=limits,
        ).to_dict(),
        capability_registry=registry,
    )

    assert first == {"finished": False}
    checkpoint = store.load_checkpoint(task.task_id)
    assert checkpoint is not None and checkpoint.budget_usage is not None
    assert checkpoint.budget_usage.cycles == 1
    assert checkpoint.budget_usage.total_tokens == 10

    second = run_single_cycle(
        envelope_dict=DistributedRunEnvelope.for_cycle(
            task=task,
            recipe=recipe,
            cycle_index=2,
            run_id="run-worker-budget",
            deadline_unix_ms=2_000_000_000_000,
            budget_limits=limits,
        ).to_dict(),
        capability_registry=registry,
    )

    assert second["finished"] is True
    assert second["result"]["status"] == "failed"
    assert second["result"]["completion_reason"] == "budget_exhausted"
    assert second["result"]["budget_exhaustion"]["enforcement_boundary"] == "cycle_start"
    assert model_calls == 1
    budget_events = [event for event in events if isinstance(event, (BudgetSnapshotEvent, BudgetExhaustedEvent))]
    initial, *elapsed_updates, llm_update, exhausted = budget_events
    assert isinstance(initial, BudgetSnapshotEvent)
    assert initial.enforcement_boundary is BudgetEnforcementBoundary.RUN_START
    assert initial.budget_usage.cycles == 0
    assert initial.budget_usage.total_tokens == 0
    assert len(elapsed_updates) <= 1
    assert all(isinstance(event, BudgetSnapshotEvent) for event in elapsed_updates)
    assert all(event.enforcement_boundary is BudgetEnforcementBoundary.CYCLE_START for event in elapsed_updates)
    assert all(event.budget_usage.cycles == 0 for event in elapsed_updates)
    assert all(event.budget_usage.total_tokens == 0 for event in elapsed_updates)
    assert isinstance(llm_update, BudgetSnapshotEvent)
    assert llm_update.enforcement_boundary is BudgetEnforcementBoundary.LLM_COMPLETE
    assert llm_update.budget_usage.cycles == 1
    assert llm_update.budget_usage.total_tokens == 10
    assert isinstance(exhausted, BudgetExhaustedEvent)
    assert exhausted.enforcement_boundary is BudgetEnforcementBoundary.CYCLE_START
    assert exhausted.budget_usage.cycles == 1
    assert exhausted.budget_usage.total_tokens == 10
    elapsed_observations = [event.budget_usage.elapsed_ms for event in budget_events]
    assert elapsed_observations == sorted(elapsed_observations)


def test_distributed_worker_resolves_host_cost_meter_and_reports_overshoot(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "host-cost-checkpoints.sqlite3")
    task = AgentTask(
        task_id="worker-host-cost",
        model="model-x",
        system_prompt="system",
        user_prompt="prompt",
        no_tool_policy="finish",
    )
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )

    class Meter:
        def __init__(self) -> None:
            self.readings = iter(
                [
                    HostCost(unit="credits", amount_microunits=0),
                    HostCost(unit="credits", amount_microunits=0),
                    HostCost(unit="credits", amount_microunits=120),
                ]
            )
            self.last = HostCost(unit="credits", amount_microunits=120)

        def read(self) -> HostCost:
            self.last = next(self.readings, self.last)
            return self.last

    llm_ref = CapabilityRef("llm.host-cost", "1")
    meter_ref = CapabilityRef("cost.run", "1")
    registry = DistributedCapabilityRegistry()
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(
            steps=[
                LLMResponse(
                    content="costly response",
                    raw={
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        }
                    },
                )
            ]
        ),
    )
    registry.register("host_cost_meter", meter_ref, Meter())
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="backend-x",
        model="model-x",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            host_cost_meter_ref=meter_ref,
        ),
    )

    dispatch = run_single_cycle(
        envelope_dict=DistributedRunEnvelope.for_cycle(
            task=task,
            recipe=recipe,
            cycle_index=1,
            run_id="run-worker-host-cost",
            deadline_unix_ms=2_000_000_000_000,
            budget_limits=RunBudgetLimits(
                max_host_cost=HostCost(unit="credits", amount_microunits=100),
            ),
        ).to_dict(),
        capability_registry=registry,
    )

    assert dispatch["finished"] is True
    assert dispatch["result"]["completion_reason"] == "budget_exhausted"
    exhaustion = dispatch["result"]["budget_exhaustion"]
    assert exhaustion["dimension"] == "host_cost"
    assert exhaustion["observed"] == 120
    assert exhaustion["overshoot"] == 20
    assert exhaustion["enforcement_boundary"] == "llm_complete"


def test_distributed_worker_resolves_every_capability_before_claiming_checkpoint(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "state.sqlite3")
    task = AgentTask(task_id="worker-fail-closed", model="model", system_prompt="system", user_prompt="prompt")
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "missing-settings.py"),
        backend="missing",
        model="missing",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(hook_refs=(CapabilityRef("hook.missing", "1"),)),
    )
    envelope = DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=recipe,
        cycle_index=1,
        deadline_unix_ms=2_000_000_000_000,
    )

    with pytest.raises(
        DistributedCapabilityError,
        match=re.escape("unknown distributed capability hook hook.missing@1"),
    ):
        run_single_cycle(envelope_dict=envelope.to_dict())

    checkpoint = store.load_checkpoint(task.task_id)
    assert checkpoint is not None
    assert checkpoint.revision == 0
    assert checkpoint.claim_token is None


def test_initial_renewal_failure_prevents_cycle_model_tool_and_commit_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _worker_case("initial_renewal_failure_has_no_side_effects")
    expected = case["expected"]
    calls = {"cycle": 0, "model": 0, "tool": 0, "commit": 0}
    store = SqliteStateStore(tmp_path / "initial-renewal.sqlite3")
    task = AgentTask(
        task_id="worker-initial-renewal-failed",
        model="model",
        system_prompt="system",
        user_prompt="prompt",
    )
    task.extra_tool_names.append("side_effect_probe")
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )

    def model_step(_request: Any) -> LLMResponse:
        calls["model"] += 1
        return LLMResponse(
            content="run probe",
            tool_calls=[ToolCall(id="probe-1", name="side_effect_probe", arguments={})],
        )

    def side_effect_probe(_context: Any, _arguments: dict[str, Any]) -> ToolExecutionResult:
        calls["tool"] += 1
        return ToolExecutionResult(tool_call_id="", content="unexpected")

    tools = build_default_registry()
    tools.register_tool("side_effect_probe", side_effect_probe, "Record an unexpected side effect.")
    toolset = ToolsetRef(id="toolset.renewal-failure", version="1", schema_digest=toolset_schema_digest(tools))
    llm_ref = CapabilityRef("llm.renewal-failure", "1")
    registry = DistributedCapabilityRegistry()
    registry.register_toolset(toolset, tools)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[model_step]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused.py"),
        backend="test",
        model="model",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(toolset_ref=toolset, llm_client_ref=llm_ref),
    )
    envelope = DistributedRunEnvelope.for_cycle(task=task, recipe=recipe, cycle_index=1)

    original_build_cycle_executor = AgentRuntime._build_cycle_executor
    original_commit = SqliteStateStore.commit_checkpoint

    def build_counted_cycle_executor(self: AgentRuntime, *args: Any, **kwargs: Any) -> Any:
        executor = original_build_cycle_executor(self, *args, **kwargs)

        def counted_executor(*executor_args: Any, **executor_kwargs: Any) -> Any:
            calls["cycle"] += 1
            return executor(*executor_args, **executor_kwargs)

        return counted_executor

    def fail_initial_renewal(self: SqliteStateStore, *args: Any, **kwargs: Any) -> bool:
        return False

    def count_commit(self: SqliteStateStore, *args: Any, **kwargs: Any) -> bool:
        calls["commit"] += 1
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(AgentRuntime, "_build_cycle_executor", build_counted_cycle_executor)
    monkeypatch.setattr(SqliteStateStore, "renew_checkpoint_claim", fail_initial_renewal)
    monkeypatch.setattr(SqliteStateStore, "commit_checkpoint", count_commit)

    with pytest.raises(CheckpointConflictError, match=re.escape(expected["outcome"])):
        run_single_cycle(envelope_dict=envelope.to_dict(), capability_registry=registry)

    assert calls == {
        "cycle": expected["cycle_calls"],
        "model": expected["model_calls"],
        "tool": expected["tool_calls"],
        "commit": expected["commit_calls"],
    }


def test_operation_unwind_stops_heartbeat_without_committing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _worker_case("operation_unwind_stops_heartbeat")
    expected = case["expected"]
    store = SqliteStateStore(tmp_path / "operation-unwind.sqlite3")
    task = AgentTask(
        task_id="worker-operation-unwind",
        model="model",
        system_prompt="system",
        user_prompt="prompt",
    )
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )

    def unwind(_request: Any) -> LLMResponse:
        raise KeyboardInterrupt("operation unwind")

    llm_ref = CapabilityRef("llm.operation-unwind", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[unwind]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused.py"),
        backend="test",
        model="model",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(llm_client_ref=llm_ref),
    )
    envelope = DistributedRunEnvelope.for_cycle(task=task, recipe=recipe, cycle_index=1)
    stopped_heartbeats: list[_LeaseHeartbeat] = []
    commit_calls = 0
    original_stop = _LeaseHeartbeat.stop
    original_commit = SqliteStateStore.commit_checkpoint

    def recording_stop(self: _LeaseHeartbeat) -> None:
        original_stop(self)
        stopped_heartbeats.append(self)

    def count_commit(self: SqliteStateStore, *args: Any, **kwargs: Any) -> bool:
        nonlocal commit_calls
        commit_calls += 1
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(_LeaseHeartbeat, "stop", recording_stop)
    monkeypatch.setattr(SqliteStateStore, "commit_checkpoint", count_commit)

    with pytest.raises(KeyboardInterrupt, match="operation unwind"):
        run_single_cycle(envelope_dict=envelope.to_dict(), capability_registry=registry)

    assert len(stopped_heartbeats) == 1
    assert not stopped_heartbeats[0]._thread.is_alive()
    assert expected["renewals_after_stop"] == 0
    assert commit_calls == expected["commit_calls"]
    assert expected["outcome"] == "unwind"


def test_distributed_worker_heartbeat_prevents_claim_theft_through_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _worker_case("commit_barrier_keeps_heartbeat_active")
    expected = case["expected"]
    lease_duration_ms = 3_000
    db_path = tmp_path / "heartbeat.sqlite3"
    store = SqliteStateStore(db_path)
    task = AgentTask(
        task_id="worker-heartbeat",
        model="model",
        system_prompt="system",
        user_prompt="prompt",
        no_tool_policy="finish",
    )
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )
    llm_started = Event()
    release_llm = Event()
    commit_started = Event()
    release_commit = Event()
    commit_calls = 0
    renewals_during_commit = 0

    def slow_step(_request):
        llm_started.set()
        assert release_llm.wait(30), "release slow LLM"
        return LLMResponse(content="done")

    llm_ref = CapabilityRef("llm.slow", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[slow_step]))
    original_commit = SqliteStateStore.commit_checkpoint
    original_renew = SqliteStateStore.renew_checkpoint_claim

    def blocking_commit(self, checkpoint, *, claim_token, expected_revision):
        nonlocal commit_calls
        commit_calls += 1
        commit_started.set()
        assert release_commit.wait(30), "release checkpoint commit"
        return original_commit(
            self,
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )

    def counting_renewal(self, *args, **kwargs):
        nonlocal renewals_during_commit
        renewed = original_renew(self, *args, **kwargs)
        if renewed and commit_started.is_set() and not release_commit.is_set():
            renewals_during_commit += 1
        return renewed

    monkeypatch.setattr(SqliteStateStore, "commit_checkpoint", blocking_commit)
    monkeypatch.setattr(SqliteStateStore, "renew_checkpoint_claim", counting_renewal)
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused.py"),
        backend="test",
        model="model",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(llm_client_ref=llm_ref),
    )
    envelope = DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=recipe,
        cycle_index=1,
        deadline_unix_ms=2_000_000_000_000,
        lease_duration_ms=lease_duration_ms,
    )
    outcome: dict[str, Any] = {}

    def run_worker() -> None:
        try:
            outcome["result"] = run_single_cycle(
                envelope_dict=envelope.to_dict(),
                capability_registry=registry,
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker_thread = Thread(target=run_worker)
    worker_thread.start()
    try:
        assert llm_started.wait(30)
        contender = SqliteStateStore(db_path)

        claimed_checkpoint = contender.load_checkpoint(task.task_id)
        assert claimed_checkpoint is not None
        assert claimed_checkpoint.lease_expires_at_ms is not None

        def wait_for_lease_after(previous_expiry: int) -> int:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                checkpoint = contender.load_checkpoint(task.task_id)
                assert checkpoint is not None
                assert checkpoint.lease_expires_at_ms is not None
                if checkpoint.lease_expires_at_ms > previous_expiry:
                    return checkpoint.lease_expires_at_ms
                time.sleep(0.001)
            raise AssertionError(f"heartbeat did not extend lease beyond {previous_expiry}")

        first_renewed_expiry = wait_for_lease_after(claimed_checkpoint.lease_expires_at_ms)
        with pytest.raises(CheckpointConflictError, match="already claimed"):
            contender.claim_checkpoint(
                task.task_id,
                1,
                claim_token="contender",
                lease_expires_at_ms=first_renewed_expiry + lease_duration_ms,
                now_ms=claimed_checkpoint.lease_expires_at_ms,
            )
        release_llm.set()
        assert commit_started.wait(30)
        commit_phase_expiry = contender.load_checkpoint(task.task_id)
        assert commit_phase_expiry is not None
        assert commit_phase_expiry.lease_expires_at_ms is not None
        renewed_during_commit = wait_for_lease_after(commit_phase_expiry.lease_expires_at_ms)
        assert expected["periodic_renewals_during_commit_min"] == 1
        with pytest.raises(CheckpointConflictError, match="already claimed"):
            contender.claim_checkpoint(
                task.task_id,
                1,
                claim_token="commit-contender",
                lease_expires_at_ms=renewed_during_commit + lease_duration_ms,
                now_ms=commit_phase_expiry.lease_expires_at_ms,
            )
    finally:
        release_llm.set()
        release_commit.set()
        worker_thread.join(30)

    assert not worker_thread.is_alive()
    assert "error" not in outcome
    assert expected["contender_claimed"] is False
    assert commit_calls == expected["commit_calls"]
    assert renewals_during_commit >= expected["periodic_renewals_during_commit_min"]
    assert expected["outcome"] == "success"
    assert outcome["result"]["result"]["status"] == "completed"


def _claimed_heartbeat(
    tmp_path: Path,
    *,
    task_id: str,
    initial_lease_ms: int,
    heartbeat_lease_ms: int = 30_000,
    deadline_unix_ms: int | None = None,
) -> tuple[InMemoryStateStore, AgentTask, Checkpoint, DistributedRunEnvelope, int]:
    store = InMemoryStateStore()
    task = AgentTask(task_id=task_id, model="model", system_prompt="system", user_prompt="prompt")
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )
    now_ms = time.time_ns() // 1_000_000
    initial_expiry = now_ms + initial_lease_ms
    claimed = store.claim_checkpoint(
        task.task_id,
        1,
        claim_token="owner",
        lease_expires_at_ms=initial_expiry,
        now_ms=now_ms,
    )
    assert claimed is not None
    envelope = DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=RuntimeRecipe(
            settings_file=str(tmp_path / "unused.py"),
            backend="test",
            model="model",
            workspace=str(tmp_path / "workspace"),
        ),
        cycle_index=1,
        deadline_unix_ms=deadline_unix_ms,
        lease_duration_ms=heartbeat_lease_ms,
    )
    return store, task, claimed, envelope, initial_expiry


def test_lease_heartbeat_renews_before_start_returns(tmp_path: Path) -> None:
    case = _worker_case("initial_renewal_precedes_operation")
    store, task, claimed, envelope, initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-ready",
        initial_lease_ms=10_000,
    )
    heartbeat = _LeaseHeartbeat(
        store=store,
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    heartbeat.start()
    try:
        renewed = store.load_checkpoint(task.task_id)
        assert renewed is not None
        assert renewed.lease_expires_at_ms is not None
        assert renewed.lease_expires_at_ms >= initial_expiry
        assert heartbeat._known_lease_expires_at_ms == renewed.lease_expires_at_ms
        assert case["expected"]["operation_calls"] == 1
    finally:
        heartbeat.stop()


def test_lease_heartbeat_initial_failure_prevents_start(tmp_path: Path) -> None:
    case = _worker_case("initial_renewal_failure_has_no_side_effects")
    store, _task, claimed, envelope, _initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-failed",
        initial_lease_ms=3_000,
    )
    heartbeat = _LeaseHeartbeat(
        store=store,
        envelope=envelope,
        claim_token="wrong-owner",
        expected_revision=claimed.revision,
    )

    with pytest.raises(CheckpointConflictError, match=re.escape(case["expected"]["outcome"])):
        heartbeat.start()
    heartbeat.stop()
    heartbeat.stop()


@pytest.mark.parametrize("renewed", [True, False])
def test_lease_heartbeat_delayed_result_after_new_expiry_prevents_start(tmp_path: Path, renewed: bool) -> None:
    store, _task, claimed, envelope, _initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-delayed-success",
        initial_lease_ms=30_000,
        heartbeat_lease_ms=1,
    )

    class DelayedRenewStore:
        def load_checkpoint(self, task_id: str) -> Checkpoint | None:
            return store.load_checkpoint(task_id)

        def renew_checkpoint_claim(self, *_args: Any, **_kwargs: Any) -> bool:
            time.sleep(0.02)
            return renewed

    heartbeat = _LeaseHeartbeat(
        store=DelayedRenewStore(),
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    with pytest.raises(CheckpointConflictError, match="heartbeat failed: claim lease expired"):
        heartbeat.start()
    assert heartbeat._thread.ident is None
    heartbeat.stop()


@pytest.mark.parametrize("renewed", [True, False])
def test_lease_heartbeat_delayed_result_after_known_expiry_prevents_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renewed: bool,
) -> None:
    store, _task, claimed, envelope, initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id=f"heartbeat-known-expiry-{renewed}",
        initial_lease_ms=1,
        heartbeat_lease_ms=30_000,
    )
    monkeypatch.setattr(
        "vv_agent.runtime.backends.celery_tasks.time.time_ns",
        lambda: (initial_expiry - 1) * 1_000_000,
    )

    class DelayedRenewStore:
        def load_checkpoint(self, task_id: str) -> Checkpoint | None:
            return store.load_checkpoint(task_id)

        def renew_checkpoint_claim(self, *_args: Any, **_kwargs: Any) -> bool:
            time.sleep(0.02)
            return renewed

    heartbeat = _LeaseHeartbeat(
        store=DelayedRenewStore(),
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    with pytest.raises(CheckpointConflictError, match="heartbeat failed: claim lease expired"):
        heartbeat.start()
    assert heartbeat._thread.ident is None
    heartbeat.stop()


def test_successful_commit_suppresses_claim_consumed_renewal_error(tmp_path: Path) -> None:
    case = _worker_case("successful_commit_beats_inflight_renewal_rejection")
    store, task, claimed, envelope, _initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-commit-race",
        initial_lease_ms=30_000,
    )
    heartbeat = _LeaseHeartbeat(
        store=store,
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    heartbeat.begin_commit()
    checkpoint = store.load_checkpoint(task.task_id)
    assert checkpoint is not None
    checkpoint.cycle_index = 1
    assert store.commit_checkpoint(
        checkpoint,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    heartbeat.mark_commit_succeeded()
    heartbeat._record_error(
        CheckpointConflictError("claim is no longer active"),
        renew_started_during_commit=True,
    )

    heartbeat.raise_if_failed()
    assert case["expected"]["durable_commit"] is True
    assert case["expected"]["heartbeat_error_suppressed"] is True
    heartbeat.stop()


def test_successful_commit_never_suppresses_store_reported_lease_expiry(tmp_path: Path) -> None:
    store, _task, claimed, envelope, _initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-store-reported-expiry",
        initial_lease_ms=30_000,
    )
    heartbeat = _LeaseHeartbeat(
        store=store,
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    heartbeat.begin_commit()
    heartbeat.mark_commit_succeeded()
    heartbeat._record_error(
        CheckpointConflictError("claim lease expired"),
        renew_started_during_commit=True,
    )

    with pytest.raises(CheckpointConflictError, match="heartbeat failed: claim lease expired"):
        heartbeat.raise_if_failed()
    heartbeat.stop()


def test_failure_observed_before_commit_cannot_be_suppressed(tmp_path: Path) -> None:
    store, _task, claimed, envelope, _initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-precommit-failure",
        initial_lease_ms=30_000,
    )
    heartbeat = _LeaseHeartbeat(
        store=store,
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    heartbeat._record_error(
        CheckpointConflictError("claim is no longer active"),
        renew_started_during_commit=False,
    )

    with pytest.raises(CheckpointConflictError, match="heartbeat failed: claim is no longer active"):
        heartbeat.begin_commit()
    heartbeat.stop()


def test_renewal_started_before_commit_cannot_be_suppressed_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, task, claimed, envelope, initial_expiry = _claimed_heartbeat(
        tmp_path,
        task_id="heartbeat-renew-before-commit",
        initial_lease_ms=30_000,
    )
    renewal_started = Event()
    release_renewal = Event()

    class BlockingHeartbeatStore:
        def load_checkpoint(self, task_id: str) -> Checkpoint | None:
            return store.load_checkpoint(task_id)

        def renew_checkpoint_claim(
            self,
            task_id: str,
            *,
            claim_token: str,
            expected_revision: int,
            lease_expires_at_ms: int,
            now_ms: int,
        ) -> bool:
            renewal_started.set()
            assert release_renewal.wait(5), "release pre-commit heartbeat renewal"
            return store.renew_checkpoint_claim(
                task_id,
                claim_token=claim_token,
                expected_revision=expected_revision,
                lease_expires_at_ms=lease_expires_at_ms,
                now_ms=now_ms,
            )

    monkeypatch.setattr(
        "vv_agent.runtime.backends.celery_tasks.time.time_ns",
        lambda: (initial_expiry - 1) * 1_000_000,
    )
    heartbeat = _LeaseHeartbeat(
        store=BlockingHeartbeatStore(),
        envelope=envelope,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    heartbeat._load_claim_lease_expiry()
    renewal_errors: list[BaseException] = []

    def renew_once() -> None:
        try:
            heartbeat._renew_once()
        except BaseException as exc:
            renewal_errors.append(exc)

    renewal = Thread(target=renew_once)
    renewal.start()
    try:
        assert renewal_started.wait(5)
        heartbeat.begin_commit()
        checkpoint = store.load_checkpoint(task.task_id)
        assert checkpoint is not None
        checkpoint.cycle_index = 1
        assert store.commit_checkpoint(
            checkpoint,
            claim_token="owner",
            expected_revision=claimed.revision,
        )
        heartbeat.mark_commit_succeeded()
        release_renewal.set()
    finally:
        release_renewal.set()
        renewal.join(5)

    assert not renewal.is_alive()
    assert len(renewal_errors) == 1
    assert isinstance(renewal_errors[0], CheckpointConflictError)
    assert str(renewal_errors[0]) == "claim is no longer active"
    with pytest.raises(CheckpointConflictError, match="heartbeat failed: claim is no longer active"):
        heartbeat.raise_if_failed()
    heartbeat.stop()


def test_expired_renewal_during_blocked_commit_remains_coordination_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_duration_ms = 3_000
    clock_ms = 1_000
    clock_lock = Lock()

    def current_time_ns() -> int:
        with clock_lock:
            return clock_ms * 1_000_000

    monkeypatch.setattr("vv_agent.runtime.backends.celery_tasks.time.time_ns", current_time_ns)
    db_path = tmp_path / "expired-during-commit.sqlite3"
    store = SqliteStateStore(db_path)
    task = AgentTask(
        task_id="worker-expired-during-commit",
        model="model",
        system_prompt="system",
        user_prompt="prompt",
        no_tool_policy="finish",
    )
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
            cycles=[],
        )
    )
    llm_ref = CapabilityRef("llm.expired-during-commit", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="done")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused.py"),
        backend="test",
        model="model",
        workspace=str(tmp_path / "workspace"),
        state_store=store.state_store_spec(),
        capabilities=DistributedCapabilities(llm_client_ref=llm_ref),
    )
    envelope = DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=recipe,
        cycle_index=1,
        lease_duration_ms=lease_duration_ms,
    )
    commit_started = Event()
    release_commit = Event()
    commit_succeeded = Event()
    renewal_started = Event()
    release_renewal = Event()
    heartbeat_error_recorded = Event()
    renewal_call_times: list[int] = []
    commit_calls = 0
    original_commit = SqliteStateStore.commit_checkpoint
    original_renew = SqliteStateStore.renew_checkpoint_claim
    original_record_error = _LeaseHeartbeat._record_error

    def blocking_commit(self, checkpoint, *, claim_token, expected_revision):
        nonlocal commit_calls
        commit_calls += 1
        commit_started.set()
        assert release_commit.wait(5), "release durable commit"
        committed = original_commit(
            self,
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if committed:
            commit_succeeded.set()
        return committed

    def blocking_renewal(self, *args, **kwargs):
        if commit_started.is_set():
            renewal_call_times.append(kwargs["now_ms"])
            renewal_started.set()
            assert release_renewal.wait(5), "release blocked heartbeat renewal"
            return False
        return original_renew(self, *args, **kwargs)

    def recording_heartbeat_error(self, error, *, renew_started_during_commit):
        original_record_error(
            self,
            error,
            renew_started_during_commit=renew_started_during_commit,
        )
        if isinstance(error, CheckpointConflictError) and str(error) == "claim lease expired":
            heartbeat_error_recorded.set()

    monkeypatch.setattr(SqliteStateStore, "commit_checkpoint", blocking_commit)
    monkeypatch.setattr(SqliteStateStore, "renew_checkpoint_claim", blocking_renewal)
    monkeypatch.setattr(_LeaseHeartbeat, "_record_error", recording_heartbeat_error)
    outcome: dict[str, Any] = {}

    def run_worker() -> None:
        try:
            outcome["result"] = run_single_cycle(
                envelope_dict=envelope.to_dict(),
                capability_registry=registry,
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = Thread(target=run_worker)
    worker.start()
    try:
        assert commit_started.wait(5)
        assert renewal_started.wait(5)
        assert renewal_call_times == [1_000]
        with clock_lock:
            clock_ms = 1_000 + lease_duration_ms
        release_renewal.set()
        assert heartbeat_error_recorded.wait(5)
        release_commit.set()
        assert commit_succeeded.wait(5)
    finally:
        release_commit.set()
        release_renewal.set()
        worker.join(5)

    assert not worker.is_alive()
    assert commit_calls == 1
    assert "result" not in outcome
    error = outcome.get("error")
    assert isinstance(error, CheckpointConflictError)
    assert str(error) == "checkpoint lease heartbeat failed: claim lease expired"
    durable = store.load_checkpoint(task.task_id)
    assert durable is not None
    assert durable.claim_token is None
    assert durable.terminal_result is not None
    assert durable.terminal_result.status is AgentStatus.COMPLETED


def test_lease_expiry_cases_match_shared_contract() -> None:
    for case in _lease_lifecycle()["expiry_cases"]:
        arguments = {
            "now_ms": case["now_ms"],
            "lease_duration_ms": case["lease_duration_ms"],
            "deadline_unix_ms": case["deadline_unix_ms"],
        }
        if case["expected_error"] is None:
            assert _lease_expiry_at(**arguments) == case["expected_expiry_ms"]
        else:
            with pytest.raises(OverflowError, match=re.escape(case["expected_error"])):
                _lease_expiry_at(**arguments)


def test_deadline_clamped_lease_drives_heartbeat_interval() -> None:
    case = _lease_lifecycle()["expiry_cases"][0]
    expiry_ms = _lease_expiry_at(
        now_ms=case["now_ms"],
        lease_duration_ms=case["lease_duration_ms"],
        deadline_unix_ms=case["deadline_unix_ms"],
    )
    effective_lease_ms = expiry_ms - case["now_ms"]

    assert 0 < _lease_heartbeat_interval_seconds(effective_lease_ms) < effective_lease_ms / 1_000


def test_lease_heartbeat_intervals_match_shared_contract(tmp_path: Path) -> None:
    for lease_duration_ms in _lease_lifecycle()["interval_lease_ms_cases"]:
        store, _task, claimed, envelope, _initial_expiry = _claimed_heartbeat(
            tmp_path,
            task_id=f"heartbeat-interval-{lease_duration_ms}",
            initial_lease_ms=30_000,
            heartbeat_lease_ms=lease_duration_ms,
        )
        heartbeat = _LeaseHeartbeat(
            store=store,
            envelope=envelope,
            claim_token="owner",
            expected_revision=claimed.revision,
        )

        assert 0 < heartbeat._interval_seconds < lease_duration_ms / 1_000
        heartbeat.stop()


def test_claim_expiry_boundary_matches_shared_contract(tmp_path: Path) -> None:
    case = _lease_lifecycle()["claim_boundary_case"]
    stores = [InMemoryStateStore(), SqliteStateStore(tmp_path / "claim-boundary.sqlite3")]
    for index, store in enumerate(stores):
        task_id = f"claim-boundary-{index}"
        assert store.create_checkpoint(
            Checkpoint(
                task_id=task_id,
                cycle_index=0,
                status=AgentStatus.RUNNING,
                messages=[Message(role="system", content="system"), Message(role="user", content="prompt")],
                cycles=[],
            )
        )
        owner = store.claim_checkpoint(
            task_id,
            1,
            claim_token="owner",
            lease_expires_at_ms=case["lease_expires_at_ms"],
            now_ms=case["claim_now_ms"],
        )
        assert owner is not None
        renewed = store.renew_checkpoint_claim(
            task_id,
            claim_token="owner",
            expected_revision=owner.revision,
            lease_expires_at_ms=case["boundary_now_ms"] + 100,
            now_ms=case["boundary_now_ms"],
        )
        assert renewed is case["owner_renewed"]
        contender = store.claim_checkpoint(
            task_id,
            1,
            claim_token="contender",
            lease_expires_at_ms=case["boundary_now_ms"] + 100,
            now_ms=case["boundary_now_ms"],
        )
        assert (contender is not None) is case["contender_reclaimed"]
