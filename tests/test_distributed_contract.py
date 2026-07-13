from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.runtime.backends.celery_tasks import run_single_cycle
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
from vv_agent.runtime.state import Checkpoint, CheckpointConflictError
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
    rust_copy = (
        Path(__file__).resolve().parents[2]
        / "vv-agent-rs"
        / "crates"
        / "vv-agent"
        / "tests"
        / "fixtures"
        / "parity"
        / FIXTURE_PATH.name
    )
    assert rust_copy.read_bytes() == FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == (
        "9d70267266a6f632b3c269c8ea91fcef477e15b26a5722ddd450b48d4d13b606"
    )


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


def test_distributed_worker_heartbeat_prevents_claim_theft_during_long_cycle(tmp_path: Path) -> None:
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

    def slow_step(_request):
        llm_started.set()
        time.sleep(0.15)
        return LLMResponse(content="done")

    llm_ref = CapabilityRef("llm.slow", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[slow_step]))
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
        lease_duration_ms=30,
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
    assert llm_started.wait(1)
    time.sleep(0.08)
    contender = SqliteStateStore(db_path)
    now_ms = time.time_ns() // 1_000_000
    with pytest.raises(CheckpointConflictError, match="already claimed"):
        contender.claim_checkpoint(
            task.task_id,
            1,
            claim_token="contender",
            lease_expires_at_ms=now_ms + 100,
            now_ms=now_ms,
        )
    worker_thread.join(2)

    assert not worker_thread.is_alive()
    assert "error" not in outcome
    assert outcome["result"]["result"]["status"] == "completed"
