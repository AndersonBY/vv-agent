from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from support import FactoryModelProvider

from vv_agent import (
    AfterCycleDecision,
    AfterCycleSnapshot,
    Agent,
    CheckpointConfig,
    MemorySession,
    RunConfig,
    Runner,
    ToolPolicy,
)
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalProvider, ApprovalRequest
from vv_agent.budget import HostCost, RunBudgetLimits
from vv_agent.checkpoint import AmbiguousModelPolicy, AmbiguousToolPolicy, CheckpointError, ResumePolicy, ToolIdempotency
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime.backends.celery import CeleryBackend, register_cycle_task
from vv_agent.runtime.backends.celery_tasks import run_single_cycle
from vv_agent.runtime.backends.distributed import (
    DISTRIBUTED_RUN_SCHEMA_VERSION,
    DISTRIBUTED_WORKER_RESPONSE_SCHEMA_VERSION,
    CapabilityRef,
    CheckpointExtensionRef,
    DistributedAdvanceDecision,
    DistributedCapabilities,
    DistributedCapabilityError,
    DistributedCapabilityRegistry,
    DistributedCheckpointConfig,
    DistributedContractError,
    DistributedDeliveryOutcome,
    DistributedRunEnvelope,
    DistributedRunHandle,
    DistributedToolPolicy,
    DistributedWorkerResponse,
    RuntimeRecipe,
    ToolsetRef,
    toolset_schema_digest,
)
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.controller import ControllerCommand, HostInteractionAdmissionContext, HostInteractionRequest
from vv_agent.runtime.run_definition import build_run_definition
from vv_agent.runtime.state import CheckpointRenewal
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.tools import (
    FunctionTool,
    ToolMetadata,
    ToolOutputText,
    ToolRegistry,
    ToolSideEffect,
    build_default_registry,
)
from vv_agent.tools.executor import FunctionToolExecutor
from vv_agent.types import AgentResult, AgentStatus, AgentTask, LLMResponse, Message, SubAgentConfig, ToolArtifactRef, ToolCall

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "distributed_run_envelope.json"
WORKER_RESPONSE_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "distributed_worker_response.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _worker_response_case_payload(fixture: dict[str, Any], case: dict[str, Any]) -> Any:
    if "response" in case:
        return copy.deepcopy(case["response"])
    valid = {item["name"]: item["response"] for item in fixture["valid_cases"]}
    payload = copy.deepcopy(valid[case["base_valid_case"]])
    mutation = case["mutation"]
    target = payload
    for field_name in mutation["path"][:-1]:
        target = target[field_name]
    field_name = mutation["path"][-1]
    if mutation["operation"] == "remove":
        del target[field_name]
    else:
        target[field_name] = copy.deepcopy(mutation["value"])
    return payload


def _strict_envelope() -> DistributedRunEnvelope:
    checkpoint_store_ref = CapabilityRef("checkpoint.strict", "2")
    checkpoint_extension_ref = CapabilityRef("extension.audit", "1")
    task = AgentTask(
        task_id="strict-run",
        model="test-model",
        prompt_bundle=build_raw_system_prompt_bundle("Use current wire only."),
        user_prompt="Inspect the payload.",
        sub_agents={
            "research": SubAgentConfig(
                model="test-model",
                description="Inspect one source.",
                backend="test",
                system_prompt="Return evidence.",
                max_cycles=4,
                exclude_tools=["bash"],
                denied_side_effects=["execute"],
                denied_capability_tags=["filesystem.delete"],
                deny_terminal_tools=False,
                denied_cost_dimensions=["gpu.second"],
                metadata={"scope": "read-only"},
            )
        },
        model_settings=ModelSettings(
            temperature=0.2,
            tool_choice={
                "type": "function",
                "function": {"name": "search"},
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
            retry=RetrySettings(max_attempts=2, backoff_seconds=0.5),
            extra_body={"provider": {"mode": "strict"}},
        ),
        initial_messages=[
            Message(
                role="assistant",
                content="Checking.",
                tool_calls=[
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"current contract"}',
                        },
                        "extra_content": {"provider": "test"},
                    }
                ],
                metadata={"sequence": 1},
            )
        ],
        initial_shared_state={"attempt": 1},
        metadata={"_vv_agent_run_id": "strict-run", "language": "en-US"},
    )
    recipe = RuntimeRecipe(
        settings_file="/srv/settings.json",
        backend="test",
        model="test-model",
        workspace="/srv/workspace",
        timeout_seconds=120,
        log_preview_chars=256,
        capabilities=DistributedCapabilities(
            tool_policy=DistributedToolPolicy(
                allowed_tools=("read_file", "task_finish"),
                disallowed_tools=("bash",),
                approval="on_request",
                predicate_ref=CapabilityRef("policy.strict", "1"),
                denied_side_effects=("execute",),
                denied_capability_tags=("filesystem.delete",),
                denied_cost_dimensions=("gpu.second",),
            ),
            llm_client_ref=CapabilityRef("llm.strict", "1"),
            memory_provider_refs=(CapabilityRef("memory.strict", "1"),),
            checkpoint_store_ref=checkpoint_store_ref,
            checkpoint_extension_refs=(
                CheckpointExtensionRef(
                    namespace="com.example.audit",
                    reference=checkpoint_extension_ref,
                    required=True,
                ),
            ),
        ),
    )
    return DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=recipe,
        cycle_index=1,
        root_run_id="strict-run",
        trace_id="trace-strict-run",
        run_definition_digest="c" * 64,
        claim_mode="continue",
        resume_attempt=1,
        checkpoint_config=DistributedCheckpointConfig(
            key="strict-run",
            resume_policy=ResumePolicy.REQUIRE_EXISTING,
            ambiguous_model_policy=AmbiguousModelPolicy.REQUIRE_RECONCILIATION,
            ambiguous_tool_policy=AmbiguousToolPolicy.RETRY_IDEMPOTENT_ONLY,
            required_extension_namespaces=("com.example.audit",),
            max_extension_state_bytes=262_144,
            credential_slots=("/model/settings/extra_headers/authorization",),
        ),
        deadline_unix_ms=2_000_000_000_000,
        lease_duration_ms=120_000,
        budget_limits=RunBudgetLimits(
            max_total_tokens=5_000,
            max_uncached_input_tokens=4_000,
            max_tool_calls=20,
            max_tool_calls_by_name={"search": 8},
            max_wall_time_ms=60_000,
            max_host_cost=HostCost(
                unit="credits",
                currency="CNY",
                amount_microunits=500_000,
            ),
        ),
    )


def _strict_payload() -> dict[str, Any]:
    return _strict_envelope().to_dict()


def _set_path(payload: dict[str, Any], path: list[str | int], value: Any) -> None:
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _remove_path(payload: dict[str, Any], path: list[str | int]) -> None:
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]


def _add_unknown_field(payload: dict[str, Any], path: list[str | int]) -> None:
    target: Any = payload
    for key in path:
        target = target[key]
    target["future_behavior"] = True


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(
        endpoint_id="test",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        function_call_available=True,
    )


def _provider(
    factory: Callable[[], Any],
    *,
    settings: ModelSettings | None = None,
) -> FactoryModelProvider:
    return FactoryModelProvider(
        factory=factory,
        resolved=_resolved(),
        settings=settings or ModelSettings(),
    )


class _ImmediateResult:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def get(self, *, timeout: float) -> dict[str, Any]:
        del timeout
        return self.value

    def revoke(self, *, terminate: bool = False) -> None:
        del terminate


class _ImmediateApp:
    def __init__(
        self,
        *,
        registry: DistributedCapabilityRegistry,
        store: InMemoryCheckpointStore,
        transport_redelivered: bool = False,
        mutate_envelope: Callable[[dict[str, Any]], None] | None = None,
        fail_first_dispatch: bool = False,
        pending_first_dispatch: bool = False,
    ) -> None:
        self.registry = registry
        self.store = store
        self.transport_redelivered = transport_redelivered
        self.mutate_envelope = mutate_envelope
        self.fail_first_dispatch = fail_first_dispatch
        self.pending_first_dispatch = pending_first_dispatch
        self.envelopes: list[dict[str, Any]] = []
        self.worker_snapshots: list[Any] = []
        self.worker_responses: list[dict[str, Any]] = []

    def send_task(
        self,
        _name: str,
        *,
        kwargs: dict[str, Any],
        serializer: str,
        task_id: str,
    ) -> _ImmediateResult:
        assert serializer == "json"
        assert task_id == kwargs["envelope_dict"]["job_id"]
        envelope = copy.deepcopy(kwargs["envelope_dict"])
        if self.mutate_envelope is not None:
            self.mutate_envelope(envelope)
        self.envelopes.append(envelope)
        if self.fail_first_dispatch and len(self.envelopes) == 1:
            raise RuntimeError("transient broker delivery failure")
        if self.pending_first_dispatch and len(self.envelopes) == 1:
            response = DistributedWorkerResponse.pending().to_dict()
            self.worker_responses.append(copy.deepcopy(response))
            return _ImmediateResult(response)
        result = run_single_cycle(
            envelope_dict=envelope,
            capability_registry=self.registry,
            transport_redelivered=self.transport_redelivered,
        )
        self.worker_responses.append(copy.deepcopy(result))
        key = envelope["checkpoint_config"]["key"]
        self.worker_snapshots.append(self.store.load_checkpoint(key))
        return _ImmediateResult(result)


class _FailOnceTerminalDeliveryStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_terminal_delivery = True

    def record_event_delivery(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: Any,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool:
        checkpoint = self.load_checkpoint(checkpoint_key)
        if self.fail_next_terminal_delivery and checkpoint is not None and checkpoint.terminal_result is not None:
            self.fail_next_terminal_delivery = False
            raise RuntimeError("terminal event delivery crash")
        return super().record_event_delivery(
            checkpoint_key,
            event_id=event_id,
            payload_digest=payload_digest,
            cursor=cursor,
            expected_revision=expected_revision,
            claim_token=claim_token,
        )


def _terminal_replay_window(
    tmp_path: Path,
    *,
    transport_redelivered: bool = False,
) -> tuple[_FailOnceTerminalDeliveryStore, DistributedCapabilityRegistry, CeleryBackend, dict[str, Any], list[int]]:
    store = _FailOnceTerminalDeliveryStore()
    checkpoint_ref = CapabilityRef("checkpoint.terminal-replay-repair", "1")
    llm_ref = CapabilityRef("llm.terminal-replay-repair", "1")
    model_calls = [0]

    def worker_complete(_request: Any) -> LLMResponse:
        model_calls[0] += 1
        return LLMResponse(content="worker answer")

    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[worker_complete]))
    recipe = RuntimeRecipe(
        settings_file="",
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _ImmediateApp(registry=registry, store=store, transport_redelivered=transport_redelivered)
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=5,
    )
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="terminal-replay-repair",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )

    with pytest.raises(RuntimeError, match="terminal event delivery crash"):
        Runner.run_sync(
            Agent(
                name="terminal-replay-repair-agent",
                instructions="Return one answer.",
                model="test-model",
            ),
            "answer",
            run_config=run_config,
        )

    checkpoint = store.load_checkpoint("terminal-replay-repair")
    assert checkpoint is not None
    assert checkpoint.terminal_result is not None
    assert checkpoint.terminal_acknowledged is False
    assert any(entry.state == "pending" for entry in checkpoint.event_outbox)
    assert len(app.envelopes) == 1
    return store, registry, backend, app.envelopes[0], model_calls


class _EnqueueOnlyResult:
    def get(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("nonblocking dispatch must not wait for a Celery result")


class _EnqueueOnlyApp:
    def __init__(self) -> None:
        self.envelopes: list[dict[str, Any]] = []
        self.options: list[dict[str, Any]] = []

    def send_task(self, _name: str, *, kwargs: dict[str, Any], **options: Any) -> _EnqueueOnlyResult:
        self.envelopes.append(copy.deepcopy(kwargs["envelope_dict"]))
        self.options.append(copy.deepcopy(options))
        return _EnqueueOnlyResult()


class _StartProbeBackend(InlineBackend):
    def start(self, **_kwargs: Any) -> None:
        return None


class _CompiledStartProbeBackend(InlineBackend):
    def __init__(self) -> None:
        self.task: AgentTask | None = None
        self.start_calls = 0
        self.continuation: Any | None = None

    def start(
        self,
        *,
        task: AgentTask,
        checkpoint_controller: Any,
        continuation: Any | None = None,
        **_kwargs: Any,
    ) -> DistributedRunHandle:
        self.start_calls += 1
        self.task = task
        self.continuation = continuation
        checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
        assert checkpoint is not None
        return DistributedRunHandle(
            checkpoint_key=checkpoint.checkpoint_key,
            run_id=checkpoint.root_run_id,
            trace_id=checkpoint.trace_id,
        )


def test_distributed_current_writer_and_reader_round_trip_strict_nested_wire() -> None:
    payload = _strict_payload()

    restored = DistributedRunEnvelope.from_dict(payload)

    assert restored.to_dict() == payload
    assert set(payload) == {
        "schema_version",
        "job_id",
        "run_id",
        "task",
        "budget_limits",
        "recipe",
        "cycle_name",
        "cycle_index",
        "idempotency_key",
        "deadline_unix_ms",
        "lease_duration_ms",
        "root_run_id",
        "trace_id",
        "run_definition_schema",
        "run_definition_digest",
        "claim_mode",
        "resume_attempt",
        "checkpoint_config",
    }


def test_distributed_round_trip_preserves_message_artifact_ref() -> None:
    envelope = _strict_envelope()
    artifact_ref = ToolArtifactRef(
        path=".vv-agent/artifacts/run-7/call-search.txt",
        media_type="text/plain",
        encoding="utf-8",
        size_bytes=42,
        sha256="0" * 64,
    )
    envelope.task.initial_messages.append(
        Message(
            role="tool",
            content="<Tool Result Compact>\nartifact_path: .vv-agent/artifacts/run-7/call-search.txt",
            tool_call_id="call-search",
            artifact_ref=artifact_ref,
        )
    )

    restored = DistributedRunEnvelope.from_dict(envelope.to_dict())

    assert restored.task.initial_messages[-1].artifact_ref == artifact_ref


def test_distributed_worker_response_matches_all_canonical_and_invalid_cases() -> None:
    fixture = json.loads(WORKER_RESPONSE_FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in fixture["valid_cases"]:
        response = DistributedWorkerResponse.from_dict(case["response"])
        assert response.to_dict() == case["response"], case["name"]

    for case in fixture["invalid_cases"]:
        with pytest.raises(DistributedContractError, match=case["error"]):
            DistributedWorkerResponse.from_dict(_worker_response_case_payload(fixture, case))


def test_distributed_worker_response_producer_preserves_bounded_tool_result_fields() -> None:
    fixture = json.loads(WORKER_RESPONSE_FIXTURE_PATH.read_text(encoding="utf-8"))
    payload = next(case["response"] for case in fixture["valid_cases"] if case["name"] == "terminal_candidate")
    decoded = DistributedWorkerResponse.from_dict(payload)
    assert decoded.result is not None
    assert decoded.checkpoint_revision is not None

    produced = DistributedWorkerResponse.terminal_candidate(
        checkpoint_revision=decoded.checkpoint_revision,
        result=decoded.result,
    ).to_dict()
    expected_result = payload["result"]["cycles"][0]["tool_results"][0]

    assert produced == payload
    assert produced["result"]["cycles"][0]["tool_results"][0] == expected_result


@pytest.mark.parametrize(
    "object_path",
    [
        [],
        ["task"],
        ["task", "sub_agents", "research"],
        ["task", "model_settings"],
        ["task", "model_settings", "retry"],
        ["task", "model_settings", "tool_choice"],
        ["task", "model_settings", "tool_choice", "function"],
        ["task", "model_settings", "response_format"],
        ["task", "initial_messages", 0],
        ["task", "initial_messages", 0, "tool_calls", 0],
        ["task", "initial_messages", 0, "tool_calls", 0, "function"],
        ["budget_limits"],
        ["budget_limits", "max_host_cost"],
        ["recipe"],
        ["recipe", "capabilities"],
        ["recipe", "capabilities", "toolset_ref"],
        ["recipe", "capabilities", "tool_policy"],
        ["recipe", "capabilities", "tool_policy", "predicate_ref"],
        ["recipe", "capabilities", "memory_provider_refs", 0],
        ["recipe", "capabilities", "checkpoint_extension_refs", 0],
        ["recipe", "capabilities", "checkpoint_extension_refs", 0, "reference"],
        ["checkpoint_config"],
    ],
    ids=lambda path: "/".join(str(part) for part in path) or "envelope",
)
def test_distributed_reader_rejects_unknown_field_at_every_closed_layer(
    object_path: list[str | int],
) -> None:
    payload = _strict_payload()
    _add_unknown_field(payload, object_path)

    with pytest.raises(DistributedContractError, match="unknown"):
        DistributedRunEnvelope.from_dict(payload)


@pytest.mark.parametrize(
    "field_path",
    [
        ["schema_version"],
        ["run_definition_schema"],
        ["job_id"],
        ["task", "max_cycles"],
        ["task", "sub_agents", "research", "description"],
        ["task", "model_settings", "retry", "backoff_seconds"],
        ["task", "model_settings", "tool_choice", "type"],
        ["task", "model_settings", "tool_choice", "function", "name"],
        ["task", "model_settings", "response_format", "type"],
        ["task", "initial_messages", 0, "content"],
        ["task", "initial_messages", 0, "tool_calls", 0, "id"],
        ["task", "initial_messages", 0, "tool_calls", 0, "function", "arguments"],
        ["budget_limits", "max_total_tokens"],
        ["budget_limits", "max_host_cost", "currency"],
        ["recipe", "settings_file"],
        ["recipe", "timeout_seconds"],
        ["recipe", "capabilities", "observer_refs"],
        ["recipe", "capabilities", "toolset_ref", "schema_digest"],
        ["recipe", "capabilities", "tool_policy", "approval"],
        ["recipe", "capabilities", "tool_policy", "predicate_ref", "version"],
        ["recipe", "capabilities", "memory_provider_refs", 0, "id"],
        ["recipe", "capabilities", "checkpoint_extension_refs", 0, "required"],
        ["recipe", "capabilities", "checkpoint_extension_refs", 0, "reference", "version"],
        ["checkpoint_config", "credential_slots"],
    ],
    ids=lambda path: "/".join(str(part) for part in path),
)
def test_distributed_reader_rejects_missing_current_wire_field(
    field_path: list[str | int],
) -> None:
    payload = _strict_payload()
    _remove_path(payload, field_path)

    with pytest.raises(DistributedContractError):
        DistributedRunEnvelope.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("schema_version", None),
        ("schema_version", "vv-agent.distributed-run.v1"),
        ("schema_version", "vv-agent.distributed-run.v4"),
        ("run_definition_schema", None),
        ("run_definition_schema", "vv-agent.run-definition.v0"),
        ("run_definition_schema", "vv-agent.run-definition.v4"),
    ],
)
def test_distributed_reader_rejects_missing_stale_or_unknown_versions(
    field_name: str,
    value: Any,
) -> None:
    payload = _strict_payload()
    payload[field_name] = value

    with pytest.raises(DistributedContractError):
        DistributedRunEnvelope.from_dict(payload)


def test_distributed_declared_extension_maps_are_open_but_json_only() -> None:
    payload = _strict_payload()
    payload["task"]["metadata"]["host_extension"] = {"enabled": True}
    payload["task"]["initial_shared_state"]["custom_state"] = [1, 2, 3]
    payload["task"]["model_settings"]["extra_body"]["future_provider_field"] = {"mode": "enabled"}
    payload["task"]["initial_messages"][0]["tool_calls"][0]["extra_content"]["future_provider_field"] = "preserved"

    assert DistributedRunEnvelope.from_dict(payload).to_dict() == payload

    payload["task"]["metadata"]["not_json"] = ("tuple",)
    with pytest.raises(DistributedContractError, match="non-JSON wire value"):
        DistributedRunEnvelope.from_dict(payload)


def test_distributed_fixture_round_trips_with_discriminator() -> None:
    canonical = _fixture()["canonical_envelope"]

    envelope = DistributedRunEnvelope.from_dict(canonical)

    assert envelope.schema_version == DISTRIBUTED_RUN_SCHEMA_VERSION
    assert envelope.to_dict() == canonical


@pytest.mark.parametrize("case", _fixture()["runtime_recipe_cases"], ids=lambda case: case["name"])
def test_distributed_recipe_settings_file_depends_on_client_reference(case: dict[str, Any]) -> None:
    payload = copy.deepcopy(_fixture()["canonical_envelope"])
    payload["recipe"]["settings_file"] = case["settings_file"]
    payload["recipe"]["capabilities"]["llm_client_ref"] = case["llm_client_ref"]
    if case["valid"]:
        assert DistributedRunEnvelope.from_dict(payload).to_dict() == payload
    else:
        with pytest.raises(DistributedContractError):
            DistributedRunEnvelope.from_dict(payload)


def test_distributed_tool_policy_round_trips_metadata_denials() -> None:
    payload = {
        "allowed_tools": ["read_file"],
        "disallowed_tools": ["bash"],
        "approval": "never",
        "predicate_ref": None,
        "denied_side_effects": ["execute"],
        "denied_capability_tags": ["filesystem.delete"],
        "deny_terminal_tools": True,
        "denied_cost_dimensions": ["gpu.second"],
    }

    policy = DistributedToolPolicy.from_dict(payload)

    assert policy.to_dict() == payload
    resolved = policy.resolve(DistributedCapabilityRegistry())
    assert resolved.denied_side_effects == ["execute"]
    assert resolved.denied_capability_tags == ["filesystem.delete"]
    assert resolved.deny_terminal_tools is True
    assert resolved.denied_cost_dimensions == ["gpu.second"]

    noncanonical = {**payload, "denied_capability_tags": [" filesystem.delete "]}
    with pytest.raises(DistributedContractError, match="canonical"):
        DistributedToolPolicy.from_dict(noncanonical)


def test_celery_projects_effective_metadata_policy_into_envelope(
    tmp_path: Path,
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.metadata-policy", "2")
    llm_ref = CapabilityRef("llm.metadata-policy", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="done")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _ImmediateApp(registry=registry, store=store)
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=5,
    )

    result = Runner.run_sync(
        Agent(
            name="metadata-policy-agent",
            instructions="Return one answer.",
            model="test-model",
        ),
        "answer without tools",
        run_config=RunConfig(
            model_provider=_provider(lambda: ScriptedLLM(steps=[])),
            execution_backend=backend,
            max_cycles=1,
            no_tool_policy="finish",
            tool_policy=ToolPolicy(
                denied_side_effects=["execute"],
                denied_capability_tags=["filesystem.delete"],
                deny_terminal_tools=True,
                denied_cost_dimensions=["gpu.second"],
            ),
            checkpoint_config=CheckpointConfig(
                key="distributed-metadata-policy",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            ),
        ),
    )

    assert result.status is AgentStatus.COMPLETED
    assert len(app.envelopes) == 1
    assert app.envelopes[0]["recipe"]["capabilities"]["tool_policy"] == {
        "allowed_tools": None,
        "disallowed_tools": [],
        "approval": "default",
        "predicate_ref": None,
        "denied_side_effects": ["execute"],
        "denied_capability_tags": ["filesystem.delete"],
        "deny_terminal_tools": True,
        "denied_cost_dimensions": ["gpu.second"],
    }


def test_celery_generic_execute_is_guarded_and_local_path_is_explicit() -> None:
    backend = object.__new__(CeleryBackend)
    task = _strict_envelope().task
    with pytest.raises(DistributedContractError, match="execute_local"):
        backend.execute(
            task=task,
            initial_messages=[],
            shared_state={},
            cycle_executor=lambda *_args: None,
            ctx=None,
            max_cycles=1,
        )
    assert callable(backend.execute_local)


@pytest.mark.parametrize("resume_policy", [ResumePolicy.NEW, ResumePolicy.RESUME_IF_PRESENT])
@pytest.mark.parametrize("extra_finalizer_tool", [False, True])
def test_nonblocking_celery_start_and_terminal_finalize_never_wait_for_result(
    tmp_path: Path, resume_policy: ResumePolicy, extra_finalizer_tool: bool
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-terminal", "1")
    llm_ref = CapabilityRef("llm.nonblocking-terminal", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="worker answer")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        capability_registry=registry,
        dispatch_timeout_seconds=5,
    )
    agent = Agent(
        name="nonblocking-terminal-agent",
        instructions="Return one answer.",
        model="test-model",
        output_guardrails=[lambda _context, output: GuardrailResult.rewrite(f"guarded: {output}")],
    )
    default_tools = build_default_registry()
    finalizer_tools = ToolRegistry()
    for name in default_tools.list_tool_names():
        finalizer_tools.register_executor(default_tools.get_executor(name), planner_extra=name == "sub_task_status")
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        tool_registry_factory=lambda: finalizer_tools,
        execution_backend=backend,
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="nonblocking-terminal",
            resume_policy=resume_policy,
            store=store,
            capability_refs={
                "output_guardrail:0": {"id": "guardrail.nonblocking-terminal", "version": "1"},
                "tool_registry_factory": {"id": "tools.nonblocking-terminal", "version": "1"},
            },
        ),
    )

    callback: dict[str, Any] = {"task": "advance-live-room-agent"}
    continuation_call: dict[str, Any] = {}

    def continuation(handle: DistributedRunHandle, envelope: DistributedRunEnvelope) -> dict[str, Any]:
        continuation_call["handle"] = handle
        continuation_call["envelope"] = envelope
        return callback

    handle = Runner.start_distributed(
        agent,
        "answer",
        run_config=run_config,
        continuation=continuation,
    )

    assert isinstance(handle, DistributedRunHandle)
    assert handle.checkpoint_key == "nonblocking-terminal"
    assert len(app.envelopes) == 1
    assert continuation_call == {
        "handle": handle,
        "envelope": DistributedRunEnvelope.from_dict(app.envelopes[0]),
    }
    assert app.options[0]["link"] == callback
    checkpoint = store.load_checkpoint(handle.checkpoint_key)
    assert checkpoint is not None
    assert checkpoint.cycle_index == 0
    assert checkpoint.claim_token is None

    response = run_single_cycle(
        envelope_dict=app.envelopes[0],
        capability_registry=registry,
    )
    decision = backend.advance(
        previous_envelope=app.envelopes[0],
        outcome=DistributedDeliveryOutcome.worker(response),
    )

    assert isinstance(decision, DistributedAdvanceDecision)
    assert decision.action == "finalize_required"
    assert DistributedAdvanceDecision.from_dict(decision.to_dict()) == decision
    assert len(app.envelopes) == 1
    if extra_finalizer_tool:
        finalizer_tools.register_executor(
            FunctionToolExecutor(
                FunctionTool(
                    name="finalizer_local_tool",
                    description="Read local host state.",
                    params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    on_invoke=lambda _context, _arguments: ToolOutputText(text="unused"),
                )
            )
        )
    result = Runner.finalize_distributed(
        agent,
        "answer",
        decision=decision.to_dict(),
        run_config=run_config,
    )
    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "guarded: worker answer"
    terminal = store.load_checkpoint(handle.checkpoint_key)
    assert terminal is not None
    assert terminal.terminal_result is not None
    assert terminal.terminal_acknowledged
    assert terminal.claim_token is None

    replayed = Runner.finalize_distributed(
        agent,
        "answer",
        decision=decision,
        run_config=run_config,
    )
    assert replayed.status is AgentStatus.COMPLETED
    assert replayed.final_output == "guarded: worker answer"
    replayed_terminal = store.load_checkpoint(handle.checkpoint_key)
    assert replayed_terminal is not None
    assert replayed_terminal.revision == terminal.revision


def test_nonblocking_computer_start_uses_canonical_tool_schemas(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.computer-schema", "1")
    llm_ref = CapabilityRef("llm.computer-schema", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="worker answer")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _ImmediateApp(registry=registry, store=store)
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        capability_registry=registry,
        dispatch_timeout_seconds=5,
    )
    agent = Agent(
        name="computer-schema-agent",
        instructions="Return one answer.",
        model="test-model",
    )
    task = AgentCompiler().compile(
        agent=agent,
        input="answer",
        run_config=RunConfig(max_cycles=1, no_tool_policy="finish"),
        resolved=_resolved(),
        trace_id="trace-computer-schema",
    )
    task.agent_type = "computer"
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="computer-schema",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )

    handle = Runner.start_distributed_compiled(agent, task, run_config=run_config)

    assert isinstance(handle, DistributedRunHandle)
    assert len(app.worker_responses) == 1
    response = DistributedWorkerResponse.from_dict(app.worker_responses[0])
    assert response.response_type == "terminal_candidate"
    checkpoint = store.load_checkpoint(handle.checkpoint_key)
    assert checkpoint is not None
    assert checkpoint.run_definition["tools"]
    bash_schema = next(
        item["schema"] for item in checkpoint.run_definition["tools"] if item["schema"]["function"]["name"] == "bash"
    )
    assert "Runtime shell hint" not in bash_schema["function"]["description"]


@pytest.mark.parametrize(
    ("envelope_tool_names", "expect_rejection"),
    [
        pytest.param(["planner_extra_alpha"], False, id="repairs-agreed-omission"),
        pytest.param(
            ["planner_extra_alpha", "planner_extra_unagreed"],
            True,
            id="rejects-definition-unagreed-extra",
        ),
    ],
)
def test_celery_worker_merges_registry_planner_extras_before_schema_validation(
    tmp_path: Path,
    envelope_tool_names: list[str],
    expect_rejection: bool,
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.planner-extras", "1")
    llm_ref = CapabilityRef("llm.planner-extras", "1")
    worker_model_calls = 0
    tool_names = ("planner_extra_alpha", "planner_extra_beta")
    unagreed_tool_name = "planner_extra_unagreed"
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def invoke(_context: Any, _arguments: dict[str, Any]) -> ToolOutputText:
        return ToolOutputText(text="ok")

    local_tools = [
        FunctionTool(
            name=name,
            description=f"Run {name}.",
            params_json_schema=schema,
            on_invoke=invoke,
        )
        for name in tool_names
    ]
    worker_tools = build_default_registry()
    worker_tools.register_executor(FunctionToolExecutor(local_tools[0]))
    worker_tools.register_executor(FunctionToolExecutor(local_tools[1]))
    worker_tools.register_executor(
        FunctionToolExecutor(
            FunctionTool(
                name=unagreed_tool_name,
                description=f"Run {unagreed_tool_name}.",
                params_json_schema=schema,
                on_invoke=invoke,
            )
        )
    )
    toolset_ref = ToolsetRef(
        id="toolset.planner-extras",
        version="1",
        schema_digest=toolset_schema_digest(worker_tools),
    )
    registry = DistributedCapabilityRegistry()
    registry.register_toolset(toolset_ref, worker_tools)
    registry.register("checkpoint_store", checkpoint_ref, store)

    def worker_complete(_request: Any) -> LLMResponse:
        nonlocal worker_model_calls
        worker_model_calls += 1
        return LLMResponse(content="worker answer")

    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[worker_complete]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            toolset_ref=toolset_ref,
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )

    def omit_one_planner_extra(envelope: dict[str, Any]) -> None:
        task = envelope["task"]
        assert task["extra_tool_names"] == list(tool_names)
        task["extra_tool_names"] = envelope_tool_names

    app = _ImmediateApp(
        registry=registry,
        store=store,
        mutate_envelope=omit_one_planner_extra,
    )
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=5,
    )

    def run_agent() -> Any:
        return Runner.run_sync(
            Agent(
                name="planner-extras-agent",
                instructions="Return one answer.",
                model="test-model",
                tools=local_tools,
            ),
            "answer without tools",
            run_config=RunConfig(
                model_provider=_provider(lambda: ScriptedLLM(steps=[])),
                execution_backend=backend,
                max_cycles=1,
                no_tool_policy="finish",
                checkpoint_config=CheckpointConfig(
                    key="distributed-planner-extras",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                ),
            ),
        )

    if expect_rejection:
        with pytest.raises(CheckpointError) as caught:
            run_agent()
        assert caught.value.code == "checkpoint_definition_mismatch"
        checkpoint = store.load_checkpoint("distributed-planner-extras")
        assert checkpoint is not None
        assert checkpoint.claim_token is None
        assert checkpoint.model_call_journal == []
        assert worker_model_calls == 0
    else:
        result = run_agent()
        assert result.status is AgentStatus.COMPLETED
        assert result.final_output == "worker answer"

    assert len(app.envelopes) == 1
    assert app.envelopes[0]["task"]["extra_tool_names"] == envelope_tool_names


def test_nonblocking_distributed_start_requires_explicit_checkpoint_key() -> None:
    store = InMemoryCheckpointStore()
    backend = _StartProbeBackend()

    with pytest.raises(CheckpointError) as exc_info:
        Runner.start_distributed(
            Agent(name="distributed-explicit-key", instructions="Return one answer."),
            "answer",
            run_config=RunConfig(
                execution_backend=backend,
                checkpoint_config=CheckpointConfig(store=store),
            ),
        )

    assert exc_info.value.code == "checkpoint_key_required"


def test_nonblocking_distributed_start_forwards_compiled_task_without_waiting() -> None:
    agent = Agent(name="compiled-start", instructions="Original instructions.", model="test-model")
    task = AgentCompiler().compile(
        agent=agent,
        input="compiled input",
        run_config=RunConfig(max_cycles=1),
        resolved=_resolved(),
        trace_id="trace-compiled-start",
    )
    task.initial_shared_state["prepared_marker"] = "compiled-task"
    expected_task = task.to_dict()
    backend = _CompiledStartProbeBackend()
    store = InMemoryCheckpointStore()
    continuation = object()

    handle = Runner.start_distributed_compiled(
        agent,
        task,
        run_config=RunConfig(
            model_provider=_provider(lambda: ScriptedLLM(steps=[])),
            execution_backend=backend,
            max_cycles=1,
            checkpoint_config=CheckpointConfig(
                key="compiled-start",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            ),
        ),
        continuation=continuation,
    )

    assert isinstance(handle, DistributedRunHandle)
    assert backend.start_calls == 1
    assert backend.continuation is continuation
    assert backend.task is not None
    assert backend.task.to_dict() == expected_task


def test_nonblocking_celery_advance_waits_for_host_interaction_and_suspended_state(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-wait", "1")
    llm_ref = CapabilityRef("llm.nonblocking-wait", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="unused")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        capability_registry=registry,
    )
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=1,
        checkpoint_config=CheckpointConfig(
            key="nonblocking-wait",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )
    agent = Agent(name="nonblocking-wait-agent", instructions="Wait.", model="test-model")
    Runner.start_distributed(agent, "work", run_config=run_config)
    envelope = app.envelopes[0]
    claimed = store.claim_checkpoint(
        "nonblocking-wait",
        1,
        claim_token="host-producer",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    request = HostInteractionRequest(
        interaction_id="wait-interaction",
        logical_cycle=1,
        operation_id="wait-operation",
        tool_call_id="wait-tool",
        prompt="Choose.",
    )
    store.produce_host_interaction(
        request,
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key="nonblocking-wait",
            claim_token="host-producer",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    host_wait = backend.advance(
        previous_envelope=envelope,
        outcome=DistributedWorkerResponse.pending(),
    )
    assert host_wait.action == "wait"
    assert host_wait.reason == "host_interaction"
    assert len(app.envelopes) == 1
    current = store.load_checkpoint("nonblocking-wait")
    assert current is not None
    unmatched_replay = DistributedWorkerResponse.terminal_replay(
        checkpoint_revision=current.revision,
        result=AgentResult(status=AgentStatus.COMPLETED, messages=[], cycles=[], final_answer="Uncommitted"),
    )
    with pytest.raises(CheckpointError, match="no matching durable terminal"):
        backend.advance(previous_envelope=envelope, outcome=unmatched_replay)
    assert store.load_checkpoint("nonblocking-wait") == current
    suspend = ControllerCommand(
        command_id="suspend-wait",
        handle=DistributedRunHandle("nonblocking-wait", current.root_run_id, current.trace_id),
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={"kind": "suspend"},
    )
    assert store.resolve_controller_command(suspend).kind == "applied"
    suspended_wait = backend.advance(
        previous_envelope=envelope,
        outcome=DistributedWorkerResponse.pending(),
    )
    assert suspended_wait.action == "wait"
    assert suspended_wait.reason == "suspended"
    assert len(app.envelopes) == 1
    current = store.load_checkpoint("nonblocking-wait")
    with pytest.raises(CheckpointError, match="no matching durable terminal"):
        backend.advance(previous_envelope=envelope, outcome=unmatched_replay)
    assert store.load_checkpoint("nonblocking-wait") == current
    assert len(app.envelopes) == 1


def test_nonblocking_celery_advance_ignores_host_response_for_other_checkpoint(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.shared-wake", "1")
    llm_ref = CapabilityRef("llm.shared-wake", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(celery_app=app, runtime_recipe=recipe, capability_registry=registry)

    def start_checkpoint(key: str) -> dict[str, Any]:
        Runner.start_distributed(
            Agent(name=f"{key}-agent", instructions="Continue.", model="test-model"),
            "work",
            run_config=RunConfig(
                model_provider=_provider(lambda: ScriptedLLM(steps=[])),
                execution_backend=backend,
                max_cycles=1,
                checkpoint_config=CheckpointConfig(
                    key=key,
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                ),
            ),
        )
        return app.envelopes[-1]

    current_key = "shared-current"
    other_key = "shared-other"
    current_envelope = start_checkpoint(current_key)
    start_checkpoint(other_key)
    claimed = store.claim_checkpoint(
        other_key,
        1,
        claim_token="other-host-producer",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    request = HostInteractionRequest(
        interaction_id="other-interaction",
        logical_cycle=1,
        operation_id="other-operation",
        tool_call_id="other-tool",
        prompt="Choose.",
    )
    store.produce_host_interaction(
        request,
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=other_key,
            claim_token="other-host-producer",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    other = store.load_checkpoint(other_key)
    assert other is not None
    command = ControllerCommand(
        command_id="other-host-response",
        handle=DistributedRunHandle(other_key, other.root_run_id, other.trace_id),
        resume_attempt=other.resume_attempt,
        expected_revision=other.revision,
        command={
            "kind": "host_interaction_response",
            "interaction_id": request.interaction_id,
            "logical_cycle": request.logical_cycle,
            "operation_id": request.operation_id,
            "tool_call_id": request.tool_call_id,
            "request_digest": request.request_digest,
            "response": {"role": "user", "content": "approved"},
        },
    )
    resolution = store.resolve_controller_command(command)
    assert resolution.kind == "applied"
    assert resolution.receipt is not None and resolution.receipt.outbox_state == "pending"

    before_current = store.load_checkpoint(current_key)
    assert before_current is not None
    decision = backend.advance(
        previous_envelope=current_envelope,
        outcome=DistributedWorkerResponse.pending(),
        enqueue=False,
    )
    assert decision.action == "dispatch"
    assert decision.envelope is not None and decision.envelope.checkpoint_config.key == current_key
    after_current = store.load_checkpoint(current_key)
    assert after_current == before_current
    other_after = store.load_checkpoint(other_key)
    assert other_after is not None and other_after.claim_token is None
    receipt = store.get_controller_command_receipt(command.command_id)
    assert receipt is not None and receipt.outbox_state == "pending"


@pytest.mark.parametrize("suspended", [False, True], ids=["active", "suspended"])
def test_nonblocking_celery_response_recovery_consumes_once_before_continue(
    tmp_path: Path,
    suspended: bool,
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.host-recovery", "1")
    llm_ref = CapabilityRef("llm.host-recovery", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)

    def complete(_request: Any) -> LLMResponse:
        before = store.load_checkpoint("host-recovery")
        duplicate = run_single_cycle(envelope_dict=app.envelopes[1], capability_registry=registry)
        assert duplicate["type"] == "pending"
        assert store.load_checkpoint("host-recovery") == before
        return LLMResponse(content="continued")

    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[complete]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(celery_app=app, runtime_recipe=recipe, capability_registry=registry)
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=2,
        no_tool_policy="continue",
        checkpoint_config=CheckpointConfig(
            key="host-recovery",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )
    agent = Agent(name="host-recovery-agent", instructions="Continue after input.", model="test-model")
    Runner.start_distributed(agent, "work", run_config=run_config)
    first_envelope = app.envelopes[0]
    claimed = store.claim_checkpoint(
        "host-recovery",
        1,
        claim_token="host-producer",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    request = HostInteractionRequest(
        interaction_id="host-interaction",
        logical_cycle=1,
        operation_id="host-operation",
        tool_call_id="host-tool",
        prompt="Choose.",
    )
    store.produce_host_interaction(
        request,
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key="host-recovery",
            claim_token="host-producer",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    current = store.load_checkpoint("host-recovery")
    assert current is not None
    handle = DistributedRunHandle("host-recovery", current.root_run_id, current.trace_id)
    if suspended:
        suspend = ControllerCommand(
            command_id="suspend-host-recovery",
            handle=handle,
            resume_attempt=current.resume_attempt,
            expected_revision=current.revision,
            command={"kind": "suspend"},
        )
        suspended_resolution = store.resolve_controller_command(suspend)
        assert suspended_resolution.kind == "applied"
        assert suspended_resolution.receipt is not None and suspended_resolution.receipt.outbox_action == "none"
        current = store.load_checkpoint("host-recovery")
        assert current is not None and current.status is AgentStatus.SUSPENDED
    command = ControllerCommand(
        command_id="host-response-command",
        handle=handle,
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={
            "kind": "host_interaction_response",
            "interaction_id": request.interaction_id,
            "logical_cycle": request.logical_cycle,
            "operation_id": request.operation_id,
            "tool_call_id": request.tool_call_id,
            "request_digest": request.request_digest,
            "response": {"role": "user", "content": "approved"},
        },
    )
    resolution = store.resolve_controller_command(command)
    assert resolution.kind == "applied"
    assert resolution.receipt is not None
    if suspended:
        assert resolution.receipt.outbox_action == "none"
        current = store.load_checkpoint("host-recovery")
        assert current is not None and current.status is AgentStatus.SUSPENDED
        resume = ControllerCommand(
            command_id="resume-host-recovery",
            handle=handle,
            resume_attempt=current.resume_attempt,
            expected_revision=current.revision,
            command={"kind": "resume"},
        )
        resume_resolution = store.resolve_controller_command(resume)
        assert resume_resolution.kind == "applied"
        assert resume_resolution.receipt is not None and resume_resolution.receipt.outbox_state == "pending"
        wake_command = resume
    else:
        assert resolution.receipt.outbox_state == "pending"
        wake_command = command
    crashed_wake = store.claim_controller_command_wake(
        command_id=wake_command.command_id,
        command_digest=wake_command.command_digest or "",
        claim_token="crashed-wake-owner",
        lease_expires_at_ms=2,
        now_ms=1,
    )
    assert crashed_wake is not None and crashed_wake["outbox_state"] == "claimed"

    admission = store.load_checkpoint("host-recovery")
    with patch.object(store, "load_checkpoint", wraps=store.load_checkpoint) as loads:
        decision = backend.advance(
            previous_envelope=first_envelope,
            outcome=DistributedWorkerResponse.pending(),
        )
    assert loads.call_count == 1
    assert store.load_checkpoint("host-recovery") == admission
    assert decision.action == "dispatch"
    assert decision.envelope is not None and decision.envelope.claim_mode == "recovery"
    assert len(app.envelopes) == 2
    worker_response = run_single_cycle(envelope_dict=app.envelopes[1], capability_registry=registry)
    assert worker_response["type"] == "committed"
    consumed = store.load_checkpoint("host-recovery")
    assert consumed is not None
    assert [message.content for message in consumed.messages].count("approved") == 1
    assert consumed.claim_token is None
    receipt = store.get_controller_command_receipt(wake_command.command_id)
    assert receipt is not None and receipt.outbox_state == "delivered"
    if suspended:
        response_receipt = store.get_controller_command_receipt(command.command_id)
        assert response_receipt is not None and response_receipt.outbox_action == "none"
    consumed_revision = consumed.revision

    replay = backend.advance(
        previous_envelope=first_envelope,
        outcome=DistributedWorkerResponse.pending(),
        enqueue=False,
    )
    assert replay.action in {"dispatch", "retry_at"}
    replayed = store.load_checkpoint("host-recovery")
    assert replayed is not None
    assert replayed.revision == consumed_revision
    assert [message.content for message in replayed.messages].count("approved") == 1

    committed = store.load_checkpoint("host-recovery")
    assert committed is not None and committed.cycle_index == 1 and committed.claim_token is None


def test_nonblocking_celery_advance_enqueues_one_cycle_per_committed_callback(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-chain", "1")
    llm_ref = CapabilityRef("llm.nonblocking-chain", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(
            steps=[
                LLMResponse(content="cycle one"),
                LLMResponse(content="cycle two"),
            ]
        ),
    )
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        capability_registry=registry,
        dispatch_timeout_seconds=5,
    )
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=2,
        no_tool_policy="continue",
        checkpoint_config=CheckpointConfig(
            key="nonblocking-chain",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )

    agent = Agent(name="nonblocking-chain-agent", instructions="Continue.", model="test-model")
    Runner.start_distributed(
        agent,
        "work",
        run_config=run_config,
    )
    first_response = run_single_cycle(envelope_dict=app.envelopes[0], capability_registry=registry)
    first_decision = backend.advance(previous_envelope=app.envelopes[0], outcome=first_response)

    assert first_decision.action == "dispatch"
    assert len(app.envelopes) == 2
    assert app.envelopes[1]["cycle_index"] == 2
    second_response = run_single_cycle(envelope_dict=app.envelopes[1], capability_registry=registry)
    second_decision = backend.advance(previous_envelope=app.envelopes[1], outcome=second_response)
    assert second_decision.action == "finalize_required"
    assert second_decision.result is not None
    assert second_decision.result.status is AgentStatus.MAX_CYCLES
    assert len(app.envelopes) == 2

    result = Runner.finalize_distributed(
        agent,
        "work",
        decision=second_decision,
        run_config=run_config,
    )
    assert result.status is AgentStatus.MAX_CYCLES
    terminal = store.load_checkpoint("nonblocking-chain")
    assert terminal is not None
    assert terminal.terminal_result is not None
    assert terminal.terminal_acknowledged


def test_nonblocking_celery_duplicate_and_out_of_order_callbacks_do_not_skip_cycles(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-callback-order", "1")
    llm_ref = CapabilityRef("llm.nonblocking-callback-order", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(steps=[LLMResponse(content="cycle one"), LLMResponse(content="cycle two")]),
    )
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        capability_registry=registry,
        dispatch_timeout_seconds=5,
        dispatch_outbox_store=store,
    )
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=3,
        no_tool_policy="continue",
        checkpoint_config=CheckpointConfig(
            key="nonblocking-callback-order",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )
    Runner.start_distributed(
        Agent(name="callback-order-agent", instructions="Continue.", model="test-model"),
        "work",
        run_config=run_config,
    )

    first_envelope = app.envelopes[0]
    first_response = run_single_cycle(envelope_dict=first_envelope, capability_registry=registry)
    checkpoint_after_first_delivery = store.load_checkpoint(first_envelope["checkpoint_config"]["key"])
    assert checkpoint_after_first_delivery is not None
    journal_count_after_first_delivery = len(checkpoint_after_first_delivery.model_call_journal)
    duplicate_worker_response = DistributedWorkerResponse.from_dict(
        run_single_cycle(envelope_dict=first_envelope, capability_registry=registry)
    )
    checkpoint_after_duplicate_delivery = store.load_checkpoint(first_envelope["checkpoint_config"]["key"])
    assert duplicate_worker_response.response_type == "committed"
    assert checkpoint_after_duplicate_delivery is not None
    assert len(checkpoint_after_duplicate_delivery.model_call_journal) == journal_count_after_first_delivery
    first = backend.advance(previous_envelope=first_envelope, outcome=first_response)
    duplicate = backend.advance(previous_envelope=first_envelope, outcome=first_response)

    assert first.action == duplicate.action == "dispatch"
    assert first.envelope is not None and duplicate.envelope is not None
    assert first.envelope.job_id == duplicate.envelope.job_id
    assert first.envelope.idempotency_key == duplicate.envelope.idempotency_key
    assert first.envelope.cycle_index == 2
    assert len(app.envelopes) == 2

    second_envelope = app.envelopes[1]
    second_response = run_single_cycle(envelope_dict=second_envelope, capability_registry=registry)
    second = backend.advance(previous_envelope=second_envelope, outcome=second_response)
    enqueued_before_stale_callback = len(app.envelopes)
    stale = backend.advance(previous_envelope=first_envelope, outcome=first_response)

    assert second.action == "dispatch"
    assert second.envelope is not None and second.envelope.cycle_index == 3
    assert stale.action == "wait"
    assert stale.reason == "superseded_delivery"
    assert len(app.envelopes) == enqueued_before_stale_callback


def test_nonblocking_start_admission_can_own_first_delivery(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.start-admission", "1")
    llm_ref = CapabilityRef("llm.start-admission", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="unused")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(llm_client_ref=llm_ref, checkpoint_store_ref=checkpoint_ref),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(celery_app=app, runtime_recipe=recipe, capability_registry=registry)
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        checkpoint_config=CheckpointConfig(
            key="start-admission",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )
    admitted: list[tuple[DistributedRunHandle, DistributedRunEnvelope]] = []

    def admit(handle: DistributedRunHandle, envelope: DistributedRunEnvelope) -> bool:
        admitted.append((handle, envelope))
        return False

    handle = Runner.start_distributed(
        Agent(name="start-admission-agent", instructions="Return one answer.", model="test-model"),
        "start",
        run_config=run_config,
        start_admission=admit,
    )

    assert admitted and admitted[0][0] == handle
    assert admitted[0][1].checkpoint_config.key == handle.checkpoint_key
    assert app.envelopes == []


def test_nonblocking_celery_rejects_brokered_approval_before_checkpoint_creation(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-approval", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    backend = CeleryBackend(
        celery_app=_EnqueueOnlyApp(),
        runtime_recipe=RuntimeRecipe(
            settings_file=str(tmp_path / "unused-settings.py"),
            backend="test",
            model="test-model",
            workspace=str(tmp_path / "workspace"),
            capabilities=DistributedCapabilities(checkpoint_store_ref=checkpoint_ref),
        ),
        capability_registry=registry,
    )

    class BlockingApprovalProvider(ApprovalProvider):
        def should_request(self, request: ApprovalRequest) -> bool:
            del request
            return True

        def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
            del request
            return None

    provider = BlockingApprovalProvider()

    with pytest.raises(ValueError, match="do not support brokered approval waits"):
        Runner.start_distributed(
            Agent(name="approval-agent", instructions="Ask first.", model="test-model"),
            "work",
            run_config=RunConfig(
                model_provider=_provider(lambda: ScriptedLLM(steps=[])),
                execution_backend=backend,
                approval_provider=provider,
                checkpoint_config=CheckpointConfig(
                    key="nonblocking-approval",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                ),
            ),
        )

    assert store.load_checkpoint("nonblocking-approval") is None


def test_nonblocking_celery_rejects_brokered_approval_recipe(tmp_path: Path) -> None:
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-recipe-approval", "1")
    backend = CeleryBackend(
        celery_app=_EnqueueOnlyApp(),
        runtime_recipe=RuntimeRecipe(
            settings_file=str(tmp_path / "unused-settings.py"),
            backend="test",
            model="test-model",
            workspace=str(tmp_path / "workspace"),
            capabilities=DistributedCapabilities(
                approval_provider_ref=CapabilityRef("approval.provider", "1"),
                approval_broker_ref=CapabilityRef("approval.broker", "1"),
                checkpoint_store_ref=checkpoint_ref,
            ),
        ),
        capability_registry=DistributedCapabilityRegistry(),
    )

    with pytest.raises(DistributedContractError, match="do not support brokered approval waits"):
        backend.advance(previous_envelope={}, outcome="transport failed")


def test_celery_worker_rejects_brokered_approval_before_worker_effects(tmp_path: Path) -> None:
    handler_calls: list[dict[str, Any]] = []

    def invoke(_context: Any, arguments: dict[str, Any]) -> ToolOutputText:
        handler_calls.append(arguments)
        return ToolOutputText(text="handler ran")

    agent = Agent(
        name="direct-worker-approval-agent",
        instructions="Call the protected tool.",
        model="test-model",
        tools=[
            FunctionTool(
                name="protected",
                description="Run the protected operation.",
                params_json_schema={"type": "object", "properties": {}, "required": []},
                on_invoke=invoke,
            )
        ],
    )

    class BlockingApprovalProvider(ApprovalProvider):
        def should_request(self, request: ApprovalRequest) -> bool:
            del request
            return True

        def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
            del request
            return None

    approval_provider_ref = CapabilityRef("approval.provider.direct-worker", "1")
    approval_broker_ref = CapabilityRef("approval.broker.direct-worker", "1")
    checkpoint_ref = CapabilityRef("checkpoint.direct-worker", "1")
    session_ref = CapabilityRef("session.direct-worker", "1")
    llm_ref = CapabilityRef("llm.direct-worker", "1")
    checkpoint_key = "direct-worker-approval"
    model_settings = ModelSettings()
    store = InMemoryCheckpointStore()
    session = MemorySession("direct-worker-approval-session")
    checkpoint_config = CheckpointConfig(
        store=store,
        key=checkpoint_key,
        resume_policy=ResumePolicy.RESUME_IF_PRESENT,
        capability_refs={
            "approval_provider": approval_provider_ref.to_dict(),
            "approval_broker": approval_broker_ref.to_dict(),
            "session": session_ref.to_dict(),
        },
    )
    run_config = RunConfig(
        max_cycles=1,
        max_handoffs=8,
        no_tool_policy="finish",
        model_settings=model_settings,
        tool_policy=ToolPolicy(approval="default"),
        approval_provider=BlockingApprovalProvider(),
        approval_broker=ApprovalBroker(),
        session=session,
        checkpoint_config=checkpoint_config,
    )
    tool_registry = Runner._build_tool_registry(agent=agent, run_config=run_config)
    task = AgentCompiler().compile(
        agent=agent,
        input="work",
        run_config=run_config,
        resolved=_resolved(),
        trace_id="trace-direct-worker-approval",
        run_id="run-direct-worker-approval",
    )
    run_definition, run_definition_digest = build_run_definition(
        agent=agent,
        root_input="work",
        run_config=run_config,
        resolved=_resolved(),
        model_settings=model_settings,
        task=task,
        registry=tool_registry,
        initial_messages=[],
    )
    controller = CheckpointResumeController(
        config=checkpoint_config,
        task_id=task.task_id,
        run_id="run-direct-worker-approval",
        trace_id="trace-direct-worker-approval",
        run_definition=run_definition,
        run_definition_digest=run_definition_digest,
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )
    assert controller.admit() is None
    checkpoint_before = store.load_checkpoint(checkpoint_key)
    assert checkpoint_before is not None
    session_before = session.get_items()
    workspace = tmp_path / "worker-workspace"
    base_toolset_ref = ToolsetRef(
        id="toolset.direct-worker",
        version="1",
        schema_digest=toolset_schema_digest(tool_registry),
    )
    task_toolset_ref = ToolsetRef(
        id=base_toolset_ref.id,
        version=base_toolset_ref.version,
        schema_digest=toolset_schema_digest(tool_registry, task=task),
    )
    registry = DistributedCapabilityRegistry(include_defaults=False)
    registry.register_toolset(base_toolset_ref, tool_registry)
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-protected", name="protected", arguments={})],
                )
            ]
        ),
    )
    registry.register("approval_provider", approval_provider_ref, run_config.approval_provider)
    registry.register("approval_broker", approval_broker_ref, run_config.approval_broker)
    envelope = DistributedRunEnvelope.for_cycle(
        task=task,
        recipe=RuntimeRecipe(
            settings_file=str(tmp_path / "unused-settings.py"),
            backend="test",
            model="test-model",
            workspace=str(workspace),
            capabilities=DistributedCapabilities(
                toolset_ref=task_toolset_ref,
                tool_policy=DistributedToolPolicy(approval="default"),
                llm_client_ref=llm_ref,
                approval_provider_ref=approval_provider_ref,
                approval_broker_ref=approval_broker_ref,
                checkpoint_store_ref=checkpoint_ref,
            ),
        ),
        cycle_index=1,
        root_run_id="run-direct-worker-approval",
        trace_id="trace-direct-worker-approval",
        run_definition_digest=run_definition_digest,
        claim_mode="continue",
        resume_attempt=1,
        checkpoint_config=DistributedCheckpointConfig.from_checkpoint_config(checkpoint_config),
    )

    with pytest.raises(DistributedContractError, match="do not support brokered approval waits"):
        run_single_cycle(envelope_dict=envelope.to_dict(), capability_registry=registry)

    assert handler_calls == []
    assert store.load_checkpoint(checkpoint_key) == checkpoint_before
    assert session.get_items() == session_before
    assert not workspace.exists()


def test_nonblocking_celery_resolves_recipe_before_first_enqueue(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-capabilities", "1")
    missing_llm_ref = CapabilityRef("llm.nonblocking-missing", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=RuntimeRecipe(
            settings_file="",
            backend="test",
            model="test-model",
            workspace=str(tmp_path / "workspace"),
            capabilities=DistributedCapabilities(
                llm_client_ref=missing_llm_ref,
                checkpoint_store_ref=checkpoint_ref,
            ),
        ),
        capability_registry=registry,
    )

    with pytest.raises(DistributedCapabilityError, match="unknown distributed capability llm_client"):
        Runner.start_distributed(
            Agent(name="capability-agent", instructions="Answer.", model="test-model"),
            "work",
            run_config=RunConfig(
                model_provider=_provider(lambda: ScriptedLLM(steps=[])),
                execution_backend=backend,
                checkpoint_config=CheckpointConfig(
                    key="nonblocking-capabilities",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                ),
            ),
        )

    assert app.envelopes == []


def test_distributed_advance_decision_rejects_nonterminal_finalize_result() -> None:
    with pytest.raises(DistributedContractError, match="must have a terminal status"):
        DistributedAdvanceDecision(
            action="finalize_required",
            handle=DistributedRunHandle(
                checkpoint_key="nonterminal-finalize",
                run_id="run-nonterminal-finalize",
                trace_id="trace-nonterminal-finalize",
            ),
            checkpoint_revision=1,
            result=AgentResult(
                status=AgentStatus.RUNNING,
                messages=[],
                cycles=[],
            ),
        )


def test_nonblocking_celery_requires_the_recipe_checkpoint_fact_source(tmp_path: Path) -> None:
    controller_store = InMemoryCheckpointStore()
    recipe_store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-mismatch", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, recipe_store)
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=RuntimeRecipe(
            settings_file=str(tmp_path / "unused-settings.py"),
            backend="test",
            model="test-model",
            workspace=str(tmp_path / "workspace"),
            capabilities=DistributedCapabilities(checkpoint_store_ref=checkpoint_ref),
        ),
        capability_registry=registry,
    )

    with pytest.raises(CheckpointError, match="does not match the runtime recipe fact source") as caught:
        Runner.start_distributed(
            Agent(name="store-mismatch-agent", instructions="Answer.", model="test-model"),
            "work",
            run_config=RunConfig(
                model_provider=_provider(lambda: ScriptedLLM(steps=[])),
                execution_backend=backend,
                checkpoint_config=CheckpointConfig(
                    key="nonblocking-store-mismatch",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=controller_store,
                ),
            ),
        )

    assert caught.value.code == "checkpoint_store_conflict"
    assert app.envelopes == []


def test_nonblocking_celery_transport_failure_schedules_recovery_after_live_lease(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.nonblocking-recovery", "1")
    llm_ref = CapabilityRef("llm.nonblocking-recovery", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[LLMResponse(content="unused")]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        capability_registry=registry,
        dispatch_timeout_seconds=5,
    )
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="nonblocking-recovery",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )
    Runner.start_distributed(
        Agent(name="nonblocking-recovery-agent", instructions="Answer.", model="test-model"),
        "work",
        run_config=run_config,
    )
    now_ms = 1_000
    lease_expires_at_ms = 6_000
    claimed = store.claim_checkpoint(
        "nonblocking-recovery",
        1,
        claim_token="live-worker",
        lease_expires_at_ms=lease_expires_at_ms,
        now_ms=now_ms,
        claim_mode="continue",
    )
    assert claimed is not None

    decision = backend.advance(
        previous_envelope=app.envelopes[0],
        outcome=DistributedDeliveryOutcome.transport_failure("connection lost"),
        now_unix_ms=now_ms,
    )

    assert decision.action == "retry_at"
    assert decision.not_before_unix_ms == lease_expires_at_ms
    assert decision.envelope is not None
    assert decision.envelope.claim_mode == "recovery"
    assert decision.envelope.deadline_unix_ms == lease_expires_at_ms + 5_000
    assert len(app.envelopes) == 2
    assert app.options[-1]["countdown"] >= 0
    transport_outcome = DistributedDeliveryOutcome.transport_failure("connection lost")
    assert DistributedDeliveryOutcome.from_dict(transport_outcome.to_dict()) == transport_outcome


def test_celery_rejects_resolved_tool_metadata_drift_before_claim(
    tmp_path: Path,
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.metadata-drift", "2")
    llm_ref = CapabilityRef("llm.metadata-drift", "1")
    worker_model_calls = 0

    def worker_complete(_request: Any) -> LLMResponse:
        nonlocal worker_model_calls
        worker_model_calls += 1
        return LLMResponse(content="must not run")

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def invoke(_context: Any, _arguments: dict[str, Any]) -> ToolOutputText:
        return ToolOutputText(text="ok")

    local_tool = FunctionTool(
        name="inspect_source",
        description="Inspect one source.",
        params_json_schema=schema,
        on_invoke=invoke,
        tool_metadata=ToolMetadata(
            side_effect=ToolSideEffect.READ,
            idempotency=ToolIdempotency.SUPPORTED,
            capability_tags=["source.inspect"],
        ),
    )
    worker_tool = FunctionTool(
        name="inspect_source",
        description="Inspect one source.",
        params_json_schema=schema,
        on_invoke=invoke,
        tool_metadata=ToolMetadata(
            side_effect=ToolSideEffect.WRITE,
            idempotency=ToolIdempotency.SUPPORTED,
            capability_tags=["source.inspect"],
        ),
    )
    worker_tools = build_default_registry()
    worker_tools.register_executor(FunctionToolExecutor(worker_tool))
    toolset_ref = ToolsetRef(
        id="toolset.metadata-drift",
        version="1",
        schema_digest=toolset_schema_digest(worker_tools),
    )
    registry = DistributedCapabilityRegistry()
    registry.register_toolset(toolset_ref, worker_tools)
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(steps=[worker_complete]),
    )
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            toolset_ref=toolset_ref,
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _ImmediateApp(registry=registry, store=store)
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=1,
    )

    with pytest.raises(Exception) as caught:
        Runner.run_sync(
            Agent(
                name="metadata-drift-agent",
                instructions="Inspect only when needed.",
                model="test-model",
                tools=[local_tool],
            ),
            "answer without executing",
            run_config=RunConfig(
                model_provider=_provider(lambda: ScriptedLLM(steps=[])),
                execution_backend=backend,
                max_cycles=1,
                no_tool_policy="finish",
                checkpoint_config=CheckpointConfig(
                    key="distributed-metadata-drift",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                ),
            ),
        )

    assert getattr(caught.value, "code", None) == "checkpoint_definition_mismatch"
    error_message = str(caught.value)
    assert "actual_len=9" in error_message
    assert "expected_len=9" in error_message
    assert "actual_names=" in error_message
    assert "expected_names=" in error_message
    assert "task_extra_tool_names=" in error_message
    assert "task_exclude_tools=" in error_message
    assert "registry_planner_extra_tool_names=" in error_message
    assert "registry_tool_names=" in error_message
    assert "'inspect_source'" in error_message
    checkpoint = store.load_checkpoint("distributed-metadata-drift")
    assert checkpoint is not None
    assert checkpoint.claim_token is None
    assert checkpoint.model_call_journal == []
    assert checkpoint.tool_journal == []
    assert worker_model_calls == 0


def test_distributed_static_invalid_cases_fail_before_worker_resolution() -> None:
    fixture = _fixture()
    parser_cases = {
        "unsupported_schema_version",
        "missing_run_definition_schema",
        "unsupported_run_definition_schema",
        "missing_checkpoint_config",
        "missing_checkpoint_store_ref",
        "missing_required_extension_ref",
        "missing_claim_mode",
        "unknown_claim_mode",
        "unsafe_extension_limit",
    }

    for case in fixture["invalid_cases"]:
        if case["name"] not in parser_cases:
            continue
        payload = copy.deepcopy(fixture["canonical_envelope"])
        _set_path(payload, case["path"], case["value"])
        with pytest.raises(DistributedContractError, match=case["error"]):
            DistributedRunEnvelope.from_dict(payload)


def test_celery_resolves_and_restores_stateful_after_cycle_hook(
    tmp_path: Path,
) -> None:
    class StatefulLifecycleHook:
        namespace = "com.example.lifecycle"
        version = "1"
        required = True

        def __init__(self) -> None:
            self.observed_cycles = 0
            self.restored_states: list[int] = []
            self.snapshot_cycles: list[int] = []

        def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
            self.observed_cycles += 1
            self.snapshot_cycles.append(snapshot.cycle_index)
            if self.observed_cycles == 1:
                return AfterCycleDecision.steer(
                    ["Verify the candidate answer."],
                    disallow_tools=["task_finish"],
                )
            return AfterCycleDecision.continue_run()

        def snapshot(self) -> dict[str, int]:
            return {"observed_cycles": self.observed_cycles}

        def restore(self, state: Any) -> None:
            assert isinstance(state, dict)
            value = state.get("observed_cycles")
            assert isinstance(value, int)
            self.observed_cycles = value
            self.restored_states.append(value)

    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.lifecycle", "2")
    llm_ref = CapabilityRef("llm.lifecycle", "1")
    hook_ref = CapabilityRef("lifecycle.policy", "1")
    extension_ref = CapabilityRef("lifecycle.policy-state", "1")
    hook = StatefulLifecycleHook()
    requests: list[Any] = []

    def first_answer(request: Any) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content="candidate")

    def verified_answer(request: Any) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content="verified")

    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(steps=[first_answer, verified_answer]),
    )
    registry.register("after_cycle_hook", hook_ref, hook)
    registry.register("checkpoint_extension", extension_ref, hook)
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            after_cycle_hook_refs=(hook_ref,),
            checkpoint_store_ref=checkpoint_ref,
            checkpoint_extension_refs=(
                CheckpointExtensionRef(
                    namespace=hook.namespace,
                    reference=extension_ref,
                    required=True,
                ),
            ),
        ),
    )
    app = _ImmediateApp(registry=registry, store=store)
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=5,
    )
    config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=3,
        no_tool_policy="finish",
        after_cycle_hooks=[hook],
        checkpoint_config=CheckpointConfig(
            key="distributed-after-cycle",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            required_extension_namespaces=[hook.namespace],
            capability_refs={
                "after_cycle_hook:0": hook_ref.to_dict(),
            },
        ),
        checkpoint_extensions=[hook],
    )

    result = Runner.run_sync(
        Agent(
            name="distributed-lifecycle-agent",
            instructions="Return one checked answer.",
            model="test-model",
        ),
        "answer",
        run_config=config,
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "verified"
    assert hook.snapshot_cycles == [1, 2]
    assert 1 in hook.restored_states
    assert len(requests) == 2
    assert [response["type"] for response in app.worker_responses] == [
        "committed",
        "terminal_candidate",
    ]
    assert requests[1].messages[-1].content == "Verify the candidate answer."
    assert all(envelope["recipe"]["capabilities"]["after_cycle_hook_refs"] == [hook_ref.to_dict()] for envelope in app.envelopes)
    terminal = store.load_checkpoint("distributed-after-cycle")
    assert terminal is not None
    assert terminal.shared_state["_vv_agent_after_cycle_control"] == {
        "schema_version": "vv-agent.after-cycle-control.v1",
        "disallowed_tools": ["task_finish"],
    }
    assert terminal.extension_state[hook.namespace].state == {"observed_cycles": 2}


@pytest.mark.parametrize("transport_redelivered", [False, True])
def test_celery_returns_candidate_then_runner_owns_terminal_order(
    tmp_path: Path,
    transport_redelivered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCheckpointStore()
    renewal_arguments: list[set[str]] = []
    original_renew = store.renew_checkpoint_claim

    def recording_renew(checkpoint_key: str, **kwargs: Any) -> CheckpointRenewal:
        renewal_arguments.append(set(kwargs))
        return original_renew(checkpoint_key, **kwargs)

    monkeypatch.setattr(store, "renew_checkpoint_claim", recording_renew)
    checkpoint_ref = CapabilityRef("checkpoint.test", "2")
    llm_ref = CapabilityRef("llm.test", "1")
    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    observed_temperatures: list[float | None] = []

    def worker_complete(request: Any) -> LLMResponse:
        observed_temperatures.append(request.model_settings.temperature)
        return LLMResponse(content="raw worker answer")

    registry.register(
        "llm_client",
        llm_ref,
        ScriptedLLM(steps=[worker_complete]),
    )
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _ImmediateApp(
        registry=registry,
        store=store,
        transport_redelivered=transport_redelivered,
    )
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=5,
        lease_duration_ms=10_000,
    )
    guardrail_observations: list[tuple[bool, bool]] = []

    def output_guardrail(_context: Any, value: Any) -> GuardrailResult:
        checkpoint = store.load_checkpoint("distributed-v2")
        assert checkpoint is not None
        guardrail_observations.append((checkpoint.terminal_result is not None, checkpoint.claim_token is not None))
        return GuardrailResult.rewrite(f"guarded: {value}")

    session = MemorySession("distributed-v2-session")
    agent = Agent(
        name="distributed-v2-agent",
        instructions="Return one answer.",
        model="test-model",
        output_guardrails=[output_guardrail],
    )
    provider = _provider(
        lambda: ScriptedLLM(steps=[]),
        settings=ModelSettings(temperature=0.25),
    )
    result = Runner.run_sync(
        agent,
        "answer once",
        run_config=RunConfig(
            model_provider=provider,
            execution_backend=backend,
            max_cycles=1,
            no_tool_policy="finish",
            session=session,
            checkpoint_config=CheckpointConfig(
                key="distributed-v2",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
                capability_refs={
                    "output_guardrail:0": {"id": "guardrail.test", "version": "1"},
                    "session": {"id": "session.test", "version": "1"},
                },
            ),
        ),
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "guarded: raw worker answer"
    assert observed_temperatures == [0.25]
    assert guardrail_observations == [(False, True)]
    assert len(app.envelopes) == 1
    assert app.envelopes[0]["schema_version"] == DISTRIBUTED_RUN_SCHEMA_VERSION
    assert app.worker_responses[0]["schema_version"] == DISTRIBUTED_WORKER_RESPONSE_SCHEMA_VERSION
    assert app.worker_responses[0]["type"] == "terminal_candidate"
    assert set(app.worker_responses[0]) == {
        "schema_version",
        "type",
        "checkpoint_revision",
        "result",
    }
    worker = app.worker_snapshots[0]
    assert worker is not None
    assert worker.terminal_result is None
    assert worker.claim_token is not None
    assert worker.model_call_journal[0].state.value == "succeeded"
    assert worker.resume_attempt == (2 if transport_redelivered else 1)
    assert len(renewal_arguments) >= 2
    assert all(arguments == {"claim_token", "lease_expires_at_ms", "now_ms"} for arguments in renewal_arguments)

    terminal = store.load_checkpoint("distributed-v2")
    assert terminal is not None
    assert terminal.terminal_result is not None
    assert terminal.terminal_result.final_answer == "guarded: raw worker answer"
    assert terminal.claim_token is None
    assert terminal.terminal_acknowledged
    assert all(entry.state == "delivered" for entry in terminal.event_outbox)
    assert len(session.get_items()) > 0


@pytest.mark.parametrize("transport_redelivered", [False, True])
def test_celery_worker_terminal_replay_repairs_pending_delivery_without_model_calls(
    tmp_path: Path, transport_redelivered: bool
) -> None:
    store, registry, _backend, envelope, model_calls = _terminal_replay_window(
        tmp_path, transport_redelivered=transport_redelivered
    )
    before = store.load_checkpoint("terminal-replay-repair")
    assert before is not None
    assert before.resume_attempt == envelope["resume_attempt"] + int(transport_redelivered)
    event_ids = tuple(entry.event_id for entry in before.event_outbox)

    first = DistributedWorkerResponse.from_dict(
        run_single_cycle(
            envelope_dict=envelope,
            capability_registry=registry,
        )
    )

    assert first.response_type == "terminal_replay"
    assert first.result is not None
    assert first.result.final_answer == "worker answer"
    repaired = store.load_checkpoint("terminal-replay-repair")
    assert repaired is not None
    assert first.checkpoint_revision == repaired.revision
    assert repaired.claim_token is None
    assert repaired.terminal_acknowledged
    assert all(entry.state == "delivered" for entry in repaired.event_outbox)
    assert tuple(entry.event_id for entry in repaired.event_outbox) == event_ids
    assert model_calls == [1]

    second = DistributedWorkerResponse.from_dict(
        run_single_cycle(
            envelope_dict=envelope,
            capability_registry=registry,
        )
    )
    assert second.response_type == "terminal_replay"
    assert second.checkpoint_revision == repaired.revision
    replayed = store.load_checkpoint("terminal-replay-repair")
    assert replayed is not None
    assert replayed.revision == repaired.revision
    assert tuple(entry.event_id for entry in replayed.event_outbox) == event_ids
    assert model_calls == [1]


def test_celery_local_terminal_replay_repairs_pending_delivery_without_model_calls(tmp_path: Path) -> None:
    store, _registry, backend, envelope, model_calls = _terminal_replay_window(tmp_path)
    checkpoint = store.load_checkpoint("terminal-replay-repair")
    assert checkpoint is not None
    assert checkpoint.terminal_result is not None
    event_ids = tuple(entry.event_id for entry in checkpoint.event_outbox)
    controller = CheckpointResumeController(
        config=CheckpointConfig(
            store=store,
            key=checkpoint.checkpoint_key,
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
        ),
        task_id=checkpoint.task_id,
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        run_definition=checkpoint.run_definition,
        run_definition_digest=checkpoint.run_definition_digest,
        initial_messages=checkpoint.messages,
        initial_shared_state=checkpoint.shared_state,
        initial_budget_usage=checkpoint.budget_usage,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
        preloaded_checkpoint=checkpoint,
    )
    controller.checkpoint = checkpoint
    decoded = DistributedRunEnvelope.from_dict(envelope)
    try:
        replay = backend.execute_local(
            task=decoded.task,
            initial_messages=[],
            shared_state={},
            cycle_executor=lambda *_args: None,
            ctx=ExecutionContext(metadata={"_vv_agent_checkpoint_controller": controller}),
            max_cycles=decoded.task.max_cycles,
        )
    finally:
        controller.close()

    assert replay.final_answer == "worker answer"
    replayed = store.load_checkpoint("terminal-replay-repair")
    assert replayed is not None
    assert replayed.revision == checkpoint.revision + 2
    assert replayed.claim_token is None
    assert replayed.terminal_acknowledged
    assert all(entry.state == "delivered" for entry in replayed.event_outbox)
    assert tuple(entry.event_id for entry in replayed.event_outbox) == event_ids
    assert model_calls == [1]


@pytest.mark.parametrize(
    ("case_name", "expected_error"),
    [
        ("definition_digest", "checkpoint_definition_mismatch"),
        ("resume_attempt", "checkpoint_resume_attempt_mismatch"),
        ("root_identity", "checkpoint_definition_mismatch"),
        ("checkpoint_policy", "checkpoint_definition_mismatch"),
        ("metadata_policy", "checkpoint_definition_mismatch"),
        ("missing_capability", "unknown distributed capability hook hook.missing@1"),
        (
            "missing_after_cycle_capability",
            "unknown distributed capability after_cycle_hook lifecycle.missing@1",
        ),
    ],
)
def test_celery_rejects_identity_definition_config_and_capability_before_claim(
    tmp_path: Path,
    case_name: str,
    expected_error: str,
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.validation", "2")
    llm_ref = CapabilityRef("llm.validation", "1")
    model_calls = 0

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(content="must not run")

    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[complete]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )

    def mutate(envelope: dict[str, Any]) -> None:
        if case_name == "definition_digest":
            envelope["run_definition_digest"] = "d" * 64
        elif case_name == "resume_attempt":
            envelope["resume_attempt"] += 1
        elif case_name == "root_identity":
            envelope["root_run_id"] = "run-other"
        elif case_name == "checkpoint_policy":
            envelope["checkpoint_config"]["ambiguous_tool_policy"] = "retry_idempotent_only"
        elif case_name == "metadata_policy":
            envelope["recipe"]["capabilities"]["tool_policy"]["deny_terminal_tools"] = True
        elif case_name == "missing_capability":
            envelope["recipe"]["capabilities"]["hook_refs"] = [{"id": "hook.missing", "version": "1"}]
        else:
            envelope["recipe"]["capabilities"]["after_cycle_hook_refs"] = [{"id": "lifecycle.missing", "version": "1"}]

    app = _ImmediateApp(
        registry=registry,
        store=store,
        mutate_envelope=mutate,
    )
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=1,
    )
    agent = Agent(
        name="distributed-validation-agent",
        instructions="Return one answer.",
        model="test-model",
    )
    config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key=f"distributed-validation-{case_name}",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )

    if case_name in {"missing_capability", "missing_after_cycle_capability"}:
        with pytest.raises(DistributedCapabilityError, match=expected_error):
            Runner.run_sync(agent, "validate first", run_config=config)
    else:
        with pytest.raises(Exception) as caught:
            Runner.run_sync(agent, "validate first", run_config=config)
        assert getattr(caught.value, "code", None) == expected_error

    checkpoint = store.load_checkpoint(f"distributed-validation-{case_name}")
    assert checkpoint is not None
    assert checkpoint.claim_token is None
    assert checkpoint.model_call_journal == []
    assert checkpoint.tool_journal == []
    assert checkpoint.terminal_result is None
    assert model_calls == 0


@pytest.mark.parametrize("first_outcome", ["transport_error", "pending"])
def test_celery_scheduler_retry_redispatches_as_recovery(
    tmp_path: Path,
    first_outcome: str,
) -> None:
    store = InMemoryCheckpointStore()
    checkpoint_ref = CapabilityRef("checkpoint.retry", "2")
    llm_ref = CapabilityRef("llm.retry", "1")
    model_calls = 0

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(content="recovered")

    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[complete]))
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "unused-settings.py"),
        backend="test",
        model="test-model",
        workspace=str(tmp_path / "workspace"),
        capabilities=DistributedCapabilities(
            llm_client_ref=llm_ref,
            checkpoint_store_ref=checkpoint_ref,
        ),
    )
    app = _ImmediateApp(
        registry=registry,
        store=store,
        fail_first_dispatch=first_outcome == "transport_error",
        pending_first_dispatch=first_outcome == "pending",
    )
    backend = CeleryBackend(
        celery_app=app,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=2,
        lease_duration_ms=10_000,
    )
    result = Runner.run_sync(
        Agent(
            name="distributed-retry-agent",
            instructions="Return one answer.",
            model="test-model",
        ),
        "recover delivery",
        run_config=RunConfig(
            model_provider=_provider(lambda: ScriptedLLM(steps=[])),
            execution_backend=backend,
            max_cycles=1,
            no_tool_policy="finish",
            checkpoint_config=CheckpointConfig(
                key="distributed-retry",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            ),
        ),
    )

    assert result.final_output == "recovered"
    assert model_calls == 1
    assert len(app.envelopes) == 2
    assert app.envelopes[0]["claim_mode"] == "continue"
    assert app.envelopes[1]["claim_mode"] == "recovery"
    if first_outcome == "pending":
        assert app.worker_responses[0] == DistributedWorkerResponse.pending().to_dict()
    terminal = store.load_checkpoint("distributed-retry")
    assert terminal is not None
    assert terminal.resume_attempt == 2


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "produce_host_interaction",
        "claim_and_consume_host_interaction_response",
        "model_receipt",
        "concurrent_recovery",
        "unclaimed_recovery",
        "completed_wake_race",
    ],
)
def test_real_tool_host_interaction_retains_receipt_before_waiting(tmp_path: Path, fault: str | None) -> None:
    import multiprocessing
    import os
    import time

    from vv_agent import ToolCallOutcome, ToolExecutionResult
    from vv_agent.runtime.stores.redis import RedisCheckpointStore
    from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore

    exchange_write_dir = os.environ.get("VV_AGENT_CROSS_HOST_WRITE_DIR")
    if exchange_write_dir is not None:
        assert fault is None, "cross-language producer requires the no-fault case"
        tmp_path = Path(exchange_write_dir)
        assert tmp_path.is_dir()
    if fault is not None and "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("process-exit verification requires fork")
    process_context = (
        multiprocessing.context.ForkContext()
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.context.SpawnContext()
    )
    redis_url = os.environ.get("VV_AGENT_CROSS_HOST_REDIS_URL")
    if redis_url is not None:
        assert exchange_write_dir is not None and fault is None
    store: Any = RedisCheckpointStore(redis_url) if redis_url else SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
    calls = process_context.Array("i", [0, 0, 0])
    model_calls = process_context.Value("i", 0)
    model_entered = process_context.Event()
    release_model = process_context.Event()

    def request_choice(context: Any, _arguments: dict[str, Any]) -> ToolCallOutcome:
        with calls.get_lock():
            calls[0] += 1
        plan = context.metadata["_vv_agent_checkpoint_plan"]
        outcome = ToolCallOutcome.HostInteraction(
            ToolExecutionResult(tool_call_id=context.tool_call_id, content="Region choice requested."),
            HostInteractionRequest(
                interaction_id="region-choice",
                logical_cycle=context.cycle_index,
                operation_id=plan.operation_id,
                tool_call_id=context.tool_call_id,
                prompt="Choose a region: https://example.invalid/?keep=1",
            ),
        )
        active = store.load_checkpoint("host-tool")
        assert active is not None and active.claim_token is not None
        assert active.status is AgentStatus.RUNNING
        with calls.get_lock():
            calls[1] += 1
        return outcome

    def later(_context: Any, _arguments: dict[str, Any]) -> ToolOutputText:
        with calls.get_lock():
            calls[2] += 1
        return ToolOutputText(text="Later result.")

    tools = [
        FunctionTool(
            name=name,
            description=name,
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke=callback,
        )
        for name, callback in (("request_choice", request_choice), ("later", later))
    ]
    worker_tools = build_default_registry()
    for tool in tools:
        worker_tools.register_executor(FunctionToolExecutor(tool))
    toolset = ToolsetRef("host-tool", "1", toolset_schema_digest(worker_tools))
    registry = DistributedCapabilityRegistry()
    checkpoint_ref, llm_ref = CapabilityRef("host-tool-store", "1"), CapabilityRef("host-tool-llm", "1")
    registry.register("checkpoint_store", checkpoint_ref, store)
    registry.register_toolset(toolset, worker_tools)

    def complete(request: Any) -> LLMResponse:
        with model_calls.get_lock():
            model_calls.value += 1
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="choice", name="request_choice", arguments={}),
                ToolCall(id="later", name="later", arguments={}),
            ],
        )

    registry.register("llm_client", llm_ref, ScriptedLLM(steps=[complete]))
    app = _EnqueueOnlyApp()
    backend = CeleryBackend(
        celery_app=app,
        capability_registry=registry,
        runtime_recipe=RuntimeRecipe(
            settings_file="",
            backend="test",
            model="test-model",
            workspace=str(tmp_path),
            capabilities=DistributedCapabilities(
                toolset_ref=toolset,
                llm_client_ref=llm_ref,
                checkpoint_store_ref=checkpoint_ref,
            ),
        ),
    )
    agent = Agent(name="region", model="test-model", instructions="Ask before continuing.", tools=tools)
    run_config = RunConfig(
        model_provider=_provider(lambda: ScriptedLLM(steps=[])),
        execution_backend=backend,
        max_cycles=3,
        checkpoint_config=CheckpointConfig(key="host-tool", store=store, resume_policy=ResumePolicy.NEW),
    )

    def race_recovery(envelope: dict[str, Any]) -> dict[str, Any]:
        barrier = process_context.Barrier(2, timeout=10)
        results = process_context.Queue()
        release_owner = process_context.Event()
        release_loser_claim = process_context.Event()
        delayed_claim = fault == "completed_wake_race"
        before = store.load_checkpoint("host-tool")
        assert before is not None and before.claim_token is None
        original_claim = SqliteCheckpointStore.claim_controller_command_wake
        original_consume = SqliteCheckpointStore.claim_and_consume_host_interaction_response

        def contender(index: int) -> None:
            separate = SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
            registry.register("checkpoint_store", checkpoint_ref, separate)

            def synchronized_claim(*args: Any, **kwargs: Any) -> Any:
                assert separate.load_checkpoint("host-tool") == before
                barrier.wait()
                if delayed_claim and index == 1:
                    assert release_loser_claim.wait(15)
                return original_claim(*args, **kwargs)

            def hold_owner(*args: Any, **kwargs: Any) -> Any:
                result = original_consume(*args, **kwargs)
                if result.kind == "applied" and not delayed_claim:
                    results.put(("admitted", separate.load_checkpoint("host-tool")))
                    assert release_owner.wait(15), "contender did not finish"
                return result

            try:
                with (
                    patch.object(SqliteCheckpointStore, "claim_controller_command_wake", synchronized_claim),
                    patch.object(SqliteCheckpointStore, "claim_and_consume_host_interaction_response", hold_owner),
                ):
                    results.put(("result", run_single_cycle(envelope_dict=envelope, capability_registry=registry)))
            finally:
                separate.close()

        processes = [process_context.Process(target=contender, args=(index,)) for index in range(2)]
        try:
            for process in processes:
                process.start()
            if delayed_claim:
                assert model_entered.wait(15)
                admitted = store.load_checkpoint("host-tool")
                release_loser_claim.set()
            else:
                kind, admitted = results.get(timeout=15)
                assert kind == "admitted"
            kind, loser = results.get(timeout=15)
            assert kind == "result" and loser["type"] == "pending"
            assert store.load_checkpoint("host-tool") == admitted
            assert admitted is not None
            if not delayed_claim:
                assert admitted.revision == before.revision + 1
            assert admitted.resume_attempt == before.resume_attempt + 1
            assert admitted.claimed_cycle == 2
            assert model_calls.value == (2 if delayed_claim else 1) and calls[:] == [1, 1, 0]
            release_owner.set()
            release_model.set()
            kind, winner = results.get(timeout=15)
            assert kind == "result" and winner["type"] == "terminal_candidate"
            for process in processes:
                process.join(timeout=5)
                assert process.exitcode == 0
            return winner
        finally:
            release_owner.set()
            release_loser_claim.set()
            release_model.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            results.close()
            results.join_thread()

    def deliver(envelope: dict[str, Any], crash_at: str | None = None) -> dict[str, Any] | None:
        nonlocal store
        if fault is None:
            return run_single_cycle(envelope_dict=envelope, capability_registry=registry)
        store.close()
        receive, send = process_context.Pipe(duplex=False)

        def child() -> None:
            nonlocal store
            receive.close()
            store = SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
            registry.register("checkpoint_store", checkpoint_ref, store)
            try:
                if crash_at is not None:
                    method = "progress_checkpoint" if crash_at == "model_receipt" else crash_at
                    original = getattr(SqliteCheckpointStore, method)

                    def exit_after_commit(*args: Any, **kwargs: Any) -> Any:
                        value = original(*args, **kwargs)
                        if crash_at == "model_receipt" and not any(
                            entry.cycle_index == 2 and entry.state.value == "succeeded" for entry in args[1].model_call_journal
                        ):
                            return value
                        os._exit(23)

                    with patch.object(SqliteCheckpointStore, method, exit_after_commit):
                        if crash_at in {"claim_and_consume_host_interaction_response", "model_receipt"}:
                            expired_clock = time.time_ns() - 600_000_000_000
                            with patch("time.time_ns", return_value=expired_clock):
                                run_single_cycle(envelope_dict=envelope, capability_registry=registry)
                        else:
                            run_single_cycle(envelope_dict=envelope, capability_registry=registry)
                    raise AssertionError("worker did not reach the committed fault")
                send.send(run_single_cycle(envelope_dict=envelope, capability_registry=registry))
            finally:
                store.close()
                send.close()

        process = process_context.Process(target=child)
        process.start()
        send.close()
        try:
            if fault == "concurrent_recovery" and envelope["cycle_index"] == 2:
                assert model_entered.wait(10), "recovery worker did not enter the model"
                probe = SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
                before = probe.load_checkpoint("host-tool")
                assert before is not None and before.claim_token is not None
                assert model_calls.value == 2
                duplicate_receive, duplicate_send = process_context.Pipe(duplex=False)

                def duplicate() -> None:
                    separate = SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
                    registry.register("checkpoint_store", checkpoint_ref, separate)
                    try:
                        duplicate_send.send(run_single_cycle(envelope_dict=envelope, capability_registry=registry))
                    finally:
                        separate.close()
                        duplicate_send.close()

                contender = process_context.Process(target=duplicate)
                contender.start()
                duplicate_send.close()
                try:
                    assert duplicate_receive.poll(10), "duplicate worker did not finish"
                    assert duplicate_receive.recv()["type"] == "pending"
                    contender.join(timeout=5)
                    assert contender.exitcode == 0
                    assert probe.load_checkpoint("host-tool") == before
                    assert model_calls.value == 2 and calls[:] == [1, 1, 0]
                finally:
                    if contender.is_alive():
                        contender.terminate()
                        contender.join(timeout=5)
                    duplicate_receive.close()
                    probe.close()
                    release_model.set()
            assert receive.poll(30), "worker did not return or exit"
            try:
                delivered = receive.recv()
            except EOFError:
                delivered = None
            process.join(timeout=5)
            assert process.exitcode == (23 if crash_at is not None else 0)
            return delivered
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            receive.close()
            store = SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
            registry.register("checkpoint_store", checkpoint_ref, store)
            run_config.checkpoint_config = CheckpointConfig(
                key="host-tool", store=store, resume_policy=ResumePolicy.REQUIRE_EXISTING
            )

    try:
        handle = Runner.start_distributed(agent, "Choose a region.", run_config=run_config)
        if fault == "produce_host_interaction":
            assert deliver(app.envelopes[0], crash_at=fault) is None
        response = deliver(app.envelopes[0])
        assert response is not None and response["type"] == "pending"
        assert calls[:] == [1, 1, 0]
        assert model_calls.value == 1
        checkpoint = store.load_checkpoint(handle.checkpoint_key)
        assert checkpoint is not None
        assert checkpoint.status is AgentStatus.HOST_INTERACTION
        assert checkpoint.claim_token is None
        assert checkpoint.cycle_index == 1
        assert checkpoint.tool_journal == checkpoint.model_call_journal == []
        assert checkpoint.cycles[-1].tool_results[0].content == "Region choice requested."
        assert checkpoint.cycles[-1].tool_results[1].error_code == "skipped_due_to_host_interaction"
        if exchange_write_dir is not None:
            return
        decision = backend.advance(
            previous_envelope=app.envelopes[0],
            outcome=DistributedDeliveryOutcome.worker(response),
        )
        assert decision.action == "wait"
        request = HostInteractionRequest.from_dict(checkpoint.active_host_interaction)
        response_text = "Europe https://example.invalid/?keep=1"
        command = ControllerCommand(
            command_id="region-response",
            handle=handle,
            resume_attempt=checkpoint.resume_attempt,
            expected_revision=checkpoint.revision,
            command={
                "kind": "host_interaction_response",
                "interaction_id": request.interaction_id,
                "logical_cycle": request.logical_cycle,
                "operation_id": request.operation_id,
                "tool_call_id": request.tool_call_id,
                "request_digest": request.request_digest,
                "response": {"role": "user", "content": response_text},
            },
        )
        assert store.resolve_controller_command(command).kind == "applied"
        resolved = store.load_checkpoint(handle.checkpoint_key)
        assert store.resolve_controller_command(command).kind == "replayed"
        assert store.load_checkpoint(handle.checkpoint_key) == resolved
        store.close()
        store = SqliteCheckpointStore(tmp_path / "host-tool.sqlite")
        registry.register("checkpoint_store", checkpoint_ref, store)
        run_config.checkpoint_config = CheckpointConfig(key="host-tool", store=store, resume_policy=ResumePolicy.REQUIRE_EXISTING)

        def resume_model(request: Any) -> LLMResponse:
            with model_calls.get_lock():
                model_calls.value += 1
            assert sum(message.content == response_text for message in request.messages) == 1
            assert any(message.content == "Region choice requested." for message in request.messages)
            if fault in {"concurrent_recovery", "completed_wake_race"}:
                model_entered.set()
                assert release_model.wait(15), "duplicate worker did not release the model"
            return LLMResponse(content="Europe selected.")

        registry.register("llm_client", llm_ref, ScriptedLLM(steps=[resume_model]))
        recovery = backend.advance(previous_envelope=app.envelopes[0], outcome=DistributedWorkerResponse.pending())
        assert recovery.action == "dispatch" and recovery.envelope is not None
        if fault in {"claim_and_consume_host_interaction_response", "model_receipt"}:
            assert deliver(recovery.envelope.to_dict(), crash_at=fault) is None
            consumed = store.load_checkpoint(handle.checkpoint_key)
            assert consumed is not None and consumed.claim_token is not None
            assert consumed.lease_expires_at_ms is not None
            assert consumed.lease_expires_at_ms < time.time_ns() // 1_000_000
            assert model_calls.value == (2 if fault == "model_receipt" else 1)
            if fault == "model_receipt":
                assert consumed.model_call_journal[0].state.value == "succeeded"
            assert sum(message.content == response_text for message in consumed.messages) == 1
            with pytest.raises(CheckpointError, match="resume_attempt"):
                run_single_cycle(envelope_dict=recovery.envelope.to_dict(), capability_registry=registry)
            assert store.load_checkpoint(handle.checkpoint_key) == consumed
            recovery = backend.advance(
                previous_envelope=recovery.envelope, outcome=RuntimeError("worker exited after response consumption")
            )
            assert recovery.action == "dispatch" and recovery.envelope is not None
            assert store.load_checkpoint(handle.checkpoint_key) == consumed
        recovered = (
            race_recovery(recovery.envelope.to_dict())
            if fault in {"unclaimed_recovery", "completed_wake_race"}
            else deliver(recovery.envelope.to_dict())
        )
        assert recovered is not None
        assert recovered["type"] == "terminal_candidate"
        assert recovered["result"]["status"] == "completed", recovered["result"].get("error")
        assert recovered["result"]["final_answer"] == "Europe selected."
        assert calls[:] == [1, 1, 0]
        assert model_calls.value == 2
        awaiting_finalization = store.load_checkpoint(handle.checkpoint_key)
        assert awaiting_finalization is not None
        assert recovered["result"]["token_usage"]["model_calls"] == [call.to_dict() for call in awaiting_finalization.model_calls]
        assert run_single_cycle(envelope_dict=recovery.envelope.to_dict(), capability_registry=registry)["type"] == "pending"
        assert store.load_checkpoint(handle.checkpoint_key) == awaiting_finalization
        finish = backend.advance(previous_envelope=recovery.envelope, outcome=recovered)
        assert finish.action == "finalize_required"
        result = Runner.finalize_distributed(agent, "Choose a region.", decision=finish, run_config=run_config)
        assert result.status is AgentStatus.COMPLETED
        assert result.final_output == "Europe selected."
        terminal = store.load_checkpoint(handle.checkpoint_key)
        assert terminal is not None
        assert terminal.terminal_acknowledged and terminal.claim_token is None
        assert len(terminal.model_calls) == 2
        assert (
            Runner.finalize_distributed(agent, "Choose a region.", decision=finish, run_config=run_config).final_output
            == result.final_output
        )
        assert store.load_checkpoint(handle.checkpoint_key) == terminal
        assert calls[:] == [1, 1, 0] and model_calls.value == 2
    finally:
        if isinstance(store, SqliteCheckpointStore):
            store.close()
        else:
            store._client.close()


def test_cross_language_host_interaction_store() -> None:
    import os

    from vv_agent.runtime.controller import HostInteractionRecoveryEnvelope, derive_host_interaction_record_id
    from vv_agent.runtime.stores.redis import RedisCheckpointStore
    from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore

    directory = os.environ.get("VV_AGENT_CROSS_HOST_DIR")
    if directory is None:
        pytest.skip("requires a real cross-language host-interaction checkpoint")
    mode = os.environ["VV_AGENT_CROSS_HOST_MODE"]
    assert mode in {"respond", "read"}
    redis_url = os.environ.get("VV_AGENT_CROSS_HOST_REDIS_URL")
    store = RedisCheckpointStore(redis_url) if redis_url else SqliteCheckpointStore(Path(directory) / "host-tool.sqlite")
    text = "Europe 中文 https://example.invalid/?keep=1"
    try:
        checkpoint = store.load_checkpoint("host-tool")
        assert checkpoint is not None
        assert checkpoint.cycle_index == 1 and len(checkpoint.model_calls) == 1
        assert checkpoint.cycles[0].tool_results[0].content == "Region choice requested."
        assert checkpoint.cycles[0].tool_results[1].error_code == "skipped_due_to_host_interaction"
        assert not checkpoint.tool_journal and not checkpoint.model_call_journal
        if mode == "respond":
            request = HostInteractionRequest.from_dict(checkpoint.active_host_interaction)
            assert request.request_digest is not None
            assert request.prompt == "Choose a region: https://example.invalid/?keep=1"
            command = ControllerCommand(
                command_id="cross-response",
                handle=DistributedRunHandle("host-tool", checkpoint.root_run_id, checkpoint.trace_id),
                resume_attempt=checkpoint.resume_attempt,
                expected_revision=checkpoint.revision,
                command={
                    "kind": "host_interaction_response",
                    "interaction_id": request.interaction_id,
                    "logical_cycle": request.logical_cycle,
                    "operation_id": request.operation_id,
                    "tool_call_id": request.tool_call_id,
                    "request_digest": request.request_digest,
                    "response": {"role": "user", "content": text},
                },
            )
            assert store.resolve_controller_command(command).kind == "applied"
            resolved = store.load_checkpoint("host-tool")
            assert resolved is not None
            assert store.resolve_controller_command(command).kind == "replayed"
            assert store.load_checkpoint("host-tool") == resolved
            recovery = HostInteractionRecoveryEnvelope(
                record_id=derive_host_interaction_record_id("host-tool", request),
                checkpoint_key="host-tool",
                run_id=checkpoint.root_run_id,
                trace_id=checkpoint.trace_id,
                claim_mode="recovery",
                resume_attempt=resolved.resume_attempt,
                expected_revision=resolved.revision,
                logical_cycle=request.logical_cycle,
                interaction_id=request.interaction_id,
                operation_id=request.operation_id,
                tool_call_id=request.tool_call_id,
                request_digest=request.request_digest,
                command_id=command.command_id,
            )
            assert store.claim_and_consume_host_interaction_response(recovery.to_dict()).kind == "applied"
            consumed = store.load_checkpoint("host-tool")
            assert store.claim_and_consume_host_interaction_response(recovery.to_dict()).kind == "replayed"
            assert store.load_checkpoint("host-tool") == consumed
        checkpoint = store.load_checkpoint("host-tool")
        assert checkpoint is not None
        assert checkpoint.status is AgentStatus.RUNNING
        assert checkpoint.claimed_cycle == 2 and checkpoint.claim_token is not None
        assert sum(message.content == text for message in checkpoint.messages) == 1
        assert checkpoint.terminal_result is None
    finally:
        if isinstance(store, SqliteCheckpointStore):
            store.close()
        else:
            store._client.close()


def test_registered_celery_task_forwards_transport_redelivery_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_single_cycle(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return DistributedWorkerResponse.pending().to_dict()

    monkeypatch.setattr(
        "vv_agent.runtime.backends.celery_tasks.run_single_cycle",
        fake_run_single_cycle,
    )

    class FakeCeleryApp:
        options: dict[str, Any]
        worker: Any

        def task(self, **options: Any) -> Callable[[Any], Any]:
            self.options = options

            def decorate(worker: Any) -> Any:
                self.worker = worker
                return worker

            return decorate

    app = FakeCeleryApp()
    registry = DistributedCapabilityRegistry()
    register_cycle_task(app, capability_registry=registry)
    bound_task = SimpleNamespace(
        request=SimpleNamespace(
            delivery_info={"redelivered": True},
            retries=3,
        )
    )

    result = app.worker(bound_task, envelope_dict={"schema_version": "test"})

    assert result == DistributedWorkerResponse.pending().to_dict()
    assert app.options["bind"] is True
    assert app.options["acks_late"] is True
    assert app.options["reject_on_worker_lost"] is True
    assert captured == {
        "envelope_dict": {"schema_version": "test"},
        "capability_registry": registry,
        "transport_redelivered": True,
        "transport_retry_count": 3,
    }
