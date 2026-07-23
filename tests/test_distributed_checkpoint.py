from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from vv_agent.budget import HostCost, RunBudgetLimits
from vv_agent.checkpoint import AmbiguousModelPolicy, AmbiguousToolPolicy, ResumePolicy, ToolIdempotency
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.runtime.backends.celery import CeleryBackend, register_cycle_task
from vv_agent.runtime.backends.celery_tasks import run_single_cycle
from vv_agent.runtime.backends.distributed import (
    DISTRIBUTED_RUN_SCHEMA_VERSION,
    DISTRIBUTED_WORKER_RESPONSE_SCHEMA_VERSION,
    CapabilityRef,
    CheckpointExtensionRef,
    DistributedCapabilities,
    DistributedCapabilityError,
    DistributedCapabilityRegistry,
    DistributedCheckpointConfig,
    DistributedContractError,
    DistributedRunEnvelope,
    DistributedToolPolicy,
    DistributedWorkerResponse,
    RuntimeRecipe,
    ToolsetRef,
    toolset_schema_digest,
)
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.tools import (
    FunctionTool,
    ToolMetadata,
    ToolOutputText,
    ToolSideEffect,
    build_default_registry,
)
from vv_agent.tools.executor import FunctionToolExecutor
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, Message, SubAgentConfig

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
        system_prompt="Use current wire only.",
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


def test_distributed_worker_response_matches_all_canonical_and_invalid_cases() -> None:
    fixture = json.loads(WORKER_RESPONSE_FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in fixture["valid_cases"]:
        response = DistributedWorkerResponse.from_dict(case["response"])
        assert response.to_dict() == case["response"], case["name"]

    for case in fixture["invalid_cases"]:
        with pytest.raises(DistributedContractError, match=case["error"]):
            DistributedWorkerResponse.from_dict(_worker_response_case_payload(fixture, case))


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
        ("schema_version", "vv-agent.distributed-run.v3"),
        ("run_definition_schema", None),
        ("run_definition_schema", "vv-agent.run-definition.v0"),
        ("run_definition_schema", "vv-agent.run-definition.v3"),
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
        dispatch_timeout_seconds=1,
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
        dispatch_timeout_seconds=1,
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

    def recording_renew(checkpoint_key: str, **kwargs: Any) -> bool:
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
    assert captured == {
        "envelope_dict": {"schema_version": "test"},
        "capability_registry": registry,
        "transport_redelivered": True,
        "transport_retry_count": 3,
    }
