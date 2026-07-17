from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent import Agent, CheckpointConfig, MemorySession, RunConfig, Runner
from vv_agent.checkpoint import ResumePolicy
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.backends.celery import CeleryBackend, register_cycle_task
from vv_agent.runtime.backends.celery_tasks import run_single_cycle
from vv_agent.runtime.backends.distributed import (
    DISTRIBUTED_RUN_SCHEMA_VERSION_V2,
    CapabilityRef,
    DistributedCapabilities,
    DistributedCapabilityError,
    DistributedCapabilityRegistry,
    DistributedContractError,
    DistributedRunEnvelope,
    RuntimeRecipe,
)
from vv_agent.runtime.state import InMemoryStateStore
from vv_agent.types import AgentStatus, LLMResponse

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "distributed_run_envelope_v2.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _set_path(payload: dict[str, Any], path: list[str], value: Any) -> None:
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


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


def _provider(factory: Callable[[], Any]) -> Callable[[Agent, RunConfig], tuple[Any, ResolvedModelConfig]]:
    def resolve(_agent: Agent, _run_config: RunConfig) -> tuple[Any, ResolvedModelConfig]:
        return factory(), _resolved()

    return resolve


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
        store: InMemoryStateStore,
        transport_redelivered: bool = False,
        mutate_envelope: Callable[[dict[str, Any]], None] | None = None,
        fail_first_dispatch: bool = False,
    ) -> None:
        self.registry = registry
        self.store = store
        self.transport_redelivered = transport_redelivered
        self.mutate_envelope = mutate_envelope
        self.fail_first_dispatch = fail_first_dispatch
        self.envelopes: list[dict[str, Any]] = []
        self.worker_snapshots: list[Any] = []

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
        result = run_single_cycle(
            envelope_dict=envelope,
            capability_registry=self.registry,
            transport_redelivered=self.transport_redelivered,
        )
        key = envelope["checkpoint_config"]["key"]
        self.worker_snapshots.append(self.store.load_checkpoint_v2(key))
        return _ImmediateResult(result)


def test_distributed_v2_fixture_round_trips_with_discriminator() -> None:
    canonical = _fixture()["canonical_envelope"]

    envelope = DistributedRunEnvelope.from_dict(canonical)

    assert envelope.schema_version == DISTRIBUTED_RUN_SCHEMA_VERSION_V2
    assert envelope.to_dict() == canonical


def test_distributed_v2_static_invalid_cases_fail_before_worker_resolution() -> None:
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


@pytest.mark.parametrize("transport_redelivered", [False, True])
def test_celery_v2_returns_candidate_then_runner_owns_terminal_order(
    tmp_path: Path,
    transport_redelivered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStateStore()
    renewal_arguments: list[set[str]] = []
    original_renew = store.renew_checkpoint_claim_v2

    def recording_renew(checkpoint_key: str, **kwargs: Any) -> bool:
        renewal_arguments.append(set(kwargs))
        return original_renew(checkpoint_key, **kwargs)

    monkeypatch.setattr(store, "renew_checkpoint_claim_v2", recording_renew)
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
        state_store=store,
        runtime_recipe=recipe,
        dispatch_timeout_seconds=5,
        lease_duration_ms=10_000,
    )
    guardrail_observations: list[tuple[bool, bool]] = []

    def output_guardrail(_context: Any, value: Any) -> GuardrailResult:
        checkpoint = store.load_checkpoint_v2("distributed-v2")
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
    provider = _provider(lambda: ScriptedLLM(steps=[]))
    provider_with_defaults: Any = provider
    provider_with_defaults.default_settings = lambda _resolved: ModelSettings(temperature=0.25)
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
    assert app.envelopes[0]["schema_version"] == DISTRIBUTED_RUN_SCHEMA_VERSION_V2
    worker = app.worker_snapshots[0]
    assert worker is not None
    assert worker.terminal_result is None
    assert worker.claim_token is not None
    assert worker.model_call_journal[0].state.value == "succeeded"
    assert worker.resume_attempt == (2 if transport_redelivered else 1)
    assert len(renewal_arguments) >= 2
    assert all(arguments == {"claim_token", "lease_expires_at_ms", "now_ms"} for arguments in renewal_arguments)

    terminal = store.load_checkpoint_v2("distributed-v2")
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
        ("missing_capability", "unknown distributed capability hook hook.missing@1"),
    ],
)
def test_celery_v2_rejects_identity_definition_config_and_capability_before_claim(
    tmp_path: Path,
    case_name: str,
    expected_error: str,
) -> None:
    store = InMemoryStateStore()
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
        else:
            envelope["recipe"]["capabilities"]["hook_refs"] = [{"id": "hook.missing", "version": "1"}]

    app = _ImmediateApp(
        registry=registry,
        store=store,
        mutate_envelope=mutate,
    )
    backend = CeleryBackend(
        celery_app=app,
        state_store=store,
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

    if case_name == "missing_capability":
        with pytest.raises(DistributedCapabilityError, match=expected_error):
            Runner.run_sync(agent, "validate first", run_config=config)
    else:
        with pytest.raises(Exception) as caught:
            Runner.run_sync(agent, "validate first", run_config=config)
        assert getattr(caught.value, "code", None) == expected_error

    checkpoint = store.load_checkpoint_v2(f"distributed-validation-{case_name}")
    assert checkpoint is not None
    assert checkpoint.claim_token is None
    assert checkpoint.model_call_journal == []
    assert checkpoint.tool_journal == []
    assert checkpoint.terminal_result is None
    assert model_calls == 0


def test_celery_v2_scheduler_retry_redispatches_as_recovery(tmp_path: Path) -> None:
    store = InMemoryStateStore()
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
        fail_first_dispatch=True,
    )
    backend = CeleryBackend(
        celery_app=app,
        state_store=store,
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
    terminal = store.load_checkpoint_v2("distributed-retry")
    assert terminal is not None
    assert terminal.resume_attempt == 2


def test_registered_celery_task_forwards_transport_redelivery_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_single_cycle(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"finished": False}

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

    assert result == {"finished": False}
    assert app.options["bind"] is True
    assert captured == {
        "envelope_dict": {"schema_version": "test"},
        "capability_registry": registry,
        "transport_redelivered": True,
        "transport_retry_count": 3,
    }
