"""The SDK produces and dispatches one bound level-3 bash command."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_session_records
from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.kernel._driver_planning import InvalidW3Command, InvalidW3CommandProvider
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.runtime.hooks import BeforeToolCallEvent
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.environments import (
    EnvironmentCapabilities,
    IsolationMode,
    LookupStatus,
    NetworkMode,
    ProviderLookupResult,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.runtime import (
    TrustedEnvironmentBinding,
    static_session_requirements,
    static_worker_supports,
)
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools import ToolRegistry, ToolSpec, build_default_registry
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .conftest import open_store

MASTER_KEY = b"w" * 32
SESSION_ID = "w3-command"
TENANT_ID = "default"
COMMAND_ARGS = {
    "command": "printf 'hello\\n'",
    "exec_dir": "work",
    "stdin": "input",
    "timeout_seconds": 7,
    "yield_time_ms": 0,
}


def _not_called(context, arguments):
    del context, arguments
    raise AssertionError("level-3 bash reached the local handler")


def _command_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_schema("bash", build_default_registry().get_schema("bash"))
    registry.register(ToolSpec(name="bash", handler=_not_called))
    registry._add_planner_extra_tool_name("bash")
    return registry


def _bash_tool():
    return _command_registry().get_executor("bash")


class HostCommandProbe:
    def __init__(self) -> None:
        self.submitted_plans: list[Any] = []
        self.submitted_calls: list[ToolCall] = []
        self.ledger_at_submit: list[str] = []
        self.ledger_reader = None

    def __deepcopy__(self, memo):
        del memo
        return self


class HostCommandProvider:
    def __init__(self, outcome: Any | None = None) -> None:
        self.outcome = outcome
        self.probe = HostCommandProbe()

    def submit(self, plan, *, context):
        from vv_agent.session.providers import Definitive

        self.probe.submitted_plans.append(plan)
        call = ToolCall.from_dict(plan.payload["request"])
        self.probe.submitted_calls.append(call)
        if callable(self.probe.ledger_reader):
            self.probe.ledger_at_submit = [stored.record.kind for stored in self.probe.ledger_reader()]
        if self.outcome is not None:
            return self.outcome
        return Definitive(ToolExecutionResult(tool_call_id=call.id, content="provider-ok").to_dict(), ("test-host",))

    def query(self, handle):
        from vv_agent.session.providers import Unknown

        return Unknown("test host has no query receipt")

    def cancel(self, handle):
        from vv_agent.session.providers import Unknown

        return Unknown("test host has no cancel receipt")

    def authenticate(self, item, plan):
        return False

    def lookup(self, execution_id, binding):
        from vv_agent.session.providers import Unknown

        return ProviderLookupResult(
            binding,
            LookupStatus.UNKNOWN,
            Unknown("test host lookup is intentionally outside this slice"),
            ("test-host-lookup-not-wired",),
            feature_level=3,
        )


class IncompatibleCommandProvider(HostCommandProvider):
    lookup = None


class SwitchCallHook:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def before_tool_call(self, event: BeforeToolCallEvent) -> ToolCall:
        if self.mode == "id":
            return ToolCall("other-call", event.call.name, dict(event.call.arguments))
        if self.mode == "name":
            return ToolCall(event.call.id, "other", dict(event.call.arguments))
        return ToolCall(
            event.call.id,
            event.call.name,
            dict(event.call.arguments),
            extra_content={"environment_id": "other-environment"},
        )


def _agent(*, hooks=None, metadata: dict[str, Any] | None = None) -> Agent:
    return Agent(
        "w3-command",
        "Run the bound command.",
        model="scripted",
        tools=[_bash_tool()],
        hooks=list(hooks or []),
        metadata=dict(metadata or {}),
    )


def _config(provider: HostCommandProvider, *, register: bool = True) -> RunConfig:
    return RunConfig(
        model_provider=ScriptedModelProvider.new(
            "scripted",
            "scripted",
            [LLMResponse("", [ToolCall("bash-1", "bash", dict(COMMAND_ARGS))]), LLMResponse("done")],
        ),
        workspace="/workspace",
        tool_providers={"bash": provider} if register else {},
        tool_registry_factory=ToolRegistry,
    )


def _capabilities() -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id="env-command-1",
        generation=7,
        isolation=IsolationMode.CONTAINER,
        network=NetworkMode.NONE,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=("command.execute", "filesystem.workspace", "network.deny"),
        network_policy_ref_digest="1" * 64,
        isolation_policy_ref_digest="2" * 64,
    )


def _binding() -> TrustedEnvironmentBinding:
    return TrustedEnvironmentBinding(
        capabilities=_capabilities(),
        descriptor={"provider": "bash", "workspace": "/workspace"},
        operational_policy={
            "network_policy_digest": canonical_json_sha256({"network": "none"}),
            "isolation_policy_digest": canonical_json_sha256({"cpu_millis": 1000}),
        },
    )


def _requirements(agent: Agent, config: RunConfig, level: int = 3) -> SessionRequirements:
    requirements = static_session_requirements(agent, config, resolved=None, handler_version="1")
    return SessionRequirements(level, requirements.handler_version, requirements.capabilities)


def _create_level_three_session(
    database: str | Path,
    key_provider: LocalKeyProvider,
    agent: Agent,
    config: RunConfig,
) -> None:
    with open_store(database) as store:
        store.install_schema()
        store._rows("UPDATE sk_settings SET write_level=3 WHERE singleton=1", ())
        store.configure_reader(level=3)
        sessions = SessionDriver(store=store, content_key_provider=key_provider, worker_level=3)
        try:
            sessions._configure_static_worker(
                level=3,
                capabilities=static_worker_supports(agent, config, resolved=None),
                requirements=_requirements(agent, config),
            )
            sessions.create(SESSION_ID, "/workspace")
        finally:
            sessions.close()


def _run(
    database: str | Path,
    key_provider: LocalKeyProvider,
    provider: HostCommandProvider,
    *,
    hooks=None,
    metadata: dict[str, Any] | None = None,
    runtime_hook=None,
) -> tuple[Any, list[Any], list[Any], list[list[str]], int]:
    agent = _agent(hooks=hooks, metadata=metadata)
    config = _config(provider)
    _create_level_three_session(database, key_provider, agent, config)
    committed_at_start = 0
    ledger_snapshots: list[list[str]] = []
    holder: dict[str, SessionDriver] = {}

    with open_store(database) as store:
        store.configure_reader(level=3)

        def ledger_reader():
            return store.read(SESSION_ID, limit=200).records

        def runtime_factory(drive_store, run_agent, run_config, task, session_id):
            runtime = holder["sessions"].runtime(
                run_agent,
                run_config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            runtime.trusted_environment_binding = _binding()
            runtime.providers["bash"] = provider
            provider.probe.ledger_reader = ledger_reader

            def observe(point, record):
                nonlocal committed_at_start
                if (
                    point == "after_commit"
                    and record is not None
                    and record.kind == "op_started"
                    and "/tool/" in str(record.operation_id)
                ):
                    committed_at_start = len(provider.probe.submitted_plans)
                    ledger_snapshots.append([stored.record.kind for stored in ledger_reader()])
                if runtime_hook is not None:
                    runtime_hook(point, record, ledger_reader)

            runtime.hook = observe
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["sessions"] = sessions
        try:
            handle = sessions.start(SESSION_ID, agent, config, "run")
            result = handle.result(timeout=30)
            records = list(store.read(SESSION_ID, limit=200).records)
            hydration = SessionHydration.from_store(store, tenant_id=TENANT_ID, session_id=SESSION_ID)
            hydrated = hydrate_session_records(tuple(records), hydration)
            return result, records, hydrated, ledger_snapshots, committed_at_start
        finally:
            sessions.close()


@pytest.mark.persistent_store
def test_level_three_bash_is_bound_after_preparation_and_submitted_after_start(
    tmp_path: Path,
    store_backend: str,
    database: str | Path,
) -> None:
    del store_backend
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    provider = HostCommandProvider()

    result, records, hydrated, ledger_snapshots, committed_at_start = _run(database, key_provider, provider)

    assert result.final_output == "done"
    assert committed_at_start == 0
    assert any({"op_planned", "op_prepared", "op_started"}.issubset(snapshot) for snapshot in ledger_snapshots)
    assert len(provider.probe.submitted_plans) == len(provider.probe.submitted_calls) == 1

    w3_records = [
        record.record
        for record in records
        if record.record.kind in {"op_planned", "op_prepared"} and "/tool/" in str(record.record.operation_id)
    ]
    assert [record.feature_level for record in w3_records] == [3, 3]
    assert [record.record.kind for record in records].count("op_prepared") == 1
    plan = next(
        record for record in hydrated if record.record.kind == "op_planned" and "/tool/" in str(record.record.operation_id)
    ).record
    prepared = next(record for record in hydrated if record.record.kind == "op_prepared").record
    submitted = ToolCall.from_dict(provider.probe.submitted_plans[0].payload["request"])
    envelope = provider.probe.submitted_plans[0].payload["request"]["_vv_command_binding"]

    assert submitted.to_dict() == {"id": "bash-1", "name": "bash", "arguments": COMMAND_ARGS}
    assert provider.probe.submitted_calls == [submitted]
    assert envelope["command_spec"] == {
        "command_text": COMMAND_ARGS["command"],
        "argv": [],
        "cwd": "/workspace/work",
        "stdin_base64": base64.b64encode(b"input").decode("ascii"),
        "stdin_mode": "provided",
        "secret_ref_names": [],
        "limits": {"wall_time_ms": 7000, "output_bytes": 65536},
    }
    assert envelope["environment"] == _capabilities().to_dict()
    assert envelope["operational_policy"] == _binding().operational_policy
    assert envelope["execution_identity"]["execution_id"] == execution_id_for(
        TENANT_ID,
        _capabilities().environment_id,
        stable_operation_key(SESSION_ID, plan.operation_id),
    )
    assert plan._payload["execution_identity"] == envelope["execution_identity"]
    assert plan._payload["command_digest"] == envelope["command_digest"]
    assert plan._payload["idempotency_key"] == stable_operation_key(SESSION_ID, plan.operation_id)
    assert plan._payload["tool"] == prepared._payload["tool"]
    assert prepared._payload["command_digest"] == provider.probe.submitted_plans[0].payload["command_digest"]
    assert "session_data_key" not in envelope and "key" not in envelope

    cycle = result.raw_result.cycles[0]
    assert [call.to_dict() for call in cycle.tool_calls] == [{"id": "bash-1", "name": "bash", "arguments": COMMAND_ARGS}]
    assert cycle.tool_results[0].content == "provider-ok"


@pytest.mark.persistent_store
def test_post_start_hook_cannot_mutate_submitted_command(
    tmp_path: Path,
    store_backend: str,
    database: str | Path,
) -> None:
    del store_backend
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "post-start-keys.sqlite3")
    provider = HostCommandProvider()
    observed: dict[str, Any] = {}

    def mutate_after_start(point, record, ledger_reader):
        if point != "before_external_call" or record is None or "/tool/" not in str(record.operation_id):
            return
        ledger = [stored.record.kind for stored in ledger_reader()]
        assert "op_started" in ledger
        request = record._payload["request"]
        request["arguments"]["command"] = "printf 'mutated\\n'"
        request["_vv_command_binding"]["command_spec"]["command_text"] = "printf 'mutated\\n'"
        observed["command"] = request["arguments"]["command"]

    result, records, hydrated, _, _ = _run(
        database,
        key_provider,
        provider,
        runtime_hook=mutate_after_start,
    )

    assert observed["command"] == "printf 'mutated\\n'"
    assert result.final_output == "done"
    assert sum(stored.record.kind == "op_started" and "/tool/" in str(stored.record.operation_id) for stored in records) == 1
    assert len(provider.probe.submitted_calls) == 1
    assert provider.probe.submitted_calls[0].to_dict() == {
        "id": "bash-1",
        "name": "bash",
        "arguments": COMMAND_ARGS,
    }
    submitted = provider.probe.submitted_plans[0]
    prepared = next(
        record.record
        for record in hydrated
        if record.record.kind == "op_prepared" and "/tool/" in str(record.record.operation_id)
    )
    envelope = submitted._payload["request"]["_vv_command_binding"]
    assert submitted._payload["request"] == prepared._payload["request"]
    assert envelope["command_spec"]["command_text"] == COMMAND_ARGS["command"]
    assert envelope["command_digest"] == prepared._payload["command_digest"]
    assert "op_started" in provider.probe.ledger_at_submit


@pytest.mark.parametrize("mode", ["id", "name", "environment"])
def test_level_three_before_tool_cannot_switch_identity_name_or_environment(tmp_path: Path, mode: str) -> None:
    provider = HostCommandProvider()
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{mode}-keys.sqlite3")
    agent = _agent(hooks=[SwitchCallHook(mode)])
    config = _config(provider)
    _create_level_three_session(tmp_path / f"{mode}.sqlite3", key_provider, agent, config)
    holder: dict[str, SessionDriver] = {}
    with open_store(tmp_path / f"{mode}.sqlite3") as store:
        store.configure_reader(level=3)

        def runtime_factory(drive_store, run_agent, run_config, task, session_id):
            runtime = holder["sessions"].runtime(
                run_agent,
                run_config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            runtime.trusted_environment_binding = _binding()
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["sessions"] = sessions
        try:
            with pytest.raises(InvalidW3Command):
                sessions.start(SESSION_ID, agent, config, "run").result(timeout=8)
        finally:
            sessions.close()
    assert provider.probe.submitted_plans == []


def test_level_three_bash_requires_a_registered_compatible_host_provider(tmp_path: Path) -> None:
    def provider_failure(path: Path, key_path: Path, provider: HostCommandProvider, *, register: bool) -> None:
        key_provider = LocalKeyProvider(MASTER_KEY, state_path=key_path)
        agent = _agent()
        config = _config(provider, register=register)
        _create_level_three_session(path, key_provider, agent, config)
        holder: dict[str, SessionDriver] = {}
        with open_store(path) as store:
            store.configure_reader(level=3)

            def runtime_factory(drive_store, run_agent, run_config, task, session_id):
                runtime = holder["sessions"].runtime(
                    run_agent,
                    run_config,
                    task,
                    session_id=session_id,
                    store=drive_store,
                    handler_version="1",
                )
                runtime.trusted_environment_binding = _binding()
                return runtime

            sessions = SessionDriver(
                store=store,
                content_key_provider=key_provider,
                runtime_factory=runtime_factory,
                worker_level=3,
            )
            holder["sessions"] = sessions
            try:
                with pytest.raises(InvalidW3CommandProvider):
                    sessions.start(SESSION_ID, agent, config, "run").result(timeout=8)
            finally:
                sessions.close()

    provider_failure(
        tmp_path / "missing-provider.sqlite3",
        tmp_path / "missing-provider-keys.sqlite3",
        HostCommandProvider(),
        register=False,
    )
    provider_failure(
        tmp_path / "incompatible-provider.sqlite3",
        tmp_path / "incompatible-provider-keys.sqlite3",
        IncompatibleCommandProvider(),
        register=True,
    )


def test_level_three_bash_rejects_hidden_bash_env(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "bash-env-keys.sqlite3")
    agent = _agent(metadata={"bash_env": {"TOKEN": "hidden"}})
    config = _config(provider)
    path = tmp_path / "bash-env.sqlite3"
    _create_level_three_session(path, key_provider, agent, config)
    holder: dict[str, SessionDriver] = {}
    with open_store(path) as store:
        store.configure_reader(level=3)

        def runtime_factory(drive_store, run_agent, run_config, task, session_id):
            runtime = holder["sessions"].runtime(
                run_agent,
                run_config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            runtime.trusted_environment_binding = _binding()
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["sessions"] = sessions
        try:
            with pytest.raises(InvalidW3Command, match="hidden shell context"):
                sessions.start(SESSION_ID, agent, config, "run").result(timeout=8)
        finally:
            sessions.close()
    assert provider.probe.submitted_plans == []
