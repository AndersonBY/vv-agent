"""Trusted environment bindings stay on the level-3 parent runtime."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from vv_agent import Agent, RunConfig
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.base import LLMClient
from vv_agent.model import ScriptedModelProvider, resolve_run_model
from vv_agent.run_config import ToolPolicy
from vv_agent.session.environments import EnvironmentCapabilities, IsolationMode, NetworkMode
from vv_agent.session.runtime import Runtime, TrustedEnvironmentBinding


class _ToolProvider:
    def submit(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"unexpected submit: {args!r} {kwargs!r}")

    def query(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"unexpected query: {args!r} {kwargs!r}")

    def cancel(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"unexpected cancel: {args!r} {kwargs!r}")

    def authenticate(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"unexpected authenticate: {args!r} {kwargs!r}")


def _binding() -> TrustedEnvironmentBinding:
    return TrustedEnvironmentBinding(
        capabilities=EnvironmentCapabilities.create(
            environment_id="environment/ordinary-child-parent",
            generation=3,
            isolation=IsolationMode.CONTAINER,
            network=NetworkMode.NONE,
            enforces_network_policy=True,
            enforces_resource_limits=True,
            persistent=False,
            environment_capabilities=("command.execute",),
            network_policy_ref_digest="1" * 64,
            isolation_policy_ref_digest="2" * 64,
        ),
        descriptor={"provider": "w3-host", "workspace": "/workspace/parent"},
        operational_policy={
            "network_policy_digest": "3" * 64,
            "isolation_policy_digest": "4" * 64,
        },
    )


def test_for_agent_drops_w3_binding_from_ordinary_child_runtime(tmp_path):
    model_provider = ScriptedModelProvider.new("scripted", "parent", [])
    parent_agent = Agent("parent", "Parent.", model="parent")
    parent_config = RunConfig(workspace=tmp_path / "parent", model_provider=model_provider)
    llm, resolved = resolve_run_model(agent=parent_agent, run_config=parent_config)
    parent_provider = _ToolProvider()
    binding = _binding()

    def resolve_selection(_selection: dict[str, Any]) -> tuple[LLMClient, ResolvedModelConfig]:
        return llm, resolved

    parent_runtime = Runtime(
        parent_agent,
        parent_config,
        resolved,
        llm,
        lambda: nullcontext(None),
        providers={"host/provider": parent_provider},
        host_bindings={"host_state": {"tenant": "default"}},
        trusted_environment_binding=binding,
        model_resolver=resolve_selection,
        handler_version="test-handler",
        ttl_ms=120_000,
        heartbeat_seconds=0.5,
        poll_ms=250,
        cancellation_grace=0.5,
        lease_retry_attempts=7,
        lease_retry_base_seconds=0.02,
        lease_retry_cap_seconds=2.0,
        model_timeout=123,
        tool_timeout=234,
    )
    child_config = RunConfig(
        workspace=tmp_path / "child",
        model_provider=model_provider,
        tool_providers={"child/provider": _ToolProvider()},
        tool_policy=ToolPolicy(disallowed_tools=["forbidden"]),
        shared_state={"seed": "child"},
    )
    child_agent = Agent("worker", "Worker.", model="child")

    child_runtime = parent_runtime.for_agent(child_agent, child_config)

    assert parent_runtime.trusted_environment_binding is binding
    assert parent_runtime.turn_environment_binding() == binding
    assert child_runtime.trusted_environment_binding is None
    assert child_runtime.turn_environment_binding() is None

    assert child_runtime.agent is child_agent
    assert child_runtime.resolved.model_id == "child"
    assert child_runtime.llm is llm
    assert child_runtime.model_resolver is parent_runtime.model_resolver
    assert child_runtime.providers["host/provider"] is parent_provider
    assert child_runtime.providers["child/provider"] is child_config.tool_providers["child/provider"]
    assert child_runtime.host_bindings is parent_runtime.host_bindings
    assert child_runtime.children is parent_runtime.children
    assert child_runtime.clock is parent_runtime.clock
    assert child_runtime.lease_keeper is parent_runtime.lease_keeper
    assert child_runtime.invoker is parent_runtime.invoker
    assert child_runtime.lease_retry_jitter is parent_runtime.lease_retry_jitter
    assert child_runtime.lease_retry_sleep is parent_runtime.lease_retry_sleep
    assert child_runtime.ttl_ms == parent_runtime.ttl_ms
    assert child_runtime.heartbeat_seconds == parent_runtime.heartbeat_seconds
    assert child_runtime.poll_ms == parent_runtime.poll_ms
    assert child_runtime.cancellation_grace == parent_runtime.cancellation_grace
    assert child_runtime.lease_retry_attempts == parent_runtime.lease_retry_attempts
    assert child_runtime.lease_retry_base_seconds == parent_runtime.lease_retry_base_seconds
    assert child_runtime.lease_retry_cap_seconds == parent_runtime.lease_retry_cap_seconds
    assert child_runtime.model_timeout == parent_runtime.model_timeout
    assert child_runtime.tool_timeout == parent_runtime.tool_timeout
    assert child_runtime.handler_version == parent_runtime.handler_version
    assert child_runtime.config.workspace == child_config.workspace
    assert child_runtime.config.tool_policy == child_config.tool_policy
    assert child_runtime.config.shared_state == child_config.shared_state
    assert child_runtime.memory_manager == replace(parent_runtime.memory_manager)
