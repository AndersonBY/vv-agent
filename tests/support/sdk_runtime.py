"""Small real-SDK assembly for producer tests over the shared SimWorld facts."""

from __future__ import annotations

from pathlib import Path

import pytest
from sim.assembly import ScheduledLeaseKeeper
from sim.world import InvariantViolation, SimWorld

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._provider_outcomes import Unknown
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.session import kernel
from vv_agent.session.runtime import Runtime, RuntimeNotReady
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
from vv_agent.tools.registry import ToolRegistry


class _SynchronousInvoker:
    def __init__(self, world):
        self.world = world

    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        del timeout, token, cancellation_grace, clock
        self.world.checkpoint("invoke")
        try:
            return callback()
        except InvariantViolation:
            raise
        except Exception as exc:
            return Unknown(f"{type(exc).__name__}: scripted outcome unavailable")


def sdk_runtime(world: SDKWorld, sid: str, participant: str, *, provider=None, children=None) -> Runtime:
    """Construct the real SDK runtime from SimWorld-owned scripted facts."""
    model_fact = world.scripted_model_callback(sid)

    class Model(ScriptedLLM):
        def complete(self, request):
            return model_fact(request)

    def forbidden(_ctx, _args):
        raise AssertionError("external effects must go through the scripted provider")

    metadata = ToolMetadata(
        idempotency=ToolIdempotency.SUPPORTED if world.scenario == "unknown_idempotent" else ToolIdempotency.UNSUPPORTED
    )
    tool = FunctionTool(
        name="effect",
        description="scripted effect",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke=forbidden,
        needs_approval=world.scenario == "approval_reply" or (world.scenario == "cancel_descendants" and sid == "grandchild"),
        tool_metadata=metadata,
    )
    keeper = world.keepers.setdefault(participant, ScheduledLeaseKeeper(world, participant))
    return Runtime(
        agent=Agent(name="sim", instructions="Be precise.", tools=[tool]),
        config=RunConfig(workspace=Path("/tmp"), session_memory_enabled=False, tool_registry_factory=ToolRegistry),
        resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
        llm=Model(),
        heartbeat_store=lambda: world.heartbeat_connection(participant),
        providers={"effect": provider} if provider is not None and not children else {},
        children=children or {},
        clock=world.clock,
        lease_keeper=keeper,
        invoker=_SynchronousInvoker(world),
        hook=world.scripted_hook(participant),
        lease_retry_sleep=world.clock.sleep,
        lease_retry_jitter=lambda first, second: first,
        wake=world.queue.append,
    )


class SDKWorld(SimWorld):
    """Use the shared scheduler/SQL/invariants with an explicitly real SDK runtime."""

    def runtime(self, sid, participant, *, provider=None, children=None):
        return sdk_runtime(self, sid, participant, provider=provider, children=children)

    def _drive_runtime(self, store, sid, *, runtime, factory):
        if factory is not None:
            return kernel._drive(store, sid, runtime=factory, clock=self.clock, lease_keeper=runtime.lease_keeper)
        return kernel.drive(store, sid, runtime=runtime)

    def not_ready(self, sid):
        del sid
        raise RuntimeNotReady(1)


def sdk_scope(world: SDKWorld, sid="s", *, participant="manual"):
    store = world.connection(participant)
    runtime = world.runtime(sid, participant)
    lease = store.acquire(sid, owner=participant, ttl_ms=runtime.ttl_ms)
    assert lease is not None
    return kernel._Scope(lease, runtime, now_ms=store._now())


def sdk_driver(world: SDKWorld, sid="s", *, participant="manual"):
    scope = sdk_scope(world, sid, participant=participant)
    worker = kernel._Driver(world.connection(participant), sid, scope.runtime, scope)
    worker.refresh()
    return worker


def until(worker, predicate):
    for _ in range(40):
        if predicate():
            return
        assert worker.step(), "driver became idle before the scheduled cut"
    pytest.fail("bounded driver schedule did not reach its cut")
