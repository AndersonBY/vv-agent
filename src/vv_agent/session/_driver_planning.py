"""SDK observations for the single core operation/model planner."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, cast

from vv_agent.kernel import _driver_planning as core
from vv_agent.kernel._records import Record

if TYPE_CHECKING:
    from .kernel import _Driver


def _drain[T](driver: _Driver, steps: Generator[core.PlanningRequest, Any, T]) -> T:
    route: tuple[Any, Any, type] | None = None
    before_llm: Callable[..., Any] | None = None
    response = None
    try:
        while True:
            try:
                request = steps.send(response)
            except StopIteration as done:
                return done.value
            except core._PlanningStopIteration as transport:
                error = transport.original
                original_context = error.__context__
                try:
                    raise error
                finally:
                    error.__context__ = original_context
            response = None
            if isinstance(request, core.ResolveModel):
                from vv_agent.llm.vv_llm_client import VvLlmClient

                client, resolved = (
                    driver.runtime.resolve_model(request.request)
                    if request.purpose == "primary"
                    else driver.runtime.model_route(request.purpose)
                )
                route = client, resolved, VvLlmClient
            elif request is core.PlanningRead.MODEL_BINDING:
                client, resolved, _ = cast(tuple[Any, Any, type], route)
                response = driver.runtime.model_binding(client, resolved)
            elif request is core.PlanningRead.ENDPOINT_SUPPORT:
                client, _, client_type = cast(tuple[Any, Any, type], route)
                response = isinstance(client, client_type)
            elif isinstance(request, core.OrderedEndpoints):
                client, _, _ = cast(tuple[Any, Any, type], route)
                preferred = request.preferred
                if preferred not in {t.endpoint_id for t in client.endpoint_targets}:
                    preferred = None
                response = [t.endpoint_id for t in client.ordered_targets(preferred)]
            elif isinstance(request, core.ProviderAvailable):
                response = request.name in driver.runtime.providers
            elif request is core.PlanningRead.HAS_HOOKS:
                response = driver.runtime.hooks.has_hooks()
            elif request is core.PlanningRead.SHARED_STATE:
                response = driver.shared_state()
            elif request is core.PlanningRead.INTERRUPTION_MESSAGES:
                response = (
                    (True, driver.runtime.config.interruption_messages())
                    if driver.runtime.config.interruption_messages
                    else (False, None)
                )
            elif isinstance(request, core.BeforeCycle):
                response = (
                    (True, driver.runtime.config.before_cycle_messages(request.cycle, list(request.messages), request.shared))
                    if driver.runtime.config.before_cycle_messages
                    else (False, None)
                )
            elif request is core.PlanningRead.PREPARE_BEFORE_LLM:
                before_llm = driver.runtime.hooks.apply_before_llm
                response = driver.runtime.hooks.has_hooks()
            elif isinstance(request, core.BeforeLLM):
                response = cast(Callable[..., Any], before_llm)(
                    task=request.task,
                    cycle_index=request.cycle,
                    messages=request.messages,
                    tool_schemas=request.schemas,
                    shared_state=request.shared,
                )
            elif request is core.PlanningRead.DEFAULT_MODEL_BINDING:
                response = driver.runtime.model_binding(driver.runtime.llm, driver.runtime.resolved)
            elif isinstance(request, core.DurableState):
                response = driver.runtime.durable_state(request.shared)
            else:
                assert isinstance(request, core.ClosePlanning)
                driver.close(request.status, request.reason)
    finally:
        steps.close()


def plan(
    self: _Driver,
    oid: str,
    request: dict[str, Any],
    kind: str,
    *,
    number: int = 1,
    dependencies: list[str] | None = None,
    capability: dict[str, Any] | None = None,
    purpose: str = "primary",
) -> Record:
    return _drain(
        self,
        core.plan(self, oid, request, kind, number=number, dependencies=dependencies, capability=capability, purpose=purpose),
    )


def model_plan(self: _Driver) -> Record | None:
    return _drain(self, core.model_plan(self, self.transcript()))


def effective_model(self: _Driver) -> dict[str, Any]:
    return _drain(self, core.effective_model(self))
