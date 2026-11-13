"""SDK operations requested by the shared inbox and turn-admission algorithms."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._driver_requests import CommitRecords
from vv_agent.kernel._token_count import _StopIterationTransport
from vv_agent.runtime.cancellation import CancellationToken

from .delegation import result_for_children
from .records import InboxItem, Record

if TYPE_CHECKING:
    from .kernel import _Driver


def _drain[T](driver: _Driver, steps: Generator[object, Any, T]) -> T:
    response = None
    error = None
    context = None
    try:
        while True:
            try:
                request = steps.throw(error) if error is not None else steps.send(response)
            except StopIteration as done:
                return done.value
            except _StopIterationTransport as transport:
                error = transport.original
                break
            response, error = None, None
            try:
                if isinstance(request, inputs.CommitInput):
                    driver.commit(request.records, (request.input_id,), guarded=True)
                elif isinstance(request, CommitRecords):
                    driver.commit(request.records, guarded=True)
                elif isinstance(request, inputs.CompileTurn):
                    response = driver.runtime.compile(
                        request.content, request.turn_id, seed=request.seed, control=request.control
                    )
                elif isinstance(request, inputs.CurrentDefinition):
                    response = inputs.DefinitionFacts(
                        driver.runtime._definitions,
                        driver.runtime._definition_bindings(request.task),
                        driver.runtime.registry,
                    )
                elif isinstance(request, inputs.ResolveModelSelection):
                    client, resolved = driver.runtime.resolve_model(request.selection)
                    response = driver.runtime.model_binding(client, resolved)
                elif isinstance(request, inputs.AuthenticateProvider):
                    response = driver.provider(request.plan).authenticate(request.item, request.plan)
                elif isinstance(request, inputs.ProjectProviderResult):
                    from .tool_results import project_tool_content

                    response = project_tool_content(
                        request.plan,
                        driver.compaction_intent(request.plan, request.result),
                        driver.runtime.workspace_backend,
                        selected_artifact=request.selected_artifact,
                    )
                elif isinstance(request, inputs.CompleteResult):
                    response = driver.completed(request.plan, request.outcome)
                elif isinstance(request, inputs.PrepareToolContext):
                    if request.include_store:
                        context = driver.runtime.context(
                            driver.task(),
                            request.plan,
                            driver.scope.token,
                            shared_state=driver.shared_state(),
                            store=driver.store,
                            session_id=driver.sid,
                        )
                    else:
                        context = driver.runtime.context(
                            driver.task(), request.plan, driver.scope.token, shared_state=driver.shared_state()
                        )
                elif isinstance(request, inputs.FinalizeTool):
                    response = driver.finalize_tool(request.plan, context, request.outcome)
                elif isinstance(request, inputs.ChildResult):
                    response = result_for_children(driver.store, request.plan, request.handles, driver.runtime).to_dict()
                elif isinstance(request, inputs.YieldResults):
                    response = driver.yield_results(request.attempts, request.item)
                elif request is inputs.InputSetting.POLL_NOW:
                    response = driver.scope.poll(driver.store).db_now_ms
                elif request is inputs.InputSetting.POLL_MS:
                    response = driver.runtime.poll_ms
                elif request is inputs.InputSetting.CANCELLED:
                    response = driver.scope.token.cancelled
                elif request is inputs.InputSetting.BUDGET_LIMITS:
                    response = driver.runtime.config.budget_limits
                elif request is inputs.InputSetting.HANDLER_VERSION:
                    response = driver.runtime.handler_version
                elif request is inputs.InputSetting.RESET_TOKEN:
                    driver.scope.token = CancellationToken()
                else:
                    raise TypeError(f"unsupported input request: {type(request).__name__}")
            except Exception as exc:
                error = _StopIterationTransport(exc) if isinstance(exc, StopIteration) else exc
    finally:
        steps.close()
    assert error is not None
    raise error


def apply_input(self: _Driver) -> bool:
    return _drain(self, inputs.apply_input(self, one_turn=self.one_turn))


def apply_child(self: _Driver, item: InboxItem) -> tuple[str, list[Record], str, str | None]:
    return _drain(self, inputs.apply_child(self, item))


def start_turn(self: _Driver) -> bool:
    return _drain(self, inputs.start_turn(self))
