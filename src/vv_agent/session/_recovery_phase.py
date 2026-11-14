"""Concrete SDK leaves for the shared recovery/close generator composition."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast

from vv_agent.approval import ApprovalDecision, ApprovalRequest
from vv_agent.kernel import _after_cycle as cycle
from vv_agent.kernel import _compaction_phase as compaction_core
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel import _driver_outcomes as outcomes
from vv_agent.kernel import _driver_planning as planning
from vv_agent.kernel import _driver_recovery as core
from vv_agent.kernel import _output_phase as output_core
from vv_agent.kernel._budget import BudgetObservation
from vv_agent.kernel._child_store import cancel_children
from vv_agent.kernel._cycles import project_cycles, turn_result_prefix
from vv_agent.kernel._driver_requests import CommitRecords
from vv_agent.kernel._microcompaction_apply import ArchiveToolMessage, ToolMarkerInput
from vv_agent.kernel._token_count import TokenCount, _StopIterationTransport
from vv_agent.kernel._usage import project_model_usage
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.lifecycle import AfterCycleHookManager
from vv_agent.types import AgentResult, AgentStatus, CompletionReason, Message

from . import compaction, memory
from . import output as output_host
from ._driver_scope import _invoke
from .delegation import result_for_children
from .projection import project_records
from .tool_results import prepare_tool_resources, project_tool_content

if TYPE_CHECKING:
    from .kernel import _Driver


class _Host:
    def __init__(self, driver, dispatch_host=None, *, manager=None, after_cycle=None, output_agent=None):
        self.driver = driver
        self.dispatch_host = dispatch_host
        self.manager = manager
        self.memory = None
        self.route = None
        self.before_llm = None
        self.after_cycle = after_cycle
        self.output_agent = output_agent
        self.provider = None
        self.approval_provider = None
        self.broker = None
        self.approval_handle = None
        self.approval_request = None

    def dispatch(self):
        if self.dispatch_host is None:
            from ._driver_dispatch import _DispatchHost

            self.dispatch_host = _DispatchHost(self.driver)
        return self.dispatch_host

    def steps[T](self, steps: Generator[object, Any, T]) -> Generator[object, Any, T]:
        response = None
        error = None
        try:
            while True:
                try:
                    request = steps.throw(error) if error is not None else steps.send(response)
                except StopIteration as done:
                    return done.value
                except (core._RecoveryStopIteration, _StopIterationTransport) as transport:
                    raise outcomes._OutcomeStopIteration(transport.original) from transport.original
                response, error = None, None
                try:
                    if isinstance(
                        request,
                        (
                            outcomes.OutcomeRead,
                            outcomes.PrepareToolResources,
                            outcomes.LookupYieldCallback,
                            outcomes.InvokeYieldCallback,
                            BudgetObservation,
                        ),
                    ):
                        response = yield request
                    else:
                        response = self.handle(request)
                except Exception as exc:
                    error = _StopIterationTransport(exc) if isinstance(exc, StopIteration) else exc
        finally:
            if self.dispatch_host is not None and self.dispatch_host.preflight is not None:
                self.dispatch_host.preflight.close()
            steps.close()

    def handle(self, request):
        driver = self.driver
        if request is core.Read.REFRESH:
            return driver.refresh()
        if request in (core.Read.CANCELLED, inputs.InputSetting.CANCELLED):
            return driver.scope.token.cancelled
        if request in (core.Read.RESET_TOKEN, inputs.InputSetting.RESET_TOKEN):
            driver.scope.token = CancellationToken()
            return None
        if request in (core.Read.HANDLER_VERSION, inputs.InputSetting.HANDLER_VERSION):
            return driver.runtime.handler_version
        if request is core.Read.AFTER_CYCLE_CONFIGURED:
            return bool(driver.runtime.config.after_cycle_hooks)
        if request is core.Read.START_AFTER_CYCLE:
            self.after_cycle = AfterCycleHookManager(driver.runtime.config.after_cycle_hooks)
            return self.after_cycle.has_hooks()
        if request is core.Read.START_OUTPUT:
            self.output_agent = driver.runtime.agent
            return None
        if request is core.Read.MEMORY_HOOKS:
            return driver.runtime.hooks.has_hooks()
        if request is core.Read.MAX_PTL_RETRIES:
            return compaction.MAX_PTL_RETRIES
        if request is core.Read.NEXT_DRIVE_DELAY:
            with driver.scope.committing() as lease:
                return driver.store.next_drive_delay_ms(driver.sid, lease=lease)
        if isinstance(request, (core.Definition, inputs.CurrentDefinition)):
            task = request.start if isinstance(request, core.Definition) else request.task
            return inputs.DefinitionFacts(
                driver.runtime._definitions, driver.runtime._definition_bindings(task), driver.runtime.registry
            )
        if isinstance(request, core.BindHostState):
            return driver.runtime.bind_state({}, request.task)
        if isinstance(request, core.BeforeMemory):
            return driver.runtime.hooks.apply_before_memory_compact(
                task=request.task, cycle_index=request.cycle_index, messages=request.messages, shared_state=request.shared_state
            )
        if isinstance(request, core.Hook):
            return driver.runtime.hook(request.point, request.record)
        if isinstance(request, core.SelectProvider):
            self.provider = driver.provider(request.plan)
            return None
        if isinstance(request, core.ProviderCall):
            provider, handle = self.provider, request.handle
            assert provider is not None
            callback = (lambda: provider.query(handle)) if request.method == "query" else (lambda: provider.cancel(handle))
            return _invoke(driver.scope, callback, request.timeout)
        if isinstance(request, core.CommitStopping):
            return driver.commit(
                request.records,
                guarded=True,
                prepare=(lambda tx: cancel_children(driver.store, tx, driver.state, driver.sid, request.turn_id))
                if request.stopping
                else None,
            )
        if isinstance(request, CommitRecords):
            return driver.commit(request.records, guarded=True)
        if isinstance(request, inputs.CommitInput):
            return driver.commit(request.records, (request.input_id,), guarded=True)
        if isinstance(request, inputs.CompileTurn):
            return driver.runtime.compile(request.content, request.turn_id, seed=request.seed, control=request.control)
        if isinstance(request, inputs.ResolveModelSelection):
            client, resolved = driver.runtime.resolve_model(request.selection)
            return driver.runtime.model_binding(client, resolved)
        if isinstance(request, inputs.AuthenticateProvider):
            return driver.provider(request.plan).authenticate(request.item, request.plan)
        if isinstance(request, inputs.ProjectProviderResult):
            return project_tool_content(
                request.plan,
                outcomes.compaction_intent(driver, request.plan, request.result),
                driver.runtime.workspace_backend,
                selected_artifact=request.selected_artifact,
            )
        if isinstance(request, inputs.PrepareToolContext):
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
            self.dispatch().context = context
            return None
        if isinstance(request, inputs.ChildResult):
            return result_for_children(driver.store, request.plan, request.handles, driver.runtime).to_dict()
        if request is inputs.InputSetting.POLL_NOW:
            return driver.scope.poll(driver.store).db_now_ms
        if request is inputs.InputSetting.POLL_MS:
            return driver.runtime.poll_ms
        if request is inputs.InputSetting.BUDGET_LIMITS:
            return driver.runtime.config.budget_limits
        if isinstance(request, planning.ResolveModel):
            from vv_agent.llm.vv_llm_client import VvLlmClient

            client, resolved = (
                driver.runtime.resolve_model(request.request)
                if request.purpose == "primary"
                else driver.runtime.model_route(request.purpose)
            )
            self.route = client, resolved, VvLlmClient
            return None
        if request is planning.PlanningRead.MODEL_BINDING:
            assert self.route is not None
            client, resolved, _ = self.route
            return driver.runtime.model_binding(client, resolved)
        if request is planning.PlanningRead.ENDPOINT_SUPPORT:
            assert self.route is not None
            client, _, client_type = self.route
            return isinstance(client, client_type)
        if isinstance(request, planning.OrderedEndpoints):
            assert self.route is not None
            client, _, _ = self.route
            preferred = request.preferred
            if preferred not in {t.endpoint_id for t in client.endpoint_targets}:
                preferred = None
            return [t.endpoint_id for t in client.ordered_targets(preferred)]
        if isinstance(request, planning.ProviderAvailable):
            return request.name in driver.runtime.providers
        if request is planning.PlanningRead.HAS_HOOKS:
            return driver.runtime.hooks.has_hooks()
        if request is planning.PlanningRead.SHARED_STATE or isinstance(request, cycle.ReadSharedState):
            return driver.shared_state()
        if request is planning.PlanningRead.INTERRUPTION_MESSAGES:
            return (
                (True, driver.runtime.config.interruption_messages())
                if driver.runtime.config.interruption_messages
                else (False, None)
            )
        if isinstance(request, planning.BeforeCycle):
            return (
                (True, driver.runtime.config.before_cycle_messages(request.cycle, list(request.messages), request.shared))
                if driver.runtime.config.before_cycle_messages
                else (False, None)
            )
        if request is planning.PlanningRead.PREPARE_BEFORE_LLM:
            self.before_llm = driver.runtime.hooks.apply_before_llm
            return driver.runtime.hooks.has_hooks()
        if isinstance(request, planning.BeforeLLM):
            assert self.before_llm is not None
            return self.before_llm(
                task=request.task,
                cycle_index=request.cycle,
                messages=request.messages,
                tool_schemas=request.schemas,
                shared_state=request.shared,
            )
        if request is planning.PlanningRead.DEFAULT_MODEL_BINDING:
            return driver.runtime.model_binding(driver.runtime.llm, driver.runtime.resolved)
        if isinstance(request, planning.DurableState):
            return driver.runtime.durable_state(request.shared)
        if isinstance(request, cycle.ReadTranscript) or request is compaction_core.CompactionRequest.TRANSCRIPT:
            return driver.transcript()
        if isinstance(request, cycle.ReadCycleProjection):
            state, records, _ = driver.store.read_state(driver.sid)
            _, records = turn_result_prefix(records, request.turn_id)
            usage, _ = project_model_usage(state, request.turn_id)
            cycles = project_cycles(state, records, request.turn_id, usage)
            project_records(records)
            return cycles, usage
        if isinstance(request, cycle.ApplyAfterCycleHooks):
            assert self.after_cycle is not None
            return self.after_cycle.apply(request.snapshot)
        if request is compaction_core.CompactionRequest.INBOX:
            return driver.store.peek_inbox(driver.sid, through_input_seq=driver.watermark)
        if request is compaction_core.CompactionRequest.MEMORY:
            self.manager = compaction.manager_for(driver)
            return compaction._settings(self.manager)
        if request is compaction_core.CompactionRequest.RESTORE_SESSION_MEMORY:
            self.memory = memory.restore_memory(driver)
            return None
        if request is compaction_core.CompactionRequest.SAVE_MEMORY_PROJECTION:
            return memory.save_projection(driver)
        if isinstance(request, compaction_core.ModelRoute):
            _, resolved = driver.runtime.model_route(request.purpose)
            return resolved.model_id
        if isinstance(request, compaction_core.MicroArtifacts):
            return driver.runtime.hook("after_microcompact_artifacts", request.record)
        if isinstance(request, compaction_core.SessionMemoryCompacted):
            return memory.session_memory_compacted(driver, request.current_tokens)
        if isinstance(request, compaction_core.SelectExtractionMessages):
            assert self.memory is not None
            return memory.select_extraction_messages(self.memory, request.source, request.current_tokens)
        if isinstance(request, compaction_core.ExtractionPrompt):
            assert self.memory is not None
            return self.memory.extraction_prompt(request.messages)
        if isinstance(request, compaction_core.ApplyExtractionReceipt):
            assert self.memory is not None
            return memory.apply_extraction_receipt(
                self.memory, request.source, request.current_tokens, request.cycle, request.content
            )
        if isinstance(request, compaction_core.CountContext):
            assert self.manager is not None
            return compaction.tokens(self.manager, request.messages)
        if isinstance(request, TokenCount) and (self.dispatch_host is None or self.dispatch_host.manager is None):
            assert self.manager is not None
            return (
                self.manager._estimate_message_tokens(request.messages)
                if isinstance(request.messages, Message)
                else self.manager._calculate_message_length(request.messages)
            )
        if isinstance(request, ArchiveToolMessage):
            assert self.manager is not None
            return self.manager._archive_tool_message(request.message, request.candidate)
        if isinstance(request, ToolMarkerInput):
            assert self.manager is not None
            return (
                self.manager._compaction_excerpt(request.message),
                self.manager.tool_result_excerpt_head,
                self.manager.tool_result_excerpt_tail,
            )
        if isinstance(request, compaction_core.BeforeCompaction):
            return memory.before_compaction_event(
                driver,
                self.manager,
                request.source,
                key=request.key,
                cycle=request.cycle,
                trigger=request.trigger,
                prune=request.prune,
            )
        if isinstance(request, compaction_core.AfterCompaction):
            return memory.after_compaction_event(
                driver,
                self.manager,
                request.source,
                key=request.key,
                start=request.start,
                mode=request.mode,
                changed=request.changed,
                archived_count=request.archived_count,
                reclaimed_tokens=request.reclaimed_tokens,
                artifact_failure_count=request.artifact_failure_count,
            )
        if isinstance(request, core.BeginApproval):
            self.approval_provider, self.broker = driver.runtime.config.approval_provider, driver.runtime.approval_broker
            self.approval_handle = request.handle
            self.approval_request = None
            return self.approval_provider is not None, self.broker is not None
        if request is core.ApprovalRead.DRAIN:
            assert self.broker is not None and self.approval_handle is not None
            with self.broker._condition:
                request_id = self.approval_handle["request_id"]
                return self.broker.wait(request_id, timeout=0) if request_id in self.broker._decisions else None
        if request is core.ApprovalRead.DISCARD:
            assert self.broker is not None and self.approval_handle is not None
            return self.broker.discard(self.approval_handle["request_id"])
        if request is core.ApprovalRead.PENDING:
            assert self.broker is not None and self.approval_handle is not None
            return self.broker.pending_request(self.approval_handle["request_id"]) is not None
        if request is core.ApprovalRead.HAS_ANSWER:
            assert self.approval_handle is not None
            request_id = self.approval_handle["request_id"]
            return any(
                item.item.kind == "approval_answer" and item.item.payload["request_id"] == request_id
                for item in driver.store.peek_inbox(driver.sid)
            )
        if request is core.ApprovalRead.TIMEOUT:
            return ApprovalDecision.timeout("Approval request timed out.")
        if request is core.ApprovalRead.ALLOW:
            return ApprovalDecision.allow()
        if isinstance(request, core.ApprovalRequest):
            plan, handle, deadline = request.plan, request.handle, request.deadline
            executor = driver.runtime.functions.orchestrator._resolve_executor(plan._payload["request"]["name"])
            assert executor is not None
            self.approval_request = ApprovalRequest(
                request_id=handle["request_id"],
                tool_name=plan._payload["request"]["name"],
                tool_call_id=plan._payload["request"]["id"],
                arguments=plan.payload["request"]["arguments"],
                run_id=plan.turn_id or "",
                trace_id=plan.turn_id or "",
                agent_name=driver.runtime.agent.name,
                cycle_index=int(plan._payload["dependencies"][0].rsplit("/", 1)[1]),
                metadata={
                    "tool_metadata": dict(executor.metadata),
                    "session_id": driver.sid,
                    "timeout_seconds": max(0, (deadline - driver.scope.poll(driver.store).db_now_ms) / 1000)
                    if deadline is not None
                    else None,
                },
            )
            return None
        if request is core.ApprovalRead.SHOULD_REQUEST:
            assert self.approval_provider is not None and self.approval_request is not None
            return self.approval_provider.should_request(self.approval_request)
        if request is core.ApprovalRead.REGISTER:
            assert self.broker is not None and self.approval_request is not None
            return self.broker.register(self.approval_request)
        if request is core.ApprovalRead.DECIDE:
            assert self.approval_provider is not None and self.approval_request is not None
            return self.approval_provider.decide(self.approval_request)
        if isinstance(request, core.ResolveApproval):
            assert self.broker is not None and self.approval_request is not None
            return self.broker.resolve(self.approval_request.request_id, request.decision)
        if isinstance(request, core.PushApprovalAnswer):
            with driver.store.atomic() as tx:
                return tx.push(driver.sid, request.item)
        if isinstance(request, output_core.OutputSetting):
            agent = self.output_agent
            assert agent is not None
            if request is output_core.OutputSetting.HAS_GUARDRAILS:
                return bool(agent.output_guardrails)
            if request is output_core.OutputSetting.HAS_OUTPUT_TYPE:
                return agent.output_type is not None
            if request is output_core.OutputSetting.VALIDATION_ENABLED:
                return agent.output_validation_enabled
            if request is output_core.OutputSetting.HAS_VALIDATOR:
                return agent.output_validator is not None
            if request is output_core.OutputSetting.HAS_REPAIR:
                return agent.output_repair is not None
            if request is output_core.OutputSetting.MAX_REPAIRS:
                return agent.output_validation_max_repairs
            return agent.output_repair_model
        if isinstance(request, output_core.PostprocessOutput):
            assert self.output_agent is not None
            raw = AgentResult(
                status=AgentStatus.COMPLETED,
                messages=driver.transcript(),
                cycles=[],
                final_answer=request.value,
                completion_reason=CompletionReason.NO_TOOL_FINISH,
            )
            checked, error = output_host.postprocess_output(
                agent=self.output_agent,
                run_context=driver.runtime.run_context(request.turn_id),
                raw_result=raw,
                final_output=request.value,
                cancellation_token=driver.scope.token,
            )
            return output_core.PostprocessedOutput(checked, error, raw.status == AgentStatus.FAILED)
        if isinstance(request, output_core.ValidateOutput):
            agent = self.output_agent
            assert agent is not None
            return output_host.run_output_validator(
                agent=agent,
                validator=agent.output_validator,
                validation_context=output_host.OutputValidationContext(request.turn_id, agent.name, agent.output_type),
                candidate=request.value,
                output_coercion_error=request.coercion_error,
                coerce=request.coerce,
            )
        if isinstance(request, output_core.ValidationError):
            return output_host.output_validation_error(request.validation)
        if isinstance(request, output_core.SerializeOutput):
            return output_host.serializable_output(request.value)
        return self.dispatch()._handle(request)


def run[T](
    driver: _Driver,
    steps: Generator[object, Any, T],
    *,
    dispatch_host=None,
    manager=None,
    after_cycle=None,
    output_agent=None,
) -> T:
    host = _Host(
        driver,
        dispatch_host,
        manager=manager,
        after_cycle=after_cycle,
        output_agent=output_agent,
    )
    composed = cast(Generator[outcomes.OutcomeRequest, Any, T], host.steps(core._compose(driver, steps)))
    return driver._run_outcome_steps(composed, prepare_resources=prepare_tool_resources)


def step(driver: _Driver) -> bool:
    return run(driver, core.step(driver, one_turn=driver.one_turn))


def close(driver: _Driver, status: str, reason: str | None, result: Any = None, *, transferred: bool = False) -> None:
    return run(driver, core.close(driver, status, reason, result, transferred=transferred))
