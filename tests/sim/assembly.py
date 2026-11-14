"""Core-only host assembly for deterministic SimWorld execution.

The adapter supplies concrete model/tool effects and durable host observations.
All execution, planning, recovery, dispatch, lease, fold, and SQL decisions stay
in ``vv_agent.kernel``.  It intentionally imports no SDK runtime, session facade,
Agent, or ToolRegistry implementation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, cast

from vv_agent._jsonschema import schema_issues
from vv_agent._memory_settings import normalize_memory_settings
from vv_agent._token_counting import count_messages_tokens
from vv_agent.kernel import _after_cycle as cycle
from vv_agent.kernel import _child_store, _driver_dispatch, _driver_recovery
from vv_agent.kernel import _child_summary as child_summary
from vv_agent.kernel import _compaction_phase as compaction
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel import _driver_outcomes as outcomes
from vv_agent.kernel import _driver_planning as planning
from vv_agent.kernel import _output_phase as output
from vv_agent.kernel import _tool_admission as admission
from vv_agent.kernel._budget import BudgetObservation, HostCostObservation
from vv_agent.kernel._budget_values import BudgetUnavailableReason, RunBudgetLimits
from vv_agent.kernel._call_context import approved_tool_call
from vv_agent.kernel._cancellation import Cancellation, CancelledError
from vv_agent.kernel._child_store import ordinary_result_for_children
from vv_agent.kernel._definition import DefinitionBindings, DefinitionCache
from vv_agent.kernel._drive import DriveBinding, drive_core
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_requests import CommitRecords
from vv_agent.kernel._execution_values import AgentTask, LLMResponse
from vv_agent.kernel._lease_scope import HeartbeatStopped, LeaseOwnership, apply_controls, cancel_if_lost
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._model_settings import ModelSettings
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
from vv_agent.kernel._provider_outcomes import Definitive, Unknown
from vv_agent.kernel._records import Record, copy_json
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLStore
from vv_agent.kernel._store import LeaseLost, SessionStore
from vv_agent.kernel._token_count import TokenCount, _StopIterationTransport
from vv_agent.kernel._tool_metadata import ToolIdempotency, ToolMetadata
from vv_agent.kernel._tool_values import ToolExecutionResult, ToolResultStatus


class SimRuntimeNotReady(Exception):
    def __init__(self, retry_after_ms: int):
        self.retry_after_ms = retry_after_ms
        super().__init__("scripted runtime is not ready")


class ScheduledLeaseKeeper:
    """Drive the real lease poll when the seeded scheduler chooses a renewal."""

    def __init__(self, world, participant: str):
        self.world, self.participant = world, participant

    def start(self, heartbeat):
        self.poll(heartbeat)
        if not heartbeat.stop.is_set() and not heartbeat.lost:
            self.world.enqueue(f"renew/{self.participant}", lambda: self.poll(heartbeat))

    def poll(self, heartbeat):
        if heartbeat.stop.is_set() and not heartbeat.lost:
            self.close(heartbeat)
            return
        try:
            store = self.world.connection(f"{self.participant}/heartbeat")
            transient_errors = (TimeoutError,)
            if isinstance(store, PostgresStore):
                from psycopg import OperationalError

                transient_errors = (TimeoutError, OperationalError)
            heartbeat.heartbeat_step(store, transient_errors)
        except Exception as exc:
            heartbeat.heartbeat_failed(exc)
            self.close(heartbeat)
            if hasattr(heartbeat, "token"):
                cancel_if_lost(heartbeat)
            if isinstance(exc, LeaseLost) and not isinstance(exc, HeartbeatStopped):
                raise
        else:
            if heartbeat.stop.is_set() or heartbeat.lost:
                self.close(heartbeat)

    def close(self, heartbeat):
        self.world.tasks.pop(f"renew/{self.participant}", None)
        heartbeat.heartbeat_finished()

    def is_alive(self, heartbeat):
        del heartbeat
        return False


def _cancellation_factory() -> Cancellation:
    return Cancellation(signal=threading.Event(), lock=threading.Lock(), child_factory=_cancellation_factory)


class Scope(LeaseOwnership):
    """Actual core ownership rules with scheduler-provided polling and cancellation."""

    def __init__(self, lease, runtime: CoreRuntimeAssembly, *, now_ms: int | None = None):
        self.runtime = runtime
        super().__init__(
            lease,
            runtime.ttl_ms,
            runtime.heartbeat_seconds,
            clock=runtime.clock,
            stop=threading.Event(),
            ready=threading.Event(),
            lock=threading.Lock(),
            poll_lock=threading.Lock(),
            commit_lock=threading.Lock(),
            latch_entry=threading.local(),
            actor_identity=threading.get_ident,
            now_ms=now_ms,
        )
        self.token = _cancellation_factory()
        self.turn_id = None
        self.generation = None

    def start(self):
        self.runtime.lease_keeper.start(self)

    def poll(self, store):
        poll = super().poll(store)
        apply_controls(self, poll)
        return poll

    def close(self):
        self.request_stop()
        self.runtime.lease_keeper.close(self)
        self.check_closed(self.runtime.lease_keeper.is_alive(self))


class ScheduledHeartbeat(LeaseOwnership):
    """Actual ownership state with the injected SimWorld scheduler."""

    def __init__(self, world, participant, lease, ttl_ms, heartbeat_seconds, clock, *, now_ms: int):
        self.scheduler = ScheduledLeaseKeeper(world, participant)
        super().__init__(
            lease,
            ttl_ms,
            heartbeat_seconds,
            clock=clock,
            stop=threading.Event(),
            ready=threading.Event(),
            lock=threading.Lock(),
            poll_lock=threading.Lock(),
            commit_lock=threading.Lock(),
            latch_entry=threading.local(),
            actor_identity=threading.get_ident,
            now_ms=now_ms,
        )

    def start(self):
        self.scheduler.start(self)

    def close(self):
        self.request_stop()
        self.scheduler.close(self)
        self.check_closed(self.scheduler.is_alive(self))


class DeclarationCatalog:
    """Concrete tool declarations; no admission or planning decisions live here."""

    def __init__(self, schema: dict[str, Any], metadata: ToolMetadata):
        self.schema = schema
        self.metadata = metadata
        self.version = 0

    def planning_signature(self):
        return self.version, self.schema, self.metadata.to_dict()

    def has_tool(self, name: str) -> bool:
        return name == "effect"

    def has_schema(self, name: str) -> bool:
        return name == "effect"

    def has_executor(self, name: str) -> bool:
        return name == "effect"

    def tool_metadata(self, name: str):
        return self.metadata if name == "effect" else None

    def list_openai_schemas(self, *, tool_names: list[str] | None = None):
        names = ["effect"] if tool_names is None else tool_names
        return [copy_json(self.schema) for name in names if name == "effect"]

    def get_schema(self, name: str):
        assert name == "effect"
        return copy_json(self.schema)

    def parameters_schema(self, name: str):
        assert name == "effect"
        return copy_json(self.schema["function"]["parameters"])

    def register_schema(self, name: str, schema: dict[str, Any]):
        assert name == "effect"
        self.schema = copy_json(schema)
        self.version += 1


@dataclass
class HostConfig:
    budget_limits: RunBudgetLimits = field(default_factory=RunBudgetLimits)
    host_cost_meter: Any = None
    approval_mode: str = "default"
    approval_required: bool = False
    approval_timeout_seconds: float = 300.0
    after_cycle_hooks: tuple[Any, ...] = ()
    session_memory_enabled: bool = False


@dataclass
class CallContext:
    idempotency_key: str | None
    plan: Record


def fold_factory() -> Fold:
    return Fold(normalize_memory_settings=normalize_memory_settings, count_messages=count_messages_tokens)


def _count(request: TokenCount) -> int:
    messages = [request.messages] if isinstance(request.messages, Message) else request.messages
    return count_messages_tokens([message.to_openai_message() for message in messages], model=request.model)


class CoreRuntimeAssembly:
    """Host-owned facts and effects consumed by the actual core generators."""

    def __init__(
        self,
        world,
        sid: str,
        participant: str,
        *,
        provider=None,
        children=None,
        model_callback: Callable[[Any], LLMResponse],
        hook: Callable[[str, Record | None], None],
        scenario: str | None,
        memory_settings: dict[str, Any] | None = None,
        max_cycles: int = 8,
    ):
        self.world, self.sid, self.participant, self.scenario = world, sid, participant, scenario
        self.clock = world.clock
        self.ttl_ms = 15_000
        self.poll_ms = 250
        self.heartbeat_seconds = 0.25
        self.handler_version = "sim-core-v1"
        self.model_timeout = 30.0
        self.tool_timeout = 30.0
        self.model_callback = model_callback
        self.prepare_tool_resources = copy_json
        self.hook = hook
        self.providers = {"effect": provider} if provider is not None and not children else {}
        self.children = children or {}
        self.delegated_tools = {}
        self.lease_keeper = ScheduledLeaseKeeper(world, participant)
        self.config = HostConfig(
            approval_required=scenario == "approval_reply" or (scenario == "cancel_descendants" and sid == "grandchild")
        )
        self.memory_settings = memory_settings or {}
        self.max_cycles = max_cycles
        self.model_binding = {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": ["scripted"],
            "context_length": 279_000,
            "max_output_tokens": 16_000,
        }
        metadata = ToolMetadata(
            idempotency=ToolIdempotency.SUPPORTED if scenario == "unknown_idempotent" else ToolIdempotency.UNSUPPORTED
        )
        self.registry = DeclarationCatalog(
            {
                "type": "function",
                "function": {
                    "name": "effect",
                    "description": "scripted effect",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            metadata,
        )
        self._definitions = DefinitionCache()
        self._active_driver: SimDriver | None = None
        self.approval_handle = None
        self._last_model_usage = {}

    def heartbeat_store(self):
        return self.world.heartbeat_connection(self.participant)

    def definition_bindings(self, task: AgentTask | Record) -> DefinitionBindings:
        return DefinitionBindings(
            agent_name="sim",
            memory_settings=normalize_memory_settings(self.memory_settings, None),
            child_tools=list(self.children),
            model_binding=copy_json(self.model_binding),
        )

    def compile_task(self, content: str, tid: str) -> AgentTask:
        task = AgentTask(
            task_id=tid,
            model="scripted",
            prompt_bundle=PromptBundle((PromptSection("system", "Be precise."),)),
            user_prompt=content,
            max_cycles=self.max_cycles,
            allow_interruption=False,
            use_workspace=False,
            extra_tool_names=["effect"],
            model_settings=ModelSettings(),
            metadata={
                "vv_session": {},
                "memory_compact_threshold": self.memory_settings.get("compact_threshold"),
                "memory_threshold_percentage": self.memory_settings.get("warning_threshold_percentage"),
            },
        )
        return task

    def durable_state(self, shared: dict[str, Any]) -> dict[str, Any]:
        return copy_json(shared)

    def shared_state(self, task: AgentTask) -> dict[str, Any]:
        return copy_json(task.initial_shared_state)

    def driver(self, store: SessionStore, sid: str, scope: Scope) -> SimDriver:
        driver = SimDriver(store, sid, self, scope)
        self._active_driver = driver
        return driver

    def drive(self, store: SessionStore, sid: str) -> bool:
        return drive_core(
            store,
            sid,
            make_binding=lambda: self._binding(store, sid, runtime=self),
        )

    def drive_factory(self, store: SessionStore, sid: str, factory: Callable[[str], Any]) -> bool:
        return drive_core(
            store,
            sid,
            make_binding=lambda: self._binding(store, sid, runtime=None, factory=factory),
        )

    def _binding(
        self,
        store: SessionStore,
        sid: str,
        *,
        runtime: CoreRuntimeAssembly | None,
        factory: Callable[[str], Any] | None = None,
    ) -> DriveBinding:
        selected = runtime or self

        def make_factory_heartbeat(lease, ttl_ms: int):
            return ScheduledHeartbeat(
                self.world,
                f"{self.participant}/factory",
                lease,
                ttl_ms,
                self.heartbeat_seconds,
                self.clock,
                now_ms=cast(SQLStore, store)._now(),
            )

        def make_scope(lease, selected_runtime):
            runtime = cast(CoreRuntimeAssembly, selected_runtime)
            return Scope(lease, runtime, now_ms=cast(SQLStore, store)._now())

        def make_driver(selected_store, selected_sid, selected_runtime, scope):
            runtime = cast(CoreRuntimeAssembly, selected_runtime)
            driver = SimDriver(selected_store, selected_sid, runtime, cast(Scope, scope))
            runtime._active_driver = driver
            return driver

        return DriveBinding(
            clock=selected.clock,
            ttl_ms=selected.ttl_ms,
            runtime=runtime,
            make_factory_heartbeat=None if runtime is not None else make_factory_heartbeat,
            make_runtime=None if factory is None else lambda: factory(sid),
            factory_retry_after_ms=lambda error: error.retry_after_ms if isinstance(error, SimRuntimeNotReady) else None,
            heartbeat_alive=lambda heartbeat: cast(ScheduledHeartbeat, heartbeat).scheduler.is_alive(heartbeat),
            make_scope=make_scope,
            make_driver=make_driver,
            after_drive=lambda selected_runtime: cast(CoreRuntimeAssembly, selected_runtime).hook("after_drive", None),
            wake_if_runnable=lambda selected_runtime: cast(CoreRuntimeAssembly, selected_runtime).world.queue.append(sid),
        )


class SimDriver(CoreDriver):
    """Thin runner that drains the actual core generators."""

    scope: Scope

    def __init__(self, store: SessionStore, sid: str, assembly: CoreRuntimeAssembly, scope: Scope):
        self.runtime = assembly
        self.one_turn = False
        self.provider = None
        self.context_plan = None
        self.context_task = None
        self.context_shared = None
        self.context_approved = False
        self.output_agent = None
        super().__init__(
            store,
            sid,
            scope,
            fold_factory=fold_factory,
            clock=assembly.clock,
            observe_cost=self._observe_cost,
            hook=lambda point, record: assembly.hook(point, record),
            emit_phase=lambda _duration, _fields: None,
        )

    def _observe_cost(self):
        meter = self.runtime.config.host_cost_meter
        if meter is None:
            return None
        value = meter.read()
        return value.to_dict() if value is not None else None

    def task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        return self.state.turns[self.state.active_turn_id].start.task()

    def _task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        return self.state.turns[self.state.active_turn_id].start._task()

    def run(self, steps: Generator[object, Any, Any]):
        response = None
        error = None
        budget_meter = None
        try:
            while True:
                try:
                    request = steps.throw(error) if error is not None else steps.send(response)
                except StopIteration as done:
                    return done.value
                except (_driver_recovery._RecoveryStopIteration, _StopIterationTransport) as transport:
                    raise transport.original from transport.original.__cause__
                except outcomes._OutcomeStopIteration as transport:
                    raise transport.original from transport.original.__cause__
                response, error = None, None
                try:
                    if request is outcomes.OutcomeRead.BIND_BUDGET_METER:
                        # Binding is local to this run so nested receipt runs cannot
                        # replace the meter observed by the outer generator.
                        budget_meter = self.runtime.config.host_cost_meter
                    elif request is BudgetObservation.CLOCK_NS:
                        response = self.clock.monotonic_ns()
                    elif request is BudgetObservation.HOST_COST:
                        response = self._host_cost(budget_meter)
                    else:
                        response = self.handle(request)
                except Exception as exc:
                    error = _StopIterationTransport(exc) if isinstance(exc, StopIteration) else exc
        finally:
            steps.close()

    def handle(self, request):
        if request is _driver_recovery.Read.REFRESH:
            return self.refresh()
        if request in (_driver_recovery.Read.CANCELLED, inputs.InputSetting.CANCELLED):
            return self.scope.token.cancelled
        if request in (_driver_recovery.Read.RESET_TOKEN, inputs.InputSetting.RESET_TOKEN):
            self.scope.token = _cancellation_factory()
            return None
        if request in (_driver_recovery.Read.HANDLER_VERSION, inputs.InputSetting.HANDLER_VERSION):
            return self.runtime.handler_version
        if request is _driver_recovery.Read.AFTER_CYCLE_CONFIGURED:
            return bool(self.runtime.config.after_cycle_hooks)
        if request is _driver_recovery.Read.START_AFTER_CYCLE:
            return False
        if request is _driver_recovery.Read.START_OUTPUT:
            return None
        if request is _driver_recovery.Read.MEMORY_HOOKS:
            return False
        if request is _driver_recovery.Read.MAX_PTL_RETRIES:
            return 3
        if request is _driver_recovery.Read.NEXT_DRIVE_DELAY:
            with self.scope.committing() as lease:
                return self.store.next_drive_delay_ms(self.sid, lease=lease)
        if isinstance(request, (_driver_recovery.Definition, inputs.CurrentDefinition)):
            task = request.start if isinstance(request, _driver_recovery.Definition) else request.task
            return inputs.DefinitionFacts(
                self.runtime._definitions, self.runtime.definition_bindings(task), self.runtime.registry
            )
        if isinstance(request, _driver_recovery.BindHostState):
            return self.runtime.shared_state(request.task)
        if isinstance(request, _driver_recovery.Hook):
            self.runtime.hook(request.point, request.record)
            return None
        if isinstance(request, _driver_recovery.SelectProvider):
            self.provider = self.runtime.providers.get(request.plan._payload["provider_binding"])
            return None
        if isinstance(request, _driver_recovery.ProviderCall):
            provider = self.provider
            assert provider is not None
            callback = (
                (lambda: provider.query(request.handle))
                if request.method == "query"
                else (lambda: provider.cancel(request.handle))
            )
            return self.invoke(callback, request.timeout)
        if isinstance(request, _driver_recovery.CommitStopping):
            return self.commit(
                request.records,
                guarded=True,
                prepare=(lambda tx: _child_store.cancel_children(self.store, tx, self.state, self.sid, request.turn_id))
                if request.stopping
                else None,
            )
        if isinstance(request, CommitRecords):
            return self.commit(request.records, guarded=True)
        if isinstance(request, inputs.CommitInput):
            return self.commit(request.records, (request.input_id,), guarded=True)
        if isinstance(request, inputs.CompileTurn):
            return self.runtime.compile_task(request.content, request.turn_id)
        if isinstance(request, inputs.AuthenticateProvider):
            return self.runtime.providers[request.plan._payload["provider_binding"]].authenticate(request.item, request.plan)
        if isinstance(request, inputs.ProjectProviderResult):
            return copy_json(request.result)
        if isinstance(request, inputs.PrepareToolContext):
            self.context_plan = request.plan
            return None
        if isinstance(request, inputs.ChildResult):
            return ordinary_result_for_children(self.store, request.plan, request.handles).to_dict()
        if request is inputs.InputSetting.POLL_NOW:
            return self.scope.poll(self.store).db_now_ms
        if request is inputs.InputSetting.POLL_MS:
            return self.runtime.poll_ms
        if request is inputs.InputSetting.BUDGET_LIMITS:
            return self.runtime.config.budget_limits
        if isinstance(request, planning.ResolveModel):
            return None
        if request is planning.PlanningRead.MODEL_BINDING or request is planning.PlanningRead.DEFAULT_MODEL_BINDING:
            return copy_json(self.runtime.model_binding)
        if request is planning.PlanningRead.ENDPOINT_SUPPORT:
            return True
        if isinstance(request, planning.OrderedEndpoints):
            return ["scripted"]
        if isinstance(request, planning.ProviderAvailable):
            return request.name in self.runtime.providers
        if request is planning.PlanningRead.HAS_HOOKS or request is planning.PlanningRead.PREPARE_BEFORE_LLM:
            return False
        if request is planning.PlanningRead.SHARED_STATE or request is cycle.ReadSharedState:
            return self.shared_state()
        if request is planning.PlanningRead.INTERRUPTION_MESSAGES or isinstance(request, planning.BeforeCycle):
            return False, None
        if isinstance(request, planning.BeforeLLM):
            return list(request.messages), list(request.schemas)
        if isinstance(request, planning.DurableState):
            return self.runtime.durable_state(request.shared)
        if isinstance(request, cycle.ReadTranscript) or request is compaction.CompactionRequest.TRANSCRIPT:
            return self.transcript()
        if isinstance(request, cycle.ReadCycleProjection):
            raise AssertionError("after-cycle hooks are not configured")
        if request is compaction.CompactionRequest.INBOX:
            return self.store.peek_inbox(self.sid, through_input_seq=self.watermark)
        if request is compaction.CompactionRequest.MEMORY:
            settings = normalize_memory_settings(self.runtime.memory_settings, None)
            return compaction.CompactionMemory(settings, None, False)
        if request is compaction.CompactionRequest.RESTORE_SESSION_MEMORY:
            return None
        if request is compaction.CompactionRequest.SAVE_MEMORY_PROJECTION:
            return None
        if isinstance(request, compaction.ModelRoute) or request is child_summary.SummaryRead.MODEL_ROUTE:
            return "scripted"
        if isinstance(request, (compaction.CountContext, child_summary.CountSummaryPrompt)):
            messages = request.messages
            return count_messages_tokens(
                [message.to_openai_message() for message in messages], model=self.runtime.memory_settings.get("model", "")
            )
        if isinstance(request, child_summary.CountPrefix):
            messages = request.request.messages
            messages = [messages] if isinstance(messages, Message) else messages
            return request.count_messages([message.to_openai_message() for message in messages], model=request.request.model)
        if isinstance(request, TokenCount):
            return _count(request)
        if isinstance(request, compaction.MicroArtifacts):
            return None
        if isinstance(request, (compaction.BeforeCompaction, compaction.AfterCompaction)):
            return self._compaction_event(request)
        if request is child_summary.SummaryRead.PREPARE_MEMORY:
            return None
        if request is child_summary.SummaryRead.PLAN_SETTINGS:
            settings = normalize_memory_settings(self.runtime.memory_settings, None)
            return {
                "protected_start": None,
                "language": settings["language"],
                "event_limit": settings["summary_event_limit"],
            }
        if request is child_summary.SummaryRead.ACCEPT_SETTINGS:
            settings = normalize_memory_settings(self.runtime.memory_settings, None)
            return {
                "protected_start": None,
                "recovery_tool_available": False,
                "effective_context_window": settings["model_context_window"] - settings["reserved_output_tokens"],
                "model": settings["model"],
            }
        if request is child_summary.SummaryRead.MODEL_CONTEXT_WINDOW:
            return normalize_memory_settings(self.runtime.memory_settings, None)["model_context_window"]
        if isinstance(request, _driver_recovery.BeginApproval):
            self.approval_handle = request.handle
            return False, False
        if request is _driver_recovery.ApprovalRead.HAS_ANSWER:
            assert self.approval_handle is not None
            request_id = self.approval_handle["request_id"]
            return any(
                item.item.kind == "approval_answer" and item.item.payload["request_id"] == request_id
                for item in self.store.peek_inbox(self.sid)
            )
        if isinstance(request, output.OutputSetting):
            return False if request is not output.OutputSetting.REPAIR_MODEL else None
        if isinstance(request, output.SerializeOutput):
            return output.serializable_output(request.value)
        if isinstance(request, outcomes.OutcomeRead):
            return self._outcome_read(request)
        if isinstance(request, outcomes.PrepareToolResources):
            return self.runtime.prepare_tool_resources(request.result)
        if isinstance(request, outcomes.LookupYieldCallback):
            binding = (
                self.runtime.children.get(request.plan._payload["request"]["name"])
                if request.child
                else self.runtime.providers.get(request.plan._payload["provider_binding"])
            )
            return getattr(binding, "on_yield", None)
        if isinstance(request, outcomes.InvokeYieldCallback):
            try:
                return request.callback(request.plan, request.handles, "steer")
            except Exception as error:
                return outcomes.YieldCallbackError(error)
        return self._handle_dispatch(request)

    def _handle_dispatch(self, request):
        if isinstance(request, _driver_dispatch.PrepareContext):
            self.context_plan, self.context_task = request.plan, request.task
            self.context_shared = self.shared_state()
            self.context_approved = request.approved
            return None
        if isinstance(request, _driver_dispatch.FinishContext):
            return {"session_tool_names": ["effect"]}
        if isinstance(request, _driver_dispatch.SetContextCycle) or request is _driver_dispatch.DispatchRead.PREPARE_AFTER_TOOL:
            return None
        if request is _driver_dispatch.DispatchRead.HAS_HOOKS:
            return False
        if request is _driver_dispatch.DispatchRead.CONTEXT_CYCLE:
            plan = self.context_plan
            assert plan is not None
            deps = plan._payload["dependencies"]
            parent = self.state.operations[deps[0]].attempts[1].plan
            return parent._payload["request"]["metadata"].get("vv_session", {}).get("cycle_index", 1)
        if request is _driver_dispatch.DispatchRead.CONTEXT_SHARED:
            return self.context_shared
        if request is _driver_dispatch.DispatchRead.DURABLE_CONTEXT:
            return self.durable_shared_state()
        if request is _driver_dispatch.DispatchRead.HANDLER_VERSION:
            return self.runtime.handler_version
        if request is _driver_dispatch.DispatchRead.POLL_LEASE:
            return self.scope.poll(self.store)
        if request is _driver_dispatch.DispatchRead.CANCELLED:
            return self.scope.token.cancelled
        if isinstance(request, _driver_dispatch.BindPreparedState):
            return copy_json(request.shared)
        if isinstance(request, _driver_dispatch.PrepareModel):
            return True
        if isinstance(request, _driver_dispatch.SelectProvider):
            self.provider = self.runtime.providers.get(request.plan._payload["provider_binding"])
            return self.provider is not None
        if isinstance(request, _driver_dispatch.InvokeOperation):
            if request.plan._payload["op_kind"] == "model":
                try:
                    return Definitive(
                        self._model_result(request.plan, self.runtime.model_callback),
                        ("scripted-model",),
                        usage=self._last_model_usage,
                    )
                except Exception as exc:
                    return Unknown(f"{type(exc).__name__}: {exc}")
            context = CallContext(request.plan._payload["idempotency_key"], request.plan)
            provider = self.provider
            assert provider is not None
            try:
                return provider.submit(request.plan, context=context)
            except Exception as exc:
                return Unknown(f"{type(exc).__name__}: scripted outcome unavailable")
        if isinstance(request, _driver_dispatch.DispatchHook):
            self.runtime.hook(request.point, request.plan)
            return None
        if isinstance(request, _driver_dispatch.CommitDispatch):
            return self.commit(request.records, guarded=request.guarded)
        if isinstance(request, _driver_dispatch.ChildBinding):
            return False
        if isinstance(request, _driver_dispatch.HasChild):
            return request.name in self.runtime.children
        if isinstance(request, _driver_dispatch.HasDelegation):
            return request.name in self.runtime.delegated_tools
        if isinstance(request, _driver_dispatch.CallChildAdmission):
            binding = self.runtime.children.get(request.plan._payload["request"]["name"])
            return binding(request.plan) if binding is not None else request.child_specs
        if isinstance(request, _driver_dispatch.AdmitChildren):
            return self.commit(
                request.records,
                guarded=True,
                prepare=lambda tx: self.run(_driver_dispatch.admit_children(self, tx, request)),
            )
        if isinstance(request, _driver_dispatch.PrepareAfterTool):
            return None
        if isinstance(request, _driver_dispatch.AfterTool):
            return request.result
        if isinstance(request, child_summary.RequiresSummary):
            return False
        if isinstance(request, _driver_dispatch.PreflightHandler):
            return False
        if isinstance(request, _driver_dispatch.BeginPreflight):
            return ["effect"]
        if isinstance(request, _driver_dispatch.EndPreflight):
            return None
        if isinstance(request, _driver_dispatch.PrepareAdmission):
            return self.runtime.registry.has_executor(request.call.name)
        if isinstance(request, admission.Read):
            if request is admission.Read.SCHEMA:
                return self.runtime.registry.parameters_schema("effect")
            if request is admission.Read.HANDLER_MANAGED:
                return False
            if request is admission.Read.POLICY_METADATA:
                assert self.context_task is not None
                return dict(self.context_task.metadata)
            if request is admission.Read.TOOL_METADATA:
                return self.runtime.registry.metadata
            if request is admission.Read.APPROVAL_MODE:
                return self.runtime.config.approval_mode
        if isinstance(request, admission.SchemaCheck):
            return schema_issues(request)
        if isinstance(request, admission.ApprovedCall):
            assert self.context_plan is not None
            return approved_tool_call(self.context_plan) if self.context_approved else None
        if isinstance(request, admission.PolicyPredicate):
            return bool(request.predicate(request.tool_name, request.arguments))
        if isinstance(request, admission.RequiresApproval):
            return self.runtime.config.approval_required
        if isinstance(request, admission.RequestApproval):
            assert self.context_plan is not None
            return ToolExecutionResult(
                tool_call_id=self.context_plan._payload["request"]["id"],
                content="Approval required.",
                status_code=ToolResultStatus.ERROR,
                error_code="tool_approval_required",
            )
        if isinstance(request, _driver_dispatch.InvokePreflight):
            return True, None
        if isinstance(request, _driver_dispatch.FinishPreflight):
            if not isinstance(request.result, ToolExecutionResult):
                raise ValueError("preflight handler bypassed the dispatch boundary")
            return request.result
        if isinstance(request, _driver_dispatch.AdmissionException):
            return type(request.error) is CancelledError
        if isinstance(request, _driver_dispatch.FormatAdmissionException):
            message = f"Tool execution failed ({request.call.name}): {request.error}"
            return admission._error_result(request.call.id, message, error_code="tool_execution_failed")
        raise TypeError(f"unsupported core host request: {type(request).__name__}")

    def _model_result(self, plan: Record, callback: Callable[[Any], LLMResponse]) -> dict[str, Any]:
        request = _RequestFact(
            plan,
            plan._payload["request"]["metadata"]
            | {
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "call_id": f"{plan.operation_id}/{plan.attempt}",
                "purpose": plan._payload["purpose"],
            },
        )
        response = callback(request)
        self._last_model_usage = copy_json(response.raw.get("usage") or {})
        return {
            "content": response.content,
            "tool_calls": [call.to_dict() for call in response.tool_calls],
            "raw": copy_json(response.raw),
            "reasoning_content": response.raw.get("reasoning_content"),
        }

    def _compaction_event(self, request):
        now = self.records[-1].created_ms / 1000 if self.records else 0.0
        common = {
            "version": "v6",
            "event_id": f"sim/{request.key}/{'started' if isinstance(request, compaction.BeforeCompaction) else 'completed'}",
            "run_id": self.state.active_turn_id or "",
            "trace_id": self.state.active_turn_id or "",
            "created_at": now,
            "session_id": self.sid,
            "agent_name": "sim",
            "metadata": {},
        }
        if isinstance(request, compaction.BeforeCompaction):
            settings = normalize_memory_settings(self.runtime.memory_settings, None)
            manager = compaction.CompactionMemory(settings, None, False)
            effective_threshold = manager.autocompact_threshold
            policy = settings["microcompaction_policy"]
            return common | {
                "type": "memory_compact_started",
                "cycle_index": request.cycle,
                "message_count": len(request.source),
                "trigger": request.trigger,
                "configured_threshold": settings["compact_threshold"],
                "effective_threshold": effective_threshold,
                "microcompact_threshold": int(effective_threshold * policy["trigger_ratio"]),
                "microcompact_target": int(effective_threshold * policy["target_ratio"]),
                "candidate_count": request.prune.candidate_count if request.prune else 0,
                "estimated_reclaimable_tokens": request.prune.estimated_reclaimable_tokens if request.prune else 0,
                "model_context_window": settings["model_context_window"],
                "model_max_output_tokens": settings["model_max_output_tokens"],
                "reserved_output_tokens": settings["reserved_output_tokens"],
                "reserved_output_source": settings["reserved_output_source"],
                "autocompact_buffer_tokens": settings["autocompact_buffer_tokens"],
            }
        return common | {
            "type": "memory_compact_completed",
            "cycle_index": request.start["cycle_index"],
            "before_count": request.start["message_count"],
            "after_count": len(request.source),
            "mode": request.mode if request.changed else "none",
            "changed": request.changed,
            "archived_count": request.archived_count,
            "reclaimed_tokens": request.reclaimed_tokens,
            "artifact_failure_count": request.artifact_failure_count,
        }

    def _outcome_read(self, request):
        if request is outcomes.OutcomeRead.CANCELLATION_REASON:
            return self.scope.token.reason
        if request is outcomes.OutcomeRead.SCOPE_CANCELLED or request is outcomes.OutcomeRead.HOST_CANCELLED:
            return self.scope.token.cancelled
        if request is outcomes.OutcomeRead.ACTIVE_INTERVAL_NS:
            return self.clock.monotonic_ns()
        if request is outcomes.OutcomeRead.APPROVAL_TIMEOUT:
            return self.runtime.config.approval_timeout_seconds
        if request is outcomes.OutcomeRead.POLL_LEASE:
            return self.scope.poll(self.store)
        if request is outcomes.OutcomeRead.POLL_INTERVAL:
            return self.runtime.poll_ms
        raise TypeError(type(request).__name__)

    def _host_cost(self, meter):
        if meter is None:
            return HostCostObservation(unavailable_reason=BudgetUnavailableReason.METER_MISSING)
        try:
            reading = meter.read()
        except Exception:
            return HostCostObservation(unavailable_reason=BudgetUnavailableReason.METER_ERROR)
        return HostCostObservation(reading=reading)

    def invoke(self, callback, _timeout):
        self.runtime.world.checkpoint("invoke")
        return callback()

    def plan(self, oid, request, kind, *, number=1, dependencies=None, capability=None, purpose="primary"):
        return self.run(
            planning.plan(
                self, oid, request, kind, number=number, dependencies=dependencies, capability=capability, purpose=purpose
            )
        )

    def completed(self, plan, outcome):
        return self.run(outcomes.completed_steps(self, plan, outcome))

    def parked(self, plan, handle, *, after):
        return self.run(outcomes.parked_steps(self, plan, handle, after=after))

    def accepted(self, plan, outcome):
        return self.run(outcomes.accepted_steps(self, plan, outcome))

    def unknown(self, plan, reason):
        return self.run(outcomes.unknown_steps(self, plan, reason))

    def yield_results(self, attempts, item):
        return self.run(outcomes.yield_results_steps(self, attempts, item))

    def transcript(self):
        from vv_agent.kernel._context import project_context

        return project_context(self.records, self.state, _message_cache=self.fold._messages)

    def shared_state(self):
        if self.state.active_turn_id is None:
            return {}
        return self.durable_shared_state()

    def model_plan(self):
        return self.run(planning.model_plan(self, self.transcript()))

    def effective_model(self):
        return self.run(planning.effective_model(self))

    def step(self):
        return self.run(_driver_recovery.step(self, one_turn=self.one_turn))

    def close(self, status, reason, result=None, *, transferred=False):
        return self.run(_driver_recovery.close(self, status, reason, result, transferred=transferred))

    def dispatch(self, attempt):
        return self.run(_driver_dispatch.dispatch(self, attempt))

    def finalize_tool(self, plan, context, outcome):
        del context
        return self.run(_driver_dispatch.finalize_tool(self, plan, outcome))


@dataclass(frozen=True)
class _RequestFact:
    plan: Record
    metadata: dict[str, Any]
