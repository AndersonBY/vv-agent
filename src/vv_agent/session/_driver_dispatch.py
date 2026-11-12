"""Execute admitted operations and finalize tool results through the driver."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

from vv_agent.events import _project_provider_stream_payload
from vv_agent.llm.errors import (
    ModelStreamInterrupted,
    is_prompt_too_long_error,
    model_request_rejection,
    model_response_rejection,
    safe_model_error,
)
from vv_agent.output_validation import OutputRepairRequest
from vv_agent.runtime.tool_results import apply_tool_use_behavior, build_skipped_result
from vv_agent.tools.handlers.common import builtin_error
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from ._driver_scope import _invoke
from .children import ChildSession, ChildTool, InvalidChildBatch, SummaryChildTool, create_child
from .delegation import admitted_result, assemble
from .output import serializable_output
from .providers import Accepted, Definitive, Outcome, ProviderResultInvalid, Unknown, validate_acceptance
from .records import Record, copy_json, digest, inherited_seed_policies
from .reducer import Attempt
from .runtime import EndpointUnavailable, request_from_dict
from .store import SequenceConflict, SessionTx
from .summary_requests import child_summary

if TYPE_CHECKING:
    from .kernel import _Driver


def dispatch(self: _Driver, attempt: Attempt) -> None:
    plan = attempt.execution_plan
    kind = plan._payload["op_kind"]
    assert self.state.active_turn_id is not None
    task = (
        self.state.turns[self.state.active_turn_id].start._task()
        if kind == "model" and not self.runtime.hooks.has_hooks()
        else self.task()
    )
    context = self.runtime.context(
        task,
        plan,
        self.scope.token,
        approved=attempt.approval in {"approve", "allow_session"},
        shared_state=self.shared_state(),
        store=self.store,
        session_id=self.sid,
    )
    context.metadata["fence_epoch"] = self.scope.lease.epoch
    if kind != "model":
        parent = self.state.operations[plan._payload["dependencies"][0]]
        source = parent.attempts[parent.selected_attempt or max(parent.attempts)].plan
        context.metadata["session_tool_names"] = [s["function"]["name"] for s in source._payload["request"]["tools"]]
        context.cycle_index = (
            source._payload["request"].get("metadata", {}).get("vv_session", {}).get("cycle_index", context.cycle_index)
        )
    if kind != "model" and self.runtime.hooks.has_hooks() and attempt.prepared is None:
        patched, short = self.runtime.hooks.apply_before_tool_call(
            task=task, cycle_index=context.cycle_index, call=ToolCall.from_dict(plan.payload["request"]), context=context
        )
        capabilities = self.state.turns[plan.turn_id or ""].start._payload["definition"]["capabilities"]
        prepared = self.plan(
            plan.operation_id or "",
            patched.to_dict(),
            "interaction" if patched.name == "ask_user" else "tool",
            dependencies=plan._payload["dependencies"],
            capability=capabilities.get(patched.name, {}),
        )
        self.commit(
            [
                self.record(
                    "op_prepared",
                    {
                        "request": patched.to_dict(),
                        "request_digest": digest(patched.to_dict()),
                        "tool": prepared._payload["tool"],
                        "op_kind": prepared._payload["op_kind"],
                        "provider_binding": prepared._payload["provider_binding"],
                        "idempotency_key": prepared._payload["idempotency_key"],
                        "hook_result": short.to_dict() if short else None,
                        "shared_state": self.runtime.durable_state(context.shared_state),
                    },
                    plan,
                )
            ],
            guarded=True,
        )
        return
    if attempt.prepared is not None:
        context.shared_state.clear()
        context.shared_state.update(self.runtime.bind_state(copy_json(plan._payload["budget_admission"]["shared_state"]), task))
    hook_result = plan._payload["budget_admission"].get("hook_result")
    if hook_result is not None:
        self.commit(self.completed(plan, self.finalize_tool(plan, context, Definitive(hook_result))), guarded=True)
        return
    if attempt.approval in {"deny", "timeout"}:
        assert attempt.approval_answer is not None
        answer = attempt.approval_answer._payload["input"]["payload"]
        executor = self.runtime.functions.orchestrator._resolve_executor(plan._payload["request"]["name"])
        assert executor is not None
        result = ToolOrchestrator._approval_error_result(
            executor=executor,
            call=ToolCall.from_dict(plan.payload["request"]),
            request_id=answer["request_id"],
            error_code="tool_approval_timeout" if attempt.approval == "timeout" else "tool_approval_denied",
            action=attempt.approval,
            message=answer.get("reason")
            or ("Approval request timed out." if attempt.approval == "timeout" else f"Approval denied for tool {executor.name}."),
        )
        self.commit(self.completed(plan, self.finalize_tool(plan, context, Definitive(result.to_dict(), ()))))
        return
    if kind != "model":
        name = plan._payload["request"]["name"]
        if any(
            r._payload["disposition"] == "applied"
            and r._payload["input"]["kind"] == "approval_answer"
            and r._payload["input"]["payload"]["decision"] == "allow_session"
            and name in r._payload["input"]["payload"]["scope"]
            for r in self.state.applied_inputs.values()
        ):
            assert context.ctx is not None
            context.ctx._approved_tool_approval = SimpleNamespace(call=ToolCall.from_dict(plan.payload["request"]))
        preflight = self.runtime.functions.preflight(plan, context)
        if preflight is not None:
            if preflight.error_code == "tool_approval_required":
                handle = {
                    "kind": "approval",
                    "request_id": f"approval/{plan.operation_id}/{plan.attempt}",
                    "request_digest": plan._payload["request_digest"],
                    "scope": [plan._payload["request"]["name"]],
                }
                self.commit([self.parked(plan, handle, after=False)], guarded=True)
            else:
                self.commit(
                    self.completed(plan, self.finalize_tool(plan, context, Definitive(preflight.to_dict(), ()))), guarded=True
                )
            return
    prepared_client = None
    if kind == "model":
        try:
            prepared_client = self.runtime.prepare_model(plan._payload["request"], plan._payload["purpose"], plan.attempt or 1)
        except EndpointUnavailable:
            self.commit(
                self.completed(plan, Definitive({"content": "", "tool_calls": [], "error_code": "endpoint_unavailable"}, ())),
                guarded=True,
            )
            return
    evaluator = self.live_budget()
    exhaustion = None
    budget_records = []
    if evaluator:
        exhaustion = (
            (evaluator.cycle_start() if plan._payload["purpose"] == "primary" else evaluator.model_call_start())
            if kind == "model"
            else evaluator.run_start()
        )
        if self.boundary("budget", f"{plan.operation_id}/start/{plan.attempt}") is None:
            budget_records = [
                self.budget_record(f"{plan.operation_id}/start/{plan.attempt}", evaluator, exhaustion, source=plan.operation_id)
            ]
    if exhaustion:
        self.commit(budget_records, guarded=True)
        self.close("failed", "budget_exhausted")
        return
    provider = self.provider(plan) if kind != "model" else None
    poll = self.scope.poll(self.store)
    if poll.inbox_seq != self.watermark or self.scope.token.cancelled:
        raise SequenceConflict("input arrived before dispatch")
    if kind == "interaction" or plan._payload["request"].get("name") == "ask_user":
        outcome = self.runtime.functions.submit(plan, context=context)
        if not isinstance(outcome, Definitive):
            raise ValueError("ask_user must be a synchronous interaction")
        if outcome.result.get("directive") != "wait_user":
            self.commit(self.completed(plan, self.finalize_tool(plan, context, outcome)), guarded=True)
            return
        question = outcome.result.get("metadata", {}).get("question", outcome.result["content"])
        parked = self.parked(
            plan, {"kind": "user", "interaction_id": f"interaction/{plan.operation_id}", "question": question}, after=False
        )
        parked = replace(parked, payload=parked._payload | {"interaction_result": outcome.result})
        from .interactions import InteractionError, user_interaction

        assert plan.turn_id is not None
        try:
            user_interaction(
                self.sid,
                plan.turn_id,
                self.state.turns[plan.turn_id].start._payload["generation"],
                parked._payload,
                operation_id=plan.operation_id,
                attempt=plan.attempt,
            )
        except InteractionError:
            invalid = builtin_error("Interaction form exceeds the UTF-8 size or option limit.", "interaction_form_too_large")
            invalid.tool_call_id = plan._payload["request"]["id"]
            self.commit(self.completed(plan, self.finalize_tool(plan, context, Definitive(invalid.to_dict(), ()))), guarded=True)
            return
        skipped = []
        for oid, op in self.state.operations.items():
            pending = op.attempts[max(op.attempts)]
            if (
                oid != plan.operation_id
                and op.turn_id == plan.turn_id
                and op.kind != "model"
                and pending.state == "planned"
                and pending.plan._payload["dependencies"] == plan._payload["dependencies"]
            ):
                result = build_skipped_result(
                    ToolCall.from_dict(pending.execution_plan.payload["request"]),
                    error_code="skipped_due_to_wait_user",
                    message="Tool skipped because a previous tool requested user input.",
                )
                skipped.extend(self.completed(pending.execution_plan, Definitive(result.to_dict(), ())))
        self.commit([parked, *skipped], guarded=True)
        return
    started = self.record(
        "op_started",
        {
            "dispatch_id": f"{plan.operation_id}/{plan.attempt}",
            "authorization_version": self.runtime.handler_version,
            "epoch": self.scope.lease.epoch,
            "mode": "sync" if kind == "model" else "provider",
        },
        plan,
    )
    if kind == "model" and plan._payload["request"]["metadata"].get("vv_session", {}).get("endpoint_order"):
        order = plan._payload["request"]["metadata"]["vv_session"]["endpoint_order"]
        started = replace(started, payload=started._payload | {"endpoint_id": order[((plan.attempt or 1) - 1) % len(order)]})
    name = plan._payload["request"].get("name")
    summary = None
    binding = self.runtime.children.get(name)
    if kind != "model" and isinstance(binding, SummaryChildTool):
        ready, summary = child_summary(self, plan, binding)
        if not ready:
            return
    if kind != "model" and (name in self.runtime.children or name in self.runtime.delegated_tools):
        child_specs = None if name in self.runtime.children else assemble(self.runtime, plan, context, task, self.store)
        if isinstance(child_specs, ToolExecutionResult):
            child_specs.tool_call_id = plan._payload["request"]["id"]
            self.commit(self.completed(plan, self.finalize_tool(plan, context, Definitive(child_specs.to_dict()))), guarded=True)
            return
        if child_specs is not None or name in self.runtime.children:
            assert plan.turn_id is not None

            def admit_child(tx: SessionTx) -> list[Record]:
                if isinstance(binding, SummaryChildTool):
                    specs = binding.admit(plan, summary)
                elif isinstance(binding, ChildTool):
                    specs = binding.admit(plan)
                else:
                    specs = binding(plan) if binding is not None else child_specs
                if isinstance(specs, ChildSession):
                    specs = [specs]
                if not isinstance(specs, Sequence) or not specs or any(not isinstance(c, ChildSession) for c in specs):
                    raise InvalidChildBatch("Expected one ChildSession or a nonempty sequence of ChildSession")
                if any(c.background != specs[0].background for c in specs):
                    raise InvalidChildBatch("All children in a batch must have the same background flag")
                if any(type(c.yield_on_steer) is not bool or (c.background and c.yield_on_steer) for c in specs):
                    raise InvalidChildBatch("Only foreground children may declare yield_on_steer=True")
                assert specs is not None
                generation = self.state.turns[plan.turn_id or ""].start._payload["generation"]
                inherited = inherited_seed_policies(self.records[0].record._payload["attributes"])
                inherited_specs = []
                for child in specs:
                    attributes = copy_json(child.spec.attributes or {})
                    attributes.setdefault("seed", {}).update(inherited)
                    inherited_specs.append(replace(child, spec=replace(child.spec, attributes=attributes)))
                handles = [create_child(tx, child, plan, generation) for child in inherited_specs]
                handle = handles[0]
                if len(handles) > 1:
                    handle = handle | {
                        "siblings": [{k: h[k] for k in ("session_id", "turn_id", "generation")} for h in handles[1:]]
                    }
                parked = self.parked(plan, handle, after=True)
                yieldable = [c.spec.session_id for c in specs if c.yield_on_steer]
                if yieldable:
                    parked = replace(parked, payload=parked._payload | {"yield_on_steer": True, "yieldable_children": yieldable})
                admission = specs[0].spec.attributes.get("child_admission") if specs[0].spec.attributes else None
                if admission:
                    marker = {key: admission[key] for key in ("mode", "handoff_count", "max_handoffs")}
                    marker["agent_name"] = admission["definition"]["agent_name"]
                    marker["metadata"] = admission["handoff_metadata"]
                    parked = replace(parked, payload=parked._payload | {"delegation": marker})
                records = [parked]
                if specs[0].background:
                    result = admitted_result(self.store, plan, handles)
                    records.extend(
                        self.completed(
                            plan,
                            self.finalize_tool(
                                plan,
                                context,
                                Definitive(result.to_dict(), tuple(f"session/{h['session_id']}/created" for h in handles)),
                            ),
                        )
                    )
                    # completed() sees the pre-transaction state; admission starts in this commit.
                    records[-1] = replace(records[-1], payload=records[-1]._payload | {"execution_started": True})
                return records

            self.commit([*budget_records, started], guarded=True, prepare=admit_child)
            return
    self.commit([*budget_records, started], guarded=True)
    self.runtime.hook("before_external_call", plan)

    def model() -> Outcome:
        request = request_from_dict(copy_json(plan._payload["request"]))
        request.metadata.update(
            {"operation_id": plan.operation_id, "attempt": plan.attempt, "call_id": f"{plan.operation_id}/{plan.attempt}"}
        )
        request.metadata["purpose"] = plan._payload["purpose"]
        response_policy = self.records[0].record._payload["attributes"].get("seed", {}).get("response_rejection", False)
        if started._payload.get("endpoint_id"):
            request.metadata.setdefault("vv_session", {})["endpoint_id"] = started._payload["endpoint_id"]
        if plan._payload["purpose"] == "output_repair":
            repair = self.runtime.agent.output_repair
            assert repair is not None
            try:
                repaired = repair(
                    OutputRepairRequest(
                        invalid_output=request.metadata["invalid_output"],
                        validation_code=request.metadata["validation_code"],
                        validation_message=request.metadata["validation_message"],
                        model=self.runtime.agent.output_repair_model,
                        model_settings=self.runtime.agent.output_repair_model_settings,
                    )
                )
            except Exception as exc:
                return Definitive({"content": safe_model_error(exc), "tool_calls": [], "error_code": "repair_provider_error"})
            if isinstance(repaired, LLMResponse):
                rejection = model_response_rejection(repaired.raw.get("completion_metadata", {})) if response_policy else None
                if rejection is not None:
                    return Definitive(
                        {
                            "content": f"HTTP 200: {rejection['message']}",
                            "tool_calls": [],
                            "error_code": "repair_provider_error",
                        },
                        usage=repaired.raw.get("usage") or {},
                    )
                return Definitive({"content": repaired.content, "tool_calls": []}, usage=repaired.raw.get("usage") or {})
            return Definitive({"content": serializable_output(repaired), "tool_calls": []})
        if "shared_state" in request.metadata.get("vv_session", {}):
            context.shared_state.clear()
            context.shared_state.update(self.runtime.bind_state(request.metadata["vv_session"]["shared_state"], task))
        stream_observed = False
        try:
            callback = None
            stream = self.runtime.config.stream
            if stream is not None and plan._payload["purpose"] == "primary":

                def callback(payload):
                    nonlocal stream_observed
                    stream_observed = True
                    event = _project_provider_stream_payload(
                        payload | {"cycle": context.cycle_index},
                        run_id=plan.turn_id or "",
                        trace_id=plan.turn_id or "",
                        agent_name=self.runtime.agent.name,
                        session_id=self.sid,
                        parent_run_id=None,
                    )
                    if event is not None and not self.scope.token.cancelled:
                        object.__setattr__(event, "version", "v6")
                        with suppress(Exception):
                            stream(event)

            response = self.runtime.complete(
                request,
                plan.attempt or 1,
                callback,
                planned_request=plan._payload["request"],
                prepared_client=prepared_client,
            )
        except Exception as exc:
            if stream_observed or isinstance(exc, ModelStreamInterrupted) or self.scope.token.cancelled:
                raise
            if is_prompt_too_long_error(exc):
                return Definitive({"error_code": "prompt_too_long", "content": "", "tool_calls": []})
            order = plan._payload["request"].get("metadata", {}).get("vv_session", {}).get("endpoint_order", [])
            rejection = model_request_rejection(exc, endpoint_count=len(order))
            if rejection is not None:
                return Definitive(
                    {
                        "error_code": "model_request_rejected",
                        "content": f"HTTP {rejection.status_code}: {rejection.message}",
                        "tool_calls": [],
                        "rejection": rejection.to_dict(),
                    }
                )
            raise
        # A fully observed response-level conclusion is definitive even after deltas.
        # Exception-based request rejection above still yields to observed stream output.
        if response_policy:
            rejection = model_response_rejection(response.raw.get("completion_metadata", {}))
            if rejection is not None:
                return Definitive(
                    {
                        "error_code": "model_request_rejected",
                        "content": f"HTTP 200: {rejection['message']}",
                        "tool_calls": [],
                        "rejection": rejection,
                    },
                    usage=copy_json(response.raw.get("usage") or {}),
                )
        observed_usage = copy_json(response.raw.get("usage") or {})
        if plan._payload["purpose"] == "primary":
            response = self.runtime.hooks.apply_after_llm(
                task=task,
                cycle_index=context.cycle_index,
                messages=request.messages,
                tool_schemas=request.tools,
                response=response,
                shared_state=context.shared_state,
            )
        return Definitive(
            {
                "content": response.content,
                "tool_calls": [c.to_dict() for c in response.tool_calls],
                "raw": response.raw,
                "reasoning_content": response.raw.get("reasoning_content"),
            },
            usage=observed_usage,
        )

    outcome = _invoke(
        self.scope,
        model if provider is None else lambda: provider.submit(plan, context=context),
        self.runtime.model_timeout if kind == "model" else self.runtime.tool_timeout,
    )
    self.runtime.hook("after_external_call", plan)
    if provider is not None and plan._payload["provider_binding"] != "function":
        try:
            if isinstance(outcome, Definitive):
                self.check_provider_result(plan, outcome)
            elif isinstance(outcome, Accepted):
                validate_acceptance(plan, outcome)
        except ProviderResultInvalid:
            outcome = Unknown("provider_result_invalid")
    if isinstance(outcome, Definitive):
        if kind != "model":
            outcome = self.finalize_tool(plan, context, outcome)
            if plan._payload["provider_binding"] != "function":
                try:
                    self.check_provider_result(plan, outcome)
                except ProviderResultInvalid:
                    self.commit([self.unknown(plan, "provider_result_invalid")])
                    return
        outcome = replace(outcome, shared_state=self.runtime.durable_state(context.shared_state))
        self.commit(self.completed(plan, outcome))
    elif isinstance(outcome, Accepted):
        self.commit([self.accepted(plan, outcome)])
    else:
        self.commit([self.unknown(plan, outcome.reason)])


def finalize_tool(self: _Driver, plan: Record, context, outcome: Definitive) -> Definitive:
    call = ToolCall.from_dict(plan.payload["request"])
    result = self.runtime.hooks.apply_after_tool_call(
        task=self.task() if self.runtime.hooks.has_hooks() else self._task(),
        cycle_index=context.cycle_index,
        call=call,
        context=replace(context, tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments)),
        result=ToolExecutionResult.from_dict(outcome.result),
    )
    if not result.tool_call_id.strip():
        result.tool_call_id = call.id
    stop = apply_tool_use_behavior(task=self._task(), call=call, result=result)
    if stop is not None:
        result.metadata["completion_reason"] = stop.value
    return replace(outcome, result=result.to_dict(), shared_state=self.runtime.durable_state(context.shared_state))
