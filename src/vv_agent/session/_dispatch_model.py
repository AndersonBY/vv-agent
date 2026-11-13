"""The existing model invocation, stream bridge and error observation leaf."""

from __future__ import annotations

from contextlib import suppress
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
from vv_agent.types import LLMResponse

from .output import serializable_output
from .providers import Definitive, Outcome
from .records import Record, copy_json
from .runtime import request_from_dict

if TYPE_CHECKING:
    from .kernel import _Driver


def invoke_model(self: _Driver, plan: Record, task, context, started: Record, prepared_client) -> Outcome:
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
