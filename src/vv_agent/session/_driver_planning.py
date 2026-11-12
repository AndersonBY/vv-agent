"""Build operation and model plans using the driver's retained state."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vv_agent.memory.microcompact import is_microcompacted_tool_content
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.runtime.lifecycle import read_after_cycle_disallowed_tools
from vv_agent.types import Message

from .records import Record, _RetainedTools, copy_json, digest, make_record

if TYPE_CHECKING:
    from .kernel import _Driver


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
    if kind == "model":
        from vv_agent.llm.vv_llm_client import VvLlmClient

        client, resolved = self.runtime.resolve_model(request) if purpose == "primary" else self.runtime.model_route(purpose)
        session = request["metadata"].get("vv_session", {})
        if purpose != "output_repair":
            session = session | {"model_binding": self.runtime.model_binding(client, resolved)}
        if isinstance(client, VvLlmClient):
            preferred = self.state.preferred_endpoint_id
            if preferred not in {t.endpoint_id for t in client.endpoint_targets}:
                preferred = None
            order = [t.endpoint_id for t in client.ordered_targets(preferred)]
            if order:
                session = session | {"endpoint_order": order}
        request = request | {"metadata": request["metadata"] | {"vv_session": session}}
    if kind == "model" and self.state.active_turn_id is not None:
        source = self.state.turns[self.state.active_turn_id].start
        tools = request.get("tools")
        frozen = source._payload["definition"].get("tools")
        if (
            isinstance(tools, list)
            and isinstance(frozen, list)
            and len(tools) == len(frozen)
            and all(a is b for a, b in zip(tools, frozen, strict=True))
        ):
            request = request | {"tools": _RetainedTools(source)}
    if kind == "model" and isinstance(request.get("messages"), list):
        request = request | {"messages": self.fold._messages.retain_array(request["messages"])}
    binding = "model" if kind == "model" else request["name"] if request["name"] in self.runtime.providers else "function"
    return make_record(
        "op_planned",
        session_id=self.sid,
        turn_id=self.state.active_turn_id,
        operation_id=oid,
        attempt=number,
        payload={
            "op_kind": kind,
            "purpose": purpose if kind == "model" else None,
            "request": request,
            "request_digest": digest(request),
            "context_version": str(self.records[-1].seq),
            "dependencies": dependencies or [],
            "tool": capability,
            "idempotency_key": None if capability and capability.get("idempotency") == "unsupported" else f"{self.sid}/{oid}",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": binding,
            "consumed_unknowns": [
                {"operation_id": key, "attempt": n}
                for key, op in self.state.operations.items()
                for n, a in op.attempts.items()
                if kind == "model"
                and request["metadata"].get("mode") != "child_summary"
                and op.turn_id == self.state.active_turn_id
                and a.state == "unknown"
                and op.kind != "model"
            ],
        },
    )


def model_plan(self: _Driver) -> Record | None:
    messages = self.transcript()
    from vv_agent.kernel._context_messages import retained_start

    from .context import protected_start

    protection = protected_start(self.records, self.state)
    protected_source = [Message.from_dict(copy_json(m.to_dict())) for m in messages] if protection is not None else []
    tid = self.state.active_turn_id
    assert tid is not None
    start = self.state.turns[tid].start
    task = start.task() if self.runtime.hooks.has_hooks() else start._task()
    definition = start._payload["definition"]
    cycle = 1 + sum(
        op.turn_id == tid
        and op.kind == "model"
        and op.attempts[1].plan._payload["purpose"] == "primary"
        and not (
            op.selected_attempt
            and (receipt := op.attempts[op.selected_attempt].result)
            and receipt._payload["result"].get("error_code") == "prompt_too_long"
        )
        for op in self.state.operations.values()
    )
    shared = self.shared_state()
    if self.runtime.config.interruption_messages:
        messages.extend(self.runtime.config.interruption_messages())
    if self.runtime.config.before_cycle_messages:
        messages.extend(self.runtime.config.before_cycle_messages(cycle, list(messages), shared))
    messages, schemas = self.runtime.hooks.apply_before_llm(
        task=task,
        cycle_index=cycle,
        messages=messages,
        tool_schemas=copy_json(definition["tools"]) if self.runtime.hooks.has_hooks() else definition["tools"],
        shared_state=shared,
    )
    retained_start(protected_source, messages, protection)
    denied = read_after_cycle_disallowed_tools(shared)
    if denied:
        schemas = [schema for schema in schemas if schema["function"]["name"] not in denied]
    if not any(schema["function"]["name"] == "read_file" for schema in schemas) and any(
        (message.role == "tool" and is_microcompacted_tool_content(message.content))
        or any(message.metadata.get("_vv_agent_compaction", {}).get(key) for key in ("artifacts", "cursors"))
        for message in messages
    ):
        self.close("failed", "microcompaction_recovery_unavailable")
        return None
    effective = self.effective_model()
    request = {
        "model": effective["model"],
        "route": effective["route"],
        "messages": [m.to_dict() for m in messages],
        "tools": schemas,
        "metadata": dict(task.metadata)
        | {
            "vv_session": {
                "shared_state": self.runtime.durable_state(shared),
                "cycle_index": cycle,
                "model_binding": effective["binding"],
            }
        },
        "prompt_bundle": task.prompt_bundle.to_dict(),
        "model_settings": replace(
            ModelSettings.from_dict(effective.get("model_settings", {})),
            retry=RetrySettings(max_attempts=1, backoff_seconds=0),
        ).to_dict(),
    }
    number = (
        sum(
            op.kind == "model" and op.turn_id == tid and op.attempts[1].plan._payload["purpose"] == "primary"
            for op in self.state.operations.values()
        )
        + 1
    )
    return self.plan(f"{tid}/model/primary/{number}", request, "model")
