"""Author v24 vectors from the private kernel; never modify the vendored snapshot."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import subprocess
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from vv_agent import Agent, RunConfig, handoff, input_guardrail
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.app_server.schema import export_schema_bundles
from vv_agent.budget import RunBudgetLimits, UnavailableMetricPolicy
from vv_agent.events import event_from_dict
from vv_agent.guardrails import GuardrailResult
from vv_agent.interaction import HostInteractionOutcome, HostInteractionRequest
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.memory import MemoryManager
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.model import ScriptedModelProvider
from vv_agent.output_validation import OutputValidationResult
from vv_agent.runtime.hooks import BaseRuntimeHook
from vv_agent.session.children import child_delivery, child_handles
from vv_agent.session.context import project_context
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.kernel import drive
from vv_agent.session.projection import project_records
from vv_agent.session.providers import Accepted, Unknown
from vv_agent.session.records import (
    APP_SERVER_ATTRIBUTES,
    BOUNDARY_DATA,
    CHILD_ADMISSION,
    INPUT_PAYLOADS,
    INPUT_SCHEMA,
    PAYLOADS,
    RECORD_SCHEMA,
    REQUEST_SESSION_METADATA,
    SEED,
    TASK_SESSION_METADATA,
    InboxItem,
    Record,
    RecordError,
    make_record,
)
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.result import project_result
from vv_agent.session.store import Conflict, LeaseLost
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.tracing import project_spans
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.function import function_tool
from vv_agent.tools.outputs import ToolOutputText
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import LLMResponse, Message, SubAgentConfig, ToolCall, ToolDirective, ToolExecutionResult
from vv_agent.workspace.memory import MemoryWorkspaceBackend

# ECMAScript supplies RFC8785's number/string encoding and UTF-16 key ordering;
# this path neither imports nor calls the producer's canonical JSON implementation.
JCS_JS = r"""
const fs = require('fs');
function jcs(x) {
  if (x === null || typeof x !== 'object') return JSON.stringify(x);
  if (Array.isArray(x)) return '[' + x.map(jcs).join(',') + ']';
  return '{' + Object.keys(x).sort().map(k => JSON.stringify(k) + ':' + jcs(x[k])).join(',') + '}';
}
for (const x of JSON.parse(fs.readFileSync(0, 'utf8'))) {
  process.stdout.write(Buffer.from(jcs(x), 'utf8').toString('base64') + '\n');
}
"""
NOW = 1_000_000
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
APPROVED_EFFECTS = []
SUMMARY = json.dumps({"original_user_messages": ["request"], "current_work_state": "Continue the task"})


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(NOW / 1000, tz=tz or UTC)


def independent_bytes(values: list[Any]) -> list[bytes]:
    result = subprocess.run(
        ["node", "-e", JCS_JS], input=json.dumps(values, ensure_ascii=True), text=True, capture_output=True, check=True
    )
    return [base64.b64decode(line, validate=True) for line in result.stdout.splitlines()]


def expected_id(value: dict[str, Any]) -> str:
    kind, sid, tid, oid, number, p = (value[k] for k in ("kind", "session_id", "turn_id", "operation_id", "attempt", "payload"))
    if kind == "session_created":
        return f"session/{sid}/created"
    if kind in {"turn_started", "turn_ended"}:
        return f"turn/{tid}/{kind[5:]}"
    if kind == "turn_parked":
        return f"turn/{tid}/wait/{p['interaction_id']}"
    if kind == "input_applied":
        return f"input/{p['input']['input_id']}/applied"
    if kind.startswith("op_"):
        suffix = "result" if kind == "op_completed" else kind[3:]
        if kind == "op_parked":
            suffix += "/" + p["phase"]
        return f"op/{oid}/{number}/{suffix}"
    if kind == "boundary_recorded":
        return f"turn/{tid}/boundary/{p['stage']}/{p['boundary_id']}"
    if kind == "context_compacted":
        suffix = "" if p["summary_operation_id"] is None else "/" + p["summary_operation_id"]
        return f"compact/{p['source_digest']}/{p['mode']}{suffix}"
    assert kind == "usage_observed"
    return f"usage/{p['meter_id']}/{p['observation']}"


def facts(raw: bytes, *, record_id: str | None = None) -> dict[str, Any]:
    return {"bytes_base64": base64.b64encode(raw).decode(), "sha256": sha256(raw).hexdigest(), "record_id": record_id}


class Cut(BaseException):
    pass


class Hooks(BaseRuntimeHook):
    def before_tool_call(self, event):
        event.context.shared_state["prepared"] = True
        return None


class AfterCycle:
    def after_cycle(self, snapshot):
        return None


class PendingApproval:
    def should_request(self, request):
        return True

    def decide(self, request):
        return None


class Provider:
    """Independent trusted provider truth with one stable accepted job."""

    def __init__(self, *, unknown=False, usage=None):
        self.unknown = unknown
        self.handle = None
        self.result = None
        self.effects = 0
        self.usage = usage or {}

    def submit(self, plan, *, context):
        self.effects += 1
        self.result = ToolExecutionResult(tool_call_id=plan.payload["request"]["id"], content="provider done").to_dict()
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "job",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "trusted/job",
            "query_ref": "query/job",
            "cancel_ref": "cancel/job",
        }
        return Unknown("response lost") if self.unknown else Accepted(self.handle)

    def query(self, handle):
        assert handle == self.handle
        return Accepted(handle)

    def cancel(self, handle):
        return Unknown("stop unconfirmed")

    def authenticate(self, item, plan):
        if self.handle is None:
            return False
        if item.kind == "provider_evidence":
            return item.payload["handle"] == self.handle
        return (
            item.payload["result"] == self.result
            and item.payload["usage"] == self.usage
            and item.payload["evidence"] == ["trusted/job"]
        )

    def callback(self, kind, tid, input_id):
        h = self.handle
        assert h is not None
        p = {k: h[k] for k in ("operation_id", "attempt", "request_digest")}
        if kind == "provider_evidence":
            p["handle"] = h
        else:
            p.update(provider_binding=h["provider"], result=self.result, usage=self.usage, evidence=[h["evidence"]])
        return InboxItem(input_id, kind, p, tid, 0)


@function_tool
def echo(text: str) -> str:
    return text


@function_tool(needs_approval=True)
def approved() -> str:
    APPROVED_EFFECTS.append("approved")
    return "approved effect"


@function_tool
def wait_turn() -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="", content="Choose", directive=ToolDirective.WAIT_USER, metadata={"question": "Choose"}
    )


class Fixtures:
    def __init__(self):
        self.kernel = SessionDriver()
        self.clock = NOW
        self.kernel.store.connection.create_function("session_now_ms", 0, lambda: self.clock)
        self.runtimes: dict[str, Any] = {}
        self.inbox: dict[tuple[str, str], InboxItem] = {}
        self.semantics = []
        self.recovery = []
        self.invalid = []

    @staticmethod
    def registry(names=()):
        registry = ToolRegistry()
        if names:
            builtins = build_default_registry()
            for name in names:
                if builtins.has_executor(name):
                    registry.register_executor(builtins.get_executor(name))
        return registry

    @classmethod
    def config(cls, config=None, steps=()):
        config = config or RunConfig()
        names = tuple(sorted({call.name for step in steps if isinstance(step, LLMResponse) for call in step.tool_calls}))
        return replace(config, tool_registry_factory=config.tool_registry_factory or partial(cls.registry, names))

    def push(self, sid, item):
        self.inbox[(sid, item.input_id)] = item
        return self.kernel.push(sid, item)

    def admit(self, sid, steps, *, agent=None, config=None, attributes=None, memory=None, content="go", **kwargs):
        agent = agent or Agent("fixture", "Be precise.", model="m", tools=[echo])
        provider = ScriptedModelProvider("scripted", "m", ScriptedLLM(steps), context_length=None, max_output_tokens=None)
        config = replace(self.config(config, steps), model_provider=provider, workspace="/fixture")
        rt = self.kernel.runtime(agent, config)
        rt.__dict__.update(kwargs)
        if memory is not None:
            rt.memory_manager = memory
        self.runtimes[sid] = rt
        self.kernel.create(sid, "/fixture", attributes)
        self.push(sid, InboxItem("initial", "user", {"content": content}))
        return rt

    def run(self, sid):
        drive(self.kernel.store, sid, runtime=self.runtimes[sid], _one_turn=True)

    def state(self, sid):
        return self.kernel.store.read_state(sid)[0]

    def result(self, sid):
        state = self.state(sid)
        return project_result(self.kernel.store, sid, next(reversed(state.turns)), runtime=self.runtimes[sid])

    def drain_children(self, sid, rt):
        drive(self.kernel.store, sid, runtime=rt, _one_turn=True)
        for stored in self.kernel.store.read_state(sid)[1]:
            r = stored.record
            if r.kind == "op_parked" and r.payload["handle"]["kind"] == "child":
                for handle in child_handles(r.payload["handle"]):
                    child = handle["session_id"]
                    child_rt = rt.child_runtime(self.kernel.store, child)
                    self.runtimes[child] = child_rt
                    self.drain_children(child, child_rt)
                    with self.kernel.store.atomic() as tx:
                        child_delivery(self.kernel.store, tx, child)
        drive(self.kernel.store, sid, runtime=rt, _one_turn=True)

    def produce(self):
        self.provider_usage()
        opaque = {"😀": 1e-7, "\ue000": -0.0, "null": None, "bool": True, "integer": 1, "float": 1.5}
        self.admit(
            "seed",
            [LLMResponse("done", raw={"usage": USAGE})],
            attributes={"seed": {"messages": [Message("user", "history").to_dict()], "shared_state": opaque}, "opaque": opaque},
        )
        self.run("seed")
        assert self.result("seed").raw_result.shared_state == opaque
        assert [m.content for m in self.result("seed").raw_result.messages if m.role == "user"] == ["history", "go"]
        self.semantics.append({"case": "creation_seed", "session_id": "seed", "expected_shared_state": opaque})

        self.admit(
            "tools",
            [LLMResponse("", [ToolCall("echo", "echo", {"text": "ok"})]), LLMResponse("done")],
            config=RunConfig(
                hooks=[Hooks()], after_cycle_hooks=[AfterCycle()], budget_limits=RunBudgetLimits(max_total_tokens=100)
            ),
        )
        self.run("tools")

        @input_guardrail
        def block(context, text):
            return GuardrailResult.block("blocked")

        self.admit("blocked", [], agent=Agent("blocked", "Be precise.", input_guardrails=[block]))
        self.run("blocked")
        self.admit(
            "multimodal",
            [LLMResponse("done")],
            content={"text": "go", "messages": [Message("user", "multimodal input").to_dict()]},
        )
        self.run("multimodal")
        self.admit("bindings", [LLMResponse("done")], host_bindings={"host": object()})
        self.run("bindings")
        target = Agent("target", "Finish.", model="m")
        self.admit(
            "handoff",
            [LLMResponse("", [ToolCall("transfer", "transfer_to_target", {})]), LLMResponse("transferred")],
            agent=Agent("source", "Delegate.", model="m", handoffs=[handoff(agent=target)]),
        )
        self.drain_children("handoff", self.runtimes["handoff"])
        self.admit(
            "budget",
            [LLMResponse("", [ToolCall("never", "echo", {"text": "never"})], raw={"usage": USAGE})],
            config=RunConfig(budget_limits=RunBudgetLimits(max_total_tokens=1)),
        )
        self.run("budget")
        assert self.result("budget").raw_result.budget_exhaustion is not None

        for name, tool_name in (("user_wait", "ask_user"), ("turn_wait", "wait_turn")):
            calls = [ToolCall("ask", tool_name, {"question": "Choose"} if tool_name == "ask_user" else {})]
            self.admit(
                name, [LLMResponse("", calls), LLMResponse("answered")], agent=Agent("fixture", "Be precise.", tools=[wait_turn])
            )
            self.run(name)
            wait = self.kernel.waits(name, self.runtimes[name])[0][2]
            self.kernel.answer(name, self.runtimes[name], "blue", "reply")
            self.run(name)
            assert self.result(name).final_output == "answered"
            self.semantics.append({"case": name, "interaction_id": wait["interaction_id"], "same_turn": True})

        for sid, decision in (("approval", "allow_session"), ("deadline", "timeout")):
            self.admit(
                sid,
                [LLMResponse("", [ToolCall("approved", "approved", {})]), LLMResponse("done")],
                agent=Agent("fixture", "Be precise.", tools=[approved]),
                config=RunConfig(hooks=[Hooks()], approval_provider=PendingApproval(), approval_timeout_seconds=1),
            )
            self.run(sid)
            wait = self.kernel.waits(sid, self.runtimes[sid])[0][2]
            if decision == "timeout":
                self.clock += 1001
            else:
                self.kernel.approve(sid, self.runtimes[sid], wait["request_id"], decision, "answer")
            self.run(sid)
            assert self.state(sid).active_turn_id is None

        for sid, unknown in (("accepted", False), ("unknown", True)):
            provider = Provider(unknown=unknown)
            self.admit(
                sid,
                [LLMResponse("", [ToolCall("effect", "echo", {"text": "effect"})]), LLMResponse("done")],
                providers={"echo": provider},
            )
            self.run(sid)
            tid = f"{sid}/turn/initial"
            self.push(sid, provider.callback("provider_evidence", tid, "evidence"))
            self.push(sid, provider.callback("provider_result", tid, "result"))
            self.run(sid)
            assert provider.effects == 1
            self.semantics.append({"case": sid, "effects": 1, "terminal": self.state(sid).active_turn_id is None})

        self.admit(
            "children",
            [
                LLMResponse(
                    "",
                    [
                        ToolCall(
                            "children",
                            "create_sub_task",
                            {"agent_id": "worker", "tasks": [{"task_description": "first"}, {"task_description": "second"}]},
                        )
                    ],
                ),
                LLMResponse("first done"),
                LLMResponse("second done"),
                LLMResponse("parent done"),
            ],
            agent=Agent("parent", "Be precise.", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}),
        )
        self.drain_children("children", self.runtimes["children"])
        assert self.result("children").final_output == "parent done"

        history = [Message("user", "request"), Message("assistant", "old " * 1200)]
        for sid, summary in (("summary", SUMMARY), ("rejected_summary", "bad")):
            self.admit(
                sid,
                [LLMResponse(summary), LLMResponse("done")],
                config=RunConfig(initial_messages=history),
                memory=MemoryManager(compact_threshold=1000, keep_recent_messages=1),
            )
            self.run(sid)
        backend = MemoryWorkspaceBackend()
        tool_history = [
            Message("user", "request"),
            Message(
                "assistant",
                "",
                tool_calls=[
                    {"id": "old", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"old.txt"}'}}
                ],
            ),
            Message("tool", "old " * 600, tool_call_id="old"),
            Message("assistant", "recent reply"),
        ]
        manager = MemoryManager(
            compact_threshold=500, keep_recent_messages=1, microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1)
        )
        self.admit(
            "micro",
            [LLMResponse("done")],
            config=RunConfig(
                initial_messages=tool_history,
                workspace_backend=backend,
                tool_registry_factory=partial(self.registry, ("read_file",)),
            ),
            memory=manager,
        )
        self.run("micro")
        self.compaction_cases(history, backend)

        self.admit(
            "memory",
            [LLMResponse('[{"category":"user_intent","content":"retain goal","importance":8}]'), LLMResponse("done")],
            config=RunConfig(
                initial_messages=history,
                session_memory_enabled=True,
                workspace_backend=backend,
                metadata={
                    "session_id": "memory",
                    "session_memory_min_tokens": 1,
                    "session_memory_min_text_messages": 1,
                    "session_memory_storage_dir": "",
                },
            ),
        )
        # No host file projection: the memory callback still logs a real structured receipt.
        self.runtimes["memory"].config = replace(self.runtimes["memory"].config, workspace=None)
        self.run("memory")

        agent = Agent(
            "fixture",
            "Be precise.",
            output_type=list,
            output_validation_enabled=True,
            output_validator=lambda *_: OutputValidationResult.accept(),
            output_repair=lambda _: LLMResponse("[1,2]", raw={"usage": USAGE}),
        )
        self.admit("repair", [LLMResponse("bad", raw={"usage": USAGE})], agent=agent)
        self.run("repair")
        assert self.result("repair").token_usage.model_calls[-1].operation.value == "output_repair"

        self.admit("controls", [LLMResponse("first"), LLMResponse("followed")])
        rt = self.runtimes["controls"]

        def queue(point, record):
            if point == "after_commit" and record.kind == "op_planned":
                rt.hook = lambda *_: None
                self.push("controls", InboxItem("steer", "steer", {"content": "steer"}, "controls/turn/initial", 0))
                self.push("controls", InboxItem("follow", "follow_up", {"content": "follow"}))

        rt.hook = queue
        self.run("controls")
        self.run("controls")
        self.kernel.control("controls", "archive", "archive", runtime=rt)
        self.kernel.control("seed", "close", "close", runtime=self.runtimes["seed"])
        from types import SimpleNamespace

        from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

        rt = self.admit("endpoint", [])
        rt.llm = VvLlmClient(
            [EndpointTarget("endpoint", "unused", "https://example.invalid")], backend="openai", randomize_endpoints=False
        )
        complete = rt.complete

        def dispatched(request, attempt, stream_callback=None):
            self.semantics.append({"case": "dispatch_request", "metadata": deepcopy(request.metadata)})
            return complete(request, attempt, stream_callback)

        rt.complete = dispatched
        with (
            patch("vv_agent.llm.vv_llm_client.create_chat_client", return_value=SimpleNamespace()),
            patch.object(VvLlmClient, "_non_stream_completion", return_value=LLMResponse("done", raw={"usage": USAGE})),
            patch.object(VvLlmClient, "_should_use_stream", return_value=False),
        ):
            self.run("endpoint")
        self.recovery_cuts()
        self.store_semantics()
        self.admission_negatives()

    def compaction_cases(self, history, backend):
        @function_tool
        def more() -> ToolOutputText:
            return ToolOutputText("new " * 1200)

        rt = self.admit(
            "second_summary",
            [LLMResponse(SUMMARY), LLMResponse("", [ToolCall("new", "more", {})]), LLMResponse(SUMMARY), LLMResponse("done")],
            agent=Agent("fixture", "Be precise.", tools=[more]),
            config=RunConfig(initial_messages=history),
            memory=MemoryManager(
                compact_threshold=1000,
                keep_recent_messages=1,
                microcompaction_policy=MicrocompactionPolicy(min_result_chars=999999),
            ),
        )

        def steer(point, record):
            if point == "after_commit" and record.kind == "op_completed" and "/tool/" in (record.operation_id or ""):
                self.push("second_summary", InboxItem("continue", "steer", {"content": "continue"}, record.turn_id))

        rt.hook = steer
        self.run("second_summary")
        assert len(self.state("second_summary").compactions) == 2

        def too_long(_):
            raise RuntimeError("maximum context length exceeded")

        self.admit(
            "emergency",
            [too_long, LLMResponse("bad"), too_long, LLMResponse(SUMMARY), LLMResponse("done")],
            config=RunConfig(initial_messages=history),
            memory=MemoryManager(compact_threshold=1_000_000, keep_recent_messages=1),
        )
        self.run("emergency")
        assert self.state("emergency").compactions[-1].payload["mode"] == "emergency"
        for sid, summary in (("summary_receipt", SUMMARY), ("rejected_receipt", "bad")):
            rt = self.admit(
                sid,
                [LLMResponse(summary), LLMResponse("done")],
                config=RunConfig(initial_messages=history),
                memory=MemoryManager(compact_threshold=1000, keep_recent_messages=1),
            )

            def cut(point, record):
                if point == "after_commit" and record.kind == "op_completed":
                    raise Cut

            rt.hook = cut
            try:
                self.run(sid)
            except Cut:
                pass
            else:
                raise AssertionError("missing summary receipt cut")
            receipt = next(
                r.record.record_id for r in reversed(self.kernel.store.read_state(sid)[1]) if r.record.kind == "op_completed"
            )
            rt.hook = lambda *_: None
            self.kernel.store._fold_cache = None
            self.clock += 10_000
            self.run(sid)
            plans = [op for op in self.state(sid).operations.values() if op.attempts[1].plan.payload["purpose"] == "compaction"]
            assert len(plans) == 1 and not rt.llm.steps
            self.recovery.append(
                {
                    "case": sid,
                    "session_id": sid,
                    "receipt_id": receipt,
                    "summary_dispatches": 1,
                    "lost_wall_ms": 10_000,
                    "expected_terminal": True,
                }
            )
        # Verify every retained micro artifact through the actual workspace backend.
        for compact in self.state("micro").compactions:
            for message in compact.payload["replacement"]:
                if ref := message.get("artifact_ref"):
                    raw = backend.read_bytes(ref["path"])
                    assert sha256(raw).hexdigest() == ref["sha256"]
                    self.semantics.append({"case": "micro_artifact", "path": ref["path"], "artifact": facts(raw)})

    def admission_negatives(self):
        def reject(sid, item, label):
            state = self.state(sid)
            selected = {oid: op.selected_attempt for oid, op in state.operations.items()}
            self.push(sid, item)
            self.run(sid)
            after = self.state(sid)
            audit = after.applied_inputs[item.input_id]
            assert audit.payload["disposition"] == "rejected", (label, audit.payload)
            assert {oid: op.selected_attempt for oid, op in after.operations.items()} == selected
            self.invalid.append(
                {
                    "rejection_class": label,
                    "layer": "admission",
                    "session_id": sid,
                    "audit_record_id": audit.record_id,
                    "reason": audit.payload["reason"],
                    **facts(item.encode()),
                }
            )

        original = self.inbox[("accepted", "result")]
        reject(
            "accepted",
            replace(original, input_id="forged-provider", payload=original.payload | {"evidence": ["forged"]}),
            "provider_authentication",
        )
        reject(
            "accepted",
            replace(original, input_id="binding-drift", payload=original.payload | {"provider_binding": "other"}),
            "provider_binding_drift",
        )
        reject("accepted", replace(original, input_id="stale-generation", generation=99), "stale_generation")
        child = next(
            InboxItem(**r.payload["input"])
            for r in self.state("children").applied_inputs.values()
            if r.payload["input"]["kind"] == "child_result"
        )
        reject(
            "children",
            replace(child, input_id="forged-child", payload=child.payload | {"terminal_digest": "0" * 64}),
            "child_authentication",
        )
        self.admit(
            "authorization",
            [LLMResponse("", [ToolCall("approved", "approved", {})])],
            agent=Agent("fixture", "Be precise.", tools=[approved]),
            config=RunConfig(approval_provider=PendingApproval(), approval_timeout_seconds=60),
        )
        self.run("authorization")
        wait = self.kernel.waits("authorization", self.runtimes["authorization"])[0][2]
        answer = InboxItem(
            "wrong-scope",
            "approval_answer",
            {
                "operation_id": wait["operation_id"],
                "attempt": wait["attempt"],
                "request_id": wait["request_id"],
                "request_digest": wait["request_digest"],
                "scope": ["forged"],
                "decision": "approve",
            },
            "authorization/turn/initial",
            0,
        )
        reject("authorization", answer, "authorization")

    def recovery_cuts(self):
        for policy in UnavailableMetricPolicy:
            sid = "lost_wall_" + policy.value
            rt = self.admit(
                sid,
                [LLMResponse("done", raw={"usage": USAGE})],
                config=RunConfig(budget_limits=RunBudgetLimits(max_wall_time_ms=100000, unavailable_metric_policy=policy)),
            )

            def cut(point, record):
                if point == "after_commit" and record.kind == "op_started":
                    raise Cut

            rt.hook = cut
            try:
                self.run(sid)
            except Cut:
                pass
            else:
                raise AssertionError("missing wall interval cut")
            prefix = [r.record.record_id for r in self.kernel.store.read_state(sid)[1]]
            rt.hook = lambda *_: None
            self.kernel.store._fold_cache = None
            self.clock += 10_000
            self.run(sid)
            result = self.result(sid)
            assert result.budget_usage.unavailable_dimensions
            unknown = next(op.attempts[1].unknown for op in self.state(sid).operations.values() if op.attempts[1].unknown)
            assert unknown.payload["observation"]["active_interval_missing"]
            assert result.status.value == ("failed" if policy == UnavailableMetricPolicy.STOP else "completed")
            self.recovery.append(
                {
                    "case": sid,
                    "session_id": sid,
                    "prefix_ids": prefix,
                    "lost_wall_ms": 10000,
                    "expected_terminal": True,
                    "budget_usage": result.budget_usage.to_dict(),
                    "status": result.status.value,
                }
            )
        for kind in ("op_planned", "op_prepared", "op_started", "op_completed", "boundary_recorded"):
            sid = "cut_" + kind
            rt = self.admit(
                sid,
                [
                    LLMResponse("", [ToolCall("a", "echo", {"text": "a"}), ToolCall("b", "echo", {"text": "b"})]),
                    LLMResponse("done"),
                ],
                config=RunConfig(hooks=[Hooks()]),
            )

            def cut(point, record, kind=kind):
                if point == "after_commit" and record.kind == kind:
                    raise Cut

            rt.hook = cut
            try:
                self.run(sid)
            except Cut:
                pass
            else:
                raise AssertionError(f"missing cut: {kind}")
            prefix = [r.record.record_id for r in self.kernel.store.read_state(sid)[1]]
            rt.hook = lambda *_: None
            self.kernel.store._fold_cache = None
            self.run(sid)
            assert self.state(sid).active_turn_id is None
            self.recovery.append(
                {
                    "case": kind,
                    "session_id": sid,
                    "cut_after": prefix[-1],
                    "prefix_ids": prefix,
                    "expected_terminal": True,
                    "remaining_model_steps": len(rt.llm.steps),
                }
            )
        for kind in ("accepted", "parked"):
            sid = "cut_" + kind
            provider = Provider()
            rt = self.admit(
                sid,
                [LLMResponse("", [ToolCall("effect", "echo", {"text": "effect"})]), LLMResponse("done")],
                providers={"echo": provider},
            )

            def cut(point, record, kind=kind):
                if point == ("before_commit" if kind == "accepted" else "after_commit") and record.kind == "op_parked":
                    raise Cut

            rt.hook = cut
            try:
                self.run(sid)
            except Cut:
                pass
            else:
                raise AssertionError("missing provider cut")
            prefix = [r.record.record_id for r in self.kernel.store.read_state(sid)[1]]
            rt.hook = lambda *_: None
            self.kernel.store._fold_cache = None
            self.push(sid, provider.callback("provider_evidence", f"{sid}/turn/initial", "evidence"))
            self.push(sid, provider.callback("provider_result", f"{sid}/turn/initial", "result"))
            self.run(sid)
            assert provider.effects == 1 and self.state(sid).active_turn_id is None
            self.recovery.append({"case": kind, "session_id": sid, "prefix_ids": prefix, "effects": 1, "expected_terminal": True})
        sid = "cut_tool_a"
        effects = []

        @function_tool
        def counted(text: str) -> str:
            effects.append(text)
            return text

        rt = self.admit(
            sid,
            [
                LLMResponse("", [ToolCall("a", "counted", {"text": "a"}), ToolCall("b", "counted", {"text": "b"})]),
                LLMResponse("done"),
            ],
            agent=Agent("fixture", "Be precise.", tools=[counted]),
        )

        def cut(point, record):
            if point == "after_commit" and record.kind == "op_completed" and (record.operation_id or "").endswith("/tool/0"):
                raise Cut

        rt.hook = cut
        try:
            self.run(sid)
        except Cut:
            pass
        else:
            raise AssertionError("missing tool A cut")
        assert effects == ["a"]
        prefix = [r.record.record_id for r in self.kernel.store.read_state(sid)[1]]
        rt.hook = lambda *_: None
        self.kernel.store._fold_cache = None
        self.run(sid)
        assert effects == ["a", "b"]
        self.recovery.append(
            {
                "case": "tool_a_done_b_pending",
                "session_id": sid,
                "prefix_ids": prefix,
                "effects": effects,
                "expected_terminal": True,
            }
        )
        for point in ("before_commit", "after_commit"):
            sid = "boundary_" + point

            class Callback:
                calls = 0

                def after_cycle(self, snapshot):
                    self.calls += 1
                    return None

            callback = Callback()
            rt = self.admit(
                sid,
                [LLMResponse("", [ToolCall("echo", "echo", {"text": "ok"})]), LLMResponse("done")],
                config=RunConfig(after_cycle_hooks=[callback]),
            )

            def cut(actual, record, point=point):
                if actual == point and record.kind == "boundary_recorded" and record.payload["stage"] == "after_cycle":
                    raise Cut

            rt.hook = cut
            try:
                self.run(sid)
            except Cut:
                pass
            else:
                raise AssertionError("missing callback cut")
            prefix = [r.record.record_id for r in self.kernel.store.read_state(sid)[1]]
            rt.hook = lambda *_: None
            self.kernel.store._fold_cache = None
            self.run(sid)
            assert callback.calls == (3 if point == "before_commit" else 2), (point, callback.calls)
            self.recovery.append(
                {
                    "case": sid,
                    "session_id": sid,
                    "prefix_ids": prefix,
                    "callback_calls": callback.calls,
                    "expected_terminal": True,
                }
            )

    def store_semantics(self):
        store = self.kernel.store
        original = InboxItem("replay", "follow_up", {"content": "same"})
        first = self.push("tools", original)
        head = store.read("tools").head_seq
        repeated = self.push("tools", original)
        assert repeated.replayed and repeated.input_seq == first.input_seq
        try:
            self.kernel.push("tools", replace(original, payload={"content": "different"}))
        except Conflict:
            pass
        else:
            raise AssertionError("input conflict accepted")
        assert store.read("tools").head_seq == head
        lease = store.acquire("tools", owner="owner", ttl_ms=100)
        assert lease is not None
        self.clock += 101
        replacement = store.acquire("tools", owner="replacement", ttl_ms=100)
        assert replacement is not None
        try:
            with store.atomic() as tx:
                tx.append("tools", lease=lease, expected_seq=head, commit_id="stale", records=())
        except LeaseLost:
            pass
        else:
            raise AssertionError("stale lease accepted")
        store.release(replacement)
        bridge = SessionRunEventStore(store, "tools")
        delivered = []
        try:
            with store.atomic() as tx:
                batch, events = bridge.batch(tx)
                delivered.extend(e.event_id for e in events)
                tx.ack(batch)
                raise Cut
        except Cut:
            pass
        redelivered = []
        bridge.consume(lambda event: redelivered.append(event.event_id))
        assert delivered == redelivered
        with store.atomic() as tx:
            batch = tx.consumer_batch("seed", "traces", limit=1)
            assert batch is not None
            # A one-record page still returns the complete transaction.
            ids = [r.record.record_id for r in batch.records]
            assert all(r.commit_id == batch.records[0].commit_id for r in batch.records)
            tx.ack(batch)
        from vv_agent.session.tracing import deliver_spans

        class LostTrace:
            def on_span_start(self, span):
                raise RuntimeError("sink lost")

            def on_span_end(self, span):
                raise RuntimeError("sink lost")

        deliver_spans(store, "seed", [LostTrace()])
        assert deliver_spans(store, "seed", [LostTrace()]) == 0
        self.admit("prefix", [LLMResponse("first"), LLMResponse("second")])
        self.run("prefix")
        self.push("prefix", InboxItem("next", "follow_up", {"content": "next"}))
        self.run("prefix")
        first = self.result("prefix")
        state = self.state("prefix")
        first_tid = next(iter(state.turns))
        isolated = project_result(store, "prefix", first_tid)
        assert isolated.final_output == "first" and first.final_output == "second"
        self.semantics.append(
            {
                "case": "whole_commit_ack_first_trace_terminal_prefix",
                "commit_record_ids": ids,
                "trace_redelivery": 0,
                "first_output": isolated.final_output,
                "latest_output": first.final_output,
            }
        )
        self.semantics.append(
            {
                "case": "replay_conflict_fence_ack_rollback",
                "head": head,
                "new_log_writes": 0,
                "same_input_receipt": True,
                "stale_writer_rejected": True,
                "redelivered_ids": redelivered,
            }
        )

    def app_server(self):
        transport = ChannelTransport(connection_id="owner")
        provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("app done", raw={"usage": USAGE})])
        server = AppServer(
            transport=transport,
            store=self.kernel.store,
            host=DefaultAppServerHost(
                agent=Agent("app", "Be precise.", model="m"),
                run_config=RunConfig(model_provider=provider, workspace="/fixture", tool_registry_factory=ToolRegistry),
            ),
        )
        transcript = []
        notices = []

        def request(identity, method, params=None):
            wire = {
                "jsonrpc": "2.0",
                "method": method,
                **({"id": identity} if identity is not None else {}),
                **({"params": params} if params is not None else {}),
            }
            server.processor.process_message(transport.connection_id, wire)
            messages = []
            if identity is not None:
                while True:
                    response = transport.receive_outbound(timeout=5)
                    if response.get("id") == identity:
                        messages.append(response)
                        break
                    notices.append(response)
            transcript.append({"connection_id": transport.connection_id, "request": wire, "responses": messages})
            return messages[-1] if messages else None

        def drain():
            while not transport._outbound.empty():
                notices.append(transport.receive_outbound(timeout=0))
            result = list(notices)
            transcript.append({"connection_id": transport.connection_id, "notifications": result})
            notices.clear()
            return result

        assert request(0, "initialize", {"clientInfo": {"name": "fixture"}})["result"]["protocolVersion"] == "v2"
        request(None, "initialized")
        request(1, "thread/start", {"agentKey": "default", "cwd": "/fixture"})
        request(2, "turn/start", {"threadId": "thread_1", "input": [{"type": "text", "text": "go"}]})
        server.run_adapter.join()
        drain()
        assert request(3, "thread/read", {"threadId": "thread_1"})["result"]["thread"]["status"] == "idle"
        assert request(4, "thread/status", {"threadId": "thread_1"})["result"]["status"] == "idle"
        request(5, "turn/resume", {"threadId": "thread_1", "turnId": "thread_1/turn/turn_1", "checkpointKey": "old"})
        request(6, "thread/unsubscribe", {"threadId": "thread_1"})
        request(13, "thread/archive", {"threadId": "thread_1"})
        assert request(14, "thread/status", {"threadId": "thread_1"})["result"]["status"] == "closed"
        for i, method, params in (
            (7, "turn/resume", {"threadId": "thread_1", "turnId": "thread_1/turn/turn_1"}),
            (8, "turn/start", {"threadId": "thread_1", "input": []}),
            (9, "thread/resume", {"threadId": "thread_1", "subscribe": True}),
            (15, "thread/resume", {"threadId": "thread_1"}),
        ):
            error = request(i, method, params)["error"]
            assert error == {"code": -32602, "message": "Thread is closed"}
        request(10, "thread/resume", {"threadId": "thread_1", "subscribe": False})
        request(11, "thread/list")
        exported = request(12, "schema/export")["result"]
        assert exported == export_schema_bundles()
        # Schema text is retained once, rather than duplicated in the transcript.
        transcript[-1]["responses"][-1]["result"] = {"bundle_names": list(exported["jsonSchema"])}
        drain()

        transport = ChannelTransport(connection_id="child-owner")
        provider = ScriptedModelProvider.new(
            "scripted",
            "m",
            [
                LLMResponse("", [ToolCall("child", "create_sub_task", {"agent_id": "worker", "task_description": "work"})]),
                LLMResponse("", [ToolCall("ask", "wait_turn", {})]),
                LLMResponse("child done"),
                LLMResponse("parent done"),
            ],
        )
        server = AppServer(
            transport=transport,
            store=self.kernel.store,
            host=DefaultAppServerHost(
                agent=Agent(
                    "app",
                    "Delegate.",
                    model="m",
                    tools=[wait_turn],
                    sub_agents={"worker": SubAgentConfig(model="m", description="Work.")},
                ),
                run_config=RunConfig(
                    model_provider=provider,
                    workspace="/fixture",
                    tool_registry_factory=partial(self.registry, ("create_sub_task",)),
                ),
            ),
        )
        request(20, "initialize", {"clientInfo": {"name": "child-owner"}})
        request(None, "initialized")
        sid = request(21, "thread/start")["result"]["threadId"]
        request(22, "turn/start", {"threadId": sid, "input": [{"type": "text", "text": "delegate"}]})
        server.run_adapter.join()
        drain()
        status = request(23, "thread/status", {"threadId": sid})["result"]
        snapshot = request(24, "thread/read", {"threadId": sid})["result"]
        assert status["status"] == snapshot["thread"]["status"] == "interrupted"
        child_sid = status["interactions"][0]["sessionId"]
        assert child_sid != sid
        tid = self.state(sid).active_turn_id
        action = {
            "threadId": sid,
            "turnId": tid,
            "actionId": "reply-child",
            "action": {"kind": "respond", "message": {"role": "user", "content": "answer"}},
        }
        request(25, "turn/action", action)
        server.run_adapter.join()
        drain()
        snapshot = request(26, "thread/read", {"threadId": sid})["result"]
        assert snapshot["turns"][-1]["result"]["finalOutput"] == "parent done", snapshot["turns"][-1]
        body = independent_bytes(
            [
                {
                    "action_id": "reply-child",
                    "schema_version": "vv-agent.controller-command-id.v1",
                    "thread_id": sid,
                    "turn_id": tid,
                }
            ]
        )[0]
        command_id = sha256(b"vv-agent.controller-command-id.v1\0" + len(body).to_bytes(8, "big") + body).hexdigest()
        assert self.state(child_sid).applied_inputs[command_id].payload["input"]["payload"]["content"]["text"] == "answer"
        head = self.kernel.store.read(sid).head_seq
        request(27, "turn/action", action)
        assert self.kernel.store.read(sid).head_seq == head
        assert "error" in request(28, "turn/action", action | {"action": {"kind": "cancel"}})
        cursor = snapshot["items"][-1]["itemId"]
        assert request(29, "thread/resume", {"threadId": sid, "subscribe": False, "afterItemId": cursor})["result"]["items"] == []
        drain()

        transport = ChannelTransport(connection_id="approval-owner")
        effects_before = len(APPROVED_EFFECTS)
        approval_host = DefaultAppServerHost(
            agent=Agent("app", "Approve.", model="m", tools=[approved]),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new(
                    "scripted", "m", [LLMResponse("", [ToolCall("approved", "approved", {})]), LLMResponse("denied safely")]
                ),
                workspace="/fixture",
                approval_timeout_seconds=60,
                tool_registry_factory=ToolRegistry,
            ),
        )
        server = AppServer(transport=transport, store=self.kernel.store, host=approval_host)
        request(30, "initialize", {"clientInfo": {"name": "approval-owner"}})
        request(None, "initialized")
        sid = request(31, "thread/start")["result"]["threadId"]
        request(32, "turn/start", {"threadId": sid, "input": [{"type": "text", "text": "approve"}]})
        while True:
            message = transport.receive_outbound(timeout=5)
            notices.append(message)
            if message.get("method") == "approval/request":
                approval = message["params"]
                break
        drain()
        owner = transport
        observer = ChannelTransport(connection_id="observer")
        server.router.register_transport(observer)
        transport = observer
        request(33, "initialize", {"clientInfo": {"name": "observer"}})
        request(None, "initialized")
        denied = request(
            34,
            "approval/resolve",
            {"threadId": sid, "turnId": approval["turnId"], "requestId": approval["requestId"], "decision": "allow"},
        )
        assert "error" in denied
        server.router.unregister_transport(owner.connection_id)
        server.run_adapter.join()
        drain()
        # A fresh App Server has no process-local owner state; only the log supplies it.
        server = AppServer(transport=observer, store=self.kernel.store, host=approval_host)
        request(35, "initialize", {"clientInfo": {"name": "observer-restart"}})
        request(None, "initialized")
        request(36, "thread/resume", {"threadId": sid})
        server.run_adapter.join()
        drain()
        state = self.state(sid)
        deadline = next(iter(state.waits.values()))["deadline_ms"]
        assert server.run_adapter._owner(sid, state.active_turn_id) == "approval-owner"
        self.clock = deadline - 1
        server.run_adapter.recover("observer", sid)
        server.run_adapter.join()
        assert self.state(sid).active_turn_id is not None
        drain()
        self.clock = deadline
        server.run_adapter.recover("observer", sid)
        server.run_adapter.join()
        resolved = drain()
        assert not any(m.get("method") == "approval/request" for m in resolved)
        assert any(m.get("method") == "approval/resolved" and m["params"]["decision"] == "timeout" for m in resolved)
        assert self.state(sid).active_turn_id is None
        assert len(APPROVED_EFFECTS) == effects_before
        host_request = HostInteractionRequest("interaction", 1, "operation", "tool", "Choose.")
        request_value = host_request.to_dict()
        outcome_value = HostInteractionOutcome(
            interaction_id=host_request.interaction_id,
            logical_cycle=host_request.logical_cycle,
            checkpoint_revision=0,
            status="admitted",
            outbox_state="pending",
            record_id="interaction-record",
            notification_id="interaction-notification",
            notification_payload_digest=sha256(independent_bytes([request_value])[0]).hexdigest(),
            notification_outbox_action="host_interaction_notification",
            notification_outbox_destination="host_interaction_observer",
        ).to_dict()
        assert HostInteractionRequest.from_dict(request_value).to_dict() == request_value
        assert HostInteractionOutcome.from_dict(outcome_value).to_dict() == outcome_value
        return {
            "protocol_version": "v2",
            "host_interaction_values": {"request": request_value, "outcome": outcome_value},
            "transcripts": transcript,
            "schemas": exported,
            "facts": {
                "child_reply_command_id": command_id,
                "owner": "approval-owner",
                "deadline_ms": deadline,
                "observer_cannot_approve": True,
                "timeout_at_absolute_deadline": True,
            },
        }

    def provider_usage(self):
        class Cut(BaseException):
            pass

        usage = USAGE
        for retry_started in (False, True):
            sid = "late_usage_audit" if retry_started else "late_usage_normal"
            provider = Provider(usage=usage)
            response = LLMResponse("first", raw={"usage": usage})

            def lose(point, plan, provider=provider, response=response):
                if point == "after_external_call":
                    provider.handle = {
                        "kind": "provider",
                        "provider": "model",
                        "job_id": "job",
                        "operation_id": plan.operation_id,
                        "attempt": plan.attempt,
                        "request_digest": plan.payload["request_digest"],
                        "evidence": "trusted/job",
                        "query_ref": "query/job",
                        "cancel_ref": None,
                    }
                    provider.result = {"content": response.content, "tool_calls": [], "raw": response.raw}
                    raise Cut

            rt = self.admit(sid, [response, LLMResponse("second")], providers={"model": provider}, hook=lose)
            try:
                self.run(sid)
            except Cut:
                pass
            else:
                raise AssertionError("missing late usage cut")
            pushed = False

            def deliver(point, record, retry_started=retry_started, sid=sid, provider=provider):
                nonlocal pushed
                if not pushed and point == ("before_external_call" if retry_started else "after_commit") and record.attempt == 2:
                    pushed = True
                    self.push(sid, provider.callback("provider_result", f"{sid}/turn/initial", "receipt"))

            rt.hook = deliver
            self.run(sid)
            result = self.result(sid)
            assert result.final_output == ("second" if retry_started else "first")
            first = result.token_usage.model_calls[0]
            assert first.usage.input_tokens == usage["prompt_tokens"] and first.usage.output_tokens == usage["completion_tokens"]
            assert first.usage.usage_source.value == "provider_reported"
            self.push(sid, provider.callback("provider_result", f"{sid}/turn/initial", "replay"))
            rt.hook = lambda *_: None
            self.run(sid)
            state, rows, _ = self.kernel.store.read_state(sid)
            assert state.applied_inputs["replay"].payload["disposition"] == "noop"
            completions = [r.record for r in rows if r.record.kind == "op_completed" and r.record.attempt == 1]
            assert len(completions) == 1
            assert independent_bytes([completions[0].payload["usage"]])[0] == independent_bytes([usage])[0]
            events = [e for e in project_records(rows) if e.type == "model_call_completed" and e.attempt == 1]
            assert len(events) == 1 and events[0].usage == first.usage
            assert self.result(sid).token_usage == result.token_usage
            self.semantics.append(
                {
                    "case": sid,
                    "context": completions[0].payload["context"],
                    "usage": usage,
                    "projected_usage": first.usage.to_dict(),
                    "billable_call_id": first.call_id,
                    "replay_disposition": "noop",
                    "completion_count": len(completions),
                }
            )

    def collect(self):
        streams = []
        for sid in self.kernel.store.list_sessions(limit=1_000_000):
            state, stored, _ = self.kernel.store.read_state(sid)
            logical = [r.record for r in stored]
            consumed = [InboxItem(**r.payload["input"]) for r in logical if r.kind == "input_applied"]
            checked = fold(logical, consumed_inputs=consumed)
            assert checked.phase == state.phase and checked.active_turn_id == state.active_turn_id
            for item in consumed:
                self.inbox[(sid, item.input_id)] = item
            streams.append((sid, stored))
        return streams


def wire_schema(envelope, payloads, *, record=False):
    variants = []
    for kind, payload in payloads.items():
        shape = deepcopy(envelope)
        shape["properties"]["kind"] = {"const": kind}
        shape["properties"]["payload"] = deepcopy(payload)
        if record and kind == "session_created":
            attributes = {
                "type": "object",
                "properties": deepcopy({"seed": SEED, "app_server": APP_SERVER_ATTRIBUTES, "child_admission": CHILD_ADMISSION}),
            }
            attributes["properties"]["child_admission"]["properties"]["definition"] = {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "object",
                        "properties": {
                            "metadata": {"type": "object", "properties": {"vv_session": deepcopy(TASK_SESSION_METADATA)}}
                        },
                    }
                },
            }
            shape["properties"]["payload"]["properties"]["attributes"] = attributes
        if record and kind == "boundary_recorded":
            shape["properties"]["payload"]["allOf"] = [
                {"if": {"properties": {"stage": {"const": stage}}}, "then": {"properties": {"data": data}}}
                for stage, data in BOUNDARY_DATA.items()
            ]
        if record:
            payload_properties = shape["properties"]["payload"]["properties"]
            for field, metadata_path, reserved in (
                ("definition", ("task", "metadata"), TASK_SESSION_METADATA),
                ("request", ("metadata",), REQUEST_SESSION_METADATA),
            ):
                if field in payload_properties:
                    nested = {"type": "object", "properties": {"vv_session": deepcopy(reserved)}}
                    for name in reversed(metadata_path):
                        nested = {"type": "object", "properties": {name: nested}}
                    payload_properties[field] = {"allOf": [payload_properties[field], nested]}
        variants.append(shape)
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "oneOf": variants}


def fold_negatives(vectors):
    invalid = []
    by_session = {}
    for v in vectors:
        if v["type"] == "record":
            by_session.setdefault(v["session_id"], []).append(v["wire"])

    def add(label, sid, index, replacement, *, prefix=None):
        log = by_session[sid]
        prior = log[:index] if prefix is None else prefix
        trial = [Record(**r) for r in prior] + [replacement]
        consumed = [InboxItem(**r.payload["input"]) for r in trial if r.kind == "input_applied"]
        try:
            fold(trial, consumed_inputs=consumed)
        except TransitionError as exc:
            reason = str(exc)
        else:
            raise AssertionError(f"accepted fold mutation: {label}")
        raw = independent_bytes([replacement.to_dict()])[0]
        assert raw == replacement.encode()
        invalid.append(
            {
                "rejection_class": label,
                "layer": "fold",
                "session_id": sid,
                "prefix_ids": [r["record_id"] for r in prior],
                "reason": reason,
                **facts(raw, record_id=expected_id(replacement.to_dict())),
            }
        )

    def changed(wire, **changes):
        return make_record(
            wire["kind"],
            session_id=wire["session_id"],
            turn_id=wire["turn_id"],
            operation_id=wire["operation_id"],
            attempt=wire["attempt"],
            payload=wire["payload"] | changes,
        )

    log = by_session["tools"]
    plan_index = next(i for i, r in enumerate(log) if r["kind"] == "op_planned")
    add("dependencies", "tools", plan_index, changed(log[plan_index], dependencies=["missing"]))
    started_index = next(i for i, r in enumerate(log) if r["kind"] == "op_started")
    add("cross_record_order", "tools", started_index, Record(**log[started_index]), prefix=log[:plan_index])
    completed_index = next(i for i, r in enumerate(log) if r["kind"] == "op_completed")
    add("request_drift", "tools", completed_index, changed(log[completed_index], request_digest="0" * 64))
    plan = log[plan_index]
    add(
        "terminal_revival",
        "tools",
        len(log),
        make_record(
            "op_planned",
            session_id="tools",
            turn_id=plan["turn_id"],
            operation_id=plan["operation_id"] + "/revival",
            attempt=1,
            payload=plan["payload"],
        ),
    )
    for sid in ("summary", "micro"):
        log = by_session[sid]
        index = next(i for i, r in enumerate(log) if r["kind"] == "context_compacted")
        for field, value in {
            "source_digest": "0" * 64,
            "prefix_ids": ["forged"],
            "tail_ids": ["forged"],
            "replacement": [],
            "evidence_manifest": {"forged": True},
        }.items():
            add(f"{sid}_compaction_{field}", sid, index, changed(log[index], **{field: value}))
    return invalid


def valid_vectors(streams, inbox):
    vectors, values, bodies = [], [], []
    for sid, stored in streams:
        for row in stored:
            values.append(row.record.to_dict())
            bodies.append(row.record.encode())
            vectors.append(
                {
                    "type": "record",
                    "session_id": sid,
                    "seq": row.seq,
                    "commit_id": row.commit_id,
                    "created_ms": row.created_ms,
                    "writer_epoch": row.writer_epoch,
                    "wire": values[-1],
                }
            )
    for (sid, _), item in sorted(inbox.items()):
        values.append(item.to_dict())
        bodies.append(item.encode())
        vectors.append({"type": "inbox", "session_id": sid, "wire": values[-1]})
    independent = independent_bytes(values)
    nested, hashes = [], []
    for vector, raw, expected in zip(vectors, bodies, independent, strict=True):
        assert raw == expected, vector["wire"]
        identity = expected_id(vector["wire"]) if vector["type"] == "record" else None
        if identity:
            assert identity == vector["wire"]["record_id"]
        vector.update(facts(expected, record_id=identity))
        if vector["type"] == "record":
            p = vector["wire"]["payload"]
            for field in ("definition", "request", "result", "input"):
                if field + "_digest" in p and field in p:
                    nested.append(p[field])
                    hashes.append(p[field + "_digest"])
            if admission := p.get("attributes", {}).get("child_admission"):
                nested.append(admission["definition"])
                hashes.append(admission["definition_digest"])
    for raw, expected in zip(independent_bytes(nested), hashes, strict=True):
        assert sha256(raw).hexdigest() == expected
    return vectors


def coverage(vectors, semantics):
    records = [v["wire"] for v in vectors if v["type"] == "record"]
    incoming = [v["wire"] for v in vectors if v["type"] == "inbox"]
    kinds = Counter(r["kind"] for r in records)
    inbox_kinds = Counter(i["kind"] for i in incoming)
    stages = Counter(r["payload"]["stage"] for r in records if r["kind"] == "boundary_recorded")
    handles = Counter(r["payload"]["handle"]["kind"] for r in records if r["kind"] == "op_parked")
    assert set(kinds) == set(PAYLOADS), set(PAYLOADS) - set(kinds)
    assert set(inbox_kinds) == set(INPUT_PAYLOADS), set(INPUT_PAYLOADS) - set(inbox_kinds)
    assert set(stages) == set(BOUNDARY_DATA), set(BOUNDARY_DATA) - set(stages)
    assert set(handles) == {"provider", "approval", "user", "child"}
    optional = {}
    for kind, schema in PAYLOADS.items():
        for field in set(schema["properties"]) - set(schema["required"]):
            optional[kind + "." + field] = sum(field in r["payload"] for r in records if r["kind"] == kind)
    for kind, schema in INPUT_PAYLOADS.items():
        for field in set(schema["properties"]) - set(schema["required"]):
            optional["inbox." + kind + "." + field] = sum(field in r["payload"] for r in incoming if r["kind"] == kind)
    for stage, schema in BOUNDARY_DATA.items():
        for field in set(schema["properties"]) - set(schema["required"]):
            optional["boundary." + stage + "." + field] = sum(
                field in r["payload"]["data"]
                for r in records
                if r["kind"] == "boundary_recorded" and r["payload"]["stage"] == stage
            )
    optional["handle.child.siblings"] = sum("siblings" in r["payload"]["handle"] for r in records if r["kind"] == "op_parked")
    for key in TASK_SESSION_METADATA["properties"]:
        optional["task.metadata.vv_session." + key] = sum(
            key in r["payload"]["definition"]["task"].get("metadata", {}).get("vv_session", {})
            for r in records
            if r["kind"] == "turn_started"
        )
    metadata = [r["payload"]["request"].get("metadata", {}).get("vv_session", {}) for r in records if r["kind"] == "op_planned"]
    metadata += [case["metadata"]["vv_session"] for case in semantics if case["case"] == "dispatch_request"]
    for key in REQUEST_SESSION_METADATA["properties"]:
        optional["request.metadata.vv_session." + key] = sum(key in m for m in metadata)
    assert all(optional.values()), optional
    return {
        "record_kinds": dict(kinds),
        "inbox_kinds": dict(inbox_kinds),
        "boundary_stages": dict(stages),
        "handle_variants": dict(handles),
        "optional_fields": optional,
    }


def invalid_vectors(vectors):
    sample = next(v["wire"] for v in vectors if v["type"] == "record" and v["wire"]["kind"] == "op_planned")
    invalid = []

    def add(label, value=None, raw=None, *, category="codec"):
        if raw is None:
            raw = json.dumps(value, ensure_ascii=True, allow_nan=True, separators=(",", ":")).encode()
        if category == "codec":
            try:
                Record.parse(raw)
            except (RecordError, ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"accepted invalid vector: {label}")
        invalid.append({"rejection_class": label, "layer": category, **facts(raw)})

    for label, value in (
        ("missing_version", None),
        ("stale_version", 0),
        ("unknown_version", 2),
        ("malformed_version", "1"),
        ("boolean_integer", True),
        ("float_integer", 1.0),
    ):
        wire = deepcopy(sample)
        if value is None:
            del wire["schema_version"]
        else:
            wire["schema_version"] = value
        add(label, wire)
    for label, key, value in (
        ("unknown_kind", "kind", "old"),
        ("wrong_identity", "record_id", "wrong"),
        ("incorrect_nullability", "attempt", None),
        ("unsafe_integer", "attempt", 2**53),
        ("unknown_field", "old_field", True),
    ):
        add(label, sample | {key: value})
    for label, key, value in (
        ("unknown_enum", "op_kind", "old"),
        ("embedded_digest", "request_digest", "0" * 64),
        ("trailing_newline_hash", "request_digest", sample["payload"]["request_digest"] + "\n"),
        ("uppercase_hash", "request_digest", "A" * 64),
        ("required_field", "purpose", None),
    ):
        wire = deepcopy(sample)
        wire["payload"][key] = value
        add(label, wire)
    for label, value in (("lone_surrogate", "\ud800"), ("nonfinite_number", float("nan"))):
        wire = deepcopy(sample)
        wire["payload"]["request"]["content"] = value
        add(label, wire)
    add(
        "duplicate_member",
        raw=json.dumps(sample).replace('"schema_version": 1', '"schema_version":1,"schema_version":1').encode(),
    )
    add("nonobject_envelope", raw=b"[]")
    add("malformed_json", raw=b"{")
    # JSON cannot encode these host values. Retain the exact valid source bytes plus
    # a typed constructor mutation, rather than pretend JSON bytes contain an object.
    for label in ("non_string_key", "non_json_host_object"):
        raw = independent_bytes([sample])[0]
        wire = deepcopy(sample)
        wire["payload"]["request"][1 if label == "non_string_key" else "host"] = (
            "value" if label == "non_string_key" else object()
        )
        try:
            Record(**wire).encode()
        except (RecordError, ValueError, TypeError):
            pass
        else:
            raise AssertionError(label)
        invalid.append({"rejection_class": label, "layer": "constructor", "mutation": label, **facts(raw)})
    wire = deepcopy(sample)
    wire["payload"]["request"]["metadata"]["vv_session"]["extra"] = True
    add("closed_reserved_metadata", wire)
    turn = deepcopy(next(v["wire"] for v in vectors if v["type"] == "record" and v["wire"]["kind"] == "turn_started"))
    turn["payload"]["definition"]["task"]["metadata"]["vv_session"] = {"extra": True}
    turn["payload"]["definition_digest"] = sha256(independent_bytes([turn["payload"]["definition"]])[0]).hexdigest()
    add("closed_task_metadata", turn)
    child = deepcopy(
        next(
            v["wire"]
            for v in vectors
            if v["type"] == "record" and "child_admission" in v["wire"]["payload"].get("attributes", {})
        )
    )
    admission = child["payload"]["attributes"]["child_admission"]
    admission["definition"]["task"]["metadata"]["vv_session"] = {"extra": True}
    admission["definition_digest"] = sha256(independent_bytes([admission["definition"]])[0]).hexdigest()
    add("closed_child_task_metadata", child)
    inbox = next(v["wire"] for v in vectors if v["type"] == "inbox" and v["wire"]["kind"] == "provider_result")
    for label, value in (
        ("retired_inbox_kind", inbox | {"kind": "deferred_result"}),
        ("stale_inbox_version", inbox | {"schema_version": 1}),
        ("missing_provider_usage", inbox | {"payload": {k: v for k, v in inbox["payload"].items() if k != "usage"}}),
        ("extra_provider_field", inbox | {"payload": inbox["payload"] | {"extra": True}}),
        ("null_provider_usage", inbox | {"payload": inbox["payload"] | {"usage": None}}),
        ("nonobject_provider_usage", inbox | {"payload": inbox["payload"] | {"usage": []}}),
        (
            "inbox_trailing_newline_hash",
            inbox | {"payload": inbox["payload"] | {"request_digest": inbox["payload"]["request_digest"] + "\n"}},
        ),
    ):
        raw = json.dumps(value).encode()
        try:
            InboxItem.parse(raw)
        except RecordError:
            pass
        else:
            raise AssertionError(label)
        invalid.append({"rejection_class": label, "layer": "inbox_codec", **facts(raw)})
    return invalid


def write_json(output, name, value):
    (output / name).write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n")


def generate(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Fixed elapsed-time observations do not affect deadlines, which use store time.
        stack.enter_context(patch("vv_agent.session.kernel.time.monotonic_ns", return_value=100_000_000))
        stack.enter_context(patch("vv_agent.prompt.builder.datetime", FixedDatetime))
        with TemporaryDirectory(prefix="c1b-producer-") as workspace:
            stack.enter_context(
                patch(
                    "vv_agent.memory.session_memory.SessionMemory.storage_path",
                    return_value=Path(workspace) / "session_memory.json",
                )
            )
            fixtures = Fixtures()
            try:
                fixtures.produce()
                app = fixtures.app_server()
                spec = importlib.util.spec_from_file_location(
                    "_session_kernel_replacements", Path(__file__).with_name("_session_kernel_replacements.py")
                )
                assert spec is not None and spec.loader is not None
                replacements = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(replacements)
                replacement_report = replacements.generate_replacements(
                    fixtures, app, output, independent_bytes, facts, write_json
                )
                streams = fixtures.collect()
                vectors = valid_vectors(streams, fixtures.inbox)
                inventory = coverage(vectors, fixtures.semantics)
                invalid = invalid_vectors(vectors) + fold_negatives(vectors) + fixtures.invalid
                write_json(output, "session_record.schema.json", wire_schema(RECORD_SCHEMA, PAYLOADS, record=True))
                write_json(output, "session_inbox.schema.json", wire_schema(INPUT_SCHEMA, INPUT_PAYLOADS))
                for kind, name in (("record", "session_records.jsonl"), ("inbox", "session_inbox.jsonl")):
                    (output / name).write_bytes(
                        b"".join(base64.b64decode(v["bytes_base64"]) + b"\n" for v in vectors if v["type"] == kind)
                    )
                write_json(output, "session_codec_vectors.json", {"coverage": inventory, "vectors": vectors})
                write_json(output, "session_invalid.json", {"vectors": invalid})
                for cases in (fixtures.semantics, fixtures.recovery):
                    for case, raw in zip(cases, independent_bytes(cases), strict=True):
                        case.update(facts(raw))
                write_json(output, "session_semantics.json", {"cases": fixtures.semantics})
                write_json(output, "session_recovery.json", {"cases": fixtures.recovery})
                projections, compactions = [], []
                for sid, stored in streams:
                    events = project_records(stored)
                    for event in events:
                        assert event_from_dict(event.to_dict()).to_dict() == event.to_dict(), (
                            sid,
                            event.to_dict(),
                            event_from_dict(event.to_dict()).to_dict(),
                        )
                    projections.append(
                        {
                            "session_id": sid,
                            "events": [e.to_dict() for e in events],
                            "spans": [
                                {"seq": seq, "method": method, "span": span.to_dict()}
                                for seq, method, span in project_spans(stored)
                            ],
                        }
                    )
                    compactions.extend(
                        {"session_id": sid, "wire": r.record.to_dict(), **facts(r.record.encode(), record_id=r.record.record_id)}
                        for r in stored
                        if r.record.kind == "context_compacted"
                    )
                    for turn_id in fixtures.state(sid).turns:
                        result = project_result(fixtures.kernel.store, sid, turn_id, runtime=fixtures.runtimes.get(sid))
                        assert result.token_usage.to_dict()["schema_version"] == "vv-agent.task-token-usage.v3"
                    projections[-1]["prefix_states"] = []
                    for through in range(1, len(stored) + 1):
                        prefix = [r.record for r in stored[:through]]
                        consumed = [InboxItem(**r.payload["input"]) for r in prefix if r.kind == "input_applied"]
                        state = fold(prefix, consumed_inputs=consumed)
                        if through < len(stored):
                            next_record = stored[through].record
                            if (
                                next_record.kind in {"context_compacted", "boundary_recorded"}
                                and next_record.payload["source_digest"] is not None
                            ):
                                context = [m.to_dict() for m in project_context(tuple(stored[:through]), state)]
                                assert sha256(independent_bytes([context])[0]).hexdigest() == next_record.payload["source_digest"]
                        projections[-1]["prefix_states"].append(
                            {
                                "seq": through,
                                "phase": state.phase,
                                "active_turn_id": state.active_turn_id,
                                "closed": state.closed,
                                "terminal_seq": state.terminal_seq,
                            }
                        )
                for projection, raw in zip(projections, independent_bytes(projections), strict=True):
                    projection.update(facts(raw))
                for entry, raw in zip(app["transcripts"], independent_bytes(app["transcripts"]), strict=True):
                    entry.update(facts(raw))
                write_json(output, "session_projection.json", {"sessions": projections})
                write_json(output, "session_compaction.json", {"vectors": compactions})
                write_json(output, "app_server_protocol.json", app)
                spec = importlib.util.spec_from_file_location(
                    "_session_kernel_curation", Path(__file__).with_name("_session_kernel_curation.py")
                )
                assert spec is not None and spec.loader is not None
                curation = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(curation)
                values, curated_coverage = curation.curate(output, streams, fixtures.semantics, independent_bytes, facts)
                replacements.validate_replacements({name: values[name] for name in replacements.REPLACE}, independent_bytes)
                spec = importlib.util.spec_from_file_location(
                    "_session_kernel_checks", Path(__file__).with_name("_session_kernel_checks.py")
                )
                assert spec is not None and spec.loader is not None
                checks = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(checks)
                self_checks = checks.validate_outputs(values, replacements.BASE, replacements.KEEP, replacements.REPLACE)
                replacement_report["replaced"] = {
                    name: replacements.classify(replacements.load(name), values[name]) for name in replacements.REPLACE
                }
                replacements.write_report(replacement_report, Path("/tmp/c1c1b-fixture-diff.md"))
                with Path("/tmp/c1c1b-fixture-diff.md").open("a") as report:
                    report.write("\n## Coverage counts (full producer = curated)\n\n")
                    report.write("| File | Full | Curated |\n| --- | ---: | ---: |\n")
                    for name, keys in curated_coverage.items():
                        report.write(f"| {name} | {len(keys['before'])} | {len(keys['after'])} |\n")
                    report.write("\n## New-file coverage keys\n")
                    for name, keys in curated_coverage.items():
                        if name in replacements.REPLACE:
                            continue
                        report.write(f"\n### {name} ({len(keys['before'])} = {len(keys['after'])})\n\n")
                        report.write("; ".join(keys["after"]) + "\n")
            finally:
                fixtures.kernel.close()
    return {"inventory": inventory, "coverage": curated_coverage, "self_checks": self_checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output), sort_keys=True))
