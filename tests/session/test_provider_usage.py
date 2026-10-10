"""Authenticated receipts and provider queries retain canonical attempt usage."""

import json
from dataclasses import replace
from threading import Event

import pytest

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.events import ModelCallCompletedEvent
from vv_agent.session.kernel import drive
from vv_agent.session.projection import project_records
from vv_agent.session.providers import Accepted, Definitive, Unknown
from vv_agent.session.records import InboxItem, RecordError
from vv_agent.session.result import project_result
from vv_agent.session.runtime import model_usage
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_recovery_matrix import MODEL_USAGE, runtime, start


def records_of(store, kind, sid="s"):
    return [r.record for r in store.read_state(sid)[1] if r.record.kind == kind]


class WorkerDeath(BaseException):
    pass


class ReceiptProvider:
    def __init__(self, result, usage):
        self.result, self.usage = result, usage
        self.item: InboxItem
        self.submits = self.queries = 0

    def bind(self, plan):
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "retained",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "authenticated-receipt",
            "query_ref": "retained",
            "cancel_ref": None,
        }
        self.item = InboxItem(
            "receipt",
            "provider_result",
            {k: self.handle[k] for k in ("operation_id", "attempt", "request_digest")}
            | {
                "provider_binding": self.handle["provider"],
                "result": self.result,
                "usage": self.usage,
                "evidence": [self.handle["evidence"]],
            },
            target_turn_id=plan.turn_id,
            generation=0,
        )

    def submit(self, plan, *, context):
        self.submits += 1
        self.bind(plan)
        return Accepted(self.handle)

    def query(self, handle):
        assert handle == self.handle
        self.queries += 1
        return Definitive(self.result, (handle["evidence"],), usage=self.usage)

    def cancel(self, handle):
        return Unknown("no cancellation evidence")

    def authenticate(self, item, plan):
        return (
            item.encode() == self.item.encode()
            and item.target_turn_id == plan.turn_id
            and item.payload["operation_id"] == plan.operation_id
            and item.payload["attempt"] == plan.attempt
            and item.payload["request_digest"] == plan.payload["request_digest"]
        )


def test_model_usage_bytes_match_sync_receipt_and_query_and_replay(store, database):
    upstream = LLMResponse("first", raw={"usage": MODEL_USAGE})
    provider_result = {"content": upstream.content, "tool_calls": [], "raw": upstream.raw}
    usage_bytes = []
    for path in ("sync", "receipt", "query"):
        start(store, path)
        provider = ReceiptProvider(provider_result, MODEL_USAGE)

        def lose(point, plan, provider=provider):
            if point == "after_external_call" and plan.payload.get("op_kind") == "model":
                provider.bind(plan)
                raise WorkerDeath

        rt = runtime(database, [upstream], providers={"model": provider}, **({"hook": lose} if path != "sync" else {}))
        if path == "sync":
            drive(store, path, runtime=rt)
            provider.bind(records_of(store, "op_planned", path)[0])
        else:
            with pytest.raises(WorkerDeath):
                drive(store, path, runtime=rt)
            if path == "query":
                provider.item = replace(
                    provider.item,
                    kind="provider_evidence",
                    payload={k: provider.handle[k] for k in ("operation_id", "attempt", "request_digest")}
                    | {"handle": provider.handle},
                )
            with store.atomic() as tx:
                tx.push(path, provider.item)
            resumed = runtime(database, [], providers={"model": provider}, poll_ms=1)
            drive(store, path, runtime=resumed)
            if path == "query":
                Event().wait(0.01)
                drive(store, path, runtime=resumed)
        completion = records_of(store, "op_completed", path)[0]
        usage_bytes.append(canonical_json_bytes(completion.payload["usage"]))
        result = project_result(store, path, completion.turn_id)
        assert result.final_output == "first"
        assert len(result.token_usage.model_calls) == 1
        call = result.token_usage.model_calls[0]
        assert call.usage == model_usage(MODEL_USAGE)
        assert provider.queries == (1 if path == "query" else 0)
        provider.result = completion.payload["result"]
        provider.bind(records_of(store, "op_planned", path)[0])
        provider.item = replace(provider.item, input_id="replay")
        with store.atomic() as tx:
            receipt = tx.push(path, provider.item)
            assert tx.push(path, provider.item) == replace(receipt, replayed=True)
        drive(store, path, runtime=runtime(database, [], providers={"model": provider}))
        state, rows, _ = store.read_state(path)
        assert state.applied_inputs["replay"].payload["disposition"] == "noop"
        assert project_result(store, path, completion.turn_id).token_usage == result.token_usage
        events = [e for e in project_records(rows) if isinstance(e, ModelCallCompletedEvent)]
        assert len(events) == 1 and events[0].call_id == call.call_id and events[0].usage == call.usage
        assert len(records_of(store, "op_completed", path)) == 1
    assert usage_bytes == [canonical_json_bytes(MODEL_USAGE)] * 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("usage", {"prompt_tokens": 12001}),
        ("usage", {"prompt_tokens": True}),
        ("result", {"content": "changed"}),
        ("result", {"content": "first", "tool_calls": [], "raw": {"usage": {"prompt_tokens": True}}, "reasoning_content": None}),
    ],
)
def test_authenticated_completed_attempt_conflicts_on_changed_result_or_usage(store, database, field, value):
    start(store)
    provider = ReceiptProvider({"content": "first", "tool_calls": []}, {"prompt_tokens": 1})
    drive(
        store,
        "s",
        runtime=runtime(database, [LLMResponse("first", raw={"usage": provider.usage})], providers={"model": provider}),
    )
    provider.result = records_of(store, "op_completed")[0].payload["result"]
    provider.bind(records_of(store, "op_planned")[0])
    # The authority authenticates the changed receipt; the kernel must still conflict.
    provider.item = replace(provider.item, payload=provider.item.payload | {field: value})
    with store.atomic() as tx:
        tx.push("s", provider.item)
    drive(store, "s", runtime=runtime(database, [], providers={"model": provider}))
    state, _, _ = store.read_state("s")
    applied = state.applied_inputs[provider.item.input_id].payload
    assert applied["disposition"] == "rejected" and applied["reason"] == "result conflict"
    assert len(records_of(store, "op_completed")) == 1


def test_tool_provider_query_preserves_definitive_usage(store, database):
    @function_tool
    def effect() -> str:
        raise AssertionError("provider owns execution")

    provider = ReceiptProvider(ToolExecutionResult(tool_call_id="a", content="done").to_dict(), MODEL_USAGE)
    start(store)
    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
        [effect],
        providers={"effect": provider},
        poll_ms=1,
    )
    drive(store, "s", runtime=rt)
    Event().wait(0.01)
    drive(store, "s", runtime=rt)
    completed = next(r for r in records_of(store, "op_completed") if "/tool/" in r.operation_id)
    assert canonical_json_bytes(completed.payload["usage"]) == canonical_json_bytes(MODEL_USAGE)
    assert provider.submits == provider.queries == 1


@pytest.mark.parametrize("usage", [{}, MODEL_USAGE])
def test_provider_result_usage_codec(usage):
    provider = ReceiptProvider({}, usage)
    from .helpers import record

    provider.bind(record("op_planned"))
    assert InboxItem.parse(provider.item.encode()) == provider.item
    payload = provider.item.payload
    for invalid in [
        {k: v for k, v in payload.items() if k != "usage"},
        payload | {"extra": True},
        *[payload | {"usage": value} for value in (None, [], 1, True, "usage")],
    ]:
        with pytest.raises(RecordError):
            replace(provider.item, payload=invalid).encode()
        with pytest.raises(RecordError):
            InboxItem.parse(json.dumps(provider.item.to_dict() | {"payload": invalid}).encode())
