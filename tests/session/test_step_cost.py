"""Cached execution preserves records, callbacks, routing and mutable boundaries."""

import json
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from pathlib import Path

import httpx2
import pytest
from support.token_counting import PROVIDER_MODELS, legacy_count, provider_counting_settings

from vv_agent.memory import MemoryManager, token_utils
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.session import compaction, kernel, reducer
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem, Record, _MessageCache, canonical_json_bytes
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, Message, ToolCall
from vv_agent.workspace import MemoryWorkspaceBackend

from .test_cold_fold import snapshot_value
from .test_compaction import SUMMARY, tool_history
from .test_endpoint_routing import transport as transport
from .test_recovery_matrix import runtime, start

RELEASE_HISTORIES = json.loads(Path(__file__).with_name("fixtures").joinpath("step_cost_release_histories.json").read_text())


@pytest.mark.parametrize("history", RELEASE_HISTORIES, ids=lambda value: value["version"])
def test_released_histories_load_to_the_0301_state_without_byte_changes(history):
    log = [Record.parse(body.encode()) for body in history["records"]]
    state = reducer.fold(log, consumed_inputs=[InboxItem(**r._payload["input"]) for r in log if r.kind == "input_applied"])
    assert snapshot_value(state) == history["state"]
    assert [record.encode().decode() for record in log] == history["records"]


@pytest.mark.parametrize("mode", ["summary", "micro", "memory"])
@pytest.mark.parametrize("counting_model", ["deepseek-v4-pro", "gpt-4o", "gpt-3.5-turbo", *PROVIDER_MODELS])
@pytest.mark.parametrize("global_server", [False, True])
def test_replayed_decisions_boundaries_and_all_record_bytes_match_uncached(
    store, database, tmp_path, monkeypatch, mode, counting_model, global_server
):
    from vv_llm.chat_clients import utils
    from vv_llm.settings import Server

    from vv_agent.memory import manager as memory_manager
    from vv_agent.memory import session_memory

    monkeypatch.setattr(utils.settings, "token_server", None)
    backend, model_setting, endpoint = provider_counting_settings(counting_model)
    monkeypatch.setattr(type(utils.settings), "get_backend", lambda self, _: backend)
    monkeypatch.setattr(type(utils.settings), "get_endpoint", lambda self, _: endpoint)
    # Session Memory's separate text-budget counter is outside the compaction-count policy.
    monkeypatch.setattr(
        session_memory, "count_tokens", lambda text, **_: utils.get_token_counts(text, "gpt-4o", use_token_server_first=False)
    )
    # Freeze observations, not the decisions being compared; both stores use the same clock.
    monkeypatch.setattr(PostgresStore, "clock_sql", "100000")
    monkeypatch.setattr(SQLiteStore, "clock_sql", "100000")
    monkeypatch.setattr(kernel.time, "monotonic_ns", lambda: 100_000_000)
    decisions = []
    micro = MemoryManager.plan_microcompaction
    summary = MemoryManager.plan_summary
    extract = compaction.extract_memory

    def plan_micro(self, messages, **kwargs):
        result = micro(self, messages, **kwargs)
        decisions.append(("micro", self.autocompact_threshold, kwargs, snapshot_value(result)))
        return result

    def plan_summary(self, messages, **kwargs):
        result = summary(self, messages, **kwargs)
        decisions.append(("summary", self.autocompact_threshold, kwargs, snapshot_value(result)))
        return result

    def extract_memory(*args):
        result = extract(*args)
        decisions.append(("extract", args[2:], result))
        return result

    monkeypatch.setattr(MemoryManager, "plan_microcompaction", plan_micro)
    monkeypatch.setattr(MemoryManager, "plan_summary", plan_summary)
    monkeypatch.setattr(compaction, "extract_memory", extract_memory)

    @function_tool
    def more() -> str:
        return "new facts 中文 " * 400

    class Observe:
        def before_memory_compact(self, _event):
            return None

    def execute(target, target_database):
        primary_calls = 0

        def model(request):
            nonlocal primary_calls
            purpose = request.metadata["purpose"]
            if purpose == "compaction":
                return LLMResponse(SUMMARY)
            if purpose == "session_memory":
                return LLMResponse('[{"category":"key_fact","content":"Keep the evidence","importance":5}]')
            primary_calls += 1
            return LLMResponse("Inspect.", [ToolCall("new", "more", {})]) if primary_calls == 1 else LLMResponse("done")

        rt = runtime(target_database, [model] * 15, tools=[more])
        rt.resolved = replace(rt.resolved, model_id=counting_model, requested_model=counting_model)
        rt.config = replace(
            rt.config,
            workspace=tmp_path,
            initial_messages=tool_history(),
            session_memory_enabled=mode == "memory",
            metadata={"session_memory_min_tokens": 1, "session_memory_min_text_messages": 1},
            workspace_backend=MemoryWorkspaceBackend(),
        )
        rt.memory_manager = MemoryManager(
            model=counting_model,
            compact_threshold={"summary": 1000, "micro": 10000, "memory": 100000}[mode],
            model_context_window=100000,
            reserved_output_tokens=0,
            autocompact_buffer_tokens=0,
            keep_recent_messages=1,
            microcompaction_policy=MicrocompactionPolicy(
                keep_recent_cycles=1, min_result_chars=500 if mode == "micro" else 1_000_000
            ),
        )
        rt.hooks.hooks.append(Observe())
        start(target)
        kernel.drive(target, "s", runtime=rt)
        state, rows, _ = target.read_state("s")
        assert state.active_turn_id is None and rows[-1].record._payload["status"] == "completed"
        bodies = [r.record.encode() for r in rows]
        for row in rows:
            assert row.record.encode() == canonical_json_bytes(row.record.to_dict())
        commits = target._rows("SELECT commit_id FROM sk_commit WHERE session_id=%s ORDER BY commit_id", ("s",))
        return bodies, commits

    with SQLiteStore.standalone(":memory:") as old, monkeypatch.context() as patch:
        old.install_schema()
        for module in (token_utils, memory_manager, compaction, reducer):
            patch.setattr(module, "count_messages_tokens", legacy_count)
        patch.setattr(_MessageCache, "retain_array", lambda self, values: values)
        expected_bodies, expected_commits = execute(old, old)
    expected_decisions = deepcopy(decisions)
    decisions.clear()
    for file in tmp_path.rglob("session_memory.json"):
        file.unlink()
    token_utils._TOKEN_CACHE.clear()
    server = Server(host="token-server.invalid", port=8188, url="https://token-server.invalid") if global_server else None
    monkeypatch.setattr(utils.settings, "token_server", server)
    model_setting.endpoints = ["fake-enabled"]

    def unexpected_http(request):
        pytest.fail(f"Compaction attempted HTTP: {request.method} {request.url}")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda self, request: unexpected_http(request))
    with httpx2.Client(transport=httpx2.MockTransport(unexpected_http)) as client:

        def post(*args, proxy=None, **kwargs):
            assert proxy is None
            return client.post(*args, **kwargs)

        monkeypatch.setattr(utils.httpx2, "post", post)
        bodies, commits = execute(store, database)
    assert utils.settings.token_server is server
    assert decisions == expected_decisions
    assert bodies == expected_bodies
    assert commits == expected_commits
    _, rows, _ = store.read_state("s")
    if mode != "memory":
        assert any(r.record.kind == "context_compacted" and r.record._payload["mode"] == mode for r in rows)
    else:
        assert any(r.record.kind == "op_planned" and r.record._payload["purpose"] == "session_memory" for r in rows)
    assert any(r.record.kind == "boundary_recorded" and r.record._payload["stage"] == "before_memory" for r in rows)
    store._fold_cache = None
    _, cold_rows, _ = store.read_state("s")
    assert [r.record.encode() for r in cold_rows] == expected_bodies


def test_producer_fragments_detach_callers_and_encode_only_new_messages(monkeypatch):
    from vv_agent.session import records

    cache = _MessageCache()
    original = records.canonical_json_bytes
    serialized = []

    def counted(value):
        if isinstance(value, dict) and "role" in value:
            serialized.append(value["content"])
        return original(value)

    monkeypatch.setattr(records, "canonical_json_bytes", counted)
    first = {"role": "assistant", "content": "old", "metadata": {"nested": ["original"]}}
    cache.retain_array([first])
    serialized.clear()
    retained = cache.retain_array([first, {"role": "user", "content": "new"}])
    assert not isinstance(retained, list)
    assert serialized == ["new"]
    first["metadata"]["nested"].append("caller")
    assert retained.body == original(retained.value)
    assert retained.value[0]["metadata"]["nested"] == ["original"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 2**100])
@pytest.mark.parametrize("warm", [True, False])
def test_producer_fragments_never_hide_invalid_json_or_poison_a_valid_entry(invalid, warm):
    cache = _MessageCache()
    valid = [{"role": "user", "content": "same", "metadata": {"value": None}}]
    if warm:
        cache.retain_array(valid)
    malformed = [{"role": "user", "content": "same", "metadata": {"value": invalid}}]
    # The original canonical encoder must still see and reject the original input.
    retained = cache.retain_array(malformed)
    assert retained is malformed
    with pytest.raises(ValueError, match="RFC 8785 I-JSON"):
        canonical_json_bytes(retained)
    fresh = cache.retain_array(valid)
    assert not isinstance(fresh, list)
    assert fresh.body == canonical_json_bytes(valid)


def test_producer_fragments_keep_the_original_finite_float_and_mapping_encoding():
    from collections import UserDict

    cache = _MessageCache()
    values = [{"role": "user", "content": "valid", "metadata": {"value": UserDict({"float": 1e-7})}}]
    retained = cache.retain_array(values)
    assert retained is values
    assert canonical_json_bytes(retained) == canonical_json_bytes(values)


def test_tool_completion_does_not_copy_frozen_history_for_a_read_only_policy(store, database, monkeypatch):
    from vv_agent.session.records import Record

    @function_tool
    def more() -> str:
        return "ok"

    copied = []
    task = Record.task

    def copy_task(self):
        copied.append(self.record_id)
        return task(self)

    monkeypatch.setattr(Record, "task", copy_task)
    start(store)
    rt = runtime(database, [LLMResponse("inspect", [ToolCall("a", "more", {})]), LLMResponse("done")], tools=[more])
    rt.config = replace(rt.config, initial_messages=[Message("user", "large 中文 " * 100)])
    kernel.drive(store, "s", runtime=rt)
    # The tool's context gets one detached task. Internal policy/close reads borrow it.
    assert len(copied) == 1


def test_recorded_steps_tokenize_only_the_new_model_and_tool_messages(store, database, monkeypatch):
    from deepseek_tokenizer import deepseek_tokenizer
    from vv_llm.chat_clients import utils
    from vv_llm.types.enums import BackendType

    tokenized = 0
    calls = 0
    encode = deepseek_tokenizer.encode

    @wraps(encode)
    def counted(text, *args, **kwargs):
        nonlocal tokenized
        tokenized += len(text)
        return encode(text, *args, **kwargs)

    monkeypatch.setattr(deepseek_tokenizer, "encode", counted)

    @function_tool
    def more(step: int) -> str:
        return f"new information {step} 中文 " * 100

    def model(request):
        nonlocal tokenized, calls
        if calls:
            formatted = utils.format_messages(
                [m.to_openai_message() for m in request.messages[-2:]], backend=BackendType.OpenAI, native_multimodal=True
            )
            assert 0 < tokenized <= 2 * sum(len(m["content"]) for m in formatted) + 2
        tokenized = 0
        calls += 1
        return (
            LLMResponse(f"step {calls}", [ToolCall(f"a{calls}", "more", {"step": calls})]) if calls <= 3 else LLMResponse("done")
        )

    start(store)
    rt = runtime(database, [model] * 4, tools=[more])
    rt.resolved = replace(rt.resolved, model_id="deepseek-v4-pro", requested_model="deepseek-v4-pro")
    rt.config = replace(rt.config, initial_messages=[Message("user", "large history 中文 " * 4000)])
    rt.memory_manager = MemoryManager(model="deepseek-v4-pro", compact_threshold=1_000_000)
    kernel.drive(store, "s", runtime=rt)
    assert calls == 4


def test_routing_is_resolved_again_after_a_committed_plan_in_the_same_drive(store, database, transport):
    from .test_endpoint_routing import host

    calls, replies = transport
    settings = {"scripted": ["a", "b"]}
    changed = False

    def change(point, record):
        nonlocal changed
        if not changed and point == "after_commit" and record.kind == "op_planned" and record._payload["op_kind"] == "model":
            settings["scripted"] = ["b"]
            changed = True

    start(store)
    replies.append({"content": "done"})
    kernel.drive(store, "s", runtime=host(database, settings, hook=change))
    assert calls == [("scripted", "b")]
    _, rows, _ = store.read_state("s")
    unavailable = [
        r.record
        for r in rows
        if r.record.kind == "op_completed" and r.record._payload["result"].get("error_code") == "endpoint_unavailable"
    ]
    assert len(unavailable) == 1 and unavailable[0]._payload["execution_started"] is False
