"""Real current producers exercise suspended counting without external snapshots."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from session.test_cold_fold import snapshot_value
from session.test_compaction import SUMMARY, configured, start, tool_history
from test_microcompact import _tool_turn

from vv_agent.kernel._compaction_text import build_compacted_tool_content
from vv_agent.kernel._microcompaction import estimate_tool_artifact_path
from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.memory import MemoryManager, token_utils
from vv_agent.memory import manager as manager_module
from vv_agent.memory.microcompact import plan_microcompact
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.session import reducer
from vv_agent.session.context import project_context
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, Message
from vv_agent.workspace import MemoryWorkspaceBackend


def assert_core_suspended():
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    suspended = set()
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        assert not (module.startswith("vv_agent.kernel.") and module != "vv_agent.kernel._sql"), (module, frame.f_code.co_name)
        steps = frame.f_locals.get("steps")
        while inspect.isgenerator(steps):
            assert inspect.getgeneratorstate(steps) == inspect.GEN_SUSPENDED
            suspended.add(steps.gi_code.co_name)
            steps = steps.gi_yieldfrom
        frame = frame.f_back
    assert suspended
    return suspended


@pytest.fixture(params=["summary", "micro"])
def produced_session(request, tmp_path):
    mode = request.param
    with SQLiteStore.standalone(str(tmp_path / "current.sqlite")) as store:
        store.install_schema()
        start(store)
        runtime = configured(store, [LLMResponse(SUMMARY), LLMResponse("done")] if mode == "summary" else [LLMResponse("done")])
        if mode == "micro":
            runtime.config = replace(runtime.config, initial_messages=tool_history(), workspace_backend=MemoryWorkspaceBackend())
            runtime.memory_manager = MemoryManager(
                compact_threshold=1000,
                keep_recent_messages=1,
                microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1),
            )
        drive(store, "s", runtime=runtime)
        state, rows, _ = store.read_state("s")
        records = [row.record for row in rows]
        compactions = [r for r in records if r.kind == "context_compacted"]
        assert len(compactions) == 1 and compactions[0]._payload["mode"] == mode
        inputs = [InboxItem(**r._payload["input"]) for r in records if r.kind == "input_applied"]
        prefix = reducer.Fold()
        prefix.extend(records[: records.index(compactions[0])], consumed_inputs=inputs)
        source = project_context(prefix.history, prefix.state)
        definition = next(r for r in records if r.kind == "turn_started")._payload["definition"]
        settings = reducer._normalize_memory_settings(definition["memory_settings"], None)
        if mode == "summary":
            replacement = [Message.from_dict(value) for value in compactions[0]._payload["replacement"]]
            expected = [replacement, source]
        else:
            old = source[3]
            assert old.role == "tool" and old.tool_call_id == "old"
            marker = replace(
                old,
                content=build_compacted_tool_content(
                    old.content,
                    artifact_path=estimate_tool_artifact_path("old"),
                    tool_name="read_file",
                    excerpt_head_chars=settings["tool_result_excerpt_head"],
                    excerpt_tail_chars=settings["tool_result_excerpt_tail"],
                ),
            )
            expected = [source, [old], [marker]]
        counts = [(settings["model"], [m.to_openai_message() for m in messages]) for messages in expected]
        yield store, records, inputs, snapshot_value(state), counts


def test_default_fold_and_cold_sql_count_once_at_each_original_point(produced_session, monkeypatch):
    store, records, inputs, expected_state, expected_counts = produced_session
    calls, chains, visited, encoded = [], set(), [], []

    def count(messages, *, model):
        chains.update(assert_core_suspended())
        calls.append((model, messages))
        return token_utils.count_messages_tokens(messages, model=model)

    monkeypatch.setattr(reducer, "count_messages_tokens", count)

    def once(values, visits):
        for value in values:
            visits.append(value)
            yield value

    bodies = [record.encode() for record in records]
    fold = reducer.Fold()
    state = fold.extend(once(records, visited), consumed_inputs=inputs, bodies=once(bodies, encoded))
    assert snapshot_value(state) == expected_state
    assert visited == records and encoded == bodies
    assert [record.encode() for record in records] == bodies
    assert calls == expected_counts
    assert fold.fork()._count_messages is count
    assert {"extend_steps", "_compacted"} <= chains
    assert (
        ("accept_summary" in chains)
        if len(expected_counts) == 2
        else ({"plan_memory_microcompact", "plan_microcompact"} <= chains)
    )
    for _ in range(2):
        calls.clear()
        store._fold_cache = store._previous_prefix = None
        cold_state, cold_rows, _ = store.read_state("s")
        assert snapshot_value(cold_state) == expected_state
        assert [row.record.encode() for row in cold_rows] == bodies
        assert calls == expected_counts
        assert store._fold_cache is not None and store._fold_cache.reducer.fork()._count_messages is count


@pytest.mark.parametrize("size", [1, 8, 64])
def test_micro_planner_is_linear_and_preserves_estimate_hooks(size, monkeypatch):
    messages = [Message("user", "start")]
    for i in range(size):
        messages.extend(_tool_turn(str(i), "read_file", "事实😀 " * 1000))
    messages.append(Message("assistant", "recent"))
    counts = []

    def estimate(self, message):
        assert {"plan_microcompact", "plan_memory_microcompact"} <= assert_core_suspended()
        counts.append(message)
        return len(message.content)

    monkeypatch.setattr(MemoryManager, "_estimate_message_tokens", estimate)
    manager = MemoryManager(
        compact_threshold=1000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        keep_recent_messages=1,
        recovery_tool_available=True,
        microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1),
    )
    assert manager.plan_microcompaction(messages, cycle_index=None, current_tokens=9000) is None
    assert manager.plan_microcompaction(messages, cycle_index=99, current_tokens=0) is None
    assert counts == []
    plan = manager.plan_microcompaction(messages, cycle_index=99, current_tokens=9000)
    assert plan is not None and plan.candidate_count == size
    assert len(counts) == 2 * size
    for i in range(size):
        assert counts[2 * i] is messages[2 * i + 2]
        assert counts[2 * i + 1].tool_call_id == str(i)
        assert counts[2 * i + 1].content.startswith("<Tool Result Compact>")


@pytest.mark.parametrize("require_reduction", [False, True])
def test_summary_count_order_and_early_rejection(require_reduction, monkeypatch):
    messages = [Message("user", "request"), Message("assistant", "facts " * 400), Message("user", "tail")]
    calls = []

    def count(self, values):
        assert "accept_summary" in assert_core_suspended()
        calls.append(values)
        return 1000 if values == messages else 77

    monkeypatch.setattr(MemoryManager, "_calculate_message_length", count)
    manager = MemoryManager(keep_recent_messages=1)
    plan = manager.plan_summary(messages)
    assert plan is not None
    assert manager.accept_summary(plan, "{}") == (messages, False)
    assert calls == []
    replacement, accepted = manager.accept_summary(plan, SUMMARY, require_reduction=require_reduction)
    assert accepted
    assert calls == ([replacement, messages] if require_reduction else [replacement])


def raising_callback(kind, observed, *, suspended=False):
    original = StopIteration("callback stopped")

    def callback(*args, **kwargs):
        if suspended:
            assert_core_suspended()
        try:
            if kind == "nested":

                def stream():
                    yield None
                    raise original

                values = stream()
                next(values)
                next(values)
            elif kind == "runtime":
                raise RuntimeError("generator raised StopIteration") from original
            else:
                raise original
        except BaseException as error:
            observed.append(error)
            raise

    return callback


@pytest.mark.parametrize("kind", ["stop", "nested", "runtime"])
@pytest.mark.parametrize("driver", ["public", "manager"])
@pytest.mark.parametrize("point", ["path", "count"])
def test_micro_callbacks_keep_the_actual_exception_object(kind, driver, point, monkeypatch):
    messages = [Message("user", "start"), *_tool_turn("old", "read_file", "body " * 1000), Message("assistant", "recent")]
    observed = []
    callback = raising_callback(kind, observed, suspended=point == "count")
    expected = StopIteration if kind == "stop" else RuntimeError
    with pytest.raises(expected) as caught:
        if driver == "public":
            plan_microcompact(
                messages,
                current_cycle=99,
                current_tokens=9000,
                target_tokens=100,
                policy=MicrocompactionPolicy(keep_recent_cycles=1),
                result_retentions={},
                artifact_path_estimate_for=callback if point == "path" else estimate_tool_artifact_path,
                estimate_message_tokens=callback if point == "count" else lambda m: len(m.content),
            )
        else:
            if point == "path":
                monkeypatch.setattr(manager_module, "estimate_tool_artifact_path", callback)
            else:
                monkeypatch.setattr(MemoryManager, "_estimate_message_tokens", callback)
            manager = MemoryManager(
                compact_threshold=1000,
                recovery_tool_available=True,
                microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1),
            )
            manager.plan_microcompaction(messages, cycle_index=99, current_tokens=9000)
    assert len(observed) == 1 and caught.value is observed[0]
    if kind != "stop":
        assert str(caught.value) == "generator raised StopIteration"
        assert isinstance(caught.value.__cause__, StopIteration)
    else:
        assert caught.value.__cause__ is None


@pytest.mark.parametrize("kind", ["stop", "nested", "runtime"])
@pytest.mark.parametrize("point", ["normalize", "count", "core-normalize"])
def test_fold_ports_keep_the_actual_exception_object(produced_session, kind, point):
    _, records, inputs, _, _ = produced_session
    observed = []
    callback = raising_callback(kind, observed, suspended=point == "count")
    normalize = reducer._normalize_memory_settings if point == "count" else callback
    count = callback if point == "count" else token_utils.count_messages_tokens
    fold = (CoreFold if point == "core-normalize" else reducer.Fold)(normalize_memory_settings=normalize, count_messages=count)
    with pytest.raises(StopIteration if kind == "stop" else RuntimeError) as caught:
        fold.extend(records, consumed_inputs=inputs)
    assert len(observed) == 1 and caught.value is observed[0]
    assert isinstance(caught.value.__cause__, StopIteration) if kind != "stop" else caught.value.__cause__ is None


@pytest.mark.parametrize("fold_type", [CoreFold, reducer.Fold])
def test_short_body_iterator_preserves_stop_iteration(fold_type):
    def unused(*args, **kwargs):
        raise AssertionError("creation must not use ports")

    fold = fold_type(normalize_memory_settings=unused, count_messages=unused)
    with pytest.raises(StopIteration) as caught:
        fold.extend([SessionSpec("s", "p", "w").record()], bodies=[])
    assert caught.value.__cause__ is None and not fold.history


@pytest.mark.parametrize("scenario", ["cold", "first-failure", "custom", "qwen-new", "qwen-old"])
def test_real_counter_first_use_in_a_fresh_process(tmp_path, scenario):
    root = Path(__file__).resolve().parents[1]
    script = """
import os, runpy, sys
from pathlib import Path
class ForbiddenIO(BaseException): pass
def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise ForbiddenIO(event)
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = Path(os.fsdecode(args[0])).name
        if name == "local_settings.py" or name.startswith(".env") or name == "credentials":
            raise ForbiddenIO(name)
sys.addaudithook(audit)
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/tests"]
runpy.run_path(sys.argv[2])["exercise_cold_counter"](sys.argv[3])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(root), str(Path(__file__).resolve()), scenario],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise_cold_counter(scenario):
    from vv_llm.chat_clients import utils

    calls, attempts, loads = [], [], []
    original_count, original_get = token_utils.count_messages_tokens, utils.get_gpt_4o_encoding
    assert utils.gpt_4o_encoding is None and not utils._qwen_tokenizer_cache

    def count(messages, *, model):
        assert "accept_summary" in assert_core_suspended()
        value = original_count(messages, model=model)
        calls.append((messages, value))
        return value

    def getter():
        assert_core_suspended()
        attempts.append("encoding")
        if scenario == "first-failure" and len(attempts) == 1:
            raise OSError("first initialization unavailable")
        return original_get()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager_module, "count_messages_tokens", count)
        patch.setattr(utils, "get_gpt_4o_encoding", getter)
        if scenario == "custom":

            def custom(**kwargs):
                assert_core_suspended()
                attempts.append("custom")
                if len(attempts) == 1:
                    raise OSError("custom unavailable")
                return 123

            patch.setattr(utils, "get_message_token_counts", custom)
        models = ["gpt-4o", "gpt-4o"]
        if scenario.startswith("qwen"):
            import qwen_tokenizer

            original_qwen = qwen_tokenizer.get_tokenizer

            def load(model):
                assert_core_suspended()
                loads.append(model)
                return original_qwen(model)

            patch.setattr(qwen_tokenizer, "get_tokenizer", load)
            models = ["qwen3.6-27b", "qwen-plus"]
            if scenario == "qwen-old":
                models.reverse()
        messages = [Message("user", "request"), Message("assistant", "facts " * 400), Message("user", "tail")]
        selected = None
        for model in models:
            manager = MemoryManager(model=model, keep_recent_messages=1)
            plan = manager.plan_summary(messages)
            assert plan is not None
            before = (len(calls), len(attempts), len(loads))
            assert not manager.accept_summary(plan, "{}")[1]
            assert (len(calls), len(attempts), len(loads)) == before
            assert manager.accept_summary(plan, SUMMARY, require_reduction=False)[1]
            if scenario.startswith("qwen"):
                current = utils._qwen_tokenizer_cache["qwen"]
                assert selected is None or current is selected
                selected = current
        assert len(calls) == 2 and calls[0][0] == calls[1][0]
        if scenario in {"first-failure", "custom"}:
            assert calls[0][1] == token_utils._estimate_tokens(json.dumps(calls[0][0], ensure_ascii=False))
            assert calls[1][1] != calls[0][1]
        if scenario == "custom":
            assert attempts == ["custom", "custom"] and calls[1][1] == 123
        elif scenario.startswith("qwen"):
            assert loads == [models[0]] and list(utils._qwen_tokenizer_cache) == ["qwen"]
        else:
            assert len(attempts) >= 2
