"""Real current producers exercise suspended counting without external snapshots."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from session.helpers import (
    content_ref,
    create_fixture_session,
    fixture_content_reader,
    principal,
    push_fixture_input,
    read_fixture_state,
    wrap_fixture_store,
)
from session.test_cold_fold import snapshot_value
from session.test_compaction import SUMMARY, configured, history, tool_history
from test_microcompact import _tool_turn

from vv_agent.kernel._boundary_hydration import hydrate_and_fold_records
from vv_agent.kernel._compaction_text import build_compacted_tool_content
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._microcompaction import estimate_tool_artifact_path
from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.memory import MemoryManager, token_utils
from vv_agent.memory import manager as manager_module
from vv_agent.memory.microcompact import plan_microcompact
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.session import compaction, reducer
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.context import project_context
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.projection import hydration_for_session_store
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, Message
from vv_agent.workspace import MemoryWorkspaceBackend


def assert_core_suspended():
    """Reject running core generators and require reached core generators to be suspended."""
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    suspended: set[str] = set()
    visited: set[int] = set()

    def visit_generator(generator):
        while inspect.isgenerator(generator) and id(generator) not in visited:
            visited.add(id(generator))
            generator_frame = generator.gi_frame
            if generator_frame is None:
                return
            generator_module = generator_frame.f_globals.get("__name__", "")
            if generator_module.startswith("vv_agent.kernel."):
                assert inspect.getgeneratorstate(generator) == inspect.GEN_SUSPENDED, (
                    generator_module,
                    generator.gi_code.co_name,
                )
                suspended.add(generator.gi_code.co_name)
            visit_generator(generator_frame.f_locals.get("steps"))
            generator = generator.gi_yieldfrom

    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        assert not (module.startswith("vv_agent.kernel.") and frame.f_code.co_flags & inspect.CO_GENERATOR), (
            module,
            frame.f_code.co_name,
        )
        visit_generator(frame.f_locals.get("steps"))
        frame = frame.f_back
    assert suspended
    return suspended


def test_core_suspension_probe_rejects_a_running_compaction_generator():
    namespace: dict[str, object] = {
        "__name__": "vv_agent.kernel._compaction_phase",
        "assert_core_suspended": assert_core_suspended,
    }
    exec("def running_compaction_generator():\n    assert_core_suspended()\n    yield None\n", namespace)
    probe = cast(Callable[[], Generator[object, object, object]], namespace["running_compaction_generator"])

    with pytest.raises(AssertionError, match=r"vv_agent\.kernel\._compaction_phase"):
        next(probe())


def test_cold_core_driver_core_fold_replay_keeps_generators_suspended(produced_session, monkeypatch):
    store, *_ = produced_session
    content_store = wrap_fixture_store(store)
    messages = history()
    create_fixture_session(
        content_store,
        "cold",
        attributes={
            "seed": {
                "messages": [message.to_dict() for message in messages],
                "shared_state": {},
                "unseen_input_protection": False,
                "response_rejection": True,
            }
        },
    )
    push_fixture_input(content_store, "cold", InboxItem("initial", "user", {"content": "go"}))

    runtime = configured(Path(store.path), [LLMResponse(SUMMARY), LLMResponse("done")])
    runtime.content_reader = fixture_content_reader(content_store, "cold")
    observed: list[set[str]] = []
    calls: list[tuple[str, list[dict[str, object]]]] = []
    folds: list[CoreFold] = []

    def count(messages, *, model):
        observed.append(assert_core_suspended())
        calls.append((model, messages))
        return token_utils.count_messages_tokens(messages, model=model)

    def fold_factory():
        fold = CoreFold(
            normalize_memory_settings=reducer._normalize_memory_settings,
            count_messages=count,
        )
        folds.append(fold)
        return fold

    store.fold_factory = content_store.fold_factory = fold_factory
    monkeypatch.setattr(compaction, "count_messages_tokens", count)
    lease = content_store.acquire("cold", owner="cold-core-driver", ttl_ms=15_000)
    assert lease is not None
    scope = _Scope(lease, runtime, now_ms=store._now())
    scope.start()
    try:
        assert scope.ready.wait(runtime.ttl_ms / 1000)
        assert scope.successful_ready
        assert content_store._root is store
        store._fold_cache = store._previous_prefix = None
        assert store._fold_cache is None and store._previous_prefix is None
        driver = _Driver(content_store, "cold", runtime, scope)
        assert isinstance(driver, CoreDriver)
        assert driver.fold._count_messages is count
        assert not calls
        for _ in range(20):
            driver.step()
            if any(row.record.kind == "context_compacted" for row in driver.records):
                break
        else:
            raise AssertionError("cold CoreDriver did not complete compaction")
    finally:
        try:
            scope.close()
        finally:
            assert content_store.release(lease)

    assert calls
    assert any("compact_context" in chains for chains in observed)
    assert any("_compacted" in chains for chains in observed)
    assert folds and all(fold._count_messages is count for fold in folds)
    raw_compaction = next(row.record for row in store.read("cold").records if row.record.kind == "context_compacted")
    assert raw_compaction._payload["replacement"]
    assert all(declared_content_ref(value) is not None for value in raw_compaction._payload["replacement"])


@pytest.fixture(params=["summary", "micro"])
def produced_session(request, tmp_path):
    mode = request.param
    database = tmp_path / "current.sqlite"
    with SQLiteStore.standalone(str(database)) as store:
        store.install_schema()
        content_store = wrap_fixture_store(store)
        messages = tool_history() if mode == "micro" else history()
        create_fixture_session(
            content_store,
            "s",
            attributes={
                "seed": {
                    "messages": [message.to_dict() for message in messages],
                    "shared_state": {},
                    "unseen_input_protection": False,
                    "response_rejection": True,
                }
            },
        )
        push_fixture_input(content_store, "s", InboxItem("initial", "user", {"content": "go"}))
        runtime = configured(
            database,
            [LLMResponse(SUMMARY), LLMResponse("done")] if mode == "summary" else [LLMResponse("done")],
        )
        runtime.content_reader = fixture_content_reader(content_store, "s")
        if mode == "micro":
            runtime.config = replace(runtime.config, initial_messages=tool_history(), workspace_backend=MemoryWorkspaceBackend())
            runtime.memory_manager = MemoryManager(
                compact_threshold=1000,
                keep_recent_messages=1,
                microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1),
            )
        drive(content_store, "s", runtime=runtime)
        raw_rows = store.read("s").records
        raw_compaction = next(row.record for row in raw_rows if row.record.kind == "context_compacted")
        assert raw_compaction._payload["replacement"]
        assert all(declared_content_ref(value) is not None for value in raw_compaction._payload["replacement"])
        assert declared_content_ref(raw_compaction._payload["evidence_manifest"]) is not None
        state, hydrated, _ = read_fixture_state(store, "s")
        records = [row.record for row in hydrated]
        compactions = [r for r in records if r.kind == "context_compacted"]
        assert len(compactions) == 1 and compactions[0]._payload["mode"] == mode
        inputs = [InboxItem(**r.record._payload["input"]) for r in raw_rows if r.record.kind == "input_applied"]
        raw_index = next(index for index, row in enumerate(raw_rows) if row.record.kind == "context_compacted")
        prefix = hydrate_and_fold_records(raw_rows[:raw_index], reducer.Fold, hydration_for_session_store(store, "s"))
        source = project_context(prefix.fold.history, prefix.state)
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
        assert [row.record.encode() for row in raw_rows] == [record.encode() for record in records]
        yield store, records, inputs, snapshot_value(state), counts


def test_default_fold_and_cold_sql_count_once_at_each_original_point(produced_session, monkeypatch):
    store, records, _, expected_state, expected_counts = produced_session
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

    rows = store.read("s").records
    hydration = hydration_for_session_store(store, "s")
    bodies = [row.record.encode() for row in rows]
    fold = reducer.Fold()
    result = hydrate_and_fold_records(
        once(rows, visited),
        lambda: fold,
        hydration,
        record_bodies=once(bodies, encoded),
    )
    state = result.state
    assert snapshot_value(state) == expected_state
    assert visited == list(rows) and encoded == bodies
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
        _, cold_rows, _ = store.read_state("s")
        cold_result = hydrate_and_fold_records(cold_rows, reducer.Fold, hydration)
        cold_state = cold_result.state
        assert snapshot_value(cold_state) == expected_state
        assert [row.record.encode() for row in cold_rows] == bodies
        assert calls == expected_counts
        assert cold_result.fold.fork()._count_messages is count


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
    store, _, _, _, _ = produced_session
    observed = []
    callback = raising_callback(kind, observed, suspended=point == "count")
    normalize = reducer._normalize_memory_settings if point == "count" else callback
    count = callback if point == "count" else token_utils.count_messages_tokens
    fold_type = CoreFold if point == "core-normalize" else reducer.Fold

    def fold_factory():
        return fold_type(normalize_memory_settings=normalize, count_messages=count)

    with pytest.raises(StopIteration if kind == "stop" else RuntimeError) as caught:
        hydrate_and_fold_records(
            store.read("s").records,
            fold_factory,
            hydration_for_session_store(store, "s"),
        )
    assert len(observed) == 1 and caught.value is observed[0]
    assert isinstance(caught.value.__cause__, StopIteration) if kind != "stop" else caught.value.__cause__ is None


@pytest.mark.parametrize("fold_type", [CoreFold, reducer.Fold])
def test_short_body_iterator_preserves_stop_iteration(fold_type):
    def unused(*args, **kwargs):
        raise AssertionError("creation must not use ports")

    fold = fold_type(normalize_memory_settings=unused, count_messages=unused)
    with pytest.raises(StopIteration) as caught:
        fold.extend(
            [SessionSpec("s", principal("p"), content_ref("w"), attributes=content_ref("a")).record()],
            bodies=[],
        )
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
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/tests", sys.argv[1]]
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
