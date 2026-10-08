"""Compaction receipts and context projections across session stores."""

import json
import multiprocessing
from dataclasses import replace

import pytest

from vv_agent.memory import MemoryManager
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, digest
from vv_agent.types import LLMResponse, Message

from .conftest import open_store
from .test_recovery_matrix import kill, receive, records_of, runtime, start

SUMMARY = json.dumps({"original_user_messages": ["original request"], "current_work_state": "Continue the task"})


def history():
    return [Message("system", "Frozen system"), Message("user", "original request"), Message("assistant", "old facts " * 4000)]


def compiled_system():
    return Message(
        "system",
        "Be precise.",
        metadata={
            "session_memory_enabled": False,
            "trace_id": "s/turn/initial",
            "_vv_agent_tool_use_behavior": "run_llm_again",
            "_vv_agent_tool_policy_approval": "default",
        },
    )


def configured(database, steps, **kwargs):
    rt = runtime(database, steps, **kwargs)
    rt.config = replace(rt.config, initial_messages=history())
    rt.memory_manager = MemoryManager(compact_threshold=1000, keep_recent_messages=1)
    return rt


def test_threshold_summary_is_logged_and_applied(store, database):
    seen = []

    def summary(request):
        seen.append(request)
        assert request.metadata["purpose"] == "compaction"
        assert "old facts" in request.messages[0].content
        assert not request.tools
        return LLMResponse(SUMMARY)

    def answer(request):
        seen.append(request)
        assert [m.name for m in request.messages].count("memory_summary") == 1
        assert request.messages[-1].content == "go"
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "Be precise."
        return LLMResponse("done")

    start(store)
    drive(store, "s", runtime=configured(database, [summary, answer]))
    compact = records_of(store, "context_compacted")
    assert len(compact) == 1 and compact[0].payload["mode"] == "summary"
    assert len(seen) == 2
    plans = records_of(store, "op_planned")
    assert [p.payload["purpose"] for p in plans] == ["compaction", "primary"]
    assert compact[0].payload["summary_operation_id"] == plans[0].operation_id
    assert "old facts" in str(records_of(store, "turn_started")[0].payload)


def summary_worker(database, pipe):
    def summary(request):
        with open_store(database) as counter:
            counter.connection.execute("INSERT INTO summary_calls VALUES (1)")
        return LLMResponse(SUMMARY)

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_completed":
            pipe.send(("barrier", None))
            pipe.recv()

    with open_store(database) as store:
        drive(store, "s", runtime=configured(database, [summary], hook=hook))


@pytest.mark.persistent_store
def test_kill_after_summary_receipt_reuses_result_once(store, database):
    start(store)
    store.connection.execute("CREATE TABLE summary_calls (n integer)")
    ctx = multiprocessing.get_context("spawn")
    pipe, child = ctx.Pipe()
    proc = ctx.Process(target=summary_worker, args=(database, child))
    proc.start()
    child.close()
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        assert not records_of(store, "context_compacted")
        drive(store, "s", runtime=configured(database, [LLMResponse("done")]))
        drive(store, "s", runtime=configured(database, []))
        assert store.connection.execute("SELECT count(*) FROM summary_calls").fetchone()[0] == 1
        assert len(records_of(store, "context_compacted")) == 1
        assert records_of(store, "turn_ended")[0].payload["status"] == "completed"
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_rejected_summary_preserves_history_and_continues(store, database):
    seen = []

    def answer(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    start(store)
    drive(store, "s", runtime=configured(database, [LLMResponse('{"user_constraints":["alone"]}'), answer]))
    assert seen == [compiled_system(), *history()[1:], Message("user", "go")]
    assert not records_of(store, "context_compacted")
    assert len(records_of(store, "op_planned")) == 2


@pytest.mark.parametrize("accepted", [True, False])
def test_prompt_too_long_emergency_and_exhaustion(store, database, accepted):
    def too_long(_request):
        raise RuntimeError("maximum context length exceeded")

    start(store)
    steps = (
        [too_long, LLMResponse(SUMMARY), LLMResponse("done")]
        if accepted
        else [
            too_long,
            LLMResponse("invalid"),
            too_long,
            LLMResponse("invalid"),
            too_long,
            too_long,
        ]
    )
    rt = configured(database, steps)
    rt.memory_manager.compact_threshold = 999999
    rt.memory_manager.keep_recent_messages = 2
    drive(store, "s", runtime=rt)
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["status"] == ("completed" if accepted else "failed")
    if accepted:
        # First PTL retry keeps the configured tail, as Runner force compaction does.
        assert not records_of(store, "context_compacted")
    else:
        assert terminal["reason"] == "CompactionExhaustedError"
        assert not records_of(store, "context_compacted")
        plans = records_of(store, "op_planned")
        assert sum(p.payload["purpose"] == "primary" for p in plans) == 4
        assert sum(p.payload["purpose"] == "compaction" for p in plans) == 2


def test_retry_after_compaction_keeps_frozen_context_with_late_steer(store, database):
    seen = []

    def lost(request):
        seen.append([m.to_dict() for m in request.messages])
        with store.atomic() as tx:
            tx.push("s", InboxItem("late", "steer", {"content": "late input"}, "s/turn/initial"))
        raise TimeoutError("lost receipt")

    def retry(request):
        seen.append([m.to_dict() for m in request.messages])
        return LLMResponse("first answer")

    start(store)
    drive(store, "s", runtime=configured(database, [LLMResponse(SUMMARY), lost, retry, LLMResponse("steered answer")]))
    assert len(seen) == 2 and seen[0] == seen[1]
    plans = [p for p in records_of(store, "op_planned") if p.payload["purpose"] == "primary"]
    assert plans[0].payload["context_version"] == plans[1].payload["context_version"]
    assert digest(plans[0].payload["request"]) == digest(plans[1].payload["request"])
    assert plans[2].payload["request"]["messages"][-1]["content"] == "late input"
    assert read_state(store, "s")[0].active_turn_id is None


def tool_history():
    return [
        Message("system", "Frozen system"),
        Message("user", "original request"),
        Message(
            "assistant",
            "",
            tool_calls=[{"id": "old", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"old.txt"}'}}],
        ),
        Message("tool", "durable old facts " * 5000, tool_call_id="old"),
        Message("assistant", "recent reply"),
    ]


def micro_runtime(database, root, steps, **kwargs):
    from pathlib import Path

    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.workspace.local import LocalWorkspaceBackend

    class CountedBackend(LocalWorkspaceBackend):
        def write_text_exclusive(self, path, content):
            count = super().write_text_exclusive(path, content)
            with open_store(database) as counter:
                counter.connection.execute("INSERT INTO artifact_writes VALUES (1)")
            return count

    rt = configured(database, steps, **kwargs)
    rt.config = replace(rt.config, initial_messages=tool_history(), workspace_backend=CountedBackend(Path(root)))
    rt.memory_manager.microcompaction_policy = MicrocompactionPolicy(keep_recent_cycles=1)
    return rt


def micro_worker(database, root, pipe):
    def hook(point, record):
        if point == "after_microcompact_artifacts":
            pipe.send(("barrier", record.payload))
            pipe.recv()

    with open_store(database) as store:
        drive(store, "s", runtime=micro_runtime(database, root, [], hook=hook))


@pytest.mark.persistent_store
def test_micro_artifact_reused_after_kill_before_log_commit(store, database, tmp_path):
    start(store)
    store.connection.execute("CREATE TABLE artifact_writes (n integer)")
    ctx = multiprocessing.get_context("spawn")
    pipe, child = ctx.Pipe()
    proc = ctx.Process(target=micro_worker, args=(database, str(tmp_path), child))
    proc.start()
    child.close()
    try:
        proposed = receive(pipe, "barrier")
        kill(proc, pipe, database)
        assert not records_of(store, "context_compacted")
        rt = micro_runtime(database, str(tmp_path), [LLMResponse("done")])
        drive(store, "s", runtime=rt)
        assert store.connection.execute("SELECT count(*) FROM artifact_writes").fetchone()[0] == 1
        accepted = records_of(store, "context_compacted")
        assert len(accepted) == 1 and accepted[0].payload == proposed
        artifact = accepted[0].payload["replacement"][3]["artifact_ref"]
        assert rt.config.workspace_backend.read_bytes(artifact["path"]) == tool_history()[3].content.encode()
        assert len(records_of(store, "op_planned")) == 1
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_two_successive_summaries_keep_artifact_and_cursor_evidence(store, database):
    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.tools.function import function_tool
    from vv_agent.tools.outputs import ToolOutputText
    from vv_agent.types import ToolArtifactRef, ToolCall, ToolExecutionResult, ToolResultCursor

    artifact = ToolArtifactRef(".vv-agent/artifacts/previous.txt", "text/plain", "utf-8", 5, "a" * 64)
    original = tool_history()
    original[3].artifact_ref = artifact
    cursor = ToolResultCursor("read_file", "continued.txt", 7, "b" * 64)
    original[3] = ToolExecutionResult(
        "old",
        original[3].content,
        truncated=True,
        truncation_reason="read_limit",
        original_bytes=len(original[3].content.encode()) + 7,
        visible_bytes=len(original[3].content.encode()),
        cursor=cursor,
    ).to_tool_message()
    original[3].artifact_ref = artifact

    @function_tool
    def more() -> ToolOutputText:
        return ToolOutputText("new facts " * 5000)

    seen = []

    def summary(request):
        seen.append(request)
        return LLMResponse(SUMMARY)

    start(store)
    rt = configured(
        database, [summary, LLMResponse("", [ToolCall("new", "more", {})]), summary, LLMResponse("done")], tools=[more]
    )
    rt.config = replace(rt.config, initial_messages=original)
    rt.memory_manager.microcompaction_policy = MicrocompactionPolicy(min_result_chars=999999)

    # Keep the new completed tool block out of the next raw tail by steering after its result.
    def hook(point, record):
        if point == "after_commit" and record.kind == "op_completed" and "/tool/" in (record.operation_id or ""):
            with store.atomic() as tx:
                tx.push("s", InboxItem("continue", "steer", {"content": "continue"}, "s/turn/initial"))

    rt.hook = hook
    drive(store, "s", runtime=rt)
    accepted = records_of(store, "context_compacted")
    assert len(accepted) == 2
    assert len(seen) == 2 and "Previous Summary" in seen[1].messages[0].content
    first, second = [r.payload["evidence_manifest"] for r in accepted]
    assert first == second and first["artifacts"][0]["artifact_ref"] == artifact.to_dict()
    assert first["cursors"][0]["cursor"] == cursor.to_dict()
    plans = [r for r in records_of(store, "op_planned") if r.payload["purpose"] == "primary"]
    assert plans[-1].payload["request"]["messages"][1]["metadata"]["_vv_agent_compaction"] == first


@pytest.mark.parametrize(
    "field", ["source_digest", "prefix_ids", "tail_ids", "replacement", "evidence_manifest", "summary_operation_id"]
)
def test_compaction_log_rejects_forged_source_or_replacement(store, database, field):
    from vv_agent.session.records import make_record
    from vv_agent.session.reducer import TransitionError, fold

    start(store)
    drive(store, "s", runtime=configured(database, [LLMResponse(SUMMARY), LLMResponse("done")]))
    log = [r.record for r in read_state(store, "s")[1]]
    index = next(i for i, r in enumerate(log) if r.kind == "context_compacted")
    record = log[index]
    changes = {
        "source_digest": "0" * 64,
        "prefix_ids": ["forged"],
        "tail_ids": ["forged"],
        "replacement": [],
        "evidence_manifest": {"artifacts": [], "cursors": [], "forged": True},
        "summary_operation_id": None,
    }
    log[index] = make_record(
        "context_compacted", session_id="s", turn_id=record.turn_id, payload=record.payload | {field: changes[field]}
    )
    consumed = [InboxItem(**r.payload["input"]) for r in log if r.kind == "input_applied"]
    with pytest.raises((TransitionError, ValueError)):
        fold(log, consumed_inputs=consumed)


def test_rejected_summary_receipt_after_restart_is_not_called_again(store, database):
    class Crash(BaseException):
        pass

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_completed":
            raise Crash

    start(store)
    with pytest.raises(Crash):
        drive(store, "s", runtime=configured(database, [LLMResponse("not a summary")], hook=hook))
    seen = []

    def primary(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    drive(store, "s", runtime=configured(database, [primary]))
    assert seen == [compiled_system(), *history()[1:], Message("user", "go")]
    assert not records_of(store, "context_compacted")
    assert len([r for r in records_of(store, "op_planned") if r.payload["purpose"] == "compaction"]) == 1


def test_summary_provider_failure_keeps_history_and_continues(store, database):
    def unavailable(_request):
        raise TimeoutError("provider unavailable")

    start(store)
    drive(store, "s", runtime=configured(database, [unavailable, unavailable, LLMResponse("done")]))
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"
    assert not records_of(store, "context_compacted")
    assert len(records_of(store, "op_unknown")) == 2


def test_summary_input_window_rejects_without_truncating_or_calling(store, database):
    seen = []

    def primary(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    start(store)
    rt = configured(database, [primary])
    rt.memory_manager.model_context_window = 1000
    drive(store, "s", runtime=rt)
    assert seen == [compiled_system(), *history()[1:], Message("user", "go")]
    assert [r.payload["purpose"] for r in records_of(store, "op_planned")] == ["primary"]


def test_summary_is_internal_telemetry_not_an_app_answer(store, database):
    from vv_agent.events import ModelCallCompletedEvent, RunCompletedEvent
    from vv_agent.session.projection import project_records
    from vv_agent.types import ModelCallOperation

    start(store)
    drive(store, "s", runtime=configured(database, [LLMResponse(SUMMARY), LLMResponse("done")]))
    events = project_records(read_state(store, "s")[1])
    assert [e.final_output for e in events if isinstance(e, RunCompletedEvent)] == ["done"]
    assert [e.operation for e in events if isinstance(e, ModelCallCompletedEvent)] == [
        ModelCallOperation.MEMORY_COMPACTION,
        ModelCallOperation.AGENT_CYCLE,
    ]


def test_prompt_too_long_does_not_spend_successful_cycle_limit(store, database):
    def too_long(_request):
        raise RuntimeError("maximum context length exceeded")

    start(store)
    rt = configured(database, [too_long, LLMResponse(SUMMARY), LLMResponse("done")])
    rt.config = replace(rt.config, max_cycles=1)
    rt.memory_manager.compact_threshold = 999999
    drive(store, "s", runtime=rt)
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def test_emergency_resummarizes_previous_summary_with_smaller_raw_tail(store, database):
    def too_long(_request):
        raise RuntimeError("maximum context length exceeded")

    requests = []

    def summary(request):
        requests.append(request)
        return LLMResponse(SUMMARY)

    start(store)
    rt = configured(database, [summary, too_long, too_long, summary, LLMResponse("done")])
    rt.config = replace(
        rt.config,
        initial_messages=[*history(), Message("assistant", "tail one " * 2000), Message("assistant", "tail two " * 2000)],
    )
    rt.memory_manager.keep_recent_messages = 3
    drive(store, "s", runtime=rt)
    records = records_of(store, "context_compacted")
    assert [r.payload["mode"] for r in records] == ["summary", "emergency"]
    assert len(records[1].payload["tail_ids"]) < len(records[0].payload["tail_ids"])
    assert "Compressed Agent Memory" in requests[1].messages[0].content
    assert "tail one" in requests[1].messages[0].content
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


@pytest.mark.parametrize("source", ["manager", "tool_metadata"])
def test_preserve_retention_blocks_micro_but_allows_summary(store, database, tmp_path, source):
    from vv_agent.tools.function import function_tool
    from vv_agent.tools.metadata import ToolMetadata, ToolResultRetention

    @function_tool(
        tool_metadata=ToolMetadata(
            result_retention=(ToolResultRetention.PRESERVE if source == "tool_metadata" else ToolResultRetention.ARCHIVE)
        )
    )
    def protected(path: str) -> str:
        return path

    store.connection.execute("CREATE TABLE artifact_writes (n integer)")
    start(store)
    rt = micro_runtime(
        database,
        str(tmp_path),
        [LLMResponse(SUMMARY), LLMResponse("done")],
        tools=[protected],
    )
    if source == "manager":
        rt.memory_manager.tool_result_retentions = {"protected": ToolResultRetention.PRESERVE}
    rt.config.initial_messages[2].tool_calls[0]["function"]["name"] = "protected"
    drive(store, "s", runtime=rt)
    assert [r.payload["mode"] for r in records_of(store, "context_compacted")] == ["summary"]
    assert store.connection.execute("SELECT count(*) FROM artifact_writes").fetchone()[0] == 0
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def test_invalid_blocks_below_summary_threshold_do_not_prune_or_abort_turn(store, database, tmp_path):
    from vv_agent.memory.token_utils import count_messages_tokens

    store.connection.execute("CREATE TABLE artifact_writes (n integer)")
    start(store)
    seen = []

    def primary(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    rt = micro_runtime(database, str(tmp_path), [primary])
    messages = tool_history()
    messages.insert(4, Message("tool", messages[3].content, tool_call_id="old"))
    rt.config = replace(rt.config, initial_messages=messages)
    source = [compiled_system(), *messages[1:], Message("user", "go")]
    rt.memory_manager.compact_threshold = count_messages_tokens([m.to_openai_message() for m in source]) + 1
    drive(store, "s", runtime=rt)
    assert seen == source
    assert not records_of(store, "context_compacted")
    assert store.connection.execute("SELECT count(*) FROM artifact_writes").fetchone()[0] == 0
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"
