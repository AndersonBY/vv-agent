"""Core decisions consume real ledger facts; the SDK performs each host request."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vv_agent.kernel import _after_cycle as phase
from vv_agent.kernel._cycles import project_cycles
from vv_agent.runtime.lifecycle import AfterCycleDecision

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_phase_and_cycle_projection_without_sdk(tmp_path, flags):
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root / "src"))
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _after_cycle as phase
        from vv_agent.kernel._context import project_context
        from vv_agent.kernel._cycles import project_cycles
        from vv_agent.kernel._cycle_values import AfterCycleDecision
        from vv_agent.kernel._records import InboxItem, Record
        from vv_agent.kernel._reducer import Fold
        from vv_agent.kernel._store import StoredRecord
        from vv_agent.kernel._usage import project_model_usage

        history = json.loads((root / "tests/session/fixtures/step_cost_release_histories.json").read_text())[-1]
        records = [Record.parse(body.encode()) for body in history["records"]]
        # The actual producer prefix immediately before its terminal record.
        records = records[:next(i for i, r in enumerate(records) if r.kind == "turn_ended")]
        def unused(*args, **kwargs):
            raise AssertionError("this retained history has no compaction")
        state = Fold(normalize_memory_settings=unused, count_messages=unused).extend(
            records, consumed_inputs=[InboxItem(**r._payload["input"]) for r in records if r.kind == "input_applied"]
        )
        tid = state.active_turn_id
        rows = tuple(StoredRecord(r, i + 1, "retained", 1, 0) for i, r in enumerate(records))
        usage, _ = project_model_usage(state, tid)
        cycles = project_cycles(state, rows, tid, usage)
        assert [c.index for c in cycles] == [1, 2]
        assert cycles[0].tool_results[0].content == "Verified evidence: 中文 and English."
        assert cycles[-1].assistant_message == "done"
        steps = phase.after_cycle(state, has_hooks=True)
        assert next(steps) == phase.ReadCycleProjection(tid)
        assert isinstance(steps.send((cycles, usage)), phase.ReadSharedState)
        shared = {}
        assert isinstance(steps.send(shared), phase.ReadTranscript)
        call = steps.send(project_context(rows, state))
        assert isinstance(call, phase.ApplyAfterCycleHooks)
        assert call.snapshot.cycle.to_dict() == cycles[-1].to_dict()
        assert call.snapshot.native_outcome.kind.value == "completed"
        call.snapshot.cycle.assistant_message = "detached"
        assert cycles[-1].assistant_message == "done"
        request = steps.send(AfterCycleDecision.steer(["Review 中文"], disallow_tools=["z", "a"]))
        assert isinstance(request, phase.RecordAfterCycle)
        assert request.data["error"] is None
        assert request.data["steering_messages"] == ["Review 中文"]
        assert request.data["shared_state"]["_vv_agent_after_cycle_control"]["disallowed_tools"] == ["a", "z"]
        assert not any(stage == "after_cycle" for _, stage, _ in state.boundaries)
        try:
            steps.send(None)
        except StopIteration as done:
            assert done.value is True
        else:
            raise AssertionError("phase must finish after boundary commit")
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_finish_skips_unexecuted_tool_and_shares_cycle_projector(monkeypatch, tmp_path):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
    from vv_agent.session import lifecycle, result
    from vv_agent.session.kernel import drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse, ToolCall, ToolDirective, ToolExecutionResult

    seen = []
    snapshots = []
    projected = []
    original = lifecycle._perform

    def perform(driver, manager, request):
        seen.append(type(request))
        return original(driver, manager, request)

    def project(*args):
        value = project_cycles(*args)
        projected.append(value)
        return value

    assert result.project_cycles is lifecycle.project_cycles is project_cycles
    monkeypatch.setattr(lifecycle, "_perform", perform)
    monkeypatch.setattr(lifecycle, "project_cycles", project)
    monkeypatch.setattr(result, "project_cycles", project)

    @function_tool
    def finish():
        return ToolExecutionResult("", "finished", directive=ToolDirective.FINISH)

    @function_tool
    def unexecuted():
        raise AssertionError("finish must skip the pending tool")

    class Hook:
        def after_cycle(self, snapshot):
            snapshots.append(snapshot)
            return AfterCycleDecision.continue_run()

    surface = SessionDriver()
    try:
        runtime = surface.runtime(
            Agent("phase", "Answer.", tools=[finish, unexecuted]),
            RunConfig(
                workspace=tmp_path,
                after_cycle_hooks=[Hook()],
                model_provider=ScriptedModelProvider.from_steps(
                    "test", "m", [LLMResponse("done", [ToolCall("a", "finish", {}), ToolCall("b", "unexecuted", {})])]
                ),
            ),
        )
        surface.create("phase", str(tmp_path))
        surface.push("phase", InboxItem("initial", "user", {"content": "go"}))
        drive(surface.store, "phase", runtime=runtime)
        output = result.project_result(surface.store, "phase", "phase/turn/initial", runtime=runtime)
        assert output.raw_result.cycles is projected[-1]
        assert len(snapshots) == 1
        assert snapshots[0].cycle.to_dict() == output.raw_result.cycles[-1].to_dict()
        assert snapshots[0].native_outcome.completion_reason.value == "tool_finish"
        assert snapshots[0].cycle.tool_results[1].error_code == "skipped_due_to_finish"
        assert seen.index(phase.CompleteSkippedTool) < seen.index(phase.CommitSkippedTools)
        assert seen.index(phase.CommitSkippedTools) < seen.index(phase.ApplyAfterCycleHooks) < seen.index(phase.RecordAfterCycle)
        assert output.final_output == "finished"
    finally:
        surface.close()


@pytest.mark.parametrize("failure_stage", [phase.ReadCycleProjection, phase.ReadTranscript, phase.RecordAfterCycle])
def test_host_stop_iteration_is_not_phase_completion(monkeypatch, tmp_path, failure_stage):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.session import lifecycle
    from vv_agent.session.kernel import drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse

    original = lifecycle._perform
    failure = StopIteration("host failure")

    def perform(driver, manager, request):
        if isinstance(request, failure_stage):
            raise failure
        return original(driver, manager, request)

    class Hook:
        def after_cycle(self, snapshot):
            return None

    monkeypatch.setattr(lifecycle, "_perform", perform)
    surface = SessionDriver()
    try:
        runtime = surface.runtime(
            Agent("phase", "Answer."),
            RunConfig(
                workspace=tmp_path,
                after_cycle_hooks=[Hook()],
                model_provider=ScriptedModelProvider.from_steps("test", "m", [LLMResponse("done")]),
            ),
        )
        surface.create("phase", str(tmp_path))
        surface.push("phase", InboxItem("initial", "user", {"content": "go"}))
        with pytest.raises(StopIteration) as raised:
            drive(surface.store, "phase", runtime=runtime)
        assert raised.value is failure
        state, _, _ = surface.store.read_state("phase")
        assert not any(stage == "after_cycle" for _, stage, _ in state.boundaries)
    finally:
        surface.close()


@pytest.mark.parametrize("call_id_length", [2500, 3000], ids=["descriptor-fits", "descriptor-overflow"])
def test_retained_event_descriptor_failure_precedes_hooks_and_next_model(tmp_path, call_id_length):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
    from vv_agent.app_server import ToolResultRef
    from vv_agent.session.kernel import drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.reducer import Fold
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse, ToolCall

    calls = []
    hooks = []

    @function_tool
    def echo():
        calls.append("echo")
        return "ok"

    class Hook:
        def after_cycle(self, snapshot):
            hooks.append(snapshot.cycle_index)

    # Both IDs fit the public ref's 4096-byte bound; JSON escaping expands them.
    call_id = "\x01" * call_id_length
    overflow = call_id_length == 3000
    surface = SessionDriver()
    try:
        runtime = surface.runtime(
            Agent("phase", "Answer.", tools=[echo]),
            RunConfig(
                workspace=tmp_path,
                after_cycle_hooks=[Hook()],
                model_provider=ScriptedModelProvider.from_steps(
                    "test", "m", [LLMResponse("working", [ToolCall(call_id, "echo", {})]), LLMResponse("done")]
                ),
            ),
        )
        surface.create("phase", str(tmp_path))
        surface.push("phase", InboxItem("initial", "user", {"content": "go"}))
        if overflow:
            with pytest.raises(ValueError, match=r"^result identity exceeds descriptor bound$"):
                drive(surface.store, "phase", runtime=runtime)
        else:
            drive(surface.store, "phase", runtime=runtime)
        state, rows, _ = surface.store.read_state("phase")
        logical = [row.record for row in rows]
        replay = Fold().extend(
            logical,
            consumed_inputs=[InboxItem(**r._payload["input"]) for r in logical if r.kind == "input_applied"],
        )
        assert len(replay.operations) == len(state.operations)
        assert calls == ["echo"]
        assert hooks == ([] if overflow else [1, 2])
        assert sum(op.kind == "model" for op in state.operations.values()) == (1 if overflow else 2)
        assert sum(stage == "after_cycle" for _, stage, _ in state.boundaries) == (0 if overflow else 2)
        assert any(r.kind == "turn_ended" for r in logical) == (not overflow)
        row = next(
            row
            for row in rows
            if row.record.kind == "op_completed"
            and row.record.operation_id is not None
            and state.operations[row.record.operation_id].kind != "model"
        )
        record = row.record
        assert record.turn_id is not None and record.operation_id is not None and record.attempt is not None
        ref = ToolResultRef(
            record.session_id,
            record.turn_id,
            record.operation_id,
            record.attempt,
            record._payload["result"]["tool_call_id"],
            row.seq,
            hashlib.sha256(record.encode()).hexdigest(),
        )
        assert ToolResultRef.from_dict(ref.to_dict()) == ref
    finally:
        surface.close()
