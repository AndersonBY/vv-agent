"""The shared attempt ledger uses core values and the SDK consumes its result."""

import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.kernel import _usage
from vv_agent.kernel._records import InboxItem, Record, digest
from vv_agent.kernel._reducer import Fold

_ROOT = Path(__file__).resolve().parents[1]
_HISTORY = _ROOT / "tests/session/fixtures/step_cost_release_histories.json"


def _history_state():
    def unused(*args, **kwargs):
        raise AssertionError("this history does not compact")

    history = json.loads(_HISTORY.read_text())[-1]
    records = [Record.parse(body.encode()) for body in history["records"]]
    fold = Fold(normalize_memory_settings=unused, count_messages=unused)
    state = fold.extend(records, consumed_inputs=[InboxItem(**r._payload["input"]) for r in records if r.kind == "input_applied"])
    return state, next(iter(state.turns))


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_usage_core_imports_and_runs_without_outer_sdk(tmp_path, flags):
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
        from vv_agent.kernel import _usage
        from vv_agent.kernel._records import Record, InboxItem
        from vv_agent.kernel._reducer import Fold

        fixture = json.loads((root / "tests/fixtures/parity/token_usage.json").read_text())
        case = fixture["normalization_cases"][0]
        values = case["input"]
        assert _usage.normalize_token_usage(values["raw_usage"], usage_source=values["usage_source_hint"],
                                            cache_status=values["cache_status_hint"]).to_dict() == case["expected"]
        assert _usage.model_usage({"prompt_tokens": 3, "session_elapsed_ms": 99}).provider_usage == {"prompt_tokens": 3}
        history = json.loads((root / "tests/session/fixtures/step_cost_release_histories.json").read_text())[-1]
        records = [Record.parse(body.encode()) for body in history["records"]]
        def unused(*args, **kwargs):
            raise AssertionError("history unexpectedly used compaction ports")
        fold = Fold(normalize_memory_settings=unused, count_messages=unused)
        state = fold.extend(records, consumed_inputs=[InboxItem(**r._payload["input"])
                                                     for r in records if r.kind == "input_applied"])
        usage, ledger = _usage.project_model_usage(state, next(iter(state.turns)))
        assert len(usage.model_calls) == len(ledger) == 2
        assert [c.cycle_index for c in usage.model_calls] == [1, 2]
        assert usage.input_tokens is None and usage.output_tokens is None
        assert all(c.usage.usage_source.value == "accounting_missing" for c in usage.model_calls)
        try:
            _usage.summarize_task_token_usage([usage.model_calls[0], usage.model_calls[0]])
        except ValueError as error:
            assert str(error) == "model_call_id_duplicate"
        else:
            raise AssertionError("duplicate call IDs must still be rejected")
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_usage_facades_share_the_actual_core_functions_and_values():
    from vv_agent import types
    from vv_agent.runtime import token_usage
    from vv_agent.session import result, runtime

    assert token_usage.normalize_token_usage is _usage.normalize_token_usage
    assert token_usage.summarize_task_token_usage is _usage.summarize_task_token_usage
    assert runtime.model_usage is _usage.model_usage
    assert result.project_model_usage is _usage.project_model_usage
    assert type(_usage.model_usage(None)) is types.TokenUsage
    assert type(_usage.summarize_task_token_usage([])) is types.TaskTokenUsage


def test_ledger_preserves_insertion_order_and_bills_every_started_receipt():
    state, tid = _history_state()
    model_ids = [oid for oid, op in state.operations.items() if op.kind == "model"]
    oid, other_id = model_ids
    operation = state.operations[oid]
    first = operation.attempts[1]
    assert first.result is not None and first.dispatch is not None
    failed_result = {"content": "", "tool_calls": [], "error_code": "rate_limited"}
    failed = replace(
        first,
        result=replace(
            first.result,
            payload=first.result.payload
            | {
                "result": failed_result,
                "result_digest": digest(failed_result),
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ),
    )
    retry = replace(
        first,
        plan=replace(first.plan, record_id="retry-plan", attempt=2),
        dispatch=replace(first.dispatch, record_id="retry-start", attempt=2),
        result=replace(
            first.result,
            record_id="retry-result",
            attempt=2,
            payload=first.result.payload | {"usage": {"prompt_tokens": 5, "completion_tokens": 1}},
        ),
    )
    unstarted = replace(first, dispatch=None, result=None)
    operation.attempts = {2: retry, 1: failed, 3: unstarted}
    operation.selected_attempt = 2
    other = state.operations[other_id]
    other.attempts[1] = replace(other.attempts[1], result=None, state="unknown")

    usage, ledger = _usage.project_model_usage(state, tid)
    assert [(c.operation_id, c.attempt) for c in usage.model_calls] == [(oid, 2), (oid, 1), (other_id, 1)]
    assert [c.usage.input_tokens for c in usage.model_calls] == [5, 3, None]
    assert [c.status.value for c in usage.model_calls] == ["completed", "failed", "ambiguous"]
    assert [entry["status"] for entry in ledger] == ["completed", "failed", "ambiguous"]
    assert usage.model_calls[-1].error_code == "model_outcome_unknown"
    assert usage.input_tokens is usage.output_tokens is usage.total_tokens is None

    other.attempts[1].dispatch = None
    known, _ = _usage.project_model_usage(state, tid)
    assert (known.input_tokens, known.output_tokens, known.total_tokens) == (8, 3, 11)


def test_sdk_result_uses_the_shared_usage_and_ledger_objects(monkeypatch):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.session import result
    from vv_agent.session.kernel import drive
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse

    driver = SessionDriver()
    try:
        runtime = driver.runtime(
            Agent("usage", "Answer."),
            RunConfig(
                model_provider=ScriptedModelProvider.from_steps(
                    "test", "m", [LLMResponse("done", raw={"usage": {"prompt_tokens": 3, "completion_tokens": 2}})]
                )
            ),
        )
        driver.create("usage", "/fixture")
        driver.push("usage", InboxItem("initial", "user", {"content": "go"}))
        drive(driver.store, "usage", runtime=runtime)
        observed = []

        def project(state, turn_id):
            value = _usage.project_model_usage(state, turn_id)
            observed.append(value)
            return value

        monkeypatch.setattr(result, "project_model_usage", project)
        projected = result.project_result(driver.store, "usage", "usage/turn/initial", runtime=runtime)
        assert len(observed) == 1
        usage, ledger = observed[0]
        assert projected.token_usage is projected.raw_result.token_usage is usage
        assert projected.metadata["session_model_calls"] is ledger
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (3, 2, 5)
    finally:
        driver.close()
