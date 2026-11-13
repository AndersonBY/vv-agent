"""Actual producer histories drive the same compaction phase without importing the SDK."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_token_count_suspension import assert_core_suspended, produced_session

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._compaction_text import build_compacted_tool_content
from vv_agent.kernel._microcompaction import estimate_tool_artifact_path
from vv_agent.memory import MemoryManager, token_utils
from vv_agent.session import compaction
from vv_agent.session.context import project_context
from vv_agent.session.reducer import Fold
from vv_agent.types import Message

# pytest discovers the imported current-producer fixture; there is no external history or skip branch.
assert produced_session is not None

SCRIPT = r"""
import encodings.utf_16_be
import importlib.abc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
data = json.loads(Path(sys.argv[2]).read_text())
class OnlyCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, OnlyCore())
from vv_agent.kernel import _compaction_phase as phase, _event_validation, _message_codec
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._context import project_context
from vv_agent.kernel._message_values import Message, ToolArtifactRef
from vv_agent.kernel._microcompaction_apply import ArchiveToolMessage, ToolMarkerInput, compaction_excerpt
from vv_agent.kernel._driver_requests import CommitRecords, PlanModel
from vv_agent.kernel._records import Record, InboxItem, make_record
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._token_count import TokenCount
records = [Record(**value) for value in data['records']]
settings = phase.CompactionMemory(**data['settings'])
counts = data['counts']
def count(messages, model):
    key = canonical_json_bytes([m.to_openai_message() for m in messages]).decode()
    assert model == settings.settings['model']
    return counts[key]
def forbidden(*args): raise AssertionError('core accessed host time')
time.monotonic = time.monotonic_ns = time.time = time.sleep = forbidden
def audit(event, args):
    if event in {'open','os.listdir','os.scandir','socket.connect','socket.getaddrinfo','subprocess.Popen'}:
        raise AssertionError('core effect: ' + event)
sys.addaudithook(audit)
for case in data['cuts']:
    prefix = records[:case['index']]
    fold = Fold(normalize_memory_settings=lambda *_: settings.settings,
                count_messages=lambda messages, *, model: counts[canonical_json_bytes(messages).decode()])
    fold.extend(prefix, consumed_inputs=[InboxItem(**r._payload['input']) for r in prefix if r.kind == 'input_applied'])
    view = SimpleNamespace(sid='s', state=fold.state, records=fold.history, watermark=0)
    target = records[case['index']]
    steps = phase.finish_summary(view) if case['method']=='finish_summary' else phase.compact_context(view,max_ptl_retries=3)
    response, trace, proposed = None, [], None
    while True:
        try: request = steps.send(response)
        except StopIteration as done:
            assert done.value is True
            break
        trace.append(request.value if isinstance(request,phase.CompactionRequest) else type(request).__name__)
        response = None
        if request is phase.CompactionRequest.INBOX: response = ()
        elif request is phase.CompactionRequest.MEMORY: response = settings
        elif request is phase.CompactionRequest.TRANSCRIPT: response = project_context(fold.history,fold.state)
        elif request is phase.CompactionRequest.SAVE_MEMORY_PROJECTION: pass
        elif isinstance(request,(phase.CountContext,TokenCount)):
            values=[request.messages] if isinstance(request.messages,Message) else request.messages
            response=count(values,request.model)
        elif isinstance(request,phase.ModelRoute):
            assert request.purpose=='compaction'
            response=target._payload['request']['model']
        elif isinstance(request,PlanModel):
            assert request.oid==target.operation_id and request.purpose=='compaction'
            expected=target._payload['request']
            for key in ('model','messages','tools','prompt_bundle','model_settings'):
                assert request.request[key]==expected[key],key
            for key,value in request.request['metadata'].items():
                expected_value = expected['metadata'][key]
                if key == 'vv_session': expected_value = {'cycle_index': expected_value['cycle_index']}
                assert value==expected_value,key
            response=proposed=target
        elif isinstance(request,ArchiveToolMessage):
            artifact=data['replacement'][request.candidate.message_index]['artifact_ref']
            response=ToolArtifactRef.from_dict(artifact)
        elif isinstance(request,ToolMarkerInput):
            response=(compaction_excerpt(request.message),settings.settings['tool_result_excerpt_head'],settings.settings['tool_result_excerpt_tail'])
        elif isinstance(request,phase.CreateRecord):
            assert request.kind==target.kind
            # Micro usage is appended by the phase after constructing its record.
            expected={k:v for k,v in target._payload.items() if k!='micro_usage'}
            assert request.payload==expected
            response=make_record(request.kind,session_id='s',turn_id=target.turn_id,payload=request.payload)
            proposed=response
        elif isinstance(request,phase.MicroArtifacts):
            assert request.record.encode()==target.encode()
        elif isinstance(request,CommitRecords):
            assert len(request.records)==1
            assert request.records[0].encode()==target.encode()
            proposed=request.records[0]
        else: raise AssertionError(type(request).__name__)
    assert proposed is not None and trace[-1]=='CommitRecords'
    assert 'memory' in trace and 'transcript' in trace
    if case['method']=='finish_summary':
        assert trace==['inbox','transcript','memory','TokenCount','TokenCount','CreateRecord','CommitRecords']
    elif data['mode']=='micro':
        assert 'ArchiveToolMessage' in trace and 'ToolMarkerInput' in trace and 'MicroArtifacts' in trace
    else:
        assert trace==['memory','transcript','CountContext','ModelRoute','CountContext','PlanModel','CommitRecords']
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_phase_replays_current_producer_cuts(produced_session, tmp_path, site_flags):
    _, records, inputs, _, _ = produced_session
    target = next(r for r in records if r.kind == "context_compacted")
    definition = next(r for r in records if r.kind == "turn_started")._payload["definition"]
    manager = MemoryManager(
        **definition["memory_settings"],
        protected_start=None,
        recovery_tool_available=any(t["function"]["name"] == "read_file" for t in definition["tools"]),
    )
    prefix = Fold()
    prefix.extend(records[: records.index(target)], consumed_inputs=inputs)
    source = project_context(prefix.history, prefix.state)
    replacement = [Message.from_dict(v) for v in target._payload["replacement"]]
    samples = [source, replacement, *[[m] for m in source], *[[m] for m in replacement]]
    cuts = [
        {
            "index": records.index(target),
            "method": "finish_summary" if target._payload["mode"] == "summary" else "compact_context",
        }
    ]
    if target._payload["mode"] == "summary":
        plan = next(r for r in records if r.kind == "op_planned" and r._payload["purpose"] == "compaction")
        samples.append([Message.from_dict(v) for v in plan._payload["request"]["messages"]])
        cuts.insert(0, {"index": records.index(plan), "method": "compact_context"})
    else:
        old = source[3]
        assert old.tool_call_id is not None
        samples.append(
            [
                replace(
                    old,
                    content=build_compacted_tool_content(
                        old.content,
                        artifact_path=estimate_tool_artifact_path(old.tool_call_id),
                        tool_name="read_file",
                        excerpt_head_chars=manager.tool_result_excerpt_head,
                        excerpt_tail_chars=manager.tool_result_excerpt_tail,
                    ),
                )
            ]
        )
    counts = {
        canonical_json_bytes([m.to_openai_message() for m in values]).decode(): token_utils.count_messages_tokens(
            [m.to_openai_message() for m in values], model=manager.model
        )
        for values in samples
    }
    data = {
        "records": [r.to_dict() for r in records],
        "settings": asdict(compaction._settings(manager)),
        "counts": counts,
        "cuts": cuts,
        "mode": target._payload["mode"],
        "replacement": target._payload["replacement"],
    }
    history = tmp_path / "current-producer.json"
    history.write_text(json.dumps(data))
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", SCRIPT, str(Path(__file__).parents[1]), str(history)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_phase_host_observations_run_while_core_is_suspended(monkeypatch, tmp_path):
    from dataclasses import replace

    from session.test_compaction import SUMMARY, configured, start, tool_history

    from vv_agent.kernel._microcompaction_apply import compaction_excerpt
    from vv_agent.memory import manager as manager_module
    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.session.kernel import drive
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.types import LLMResponse
    from vv_agent.workspace import MemoryWorkspaceBackend

    seen = []
    original = manager_module.count_messages_tokens
    archive = MemoryManager._archive_tool_message

    def count(messages, *, model=""):
        chains = assert_core_suspended()
        seen.append(("count", chains))
        return original(messages, model=model)

    def archived(self, message, candidate):
        chains = assert_core_suspended()
        seen.append(("archive", chains))
        return archive(self, message, candidate)

    def excerpt(message):
        seen.append(("excerpt", assert_core_suspended()))
        return compaction_excerpt(message)

    monkeypatch.setattr(compaction, "count_messages_tokens", count)
    monkeypatch.setattr(manager_module, "count_messages_tokens", count)
    monkeypatch.setattr(MemoryManager, "_archive_tool_message", archived)
    monkeypatch.setattr(MemoryManager, "_compaction_excerpt", staticmethod(excerpt))
    for mode in ("summary", "micro"):
        with SQLiteStore.standalone(str(tmp_path / f"{mode}.sqlite")) as store:
            store.install_schema()
            start(store)
            runtime = configured(
                store, [LLMResponse(SUMMARY), LLMResponse("done")] if mode == "summary" else [LLMResponse("done")]
            )
            if mode == "micro":
                runtime.config = replace(
                    runtime.config, initial_messages=tool_history(), workspace_backend=MemoryWorkspaceBackend()
                )
                runtime.memory_manager = MemoryManager(
                    compact_threshold=1000,
                    keep_recent_messages=1,
                    microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1),
                )
            drive(store, "s", runtime=runtime)
            assert len([r for r in store.read("s").records if r.record.kind == "context_compacted"]) == 1
    assert any("compact_context" in chains for kind, chains in seen if kind == "count")
    assert any("finish_summary" in chains for kind, chains in seen if kind == "count")
    assert [kind for kind, _ in seen if kind != "count"] == ["archive", "excerpt"]
    assert all("apply_microcompaction" in chains for kind, chains in seen if kind in {"archive", "excerpt"})


def _threshold_history():
    from session.test_compaction import SUMMARY, configured, start

    from vv_agent.session.kernel import drive
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.types import LLMResponse

    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        runtime = configured(store, [LLMResponse(SUMMARY), LLMResponse("done")])
        drive(store, "s", runtime=runtime)
        return [row.record for row in store.read_state("s")[1]], runtime


@pytest.fixture(scope="module")
def threshold_history():
    return _threshold_history()


THRESHOLD_CASES = (
    "invalid_blocks_window",
    "invalid_blocks_threshold",
    "unprotected_window",
    "protected_window_error",
    "count_error_precedes_window",
    "threshold_error_after_count",
    "completion_hook_ignores_limits",
    "replacement_ignores_limits",
)


def _threshold_probe(history, case, monkeypatch, property_owners):
    """Run actual entrypoints over producer-derived state; observe only host boundaries."""
    from copy import copy
    from types import MethodType, SimpleNamespace
    from typing import Any, cast

    from vv_agent.kernel import _summary_compaction
    from vv_agent.session.kernel import _Driver
    from vv_agent.session.records import InboxItem

    records, runtime = history
    compact = next(r for r in records if r.kind == "context_compacted")
    started = next(r for r in records if r.kind == "boundary_recorded" and r._payload["stage"] == "memory_started")
    completed = next(r for r in records if r.kind == "boundary_recorded" and r._payload["stage"] == "memory_completed")
    cut = completed if case == "completion_hook_ignores_limits" else compact if case == "replacement_ignores_limits" else started
    prefix = records[: records.index(cut)]
    fold = Fold()
    fold.extend(prefix, consumed_inputs=[InboxItem(**r._payload["input"]) for r in prefix if r.kind == "input_applied"])
    source = project_context(fold.history, fold.state)
    definition = next(r for r in prefix if r.kind == "turn_started")._payload["definition"]
    manager = MemoryManager(**definition["memory_settings"], protected_start=None)
    if case != "threshold_error_after_count" and case != "invalid_blocks_threshold":
        cast(Any, manager).model_context_window = "invalid-window"
    if case in {"invalid_blocks_threshold", "threshold_error_after_count", "completion_hook_ignores_limits"}:
        cast(Any, manager).compact_threshold = float("nan")
    if case.startswith("invalid_blocks"):
        source = [Message("tool", "orphan", tool_call_id="missing")]
    if case in {"protected_window_error", "count_error_precedes_window"}:
        manager.protected_start = 1
    trace = []
    count_error = LookupError("count failed first")
    count_error.__cause__ = ValueError("counter cause")

    def count(*args, **kwargs):
        trace.append("count")
        if case == "count_error_precedes_window":
            raise count_error
        return 0

    def memory(_driver):
        trace.append("memory")
        return manager

    def transcript():
        trace.append("transcript")
        return source

    class Observer:
        def after_compact(self, event):
            trace.append("after_hook")

    observed_runtime = copy(runtime)
    observed_runtime.config = replace(runtime.config, memory_providers=[Observer()])
    driver = cast(
        Any, SimpleNamespace(sid="s", state=fold.state, records=fold.history, runtime=observed_runtime, transcript=transcript)
    )
    for name in ("record", "boundary", "boundary_record", "_task", "task"):
        setattr(driver, name, MethodType(getattr(_Driver, name), driver))
    driver.commit = lambda rows, *, guarded: trace.append("commit")
    for owner in property_owners:
        for name in ("effective_context_window", "autocompact_threshold"):
            getter = getattr(owner, name).fget

            def observed(self, getter=getter, name=name):
                trace.append(name)
                return getter(self)

            monkeypatch.setattr(owner, name, property(observed))
    monkeypatch.setattr(compaction, "manager_for", memory)
    monkeypatch.setattr(compaction, "count_messages_tokens", count)
    monkeypatch.setattr(MemoryManager, "_calculate_message_length", count)
    result = None
    error: dict[str, Any] | None = None
    try:
        if case == "replacement_ignores_limits":
            plan = _summary_compaction.plan_summary(
                source,
                keep_recent=manager.keep_recent_messages,
                drop_ratio=0,
                protected_start=None,
                language=manager.language,
                event_limit=manager.summary_event_limit,
            )
            replacement = [Message.from_dict(value) for value in compact._payload["replacement"]]
            receipt = next(
                r for r in records if r.kind == "op_completed" and r.operation_id == compact._payload["summary_operation_id"]
            )
            record = compaction.replacement_record(
                driver, manager, source, replacement, mode="summary", summary=receipt, plan=plan
            )
            result = record.encode().decode()
        else:
            result = compaction.compact_context(driver)
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "cause": None if exc.__cause__ is None else [type(exc.__cause__).__name__, str(exc.__cause__)],
        }
        if case == "count_error_precedes_window":
            error["original_counter_error"] = exc is count_error
    return {"trace": trace, "result": result, "error": error}


@pytest.mark.parametrize("case", THRESHOLD_CASES)
def test_thresholds_are_evaluated_only_at_original_read_points(threshold_history, monkeypatch, case):
    from vv_agent.kernel._compaction_phase import CompactionMemory

    observed = _threshold_probe(threshold_history, case, monkeypatch, (MemoryManager, CompactionMemory))
    if case.startswith("invalid_blocks"):
        assert observed == {"trace": ["memory", "transcript"], "result": False, "error": None}
    elif case == "unprotected_window":
        assert observed == {
            "trace": ["memory", "transcript", "count", "autocompact_threshold", "autocompact_threshold"],
            "result": False,
            "error": None,
        }
    elif case == "completion_hook_ignores_limits":
        assert observed == {"trace": ["memory", "transcript", "count", "after_hook", "commit"], "result": True, "error": None}
    elif case == "replacement_ignores_limits":
        assert observed["trace"] == [] and observed["error"] is None and isinstance(observed["result"], str)
    elif case == "count_error_precedes_window":
        assert observed["trace"] == ["memory", "transcript", "count"]
        assert observed["error"] == {
            "type": "LookupError",
            "message": "count failed first",
            "cause": ["ValueError", "counter cause"],
            "original_counter_error": True,
        }
    else:
        property_name = "effective_context_window" if case == "protected_window_error" else "autocompact_threshold"
        assert observed["trace"] == ["memory", "transcript", "count", property_name]
        assert observed["error"]["type"] == ("TypeError" if case == "protected_window_error" else "ValueError")


def test_compaction_memory_stores_settings_once_and_derives_from_current_values():
    from dataclasses import fields

    from vv_agent.kernel import _compaction_phase as phase
    from vv_agent.kernel._compaction_threshold import compute_compaction_threshold

    assert {field.name for field in fields(phase.CompactionMemory)} == {"settings", "protected_start", "recovery_tool_available"}
    assert phase.compute_compaction_threshold is compute_compaction_threshold
    manager = MemoryManager()
    memory = compaction._settings(manager)
    assert memory.effective_context_window == manager.effective_context_window
    assert memory.autocompact_threshold == manager.autocompact_threshold
    memory.settings["model_context_window"] = 100
    memory.settings["reserved_output_tokens"] = 10
    memory.settings["autocompact_buffer_tokens"] = 20
    assert memory.effective_context_window == 90 and memory.autocompact_threshold == 70
