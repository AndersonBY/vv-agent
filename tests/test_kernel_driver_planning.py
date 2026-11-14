"""The SDK and SDK-free callers execute one planner over actual folded records."""

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vv_agent.kernel import _driver_planning as core


def folded_view(root):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from vv_agent.kernel._records import InboxItem, Record, SessionSpec
    from vv_agent.kernel._reducer import Fold

    history = json.loads((Path(root) / "tests/session/fixtures/step_cost_release_histories.json").read_text())[-1]
    records = []
    for body in history["records"]:
        record = Record.parse(body.encode())
        if record.kind == "op_planned":
            break
        records.append(record)
    first = records[0]
    records[0] = SessionSpec(first.session_id, first.payload["principal"], first.payload["workspace"]).record()

    def unused(*args, **kwargs):
        raise AssertionError("planning prefix must not count or normalize compaction settings")

    fold = Fold(normalize_memory_settings=unused, count_messages=unused)
    state = fold.extend(records, consumed_inputs=[InboxItem(**r._payload["input"]) for r in records if r.kind == "input_applied"])
    return SimpleNamespace(state=state, records=fold.history, fold=fold, sid=first.session_id)


def scripted_host(steps, view, *, hook=None, endpoints=None):
    from vv_agent.kernel import _driver_planning as core

    start = view.state.turns[view.state.active_turn_id].start
    task = start._task()
    binding = dict(start._payload["definition"]["model_binding"]) | {"endpoints": []}
    response = None
    trace = []
    try:
        while True:
            try:
                request = steps.send(response)
            except StopIteration as done:
                return done.value, trace
            trace.append(request)
            response = None
            if isinstance(request, core.ResolveModel):
                pass
            elif request is core.PlanningRead.MODEL_BINDING:
                response = binding
            elif request is core.PlanningRead.ENDPOINT_SUPPORT:
                response = endpoints is not None
            elif isinstance(request, core.OrderedEndpoints):
                response = endpoints
            elif isinstance(request, core.ProviderAvailable):
                response = request.name == "evidence"
            elif request is core.PlanningRead.HAS_HOOKS or request is core.PlanningRead.PREPARE_BEFORE_LLM:
                response = hook is not None
            elif request is core.PlanningRead.SHARED_STATE:
                response = dict(task.initial_shared_state)
            elif request is core.PlanningRead.INTERRUPTION_MESSAGES or isinstance(request, core.BeforeCycle):
                response = False, None
            elif isinstance(request, core.BeforeLLM):
                response = hook(request) if hook else (list(request.messages), list(request.schemas))
            elif request is core.PlanningRead.DEFAULT_MODEL_BINDING:
                response = binding
            elif isinstance(request, core.DurableState):
                response = dict(request.shared)
            else:
                assert isinstance(request, core.ClosePlanning)
    finally:
        steps.close()


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_normal_core_planner_builds_and_folds_real_plan_without_sdk(tmp_path, flags):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        sys.path.insert(0, sys.argv[1] + "/src")
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
    """)
    script += inspect.getsource(folded_view) + "\n" + inspect.getsource(scripted_host)
    script += textwrap.dedent("""
        from vv_agent.kernel import _driver_planning as core
        from vv_agent.kernel._context import project_context
        view = folded_view(sys.argv[1])
        start = view.state.turns[view.state.active_turn_id].start
        record, trace = scripted_host(core.model_plan(view, project_context(view.records, view.state)), view)
        assert record.kind == "op_planned" and record.payload["request"]["metadata"]["vv_session"]["cycle_index"] == 1
        assert record._payload["request"]["tools"] is start._payload["definition"]["tools"]
        cached = start._tools_body
        again, _ = scripted_host(core.model_plan(view, project_context(view.records, view.state)), view)
        assert again.encode() == record.encode() and start._tools_body is cached
        assert all(id(m) in view.fold._messages._known for m in record._payload["request"]["messages"])
        view.fold.extend([record])
        assert view.state.operations[record.operation_id].attempts[1].plan is record
        assert all(n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.")
                   for n in sys.modules if n.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).resolve().parents[1])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_planning_retains_output_repair_binding_and_empty_order():
    view = folded_view(Path(__file__).resolve().parents[1])
    start = view.state.turns[view.state.active_turn_id].start
    binding = dict(start._payload["definition"]["model_binding"]) | {"endpoints": ["repair-retained"]}
    request = {
        "model": start._task().model,
        "messages": [],
        "tools": [],
        "metadata": {"vv_session": {"model_binding": binding, "endpoint_order": ["old"]}},
    }
    record, trace = scripted_host(core.plan(view, "repair", request, "model", purpose="output_repair"), view, endpoints=[])
    assert core.PlanningRead.MODEL_BINDING not in trace
    assert record.payload["request"]["metadata"]["vv_session"] == {"model_binding": binding, "endpoint_order": ["old"]}


@pytest.mark.parametrize("mode", ["primary", "child_summary"])
def test_core_unknown_consumption_preserves_operation_and_attempt_order(mode):
    from vv_agent.kernel._records import Record
    from vv_agent.kernel._state import Attempt, Operation

    view = folded_view(Path(__file__).resolve().parents[1])
    tid = view.state.active_turn_id
    for oid, turn, kind in [
        ("later", tid, "tool"),
        ("earlier", tid, "tool"),
        ("other", "other", "tool"),
        ("model", tid, "model"),
    ]:
        plan = Record(oid, "op_planned", view.sid, turn, oid, 1, {})
        view.state.operations[oid] = Operation(turn, kind, {2: Attempt(plan, state="unknown"), 1: Attempt(plan, state="unknown")})
    request = {"model": view.state.turns[tid].start._task().model, "messages": [], "tools": [], "metadata": {"mode": mode}}
    record, _ = scripted_host(core.plan(view, "model/next", request, "model"), view)
    assert record.payload["consumed_unknowns"] == (
        [] if mode == "child_summary" else [{"operation_id": oid, "attempt": n} for oid in ("later", "earlier") for n in (2, 1)]
    )


@pytest.mark.parametrize("idempotency", [None, "supported", "unsupported"])
def test_tool_plan_keeps_provider_choice_and_idempotency_key(idempotency):
    view = folded_view(Path(__file__).resolve().parents[1])
    capability = {"idempotency": idempotency} if idempotency is not None else None
    record, _ = scripted_host(
        core.plan(view, "tool", {"id": "call", "name": "evidence", "arguments": {}}, "tool", capability=capability), view
    )
    assert record.payload["provider_binding"] == "evidence"
    assert record.payload["idempotency_key"] == (None if idempotency == "unsupported" else view.sid + "/tool")


@pytest.mark.parametrize("mode", ["protected-tail", "recovery", "denied-tools"])
def test_core_owns_protected_tail_and_recovery_decisions(mode):
    from vv_agent.kernel._compaction_text import build_compacted_tool_content
    from vv_agent.kernel._context import project_context
    from vv_agent.kernel._context_messages import UnseenInputError
    from vv_agent.kernel._cycle_values import AFTER_CYCLE_CONTROL_SCHEMA, AFTER_CYCLE_CONTROL_STATE_KEY
    from vv_agent.kernel._message_values import Message

    view = folded_view(Path(__file__).resolve().parents[1])

    def hook(request):
        if mode == "protected-tail":
            return [request.messages[0]], request.schemas
        if mode == "recovery":
            marker = Message(
                "tool", build_compacted_tool_content("text", artifact_path="a", tool_name="evidence"), tool_call_id="c"
            )
            return [*request.messages, marker], []
        request.shared[AFTER_CYCLE_CONTROL_STATE_KEY] = {
            "schema_version": AFTER_CYCLE_CONTROL_SCHEMA,
            "disallowed_tools": ["evidence"],
        }
        return request.messages, request.schemas

    steps = core.model_plan(view, project_context(view.records, view.state))
    if mode == "protected-tail":
        with pytest.raises(UnseenInputError, match="unseen_input_removed"):
            scripted_host(steps, view, hook=hook)
    else:
        record, trace = scripted_host(steps, view, hook=hook)
        if mode == "recovery":
            assert record is None
            assert trace[-1] == core.ClosePlanning("failed", "microcompaction_recovery_unavailable")
            assert core.PlanningRead.DEFAULT_MODEL_BINDING not in trace
        else:
            assert record.payload["request"]["tools"] == []


def test_sdk_passes_the_actual_core_driver_state_owner(monkeypatch):
    from vv_agent.kernel._driver import CoreDriver
    from vv_agent.session import _driver_planning as adapter
    from vv_agent.session import _recovery_phase
    from vv_agent.session.kernel import _Driver

    assert issubclass(_Driver, CoreDriver)
    current = folded_view(Path(__file__).resolve().parents[1])
    driver = object.__new__(_Driver)
    driver.state, driver.records, driver.fold, driver.sid = current.state, tuple(current.records), current.fold, current.sid
    observed = []
    original = core.plan

    def planning(view, *args, **kwargs):
        observed.append(view)
        return original(view, *args, **kwargs)

    def drain(owner, steps):
        assert owner is driver
        return scripted_host(steps, owner)[0]

    monkeypatch.setattr(core, "plan", planning)
    monkeypatch.setattr(_recovery_phase, "run", drain)
    first = adapter.plan(driver, "tool", {"id": "c", "name": "evidence", "arguments": {}}, "tool")
    replacement = folded_view(Path(__file__).resolve().parents[1])
    driver.state, driver.records, driver.fold = replacement.state, tuple(replacement.records), replacement.fold
    second = adapter.plan(driver, "tool", {"id": "c", "name": "evidence", "arguments": {}}, "tool")
    assert observed == [driver, driver]
    assert all(isinstance(view, CoreDriver) for view in observed)
    assert observed[-1].state is replacement.state and observed[-1].fold is replacement.fold
    assert first.encode() == second.encode()


@pytest.mark.parametrize("mode", ["default", "fork", "selected", "selected-over-fork"])
def test_effective_model_core_owns_selected_fork_default_precedence(mode):
    from dataclasses import replace

    from vv_agent.kernel._records import InboxItem, make_record

    view = folded_view(Path(__file__).resolve().parents[1])
    fork = {"model": "fork", "route": {}, "model_settings": {"max_tokens": 13}, "binding": {"model": "fork"}}
    if "fork" in mode:
        first = view.records[0]
        payload = first.record.payload
        payload["attributes"]["app_server_fork"] = {"effective_model": fork}
        view.records[0] = replace(first, record=replace(first.record, payload=payload))
    if mode.startswith("selected"):
        selection = {"model": "selected", "model_settings": {"max_tokens": 17}, "route": {}}
        item = InboxItem("selection", "model_selection", {"selection": selection})
        binding = {"backend": "test", "model": "selected", "endpoints": [], "context_length": None, "max_output_tokens": None}
        record = make_record(
            "input_applied",
            session_id=view.sid,
            turn_id=view.state.active_turn_id,
            payload={
                "input": item.to_dict(),
                "input_digest": item.digest,
                "disposition": "applied",
                "reason": None,
                "target_operation_id": None,
                "target_wait_id": None,
                "position": "between_actions",
                "model_binding": binding,
            },
        )
        view.fold.extend([record], consumed_inputs=[item])
    result, trace = scripted_host(core.effective_model(view), view)
    if mode.startswith("selected"):
        assert result["model"] == "selected" and trace == []
        assert result["binding"] is view.state.model_selection._payload["model_binding"]
    elif mode == "fork":
        assert result == fork and trace == []
        assert result["model_settings"] is not fork["model_settings"]
    else:
        assert result["model"] == view.state.turns[view.state.active_turn_id].start._task().model
        assert trace == [core.PlanningRead.DEFAULT_MODEL_BINDING]
