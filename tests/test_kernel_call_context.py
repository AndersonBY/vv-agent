"""Pure call facts preserve SDK evaluation points and Python approval equality."""

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest

from vv_agent.kernel import _call_context as core
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
from vv_agent.kernel._records import Record
from vv_agent.kernel._store import SessionStore


def task():
    return AgentTask(
        prompt_bundle=PromptBundle((PromptSection("instructions", "Work."),)),
        user_prompt="work",
        task_id="task",
        model="m",
        initial_shared_state={"nested": []},
    )


def plan(*, kind="tool", request=None, dependencies=None, operation_id="tool", turn_id="turn"):
    return Record(
        "plan",
        "op_planned",
        "s",
        turn_id,
        operation_id,
        1,
        {
            "op_kind": kind,
            "request": request if request is not None else {"id": "c", "name": "lookup", "arguments": {"nested": [1]}},
            "dependencies": dependencies if dependencies is not None else ["model/2"],
            "idempotency_key": "key",
        },
    )


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_call_facts_import_and_execute_with_only_core(tmp_path, flags):
    source = textwrap.dedent("""
        import importlib.abc
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, sys.argv[1])
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel._call_context import CompiledCallFacts, approved_tool_call, matches_preapproved_call
        from vv_agent.kernel._execution_values import AgentTask
        from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
        from vv_agent.kernel._records import Record
        task = AgentTask(prompt_bundle=PromptBundle((PromptSection("instructions", "Work."),)), user_prompt="work", task_id="task", model="m")
        plan = Record("p", "op_planned", "s", None, "tool", 1, {
            "op_kind": "tool", "dependencies": ["model/2"], "idempotency_key": None,
            "request": {"id": "c", "name": "lookup", "arguments": {"nested": [1]},
                        "tools": [{"function": {"name": "one"}}, {"function": {"name": "one"}}]}})
        facts = CompiledCallFacts(task, plan)
        assert facts.cycle_index == 2 and facts.task_id == "task" and facts.idempotency_key is None
        assert facts.metadata(None) == {"session_id": None, "operation_id": "tool", "attempt": 1,
                                        "session_tool_names": ["one", "one"]}
        call = approved_tool_call(plan)
        call.arguments["nested"].append(2)
        assert plan._payload["request"]["arguments"] == {"nested": [1]}
        assert matches_preapproved_call(SimpleNamespace(id="c", name="lookup", arguments={"value": True}),
                                        tool_call_id="c", tool_name="lookup", arguments={"value": 1})
        assert all(n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.")
                   for n in sys.modules if n.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", source, str(Path(__file__).resolve().parents[1] / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fact_reads_follow_host_mutations_without_recomputing_source():
    current = task()
    record = plan(turn_id=None)
    facts = core.CompiledCallFacts(current, record)
    record.payload["dependencies"][0] = "changed/99"
    current.task_id = "changed"
    record.payload["idempotency_key"] = None
    assert facts.cycle_index == 2
    assert facts.task_id == "changed" and facts.idempotency_key is None
    record.payload["request"]["tools"] = [{"function": {"name": "b"}}, {"function": {"name": "a"}}]
    assert facts.metadata("s")["session_tool_names"] == ["b", "a"]


@pytest.mark.parametrize("cycle", [None, True, "unchanged", 0])
def test_model_cycle_fact_does_not_add_validation(cycle):
    record = plan(kind="model", operation_id="model/bad", request={"metadata": {"vv_session": {"cycle_index": cycle}}})
    assert core.CompiledCallFacts(task(), record).cycle_index is cycle


@pytest.mark.parametrize("failed_field", [None, "id", "name", "arguments", "raises"])
def test_preapproval_retains_duck_matching_and_short_circuit(failed_field):
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.tools.base import ToolContext, is_tool_call_preapproved

    seen = []
    failure = RuntimeError("property error")

    class Call:
        def __bool__(self):
            raise AssertionError("the approved object is checked for None, not truthiness")

        @property
        def id(self):
            seen.append("id")
            return "other" if failed_field == "id" else "c"

        @property
        def name(self):
            seen.append("name")
            if failed_field == "raises":
                raise failure
            return "other" if failed_field == "name" else "lookup"

        @property
        def arguments(self):
            seen.append("arguments")
            return {"value": 2 if failed_field == "arguments" else True}

    call = Call()
    context = object.__new__(ToolContext)
    context.ctx = ExecutionContext(_approved_tool_approval=SimpleNamespace(call=call))
    if failed_field == "raises":
        with pytest.raises(RuntimeError) as caught:
            is_tool_call_preapproved(context, tool_call_id="c", tool_name="lookup", arguments={"value": 1})
        assert caught.value is failure
    else:
        assert is_tool_call_preapproved(context, tool_call_id="c", tool_name="lookup", arguments={"value": 1}) is (
            failed_field is None
        )
    assert seen == (
        ["id"] if failed_field == "id" else ["id", "name"] if failed_field in {"name", "raises"} else ["id", "name", "arguments"]
    )


@pytest.mark.parametrize(
    "cut,expected,events",
    [
        ("approved", KeyError, []),
        ("workspace", RuntimeError, ["workspace"]),
        ("source", IndexError, ["workspace"]),
        ("cycle", ValueError, ["workspace", "run_context"]),
        ("names", KeyError, ["workspace", "run_context", "backend", "child"]),
        ("none", None, ["workspace", "run_context", "backend", "child"]),
    ],
)
def test_real_context_retains_host_order_and_aliases(tmp_path, monkeypatch, cut, expected, events):
    from vv_agent import Agent, RunConfig
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.llm.scripted import ScriptedLLM
    from vv_agent.runtime.cancellation import CancellationToken
    from vv_agent.session import runtime
    from vv_agent.tools import base

    assert runtime.CompiledCallFacts is core.CompiledCallFacts
    assert runtime.approved_tool_call is core.approved_tool_call
    assert base.matches_preapproved_call is core.matches_preapproved_call
    seen = []
    backend = object()
    child = object()

    class Workspace:
        def __fspath__(self):
            seen.append("workspace")
            if cut == "workspace":
                raise RuntimeError("bad workspace")
            return str(tmp_path)

    class ObservedRuntime(runtime.Runtime):
        def run_context(self, tid):
            seen.append("run_context")
            return super().run_context(tid)

        @property
        def workspace_backend(self):
            seen.append("backend")
            return backend

        def child_tasks(self, store, session_id):
            seen.append("child")
            return child

    def no_heartbeat() -> NoReturn:
        raise AssertionError("context construction must not acquire a lease")

    rt = ObservedRuntime(
        Agent("context", "Work."), RunConfig(), ResolvedModelConfig("test", "m", "m", "m", []), ScriptedLLM([]), no_heartbeat
    )
    monkeypatch.setattr(rt.config, "workspace", Workspace())
    current = task()
    live = object()
    current.metadata.update(live=live, nested={"values": []})
    record = plan(request={"id": "c", "name": "sub_task_status", "arguments": {}, "tools": []})
    if cut == "approved":
        record.payload["request"].pop("id")
    elif cut == "source":
        record.payload["dependencies"] = []
    elif cut == "cycle":
        record.payload["dependencies"] = ["model/bad"]
    elif cut == "names":
        record.payload["request"]["tools"] = [{}]
    token = CancellationToken()
    shared = {"live": live}
    if expected is not None:
        with pytest.raises(expected):
            rt.context(
                current,
                record,
                token,
                approved=cut == "approved",
                shared_state=shared,
                store=cast(SessionStore, object()),
                session_id="s",
            )
    else:
        result = rt.context(current, record, token, shared_state=shared, store=cast(SessionStore, object()), session_id="s")
        assert result.ctx is not None and result.ctx.metadata is result.task_metadata
        assert result.task_metadata is not current.metadata
        assert result.task_metadata["nested"] is current.metadata["nested"] and result.task_metadata["live"] is live
        assert result.shared_state is shared and result.ctx.cancellation_token is token
        assert result.workspace_backend is backend and result.sub_task_manager is child
    assert seen == events
