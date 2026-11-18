"""The SDK compiles and consumes the same standard-library policy implementation."""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, NoReturn

import pytest

from vv_agent.kernel import _tool_policy as core


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_tool_policy_imports_without_outer_or_third_party(tmp_path, flags):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        sys.path.insert(0, sys.argv[1])

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel._tool_policy import ToolPolicy, apply_tool_context_policy, merge_tool_policy_layers
        from vv_agent.kernel._cycle_values import AFTER_CYCLE_CONTROL_STATE_KEY, AFTER_CYCLE_CONTROL_SCHEMA
        trace = []
        predicate = lambda name, args: trace.append(name) or True
        live = object()
        metadata = {"live": live, "_vv_agent_allowed_tools": ["a", "b"], "_vv_agent_tool_policy_approval": "always"}
        apply_tool_context_policy(metadata, agent_policy=None,
            run_policy=ToolPolicy(allowed_tools=["b"], approval="never", can_use_tool=predicate),
            shared_state={AFTER_CYCLE_CONTROL_STATE_KEY: {
                "schema_version": AFTER_CYCLE_CONTROL_SCHEMA, "disallowed_tools": ["later"]}})
        assert metadata["_vv_agent_allowed_tools"] == ["b"]
        assert metadata["_vv_agent_tool_policy_approval"] == "always"
        assert metadata["_vv_agent_disallowed_tools"] == ["later"]
        assert metadata["live"] is live and trace == []
        assert metadata["_vv_agent_tool_policy_can_use_tool"]("b", {}) and trace == ["b"]
        assert merge_tool_policy_layers(None, None, None) is None
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).resolve().parents[1] / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_policy_facades_bind_the_actual_core_objects():
    import vv_agent
    from vv_agent import run_config
    from vv_agent.runtime import compiler
    from vv_agent.session import _dispatch_context

    assert vv_agent.ToolPolicy is run_config.ToolPolicy is core.ToolPolicy
    assert vv_agent.ApprovalPolicy is run_config.ApprovalPolicy is core.ApprovalPolicy
    for name in ("CanUseTool", "merge_tool_policies", "merge_tool_policy_layers"):
        assert getattr(run_config, name) is getattr(core, name)
    assert compiler._apply_tool_policy_metadata is core._apply_tool_policy_metadata
    assert _dispatch_context.apply_tool_context_policy is core.apply_tool_context_policy
    assert core.ToolPolicy.__module__ == core.__name__


@pytest.mark.parametrize("stop", [None, "agent", "runner", "run", "raise"])
def test_policy_predicates_remain_lazy_ordered_and_shallow(stop):
    trace = []
    failure = RuntimeError("predicate failure")

    def predicate(layer):
        def check(name, arguments):
            trace.append((layer, name))
            assert "private" not in arguments
            arguments["private"] = layer
            arguments["nested"].append(layer)
            if stop == "raise":
                raise failure
            return stop != layer

        return check

    merged = core.merge_tool_policy_layers(
        *(core.ToolPolicy(can_use_tool=predicate(layer)) for layer in ("agent", "runner", "run"))
    )
    assert trace == [] and merged is not None and merged.can_use_tool is not None
    arguments = {"nested": []}
    if stop == "raise":
        with pytest.raises(RuntimeError) as caught:
            merged.can_use_tool("tool", arguments)
        assert caught.value is failure
    else:
        assert merged.can_use_tool("tool", arguments) is (stop is None)
    expected = (
        ["agent"] if stop in {"agent", "raise"} else ["agent", "runner"] if stop == "runner" else ["agent", "runner", "run"]
    )
    assert trace == [(layer, "tool") for layer in expected]
    assert arguments == {"nested": expected}


def test_policy_defaults_and_metadata_cleanup_preserve_live_values():
    first, second = core.ToolPolicy(), core.ToolPolicy()
    first.disallowed_tools.append("first")
    assert second.disallowed_tools == [] and first.allowed_tools is None and first.approval == "default"
    metadata: dict[str, Any] = dict.fromkeys(core._TASK_TOOL_POLICY_METADATA_KEYS, "stale")
    live = object()
    metadata.update(live=live, _vv_agent_tool_policy_approval="always", _vv_agent_tool_policy_can_use_tool=live)
    core._apply_tool_policy_metadata(metadata, None)
    assert metadata == {"live": live, "_vv_agent_tool_policy_approval": "always", "_vv_agent_tool_policy_can_use_tool": live}
    assert core.merge_tool_policies(None, None) is None
    current = core.ToolPolicy(allowed_tools=[])
    merged = core.merge_tool_policies(core.ToolPolicy(allowed_tools=["a"]), current)
    assert merged is not None and merged.allowed_tools == [] and merged.allowed_tools is not current.allowed_tools


@pytest.mark.parametrize("kind", ["model", "tool"])
def test_real_compiler_and_context_call_the_shared_helpers(tmp_path, monkeypatch, kind):
    from vv_agent import Agent, RunConfig
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.llm.scripted import ScriptedLLM
    from vv_agent.runtime import compiler
    from vv_agent.runtime.cancellation import CancellationToken
    from vv_agent.session import _dispatch_context, runtime
    from vv_agent.session.records import digest, make_record

    compiled, composed = [], []

    def compile_metadata(metadata, policy):
        compiled.append(metadata)
        return core._apply_tool_policy_metadata(metadata, policy)

    def context_policy(metadata, **kwargs):
        composed.append(metadata)
        return core.apply_tool_context_policy(metadata, **kwargs)

    monkeypatch.setattr(compiler, "_apply_tool_policy_metadata", compile_metadata)
    monkeypatch.setattr(_dispatch_context, "apply_tool_context_policy", context_policy)
    agent = Agent("policy", "Work.")
    config = RunConfig(workspace=tmp_path, tool_policy=core.ToolPolicy(allowed_tools=["lookup"], approval="always"))
    resolved = ResolvedModelConfig("test", "m", "m", "m", [])
    task = compiler.AgentCompiler().compile(agent=agent, input="work", run_config=config, resolved=resolved, trace_id="trace")
    assert len(compiled) == 1
    assert task.metadata["_vv_agent_allowed_tools"] == ["lookup"]
    live = object()
    task.metadata.update(live=live, nested={"values": []})
    if kind == "model":
        task.metadata["_vv_agent_tool_policy_approval"] = "malformed but unused for model"
    task.task_id = "turn"
    request = (
        {"metadata": {"vv_session": {"cycle_index": 7}}, "tools": []}
        if kind == "model"
        else {"id": "call", "name": "lookup", "arguments": {"nested": {"value": 1}}}
    )
    plan = make_record(
        "op_planned",
        session_id="session",
        turn_id="turn",
        operation_id="turn/model/primary/2" if kind == "model" else "turn/tool/1",
        attempt=1,
        payload={
            "op_kind": kind,
            "purpose": "primary" if kind == "model" else None,
            "request": request,
            "request_digest": digest(request),
            "context_version": "1",
            "dependencies": [] if kind == "model" else ["turn/model/primary/2"],
            "tool": None,
            "idempotency_key": "key",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "provider",
            "consumed_unknowns": [],
        },
    )

    def no_heartbeat() -> NoReturn:
        raise AssertionError("context construction must not acquire a lease")

    rt = runtime.Runtime(agent, config, resolved, ScriptedLLM([]), no_heartbeat)
    token = CancellationToken()
    shared = {"nested": []}
    context = rt.context(task, plan, token, approved=kind == "tool", shared_state=shared, session_id="session")
    assert len(composed) == (kind == "tool")
    assert context.ctx is not None and context.ctx.metadata is context.task_metadata
    assert context.task_metadata is not task.metadata
    assert context.task_metadata["live"] is live and context.task_metadata["nested"] is task.metadata["nested"]
    assert context.shared_state is shared and context.ctx.cancellation_token is token
    assert context.cycle_index == (7 if kind == "model" else 2)
    if kind == "tool":
        assert context.ctx._approved_tool_approval is not None
        assert context.ctx._approved_tool_approval.call.to_dict() == request
        assert context.task_metadata["_vv_agent_tool_policy_approval"] == "always"
    else:
        assert context.task_metadata["_vv_agent_tool_policy_approval"] == "malformed but unused for model"
