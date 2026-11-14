"""The complete execution DTO implementation imports and validates inside core."""

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("facade_first", [False, True], ids=["core-only", "existing-types-first"])
def test_execution_values_without_site_packages(tmp_path, facade_first):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        root = Path(sys.argv[1])
        facade_first = sys.argv[2] == "True"
        sys.path.insert(0, str(root / "src"))

        class ExecutionOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                if facade_first and fullname in {"vv_agent.types", "vv_agent.core_types"}:
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, ExecutionOnly())
        if facade_first:
            import vv_agent.types
        from vv_agent.kernel import _execution_values as core
        from vv_agent.kernel._tool_values import ToolCall
        import vv_agent

        assert Path(core.__file__) == root / "src/vv_agent/kernel/_execution_values.py"
        assert Path(vv_agent.__file__) == root / "src/vv_agent/__init__.py"
        assert "Agent" not in vars(vv_agent)
        fixtures = root / "tests/fixtures/parity"
        definitions = json.loads((fixtures / "run_definition.json").read_text())
        for case in definitions["golden_cases"]:
            payload = case["definition"]["task"]
            task = core.AgentTask.from_dict(payload)
            assert task.to_dict() == payload, case["name"]
            assert type(task) is core.AgentTask
            assert all(type(child) is core.SubAgentConfig for child in task.sub_agents.values())

        usage = json.loads((fixtures / "token_usage.json").read_text())
        for case in usage["task_aggregation_cases"]:
            summary = core.TaskTokenUsage()
            for payload in case["model_calls"]:
                summary.add_model_call(core.ModelCallRecord.from_dict(payload))
            assert core.TaskTokenUsage.from_dict(summary.to_dict()) == summary
        assert core.TokenUsage(input_tokens=1 << 64).input_tokens == 1 << 64
        try:
            core.TaskTokenUsageTotals(input_tokens=1 << 53)
        except ValueError as exc:
            assert str(exc) == "token usage totals must be JSON-safe integers"
        else:
            raise AssertionError("safe-integer accounting totals must stay bounded")

        payload = json.loads((fixtures / "result_public.json").read_text())["agent_result"]
        result = core.AgentResult.from_dict(payload)
        assert result.to_dict() == payload
        assert type(result) is core.AgentResult
        assert type(result.token_usage) is core.TaskTokenUsage
        assert all(type(cycle) is core.CycleRecord for cycle in result.cycles)
        try:
            core.AgentResult.from_dict(payload | {"future_field": True})
        except ValueError as exc:
            assert "unknown=['future_field']" in str(exc)
        else:
            raise AssertionError("result wire must stay closed")

        response = core.LLMResponse("done", [ToolCall("call", "tool", {})])
        assert response.tool_calls[0].to_dict()["name"] == "tool"
        assert response.raw is not core.LLMResponse("next").raw
        assert core.SubTaskRequest("child", "work").metadata == {}
        outcome = core.SubTaskOutcome("child-task", "child", core.AgentStatus.COMPLETED)
        assert outcome.to_dict()["status"] == "completed"
        assert core.TOKEN_USAGE_SCHEMA_VERSION == "vv-agent.token-usage.v1"
        assert core.MODEL_CALL_SCHEMA_VERSION == "vv-agent.model-call.v2"
        assert core.TASK_TOKEN_USAGE_SCHEMA_VERSION == "vv-agent.task-token-usage.v3"
        assert {
            name for name in sys.modules if name.startswith("vv_agent")
            and name not in {"vv_agent", "vv_agent.kernel"}
            and not name.startswith("vv_agent.kernel.")
        } == ({"vv_agent.types", "vv_agent.core_types"} if facade_first else set())
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root), str(facade_first)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_types_and_root_bindings_are_actual_execution_objects():
    import vv_agent
    from vv_agent import types
    from vv_agent.kernel import _execution_values as core
    from vv_agent.kernel._message_values import _MESSAGE_FIELDS

    for name, value in vars(core).items():
        if not name.startswith("_") and getattr(value, "__module__", None) == core.__name__:
            assert getattr(types, name) is value, name
            if name in vv_agent.__all__:
                assert getattr(vv_agent, name) is value, name
    for name in ("NoToolPolicy", "TOKEN_USAGE_SCHEMA_VERSION", "MODEL_CALL_SCHEMA_VERSION", "TASK_TOKEN_USAGE_SCHEMA_VERSION"):
        assert getattr(types, name) is getattr(core, name), name
    for name in (
        "_aggregate_cache_usage",
        "_trim_portable_whitespace",
        "_validate_no_tool_policy",
        "_agent_result_error_text",
        "_last_assistant_output",
    ):
        assert getattr(types, name) is getattr(core, name), name
    assert types._MESSAGE_FIELDS is _MESSAGE_FIELDS
    assert importlib.import_module("vv_agent.types").AgentResult is core.AgentResult
    assert core.AgentResult.__module__ == core.__name__
