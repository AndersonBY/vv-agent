"""After-cycle values validate and capture state without loading the SDK."""

import json
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.kernel import _cycle_values as core
from vv_agent.kernel._execution_values import CycleRecord, TaskTokenUsage, TaskTokenUsageTotals
from vv_agent.kernel._message_values import Message


def test_cycle_values_without_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[1]
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
        from vv_agent.kernel import _cycle_values as core
        from vv_agent.kernel._execution_values import CycleRecord, TaskTokenUsage
        from vv_agent.kernel._message_values import Message
        import vv_agent

        assert Path(core.__file__) == root / "src/vv_agent/kernel/_cycle_values.py"
        assert Path(vv_agent.__file__) == root / "src/vv_agent/__init__.py"
        assert not hasattr(core, "AfterCycleHookManager")
        assert not hasattr(core, "AfterCycleHook")
        fixture = json.loads((root / "tests/fixtures/parity/after_cycle_hook.json").read_text())
        assert core.AFTER_CYCLE_CONTROL_SCHEMA == fixture["permission_state"]["schema_version"]
        for case in fixture["invalid_decisions"]:
            raw = case["decision"]
            try:
                stop = core.AfterCycleStop(**raw["stop"]) if raw["stop"] else None
                core.AfterCycleDecision(**(raw | {"stop": stop}))
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(case["name"])
        state = {}
        assert core.persist_after_cycle_disallowed_tools(state, ("z", "a")) == ("a", "z")
        assert core.read_after_cycle_disallowed_tools(state) == ("a", "z")
        snapshot = core.AfterCycleSnapshot.capture(
            task_id="task", cycle_index=1, max_cycles=3,
            cycle=CycleRecord(1, "answer"), messages=[Message("user", "question")],
            shared_state=state, cumulative_token_usage=TaskTokenUsage(),
            available_tool_names=["z"], disallowed_tool_names=["a"],
            native_outcome=core.NativeCycleOutcome("completed", "no_tool_finish", steer_allowed=True),
        )
        assert snapshot.remaining_cycles == 2
        assert snapshot.native_outcome.kind is core.NativeCycleOutcomeKind.COMPLETED
        assert {name for name in sys.modules if name.startswith("vv_agent")}
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_cycle_bindings_are_actual_core_objects():
    import vv_agent
    from vv_agent import runtime
    from vv_agent.runtime import lifecycle

    for name in (
        "AfterCycleAction",
        "AfterCycleDecision",
        "AfterCycleSnapshot",
        "AfterCycleStop",
        "NativeCycleOutcome",
        "NativeCycleOutcomeKind",
        "AfterCycleHookError",
        "AFTER_CYCLE_CONTROL_SCHEMA",
        "AFTER_CYCLE_CONTROL_STATE_KEY",
        "read_after_cycle_disallowed_tools",
        "persist_after_cycle_disallowed_tools",
    ):
        assert getattr(lifecycle, name) is getattr(core, name), name
        if name in vv_agent.__all__:
            assert getattr(vv_agent, name) is getattr(core, name), name
    assert runtime.AfterCycleStop is core.AfterCycleStop
    assert lifecycle.AfterCycleHookManager.__module__ == lifecycle.__name__
    assert lifecycle.AfterCycleHook.__module__ == lifecycle.__name__


@pytest.mark.parametrize("usage_type", [TaskTokenUsage, TaskTokenUsageTotals])
def test_snapshot_copies_and_control_state_mutations(usage_type):
    cycle = CycleRecord(1, "answer")
    messages = [Message("user", "question", metadata={"nested": []})]
    state = {"nested": {"values": []}}
    available = ["visible"]
    denied = ["hidden"]
    native = core.NativeCycleOutcome(core.NativeCycleOutcomeKind.CONTINUE)
    usage = usage_type(input_tokens=3)
    snapshot = core.AfterCycleSnapshot.capture(
        task_id="task",
        cycle_index=4,
        max_cycles=3,
        cycle=cycle,
        messages=messages,
        shared_state=state,
        cumulative_token_usage=usage,
        available_tool_names=available,
        disallowed_tool_names=denied,
        native_outcome=native,
    )
    assert snapshot.remaining_cycles == 0
    assert snapshot.native_outcome is native
    assert type(snapshot.cumulative_token_usage) is TaskTokenUsageTotals
    assert snapshot.cumulative_token_usage is not usage
    snapshot.cycle.assistant_message = "changed copy"
    snapshot.messages[0].metadata["nested"].append("copy")
    snapshot.shared_state["nested"]["values"].append("copy")
    snapshot.cumulative_token_usage.input_tokens = 9
    with pytest.raises(TypeError):
        cast(Any, snapshot.shared_state)["new"] = True
    with pytest.raises(FrozenInstanceError):
        cast(Any, snapshot).remaining_cycles = 1
    assert cycle.assistant_message == "answer"
    assert messages[0].metadata == {"nested": []}
    assert state == {"nested": {"values": []}}
    assert usage.input_tokens == 3
    assert snapshot.available_tool_names == ("visible",)
    assert snapshot.disallowed_tool_names == ("hidden",)
    available.append("later")
    denied.append("later")
    assert snapshot.available_tool_names == ("visible",)
    assert snapshot.disallowed_tool_names == ("hidden",)

    assert core.persist_after_cycle_disallowed_tools(state, ()) == ()
    assert core.AFTER_CYCLE_CONTROL_STATE_KEY not in state
    assert core.persist_after_cycle_disallowed_tools(state, ("\ue000", "😀", "a")) == ("a", "😀", "\ue000")
    assert core.persist_after_cycle_disallowed_tools(state, ("a", "b")) == ("a", "b", "😀", "\ue000")
    assert state["nested"] == {"values": []}
    assert core.read_after_cycle_disallowed_tools(state) == ("a", "b", "😀", "\ue000")
    before = json.dumps(state, ensure_ascii=False)
    state[core.AFTER_CYCLE_CONTROL_STATE_KEY]["disallowed_tools"].reverse()
    malformed = json.dumps(state, ensure_ascii=False)
    with pytest.raises(core.AfterCycleHookError, match="sorted and unique") as error:
        core.persist_after_cycle_disallowed_tools(state, ("c",))
    assert error.value.code == "after_cycle_control_state_invalid"
    assert json.dumps(state, ensure_ascii=False) == malformed != before
