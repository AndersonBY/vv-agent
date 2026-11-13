"""Pure thresholds and the sole MemoryManager definition projection."""

import inspect
import subprocess
import sys
import textwrap
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import pytest

from vv_agent import _token_counting
from vv_agent.kernel import _compaction_threshold as core
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.memory import MemoryManager
from vv_agent.microcompaction import MicrocompactionPolicy


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_threshold_core_import_and_coercion_without_outer_sdk(tmp_path, site_flags):
    root = Path(__file__).resolve().parents[1]
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
        from vv_agent.kernel._compaction_threshold import compute_compaction_threshold as threshold

        assert threshold(configured_threshold=" 128000 ", model_context_window=64000.9,
                         reserved_output_tokens="8000", autocompact_buffer_tokens=13000) == 43000
        assert threshold(configured_threshold=True, model_context_window=" 100 ",
                         reserved_output_tokens=False, autocompact_buffer_tokens=-10) == 100
        assert threshold(configured_threshold="bad", model_context_window=None,
                         reserved_output_tokens=0, autocompact_buffer_tokens=0) == 0
        for value, error in ((float("nan"), ValueError), (float("inf"), OverflowError)):
            for name in ("configured_threshold", "model_context_window",
                         "reserved_output_tokens", "autocompact_buffer_tokens"):
                inputs = dict(configured_threshold=100, model_context_window=100,
                              reserved_output_tokens=0, autocompact_buffer_tokens=0)
                inputs[name] = value
                try:
                    threshold(**inputs)
                except error:
                    pass
                else:
                    raise AssertionError((name, value))
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._compaction_threshold"
        }
    """)
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_threshold_actual_consumers_share_core_functions():
    from vv_agent.memory import manager

    assert manager.compute_compaction_threshold is core.compute_compaction_threshold
    assert _token_counting._coerce_positive_int is core._coerce_positive_int


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"compact_threshold": 64_000, "language": "en-US", "artifact_scope": ("run", "turn")},
        {"microcompaction_policy": MicrocompactionPolicy(keep_recent_cycles=4).to_dict()},
    ],
    ids=["defaults", "sparse", "wire-policy"],
)
def test_memory_settings_projection_restores_defaults_and_sparse_values(settings):
    manager = MemoryManager(**settings)
    projected = manager._definition_settings()
    excluded = {
        "protected_start",
        "workspace_backend",
        "summary_callback",
        "session_memory",
        "microcompaction_policy",
        "recovery_tool_available",
    }
    declared = fields(MemoryManager)
    assert len(declared) == 25
    assert MemoryManager.__slots__ == tuple(f.name for f in declared)
    parameters = inspect.signature(MemoryManager).parameters
    assert all(parameters[f.name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for f in declared[:-1])
    assert parameters["protected_start"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not hasattr(manager, "__dict__")
    assert list(projected) == [f.name for f in declared if f.name not in excluded] + ["microcompaction_policy"]
    assert projected["microcompaction_policy"] == manager.microcompaction_policy.to_dict()
    assert projected["tool_result_retentions"] is manager.tool_result_retentions
    assert MemoryManager(**projected) == manager
    assert manager._definition_settings() is not projected


def test_runtime_definition_reads_current_projected_settings():
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.tools.metadata import ToolResultRetention

    driver = SessionDriver()
    try:
        runtime = driver.runtime(Agent("settings", "Work."), RunConfig(model_provider=ScriptedModelProvider.new("test", "m", [])))
        task = runtime.compile("go", "settings/turn/initial")
        original = runtime.definition(task)
        original_digest = runtime.definition_digest(task)
        manager = runtime.memory_manager
        manager.compact_threshold += 1
        manager.microcompaction_policy = MicrocompactionPolicy(keep_recent_cycles=4)
        manager.tool_result_retentions["custom"] = ToolResultRetention.PRESERVE
        manager.protected_start = 1
        manager.recovery_tool_available = True
        current = runtime.definition(task)
        assert current["memory_settings"] == manager._definition_settings()
        assert current["memory_settings"]["compact_threshold"] == original["memory_settings"]["compact_threshold"] + 1
        assert runtime.definition_digest(task) == sha256(canonical_json_bytes(current)).hexdigest() != original_digest
        assert original["memory_settings"]["tool_result_retentions"] == {}
    finally:
        driver.close()
