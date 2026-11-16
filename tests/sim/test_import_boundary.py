"""Default Sim scenarios import with only the core and three accepted host leaves."""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
import importlib
import importlib.abc
import json
import sys
import tempfile
from pathlib import Path

source, root, purelib = sys.argv[1:]
sys.path[:0] = [source, root, str(Path(root) / "tests"), purelib]
allowed = {
    "vv_agent",
    "vv_agent.kernel",
    "vv_agent._jsonschema",
    "vv_agent._memory_settings",
    "vv_agent._token_counting",
}

class ForbiddenSDKImport(AssertionError):
    pass

blocked = []

class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vv_agent" or fullname.startswith("vv_agent."):
            if fullname not in allowed and not fullname.startswith("vv_agent.kernel."):
                blocked.append(fullname)
                raise ForbiddenSDKImport(f"forbidden SDK import: {fullname}")

core_only = CoreOnly()
sys.meta_path.insert(0, core_only)
try:
    importlib.import_module("vv_agent.session.runtime")
except ForbiddenSDKImport as error:
    assert error.args == ("forbidden SDK import: vv_agent.session",)
else:
    raise AssertionError("forbidden SDK import was not blocked")
assert blocked == ["vv_agent.session"]

from vv_agent import _jsonschema, _memory_settings, _token_counting
from tests.sim.scenarios import SCENARIOS, run_scenario

assert callable(run_scenario)
assert len(SCENARIOS) == 21

boundary_modules = sorted(name for name in sys.modules if name.startswith("vv_agent"))
assert all(name in allowed or name.startswith("vv_agent.kernel.") for name in boundary_modules)
sys.meta_path.remove(core_only)

# S1 content adapters are an SDK-capable host dependency after the import gate.
from tests.sim.world import SimWorld

with tempfile.TemporaryDirectory() as directory:
    for scenario in SCENARIOS:
        with SimWorld(Path(directory) / f"{scenario}.sqlite", seed=17) as world:
            summary = run_scenario(world, scenario)
        assert summary["backend"] == "sqlite"
        assert summary["calls"]

print(json.dumps({
    "blocked_imports": blocked,
    "boundary_modules": len(boundary_modules),
    "leaves": sorted(name for name in boundary_modules if name in allowed),
    "kernel_modules": len([name for name in boundary_modules if name.startswith("vv_agent.kernel.")]),
    "scenarios": len(SCENARIOS),
}))
"""


def test_default_simulation_scenarios_import_without_non_core_sdk(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            PROBE,
            str(ROOT / "src"),
            str(ROOT),
            sysconfig.get_paths()["purelib"],
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["blocked_imports"] == ["vv_agent.session"]
    assert evidence["boundary_modules"] > 0
    assert {"vv_agent._jsonschema", "vv_agent._memory_settings", "vv_agent._token_counting"} <= set(evidence["leaves"])
    assert evidence["kernel_modules"] > 0
    assert evidence["scenarios"] == 21
