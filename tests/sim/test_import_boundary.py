"""Sim scenarios execute under the strict core-only import guard."""

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
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._sqlite import SQLiteStore
from tests.sim.scenarios import SCENARIOS, run_scenario
from tests.sim.world import SimPostgresStore, SimSQLiteStore, SimWorld

assert callable(run_scenario)
assert len(SCENARIOS) == 22
assert issubclass(SimSQLiteStore, SQLiteStore)
assert issubclass(SimPostgresStore, PostgresStore)


def storage_state(world):
    store = world.connection("observer")
    return {
        sid: world.state_signature(store, sid)
        for sid in store.list_sessions()
    }


def run_round(directory):
    directory.mkdir()
    results = {}
    for scenario in SCENARIOS:
        with SimWorld(Path(directory) / f"{scenario}.sqlite", seed=17) as world:
            summary = run_scenario(world, scenario)
            assert summary["backend"] == "sqlite"
            assert summary["calls"]
            results[scenario] = {
                "summary": summary,
                "storage": storage_state(world),
            }
    return results


with tempfile.TemporaryDirectory() as directory:
    first = run_round(Path(directory) / "first")
    replay = run_round(Path(directory) / "replay")
    assert first == replay

loaded = {name for name in sys.modules if name.startswith("vv_agent")}
assert all(name in allowed or name.startswith("vv_agent.kernel.") for name in loaded)
leaves = sorted(
    name
    for name in loaded
    if name not in {"vv_agent", "vv_agent.kernel"} and not name.startswith("vv_agent.kernel.")
)
assert leaves == [
    "vv_agent._jsonschema",
    "vv_agent._memory_settings",
    "vv_agent._token_counting",
]
sys.meta_path.remove(core_only)

print(json.dumps({
    "blocked_imports": blocked,
    "boundary_modules": len(loaded),
    "leaves": leaves,
    "kernel_modules": len([name for name in loaded if name.startswith("vv_agent.kernel.")]),
    "scenarios": len(SCENARIOS),
    "replayed_scenarios": len(first),
}))
"""


def test_all_simulation_scenarios_execute_under_core_only_guard(tmp_path):
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
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["blocked_imports"] == ["vv_agent.session"]
    assert evidence["boundary_modules"] > 0
    assert evidence["leaves"] == [
        "vv_agent._jsonschema",
        "vv_agent._memory_settings",
        "vv_agent._token_counting",
    ]
    assert evidence["kernel_modules"] > 0
    assert evidence["scenarios"] == 22
    assert evidence["replayed_scenarios"] == 22
