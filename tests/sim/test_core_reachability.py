"""Executable reachability gate for the P1 core execution spine and public values.

The fixed execution registry below is the small set of cross-module functions that
existing SimWorld participants actually invoke.  It is not a scan of private
helpers.  ``sys.setprofile`` observes exact ``__code__`` identities while seven
fixed SQLite scenarios and one real projection failure/retry participant run in a
fresh SDK-free process.  New projection evidence is required because SimWorld did
not previously have a projection participant.

The central public API manifest is checked separately in an SDK-enabled process.
Only explicit core-origin callables are counted: value methods are owned by real
SDK-free producers, SessionStore backoff protocols map to actual SQLStore calls,
``child_delivery`` maps to natural SimWorld execution, and ``HostCostMeter.read``
is a host port outside the core default algorithm.  A newly introduced core-origin
contract callable without an ownership entry fails this gate.  An import alone is
never evidence that a callable ran.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest

from tests.support.contract import fixture_root

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
PURELIB = Path(sysconfig.get_paths()["purelib"])
CONTRACT_PUBLIC_API = fixture_root() / "public_api.json"

FIXED_SCENARIOS = (
    "accepted",
    "children",
    "factory_not_ready",
    "future_input",
    "tools",
    "unknown_unsafe",
    "context_compacted",
)

EXECUTION_ENTRY_FUNCTIONS = (
    ("vv_agent.kernel._drive", "drive_core"),
    ("vv_agent.kernel._driver", "CoreDriver.load_state"),
    ("vv_agent.kernel._driver", "CoreDriver.refresh"),
    ("vv_agent.kernel._driver", "CoreDriver.bind_turn"),
    ("vv_agent.kernel._driver", "CoreDriver.commit"),
    ("vv_agent.kernel._reducer", "Fold.fork"),
    ("vv_agent.kernel._reducer", "Fold.extend"),
    ("vv_agent.kernel._reducer", "Fold.snapshot"),
    ("vv_agent.kernel._driver_recovery", "step"),
    ("vv_agent.kernel._driver_recovery", "close"),
    ("vv_agent.kernel._driver_dispatch", "dispatch"),
    ("vv_agent.kernel._driver_inputs", "apply_input"),
    ("vv_agent.kernel._driver_inputs", "start_turn"),
    ("vv_agent.kernel._driver_planning", "plan"),
    ("vv_agent.kernel._driver_planning", "model_plan"),
    ("vv_agent.kernel._driver_outcomes", "completed_steps"),
    ("vv_agent.kernel._driver_outcomes", "accepted_steps"),
    ("vv_agent.kernel._driver_outcomes", "unknown_steps"),
    ("vv_agent.kernel._driver_outcomes", "parked_steps"),
    ("vv_agent.kernel._sql", "SQLSessionTx.create"),
    ("vv_agent.kernel._sql", "SQLSessionTx.push"),
    ("vv_agent.kernel._sql", "SQLSessionTx.append"),
    ("vv_agent.kernel._sql", "SQLSessionTx.ack"),
    ("vv_agent.kernel._sql", "SQLStore.acquire"),
    ("vv_agent.kernel._sql", "SQLStore.renew"),
    ("vv_agent.kernel._sql", "SQLStore.release"),
    ("vv_agent.kernel._sql", "SQLStore.defer_drive"),
    ("vv_agent.kernel._sql", "SQLStore.defer_projection"),
    ("vv_agent.kernel._sql", "SQLStore.read_state"),
)
CORE_EXECUTION_PATHS = tuple(f"{module}.{qualname}" for module, qualname in EXECUTION_ENTRY_FUNCTIONS)

CHILD_DELIVERY_PATH = "vv_agent.kernel._child_store.child_delivery"
PROMPT_FLATTEN_PATH = "vv_agent.kernel._prompt_values.PromptBundle.flatten"
INTERACTION_PRODUCER = "tests/test_kernel_input_dependencies.py"
CONTRACT_PROFILE_FUNCTIONS = (
    ("vv_agent.kernel._child_store", "child_delivery"),
    ("vv_agent.kernel._prompt_values", "PromptBundle.flatten"),
)

CONTRACT_EXPECTATIONS = {
    "capability:session:child_delivery": (
        CHILD_DELIVERY_PATH,
        CHILD_DELIVERY_PATH,
    ),
    "surface:prompt_bundle.flatten": (PROMPT_FLATTEN_PATH, "scenario"),
    "surface:host_cost_meter.read": (
        "vv_agent.kernel._budget_values.HostCostMeter.read",
        "host_port",
    ),
    "surface:session_SessionStore.defer_drive": (
        "vv_agent.kernel._store.SessionStore.defer_drive",
        "vv_agent.kernel._sql.SQLStore.defer_drive",
    ),
    "surface:session_SessionStore.defer_projection": (
        "vv_agent.kernel._store.SessionStore.defer_projection",
        "vv_agent.kernel._sql.SQLStore.defer_projection",
    ),
    "surface:interaction_AppServerUserInteraction.to_dict": (
        "vv_agent.kernel._interaction_values.AppServerUserInteraction.to_dict",
        INTERACTION_PRODUCER,
    ),
    "surface:interaction_InteractionAnswerSpec.to_dict": (
        "vv_agent.kernel._interaction_values.InteractionAnswerSpec.to_dict",
        INTERACTION_PRODUCER,
    ),
    "surface:interaction_InteractionReplyTarget.to_dict": (
        "vv_agent.kernel._interaction_values.InteractionReplyTarget.to_dict",
        INTERACTION_PRODUCER,
    ),
    "surface:interaction_InteractionAnswer.to_dict": (
        "vv_agent.kernel._interaction_values.InteractionAnswer.to_dict",
        INTERACTION_PRODUCER,
    ),
}
CONTRACT_ORIGINS = {key: origin for key, (origin, _owner) in CONTRACT_EXPECTATIONS.items()}
CONTRACT_OWNERSHIP = {key: owner for key, (_origin, owner) in CONTRACT_EXPECTATIONS.items()}

CORE_PROBE = r"""
import importlib
import importlib.abc
import json
import sys
import tempfile
from pathlib import Path

source, purelib, root, entrypoints_raw, contract_profiles_raw, scenarios_raw = sys.argv[1:]
entrypoints = json.loads(entrypoints_raw)
contract_profiles = json.loads(contract_profiles_raw)
scenarios = json.loads(scenarios_raw)
sys.path[:0] = [source, root, purelib]

allowed = {
    "vv_agent",
    "vv_agent.kernel",
    "vv_agent._jsonschema",
    "vv_agent._memory_settings",
    "vv_agent._token_counting",
}

class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vv_agent" or fullname.startswith("vv_agent."):
            if fullname not in allowed and not fullname.startswith("vv_agent.kernel."):
                raise AssertionError(f"forbidden SDK import: {fullname}")

sys.meta_path.insert(0, CoreOnly())


def resolve(module_name, qualname):
    value = importlib.import_module(module_name)
    for attribute in qualname.split("."):
        value = getattr(value, attribute)
    return value


from tests.sim.scenarios import run_scenario
from tests.sim.world import SimWorld
from vv_agent.kernel._records import InboxItem, SessionSpec

functions = {}
for module_name, qualname in (*entrypoints, *contract_profiles):
    path = f"{module_name}.{qualname}"
    functions[path] = resolve(module_name, qualname)
codes = {function.__code__: key for key, function in functions.items()}
assert len(codes) == len(functions), "registry callables must have distinct __code__ identities"

trace = {key: [] for key in functions}
active_evidence = "startup"


def profiler(frame, event, arg):
    del arg
    if event == "call" and frame.f_code in codes:
        trace[codes[frame.f_code]].append(active_evidence)


def work(store):
    return {(item.kind, item.consumer) for item in store.list_runnable()}


class ExpectedProjectionSinkError(Exception):
    pass


def projection_failure_then_ack(world):
    store = world.connection("projection-host")
    with store.atomic() as tx:
        tx.create(SessionSpec("projection", "sim", "/tmp"), consumers=("events", "other"))
    with store.atomic() as tx:
        tx.push("projection", InboxItem("initial", "user", {"content": "go"}))

    observer = world.connection("projection-observer")
    initial = work(observer)
    assert initial == {("drive", ""), ("project", "events"), ("project", "other")}

    failed_batch = None

    def failing_sink(batch):
        nonlocal failed_batch
        failed_batch = batch
        assert (batch.session_id, batch.consumer, batch.from_seq) == ("projection", "events", 1)
        raise ExpectedProjectionSinkError

    try:
        with store.atomic() as tx:
            failing_sink(tx.consumer_batch("projection", "events"))
    except ExpectedProjectionSinkError:
        store.defer_projection("projection", "events", retry_after_ms=100)

    assert failed_batch is not None
    during = work(observer)
    assert ("project", "events") not in during
    assert {("drive", ""), ("project", "other")} <= during

    world.advance(99)
    assert ("project", "events") not in work(observer)
    world.advance(1)
    due = work(observer)
    assert {("drive", ""), ("project", "events"), ("project", "other")} <= due

    delivered = []

    def successful_sink(batch):
        assert batch is not None
        assert (batch.from_seq, batch.through_seq) == (failed_batch.from_seq, failed_batch.through_seq)
        delivered.extend(row.record.record_id for row in batch.records)

    with store.atomic() as tx:
        retry_batch = tx.consumer_batch("projection", "events")
        successful_sink(retry_batch)
        tx.ack(retry_batch)

    assert delivered == [row.record.record_id for row in failed_batch.records]
    after_ack = work(observer)
    assert ("project", "events") not in after_ack
    assert {("drive", ""), ("project", "other")} <= after_ack


summaries = []
sys.setprofile(profiler)
try:
    with tempfile.TemporaryDirectory() as directory:
        for scenario in scenarios:
            active_evidence = f"scenario:{scenario}"
            with SimWorld(Path(directory) / f"{scenario}.sqlite", seed=17) as world:
                summary = run_scenario(world, scenario)
            assert summary["backend"] == "sqlite"
            assert summary["calls"]
            summaries.append(scenario)

        active_evidence = "projection:failure-then-ack"
        with SimWorld(Path(directory) / "projection.sqlite", seed=17) as world:
            projection_failure_then_ack(world)
finally:
    sys.setprofile(None)

loaded = {name for name in sys.modules if name.startswith("vv_agent")}
assert loaded <= allowed | {name for name in loaded if name.startswith("vv_agent.kernel.")}
leaves = sorted(
    name
    for name in loaded
    if name in {"vv_agent._jsonschema", "vv_agent._memory_settings", "vv_agent._token_counting"}
)
assert set(leaves) == {
    "vv_agent._jsonschema",
    "vv_agent._memory_settings",
    "vv_agent._token_counting",
}

print(
    json.dumps(
        {
            "leaves": leaves,
            "kernel_modules": len([name for name in loaded if name.startswith("vv_agent.kernel.")]),
            "scenarios": summaries,
            "trace": {key: sorted(set(values)) for key, values in trace.items()},
        },
        sort_keys=True,
    )
)
"""


CONTRACT_DISCOVERY_PROBE = r"""
import importlib
import inspect
import json
import sys
from pathlib import Path

source, manifest_path = sys.argv[1:]
sys.path.insert(0, source)


def resolve(path):
    parts = path.split(".")
    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            continue
        for name in parts[split_index:]:
            value = getattr(value, name)
        return value
    raise ModuleNotFoundError(f"No importable module prefix for {path}")


def origin(function):
    return f"{function.__module__}.{function.__qualname__}"


manifest = json.loads(Path(manifest_path).read_text())
assert manifest["contract"] == "vv-agent-public-api-v15"

discovered = {}
for domain in manifest["domains"]:
    for capability in domain["capabilities"]:
        path = capability.get("python")
        if path is None:
            continue
        value = resolve(path)
        if inspect.isfunction(value) and value.__module__.startswith("vv_agent.kernel"):
            key = f"capability:{domain['id']}:{capability['id']}"
            assert key not in discovered, key
            discovered[key] = origin(value)

for surface in manifest["surfaces"]:
    surface_target = surface.get("python_target")
    if surface_target is None:
        continue
    for group in ("members", "protocol_operations", "supporting_operations"):
        for member in surface.get(group, []):
            python = member.get("python")
            if not isinstance(python, dict) or python.get("kind") not in {"method", "function"}:
                continue
            owner = resolve(python.get("target", surface_target))
            value = getattr(owner, python["name"])
            if inspect.isfunction(value) and value.__module__.startswith("vv_agent.kernel"):
                key = f"surface:{surface['id']}.{member['id']}"
                assert key not in discovered, key
                discovered[key] = origin(value)

print(json.dumps({"contract": manifest["contract"], "discovered": discovered}, sort_keys=True))
"""


def _require_exact(label, expected, actual):
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    assert not missing and not unexpected, f"{label}: missing={missing}, unexpected={unexpected}"


def _require_executed(label, expected, trace):
    unresolved = sorted(path for path in expected if path not in trace)
    unexecuted = sorted(path for path in expected if path in trace and not trace[path])
    assert not unresolved, f"{label}: unresolved={unresolved}"
    assert not unexecuted, f"{label}: imported but not executed={unexecuted}"


def _run_probe(script, *args, tmp_path, no_site):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(SOURCE), str(ROOT), str(PURELIB)))
    flags = ["-I", "-B"]
    if no_site:
        flags.insert(1, "-S")
    result = subprocess.run(
        [sys.executable, *flags, "-c", script, *map(str, args)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_core_execution_entries_are_naturally_reached_in_fresh_sdk_free_simulation(tmp_path):
    evidence = _run_probe(
        textwrap.dedent(CORE_PROBE),
        SOURCE,
        PURELIB,
        ROOT,
        json.dumps(EXECUTION_ENTRY_FUNCTIONS),
        json.dumps(CONTRACT_PROFILE_FUNCTIONS),
        json.dumps(FIXED_SCENARIOS),
        tmp_path=tmp_path,
        no_site=True,
    )

    assert tuple(evidence["scenarios"]) == FIXED_SCENARIOS
    assert evidence["kernel_modules"] > 0
    _require_exact(
        "simulation import leaves",
        {
            "vv_agent._jsonschema",
            "vv_agent._memory_settings",
            "vv_agent._token_counting",
        },
        evidence["leaves"],
    )

    trace = evidence["trace"]
    resolved_core = {path: trace[path] for path in CORE_EXECUTION_PATHS if path in trace}
    _require_exact("resolved execution entries", CORE_EXECUTION_PATHS, resolved_core)
    _require_executed("29 core execution entries", CORE_EXECUTION_PATHS, trace)
    _require_executed("contract child_delivery", (CHILD_DELIVERY_PATH,), trace)
    _require_executed("contract PromptBundle.flatten", (PROMPT_FLATTEN_PATH,), trace)

    projection_contexts = trace["vv_agent.kernel._sql.SQLStore.defer_projection"]
    assert "projection:failure-then-ack" in projection_contexts, projection_contexts
    assert "scenario:children" in trace[CHILD_DELIVERY_PATH]
    assert any(context.startswith("scenario:") for context in trace[PROMPT_FLATTEN_PATH])


def test_contract_core_origins_have_explicit_producer_or_execution_ownership(tmp_path):
    evidence = _run_probe(
        textwrap.dedent(CONTRACT_DISCOVERY_PROBE),
        SOURCE,
        CONTRACT_PUBLIC_API,
        tmp_path=tmp_path,
        no_site=False,
    )

    assert evidence["contract"] == "vv-agent-public-api-v15"
    _require_exact("contract core-origin callables", CONTRACT_ORIGINS, evidence["discovered"])
    assert evidence["discovered"] == CONTRACT_ORIGINS
    _require_exact("contract callable ownership", CONTRACT_ORIGINS, CONTRACT_OWNERSHIP)

    for contract_key, owner in CONTRACT_OWNERSHIP.items():
        if owner.startswith("vv_agent.kernel."):
            assert owner in {*CORE_EXECUTION_PATHS, CHILD_DELIVERY_PATH}, (contract_key, owner)


def test_gate_rejects_unresolved_unmapped_and_unexecuted_fixture_entries():
    with pytest.raises(AssertionError, match=r"missing=.*new"):
        _require_exact("fixture resolution", {"existing", "new"}, {"existing"})

    new_contract_callable = "surface:new_surface.new_method"
    with pytest.raises(AssertionError, match=r"unexpected=.*new_surface"):
        _require_exact("fixture ownership", CONTRACT_ORIGINS, {*CONTRACT_OWNERSHIP, new_contract_callable})

    resolved_but_unexecuted = {"vv_agent.kernel._driver.CoreDriver.load_state"}
    with pytest.raises(AssertionError, match="imported but not executed"):
        _require_executed(
            "fixture trace",
            resolved_but_unexecuted,
            {path: [] for path in resolved_but_unexecuted},
        )


def test_gate_rejects_new_core_entry_in_supporting_operations(tmp_path):
    manifest = json.loads(CONTRACT_PUBLIC_API.read_text())
    prompt_surface = next(surface for surface in manifest["surfaces"] if surface["id"] == "prompt_bundle")
    prompt_surface.setdefault("supporting_operations", []).append(
        {
            "id": "to_dict",
            "kind": "operation",
            "python": {
                "kind": "method",
                "name": "to_dict",
                "signature": {
                    "async": False,
                    "parameters": [
                        {
                            "kind": "positional_or_keyword",
                            "name": "self",
                            "required": True,
                        }
                    ],
                },
            },
        }
    )
    fixture_copy = tmp_path / "public_api.json"
    fixture_copy.write_text(json.dumps(manifest))

    evidence = _run_probe(
        textwrap.dedent(CONTRACT_DISCOVERY_PROBE),
        SOURCE,
        fixture_copy,
        tmp_path=tmp_path,
        no_site=False,
    )
    with pytest.raises(AssertionError, match=r"unexpected=.*prompt_bundle.to_dict"):
        _require_exact("contract core-origin callables", CONTRACT_ORIGINS, evidence["discovered"])


def test_gate_rejects_unmapped_surface_function_member(tmp_path):
    manifest = json.loads(CONTRACT_PUBLIC_API.read_text())
    manifest["surfaces"].append(
        {
            "id": "future_core_function",
            "python_target": "vv_agent.kernel._child_store",
            "members": [
                {
                    "id": "child_delivery",
                    "python": {"kind": "function", "name": "child_delivery"},
                }
            ],
        }
    )
    fixture_copy = tmp_path / "public_api.json"
    fixture_copy.write_text(json.dumps(manifest))

    evidence = _run_probe(
        textwrap.dedent(CONTRACT_DISCOVERY_PROBE),
        SOURCE,
        fixture_copy,
        tmp_path=tmp_path,
        no_site=False,
    )
    assert evidence["discovered"]["surface:future_core_function.child_delivery"] == CHILD_DELIVERY_PATH
    with pytest.raises(AssertionError, match=r"unexpected=.*future_core_function.child_delivery"):
        _require_exact("contract core-origin callables", CONTRACT_ORIGINS, evidence["discovered"])


def test_gate_rejects_unmapped_surface_member_python_target_override(tmp_path):
    manifest = json.loads(CONTRACT_PUBLIC_API.read_text())
    manifest["surfaces"].append(
        {
            "id": "future_core_target",
            "python_target": "vv_agent.RunResult",
            "members": [
                {
                    "id": "to_dict",
                    "python": {
                        "kind": "method",
                        "name": "to_dict",
                        "target": "vv_agent.AppServerUserInteraction",
                    },
                }
            ],
        }
    )
    fixture_copy = tmp_path / "public_api.json"
    fixture_copy.write_text(json.dumps(manifest))

    evidence = _run_probe(
        textwrap.dedent(CONTRACT_DISCOVERY_PROBE),
        SOURCE,
        fixture_copy,
        tmp_path=tmp_path,
        no_site=False,
    )
    expected = "vv_agent.kernel._interaction_values.AppServerUserInteraction.to_dict"
    assert evidence["discovered"]["surface:future_core_target.to_dict"] == expected
    with pytest.raises(AssertionError, match=r"unexpected=.*future_core_target.to_dict"):
        _require_exact("contract core-origin callables", CONTRACT_ORIGINS, evidence["discovered"])
