"""Budget values validate without the outer SDK and retain public type identity."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_budget_values_import_and_validate_without_outer_sdk(tmp_path, site_flags):
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
        from vv_agent.kernel import _budget_values as core
        from vv_agent.kernel._canonical_json import MAX_WIRE_INTEGER, canonical_json_bytes

        assert Path(core.__file__) == root / "src/vv_agent/kernel/_budget_values.py"
        assert core.MAX_WIRE_INTEGER == MAX_WIRE_INTEGER == 9007199254740991
        fixture = json.loads((root / "tests/fixtures/parity/run_budget.json").read_text())
        for name, value_type in (
            ("limits", core.RunBudgetLimits),
            ("snapshot", core.BudgetUsageSnapshot),
            ("exhaustion", core.BudgetExhaustion),
        ):
            payload = fixture["wire_examples"][name]
            value = value_type.from_dict(payload)
            assert value.to_dict() == payload, name
            assert canonical_json_bytes(value.to_dict()) == canonical_json_bytes(payload), name

        for case in fixture["invalid_cases"]:
            try:
                core.RunBudgetLimits.from_dict(case["limits"])
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(case["name"])

        assert core.RunBudgetLimits(max_total_tokens=MAX_WIRE_INTEGER).max_total_tokens == MAX_WIRE_INTEGER
        assert core.RunBudgetLimits().unavailable_metric_policy is core.UnavailableMetricPolicy.CONTINUE_AND_MARK
        assert not core.RunBudgetLimits().has_limits
        try:
            core.HostCost(unit="credits", amount_microunits=True)
        except ValueError as exc:
            assert str(exc) == "host cost amount_microunits must be between 0 and 9007199254740991"
        else:
            raise AssertionError("boolean cost must remain invalid")

        class Meter:
            def read(self):
                raise AssertionError("pure protocol validation must not sample a meter")

        assert isinstance(Meter(), core.HostCostMeter)
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._budget_values", "vv_agent.kernel._canonical_json"
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


def test_budget_values_keep_actual_outer_and_root_bindings():
    import vv_agent
    from vv_agent import budget
    from vv_agent.kernel import _budget_values as core

    names = set(budget.__all__) - {"BudgetEvaluator", "MAX_WIRE_INTEGER"}
    for name in names:
        value_type = getattr(core, name)
        assert value_type.__module__ == core.__name__, name
        assert getattr(budget, name) is value_type, name
        assert getattr(vv_agent, name) is value_type, name
    assert budget.MAX_WIRE_INTEGER == core.MAX_WIRE_INTEGER
    assert budget.BudgetEvaluator.__module__ == budget.__name__
    assert budget.ModelCallBudgetExhausted.__module__ == budget.__name__
    assert not hasattr(core, "BudgetEvaluator")
    assert not hasattr(core, "ModelCallBudgetExhausted")
