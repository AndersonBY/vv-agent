"""Focused source-shape mutations for the two external operation dispatch leaves."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_repository_contracts.py"
CONTRACTS = runpy.run_path(str(SCRIPT))
check_dispatch_boundaries = CONTRACTS["check_dispatch_boundaries"]
check_repository = CONTRACTS["check_repository"]
DISPATCH = "src/vv_agent/kernel/_driver_dispatch.py"
INVOKER = "src/vv_agent/session/_driver_scope.py"
KERNEL_SOURCE = """class InvokeOperation:
    pass


def dispatch():
    return InvokeOperation()
"""
INVOKER_SOURCE = """def _invoke(scope):
    return scope.runtime.invoker.invoke()
"""


def write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def anchored(root: Path) -> None:
    write(root, DISPATCH, KERNEL_SOURCE)
    write(root, INVOKER, INVOKER_SOURCE)


def test_one_producer_and_one_leaf_are_accepted_with_unrelated_port_calls(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        "src/vv_agent/session/timing.py",
        "class Invoker:\n    def invoke(self): ...\n\nclass ThreadInvoker:\n    def invoke(self): ...\n",
    )
    write(
        tmp_path,
        "src/vv_agent/session/recovery.py",
        "def recover(provider):\n"
        "    provider.query('handle')\n"
        "    provider.cancel('handle')\n"
        "    commit('records')\n"
        "    wake('session')\n",
    )
    assert check_dispatch_boundaries(tmp_path) == []


def test_duplicate_construction_at_the_producer_is_rejected(tmp_path):
    anchored(tmp_path)
    write(tmp_path, DISPATCH, KERNEL_SOURCE.replace("return InvokeOperation()", "return InvokeOperation(), InvokeOperation()"))
    errors = check_dispatch_boundaries(tmp_path)
    assert any("expected exactly one InvokeOperation construction" in error for error in errors)
    assert any("found 2 total and 2 at the producer" in error for error in errors)


@pytest.mark.parametrize("bypass", ["direct", "import_alias", "module_alias", "assignment_alias"])
def test_operation_construction_bypasses_are_rejected(tmp_path, bypass):
    anchored(tmp_path)
    source = {
        "direct": "from vv_agent.kernel._driver_dispatch import InvokeOperation\n\ndef bypass():\n    return InvokeOperation()\n",
        "import_alias": (
            "from vv_agent.kernel._driver_dispatch import InvokeOperation as Request\n\ndef bypass():\n    return Request()\n"
        ),
        "module_alias": (
            "from vv_agent.kernel import _driver_dispatch as dispatch\n\ndef bypass():\n    return dispatch.InvokeOperation()\n"
        ),
        "assignment_alias": (
            "from vv_agent.kernel import _driver_dispatch as dispatch\n"
            "Request = dispatch.InvokeOperation\n"
            "\ndef bypass():\n"
            "    return Request()\n"
        ),
    }[bypass]
    write(tmp_path, "src/vv_agent/session/bypass.py", source)
    errors = check_dispatch_boundaries(tmp_path)
    assert any("InvokeOperation construction outside" in error for error in errors)


def test_construction_in_another_core_function_does_not_count_as_the_producer(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        DISPATCH,
        "class InvokeOperation: pass\n\ndef dispatch():\n    return None\n\ndef recovery():\n    return InvokeOperation()\n",
    )
    errors = check_dispatch_boundaries(tmp_path)
    assert any("found 1 total and 0 at the producer" in error for error in errors)
    assert any(f"{DISPATCH}:7 in recovery: InvokeOperation construction outside" in error for error in errors)


def test_class_method_named_dispatch_does_not_count_as_the_producer(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        DISPATCH,
        "class InvokeOperation: pass\n\n"
        "def dispatch():\n"
        "    return None\n\n"
        "class Other:\n"
        "    def dispatch(self):\n"
        "        return InvokeOperation()\n",
    )
    errors = check_dispatch_boundaries(tmp_path)
    assert any("found 1 total and 0 at the producer" in error for error in errors)
    assert any(f"{DISPATCH}:8 in dispatch: InvokeOperation construction outside" in error for error in errors)


def test_construction_in_conditionally_defined_function_is_still_rejected(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        "src/vv_agent/session/bypass.py",
        "from vv_agent.kernel._driver_dispatch import InvokeOperation\n\n"
        "if True:\n"
        "    def bypass():\n"
        "        return InvokeOperation()\n",
    )
    errors = check_dispatch_boundaries(tmp_path)
    assert any("in bypass: InvokeOperation construction outside" in error for error in errors)


@pytest.mark.parametrize("bypass", ["direct", "captured_aliases"])
def test_invoker_bypasses_are_rejected(tmp_path, bypass):
    anchored(tmp_path)
    source = {
        "direct": "def bypass(scope):\n    return other.runtime.invoker.invoke()\n",
        "captured_aliases": "def bypass(scope):\n"
        "    runtime = scope.runtime\n"
        "    invoker = runtime.invoker\n"
        "    invoke = invoker.invoke\n"
        "    return invoke()\n",
    }[bypass]
    write(tmp_path, "src/vv_agent/session/bypass.py", source)
    errors = check_dispatch_boundaries(tmp_path)
    assert any(".invoker.invoke call outside" in error for error in errors)


def test_invoker_bypass_matches_a_renamed_receiver_in_another_file(tmp_path):
    anchored(tmp_path)
    write(tmp_path, "src/vv_agent/session/bypass.py", "def bypass(cfg):\n    return cfg.invoker.invoke()\n")
    errors = check_dispatch_boundaries(tmp_path)
    assert any("src/vv_agent/session/bypass.py:2 in bypass: .invoker.invoke call outside" in error for error in errors)


def test_captured_invoker_alias_at_the_designated_leaf_is_one_actual_call(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        INVOKER,
        "def _invoke(scope):\n    runtime = scope.runtime\n    invoker = runtime.invoker\n    return invoker.invoke()\n",
    )
    assert check_dispatch_boundaries(tmp_path) == []


def test_invoker_call_in_another_session_function_does_not_count_as_the_leaf(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        INVOKER,
        "def _invoke(scope):\n    return None\n\ndef _bypass(scope):\n    return scope.runtime.invoker.invoke()\n",
    )
    errors = check_dispatch_boundaries(tmp_path)
    assert any("found 1 total and 0 at the leaf" in error for error in errors)
    assert any(f"{INVOKER}:5 in _bypass: .invoker.invoke call outside" in error for error in errors)


def test_class_method_named_invoke_does_not_count_as_the_leaf(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        INVOKER,
        "def _invoke(scope):\n"
        "    return None\n\n"
        "class Other:\n"
        "    def _invoke(self, scope):\n"
        "        return scope.invoker.invoke()\n",
    )
    errors = check_dispatch_boundaries(tmp_path)
    assert any("found 1 total and 0 at the leaf" in error for error in errors)
    assert any(f"{INVOKER}:6 in _invoke: .invoker.invoke call outside" in error for error in errors)


def test_conditionally_defined_invoker_function_is_still_rejected(tmp_path):
    anchored(tmp_path)
    write(
        tmp_path,
        "src/vv_agent/session/bypass.py",
        "if True:\n    def bypass(scope):\n        return scope.runtime.invoker.invoke()\n",
    )
    errors = check_dispatch_boundaries(tmp_path)
    assert any("in bypass: .invoker.invoke call outside" in error for error in errors)


def test_missing_expected_sources_are_reported_instead_of_skipped(tmp_path):
    errors = check_dispatch_boundaries(tmp_path)
    assert f"{DISPATCH}: missing dispatch boundary source" in errors
    assert f"{INVOKER}: missing invoker boundary source" in errors


def test_missing_expected_definitions_are_reported(tmp_path):
    write(tmp_path, DISPATCH, "class InvokeOperation: pass\n")
    write(tmp_path, INVOKER, "def other(scope): pass\n")
    errors = check_dispatch_boundaries(tmp_path)
    assert any("expected one dispatch definition, found 0" in error for error in errors)
    assert any("expected one _invoke definition, found 0" in error for error in errors)
    assert any("expected exactly one InvokeOperation construction" in error for error in errors)
    assert any("expected exactly one .invoker.invoke call" in error for error in errors)


def test_repository_gate_does_not_skip_absent_entrypoints(tmp_path):
    write(
        tmp_path,
        "docs/session-kernel-capability-matrix.md",
        "| Capability ID | Tests |\n| --- | --- |\n| `sample` | `tests/test_sample.py::test_sample` |\n",
    )
    write(
        tmp_path,
        "tests/test_sample.py",
        'import pytest\n\n@pytest.mark.capability("sample")\ndef test_sample():\n    pass\n',
    )
    errors = check_repository(tmp_path)
    assert f"{DISPATCH}: missing dispatch boundary source" in errors
    assert f"{INVOKER}: missing invoker boundary source" in errors
