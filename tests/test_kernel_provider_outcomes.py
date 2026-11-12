"""Pure receipt checks retain validation order, causes and actual facade identity."""

import json
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from vv_agent.kernel import _provider_outcomes as core
from vv_agent.kernel import _tool_result_checks as checks
from vv_agent.kernel._records import Record
from vv_agent.kernel._tool_values import ToolExecutionResult, ToolResultStatus

_ROOT = Path(__file__).resolve().parents[1]


def _tool_plan():
    history = json.loads((_ROOT / "tests/session/fixtures/step_cost_release_histories.json").read_text())[-1]
    records = [Record.parse(body.encode()) for body in history["records"]]
    return next(r for r in records if r.kind == "op_planned" and r._payload["op_kind"] == "tool")


def _handle(plan):
    return {
        "kind": "provider",
        "provider": plan._payload["provider_binding"],
        "job_id": "receipt",
        "operation_id": plan.operation_id,
        "attempt": plan.attempt,
        "request_digest": plan._payload["request_digest"],
        "evidence": "accepted",
        "query_ref": "receipt",
        "cancel_ref": None,
    }


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("layer", ["tool", "provider"])
def test_receipt_core_imports_and_validates_without_outer_dependencies(tmp_path, flags, layer):
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        root, layer = Path(sys.argv[1]), sys.argv[2]
        sys.path.insert(0, str(root / "src"))
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if layer == "tool":
                    assert not fullname.startswith(("vv_agent.kernel._record", "vv_agent.kernel._state")), fullname
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _tool_result_checks as checks
        from vv_agent.kernel._tool_values import ToolExecutionResult, ToolResultStatus

        ambiguous = ToolExecutionResult("call", "timeout", status_code=ToolResultStatus.ERROR, error_code="tool_timeout")
        try:
            checks.validate_definitive_result(ambiguous)
        except checks.DefinitiveResultInvalid as error:
            assert error.code == "deferred_resolution_result_invalid"
        else:
            raise AssertionError("ambiguous tool result accepted")
        ambiguous.metadata["definitive_outcome"] = True
        checks.validate_definitive_result(ambiguous)
        try:
            checks.validate_output_body(ToolExecutionResult("call", '{"n": NaN}', metadata={"output_type": "json"}))
        except ValueError as error:
            assert str(error) == "non-finite JSON constant"
        else:
            raise AssertionError("nonfinite structured body accepted")
        fixture = json.loads((root / "tests/fixtures/parity/bounded_tool_result.json").read_text())
        bounded = fixture["canonical_results"]["truncated_bash"]
        bounded["metadata"] = dict(bounded.get("metadata", {}), output_type="json")
        checks.validate_output_body(ToolExecutionResult.from_dict(bounded))
        if layer == "provider":
            from vv_agent.kernel import _provider_outcomes as core
            from vv_agent.kernel._records import Record
            history = json.loads((root / "tests/session/fixtures/step_cost_release_histories.json").read_text())[-1]
            records = [Record.parse(body.encode()) for body in history["records"]]
            plan = next(r for r in records if r.kind == "op_planned" and r._payload["op_kind"] == "tool")
            handle = dict(kind="provider", provider=plan._payload["provider_binding"], job_id="receipt",
                          operation_id=plan.operation_id, attempt=plan.attempt,
                          request_digest=plan._payload["request_digest"], evidence="accepted",
                          query_ref="receipt", cancel_ref=None)
            raw = ToolExecutionResult(plan._payload["request"]["id"], "ok").to_dict()
            core.validate_acceptance(plan, core.Accepted(handle))
            core.validate_provider_result(plan, core.Definitive(raw, ("accepted",)), handle)
            try:
                core.validate_acceptance(plan, core.Accepted(handle, 1))
            except core.ProviderResultInvalid as error:
                assert error.code == str(error) == "provider_result_invalid"
                assert error.__cause__ is None
            else:
                raise AssertionError("non-bool acceptance admitted")
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
        if layer == "tool":
            assert not any(name.startswith(("vv_agent.kernel._record", "vv_agent.kernel._state")) for name in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(_ROOT), layer],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_facades_keep_actual_objects_and_outer_execution_owners():
    import vv_agent
    from vv_agent.session import providers, tool_results
    from vv_agent.tools import outcomes

    for name in ("Definitive", "Accepted", "Unknown"):
        assert getattr(vv_agent, name) is getattr(providers, name) is getattr(core, name)
    for name in ("Outcome", "ProviderResultInvalid", "validate_provider_result", "validate_acceptance"):
        assert getattr(providers, name) is getattr(core, name)
    assert outcomes.DefinitiveResultInvalid is checks.DefinitiveResultInvalid
    assert outcomes.validate_definitive_result is checks.validate_definitive_result
    assert tool_results.result_kind is checks.result_kind
    assert tool_results.validate_output_body is checks.validate_output_body
    assert not hasattr(providers, "validate_provider_handle")
    assert not hasattr(outcomes, "_is_ambiguous_tool_error")
    assert not hasattr(tool_results, "_reject_json_constant")
    assert providers.Provider.__module__ == providers.FunctionProvider.__module__ == providers.__name__
    assert outcomes.HostToolOutcome.__module__ == outcomes.__name__
    first, second = core.Definitive({}), core.Definitive({})
    assert first.evidence == ("synchronous-return",) and first.shared_state is None
    assert first.usage == second.usage == {} and first.usage is not second.usage
    assert hasattr(first, "__dict__")
    with pytest.raises(FrozenInstanceError):
        first.__setattr__("evidence", ())


def test_receipt_validation_order_and_nested_causes():
    plan = _tool_plan()
    handle = _handle(plan)
    bad_handle = handle | {"operation_id": "wrong"}
    call_id = plan._payload["request"]["id"]
    raw = ToolExecutionResult(call_id, "ok").to_dict()
    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_provider_result(plan, core.Definitive(raw | {"extra": float("nan")}, ()), bad_handle)
    assert type(caught.value.__cause__) is ValueError
    assert str(caught.value.__cause__) == "value must be RFC 8785 I-JSON: non-finite number"

    ambiguous = ToolExecutionResult(
        call_id,
        "not-json",
        status_code=ToolResultStatus.ERROR,
        error_code="tool_timeout",
        metadata={"output_type": "json"},
    )
    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_provider_result(plan, core.Definitive(ambiguous.to_dict(), ()), bad_handle)
    assert type(caught.value.__cause__) is checks.DefinitiveResultInvalid
    assert caught.value.__cause__.code == "deferred_resolution_result_invalid"

    malformed = ToolExecutionResult("wrong-call", "{}", metadata={"output_type": "file"})
    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_provider_result(plan, core.Definitive(malformed.to_dict(), ()), bad_handle)
    assert type(caught.value.__cause__) is ValueError
    assert str(caught.value.__cause__) == "invalid file/image body"

    wrong_call = ToolExecutionResult("wrong-call", "ok").to_dict()
    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_provider_result(plan, core.Definitive(wrong_call, ()), bad_handle)
    assert str(caught.value.__cause__) == "provider result call ID mismatch"

    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_provider_result(plan, core.Definitive(raw, ()), bad_handle)
    assert str(caught.value.__cause__) == "provider result evidence or usage is incomplete"

    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_provider_result(plan, core.Definitive(raw, ("accepted",)), bad_handle)
    cause = caught.value.__cause__
    assert type(cause) is core.ProviderResultInvalid
    assert type(cause.__cause__) is ValueError
    assert str(cause.__cause__) == "provider handle mismatch"


def test_acceptance_handle_failure_precedes_declaration_failure():
    plan = _tool_plan()
    handle = _handle(plan)
    invalid_flag: Any = 1
    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_acceptance(plan, core.Accepted(handle | {"operation_id": "wrong"}, invalid_flag))
    assert str(caught.value.__cause__) == "provider handle mismatch"
    with pytest.raises(core.ProviderResultInvalid) as caught:
        core.validate_acceptance(plan, core.Accepted(handle, invalid_flag))
    assert caught.value.__cause__ is None
