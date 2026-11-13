"""One core admission algorithm, with the real schema primitive kept on the host."""

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_admission_core_imports_without_outer_or_third_party(tmp_path, flags):
    source = textwrap.dedent("""
        import importlib.abc
        import sys
        sys.path.insert(0, sys.argv[1])
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _tool_admission as core
        from vv_agent.kernel._tool_values import ToolCall
        arguments = {"nested": []}
        assert core._parse_arguments("c", arguments)[0] is arguments
        assert core._parse_arguments("c", "{")[1].error_code == "invalid_arguments_json"
        call = ToolCall("c", "missing", {})
        steps = core.admit_steps(call, original_call=call, executor_present=False, allowed_tool_names=None)
        assert next(steps) is core.Read.POLICY_METADATA
        try:
            steps.send({})
        except StopIteration as done:
            assert done.value.error_code == "tool_not_found"
        else:
            raise AssertionError("missing tool must finish without schema or approval requests")
        assert all(n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.")
                   for n in sys.modules if n.startswith("vv_agent"))
    """)
    completed = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", source, str(Path(__file__).resolve().parents[1] / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_sdk_free_host_uses_the_real_validator_and_core_order(tmp_path):
    source = textwrap.dedent("""
        import importlib.abc
        import sys
        sys.path.insert(0, sys.argv[1])
        class NoSDK(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("vv_agent"):
                    assert (fullname in {"vv_agent", "vv_agent.kernel", "vv_agent._jsonschema"}
                            or fullname.startswith("vv_agent.kernel.")), fullname
        sys.meta_path.insert(0, NoSDK())
        from vv_agent import _jsonschema
        from vv_agent.kernel import _tool_admission as core
        from vv_agent.kernel._tool_values import ToolCall
        schema = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
        def run(arguments, denied=False):
            call = ToolCall("c", "tool", arguments)
            steps = core.admit_steps(call, original_call=call, executor_present=True, allowed_tool_names=["tool"])
            response = None
            trace = []
            while True:
                try:
                    request = steps.send(response)
                except StopIteration as done:
                    return done.value, trace
                trace.append(type(request).__name__ if not isinstance(request, core.Read) else request.name)
                if isinstance(request, core.SchemaCheck):
                    response = _jsonschema.schema_issues(request)
                elif request is core.Read.SCHEMA:
                    response = schema
                elif request is core.Read.HANDLER_MANAGED:
                    response = False
                elif request is core.Read.POLICY_METADATA:
                    response = {"_vv_agent_allowed_tools": []} if denied else {}
                elif request is core.Read.TOOL_METADATA:
                    response = None
                elif isinstance(request, core.ApprovedCall):
                    response = None
                elif request is core.Read.APPROVAL_MODE:
                    response = "never"
                else:
                    raise AssertionError(request)
        result, trace = run({"value": "wrong"})
        assert result.error_code == "invalid_tool_arguments"
        assert trace == ["SCHEMA", "SchemaCheck"]
        result, trace = run({"value": 1}, denied=True)
        assert result.error_code == "tool_not_allowed"
        assert "ApprovedCall" not in trace and "RequestApproval" not in trace
        result, trace = run({"value": 1})
        assert isinstance(result, core.Dispatch) and result.mark_started
        info = _jsonschema._validator_for.cache_info()
        assert info.misses == 1 and info.hits == 2 and info.maxsize == 256
        assert "vv_agent.tools" not in sys.modules and "vv_agent.session" not in sys.modules
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source, str(Path(__file__).resolve().parents[1] / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_preflight_and_submit_repeat_real_admission(tmp_path, monkeypatch):
    from vv_agent import _jsonschema, function_tool
    from vv_agent.kernel import _tool_admission as core
    from vv_agent.kernel._records import Record
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.session.providers import Definitive, FunctionProvider
    from vv_agent.tools import dispatcher
    from vv_agent.tools.base import ToolContext
    from vv_agent.tools.orchestrator import ToolOrchestrator
    from vv_agent.workspace import LocalWorkspaceBackend

    assert dispatcher._parse_arguments is core._parse_arguments
    effects, checks, predicates = [], [], []

    @function_tool
    def effect() -> str:
        effects.append("ran")
        return "done"

    def predicate(name, arguments):
        predicates.append((name, arguments))
        return True

    actual = _jsonschema.schema_issues

    def schema_check(request):
        checks.append(request)
        return actual(request)

    monkeypatch.setattr(_jsonschema, "schema_issues", schema_check)
    provider = FunctionProvider(ToolOrchestrator.from_tools([effect]))
    context = ToolContext(
        tmp_path,
        {},
        1,
        LocalWorkspaceBackend(tmp_path),
        ctx=ExecutionContext(metadata={"_vv_agent_tool_policy_can_use_tool": predicate}),
        metadata={"session_tool_names": ["effect"]},
    )
    record = Record("p", "op_planned", "s", "t", "op", 1, {"request": {"id": "c", "name": "effect", "arguments": {}}})
    assert provider.preflight(record, context) is None
    assert effects == [] and len(checks) == len(predicates) == 1
    assert "_vv_agent_tool_dispatch_callback" not in context.metadata
    outcome = provider.submit(record, context=context)
    assert isinstance(outcome, Definitive)
    assert outcome.result["content"] == "done"
    assert effects == ["ran"] and len(predicates) == 2
    assert len(checks) == 3  # Preflight, submit admission, and the real FunctionToolExecutor.execute validation.


def test_approval_metadata_snapshot_is_taken_after_policy_before_requires_callback(tmp_path):
    from vv_agent import function_tool
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.tools.base import ToolContext
    from vv_agent.tools.orchestrator import ToolOrchestrator
    from vv_agent.types import ToolCall, ToolExecutionResult
    from vv_agent.workspace import LocalWorkspaceBackend

    trace = []

    class Provider:
        def should_request(self, request):
            trace.append("original provider")
            return False

    class UnexpectedProvider:
        def should_request(self, request):
            raise AssertionError("requires_approval must not replace the already selected approval metadata")

    metadata: dict[str, Any] = {"_vv_agent_approval_provider": Provider()}

    def needs_approval(context, arguments):
        metadata["_vv_agent_approval_provider"] = UnexpectedProvider()
        trace.append("requires")
        return True

    @function_tool(needs_approval=needs_approval)
    def effect() -> str:
        trace.append("handler")
        return "done"

    context = ToolContext(tmp_path, {}, 1, LocalWorkspaceBackend(tmp_path), ctx=ExecutionContext(metadata=metadata))
    result = ToolOrchestrator.from_tools([effect]).run_one(ToolCall("c", "effect", {}), context=context)
    assert isinstance(result, ToolExecutionResult)
    assert result.content == "done"
    assert trace == ["requires", "original provider", "handler"]


@pytest.mark.parametrize("mode", ["stop", "nested", "approved-accessor"])
def test_callback_stop_iteration_is_not_generator_completion_or_heuristically_unwrapped(tmp_path, mode):
    from vv_agent import function_tool
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.tools.base import ToolContext
    from vv_agent.tools.orchestrator import ToolOrchestrator
    from vv_agent.types import ToolCall, ToolExecutionResult
    from vv_agent.workspace import LocalWorkspaceBackend

    errors = []
    original = StopIteration("user stop")

    def predicate(name, arguments):
        if mode == "stop":
            raise original

        def broken():
            raise StopIteration("nested")
            yield

        next(broken())

    def format_error(error):
        errors.append(error)
        return "formatted"

    @function_tool(failure_error_function=format_error)
    def effect() -> str:
        raise AssertionError("denied call must not execute")

    context = ToolContext(
        tmp_path,
        {},
        1,
        LocalWorkspaceBackend(tmp_path),
        ctx=ExecutionContext(metadata={"_vv_agent_tool_policy_can_use_tool": None if mode == "approved-accessor" else predicate}),
    )
    if mode == "approved-accessor":

        class Approved:
            @property
            def id(self):
                raise original

        assert context.ctx is not None
        context.ctx._approved_tool_approval = SimpleNamespace(call=Approved())
    result = ToolOrchestrator.from_tools([effect]).run_one(ToolCall("c", "effect", {}), context=context)
    assert isinstance(result, ToolExecutionResult)
    assert result.error_code == "tool_execution_failed" and result.content
    assert len(errors) == 1
    if mode in {"stop", "approved-accessor"}:
        assert errors[0] is original
    else:
        assert type(errors[0]) is RuntimeError and isinstance(errors[0].__cause__, StopIteration)


def test_precomputed_result_keeps_planned_and_completed_without_admission(tmp_path, monkeypatch):
    from vv_agent import _jsonschema, function_tool
    from vv_agent.tools.base import ToolContext
    from vv_agent.tools.orchestrator import ToolOrchestrator
    from vv_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus
    from vv_agent.workspace import LocalWorkspaceBackend

    @function_tool(needs_approval=True)
    def effect(value: int) -> str:
        raise AssertionError("precomputed result must not execute")

    orchestrator = ToolOrchestrator.from_tools([effect])

    def forbidden(request):
        raise AssertionError("precomputed result must not validate schema")

    monkeypatch.setattr(_jsonschema, "schema_issues", forbidden)
    context = ToolContext(tmp_path, {}, 1, LocalWorkspaceBackend(tmp_path))
    precomputed = ToolExecutionResult("pre", "wait", directive=ToolDirective.WAIT_USER)
    events = []
    result = orchestrator.run_one(
        ToolCall("c", "effect", {"invalid": True}),
        context=context,
        event_sink=events.append,
        _precomputed_result=precomputed,
    )
    assert result is precomputed and result.status_code == ToolResultStatus.WAIT_RESPONSE
    assert [event.type for event in events] == ["tool_call_planned", "tool_call_completed"]
    assert events[-1].execution_started is False and events[-1].duration_ms is None


@pytest.mark.parametrize("source", ["approved-getter", "allowed-collection"])
@pytest.mark.parametrize("chain", ["implicit", "explicit-cause", "suppressed-context"])
@pytest.mark.parametrize("ambient", [False, True], ids=["no-ambient-error", "handling-ambient-error"])
def test_public_formatter_retains_admission_exception_chain(source, chain, ambient):
    import json

    from vv_agent import function_tool
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.tools.base import ToolContext
    from vv_agent.tools.orchestrator import ToolOrchestrator
    from vv_agent.types import ToolCall, ToolExecutionResult
    from vv_agent.workspace import MemoryWorkspaceBackend

    original = StopIteration("admission stop")
    underlying = LookupError("original underlying context")
    explicit = ValueError("explicit cause")
    outer = RuntimeError("ambient caller error")
    observed, effects = [], []

    def failure():
        try:
            raise underlying
        except LookupError:
            if chain == "explicit-cause":
                raise original from explicit
            if chain == "suppressed-context":
                raise original from None
            raise original  # noqa: B904 - this case deliberately exercises implicit chaining.

    def formatter(error):
        observed.append((error, error.__context__, error.__cause__, error.__suppress_context__))
        return json.dumps(
            {
                "context": str(error.__context__),
                "cause": str(error.__cause__) if error.__cause__ is not None else None,
                "suppress_context": error.__suppress_context__,
            },
            sort_keys=True,
        )

    @function_tool(failure_error_function=formatter)
    def effect() -> str:
        effects.append("executed")
        return "done"

    class Approved:
        @property
        def id(self):
            return failure()

    class Allowed(list):
        def __contains__(self, value):
            return failure()

    context = ToolContext(Path("/fixture"), {}, 1, MemoryWorkspaceBackend(), ctx=ExecutionContext())
    assert context.ctx is not None
    if source == "approved-getter":
        context.ctx._approved_tool_approval = SimpleNamespace(call=Approved())
    allowed = Allowed(["effect"]) if source == "allowed-collection" else None
    orchestrator = ToolOrchestrator.from_tools([effect])

    def invoke():
        return orchestrator.run_one(ToolCall("c", "effect", {}), context=context, allowed_tool_names=allowed)

    if ambient:
        try:
            raise outer
        except RuntimeError:
            result = invoke()
    else:
        result = invoke()

    expected_cause = explicit if chain == "explicit-cause" else None
    suppressed = chain != "implicit"
    assert len(observed) == 1
    error, original_context, original_cause, original_suppression = observed[0]
    assert error is original and original_context is underlying and original_cause is expected_cause
    assert original_suppression is suppressed
    assert original.__context__ is underlying and original.__cause__ is expected_cause
    assert original.__suppress_context__ is suppressed
    assert underlying.__context__ is (outer if ambient else None)
    assert effects == [] and isinstance(result, ToolExecutionResult)
    assert result.error_code == "tool_execution_failed"
    assert json.loads(result.content)["error"] == json.dumps(
        {
            "context": str(underlying),
            "cause": str(expected_cause) if expected_cause is not None else None,
            "suppress_context": suppressed,
        },
        sort_keys=True,
    )
