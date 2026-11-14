"""Core-only schema requests and the real SDK-free host are separate boundaries."""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _run(script, flags, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", textwrap.dedent(script), str(_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_schema_request_and_normalization_without_host_imports(tmp_path, flags):
    _run(
        """
        import importlib.abc
        import json
        import sys

        assert not any(name.startswith("vv_agent") for name in sys.modules)
        sys.path.insert(0, sys.argv[1] + "/src")
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _tool_schema as core

        arguments = {"value": "unchanged"}
        computation = core.validate_tool_arguments({"type": "object"}, arguments)
        request = next(computation)
        assert type(request) is core.SchemaCheck
        assert request.serialized_schema == '{"type":"object"}'
        assert request.arguments is arguments
        item = core.SchemaIssue(("b/名", "~"), ("properties", "b/名", "type"), "type")
        try:
            computation.send((item, core.SchemaIssue((), (), None), item))
        except StopIteration as done:
            issues = done.value
        else:
            raise AssertionError("normalizer must complete after one schema check")
        assert issues == [
            {"instance_path": "", "schema_path": "", "rule": "falseSchema"},
            {"instance_path": "/b~1名/~0", "schema_path": "/properties/b~1名/type", "rule": "type"},
        ]
        assert core._validation_rule("", ("allOf", 1, "type")) == "type"
        assert core._validation_rule(None, (1,)) == "falseSchema"
        assert core._validation_rule(None, ("",)) == ""
        result = core.invalid_tool_arguments_result(tool_call_id="call", issues=issues)
        assert result.error_code == "invalid_tool_arguments"
        assert result.status_code.value == "ERROR" and result.directive.value == "continue"
        assert json.loads(result.content)["issues"] == issues
        assert result.metadata == {"error_code": "invalid_tool_arguments", "issue_count": 2}
        assert core.invalid_tool_arguments_result(tool_call_id="call", issues=[]) is None
        assert "jsonschema" not in sys.modules and "vv_agent._jsonschema" not in sys.modules
        """,
        flags,
        tmp_path,
    )


def test_sdk_free_host_runs_real_jsonschema_after_blocker_is_installed(tmp_path):
    _run(
        """
        import importlib.abc
        import json
        import sys

        assert not any(name.startswith("vv_agent") for name in sys.modules)
        assert "jsonschema" not in sys.modules
        sys.path.insert(0, sys.argv[1] + "/src")
        # jsonschema's base dependencies and optional format dependencies may be imported by the library.
        dependencies = {"jsonschema", "jsonschema_specifications", "referencing", "rpds", "attrs", "attr",
                        "typing_extensions", "fqdn", "idna", "isoduration", "jsonpointer", "rfc3339_validator",
                        "rfc3987", "uri_template", "webcolors", "rfc3986_validator", "rfc3987_syntax",
                        "lark", "arrow", "dateutil", "six"}
        class HostOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel", "vv_agent._jsonschema"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names | dependencies, fullname
        sys.meta_path.insert(0, HostOnly())
        from vv_agent import _jsonschema as host
        from vv_agent.kernel import _tool_schema as core
        from jsonschema import Draft202012Validator

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": False}
        arguments = {"x": "wrong", "extra": True}
        computation = core.validate_tool_arguments(schema, arguments)
        request = next(computation)
        assert request.arguments is arguments
        host._validator_for.cache_clear()
        facts = host.schema_issues(request)
        assert isinstance(facts, tuple) and len(facts) == 2
        assert all(type(fact) is core.SchemaIssue for fact in facts)
        validator = host._validator_for(request.serialized_schema)
        assert type(validator) is Draft202012Validator
        assert validator is host._validator_for(request.serialized_schema)
        assert validator.format_checker is None
        assert host._validator_for.cache_parameters() == {"maxsize": 256, "typed": False}
        try:
            computation.send(facts)
        except StopIteration as done:
            issues = done.value
        else:
            raise AssertionError("real host reply did not finish normalization")
        assert issues == [
            {"instance_path": "", "schema_path": "/additionalProperties", "rule": "additionalProperties"},
            {"instance_path": "/x", "schema_path": "/properties/x/type", "rule": "type"},
        ]
        assert core.invalid_tool_arguments_result(tool_call_id="real", issues=issues).metadata["issue_count"] == 2
        assert all(name in {"vv_agent", "vv_agent.kernel", "vv_agent._jsonschema"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
        assert "vv_agent.tools" not in sys.modules
        """,
        [],
        tmp_path,
    )


def test_sdk_and_host_share_validator_cache_and_argument_identity(monkeypatch):
    from jsonschema import Draft202012Validator

    from vv_agent import _jsonschema as host
    from vv_agent.kernel import _tool_schema as core
    from vv_agent.tools import argument_validation as sdk

    class NoCopy(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("arguments must not be copied")

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    arguments = NoCopy(x="wrong", extra=True)
    host._validator_for.cache_clear()
    serialized = core._canonical_schema(schema)
    sdk.assert_valid_tool_schema(schema)
    validator = host._validator_for(serialized)
    seen = []
    original = Draft202012Validator.iter_errors

    def observe(self, instance, *args, **kwargs):
        seen.append((self, instance))
        return original(self, instance, *args, **kwargs)

    monkeypatch.setattr(Draft202012Validator, "iter_errors", observe)
    facts = host.schema_issues(core.SchemaCheck(serialized, arguments))
    issues = sdk.validate_tool_arguments(schema, arguments)
    assert len(facts) == len(issues) == 1  # The open object was not silently closed.
    assert all(owner is validator and instance is arguments for owner, instance in seen)
    assert len(seen) == 2 and host._validator_for.cache_info().currsize == 1
    assert "additionalProperties" not in schema
    closed = sdk.close_object_schemas(schema)
    assert closed["additionalProperties"] is False and "additionalProperties" not in schema


@pytest.mark.parametrize("schema", [True, False])
def test_boolean_schemas_use_the_real_host(schema: Any):
    from vv_agent.tools.argument_validation import (
        assert_valid_tool_schema,
        invalid_tool_arguments_result,
        validate_tool_arguments,
    )

    assert_valid_tool_schema(schema)
    issues = validate_tool_arguments(schema, {})
    result = invalid_tool_arguments_result(tool_call_id="boolean", schema=schema, arguments={})
    if schema:
        assert issues == [] and result is None
    else:
        assert issues == [{"instance_path": "", "schema_path": "", "rule": "falseSchema"}]
        assert result is not None and result.metadata["issue_count"] == 1


@pytest.mark.parametrize("failure", [RuntimeError("iteration failed"), StopIteration("host iterator failed")])
def test_host_iteration_exceptions_are_not_converted_to_empty_issues(monkeypatch, failure):
    from jsonschema import Draft202012Validator

    from vv_agent import _jsonschema as host
    from vv_agent.kernel import _tool_schema as core
    from vv_agent.tools.argument_validation import invalid_tool_arguments_result, validate_tool_arguments

    schema = {"type": "object", "title": "exception propagation"}
    host._validator_for(core._canonical_schema(schema))

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(Draft202012Validator, "iter_errors", fail)
    with pytest.raises(type(failure)) as caught:
        validate_tool_arguments(schema, {})
    assert caught.value is failure
    with pytest.raises(type(failure)) as caught:
        invalid_tool_arguments_result(tool_call_id="failure", schema=schema, arguments={})
    assert caught.value is failure


def test_serialization_schema_and_reference_errors_propagate():
    from jsonschema import Draft202012Validator, SchemaError

    from vv_agent.tools.argument_validation import assert_valid_tool_schema, validate_tool_arguments

    with pytest.raises(TypeError, match="not JSON serializable"):
        validate_tool_arguments({"enum": {1}}, {})
    with pytest.raises(SchemaError):
        assert_valid_tool_schema({"type": 17})
    with pytest.raises(SchemaError):
        validate_tool_arguments({"type": 17}, {})
    schema = {"$ref": "#/$defs/missing"}
    with pytest.raises(Exception) as baseline:
        tuple(Draft202012Validator(schema).iter_errors({}))
    with pytest.raises(type(baseline.value)) as caught:
        validate_tool_arguments(schema, {})
    assert str(caught.value) == str(baseline.value)


def test_schema_rejection_precedes_policy_approval_broker_and_handler(tmp_path):
    from support import require_tool_result

    from vv_agent.approval import ApprovalBroker, ApprovalDecision
    from vv_agent.tools import ToolContext
    from vv_agent.tools.function import function_tool
    from vv_agent.tools.orchestrator import ToolOrchestrator
    from vv_agent.types import ToolCall
    from vv_agent.workspace import MemoryWorkspaceBackend

    seen = []

    def policy(name, arguments):
        seen.append("policy")
        return True

    def approval(context, arguments):
        seen.append("requires_approval")
        return True

    class Decisions:
        def should_request(self, request):
            seen.append("should_request")
            return True

        def decide(self, request):
            seen.append("decide")
            return ApprovalDecision.allow()

    class Broker(ApprovalBroker):
        def register(self, request):
            seen.append("broker")
            super().register(request)

    @function_tool(needs_approval=approval)
    def effect(value: int) -> str:
        seen.append("handler")
        return str(value)

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
        metadata={
            "_vv_agent_tool_policy_can_use_tool": policy,
            "_vv_agent_approval_provider": Decisions(),
            "_vv_agent_approval_broker": Broker(),
        },
    )
    orchestrator = ToolOrchestrator.from_tools([effect])
    events = []
    denied = require_tool_result(
        orchestrator.run_one(
            ToolCall("bad", "effect", {"value": "wrong", "extra": True}), context=context, event_sink=events.append
        )
    )
    assert denied.error_code == "invalid_tool_arguments" and seen == []
    assert [event.type for event in events] == ["tool_call_planned", "tool_call_completed"]
    assert events[-1].execution_started is False
    accepted = require_tool_result(orchestrator.run_one(ToolCall("good", "effect", {"value": 7}), context=context))
    assert accepted.content == "7" and seen.count("handler") == 1
    assert seen.index("policy") < seen.index("requires_approval") < seen.index("broker") < seen.index("handler")
    assert seen.count("broker") == seen.count("decide") == 1


def test_session_schema_rejection_does_not_call_admission_callbacks(tmp_path):
    from vv_agent import Agent, RunConfig, Runner, ScriptedModelProvider
    from vv_agent.approval import ApprovalBroker, ApprovalDecision
    from vv_agent.run_config import ToolPolicy
    from vv_agent.tools.function import function_tool
    from vv_agent.types import LLMResponse, ToolCall

    seen = []

    def policy(name, arguments):
        seen.append("policy")
        return True

    def approval(context, arguments):
        seen.append("requires_approval")
        return True

    class Decisions:
        def should_request(self, request):
            seen.append("should_request")
            return True

        def decide(self, request):
            seen.append("decide")
            return ApprovalDecision.allow()

    class Broker(ApprovalBroker):
        def is_session_allowed(self, tool_name):
            seen.append("broker")
            return super().is_session_allowed(tool_name)

        def register(self, request):
            seen.append("register")
            super().register(request)

    @function_tool(needs_approval=approval)
    def effect(value: int) -> str:
        seen.append("handler")
        return str(value)

    result = Runner.run_sync(
        Agent("schema", "Answer.", tools=[effect]),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=ScriptedModelProvider.from_steps(
                "test", "m", [LLMResponse("", [ToolCall("bad", "effect", {"value": "wrong"})]), LLMResponse("done")]
            ),
            tool_policy=ToolPolicy(can_use_tool=policy),
            approval_provider=Decisions(),
            approval_broker=Broker(),
        ),
    )
    assert seen == []
    assert result.final_output == "done"
    results = [value for cycle in result.raw_result.cycles for value in cycle.tool_results]
    assert len(results) == 1 and results[0].error_code == "invalid_tool_arguments"
    assert not any(event.type == "tool_call_started" for event in result.events)
