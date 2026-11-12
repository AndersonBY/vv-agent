"""The complete Tool value implementation executes without the outer SDK."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_tool_values_import_and_validate_without_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from copy import deepcopy
        from pathlib import Path

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root / "src"))

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _tool_values
        from vv_agent.kernel._message_values import Message, ToolArtifactRef, ToolResultCursor
        from vv_agent.kernel._tool_values import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

        assert Path(_tool_values.__file__) == root / "src/vv_agent/kernel/_tool_values.py"
        call = ToolCall.from_dict({"id": "call", "name": "tool", "arguments": {}, "extra_content": {"nested": [1]}})
        exported = call.to_dict()
        exported["extra_content"]["nested"].append(2)
        assert call.extra_content == {"nested": [1]}
        assert ToolExecutionResult("call", "ok").directive is ToolDirective.CONTINUE
        assert ToolExecutionResult("call", "ok").status_code is ToolResultStatus.SUCCESS

        fixture = json.loads((root / "tests/fixtures/parity/bounded_tool_result.json").read_text())
        # Existing canonical vectors exercise both recovery pointers and retained provenance.
        for case in fixture["tool_message_projection"]["cases"]:
            payload = fixture["canonical_results"][case["result_ref"]]
            result = ToolExecutionResult.from_dict(payload)
            assert result.to_dict() == payload, case["name"]
            message = result.to_tool_message()
            assert type(message) is Message
            assert message.content == case["expected_message"], case["name"]
            assert Message.from_dict(message.to_dict()).to_dict() == message.to_dict()
            if result.artifact is not None:
                assert type(result.artifact) is ToolArtifactRef
            if result.cursor is not None:
                assert type(result.cursor) is ToolResultCursor

        invalid = deepcopy(fixture["canonical_results"]["truncated_bash"])
        invalid["metadata"]["vv_artifact"]["source"]["arguments_sha256"] = "invalid"
        try:
            ToolExecutionResult.from_dict(invalid)
        except ValueError as exc:
            assert str(exc) == "tool_result_invalid"
            assert exc.__cause__ is not None
            assert "lowercase SHA-256" in str(exc.__cause__)
        else:
            raise AssertionError("core must retain full provenance validation")

        result = ToolExecutionResult("call", "ok")
        result.metadata["request_compaction"] = False
        try:
            result.to_tool_message()
        except ValueError as exc:
            assert "request_compaction must be true" in str(exc)
        else:
            raise AssertionError("projection must revalidate mutated values")

        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._tool_values",
            "vv_agent.kernel._message_values", "vv_agent.kernel._message_codec", "vv_agent.kernel._canonical_json"
        }
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


def test_tool_values_keep_real_types_and_public_bindings():
    import vv_agent
    from vv_agent import types
    from vv_agent.kernel import _tool_values as core
    from vv_agent.memory import manager

    for name in ("ToolCall", "ToolDirective", "ToolResultStatus", "ToolExecutionResult"):
        assert getattr(types, name) is getattr(core, name), name
    for name in ("ToolDirective", "ToolResultStatus", "ToolExecutionResult"):
        assert getattr(vv_agent, name) is getattr(core, name), name
    assert manager.ToolExecutionResult is core.ToolExecutionResult
