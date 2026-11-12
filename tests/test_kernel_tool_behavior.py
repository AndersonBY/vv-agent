"""Tool behavior executes through the real package without outer SDK imports."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_tool_behavior_without_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.abc
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
        from vv_agent.kernel import _tool_behavior as core
        from vv_agent.kernel._execution_values import AgentTask, CompletionReason
        from vv_agent.kernel._message_values import Message, ToolResultCursor
        from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
        from vv_agent.kernel._tool_values import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus
        import vv_agent

        assert Path(core.__file__) == root / "src/vv_agent/kernel/_tool_behavior.py"
        assert Path(vv_agent.__file__) == root / "src/vv_agent/__init__.py"
        task = AgentTask("task", "model", PromptBundle((PromptSection("system", "system", True),)), "go",
                         metadata={"_vv_agent_tool_use_behavior": "stop_on_first_tool"})
        call = ToolCall("call", "lookup", {})
        cursor = ToolResultCursor("read_file", "恢复.txt", 1, "0" * 64)
        result = ToolExecutionResult("call", "恢复😀", truncated=True, truncation_reason="read_limit",
                                     original_bytes=20, visible_bytes=10, cursor=cursor)
        before = result.to_dict()
        recovery_text = result.to_tool_message().content
        assert core.apply_tool_use_behavior(task=task, call=call, result=result) is CompletionReason.STOP_ON_FIRST_TOOL
        assert result.directive is ToolDirective.FINISH
        assert result.to_dict() == before | {"directive": "finish"}
        assert result.cursor is cursor
        assert result.to_tool_message().content == recovery_text

        skipped = core.build_skipped_result(call, error_code="tool_skipped", message="跳过😀")
        assert type(skipped) is ToolExecutionResult
        assert skipped.status_code is ToolResultStatus.ERROR
        assert skipped.directive is ToolDirective.CONTINUE
        assert skipped.content == '{"ok": false, "error": "跳过😀", "error_code": "tool_skipped"}'
        assert core.apply_tool_use_behavior(task=task, call=call, result=skipped) is None
        assert skipped.directive is ToolDirective.CONTINUE

        for url, path, content in [
            ("data:image/png;base64,AA==", None, ""),
            ("data:image/png;base64,AA==", "图😀.png", "[Image loaded] 图😀.png"),
            ("https://example.invalid/a.png", None, ""),
            (None, "图😀.png", "[Image loaded] 图😀.png"),
        ]:
            image_result = ToolExecutionResult("call", "unchanged", image_url=url, image_path=path)
            before = image_result.to_dict()
            image = core.build_image_notification(result=image_result, include_image=True)
            assert type(image) is Message
            assert (image.role, image.content, image.image_url) == ("user", content, url)
            assert core.build_image_notification(result=image_result, include_image=False) is None
            assert image_result.to_dict() == before
        assert core.build_image_notification(result=skipped, include_image=True) is None
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
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


def test_existing_tool_helpers_are_actual_core_objects():
    from vv_agent import types
    from vv_agent.kernel import _tool_behavior as core
    from vv_agent.kernel import _tool_values
    from vv_agent.kernel._message_values import Message
    from vv_agent.runtime import tool_results
    from vv_agent.session import context

    for name in ("apply_tool_use_behavior", "build_image_notification", "build_skipped_result"):
        assert getattr(tool_results, name) is getattr(core, name), name
    assert context.build_image_notification is core.build_image_notification
    assert types.ToolExecutionResult is _tool_values.ToolExecutionResult
    assert types.ToolDirective is _tool_values.ToolDirective
    assert types.CompletionReason is core.CompletionReason
    assert types.AgentTask is core.AgentTask
    assert types.Message is Message is core.Message
