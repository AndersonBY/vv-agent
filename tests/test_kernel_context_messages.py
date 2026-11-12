"""Protected suffixes and resume sanitizers execute without outer SDK imports."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_context_messages_import_and_run_without_outer_sdk(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _context_messages as core
        from vv_agent.kernel._message_values import Message

        assert Path(core.__file__) == Path(sys.argv[1]) / "vv_agent/kernel/_context_messages.py"
        assert core.Message is Message
        assistant = Message("assistant", "", tool_calls=[{"id": "a"}, {"id": "b"}])
        source = [
            Message("user", "seen"), assistant,
            Message("tool", "A", tool_call_id="a"),
            Message("tool", "B", tool_call_id="b"),
            Message("user", "protected"),
        ]
        replacement = [Message("assistant", "summary"), *source[1:]]
        assert core.atomic_start(source, 3) == 1
        assert core.retained_start(source, replacement, 3) == 1
        assert core.retained_start(source, [], None) is None
        assert core.retained_start(source, [], len(source)) is None
        try:
            core.retained_start(source, source[2:], 3)
        except core.UnseenInputError as exc:
            assert exc.code == "unseen_input_removed"
            assert str(exc) == "unseen_input_removed: replacement must retain the protected suffix verbatim"
        else:
            raise AssertionError("an incomplete protected tool block must be rejected")
        assert core.UnseenInputCapacityError.code == "unseen_input_capacity_exceeded"

        reasoning = Message("assistant", "", reasoning_content="thinking")
        partial = [assistant, source[2]]
        history = [Message("tool", "orphan", tool_call_id="orphan"), Message("assistant", " "), reasoning, *partial]
        sanitized = core.sanitize_for_resume(history)
        assert sanitized == [reasoning, Message("assistant", "", tool_calls=[{"id": "a"}]), source[2]]
        assert sanitized[0] is reasoning and sanitized[-1] is source[2]
        assert assistant.tool_calls == [{"id": "a"}, {"id": "b"}]
        assert len(history) == 5
        assert core.filter_orphan_tool_results(partial) == partial
        assert core.filter_unresolved_tool_uses(partial) == sanitized[1:]
        assert core.filter_thinking_only_messages([reasoning, source[0]]) == [source[0]]
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._context_messages",
            "vv_agent.kernel._message_values", "vv_agent.kernel._canonical_json"
        }
    """)
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_context_messages_keep_public_exports_and_active_callers_identical():
    import vv_agent
    from vv_agent import memory
    from vv_agent.kernel import _context as context
    from vv_agent.kernel import _context_messages as core
    from vv_agent.kernel import _reducer as reducer
    from vv_agent.memory import input_protection, manager, message_sanitizer
    from vv_agent.session import compaction

    for facade in (input_protection, message_sanitizer):
        for name in facade.__all__:
            value = getattr(core, name)
            assert value.__module__ == core.__name__, name
            assert getattr(facade, name) is value, name
        assert not any(name.startswith("_") and not name.startswith("__") for name in vars(facade))
    for name in ("UnseenInputError", "UnseenInputCapacityError"):
        assert getattr(vv_agent, name) is getattr(core, name), name
    assert "sanitize_for_resume" in memory.__all__
    assert memory.sanitize_for_resume is core.sanitize_for_resume
    for module in (manager, context, reducer):
        assert module.retained_start is core.retained_start
        assert module.filter_empty_assistant_messages is core.filter_empty_assistant_messages
    assert manager.atomic_start is context.atomic_start is core.atomic_start
    assert compaction.retained_start is core.retained_start
    assert compaction.UnseenInputCapacityError is core.UnseenInputCapacityError
