"""Compaction text keeps its exact bytes and imports without the outer SDK."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vv_agent.kernel._compaction_text import (
    build_compacted_tool_content,
    content_without_recovery_envelope,
    has_recovery_envelope,
    is_microcompacted_tool_content,
)


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_compaction_text_package_import_is_isolated(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        class KernelOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, KernelOnly())
        from vv_agent.kernel import _compaction_text as text
        import vv_agent

        assert Path(text.__file__) == Path(sys.argv[1]) / "vv_agent/kernel/_compaction_text.py"
        assert "Agent" not in vars(vv_agent)
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._compaction_text"
        }
        assert text.COMPACT_MARKER_OPENING == "<Tool Result Compact>"
        assert text.COMPACT_MARKER_CLOSING == "</Tool Result Compact>"
        assert text.has_recovery_envelope('body\\n{"vv_agent_recovery": null}')
        assert text.content_without_recovery_envelope('body\\n{"vv_agent_recovery": null}') == "body"
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


@pytest.mark.parametrize(
    ("content", "head", "tail", "excerpt"),
    [
        ('  clean excerpt  \n{"vv_agent_recovery":{"sha256":"private"}}', 200, 200, "clean excerpt"),
        ("前缀🙂中段尾部", 3, 2, "前缀🙂\n...<snip>...\n尾部"),
        ("abcdef", 4, 4, "abcd\n...<snip>...\ncdef"),
        ("abcdef", 0, 2, "ef"),
        ("abc", 3, 3, "abc"),
        ("abc", -1, -2, ""),
    ],
)
def test_compaction_marker_exact_utf8_bytes(content, head, tail, excerpt):
    actual = build_compacted_tool_content(
        content,
        artifact_path=".vv-agent/artifacts/run/片段.txt",
        tool_name="custom_search",
        excerpt_head_chars=head,
        excerpt_tail_chars=tail,
    )
    expected = (
        "<Tool Result Compact>\n"
        "tool_name: custom_search\n"
        "artifact_path: .vv-agent/artifacts/run/片段.txt\n"
        "retrieval_hint: use read_file on artifact_path if needed\n"
        "excerpt:\n"
        f"{excerpt}\n"
        "</Tool Result Compact>"
    )
    assert actual.encode("utf-8") == expected.encode("utf-8")


@pytest.mark.parametrize(
    ("content", "present", "stripped"),
    [
        (' body \n{"vv_agent_recovery":false}', True, " body "),
        ('body\r\n{"vv_agent_recovery":null}', True, "body\r"),
        ('body\n\n{"vv_agent_recovery":{}}', True, "body\n"),
        ('{"vv_agent_recovery":{}}', False, '{"vv_agent_recovery":{}}'),
        ('body\n{"vv_agent_recovery":{}}\n', False, 'body\n{"vv_agent_recovery":{}}\n'),
        ('body\n{"other":{}}', False, 'body\n{"other":{}}'),
        ('body\n["vv_agent_recovery"]', False, 'body\n["vv_agent_recovery"]'),
        ("body\nnull", False, "body\nnull"),
        ("body\ninvalid JSON", False, "body\ninvalid JSON"),
        ("", False, ""),
        (None, False, None),
    ],
)
def test_recovery_envelope_keeps_last_line_and_newline_semantics(content, present, stripped):
    assert has_recovery_envelope(content) is present
    assert content_without_recovery_envelope(content) == stripped
    if not present:
        assert content_without_recovery_envelope(content) is content


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<Tool Result Compact>\nanything", True),
        ("<Tool Result Compact>unfinished", True),
        (" <Tool Result Compact>", False),
        ("</Tool Result Compact>", False),
        ("", False),
        (None, False),
    ],
)
def test_compacted_marker_recognition_preserves_prefix_semantics(content, expected):
    assert is_microcompacted_tool_content(content) is expected
