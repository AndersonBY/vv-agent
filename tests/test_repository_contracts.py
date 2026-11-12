from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repository_contracts.py"
check_repository = runpy.run_path(str(SCRIPT))["check_repository"]
MATRIX = "docs/session-kernel-capability-matrix.md"
REFERENCE = "tests/test_sample.py::test_kept"
TEST = """import pytest

@pytest.mark.capability("kept")
def test_kept():
    raise AssertionError("the checker must never execute tests")
"""
TABLE = f"""| Capability ID | Tests |
| --- | --- |
| `kept` | `{REFERENCE}` |
"""


def write(root: Path, path: str, text: str) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    write(tmp_path, MATRIX, TABLE)
    write(tmp_path, "tests/test_sample.py", TEST)
    return tmp_path


def test_valid_bidirectional_mapping_never_imports_source(repository):
    write(repository, "src/sample.py", "raise RuntimeError('do not import')\nclass Kept:\n    async def method(self): pass\n")
    write(
        repository,
        "docs/reference.md",
        "`src/sample.py` `tests/` `py:src/sample.py::Kept.method` `repo:pyproject.toml`\n",
    )
    write(repository, "pyproject.toml", "")
    assert check_repository(repository) == []


@pytest.mark.parametrize(
    ("source", "table", "message"),
    [
        (TEST.replace("def test_kept", "def test_moved"), TABLE, "missing test definition"),
        (TEST.replace('@pytest.mark.capability("kept")\n', ""), TABLE, "missing capability marker"),
        (TEST.replace('capability("kept")', 'capability("other")'), TABLE, "capability marker absent from matrix: other"),
        (TEST, TABLE.replace("| `kept` |", "| `other` |"), "missing capability marker: other"),
        (TEST, TABLE.replace(REFERENCE, "tests/test_missing.py::test_kept"), "missing test definition"),
        (TEST, TABLE.replace("| `kept`", "| `Bad ID`"), "invalid capability ID"),
        (TEST, TABLE + TABLE.splitlines()[-1] + "\n", "duplicate capability ID"),
        (TEST, TABLE.replace(f"`{REFERENCE}`", f"`{REFERENCE}`, `{REFERENCE}`"), "duplicate capability test reference"),
        (TEST, TABLE.replace(REFERENCE, "test_kept"), "matrix test name needs an explicit"),
        (TEST, TABLE.replace(REFERENCE, "tests/test_sample.py"), "capability has no explicit test nodes"),
        (TEST, "# No capability table\n", "no capability rows found"),
        (TEST, TABLE.replace("| `kept` |", "| `kept` | broken |"), "malformed capability row"),
        (TEST.replace('capability("kept")', "capability(CAPABILITY)"), TABLE, "one literal kebab-case ID"),
        (TEST.replace('capability("kept")', 'capability("kept", "other")'), TABLE, "one literal kebab-case ID"),
        (TEST.replace('capability("kept")', 'capability(id="kept")'), TABLE, "one literal kebab-case ID"),
        (TEST.replace('capability("kept")', "capability"), TABLE, "one literal kebab-case ID"),
        (TEST.replace("def test_kept", '@pytest.mark.capability("kept")\ndef test_kept'), TABLE, "duplicate capability marker"),
        (TEST + "\ndef test_kept(): pass\n", TABLE, "duplicate test definition"),
        ("pytestmark = pytest.mark.capability('kept')\ndef test_kept(): pass\n", TABLE, "must be direct decorators"),
        ("@pytest.mark.capability('kept')\nclass TestKept:\n    def test_kept(self): pass\n", TABLE, "must be direct decorators"),
        ("def helper():\n    @pytest.mark.capability('kept')\n    def test_kept(): pass\n", TABLE, "missing test definition"),
    ],
)
def test_deliberate_matrix_violations(repository, source, table, message):
    write(repository, "tests/test_sample.py", source)
    write(repository, MATRIX, table)
    assert any(message in error for error in check_repository(repository))


def test_same_capability_on_unlisted_test_is_not_covered_by_another_node(repository):
    write(repository, "tests/test_other.py", TEST)
    errors = check_repository(repository)
    assert errors == ["tests/test_other.py::test_kept: capability marker absent from matrix: kept"]


def test_async_class_tests_and_multiple_capabilities(repository):
    write(
        repository,
        "tests/test_sample.py",
        'class TestGroup:\n    @pytest.mark.capability("kept")\n'
        '    @pytest.mark.capability("second")\n    async def test_kept(self): pass\n',
    )
    node = "tests/test_sample.py::TestGroup::test_kept"
    write(repository, MATRIX, TABLE.replace(REFERENCE, node) + f"| `second` | `{node}` |\n")
    assert check_repository(repository) == []


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("src/moved.py", "missing repository path"),
        ("docs/moved.md", "missing repository path"),
        ("scripts/moved.py", "missing repository path"),
        ("tests/moved.py", "missing repository path"),
        ("examples/moved.py", "missing repository path"),
        ("examples/", "missing repository path"),
        ("repo:missing.toml", "missing repository path"),
        ("py:src/sample.py::Moved", "missing Python definition"),
        ("py:src/sample.py::Kept.moved", "missing Python definition"),
        ("py:src/sample.py::Imported", "missing Python definition"),
        ("py:src/sample.py::Alias", "missing Python definition"),
        ("py:src/sample.py", "invalid Python reference"),
        ("py:local_settings.py::secret", "invalid Python reference"),
        ("tests/test_sample.py::test_moved", "missing Python definition"),
        ("tests/test_sample.py::test_kept[case]", "invalid Python reference"),
        ("repo:../outside", "repository path escapes root"),
        ("repo:/outside", "invalid repository reference"),
        ("repo:docs/<name>.md", "invalid repository reference"),
    ],
)
def test_deliberate_document_reference_violations(repository, reference, message):
    write(repository, "src/sample.py", "from absent import Imported\nclass Kept: pass\nAlias = Kept\n")
    write(repository, "docs/reference.md", f"`{reference}`\n")
    assert any(message in error for error in check_repository(repository))


def test_symbol_check_ignores_nested_local_functions(repository):
    write(repository, "src/sample.py", "def outer():\n    def local(): pass\n")
    write(repository, "docs/reference.md", "`py:src/sample.py::outer.local`\n")
    assert any("missing Python definition" in error for error in check_repository(repository))


def test_release_notes_root_is_excluded(repository):
    write(repository, "docs/releases/nested/old.md", "`src/removed.py` `py:src/removed.py::old`\n")
    assert check_repository(repository) == []


@pytest.mark.parametrize(
    "directory",
    ["templates", "examples", "reference/templates", "design/examples", "guide/releases", "history", "historical", "archive"],
)
def test_active_document_directories_are_checked(repository, directory):
    path = f"docs/{directory}/current.md"
    write(repository, path, "`src/removed.py`\n")
    assert check_repository(repository) == [f"{path}:1: missing repository path: src/removed.py"]


@pytest.mark.parametrize("inline", ["```example```", "```python `example`"])
def test_backticks_in_fence_info_cannot_hide_following_reference(repository, inline):
    write(repository, "docs/reference.md", f"{inline}\n`src/deleted.py`\n")
    assert check_repository(repository) == ["docs/reference.md:2: missing repository path: src/deleted.py"]


def test_fenced_examples_and_ambiguous_prose_are_not_guessed(repository):
    write(
        repository,
        "docs/reference.md",
        "```python\n`src/missing.py`\n````\n"
        "~~~text\n`py:src/missing.py::gone`\n~~~\n"
        "`uv run python scripts/missing.py` `tests/test_*.py` `src/<package>/file.py` "
        "`Class.method()` `example.txt` `session/kernel.py`\n",
    )
    assert check_repository(repository) == []
    with (repository / "docs/reference.md").open("a") as stream:
        stream.write("`src/missing.py`\n")
    assert any("missing repository path" in error for error in check_repository(repository))


def test_bad_python_is_reported_without_execution(repository):
    write(repository, "tests/test_sample.py", "def broken(:\n")
    assert any("cannot parse Python file" in error for error in check_repository(repository))


def test_symbol_reference_cannot_read_outside_the_repository(repository, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "source.py"
    outside.write_text("class Secret: pass\n")
    link = repository / "src/link.py"
    link.parent.mkdir()
    link.symlink_to(outside)
    write(repository, "docs/reference.md", "`py:src/link.py::Secret`\n")
    assert any("repository path escapes root" in error for error in check_repository(repository))


def test_cli_returns_nonzero_with_actionable_location(repository):
    script = repository / "scripts/check_repository_contracts.py"
    script.parent.mkdir()
    shutil.copyfile(SCRIPT, script)
    write(repository, "docs/reference.md", "Reference moved.\n`src/moved.py`\n")
    result = subprocess.run([sys.executable, str(script)], cwd=repository, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "docs/reference.md:2: missing repository path: src/moved.py" in result.stdout
    assert "1 violation(s)" in result.stdout
    write(repository, "src/moved.py", "")
    result = subprocess.run([sys.executable, str(script)], cwd=repository, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout


def test_matrix_example_tables_do_not_declare_capabilities(repository):
    write(repository, MATRIX, TABLE + "\n```text\n" + TABLE.replace("kept", "example") + "```\n")
    assert check_repository(repository) == []


def test_external_scope_is_explicit_and_not_guessed_from_prose(repository):
    write(repository, "docs/reference.md", "The contract repository owns `docs/missing.md`.\n")
    assert check_repository(repository) == ["docs/reference.md:1: missing repository path: docs/missing.md"]
    write(repository, "docs/reference.md", "The contract owns `../vv-agent-contract/docs/missing.md`.\n")
    assert check_repository(repository) == []
