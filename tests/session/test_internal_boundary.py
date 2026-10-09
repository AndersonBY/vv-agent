"""The internal promotion neither imports retired orchestration nor changes defaults."""

import ast
import os
import subprocess
import sys
from pathlib import Path


def test_kernel_imports_use_f1_modules():
    root = Path(__file__).resolve().parents[2] / "src" / "vv_agent" / "session"
    retired = (
        "vv_agent.checkpoint",
        "vv_agent.runtime.controller",
        "vv_agent.runtime.cycle_runner",
        "vv_agent.deferred",
        "vv_agent.runtime.state",
        "vv_agent.runtime.stores",
        "vv_agent.runtime.backends",
    )
    for source in root.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules += [f"{node.module}.{alias.name}" for alias in node.names]
            assert not any(m == old or m.startswith(old + ".") for m in modules for old in retired), source.name


def test_defaults_and_pure_suite_without_postgres_import():
    script = """
import importlib.abc
import sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'psycopg', 'django'}:
            raise AssertionError(f'unexpected import: {fullname}')
sys.meta_path.insert(0, Forbidden())
import vv_agent
assert not any(name.split('.')[0] in {'psycopg', 'django'} for name in sys.modules)
import pytest
raise SystemExit(pytest.main(['tests/session/test_records_reducer.py', '-q', '-p', 'no:cacheprovider']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
