"""Boundary mutations use inert source text; the checker never imports it."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_repository_contracts.py"
check_kernel_boundaries = runpy.run_path(str(SCRIPT))["check_kernel_boundaries"]


def source(root: Path, text: str, name: str = "probe.py") -> Path:
    file = root / "src/vv_agent/kernel" / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file


def test_extracted_kernel_passes_static_boundaries():
    assert check_kernel_boundaries(ROOT) == []


def test_stdlib_and_internal_imports_are_allowed_without_execution(tmp_path):
    source(tmp_path, 'raise AssertionError("never execute source")\n', "__init__.py")
    source(
        tmp_path,
        "import re as regex\n"
        "from collections.abc import Callable\n"
        "from typing import Any\n"
        "from vv_agent import kernel as core\n"
        "from vv_agent.kernel._schema import _compile_check as check\n"
        "from . import _record_schema\n"
        "from .._schema import _compile_check\n"
        "regex.compile('test')\n",
        "nested/probe.py",
    )
    assert check_kernel_boundaries(tmp_path) == []


@pytest.mark.parametrize(
    "statement",
    [
        "import vv_agent",
        "import vv_agent.session.records",
        "import vv_agent.kernel_extra",
        "from vv_agent import Runner",
        "from vv_agent import kernel, Runner",
        "from ..session import records",
        "from .. import canonical_json",
        "from ...outside import something",
        "import orjson",
        "from jsonschema import Draft202012Validator",
        "if False:\n    import httpx",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vv_agent import Agent",
        "def connect():\n    import psycopg",
        "from math import *",
    ],
)
def test_forbidden_dependencies_include_conditional_and_lazy_imports(tmp_path, statement):
    source(tmp_path, statement + "\n")
    errors = check_kernel_boundaries(tmp_path)
    assert errors and all(error.startswith("src/vv_agent/kernel/probe.py:") for error in errors)
    assert any("import" in error for error in errors)


@pytest.mark.parametrize(
    "statement",
    [
        "import threading\nthreading.Thread(target=lambda: None).start()",
        "import threading as workers\nworkers.Timer(1, lambda: None)",
        "from threading import Thread as Worker\nWorker()",
        "from _thread import start_new_thread as spawn\nspawn(lambda: None, ())",
        "from concurrent.futures import ThreadPoolExecutor as Pool\nPool()",
        "import multiprocessing as workers\nworkers.Process()",
        "import time as clock\nclock.monotonic()",
        "from time import time_ns as now\nnow()",
        "from time import sleep as pause\npause(1)",
        "from datetime import datetime as Date\nDate.now()",
        "from datetime import date as Date\nDate.today()",
        "import socket as net\nnet.socket()",
        "from socket import create_connection as connect\nconnect(('example.invalid', 80))",
        "import urllib.request as net\nnet.urlopen('https://example.invalid')",
        "from urllib import request as net\nnet.urlopen('https://example.invalid')",
        "from http.client import HTTPConnection as Client\nClient('example.invalid')",
        "open('never-open-this')",
        "from builtins import open as read\nread('never-open-this')",
        "import io\nio.open('never-open-this')",
        "from os import stat as inspect\ninspect('never-open-this')",
        "import pathlib as paths\npaths.Path('never-open-this').read_text()",
        "from pathlib import Path as File\nFile('never-open-this').write_text('bad')",
        "import tempfile\ntempfile.TemporaryFile()",
        "import sqlite3\nsqlite3.connect(':memory:')",
        "import threading as workers\nspawn = workers.Thread\nrun = spawn\nrun()",
        "import time\nnow = time.time\nnow()",
        "read = open\nread('never-open-this')",
        "import subprocess as process\nprocess.run(['false'])",
        "__import__('vv_agent.session.records')",
        "from importlib import import_module as load\nload('vv_agent.session.records')",
        "exec('import socket')",
    ],
)
def test_forbidden_calls_include_import_and_direct_assignment_aliases(tmp_path, statement):
    source(tmp_path, statement + "\n")
    assert any("forbidden kernel effect call" in error for error in check_kernel_boundaries(tmp_path))


@pytest.mark.parametrize(
    ("statement", "target"),
    [
        ("print('output')", "print"),
        ("import builtins\nbuiltins.print('output')", "builtins.print"),
        ("import sys\nsys.stdout.write('output')", "sys.stdout.write"),
        ("import sys\nsys.stdout.writelines(['output'])", "sys.stdout.writelines"),
        ("import sys\nsys.stderr.write('output')", "sys.stderr.write"),
        ("import sys\nsys.stderr.writelines(['output'])", "sys.stderr.writelines"),
        ("from builtins import print as emit\nemit('output')", "builtins.print"),
        ("from sys import stdout as output\noutput.write('output')", "sys.stdout.write"),
        ("import sys\nemit = sys.stderr.writelines\nemit(['output'])", "sys.stderr.writelines"),
        ("emit = print\nemit('output')", "print"),
    ],
)
def test_standard_output_calls_and_direct_aliases_are_forbidden(tmp_path, statement, target):
    source(tmp_path, statement + "\n")
    assert any(error.endswith(f"forbidden kernel effect call: {target}") for error in check_kernel_boundaries(tmp_path))


def test_alias_calls_are_checked_before_later_assignments(tmp_path):
    source(tmp_path, "import time as clock\nclock.time()\nimport re\nclock = re\n")
    assert check_kernel_boundaries(tmp_path) == ["src/vv_agent/kernel/probe.py:2: forbidden kernel effect call: time.time"]


def test_new_unimported_nested_module_is_scanned(tmp_path):
    source(tmp_path, "", "__init__.py")
    source(tmp_path, "from vv_agent.session.records import Record\n", "unused/new_module.py")
    assert check_kernel_boundaries(tmp_path) == [
        "src/vv_agent/kernel/unused/new_module.py:1: forbidden kernel import: vv_agent.session.records"
    ]


def test_store_name_does_not_create_a_provisional_exception(tmp_path):
    source(tmp_path, "def connect():\n    import psycopg\n", "storage/postgres.py")
    assert check_kernel_boundaries(tmp_path) == ["src/vv_agent/kernel/storage/postgres.py:2: forbidden kernel import: psycopg"]


def test_outer_session_sources_are_not_claimed_as_pure_core(tmp_path):
    file = tmp_path / "src/vv_agent/session/records.py"
    file.parent.mkdir(parents=True)
    file.write_text("import orjson\n", encoding="utf-8")
    assert check_kernel_boundaries(tmp_path) == []


def test_malformed_unimported_core_source_is_reported(tmp_path):
    source(tmp_path, "def broken(:\n", "unused.py")
    assert any("cannot parse kernel Python" in error for error in check_kernel_boundaries(tmp_path))


def test_repository_gate_cli_enforces_kernel_rules(tmp_path):
    script = tmp_path / "scripts/check_repository_contracts.py"
    script.parent.mkdir()
    script.write_bytes(SCRIPT.read_bytes())
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        'import pytest\n@pytest.mark.capability("sample")\ndef test_sample(): pass\n', encoding="utf-8"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "session-kernel-capability-matrix.md").write_text(
        "| Capability ID | Tests |\n| --- | --- |\n| sample | `tests/test_sample.py::test_sample` |\n", encoding="utf-8"
    )
    source(tmp_path, "import socket\nsocket.socket()\n", "unimported.py")
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 1
    assert "src/vv_agent/kernel/unimported.py:2: forbidden kernel effect call: socket.socket" in result.stdout
