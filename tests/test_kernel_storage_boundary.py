"""Static storage-boundary mutations use inert source text and never execute it."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_repository_contracts.py"
check_kernel_boundaries = runpy.run_path(str(SCRIPT))["check_kernel_boundaries"]


def source(root: Path, text: str, name: str = "policy.py") -> Path:
    file = root / "src/vv_agent/kernel" / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file


def test_kernel_storage_boundary_passes_current_repository():
    assert check_kernel_boundaries(ROOT) == []


def test_storage_protocol_and_generic_commit_append_calls_remain_allowed(tmp_path):
    source(
        tmp_path,
        "def update(store, transaction, driver, ledger, cache):\n"
        "    lease = store.acquire('session', owner='worker', ttl_ms=1000)\n"
        "    store.append('session', lease=lease, expected_seq=0, commit_id='c', records=())\n"
        "    store._defer_idle_drive(lease, poll_ms=10)\n"
        "    transaction.ack(batch)\n"
        "    driver.commit(records, guarded=True)\n"
        "    ledger.commit()\n"
        "    cache.append(record)\n",
    )
    assert check_kernel_boundaries(tmp_path) == []


@pytest.mark.parametrize("filename", ["_sql.py", "_sqlite.py", "_postgres.py"])
def test_only_the_exact_storage_implementations_may_use_low_level_storage_apis(tmp_path, filename):
    source(
        tmp_path,
        "def run(store, connection):\n"
        "    connection.execute('UPDATE sk_session')\n"
        "    connection.executemany('UPDATE sk_session', [])\n"
        "    connection.executescript('UPDATE sk_session')\n"
        "    store._rows('UPDATE sk_session')\n"
        "    store._one('UPDATE sk_session')\n"
        "    store.connection.transaction()\n"
        "    store.connection.cursor()\n",
        filename,
    )
    assert check_kernel_boundaries(tmp_path) == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("driver.execute('UPDATE sk_session')", "driver.execute"),
        ("driver.executemany('UPDATE sk_session', [])", "driver.executemany"),
        ("driver.executescript('UPDATE sk_session')", "driver.executescript"),
        ("driver._rows('UPDATE sk_session')", "driver._rows"),
        ("driver._one('UPDATE sk_session')", "driver._one"),
        ("driver.connection", "driver.connection"),
        ("driver._conn", "driver._conn"),
        ("driver.cursor()", "driver.cursor"),
        ("with driver.transaction(): pass", "driver.transaction"),
        ("run = driver.execute\nrun('UPDATE sk_session')", "driver.execute"),
    ],
)
def test_driver_and_cursor_bypasses_are_forbidden_outside_storage(tmp_path, code, expected):
    source(tmp_path, "def drive(driver):\n    " + code.replace("\n", "\n    ") + "\n")
    errors = check_kernel_boundaries(tmp_path)
    assert errors == [f"src/vv_agent/kernel/policy.py:2: forbidden kernel storage API: {expected}"]


def test_captured_cursor_acquisition_alias_is_forbidden_outside_storage(tmp_path):
    source(tmp_path, "def drive(driver):\n    get_cursor = driver.cursor\n    get_cursor()\n")
    assert check_kernel_boundaries(tmp_path) == ["src/vv_agent/kernel/policy.py:3: forbidden kernel storage API: driver.cursor"]


@pytest.mark.parametrize("filename", ["nested/_sqlite.py", "storage/_sql.py", "driver/_postgres.py"])
def test_storage_allowlist_does_not_apply_to_nested_or_similarly_named_modules(tmp_path, filename):
    source(tmp_path, "def drive(driver):\n    driver.execute('UPDATE sk_session')\n", filename)
    assert check_kernel_boundaries(tmp_path) == [
        f"src/vv_agent/kernel/{filename}:2: forbidden kernel storage API: driver.execute"
    ]
