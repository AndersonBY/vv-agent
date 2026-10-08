from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "contract_snapshot.py"


@pytest.fixture
def matrix(tmp_path: Path) -> dict[str, Any]:
    shutil.copyfile(REPO_ROOT / "contract.lock.json", tmp_path / "contract.lock.json")
    version = json.loads((tmp_path / "contract.lock.json").read_text())["contract_version"]
    return {
        "schema_version": 2,
        "contract_version": version,
        "status": "verified",
        "required_implementations": ["python"],
        "implementations": {
            "python": {
                "contract_version": version,
                "status": "verified",
                "verified_revision": "a" * 40,
            },
            "rust": {
                "contract_version": "23.0.0",
                "package_series": "0.21.x",
                "status": "frozen",
                "verified_revision": "b" * 40,
            },
        },
        "cross_repository_run": "https://example.invalid/actions/runs/1",
    }


def _adoption(tmp_path: Path, matrix: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "adoption",
            "--implementation",
            "python",
            "--matrix",
            str(matrix_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("include_rust", [True, False], ids=["frozen-rust", "no-rust"])
def test_python_verified_adoption_does_not_require_rust(tmp_path: Path, matrix: dict[str, Any], include_rust: bool) -> None:
    if not include_rust:
        del matrix["implementations"]["rust"]

    result = _adoption(tmp_path, matrix)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "contract_version": matrix["contract_version"],
        "implementation": "python",
        "status": "verified",
        "verified_revision": "a" * 40,
        "release_revision": None,
        "cross_repository_run": matrix["cross_repository_run"],
    }


def test_python_pending_adoption_fails(tmp_path: Path, matrix: dict[str, Any]) -> None:
    matrix["implementations"]["python"]["status"] = "pending"

    result = _adoption(tmp_path, matrix)

    assert result.returncode == 1
    assert "python implementation is not centrally verified" in result.stderr


@pytest.mark.parametrize("schema_version", [1, True, 2.0, "2"])
def test_adoption_rejects_old_or_malformed_schema(tmp_path: Path, matrix: dict[str, Any], schema_version: object) -> None:
    matrix["schema_version"] = schema_version

    result = _adoption(tmp_path, matrix)

    assert result.returncode == 1
    assert "support matrix must be a schema_version=2 object" in result.stderr


def test_python_adoption_requires_locked_version(tmp_path: Path, matrix: dict[str, Any]) -> None:
    matrix["implementations"]["python"]["contract_version"] = "0.0.0"

    result = _adoption(tmp_path, matrix)

    assert result.returncode == 1
    assert "python pinned version does not match contract.lock.json" in result.stderr


def test_python_adoption_requires_central_verification(tmp_path: Path, matrix: dict[str, Any]) -> None:
    matrix["status"] = "pending-adoption"

    result = _adoption(tmp_path, matrix)

    assert result.returncode == 1
    assert f"contract {matrix['contract_version']} is not centrally verified" in result.stderr
