from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "record_codec_fixtures.py"
sys.path.insert(0, str(REPO_ROOT))
try:
    RECORD_CODEC = runpy.run_path(str(SCRIPT))
finally:
    sys.path.remove(str(REPO_ROOT))
SOURCE_FILES = {
    "src/vv_agent/module.py": b"module = 'source'\n",
    "scripts/record_codec_fixtures.py": b"capture = 'source'\n",
    "scripts/s1_fixture_values.py": b"values = 'source'\n",
}


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for relative, body in SOURCE_FILES.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    git(repo, "add", "--", *SOURCE_FILES)
    git(
        repo,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    return repo


@pytest.mark.parametrize("relative", SOURCE_FILES)
def test_source_digest_and_dirty_cover_each_capture_source(source_repo: Path, relative: str) -> None:
    digest = RECORD_CODEC["_source_tree_digest"](source_repo)
    (source_repo / relative).write_bytes((source_repo / relative).read_bytes() + b"changed\n")
    assert RECORD_CODEC["_source_tree_digest"](source_repo) != digest
    assert RECORD_CODEC["_source_worktree_dirty"](source_repo)


def test_source_digest_covers_untracked_source_files(source_repo: Path) -> None:
    digest = RECORD_CODEC["_source_tree_digest"](source_repo)
    untracked = source_repo / "src/vv_agent/untracked.py"
    untracked.write_bytes(b"untracked = True\n")
    assert RECORD_CODEC["_source_tree_digest"](source_repo) != digest
    assert RECORD_CODEC["_source_worktree_dirty"](source_repo)


def test_source_digest_includes_source_path_names(source_repo: Path) -> None:
    digest = RECORD_CODEC["_source_tree_digest"](source_repo)
    (source_repo / "src/vv_agent/module.py").rename(source_repo / "src/vv_agent/renamed.py")
    assert RECORD_CODEC["_source_tree_digest"](source_repo) != digest


def test_fixture_output_does_not_dirty_source_provenance(source_repo: Path) -> None:
    digest = RECORD_CODEC["_source_tree_digest"](source_repo)
    output = source_repo / "tests/fixtures/parity/trusted_storage_s1.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"{}\n")
    assert RECORD_CODEC["_source_tree_digest"](source_repo) == digest
    assert not RECORD_CODEC["_source_worktree_dirty"](source_repo)


def test_app_server_capture_uses_explicit_unsigned_local_policy() -> None:
    first = RECORD_CODEC["app_server_error"]()
    second = RECORD_CODEC["app_server_error"]()

    assert first == second
    assert first == {
        "code": -32602,
        "message": "stored record chain digest mismatch at seq 1",
    }
