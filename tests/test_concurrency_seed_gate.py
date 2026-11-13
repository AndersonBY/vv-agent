from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_concurrency_seeds.py"


def git(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Seed Gate Tests")
    git(repo, "config", "user.email", "seed-gate@example.invalid")
    return repo


def commit(repo: Path, message: str) -> str:
    sequence = len(list((repo / "history").glob("*.txt"))) if (repo / "history").exists() else 0
    path = repo / "history" / f"{sequence:03}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"{sequence}\n", encoding="utf-8")
    git(repo, "add", str(path.relative_to(repo)))
    git(repo, "commit", "-q", "-F", "-", input_text=message + "\n")
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def run_gate(repo: Path, base: str, head: str = "HEAD", env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo), "--base", base, "--head", head],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env=env,
    )


def test_classified_fix_without_seed_fails(tmp_path):
    repo = init_repo(tmp_path)
    base = commit(repo, "test: establish base")
    commit(repo, "fix(concurrency): valid earlier fix\n\nSimulation-Seed: 17")
    commit(repo, "fix(concurrency): preserve stale lease fencing\n\nNo reproducing seed is recorded.")

    result = run_gate(repo, base)

    assert result.returncode == 1
    assert "requires Simulation-Seed: <nonnegative decimal>" in result.stderr


def test_zero_and_multiple_seeds_pass(tmp_path):
    repo = init_repo(tmp_path)
    base = commit(repo, "test: establish base")
    commit(repo, "fix(concurrency)!: retain control binding\n\nSimulation-Seed: 0\nSimulation-Seed: 17")

    result = run_gate(repo, base)

    assert result.returncode == 0, result.stderr
    assert "checked 1 commit(s); 1 classified" in result.stdout


def test_malformed_seed_is_rejected(tmp_path):
    repo = init_repo(tmp_path)
    base = commit(repo, "test: establish base")
    commit(repo, "fix(concurrency): preserve fencing\n\nSimulation-Seed: not-a-seed")

    result = run_gate(repo, base)

    assert result.returncode == 1
    assert "requires Simulation-Seed" in result.stderr


def test_identical_base_and_head_pass_with_zero_commits(tmp_path):
    repo = init_repo(tmp_path)
    head = commit(repo, "fix(concurrency): historical commit before selected range")

    result = run_gate(repo, head, head)

    assert result.returncode == 0, result.stderr
    assert "checked 0 commit(s); 0 classified" in result.stdout


def test_ordinary_refactor_touching_concurrency_files_is_unaffected(tmp_path):
    repo = init_repo(tmp_path)
    base = commit(repo, "test: establish base")
    commit(repo, "refactor(kernel): extract lease recovery without behavior change")
    commit(repo, "test(concurrency): cover existing scheduler behavior")

    result = run_gate(repo, base)

    assert result.returncode == 0, result.stderr
    assert "checked 2 commit(s); 0 classified" in result.stdout


def test_merge_and_force_updated_ranges_use_head_set_difference(tmp_path):
    repo = init_repo(tmp_path)
    root = commit(repo, "test: establish repository root")
    base = commit(repo, "test: establish base")
    git(repo, "switch", "-q", "-c", "feature")
    commit(repo, "fix(concurrency): serialize renewal publication\n\nSimulation-Seed: 17")
    git(repo, "switch", "-q", "main")
    commit(repo, "refactor(session): unrelated mainline extraction")
    git(repo, "merge", "-q", "--no-ff", "feature", "-m", "Merge feature")
    merged = git(repo, "rev-parse", "HEAD").stdout.strip()
    git(repo, "switch", "-q", "-c", "old-head", root)
    old_head = commit(repo, "refactor(session): abandoned push base")
    git(repo, "switch", "-q", "main")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", old_head, merged],
        check=False,
    )

    merged_result = run_gate(repo, base, merged)
    force_update_result = run_gate(repo, old_head, merged)

    assert ancestor.returncode == 1
    assert merged_result.returncode == 0, merged_result.stderr
    assert "checked 3 commit(s); 1 classified" in merged_result.stdout
    assert force_update_result.returncode == 0, force_update_result.stderr
    assert "checked 4 commit(s); 1 classified" in force_update_result.stdout


@pytest.mark.skipif(os.name == "nt", reason="POSIX Git wrapper fixture")
def test_git_is_found_from_inherited_non_system_path(tmp_path):
    repo = init_repo(tmp_path)
    base = commit(repo, "test: establish base")
    commit(repo, "fix(concurrency): preserve fencing\n\nSimulation-Seed: 0")
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    marker = tmp_path / "git-wrapper-used"
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        f'#!/bin/sh\nprintf used > "$GIT_WRAPPER_MARKER"\nexec {shlex.quote(real_git)} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env = {**os.environ, "PATH": f"{wrapper_dir}{os.pathsep}{os.environ['PATH']}", "GIT_WRAPPER_MARKER": str(marker)}

    result = run_gate(repo, base, env=env)

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "used"
