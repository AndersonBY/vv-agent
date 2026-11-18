"""Require a simulation-seed reference on explicit concurrency-fix commits."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CONCURRENCY_FIX = re.compile(r"fix\(concurrency\)!?:[ \t]+\S")
SEED_TRAILER = re.compile(r"Simulation-Seed:[ \t]+(?:0|[1-9][0-9]*)[ \t]*")
ZERO_COMMIT = re.compile(r"0{40}(?:0{24})?")
LOG_FORMAT = "%H%x1f%s%x1f%b"


class GateError(RuntimeError):
    pass


def git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "--no-pager", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def range_commits(repo_root: Path, base: str, head: str) -> list[tuple[str, str, str]]:
    base = git(repo_root, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
    head = git(repo_root, "rev-parse", "--verify", f"{head}^{{commit}}").strip()
    output = git(repo_root, "log", "--reverse", "-z", f"--format={LOG_FORMAT}", f"{base}..{head}")
    if not output:
        return []
    commits = []
    for record in output.rstrip("\x00").split("\x00"):
        fields = record.split("\x1f", 2)
        if len(fields) != 3:
            raise GateError("git log returned an unexpected commit-message format")
        commits.append((fields[0], fields[1], fields[2]))
    return commits


def missing_seeds(repo_root: Path, base: str, head: str) -> tuple[int, int, list[str]]:
    commits = range_commits(repo_root, base, head)
    classified = 0
    missing = []
    for commit, subject, body in commits:
        if CONCURRENCY_FIX.match(subject) is None:
            continue
        classified += 1
        if not any(SEED_TRAILER.fullmatch(line) for line in body.splitlines()):
            missing.append(commit)
    return len(commits), classified, missing


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    command_parser.add_argument("--base", required=True, help="base commit/ref supplied by CI")
    command_parser.add_argument("--head", default="HEAD", help="head commit/ref supplied by CI")
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if ZERO_COMMIT.fullmatch(args.base):
        print("Concurrency seed gate skipped: no previous commit for this push.")
        return 0
    try:
        checked, classified, missing = missing_seeds(args.repo_root, args.base, args.head)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or f"exit {error.returncode}"
        print(f"Concurrency seed gate: git command failed: {detail}", file=sys.stderr)
        return 2
    except GateError as error:
        print(f"Concurrency seed gate: {error}", file=sys.stderr)
        return 2
    if missing:
        print("Concurrency fix commits must cite a reproducing simulation seed:", file=sys.stderr)
        for commit in missing:
            print(f"  {commit[:12]} requires Simulation-Seed: <nonnegative decimal>", file=sys.stderr)
        return 1
    print(f"Concurrency seed gate: checked {checked} commit(s); {classified} classified concurrency fix(es).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
