#!/usr/bin/env python3
"""Start or resume at most one nightly Colab campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TERMINAL_STATES = frozenset({"terminal_complete", "terminal_failed"})


def _revision(value: str) -> str:
    value = value.lower()
    if REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a full 40-character Git hash")
    return value


def _write_pointer(state_dir: Path, job_name: str) -> None:
    temporary = state_dir / f".current-job.{secrets.token_hex(4)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(job_name + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_dir / "current-job")
    finally:
        if temporary.exists():
            temporary.unlink()


def _current_job(state_dir: Path) -> str | None:
    pointer = state_dir / "current-job"
    try:
        job_name = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if JOB_RE.fullmatch(job_name) is None:
        raise ValueError("current-job must contain a safe job basename")
    return job_name


def _is_released(job_dir: Path) -> bool:
    released = job_dir / "released.json"
    if not released.is_file():
        return False
    value = json.loads(released.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("released.json must contain an object")
    acceptance = value.get("acceptance")
    if (
        value.get("terminal") is not True
        or value.get("state") not in TERMINAL_STATES
        or not isinstance(acceptance, dict)
        or not isinstance(acceptance.get("accepted"), bool)
    ):
        raise ValueError("released.json must contain terminal acceptance")
    return True


def _new_job(state_dir: Path) -> tuple[str, Path, str]:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    job_name = f"{stamp}-{secrets.token_hex(4)}"
    return job_name, state_dir / "jobs" / job_name, f"vvp1-nightly-{job_name}"


def _log_path(state_dir: Path) -> Path:
    logs = state_dir / "logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = logs / f"{stamp}-{secrets.token_hex(4)}.log"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--vv-agent-revision", required=True, type=_revision)
    parser.add_argument("--contract-revision", required=True, type=_revision)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, controller: Path | None = None) -> int:
    args = parse_args(argv)
    controller = controller or Path(__file__).with_name("colab_campaign.py")
    try:
        state_dir = args.state_dir.resolve()
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = os.open(state_dir / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("colab nightly: another invocation holds the state lock", file=sys.stderr)
                return 75

            job_name = _current_job(state_dir)
            if job_name is None:
                job_name, job_dir, session = _new_job(state_dir)
                _write_pointer(state_dir, job_name)
                action = "run"
            else:
                job_dir = state_dir / "jobs" / job_name
                if _is_released(job_dir):
                    job_name, job_dir, session = _new_job(state_dir)
                    _write_pointer(state_dir, job_name)
                    action = "run"
                else:
                    action = "wait"

            log_path = _log_path(state_dir)
            command = [sys.executable, str(controller), action, "--job-dir", str(job_dir)]
            if action == "run":
                command.extend(
                    [
                        "--session",
                        session,
                        "--vv-agent-revision",
                        args.vv_agent_revision,
                        "--contract-revision",
                        args.contract_revision,
                        "--sqlite-chunk-size",
                        "10000",
                        "--postgres-chunk-size",
                        "1000",
                    ]
                )
            print(f"colab nightly {action} {job_name}: {log_path}", file=sys.stderr, flush=True)
            with log_path.open("ab") as stream:
                stream.write(f"action={action} job={job_name}\n".encode())
                stream.flush()
                try:
                    result = subprocess.run(
                        command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=False
                    )
                except OSError as error:
                    stream.write(f"controller failed to start: {error}\n".encode())
                    return 1
            return result.returncode
        finally:
            os.close(lock)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"colab nightly: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
