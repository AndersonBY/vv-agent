from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class CapturedProcess:
    process: subprocess.Popen[str]
    output_path: Path


def start_captured_process(
    command: list[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    env: Mapping[str, str] | None = None,
) -> CapturedProcess:
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            errors="replace",
            delete=False,
            prefix="vv_agent_process_",
            suffix=".log",
        ) as output_handle:
            output_path = Path(output_handle.name)
            stdin_target: int = subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=stdin_target,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=dict(env) if env is not None else None,
                start_new_session=True,
            )
    except Exception:
        if output_path is not None:
            with suppress(Exception):
                output_path.unlink(missing_ok=True)
        raise

    if stdin_text is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin_text)
        finally:
            process.stdin.close()
        process.stdin = None

    return CapturedProcess(process=process, output_path=output_path)


def read_captured_output(path: Path, *, limit_chars: int) -> str:
    if limit_chars <= 0 or not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit_chars)
    except Exception:
        return ""


def remove_captured_output(path: Path) -> None:
    with suppress(Exception):
        path.unlink(missing_ok=True)


def kill_process_tree(process: subprocess.Popen[str], *, wait_seconds: float = 1.0) -> None:
    if process.poll() is not None:
        return

    normalized_wait = max(float(wait_seconds or 0.0), 0.1)

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(normalized_wait, 1.0),
                check=False,
            )
        except Exception:
            try:
                process.kill()
            except Exception:
                return
        try:
            process.wait(timeout=normalized_wait)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                return
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=normalized_wait)
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.terminate()
        except Exception:
            return

    try:
        process.wait(timeout=normalized_wait)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.kill()
        except Exception:
            return

    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=normalized_wait)
