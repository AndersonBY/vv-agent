#!/usr/bin/env python3
"""Lifecycle controller for one detached Colab simulation job."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

FULL_SQLITE = 1_000_000
FULL_POSTGRES = 10_000
SQLITE_CHUNK = 100_000
POSTGRES_CHUNK = 1_000
REQUIRED_COVERED = frozenset({"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I10"})
EXPECTED_PENDING = frozenset({"I9", "I11"})
FULL_SCENARIOS = 21
PART_RE = re.compile(r"^(sqlite|postgres)-(\d{10})-(\d{10})\.json$")
JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
FIXED_ARTIFACTS = frozenset({"process-status.json", "child.stdout.log", "child.stderr.log"})


def run_cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["colab", *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def _check(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit {result.returncode}: {result.stderr.strip()[:500]}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def job_lock(job_dir: Path):
    with (job_dir / ".lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _state(job_dir: Path) -> dict:
    state = _json(job_dir / "state.json")
    if state.get("job_dir") != str(job_dir.resolve()):
        raise ValueError("state job directory mismatch")
    return state


SUPERVISOR = r"""import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


CONFIG = __CONFIG__


def ident(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        return {"pid": pid, "pgid": int(fields[2]), "start_ticks": int(fields[19]), "live": fields[0] != "Z"}
    except (FileNotFoundError, IndexError, ValueError):
        return {"pid": pid, "live": False}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def put(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


job = Path(CONFIG["job_dir"])
try:
    job.mkdir(parents=True, mode=0o700)
except FileExistsError:
    raise SystemExit(3)
source = Path(CONFIG["staged_script"])
script = job / "colab_simulation.py"
if digest(source) != CONFIG["script_sha256"]:
    raise SystemExit(4)
shutil.copyfile(source, script)
if digest(script) != CONFIG["script_sha256"]:
    raise SystemExit(4)
status_path = job / "process-status.json"
with (job / "child.stdout.log").open("xb") as stdout, (job / "child.stderr.log").open("xb") as stderr:
    try:
        child = subprocess.Popen(
            CONFIG["command"],
            cwd=job,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    except Exception as error:
        put(status_path, {"state": "launch_failed", "error_type": type(error).__name__})
        raise SystemExit(5)
    identity = ident(child.pid)
    put(status_path, {"state": "running", "child_identity": identity, "command": CONFIG["command"]})
    try:
        code = child.wait()
        put(
            status_path,
            {
                "state": "exited",
                "exit_code": code,
                "child_identity": identity,
                "command": CONFIG["command"],
                "finished_at": time.time(),
            },
        )
    except BaseException as error:
        put(status_path, {"state": "supervisor_failed", "error_type": type(error).__name__})
        raise
"""


LAUNCH_CELL = r"""import json
from pathlib import Path
import subprocess
import sys
import time


CONFIG = __CONFIG__
supervisor = subprocess.Popen(
    [sys.executable, CONFIG["supervisor_path"]],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
status_path = Path(CONFIG["job_dir"]) / "process-status.json"
deadline = time.monotonic() + 12
while time.monotonic() < deadline:
    try:
        status = json.loads(status_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(0.05)
        continue
    launch = {"status": status}
    print("LAUNCH_JSON=" + json.dumps(launch, sort_keys=True), flush=True)
    break
else:
    print('LAUNCH_JSON={"status":"launch_pending"}', flush=True)
    raise SystemExit(2)
"""


OBSERVER = r"""import hashlib
import json
import os
from pathlib import Path
import re


CONFIG = __CONFIG__
PART_RE = re.compile(r"^(sqlite|postgres)-(\d{10})-(\d{10})\.json$")


def ident(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        return {"pid": pid, "pgid": int(fields[2]), "start_ticks": int(fields[19]), "live": fields[0] != "Z"}
    except (FileNotFoundError, IndexError, ValueError):
        return {"pid": pid, "live": False}


def same(left, right):
    keys = ("pid", "pgid", "start_ticks")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return all(k in left and k in right for k in keys) and tuple(
        left[k] for k in keys
    ) == tuple(right[k] for k in keys)


def record(path, relative):
    if not path.is_file() or path.is_symlink():
        return None
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return {"relative_path": relative, "sha256": value.hexdigest(), "size": path.stat().st_size}


job = Path(CONFIG["job_dir"])
reports = Path(CONFIG["reports_dir"])
status = None
result = None
try:
    status = json.loads((job / "process-status.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    pass
try:
    value = json.loads((reports / "result.json").read_text())
    result = {"status": value.get("status"), "full_acceptance": value.get("full_acceptance"), "has_error": "error" in value}
except (FileNotFoundError, json.JSONDecodeError):
    pass
recorded = (status or {}).get("child_identity")
current = ident(recorded["pid"]) if isinstance(recorded, dict) and isinstance(recorded.get("pid"), int) else None
launch = CONFIG.get("launch_identity")
artifacts = []
for relative in ("process-status.json", "child.stdout.log", "child.stderr.log"):
    meta = record(job / relative, relative)
    if meta:
        artifacts.append(meta)
for entry in os.scandir(reports) if reports.is_dir() else ():
    if entry.is_file(follow_symlinks=False) and (
        entry.name == "result.json" or PART_RE.fullmatch(entry.name)
    ):
        meta = record(Path(entry.path), "reports/" + entry.name)
        if meta:
            artifacts.append(meta)
launch_matches = launch is None or same(launch, recorded)
print("OBSERVE_JSON=" + json.dumps({
    "session": CONFIG["session"],
    "paths": {"job_dir": str(job), "reports_dir": str(reports)},
    "status": status,
    "recorded": recorded,
    "current": current,
    "identity_matches": same(recorded, current),
    "launch_matches": launch_matches,
    "result": result,
    "artifacts": artifacts,
}, sort_keys=True), flush=True)
"""


def _render(template: str, config: dict) -> str:
    return template.replace("__CONFIG__", repr(config))


def _marker(text: str, prefix: str) -> dict | None:
    for line in reversed(text.splitlines()):
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _revision(value: str) -> str:
    value = value.lower()
    if REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a full 40-character Git hash")
    return value


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _interval(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def start(
    job_dir: Path,
    session: str,
    vv_agent_revision: str,
    contract_revision: str,
    sqlite_seeds: int,
    postgres_seeds: int,
    sqlite_chunk_size: int = SQLITE_CHUNK,
    postgres_chunk_size: int = POSTGRES_CHUNK,
    *,
    cli=run_cli,
) -> dict:
    job_dir = job_dir.resolve()
    if JOB_RE.fullmatch(job_dir.name) is None:
        raise ValueError("job directory basename must be a safe job id")
    script = Path(__file__).with_name("colab_simulation.py")
    if not script.is_file():
        raise FileNotFoundError(script)
    job_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    remote_job = f"/content/vv-agent-campaigns/{job_dir.name}"
    remote_reports = remote_job + "/reports"
    staged_script = f"/content/{job_dir.name}-colab_simulation.py"
    supervisor = f"/content/{job_dir.name}-supervisor.py"
    command = [
        "python3",
        remote_job + "/colab_simulation.py",
        "--vv-agent-revision",
        vv_agent_revision,
        "--contract-revision",
        contract_revision,
        "--report-dir",
        remote_reports,
        "--sqlite-seeds",
        str(sqlite_seeds),
        "--postgres-seeds",
        str(postgres_seeds),
        "--sqlite-chunk-size",
        str(sqlite_chunk_size),
        "--postgres-chunk-size",
        str(postgres_chunk_size),
        "--seed",
        "0",
        "--workers",
        "8",
    ]
    state = {
        "job_dir": str(job_dir),
        "session": session,
        "vv_agent_revision": vv_agent_revision,
        "contract_revision": contract_revision,
        "script_sha256": _sha(script),
        "sqlite_seeds": sqlite_seeds,
        "postgres_seeds": postgres_seeds,
        "sqlite_chunk_size": sqlite_chunk_size,
        "postgres_chunk_size": postgres_chunk_size,
        "remote_job_dir": remote_job,
        "remote_reports_dir": remote_reports,
        "command": command,
        "launch_identity": None,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    _write_json(job_dir / "state.json", state)
    _check(cli(["new", "--session", session, "--high-mem"]), "new")
    supervisor_config = {
        "job_dir": remote_job,
        "staged_script": staged_script,
        "script_sha256": state["script_sha256"],
        "command": command,
    }
    supervisor_path = job_dir / "supervisor.py"
    supervisor_path.write_text(_render(SUPERVISOR, supervisor_config), encoding="utf-8")
    _check(cli(["upload", "--session", session, str(script), staged_script]), "upload simulation")
    _check(cli(["upload", "--session", session, str(supervisor_path), supervisor]), "upload supervisor")
    launch_config = {"job_dir": remote_job, "supervisor_path": supervisor}
    launch_path = job_dir / "launch_cell.py"
    launch_path.write_text(_render(LAUNCH_CELL, launch_config), encoding="utf-8")
    launched = cli(["exec", "--session", session, "--file", str(launch_path), "--timeout", "30"])
    _check(launched, "launch exec")
    marker = _marker(launched.stdout, "LAUNCH_JSON=")
    if marker is None or not isinstance(marker.get("status"), dict):
        raise RuntimeError("launch exec did not return the supervisor process status")
    state["launch_identity"] = marker["status"].get("child_identity")
    _write_json(job_dir / "state.json", state)
    return state


def _unknown(reason: str) -> dict:
    return {"state": "unknown", "terminal": False, "reason": reason}


def observe(job_dir: Path, *, cli=run_cli) -> dict:
    job_dir = job_dir.resolve()
    state = _state(job_dir)
    released = job_dir / "released.json"
    if released.is_file():
        value = _json(released)
        acceptance = value.get("acceptance") or {}
        return {
            **value,
            "state": "released",
            "terminal": True,
            "fresh": False,
            "accepted": acceptance.get("accepted") is True,
            "acceptance": acceptance,
        }
    try:
        session_status = cli(["status", "--session", state["session"]])
        if session_status.returncode:
            return _unknown("session status failed")
        if f"Session '{state['session']}' not found" in session_status.stdout + session_status.stderr:
            return _unknown("session is unavailable")
        observer = job_dir / "observer.py"
        config = {
            "job_dir": state["remote_job_dir"],
            "reports_dir": state["remote_reports_dir"],
            "session": state["session"],
            "launch_identity": state["launch_identity"],
        }
        observer.write_text(_render(OBSERVER, config), encoding="utf-8")
        result = cli(["exec", "--session", state["session"], "--file", str(observer), "--timeout", "30"])
        if result.returncode:
            return _unknown("observer exec failed")
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        return _unknown(f"observation failed: {type(error).__name__}")
    observed = _marker(result.stdout, "OBSERVE_JSON=")
    if observed is None or observed.get("session") != state["session"]:
        return _unknown("observer marker is missing")
    expected_paths = {"job_dir": state["remote_job_dir"], "reports_dir": state["remote_reports_dir"]}
    if observed.get("paths") != expected_paths or observed.get("launch_matches") is not True:
        return _unknown("observer paths or launch identity do not match")
    status = observed.get("status")
    if not isinstance(status, dict):
        return _unknown("process status is missing")
    if status.get("state") == "launch_failed":
        observed.update(state="terminal_failed", terminal=True, accepted=False, reason="child launch failed")
        return observed
    current = observed.get("current")
    live_match = observed.get("identity_matches") is True and isinstance(current, dict) and current.get("live")
    if status.get("state") == "running":
        if live_match:
            observed.update(state="running", terminal=False, fresh=True)
        else:
            observed.update(_unknown("running status has no matching live child"))
        return observed
    if status.get("state") != "exited":
        observed.update(_unknown("job has no live child or terminal exit status"))
        return observed
    if isinstance(current, dict) and current.get("live") and not observed.get("identity_matches"):
        observed.update(_unknown("pid was reused after terminal status"))
        return observed
    code = status.get("exit_code")
    if not isinstance(code, int):
        observed.update(_unknown("terminal status has no integer exit code"))
        return observed
    observed["exit_code"] = code
    if code != 0:
        observed.update(state="terminal_failed", terminal=True, accepted=False, reason=f"child exited {code}")
        return observed
    result_meta = observed.get("result")
    if not isinstance(result_meta, dict):
        observed.update(_unknown("child exited zero without reports/result.json"))
    elif result_meta.get("status") == "complete":
        observed.update(state="terminal_complete", terminal=True, fresh=True, reason="complete result")
    else:
        observed.update(state="terminal_failed", terminal=True, accepted=False, reason="result reports failure")
    return observed


def _artifacts(observation: dict) -> list[dict]:
    artifacts = observation.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("observer artifact list is missing")
    safe = []
    seen = set()
    for item in artifacts:
        relative, digest = item.get("relative_path"), item.get("sha256")
        if relative in seen or not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("observer artifact is malformed or duplicated")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("observer artifact hash is malformed")
        allowed_part = relative.startswith("reports/") and PART_RE.fullmatch(relative[8:])
        if relative in FIXED_ARTIFACTS or relative == "reports/result.json" or allowed_part:
            pass
        else:
            raise ValueError("observer returned an unowned artifact path")
        seen.add(relative)
        safe.append(item)
    return safe


def _download(job_dir: Path, state: dict, observation: dict, *, terminal: bool, cli=run_cli) -> list[dict]:
    evidence = job_dir / "evidence"
    transfers = []
    for item in _artifacts(observation):
        relative = item["relative_path"]
        if not terminal and relative in FIXED_ARTIFACTS:
            continue
        local = evidence / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file():
            if _sha(local) != item["sha256"]:
                raise ValueError(f"local evidence conflict: {relative}")
            transfers.append({"relative_path": relative, "sha256": item["sha256"], "status": "matched"})
            continue
        incoming = local.with_name(f".{local.name}.{secrets.token_hex(4)}")
        result = cli(
            [
                "download",
                "--session",
                state["session"],
                state["remote_job_dir"] + "/" + relative,
                str(incoming),
            ]
        )
        try:
            _check(result, f"download {relative}")
            if not incoming.is_file() or _sha(incoming) != item["sha256"]:
                raise ValueError(f"download hash mismatch: {relative}")
            os.replace(incoming, local)
        finally:
            with contextlib.suppress(FileNotFoundError):
                incoming.unlink()
        transfers.append({"relative_path": relative, "sha256": item["sha256"], "status": "downloaded"})
    _write_json(evidence / "collection-manifest.json", {"terminal": terminal, "transfers": transfers})
    return transfers


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_evidence(evidence: Path, state: dict, observation: dict) -> dict:
    _require(observation.get("exit_code") == 0, "child exit code is not zero")
    result = _json(evidence / "reports/result.json")
    _require(result.get("status") == "complete", "result is not complete")
    identity = result.get("identity") or {}
    _require(identity.get("vv_agent_revision") == state["vv_agent_revision"], "source revision mismatch")
    _require(identity.get("contract_revision") == state["contract_revision"], "contract revision mismatch")
    campaigns = result.get("campaigns")
    if not isinstance(campaigns, dict) or set(campaigns) != {"sqlite", "postgres"}:
        raise ValueError("both result campaigns are required")
    configured = {"sqlite": state["sqlite_seeds"], "postgres": state["postgres_seeds"]}
    chunks = {
        "sqlite": state["sqlite_chunk_size"],
        "postgres": state["postgres_chunk_size"],
    }
    parts: dict[str, list[dict]] = {"sqlite": [], "postgres": []}
    code_digests = set()
    contract_manifests = set()
    files = sorted(path for path in (evidence / "reports").iterdir() if path.is_file() and path.name != "result.json")
    for path in files:
        match = PART_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected part filename {path.name}")
        backend, start_text, count_text = match.groups()
        start, count = int(start_text), int(count_text)
        part = _json(path)
        _require(
            part.get("backend") == backend and part.get("seed") == start and part.get("seeds") == count,
            f"range mismatch {path.name}",
        )
        _require("error" not in part, f"producer error in {path.name}")
        _require(part.get("code_revision") == state["vv_agent_revision"], f"source mismatch in {path.name}")
        _require(part.get("contract_revision") == state["contract_revision"], f"contract mismatch in {path.name}")
        _require(part.get("dirty") is False and part.get("contract_dirty") is False, f"dirty identity in {path.name}")
        code_digest = part.get("code_digest")
        contract_manifest = part.get("contract_manifest_sha256")
        if not isinstance(code_digest, str) or re.fullmatch(r"[0-9a-f]{64}", code_digest) is None:
            raise ValueError(f"code digest malformed in {path.name}")
        if not isinstance(contract_manifest, str) or re.fullmatch(r"[0-9a-f]{64}", contract_manifest) is None:
            raise ValueError(f"contract manifest malformed in {path.name}")
        _require(part.get("in_memory_mutations") == [], f"source mutations present in {path.name}")
        code_digests.add(code_digest)
        contract_manifests.add(contract_manifest)
        _require(
            isinstance(part.get("pending"), dict) and frozenset(part["pending"]) == EXPECTED_PENDING,
            f"pending mismatch in {path.name}",
        )
        scenarios = part.get("scenarios")
        if not isinstance(scenarios, dict) or not all(isinstance(value, int) and value > 0 for value in scenarios.values()):
            raise ValueError(f"scenarios malformed in {path.name}")
        if sum(scenarios.values()) != count:
            raise ValueError(f"scenario count mismatch in {path.name}")
        parts[backend].append({"start": start, "count": count, "name": path.name, "data": part})
    _require(len(code_digests) == 1 and len(contract_manifests) == 1, "parts do not share one code and contract identity")
    summary: dict[str, dict[str, Any]] = {}
    for backend in ("sqlite", "postgres"):
        ordered = sorted(parts[backend], key=lambda item: item["start"])
        if not ordered:
            raise ValueError(f"missing {backend} parts")
        cursor, remaining = 0, configured[backend]
        for part in ordered:
            _require(part["start"] == cursor, f"{backend} ranges are not contiguous")
            _require(part["count"] == min(chunks[backend], remaining), f"{backend} chunk count mismatch")
            cursor += part["count"]
            remaining -= part["count"]
        _require(cursor == configured[backend], f"{backend} actual total mismatch")
        aggregate = {}
        covered = set()
        for part in ordered:
            for name, count in part["data"]["scenarios"].items():
                aggregate[name] = aggregate.get(name, 0) + count
            covered.update(part["data"].get("covered", []))
        campaign = campaigns[backend]
        _require(campaign.get("seed") == 0 and campaign.get("seeds") == configured[backend], f"result {backend} total mismatch")
        _require(campaign.get("scenarios") == aggregate, f"result {backend} scenarios differ from part sum")
        _require(set(campaign.get("covered", [])) == covered, f"result {backend} coverage differs from part union")
        _require(sorted(campaign.get("reports", [])) == [part["name"] for part in ordered], f"result {backend} manifest mismatch")
        _require(
            isinstance(campaign.get("pending"), dict) and frozenset(campaign["pending"]) == EXPECTED_PENDING,
            f"result {backend} pending mismatch",
        )
        summary[backend] = {"seeds": cursor, "reports": len(ordered), "scenarios": aggregate}
    postgres_cleanup = (result.get("runtime") or {}).get("postgres") or {}
    _require(
        postgres_cleanup.get("temporary_role_removed") is True and postgres_cleanup.get("temporary_databases_removed") is True,
        "PostgreSQL cleanup is unverified",
    )
    full = configured["sqlite"] >= FULL_SQLITE and configured["postgres"] >= FULL_POSTGRES
    claimed_full = result.get("full_acceptance")
    _require(isinstance(claimed_full, bool), "full_acceptance is missing")
    if not full:
        _require(claimed_full is False, "partial totals must not claim full acceptance")
        return {"accepted": True, "full_acceptance": False, "backends": summary}
    _require(claimed_full is True, "full totals did not obtain full acceptance")
    for backend in ("sqlite", "postgres"):
        _require(REQUIRED_COVERED.issubset(campaigns[backend].get("covered", [])), f"result {backend} coverage is incomplete")
        _require(len(summary[backend]["scenarios"]) == FULL_SCENARIOS, f"{backend} scenario count is not {FULL_SCENARIOS}")
    for part in parts["sqlite"]:
        storage = (part["data"].get("environment") or {}).get("sqlite_temporary_storage") or {}
        observations = storage.get("observations")
        if storage.get("verified") is not True or not isinstance(observations, list) or not observations:
            raise ValueError(f"SQLite storage unexplained in {part['name']}")
        if not all(item.get("filesystem") not in (None, "tmpfs") for item in observations):
            raise ValueError(f"SQLite storage is not ordinary filesystem in {part['name']}")
    return {"accepted": True, "full_acceptance": True, "backends": summary}


def collect(job_dir: Path, observation: dict | None = None, *, cli=run_cli) -> dict:
    job_dir = job_dir.resolve()
    state = _state(job_dir)
    with job_lock(job_dir):
        released = job_dir / "released.json"
        if released.is_file():
            value = _json(released)
            acceptance = value.get("acceptance") or {}
            return {
                **value,
                "state": "released",
                "terminal": True,
                "accepted": acceptance.get("accepted") is True,
                "acceptance": acceptance,
            }
        observation = observation or observe(job_dir, cli=cli)
        if observation.get("state") == "unknown":
            return observation
        terminal = observation.get("state") in {"terminal_complete", "terminal_failed"}
        _download(job_dir, state, observation, terminal=terminal, cli=cli)
        if not terminal:
            return {**observation, "evidence_collected": True}
        try:
            acceptance = validate_evidence(job_dir / "evidence", state, observation)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            acceptance = {"accepted": False, "full_acceptance": False, "reason": str(error)}
        _check(cli(["stop", "--session", state["session"]]), "stop")
        release = {
            "terminal": True,
            "state": observation["state"],
            "acceptance": acceptance,
            "released_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        _write_json(job_dir / "released.json", release)
        return {**observation, **release, "accepted": acceptance["accepted"]}


def wait(job_dir: Path, *, interval: float = 60.0, max_observations: int = 0, cli=run_cli) -> dict:
    count = 0
    while True:
        observation = observe(job_dir, cli=cli)
        count += 1
        if observation.get("state") in {"terminal_complete", "terminal_failed"}:
            try:
                result = collect(job_dir, observation, cli=cli)
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
                return {"state": "unknown", "terminal": True, "reason": f"terminal collection failed: {error}"}
            return result
        if observation.get("state") == "unknown":
            return observation
        if observation.get("state") != "running":
            return observation
        if observation.get("state") == "running":
            try:
                collect(job_dir, observation, cli=cli)
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
                return {"state": "unknown", "terminal": False, "reason": f"partial collection failed: {error}"}
        if count >= max_observations > 0:
            return observation
        time.sleep(interval)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-dir", type=Path, required=True)


def _add_start(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", required=True)
    parser.add_argument("--vv-agent-revision", required=True, type=_revision)
    parser.add_argument("--contract-revision", required=True, type=_revision)
    parser.add_argument("--sqlite-seeds", type=_positive, default=FULL_SQLITE)
    parser.add_argument("--postgres-seeds", type=_positive, default=FULL_POSTGRES)
    parser.add_argument("--sqlite-chunk-size", type=_positive, default=SQLITE_CHUNK)
    parser.add_argument("--postgres-chunk-size", type=_positive, default=POSTGRES_CHUNK)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("start", "run"):
        command = commands.add_parser(name)
        _add_common(command)
        _add_start(command)
        if name == "run":
            command.add_argument("--interval", type=_interval, default=60.0)
            command.add_argument("--max-observations", type=_nonnegative, default=0)
    for name in ("status", "collect", "wait"):
        command = commands.add_parser(name)
        _add_common(command)
        if name == "wait":
            command.add_argument("--interval", type=_interval, default=60.0)
            command.add_argument("--max-observations", type=_nonnegative, default=0)
    return parser.parse_args(argv)


def _public(value: dict) -> dict:
    return {
        key: value[key] for key in ("state", "terminal", "fresh", "accepted", "acceptance", "reason", "exit_code") if key in value
    }


def main(argv: list[str] | None = None, *, cli=run_cli) -> int:
    args = parse_args(argv)
    try:
        if args.action in {"start", "run"}:
            start(
                args.job_dir,
                args.session,
                args.vv_agent_revision,
                args.contract_revision,
                args.sqlite_seeds,
                args.postgres_seeds,
                args.sqlite_chunk_size,
                args.postgres_chunk_size,
                cli=cli,
            )
        if args.action in {"run", "wait"}:
            result = wait(args.job_dir, interval=args.interval, max_observations=args.max_observations, cli=cli)
        elif args.action == "status":
            result = observe(args.job_dir, cli=cli)
        elif args.action == "collect":
            result = collect(args.job_dir, cli=cli)
        else:
            result = observe(args.job_dir, cli=cli)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error), "action": args.action}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(_public(result), sort_keys=True))
    if result.get("state") == "unknown":
        return 2
    if args.action in {"wait", "run"} and result.get("terminal") is not True:
        return 2
    if result.get("terminal") and result.get("accepted") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
