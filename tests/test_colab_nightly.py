from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_nightly", ROOT / "scripts/colab_nightly.py")
assert SPEC is not None and SPEC.loader is not None
nightly_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly_module)
nightly_main: Any = cast(Any, nightly_module).main

FIRST = "a" * 40
SECOND = "b" * 40


def fake_controller(
    tmp_path: Path, *, run_code: int = 0, wait_code: int = 2, released: dict[str, object] | None = None
) -> tuple[Path, Path]:
    calls = tmp_path / "calls.jsonl"
    script = tmp_path / "fake_controller.py"
    template = r"""import json
import sys
from pathlib import Path

CALLS = Path(__CALLS__)
RUN_CODE = __RUN_CODE__
WAIT_CODE = __WAIT_CODE__
RELEASED = __RELEASED__
args = sys.argv[1:]
action = args[0]
job = Path(args[args.index("--job-dir") + 1]).resolve()
pointer = job.parent.parent / "current-job"
number = len(CALLS.read_text().splitlines()) + 1 if CALLS.exists() else 1
record = {"args": args, "number": number, "pointer": pointer.read_text().strip() if pointer.exists() else None}
with CALLS.open("a") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
if action == "run":
    job.mkdir(parents=True, exist_ok=False)
print(f"stdout call {number}", flush=True)
print(f"stderr call {number}", file=sys.stderr, flush=True)
if action == "wait" and RELEASED is not None:
    (job / "released.json").write_text(json.dumps(RELEASED))
raise SystemExit(RUN_CODE if action == "run" else WAIT_CODE)
"""
    script.write_text(
        template.replace("__CALLS__", repr(str(calls)))
        .replace("__RUN_CODE__", str(run_code))
        .replace("__WAIT_CODE__", str(wait_code))
        .replace("__RELEASED__", repr(released))
    )
    return script, calls


def invoke(state_dir: Path, controller: Path, *, vv_agent_revision: str = FIRST, contract_revision: str = SECOND) -> int:
    return nightly_main(
        [
            "--state-dir",
            str(state_dir),
            "--vv-agent-revision",
            vv_agent_revision,
            "--contract-revision",
            contract_revision,
        ],
        controller=controller,
    )


def calls(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def set_current(state_dir: Path, job_name: str, *, released: object | None = None) -> Path:
    job_dir = state_dir / "jobs" / job_name
    job_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "current-job").write_text(job_name + "\n")
    if released is not None:
        (job_dir / "released.json").write_text(json.dumps(released))
    return job_dir


def test_fresh_state_runs_one_full_campaign(tmp_path: Path):
    controller, call_log = fake_controller(tmp_path)
    state_dir = tmp_path / "state"

    assert invoke(state_dir, controller) == 0

    recorded = calls(call_log)
    assert len(recorded) == 1 and recorded[0]["args"][0] == "run"
    arguments = recorded[0]["args"]
    assert arguments[arguments.index("--vv-agent-revision") + 1] == FIRST
    assert arguments[arguments.index("--contract-revision") + 1] == SECOND
    assert arguments[arguments.index("--sqlite-chunk-size") + 1] == "10000"
    assert arguments[arguments.index("--postgres-chunk-size") + 1] == "1000"
    assert "--sqlite-seeds" not in arguments and "--postgres-seeds" not in arguments and "--workers" not in arguments
    job_name = (state_dir / "current-job").read_text().strip()
    assert recorded[0]["pointer"] == job_name
    assert (state_dir / "jobs" / job_name).is_dir()


def test_unreleased_job_only_waits_and_ignores_changed_revisions(tmp_path: Path):
    controller, call_log = fake_controller(tmp_path)
    state_dir = tmp_path / "state"
    job_dir = set_current(state_dir, "old-job")
    state_path = job_dir / "state.json"
    state_path.write_text('{"state": "running"}\n')
    before = state_path.read_bytes()

    assert invoke(state_dir, controller, vv_agent_revision=SECOND, contract_revision=FIRST) == 2

    recorded = calls(call_log)
    assert [item["args"][0] for item in recorded] == ["wait"]
    assert recorded[0]["args"][recorded[0]["args"].index("--job-dir") + 1] == str(job_dir)
    assert (state_dir / "current-job").read_text().strip() == "old-job"
    assert state_path.read_bytes() == before


def test_unknown_run_is_waited_on_next_invocation(tmp_path: Path):
    controller, call_log = fake_controller(tmp_path, run_code=2, wait_code=2)
    state_dir = tmp_path / "state"

    assert invoke(state_dir, controller) == 2
    first_job = (state_dir / "current-job").read_text().strip()
    (state_dir / "jobs" / first_job / "state.json").write_text('{"state": "unknown"}\n')
    assert invoke(state_dir, controller, vv_agent_revision=SECOND, contract_revision=FIRST) == 2

    recorded = calls(call_log)
    assert [item["args"][0] for item in recorded] == ["run", "wait"]
    assert all(item["pointer"] == first_job for item in recorded)
    assert [path.name for path in (state_dir / "jobs").iterdir()] == [first_job]


def test_controller_start_oserror_keeps_pointer_for_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    controller, call_log = fake_controller(tmp_path, wait_code=2)
    state_dir = tmp_path / "state"

    def fail_run(*args: object, **kwargs: object) -> None:
        raise OSError("controller could not start")

    with monkeypatch.context() as patcher:
        patcher.setattr(cast(Any, nightly_module).subprocess, "run", fail_run)
        assert invoke(state_dir, controller) == 1

    first_job = (state_dir / "current-job").read_text().strip()
    assert not call_log.exists()
    assert not (state_dir / "jobs").exists()
    assert invoke(state_dir, controller) == 2

    recorded = calls(call_log)
    assert [item["args"][0] for item in recorded] == ["wait"]
    assert recorded[0]["pointer"] == first_job
    assert recorded[0]["args"][recorded[0]["args"].index("--job-dir") + 1].endswith(first_job)
    assert (state_dir / "current-job").read_text().strip() == first_job
    assert not (state_dir / "jobs").exists()


@pytest.mark.parametrize(
    ("released", "wait_code"),
    [
        ({"terminal": True, "state": "terminal_complete", "acceptance": {"accepted": True}}, 0),
        ({"terminal": True, "state": "terminal_failed", "acceptance": {"accepted": False}}, 1),
    ],
)
def test_wait_that_releases_finishes_current_invocation_and_runs_next_time(
    tmp_path: Path, released: dict[str, object], wait_code: int
):
    controller, call_log = fake_controller(tmp_path, wait_code=wait_code, released=released)
    state_dir = tmp_path / "state"
    set_current(state_dir, "old-job")

    assert invoke(state_dir, controller) == wait_code
    assert [item["args"][0] for item in calls(call_log)] == ["wait"]
    assert (state_dir / "current-job").read_text().strip() == "old-job"

    assert invoke(state_dir, controller) == 0
    recorded = calls(call_log)
    assert [item["args"][0] for item in recorded] == ["wait", "run"]
    new_job = (state_dir / "current-job").read_text().strip()
    assert new_job != "old-job" and recorded[-1]["pointer"] == new_job


def test_lock_contention_does_not_dispatch(tmp_path: Path):
    controller, call_log = fake_controller(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    with (state_dir / ".lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert invoke(state_dir, controller) == 75

    assert not call_log.exists()
    assert not (state_dir / "current-job").exists()


@pytest.mark.parametrize(
    "released",
    [
        "{bad json",
        '{"terminal": true, "state": "terminal_complete", "acceptance": {}}',
    ],
)
def test_malformed_release_is_rejected(tmp_path: Path, released: str):
    controller, call_log = fake_controller(tmp_path)
    state_dir = tmp_path / "state"
    job_dir = set_current(state_dir, "old-job")
    (job_dir / "released.json").write_text(released)

    assert invoke(state_dir, controller) == 1
    assert not call_log.exists()
    assert (state_dir / "current-job").read_text().strip() == "old-job"


def test_invocation_logs_are_unique_and_keep_stdout_and_stderr(tmp_path: Path):
    controller, _ = fake_controller(tmp_path, run_code=2, wait_code=2)
    state_dir = tmp_path / "state"

    assert invoke(state_dir, controller) == 2
    assert invoke(state_dir, controller) == 2

    logs = sorted((state_dir / "logs").glob("*.log"))
    assert len(logs) == 2
    contents = [path.read_text() for path in logs]
    assert any("stdout call 1" in text and "stderr call 1" in text for text in contents)
    assert any("stdout call 2" in text and "stderr call 2" in text for text in contents)
