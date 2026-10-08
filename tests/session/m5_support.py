"""M5 repetition plugin; each trial retains the normal disposable DB fixture."""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

RACES = {
    "test_concurrent_acquire_and_twenty_identical_pushes",
    "test_twenty_concurrent_callback_replays_and_different_bytes_conflict",
    "test_concurrent_same_and_different_callback_bytes",
    "test_zombie_after_admission_is_fenced_but_external_call_can_still_happen",
    "test_late_first_model_result_respects_retry_dispatch_boundary",
    "test_twenty_concurrent_completion_duplicates_and_conflict",
    "test_twenty_concurrent_child_consumers_deliver_once",
}


def pytest_addoption(parser):
    parser.addoption("--m5-output")
    parser.addoption("--m5-repeat", type=int, default=20)
    parser.addoption("--m5-race-repeat", type=int, default=100)
    parser.addoption("--m5-seed", type=int, default=51008)
    parser.addoption("--m5-group", choices=["all", "deterministic", "race"], default="all")


def pytest_configure(config):
    config._m5_results = {}
    config._m5_started = time.time()


def pytest_generate_tests(metafunc):
    count = metafunc.config.getoption("--m5-race-repeat" if metafunc.function.__name__ in RACES else "--m5-repeat")
    metafunc.parametrize("m5_trial", range(count), indirect=True, ids=lambda i: f"m5-{i:03d}")


@pytest.fixture(autouse=True)
def m5_trial(request):
    case = request.node.nodeid.split("m5-")[0]
    offset = int.from_bytes(hashlib.sha256(case.encode()).digest()[:4], "big")
    seed = request.config.getoption("--m5-seed") + offset + request.param
    random.seed(seed)
    request.node._m5_seed = seed
    return seed


def pytest_collection_modifyitems(config, items):
    group = config.getoption("--m5-group")
    removed = [
        i
        for i in items
        if (group == "race" and i.originalname not in RACES) or (group == "deterministic" and i.originalname in RACES)
    ]
    items[:] = [i for i in items if i not in removed]
    config.hook.pytest_deselected(items=removed)


def pytest_runtest_logreport(report):
    # The config is supplied by the session-scoped hook below.
    assert _config is not None
    results = _config._m5_results
    trial = results.setdefault(report.nodeid, {"nodeid": report.nodeid, "phases": {}, "duration_s": 0.0})
    trial["phases"][report.when] = report.outcome
    trial["duration_s"] += report.duration
    if report.failed:
        trial.setdefault("failures", []).append(str(report.longrepr))
    if report.when == "teardown":
        _save(_config)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.user_properties.append(("m5_seed", getattr(item, "_m5_seed", None)))
    item.config._m5_results.setdefault(item.nodeid, {"nodeid": item.nodeid, "phases": {}, "duration_s": 0.0})["seed"] = getattr(
        item, "_m5_seed", None
    )


_config: Any = None


def pytest_sessionstart(session):
    global _config
    _config = session.config


def _save(config, exitstatus=None):
    trials = list(config._m5_results.values())
    cases = {}
    for trial in trials:
        # m5 id can occur before or after existing parametrization ids.
        import re

        case = re.sub(r"m5-\d+-?", "", trial["nodeid"]).replace("[]", "")
        aggregate = cases.setdefault(case, {"pass": 0, "fail": 0, "skip": 0, "incomplete": 0, "duration_s": 0.0})
        phases = trial["phases"]
        status = (
            "fail"
            if "failed" in phases.values()
            else "skip"
            if "skipped" in phases.values()
            else "pass"
            if len(phases) == 3
            else "incomplete"
        )
        aggregate[status] += 1
        aggregate["duration_s"] += trial["duration_s"]
    path = Path(config.getoption("--m5-output"))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "store": "real PostgreSQL",
        "started_unix_s": config._m5_started,
        "elapsed_s": time.time() - config._m5_started,
        "base_seed": config.getoption("--m5-seed"),
        "deterministic_repetitions": config.getoption("--m5-repeat"),
        "concurrency_repetitions": config.getoption("--m5-race-repeat"),
        "exitstatus": exitstatus,
        "cases": cases,
        "trials": trials,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def pytest_sessionfinish(session, exitstatus):
    _save(session.config, int(exitstatus))
