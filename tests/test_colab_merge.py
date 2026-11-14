from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_merge", ROOT / "scripts/colab_merge.py")
assert SPEC is not None and SPEC.loader is not None
merge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge
SPEC.loader.exec_module(merge)

CODE = "a" * 40
OTHER_CODE = "c" * 40
CONTRACT = "b" * 40
CODE_DIGEST = "d" * 64
CONTRACT_MANIFEST = "e" * 64
PENDING = {"I9": "future", "I11": "future"}
REQUIRED_COVERAGE = tuple(sorted(merge.REQUIRED_COVERED))
SCENARIOS = tuple(f"scenario-{index:02d}" for index in range(merge.FULL_SCENARIOS))


def scenario_counts(total: int, names=SCENARIOS) -> dict[str, int]:
    if total < len(names):
        raise ValueError("fixture total is too small for its scenario names")
    counts = {name: 1 for name in names}
    counts[names[0]] += total - len(names)
    return counts


def part(
    backend: str,
    start: int,
    count: int,
    *,
    source: str,
    scenarios: dict[str, int],
    coverage: tuple[str, ...],
    pending: dict[str, str],
) -> dict:
    value = {
        "backend": backend,
        "seed": start,
        "seeds": count,
        "code_revision": source,
        "contract_revision": CONTRACT,
        "dirty": False,
        "contract_dirty": False,
        "code_digest": CODE_DIGEST,
        "contract_manifest_sha256": CONTRACT_MANIFEST,
        "in_memory_mutations": [],
        "pending": dict(pending),
        "scenarios": dict(scenarios),
        "covered": list(coverage),
        "environment": {},
    }
    if backend == "sqlite":
        value["environment"]["sqlite_temporary_storage"] = {
            "verified": True,
            "observations": [{"directory": "/tmp/shard", "filesystem": "ext4"}],
        }
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rewrite_manifest(job_dir: Path) -> None:
    evidence = job_dir / "evidence"
    transfers = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file() or path.name == "collection-manifest.json":
            continue
        transfers.append(
            {
                "relative_path": path.relative_to(evidence).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": "downloaded",
            }
        )
    write_json(evidence / "collection-manifest.json", {"terminal": True, "transfers": transfers})


def make_shard(
    root: Path,
    name: str,
    *,
    seed: int,
    sqlite_total: int = 21,
    postgres_total: int = 21,
    sqlite_coverage: tuple[str, ...] = REQUIRED_COVERAGE,
    postgres_coverage: tuple[str, ...] = REQUIRED_COVERAGE,
    source: str = CODE,
    pending: dict[str, str] | None = None,
    scenario_names=SCENARIOS,
) -> Path:
    job_dir = root / name
    pending = PENDING if pending is None else pending
    write_json(
        job_dir / "state.json",
        {
            "vv_agent_revision": source,
            "contract_revision": CONTRACT,
            "seed": seed,
            "sqlite_seeds": sqlite_total,
            "postgres_seeds": postgres_total,
            "sqlite_chunk_size": sqlite_total,
            "postgres_chunk_size": postgres_total,
        },
    )
    result = {
        "status": "complete",
        "full_acceptance": False,
        "identity": {"vv_agent_revision": source, "contract_revision": CONTRACT},
        "campaigns": {},
        "runtime": {
            "temporary_directory_removed": True,
            "postgres": {
                "password_authentication_verified": True,
                "temporary_role_removed": True,
                "temporary_databases_removed": True,
            },
        },
    }
    reports = {}
    for backend, total, coverage in (
        ("sqlite", sqlite_total, sqlite_coverage),
        ("postgres", postgres_total, postgres_coverage),
    ):
        report_name = f"{backend}-{seed:010d}-{total:010d}.json"
        scenarios = scenario_counts(total, scenario_names)
        reports[backend] = report_name
        write_json(
            job_dir / "evidence/reports" / report_name,
            part(
                backend,
                seed,
                total,
                source=source,
                scenarios=scenarios,
                coverage=coverage,
                pending=pending,
            ),
        )
        result["campaigns"][backend] = {
            "seed": seed,
            "seeds": total,
            "scenarios": scenarios,
            "covered": list(coverage),
            "pending": dict(pending),
            "reports": [report_name],
        }
    write_json(job_dir / "evidence/reports/result.json", result)
    write_json(
        job_dir / "evidence/process-status.json",
        {"state": "exited", "exit_code": 0, "child_identity": {"pid": 4321}},
    )
    write_json(job_dir / "evidence/child.stdout.log", {})
    write_json(job_dir / "evidence/child.stderr.log", {})
    rewrite_manifest(job_dir)
    return job_dir


def mutate_result(job_dir: Path, update) -> None:
    path = job_dir / "evidence/reports/result.json"
    value = json.loads(path.read_text())
    update(value)
    write_json(path, value)
    rewrite_manifest(job_dir)


def test_merge_formal_totals_is_full_acceptance(tmp_path: Path):
    first = make_shard(
        tmp_path,
        "job-a",
        seed=0,
        sqlite_total=5_000,
        postgres_total=5_000,
    )
    second = make_shard(
        tmp_path,
        "job-b",
        seed=5_000,
        sqlite_total=995_000,
        postgres_total=5_000,
    )

    result = merge.merge_job_dirs([first, second])

    assert result["accepted"] is True
    assert result["full_acceptance"] is True
    assert result["identity"]["code_digest"] == CODE_DIGEST
    assert result["campaigns"]["sqlite"]["seed"] == 0
    assert result["campaigns"]["sqlite"]["seeds"] == 1_000_000
    assert result["campaigns"]["postgres"]["seeds"] == 10_000
    assert result["campaigns"]["sqlite"]["scenarios"]["scenario-20"] == 2
    assert result["campaigns"]["postgres"]["scenarios"]["scenario-20"] == 2
    assert result["campaigns"]["sqlite"]["covered"] == sorted(merge.REQUIRED_COVERED)
    assert result["campaigns"]["sqlite"]["pending"] == PENDING
    assert result["campaigns"]["sqlite"]["reports"] == [
        "sqlite-0000000000-0000005000.json",
        "sqlite-0000005000-0000995000.json",
    ]
    assert result["runtime"]["shards"] == 2
    assert result["runtime"]["job_names"] == ["job-a", "job-b"]
    assert result["runtime"]["backends"]["sqlite"]["total"] == 1_000_000
    assert result["runtime"]["backends"]["postgres"]["total"] == 10_000


def test_custom_target_is_accepted_without_claiming_formal_full(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)

    result = merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)

    assert result["accepted"] is True
    assert result["full_acceptance"] is False
    assert result["campaigns"]["sqlite"]["seeds"] == 42


@pytest.mark.parametrize(
    ("second_seed", "target", "message"),
    [
        (42, 63, "missing seed range"),
        (10, 42, "overlaps"),
    ],
)
def test_merge_rejects_gaps_and_overlaps(tmp_path: Path, second_seed: int, target: int, message: str):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=second_seed)

    with pytest.raises(ValueError, match=message):
        merge.merge_job_dirs([first, second], sqlite_total=target, postgres_total=target)


def test_merge_rejects_cross_shard_identity_drift(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21, source=OTHER_CODE)

    with pytest.raises(ValueError, match="identity drift in vv_agent_revision"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_merge_rejects_pending_drift(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21, pending={"I9": "later", "I11": "future"})

    with pytest.raises(ValueError, match="identity drift in pending"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_result_pending_must_match_shard_parts(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)
    mutate_result(second, lambda value: value["campaigns"]["sqlite"]["pending"].update(I9="later"))

    with pytest.raises(ValueError, match="result sqlite pending differs from part values"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_partial_shard_cannot_claim_full_acceptance(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)
    mutate_result(second, lambda value: value.update(full_acceptance=True))

    with pytest.raises(ValueError, match="partial totals must not claim full acceptance"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_manifest_hash_drift_does_not_write_output(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)
    manifest_path = first / "evidence/collection-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["transfers"][0]["sha256"] = "f" * 64
    write_json(manifest_path, manifest)
    output = tmp_path / "merged.json"

    with pytest.raises(ValueError, match="collection manifest"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42, output=output)
    assert not output.exists()


def test_merge_requires_exactly_21_scenario_names(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0, scenario_names=SCENARIOS[:-1])
    second = make_shard(tmp_path, "job-b", seed=21, scenario_names=SCENARIOS[:-1])

    with pytest.raises(ValueError, match="exactly 21 scenario names"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_merge_requires_every_backend_coverage(tmp_path: Path):
    first = make_shard(
        tmp_path,
        "job-a",
        seed=0,
        sqlite_coverage=("I1", "I2", "I3", "I4", "I5"),
        postgres_coverage=("I6", "I7", "I8"),
    )
    second = make_shard(
        tmp_path,
        "job-b",
        seed=21,
        sqlite_coverage=("I6", "I7", "I8"),
        postgres_coverage=("I1", "I2", "I3", "I4", "I5"),
    )

    with pytest.raises(ValueError, match="aggregate sqlite coverage is incomplete"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_merge_rejects_missing_postgres_cleanup(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)
    mutate_result(second, lambda value: value["runtime"].pop("postgres"))

    with pytest.raises(ValueError, match="PostgreSQL authentication or cleanup"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_merge_rejects_missing_backend(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)
    mutate_result(second, lambda value: value["campaigns"].pop("postgres"))

    with pytest.raises(ValueError, match="both result campaigns are required"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (41, "exceeds target"),
        (43, "total is below target"),
    ],
)
def test_merge_rejects_total_above_or_below_target(tmp_path: Path, target: int, message: str):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)

    with pytest.raises(ValueError, match=message):
        merge.merge_job_dirs([first, second], sqlite_total=target, postgres_total=42)


def test_merge_rejects_duplicate_job_directory(tmp_path: Path):
    job = make_shard(tmp_path, "job-a", seed=0)

    with pytest.raises(ValueError, match="duplicate job directory"):
        merge.merge_job_dirs([job, job], sqlite_total=42, postgres_total=42)


def test_merge_rejects_duplicate_report_filename(tmp_path: Path):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=0)

    with pytest.raises(ValueError, match="duplicate report filename"):
        merge.merge_job_dirs([first, second], sqlite_total=42, postgres_total=42)


def test_cli_writes_only_the_requested_merge_output(tmp_path: Path, capsys):
    first = make_shard(tmp_path, "job-a", seed=0)
    second = make_shard(tmp_path, "job-b", seed=21)
    output = tmp_path / "merged.json"

    status = merge.main(
        [
            "--job-dir",
            str(first),
            "--job-dir",
            str(second),
            "--sqlite-total",
            "42",
            "--postgres-total",
            "42",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text())["runtime"]["shards"] == 2
    assert not (first / "merged.json").exists()
    assert capsys.readouterr().err == ""
