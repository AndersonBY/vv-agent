from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

FULL_SQLITE = 1_000_000
FULL_POSTGRES = 10_000
FULL_SCENARIOS = 21
BACKENDS = ("sqlite", "postgres")
REQUIRED_COVERED = frozenset({"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I10"})
EXPECTED_PENDING = frozenset({"I9", "I11"})
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PART_RE = re.compile(r"^(sqlite|postgres)-(\d{10})-(\d{10})\.json$")


@dataclass(frozen=True)
class _Part:
    backend: str
    start: int
    count: int
    name: str
    data: dict[str, Any]

    @property
    def end(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class _Shard:
    path: Path
    name: str
    state: dict[str, Any]
    result: dict[str, Any]
    reports: dict[str, tuple[_Part, ...]]

    def total(self, backend: str) -> int:
        return sum(part.count for part in self.reports[backend])

    def part_identity(self, field: str) -> Any:
        values = [part.data[field] for backend in BACKENDS for part in self.reports[backend]]
        frozen = {_freeze(value) for value in values}
        _require(len(frozen) == 1, f"shard {self.name}: part {field} differs")
        return values[0]


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@cache
def _campaign():
    path = Path(__file__).with_name("colab_campaign.py")
    spec = importlib.util.spec_from_file_location("_colab_campaign_for_merge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load controller from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_state(state: dict[str, Any], shard_name: str) -> None:
    for field in ("vv_agent_revision", "contract_revision"):
        value = state.get(field)
        _require(
            isinstance(value, str) and REVISION_RE.fullmatch(value) is not None,
            f"shard {shard_name}: state {field} is malformed",
        )
    seed = state.get("seed")
    _require(
        type(seed) is int and seed >= 0,
        f"shard {shard_name}: state seed must be a non-negative integer",
    )
    for backend in BACKENDS:
        total = state.get(f"{backend}_seeds")
        _require(
            type(total) is int and total > 0,
            f"shard {shard_name}: state {backend}_seeds must be positive",
        )
        chunk = state.get(f"{backend}_chunk_size")
        _require(
            type(chunk) is int and chunk > 0,
            f"shard {shard_name}: state {backend}_chunk_size must be positive",
        )


def _validate_released(job_dir: Path, shard_name: str) -> None:
    path = job_dir / "released.json"
    if not path.exists():
        return
    released = _json(path)
    acceptance = released.get("acceptance")
    _require(released.get("terminal") is True, f"shard {shard_name}: released.json is not terminal")
    _require(
        isinstance(acceptance, dict) and acceptance.get("accepted") is True,
        f"shard {shard_name}: released.json is not accepted",
    )
    _require(
        acceptance.get("full_acceptance") is False,
        f"shard {shard_name}: released.json claims full acceptance for a shard",
    )


def _load_parts(evidence: Path, shard_name: str) -> dict[str, tuple[_Part, ...]]:
    reports: dict[str, list[_Part]] = {backend: [] for backend in BACKENDS}
    report_dir = evidence / "reports"
    for path in sorted(report_dir.iterdir(), key=lambda item: item.name):
        if path.name == "result.json" or not path.is_file():
            continue
        match = PART_RE.fullmatch(path.name)
        _require(match is not None, f"shard {shard_name}: unexpected report filename {path.name}")
        backend, start_text, count_text = match.groups()
        data = _json(path)
        reports[backend].append(
            _Part(
                backend=backend,
                start=int(start_text),
                count=int(count_text),
                name=path.name,
                data=data,
            )
        )
    return {backend: tuple(sorted(reports[backend], key=lambda part: (part.start, part.name))) for backend in BACKENDS}


def _load_shard(job_dir: Path) -> _Shard:
    job_dir = job_dir.resolve()
    _require(job_dir.is_dir(), f"job directory does not exist: {job_dir}")
    shard_name = job_dir.name
    state = _json(job_dir / "state.json")
    _validate_state(state, shard_name)
    evidence = job_dir / "evidence"
    _require(evidence.is_dir(), f"shard {shard_name}: evidence directory is missing")
    try:
        acceptance = _campaign().validate_evidence(evidence, state, {"exit_code": 0})
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"shard {shard_name}: {error}") from error
    _require(acceptance.get("accepted") is True, f"shard {shard_name}: controller rejected evidence")
    _require(
        acceptance.get("full_acceptance") is False,
        f"shard {shard_name}: partial shard must not claim full acceptance",
    )
    _validate_released(job_dir, shard_name)
    result = _json(evidence / "reports/result.json")
    reports = _load_parts(evidence, shard_name)
    for backend in BACKENDS:
        _require(reports[backend], f"shard {shard_name}: missing {backend} reports")
        campaign = result["campaigns"][backend]
        _require(
            isinstance(campaign.get("covered"), list) and all(isinstance(value, str) for value in campaign["covered"]),
            f"shard {shard_name}: result {backend} coverage is malformed",
        )
        _require(
            isinstance(campaign.get("pending"), dict) and frozenset(campaign["pending"]) == EXPECTED_PENDING,
            f"shard {shard_name}: result {backend} pending mismatch",
        )
        for part in reports[backend]:
            _require(
                isinstance(part.data.get("covered"), list) and all(isinstance(value, str) for value in part.data["covered"]),
                f"shard {shard_name}: coverage is malformed in {part.name}",
            )
            _require(
                isinstance(part.data.get("pending"), dict) and frozenset(part.data["pending"]) == EXPECTED_PENDING,
                f"shard {shard_name}: pending mismatch in {part.name}",
            )
        _require(
            campaign["pending"] == _shard_part_pending(reports[backend], shard_name, backend),
            f"shard {shard_name}: result {backend} pending differs from part values",
        )
    shard = _Shard(path=job_dir, name=shard_name, state=state, result=result, reports=reports)
    for backend in BACKENDS:
        configured = state[f"{backend}_seeds"]
        _require(
            shard.total(backend) == configured,
            f"shard {shard_name}: {backend} report total differs from configured total",
        )
        for part in reports[backend]:
            _require(
                part.start >= state["seed"] and part.end <= state["seed"] + configured,
                f"shard {shard_name}: {backend} report escapes the configured seed interval",
            )
        shard.part_identity("code_digest")
        shard.part_identity("contract_manifest_sha256")
        shard.part_identity("in_memory_mutations")
        shard.part_identity("dirty")
        shard.part_identity("contract_dirty")
        shard.part_identity("pending")
    return shard


def _shard_part_pending(parts: Sequence[_Part], shard_name: str, backend: str) -> dict[str, Any]:
    values = [part.data["pending"] for part in parts]
    frozen = {_freeze(value) for value in values}
    _require(len(frozen) == 1, f"shard {shard_name}: pending differs within {backend} parts")
    return values[0]


def _identity(shards: Sequence[_Shard]) -> dict[str, Any]:
    fields = (
        ("vv_agent_revision", lambda shard: shard.result["identity"]["vv_agent_revision"]),
        ("contract_revision", lambda shard: shard.result["identity"]["contract_revision"]),
        ("code_digest", lambda shard: shard.part_identity("code_digest")),
        ("contract_manifest_sha256", lambda shard: shard.part_identity("contract_manifest_sha256")),
        ("in_memory_mutations", lambda shard: shard.part_identity("in_memory_mutations")),
        ("dirty", lambda shard: shard.part_identity("dirty")),
        ("contract_dirty", lambda shard: shard.part_identity("contract_dirty")),
        ("pending", lambda shard: shard.part_identity("pending")),
    )
    identity: dict[str, Any] = {}
    for field, read in fields:
        values = [_freeze(read(shard)) for shard in shards]
        _require(len(set(values)) == 1, f"cross-shard identity drift in {field}")
        identity[field] = read(shards[0])
    return identity


def _aggregate_ranges(shards: Sequence[_Shard], backend: str, target: int) -> list[dict[str, Any]]:
    cursor = 0
    intervals = []
    for shard in sorted(shards, key=lambda item: (item.state["seed"], item.name)):
        start = shard.state["seed"]
        total = shard.total(backend)
        end = start + total
        if start < cursor:
            raise ValueError(f"aggregate {backend} shard {shard.name} overlaps at seed {start}; expected {cursor} or later")
        if start != cursor:
            raise ValueError(f"aggregate {backend} is missing seed range [{cursor}, {start})")
        if end > target:
            raise ValueError(f"aggregate {backend} shard {shard.name} exceeds target: [{start}, {end}) > {target}")
        intervals.append({"job_name": shard.name, "start": start, "end": end, "count": total})
        cursor = end
    if cursor < target:
        raise ValueError(f"aggregate {backend} total is below target: expected {target}, got {cursor}")
    _require(cursor == target, f"aggregate {backend} total mismatch: expected {target}, got {cursor}")
    return intervals


def _aggregate_campaign(shards: Sequence[_Shard], backend: str, target: int) -> dict[str, Any]:
    scenarios: dict[str, int] = {}
    covered: set[str] = set()
    reports = []
    for shard in sorted(shards, key=lambda item: (item.state["seed"], item.name)):
        for part in shard.reports[backend]:
            for name, count in part.data["scenarios"].items():
                scenarios[name] = scenarios.get(name, 0) + count
            covered.update(part.data["covered"])
            reports.append(part.name)
    _require(
        len(scenarios) == FULL_SCENARIOS,
        f"aggregate {backend} must have exactly {FULL_SCENARIOS} scenario names; got {len(scenarios)}",
    )
    _require(
        all(type(count) is int and count > 0 for count in scenarios.values()),
        f"aggregate {backend} scenario counts must all be positive",
    )
    _require(
        REQUIRED_COVERED.issubset(covered),
        f"aggregate {backend} coverage is incomplete; missing {sorted(REQUIRED_COVERED - covered)}",
    )
    pending = _identity(shards)["pending"]
    return {
        "seed": 0,
        "seeds": target,
        "scenarios": dict(sorted(scenarios.items())),
        "covered": sorted(covered),
        "pending": dict(sorted(pending.items())),
        "reports": reports,
    }


def merge_job_dirs(
    job_dirs: Sequence[Path],
    *,
    sqlite_total: int = FULL_SQLITE,
    postgres_total: int = FULL_POSTGRES,
    output: Path | None = None,
) -> dict[str, Any]:
    _require(len(job_dirs) >= 2, "at least two --job-dir values are required")
    _require(type(sqlite_total) is int and sqlite_total > 0, "sqlite_total must be positive")
    _require(type(postgres_total) is int and postgres_total > 0, "postgres_total must be positive")
    resolved = [Path(job_dir).resolve() for job_dir in job_dirs]
    _require(len(set(resolved)) == len(resolved), "duplicate job directory")
    names = [path.name for path in resolved]
    _require(len(set(names)) == len(names), "duplicate job name")
    if output is not None:
        output_path = Path(output).resolve()
        for job_dir in resolved:
            try:
                output_path.relative_to(job_dir)
            except ValueError:
                continue
            raise ValueError("output path must be outside every shard job directory")

    shards = [_load_shard(job_dir) for job_dir in resolved]
    seen_reports = set()
    duplicate_reports = []
    for shard in shards:
        for backend in BACKENDS:
            for part in shard.reports[backend]:
                if part.name in seen_reports:
                    duplicate_reports.append(part.name)
                seen_reports.add(part.name)
    duplicate_reports = sorted(set(duplicate_reports))
    if duplicate_reports:
        raise ValueError(f"duplicate report filename: {duplicate_reports[0]}")
    identity = _identity(shards)
    intervals = {
        "sqlite": _aggregate_ranges(shards, "sqlite", sqlite_total),
        "postgres": _aggregate_ranges(shards, "postgres", postgres_total),
    }
    campaigns = {
        backend: _aggregate_campaign(shards, backend, sqlite_total if backend == "sqlite" else postgres_total)
        for backend in BACKENDS
    }
    result = {
        "accepted": True,
        "full_acceptance": sqlite_total == FULL_SQLITE and postgres_total == FULL_POSTGRES,
        "identity": identity,
        "campaigns": campaigns,
        "runtime": {
            "shards": len(shards),
            "job_names": [shard.name for shard in shards],
            "jobs": [
                {
                    "job_name": shard.name,
                    "seed": shard.state["seed"],
                    "sqlite_seeds": shard.state["sqlite_seeds"],
                    "postgres_seeds": shard.state["postgres_seeds"],
                }
                for shard in shards
            ],
            "backends": {
                "sqlite": {"total": sqlite_total, "intervals": intervals["sqlite"]},
                "postgres": {"total": postgres_total, "intervals": intervals["postgres"]},
            },
        },
    }
    if output is not None:
        output_path = Path(output)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only merge of independently validated Colab seed shards")
    parser.add_argument("--job-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sqlite-total", type=_positive, default=FULL_SQLITE)
    parser.add_argument("--postgres-total", type=_positive, default=FULL_POSTGRES)
    args = parser.parse_args(argv)
    try:
        merge_job_dirs(
            args.job_dir,
            sqlite_total=args.sqlite_total,
            postgres_total=args.postgres_total,
            output=args.output,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
