#!/usr/bin/env python3
"""Sync and verify a vendored vv-agent contract snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class SnapshotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read valid JSON from {path}: {exc}") from exc


def load_json_location(location: str) -> Any:
    if location.startswith(("https://", "http://")):
        try:
            with urllib.request.urlopen(location) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read valid JSON from {location}: {exc}") from exc
    return load_json(Path(location).expanduser().resolve())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def snapshot_files(fixtures: Path) -> list[Path]:
    return sorted(
        (path for path in fixtures.rglob("*") if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(fixtures).as_posix(),
    )


def validate_snapshot(fixtures: Path) -> dict[str, Any]:
    fixtures = fixtures.resolve()
    manifest = fixtures / "SHA256SUMS"
    if not manifest.is_file():
        raise SnapshotError(f"missing fixture manifest: {manifest}")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if lines != sorted(lines, key=lambda line: line.split("  ", 1)[-1]):
        raise SnapshotError("SHA256SUMS entries must be sorted by relative path")

    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise SnapshotError(f"invalid SHA256SUMS line {line_number}: {line!r}")
        digest, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in relative:
            raise SnapshotError(f"unsafe fixture path in SHA256SUMS: {relative}")
        if relative == "SHA256SUMS" or relative in entries:
            raise SnapshotError(f"duplicate or self-referential fixture path: {relative}")
        entries[relative] = digest

    actual = {path.relative_to(fixtures).as_posix() for path in snapshot_files(fixtures)}
    if actual != set(entries):
        raise SnapshotError(
            f"fixture coverage mismatch: missing={sorted(actual - set(entries))}, "
            f"stale={sorted(set(entries) - actual)}"
        )
    for relative, expected in entries.items():
        actual_digest = sha256_file(fixtures / relative)
        if actual_digest != expected:
            raise SnapshotError(f"fixture digest mismatch for {relative}: expected {expected}, got {actual_digest}")
    return {
        "fixture_files": len(entries) + 1,
        "manifest_entries": len(entries),
        "manifest_sha256": sha256_file(manifest),
    }


def git_revision(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or REVISION_RE.fullmatch(revision) is None:
        raise SnapshotError(f"cannot resolve a full Git revision from contract source {source}")
    return revision


def contract_source(source: Path, revision: str | None = None) -> dict[str, Any]:
    source = source.resolve()
    contract = load_json(source / "contract.json")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise SnapshotError("contract source contract.json must be a schema_version=1 object")
    version = contract.get("version")
    if contract.get("name") != "vv-agent-contract" or not isinstance(version, str):
        raise SnapshotError("contract source has unexpected name or version")
    if SEMVER_RE.fullmatch(version) is None:
        raise SnapshotError(f"invalid contract version: {version!r}")
    repository = contract.get("repository")
    if not isinstance(repository, str) or not repository.startswith("https://github.com/"):
        raise SnapshotError("contract source repository must be an HTTPS GitHub URL")
    fixtures_config = contract.get("fixtures")
    if not isinstance(fixtures_config, dict) or fixtures_config.get("path") != "fixtures":
        raise SnapshotError("contract source fixtures.path must be fixtures")
    fixtures = source / "fixtures"
    report = validate_snapshot(fixtures)
    if fixtures_config.get("manifest_sha256") != report["manifest_sha256"]:
        raise SnapshotError("contract source manifest digest does not match contract.json")
    resolved_revision = revision or git_revision(source)
    if REVISION_RE.fullmatch(resolved_revision) is None:
        raise SnapshotError("contract revision must be a full 40-character hexadecimal Git commit")
    return {
        "root": source,
        "fixtures": fixtures,
        "contract": contract,
        "version": version,
        "repository": repository,
        "revision": resolved_revision,
        **report,
    }


def compare_trees(expected: Path, actual: Path) -> None:
    expected_files = {
        path.relative_to(expected).as_posix(): path for path in expected.rglob("*") if path.is_file()
    }
    actual_files = {path.relative_to(actual).as_posix(): path for path in actual.rglob("*") if path.is_file()}
    if set(expected_files) != set(actual_files):
        raise SnapshotError(
            f"snapshot file-set drift: missing={sorted(set(expected_files) - set(actual_files))}, "
            f"extra={sorted(set(actual_files) - set(expected_files))}"
        )
    for relative, expected_path in expected_files.items():
        if expected_path.read_bytes() != actual_files[relative].read_bytes():
            raise SnapshotError(f"vendored fixture differs from canonical source: {relative}")


def replace_snapshot(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".contract-snapshot-", dir=target.parent))
    backup = target.with_name(f".{target.name}.backup")
    try:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            destination = staging / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        validate_snapshot(staging)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        staging.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if target.exists() and backup.exists():
            shutil.rmtree(target)
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@contextmanager
def local_artifact(location: str) -> Iterator[Path]:
    if location.startswith(("https://", "http://")):
        with tempfile.TemporaryDirectory(prefix="vv-agent-contract-download-") as temporary:
            destination = Path(temporary) / "contract.zip"
            try:
                urllib.request.urlretrieve(location, destination)
            except OSError as exc:
                raise SnapshotError(f"cannot download contract artifact {location}: {exc}") from exc
            yield destination
    else:
        path = Path(location).expanduser().resolve()
        if not path.is_file():
            raise SnapshotError(f"contract artifact does not exist: {path}")
        yield path


@contextmanager
def extracted_artifact(artifact: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="vv-agent-contract-artifact-") as temporary:
        root = Path(temporary)
        try:
            with zipfile.ZipFile(artifact) as archive:
                for member in archive.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise SnapshotError(f"unsafe path in contract artifact: {member.filename}")
                archive.extractall(root)
        except (OSError, zipfile.BadZipFile) as exc:
            raise SnapshotError(f"cannot extract contract artifact {artifact}: {exc}") from exc
        yield root


def load_lock(repo_root: Path, lock_name: str) -> tuple[Path, dict[str, Any]]:
    lock_path = repo_root / lock_name
    lock = load_json(lock_path)
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise SnapshotError(f"{lock_name} must be a schema_version=1 object")
    required_strings = [
        "contract_version",
        "contract_revision",
        "source_repository",
        "artifact_url",
        "artifact_sha256",
        "snapshot_path",
        "fixture_manifest_sha256",
    ]
    for field in required_strings:
        if not isinstance(lock.get(field), str) or not lock[field]:
            raise SnapshotError(f"{lock_name} field {field} must be a non-empty string")
    if SEMVER_RE.fullmatch(lock["contract_version"]) is None:
        raise SnapshotError(f"invalid locked contract version: {lock['contract_version']!r}")
    if REVISION_RE.fullmatch(lock["contract_revision"]) is None:
        raise SnapshotError("locked contract revision must be a full hexadecimal Git commit")
    for digest_field in ("artifact_sha256", "fixture_manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", lock[digest_field]) is None:
            raise SnapshotError(f"{digest_field} must be a lowercase SHA-256 digest")
    return lock_path, lock


def check_lock(
    repo_root: Path,
    lock_name: str,
    source: Path | None = None,
    artifact: str | None = None,
) -> dict[str, Any]:
    _, lock = load_lock(repo_root, lock_name)
    snapshot = (repo_root / lock["snapshot_path"]).resolve()
    try:
        snapshot.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SnapshotError("snapshot_path must stay inside the implementation repository") from exc
    report = validate_snapshot(snapshot)
    if report["manifest_sha256"] != lock["fixture_manifest_sha256"]:
        raise SnapshotError("vendored fixture manifest does not match contract.lock.json")

    if source is not None:
        canonical = contract_source(source)
        if canonical["version"] != lock["contract_version"]:
            raise SnapshotError("locked version does not match canonical contract source")
        if canonical["revision"] != lock["contract_revision"]:
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(canonical["root"]),
                    "merge-base",
                    "--is-ancestor",
                    lock["contract_revision"],
                    canonical["revision"],
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if ancestor.returncode != 0:
                raise SnapshotError("canonical checkout does not contain the locked contract revision")
        if canonical["repository"] != lock["source_repository"]:
            raise SnapshotError("locked repository does not match canonical contract source")
        compare_trees(canonical["fixtures"], snapshot)

    if artifact is not None:
        with local_artifact(artifact) as artifact_path:
            digest = sha256_file(artifact_path)
            if digest != lock["artifact_sha256"]:
                raise SnapshotError(f"contract artifact digest mismatch: expected {lock['artifact_sha256']}, got {digest}")
            with extracted_artifact(artifact_path) as artifact_root:
                artifact_contract = load_json(artifact_root / "contract.json")
                if artifact_contract.get("version") != lock["contract_version"]:
                    raise SnapshotError("artifact version does not match contract.lock.json")
                compare_trees(artifact_root / "fixtures", snapshot)

    return {
        "contract_version": lock["contract_version"],
        "contract_revision": lock["contract_revision"],
        "snapshot_path": lock["snapshot_path"],
        **report,
    }


def verify_adoption(
    repo_root: Path,
    lock_name: str,
    implementation: str,
    matrix_location: str,
    revision: str | None = None,
) -> dict[str, Any]:
    _, lock = load_lock(repo_root, lock_name)
    matrix = load_json_location(matrix_location)
    if not isinstance(matrix, dict) or matrix.get("schema_version") != 1:
        raise SnapshotError("support matrix must be a schema_version=1 object")
    if matrix.get("contract_version") != lock["contract_version"]:
        raise SnapshotError("support matrix version does not match contract.lock.json")
    if matrix.get("status") != "verified":
        raise SnapshotError(f"contract {lock['contract_version']} is not centrally verified")
    implementations = matrix.get("implementations")
    if not isinstance(implementations, dict) or not isinstance(implementations.get(implementation), dict):
        raise SnapshotError(f"support matrix has no {implementation} implementation record")
    record = implementations[implementation]
    verified_revision = record.get("verified_revision")
    if record.get("status") != "verified" or not isinstance(verified_revision, str):
        raise SnapshotError(f"{implementation} implementation is not centrally verified")
    if REVISION_RE.fullmatch(verified_revision) is None:
        raise SnapshotError(f"{implementation} verified revision is not a full Git commit")

    if revision is not None:
        if REVISION_RE.fullmatch(revision) is None:
            raise SnapshotError("release revision must be a full Git commit")
        result = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", verified_revision, revision],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SnapshotError(
                f"release revision {revision} does not contain centrally verified {implementation} revision "
                f"{verified_revision}"
            )

    return {
        "contract_version": lock["contract_version"],
        "implementation": implementation,
        "verified_revision": verified_revision,
        "release_revision": revision,
        "cross_repository_run": matrix.get("cross_repository_run"),
    }


def sync_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    canonical = contract_source(args.source, revision=args.revision)
    with local_artifact(args.artifact) as artifact_path:
        artifact_digest = sha256_file(artifact_path)
        with extracted_artifact(artifact_path) as artifact_root:
            artifact_contract = load_json(artifact_root / "contract.json")
            if artifact_contract.get("version") != canonical["version"]:
                raise SnapshotError("artifact version differs from the selected contract source")
            compare_trees(canonical["fixtures"], artifact_root / "fixtures")

    existing_lock = None
    lock_path = repo_root / args.lock
    if lock_path.exists():
        _, existing_lock = load_lock(repo_root, args.lock)
    snapshot_path = args.snapshot_path or (existing_lock and existing_lock["snapshot_path"])
    if not snapshot_path:
        raise SnapshotError("--snapshot-path is required when no contract.lock.json exists")
    snapshot = (repo_root / snapshot_path).resolve()
    try:
        snapshot.relative_to(repo_root)
    except ValueError as exc:
        raise SnapshotError("snapshot_path must stay inside the implementation repository") from exc

    replace_snapshot(canonical["fixtures"], snapshot)
    lock = {
        "schema_version": 1,
        "contract_version": canonical["version"],
        "contract_revision": canonical["revision"],
        "source_repository": canonical["repository"],
        "artifact_url": args.artifact_url,
        "artifact_sha256": artifact_digest,
        "snapshot_path": Path(snapshot_path).as_posix(),
        "fixture_manifest_sha256": canonical["manifest_sha256"],
    }
    write_json_atomic(lock_path, lock)
    report = check_lock(repo_root, args.lock, artifact=args.artifact)
    compare_trees(canonical["fixtures"], snapshot)
    return report


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    command_parser.add_argument("--lock", default="contract.lock.json")
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="verify the lock and vendored fixture snapshot")
    check.add_argument("--source", type=Path, help="optional canonical checkout for byte comparison")
    check.add_argument("--artifact", help="optional local path or URL for release artifact verification")

    adoption = subparsers.add_parser("adoption", help="require central verified adoption before release")
    adoption.add_argument("--implementation", choices=["python", "rust"], required=True)
    adoption.add_argument(
        "--matrix",
        default="https://raw.githubusercontent.com/AndersonBY/vv-agent-contract/main/support-matrix.json",
        help="support-matrix.json path or URL",
    )
    adoption.add_argument("--revision", help="release commit that must contain the verified implementation revision")

    sync = subparsers.add_parser("sync", help="replace the vendored snapshot from a canonical checkout")
    sync.add_argument("--source", type=Path, required=True)
    sync.add_argument("--artifact", required=True, help="local path or URL of the deterministic release zip")
    sync.add_argument("--artifact-url", required=True, help="immutable release URL recorded in the lock")
    sync.add_argument("--snapshot-path", help="repository-relative vendored fixture directory")
    sync.add_argument("--revision", help="override source Git revision; must be a full commit hash")
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "check":
            report = check_lock(args.repo_root.resolve(), args.lock, source=args.source, artifact=args.artifact)
        elif args.command == "adoption":
            report = verify_adoption(
                args.repo_root.resolve(),
                args.lock,
                args.implementation,
                args.matrix,
                revision=args.revision,
            )
        else:
            report = sync_snapshot(args)
    except SnapshotError as exc:
        print(f"contract snapshot error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
