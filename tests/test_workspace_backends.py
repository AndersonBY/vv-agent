from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from vv_agent.workspace import (
    INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
    DiscoveryFilteredWorkspaceBackend,
    InvalidPortableRegexError,
    LocalWorkspaceBackend,
    MemoryWorkspaceBackend,
)
from vv_agent.workspace.s3 import S3WorkspaceBackend


class _FakeClientError(Exception):
    pass


class _FakeNoSuchKey(_FakeClientError):
    pass


class _FakeS3Exceptions:
    NoSuchKey = _FakeNoSuchKey
    ClientError = _FakeClientError


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def paginate(self, **kwargs: Any) -> list[dict[str, list[dict[str, str]]]]:
        prefix = str(kwargs["Prefix"])
        contents = [{"Key": key} for key in sorted(self._objects) if key.startswith(prefix)]
        return [{"Contents": contents}]


class _FakeS3Client:
    def __init__(self) -> None:
        self.exceptions = _FakeS3Exceptions()
        self.objects: dict[str, bytes] = {}

    def get_paginator(self, _name: str) -> _FakePaginator:
        return _FakePaginator(self.objects)

    def get_object(self, **kwargs: Any) -> dict[str, BytesIO]:
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise _FakeNoSuchKey(key)
        return {"Body": BytesIO(self.objects[key])}

    def put_object(self, **kwargs: Any) -> None:
        key = str(kwargs["Key"])
        body = bytes(kwargs["Body"])
        assert kwargs["ContentLength"] == len(body)
        self.objects[key] = body

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise _FakeClientError(key)
        return {
            "ContentLength": len(self.objects[key]),
            "LastModified": datetime.now(tz=UTC),
        }


def _make_s3_backend(prefix: str = "tenant/workspace") -> S3WorkspaceBackend:
    backend = object.__new__(S3WorkspaceBackend)
    backend._client = _FakeS3Client()
    backend._bucket = "test-bucket"
    backend._prefix = prefix
    return backend


def test_workspace_backends_report_utf8_bytes_written(tmp_path: Path) -> None:
    backends = [
        LocalWorkspaceBackend(tmp_path / "local"),
        MemoryWorkspaceBackend(),
        _make_s3_backend(),
    ]
    content = "你好, world"

    for backend in backends:
        assert backend.write_text("unicode.txt", content) == len(content.encode("utf-8"))
        assert backend.read_text("unicode.txt") == content


def test_local_workspace_backend_replaces_invalid_utf8_when_reading_text(tmp_path: Path) -> None:
    (tmp_path / "mixed.log").write_bytes(b"ok\xffdone")
    backend = LocalWorkspaceBackend(tmp_path)

    assert backend.read_text("mixed.log") == "ok\ufffddone"


def test_workspace_backends_normalize_inner_dot_segments(tmp_path: Path) -> None:
    backends = [
        LocalWorkspaceBackend(tmp_path / "local"),
        MemoryWorkspaceBackend(),
        _make_s3_backend(),
    ]

    for backend in backends:
        backend.write_text("nested/../canonical.txt", "value")
        assert backend.read_text("./canonical.txt") == "value"
        assert backend.list_files(".", "*.txt") == ["canonical.txt"]


def test_local_workspace_parent_escape_matches_rust_contract(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(tmp_path)

    with pytest.raises(ValueError, match="escapes workspace"):
        backend.write_text("../escape.txt", "blocked")
    with pytest.raises(ValueError, match="escapes workspace"):
        backend.read_text("../escape.txt")
    with pytest.raises(ValueError, match="escapes workspace"):
        backend.list_files("..", "*.txt")
    with pytest.raises(ValueError, match="escapes workspace"):
        backend.file_info("../escape.txt")
    with pytest.raises(ValueError, match="escapes workspace"):
        backend.mkdir("../escape")

    assert backend.exists("../escape.txt") is False
    assert backend.is_file("../escape.txt") is False


def test_virtual_workspace_parent_escape_is_clamped_to_root() -> None:
    backends = [MemoryWorkspaceBackend(), _make_s3_backend()]

    for backend in backends:
        backend.write_text("../../escape.txt", "contained")
        assert backend.read_text("/escape.txt") == "contained"
        assert backend.list_files("..", "*.txt") == ["escape.txt"]
        info = backend.file_info("folder/../escape.txt")
        assert info is not None
        assert info.path == "escape.txt"


def test_memory_workspace_dot_and_parent_paths_address_root() -> None:
    backend = MemoryWorkspaceBackend()

    backend.mkdir("../safe")
    assert backend.exists("./safe") is True
    root_info = backend.file_info("..")
    assert root_info is not None
    assert root_info.path == "."
    assert root_info.is_dir is True


def test_s3_workspace_append_reports_uploaded_utf8_byte_count() -> None:
    backend = _make_s3_backend()
    initial = "你好"
    appended = "世界"

    assert backend.write_text("note.txt", initial) == len(initial.encode("utf-8"))
    assert backend.write_text("note.txt", appended, append=True) == len((initial + appended).encode("utf-8"))
    assert backend.read_text("note.txt") == initial + appended


def test_discovery_filtered_workspace_hides_only_listed_paths() -> None:
    backend = MemoryWorkspaceBackend()
    backend.write_text("notes/readme.md", "notes")
    backend.write_text("src/main.py", "main")
    backend.write_text("generated/cache.bin", "cache")
    backend.write_text("logs/run.log", "log")
    filtered = DiscoveryFilteredWorkspaceBackend(backend, r"^(?:generated|logs)/")

    assert filtered.list_files(".", "**/*") == ["notes/readme.md", "src/main.py"]
    assert filtered.read_text("generated/cache.bin") == "cache"
    assert filtered.read_bytes("logs/run.log") == b"log"
    assert filtered.exists("generated/cache.bin") is True
    assert filtered.is_file("logs/run.log") is True
    assert filtered.file_info("generated/cache.bin") is not None
    assert filtered.write_text("generated/new.bin", "new") == 3
    filtered.mkdir("logs/archive")
    assert backend.exists("generated/new.bin") is True
    assert backend.exists("logs/archive") is True


@pytest.mark.parametrize("pattern", [r"(?=secret)", r"(a)\1", r"\p{Greek}"])
def test_discovery_filtered_workspace_rejects_non_portable_regex(pattern: str) -> None:
    with pytest.raises(InvalidPortableRegexError) as error:
        DiscoveryFilteredWorkspaceBackend(MemoryWorkspaceBackend(), pattern)

    assert str(error.value) == INVALID_EXCLUDE_FILES_PATTERN_MESSAGE


# ---------------------------------------------------------------------------
# LocalWorkspaceBackend
# ---------------------------------------------------------------------------


class TestLocalWorkspaceBackend:
    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(tmp_path)

    @pytest.fixture
    def tmp(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_write_and_read_text(self, backend: LocalWorkspaceBackend) -> None:
        backend.write_text("hello.txt", "world")
        assert backend.read_text("hello.txt") == "world"

    def test_write_append(self, backend: LocalWorkspaceBackend) -> None:
        backend.write_text("log.txt", "a")
        backend.write_text("log.txt", "b", append=True)
        assert backend.read_text("log.txt") == "ab"

    def test_read_bytes(self, backend: LocalWorkspaceBackend, tmp: Path) -> None:
        (tmp / "bin.dat").write_bytes(b"\x00\x01\x02")
        assert backend.read_bytes("bin.dat") == b"\x00\x01\x02"

    def test_list_files(self, backend: LocalWorkspaceBackend) -> None:
        backend.write_text("a.txt", "1")
        backend.write_text("sub/b.txt", "2")
        files = backend.list_files(".", "**/*")
        assert "a.txt" in files
        assert "sub/b.txt" in files

    def test_list_files_with_glob(self, backend: LocalWorkspaceBackend) -> None:
        backend.write_text("a.py", "x")
        backend.write_text("b.txt", "y")
        files = backend.list_files(".", "**/*.py")
        assert files == ["a.py"]

    def test_file_info(self, backend: LocalWorkspaceBackend) -> None:
        backend.write_text("info.txt", "data")
        info = backend.file_info("info.txt")
        assert info is not None
        assert info.is_file is True
        assert info.is_dir is False
        assert info.size == 4
        assert info.suffix == ".txt"

    def test_file_info_missing(self, backend: LocalWorkspaceBackend) -> None:
        assert backend.file_info("nope.txt") is None

    def test_exists_and_is_file(self, backend: LocalWorkspaceBackend) -> None:
        backend.write_text("e.txt", "")
        assert backend.exists("e.txt") is True
        assert backend.is_file("e.txt") is True
        assert backend.exists("nope") is False

    def test_mkdir(self, backend: LocalWorkspaceBackend, tmp: Path) -> None:
        backend.mkdir("deep/nested/dir")
        assert (tmp / "deep" / "nested" / "dir").is_dir()

    def test_path_escape_raises(self, backend: LocalWorkspaceBackend) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            backend.read_text("../../etc/passwd")

    def test_allow_outside_root_reads_parent_file(self, tmp: Path) -> None:
        backend = LocalWorkspaceBackend(tmp, allow_outside_root=True)
        outside = (tmp.parent / f"{tmp.name}_outside.txt").resolve()
        outside.write_text("external", encoding="utf-8")
        assert backend.read_text(str(outside)) == "external"

    def test_allow_outside_root_list_files_returns_absolute_paths(self, tmp: Path) -> None:
        backend = LocalWorkspaceBackend(tmp, allow_outside_root=True)
        outside_dir = (tmp.parent / f"{tmp.name}_outside_dir").resolve()
        outside_dir.mkdir(parents=True, exist_ok=True)
        target = outside_dir / "a.txt"
        target.write_text("a", encoding="utf-8")

        files = backend.list_files(str(outside_dir), "**/*")
        assert str(target.resolve()) in files

    def test_allow_outside_root_file_info_returns_absolute_path(self, tmp: Path) -> None:
        backend = LocalWorkspaceBackend(tmp, allow_outside_root=True)
        outside = (tmp.parent / f"{tmp.name}_outside_info.txt").resolve()
        outside.write_text("info", encoding="utf-8")

        info = backend.file_info(str(outside))
        assert info is not None
        assert info.path == str(outside)


# ---------------------------------------------------------------------------
# MemoryWorkspaceBackend
# ---------------------------------------------------------------------------


class TestMemoryWorkspaceBackend:
    @pytest.fixture
    def backend(self) -> MemoryWorkspaceBackend:
        return MemoryWorkspaceBackend()

    def test_write_and_read_text(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("hello.txt", "world")
        assert backend.read_text("hello.txt") == "world"

    def test_write_append(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("log.txt", "a")
        backend.write_text("log.txt", "b", append=True)
        assert backend.read_text("log.txt") == "ab"

    def test_read_bytes(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("data.txt", "abc")
        assert backend.read_bytes("data.txt") == b"abc"

    def test_read_missing_raises(self, backend: MemoryWorkspaceBackend) -> None:
        with pytest.raises(FileNotFoundError):
            backend.read_text("nope.txt")

    def test_list_files(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("a.txt", "1")
        backend.write_text("sub/b.txt", "2")
        files = backend.list_files(".", "**/*")
        assert "a.txt" in files
        assert "sub/b.txt" in files

    def test_list_files_with_glob(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("a.py", "x")
        backend.write_text("b.txt", "y")
        files = backend.list_files(".", "**/*.py")
        assert files == ["a.py"]

    def test_file_info(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("info.txt", "data")
        info = backend.file_info("info.txt")
        assert info is not None
        assert info.is_file is True
        assert info.is_dir is False
        assert info.size == 4
        assert info.suffix == ".txt"

    def test_file_info_dir(self, backend: MemoryWorkspaceBackend) -> None:
        backend.mkdir("mydir")
        info = backend.file_info("mydir")
        assert info is not None
        assert info.is_dir is True
        assert info.is_file is False

    def test_file_info_missing(self, backend: MemoryWorkspaceBackend) -> None:
        assert backend.file_info("nope.txt") is None

    def test_exists_and_is_file(self, backend: MemoryWorkspaceBackend) -> None:
        backend.write_text("e.txt", "")
        assert backend.exists("e.txt") is True
        assert backend.is_file("e.txt") is True
        assert backend.exists("nope") is False

    def test_mkdir(self, backend: MemoryWorkspaceBackend) -> None:
        backend.mkdir("deep/nested/dir")
        assert backend.exists("deep") is True
        assert backend.exists("deep/nested") is True
        assert backend.exists("deep/nested/dir") is True
