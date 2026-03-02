from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent.workspace import (
    LocalWorkspaceBackend,
    MemoryWorkspaceBackend,
)

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
