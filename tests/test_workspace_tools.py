from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent.runtime.local_environment import LocalEnvironment
from vv_agent.tools import ToolContext
from vv_agent.tools.handlers import search as search_handler
from vv_agent.tools.handlers import workspace_io
from vv_agent.workspace import LocalWorkspaceBackend


class _SearchHandle:
    def __init__(self, stdout: Any) -> None:
        self.stdout = stdout
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _context(workspace: Path, environment: LocalEnvironment) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(workspace),
        metadata={"local_environment": environment},
    )


def test_find_files_and_search_files_use_injected_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class TrackingEnvironment(LocalEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.search_calls: list[tuple[list[str], bool, bool]] = []

        def start_search(self, command: list[str], *, cwd: Path, text: bool, capture_stderr: bool, **_: Any):
            assert cwd == tmp_path
            self.search_calls.append((command, text, capture_stderr))
            if text:
                events = [
                    {"type": "begin", "data": {"path": {"text": "hit.txt"}}},
                    {
                        "type": "match",
                        "data": {
                            "path": {"text": "hit.txt"},
                            "line_number": 1,
                            "lines": {"text": "needle\n"},
                            "submatches": [{"match": {"text": "needle"}, "start": 0, "end": 6}],
                        },
                    },
                ]
                return _SearchHandle(io.StringIO("".join(f"{json.dumps(event)}\n" for event in events)))
            return _SearchHandle(io.BytesIO(b"hit.txt\x00"))

    environment = TrackingEnvironment()
    context = _context(tmp_path, environment)
    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")

    found = workspace_io.find_files(context, {"path": "."})
    searched = search_handler.search_files(context, {"pattern": "needle", "output_mode": "files_with_matches"})

    assert json.loads(found.content)["files"] == ["hit.txt"]
    assert searched.metadata["files"] == ["hit.txt"]
    assert len(environment.search_calls) == 2
    assert environment.search_calls[0][1:] == (False, True)
    assert environment.search_calls[1][1:] == (True, False)


def test_builtin_tool_handlers_contain_no_direct_subprocess_popen_calls() -> None:
    source_root = Path(__file__).parents[1] / "src/vv_agent/tools/handlers"
    for filename in ("bash.py", "workspace_io.py", "search.py"):
        tree = ast.parse((source_root / filename).read_text(encoding="utf-8"))
        direct_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        assert direct_calls == [], filename
