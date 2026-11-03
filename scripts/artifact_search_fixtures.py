"""Deterministic artifact search vectors from the real built-in handler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.search import search_files
from vv_agent.workspace import MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import persist_text_artifact, text_artifact_ref

TASK = "artifact-search/turn/initial"
CASES = {
    "hit": {"unit": "first\nneedle here\nlast\n", "repeat": 1, "pattern": "needle"},
    "no_hit": {"unit": "nothing\n", "repeat": 1, "pattern": "needle"},
    "denied": {"unit": "needle\n", "repeat": 1, "pattern": "needle", "task": "other"},
    "missing": {"unit": "needle\n", "repeat": 1, "pattern": "needle", "missing": True},
    "invalid_path": {"unit": "needle\n", "repeat": 1, "pattern": "needle", "invalid_path": True},
    "long_line": {"unit": "x", "repeat": 1_000_000, "pattern": "x$"},
    "cjk": {"unit": "开头\n找到汉字\n结尾\n", "repeat": 1, "pattern": "汉字"},
    "emoji": {"unit": "first\n🙂🚀\nlast\n", "repeat": 1, "pattern": "🙂🚀"},
    "crlf": {"unit": "first\r\nneedle\r\nlast\r\n", "repeat": 1, "pattern": "^needle$"},
    "line_numbers": {"unit": "head\n", "repeat": 3000, "suffix": "needle\n", "pattern": "needle"},
    "budget_lines": {"unit": "needle\n", "repeat": 700, "pattern": "needle"},
    "budget_chars": {"unit": "needle" + "x" * 1000 + "\n", "repeat": 100, "pattern": "needle"},
    "invalid_regex": {"unit": "needle\n", "repeat": 1, "pattern": "["},
}


def produce(case):
    backend = MemoryWorkspaceBackend()
    text = case["unit"] * case["repeat"] + case.get("suffix", "")
    task = case.get("task", TASK)
    ref = (
        text_artifact_ref(task, "result", text)
        if case.get("missing")
        else persist_text_artifact(backend, task, "result", text, reuse_existing=True)
    )
    path = ref.path if not case.get("invalid_path") else ".vv-agent/artifacts/../outside.txt"
    arguments = {
        "path": path,
        "pattern": case["pattern"],
        "output_mode": "content",
        "head_limit": 0,
        "glob": "ignored-no-match",
        "type": "ignored-invalid-type",
    }
    context = ToolContext(Path("/fixture-artifact-search"), {}, 0, backend, task_id=TASK)
    result = search_files(context, arguments)
    return result, backend, arguments


def generate():
    return {
        "cases": {
            name: {"input": case, "arguments": args, "result": result.to_dict()}
            for name, case in CASES.items()
            for result, _backend, args in [produce(case)]
        },
        "rules": {
            "route": "reserved_path_before_directory_options",
            "scope": "current_task_segment_only",
            "line_numbers": "original_artifact_read_file_start_line_end_line",
            "max_result_lines": 500,
            "max_result_chars": 30000,
            "oversized_line": "not_matched_search_explicitly_incomplete_read_file_range_and_cursor",
            "writes_during_search": 0,
            "completion": "ordinary_unified_projection_no_search_exception",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-root", type=Path, required=True)
    args = parser.parse_args()
    (args.contract_root / "fixtures/artifact_search.json").write_text(
        json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
