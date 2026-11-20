"""Fixture writers share real cases while keeping one owner per output file."""

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from support.contract import fixture_root

from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ARTIFACTS = runpy.run_path(str(SCRIPTS / "tool_result_artifact_fixtures.py"))
SESSION = runpy.run_path(str(SCRIPTS / "session_kernel_fixtures.py"))
REPLACEMENTS = runpy.run_path(str(SCRIPTS / "_session_kernel_replacements.py"))


@pytest.mark.parametrize("existing_bounded", [False, True])
def test_artifact_cli_writes_only_its_own_fixture(tmp_path, monkeypatch, existing_bounded):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    bounded = fixtures / "bounded_tool_result.json"
    if existing_bounded:
        bounded.write_bytes(b"session-generator-owned\n")
    monkeypatch.setattr(sys, "argv", ["tool_result_artifact_fixtures.py", "--contract-root", str(tmp_path)])
    ARTIFACTS["main"]()
    assert json.loads((fixtures / "tool_result_artifact.json").read_text()) == ARTIFACTS["generate"]()
    if existing_bounded:
        assert bounded.read_bytes() == b"session-generator-owned\n"
    else:
        assert not bounded.exists()


def test_session_bounded_regenerates_projection_and_capture_provenance(tmp_path):
    fixtures = SESSION["Fixtures"]()
    try:
        author = REPLACEMENTS["Author"](fixtures, SESSION["independent_bytes"], SESSION["facts"])
        bounded = author.outputs["bounded_tool_result.json"]
        bounded["unified_projection"] = {"stale": True}
        # The full producer's children() pass raises the store level before
        # bounded sessions are created; this isolated gate needs the same setup.
        fixtures._raw_store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        author.bounded()
        cases = ARTIFACTS["generate"]()["cases"]
        projection = bounded["unified_projection"]
        assert projection["limit_chars"] == 12000
        assert projection["ordinary_unchanged"] == cases["short"]
        assert projection["newly_truncated"] == cases["long_ascii"]
        assert projection["already_truncated"] == cases["already_truncated"]
        assert projection["marker"] == {
            "source_character_count_excludes_notice": True,
            "includes_artifact_path": True,
            "read_range": "read_file(start_line, end_line)",
            "search_regex": "search_files(pattern)",
            "head_chars": 6000,
            "tail_chars": "12000 - head_chars - marker_chars",
            "already_truncated": "preserve_producer_marker_and_recovery",
        }
        assert "stale" not in projection
        for name in ("truncated_bash", "truncated_error"):
            result = bounded["canonical_results"][name]
            entry = result["metadata"]["vv_artifact"]
            _, rows, _ = fixtures.kernel.store.read_state("c1c/bounded_" + name)
            plan_row = next(
                row
                for row in rows
                if row.record.kind == "op_planned" and row.record.operation_id == entry["source"]["operation_id"]
            )
            plan = plan_row.record
            plan_payload = hydrate_record_payload(
                plan_row, hydration_for_session_store(fixtures.kernel.store, "c1c/bounded_" + name)
            )
            assert plan_payload["request"]["arguments"] == {}
            assert entry["source"] == {
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "tool_call_id": plan_payload["request"]["id"],
                "arguments_sha256": sha256(b"{}").hexdigest(),
            }
            assert entry["source_complete"] is False
            assert entry["capture_incomplete_reason"] == "capture_completeness_unverified"
            assert {key: entry[key] for key in result["artifact"]} == result["artifact"]
        SESSION["write_json"](tmp_path, "bounded_tool_result.json", bounded)
        assert (tmp_path / "bounded_tool_result.json").read_bytes() == (fixture_root() / "bounded_tool_result.json").read_bytes()
    finally:
        fixtures.kernel.close()
