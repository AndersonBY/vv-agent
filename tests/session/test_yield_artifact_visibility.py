from pathlib import PurePosixPath

import pytest

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.context import project_context
from vv_agent.session.kernel import _Driver, read_state
from vv_agent.session.tool_results import prepare_tool_resources, project_tool_result

from .test_yield_review_regressions import test_pure_long_renderer_retry_should_retain_the_committed_body as run_long_retry


@pytest.mark.sqlite_file
def test_unselected_candidate_not_projected_and_same_body_reuses(store, database, tmp_path, monkeypatch):
    original = _Driver.completed
    candidates = []

    def capture(driver, plan, outcome):
        receipts = original(driver, plan, outcome)
        if driver.sid == "s" and len(outcome.result.get("content", "")) > 12000:
            candidates.append((driver.runtime.workspace_backend, plan, outcome.result, receipts[0].payload["result"]))
        return receipts

    monkeypatch.setattr(_Driver, "completed", capture)
    run_long_retry(store, database, tmp_path, monkeypatch)
    assert len(candidates) == 2
    backend, plan, raw, selected = candidates[1]
    orphan = candidates[0][3]["artifact"]
    state, retained, _ = read_state(store, "s")
    completion = next(r for r in retained if r.record.kind == "op_completed" and "yielded_by" in r.record.payload)
    descriptor = project_tool_result(completion)
    model_context = [m.to_dict() for m in project_context(retained, state)]
    projections = canonical_json_bytes(
        {"records": [r.record.to_dict() for r in retained], "context": model_context, "descriptor": descriptor}
    )
    assert orphan["path"].encode() not in projections
    assert PurePosixPath(orphan["path"]).name.encode() not in projections
    assert orphan["sha256"].encode() not in projections
    assert len(descriptor["resources"]) == 1
    assert descriptor["resources"][0]["sha256"] == selected["artifact"]["sha256"]
    files = {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in backend._artifact_root.rglob("*.txt")}
    repeated = prepare_tool_resources(plan, raw, backend)
    assert repeated == selected
    assert all((p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) == value for p, value in files.items())
    assert backend.read_text(orphan["path"]) == candidates[0][2]["content"]
    assert len(files) == 2
    assert backend.list_files(".vv-agent/artifacts", "**/*") == []
    assert not any(".vv-agent/artifacts/" in path for path in backend.list_files(".", "**/*"))
    print(
        "CANDIDATE_VISIBILITY",
        {
            "files": len(files),
            "resources": len(descriptor["resources"]),
            "orphan_in_records_or_context_or_resources": False,
            "same_body_same_path": True,
            "same_body_no_rewrite": True,
        },
    )
