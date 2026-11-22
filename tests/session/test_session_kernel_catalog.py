from __future__ import annotations

import base64
import json
import runpy
import shutil
from pathlib import Path

import pytest

from scripts import _session_kernel_catalog as catalog
from vv_agent.kernel._content_runtime import SessionContentWriter
from vv_agent.session.records import SessionSpec

AUTHOR = runpy.run_path(str(Path(__file__).parents[2] / "scripts/session_kernel_fixtures.py"))


@pytest.mark.skipif(shutil.which("node") is None, reason="Independent RFC 8785 encoder requires Node.js")
def test_content_capture_uses_only_curated_fixture_refs_and_preserves_raw_bytes(tmp_path: Path) -> None:
    output = tmp_path / "curated"
    output.mkdir()
    fixtures = AUTHOR["Fixtures"]()
    try:
        tenant_id = fixtures.kernel.tenant_id
        session_id = "catalog-fixture"
        with fixtures.kernel.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)

            def spec(content: SessionContentWriter) -> SessionSpec:
                workspace = content.put_json("/workspace/😀")
                actor, _ = content.service_principal("fixture/catalog")
                return SessionSpec(
                    session_id,
                    actor,
                    workspace.to_dict(),
                    attributes=content.put_json({"ref": workspace.to_dict(), "😀": [1e-7, 1e21]}).to_dict(),
                )

            writer.create_session(spec, consumers=())
        created = fixtures.kernel.store.read(session_id, limit=1).records[0].record.payload
        outer = created["attributes"]
        nested = created["workspace"]
        (output / "sample.jsonl").write_text(json.dumps({"attributes": outer}) + "\n", encoding="utf-8")

        def capture(independent_bytes=AUTHOR["independent_bytes"]):
            return catalog.capture_content_catalog(
                fixtures._raw_store,
                fixtures.content_key_provider,
                output,
                {"contents": {}},
                independent_bytes=independent_bytes,
                fixture_values=AUTHOR["S1_FIXTURE_VALUES"],
            )

        actual = capture()
        assert actual == capture()
        assert set(actual["contents"]) == {outer["content_id"], nested["content_id"]}
        assert [key["key_id"] for key in actual["keys"]] == [outer["key_id"]]
        entry = actual["contents"][outer["content_id"]]
        assert (entry["tenant_id"], entry["session_id"], entry["key_id"]) == (tenant_id, session_id, outer["key_id"])
        raw = base64.b64decode(entry["bytes_base64"], validate=True)
        assert b"1e-7,1e+21" in raw and "😀".encode() in raw
        assert entry["value"]["ref"] == nested

        with pytest.raises(ValueError, match="not canonical"):
            capture(lambda values: [b"{}"] * len(values))
        # A different deterministic run has its own issued token inventory.
        with (
            AUTHOR["S1_FIXTURE_VALUES"].deterministic_content_secrets(),
            pytest.raises(ValueError, match="unexpected fixture key binding"),
        ):
            capture()
    finally:
        fixtures.kernel.close()
