"""Canonical rejection facts must be produced by the actual kernel."""

import json
import subprocess
import sys
from pathlib import Path

from support.contract import fixture_root


def test_real_rejection_producer_matches_contract(tmp_path):
    output = tmp_path / "model_rejection.json"
    subprocess.run(
        [sys.executable, "-m", "scripts.model_rejection_fixtures", "--out", str(output)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected = json.loads((fixture_root() / "model_rejection.json").read_text())
    assert len(expected["vectors"]) >= 81
    assert output.read_bytes() == (fixture_root() / "model_rejection.json").read_bytes()
    assert json.loads(output.read_text()) == expected
