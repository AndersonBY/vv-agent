from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_builtin_tool_schemas.py"
GENERATED_SOURCE = ROOT / "src" / "vv_agent" / "constants" / "workspace.py"


def test_builtin_tool_schema_source_is_reproducible(tmp_path: Path) -> None:
    generated = tmp_path / "workspace.py"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(generated)],
        cwd=tmp_path,
        check=True,
    )

    if generated.read_bytes() != GENERATED_SOURCE.read_bytes():
        raise AssertionError(
            "generated builtin tool schemas are stale; run `uv run python scripts/generate_builtin_tool_schemas.py`"
        )
