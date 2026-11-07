"""Generate completion verdict evidence through local HTTP and the actual kernel."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.session.test_model_completion_evidence import CASES, run_completion_case
from vv_agent.session.sqlite import SQLiteStore


def generate():
    cases = []
    with patch("vv_agent.session.kernel.time.monotonic_ns", return_value=100_000_000):
        for case in CASES:
            for declared in (None, False, True):
                with SQLiteStore.standalone(":memory:") as store:
                    store.install_schema()
                    store.connection.create_function("session_now_ms", 0, lambda: 1_000_000)
                    cases.append(run_completion_case(store, store, case, declared))
    return {"producer": "scripts/model_completion_fixtures.py", "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + "\n")
