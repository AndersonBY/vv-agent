#!/usr/bin/env python3
"""Minimal stdio JSONL client for the App Server process."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "vv_agent", "app-server", "--listen", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    for payload in [
        {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "stdio-example"}}},
        {"id": 1, "method": "model/list", "params": {}},
    ]:
        process.stdin.write(_json_line(payload))
    process.stdin.close()

    for line in process.stdout:
        print(line.rstrip())

    return_code = process.wait(timeout=10)
    if return_code != 0:
        raise SystemExit(process.stderr.read())


if __name__ == "__main__":
    main()
