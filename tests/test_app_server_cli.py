from __future__ import annotations

import json
import subprocess
import sys


def test_app_server_help_lists_stdio_listener() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "app-server", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "--listen stdio" in result.stdout


def test_app_server_generate_schema_cli_writes_files(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "app-server", "generate-json-schema", "--out", str(tmp_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert (tmp_path / "json" / "vv_agent_app_server.schemas.json").is_file()


def test_debug_app_server_send_message_prints_jsonl_flow() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "debug", "app-server", "send-message", "hello"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(message.get("method") == "turn/completed" for message in messages)
