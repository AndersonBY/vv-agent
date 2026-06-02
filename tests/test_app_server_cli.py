from __future__ import annotations

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
