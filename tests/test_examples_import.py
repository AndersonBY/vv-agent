"""Examples resolve current imports without executing their network entrypoints."""

import runpy
import socket
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").rglob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_compiles_and_imports(path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("example import attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    compile(path.read_text(), str(path), "exec")
    namespace = runpy.run_path(str(path), run_name=f"example_{path.stem}")
    for name, value in namespace.items():
        if name.endswith("_CODE") and isinstance(value, str):
            compile(value, f"{path}:{name}", "exec")
