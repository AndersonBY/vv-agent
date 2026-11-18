"""The real memory settings defaults are available without importing the SDK."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def test_actual_normalization_is_host_importable_and_matches_manager():
    from vv_agent import _memory_settings
    from vv_agent.memory import MemoryManager
    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.session import reducer

    assert reducer._normalize_memory_settings is _memory_settings.normalize_memory_settings
    cases: tuple[dict[str, Any], ...] = (
        {},
        {"compact_threshold": 64_000, "language": "en-US", "artifact_scope": ("run", "turn")},
        {"microcompaction_policy": MicrocompactionPolicy(keep_recent_cycles=4).to_dict()},
    )
    for settings in cases:
        host = _memory_settings.normalize_memory_settings(settings, 7)
        sdk = MemoryManager(**settings, protected_start=7)._definition_settings()
        assert host == sdk
        assert "protected_start" not in host


def test_normalization_rejects_unknown_settings():
    from vv_agent import _memory_settings

    with pytest.raises(TypeError):
        _memory_settings.normalize_memory_settings({"unknown_setting": True}, None)


def test_normalization_rejects_protected_start_in_stored_settings():
    from vv_agent import _memory_settings

    with pytest.raises(TypeError):
        _memory_settings.normalize_memory_settings({"protected_start": 2}, 3)


@pytest.mark.parametrize(
    ("policy", "error"),
    [
        ("not-a-policy", TypeError),
        ({"unknown": True}, ValueError),
    ],
)
def test_normalization_rejects_invalid_wire_policy(policy: object, error: type[Exception]):
    from vv_agent import _memory_settings

    with pytest.raises(error):
        _memory_settings.normalize_memory_settings({"microcompaction_policy": policy}, None)


def test_host_leaf_import_blocks_every_other_sdk_module(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import importlib.abc
import json
import sys

sys.path.insert(0, sys.argv[1])

class SDKBlocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vv_agent" or fullname.startswith("vv_agent."):
            allowed = fullname in {"vv_agent", "vv_agent.kernel", "vv_agent._memory_settings"}
            assert allowed or fullname.startswith("vv_agent.kernel."), fullname

sys.meta_path.insert(0, SDKBlocked())
from vv_agent import _memory_settings
from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy

cases = (
    {},
    {"compact_threshold": 64000, "language": "en-US", "artifact_scope": ("run", "turn")},
    {"microcompaction_policy": MicrocompactionPolicy(keep_recent_cycles=4).to_dict()},
)
print(json.dumps([_memory_settings.normalize_memory_settings(value, 7) for value in cases], sort_keys=True))
assert "vv_agent.memory" not in sys.modules
assert "vv_agent.session" not in sys.modules
assert "vv_agent._token_counting" not in sys.modules
assert "vv_llm" not in sys.modules
assert "tiktoken" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout)) == 3


def test_custom_post_init_observes_the_actual_policy_factory_once(monkeypatch: pytest.MonkeyPatch):
    from vv_agent.memory import manager
    from vv_agent.microcompaction import MicrocompactionPolicy

    events: list[str] = []
    original = MicrocompactionPolicy.__init__

    class CustomManager(manager.MemoryManager):
        def __post_init__(self) -> None:
            events.append("post_init")
            super().__post_init__()

    def tracked(self: MicrocompactionPolicy, *args: Any, **kwargs: Any) -> None:
        events.append("policy")
        original(self, *args, **kwargs)

    monkeypatch.setattr(MicrocompactionPolicy, "__init__", tracked)
    value = CustomManager()
    assert events == ["policy", "post_init"]
    assert value.microcompaction_policy == MicrocompactionPolicy()


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"compact_threshold": 64_000, "language": "en-US", "artifact_scope": ("run", "turn")},
    ],
)
def test_public_constructor_and_type_hints_match_expected_regressions(settings: dict[str, Any]):
    from dataclasses import fields
    from typing import get_type_hints

    from vv_agent.memory import MemoryManager
    from vv_agent.memory.session_memory import SessionMemory
    from vv_agent.workspace.base import WorkspaceBackend

    class_hints = get_type_hints(MemoryManager)
    init_hints = get_type_hints(MemoryManager.__init__)
    assert class_hints["workspace_backend"] == WorkspaceBackend | None
    assert class_hints["session_memory"] == SessionMemory | None
    assert init_hints["workspace_backend"] == WorkspaceBackend | None
    assert init_hints["session_memory"] == SessionMemory | None
    value = MemoryManager(**settings)
    assert type(value) is MemoryManager
    assert [field.name for field in fields(value)] == list(class_hints)
