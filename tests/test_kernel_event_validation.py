"""Shared event validation keeps the public decoder's acceptance and diagnostics."""

import base64
import json
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vv_agent import events
from vv_agent.kernel import _event_validation as core

FIXTURES = Path(__file__).parent / "fixtures/parity"


def canonical_events() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for name in ("run_events.jsonl", "budget_events.jsonl")
        for line in (FIXTURES / name).read_text().splitlines()
    ]


def outcome(function, payload):
    try:
        function(payload)
    except Exception as exc:
        return type(exc), str(exc)
    return None


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_event_validation_import_and_canonical_vectors_without_outer_sdk(tmp_path, site_flags):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import base64
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root / "src"))

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _event_validation as core
        from vv_agent.kernel._message_values import Message

        assert Path(core.__file__) == root / "src/vv_agent/kernel/_event_validation.py"
        fixtures = root / "tests/fixtures/parity"
        for name in ("run_events.jsonl", "budget_events.jsonl"):
            for line in (fixtures / name).read_text().splitlines():
                payload = json.loads(line)
                assert core.validate_retained_event(payload) is None
                if payload["type"] == "memory_compact_started":
                    live = payload | {"cycle_index": 2**53 + 1, "estimated_tokens": None,
                                      "metadata": {"messages": [Message("user", "live")]}}
                    assert core.validate_retained_event(live) is None
                    assert core.validate_retained_event(live | {"session_id": None}) is None
        for case in json.loads((fixtures / "run_events_invalid.json").read_text())["reject"]:
            payload = json.loads(base64.b64decode(case["bytes_base64"]))
            try:
                core.validate_retained_event(payload)
            except ValueError:
                pass
            else:
                raise AssertionError(case["id"])
        assert "vv_agent.events" not in sys.modules
        assert not hasattr(core, "RunEvent")
        assert not hasattr(core, "new_event_id")
        assert not hasattr(core, "event_created_at")
        assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
                   for name in sys.modules if name.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_canonical_acceptance_and_negative_diagnostics_match_public_decoder():
    for payload in canonical_events():
        assert core.validate_retained_event(payload) is None
        event = events.event_from_dict(payload)
        assert (event.type, event.run_id, event.session_id) == (payload["type"], payload["run_id"], payload["session_id"])
    for case in json.loads((FIXTURES / "run_events_invalid.json").read_text())["reject"]:
        payload = json.loads(base64.b64decode(case["bytes_base64"]))
        expected = outcome(events.event_from_dict, payload)
        assert expected is not None, case["id"]
        assert outcome(core.validate_retained_event, payload) == expected, case["id"]


def test_bounded_single_field_mutations_keep_decoder_acceptance_and_diagnostics():
    by_kind = {payload["type"]: payload for payload in canonical_events()}
    for kind, payload in by_kind.items():
        for field in payload:
            variants = [{key: value for key, value in payload.items() if key != field}]
            variants += [payload | {field: value} for value in (None, False, True, -1, 1.5, "", [], {})]
            for variant in variants:
                assert outcome(core.validate_retained_event, variant) == outcome(events.event_from_dict, variant), (
                    kind,
                    field,
                    variant,
                )


@pytest.mark.parametrize("logical_cycle", [True, 1.0])
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({}, "cycle_aborted logical_cycle must equal cycle_index + 1"),
        ({"metadata": None}, "Run event metadata must be an object"),
        ({"metadata": None, "future": 1}, "Run event cycle_aborted contains unknown fields: future"),
    ],
)
def test_constructor_only_cycle_gap_runs_after_complete_wire_validation(logical_cycle, extra, message):
    payload = {
        "version": "v6",
        "type": "cycle_aborted",
        "run_id": "turn",
        "session_id": "session",
        "event_id": "evt_cycle",
        "trace_id": "trace",
        "created_at": 0,
        "cycle_index": 0,
        "logical_cycle": logical_cycle,
        "reason": "cancelled",
    } | extra
    if not extra:
        assert core._validate_event_wire(payload) is None
    assert outcome(events.event_from_dict, payload) == (ValueError, message)
    assert outcome(core.validate_retained_event, payload) == (ValueError, message)


def test_memory_public_classes_defaults_and_constructor_validation_remain_outer(monkeypatch):
    import vv_agent
    from vv_agent import memory
    from vv_agent.memory import provider
    from vv_agent.types import Message

    for name in ("MemoryCompactStarted", "MemoryCompactCompleted"):
        event_type = getattr(events, name)
        assert event_type.__module__ == "vv_agent.events"
        assert issubclass(event_type, events.RunEvent)
        assert getattr(vv_agent, name) is getattr(memory, name) is getattr(provider, name) is event_type
    assert events._validate_event_wire is core._validate_event_wire
    assert events._memory_compact_counter is core._memory_compact_counter
    assert events._validate_cycle_aborted_fields is core._validate_cycle_aborted_fields
    assert events.OperationKind is core.OperationKind
    calls = []
    monkeypatch.setattr(events, "new_event_id", lambda: calls.append("id") or "evt_default")
    monkeypatch.setattr(events, "event_created_at", lambda: calls.append("time") or 12.5)
    payload = deepcopy(next(p for p in canonical_events() if p["type"] == "memory_compact_started"))
    kwargs: dict[str, Any] = {
        key: value for key, value in payload.items() if key not in {"version", "type", "event_id", "created_at"}
    }
    live = [Message("user", "live")]
    metadata = {"messages": live}
    unchecked: dict[str, Any] = kwargs | {"message_count": True, "estimated_tokens": "raw", "metadata": metadata}
    event = events.MemoryCompactStarted(**unchecked)
    assert calls == ["id", "time"]
    assert (event.event_id, event.created_at) == ("evt_default", 12.5)
    assert event.message_count is True and event.estimated_tokens == "raw"
    assert event.metadata is not metadata and event.metadata["messages"] is live
    assert outcome(events.event_from_dict, event.to_dict()) == (
        ValueError,
        "Run event message_count must be a non-negative JSON-safe integer or null",
    )
    calls.clear()
    with pytest.raises(ValueError, match="Unsupported memory compact trigger: 'bad'"):
        invalid: dict[str, Any] = kwargs | {"trigger": "bad", "configured_threshold": -1}
        events.MemoryCompactStarted(**invalid)
    assert calls == ["id", "time"]
    raw_tokens: Any = "raw"
    completed = events.MemoryCompactCompleted(
        run_id="turn",
        before_count=True,
        after_count=-1,
        summary_tokens=raw_tokens,
        mode="none",
        changed=False,
        archived_count=0,
        reclaimed_tokens=0,
        artifact_failure_count=0,
    )
    assert completed.trace_id == ""
    assert completed.before_count is True and completed.after_count == -1 and completed.summary_tokens == "raw"
