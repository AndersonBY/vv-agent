from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent import event_from_dict

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_events_v1_invalid.json"
FIXTURE_SHA256 = "55e3be856d8c1cc1c522cefa8bb0d0aa05b4552e7eda34c0a6c5c04172394e06"


def _contract() -> dict[str, Any]:
    fixture_bytes = FIXTURE.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256
    return json.loads(fixture_bytes)


def test_run_event_v1_compatibility_inputs_canonicalize_to_fixture() -> None:
    contract = _contract()

    for case in contract["canonicalize"]:
        event = event_from_dict(case["input"])
        assert event.to_dict() == case["output"], case["id"]


def test_run_event_v1_invalid_inputs_are_rejected() -> None:
    contract = _contract()

    for case in contract["reject"]:
        with pytest.raises(ValueError, match=r".+"):
            event_from_dict(case["input"])
