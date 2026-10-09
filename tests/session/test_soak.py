"""Small normal-suite repetition of the same soak that runs at release scale."""

import json
import runpy
from pathlib import Path

import pytest

from vv_agent.session.records import InboxItem
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .transport import PollingProvider, Transport

run_soak = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/session_kernel_soak.py"))["run_soak"]

pytestmark = pytest.mark.persistent_store


@pytest.mark.parametrize("seed", [42, 97])
def test_seeded_fault_soak(database, seed):
    summary = run_soak(database, seed=seed, sessions=8)
    assert summary["counts"]["terminal_sessions"] == 9
    assert summary["counts"]["abandoned_workers"] == 2
    assert summary["queue_remaining"] == 0


def test_duplicate_wake_cannot_poll_accepted_before_due(store, database, monkeypatch):
    """A duplicate delivery must respect the stored poll deferral."""
    monkeypatch.setattr(type(store), "clock_sql", "1000000")
    transport = Transport(store, database)
    provider = PollingProvider(database)
    provider.install()

    @function_tool
    def effect() -> str:
        raise AssertionError("provider must own the external operation")

    transport.create(steps=[LLMResponse("", [ToolCall("job", "effect", {})]), LLMResponse("done")], tools=[effect])
    transport.runtimes["s"].providers["effect"] = provider
    transport.runtimes["s"].poll_ms = 1000
    transport.push("s", InboxItem("initial", "user", {"content": "go"}))
    transport.deliver()
    assert provider.queries == []
    monkeypatch.setattr(type(store), "clock_sql", "1001000")
    transport.tick()
    before = tuple(row.record.encode() for row in store.read_state("s")[1])
    assert provider.queries == [1_001_000]
    assert store._one("SELECT next_drive_ms FROM sk_session WHERE session_id='s'") == (1_002_000,)
    assert not store.is_runnable("s")
    transport.wake("s")  # One duplicate broker delivery, with no new inbox item or clock advance.
    transport.deliver()
    assert before == tuple(row.record.encode() for row in store.read_state("s")[1])
    assert provider.counts() == (1, 1)
    assert transport.queue == [] and not store.is_runnable("s")
    print(
        json.dumps(
            {
                "queries_ms": provider.queries,
                "period_ms": 1000,
                "elapsed_ms": 0,
                "bound": 1,
                "submits": 1,
                "log_unchanged": True,
                "queue_remaining": 0,
            }
        )
    )
    assert len(provider.queries) == 1, f"duplicate wake exceeded F3d bound: {provider.queries}"
