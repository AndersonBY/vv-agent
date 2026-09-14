from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from test_checkpoint_runner import _config, _provider

from vv_agent import Agent, Runner, function_tool
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.lifecycle import AfterCycleDecision, AfterCycleSnapshot
from vv_agent.runtime.state import CheckpointStore
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.redis import RedisCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


class _ObservedStore:
    def __init__(self, inner: Any, crash_before_commit: bool) -> None:
        self.inner = inner
        self.history_reads = 0
        self.model_calls = 0
        self.crash_after_commit = 5
        self.crash_before_commit = crash_before_commit

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def load_checkpoint(self, key: str) -> Any:
        checkpoint = self.inner.load_checkpoint(key)
        if checkpoint is not None:
            assert len(checkpoint.cycles) <= 2
            assert len(checkpoint.model_calls) <= 2
        return checkpoint

    def load_checkpoint_history(self, key: str) -> Any:
        assert self.model_calls == 12, "execution or recovery loaded complete history before the result boundary"
        self.history_reads += 1
        return self.inner.load_checkpoint_history(key)

    def commit_checkpoint(self, checkpoint: Any, **kwargs: Any) -> bool:
        if self.crash_before_commit and checkpoint.cycle_index == self.crash_after_commit:
            self.crash_after_commit = 0
            raise SystemExit("crash before atomic cycle and history commit")
        written = self.inner.commit_checkpoint(checkpoint, **kwargs)
        if written:
            self.load_checkpoint(checkpoint.checkpoint_key)
            if checkpoint.cycle_index == self.crash_after_commit:
                self.crash_after_commit = 0
                raise SystemExit("crash after atomic cycle and history commit")
        return written


@pytest.mark.parametrize("kind", ["memory", "sqlite", "redis"])
@pytest.mark.parametrize("crash_before_commit", [False, True])
def test_runner_archives_history_recovers_committed_cycle_and_replays_complete_result(
    kind: str, crash_before_commit: bool, tmp_path: Path
) -> None:
    if kind == "sqlite":
        inner: Any = SqliteCheckpointStore(tmp_path / "runner-history.sqlite3")
    elif kind == "redis":
        redis_url = os.getenv("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for real Redis producer history")
        inner = RedisCheckpointStore(redis_url)
    else:
        inner = InMemoryCheckpointStore()
    store = _ObservedStore(inner, crash_before_commit)
    key = f"runner-history-{uuid4().hex}"
    effects: list[int] = []
    observations: list[tuple[int, int | None]] = []

    @function_tool(name="record_cycle", tool_metadata={"idempotency": "supported"})
    def record_cycle(index: int) -> str:
        effects.append(index)
        return f"recorded {index}"

    class ObserveUsage:
        def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
            assert store.history_reads == 0
            observations.append((snapshot.cycle_index, snapshot.cumulative_token_usage.total_tokens))
            assert not hasattr(snapshot.cumulative_token_usage, "model_calls")
            return AfterCycleDecision.continue_run()

    def complete(_request: Any) -> LLMResponse:
        assert store.history_reads == 0
        store.model_calls += 1
        index = store.model_calls
        return LLMResponse(
            content=f"cycle {index}",
            tool_calls=[ToolCall(id=f"tool-{index}", name="record_cycle", arguments={"index": index})] if index < 12 else [],
            raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )

    agent = Agent(name="history-producer", instructions="Record each requested cycle.", model="test-model", tools=[record_cycle])
    config = _config(
        cast(CheckpointStore, store),
        key=key,
        provider=_provider(lambda: ScriptedLLM(steps=[complete] * 12)),
        max_cycles=12,
        capability_refs={"after_cycle_hook:0": {"id": "history.observer", "version": "1"}},
    )
    config.after_cycle_hooks = [ObserveUsage()]
    try:
        with pytest.raises(SystemExit, match=r"crash .* atomic"):
            Runner.run_sync(agent, "record cycles", run_config=config)
        assert store.model_calls == 5
        assert effects == list(range(1, 6))
        assert store.history_reads == 0
        retained = store.load_checkpoint(key)
        assert retained.cycle_index == (4 if crash_before_commit else 5)
        assert retained.history["cycle_count"] == (3 if crash_before_commit else 4)
        if crash_before_commit:
            # Expire only this disposable store's abandoned claim.
            if isinstance(inner, SqliteCheckpointStore):
                with inner._conn:
                    inner._conn.execute("UPDATE checkpoints SET lease_expires_at_ms = 1 WHERE checkpoint_key = ?", (key,))
            elif isinstance(inner, RedisCheckpointStore):
                inner._client.set(inner._keys(key)[1], "1")
            else:
                assert isinstance(inner, InMemoryCheckpointStore)
                inner._store[key].lease_expires_at_ms = 1
        else:
            assert retained.claim_token is None

        result = Runner.run_sync(agent, "record cycles", run_config=config)
        assert result.status is AgentStatus.COMPLETED
        assert result.final_output == "cycle 12"
        assert store.model_calls == 12
        assert effects == list(range(1, 12))
        assert [cycle.index for cycle in result.raw_result.cycles] == list(range(1, 13))
        assert [record.cycle_index for record in result.token_usage.model_calls] == list(range(1, 13))
        assert len({record.call_id for record in result.token_usage.model_calls}) == 12
        assert result.token_usage.total_tokens == 180
        assert observations and observations[-1] == (12, 180)
        assert all(total == index * 15 for index, total in observations)
        assert store.history_reads > 0
        terminal = store.load_checkpoint(key)
        assert len(terminal.cycles) == 1
        assert len(terminal.model_calls) == 1
        assert terminal.history["cycle_count"] == 11

        replay = Runner.run_sync(agent, "record cycles", run_config=config)
        assert replay.status is AgentStatus.COMPLETED
        assert replay.token_usage == result.token_usage
        assert replay.raw_result.cycles == result.raw_result.cycles
        assert store.model_calls == 12
        assert effects == list(range(1, 12))
    finally:
        inner.delete_checkpoint(key)
        if isinstance(inner, SqliteCheckpointStore):
            inner.close()
