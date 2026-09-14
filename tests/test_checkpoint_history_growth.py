from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from test_checkpoint import _minimal_checkpoint
from test_checkpoint_history_stores import _candidate, _commit
from test_checkpoint_runner import _config, _provider

import vv_agent.checkpoint as canonical
import vv_agent.runtime.checkpoint_codec as codec
import vv_agent.runtime.stores.redis as redis_store
import vv_agent.runtime.stores.sqlite as sqlite_store
from vv_agent import Agent, Runner
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse


def _count_sqlite_json_writes(monkeypatch: pytest.MonkeyPatch, counts: Counter[str]) -> None:
    original_connect = sqlite3.connect

    class CountingSqliteConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> Any:
            if sql.lstrip().startswith(("INSERT", "UPDATE")):
                counts["sqlite_bound_json_bytes"] += sum(
                    len(value.encode("utf-8")) for value in parameters if isinstance(value, str) and value.startswith(("[", "{"))
                )
                counts["write_operations"] += 1
            return super().execute(sql, parameters)

    def connect(database: str | Path, *, check_same_thread: bool = True) -> CountingSqliteConnection:
        return original_connect(database, check_same_thread=check_same_thread, factory=CountingSqliteConnection)

    monkeypatch.setattr(sqlite3, "connect", connect)


@pytest.mark.parametrize("kind", ["sqlite", "redis"])
def test_fixed_context_history_cumulative_serialization_and_storage_bytes_are_linear(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: Counter[str] = Counter()
    original_canonical = canonical._canonical_json
    original_decode = codec._strict_json_loads
    original_sql_json = sqlite_store._json_dump

    def encode(value: Any, field_name: str) -> str:
        payload = original_canonical(value, field_name)
        if isinstance(value, dict):
            schema = value.get("schema_version", "")
            if schema.startswith("vv-agent.checkpoint-history."):
                counts["archive_encoded_bytes"] += len(payload.encode("utf-8"))
            elif schema.startswith("vv-agent.checkpoint."):
                counts["checkpoint_encoded_bytes"] += len(payload.encode("utf-8"))
        return payload

    def decode(payload: str | bytes) -> Any:
        counts["json_decoded_bytes"] += len(payload.encode("utf-8") if isinstance(payload, str) else payload)
        return original_decode(payload)

    def sql_json(value: Any) -> str:
        payload = original_sql_json(value)
        counts["sqlite_json_encoded_bytes"] += len(payload.encode("utf-8"))
        return payload

    monkeypatch.setattr(canonical, "_canonical_json", encode)
    for module in (codec, sqlite_store, redis_store):
        monkeypatch.setattr(module, "_strict_json_loads", decode)
    monkeypatch.setattr(sqlite_store, "_json_dump", sql_json)

    if kind == "sqlite":
        _count_sqlite_json_writes(monkeypatch, counts)
        store: sqlite_store.SqliteCheckpointStore | redis_store.RedisCheckpointStore = sqlite_store.SqliteCheckpointStore(
            tmp_path / "history-growth.sqlite3"
        )
    else:
        redis_url = os.getenv("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for actual Redis socket byte accounting")
        import redis

        class CountingSocket:
            def __init__(self, socket: Any) -> None:
                self.socket = socket

            def __getattr__(self, name: str) -> Any:
                return getattr(self.socket, name)

            def sendall(self, data: bytes, *args: Any, **kwargs: Any) -> Any:
                result = self.socket.sendall(data, *args, **kwargs)
                counts["redis_sent_bytes"] += len(data)
                return result

            def recv(self, *args: Any, **kwargs: Any) -> bytes:
                data = self.socket.recv(*args, **kwargs)
                counts["redis_received_bytes"] += len(data)
                return data

            def recv_into(self, *args: Any, **kwargs: Any) -> int:
                received = self.socket.recv_into(*args, **kwargs)
                counts["redis_received_bytes"] += received
                return received

        class CountingRedisConnection(redis.Connection):
            def _connect(self) -> Any:
                return CountingSocket(super()._connect())

        pool = redis.ConnectionPool.from_url(redis_url, connection_class=CountingRedisConnection, decode_responses=True)
        store = redis_store.RedisCheckpointStore(redis_url)
        store._client.close()
        store._client = redis.Redis(connection_pool=pool)

    key = f"history-growth-{uuid4().hex}"
    retained_sizes: list[int] = []
    samples: dict[int, dict[str, int]] = {}
    try:
        initial = _minimal_checkpoint(key=key)
        assert store.create_checkpoint(initial)
        for index in range(1, 41):
            candidate = _candidate(store, key, index)
            assert candidate is not None
            candidate.cycle_index = index - 1
            # Exercise repeated progress writes as well as the atomic commit.
            for _ in range(2):
                assert candidate.claim_token is not None
                assert store.progress_checkpoint(
                    candidate, claim_token=candidate.claim_token, expected_revision=candidate.revision
                )
                candidate = store.load_checkpoint(key)
                assert candidate is not None
            candidate.cycle_index = index
            assert _commit(store, candidate)
            current = store.load_checkpoint(key)
            assert current is not None
            assert current.messages == initial.messages
            assert len(current.cycles) == len(current.model_calls) == 1
            # Size observation does not add another checkpoint serialization to
            # the workload counters; actual codec and storage calls are above.
            retained_sizes.append(
                len(original_canonical(codec.checkpoint_to_dict(current), "retained checkpoint").encode("utf-8"))
            )
            if index in (20, 40):
                samples[index] = dict(counts)
        required = {"checkpoint_encoded_bytes", "archive_encoded_bytes", "json_decoded_bytes"}
        required |= (
            {"sqlite_json_encoded_bytes", "sqlite_bound_json_bytes"}
            if kind == "sqlite"
            else {"redis_sent_bytes", "redis_received_bytes"}
        )
        for metric in required:
            assert samples[20][metric] > 0
            assert samples[40][metric] <= samples[20][metric] * 2.2, (metric, samples)
        assert max(retained_sizes[1:]) < min(retained_sizes[1:]) * 1.02
        print(json.dumps({"backend": kind, "bytes_at_20": samples[20], "bytes_at_40": samples[40]}, sort_keys=True))
    finally:
        store.delete_checkpoint(key)
        if isinstance(store, sqlite_store.SqliteCheckpointStore):
            store.close()
        else:
            store._client.close()
            store._client.connection_pool.disconnect()


def test_runner_growing_context_reports_cumulative_sqlite_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts: Counter[str] = Counter()
    _count_sqlite_json_writes(monkeypatch, counts)
    samples: list[dict[str, Any]] = []
    output = "x" * 4096
    for cycle_count in (20, 40):
        counts.clear()
        store = sqlite_store.SqliteCheckpointStore(tmp_path / f"growing-context-{cycle_count}.sqlite3")
        key = f"growing-context-{cycle_count}"
        request_message_counts: list[int] = []

        def complete(request: Any, observed_counts: list[int] = request_message_counts) -> LLMResponse:
            observed_counts.append(len(request.messages))
            return LLMResponse(
                content=output,
                raw={"usage": {"prompt_tokens": 100, "completion_tokens": 1024, "total_tokens": 1124}},
            )

        config = _config(
            store,
            key=key,
            provider=_provider(lambda count=cycle_count: ScriptedLLM(steps=[complete] * count)),
            max_cycles=cycle_count,
            no_tool_policy="continue",
        )
        try:
            result = Runner.run_sync(
                Agent(name="growing-context", instructions="Continue.", model="test-model"),
                "Measure current context growth.",
                run_config=config,
            )
            checkpoint = store.load_checkpoint(key)
            assert checkpoint is not None
            assert request_message_counts == list(range(2, cycle_count * 2 + 1, 2))
            assert len(checkpoint.messages) == cycle_count * 2 + 1
            # All generated content survives: this workload must not silently
            # become the fixed-context/compacted benchmark above.
            assert sum(message.content == output for message in checkpoint.messages) == cycle_count
            assert len(checkpoint.cycles) == len(checkpoint.model_calls) == 1
            assert checkpoint.history["cycle_count"] == cycle_count - 1
            assert [cycle.index for cycle in result.raw_result.cycles] == list(range(1, cycle_count + 1))
            assert len(result.token_usage.model_calls) == cycle_count
            assert result.token_usage.total_tokens == cycle_count * 1124
            assert counts["sqlite_bound_json_bytes"] > 0
            samples.append(
                {
                    "cycles": cycle_count,
                    "message_counts_first_last": [request_message_counts[0], request_message_counts[-1]],
                    "checkpoint_message_count": len(checkpoint.messages),
                    "checkpoint_bytes": len(json.dumps(codec.checkpoint_to_dict(checkpoint)).encode("utf-8")),
                    "retained_cycles": len(checkpoint.cycles),
                    "archived_cycles": checkpoint.history["cycle_count"],
                    "public_cycles": len(result.raw_result.cycles),
                    **counts,
                }
            )
        finally:
            store.close()
    # This reports the unbounded-context cost, not a linear-growth acceptance
    # threshold. SQL binding bytes are neither disk/WAL nor network bytes.
    print(
        json.dumps(
            {
                "backend": "sqlite",
                "workload": "real_runner_growing_context",
                "per_cycle_output_bytes": len(output.encode("utf-8")),
                "samples": samples,
                "write_bytes_ratio_40_over_20": (samples[1]["sqlite_bound_json_bytes"] / samples[0]["sqlite_bound_json_bytes"]),
            },
            sort_keys=True,
        )
    )
