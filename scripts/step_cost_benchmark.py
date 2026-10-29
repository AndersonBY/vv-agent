"""Measure ten zero-latency tool steps through the real PostgreSQL record path."""

from __future__ import annotations

import argparse
import cProfile
import fcntl
import io
import json
import os
import pstats
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.session.test_recovery_matrix import runtime
from vv_agent.memory import MemoryManager
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, Message, ToolCall

HEAVY_LOCK = Path("/home/makerbi/vectorvein/tmp/codex-runs/perf/heavy.lock")


class ObservationHook:
    def before_memory_compact(self, _event):
        return None


@contextmanager
def heavy_lock():
    """All measured work shares the host's exclusive heavy-work lock."""
    with HEAVY_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.run(["pgrep", "-f", "[b]2-m0-harness/run.py"], capture_output=True).returncode == 0:
            time.sleep(1)
        yield


def text_payload(size: int, seed: int) -> str:
    rng = random.Random(seed)
    words = ["记录", "工具", "验证", "中文", "history", "result", "context", "payload", "step", "exact"]
    lines = [f"line {i}: {' '.join(rng.choices(words, k=14))} {rng.randrange(1000000):06d}.\n" for i in range(size // 60 + 1)]
    return "".join(lines).encode()[:size].decode("utf-8", errors="ignore")


def live_history() -> list[Message]:
    messages = [Message("user", "检查这些资料并继续执行。Check the evidence and continue.")]
    for index, size in enumerate([30, 60, 90, 45, 75, 30, 90, 60, 75, 90, 60, 45]):
        call = f"old-{index}"
        messages.extend(
            [
                Message(
                    "assistant",
                    f"Inspect item {index}.",
                    tool_calls=[{"id": call, "type": "function", "function": {"name": "payload", "arguments": '{"step":0}'}}],
                ),
                Message("tool", text_payload(size * 1024, index), tool_call_id=call, name="payload"),
            ]
        )
    return messages


def run_case(store, dsn: str, profiler: cProfile.Profile | None = None) -> dict:
    from deepseek_tokenizer import deepseek_tokenizer

    instants = []
    tokenized = []
    pending_chars = 0
    tokenizer_calls = 0
    resolve_calls = 0
    original_encode = deepseek_tokenizer.encode
    outputs = [text_payload((30 + step % 3 * 30) * 1024, step + 100) for step in range(10)]

    def encode(text, *args, **kwargs):
        nonlocal pending_chars, tokenizer_calls
        pending_chars += len(text)
        tokenizer_calls += 1
        return original_encode(text, *args, **kwargs)

    @function_tool
    def payload(step: int) -> str:
        return outputs[step]

    def model(request):
        nonlocal pending_chars
        instants.append(time.perf_counter())
        tokenized.append(pending_chars)
        pending_chars = 0
        index = len(instants) - 1
        if index < 10:
            return LLMResponse(f"Read step {index}.", [ToolCall(f"new-{index}", "payload", {"step": index})])
        return LLMResponse("done")

    rt = runtime(dsn, [model] * 11, tools=[payload])
    rt.config = replace(rt.config, max_cycles=20)
    rt.hooks.hooks.append(ObservationHook())
    rt.resolved = replace(rt.resolved, model_id="deepseek-v4-pro", requested_model="deepseek-v4-pro")
    rt.memory_manager = MemoryManager(
        model="deepseek-v4-pro",
        compact_threshold=10_000_000,
        model_context_window=20_000_000,
        microcompaction_policy=MicrocompactionPolicy(min_result_chars=10_000_000),
    )

    def resolver(_request):
        nonlocal resolve_calls
        resolve_calls += 1
        return rt.llm, rt.resolved

    rt.model_resolver = resolver
    seed = live_history()
    seed_bytes = sum(len(m.content.encode()) for m in seed)
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                "s", "benchmark", "/tmp", attributes={"seed": {"messages": [m.to_dict() for m in seed], "shared_state": {}}}
            ),
            consumers=("events",),
        )
        tx.push("s", InboxItem("initial", "user", {"content": "continue"}))
    with patch.object(deepseek_tokenizer, "encode", encode):
        begin = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        try:
            drive(store, "s", runtime=rt)
        finally:
            if profiler is not None:
                profiler.disable()
        total = time.perf_counter() - begin
    state, rows, _ = read_state(store, "s")
    assert state.active_turn_id is None and len(instants) == 11, (len(instants), rows[-1].record._payload)
    assert rows[-1].record._payload["status"] == "completed"
    values = [(b - a) * 1000 for a, b in pairwise(instants)]
    return {
        "seed_content_bytes": seed_bytes,
        "step_ms": values,
        "median_step_ms": statistics.median(values),
        "mean_step_ms": statistics.mean(values),
        "first_model_ms": (instants[0] - begin) * 1000,
        "drive_ms": total * 1000,
        "tokenized_chars_before_each_model": tokenized,
        "tokenizer_calls": tokenizer_calls,
        "resolver_calls": resolve_calls,
        "records": len(rows),
        "record_bytes": sum(len(row.record.encode()) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--dsn", default=os.environ.get("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"timings": [], "profile": {}}
    with heavy_lock(), psycopg.connect(args.dsn, autocommit=True) as admin:
        print("Exclusive heavy.lock acquired; b2-m0 harness absent.", flush=True)
        for index in range(args.samples + 1):
            name = f"vvsk_test_vtok_{uuid4().hex}"
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            try:
                dsn = make_conninfo(args.dsn, dbname=name)
                with PostgresStore.standalone(dsn) as store:
                    store.install_schema()
                    if index == args.samples:
                        profiler = cProfile.Profile()
                        result = run_case(store, dsn, profiler)
                        profiler.dump_stats(str(args.output / "drive.prof"))
                        stream = io.StringIO()
                        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(65)
                        (args.output / "profile.txt").write_text(stream.getvalue())
                        results["profile"] = result
                    else:
                        result = run_case(store, dsn)
                        results["timings"].append(result)
                    print(json.dumps({"sample": index, **result}), flush=True)
            finally:
                admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
