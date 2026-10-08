"""End-to-end scripted short-run cost. Run from the repository root with uv run python."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from vv_agent import Agent, MemorySession, RunConfig, Runner
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.kernel import Runtime, drive, read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall


@function_tool
def echo(text: str) -> str:
    return text


def run_case(path: str, scenario: str, workspace: Path) -> None:
    ready, release = threading.Event(), threading.Event()

    def blocking(_request):
        ready.set()
        if not release.wait(5):
            raise TimeoutError("benchmark cancellation did not release the provider")
        return LLMResponse("cancelled")

    turns = 10 if scenario == "ten_turns" else 1
    steps = [LLMResponse("done") for _ in range(turns)]
    if scenario == "two_tools":
        steps.insert(0, LLMResponse("", [ToolCall("a", "echo", {"text": "a"}), ToolCall("b", "echo", {"text": "b"})]))
    llm = ScriptedLLM([blocking] if scenario == "start_cancel" else steps)
    provider = ScriptedModelProvider("scripted", "m", llm, context_length=None, max_output_tokens=None)
    config = RunConfig(model_provider=provider, workspace=workspace)
    agent = Agent("bench", "Be concise.", tools=[echo])
    if path == "runner":
        if scenario == "start_cancel":
            handle = Runner.start(agent, "go", run_config=config)
            try:
                assert ready.wait(5), "Runner did not enter provider"
                handle.cancel()
            finally:
                release.set()
            assert handle.result(timeout=5).completion_reason.value == "cancelled"
        else:
            config.session = MemorySession("bench")
            for _ in range(turns):
                assert Runner.run_sync(agent, "go", run_config=config).final_output == "done"
    else:
        with SQLiteStore.standalone(":memory:") as store:
            store.install_schema()
            with store.atomic() as tx:
                tx.create(SessionSpec("bench", "test", str(workspace)), consumers=("events",))
            runtime = Runtime(agent, config, provider.resolve(provider.default_model_ref()), llm, lambda: nullcontext(store))
            for i in range(turns):
                with store.atomic() as tx:
                    tx.push("bench", InboxItem(str(i), "user", {"content": "go"}))
                if scenario == "start_cancel":
                    errors = []

                    def run(errors=errors):
                        try:
                            drive(store, "bench", runtime=runtime)
                        except BaseException as exc:
                            errors.append(exc)

                    thread = threading.Thread(target=run, name="benchmark-driver")
                    thread.start()
                    try:
                        assert ready.wait(5), "kernel did not enter provider"
                        with store.atomic() as tx:
                            tx.push("bench", InboxItem("cancel", "control", {"action": "cancel"}, "bench/turn/0"))
                    finally:
                        release.set()
                        thread.join(timeout=5)
                    assert not thread.is_alive() and not errors, errors
                else:
                    drive(store, "bench", runtime=runtime)
            state, records, _ = read_state(store, "bench")
            assert state.active_turn_id is None
            terminal = [r.record.payload for r in records if r.record.kind == "turn_ended"]
            assert len(terminal) == turns
            assert terminal[-1]["status"] == ("cancelled" if scenario == "start_cancel" else "completed")
            if scenario != "start_cancel":
                assert terminal[-1]["result"] == "done"
    assert not llm.steps


def rss_bytes() -> int:
    # Linux current RSS, not the monotonic high-water mark returned by getrusage.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("RSS is unavailable")


def benchmark(runs: int, warmup: int) -> dict:
    rows = []
    with TemporaryDirectory(prefix="vv-kernel-bench-") as temporary:
        workspace = Path(temporary)
        for scenario in ("no_tool", "two_tools", "ten_turns", "start_cancel"):
            measurements = {}
            for path in ("runner", "kernel"):
                for _ in range(warmup):
                    run_case(path, scenario, workspace)
                gc.collect()
                before_rss = rss_bytes()
                before_threads = set(threading.enumerate())
                samples = []
                for _ in range(runs):
                    start = time.perf_counter_ns()
                    run_case(path, scenario, workspace)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                gc.collect()
                remaining = set(threading.enumerate())
                measurements[path] = {
                    "p50_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[math.ceil(runs * 0.95) - 1],
                    "threads_after": len(remaining),
                    "leaked_threads": sorted(t.name for t in remaining - before_threads),
                    "rss_delta_bytes": rss_bytes() - before_rss,
                }
            added = measurements["kernel"]["p95_ms"] - measurements["runner"]["p95_ms"]
            target = 80 if scenario == "ten_turns" else 50
            row = {"scenario": scenario, **measurements, "p95_added_ms": added, "target_ms": target}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {
        "runs": runs,
        "warmup": warmup,
        "single_turn_target_ms": 50,
        "per_turn_target_ms": 8,
        "rows": rows,
        "passed": all(r["p95_added_ms"] <= r["target_ms"] and not r["kernel"]["leaked_threads"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 200 or args.warmup < 0:
        parser.error("use at least 200 measured runs and non-negative warmup")
    result = benchmark(args.runs, args.warmup)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
