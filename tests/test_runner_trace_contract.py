from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.kernel_runtime import start_runner

from vv_agent import Agent, RunBudgetLimits, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.runtime.hooks import BaseRuntimeHook
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.tracing import project_spans
from vv_agent.types import LLMResponse, ToolCall


def test_real_runner_trace_matches_current_producer_fixture(tmp_path):
    class Hooks(BaseRuntimeHook):
        def before_tool_call(self, event):
            event.context.shared_state["prepared"] = True

    class AfterCycle:
        def after_cycle(self, snapshot):
            return None

    @function_tool
    def echo(text: str) -> str:
        return text

    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "tools",
            Agent("fixture", "Be precise.", model="m", tools=[echo]),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                hooks=[Hooks()],
                after_cycle_hooks=[AfterCycle()],
                budget_limits=RunBudgetLimits(max_total_tokens=100),
                model_provider=ScriptedModelProvider.from_steps(
                    "scripted", "m", [LLMResponse("", [ToolCall("echo", "echo", {"text": "ok"})]), LLMResponse("done")]
                ),
            ),
        )
        assert handle.result().final_output == "done"
        actual: list[dict[str, Any]] = [
            {"seq": seq, "method": method, "span": span.to_dict()}
            for seq, method, span in project_spans(driver.store.read_state("tools")[1])
        ]
        expected = [
            json.loads(line) for line in (Path(__file__).parent / "fixtures/parity/runner_trace.jsonl").read_text().splitlines()
        ]

        def stable(row):
            return row | {"span": {k: v for k, v in row["span"].items() if k not in {"started_at", "ended_at"}}}

        by_id = {(r["method"], r["span"]["span_id"]): r for r in actual}
        assert [stable(by_id[(r["method"], r["span"]["span_id"])]) for r in expected] == [stable(r) for r in expected]
        assert len(actual) == 6
    finally:
        driver.close()
