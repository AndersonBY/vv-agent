# Run Budgets

`RunBudgetLimits` provides optional, task-neutral resource controls for one
Agent run. A budget limits resources; it does not inspect the prompt, decide
whether an answer is correct, or force a task-specific research phase.

## Public API

```python
from vv_agent import Agent, HostCost, RunBudgetLimits, RunConfig, Runner

result = Runner.run_sync(
    Agent(name="assistant", instructions="Complete the request."),
    "Do the work",
    run_config=RunConfig(
        budget_limits=RunBudgetLimits(
            max_total_tokens=20_000,
            max_uncached_input_tokens=12_000,
            max_tool_calls=40,
            max_tool_calls_by_name={"web_search": 12},
            max_wall_time_ms=300_000,
            max_host_cost=HostCost(unit="credits", amount_microunits=2_000_000),
        )
    ),
)
```

All limits are optional. `None` means unlimited, and an empty limits object has
no runtime effect. Wire integers are limited to `0..9007199254740991`.

`result.budget_usage` exposes the cumulative observation. A stop caused by a
budget has `status == failed`, `completion_reason == budget_exhausted`, and a
typed `result.budget_exhaustion`. Missing accounting stays missing; it is never
converted to zero.

## Enforcement

- Cancellation visible before admission wins over a budget stop.
- Token and host-cost readings are observed after an atomic model call, so one
  completed call may exceed its limit.
- Tool batches are checked and fully reserved before the first side effect. A
  rejected batch executes no tools.
- A tool or model operation error that already occurred is not replaced by a
  later budget observation.
- A natural terminal exactly at a limit remains valid. The next atomic
  operation is rejected because no capacity remains.
- `continue_and_mark` records unavailable accounting and keeps running.
  `stop` converts a configured unavailable dimension into a typed stop.

Configured runs emit `budget_snapshot` observations when accounting changes.
A budget stop emits exactly one `budget_exhausted` event followed by the normal
`run_failed` terminal. Runs without limits emit no budget events and preserve
the previous event order.

## Host Cost

`HostCostMeter.read()` returns a host-scoped cumulative reading. The SDK does
not contain a price table, convert currencies, or subtract an implicit
baseline. Unit, optional currency, and monotonicity must match the configured
limit exactly.

For distributed execution, register the meter in the worker-local capability
registry and set `RuntimeRecipe.capabilities.host_cost_meter_ref`. Process-local
meter objects are not serialized into Celery envelopes.

## Resume And Child Runs

Approval resume preserves the source run's usage while excluding approval wait
time. The resumed model loop still receives the normal fresh `max_cycles`
allowance. Independent Runner calls start fresh.

Framework-created child runs inherit limits but use fresh token, tool, cycle,
and elapsed counters. A parent host meter is not propagated implicitly. Share a
host-scoped meter explicitly when parent and child work must consume one global
ledger.

Distributed workers persist `budget_usage` in the existing checkpoint and add
only each active monotonic worker segment. Queue time is excluded. This is an
additive checkpoint-v1 field and does not claim exact resume or exactly-once
external effects.

## Verification

```bash
uv run pytest tests/test_run_budget.py
uv run pytest tests/test_distributed_contract.py tests/test_checkpoint_contract.py
uv run pytest tests/test_app_server_contract_parity.py
```

The normative cross-language behavior is pinned by `contract.lock.json` and
the vendored `run_budget_v1.json` and `budget_events_v1.jsonl` fixtures.
