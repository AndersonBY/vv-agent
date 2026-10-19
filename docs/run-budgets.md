# Run budgets

`RunBudgetLimits` sets optional resource limits for one Agent run. A budget
limits resources only. It does not read the prompt, judge whether an answer is
correct, or force the agent through any task-specific phase.

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

All limits are optional. `None` means unlimited, and an empty limits object
changes nothing. Integer limits must be in `0..9007199254740991`
(2^53 - 1, the largest integer that every JSON parser reads exactly).

`result.budget_usage` holds the cumulative usage. A run stopped by a budget
has `status == failed`, `completion_reason == budget_exhausted`, error code
`run_budget_exhausted`, and a typed `result.budget_exhaustion`. Usage the
provider did not report stays missing; it is never counted as zero.

## Enforcement

The runtime checks the budget before it starts every model call: primary
calls, compaction summaries, Session Memory extraction, and output repair.

- A model call cannot be split, so token and host-cost usage is read after the
  call completes. One completed call may therefore go over its limit; the next
  call is then refused.
- A tool batch is checked and reserved as a whole before the first tool runs.
  If the batch does not fit, no tool in it runs.
- Cancellation seen before an operation is admitted wins over a budget stop.
- An error that a tool or model call already produced is not replaced by a
  later budget stop.
- A run that finishes exactly at a limit still succeeds. Only the next
  operation is refused, because no capacity is left.
- `RunBudgetLimits.unavailable_metric_policy` decides what happens when a
  limited dimension cannot be measured. The default, `continue_and_mark`,
  records the dimension as unavailable and keeps running. `stop` turns it
  into a budget stop.

Runs with limits emit a `budget_snapshot` event when usage changes. A budget
stop emits exactly one `budget_exhausted` event, followed by the normal
`run_failed` terminal event. Runs without limits emit no budget events.

## Host cost

Pass a meter in `RunConfig.host_cost_meter`. `HostCostMeter.read()` returns
the host's cumulative cost reading as a `HostCost`. The SDK has no price
table, does not convert currencies, and does not subtract a starting value.
The reading's `unit` and optional `currency` must match the limit exactly,
and readings must never decrease. A missing reading, a mismatch, or a
decrease marks host cost as unavailable, and `unavailable_metric_policy`
decides what happens next.

The meter object lives only in the host process; session records do not store
it. A durable host that resumes a turn must pass the meter again when it
rebuilds the runtime.

## Resume and child runs

An approval or user reply continues the same turn. Usage and the cycle
allowance carry over, and the time spent waiting does not count toward
`max_wall_time_ms`. A new turn starts with fresh counters.

Child runs created by the framework inherit the parent's limits but start
with fresh token, tool, cycle, and elapsed-time counters. Children get no host
meter, so `max_host_cost` is enforced on the parent only. If child spending
must count against the parent's limit, have the parent's meter read a
host-side ledger that also records the children's cost.

The session log records usage for each stretch of active execution, measured
with a monotonic clock. Time spent queued is not counted. If the process dies
during a stretch, that stretch's wall time is marked unavailable. A model or
tool result already stored in the log is reused without calling it again and
without charging its usage twice.

## Tests

```bash
uv run pytest tests/test_run_budget.py
uv run pytest tests/session/test_capability_parity.py tests/test_run_resume.py
uv run pytest tests/test_app_server_contract_parity.py
```

The shared behavior is defined by the `run_budget.json` and
`budget_events.jsonl` contract fixtures, pinned in `contract.lock.json`.
