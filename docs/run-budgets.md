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

A durable host must reconstruct the named host binding for the retained turn.
Process-local meter objects are not serialized into session records.

## Resume And Child Runs

Approval and user replies continue the same retained turn, preserving usage and
cycle allowances while excluding wait time. Fresh turns start fresh counters.

Framework-created child runs inherit limits but use fresh token, tool, cycle,
and elapsed counters. A parent host meter is not propagated implicitly. Share a
host-scoped meter explicitly when parent and child work must consume one global
ledger.

The session log retains accounting observations for each active monotonic
segment. Queue time is excluded. Lost intervals remain unavailable; a retained
receipt is reused without dispatch or another usage charge.

## Verification

```bash
uv run pytest tests/test_run_budget.py
uv run pytest tests/session/test_capability_parity.py tests/test_run_resume.py
uv run pytest tests/test_app_server_contract_parity.py
```

The normative cross-language behavior is pinned by `contract.lock.json` and
the vendored `run_budget.json` and `budget_events.jsonl` fixtures.

Model admission is checked before every primary, compaction, Session Memory and
output-repair operation. Atomic completion accounts for failures and unknown usage,
and checks overshoot before the next operation. A reserved tool batch completes as
a whole. Cancellation and an existing operation failure keep terminal priority.
Completed retained receipts replay without another dispatch or usage charge.
