# Optional output validation

Output validation lets a host check a run's final output and, if the check
fails, make one repair attempt without tools. It is off by default. It runs
after the agent loop has produced its final output and never calls the
primary model again.

## Python API

Register the callbacks on `Agent` and turn the feature on with
`output_validation_enabled=True`:

```python
from vv_agent import (
    Agent,
    OutputRepairRequest,
    OutputValidationContext,
    OutputValidationResult,
)


def validate_output(
    output: object,
    context: OutputValidationContext,
) -> OutputValidationResult:
    if isinstance(output, dict) and "answer" in output:
        return OutputValidationResult.accept()
    return OutputValidationResult.reject("answer_missing", "Expected an answer field.")


def repair_output(request: OutputRepairRequest) -> object:
    assert request.tools == ()
    return {"answer": request.invalid_output}


agent = Agent(
    name="typed-agent",
    instructions="Return the answer.",
    output_type=dict,
    output_validation_enabled=True,
    output_validator=validate_output,
    output_repair=repair_output,
    output_validation_max_repairs=1,
    output_repair_model="host-selected-repair-model",
    output_repair_model_settings={"temperature": 0},
)
```

Registering callbacks alone does not enable validation. The `Agent`
constructor also checks the configuration:

- `output_validation_enabled=True` requires an `output_validator`.
- `output_repair` requires an `output_validator`.
- `output_validation_max_repairs` must be `0` or `1`.

## Lifecycle and failure

The steps run in this order:

1. The runner applies the output guardrails and `output_type` coercion to get
   a candidate output.
2. If validation is enabled, the validator checks the candidate.
3. If the candidate is invalid and repair is enabled, the repair callback
   receives one `OutputRepairRequest`. Its replacement goes through
   `output_type` coercion and the same validator again.
4. The runner commits the turn's terminal result and emits its terminal
   event.

Because validation happens before step 4, the stored terminal result is
either the validated output or a validation failure. There is at most one
repair, and these cases end the run as failed without another attempt:

- the repaired output is still invalid;
- the repair callback raises (`repair_provider_error`);
- the validator raises (`output_validator_error`) or returns something other
  than `OutputValidationResult` (`output_validator_contract_invalid`). These
  two are not repaired at all.

An `output_type` coercion failure counts as a rejection with code
`output_type_invalid`, so it can be repaired. When validation is
disabled or the validator accepts the output, events and traces look exactly
as they would without this feature.

Replaying a finished turn returns its stored terminal result. The primary
model, the validator, and the repair callback are not called again. A turn
resumed after approval goes through the same steps before its terminal result
is committed.

A validation failure returns a normal `RunResult` with:

- `status == AgentStatus.FAILED`;
- `error_code == "output_validation_failed"`;
- the invalid candidate in `partial_output`, when it can be represented;
- validation or provider details in `raw_result.error`.

The run emits exactly one terminal event for the final result:
`run_failed` after a rejection, or `run_completed` with the repaired output
after a successful repair. There is no separate validation event.

## Safety boundary

`OutputValidationContext` holds only `run_id`, `agent_name`, and the declared
`output_type`. The framework does not look at what the task is about or what
the answer means; that judgment belongs to the host's validator.

`OutputRepairRequest.tools` is always an empty tuple (the contract's empty
tool list). `model` and `model_settings` come from `output_repair_model` and
`output_repair_model_settings` and are passed to the callback as-is. The
runner does not resolve them, add tools, or run repair as another agent cycle.

In a durable session, repair is logged as one `output_repair` model operation
with no tools, and its usage counts toward the run budget. If the outcome of a
repair is unknown, for example because the process died during the call, the
runner does not retry it automatically. A durable host must register the same
validator and repair callbacks when it rebuilds the agent for a turn.

## Tests

```bash
uv run pytest tests/test_output_validation_contract.py
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```
