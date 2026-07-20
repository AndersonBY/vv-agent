# Optional Output Validation

Contract `0.9.0` adds a default-off host extension that validates a completed
output and may make one tools-free repair request. It runs outside the normal
agent loop and does not call the primary model again.

## Python API

Register callbacks on `Agent` and opt in explicitly with
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

Merely registering callbacks does not enable them. The default remains off,
and `output_validation_max_repairs` accepts only `0` or `1`.

## Lifecycle And Failure

The Runner first applies the existing output guardrail and `output_type`
coercion to form a terminal candidate. When this extension is enabled, the
validator and optional one-shot repair run before session persistence,
checkpoint finalization, and terminal-event emission. The committed terminal
is therefore either the validated success or a typed validation failure.
Disabled and accepted validators preserve the existing event and trace shape.
A checkpoint terminal replay reuses that authoritative terminal and does not
call the primary model, validator, or repair callback again.

If the initial output is invalid and repair is enabled, the callback receives
one `OutputRepairRequest`. A replacement is passed through `output_type` and
the same validator again. A second invalid result, a repair exception, or a
malformed validator result cannot start another repair.

Validation failure returns a normal `RunResult` with:

- `status == AgentStatus.FAILED`;
- `error_code == "output_validation_failed"`;
- the invalid candidate in `partial_output` when it can be represented;
- diagnostic validation or provider detail in `raw_result.error`.

Exactly one terminal event is emitted for the final validated observation. A
rejection emits `run_failed`; a successful repair emits `run_completed` with
the repaired output. Approval resume follows the same ordering before its
fresh terminal is committed. No separate validation event is added.

## Safety Boundary

`OutputValidationContext` contains only `run_id`, `agent_name`, and the
declared `output_type`. The framework does not inspect task categories, answer
meaning, domain milestones, or stopping rules.

`OutputRepairRequest.tools` is an immutable empty tuple, the Python adaptation
of the contract's empty tool collection. `model` and `model_settings` are
host-selected descriptors passed to the callback; the Runner does not resolve
them, inject tools, or turn repair into another agent cycle.

Checkpoint v2 requires stable `output_validator` and `output_repair`
capability refs when those callbacks are enabled.

## Verification

```bash
uv run pytest tests/test_output_validation_contract.py
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```
