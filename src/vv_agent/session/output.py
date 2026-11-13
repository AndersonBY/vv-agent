"""SDK observations and guarded ledger operations for the core output phase."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _output_phase as phase
from vv_agent.kernel._output_phase import serializable_output as serializable_output
from vv_agent.output_validation import (
    OutputValidationContext as OutputValidationContext,
)
from vv_agent.output_validation import (
    output_validation_error as output_validation_error,
)
from vv_agent.output_validation import (
    postprocess_output as postprocess_output,
)
from vv_agent.output_validation import (
    run_output_validator as run_output_validator,
)

if TYPE_CHECKING:
    from .kernel import _Driver


def prepare_output(driver: _Driver, value: Any) -> tuple[str, str | None, Any] | None:
    from ._recovery_phase import run

    assert driver.state.active_turn_id is not None
    agent = driver.runtime.agent
    steps = phase.prepare_output(driver.state, value)
    return run(driver, steps, output_agent=agent)
