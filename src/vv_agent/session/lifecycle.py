"""Drive after-cycle host requests against the real session driver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.kernel import _after_cycle as phase
from vv_agent.runtime.lifecycle import AfterCycleHookManager

if TYPE_CHECKING:
    from .kernel import _Driver


def after_cycle(driver: _Driver) -> bool:
    from ._recovery_phase import run

    manager = AfterCycleHookManager(driver.runtime.config.after_cycle_hooks)
    steps = phase.after_cycle(driver.state, has_hooks=manager.has_hooks())
    return run(driver, steps, after_cycle=manager)
