"""ApprovalProvider/Broker transport; the parked operation and inbox own decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kernel import _Driver
    from .reducer import Attempt


def resolve_approval(driver: _Driver, attempt: Attempt) -> bool:
    from vv_agent.kernel._driver_recovery import _approval

    from ._recovery_phase import run

    return run(driver, _approval(driver, attempt))
