"""Advance, recover and close turns using the driver's retained state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _driver_outcomes

from .records import Record

if TYPE_CHECKING:
    from .kernel import _Driver


def unknown(self: _Driver, plan: Record, reason: str) -> Record:
    return self._run_outcome_steps(_driver_outcomes.unknown_steps(self, plan, reason))


def step(self: _Driver) -> bool:
    from . import _recovery_phase

    return _recovery_phase.step(self)


def close(self: _Driver, status: str, reason: str | None, result: Any = None, *, transferred: bool = False) -> None:
    from . import _recovery_phase

    _recovery_phase.close(self, status, reason, result, transferred=transferred)
