"""SDK observations for the single core operation/model planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _driver_planning as core
from vv_agent.kernel._records import Record

if TYPE_CHECKING:
    from .kernel import _Driver


def plan(
    self: _Driver,
    oid: str,
    request: dict[str, Any],
    kind: str,
    *,
    number: int = 1,
    dependencies: list[str] | None = None,
    capability: dict[str, Any] | None = None,
    purpose: str = "primary",
) -> Record:
    from ._recovery_phase import run

    return run(
        self,
        core.plan(self, oid, request, kind, number=number, dependencies=dependencies, capability=capability, purpose=purpose),
    )


def model_plan(self: _Driver) -> Record | None:
    from ._recovery_phase import run

    return run(self, core.model_plan(self, self.transcript()))


def effective_model(self: _Driver) -> dict[str, Any]:
    from ._recovery_phase import run

    return run(self, core.effective_model(self))
