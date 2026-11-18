"""SDK operations requested by the shared inbox and turn-admission algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.kernel import _driver_inputs as inputs

from .records import InboxItem, Record

if TYPE_CHECKING:
    from .kernel import _Driver


def apply_input(self: _Driver) -> bool:
    from ._recovery_phase import run

    return run(self, inputs.apply_input(self, one_turn=self.one_turn))


def apply_child(self: _Driver, item: InboxItem) -> tuple[str, list[Record], str, str | None]:
    from ._recovery_phase import run

    return run(self, inputs.apply_child(self, item))


def start_turn(self: _Driver) -> bool:
    from ._recovery_phase import run

    return run(self, inputs.start_turn(self))
