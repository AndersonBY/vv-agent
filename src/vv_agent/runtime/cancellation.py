from __future__ import annotations

import threading
from typing import cast

from vv_agent.kernel._cancellation import Cancellation
from vv_agent.kernel._cancellation import CancelledError as CancelledError


class CancellationToken(Cancellation):
    """Thread-safe cancellation primitive with parent-child propagation."""

    def __init__(self) -> None:
        super().__init__(signal=threading.Event(), lock=threading.Lock(), child_factory=CancellationToken)

    def child(self) -> CancellationToken:
        return cast(CancellationToken, super().child())
