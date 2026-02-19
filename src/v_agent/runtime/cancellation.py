from __future__ import annotations

import threading
from collections.abc import Callable


class CancelledError(Exception):
    """Raised when a cancelled CancellationToken is checked."""


class CancellationToken:
    """Thread-safe cancellation primitive with parent-child propagation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            cb()

    def check(self) -> None:
        if self._event.is_set():
            raise CancelledError("Operation was cancelled")

    def on_cancel(self, cb: Callable[[], None]) -> None:
        with self._lock:
            self._callbacks.append(cb)
        if self._event.is_set():
            cb()

    def child(self) -> CancellationToken:
        child_token = CancellationToken()
        self.on_cancel(child_token.cancel)
        return child_token
