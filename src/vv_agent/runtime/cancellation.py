from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import suppress


class CancelledError(Exception):
    """Raised when a cancelled CancellationToken is checked."""


class CancellationToken:
    """Thread-safe cancellation primitive with parent-child propagation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "Operation was cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason.strip() or "Operation was cancelled"
            self._event.set()
            callbacks = self._callbacks
            self._callbacks = []
        for cb in callbacks:
            self._invoke_callback(cb)

    def check(self) -> None:
        if self._event.is_set():
            raise CancelledError(self.reason or "Operation was cancelled")

    def on_cancel(self, cb: Callable[[], None]) -> None:
        with self._lock:
            call_immediately = self._event.is_set()
            if not call_immediately:
                self._callbacks.append(cb)
        if call_immediately:
            self._invoke_callback(cb)

    def child(self) -> CancellationToken:
        child_token = CancellationToken()

        def cancel_child() -> None:
            child_token.cancel(self.reason or "Operation was cancelled")

        self.on_cancel(cancel_child)
        return child_token

    @staticmethod
    def _invoke_callback(callback: Callable[[], None]) -> None:
        with suppress(BaseException):
            callback()
