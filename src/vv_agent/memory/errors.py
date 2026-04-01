from __future__ import annotations


class CompactionExhaustedError(RuntimeError):
    """Raised when repeated PTL-triggered compaction retries still cannot fit the prompt."""

    def __init__(self, attempts: int, last_error: Exception | None = None) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Context compaction failed after {attempts} consecutive attempts. "
            f"Last error: {last_error}"
        )
