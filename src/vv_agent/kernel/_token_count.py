"""A suspended compaction requests a count from its caller, without invoking it."""

from dataclasses import dataclass

from vv_agent.kernel._message_values import Message


@dataclass(frozen=True, slots=True)
class TokenCount:
    messages: Message | list[Message]
    model: str = ""


class _StopIterationTransport(Exception):
    """Carry a raw synchronous StopIteration across a core generator boundary."""

    def __init__(self, original: StopIteration) -> None:
        super().__init__()
        self.original = original
