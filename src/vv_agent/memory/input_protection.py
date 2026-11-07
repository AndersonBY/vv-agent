"""Shared replacement guard for the exact suffix selected by the session log."""

from vv_agent.types import Message


class UnseenInputError(ValueError):
    code = "unseen_input_removed"


class UnseenInputCapacityError(ValueError):
    code = "unseen_input_capacity_exceeded"


def atomic_start(messages: list[Message], start: int) -> int:
    for index, message in enumerate(messages):
        if message.tool_calls and index < start <= index + len(message.tool_calls):
            return index
    return start


def retained_start(source: list[Message], replacement: list[Message], start: int | None) -> int | None:
    if start is None or start >= len(source):
        return None
    start = atomic_start(source, start)
    size = len(source) - start
    for index in range(len(replacement) - size + 1):
        if replacement[index] == source[start] and all(
            replacement[index + offset] == source[start + offset] for offset in range(1, size)
        ):
            return index
    raise UnseenInputError("unseen_input_removed: replacement must retain the protected suffix verbatim")
