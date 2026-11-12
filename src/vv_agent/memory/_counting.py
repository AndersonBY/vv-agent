"""Resolve count requests outside suspended core generators."""

from collections.abc import Callable, Generator

from vv_agent.kernel._token_count import TokenCount, _StopIterationTransport


def _drain_counts[T](steps: Generator[TokenCount, int, T], counter: Callable[[TokenCount], int]) -> T:
    count = None
    try:
        while True:
            try:
                request = next(steps) if count is None else steps.send(count)
            except StopIteration as done:
                return done.value
            except _StopIterationTransport as transport:
                raise transport.original from transport.original.__cause__
            count = counter(request)
    finally:
        steps.close()
