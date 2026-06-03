from __future__ import annotations

import threading

import pytest

from vv_agent.app_server.request_serialization import RequestQueueOverloaded, RequestScope, RequestSerializationQueues


def test_same_key_exclusive_requests_run_fifo() -> None:
    queues = RequestSerializationQueues()
    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def first() -> str:
        order.append("first:start")
        first_started.set()
        assert release_first.wait(timeout=1)
        order.append("first:end")
        return "first"

    def second() -> str:
        order.append("second")
        return "second"

    first_future = queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=first)
    assert first_started.wait(timeout=1)
    second_future = queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=second)

    assert order == ["first:start"]
    release_first.set()
    assert first_future.result(timeout=1) == "first"
    assert second_future.result(timeout=1) == "second"
    assert order == ["first:start", "first:end", "second"]


def test_consecutive_shared_reads_for_same_key_can_run_together() -> None:
    queues = RequestSerializationQueues()
    first_started = threading.Event()
    second_started = threading.Event()
    release_reads = threading.Event()

    def first() -> str:
        first_started.set()
        assert release_reads.wait(timeout=1)
        return "first"

    def second() -> str:
        second_started.set()
        assert release_reads.wait(timeout=1)
        return "second"

    first_future = queues.enqueue(key=RequestScope.thread_read("thread_1"), access="shared_read", fn=first)
    assert first_started.wait(timeout=1)
    second_future = queues.enqueue(key=RequestScope.thread_read("thread_1"), access="shared_read", fn=second)

    assert second_started.wait(timeout=1)
    release_reads.set()
    assert first_future.result(timeout=1) == "first"
    assert second_future.result(timeout=1) == "second"


def test_exclusive_request_waits_for_preceding_shared_reads() -> None:
    queues = RequestSerializationQueues()
    read_started = threading.Event()
    release_read = threading.Event()
    exclusive_started = threading.Event()

    def read() -> str:
        read_started.set()
        assert release_read.wait(timeout=1)
        return "read"

    def exclusive() -> str:
        exclusive_started.set()
        return "exclusive"

    read_future = queues.enqueue(key=RequestScope.thread_read("thread_1"), access="shared_read", fn=read)
    assert read_started.wait(timeout=1)
    exclusive_future = queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=exclusive)

    assert not exclusive_started.wait(timeout=0.05)
    release_read.set()
    assert read_future.result(timeout=1) == "read"
    assert exclusive_future.result(timeout=1) == "exclusive"
    assert exclusive_started.is_set()


def test_following_shared_read_waits_behind_exclusive_request() -> None:
    queues = RequestSerializationQueues()
    exclusive_started = threading.Event()
    release_exclusive = threading.Event()
    read_started = threading.Event()

    def exclusive() -> str:
        exclusive_started.set()
        assert release_exclusive.wait(timeout=1)
        return "exclusive"

    def read() -> str:
        read_started.set()
        return "read"

    exclusive_future = queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=exclusive)
    assert exclusive_started.wait(timeout=1)
    read_future = queues.enqueue(key=RequestScope.thread_read("thread_1"), access="shared_read", fn=read)

    assert not read_started.wait(timeout=0.05)
    release_exclusive.set()
    assert exclusive_future.result(timeout=1) == "exclusive"
    assert read_future.result(timeout=1) == "read"
    assert read_started.is_set()


def test_queue_limit_rejects_excess_work() -> None:
    queues = RequestSerializationQueues(max_queued_per_scope=1)
    first_started = threading.Event()
    release_first = threading.Event()

    def blocking_work() -> str:
        first_started.set()
        assert release_first.wait(timeout=2)
        return "first"

    first = queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=blocking_work)
    assert first_started.wait(timeout=1)

    second = queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=lambda: "second")
    with pytest.raises(RequestQueueOverloaded):
        queues.enqueue(key=RequestScope.thread("thread_1"), access="exclusive", fn=lambda: "third")

    release_first.set()
    assert first.result(timeout=1) == "first"
    assert second.result(timeout=1) == "second"
