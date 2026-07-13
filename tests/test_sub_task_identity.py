from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vv_agent.runtime.sub_task_identity import assigned_sub_task_identity, take_sub_task_identity


def _take_pair() -> tuple[str, str] | None:
    identity = take_sub_task_identity()
    if identity is None:
        return None
    return identity.task_id, identity.session_id


def test_assigned_sub_task_identity_is_one_shot_and_cleans_up_after_scope() -> None:
    assert _take_pair() is None
    with assigned_sub_task_identity("task-one", "session-one"):
        assert _take_pair() == ("task-one", "session-one")
        assert _take_pair() is None
    assert _take_pair() is None


def test_assigned_sub_task_identity_nesting_restores_only_unconsumed_outer_value() -> None:
    with assigned_sub_task_identity("outer-task", "outer-session"):
        with assigned_sub_task_identity("inner-task", "inner-session"):
            assert _take_pair() == ("inner-task", "inner-session")
        assert _take_pair() == ("outer-task", "outer-session")

    with assigned_sub_task_identity("consumed-task", "consumed-session"):
        assert _take_pair() == ("consumed-task", "consumed-session")
        with assigned_sub_task_identity("inner-task", "inner-session"):
            assert _take_pair() == ("inner-task", "inner-session")
        assert _take_pair() is None
    assert _take_pair() is None


def test_assigned_sub_task_identity_cleans_up_after_exception_and_restores_outer_scope() -> None:
    with pytest.raises(RuntimeError, match="scope failed"), assigned_sub_task_identity(
        "failed-task", "failed-session"
    ):
        raise RuntimeError("scope failed")
    assert _take_pair() is None

    with assigned_sub_task_identity("outer-task", "outer-session"):
        with pytest.raises(RuntimeError, match="inner failed"), assigned_sub_task_identity(
            "inner-task", "inner-session"
        ):
            raise RuntimeError("inner failed")
        assert _take_pair() == ("outer-task", "outer-session")
    assert _take_pair() is None


def test_assigned_sub_task_identity_is_isolated_across_async_tasks_and_threads() -> None:
    async def async_worker(index: int) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
        with assigned_sub_task_identity(f"async-task-{index}", f"async-session-{index}"):
            await asyncio.sleep(0)
            first = _take_pair()
            await asyncio.sleep(0)
            return first, _take_pair()

    async def run_async_workers() -> list[tuple[tuple[str, str] | None, tuple[str, str] | None]]:
        return list(await asyncio.gather(*(async_worker(index) for index in range(3))))

    assert asyncio.run(run_async_workers()) == [
        ((f"async-task-{index}", f"async-session-{index}"), None)
        for index in range(3)
    ]

    barrier = threading.Barrier(2)

    def thread_worker(index: int) -> tuple[tuple[str, str] | None, tuple[str, str] | None, bool]:
        with assigned_sub_task_identity(f"thread-task-{index}", f"thread-session-{index}"):
            barrier.wait(timeout=2)
            first = _take_pair()
            second = _take_pair()
        return first, second, _take_pair() is None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(thread_worker, range(2)))

    assert results == [
        ((f"thread-task-{index}", f"thread-session-{index}"), None, True)
        for index in range(2)
    ]
    assert _take_pair() is None
