from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssignedSubTaskIdentity:
    task_id: str
    session_id: str


def normalize_identity_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


_ASSIGNED_SUB_TASK_IDENTITY: ContextVar[AssignedSubTaskIdentity | None] = ContextVar(
    "vv_agent_assigned_sub_task_identity",
    default=None,
)


@contextmanager
def assigned_sub_task_identity(task_id: str, session_id: str) -> Iterator[None]:
    token = _ASSIGNED_SUB_TASK_IDENTITY.set(
        AssignedSubTaskIdentity(task_id=task_id, session_id=session_id)
    )
    try:
        yield
    finally:
        _ASSIGNED_SUB_TASK_IDENTITY.reset(token)


def take_sub_task_identity() -> AssignedSubTaskIdentity | None:
    identity = _ASSIGNED_SUB_TASK_IDENTITY.get()
    _ASSIGNED_SUB_TASK_IDENTITY.set(None)
    return identity
