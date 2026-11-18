"""Shared authorization boundary for session resource operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from vv_agent.kernel._content import Principal

type SessionResourceAction = Literal["export", "forget", "purge"]
type SessionResourceAuthorization = Callable[[Principal, str, SessionResourceAction], bool]


def authorize_session_resource_access(
    *,
    tenant_id: str,
    session_id: str,
    actor: Principal | None,
    authorize: SessionResourceAuthorization | None,
    action: SessionResourceAction,
) -> None:
    """Enforce the trusted-service and authorized-user session access boundaries."""

    if actor is None:
        if action == "export" and authorize is None:
            return
        raise PermissionError(f"session {action} access requires an authenticated Principal")
    if not isinstance(actor, Principal):
        raise PermissionError(f"session {action} access requires a Principal")
    if actor.tenant_id != tenant_id:
        raise PermissionError(f"session {action} actor belongs to another tenant")
    if actor.subject_kind not in {"user", "service", "agent"}:
        raise PermissionError(f"session {action} actor has an invalid subject kind")

    if authorize is None:
        if actor.subject_kind == "user":
            raise PermissionError(f"session {action} user access requires an authorization hook")
        return
    if not callable(authorize):
        raise PermissionError(f"session {action} authorization hook is invalid")
    if authorize(actor, session_id, action) is not True:
        raise PermissionError(f"session {action} is not authorized")
