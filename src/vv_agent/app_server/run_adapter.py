from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vv_agent.session.app_server import _KernelRunAdapter, _KernelThreadStore

from vv_agent.app_server.host import AppServerHost
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadRecord, TurnRecord
from vv_agent.run_handle import RunHandle


@dataclass(frozen=True, slots=True)
class StartedTurn:
    thread: ThreadRecord
    turn: TurnRecord
    handle: RunHandle
    is_durable_resume: bool = False


class TurnResumeError(ValueError):
    pass


class _RunAdapterFormatting:
    def __init__(
        self,
        *,
        host: AppServerHost,
        store: _KernelThreadStore,
        state_manager: ThreadStateManager,
        router: OutgoingRouter,
    ) -> None:
        self._host = host
        self._store = store
        self._state_manager = state_manager
        self._router = router

    @staticmethod
    def _validate_public_controller_action_identity(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise TurnResumeError(f"{field_name} must be a non-empty string")
        if len(value.encode("utf-8")) > 512:
            raise TurnResumeError(f"{field_name} exceeds the UTF-8 byte limit")

    @staticmethod
    def _validate_public_controller_action(action: Any) -> None:
        if not isinstance(action, dict):
            raise TurnResumeError("action must be an object")
        kind = action.get("kind")
        if kind not in {"respond", "suspend", "resume", "cancel", "abort"}:
            raise TurnResumeError("action kind is unsupported")
        from vv_agent.app_server.protocol.turn import TurnActionParams

        try:
            TurnActionParams("thread", "turn", "action", action)
        except (ValueError, TypeError) as exc:
            raise TurnResumeError(str(exc)) from exc

    @staticmethod
    def _result_error_text(error: dict[str, Any] | None, wait_reason: str | None = None) -> str:
        value = error.get("message") or error.get("code") if error is not None else None
        if isinstance(value, str) and value:
            return value
        return wait_reason or "Turn failed"

    def _prompt_from_input(self, input: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in input:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    def _notify_subscribers(
        self,
        thread_id: str,
        method: str,
        params: dict[str, Any],
        *,
        exclude: set[str] | None = None,
    ) -> None:
        excluded = exclude or set()
        for subscriber in self._state_manager.subscribers(thread_id):
            if subscriber not in excluded:
                self._router.send_notification(subscriber, method, params)


class RunAdapter:
    def __new__(cls, *, host, store, state_manager, router) -> _KernelRunAdapter:
        from vv_agent.session.app_server import _KernelRunAdapter

        return _KernelRunAdapter(kernel=store.kernel, host=host, store=store, state_manager=state_manager, router=router)
