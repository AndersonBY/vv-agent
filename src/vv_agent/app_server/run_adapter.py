from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vv_agent.session.app_server import _KernelRunAdapter

from vv_agent.app_server.host import AppServerApprovalProvider, AppServerHost
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadRecord, ThreadStore, TurnRecord
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
        store: ThreadStore,
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
        expected = {"kind", "message"} if kind == "respond" else {"kind"}
        actual = set(action)
        if actual != expected:
            missing = expected - actual
            unknown = actual - expected
            raise TurnResumeError(
                f"action fields do not match the public schema: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if kind != "respond":
            return
        message = action["message"]
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise TurnResumeError("respond message must contain exactly role and content")
        if message["role"] != "user":
            raise TurnResumeError("respond message role must be user")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise TurnResumeError("respond message content must be a non-empty string")
        if len(content.encode("utf-8")) > 65536:
            raise TurnResumeError("respond message content exceeds the UTF-8 byte limit")

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

    def _with_app_server_controls(self, run_config, *, connection_id: str, thread_id: str, turn_id: str):
        return replace(
            run_config,
            metadata={
                **run_config.metadata,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "session_id": thread_id,
            },
            approval_provider=run_config.approval_provider
            or AppServerApprovalProvider(
                connection_id=connection_id,
                thread_id=thread_id,
                turn_id=turn_id,
                router=self._router,
                timeout_seconds=run_config.approval_timeout_seconds,
            ),
        )

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
