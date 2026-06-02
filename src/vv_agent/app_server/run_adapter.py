from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any

from vv_agent.app_server.host import AgentResolutionRequest, AppServerApprovalProvider, AppServerHost, RunConfigResolutionRequest
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import RequestId
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadRecord, ThreadStore, TurnRecord
from vv_agent.result import RunResult
from vv_agent.run_handle import RunHandle
from vv_agent.runner import Runner
from vv_agent.types import Message


@dataclass(frozen=True, slots=True)
class StartedTurn:
    thread: ThreadRecord
    turn: TurnRecord
    handle: RunHandle


class RunAdapter:
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

    def start_turn(
        self,
        *,
        connection_id: str,
        thread_id: str,
        input: list[dict[str, Any]],
        request_id: RequestId | None = None,
    ) -> StartedTurn:
        thread = self._store.read_thread(thread_id).thread
        agent_request = AgentResolutionRequest(
            thread_id=thread.thread_id,
            agent_key=thread.agent_key,
            cwd=thread.cwd,
            metadata=thread.metadata,
        )
        config_request = RunConfigResolutionRequest(
            thread_id=thread.thread_id,
            agent_key=thread.agent_key,
            cwd=thread.cwd,
            metadata=thread.metadata,
        )
        agent = self._host.resolve_agent(agent_request)
        run_config = self._host.build_run_config(config_request)
        turn = self._store.create_turn(thread_id=thread.thread_id, input=input, status="running")
        run_config = self._with_app_server_controls(
            run_config,
            connection_id=connection_id,
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
        )
        handle = Runner.start(agent, self._prompt_from_input(input), run_config=run_config)
        self._state_manager.set_active_turn(thread_id=thread.thread_id, turn_id=turn.turn_id, handle=handle)
        started = StartedTurn(thread=thread, turn=turn, handle=handle)
        if request_id is not None:
            self._router.send_response(
                connection_id,
                request_id,
                {"threadId": thread.thread_id, "turnId": turn.turn_id, "status": "running"},
            )
        self._router.send_notification(
            connection_id,
            "turn/started",
            {"threadId": thread.thread_id, "turnId": turn.turn_id},
        )
        self._notify_subscribers(
            thread.thread_id,
            "turn/started",
            {"threadId": thread.thread_id, "turnId": turn.turn_id},
            exclude={connection_id},
        )
        threading.Thread(target=self._pump_events, args=(connection_id, started), daemon=True).start()
        return started

    def _pump_events(self, connection_id: str, started: StartedTurn) -> None:
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            for event in started.handle.events():
                projection = map_run_event(event, thread_id=started.thread.thread_id, turn_id=started.turn.turn_id)
                if projection.item is not None:
                    self._store.append_item(projection.item, run_event_id=event.event_id)
                if projection.notification_method is not None:
                    self._notify_subscribers(
                        started.thread.thread_id,
                        projection.notification_method,
                        projection.notification_params,
                    )
            result = started.handle.result(timeout=0)
        except BaseException as exc:
            error = exc
        finally:
            self._complete_turn(connection_id, started, result=result, error=error)

    def _complete_turn(
        self,
        connection_id: str,
        started: StartedTurn,
        *,
        result: RunResult | None,
        error: BaseException | None,
    ) -> None:
        if result is not None:
            status = result.status.value
            token_usage = result.token_usage.to_dict()
            payload: dict[str, Any] = {
                "threadId": started.thread.thread_id,
                "turnId": started.turn.turn_id,
                "runId": result.run_id,
                "status": status,
                "finalOutput": result.final_output,
                "tokenUsage": token_usage,
            }
            self._store.update_turn(
                started.turn.turn_id,
                status=status,
                run_id=result.run_id,
                result={"finalOutput": result.final_output, "tokenUsage": token_usage},
            )
        else:
            status = "failed"
            payload = {
                "threadId": started.thread.thread_id,
                "turnId": started.turn.turn_id,
                "status": status,
                "error": str(error) if error is not None else "Turn failed",
            }
            self._store.update_turn(started.turn.turn_id, status=status, result={"error": payload["error"]})
        self._state_manager.clear_active_turn(started.thread.thread_id, started.turn.turn_id)
        self._notify_subscribers(started.thread.thread_id, "turn/completed", payload)
        follow_up = self._state_manager.pop_next_follow_up(started.thread.thread_id)
        if follow_up is not None and status == "completed":
            self.start_turn(connection_id=connection_id, thread_id=started.thread.thread_id, input=follow_up)

    def _prompt_from_input(self, input: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in input:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    def _with_app_server_controls(self, run_config, *, connection_id: str, thread_id: str, turn_id: str):
        existing_before_cycle = run_config.before_cycle_messages

        def before_cycle_messages(cycle_index: int, messages: list[Message], shared_state: dict[str, Any]) -> list[Message]:
            injected: list[Message] = []
            if existing_before_cycle is not None:
                injected.extend(existing_before_cycle(cycle_index, messages, shared_state))
            for queued_input in self._state_manager.drain_steering(thread_id):
                injected.extend(self._messages_from_input(queued_input))
            return injected

        return replace(
            run_config,
            metadata={**run_config.metadata, "session_id": thread_id},
            before_cycle_messages=before_cycle_messages,
            approval_provider=run_config.approval_provider
            or AppServerApprovalProvider(
                connection_id=connection_id,
                thread_id=thread_id,
                turn_id=turn_id,
                router=self._router,
                timeout_seconds=run_config.approval_timeout_seconds,
            ),
        )

    def _messages_from_input(self, input: list[dict[str, Any]]) -> list[Message]:
        prompt = self._prompt_from_input(input)
        if not prompt:
            return []
        return [Message(role="user", content=prompt)]

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
