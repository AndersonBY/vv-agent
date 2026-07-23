from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from vv_agent.app_server.host import AgentResolutionRequest, AppServerApprovalProvider, AppServerHost, RunConfigResolutionRequest
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import (
    CheckpointSummary,
    InterruptionSummary,
    RequestId,
    TurnResumeResponse,
    WarningParams,
)
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadRecord, ThreadStore, TurnRecord
from vv_agent.app_server.usage_projection import task_token_usage_to_wire
from vv_agent.checkpoint import CheckpointError, ResumeObservation, ResumePolicy
from vv_agent.result import RunResult
from vv_agent.run_handle import RunHandle
from vv_agent.runner import Runner
from vv_agent.runtime.state import Checkpoint
from vv_agent.types import AgentStatus, Message


@dataclass(frozen=True, slots=True)
class StartedTurn:
    thread: ThreadRecord
    turn: TurnRecord
    handle: RunHandle
    checkpoint_store: Any | None = None
    checkpoint_key: str | None = None
    is_durable_resume: bool = False


class TurnResumeError(ValueError):
    pass


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
        metadata: dict[str, Any] | None = None,
        request_id: RequestId | None = None,
    ) -> StartedTurn:
        thread = self._store.read_thread(thread_id).thread
        turn_metadata = dict(metadata or {})
        effective_metadata = {**thread.metadata, **turn_metadata}
        agent_request = AgentResolutionRequest(
            thread_id=thread.thread_id,
            agent_key=thread.agent_key,
            cwd=thread.cwd,
            metadata=effective_metadata,
        )
        config_request = RunConfigResolutionRequest(
            thread_id=thread.thread_id,
            agent_key=thread.agent_key,
            cwd=thread.cwd,
            metadata=effective_metadata,
        )
        agent = self._host.resolve_agent(agent_request)
        run_config = self._host.build_run_config(config_request)
        run_config = replace(
            run_config,
            metadata={**run_config.metadata, **effective_metadata},
        )
        turn = self._store.create_turn(thread_id=thread.thread_id, input=input, status="running")
        run_config = self._with_app_server_controls(
            run_config,
            connection_id=connection_id,
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
        )
        handle = Runner.start(agent, self._prompt_from_input(input), run_config=run_config)
        checkpoint_config = run_config.checkpoint_config
        self._state_manager.set_active_turn(
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
            handle=handle,
            checkpoint_key=checkpoint_config.key if checkpoint_config is not None else None,
        )
        self._state_manager.set_status(thread.thread_id, "running")
        started = StartedTurn(
            thread=thread,
            turn=turn,
            handle=handle,
            checkpoint_store=(checkpoint_config.store if checkpoint_config is not None else None),
            checkpoint_key=(checkpoint_config.key if checkpoint_config is not None else None),
        )
        if request_id is not None:
            self._router.send_response(
                connection_id,
                request_id,
                {"threadId": thread.thread_id, "turnId": turn.turn_id, "status": "running"},
            )
        self._notify_subscribers(
            thread.thread_id,
            "thread/status/changed",
            {"threadId": thread.thread_id, "status": "running"},
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

    def resume_turn(
        self,
        *,
        connection_id: str,
        thread_id: str,
        turn_id: str,
        checkpoint_key: str,
        request_id: RequestId,
    ) -> StartedTurn | None:
        snapshot = self._store.read_thread(thread_id)
        if snapshot.thread.archived_at is not None:
            raise TurnResumeError("Cannot resume a turn in an archived thread")
        turn = next((candidate for candidate in snapshot.turns if candidate.turn_id == turn_id), None)
        if turn is None:
            raise TurnResumeError("Turn does not belong to the requested thread")

        effective_metadata = dict(snapshot.thread.metadata)
        agent = self._host.resolve_agent(
            AgentResolutionRequest(
                thread_id=thread_id,
                agent_key=snapshot.thread.agent_key,
                cwd=snapshot.thread.cwd,
                metadata=effective_metadata,
            )
        )
        run_config = self._host.build_run_config(
            RunConfigResolutionRequest(
                thread_id=thread_id,
                agent_key=snapshot.thread.agent_key,
                cwd=snapshot.thread.cwd,
                metadata=effective_metadata,
            )
        )
        run_config = replace(run_config, metadata={**run_config.metadata, **effective_metadata})
        run_config = self._with_app_server_controls(
            run_config,
            connection_id=connection_id,
            thread_id=thread_id,
            turn_id=turn_id,
        )
        checkpoint_config = run_config.checkpoint_config
        if checkpoint_config is None or checkpoint_config.store is None:
            raise TurnResumeError("turn/resume requires a process-local checkpoint v2 store")
        checkpoint_config = replace(
            checkpoint_config,
            key=checkpoint_key,
            resume_policy=ResumePolicy.REQUIRE_EXISTING,
        )
        run_config = replace(run_config, checkpoint_config=checkpoint_config)
        checkpoint = checkpoint_config.store.load_checkpoint(checkpoint_key)
        if checkpoint is None:
            raise TurnResumeError("Checkpoint does not exist")
        self._validate_checkpoint_binding(checkpoint, thread=snapshot.thread, turn=turn)

        active = self._state_manager.active_turn(thread_id)
        if active is not None:
            if active.turn_id != turn_id or active.checkpoint_key != checkpoint_key:
                raise TurnResumeError("Thread has a different active turn")
            self._router.send_response(
                connection_id,
                request_id,
                TurnResumeResponse(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    run_id=active.run_id or checkpoint.root_run_id,
                    status="running",
                    checkpoint=(self._checkpoint_summary(checkpoint) if self._has_live_claim(checkpoint) else None),
                ).to_dict(),
            )
            return None

        if self._has_live_claim(checkpoint):
            self._store.resume_turn(turn_id, run_id=checkpoint.root_run_id)
            self._store.set_active_turn(thread_id, turn_id, "running")
            self._state_manager.set_status(thread_id, "running")
            self._router.send_response(
                connection_id,
                request_id,
                TurnResumeResponse(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    run_id=checkpoint.root_run_id,
                    status="running",
                    checkpoint=self._checkpoint_summary(checkpoint),
                ).to_dict(),
            )
            return None

        prompt = self._prompt_from_input(turn.input)
        if checkpoint.terminal_result is not None:
            result = Runner.run_sync(agent, prompt, run_config=run_config)
            retained = checkpoint_config.store.load_checkpoint(checkpoint_key)
            if retained is None or retained.terminal_result is None:
                raise TurnResumeError("Terminal checkpoint disappeared during replay")
            response = self._resume_response(
                thread_id=thread_id,
                turn_id=turn_id,
                result=result,
                checkpoint=retained,
            )
            self._store.update_turn(
                turn_id,
                status=response.status,
                run_id=result.run_id,
                completed_at=turn.completed_at,
                result=self._stored_result_from_response(response),
            )
            self._store.set_active_turn(thread_id, None, "idle")
            self._state_manager.set_status(thread_id, "idle")
            self._router.send_response(connection_id, request_id, response.to_dict())
            return None

        resumed_turn = self._store.resume_turn(turn_id, run_id=checkpoint.root_run_id)
        handle = Runner.start(agent, prompt, run_config=run_config)
        self._state_manager.set_active_turn(
            thread_id=thread_id,
            turn_id=turn_id,
            handle=handle,
            checkpoint_key=checkpoint_key,
            run_id=checkpoint.root_run_id,
        )
        self._state_manager.set_status(thread_id, "running")
        self._state_manager.subscribe(thread_id, connection_id)
        started = StartedTurn(
            thread=snapshot.thread,
            turn=resumed_turn,
            handle=handle,
            checkpoint_store=checkpoint_config.store,
            checkpoint_key=checkpoint_key,
            is_durable_resume=True,
        )
        self._router.send_response(
            connection_id,
            request_id,
            TurnResumeResponse(
                thread_id=thread_id,
                turn_id=turn_id,
                run_id=checkpoint.root_run_id,
                status="running",
            ).to_dict(),
        )
        self._notify_subscribers(
            thread_id,
            "thread/status/changed",
            {"threadId": thread_id, "status": "running"},
        )
        self._notify_subscribers(
            thread_id,
            "turn/started",
            {
                "threadId": thread_id,
                "turnId": turn_id,
                "runId": checkpoint.root_run_id,
                "status": "running",
            },
        )
        threading.Thread(target=self._pump_events, args=(connection_id, started), daemon=True).start()
        return started

    def _pump_events(self, connection_id: str, started: StartedTurn) -> None:
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            for event in started.handle.events():
                if started.is_durable_resume and event.type == "run_started":
                    continue
                projection = map_run_event(event, thread_id=started.thread.thread_id, turn_id=started.turn.turn_id)
                should_notify = True
                if projection.item is not None:
                    should_notify = self._store.append_item(
                        projection.item,
                        run_event_id=event.event_id,
                    )
                if should_notify and projection.notification_method is not None:
                    self._notify_subscribers(
                        started.thread.thread_id,
                        projection.notification_method,
                        projection.notification_params,
                    )
                if should_notify:
                    for method, params in projection.additional_notifications:
                        self._notify_subscribers(started.thread.thread_id, method, params)
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
            status = self._turn_status(result.status)
            payload: dict[str, Any] = {
                "threadId": started.thread.thread_id,
                "turnId": started.turn.turn_id,
                "runId": result.run_id,
                "status": status,
            }
            if result.status is not AgentStatus.RECONCILIATION_REQUIRED:
                payload["tokenUsage"] = task_token_usage_to_wire(result.token_usage)
            if result.budget_usage is not None and result.status is not AgentStatus.RECONCILIATION_REQUIRED:
                budget_usage = result.budget_usage.to_dict()
                payload["budgetUsage"] = budget_usage
            if result.budget_exhaustion is not None and result.status is not AgentStatus.RECONCILIATION_REQUIRED:
                budget_exhaustion = result.budget_exhaustion.to_dict()
                payload["budgetExhaustion"] = budget_exhaustion
            if result.completion_reason is not None:
                payload["completionReason"] = result.completion_reason.value
            if result.completion_tool_name is not None:
                payload["completionToolName"] = result.completion_tool_name
            if result.partial_output is not None:
                payload["partialOutput"] = result.partial_output
            if result.final_output is not None:
                payload["finalOutput"] = result.final_output
            checkpoint = self._load_result_checkpoint(started, result)
            if checkpoint is not None:
                payload["checkpoint"] = self._checkpoint_summary(checkpoint).to_dict()
            if result.resume_observation is not None:
                payload["interruption"] = self._interruption_summary(result.resume_observation).to_dict()
            if status == "failed":
                result_error = result.raw_result.error or result.raw_result.wait_reason or "Turn failed"
                payload["error"] = result_error
            self._store.update_turn(
                started.turn.turn_id,
                status=status,
                run_id=result.run_id,
                result=self._stored_result_from_payload(payload),
            )
        else:
            status = "failed"
            error_message = self._safe_run_error(error, durable_resume=started.is_durable_resume)
            payload = {
                "threadId": started.thread.thread_id,
                "turnId": started.turn.turn_id,
                "status": status,
                "completionReason": "failed",
                "error": error_message,
            }
            self._store.update_turn(
                started.turn.turn_id,
                status=status,
                result={"completionReason": "failed", "error": payload["error"]},
            )
            self._notify_subscribers(
                started.thread.thread_id,
                "error/warning",
                WarningParams(
                    message=error_message,
                    code="event_stream",
                ).to_dict(),
            )
        self._state_manager.clear_active_turn(started.thread.thread_id, started.turn.turn_id)
        self._state_manager.set_status(started.thread.thread_id, "idle")
        self._notify_subscribers(
            started.thread.thread_id,
            "thread/status/changed",
            {"threadId": started.thread.thread_id, "status": "idle"},
        )
        self._notify_subscribers(started.thread.thread_id, "turn/completed", payload)
        follow_up = self._state_manager.pop_next_follow_up(started.thread.thread_id)
        if follow_up is not None and status == "completed":
            self.start_turn(connection_id=connection_id, thread_id=started.thread.thread_id, input=follow_up)

    @staticmethod
    def _turn_status(status: AgentStatus) -> str:
        if status is AgentStatus.COMPLETED:
            return "completed"
        if status in {AgentStatus.WAIT_USER, AgentStatus.RECONCILIATION_REQUIRED}:
            return "interrupted"
        return "failed"

    @staticmethod
    def _has_live_claim(checkpoint: Checkpoint) -> bool:
        now_ms = time.time_ns() // 1_000_000
        return bool(
            checkpoint.claim_token is not None
            and checkpoint.lease_expires_at_ms is not None
            and checkpoint.lease_expires_at_ms > now_ms
        )

    def _validate_checkpoint_binding(
        self,
        checkpoint: Checkpoint,
        *,
        thread: ThreadRecord,
        turn: TurnRecord,
    ) -> None:
        run_definition = checkpoint.run_definition
        metadata = run_definition.get("run_metadata")
        expected_metadata = {
            "thread_id": thread.thread_id,
            "turn_id": turn.turn_id,
            "session_id": thread.thread_id,
        }
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise TurnResumeError("Checkpoint is not bound to the requested turn")
        if run_definition.get("root_input") != self._prompt_from_input(turn.input):
            raise TurnResumeError("Checkpoint input does not match the requested turn")

    @staticmethod
    def _checkpoint_summary(checkpoint: Checkpoint) -> CheckpointSummary:
        return CheckpointSummary(
            key=checkpoint.checkpoint_key,
            resume_attempt=checkpoint.resume_attempt,
            cycle_index=checkpoint.cycle_index,
            status=checkpoint.status.value,
            terminal_acknowledged=checkpoint.terminal_acknowledged,
        )

    @staticmethod
    def _interruption_summary(observation: ResumeObservation) -> InterruptionSummary:
        return InterruptionSummary(
            reason="resume_requires_reconciliation",
            operation_id=observation.operation_id,
            operation_kind=observation.operation_kind.value,
            cycle_index=observation.cycle_index,
            risk=observation.risk,
            idempotency_support=(observation.idempotency_support.value if observation.idempotency_support is not None else None),
        )

    def _resume_response(
        self,
        *,
        thread_id: str,
        turn_id: str,
        result: RunResult,
        checkpoint: Checkpoint,
    ) -> TurnResumeResponse:
        status = self._turn_status(result.status)
        error = None
        if status == "failed":
            error = result.raw_result.error or result.raw_result.wait_reason or "Turn failed"
        return TurnResumeResponse(
            thread_id=thread_id,
            turn_id=turn_id,
            run_id=result.run_id,
            status=status,
            final_output=result.final_output,
            completion_reason=(result.completion_reason.value if result.completion_reason is not None else None),
            completion_tool_name=result.completion_tool_name,
            partial_output=result.partial_output,
            checkpoint=self._checkpoint_summary(checkpoint),
            interruption=(
                self._interruption_summary(result.resume_observation) if result.resume_observation is not None else None
            ),
            error=error,
        )

    @staticmethod
    def _stored_result_from_response(response: TurnResumeResponse) -> dict[str, Any]:
        return RunAdapter._stored_result_from_payload(response.to_dict())

    @staticmethod
    def _stored_result_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
        envelope_fields = {"threadId", "turnId", "runId", "status"}
        return {name: value for name, value in payload.items() if name not in envelope_fields}

    @staticmethod
    def _safe_run_error(error: BaseException | None, *, durable_resume: bool) -> str:
        if not durable_resume:
            return str(error) if error is not None else "Turn failed"
        if isinstance(error, CheckpointError):
            return f"Checkpoint resume failed ({error.code})"
        return "Checkpoint resume failed"

    @staticmethod
    def _load_result_checkpoint(started: StartedTurn, result: RunResult) -> Checkpoint | None:
        checkpoint_key = result.checkpoint_key or started.checkpoint_key
        if started.checkpoint_store is None or checkpoint_key is None:
            return None
        return started.checkpoint_store.load_checkpoint(checkpoint_key)

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
            metadata={
                **run_config.metadata,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "session_id": thread_id,
            },
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
