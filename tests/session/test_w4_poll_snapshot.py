"""W4 poll reads use complete keeper snapshots and token-safe turn binding."""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, get_ident
from time import monotonic, sleep

import pytest

from vv_agent.kernel import _driver_dispatch as dispatch_core
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel import _driver_recovery as core
from vv_agent.kernel._sql import SQLStore
from vv_agent.kernel._store import LeasePoll
from vv_agent.session import kernel
from vv_agent.session._driver_dispatch import _DispatchHost
from vv_agent.session._driver_scope import _LeaseHeartbeat, _Scope
from vv_agent.session._recovery_phase import _Host
from vv_agent.session.timing import ThreadInvoker
from vv_agent.types import LLMResponse, ToolCall

from .helpers import fixture_content_reader, item, push_fixture_input, read_fixture_state, wrap_fixture_store
from .test_recovery_matrix import effect_tool, runtime, start

pytestmark = pytest.mark.persistent_store


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


@dataclass(frozen=True)
class _Bind:
    turn_id: str | None
    generation: int | None
    token_id: int
    cancelled: bool
    reason: str | None
    applied_ids: tuple[str, ...]
    direct_state_mapping: bool


@dataclass(frozen=True)
class _Reset:
    before_id: int
    after_id: int
    cancelled: bool
    reason: str | None


class _ObservedInvoker:
    def __init__(self) -> None:
        self.thread = ThreadInvoker()
        self.tokens = []
        self.outcomes = []

    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        self.tokens.append(token)
        outcome = self.thread.invoke(
            callback,
            timeout=timeout,
            token=token,
            cancellation_grace=cancellation_grace,
            clock=clock,
        )
        self.outcomes.append(outcome)
        return outcome


def _wait_until(predicate, *, timeout: float = 10) -> None:
    deadline = monotonic() + timeout
    while not predicate():
        assert monotonic() < deadline, "condition did not become true"
        sleep(0.01)


def test_factory_handoff_controls_bind_new_token_without_driver_renewal(store, database, monkeypatch) -> None:
    start(store)
    old_turn = "s/turn/old"
    new_turn = "s/turn/initial"
    stale_id = "stale-suspend"
    consumed_id = "consumed-suspend"
    push_fixture_input(
        store,
        "s",
        item(stale_id, "control", action="suspend", target_turn_id=old_turn, generation=7),
    )
    push_fixture_input(
        store,
        "s",
        item(consumed_id, "control", action="suspend", target_turn_id=new_turn, generation=0),
    )
    control_ids = {stale_id, consumed_id}

    content_reader = fixture_content_reader(store, "s")
    root = store._root
    driver_connection = id(root.connection)
    renew = SQLStore.renew
    renewals: list[tuple[int, int, str]] = []
    main_thread = get_ident()
    driver_renewal_stacks: list[str] = []
    connections: list[tuple[int, int]] = []
    active_connections = 0
    max_active_connections = 0

    def observed_renew(self, lease, *, ttl_ms):
        thread = get_ident()
        renewals.append((id(self.connection), thread, lease.owner))
        if thread == main_thread:
            driver_renewal_stacks.append("".join(traceback.format_stack()))
        return renew(self, lease, ttl_ms=ttl_ms)

    heartbeat_factory = root.heartbeat_store_factory

    def observed_factory():
        renewal_factory = heartbeat_factory()

        @contextmanager
        def observed():
            nonlocal active_connections, max_active_connections
            active_connections += 1
            max_active_connections = max(max_active_connections, active_connections)
            try:
                with renewal_factory() as heartbeat_store:
                    connections.append((id(heartbeat_store.connection), get_ident()))
                    yield heartbeat_store
            finally:
                active_connections -= 1

        return observed

    confirms: list[LeasePoll] = []
    heartbeat_confirmed = _LeaseHeartbeat.heartbeat_confirmed

    def observed_heartbeat_confirmed(self, poll: LeasePoll) -> None:
        heartbeat_confirmed(self, poll)
        confirms.append(poll)

    adopted: list[LeasePoll] = []
    adopt_confirmed = _Scope.adopt_confirmed

    def observed_adopt(self, poll: LeasePoll, *, expires_deadline: float) -> None:
        adopted.append(poll)
        adopt_confirmed(self, poll, expires_deadline=expires_deadline)

    binds: list[_Bind] = []
    bind_turn = kernel._Driver.bind_turn

    def observed_bind_turn(self) -> None:
        bind_turn(self)
        binding = self.scope.control_binding
        assert binding is not None
        state = self.fold.state if self._logical_dirty else self.state
        binds.append(
            _Bind(
                binding.turn_id,
                binding.generation,
                id(binding.token),
                binding.token.cancelled,
                binding.token.reason,
                tuple(binding.applied_inputs),
                binding.applied_inputs is state.applied_inputs,
            )
        )

    resets: list[_Reset] = []
    reset_token = _Scope.reset_token

    def observed_reset_token(self) -> None:
        before = self.control_binding
        assert before is not None
        reset_token(self)
        after = self.control_binding
        assert after is not None
        resets.append(_Reset(id(before.token), id(after.token), after.token.cancelled, after.token.reason))

    observed_invoker = _ObservedInvoker()
    model_renewals = Event()
    model_renewal_window: list[tuple[int, int, str]] = []
    model_errors: list[str] = []
    heartbeat_attempts: list[tuple[int, int | None, int]] = []
    heartbeat_failures: list[tuple[int, str]] = []
    heartbeat_finished: list[int] = []
    heartbeat_once = _LeaseHeartbeat.heartbeat_once
    heartbeat_failed = _LeaseHeartbeat.heartbeat_failed
    heartbeat_observed_finished = _LeaseHeartbeat.heartbeat_finished

    def observed_heartbeat_once(self, heartbeat_store):
        backend = heartbeat_store._root if hasattr(heartbeat_store, "_root") else heartbeat_store
        connection = getattr(backend, "connection", None)
        heartbeat_attempts.append((id(self), None if connection is None else id(connection), get_ident()))
        heartbeat_once(self, heartbeat_store)

    def observed_heartbeat_failed(self, error: Exception) -> None:
        heartbeat_failures.append((id(self), f"{type(error).__name__}: {error}"))
        heartbeat_failed(self, error)

    def observed_heartbeat_finished(self) -> None:
        heartbeat_finished.append(id(self))
        heartbeat_observed_finished(self)

    def long_model(_request):
        before = len(renewals)
        try:
            _wait_until(lambda: len(renewals) >= before + 2)
            model_renewal_window.extend(renewals[before:])
            model_renewals.set()
            return LLMResponse("done")
        except Exception as exc:
            model_errors.append(f"{type(exc).__name__}: {exc}")
            return LLMResponse("renewal-not-observed")

    def make_runtime(session_id: str):
        assert session_id == "s"
        selected = runtime(
            database,
            [long_model],
            content_reader=content_reader,
            ttl_ms=3_000,
            heartbeat_seconds=0.04,
        )
        selected.heartbeat_store = store.heartbeat_store
        selected.invoker = observed_invoker
        return selected

    monkeypatch.setattr(SQLStore, "renew", observed_renew)
    monkeypatch.setattr(root, "heartbeat_store_factory", observed_factory)
    monkeypatch.setattr(_LeaseHeartbeat, "heartbeat_once", observed_heartbeat_once)
    monkeypatch.setattr(_LeaseHeartbeat, "heartbeat_failed", observed_heartbeat_failed)
    monkeypatch.setattr(_LeaseHeartbeat, "heartbeat_finished", observed_heartbeat_finished)
    monkeypatch.setattr(_LeaseHeartbeat, "heartbeat_confirmed", observed_heartbeat_confirmed)
    monkeypatch.setattr(_Scope, "adopt_confirmed", observed_adopt)
    monkeypatch.setattr(kernel._Driver, "bind_turn", observed_bind_turn)
    monkeypatch.setattr(_Scope, "reset_token", observed_reset_token)

    try:
        assert kernel.drive(store, "s", runtime=make_runtime)
    except Exception as exc:
        raise AssertionError(
            {
                "outcomes": [repr(outcome) for outcome in observed_invoker.outcomes],
                "tokens": [(token.cancelled, token.reason) for token in observed_invoker.tokens],
                "binds": binds,
                "resets": resets,
                "model_errors": model_errors,
                "renewals": renewals,
                "heartbeat_attempts": heartbeat_attempts,
                "heartbeat_failures": heartbeat_failures,
                "heartbeat_finished": heartbeat_finished,
                "connections": connections,
                "records": [row.record.kind for row in read_fixture_state(store, "s")[1][-20:]],
            }
        ) from exc
    assert model_renewals.is_set(), {
        "model_errors": model_errors,
        "renewals": renewals,
        "heartbeat_attempts": heartbeat_attempts,
        "heartbeat_failures": heartbeat_failures,
        "heartbeat_finished": heartbeat_finished,
        "connections": connections,
    }

    controls_seen = [poll for poll in confirms if control_ids <= {control.input_id for control in poll.controls}]
    assert controls_seen
    assert adopted and control_ids <= {control.input_id for control in adopted[0].controls}
    assert adopted[0] in controls_seen

    new_binds = [bind for bind in binds if bind.turn_id == new_turn]
    assert new_binds
    assert control_ids <= set(new_binds[-1].applied_ids)
    assert all(not bind.cancelled for bind in new_binds)
    assert all(bind.direct_state_mapping for bind in binds)

    assert resets
    assert any(reset.before_id != reset.after_id and not reset.cancelled and reset.reason is None for reset in resets)
    assert observed_invoker.tokens
    long_token = observed_invoker.tokens[-1]
    assert not long_token.cancelled and long_token.reason is None

    assert len(renewals) >= 3
    renewal_threads = {thread for _, thread, _ in renewals}
    assert main_thread not in renewal_threads, "\n".join(driver_renewal_stacks)
    assert model_renewal_window
    assert len({thread for _, thread, _ in model_renewal_window}) == 1
    assert max_active_connections == 1
    assert all(connection_id != driver_connection for connection_id, _ in connections)

    final_state, records, _ = read_fixture_state(store, "s")
    assert final_state.active_turn_id is None
    assert records[-1].record.kind == "turn_ended"
    assert records[-1].record.payload["status"] == "completed"


def test_confirmed_snapshot_binds_controls_and_survives_handler_reads(store, database, monkeypatch) -> None:
    start(store)
    content_reader = fixture_content_reader(store, "s")
    parked_runtime = runtime(
        database,
        [LLMResponse("", [ToolCall("approval-call", "effect", {})])],
        [effect_tool(approval=True)],
        content_reader=content_reader,
        ttl_ms=15_000,
        heartbeat_seconds=0.04,
    )
    parked_runtime.config.approval_timeout_seconds = 5
    parked_runtime.heartbeat_store = store.heartbeat_store
    assert kernel.drive(store, "s", runtime=parked_runtime)

    state, _, _ = read_fixture_state(store, "s")
    current_turn = state.active_turn_id
    assert current_turn is not None
    generation = state.turns[current_turn].start._payload["generation"]
    operation_id, operation = next((operation_id, op) for operation_id, op in state.operations.items() if op.kind == "tool")
    attempt_number = max(operation.attempts)
    attempt = operation.attempts[attempt_number]
    assert attempt.wait is not None
    deadline = attempt.wait["deadline_ms"]
    assert deadline is not None

    current_id = "current-cancel"
    stale_id = "stale-suspend"
    other_id = "other-suspend"
    push_fixture_input(store, "s", item(stale_id, "control", action="suspend", target_turn_id="s/turn/old", generation=7))
    push_fixture_input(
        store, "s", item(current_id, "control", action="cancel", target_turn_id=current_turn, generation=generation)
    )
    push_fixture_input(store, "s", item(other_id, "control", action="suspend", target_turn_id="s/turn/other", generation=0))

    lease = store.acquire("s", owner="poll-snapshot", ttl_ms=15_000)
    assert lease is not None
    scope = _Scope(lease, parked_runtime, now_ms=store._now())
    driver = kernel._Driver(store, "s", parked_runtime, scope)
    driver_operation = driver.state.operations[operation_id]
    driver_attempt = driver_operation.attempts[attempt_number]
    assert driver_attempt.wait is not None
    deadline = driver_attempt.wait["deadline_ms"]
    assert deadline is not None
    poll = scope.poll(store)
    assert scope.confirmed_poll() is poll
    assert {current_id, stale_id, other_id} == {control.input_id for control in poll.controls}

    scope.bind_turn(current_turn, generation, {})
    cancelled_token = scope.token
    assert cancelled_token.cancelled and cancelled_token.reason == "cancel"
    applied = {control.input_id: object() for control in poll.controls}
    cancelled_before_reset = cancelled_token
    scope.bind_turn(current_turn, generation, applied)
    scope.reset_token()
    new_token = scope.token
    assert new_token is not cancelled_before_reset
    assert not new_token.cancelled and new_token.reason is None
    assert scope.control_binding is not None and scope.control_binding.applied_inputs is applied

    def forbidden_renew(_request):
        pytest.fail("driver-thread poll reopened a lease renewal")

    monkeypatch.setattr(scope, "poll", forbidden_renew)
    assert _DispatchHost(driver)._handle(dispatch_core.DispatchRead.POLL_LEASE) is poll
    host = _Host(driver)
    assert host.handle(inputs.InputSetting.POLL_NOW) == poll.db_now_ms
    assert scope.confirmed_poll() is poll

    plan = driver.hydrate_plan(driver_attempt.execution_plan)
    handle = driver.hydrate_parked_wait(operation_id, attempt_number, driver_attempt.wait)["handle"]
    assert driver_attempt.plan.attempt == attempt_number
    assert host.handle(core.ApprovalRequest(plan, handle, deadline)) is None
    assert host.approval_request is not None
    assert host.approval_request.metadata["timeout_seconds"] == pytest.approx(max(0, (deadline - poll.db_now_ms) / 1000))
    assert scope.confirmed_poll() is poll

    scope.close()
    assert store.release(poll.lease)
