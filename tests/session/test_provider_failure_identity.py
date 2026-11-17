"""Provider failure identity survives independent stores and pure observation."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.kernel._provider_outcomes import ProviderResultInvalid
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import InboxItem
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .test_worker_surface_failures import (
    _force_due,
    _InvalidQueryProvider,
    _push_in_transaction_and_wake,
    _runtime_factory_for,
    _wait_for,
    external_wait,
)

pytestmark = pytest.mark.persistent_store
MASTER_KEY = b"i" * 32


def test_provider_result_callback_recovers_original_failure(database: str | Any, tmp_path: Path) -> None:
    sid = "provider-failure-identity-callback"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "callback-keys.sqlite3")
    provider = _InvalidQueryProvider()
    model = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("external-call", "external_wait", {})]),
            LLMResponse("recovered"),
        ],
    )
    with ExitStack() as stack:
        store = stack.enter_context(open_store(database))
        store.install_schema()
        driver = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            worker_level=2,
            handler_versions=("1",),
        )
        try:
            driver.runtime_factory = _runtime_factory_for(driver, {"external_wait": provider})
            driver.create(sid, "")
            handle = driver.start(
                sid,
                Agent("provider", "Wait externally.", model="m", tools=[external_wait]),
                RunConfig(model_provider=model),
                "go",
            )
            _wait_for(lambda: bool(provider.handle))
            driver._worker.wait_idle(timeout=5)
            _force_due(driver, sid)
            _wait_for(
                lambda: (
                    (failure := driver._errors.get(sid)) is not None
                    and isinstance(failure.error, ProviderResultInvalid)
                    and failure.terminal
                )
            )
            driver._worker.wait_idle(timeout=5)
            state, _, _ = driver.store.read_state(sid)
            turn = state.turns[state.active_turn_id or ""]
            _push_in_transaction_and_wake(
                driver,
                sid,
                InboxItem(
                    "provider-result",
                    "provider_result",
                    {
                        "operation_id": provider.handle["operation_id"],
                        "attempt": provider.handle["attempt"],
                        "request_digest": provider.handle["request_digest"],
                        "provider_binding": provider.handle["provider"],
                        "result": provider.result,
                        "usage": {},
                        "evidence": [provider.handle["evidence"]],
                    },
                    state.active_turn_id,
                    turn.start._payload["generation"],
                ),
            )

            assert handle.result(timeout=8).final_output == "recovered"
            assert provider.queries == 1
            assert sid not in driver._errors
        finally:
            driver.close()
    key_provider.close()


def test_provider_failure_observer_reconciles_remote_terminal(database: str | Any) -> None:
    sid = "provider-failure-identity-persistent"
    key_provider = LocalKeyProvider(MASTER_KEY)
    provider = _InvalidQueryProvider()
    model = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("external-call", "external_wait", {})]),
            LLMResponse("remote done"),
        ],
    )
    agent = Agent("provider", "Wait externally.", model="m", tools=[external_wait])
    config = RunConfig(model_provider=model)
    providers = {"external_wait": provider}

    with ExitStack() as stack:
        first_store = stack.enter_context(open_store(database))
        first_store.install_schema()
        second_store = stack.enter_context(open_store(database))
        first = SessionDriver(
            store=first_store,
            content_key_provider=key_provider,
            worker_level=2,
            handler_versions=("1",),
        )
        second = SessionDriver(
            store=second_store,
            content_key_provider=key_provider,
            worker_level=2,
            handler_versions=("1",),
        )
        try:
            first.runtime_factory = _runtime_factory_for(first, providers)
            first.create(sid, "")
            first.start(sid, agent, config, "go")
            _wait_for(lambda: bool(provider.handle))
            first._worker.wait_idle(timeout=5)
            _force_due(first, sid)
            _wait_for(
                lambda: (
                    (failure := first._errors.get(sid)) is not None
                    and isinstance(failure.error, ProviderResultInvalid)
                    and failure.terminal
                )
            )
            first._worker.wait_idle(timeout=5)
            failure = first._errors[sid]
            failure.warning_sent = True
            old_key = (provider.handle["operation_id"], provider.handle["attempt"])
            assert failure.fenced_operations == frozenset({old_key})
            first._worker.stop()
            assert first._worker.join(5)

            state, _, _ = first.store.read_state(sid)
            turn = state.turns[failure.turn_id or ""]
            second.runtime_factory = _runtime_factory_for(second, providers)
            second._runtimes[sid] = second._static_registration(
                sid,
                agent,
                config,
                None,
                handler_version="1",
                armed=True,
            )
            second._worker.start()
            second.push(
                sid,
                InboxItem(
                    "remote-result",
                    "provider_result",
                    {
                        "operation_id": provider.handle["operation_id"],
                        "attempt": provider.handle["attempt"],
                        "request_digest": provider.handle["request_digest"],
                        "provider_binding": provider.handle["provider"],
                        "result": provider.result,
                        "usage": {},
                        "evidence": [provider.handle["evidence"]],
                    },
                    failure.turn_id,
                    turn.start._payload["generation"],
                ),
            )

            def remote_terminal() -> bool:
                current_state, _, _ = first.store.read_state(sid)
                current_turn = current_state.turns.get(failure.turn_id or "")
                return current_turn is not None and current_turn.ended

            _wait_for(remote_terminal, timeout=8)
            second._worker.wait_idle(timeout=5)
            second._worker.stop()
            assert second._worker.join(5)

            observed_state, _, _ = first._read_observed_state(first.store, sid)
            _, _, error = first._snapshot_status(
                first.store,
                sid,
                observed_state,
                failure.turn_id or "",
            )
            assert first._errors.get(sid) is None
            assert error is None
        finally:
            second.close()
            first.close()
    key_provider.close()
