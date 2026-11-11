"""Dropping the caller reference during a foreground model call leaves its run alive."""

import gc
import json
import threading
import weakref

import pytest

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, ScriptedModelProvider
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.session import kernel, ownership
from vv_agent.types import LLMResponse


@pytest.mark.persistent_store
def test_drop_session_during_foreground_model_releases_execution_owners(store, database, tmp_path, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    errors, results, scopes, events = [], [], [], []

    def respond(request):
        entered.set()
        assert finish.wait(5), "controlled foreground model was not released"
        return LLMResponse("done")

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=ScriptedModelProvider.from_steps("test", "m", [respond]),
            workspace=tmp_path,
            session_store=store,
        )
    )
    session = client.create_session(agent=Agent("held", "Answer.", model="m"), session_id="drop-held")
    session.subscribe(lambda event, payload: events.append(event))
    # Prepare imports and the actual product tokenizer before the bounded run handshake.
    runtime = session._runtime()
    definition = runtime.definition(runtime.compile("go", "drop-preparation"))
    count_messages_tokens([{"role": "user", "content": "go"}], model=definition["memory_settings"]["model"])
    initialize = kernel._Scope.__init__

    def capture(scope, lease, *args, **kwargs):
        initialize(scope, lease, *args, **kwargs)
        if lease.session_id == "drop-held":
            scopes.append(scope)

    monkeypatch.setattr(kernel._Scope, "__init__", capture)
    with ownership._ledger_lock():
        baseline = dict(ownership._active)
    reference = weakref.ref(session)

    def run(current):
        try:
            results.append(current.prompt("go"))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, args=(session,), daemon=True)
    worker.start()
    try:
        assert entered.wait(5), "foreground model handshake not reached"
        assert not errors and worker.is_alive() and scopes
        del session
        gc.collect()
        current = reference()
        assert current is not None and current.running and not current.closed
        assert all(not scope.token.cancelled and not scope.lost for scope in scopes)
        assert "session_closed" not in events
        assert not store.read_state("drop-held")[0].closed
        with ownership._ledger_lock():
            assert set(ownership._active) - set(baseline), "foreground E was not reserved"
        finish.set()
        worker.join(10)
        assert not worker.is_alive() and not errors
        assert results[0].final_output == "done"
        assert all(not scope.thread.is_alive() and not scope.lost for scope in scopes)
        assert all(handle.done() and not handle._thread.is_alive() for handle in client.driver.handles)
        with ownership._ledger_lock():
            assert ownership._active == baseline
        lease = store._one("SELECT lease_owner,lease_until_ms FROM sk_session WHERE session_id=%s", ("drop-held",))
        assert tuple(lease) == (None, None)
        assert not store.read_state("drop-held")[0].closed and "session_closed" not in events
        print(
            "FOREGROUND_DROP "
            + json.dumps(
                {
                    "entered": entered.is_set(),
                    "result": results[0].final_output,
                    "HB_stopped": all(not scope.thread.is_alive() for scope in scopes),
                    "lease": list(lease),
                    "baseline_restored": True,
                    "closed_event": "session_closed" in events,
                }
            ),
            flush=True,
        )
    finally:
        finish.set()
        worker.join(10)
        if not worker.is_alive():
            client.driver.close()
