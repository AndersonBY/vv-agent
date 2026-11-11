"""Native outer transaction while an App Server execution lease already exists."""

from threading import Event, current_thread

import pytest

from vv_agent.session.kernel import _Driver
from vv_agent.session.postgres import join_transaction
from vv_agent.session.records import InboxItem

from .test_yield_completion_errors import parked_server
from .test_yield_review6_edges import payload


@pytest.mark.postgres
def test_running_worker_never_dispatches_uncommitted_native_reply(store, tmp_path, monkeypatch):
    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    tid = store.read_state("completion")[0].active_turn_id
    entered, read, dispatched, finish_model = Event(), Event(), Event(), Event()
    outcome = Event()
    cuts = []
    model_entries = []
    model_returns = []
    execution_connections = []
    refresh = _Driver.refresh
    client = adapter._store.runtimes["completion"].llm
    callback = client.callback

    def model(request):
        model_entries.append(request)
        response = callback(request)
        model_returns.append(response)
        dispatched.set()
        outcome.set()
        assert finish_model.wait(5)
        return response

    # Observe the existing scripted client; preserve the admitted runtime binding.
    monkeypatch.setattr(client, "callback", model)

    def gate(driver):
        cut = driver.sid == "completion" and current_thread().name == "session-surface-driver" and not cuts
        watermark = driver.watermark
        if cut:
            cuts.append(driver)
            execution_connections.append(driver.store)
            entered.set()
            assert read.wait(5)
        result = refresh(driver)
        if cut and driver.watermark == watermark:
            # The real committed read saw no new input. Observe at that read,
            # before unrelated idle writes can contend with the host row lock.
            outcome.set()
            assert finish_model.wait(5)
        return result

    monkeypatch.setattr(_Driver, "refresh", gate)

    class Rollback(Exception):
        pass

    try:
        server.wake("completion")
        assert entered.wait(5)
        driver = cuts[0]
        assert driver.scope.ready.wait(5)
        assert driver.scope.lease.session_id == "completion" and not driver.scope.lost
        with pytest.raises(Rollback), store.connection.transaction():
            join_transaction(store.connection).push(
                "completion", InboxItem("native-uncommitted", "provider_result", payload(provider), tid, 0)
            )
            server.wake("completion")
            read.set()
            observed = outcome.wait(5)
            if not observed:
                import sys
                import threading
                import traceback

                print(
                    "NATIVE_DIAGNOSTIC",
                    "handle_errors",
                    [repr(h._error) for h in server.kernel.handles],
                    "driver_turn",
                    driver.state.active_turn_id,
                    "watermark",
                    driver.watermark,
                    flush=True,
                )
                frames = sys._current_frames()
                for thread in threading.enumerate():
                    if (
                        thread.name in {"session-surface-driver", "session-app-server-pump", "session-lease-heartbeat"}
                        and thread.ident in frames
                    ):
                        print("NATIVE_STACK", thread.name, "".join(traceback.format_stack(frames[thread.ident])), flush=True)
            assert observed, "neither real idle step nor actual model dispatch occurred"
            visible = dispatched.is_set()
            print(
                "FIX9_NATIVE_ACTIVE_OWNER",
                type(store).__name__,
                "uncommitted_model_dispatch",
                visible,
                "independent_execution",
                execution_connections[0] is not store,
                "model_calls_before_exit",
                len(calls),
                flush=True,
            )
            raise Rollback()
        finish_model.set()
        adapter.join()
        assert store.read_input("completion", "native-uncommitted") is None
        assert not visible, "an active worker dispatched a model from a rolled-back native transaction"
        assert all(h._error is None and h.done() for h in server.kernel.handles)
        assert not any(method == "error/warning" for _, method, _ in notices)
        assert model_entries == model_returns == []
        assert driver.scope.successful_ready
        assert len(calls) == provider.submits == 1
    finally:
        read.set()
        finish_model.set()
        adapter.close()
        adapter.join()
        server.kernel.close()
