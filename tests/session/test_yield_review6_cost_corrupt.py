"""A6 idle wake cost and controlled corruption errors on disposable SQL stores."""

from collections import Counter
from hashlib import sha256
from time import monotonic

import pytest

from vv_agent.app_server import AppServer, DefaultAppServerHost
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.run_handle import RunHandle
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse

from .test_recovery_matrix import runtime
from .test_yield_parked_owner import shutdown
from .test_yieldable_waits import waiting_runtime


def cost_tree(store, database, monkeypatch, count):
    rt, observed = waiting_runtime(database, count=count)
    server = AppServer(store=store, host=DefaultAppServerHost(agent=rt.agent, run_config=rt.config))
    server.store.create_thread_receipt(input_id="s", agent_key="default")
    with store.atomic() as tx:
        tx.push("s", InboxItem("initial", "user", {"content": "go"}))
    drive(store, "s", runtime=rt)
    for i in range(count):
        crt = runtime(database, [LLMResponse("waiting")])
        crt.config.no_tool_policy = "wait_user"
        drive(store, f"yield-child-{i}", runtime=crt)
    tid = store.read_state("s")[0].active_turn_id
    h = RunHandle(server.kernel, "s", tid, rt, None)
    server.kernel.handles.append(h)
    h._done = True

    def assemble(*args, **kw):
        result, _ = waiting_runtime(database, count=count)
        result.child_runtime = lambda *_: runtime(database, [])
        return result

    monkeypatch.setattr(server.kernel, "runtime", assemble)
    return server, h, observed


@pytest.mark.persistent_store
@pytest.mark.parametrize("count", [1, 50])
def test_idle_wake_cost(store, database, monkeypatch, count):
    server, h, observed = cost_tree(store, database, monkeypatch, count)
    calls = Counter()
    for name in ["read_state", "read", "is_runnable", "next_drive_delay_ms", "_rows"]:
        original = getattr(store, name)

        def wrap(*a, _name=name, _fn=original, **kw):
            calls[_name] += 1
            return _fn(*a, **kw)

        monkeypatch.setattr(store, name, wrap)
    try:
        n = 100
        durations = []
        before = {sid: store.read_state(sid)[1][-1].seq for sid in ["s", *[f"yield-child-{i}" for i in range(count)]]}
        calls.clear()
        t = monotonic()
        for _ in range(n):
            t0 = monotonic()
            server.wake("s")
            durations.append((monotonic() - t0) * 1000)
        seconds = monotonic() - t
        returned_calls = dict(calls)
        server.run_adapter.join()
        settled_seconds = monotonic() - t
        settled_calls = dict(calls)
        print(
            "IDLE_WAKE_COST",
            dict(
                backend=type(store).__name__,
                children=count,
                n=n,
                seconds=round(seconds, 6),
                mean_ms=round(seconds * 1000 / n, 3),
                p95_ms=round(sorted(durations)[94], 3),
                calls=returned_calls,
                settled_seconds=round(settled_seconds, 6),
                settled_calls=settled_calls,
                handles=len(server.kernel.handles),
            ),
            flush=True,
        )
        assert not observed
        assert all(store.read_state(sid)[1][-1].seq == seq for sid, seq in before.items())
        assert not any(store.peek_inbox(sid) for sid in before)
        assert 1 < len(server.kernel.handles) <= n + 1
        assert all(handle.done() for handle in server.kernel.handles)
        # Measure an actual full idle exit separately from the wake admission burst.
        rt = server.kernel.runtime(None, None)
        handle = RunHandle(server.kernel, "s", h.run_id, rt, None)
        calls.clear()
        t = monotonic()
        handle.start()
        handle.result(timeout=30)
        print(
            "NORMAL_EXIT_COST",
            dict(backend=type(store).__name__, children=count, seconds=round(monotonic() - t, 6), calls=dict(calls)),
            flush=True,
        )
        assert not handle._error and not observed
    finally:
        shutdown(server)


@pytest.mark.persistent_store
@pytest.mark.parametrize("damage", ["cycle", "missing_session", "missing_operation", "broken_handle", "record_digest"])
def test_host_wake_reports_corruption(store, database, monkeypatch, damage):
    server, _h, _ = cost_tree(store, database, monkeypatch, 1)
    warnings = []
    original_notify = server.run_adapter._notify_subscribers

    def notify(sid, method, params):
        if method == "error/warning":
            warnings.append((sid, params))
        return original_notify(sid, method, params)

    monkeypatch.setattr(server.run_adapter, "_notify_subscribers", notify)
    # Break disposable persisted facts. Recompute the local digest for structural
    # cases so the normal strict decoder/reducer, rather than the checksum, rejects them.
    rows = store.read_state("s")[1]
    record = rows[0] if damage == "record_digest" else next(r for r in rows if r.record.kind == "op_parked")
    data = record.record.to_dict()
    if damage != "record_digest":
        handle = data["payload"]["handle"]
        if damage == "cycle":
            handle["session_id"] = "s"
        elif damage == "missing_session":
            handle["session_id"] = "not-present"
        elif damage == "missing_operation":
            data["operation_id"] = "not-present"
        else:
            data["payload"].pop("handle")
        body = canonical_json_bytes(data)
        checksum = sha256(body).hexdigest()
        with store.atomic():
            store._rows(
                "UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=%s", (body, checksum, "s", record.seq)
            )
    else:
        with store.atomic():
            store._rows("UPDATE sk_record SET digest=%s WHERE session_id=%s AND seq=%s", ("0" * 64, "s", 1))
    store._fold_cache = None
    store._previous_prefix = None
    try:
        server.wake("s")
        server.run_adapter.join()
        print("CONTROLLED_CORRUPTION", dict(backend=type(store).__name__, damage=damage, warnings=warnings), flush=True)
        assert warnings and all(sid == "s" and "s:" in p["message"] for sid, p in warnings)
        assert all(p["code"] in {"drive_start", "event_stream"} for _, p in warnings)
        assert len(server.kernel.handles) <= 2
    finally:
        shutdown(server)
