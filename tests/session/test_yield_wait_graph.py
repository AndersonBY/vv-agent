"""Wait projection diagnoses path cycles and still visits healthy shared paths."""

from hashlib import sha256

import pytest

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.store import Conflict

from .test_yield_review6_cost_corrupt import cost_tree


@pytest.mark.persistent_store
def test_wait_projection_cycle_is_explicit_and_restorable(store, database, monkeypatch):
    server, _handle, observed = cost_tree(store, database, monkeypatch, 2)
    original = next(r for r in store.read_state("s")[1] if r.record.kind == "op_parked")

    def update(body):
        with store.atomic():
            store._rows(
                "UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=%s",
                (body, sha256(body).hexdigest(), "s", original.seq),
            )
        store._fold_cache = store._previous_prefix = None

    try:
        healthy = server.kernel.waits("s")
        assert {sid for sid, _, _ in healthy} == {"yield-child-0", "yield-child-1"}
        data = original.record.to_dict()
        child_id = data["payload"]["handle"]["session_id"]
        data["payload"]["handle"]["session_id"] = "s"
        data["payload"]["yieldable_children"] = ["s" if sid == child_id else sid for sid in data["payload"]["yieldable_children"]]
        update(canonical_json_bytes(data))
        with pytest.raises(Conflict, match="cyclic child ownership: s"):
            server.kernel.waits("s")
        notices = []
        notify = server.run_adapter._notify_subscribers

        def capture(sid, method, params):
            if method == "error/warning":
                notices.append((sid, params))
            return notify(sid, method, params)

        monkeypatch.setattr(server.run_adapter, "_notify_subscribers", capture)
        for _ in range(2):
            server.wake("s")
            server.run_adapter.join()
            assert server.run_adapter._state_manager.active_turn("s") is None
            assert all(h.done() for h in server.kernel.handles)
        assert len(notices) == 2 and all(sid == "s" and "cyclic child ownership: s" in p["message"] for sid, p in notices)
        update(original.record.encode())
        assert server.kernel.waits("s") == healthy
        server.wake("s")
        server.run_adapter.join()
        assert not observed and all(h.done() for h in server.kernel.handles)
        assert {sid for sid, _, _ in server.kernel.waits("s")} == {"yield-child-0", "yield-child-1"}
    finally:
        server.run_adapter.close()
        server.kernel.close()


@pytest.mark.persistent_store
def test_wait_projection_allows_repeated_acyclic_paths(store, database, monkeypatch):
    server, _handle, observed = cost_tree(store, database, monkeypatch, 2)
    try:
        state, records, watermark = store.read_state("s")
        expected = server.kernel.waits("s")
        oid, number = next(iter(state.waits))
        # Repeated edges in a projection snapshot must not become a global visited
        # filter. Both paths still read the same real, healthy child sessions.
        from copy import deepcopy

        state.operations[oid].attempts[number + 1] = deepcopy(state.operations[oid].attempts[number])
        actual = server.kernel.waits("s", snapshot=(state, records, watermark))
        assert actual == expected + expected
        assert {sid for sid, _, _ in actual} == {"yield-child-0", "yield-child-1"}
        assert not observed
    finally:
        server.run_adapter.close()
        server.kernel.close()
