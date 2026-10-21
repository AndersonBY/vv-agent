"""App Server A1 projections and admission on the real SQL backends."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from threading import Barrier

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.session.app_server import ThreadStatus, project_thread_status
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Accepted
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.replay import list_events
from vv_agent.types import LLMResponse

from .conftest import open_store
from .test_children import parent_runtime
from .test_provider_usage import ReceiptProvider
from .test_recovery_matrix import runtime


def create(store):
    with store.atomic() as tx:
        tx.create(
            SessionSpec("s", "test", "/tmp", attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}}),
            consumers=("app_server",),
        )
        tx.push("s", InboxItem("prompt", "user", {"content": "go"}))


def test_real_provider_wait_status_and_bounded_projection(store, database):
    class Pending(ReceiptProvider):
        def query(self, handle):
            return Accepted(handle)

        def authenticate(self, item, plan):
            return item.kind == "provider_evidence" and item.payload["handle"] == self.handle

    provider = Pending({"content": "done", "tool_calls": [], "reasoning_content": None, "raw": {}}, {})
    create(store)

    class Cut(BaseException):
        pass

    def lost(point, plan):
        if point == "after_external_call":
            provider.bind(plan)
            raise Cut()

    rt = runtime(database, [LLMResponse("done")], providers={"model": provider}, hook=lost)
    with pytest.raises(Cut):
        drive(store, "s", runtime=rt)
    with store.atomic() as tx:
        tx.push(
            "s",
            replace(
                provider.item,
                kind="provider_evidence",
                payload={k: provider.handle[k] for k in ("operation_id", "attempt", "request_digest")}
                | {"handle": provider.handle},
            ),
        )
    drive(store, "s", runtime=rt)
    state, _, _ = store.read_state("s")
    assert project_thread_status(state) is ThreadStatus.PROVIDER_RUNNING
    server = AppServer(store=store)
    status = server.run_adapter.public_thread_status("s")
    assert status["waitKind"] == "provider" and status["pendingInteraction"] is None
    assert status["unconfirmedCount"] == 1
    assert "retained" not in str(status) and "job_id" not in str(status)
    after = through = None
    events = []
    for _ in range(100):
        page = list_events(store, "s", limit=1, after=after, through=through)
        events.extend(x["event"] for x in page["events"])
        after, through = page["nextCursor"], page["throughCursor"]
        if not page["more"]:
            break
    assert any(e["type"] == "model_call_started" for e in events)
    assert through is not None
    assert status["projectionWatermark"] == through["seq"]

    # Historical unresolved work belongs to that turn, never the next one.
    tid = state.active_turn_id
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel-first", "control", {"action": "cancel"}, tid))
    drive(store, "s", runtime=rt, _one_turn=True)
    assert server.run_adapter.public_thread_status("s")["unconfirmedCount"] == 1
    with store.atomic() as tx:
        tx.push("s", InboxItem("second", "user", {"content": "new turn"}))
    drive(store, "s", runtime=runtime(database, [LLMResponse("new done")]), _one_turn=True)
    assert server.run_adapter.public_thread_status("s")["unconfirmedCount"] == 0


def test_real_child_and_suspended_status(store, database):
    create(store)
    rt = parent_runtime(database)
    drive(store, "s", runtime=rt)
    state = store.read_state("s")[0]
    assert project_thread_status(state) is ThreadStatus.CHILD_RUNNING
    tid = state.active_turn_id
    with store.atomic() as tx:
        tx.push("s", InboxItem("suspend", "control", {"action": "suspend"}, tid))
    drive(store, "s", runtime=rt)
    assert project_thread_status(store.read_state("s")[0]) is ThreadStatus.SUSPENDED


def test_real_retry_backoff_status(store, database):
    from .helpers import record

    create(store)

    class Cut(BaseException):
        pass

    def stop(point, plan):
        if point == "before_external_call":
            raise Cut()

    rt = runtime(database, [], hook=stop)
    with pytest.raises(Cut):
        drive(store, "s", runtime=rt)
    state, rows, _ = store.read_state("s")
    plan = next(r.record for r in rows if r.record.kind == "op_planned")
    lease = store.acquire("s", owner="recovery", ttl_ms=15000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=rows[-1].seq,
            commit_id="unknown",
            records=(record("op_unknown", tid=plan.turn_id, oid=plan.operation_id, retry_at_ms=store._now() + 60000),),
        )
    store.release(lease)
    state = store.read_state("s")[0]
    assert project_thread_status(state, now_ms=store._now()) is ThreadStatus.RETRY_BACKOFF
    assert state.next_drive_ms > store._now()


@pytest.mark.persistent_store
def test_concurrent_api_nodes_share_atomic_submission_fence(store, database):
    from test_app_server_distributed import request, submission

    provider = ScriptedModelProvider.from_callback("test", "m", lambda r: LLMResponse("done"))
    with ExitStack() as stack:
        nodes = []
        for name in ("A", "B"):
            connection = stack.enter_context(open_store(database))
            server = AppServer(
                store=connection,
                transport=ChannelTransport(connection_id=name),
                host=DefaultAppServerHost(agent=Agent("test", "Answer."), run_config=RunConfig(model_provider=provider)),
            )
            stack.callback(server.kernel.close)
            stack.callback(server.run_adapter.join)
            request(server, 0, "initialize", {"clientInfo": {"name": name}})
            nodes.append(server)
        request(nodes[0], 1, "thread/start", {"inputId": "thread"})
        barrier = Barrier(2)

        def submit(server):
            barrier.wait(timeout=5)
            return request(server, 2, "turn/start", submission())["result"]

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(submit, s) for s in nodes]
            receipts = [f.result(timeout=15) for f in futures]
        for server in nodes:
            server.run_adapter.join()
        assert sorted(r["replayed"] for r in receipts) == [False, True]
        state = store.read_state("thread")[0]
        assert len(state.turns) == 1 and state.active_turn_id is None


@pytest.mark.persistent_store
def test_record_observer_snapshot_does_not_lock_session_writers(store, database):
    from concurrent.futures import ThreadPoolExecutor

    from vv_agent.session.records import SessionSpec

    from .conftest import open_store

    with store.atomic() as tx:
        tx.create(SessionSpec("observer", "local", "."), consumers=("events",))
    with ThreadPoolExecutor(1) as pool, open_store(database) as writer, writer.atomic() as tx:
        tx.lock_session("observer")
        future = pool.submit(store.read, "observer", after_seq=0, limit=1)
        page = future.result(timeout=0.5)
        assert page.head_seq == 1 and len(page.records) == 1


def test_lease_retry_status_uses_database_time_and_no_session_lock(store):
    from vv_agent.session.records import SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("retry", "local", "."), consumers=("events",))
    lease = store.acquire("retry", owner="dead", ttl_ms=2000)
    head, epoch, remaining = store.lease_status("retry")
    assert head == 1 and epoch == lease.epoch and 0 < remaining <= 2000


def test_catalog_controls_are_consumed_facts_at_the_captured_head(store):
    from vv_agent.session.surfaces import SessionDriver

    driver = SessionDriver(store=store)
    driver.create("catalog", ".")
    assert store.control_records("catalog", through_seq=1) == ()
    driver.control("catalog", "archive", "archive")
    head = store.read("catalog").head_seq
    facts = store.control_records("catalog", through_seq=head)
    assert len(facts) == 1
    assert facts[0].record.payload["disposition"] == "applied"
    assert facts[0].record.payload["input"]["payload"]["action"] == "archive"
    assert store.control_records("catalog", through_seq=1) == ()
    driver.push("catalog", InboxItem("not-consumed", "control", {"action": "close"}))
    assert store.control_records("catalog", through_seq=head) == facts
