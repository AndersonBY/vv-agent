"""Shared host-controlled at-least-once transport for dispatch tests and soak."""

from vv_agent.session.children import child_delivery
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Definitive
from vv_agent.session.records import SessionSpec
from vv_agent.session.supervisor import tick
from vv_agent.types import ToolExecutionResult

from .conftest import open_store
from .controlled import ControlledProvider
from .test_recovery_matrix import runtime


class Transport:
    def __init__(self, store, database):
        self.store, self.database = store, database
        self.queue = []
        self.wakes = []
        self.runtimes = {}
        self.projections = []

    def wake(self, sid):
        self.wakes.append(sid)
        self.queue.append(sid)

    def create(self, sid="s", steps=(), tools=(), consumers=("host",)):
        with self.store.atomic() as tx:
            tx.create(SessionSpec(sid, "test", "/tmp"), consumers=consumers)
        self.runtimes[sid] = runtime(self.database, steps, tools, wake=self.wake)

    def push(self, sid, item):
        with self.store.atomic() as tx:
            tx.push(sid, item)
        self.wake(sid)

    def deliver(self, index=0):
        sid = self.queue.pop(index)
        drive(self.store, sid, runtime=self.runtimes[sid])

    def drain(self):
        for _ in range(20):
            if not self.queue:
                return
            self.deliver()
        raise AssertionError("transport did not reach quiescence")

    def project(self, sid, consumer):
        parent = None
        with self.store.atomic() as tx:
            batch = tx.consumer_batch(sid, consumer, limit=256 if consumer == "child_delivery" else 2)
            if batch is None:
                return
            self.projections.append((sid, consumer, batch.from_seq, batch.through_seq))
            if consumer == "child_delivery":
                child_delivery(self.store, tx, sid)
                if any(r.record.kind == "turn_ended" for r in batch.records):
                    parent = self.store.read(sid, limit=1).records[0].record.payload["parent_session_id"]
            else:
                tx.ack(batch)
        if parent:
            self.wake(parent)

    def tick(self, page_size=100):
        return tick(self.store, runtime=self.runtimes.__getitem__, project=self.project, page_size=page_size)


class WorkerKilled(BaseException):
    pass


class PollingProvider(ControlledProvider):
    def __init__(self, database):
        super().__init__(database, "accepted")
        self.queries = []
        self.deadline = None

    def query(self, handle):
        with open_store(self.database) as store:
            now = store._now()
        self.queries.append(now)
        if self.deadline is not None and now >= self.deadline:
            return Definitive(
                ToolExecutionResult(tool_call_id="job", content="deadline handled").to_dict(), (handle["evidence"],)
            )
        return super().query(handle)
