"""Closed declarations, exact baseline readers, and unchanged undeclared SQL waits."""

import sys
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from vv_agent.session import kernel, sql
from vv_agent.session.children import SummaryChildTool
from vv_agent.session.providers import Accepted, ProviderResultInvalid, validate_acceptance
from vv_agent.session.records import InboxItem, RecordError
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_batch_children import complete, deliver, rows
from .test_recovery_matrix import start
from .test_yieldable_waits import YieldProvider, provider_runtime, steer, waiting_runtime

BASELINE = "6cbc06b61d869e836ccf946984a29f70bcdc547b"
BASELINE_BLOBS = {"records": "532662aec84014693e4b0c1700ac696e9e790708", "kernel": "30c102ea0efd0e1355b4e291bc8826969286848d"}


def baseline_module(name, monkeypatch):
    path = Path(__file__).resolve().parent / "fixtures" / "baseline_0341" / f"{name}.py.txt"
    data = path.read_bytes()
    expected = BASELINE_BLOBS[name]
    actual = sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert actual == expected, f"Baseline Git blob SHA-1 mismatch for {path.name}: expected {expected}, got {actual}"
    source = data.decode("utf-8")
    module = ModuleType(f"vv_agent.session._baseline_{name}")
    module.__package__ = "vv_agent.session"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, f"{BASELINE}/{name}.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("provider_wait", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_0341_cold_reader_accepts_undeclared_and_rejects_entire_declared_session(
    store, database, monkeypatch, provider_wait, declared
):
    start(store)
    rt, _ = (
        provider_runtime(database, YieldProvider(declared=declared))
        if provider_wait
        else waiting_runtime(database, declared=declared)
    )
    kernel.drive(store, "s", runtime=rt)
    before = [r.record.encode() for r in rows(store)]
    old = baseline_module("records", monkeypatch)
    monkeypatch.setattr(sql, "Record", old.Record)
    monkeypatch.setattr(sql, "InboxItem", old.InboxItem)
    store._fold_cache = store._previous_prefix = None
    if declared:
        with pytest.raises(old.RecordError):
            store.read_state("s")
    else:
        assert [r.record.encode() for r in store.read_state("s")[1]] == before


@pytest.mark.parametrize("bad", [False, 0, 1, "true", None])
def test_wire_declaration_accepts_only_literal_true(store, database, bad):
    start(store)
    rt, _ = waiting_runtime(database)
    kernel.drive(store, "s", runtime=rt)
    parked = rows(store, kind="op_parked")[0].record
    with pytest.raises(RecordError):
        replace(parked, payload=parked.payload | {"yield_on_steer": bad}).encode()
    provider = YieldProvider()
    prt, _ = provider_runtime(database, provider)
    # The native acceptance API accepts False, but its wire representation omits it.
    plan = next(op.attempts[1].plan for op in kernel.read_state(store, "s")[0].operations.values() if op.kind == "tool")
    provider.submit(plan, context=None)
    payload = {k: provider.handle[k] for k in ("operation_id", "attempt", "request_digest")}
    payload.update(handle=provider.handle, yield_on_steer=bad)
    with pytest.raises(RecordError):
        InboxItem("bad", "provider_evidence", payload, plan.turn_id, 0).encode()
    if type(bad) is not bool:
        with pytest.raises(ProviderResultInvalid):
            validate_acceptance(plan, Accepted(provider.handle, yield_on_steer=bad))
    assert prt.providers["child_task"] is provider


@pytest.mark.parametrize("mutation", ["extra", "undeclared_members", "duplicate_members", "foreign_member", "before_dispatch"])
def test_park_declaration_closed_shape_and_membership(store, database, mutation):
    start(store)
    rt, _ = waiting_runtime(database)
    kernel.drive(store, "s", runtime=rt)
    retained = [r.record for r in rows(store)]
    index = next(i for i, r in enumerate(retained) if r.kind == "op_parked")
    parked = retained[index]
    payload = parked.payload
    if mutation == "extra":
        payload["yield_mode"] = "steer"
        with pytest.raises(RecordError):
            replace(parked, payload=payload).encode()
        return
    if mutation == "undeclared_members":
        payload.pop("yield_on_steer")
    elif mutation == "duplicate_members":
        payload["yieldable_children"] *= 2
    elif mutation == "foreign_member":
        payload["yieldable_children"] = ["not-admitted"]
    else:
        payload["phase"] = "before_dispatch"
    retained[index] = replace(parked, payload=payload)
    with pytest.raises(TransitionError):
        fold(retained)


@pytest.mark.parametrize("action", ["cancel", "abort", "suspend", "human"])
def test_control_and_human_wait_block_yield(store, database, action):
    start(store)
    rt, observed = waiting_runtime(database)
    if action == "human":
        rt.llm.steps[0] = LLMResponse(
            "", [ToolCall("child-call", "child_task", {}), ToolCall("ask", "ask_user", {"question": "Choose"})]
        )
    kernel.drive(store, "s", runtime=rt)
    if action != "human":
        with store.atomic() as tx:
            tx.push("s", InboxItem("control", "control", {"action": action}, "s/turn/initial", 0))
    steer(store)
    kernel.drive(store, "s", runtime=rt)
    assert not observed
    assert not any(r.record.payload.get("yielded_by") for r in rows(store))


def test_summary_child_renderer_uses_baseline_admission_signature(store, database):
    start(store)
    rt, observed = waiting_runtime(database)
    admit = rt.children["child_task"]
    admissions, rendered = [], []

    def summary_admit(plan, summary):
        admissions.append(summary)
        return admit(plan)

    def render(plan, handles, reason):
        rendered.append((handles, reason))
        return ToolExecutionResult(plan.payload["request"]["id"], "Summary wait interrupted")

    rt.children["child_task"] = SummaryChildTool(lambda _: False, summary_admit, render)
    kernel.drive(store, "s", runtime=rt)
    steer(store)
    kernel.drive(store, "s", runtime=rt)
    assert admissions == [None] and len(rendered) == len(observed) == 1


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(1000, tz=tz or UTC)


@pytest.mark.parametrize("provider_wait", [False, True])
def test_undeclared_wait_log_is_byte_identical_to_0341(monkeypatch, provider_wait):
    old = baseline_module("kernel", monkeypatch)
    monkeypatch.setattr("vv_agent.session.timing.time.monotonic_ns", lambda: 100_000_000)
    monkeypatch.setattr("vv_agent.prompt.builder.datetime", FixedDatetime)

    def produce(driver):
        with SQLiteStore.standalone(":memory:") as store:
            store.clock_sql = "1000000"
            store.install_schema()
            start(store)
            rt, _ = (
                provider_runtime(store, YieldProvider(declared=False))
                if provider_wait
                else waiting_runtime(store, declared=False)
            )
            driver(store, "s", runtime=rt)
            steer(store)
            driver(store, "s", runtime=rt)
            assert any(i.item.input_id == "steer" for i in store.peek_inbox("s"))
            if not provider_wait:
                complete(store, store, "yield-child-0")
                deliver(store, "yield-child-0")
                driver(store, "s", runtime=rt)
            return [r.record.encode() for r in rows(store)]

    assert produce(old.drive) == produce(kernel.drive)


def test_terminal_beyond_first_inbox_page_precedes_steer(store, database):
    start(store)
    rt, observed = waiting_runtime(database)
    kernel.drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    steer(store)
    with store.atomic() as tx:
        for index in range(255):
            tx.push("s", InboxItem(f"stale-{index}", "steer", {"content": "old"}, "s/turn/initial", 99))
    deliver(store, "yield-child-0")
    kernel.drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert not any(r.record.payload.get("yielded_by") for r in rows(store))
    applied = [r.record.payload["input"] for r in rows(store, kind="input_applied")]
    child_index = next(i for i, item in enumerate(applied) if item["kind"] == "child_result")
    steer_index = next(i for i, item in enumerate(applied) if item["input_id"] == "steer")
    assert child_index < steer_index


def test_yield_replay_does_not_redispatch_rebill_or_render(store, database):
    start(store)
    provider = YieldProvider(callback=True)
    renders = []
    render = provider.on_yield
    assert render is not None

    def observed_render(*args):
        renders.append(1)
        return render(*args)

    provider.on_yield = observed_render
    rt, requests = provider_runtime(database, provider)
    kernel.drive(store, "s", runtime=rt)
    steer(store)
    kernel.drive(store, "s", runtime=rt)
    before = [r.record.encode() for r in rows(store)]
    for _ in range(2):
        store._fold_cache = store._previous_prefix = None
        kernel.drive(store, "s", runtime=rt)
    # Every usage observation and result used for billing stays byte-identical.
    assert [r.record.encode() for r in rows(store)] == before
    assert provider.submits == 1 and provider.queries == provider.cancels == 0
    assert renders == [1] and len(requests) == 1
