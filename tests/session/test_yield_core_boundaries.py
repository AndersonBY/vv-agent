"""Closed declarations and undeclared waits at the current S1 storage boundary."""

from dataclasses import replace

import pytest

from vv_agent.session import kernel
from vv_agent.session.children import SummaryChildTool
from vv_agent.session.providers import Accepted, ProviderResultInvalid, validate_acceptance
from vv_agent.session.records import InboxItem, RecordError
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .helpers import push_fixture_input, read_fixture_state, wrap_fixture_store
from .test_batch_children import complete, deliver, rows
from .test_recovery_matrix import start
from .test_yieldable_waits import YieldProvider, provider_runtime, steer, waiting_runtime


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


@pytest.mark.parametrize("bad", [False, 0, 1, "true", None])
def test_wire_declaration_accepts_only_literal_true(store, database, bad):
    start(store)
    rt, _ = waiting_runtime(store, database)
    kernel.drive(store, "s", runtime=rt)
    parked = rows(store, kind="op_parked")[0].record
    with pytest.raises(RecordError):
        replace(parked, payload=parked.payload | {"yield_on_steer": bad}).encode()
    provider = YieldProvider()
    prt, _ = provider_runtime(store, database, provider)
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
    rt, _ = waiting_runtime(store, database)
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
    rt, observed = waiting_runtime(store, database)
    if action == "human":
        rt.llm.steps[0] = LLMResponse(
            "", [ToolCall("child-call", "child_task", {}), ToolCall("ask", "ask_user", {"question": "Choose"})]
        )
    kernel.drive(store, "s", runtime=rt)
    if action != "human":
        push_fixture_input(store, "s", InboxItem("control", "control", {"action": action}, "s/turn/initial", 0))
    steer(store)
    kernel.drive(store, "s", runtime=rt)
    assert not observed
    assert not any(r.record.payload.get("yielded_by") for r in read_fixture_state(store, "s")[1])


def test_summary_child_renderer_uses_baseline_admission_signature(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
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


@pytest.mark.parametrize("provider_wait", [False, True])
def test_current_s1_undeclared_wait_does_not_yield_or_redispatch(store, database, provider_wait):
    start(store)
    provider = YieldProvider(declared=False)
    rt, observed = (
        provider_runtime(store, database, provider) if provider_wait else waiting_runtime(store, database, declared=False)
    )
    kernel.drive(store, "s", runtime=rt)
    park = rows(store, kind="op_parked")[-1].record
    assert park.payload.get("yield_on_steer", False) is False

    steer(store)
    kernel.drive(store, "s", runtime=rt)
    state, hydrated, _ = read_fixture_state(store, "s")
    assert not observed
    assert "steer" not in state.applied_inputs
    assert any(i.item.input_id == "steer" for i in store.peek_inbox("s"))
    assert not any("yielded_by" in row.record.payload for row in hydrated)

    if provider_wait:
        assert provider.submits == 1
        kernel.drive(store, "s", runtime=rt)
        assert provider.submits == 1 and not observed
    else:
        complete(store, database, "yield-child-0")
        deliver(store, "yield-child-0")
        kernel.drive(store, "s", runtime=rt)
        assert len(observed) == 1
        kernel.drive(store, "s", runtime=rt)
        assert len(observed) == 1


def test_terminal_beyond_first_inbox_page_precedes_steer(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    kernel.drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    steer(store)
    actor = store.read_input("s", "initial").item.actor
    for index in range(255):
        push_fixture_input(
            store,
            "s",
            InboxItem(f"stale-{index}", "steer", {"content": "old"}, "s/turn/initial", 99, actor=actor),
            preserve_actor=True,
        )
    deliver(store, "yield-child-0")
    kernel.drive(store, "s", runtime=rt)
    assert len(observed) == 1
    hydrated = read_fixture_state(store, "s")[1]
    assert not any(r.record.payload.get("yielded_by") for r in hydrated)
    applied = [r.record.payload["input"] for r in hydrated if r.record.kind == "input_applied"]
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
    rt, requests = provider_runtime(store, database, provider)
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
