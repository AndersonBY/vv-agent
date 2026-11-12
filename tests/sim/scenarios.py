"""Bounded recovery scenarios using real drive(), provider receipts and child delivery."""

from __future__ import annotations

import json
from dataclasses import replace

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.session.children import ChildSession, child_delivery, completion_input
from vv_agent.session.providers import Accepted, Definitive, Unknown
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import LeaseLost
from vv_agent.types import ToolExecutionResult

from .world import FaultyStore, SimWorld, require

SCENARIOS = (
    "tools",
    "unknown_idempotent",
    "unknown_unsafe",
    "model_unknown",
    "precommit",
    "postcommit",
    "reply_loss",
    "accepted",
    "accepted_lost",
    "late_result",
    "children",
    "cancel",
    "factory_not_ready",
    "lease_takeover",
    "definition_mismatch",
    "model_result_precommit",
    "metered_tool",
)


class Provider:
    """A scripted external service with durable truth outside the sk_* ledger."""

    def __init__(self, world):
        self.world = world
        self.store = world.connection("provider")
        with self.store.atomic():
            self.store._rows("CREATE TABLE sim_effect (key text PRIMARY KEY, handle text, result text, usage text)")
        self.ready = False
        self.queries = []

    def submit(self, plan, *, context):
        w = self.world
        key = context.idempotency_key
        w.calls.append((plan.session_id, plan.operation_id, plan.attempt, "tool", key))
        effect = key or f"effect/{len(w.calls)}"
        handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": effect,
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": f"receipt/{effect}",
            "query_ref": "sim_effect",
            "cancel_ref": None,
        }
        result = ToolExecutionResult(tool_call_id=plan.payload["request"]["id"], content="effect done").to_dict()
        with self.store.atomic():
            self.store._rows(
                "INSERT INTO sim_effect VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (effect, json.dumps(handle), json.dumps(result), json.dumps(TOOL_USAGE if w.scenario == "metered_tool" else {})),
            )
        if w.scenario in {"accepted", "accepted_lost", "metered_tool"}:
            return Accepted(handle)
        if w.scenario in {"unknown_idempotent", "unknown_unsafe", "late_result"}:
            return Unknown("external receipt lost")
        usage = json.loads(self.store._one("SELECT usage FROM sim_effect WHERE key=%s", (effect,))[0])
        return Definitive(result, (handle["evidence"],), usage=usage)

    def query(self, handle):
        self.queries.append(handle)
        retained, result, usage = self.store._one("SELECT handle,result,usage FROM sim_effect WHERE key=%s", (handle["job_id"],))
        if canonical_json_bytes(handle) != canonical_json_bytes(json.loads(retained)):
            return Unknown("query handle does not match external service")
        return Definitive(json.loads(result), (handle["evidence"],), usage=json.loads(usage)) if self.ready else Accepted(handle)

    def cancel(self, handle):
        return Unknown("no trusted stop receipt")

    def authenticate(self, item, plan):
        for handle, result, usage in self.store._rows("SELECT handle,result,usage FROM sim_effect"):
            h = json.loads(handle)
            if (
                item.target_turn_id != plan.turn_id
                or item.generation != 0
                or h["operation_id"] != plan.operation_id
                or h["attempt"] != plan.attempt
                or h["request_digest"] != plan.payload["request_digest"]
            ):
                continue
            if item.kind == "provider_evidence":
                return canonical_json_bytes(item.payload["handle"]) == canonical_json_bytes(h)
            expected = {k: h[k] for k in ("operation_id", "attempt", "request_digest")}
            expected.update(
                provider_binding=h["provider"], result=json.loads(result), usage=json.loads(usage), evidence=[h["evidence"]]
            )
            return canonical_json_bytes(item.payload) == canonical_json_bytes(expected)
        return False

    def callback(self, *, evidence=False):
        handle, result, usage = self.store._one("SELECT handle,result,usage FROM sim_effect ORDER BY key LIMIT 1")
        h = json.loads(handle)
        payload = {k: h[k] for k in ("operation_id", "attempt", "request_digest")}
        if evidence:
            kind, input_id = "provider_evidence", "evidence"
            payload["handle"] = h
        else:
            kind, input_id = "provider_result", "receipt"
            payload.update(
                provider_binding=h["provider"], result=json.loads(result), usage=json.loads(usage), evidence=[h["evidence"]]
            )
        return InboxItem(input_id, kind, payload, "s/turn/initial", 0)


TOOL_USAGE = {"prompt_tokens": 12000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 0}}


def scan_after_hint_loss(w, provider, children, *, ready_active=False):
    observer = w.connection("observer")
    if ready_active:
        state, rows, _ = observer.read_state("s")
        require(state.active_turn_id is not None and observer.is_runnable("s"), "I4", "hint loss did not reach READY ACTIVE work")
        w.events.append({"lost_hints": len(w.queue), "ready_active_head": rows[-1].seq})
    w.scan(provider=provider, children=children, participant="worker-1", rounds=4)
    require(observer.read_state("s")[0].active_turn_id is None, "I4", "fair SQL scan did not finish READY ACTIVE work")


def lease_takeover(w, provider):
    """Two queued workers contend; the winner yields at real driver hooks."""
    cut = w.choices.choose("lease/cut", ["planned", "before_send", "after_model"])
    owner = []
    raced = []
    prepared = []
    heartbeats = {}
    contenders = ("worker-0", "worker-1")

    def entry(participant):
        if not owner:
            owner.append(participant)
        result = w.drive(participant=participant, provider=provider)
        if owner and owner[0] != participant and not raced:
            require(result is False, "I2", "competing worker acquired an unexpired lease")
            w.events.append({"contender_refused": participant})

    for participant in contenders:
        rt = w.runtime("s", participant, provider=provider)
        start = rt.lease_keeper.start

        def capture_start(heartbeat, participant=participant, start=start):
            heartbeats[participant] = heartbeat
            start(heartbeat)

        rt.lease_keeper.start = capture_start
        original_hook = rt.hook

        def hook(point, record, participant=participant, original_hook=original_hook):
            is_model = record is not None and record.payload.get("op_kind") == "model"
            if point == "after_commit" and record.kind == "op_planned" and is_model and owner == [participant] and not prepared:
                prepared.append(True)
                heartbeat = heartbeats[participant]
                before = heartbeat.lease
                if w.summary()["backend"] == "sqlite":
                    w.advance(w.choices.choose("lease/renew-after-ms", [1, 17, 31]))
                # Reuse the keeper's renewal action at this real driver checkpoint.
                w.enqueue(f"renew/{participant}", lambda: w.keepers[participant].poll(heartbeat))
                # The renewal and any remaining initial participants run in seeded order.
                for _ in range(4):
                    if not w.step():
                        break
                current = w.connection("observer")._one(
                    "SELECT lease_epoch,lease_owner,lease_until_ms FROM sk_session WHERE session_id='s'"
                )
                require(
                    current == (before.epoch, before.owner, heartbeat.lease.expires_at_ms), "I2", "renewal changed lease identity"
                )
                require(heartbeat.lease is not before, "I2", "scheduled renewal never reached the real store")
                w.events.append({"renewed_owner": participant, "epoch": before.epoch, "until": current[2]})
            matches = (
                (cut == "planned" and point == "after_commit" and record.kind == "op_planned" and is_model)
                or (cut == "before_send" and point == "before_external_call" and is_model)
                or (cut == "after_model" and point == "after_external_call" and is_model)
            )
            if owner == [participant] and matches and not raced:
                raced.append(cut)
                heartbeat = heartbeats[participant]
                successor = next(p for p in contenders if p != participant)
                w.events.append({"takeover_cut": cut, "owner": participant})

                def expire_then_take_over():
                    w.expire()
                    w.enqueue("lease/takeover", lambda: w.drive(participant=successor, provider=provider))

                w.enqueue("lease/expire", expire_then_take_over)
                for _ in range(4):
                    if not w.step():
                        break
                try:
                    w.keepers[participant].poll(heartbeat)
                except LeaseLost:
                    w.events.append({"stale_renewal_rejected": participant})
                else:
                    require(False, "I2", "old owner renewed after takeover")
            original_hook(point, record)

        rt.hook = hook
        w.workers[participant] = FaultyStore(w.connection(participant), w, participant)
        w.runtimes[("s", participant)] = rt
        w.enqueue(f"worker/{participant}", lambda participant=participant: entry(participant))
    for _ in range(8):
        if not w.step():
            break
    require(bool(raced) and not w.tasks, "scheduler", "lease schedule did not reach/drain its cut")
    require(any("contender_refused" in e for e in w.events), "I2", "no real competing claim was attempted")
    rows = w.connection("observer").read("s").records
    require(len({r.writer_epoch for r in rows if r.writer_epoch}) >= 2, "I2", "takeover did not commit a new epoch")


def reject_forged_tool_receipts(w, provider):
    valid = provider.callback()
    forged_usage = replace(valid, input_id="usage-forged", payload=valid.payload | {"usage": {"prompt_tokens": True}})
    forged_result = replace(valid, input_id="result-forged", payload=valid.payload | {"result": {"content": "invented"}})
    for item in (forged_usage, forged_result):
        w.push("s", item)
    w.drive(provider=provider, participant="worker-1")
    state, rows, _ = w.connection("observer").read_state("s")
    for item in (forged_usage, forged_result):
        require(
            state.applied_inputs[item.input_id].payload["disposition"] == "rejected",
            "I10",
            "external SQL truth did not reject forged settlement",
        )
    require(
        not any(r.record.kind == "op_completed" and "/tool/" in (r.record.operation_id or "") for r in rows),
        "I10",
        "forged receipt settled parked tool",
    )
    w.push("s", valid)
    w.push("s", replace(valid, input_id="usage-identical"))
    provider.ready = True


def check_metered_tool(w, provider):
    state, rows, _ = w.connection("observer").read_state("s")
    completion = next(r.record for r in rows if r.record.kind == "op_completed" and "/tool/" in (r.record.operation_id or ""))
    measured = json.loads(provider.store._one("SELECT usage FROM sim_effect")[0])
    require(
        canonical_json_bytes(completion.payload["usage"]) == canonical_json_bytes(measured),
        "I5",
        "tool usage differs from external SQL receipt",
    )
    require(
        state.applied_inputs["usage-identical"].payload["disposition"] == "noop",
        "I5",
        "identical metered receipt was charged again",
    )
    before = completion.encode()
    # The service issues an authenticated conflicting report for the same attempt.
    # Only its real service row changes; committed accounting must stay immutable.
    with provider.store.atomic():
        provider.store._rows("UPDATE sim_effect SET usage=%s", (json.dumps(measured | {"prompt_tokens": 12001}),))
    conflict = replace(provider.callback(), input_id="usage-conflict")
    plan = state.operations[completion.operation_id].attempts[completion.attempt].execution_plan
    require(provider.authenticate(conflict, plan), "I5", "conflict case never reached authenticated receipt admission")
    w.push("s", conflict)
    w.drive(provider=provider, participant="worker-1")
    state, rows, _ = w.connection("observer").read_state("s")
    applied = state.applied_inputs[conflict.input_id].payload
    require(
        applied["disposition"] == "rejected" and applied["reason"] == "result conflict",
        "I5",
        "authenticated conflicting usage replaced the original receipt",
    )
    completed = [r.record for r in rows if r.record.kind == "op_completed" and r.record.operation_id == completion.operation_id]
    require(len(completed) == 1 and completed[0].encode() == before, "I5", "conflicting redelivery changed/billed the tool twice")
    w.covered.update({"I5", "I10"})


def run_scenario(world: SimWorld, scenario=None):
    w = world
    w.scenario = w.choices.choose("scenario", [scenario] if scenario else SCENARIOS)
    w.install_write_observer()
    w.create()
    w.push("s", InboxItem("initial", "user", {"content": "go"}))
    w.stale_writer_probe()
    provider = Provider(w)
    children = {}
    if w.scenario in {"children", "cancel"}:
        children = {"effect": lambda _plan: ChildSession(SessionSpec("child", "sim", "/tmp"), "child work")}
    fault = (w.scenario, "op_completed", "tool") if w.scenario in {"precommit", "postcommit", "reply_loss"} else None
    if w.scenario == "tools":
        fault = ("postcommit", "op_planned", "model")
    elif w.scenario == "model_result_precommit":
        fault = ("precommit", "op_completed", "model")
    elif w.scenario == "definition_mismatch":
        fault = ("postcommit", "op_completed", "tool")
    if w.scenario == "accepted_lost":
        w.after_external = lambda worker: worker.die("accepted before ledger commit")
    # An independent host and scanner can run while a real driver is at a safe hook.
    w.enqueue("host/duplicate", lambda: w.push("s", InboxItem("initial", "user", {"content": "go"})))
    w.enqueue("transport/wake", lambda: w.queue.extend(["s", "s"]))
    if w.scenario == "factory_not_ready":
        w.drive(provider=provider, factory=w.not_ready)
        require(not w.calls, "factory", "RuntimeNotReady dispatched externally")
        w.advance()
    if w.scenario == "lease_takeover":
        lease_takeover(w, provider)
    else:
        w.enqueue("worker/first", lambda: w.deliver(provider=provider, children=children, fault=fault))
    for _ in range(8):
        if not w.step():
            break
    require(not w.tasks, "scheduler", "initial participant schedule did not drain")
    w.check()
    if w.scenario in {"tools", "precommit", "postcommit", "accepted_lost", "model_result_precommit", "definition_mismatch"}:
        w.expire()
    if w.scenario == "accepted_lost":
        w.push("s", provider.callback(evidence=True))
    if w.scenario in {"accepted", "accepted_lost"}:
        # Evidence and the result remain pending until the scanner alone recovers.
        provider.ready = True
        w.push("s", provider.callback())
    if w.scenario == "definition_mismatch":
        rt = w.runtime("s", "worker-1", provider=provider)
        schema = json.loads(json.dumps(rt.registry.get_schema("effect")))
        schema["function"]["description"] = "Changed tool definition after the turn was frozen."
        rt.registry.register_schema("effect", schema)
        if "worker-1" not in w.workers:
            w.workers["worker-1"] = FaultyStore(w.connection("worker-1"), w, "worker-1")
        w.runtimes[("s", "worker-1")] = rt
    if w.scenario == "metered_tool":
        reject_forged_tool_receipts(w, provider)
    if w.scenario == "late_result":
        before = sum(c[3] == "tool" for c in w.calls)
        w.drive(provider=provider, participant="worker-1")
        require(sum(c[3] == "tool" for c in w.calls) == before, "I10", "unsafe unknown retried on recovery")
        w.push("s", provider.callback())
        w.push("s", replace(provider.callback(), input_id="late-duplicate"))
    if children:
        handle = next(iter(w.connection("observer").read_state("s")[0].waits.values()))["handle"]
        if w.scenario == "cancel":
            w.push("s", InboxItem("cancel", "control", {"action": "cancel"}, "s/turn/initial"))
            w.drive(provider=provider, children=children)
        w.drive("child", participant="child-worker")
        child_state, rows, _ = w.connection("observer").read_state("child")
        terminal = next(r for r in rows if r.record.kind == "turn_ended")
        delivery = w.connection("delivery")
        for _ in range(2):
            with delivery.atomic() as tx:
                child_delivery(delivery, tx, "child")
        w.push("s", replace(completion_input(handle, terminal), input_id="child-redelivery"))
        forged = replace(
            completion_input(handle, terminal),
            input_id="child-forged",
            payload={
                **completion_input(handle, terminal).payload,
                "session_id": "s",
            },
        )
        w.push("s", forged)
        w.drive(provider=provider, children=children)
        parent = w.connection("observer").read_state("s")[0]
        require(parent.applied_inputs["child-forged"].payload["disposition"] == "rejected", "I6", "forged child accepted")
        require(parent.applied_inputs["child-redelivery"].payload["disposition"] == "noop", "I6", "child result applied twice")
        if w.scenario == "cancel":
            require(
                child_state.active_turn_id is None and terminal.record.payload["status"] == "cancelled",
                "I7",
                "descendant survived cancel",
            )
        w.covered.update({"I6", "I7"} if w.scenario == "cancel" else {"I6"})
    before_recovery = tuple(w.calls)
    scan_after_hint_loss(
        w,
        provider,
        children,
        ready_active=w.scenario in {"tools", "accepted_lost", "model_result_precommit", "definition_mismatch", "metered_tool"},
    )
    state, rows, _ = w.connection("observer").read_state("s")
    require(state.active_turn_id is None, "I4", "fair SQL scan did not finish this bounded scenario")
    if w.scenario == "definition_mismatch":
        terminal = next(r.record for r in rows if r.record.kind == "turn_ended")
        require(
            terminal.payload["status"] == "failed" and terminal.payload["reason"] == "handler_schema_or_capability_mismatch",
            "definition",
            "changed runtime did not fail its frozen definition check",
        )
        require(tuple(w.calls) == before_recovery, "definition", "definition-mismatched worker sent external work")
    if w.scenario == "model_result_precommit":
        model = next(op for op in state.operations.values() if op.kind == "model")
        first = model.attempts[1]
        require(
            first.unknown is not None
            and first.result is None
            and first.unknown.payload["measurement_missing"]
            and first.unknown.payload["duplicate_cost_risk"],
            "I5",
            "rolled-back model receipt lost its missing measurement/risk",
        )
        require(
            set(model.attempts) == {1, 2} and model.attempts[2].result is not None,
            "I1",
            "lost model result did not use one legal new attempt",
        )
    if w.scenario == "metered_tool":
        check_metered_tool(w, provider)
    effects = provider.store._one("SELECT count(*) FROM sim_effect")[0]
    calls = sum(c[3] == "tool" for c in w.calls)
    if w.scenario == "unknown_idempotent":
        require(calls == 2 and effects == 1, "I1", "idempotent retry was not deduplicated by the external service")
    elif w.scenario == "model_unknown":
        require(len(w.calls) == 2 and effects == 0, "I1", "ambiguous model did not stop at retry bound")
    elif w.scenario not in {"children", "cancel"}:
        require(calls == effects == 1, "I1", "completed or unsafe operation was redispatched")
    # Delayed duplicate broker deliveries really reach drive(), after completion.
    w.queue.extend(["s", "s"])
    before = tuple(w.calls)
    for _ in range(2):
        w.deliver(provider=provider, children=children, participant="worker-1")
    require(tuple(w.calls) == before, "I1", "delayed duplicate wakes repeated external work")
    w.queue.clear()
    w.check(quiescent=True)
    w.check_accounting()
    return w.summary()
