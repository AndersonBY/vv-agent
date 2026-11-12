"""Independent kernel review probes. No remote services or credentials."""

import hashlib
from dataclasses import replace

import pytest

from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem
from vv_agent.session.reducer import fold
from vv_agent.session.store import LeaseLost
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .conftest import open_store
from .test_batch_children import complete, deliver, rows
from .test_recovery_matrix import force_expire, runtime, start
from .test_yieldable_waits import YieldProvider, provider_runtime, steer, waiting_runtime


def yielded_rows(store):
    return [r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]


@pytest.mark.sqlite_file
def test_watermark_retry_impure_callback_uses_only_committed_result(store, database):
    calls = []

    def render(plan, handles, reason):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            # Concurrent new inbox input after the captured watermark, before commit.
            with open_store(database) as other, other.atomic() as tx:
                tx.push("s", InboxItem("extra-steer", "steer", {"content": "another steer"}, plan.turn_id, 0))
        return ToolExecutionResult(plan.payload["request"]["id"], f"render-{len(calls)}")

    start(store)
    rt, observed = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    retained = yielded_rows(store)
    print(
        "RETRY",
        {
            "callback_calls": calls,
            "yield_count": len(retained),
            "logged": retained[0].record.payload["result"]["content"],
            "model_tools": [m.content for m in observed[0].messages if m.role == "tool"],
        },
    )
    assert calls == [1, 2]
    assert len(retained) == 1
    assert retained[0].record.payload["result"]["content"] == "render-2"
    assert any(m.role == "tool" and m.content == "render-2" for m in observed[0].messages)
    assert not any(m.role == "tool" and m.content == "render-1" for m in observed[0].messages)


@pytest.mark.sqlite_file
def test_lost_lease_during_renderer_fences_all_yield_records(store, database):
    replacement = []

    def render(plan, handles, reason):
        force_expire(database)
        with open_store(database) as other:
            replacement.append(other.acquire("s", owner="replacement", ttl_ms=60000))
        return ToolExecutionResult(plan.payload["request"]["id"], "old writer")

    start(store)
    rt, _ = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    with pytest.raises(LeaseLost):
        drive(store, "s", runtime=rt)
    assert not yielded_rows(store)
    assert "steer" not in read_state(store, "s")[0].applied_inputs
    assert any(i.item.input_id == "steer" for i in store.peek_inbox("s"))
    store.release(replacement[0])
    restored, observed = waiting_runtime(database, declared=False)
    restored.llm.steps.pop(0)
    drive(store, "s", runtime=restored)
    assert len(yielded_rows(store)) == 1 and len(observed) == 1
    print("LEASE_FENCE both retained before replacement; recovery yielded exactly once")


@pytest.mark.sqlite_file
def test_undeclared_same_watermark_keeps_original_order(store, database):
    start(store)
    rt, _ = waiting_runtime(database, declared=False)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    steer(store)
    deliver(store, "yield-child-0")
    drive(store, "s", runtime=rt)
    applied = [r.record.payload["input"]["kind"] for r in rows(store, kind="input_applied")]
    assert applied.index("child_result") < applied.index("steer")
    assert not yielded_rows(store)
    print("UNDECLARED targeted steer remained deferred; natural child completion first")


@pytest.mark.sqlite_file
def test_duplicate_yielded_child_delivery_and_cold_replay(store, database):
    start(store)
    rt, _ = waiting_runtime(database, count=2)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    complete(store, database, "yield-child-1")
    for sid in ["yield-child-0", "yield-child-1"]:
        deliver(store, sid)
        deliver(store, sid)
    original = [i.item for i in store.peek_inbox("s") if i.item.kind == "child_result"]
    drive(store, "s", runtime=rt)
    with store.atomic() as tx:
        for item in original:
            tx.push("s", replace(item, input_id=item.input_id + "/duplicate"))
    drive(store, "s", runtime=rt)
    state, records, _ = read_state(store, "s")
    cold = fold(
        [r.record for r in records],
        consumed_inputs=[InboxItem(**r.record.payload["input"]) for r in records if r.record.kind == "input_applied"],
    )
    assert [a.yielded for op in state.operations.values() for a in op.attempts.values()] == [
        a.yielded for op in cold.operations.values() for a in op.attempts.values()
    ]
    assert len(yielded_rows(store)) == 1
    notices = [r for r in rows(store, kind="input_applied") if r.record.payload["reason"] == "background child notification"]
    assert len(notices) == 2
    assert all(cold.applied_inputs[i.input_id + "/duplicate"].payload["disposition"] == "noop" for i in original)
    print("CHILD_DEDUP exactly 2 notifications, 1 tool result; cold Attempt.yielded matches")


@pytest.mark.parametrize("kind,body", [("json", "not JSON"), ("file", "{}"), ("image", "[]"), ("error", '{"ok":true}')])
@pytest.mark.sqlite_file
def test_invalid_structured_renderer_should_fall_back(store, database, kind, body):
    def render(plan, handles, reason):
        return ToolExecutionResult(plan.payload["request"]["id"], body, metadata={"output_type": kind})

    start(store)
    rt, _observed = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    result = yielded_rows(store)[0].record.payload["result"]
    print("STRUCTURED_RENDERER", result)
    from vv_agent.session.providers import Definitive, ProviderResultInvalid, validate_provider_result

    state = read_state(store, "s")[0]
    receipt = yielded_rows(store)[0]
    plan = state.operations[receipt.record.operation_id].attempts[1].execution_plan
    with pytest.raises(ProviderResultInvalid):
        validate_provider_result(plan, Definitive(render(plan, [], "steer").to_dict()))
    print("STRUCTURED_RENDERER standard provider validator rejects the identical result")
    assert result["metadata"].get("yield_callback_failed") is True
    assert result["content"].startswith("Wait interrupted by new input")


@pytest.mark.sqlite_file
def test_pure_long_renderer_retry_should_retain_the_committed_body(store, database, tmp_path, monkeypatch):
    from vv_agent.session.children import child_delivery
    from vv_agent.session.kernel import _Driver
    from vv_agent.types import ToolArtifactRef
    from vv_agent.workspace.artifacts import verify_text_artifact
    from vv_agent.workspace.local import LocalWorkspaceBackend

    calls = []

    def render(plan, handles, reason):
        # A pure function of the verified handles. It changes because inputs change.
        body = "|".join(h["status"] for h in handles) + "\n" + "x" * 16000
        calls.append(body)
        return ToolExecutionResult(plan.payload["request"]["id"], body)

    start(store)
    rt, _observed = waiting_runtime(database, count=2, callback=render)
    rt.config = replace(rt.config, workspace=tmp_path, workspace_backend=LocalWorkspaceBackend(tmp_path))
    # Runtime initialized its workspace backend before the test config replacement.
    rt.workspace_backend = rt.config.workspace_backend
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    steer(store)
    completed_original = _Driver.completed
    injected = []

    def completed_and_arrival(driver, plan, outcome):
        result = completed_original(driver, plan, outcome)
        if driver.sid == "s" and outcome.result.get("content", "").startswith("running|running") and not injected:
            injected.append(1)
            # First artifact has already been prepared; child arrives before guarded append.
            with open_store(database) as other, other.atomic() as tx:
                child_delivery(other, tx, "yield-child-0")
        return result

    monkeypatch.setattr(_Driver, "completed", completed_and_arrival)
    drive(store, "s", runtime=rt)
    assert len(calls) == 2
    result = yielded_rows(store)[0].record.payload["result"]
    artifact = ToolArtifactRef.from_dict(result["artifact"])
    actual = rt.workspace_backend.read_text(artifact.path)
    print(
        "PURE_LONG_RETRY",
        {
            "rendered_prefixes": [s[:35] for s in calls],
            "logged_content_prefix": result["content"][:35],
            "artifact_prefix": actual[:35],
            "reference_sha": artifact.sha256,
            "actual_sha": hashlib.sha256(actual.encode()).hexdigest(),
            "resources": result.get("metadata", {}).get("app_server_resources"),
        },
    )
    assert actual == calls[-1], "committed recovery artifact still contains the discarded render"
    verify_text_artifact(rt.workspace_backend, artifact)


@pytest.mark.sqlite_file
def test_retrying_unknown_blocks_yield_until_definitive_result(store, database):
    from vv_agent.session.providers import Unknown
    from vv_agent.tools.function import FunctionTool
    from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata

    class Retrying(YieldProvider):
        submissions = 0

        def submit(self, plan, *, context):
            self.submissions += 1
            if self.submissions == 1:
                with open_store(database) as other:
                    steer(other)
                return Unknown("first attempt unconfirmed")
            assert not yielded_rows(store)
            assert "steer" not in read_state(store, "s")[0].applied_inputs
            return super().submit(plan, context=context)

    start(store)
    rt, observed = waiting_runtime(database)
    rt.agent.tools.append(
        FunctionTool(
            "retry_work",
            "Retry work",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda *_: ToolExecutionResult("", "unused"),
            tool_metadata=ToolMetadata(idempotency=ToolIdempotency.SUPPORTED),
        )
    )
    provider = Retrying(declared=False, callback=None)
    rt.providers["retry_work"] = provider
    rt.llm.steps[0] = LLMResponse("", [ToolCall("child-call", "child_task", {}), ToolCall("retry-call", "retry_work", {})])
    drive(store, "s", runtime=rt)
    unknowns = rows(store, kind="op_unknown")
    assert len(unknowns) == 1 and unknowns[0].record.payload["retry"] == "retry"
    assert provider.submissions == 2
    assert not observed and not yielded_rows(store)
    print("RETRY_UNKNOWN no yield in unknown/retry or undeclared accepted retry; steer retained")


@pytest.mark.sqlite_file
def test_after_commit_lease_replacement_reuses_yielded_receipt(store, database):
    callbacks = []

    def render(plan, handles, reason):
        callbacks.append(1)
        return ToolExecutionResult(plan.payload["request"]["id"], "yielded once")

    start(store)
    rt, _ = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    replacement_observed = []
    switched = []

    def hook(point, record):
        if point == "after_commit" and record and record.payload.get("yielded_by") and not switched:
            switched.append(1)
            force_expire(database)
            resumed, observed = waiting_runtime(database, declared=False, callback=render)
            resumed.llm.steps.pop(0)
            with open_store(database) as other:
                drive(other, "s", runtime=resumed)
            replacement_observed.extend(observed)

    rt.hook = hook
    with pytest.raises(LeaseLost):
        drive(store, "s", runtime=rt)
    assert callbacks == [1]
    assert len(yielded_rows(store)) == 1
    assert len(replacement_observed) == 1
    assert "steer" in read_state(store, "s")[0].applied_inputs
    print("AFTER_COMMIT_LEASE 1 renderer call, 1 receipt, replacement model saw retained steering")


@pytest.mark.parametrize("raises_on", [1, 2])
@pytest.mark.sqlite_file
def test_callback_failure_across_retry_matches_final_log_and_model(store, database, raises_on):
    calls = []

    def render(plan, handles, reason):
        calls.append(1)
        if len(calls) == 1:
            with open_store(database) as other, other.atomic() as tx:
                tx.push("s", InboxItem("extra", "steer", {"content": "extra"}, plan.turn_id, 0))
        if len(calls) == raises_on:
            raise ValueError("private renderer exception")
        return ToolExecutionResult(plan.payload["request"]["id"], "second choice")

    start(store)
    rt, observed = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    result = yielded_rows(store)[0].record.payload["result"]
    assert len(calls) == 2
    expected = "second choice" if raises_on == 1 else "Wait interrupted by new input"
    assert result["content"].startswith(expected)
    assert result.get("metadata", {}).get("yield_callback_failed", False) == (raises_on == 2)
    assert any(m.role == "tool" and m.content == result["content"] for m in observed[0].messages)
    assert "private renderer exception" not in str([r.record.payload for r in rows(store)])
    print("CALLBACK_FAILURE_RETRY", {"raises_on": raises_on, "retained_default": raises_on == 2, "model_equals_log": True})


@pytest.mark.sqlite_file
def test_yielded_child_notification_deferred_until_next_batch_boundary(store, database):
    from vv_agent.tools.function import function_tool

    @function_tool
    def parent_work() -> str:
        return "provider owns work"

    start(store)
    rt, _observed = waiting_runtime(database)
    provider = YieldProvider(declared=False, callback=None)
    rt.agent.tools.append(parent_work)
    rt.providers["parent_work"] = provider
    observer = rt.llm.steps[1]

    def parent_call(request):
        observer(request)
        return LLMResponse("", [ToolCall("new-work", "parent_work", {})])

    rt.llm.steps[1] = parent_call
    rt.llm.steps.append(LLMResponse("Parent waiting"))
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    drive(store, "s", runtime=rt)
    assert any(i.item.kind == "child_result" for i in store.peek_inbox("s"))
    assert not any(r.record.payload.get("reason") == "background child notification" for r in rows(store))
    h = provider.handle
    with store.atomic() as tx:
        tx.push(
            "s",
            InboxItem(
                "new-work-done",
                "provider_result",
                {
                    "operation_id": h["operation_id"],
                    "attempt": h["attempt"],
                    "request_digest": h["request_digest"],
                    "provider_binding": h["provider"],
                    "result": ToolExecutionResult("new-work", "done").to_dict(),
                    "usage": {},
                    "evidence": [h["evidence"]],
                },
                "s/turn/initial",
                0,
            ),
        )
    drive(store, "s", runtime=rt)
    assert len(yielded_rows(store)) == 1
    notices = [r for r in rows(store) if r.record.payload.get("reason") == "background child notification"]
    assert len(notices) == 1
    assert not any(i.item.kind == "child_result" for i in store.peek_inbox("s"))
    print("BACKGROUND_BOUNDARY child remained pending in next provider batch, then exactly 1 notification")


@pytest.mark.sqlite_file
def test_callback_receives_verified_result_while_sibling_waits_for_user(store, database):
    captured = []

    def render(plan, handles, reason):
        captured.append(handles)
        return ToolExecutionResult(plan.payload["request"]["id"], "Observed children")

    start(store)
    rt, observed = waiting_runtime(database, count=2, callback=render)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    child_rt = runtime(database, [LLMResponse("Need a human answer")])
    child_rt.config.no_tool_policy = "wait_user"
    drive(store, "yield-child-1", runtime=child_rt)
    assert read_state(store, "yield-child-1")[0].turns["yield-child-1/turn/start"].wait
    steer(store)
    drive(store, "s", runtime=rt)
    terminal = rows(store, "yield-child-0", "turn_ended")[0]
    assert captured[0][0]["status"] == "completed"
    assert captured[0][0]["result"] == terminal.record.payload["result"]
    assert captured[0][0]["terminal_seq"] == terminal.seq
    assert captured[0][0]["terminal_digest"] == terminal.record.digest
    assert captured[0][1]["status"] == "running"
    assert "result" not in captured[0][1]
    assert len(observed) == 1


class AcceptanceCut(BaseException):
    pass


@pytest.mark.sqlite_file
@pytest.mark.parametrize("declared", [False, True])
def test_recovered_acceptance_preserves_original_declaration(store, database, declared, monkeypatch):
    from vv_agent.kernel._canonical_json import canonical_json_bytes

    provider = YieldProvider(declared=declared, callback=True)
    # Authenticate every acceptance byte, including the declaration.
    accepted_bytes = []

    def authenticate(item, plan):
        return canonical_json_bytes(item.payload) in accepted_bytes

    monkeypatch.setattr(provider, "authenticate", authenticate)
    rt, observed = provider_runtime(database, provider)
    start(store)

    def cut(point, record):
        if point == "before_commit" and record and record.kind == "op_parked":
            raise AcceptanceCut()

    rt.hook = cut
    with pytest.raises(AcceptanceCut):
        drive(store, "s", runtime=rt)
    rt.hook = lambda *_: None
    state = read_state(store, "s")[0]
    op = next(op for op in state.operations.values() if op.kind == "tool")
    plan = op.attempts[1].execution_plan
    assert op.attempts[1].state == "started"
    payload = {key: provider.handle[key] for key in ("operation_id", "attempt", "request_digest")}
    payload["handle"] = provider.handle
    if declared:
        payload["yield_on_steer"] = True
    accepted_bytes.append(canonical_json_bytes(payload))
    with store.atomic() as tx:
        tx.push("s", InboxItem("recovered-acceptance", "provider_evidence", payload, plan.turn_id, 0))
    drive(store, "s", runtime=rt)
    park = rows(store, kind="op_parked")[-1].record
    assert park.payload.get("yield_on_steer", False) is declared
    # Authentic repeats are idempotent; a re-authenticated conflicting declaration is rejected.
    conflict = dict(payload)
    if declared:
        conflict.pop("yield_on_steer")
    else:
        conflict["yield_on_steer"] = True
    accepted_bytes.append(canonical_json_bytes(conflict))
    with store.atomic() as tx:
        tx.push("s", InboxItem("repeat", "provider_evidence", payload, plan.turn_id, 0))
        tx.push("s", InboxItem("conflict", "provider_evidence", conflict, plan.turn_id, 0))
    drive(store, "s", runtime=rt)
    state = read_state(store, "s")[0]
    assert state.applied_inputs["repeat"].payload["disposition"] == "noop"
    assert state.applied_inputs["conflict"].payload["disposition"] == "rejected"
    assert state.applied_inputs["conflict"].payload["reason"] == "acceptance declaration conflict"
    assert len(rows(store, kind="op_parked")) == 1
    steer(store)
    drive(store, "s", runtime=rt)
    assert bool(observed) is declared


@pytest.mark.sqlite_file
@pytest.mark.parametrize("declared", [False, True])
def test_query_cannot_replace_acceptance_declaration(store, database, declared, monkeypatch):
    from vv_agent.session.providers import Accepted, ProviderResultInvalid

    start(store)
    provider = YieldProvider(declared=declared)
    rt, _ = provider_runtime(database, provider)
    rt.poll_ms = 1
    drive(store, "s", runtime=rt)
    monkeypatch.setattr(provider, "query", lambda handle: Accepted(handle, yield_on_steer=not declared))
    from threading import Event

    Event().wait(0.02)
    with pytest.raises(ProviderResultInvalid):
        drive(store, "s", runtime=rt)
    assert rows(store, kind="op_parked")[-1].record.payload.get("yield_on_steer", False) is declared


def test_shared_tool_result_candidates_have_immutable_content_identity():
    from vv_agent.session.tool_results import prepare_tool_resources

    from .test_tool_result_artifacts import ArchiveWorkspace, produce

    backend = ArchiveWorkspace()
    plan, receipt, _ = produce("A" * 16000, backend=backend)
    first = receipt.payload["result"]
    second = prepare_tool_resources(plan, ToolExecutionResult("call-s1", "B" * 16000).to_dict(), backend)
    assert first["artifact"]["path"] != second["artifact"]["path"]
    assert backend.read_text(first["artifact"]["path"]) == "A" * 16000
    assert backend.read_text(second["artifact"]["path"]) == "B" * 16000
    assert second["metadata"]["app_server_resources"][0]["status"] == "retained"


def long_yield_candidate_worker(database, workspace, pipe):
    from vv_agent.workspace.local import LocalWorkspaceBackend

    with open_store(database) as store:

        def render(plan, handles, reason):
            body = "|".join(h["status"] for h in handles) + "\n" + "x" * 16000
            return ToolExecutionResult(plan.payload["request"]["id"], body)

        rt, _ = waiting_runtime(database, count=2, callback=render)
        rt.llm.steps.pop(0)
        rt.config = replace(rt.config, workspace=workspace, workspace_backend=LocalWorkspaceBackend(workspace))
        rt.workspace_backend = rt.config.workspace_backend

        def hook(point, record):
            if point == "before_commit" and record and record.payload.get("yielded_by"):
                pipe.send("candidate-saved")
                pipe.recv()

        rt.hook = hook
        drive(store, "s", runtime=rt)


@pytest.mark.persistent_store
def test_process_death_before_yield_commit_can_select_new_body(store, database, tmp_path):
    import multiprocessing

    from vv_agent.types import ToolArtifactRef
    from vv_agent.workspace.artifacts import verify_text_artifact
    from vv_agent.workspace.local import LocalWorkspaceBackend

    start(store)
    rt, _ = waiting_runtime(database, count=2)
    drive(store, "s", runtime=rt)
    steer(store)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=long_yield_candidate_worker, args=(database, tmp_path, child))
    process.start()
    child.close()
    try:
        assert parent.poll(20)
        assert parent.recv() == "candidate-saved"
        process.kill()
        process.join(10)
        assert process.exitcode == -9
        print("RECOVERY_KILL", {"worker": process.name, "exit": process.exitcode})
        artifact_root = LocalWorkspaceBackend(tmp_path)._artifact_root
        original = {p: p.read_bytes() for p in artifact_root.rglob("*.txt")}
        assert len(original) == 1
        assert not yielded_rows(store)
        assert any(i.item.input_id == "steer" for i in store.peek_inbox("s"))
        force_expire(database)
        complete(store, database, "yield-child-0")
        deliver(store, "yield-child-0")
        calls = []

        def render(plan, handles, reason):
            body = "|".join(h["status"] for h in handles) + "\n" + "x" * 16000
            calls.append(body)
            return ToolExecutionResult(plan.payload["request"]["id"], body)

        resumed, observed = waiting_runtime(database, count=2, callback=render)
        resumed.llm.steps.pop(0)
        resumed.config = replace(resumed.config, workspace=tmp_path, workspace_backend=LocalWorkspaceBackend(tmp_path))
        resumed.workspace_backend = resumed.config.workspace_backend
        drive(store, "s", runtime=resumed)
        raw = yielded_rows(store)[0].record.payload["result"]
        artifact = ToolArtifactRef.from_dict(raw["artifact"])
        assert calls[0].startswith("completed|running")
        assert resumed.workspace_backend.read_text(artifact.path) == calls[0]
        verify_text_artifact(resumed.workspace_backend, artifact)
        assert len(list(artifact_root.rglob("*.txt"))) == 2
        assert all(p.read_bytes() == body for p, body in original.items())
        assert len(observed) == 1
        # Committed recovery never re-renders or rewrites either candidate.
        drive(store, "s", runtime=resumed)
        assert len(calls) == 1
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        assert process.exitcode == -9
        print("RECOVERY_KILL", {"worker": process.name, "exit": process.exitcode})
        parent.close()


def acceptance_crash_worker(database, declared, pipe):
    with open_store(database) as store:
        provider = YieldProvider(declared=declared)
        rt, _ = provider_runtime(database, provider)

        def hook(point, record):
            if point == "before_commit" and record and record.kind == "op_parked":
                pipe.send(provider.handle)
                pipe.recv()

        rt.hook = hook
        drive(store, "s", runtime=rt)


@pytest.mark.persistent_store
@pytest.mark.parametrize("declared", [False, True])
def test_process_death_after_acceptance_before_park_restores_declaration(store, database, declared, monkeypatch):
    import multiprocessing

    from vv_agent.kernel._canonical_json import canonical_json_bytes

    start(store)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=acceptance_crash_worker, args=(database, declared, child))
    process.start()
    child.close()
    try:
        assert parent.poll(20)
        handle = parent.recv()
        process.kill()
        process.join(10)
        assert process.exitcode == -9
        print("RECOVERY_KILL", {"worker": process.name, "exit": process.exitcode})
        assert not rows(store, kind="op_parked")
        force_expire(database)
        provider = YieldProvider(declared=declared)
        provider.handle = handle
        payload = {k: handle[k] for k in ("operation_id", "attempt", "request_digest")}
        payload["handle"] = handle
        if declared:
            payload["yield_on_steer"] = True
        auth_bytes = canonical_json_bytes(payload)
        monkeypatch.setattr(provider, "authenticate", lambda item, plan: canonical_json_bytes(item.payload) == auth_bytes)
        rt, observed = provider_runtime(database, provider)
        rt.llm.steps.pop(0)
        with store.atomic() as tx:
            tx.push("s", InboxItem("recovered", "provider_evidence", payload, "s/turn/initial", 0))
        drive(store, "s", runtime=rt)
        assert rows(store, kind="op_parked")[-1].record.payload.get("yield_on_steer", False) is declared
        steer(store)
        drive(store, "s", runtime=rt)
        assert bool(observed) is declared
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        assert process.exitcode == -9
        print("RECOVERY_KILL", {"worker": process.name, "exit": process.exitcode})
        parent.close()


@pytest.mark.sqlite_file
@pytest.mark.parametrize("declared", [False, True])
def test_acceptance_authentication_binds_declaration_bytes(store, database, declared, monkeypatch):
    from vv_agent.kernel._canonical_json import canonical_json_bytes
    from vv_agent.session.records import digest

    start(store)
    provider = YieldProvider(declared=declared)
    rt, _ = provider_runtime(database, provider)

    def cut(point, record):
        if point == "before_commit" and record and record.kind == "op_parked":
            raise AcceptanceCut()

    rt.hook = cut
    with pytest.raises(AcceptanceCut):
        drive(store, "s", runtime=rt)
    rt.hook = lambda *_: None
    h = provider.handle
    payload = {k: h[k] for k in ("operation_id", "attempt", "request_digest")}
    payload["handle"] = h
    if declared:
        payload["yield_on_steer"] = True
    original_bytes = canonical_json_bytes(payload)
    monkeypatch.setattr(provider, "authenticate", lambda item, plan: canonical_json_bytes(item.payload) == original_bytes)
    altered = dict(payload)
    if declared:
        altered.pop("yield_on_steer")
    else:
        altered["yield_on_steer"] = True
    trusted = InboxItem("original", "provider_evidence", payload, "s/turn/initial", 0)
    untrusted = replace(trusted, input_id="altered", payload=altered)
    assert replace(trusted, payload=altered).encode() != trusted.encode()
    assert digest(altered) != digest(payload)
    with store.atomic() as tx:
        tx.push("s", untrusted)
        tx.push("s", trusted)
    drive(store, "s", runtime=rt)
    state = read_state(store, "s")[0]
    assert state.applied_inputs["altered"].payload["disposition"] == "rejected"
    assert state.applied_inputs["altered"].payload["reason"] == "untrusted or mismatched provider evidence"
    assert state.applied_inputs["original"].payload["disposition"] == "applied"
    assert rows(store, kind="op_parked")[-1].record.payload.get("yield_on_steer", False) is declared


@pytest.mark.sqlite_file
def test_unselected_candidate_not_projected_and_same_body_reuses(store, database, tmp_path, monkeypatch):
    from pathlib import PurePosixPath

    from vv_agent.kernel._canonical_json import canonical_json_bytes
    from vv_agent.session.context import project_context
    from vv_agent.session.kernel import _Driver
    from vv_agent.session.tool_results import prepare_tool_resources, project_tool_result

    original = _Driver.completed
    candidates = []

    def capture(driver, plan, outcome):
        receipts = original(driver, plan, outcome)
        if driver.sid == "s" and len(outcome.result.get("content", "")) > 12000:
            candidates.append((driver.runtime.workspace_backend, plan, outcome.result, receipts[0].payload["result"]))
        return receipts

    monkeypatch.setattr(_Driver, "completed", capture)
    test_pure_long_renderer_retry_should_retain_the_committed_body(store, database, tmp_path, monkeypatch)
    assert len(candidates) == 2
    backend, plan, raw, selected = candidates[1]
    orphan = candidates[0][3]["artifact"]
    state, retained, _ = read_state(store, "s")
    completion = next(r for r in retained if r.record.kind == "op_completed" and "yielded_by" in r.record.payload)
    descriptor = project_tool_result(completion)
    model_context = [m.to_dict() for m in project_context(retained, state)]
    projections = canonical_json_bytes(
        {"records": [r.record.to_dict() for r in retained], "context": model_context, "descriptor": descriptor}
    )
    assert orphan["path"].encode() not in projections
    assert PurePosixPath(orphan["path"]).name.encode() not in projections
    assert orphan["sha256"].encode() not in projections
    assert len(descriptor["resources"]) == 1
    assert descriptor["resources"][0]["sha256"] == selected["artifact"]["sha256"]
    files = {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in backend._artifact_root.rglob("*.txt")}
    repeated = prepare_tool_resources(plan, raw, backend)
    assert repeated == selected
    assert all((p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) == value for p, value in files.items())
    assert backend.read_text(orphan["path"]) == candidates[0][2]["content"]
    assert len(files) == 2
    assert backend.list_files(".vv-agent/artifacts", "**/*") == []
    assert not any(".vv-agent/artifacts/" in path for path in backend.list_files(".", "**/*"))
    print(
        "CANDIDATE_VISIBILITY",
        {
            "files": len(files),
            "resources": len(descriptor["resources"]),
            "orphan_in_records_or_context_or_resources": False,
            "same_body_same_path": True,
            "same_body_no_rewrite": True,
        },
    )
