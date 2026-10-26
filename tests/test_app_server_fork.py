"""Completed-boundary forks through real processors and independent API nodes."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Barrier, Event

import pytest
from jsonschema import Draft202012Validator
from test_app_server_context import create
from test_app_server_distributed import node, request, submission

from vv_agent import ScriptedModelProvider
from vv_agent.app_server.schema import json_schema_bundle
from vv_agent.session.context import project_context
from vv_agent.session.records import digest
from vv_agent.types import LLMResponse


def turn(server, identity, expected=None, text=None):
    result = request(server, identity, "turn/start", submission(identity=identity, expected=expected, text=text or identity))
    assert "result" in result, result
    server.run_adapter.join()
    return result["result"]["turnId"]


def fork(server, identity="copy", **kwargs):
    return request(server, "fork-" + identity, "thread/fork", {"threadId": "thread", "forkId": identity, **kwargs})


def context(server, sid):
    state, records, _ = server.kernel.store.read_state(sid)
    return [m.to_dict() for m in project_context(records, state)]


@pytest.mark.parametrize("cut", ["through", "before", "last", "empty"])
def test_cut_points_lineage_and_independent_new_turn(tmp_path, cut):
    calls = []
    provider = ScriptedModelProvider.from_callback("test", "m", lambda r: calls.append(r) or LLMResponse("answer"))
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        create(a)
        first = turn(a, "one")
        first_context = context(a, "thread")
        second = turn(a, "two", first)
        original = a.kernel.store.read("thread")
        cut_params = (
            {"throughTurnId": first}
            if cut == "through"
            else {"beforeTurnId": second}
            if cut == "before"
            else {"beforeTurnId": first}
            if cut == "empty"
            else {}
        )
        result = fork(b, **cut_params, workspace={"snapshot": "host-prepared", "nested": {"id": 7}})["result"]
        sid = result["threadId"]
        assert sid == "fork/" + digest(["thread", "copy"])
        assert result["accepted"] and not result["replayed"]
        lineage = result["lineage"]
        copied = context(b, sid)
        assert lineage["contextDigest"] == digest(copied)
        assert lineage["cutTurnId"] == (None if cut == "empty" else second if cut == "last" else first)
        source_record = next(r for r in original.records if r.seq == lineage["sourceRecordSeq"])
        assert lineage["sourceRecordDigest"] == source_record.record.digest
        if cut in {"through", "before"}:
            assert copied == first_context
        elif cut == "empty":
            assert all(m["role"] == "system" for m in copied)
        else:
            assert copied == context(a, "thread")
        read = request(b, "read", "thread/read", {"threadId": sid})["result"]
        assert read["turns"] == [] and read["items"] == []
        assert read["thread"]["metadata"]["lineage"] == lineage
        assert read["thread"]["metadata"]["workspace"]["nested"] == {"id": 7}
        listing = request(b, "list", "thread/list", {})["result"]["threads"]
        assert next(t for t in listing if t["threadId"] == sid)["metadata"]["lineage"] == lineage
        view = request(b, "context", "thread/context/read", {"threadId": sid})["result"]
        assert view["lineage"] == lineage and view["counts"]["pendingInputs"] == 0
        Draft202012Validator(json.loads(json_schema_bundle()["ThreadForkResponse"])).validate(result)
        Draft202012Validator(json.loads(json_schema_bundle()["ThreadContextReadResponse"])).validate(view)
        assert fork(a, **cut_params, workspace={"snapshot": "host-prepared", "nested": {"id": 7}})["result"] == result | {
            "replayed": True
        }
        binding = []
        original_resolve = b.run_adapter._host.build_run_config
        b.run_adapter._host.build_run_config = lambda r: binding.append(r) or original_resolve(r)
        result_turn = request(b, "new", "turn/start", submission(sid, "new", text="fork prompt"))
        assert "result" in result_turn
        b.run_adapter.join()
        assert binding[-1].thread_id == sid
        assert binding[-1].metadata["workspace"] == {"snapshot": "host-prepared", "nested": {"id": 7}}
        assert calls[-1].messages[-1].content == "fork prompt"
        assert a.kernel.store.read("thread").records == original.records
        assert all(x["content"] != "fork prompt" for x in context(a, "thread"))
        nested = request(b, "nested", "thread/fork", {"threadId": sid, "forkId": "nested"})["result"]
        assert nested["lineage"]["sourceThreadId"] == sid


def test_replay_before_source_changes_and_conflicts_across_nodes(tmp_path):
    with ExitStack() as stack:
        provider = ScriptedModelProvider.from_callback("test", "m", lambda r: LLMResponse("answer"))
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        create(a)
        first = turn(a, "one")
        barrier = Barrier(2)

        def submit(s):
            barrier.wait(5)
            return fork(s)["result"]

        with ThreadPoolExecutor(2) as pool:
            receipts = list(pool.map(submit, [a, b]))
        assert sorted(x["replayed"] for x in receipts) == [False, True]
        receipt = receipts[0] | {"replayed": True}
        turn(a, "two", first)
        assert fork(b)["result"] == receipt
        for change in [{"workspace": {}}, {"throughTurnId": "unknown"}, {"beforeTurnId": first}]:
            assert fork(b, **change)["error"]["code"] == -32024
        assert fork(a, "unknown", throughTurnId="unknown")["error"]["code"] == -32602
        assert request(b, "missing", "thread/fork", {"threadId": "missing", "forkId": "x"})["error"]["code"] == -32020


def test_running_turn_pending_input_and_compaction_do_not_leak(tmp_path):
    entered, release = Event(), Event()

    def complete(r):
        if r.messages[-1].content == "running":
            entered.set()
            assert release.wait(10)
        return LLMResponse("answer")

    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", complete))
        create(a)
        first = turn(a, "one")
        expected = context(a, "thread")
        active = request(a, "active", "turn/start", submission(identity="running", expected=first, text="running"))["result"][
            "turnId"
        ]
        try:
            assert entered.wait(5)
            assert "result" in request(
                a, "pending", "turn/followUp", submission(identity="pending", expected=active, text="PENDING")
            )
            assert "result" in request(
                a, "compact", "thread/compact/start", {"threadId": "thread", "actionId": "pending-compact"}
            )
            source_before = a.kernel.store.read("thread")
            for selector in ("throughTurnId", "beforeTurnId"):
                error = fork(a, selector, **{selector: active})["error"]
                assert error["code"] == -32602 and "not ended" in error["message"]
            result = fork(a)["result"]
            assert context(a, result["threadId"]) == expected
            state, records, _ = a.kernel.store.read_state(result["threadId"])
            assert len(records) == 1 and not state.turns and not state.operations and not state.waits
            assert not a.kernel.store.peek_inbox(result["threadId"])
            assert a.kernel.store.read("thread").records == source_before.records
        finally:
            release.set()


@pytest.mark.parametrize(
    "bad",
    [
        {"throughTurnId": "x", "beforeTurnId": "y"},
        {"workspace": None},
        {"workspace": []},
        {"forkId": "bad/id"},
        {"model": "other"},
        {"instructions": "override"},
        {"throughTurnId": None},
    ],
)
def test_closed_params(tmp_path, bad):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        create(a)
        assert fork(a, **bad)["error"]["code"] == -32602
        schema = Draft202012Validator(json.loads(json_schema_bundle()["ClientRequest"]))
        assert list(
            schema.iter_errors(
                {"jsonrpc": "2.0", "id": 1, "method": "thread/fork", "params": {"threadId": "thread", "forkId": "copy", **bad}}
            )
        )


def test_authorization_precedes_reads_and_creation(tmp_path, monkeypatch):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        create(a)
        monkeypatch.setattr(a.run_adapter._host, "authorize", lambda *args: False)
        monkeypatch.setattr(a.kernel.store, "read", lambda *a, **k: pytest.fail("unauthorized read"))
        assert fork(a)["error"]["code"] == -32027
        monkeypatch.undo()


def test_fork_before_any_ended_turn_uses_only_creation_baseline(tmp_path):
    entered, release = Event(), Event()

    def complete(request):
        entered.set()
        assert release.wait(5)
        return LLMResponse("done")

    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", complete))
        create(a)
        idle = fork(a, "idle")["result"]
        assert idle["lineage"]["cutTurnId"] is None and idle["lineage"]["sourceRecordSeq"] == 1
        request(a, "run", "turn/start", submission())
        try:
            assert entered.wait(5)
            active = fork(a, "active")["result"]
            assert active["lineage"] == idle["lineage"]
            assert context(a, active["threadId"]) == context(a, idle["threadId"])
        finally:
            release.set()


def test_canonical_fork_fixture_is_produced_by_real_processor(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    produced = subprocess.run(
        [sys.executable, "-m", "scripts.app_server_a2_fixtures", "--out", str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert produced.returncode == 0, produced.stderr
    assert (tmp_path / "app_server_a2.json").read_bytes() == (root / "tests/fixtures/parity/app_server_a2.json").read_bytes()
