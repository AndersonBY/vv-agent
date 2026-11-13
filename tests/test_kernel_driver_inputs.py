"""Actual input/start-turn records and SQL consume, with narrow compilation facts."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest


def produce(path, case):
    from sim.world import SimClock

    from vv_agent.agent import Agent
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.llm.scripted import ScriptedLLM
    from vv_agent.run_config import RunConfig
    from vv_agent.session._driver_scope import _Scope
    from vv_agent.session.kernel import _Driver
    from vv_agent.session.records import InboxItem, SessionSpec
    from vv_agent.session.runtime import Runtime
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.types import Message

    clock = SimClock()
    attributes = (
        {"seed": {"messages": [Message("user", "retained seed").to_dict()], "shared_state": {"nested": [True, 1]}}}
        if case == "seed"
        else None
    )
    content = {"text": "new text", "messages": [Message("user", "explicit input").to_dict()]} if case == "seed" else "work"
    item = InboxItem("i", "user", {"content": content}, target_turn_id=None)
    with SQLiteStore.standalone(str(path), clock=clock) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "p", "/tmp", attributes=attributes), consumers=())
            tx.push("s", item)
            if case == "cancel":
                tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, "s/turn/i"))
        runtime = Runtime(
            Agent("input", "Work."),
            RunConfig(session_memory_enabled=False),
            ResolvedModelConfig("test", "m", "m", "m", []),
            ScriptedLLM([]),
            store.heartbeat_store,
            clock=clock,
        )
        compiled = []
        original = runtime.compile

        def compile(*args, **kwargs):
            value = original(*args, **kwargs)
            compiled.append(value.to_dict())
            return value

        pytest.MonkeyPatch().setattr(runtime, "compile", compile)
        lease = store.acquire("s", owner="owner", ttl_ms=15000)
        assert lease is not None
        scope = _Scope(lease, runtime, now_ms=store._now())
        driver = _Driver(store, "s", runtime, scope)
        assert driver.apply_input()
        if case == "cancel":
            assert not driver.apply_input()
            pending = store.read_input("s", "cancel")
            assert pending is not None and pending.consumed_seq is None
        old_token = scope.token
        assert driver.start_turn() and scope.token is not old_token
        if case == "cancel":
            assert driver.apply_input() and driver.state.turns["s/turn/i"].cancelled

        class UnexpectedRuntime:
            def __getattr__(self, name):
                raise AssertionError(f"empty/admitted inputs accessed Runtime.{name}")

        pytest.MonkeyPatch().setattr(driver, "runtime", UnexpectedRuntime())
        assert not driver.apply_input() and not driver.start_turn()
        pytest.MonkeyPatch().setattr(driver, "runtime", runtime)
        definition = driver.state.turns["s/turn/i"].start._payload["definition"]
        return {
            "attributes": attributes,
            "item": item.to_dict(),
            "case": case,
            "compiled": compiled[0],
            "bindings": asdict(runtime._definition_bindings(driver.state.turns["s/turn/i"].start)),
            "definition": definition,
            "records": [row.record.encode().decode() for row in driver.records],
        }


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("case", ["plain", "seed", "cancel"])
def test_core_inputs_start_and_consume_match_actual_sdk(tmp_path, flags, case):
    data = produce(tmp_path / "sdk.sqlite", case)
    fixture = tmp_path / "facts.json"
    fixture.write_text(json.dumps(data))
    script = r"""
import importlib.abc, json, sys
from pathlib import Path
from threading import Event, Lock, local, get_ident

root = Path(sys.argv[1])
data = json.loads(Path(sys.argv[2]).read_text())
sys.path.insert(0, str(root / "src"))


class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (
            fullname in {"vv_agent", "vv_agent.kernel"}
            or fullname.startswith("vv_agent.kernel.")
            or fullname.split(".")[0] in sys.stdlib_module_names
        ), fullname


sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_requests import CommitRecords
from vv_agent.kernel._definition import DefinitionCache, DefinitionBindings
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._records import SessionSpec, InboxItem, copy_json
from vv_agent.kernel._tool_metadata import ToolMetadata


class Clock:
    def now_ms(self):
        return 1000000

    def monotonic(self):
        return 1000.0

    def monotonic_ns(self):
        return 1000000000000

    def sleep(self, seconds):
        raise AssertionError("unexpected sleep")


clock = Clock()


def unused(*args, **kwargs):
    raise AssertionError("unexpected compaction")


def fold():
    return Fold(normalize_memory_settings=unused, count_messages=unused)


class Scope(LeaseOwnership):
    turn_id = None
    generation = None


class Catalog:
    def planning_signature(self):
        return (0,)

    def has_tool(self, n):
        return any(t["function"]["name"] == n for t in data["definition"]["tools"])

    def has_schema(self, n):
        return self.has_tool(n)

    def has_executor(self, n):
        return n in data["definition"]["capabilities"]

    def tool_metadata(self, n):
        return ToolMetadata.from_dict(data["definition"]["capabilities"][n]) if self.has_executor(n) else None

    def list_openai_schemas(self, *, tool_names=None):
        return [copy_json(t) for t in data["definition"]["tools"] if tool_names is None or t["function"]["name"] in tool_names]


with SQLiteStore.standalone(sys.argv[3], fold_factory=fold, clock=clock) as store:
    store.install_schema()
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "/tmp", attributes=data["attributes"]), consumers=())
        tx.push("s", InboxItem(**data["item"]))
        if data["case"] == "cancel":
            tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, "s/turn/i"))
    scope = Scope(
        store.acquire("s", owner="owner", ttl_ms=15000),
        15000,
        0.25,
        clock=clock,
        stop=Event(),
        ready=Event(),
        lock=Lock(),
        poll_lock=Lock(),
        commit_lock=Lock(),
        latch_entry=local(),
        actor_identity=get_ident,
    )
    driver = CoreDriver(
        store,
        "s",
        scope,
        fold_factory=fold,
        clock=clock,
        observe_cost=lambda: None,
        hook=lambda *_: None,
        emit_phase=lambda *_: None,
    )
    cache = DefinitionCache()
    trace = []

    def run(steps):
        response = None
        while True:
            try:
                request = steps.send(response)
            except StopIteration as done:
                return done.value
            trace.append(type(request).__name__ if not isinstance(request, inputs.InputSetting) else request.value)
            if isinstance(request, inputs.CommitInput):
                driver.commit(request.records, (request.input_id,), guarded=True)
                response = None
            elif isinstance(request, CommitRecords):
                driver.commit(request.records, guarded=True)
                response = None
            elif isinstance(request, inputs.CompileTurn):
                assert request.content == ("new text" if data["case"] == "seed" else "work") and not request.control
                response = AgentTask.from_dict(copy_json(data["compiled"]))
            elif isinstance(request, inputs.CurrentDefinition):
                response = inputs.DefinitionFacts(cache, DefinitionBindings(**data["bindings"]), Catalog())
            elif request is inputs.InputSetting.BUDGET_LIMITS:
                response = None
            elif request is inputs.InputSetting.HANDLER_VERSION:
                response = "1"
            elif request is inputs.InputSetting.RESET_TOKEN:
                response = None
            else:
                raise AssertionError(type(request).__name__)

    assert run(inputs.apply_input(driver, one_turn=False)) and trace == ["CommitInput"]
    if data["case"] == "cancel":
        assert not run(inputs.apply_input(driver, one_turn=False))
    trace.clear()
    assert run(inputs.start_turn(driver))
    assert trace == ["CompileTurn", "CurrentDefinition", "budget_limits", "handler_version", "CommitRecords", "reset_token"]
    if data["case"] == "cancel":
        assert run(inputs.apply_input(driver, one_turn=False)) and driver.state.turns["s/turn/i"].cancelled
    assert [row.record.encode().decode() for row in driver.records] == data["records"]
    assert store.read_input("s", "i").consumed_seq == 2
    assert not store.peek_inbox("s")
    assert all(
        n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.") for n in sys.modules if n.startswith("vv_agent")
    )
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            *flags,
            "-B",
            "-c",
            script,
            str(Path(__file__).parents[1]),
            str(fixture),
            str(tmp_path / "core.sqlite"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def queued_driver(tmp_path):
    from session.test_recovery_matrix import runtime

    from vv_agent.session._driver_scope import _Scope
    from vv_agent.session.kernel import _Driver
    from vv_agent.session.records import InboxItem, SessionSpec
    from vv_agent.session.sqlite import SQLiteStore

    with SQLiteStore.standalone(str(tmp_path / "queued.sqlite")) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "p", "/tmp"), consumers=())
            tx.push("s", InboxItem("i", "user", {"content": "work"}))
        rt = runtime(tmp_path / "queued.sqlite", [])
        lease = store.acquire("s", owner="test", ttl_ms=rt.ttl_ms)
        assert lease is not None
        driver = _Driver(store, "s", rt, _Scope(lease, rt, now_ms=store._now()))
        assert driver.apply_input()
        yield driver


@pytest.mark.parametrize("error", [LookupError("compile failed"), StopIteration("compile stopped")])
def test_sdk_request_exceptions_preserve_identity(queued_driver, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    driver = queued_driver
    cause = ValueError("original cause")
    error.__cause__ = cause
    monkeypatch.setattr(driver.runtime, "compile", fail)
    before = [row.record.encode() for row in driver.records]
    with pytest.raises(type(error)) as caught:
        driver.start_turn()
    assert caught.value is error and caught.value.__cause__ is cause
    assert [row.record.encode() for row in driver.store.read("s").records] == before
    assert driver.state.active_turn_id is None


def test_model_selection_failure_is_rejected_inside_core_request_boundary(queued_driver, monkeypatch):
    from vv_agent.session.records import InboxItem

    driver = queued_driver

    def fail(*args):
        raise StopIteration("resolution failed")

    monkeypatch.setattr(driver.runtime, "model_resolver", fail)
    with driver.store.atomic() as tx:
        tx.push("s", InboxItem("selection", "model_selection", {"selection": {"model": "other", "route": {}}}))
    driver.refresh()
    assert driver.apply_input()
    applied = driver.state.applied_inputs["selection"]
    assert applied._payload["disposition"] == "rejected"
    assert applied._payload["reason"] == "model_selection_unresolvable"
    stored = driver.store.read_input("s", "selection")
    assert stored is not None and stored.consumed_seq == driver.records[-1].seq


@pytest.mark.parametrize("entry", ["input", "turn", "catalog"])
def test_real_ledger_and_catalog_stopiteration_preserves_original_error(queued_driver, monkeypatch, entry):
    from vv_agent.session.records import InboxItem

    driver = queued_driver
    failure = StopIteration("original boundary failure")
    cause = ValueError("original boundary cause")
    failure.__cause__ = cause

    def fail(*args, **kwargs):
        raise failure

    if entry == "catalog":
        monkeypatch.setattr(type(driver.runtime.registry), "planning_signature", fail)
    else:

        def hook(point, record):
            if point == "before_commit":
                raise failure

        monkeypatch.setattr(driver.runtime, "hook", hook)
    if entry == "input":
        with driver.store.atomic() as tx:
            tx.push("s", InboxItem("other", "follow_up", {"content": "later"}))
        driver.refresh()
    before = [row.record.encode() for row in driver.store.read("s").records]
    with pytest.raises(StopIteration) as caught:
        driver.apply_input() if entry == "input" else driver.start_turn()
    assert caught.value is failure and caught.value.__cause__ is cause
    assert [row.record.encode() for row in driver.store.read("s").records] == before
    if entry == "input":
        pending = driver.store.read_input("s", "other")
        assert pending is not None and pending.consumed_seq is None
