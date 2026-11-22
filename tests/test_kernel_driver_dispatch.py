"""Actual dispatch/receipt algorithms with real SQL and a separate schema host."""

import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest


def _produced_prefix(tmp_path, mode, monkeypatch, configure=None):
    from hashlib import sha256

    from session.controlled import ControlledProvider
    from sim.world import SimClock
    from support.s1_fixture_values import create_fixture_session, push_fixture_input

    from vv_agent import Agent, RunConfig, function_tool
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.kernel import _driver_dispatch as core
    from vv_agent.kernel import _tool_admission as admission
    from vv_agent.kernel._content import KeyDestroyOutcome, KeyDestroyStatus, WrappedDataKey
    from vv_agent.kernel._content_runtime import (
        bind_content_runtime,
        durable_content_reader,
        wrap_session_content_store,
    )
    from vv_agent.llm.scripted import ScriptedLLM
    from vv_agent.run_config import ToolPolicy
    from vv_agent.session import _driver_dispatch as host
    from vv_agent.session import _recovery_phase
    from vv_agent.session._driver_scope import _Scope
    from vv_agent.session.kernel import _Driver, drive
    from vv_agent.session.providers import Definitive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.runtime import Runtime
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
    from vv_agent.types import LLMResponse, ToolCall

    calls = []
    contexts = []

    class FixtureKeyProvider:
        provider_id = "dispatch-fixture-v1"

        @staticmethod
        def _key(tenant_id: str, session_id: str) -> bytes:
            return sha256(f"{tenant_id}\x00{session_id}".encode()).digest()

        def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
            return WrappedDataKey(tenant_id, session_id, f"key/{tenant_id}/{session_id}", self._key(tenant_id, session_id))

        def unwrap(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
            assert (wrapped_key.tenant_id, wrapped_key.session_id) == (tenant_id, session_id)
            return wrapped_key.wrapped_key

        def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome:
            return KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, f"destroy/{tenant_id}/{key_id}")

    class FixtureCipher:
        def __init__(self, counter: int = 0) -> None:
            self.counter = counter

        def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
            del key, nonce, aad
            return b"dispatch-cipher" + plaintext

        def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
            del key, nonce, aad
            assert ciphertext.startswith(b"dispatch-cipher")
            return ciphertext.removeprefix(b"dispatch-cipher")

        def random_bytes(self, n: int) -> bytes:
            output = b""
            while len(output) < n:
                output += sha256(f"dispatch-cipher:{self.counter}".encode()).digest()
                self.counter += 1
            return output[:n]

    @function_tool(tool_metadata=ToolMetadata(idempotency=ToolIdempotency.UNSUPPORTED))
    def effect(count: int = 1) -> str:
        calls.append(count)
        return "ok"

    class Planned(BaseException):
        pass

    clock = SimClock()
    database = tmp_path / "sdk.sqlite"
    core_database = tmp_path / "core.sqlite"
    data = {"mode": mode, "clock_ms": clock.now_ms(), "resources_noop": True}
    with SQLiteStore.standalone(str(database), clock=clock) as raw_store:
        raw_store.install_schema()
        cipher = FixtureCipher()
        binding = bind_content_runtime(raw_store, key_provider=FixtureKeyProvider(), cipher=cipher)
        store = wrap_session_content_store(raw_store, binding)
        create_fixture_session(
            store,
            "s",
            attributes={
                "seed": {
                    "messages": [],
                    "shared_state": {},
                    "unseen_input_protection": True,
                    "response_rejection": True,
                }
            },
        )
        push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
        runtime = Runtime(
            Agent("dispatch", "Work.", tools=[effect]),
            RunConfig(session_memory_enabled=False, workspace=tmp_path),
            ResolvedModelConfig("test", "m", "m", "m", []),
            ScriptedLLM(
                [LLMResponse("model result")]
                if mode == "model"
                else [
                    LLMResponse("", [ToolCall("a", "effect", {"count": "bad"} if mode == "schema" else {"count": 1})]),
                ]
            ),
            store.heartbeat_store,
            clock=clock,
        )
        runtime.content_reader = durable_content_reader(store, tenant_id=raw_store.tenant_id, session_id="s")
        if mode == "unknown":
            provider = ControlledProvider(database, mode="unknown")
            provider.install()
            runtime.providers["effect"] = provider
        target = "model" if mode == "model" else "tool"

        def cut(point, record):
            if point == "after_commit" and record and record.kind == "op_planned" and record._payload["op_kind"] == target:
                raise Planned

        runtime.hook = cut
        with pytest.raises(Planned):
            drive(store, "s", runtime=runtime)
        runtime.hook = lambda *_: None
        if mode == "policy":
            runtime.config.tool_policy = ToolPolicy(disallowed_tools=["effect"])
        if configure is not None:
            configure(runtime, effect)
        with sqlite3.connect(core_database) as copied:
            raw_store.connection.backup(copied)
        data["cipher_calls"] = cipher.counter
        lease = store.acquire("s", owner="dispatch-reader", ttl_ms=15000)
        assert lease is not None
        scope = _Scope(lease, runtime, now_ms=raw_store._now())
        scope.start()
        try:
            assert scope.ready.wait(runtime.ttl_ms / 1000)
            assert scope.successful_ready
            driver = _Driver(store, "s", runtime, scope)
            driver.bind_turn()
            operation = next(op for op in driver.state.operations.values() if op.kind == target and op.selected_attempt is None)
            attempt = operation.attempts[max(operation.attempts)]
            plan = attempt.execution_plan
            data.update(
                oid=plan.operation_id,
                attempt=plan.attempt,
                prefix_seq=driver.records[-1].seq,
                handler_version=runtime.handler_version,
            )
            handle = host._DispatchHost._handle

            def observe(self, request):
                if isinstance(request, core.SelectProvider):
                    assert "_vv_agent_tool_dispatch_callback" not in self.context.metadata
                response = handle(self, request)
                if isinstance(request, core.FinishContext):
                    contexts.append(self.context)
                    data["context"] = {
                        "metadata": dict(response),
                        "cycle": self.context.cycle_index,
                        "shared": runtime.durable_state(self.context.shared_state),
                    }
                elif request is admission.Read.SCHEMA:
                    data["schema"] = response
                elif request is admission.Read.POLICY_METADATA:
                    data["policy"] = response
                elif request is admission.Read.TOOL_METADATA:
                    data["tool_metadata"] = response.to_dict() if response is not None else None
                elif isinstance(request, core.InvokeOperation):
                    data["outcome"] = {
                        "type": "definitive" if isinstance(response, Definitive) else "unknown",
                        **asdict(response),
                    }
                return response

            monkeypatch.setattr(host._DispatchHost, "_handle", observe)
            resources = _recovery_phase.prepare_tool_resources

            def observe_resources(plan, raw, backend):
                result = resources(plan, raw, backend)
                assert result == raw
                return result

            monkeypatch.setattr(_recovery_phase, "prepare_tool_resources", observe_resources)
            driver.dispatch(attempt)
            if attempt.result is not None:
                data["result"] = driver.hydrate_record_field(
                    attempt.result,
                    "result",
                    attempt.result._payload["result"],
                )
            data["expected"] = [r.record.encode().decode() for r in driver.records if r.seq > data["prefix_seq"]]
            if configure is None:
                assert bool(calls) == (mode == "tool")
            data["effect_calls"] = calls
            if mode != "tool":
                assert all("_vv_agent_tool_dispatch_callback" not in context.metadata for context in contexts)
            if mode == "unknown":
                assert provider.counts() == (1, 1)
            if mode in {"schema", "policy"}:
                code = "invalid_tool_arguments" if mode == "schema" else "tool_not_allowed"
                result = data.get("result")
                assert isinstance(result, dict)
                assert result["error_code"] == code
                assert not attempt.started
            elif mode == "unknown":
                assert attempt.unknown is not None and attempt.unknown._payload["retry"] == "stop"
        finally:
            scope.close()
        assert store.release(lease)
    return core_database, data


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("mode", ["model", "tool", "schema", "policy", "unknown"])
def test_real_core_dispatch_without_sdk_imports(tmp_path, monkeypatch, flags, mode):
    from vv_agent import _jsonschema
    from vv_agent.kernel._tool_schema import SchemaCheck

    database, data = _produced_prefix(tmp_path, mode, monkeypatch)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(data))
    script = r"""
import importlib.abc, json, sys
from hashlib import sha256
from pathlib import Path
from threading import Event

sys.path.insert(0, sys.argv[1])


class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (
            fullname in {"vv_agent", "vv_agent.kernel"}
            or fullname.startswith("vv_agent.kernel.")
            or fullname.split(".")[0] in sys.stdlib_module_names
        ), fullname


sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel import _driver_dispatch as core, _tool_admission as admission
from vv_agent.kernel._content import KeyDestroyOutcome, KeyDestroyStatus, WrappedDataKey
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_and_fold_records
from vv_agent.kernel._content_runtime import (
    bind_content_runtime,
    durable_content_reader,
    wrap_session_content_store,
)
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_outcomes import OutcomeRead, PrepareToolResources
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._provider_outcomes import Definitive, Unknown
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._tool_metadata import ToolMetadata
from vv_agent.kernel._tool_schema import SchemaIssue

source = json.loads(Path(sys.argv[3]).read_text())


class FixtureKeyProvider:
    provider_id = "dispatch-fixture-v1"

    @staticmethod
    def _key(tenant_id, session_id):
        return sha256(f"{tenant_id}\x00{session_id}".encode()).digest()

    def generate_data_key(self, tenant_id, session_id):
        return WrappedDataKey(tenant_id, session_id, f"key/{tenant_id}/{session_id}", self._key(tenant_id, session_id))

    def unwrap(self, tenant_id, session_id, wrapped_key):
        assert (wrapped_key.tenant_id, wrapped_key.session_id) == (tenant_id, session_id)
        return wrapped_key.wrapped_key

    def destroy(self, tenant_id, key_id):
        return KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, f"destroy/{tenant_id}/{key_id}")


class FixtureCipher:
    def __init__(self, counter=0):
        self.counter = counter

    def encrypt(self, key, nonce, plaintext, aad):
        del key, nonce, aad
        return b"dispatch-cipher" + plaintext

    def decrypt(self, key, nonce, ciphertext, aad):
        del key, nonce, aad
        assert ciphertext.startswith(b"dispatch-cipher")
        return ciphertext.removeprefix(b"dispatch-cipher")

    def random_bytes(self, n):
        output = b""
        while len(output) < n:
            output += sha256(f"dispatch-cipher:{self.counter}".encode()).digest()
            self.counter += 1
        return output[:n]


class Clock:
    def now_ms(self):
        return source["clock_ms"]

    def monotonic(self):
        return self.now_ms() / 1000

    def monotonic_ns(self):
        return self.now_ms() * 1000000

    def sleep(self, seconds):
        raise AssertionError("unexpected scheduling")


clock = Clock()


def unused(*args, **kwargs):
    raise AssertionError("no compaction in produced prefix")


def factory():
    return Fold(normalize_memory_settings=unused, count_messages=unused)


class Token:
    def cancel(self, reason):
        raise AssertionError(f"unexpected cancellation: {reason}")


class Scope(LeaseOwnership):
    turn_id = None
    generation = None


class FixtureDriver(CoreDriver):
    def logical_state(self):
        raw_state = self.fold.state
        logical = hydrate_and_fold_records(
            self.records,
            factory,
            SessionHydration(self.content_reader, tenant_id=self.store.tenant_id, session_id="s"),
        )
        for operation_id, operation in logical.state.operations.items():
            raw_operation = raw_state.operations[operation_id]
            for attempt_number, attempt in operation.attempts.items():
                raw_attempt = raw_operation.attempts[attempt_number]
                attempt.plan = raw_attempt.plan
                attempt.prepared = raw_attempt.prepared
        return logical.state

    def extend(self, records):
        super().extend(records)
        self.state = self.logical_state()

    def hydrated_definition(self, record):
        return super().hydrated_definition(self.stored_record(record).record)


with SQLiteStore.standalone(sys.argv[2], clock=clock, fold_factory=factory) as raw_store:
    raw_store.install_schema()
    binding = bind_content_runtime(
        raw_store,
        key_provider=FixtureKeyProvider(),
        cipher=FixtureCipher(source["cipher_calls"]),
    )
    store = wrap_session_content_store(raw_store, binding)
    reader = durable_content_reader(store, tenant_id=store.tenant_id, session_id="s")
    lease = store.acquire("s", owner="dispatch-reader", ttl_ms=15000)
    scope = Scope(
        lease,
        15000,
        0.25,
        clock=clock,
        stop=Event(),
        ready=Event(),
        now_ms=source["clock_ms"],
        token_factory=Token,
    )
    driver = FixtureDriver(
        store,
        "s",
        scope,
        fold_factory=factory,
        clock=clock,
        observe_cost=lambda: None,
        hook=lambda *a: None,
        emit_phase=lambda *a: None,
        content_reader=reader,
    )
    driver.state = driver.logical_state()
    driver.bind_turn()
    attempt = driver.state.operations[source["oid"]].attempts[source["attempt"]]
    steps = core.dispatch(driver, attempt)
    value = None
    invoked = 0
    checks = 0
    after_result = None
    context = source["context"]
    while True:
        try:
            r = steps.send(value)
        except StopIteration:
            break
        value = None
        if isinstance(r, core.PrepareContext):
            pass
        elif isinstance(r, core.FinishContext):
            value = context["metadata"]
        elif r is core.DispatchRead.HAS_HOOKS:
            value = False
        elif r is core.DispatchRead.CONTEXT_CYCLE:
            value = context["cycle"]
        elif isinstance(r, core.SetContextCycle):
            context["cycle"] = r.cycle
        elif r is core.DispatchRead.CONTEXT_SHARED:
            value = context["shared"]
        elif r is core.DispatchRead.DURABLE_CONTEXT:
            value = context["shared"]
        elif r is core.DispatchRead.HANDLER_VERSION:
            value = source["handler_version"]
        elif r is core.DispatchRead.POLL_LEASE:
            value = scope.poll(store)
        elif r in (core.DispatchRead.CANCELLED, OutcomeRead.SCOPE_CANCELLED, OutcomeRead.HOST_CANCELLED):
            value = False
        elif r is OutcomeRead.CANCELLATION_REASON:
            value = None
        elif isinstance(r, core.PreflightHandler):
            value = False
        elif isinstance(r, core.BeginPreflight):
            value = context["metadata"]["session_tool_names"]
        elif isinstance(r, core.PrepareAdmission):
            value = True
        elif r is admission.Read.SCHEMA:
            value = source["schema"]
        elif isinstance(r, admission.SchemaCheck):
            checks += 1
            print(json.dumps({"schema": r.serialized_schema, "arguments": r.arguments}), flush=True)
            value = tuple(
                SchemaIssue(tuple(x["instance_path"]), tuple(x["schema_path"]), x["validator"]) for x in json.loads(input())
            )
        elif r is admission.Read.HANDLER_MANAGED:
            value = False
        elif r is admission.Read.POLICY_METADATA:
            value = source["policy"]
        elif r is admission.Read.TOOL_METADATA:
            value = ToolMetadata.from_dict(source["tool_metadata"]) if source["tool_metadata"] else None
        elif isinstance(r, admission.ApprovedCall):
            value = None
        elif r is admission.Read.APPROVAL_MODE:
            value = source["policy"].get("_vv_agent_tool_policy_approval")
        elif isinstance(r, admission.RequiresApproval):
            value = False
        elif isinstance(r, core.InvokePreflight):
            assert r.mark_started
            value = True, None
        elif isinstance(r, core.FinishPreflight):
            value = r.result
        elif r is core.EndPreflight.CLEANUP:
            pass
        elif isinstance(r, core.PrepareModel):
            value = True
        elif isinstance(r, core.SelectProvider):
            value = True
        elif isinstance(r, core.ChildBinding):
            value = False
        elif isinstance(r, (core.HasChild, core.HasDelegation)):
            value = False
        elif isinstance(r, core.CommitDispatch):
            driver.commit(r.records, guarded=r.guarded)
        elif isinstance(r, core.DispatchHook):
            pass
        elif isinstance(r, core.InvokeOperation):
            invoked += 1
            outcome = source["outcome"]
            value = (
                Unknown(outcome["reason"])
                if outcome["type"] == "unknown"
                else Definitive(outcome["result"], tuple(outcome["evidence"]), outcome["usage"], outcome["shared_state"])
            )
        elif r is core.DispatchRead.PREPARE_AFTER_TOOL:
            pass
        elif isinstance(r, core.PrepareAfterTool):
            pass
        elif isinstance(r, core.AfterTool):
            value = r.result
        elif isinstance(r, PrepareToolResources):
            assert source["resources_noop"]
            value = r.result
        else:
            raise AssertionError((type(r).__name__, repr(r)))
    actual = [r.record.encode().decode() for r in driver.records if r.seq > source["prefix_seq"]]
    assert actual == source["expected"], (actual, source["expected"])
    assert invoked == (0 if source["mode"] in {"schema", "policy"} else 1)
    assert checks == (0 if source["mode"] == "model" else 1)
    if source["mode"] == "unknown":
        current_attempt = driver.state.operations[source["oid"]].attempts[source["attempt"]]
        assert current_attempt.unknown is not None
        assert current_attempt.unknown.payload["measurement_missing"] and current_attempt.unknown.payload["retry"] == "stop"
    assert all(
        n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.") for n in sys.modules if n.startswith("vv_agent")
    )
    print(json.dumps({"done": True, "records": len(actual), "schema_checks": checks, "invocations": invoked}), flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src"), str(database), str(facts)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    done = None
    try:
        for line in process.stdout:
            request = json.loads(line)
            if request.get("done"):
                done = request
                continue
            # This host process runs the existing validator; the core process never imports it.
            issues = _jsonschema.schema_issues(SchemaCheck(request["schema"], request["arguments"]))
            process.stdin.write(json.dumps([asdict(issue) for issue in issues]) + "\n")
            process.stdin.flush()
        assert process.wait(timeout=10) == 0, process.stderr.read()
        assert done is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.parametrize("cause", [False, True])
@pytest.mark.parametrize("suppressed", [False, True])
def test_dispatch_preflight_formatter_keeps_callback_exception_chain(tmp_path, monkeypatch, cause, suppressed):
    from vv_agent.run_config import ToolPolicy

    original = StopIteration("policy stopped")
    context = LookupError("original context")
    explicit = ValueError("explicit cause") if cause else None
    original.__context__ = context
    original.__cause__ = explicit
    original.__suppress_context__ = suppressed
    observed = []

    def predicate(name, arguments):
        raise original

    def format_error(error):
        observed.append((error, error.__context__, error.__cause__, error.__suppress_context__))
        return "formatted original policy failure"

    def configure(runtime, effect):
        effect.failure_error_function = format_error
        runtime.config.tool_policy = ToolPolicy(can_use_tool=predicate)

    _, data = _produced_prefix(tmp_path, "formatter", monkeypatch, configure)
    assert observed == [(original, context, explicit, suppressed)]
    assert data["effect_calls"] == []
    result = data["result"]
    assert result["error_code"] == "tool_execution_failed"
    assert "formatted original policy failure" in result["content"]


def test_prepared_producer_publishes_request_tool_hook_and_shared_state_refs(tmp_path):
    from types import SimpleNamespace

    from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
    from vv_agent.kernel import _driver_dispatch as core
    from vv_agent.kernel._content import ContentRef
    from vv_agent.kernel._content_publication import make_record_before_publication, publish_record_content
    from vv_agent.kernel._driver import CoreDriver
    from vv_agent.kernel._driver_planning import ProviderAvailable
    from vv_agent.kernel._hydration import DefinitionContent, reference_digest
    from vv_agent.kernel._records import Record, digest
    from vv_agent.kernel._state import Attempt, ExecutionState, Operation, Turn
    from vv_agent.kernel._store import StoredRecord
    from vv_agent.kernel._tool_values import ToolCall, ToolExecutionResult
    from vv_agent.session.content_runtime import SessionContentWriter
    from vv_agent.session.content_store import LocalKeyProvider
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.session.surfaces import SessionDriver

    @function_tool
    def effect(text: str) -> str:
        return text

    sessions = SessionDriver(tmp_path / "dispatch.sqlite", content_key_provider=LocalKeyProvider(b"k" * 32))
    cast(SQLiteStore, sessions.store).configure_input_admission(allow_unsigned=True)
    sessions.create("s", "/workspace", {})
    runtime = sessions.runtime(
        Agent("dispatch", "Work.", tools=[effect], model="scripted"),
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
            workspace="/workspace",
        ),
        session_id="s",
    )
    task = runtime.compile("go", "s/turn/tool", seed=None, control=False)
    turn_id = "s/turn/tool"
    parent_id = "parent"
    parent = make_record_before_publication(
        "op_planned",
        session_id="s",
        turn_id=turn_id,
        operation_id=parent_id,
        attempt=1,
        payload={
            "op_kind": "model",
            "purpose": "primary",
            "request": {"tools": [{"function": {"name": "effect", "parameters": {"type": "object"}}}]},
            "request_digest": "0" * 64,
            "context_version": "1",
            "dependencies": [],
            "tool": None,
            "idempotency_key": "parent",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "model",
            "consumed_unknowns": [],
        },
    )
    plan = make_record_before_publication(
        "op_planned",
        session_id="s",
        turn_id=turn_id,
        operation_id="tool",
        attempt=1,
        payload={
            "op_kind": "tool",
            "purpose": None,
            "request": ToolCall("a", "effect", {"text": "original"}).to_dict(),
            "request_digest": "1" * 64,
            "context_version": "2",
            "dependencies": [parent_id],
            "tool": {},
            "idempotency_key": "s/tool",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "function",
            "consumed_unknowns": [],
        },
    )

    turn_start = Record(
        "turn/s/turn/tool/start",
        "turn_started",
        "s",
        turn_id,
        None,
        None,
        {"handler_version": "handler/h2", "generation": 0},
    )
    fixture_state = ExecutionState(
        session_id="s",
        active_turn_id=turn_id,
        turns={turn_id: Turn(turn_start)},
        operations={
            parent_id: Operation(
                turn_id=turn_id,
                kind="model",
                attempts={1: Attempt(parent)},
                selected_attempt=1,
            )
        },
    )

    class Driver(CoreDriver):
        sid = "s"
        records = (StoredRecord(turn_start, 2, "commit", 1, 0),)
        fold = None
        state = fixture_state
        scope = SimpleNamespace(lease=SimpleNamespace(epoch=7))

        def hydrated_definition(self, record: Record) -> DefinitionContent:
            del record
            return DefinitionContent(
                agent_name="dispatch",
                memory_settings={},
                child_tools=(),
                model_binding={},
                capabilities={},
                task=task,
                tools=(),
            )

        def hydrate_plan(self, record: Record) -> Record:
            return record

        def record(
            self,
            kind: str,
            payload: dict[str, Any],
            plan: Record | None = None,
            *,
            tid: str | None = None,
            feature_level: int = 1,
        ) -> Record:
            del tid
            return make_record_before_publication(
                kind,
                session_id="s",
                turn_id=plan.turn_id if plan else turn_id,
                operation_id=plan.operation_id if plan else None,
                attempt=plan.attempt if plan else None,
                payload=payload,
                feature_level=feature_level,
            )

    driver = object.__new__(Driver)
    attempt = Attempt(plan)
    steps = core.dispatch(driver, attempt)
    response = None
    captured = None
    while True:
        try:
            request = steps.send(response)
        except StopIteration:
            break
        response = None
        if isinstance(request, core.PrepareContext):
            pass
        elif isinstance(request, core.FinishContext):
            response = {}
        elif request is core.DispatchRead.HAS_HOOKS:
            response = True
        elif request is core.DispatchRead.CONTEXT_CYCLE:
            response = 1
        elif isinstance(request, core.SetContextCycle):
            pass
        elif request is core.DispatchRead.DURABLE_CONTEXT:
            response = {"shared_marker": "SHARED_MARKER"}
        elif isinstance(request, core.BeforeTool):
            response = (
                ToolCall("a", "effect", {"text": "PATCHED_MARKER"}),
                ToolExecutionResult("a", content="HOOK_MARKER"),
            )
        elif isinstance(request, ProviderAvailable):
            response = False
        elif isinstance(request, core.CommitDispatch):
            captured = request.records[0]
            break
        else:
            raise AssertionError(f"unexpected dispatch request: {request!r}")
    assert captured is not None and captured.kind == "op_prepared"

    with sessions.store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="s")
        prepared = publish_record_content(writer, (captured,)).records[0]
        values = {
            slot: writer.get_json(ContentRef.from_dict(prepared.payload[slot]))
            for slot in ("request", "tool", "hook_result", "shared_state")
        }
        request_ref = ContentRef.from_dict(prepared.payload["request"])
        assert values["request"]["arguments"] == {"text": "PATCHED_MARKER"}
        assert values["tool"] == {}
        assert values["hook_result"]["content"] == "HOOK_MARKER"
        assert values["shared_state"] == {"shared_marker": "SHARED_MARKER"}
        assert prepared.payload["request_digest"] == reference_digest(request_ref)
        assert prepared.payload["request_digest"] != digest(values["request"])
        assert b"PATCHED_MARKER" not in prepared.encode()
    sessions.close()
