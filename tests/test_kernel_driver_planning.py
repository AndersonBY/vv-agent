"""The SDK and SDK-free callers execute one planner over actual folded records."""

import inspect
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

from vv_agent.kernel import _driver_planning as core


@contextmanager
def folded_view():
    import hashlib
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any, ClassVar

    from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_and_fold_records
    from vv_agent.kernel._budget_values import RunBudgetLimits
    from vv_agent.kernel._content import KeyDestroyOutcome, KeyDestroyStatus, KeyUnavailable, WrappedDataKey
    from vv_agent.kernel._content_publication import (
        link_declared_record_content,
        make_record_before_publication,
        publish_record_content,
        publish_session_created_content,
        resolve_record_content_links,
    )
    from vv_agent.kernel._content_runtime import bind_content_runtime, wrap_session_content_store
    from vv_agent.kernel._definition import DefinitionBindings, DefinitionCache
    from vv_agent.kernel._execution_values import AgentTask, Message
    from vv_agent.kernel._hydration import _parse_definition
    from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
    from vv_agent.kernel._records import SessionSpec, digest
    from vv_agent.kernel._reducer import Fold
    from vv_agent.kernel._requirements import SessionRequirements
    from vv_agent.kernel._sqlite import SQLiteStore
    from vv_agent.kernel._tool_metadata import ToolMetadata

    class PlanningKeyProvider:
        provider_id = "planning-core-v1"

        def generate_data_key(self, tenant_id, session_id):
            return WrappedDataKey(tenant_id, session_id, f"key/{tenant_id}/{session_id}", b"planning-wrapped-key")

        def restore_data_key(self, wrapped_key):
            if not wrapped_key.key_id.startswith(f"key/{wrapped_key.tenant_id}/{wrapped_key.session_id}"):
                raise KeyUnavailable("planning key binding mismatch")

        def unwrap(self, tenant_id, session_id, wrapped_key):
            self.restore_data_key(wrapped_key)
            if (wrapped_key.tenant_id, wrapped_key.session_id) != (tenant_id, session_id):
                raise KeyUnavailable("planning key binding mismatch")
            return hashlib.sha256(f"planning:{tenant_id}:{session_id}".encode()).digest()

        def destroy(self, tenant_id, key_id):
            del tenant_id
            return KeyDestroyOutcome(False, KeyDestroyStatus.KEY_UNAVAILABLE, None, f"{key_id} is test-only")

    class PlanningCipher:
        def __init__(self):
            self.counter = 0

        @staticmethod
        def _mask(key, nonce, aad, length):
            return hashlib.shake_256(key + nonce + aad).digest(length)

        def encrypt(self, key, nonce, plaintext, aad):
            masked = bytes(
                left ^ right for left, right in zip(plaintext, self._mask(key, nonce, aad, len(plaintext)), strict=True)
            )
            return masked + hashlib.sha256(aad + nonce + masked).digest()

        def decrypt(self, key, nonce, ciphertext, aad):
            masked, tag = ciphertext[:-32], ciphertext[-32:]
            if tag != hashlib.sha256(aad + nonce + masked).digest():
                raise KeyUnavailable("planning ciphertext failed authentication")
            return bytes(left ^ right for left, right in zip(masked, self._mask(key, nonce, aad, len(masked)), strict=True))

        def random_bytes(self, n):
            self.counter += 1
            return hashlib.sha256(b"planning-nonce:" + self.counter.to_bytes(8, "big")).digest()[:n]

    class PlanningClock:
        def now_ms(self):
            return 1_000

        def monotonic(self):
            return 1.0

        def monotonic_ns(self):
            return 1_000_000_000

        def sleep(self, seconds):
            raise AssertionError(f"planning prefix does not sleep ({seconds})")

    class PlanningCatalog:
        schema: ClassVar[dict[str, Any]] = {
            "type": "function",
            "function": {
                "name": "evidence",
                "description": "Return retained evidence.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }

        def planning_signature(self):
            return ("planning-core-v1",)

        def has_tool(self, name):
            return name == "evidence"

        def has_schema(self, name):
            return name == "evidence"

        def has_executor(self, name):
            return name == "evidence"

        def tool_metadata(self, name):
            return ToolMetadata() if name == "evidence" else None

        def list_openai_schemas(self, *, tool_names=None):
            return [self.schema] if tool_names is None or "evidence" in tool_names else []

    def fold_factory():
        def normalize_memory_settings(settings, protected_start):
            del protected_start
            return dict(settings)

        def count_messages(messages, *, model):
            del model
            return sum(max(1, len(str(message.get("content", ""))) // 4) for message in messages)

        return Fold(normalize_memory_settings=normalize_memory_settings, count_messages=count_messages)

    with (
        tempfile.TemporaryDirectory(prefix="vv-planning-s1-") as directory,
        SQLiteStore.standalone(
            str(Path(directory) / "planning.sqlite"),
            fold_factory=fold_factory,
            clock=PlanningClock(),
        ) as raw_store,
    ):
        sid = "planning"
        tid = "planning/turn"
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        raw_store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        raw_store.configure_worker(level=2, handler_versions=("1",), capabilities=())
        tenant_id = raw_store.tenant_id
        binding = bind_content_runtime(raw_store, key_provider=PlanningKeyProvider(), cipher=PlanningCipher())
        store = wrap_session_content_store(raw_store, binding)

        from vv_agent.kernel._content_runtime import SessionContentWriter

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=sid)

            def session_spec(content):
                principal, _ = content.service_principal("fixture/host")
                created = publish_session_created_content(
                    content,
                    principal=principal,
                    workspace="/workspace",
                    attributes={
                        "seed": {
                            "messages": [Message("user", "xxxxxxxxxxxxxxxx").to_dict()],
                            "shared_state": {},
                            "unseen_input_protection": True,
                        }
                    },
                )
                return SessionSpec(
                    sid,
                    created.principal,
                    created.workspace.to_dict(),
                    attributes=created.attributes.to_dict(),
                    feature_level=2,
                    bootstrap=SessionRequirements(2, "1", ()),
                )

            writer.create_session(session_spec, consumers=())

        task = AgentTask(
            task_id=tid,
            model="scripted",
            prompt_bundle=PromptBundle((PromptSection("agent_instructions", "Be precise.", stable=True),)),
            user_prompt="xxxxxxxxxxxxxxxx",
            max_cycles=10,
            allow_interruption=False,
            use_workspace=False,
            extra_tool_names=["evidence"],
            initial_shared_state={"todo_list": []},
            metadata={"trace_id": tid, "session_memory_enabled": False},
        )
        memory_settings = {
            "artifact_scope": "run",
            "autocompact_buffer_tokens": 13_000,
            "base_system_prompt": "",
            "compact_threshold": 250_000,
            "include_memory_warning": False,
            "keep_recent_messages": 10,
            "language": "zh-CN",
            "microcompaction_policy": {
                "keep_recent_cycles": 3,
                "min_result_chars": 500,
                "target_ratio": 0.6,
                "trigger_ratio": 0.75,
            },
            "model": "scripted",
            "model_context_window": 279_000,
            "model_max_output_tokens": None,
            "reserved_output_source": "framework_fallback",
            "reserved_output_tokens": 16_000,
            "summary_backend": None,
            "summary_event_limit": 40,
            "summary_model": None,
            "tool_result_excerpt_head": 200,
            "tool_result_excerpt_tail": 200,
            "tool_result_retentions": {},
            "warning_threshold_percentage": 90,
        }
        definition = DefinitionCache().definition(
            task,
            DefinitionBindings(
                agent_name="kernel",
                memory_settings=memory_settings,
                child_tools=(),
                model_binding={"backend": "scripted", "context_length": None, "max_output_tokens": None, "model": "scripted"},
            ),
            PlanningCatalog(),
        )

        lease = store.acquire(sid, owner="planning-producer", ttl_ms=15_000)
        if lease is None:
            raise AssertionError("planning prefix could not acquire the SQL lease")
        try:
            head = store.read(sid, limit=1)
            with store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=sid)
                turn_started = make_record_before_publication(
                    "turn_started",
                    session_id=sid,
                    turn_id=tid,
                    payload={
                        "input_ids": [],
                        "definition": definition,
                        "definition_digest": digest(definition),
                        "handler_version": "1",
                        "budget": RunBudgetLimits().to_dict(),
                        "binding": "task",
                        "generation": 0,
                        "requirements": {"level": 2, "handler_version": "1", "capabilities": []},
                    },
                )
                publication = publish_record_content(writer, (turn_started,))
                receipt = tx.append(
                    sid,
                    lease=lease,
                    expected_seq=head.head_seq,
                    commit_id="planning-prefix",
                    records=publication.records,
                )
                links = resolve_record_content_links(publication.links, receipt.record_sequences)
                link_declared_record_content(writer, links)
        finally:
            store.release(lease)

        page = store.read(sid)
        result = hydrate_and_fold_records(
            page.records,
            fold_factory,
            SessionHydration.from_store(store, tenant_id=tenant_id, session_id=sid),
        )
        view = SimpleNamespace(
            state=result.state,
            records=result.fold.history,
            fold=result.fold,
            sid=sid,
            store=store,
        )
        view.hydrated_definition = lambda record: _parse_definition(record._payload["definition"])
        view.hydrate_record_field = lambda record, slot, ref: record._payload[slot]
        view.session_attributes = lambda: view.records[0].record.payload["attributes"]
        yield view


def append_hydrated_plan(view, producer):
    from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_session_stored_record
    from vv_agent.kernel._content_publication import (
        link_declared_record_content,
        publish_record_content,
        resolve_record_content_links,
    )
    from vv_agent.kernel._content_runtime import SessionContentWriter

    head = view.store.read(view.sid, limit=1)
    lease = view.store.acquire(view.sid, owner="planning-output", ttl_ms=15_000)
    if lease is None:
        raise AssertionError("planning output could not acquire the SQL lease")
    try:
        with view.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=view.store.tenant_id, session_id=view.sid)
            publication = publish_record_content(writer, (producer,))
            receipt = tx.append(
                view.sid,
                lease=lease,
                expected_seq=head.head_seq,
                commit_id="planning-output",
                records=publication.records,
            )
            links = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, links)
    finally:
        view.store.release(lease)

    page = view.store.read(view.sid)
    stored = next(row for row in page.records if row.record.record_id == producer.record_id)
    hydrated = hydrate_session_stored_record(
        stored,
        SessionHydration.from_store(view.store, tenant_id=view.store.tenant_id, session_id=view.sid),
    )
    view.fold.extend((hydrated.record,), bodies=(stored.record.encode(),))
    view.records.append(hydrated)
    view.state = view.fold.state
    return stored, hydrated.record


def append_hydrated_selection(view, selection, binding):
    from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_and_fold_records
    from vv_agent.kernel._content_publication import (
        link_declared_record_content,
        make_record_before_publication,
        publish_record_content,
        resolve_record_content_links,
    )
    from vv_agent.kernel._content_runtime import SessionContentWriter
    from vv_agent.kernel._records import InboxItem, digest
    from vv_agent.kernel._requirements import SessionRequirements

    head = view.store.read(view.sid, limit=1)
    lease = view.store.acquire(view.sid, owner="planning-selection", ttl_ms=15_000)
    if lease is None:
        raise AssertionError("planning selection could not acquire the SQL lease")
    try:
        with view.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=view.store.tenant_id, session_id=view.sid)
            actor, _ = writer.service_principal("fixture/model-selection")
            item = InboxItem(
                "selection",
                "model_selection",
                {"selection": writer.put_json(selection).to_dict()},
                actor=actor,
                feature_level=2,
                requirements=SessionRequirements(2, "1", ()),
            )
            tx.push(view.sid, item)
            producer = make_record_before_publication(
                "input_applied",
                session_id=view.sid,
                turn_id=view.state.active_turn_id,
                payload={
                    "input": item.to_dict(),
                    "input_digest": digest(item.to_dict()),
                    "input_proof": None,
                    "disposition": "applied",
                    "reason": None,
                    "target_operation_id": None,
                    "target_wait_id": None,
                    "position": "between_actions",
                    "model_binding": writer.put_json(binding).to_dict(),
                },
            )
            publication = publish_record_content(writer, (producer,))
            receipt = tx.append(
                view.sid,
                lease=lease,
                expected_seq=head.head_seq,
                commit_id="planning-selection",
                records=publication.records,
                consume_input_ids=("selection",),
            )
            links = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, links)
    finally:
        view.store.release(lease)

    page = view.store.read(view.sid)
    stored = next(row for row in page.records if row.record.record_id == producer.record_id)
    result = hydrate_and_fold_records(
        (stored,),
        lambda: view.fold.fork(),
        SessionHydration.from_store(view.store, tenant_id=view.store.tenant_id, session_id=view.sid),
        base_fold=view.fold,
    )
    view.fold = result.fold
    view.state = result.state
    view.records.append(result.hydrated_records[0])
    return result.state.model_selection


def scripted_host(steps, view, *, hook=None, endpoints=None):
    from vv_agent.kernel import _driver_planning as core

    start = view.state.turns[view.state.active_turn_id].start
    task = start._task()
    binding = dict(start._payload["definition"]["model_binding"]) | {"endpoints": []}
    response = None
    trace = []
    try:
        while True:
            try:
                request = steps.send(response)
            except StopIteration as done:
                return done.value, trace
            trace.append(request)
            response = None
            if isinstance(request, core.ResolveModel):
                pass
            elif request is core.PlanningRead.MODEL_BINDING:
                response = binding
            elif request is core.PlanningRead.ENDPOINT_SUPPORT:
                response = endpoints is not None
            elif isinstance(request, core.OrderedEndpoints):
                response = endpoints
            elif isinstance(request, core.ProviderAvailable):
                response = request.name == "evidence"
            elif request is core.PlanningRead.HAS_HOOKS or request is core.PlanningRead.PREPARE_BEFORE_LLM:
                response = hook is not None
            elif request is core.PlanningRead.SHARED_STATE:
                response = dict(task.initial_shared_state)
            elif request is core.PlanningRead.INTERRUPTION_MESSAGES or isinstance(request, core.BeforeCycle):
                response = False, None
            elif isinstance(request, core.BeforeLLM):
                response = hook(request) if hook else (list(request.messages), list(request.schemas))
            elif request is core.PlanningRead.DEFAULT_MODEL_BINDING:
                response = binding
            elif isinstance(request, core.DurableState):
                response = dict(request.shared)
            else:
                assert isinstance(request, core.ClosePlanning)
    finally:
        steps.close()


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_normal_core_planner_builds_and_folds_real_plan_without_sdk(tmp_path, flags):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from contextlib import contextmanager
        sys.path.insert(0, sys.argv[1] + "/src")
        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname
        sys.meta_path.insert(0, CoreOnly())
    """)
    script += inspect.getsource(folded_view) + "\n" + inspect.getsource(append_hydrated_plan) + "\n"
    script += inspect.getsource(scripted_host)
    script += textwrap.dedent("""
        from vv_agent.kernel import _driver_planning as core
        from vv_agent.kernel._context import project_context
        with folded_view() as view:
            start = view.state.turns[view.state.active_turn_id].start
            record, trace = scripted_host(core.model_plan(view, project_context(view.records, view.state)), view)
            assert record.kind == "op_planned" and record.payload["request"]["metadata"]["vv_session"]["cycle_index"] == 1
            from vv_agent.kernel._canonical_json import canonical_json_bytes
            assert record._payload["request"]["tools"] == start._payload["definition"]["tools"]
            assert canonical_json_bytes(record._payload["request"]["tools"]) == canonical_json_bytes(
                start._payload["definition"]["tools"]
            )
            cached = start._tools_body
            again, _ = scripted_host(core.model_plan(view, project_context(view.records, view.state)), view)
            assert canonical_json_bytes(again._payload) == canonical_json_bytes(record._payload)
            assert start._tools_body is cached
            before = next(item for item in trace if isinstance(item, core.BeforeLLM))
            assert record._payload["request"]["messages"] == [message.to_dict() for message in before.messages]
            stored, hydrated = append_hydrated_plan(view, record)
            assert b"Be precise." not in stored.record.encode() and b"xxxxxxxxxxxxxxxx" not in stored.record.encode()
            assert view.state.operations[record.operation_id].attempts[1].plan is hydrated
            assert hydrated._payload["request"]["tools"] == record._payload["request"]["tools"]
            assert all(n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.")
                       for n in sys.modules if n.startswith("vv_agent"))
    """)
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).resolve().parents[1])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_planning_retains_output_repair_binding_and_empty_order():
    with folded_view() as view:
        start = view.state.turns[view.state.active_turn_id].start
        binding = dict(start._payload["definition"]["model_binding"]) | {"endpoints": ["repair-retained"]}
        request = {
            "model": start._task().model,
            "messages": [],
            "tools": [],
            "metadata": {"vv_session": {"model_binding": binding, "endpoint_order": ["old"]}},
        }
        record, trace = scripted_host(core.plan(view, "repair", request, "model", purpose="output_repair"), view, endpoints=[])
        assert core.PlanningRead.MODEL_BINDING not in trace
        assert record.payload["request"]["metadata"]["vv_session"] == {"model_binding": binding, "endpoint_order": ["old"]}


@pytest.mark.parametrize("mode", ["primary", "child_summary"])
def test_core_unknown_consumption_preserves_operation_and_attempt_order(mode):
    from vv_agent.kernel._records import Record
    from vv_agent.kernel._state import Attempt, Operation

    with folded_view() as view:
        tid = view.state.active_turn_id
        for oid, turn, kind in [
            ("later", tid, "tool"),
            ("earlier", tid, "tool"),
            ("other", "other", "tool"),
            ("model", tid, "model"),
        ]:
            plan = Record(oid, "op_planned", view.sid, turn, oid, 1, {})
            view.state.operations[oid] = Operation(
                turn, kind, {2: Attempt(plan, state="unknown"), 1: Attempt(plan, state="unknown")}
            )
        request = {"model": view.state.turns[tid].start._task().model, "messages": [], "tools": [], "metadata": {"mode": mode}}
        record, _ = scripted_host(core.plan(view, "model/next", request, "model"), view)
        assert record.payload["consumed_unknowns"] == (
            []
            if mode == "child_summary"
            else [{"operation_id": oid, "attempt": n} for oid in ("later", "earlier") for n in (2, 1)]
        )


@pytest.mark.parametrize("idempotency", [None, "supported", "unsupported"])
def test_tool_plan_keeps_provider_choice_and_idempotency_key(idempotency):
    with folded_view() as view:
        capability = {"idempotency": idempotency} if idempotency is not None else None
        record, _ = scripted_host(
            core.plan(view, "tool", {"id": "call", "name": "evidence", "arguments": {}}, "tool", capability=capability), view
        )
        assert record.payload["provider_binding"] == "evidence"
        assert record.payload["idempotency_key"] == (None if idempotency == "unsupported" else view.sid + "/tool")


@pytest.mark.parametrize("mode", ["protected-tail", "recovery", "denied-tools"])
def test_core_owns_protected_tail_and_recovery_decisions(mode):
    from vv_agent.kernel._compaction_text import build_compacted_tool_content
    from vv_agent.kernel._context import project_context
    from vv_agent.kernel._context_messages import UnseenInputError
    from vv_agent.kernel._cycle_values import AFTER_CYCLE_CONTROL_SCHEMA, AFTER_CYCLE_CONTROL_STATE_KEY
    from vv_agent.kernel._message_values import Message

    with folded_view() as view:

        def hook(request):
            if mode == "protected-tail":
                return [request.messages[0]], request.schemas
            if mode == "recovery":
                marker = Message(
                    "tool", build_compacted_tool_content("text", artifact_path="a", tool_name="evidence"), tool_call_id="c"
                )
                return [*request.messages, marker], []
            request.shared[AFTER_CYCLE_CONTROL_STATE_KEY] = {
                "schema_version": AFTER_CYCLE_CONTROL_SCHEMA,
                "disallowed_tools": ["evidence"],
            }
            return request.messages, request.schemas

        steps = core.model_plan(view, project_context(view.records, view.state))
        if mode == "protected-tail":
            with pytest.raises(UnseenInputError, match="unseen_input_removed"):
                scripted_host(steps, view, hook=hook)
        else:
            record, trace = scripted_host(steps, view, hook=hook)
            if mode == "recovery":
                assert record is None
                assert trace[-1] == core.ClosePlanning("failed", "microcompaction_recovery_unavailable")
                assert core.PlanningRead.DEFAULT_MODEL_BINDING not in trace
            else:
                assert record.payload["request"]["tools"] == []


def test_sdk_passes_the_actual_core_driver_state_owner(monkeypatch):
    from vv_agent.kernel._driver import CoreDriver
    from vv_agent.session import _driver_planning as adapter
    from vv_agent.session import _recovery_phase
    from vv_agent.session.kernel import _Driver

    assert issubclass(_Driver, CoreDriver)
    with folded_view() as current, folded_view() as replacement:
        driver = object.__new__(_Driver)
        driver.state, driver.records, driver.fold, driver.sid = current.state, tuple(current.records), current.fold, current.sid
        observed = []
        original = core.plan

        def planning(view, *args, **kwargs):
            observed.append(view)
            return original(view, *args, **kwargs)

        def drain(owner, steps):
            assert owner is driver
            return scripted_host(steps, owner)[0]

        monkeypatch.setattr(core, "plan", planning)
        monkeypatch.setattr(_recovery_phase, "run", drain)
        first = adapter.plan(driver, "tool", {"id": "c", "name": "evidence", "arguments": {}}, "tool")
        driver.state, driver.records, driver.fold = replacement.state, tuple(replacement.records), replacement.fold
        second = adapter.plan(driver, "tool", {"id": "c", "name": "evidence", "arguments": {}}, "tool")
        assert observed == [driver, driver]
        assert all(isinstance(view, CoreDriver) for view in observed)
        assert observed[-1].state is replacement.state and observed[-1].fold is replacement.fold
        from vv_agent.kernel._canonical_json import canonical_json_bytes

        assert canonical_json_bytes(first._payload) == canonical_json_bytes(second._payload)


@pytest.mark.parametrize("mode", ["default", "fork", "selected", "selected-over-fork"])
def test_effective_model_core_owns_selected_fork_default_precedence(mode):
    from dataclasses import replace

    with folded_view() as view:
        fork = {"model": "fork", "route": {}, "model_settings": {"max_tokens": 13}, "binding": {"model": "fork"}}
        if "fork" in mode:
            first = view.records[0]
            payload = first.record.payload
            payload["attributes"]["app_server_fork"] = {"effective_model": fork}
            view.records[0] = replace(first, record=replace(first.record, payload=payload))
        if mode.startswith("selected"):
            selection = {"model": "selected", "model_settings": {"max_tokens": 17}, "route": {}}
            binding = {"backend": "test", "model": "selected", "endpoints": [], "context_length": None, "max_output_tokens": None}
            append_hydrated_selection(view, selection, binding)
        result, trace = scripted_host(core.effective_model(view), view)
        if mode.startswith("selected"):
            assert result["model"] == "selected" and trace == []
            assert result["binding"] == view.state.model_selection._payload["model_binding"]
        elif mode == "fork":
            assert result == fork and trace == []
            assert result["model_settings"] is not fork["model_settings"]
        else:
            assert result["model"] == view.state.turns[view.state.active_turn_id].start._task().model
            assert trace == [core.PlanningRead.DEFAULT_MODEL_BINDING]


def test_real_planner_publishes_hydratable_request_reference(tmp_path):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.kernel._content import ContentRef
    from vv_agent.kernel._hydration import reference_digest
    from vv_agent.kernel._records import digest
    from vv_agent.session._driver_scope import _Scope
    from vv_agent.session.content_store import LocalKeyProvider
    from vv_agent.session.kernel import _Driver, drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse

    class Planned(BaseException):
        pass

    marker = "PLANNER_CONTENT_MARKER_7e77"
    with SQLiteStore.standalone(str(tmp_path / "planner.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))
        sessions.create("s", "/workspace", {})
        sessions.push("s", InboxItem("input", "user", {"content": marker}))
        runtime = sessions.runtime(
            Agent("planner", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("done")]),
                workspace="/workspace",
            ),
            session_id="s",
        )

        def cut(point, record):
            if point == "after_commit" and record is not None and record.kind == "op_planned":
                raise Planned

        runtime.hook = cut
        with pytest.raises(Planned):
            drive(sessions.store, "s", runtime=runtime)

        runtime.hook = lambda *_: None
        lease = sessions.store.acquire("s", owner="planner-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        driver.bind_turn()
        operation = next(op for op in driver.state.operations.values() if op.kind == "model")
        plan = operation.attempts[1].execution_plan
        request_ref = ContentRef.from_dict(plan.payload["request"])
        budget_ref = ContentRef.from_dict(plan.payload["budget_admission"])
        request = driver.hydrate_plan(plan).payload["request"]

        assert marker in request["messages"][1]["content"]
        assert plan.payload["request_digest"] == reference_digest(request_ref)
        assert plan.payload["request_digest"] != digest(request)
        assert budget_ref.size == 2
        assert marker.encode() not in plan.encode()
        sessions.close()


def test_real_tool_planner_publishes_hydratable_request_and_capability_refs(tmp_path):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
    from vv_agent.kernel._content import ContentRef
    from vv_agent.kernel._hydration import reference_digest
    from vv_agent.kernel._records import digest
    from vv_agent.session._driver_scope import _Scope
    from vv_agent.session.content_store import LocalKeyProvider
    from vv_agent.session.kernel import _Driver, drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse, ToolCall

    class ToolPlanned(BaseException):
        pass

    @function_tool
    def effect(text: str) -> str:
        return text

    with SQLiteStore.standalone(str(tmp_path / "tool-planner.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))
        sessions.create("s", "/workspace", {})
        sessions.push("s", InboxItem("input", "user", {"content": "use the tool"}))
        runtime = sessions.runtime(
            Agent("planner", "Work.", tools=[effect], model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new(
                    "scripted",
                    "scripted",
                    [LLMResponse("", [ToolCall("call", "effect", {"text": "TOOL_MARKER_5e31"})])],
                ),
                workspace="/workspace",
            ),
            session_id="s",
        )

        def cut(point, record):
            if (
                point == "after_commit"
                and record is not None
                and record.kind == "op_planned"
                and record.payload["op_kind"] == "tool"
            ):
                raise ToolPlanned

        runtime.hook = cut
        with pytest.raises(ToolPlanned):
            drive(sessions.store, "s", runtime=runtime)

        runtime.hook = lambda *_: None
        lease = sessions.store.acquire("s", owner="tool-planner-test", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        driver.bind_turn()
        operation = next(op for op in driver.state.operations.values() if op.kind == "tool")
        plan = operation.attempts[1].execution_plan
        hydrated = driver.hydrate_plan(plan).payload
        request_ref = ContentRef.from_dict(plan.payload["request"])
        tool_ref = ContentRef.from_dict(plan.payload["tool"])
        budget_ref = ContentRef.from_dict(plan.payload["budget_admission"])

        assert hydrated["request"]["arguments"] == {"text": "TOOL_MARKER_5e31"}
        assert hydrated["tool"] == {}
        assert hydrated["budget_admission"] == {}
        assert plan.payload["request_digest"] == reference_digest(request_ref)
        assert plan.payload["request_digest"] != digest(hydrated["request"])
        assert tool_ref == budget_ref
        assert b"TOOL_MARKER_5e31" not in plan.encode()
        sessions.close()
