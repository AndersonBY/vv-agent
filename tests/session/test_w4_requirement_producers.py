"""Real contract-41 turn requirement producers and fail-closed admission."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from support import FixedModelProvider, ModelMapProvider

from vv_agent import Runner, handoff
from vv_agent.agent import Agent
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalProvider, ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.context_providers import ContextFragment
from vv_agent.guardrails import GuardrailResult
from vv_agent.kernel._content import APPROVAL_METADATA, APPROVAL_REASON, APPROVAL_SCOPE, INPUT_ACTOR_ATTRIBUTES
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_inputs import CommitInput, ResolveModelSelection, apply_input
from vv_agent.kernel._requirements import (
    SessionRequirements,
    definition_requirements,
    model_capability_tag,
    model_capability_tags,
    tool_capability_tag,
)
from vv_agent.kernel._state import ExecutionState, Turn
from vv_agent.kernel._tool_policy import ToolPolicy
from vv_agent.llm.scripted import ScriptedLLM, ScriptStep
from vv_agent.memory import MemoryManager
from vv_agent.model import resolve_model_selection
from vv_agent.run_config import RunConfig
from vv_agent.runtime.hooks import BaseRuntimeHook
from vv_agent.session.children import ChildSession, child_delivery, child_handles
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.kernel import Runtime, drive
from vv_agent.session.projection import hydrate_stored_record, hydration_for_session_store
from vv_agent.session.records import InboxItem, Record, SessionSpec
from vv_agent.session.runtime import (
    static_model_selection_requirements,
    static_session_requirements,
    static_worker_supports,
)
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import SessionStore
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, SubAgentConfig, ToolCall

from .helpers import (
    fixture_content_reader,
    push_fixture_input,
    wrap_fixture_store,
)

RESOLVED = ResolvedModelConfig(
    "scripted",
    "m",
    "m",
    "m",
    [EndpointOption(EndpointConfig("endpoint-a", "secret", "https://example.invalid"), "m")],
    context_length=8192,
    max_output_tokens=1024,
)


def principal() -> dict[str, Any]:
    return {
        "tenant_id": "default",
        "subject_id": "subject",
        "subject_kind": "user",
        "attributes": {
            "content_id": "content/test/principal",
            "mac": "0" * 64,
            "size": 1,
            "key_id": "key/test",
            "labels": {"source": ["user"], "classification": "internal"},
        },
    }


def _unused_heartbeat_store() -> AbstractContextManager[SessionStore]:
    raise AssertionError("test must install a real heartbeat store before driving")


def runtime_for(
    agent: Agent,
    config: RunConfig,
    *,
    resolved: ResolvedModelConfig = RESOLVED,
    handler_version: str = "handler/h2",
    heartbeat_store: Callable[[], AbstractContextManager[SessionStore]] | None = None,
) -> Runtime:
    provider = FixedModelProvider(ScriptedLLM([LLMResponse("done")]), resolved)
    config = replace(config, model_provider=provider)
    return Runtime(
        agent,
        config,
        resolved,
        provider.llm,
        heartbeat_store or _unused_heartbeat_store,
        handler_version=handler_version,
    )


def runtime_for_steps(
    agent: Agent,
    config: RunConfig,
    steps: list[ScriptStep],
    *,
    handler_version: str = "handler/h2",
) -> tuple[Runtime, ScriptedLLM]:
    llm = ScriptedLLM(steps)
    provider = FixedModelProvider(llm, RESOLVED)
    runtime = Runtime(
        agent,
        replace(config, model_provider=provider),
        RESOLVED,
        llm,
        _unused_heartbeat_store,
        handler_version=handler_version,
    )
    return runtime, llm


def _create_l2_session(
    raw_store: SQLiteStore,
    store: SessionStore,
    sid: str,
    runtime: Runtime,
    workspace: Path,
) -> SessionRequirements:
    bound = runtime.requirements_bound()
    raw_store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    raw_store.configure_worker(
        level=2,
        handler_versions=(runtime.handler_version,),
        capabilities=bound.capabilities,
    )
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=raw_store.tenant_id, session_id=sid)

        def session_spec(content):
            actor, _ = content.service_principal("requirements")
            return SessionSpec(
                sid,
                actor,
                content.put_json(str(workspace)).to_dict(),
                attributes={"seed": {"messages": [], "shared_state": {}}},
                feature_level=2,
                bootstrap=bound,
            )

        writer.create_session(session_spec, consumers=())
    return bound


def _assert_l2_session_requirements(store: SessionStore, sid: str) -> None:
    rows = store.read_state(sid)[1]
    created = rows[0].record
    assert created.kind == "session_created"
    assert created.feature_level == 2
    bound = SessionRequirements.from_dict(created.payload["bootstrap"])
    applied = next(row for row in rows if row.record.kind == "input_applied" and row.record.payload["input"]["kind"] == "user")
    item = applied.record.payload["input"]
    assert item["feature_level"] == 2
    assert SessionRequirements.from_dict(item["requirements"]) == bound
    started = next(row for row in rows if row.record.kind == "turn_started")
    hydrated = hydrate_stored_record(started, hydration_for_session_store(store, sid)).record
    actual = definition_requirements(
        hydrated._payload["definition"],
        handler_version=hydrated._payload["handler_version"],
    )
    assert hydrated._payload["requirements"] == actual.to_dict()
    assert actual.handler_version == bound.handler_version
    assert actual.level <= bound.level
    assert set(actual.capabilities) <= set(bound.capabilities)


def _owner_link_values(
    store: SessionStore,
    sid: str,
    *,
    record_seq: int | None = None,
    input_id: str | None = None,
) -> dict[str, Any]:
    created = store.read_state(sid)[1][0].record
    tenant_id = created._payload["principal"]["tenant_id"]
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=sid)
        links = writer.durable.read_links(
            tenant_id,
            sid,
            record_seq=record_seq,
            input_id=input_id,
        )
        return {link.slot: writer.get_json(link.ref) for link in links}


class _Restart(BaseException):
    pass


def _apply_core_item(item: InboxItem, response: dict[str, Any]) -> Record:
    turn_start = Record(
        "turn/s/turn/start",
        "turn_started",
        "s",
        "turn",
        None,
        None,
        {"handler_version": "handler/h2", "generation": 0},
    )
    state = ExecutionState(
        session_id="s",
        turns={"turn": Turn(turn_start)},
        active_turn_id="turn",
    )
    driver = object.__new__(CoreDriver)
    driver.sid = "s"
    driver.watermark = 1
    driver.state = state
    driver.store = SimpleNamespace(peek_inbox=lambda *args, **kwargs: [SimpleNamespace(item=item)])
    driver.content_reader = None
    pending = apply_input(driver, one_turn=True)
    request = next(pending)
    assert isinstance(request, ResolveModelSelection)
    request = pending.send(response)
    assert isinstance(request, CommitInput)
    with pytest.raises(StopIteration) as done:
        next(pending)
    assert done.value.value is True
    assert len(request.records) == 1
    return request.records[0]


def _commands_runtime(calls: Counter[str], *, tool_enabled: bool) -> Runtime:
    def instructions(_context, _agent):
        calls["instructions"] += 1
        return "Use the configured tools."

    def guardrail(_context, value):
        calls["guardrail"] += 1
        del value
        return GuardrailResult.allow()

    def enabled(_context, _agent):
        calls["is_enabled"] += 1
        return tool_enabled

    class Provider:
        def fragments(self, request):
            del request
            calls["context"] += 1
            return [ContextFragment(id="provider", text="trusted context")]

    @function_tool(is_enabled=enabled)
    def alpha(text: str) -> str:
        """Return the text."""

        return text

    agent = Agent(
        "requirements",
        instructions,
        model="m",
        tools=[alpha],
        input_guardrails=[guardrail],
    )
    return runtime_for(agent, RunConfig(context_providers=[Provider()]))


def test_pre_admission_bound_has_no_compile_effects_and_turn_has_one_each(tmp_path) -> None:
    calls: Counter[str] = Counter()
    runtime = _commands_runtime(calls, tool_enabled=True)

    bound = runtime.requirements_bound()
    assert bound.level == 2
    assert bound.handler_version == "handler/h2"
    assert calls == Counter()

    task = runtime.compile("go", "turn")
    assert task.task_id == "turn"
    assert calls == Counter({"guardrail": 1, "instructions": 1, "context": 1, "is_enabled": 1})


def test_static_publish_helpers_do_not_construct_runtime_or_memory_manager(monkeypatch) -> None:
    enabled_calls = Counter()

    def enabled(_context, _agent):
        enabled_calls["value"] += 1
        return False

    @function_tool(is_enabled=enabled)
    def dynamic(text: str) -> str:
        """Return text."""

        return text

    agent = Agent("requirements", "Be concise.", model="m", tools=[dynamic])
    config = RunConfig(model_provider=FixedModelProvider(ScriptedLLM([]), RESOLVED))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("static publication must not construct Runtime or MemoryManager")

    with monkeypatch.context() as patch:
        patch.setattr(Runtime, "__post_init__", unexpected)
        patch.setattr(MemoryManager, "__post_init__", unexpected)
        bound = static_session_requirements(
            agent,
            config,
            RESOLVED,
            handler_version="handler/h2",
        )
        supports = static_worker_supports(agent, config, RESOLVED)

    assert enabled_calls == Counter()
    assert bound.level == 2
    assert bound.handler_version == "handler/h2"
    assert set(bound.capabilities) <= set(supports)
    assert (
        model_capability_tag(
            {
                "backend": RESOLVED.backend,
                "model": RESOLVED.model_id,
                "context_length": RESOLVED.context_length,
                "max_output_tokens": RESOLVED.max_output_tokens,
            }
        )
        in supports
    )


def test_null_static_handler_requires_explicit_trusted_family_capability() -> None:
    agent = Agent("requirements", "Be concise.", model="m")
    config = RunConfig()

    with pytest.raises(ValueError, match="explicit trusted family capability"):
        static_session_requirements(agent, config, RESOLVED, handler_version=None)

    bound = static_session_requirements(
        agent,
        config,
        RESOLVED,
        handler_version=None,
        extra_capabilities=("family:custom",),
    )
    assert bound.handler_version is None
    assert "family:custom" in bound.capabilities


def test_static_model_facts_split_internal_bindings_without_memory_manager(monkeypatch) -> None:
    summary = ResolvedModelConfig(
        "scripted",
        "summary",
        "summary",
        "summary",
        [EndpointOption(EndpointConfig("endpoint-summary", "secret", "https://example.invalid"), "summary")],
        context_length=4096,
        max_output_tokens=512,
    )
    provider = ModelMapProvider(
        {
            "m": (ScriptedLLM([]), RESOLVED),
            "summary": (ScriptedLLM([]), summary),
        },
        "m",
    )
    agent = Agent(
        "requirements",
        "Be concise.",
        model="m",
        metadata={"memory_summary_model": "summary"},
    )
    config = RunConfig(model_provider=provider)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("static model facts must not construct MemoryManager")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryManager, "__post_init__", unexpected)
        bound = static_session_requirements(agent, config, handler_version="handler/h2")
        supports = static_worker_supports(agent, config)

    def tag(resolved: ResolvedModelConfig) -> str:
        return model_capability_tag(
            {
                "backend": resolved.backend,
                "model": resolved.model_id,
                "context_length": resolved.context_length,
                "max_output_tokens": resolved.max_output_tokens,
            }
        )

    expected = {tag(RESOLVED), tag(summary)}
    assert {item for item in bound.capabilities if item.startswith("model:")} == expected
    assert {item for item in supports if item.startswith("model:")} == expected


def test_dynamic_enabled_subset_uses_complete_worker_bound(tmp_path) -> None:
    enabled_calls = Counter()
    disabled_calls = Counter()

    def enabled_tool(_context, _agent):
        enabled_calls["value"] += 1
        return True

    def disabled_tool(_context, _agent):
        disabled_calls["value"] += 1
        return False

    @function_tool(is_enabled=enabled_tool)
    def enabled(text: str) -> str:
        """Enabled dynamic tool."""

        return text

    @function_tool(is_enabled=disabled_tool)
    def disabled(text: str) -> str:
        """Disabled dynamic tool."""

        return text

    runtime = runtime_for(
        Agent("requirements", "Be concise.", model="m", tools=[enabled, disabled]),
        RunConfig(),
    )
    bound = runtime.requirements_bound()
    assert enabled_calls == Counter() and disabled_calls == Counter()

    task = runtime.compile("go", "turn")
    assert enabled_calls == Counter({"value": 1})
    assert disabled_calls == Counter({"value": 1})
    definition = runtime.definition(task)
    actual = definition_requirements(definition, handler_version=runtime.handler_version)

    enabled_tag = tool_capability_tag(
        "enabled",
        runtime.registry.get_schema("enabled"),
        runtime.registry.tool_metadata("enabled"),
    )
    disabled_tag = tool_capability_tag(
        "disabled",
        runtime._configured_registry().get_schema("disabled"),
        runtime._configured_registry().tool_metadata("disabled"),
    )
    assert enabled_tag in actual.capabilities
    assert disabled_tag in bound.capabilities
    assert disabled_tag not in actual.capabilities
    assert set(actual.capabilities) < set(bound.capabilities)


def test_static_bound_applies_tool_policy_without_catalog_leak() -> None:
    agent = Agent("requirements", "Be concise.", model="m")
    config = RunConfig(tool_policy=ToolPolicy(allowed_tools=["read_file"]))
    runtime = runtime_for(agent, config)
    bound = runtime.requirements_bound()
    supports = static_worker_supports(agent, config, runtime.resolved)

    def tag(name: str) -> str:
        return tool_capability_tag(
            name,
            runtime.registry.get_schema(name),
            runtime.registry.tool_metadata(name),
        )

    assert tag("read_file") in bound.capabilities
    assert tag("bash") not in bound.capabilities
    assert tag("write_file") not in bound.capabilities
    assert tag("bash") in supports
    assert set(bound.capabilities) < set(supports)


def test_static_bound_uses_subagents_multimodal_and_activation_configuration() -> None:
    resolved = replace(RESOLVED, native_multimodal=True)
    agent = Agent(
        "requirements",
        "Be concise.",
        model="m",
        metadata={"available_skills": ["read-only"]},
        sub_agents={"helper": SubAgentConfig(model="m", description="Helper.")},
    )
    runtime = runtime_for(agent, RunConfig(), resolved=resolved)
    bound = runtime.requirements_bound()

    def tag(name: str) -> str:
        return tool_capability_tag(
            name,
            runtime.registry.get_schema(name),
            runtime.registry.tool_metadata(name),
        )

    for name in ("create_sub_task", "sub_task_status", "activate_skill", "read_image"):
        assert tag(name) in bound.capabilities


def test_definition_requirements_use_exact_visible_tools_and_atomic_models() -> None:
    @function_tool
    def visible(text: str) -> str:
        """Visible tool."""

        return text

    runtime = runtime_for(Agent("requirements", "Be concise.", model="m", tools=[visible]), RunConfig())
    task = runtime.compile("go", "turn")
    definition = runtime.definition(task)
    actual = definition_requirements(definition, handler_version=runtime.handler_version)

    expected_tools = {
        tool_capability_tag(
            schema["function"]["name"],
            schema,
            definition["capabilities"].get(schema["function"]["name"], {}),
        )
        for schema in definition["tools"]
    }
    expected_models = set(model_capability_tags(definition["model_binding"]))
    assert set(actual.capabilities) == expected_tools | expected_models
    assert definition["capabilities"] != actual.capabilities


def test_model_selection_requirements_use_trusted_resolution_without_compile() -> None:
    calls: Counter[str] = Counter()
    runtime = _commands_runtime(calls, tool_enabled=True)
    selection = {"model": "m", "route": {}, "metadata": {}}

    requirements = runtime.model_selection_requirements(selection)
    model_provider = runtime.config.model_provider
    assert model_provider is not None
    static_requirements = static_model_selection_requirements(
        selection,
        resolve_model=lambda request: resolve_model_selection(model_provider, request),
        handler_version=runtime.handler_version,
    )

    assert requirements.handler_version == runtime.handler_version
    assert requirements.capabilities == model_capability_tags(runtime.model_binding(runtime.llm, runtime.resolved))
    assert static_requirements == requirements
    assert all(tag.startswith("model:") for tag in requirements.capabilities)
    assert calls == Counter()


def test_model_selection_input_gate_matches_resolved_tags_and_handler() -> None:
    runtime = runtime_for(Agent("requirements", "Be concise.", model="m"), RunConfig())
    selection = {"model": "m", "route": {}, "metadata": {}}
    selection_ref = principal()["attributes"]
    requirements = runtime.model_selection_requirements(selection)
    binding = runtime.model_binding(runtime.llm, runtime.resolved)

    applied = _apply_core_item(
        InboxItem(
            "matching",
            "model_selection",
            {"selection": selection_ref},
            actor=principal(),
            feature_level=2,
            requirements=requirements,
        ),
        binding,
    )
    assert applied._payload["disposition"] == "applied"
    assert applied._payload["model_binding"] == binding

    understated = SessionRequirements.normalized(
        level=2,
        handler_version=requirements.handler_version,
        capabilities=(*requirements.capabilities, "model:" + "0" * 64),
    )
    rejected = _apply_core_item(
        InboxItem(
            "understated",
            "model_selection",
            {"selection": selection_ref},
            actor=principal(),
            feature_level=2,
            requirements=understated,
        ),
        binding,
    )
    assert rejected._payload["reason"] == "model_selection_requirements_mismatch"

    wrong_handler = replace(requirements, handler_version="handler/h1")
    handler_rejected = _apply_core_item(
        InboxItem(
            "handler",
            "model_selection",
            {"selection": selection_ref},
            actor=principal(),
            feature_level=2,
            requirements=wrong_handler,
        ),
        binding,
    )
    assert handler_rejected._payload["reason"] == "model_selection_handler_mismatch"


def test_tool_hook_cannot_change_start_turn_tool_identity() -> None:
    hook_calls = Counter()

    class Hook(BaseRuntimeHook):
        def before_tool_call(self, event):
            del event
            hook_calls["before"] += 1
            return None

    @function_tool
    def visible(text: str) -> str:
        """Visible tool."""

        return text

    plain = runtime_for(Agent("requirements", "Be concise.", model="m", tools=[visible]), RunConfig())
    hooked = runtime_for(Agent("requirements", "Be concise.", model="m", tools=[visible], hooks=[Hook()]), RunConfig())
    plain_task = plain.compile("go", "turn")
    hooked_task = hooked.compile("go", "turn")
    plain_definition = plain.definition(plain_task)
    hooked_definition = hooked.definition(hooked_task)

    assert plain_definition["tools"] == hooked_definition["tools"]
    assert definition_requirements(plain_definition, handler_version=plain.handler_version) == definition_requirements(
        hooked_definition, handler_version=hooked.handler_version
    )
    assert hook_calls == Counter()


def test_l2_session_input_drives_complete_turn_with_exact_requirements(tmp_path) -> None:
    runtime = runtime_for(
        Agent("requirements", "Be concise.", model="m"),
        RunConfig(workspace=tmp_path),
    )
    bound = runtime.requirements_bound()
    with SQLiteStore.standalone(str(tmp_path / "published.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        store.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=bound.capabilities,
        )
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="published")

            def session_spec(content):
                actor, _ = content.service_principal("requirements")
                return SessionSpec(
                    "published",
                    actor,
                    content.put_json(str(tmp_path)).to_dict(),
                    attributes={"seed": {"messages": [], "shared_state": {}}},
                    feature_level=2,
                    bootstrap=bound,
                )

            writer.create_session(session_spec, consumers=())

        push_fixture_input(
            store,
            "published",
            InboxItem(
                "input",
                "user",
                {"content": "go"},
                feature_level=2,
                requirements=bound,
            ),
        )
        runtime.heartbeat_store = store.heartbeat_store
        runtime.content_reader = fixture_content_reader(store, "published")
        drive(store, "published", runtime=runtime, _one_turn=True, _wait_for_lease=True)

        stored = next(row for row in store.read("published", limit=100).records if row.record.kind == "turn_started")
        hydrated = hydrate_stored_record(stored, hydration_for_session_store(store, "published")).record
        expected = definition_requirements(
            hydrated._payload["definition"],
            handler_version=runtime.handler_version,
        )
        assert hydrated._payload["requirements"] == expected.to_dict()
        assert expected.capabilities != ()
        kinds = [row.record.kind for row in store.read("published", limit=100).records]
        assert "op_planned" in kinds
        assert "op_completed" in kinds
        assert "turn_ended" in kinds


def test_l2_bootstrap_with_l1_user_input_drives_complete_turn(tmp_path) -> None:
    runtime = runtime_for(
        Agent("requirements", "Be concise.", model="m"),
        RunConfig(workspace=tmp_path),
    )
    bound = runtime.requirements_bound()
    with SQLiteStore.standalone(str(tmp_path / "transition.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        store.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=bound.capabilities,
        )
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="transition")

            def session_spec(content):
                actor, _ = content.service_principal("requirements")
                return SessionSpec(
                    "transition",
                    actor,
                    content.put_json(str(tmp_path)).to_dict(),
                    attributes={"seed": {"messages": [], "shared_state": {}}},
                    feature_level=2,
                    bootstrap=bound,
                )

            writer.create_session(session_spec, consumers=())

        push_fixture_input(
            store,
            "transition",
            InboxItem("input", "user", {"content": "go"}),
        )
        runtime.heartbeat_store = store.heartbeat_store
        runtime.content_reader = fixture_content_reader(store, "transition")
        drive(store, "transition", runtime=runtime, _one_turn=True, _wait_for_lease=True)

        records = store.read("transition", limit=100).records
        started = next(row for row in records if row.record.kind == "turn_started")
        hydrated = hydrate_stored_record(started, hydration_for_session_store(store, "transition")).record
        assert (
            hydrated._payload["requirements"]
            == definition_requirements(
                hydrated._payload["definition"],
                handler_version=runtime.handler_version,
            ).to_dict()
        )
        kinds = [row.record.kind for row in records]
        assert "op_planned" in kinds
        assert "op_completed" in kinds
        assert "turn_ended" in kinds


def test_l2_model_selection_is_used_by_real_drive_with_exact_tags(tmp_path) -> None:
    selected = replace(RESOLVED, backend="selected", model_id="B", requested_model="B", selected_model="B")
    primary_llm = ScriptedLLM([LLMResponse("primary")])
    selected_llm = ScriptedLLM([LLMResponse("selected")])
    provider = ModelMapProvider({"m": (primary_llm, RESOLVED), "B": (selected_llm, selected)}, "m")
    agent = Agent("requirements", "Be concise.", model="m")
    config = replace(RunConfig(workspace=tmp_path), model_provider=provider)
    runtime = Runtime(
        agent,
        config,
        RESOLVED,
        primary_llm,
        _unused_heartbeat_store,
        handler_version="handler/h2",
        model_resolver=lambda request: resolve_model_selection(provider, request),
    )
    bound = runtime.requirements_bound()
    selection = {"model": "B", "route": {}, "metadata": {}}
    selection_requirements = static_model_selection_requirements(
        selection,
        resolve_model=runtime.resolve_model,
        handler_version=runtime.handler_version,
    )
    turn_bound = SessionRequirements.normalized(
        level=max(bound.level, selection_requirements.level),
        handler_version=bound.handler_version,
        capabilities=(*bound.capabilities, *selection_requirements.capabilities),
    )

    with SQLiteStore.standalone(str(tmp_path / "selection.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        store.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=tuple(sorted(set(bound.capabilities) | set(selection_requirements.capabilities))),
        )
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="selection")

            def session_spec(content):
                actor, _ = content.service_principal("requirements")
                return SessionSpec(
                    "selection",
                    actor,
                    content.put_json(str(tmp_path)).to_dict(),
                    attributes={"seed": {"messages": [], "shared_state": {}}},
                    feature_level=2,
                    bootstrap=bound,
                )

            writer.create_session(session_spec, consumers=())

        push_fixture_input(
            store,
            "selection",
            InboxItem(
                "input",
                "user",
                {"content": "go"},
                feature_level=2,
                requirements=turn_bound,
            ),
        )
        push_fixture_input(
            store,
            "selection",
            InboxItem(
                "select",
                "model_selection",
                {"selection": selection},
                feature_level=2,
                requirements=selection_requirements,
            ),
        )
        runtime.heartbeat_store = store.heartbeat_store
        runtime.content_reader = fixture_content_reader(store, "selection")
        drive(store, "selection", runtime=runtime, _one_turn=True, _wait_for_lease=True)

        hydrated = [
            hydrate_stored_record(row, hydration_for_session_store(store, "selection")).record
            for row in store.read("selection", limit=100).records
        ]
        selection_applied = next(
            row for row in hydrated if row.kind == "input_applied" and row.payload["input"]["input_id"] == "select"
        )
        model_plan = next(row for row in hydrated if row.kind == "op_planned" and row.payload["purpose"] == "primary")
        assert selection_applied.payload["disposition"] == "applied"
        assert selection_applied.payload["model_binding"]["model"] == "B"
        assert model_plan.payload["request"]["model"] == "B"
        kinds = [record.kind for record in hydrated]
        assert "op_completed" in kinds
        assert hydrated[-1].kind == "turn_ended"
        assert selected_llm.steps == []


def test_actual_over_bound_fails_without_dispatch_or_mutation(tmp_path) -> None:
    @function_tool
    def unexpected(text: str) -> str:
        """Not declared in the pre-admission bound."""

        return text

    bound = SessionRequirements(level=2, handler_version="handler/h2", capabilities=())
    with SQLiteStore.standalone(str(tmp_path / "overflow.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        store.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=())
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="overflow")

            def session_spec(content):
                actor, _ = content.service_principal("requirements")
                return SessionSpec(
                    "overflow",
                    actor,
                    content.put_json(str(tmp_path)).to_dict(),
                    attributes={"seed": {"messages": [], "shared_state": {}}},
                    feature_level=2,
                    bootstrap=bound,
                )

            writer.create_session(session_spec, consumers=())

        push_fixture_input(
            store,
            "overflow",
            InboxItem(
                "input",
                "user",
                {"content": "go"},
                feature_level=2,
                requirements=bound,
            ),
        )
        inbox_before = store._rows("SELECT body FROM sk_inbox WHERE session_id='overflow'")
        runtime = runtime_for(
            Agent("requirements", "Be concise.", model="m", tools=[unexpected]),
            RunConfig(workspace=tmp_path),
            heartbeat_store=store.heartbeat_store,
        )
        runtime.content_reader = fixture_content_reader(store, "overflow")

        with pytest.raises(ValueError, match="unsatisfiable pre-admission requirements"):
            drive(store, "overflow", runtime=runtime, _one_turn=True, _wait_for_lease=True)

        records = store.read("overflow", limit=100).records
        assert not any(record.record.kind == "turn_started" for record in records)
        assert not any(record.record.kind == "turn_ended" for record in records)
        assert not any(record.record.kind == "op_planned" for record in records)
        assert any(
            record.record.kind == "input_applied" and record.record.payload["disposition"] == "queued" for record in records
        )
        assert store._rows("SELECT body FROM sk_inbox WHERE session_id='overflow'") == inbox_before


def test_endpoint_client_identity_uses_the_loaded_real_class() -> None:
    from vv_agent.llm.vv_llm_client import VvLlmClient
    from vv_agent.session.runtime import _is_vv_endpoint_client

    class DerivedClient(VvLlmClient):
        pass

    assert _is_vv_endpoint_client(VvLlmClient([]))
    assert _is_vv_endpoint_client(DerivedClient([]))
    assert not _is_vv_endpoint_client(object())


def test_prepare_model_keeps_real_endpoint_selection_semantics(tmp_path) -> None:
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

    class DerivedClient(VvLlmClient):
        pass

    original = DerivedClient(
        [
            EndpointTarget("a", "unused", "https://example.invalid"),
            EndpointTarget("b", "unused", "https://example.invalid"),
        ],
        backend=RESOLVED.backend,
        randomize_endpoints=False,
    )
    runtime = Runtime(
        Agent("provider", "Answer.", model="m"),
        RunConfig(workspace=tmp_path),
        RESOLVED,
        original,
        _unused_heartbeat_store,
        handler_version="handler/h2",
    )
    request = {
        "model": "m",
        "route": {},
        "metadata": {
            "vv_session": {
                "model_binding": runtime.model_binding(original, RESOLVED),
                "endpoint_order": ["b", "a"],
            }
        },
    }
    prepared = runtime.prepare_model(request, "primary", 1)
    assert isinstance(prepared, DerivedClient)
    assert [target.endpoint_id for target in prepared.endpoint_targets] == ["b"]
    assert prepared.max_retries_per_endpoint == 1
    assert prepared.randomize_endpoints is False
    assert [target.endpoint_id for target in original.endpoint_targets] == ["a", "b"]


def test_first_planning_does_not_import_optional_provider_client() -> None:
    script = r"""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from vv_agent import Agent, RunConfig, Runner, ScriptedModelProvider
from vv_agent.types import LLMResponse

with TemporaryDirectory() as root:
    provider = ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("cold ok")])
    handle = Runner.start(
        Agent("cold", "Answer.", model="scripted"),
        "go",
        run_config=RunConfig(workspace=Path(root), model_provider=provider),
    )
    result = handle.result(timeout=5)
    print(json.dumps({
        "output": result.final_output,
        "optional_client_loaded": "vv_agent.llm.vv_llm_client" in sys.modules,
    }))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value == {"output": "cold ok", "optional_client_loaded": False}


def test_runner_configured_child_publishes_l2_bootstrap_and_typed_start(tmp_path) -> None:
    provider = FixedModelProvider(
        ScriptedLLM(
            [
                LLMResponse(
                    "",
                    [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})],
                ),
                LLMResponse("child done"),
                LLMResponse("parent done"),
            ]
        ),
        RESOLVED,
    )
    parent = Agent(
        "parent",
        "Parent instructions.",
        model="m",
        sub_agents={"worker": SubAgentConfig(model="m", description="Worker instructions.")},
    )
    handle = Runner.start(parent, "go", run_config=RunConfig(workspace=tmp_path, model_provider=provider))
    try:
        handle.join(timeout=20)
        assert handle.done()
        store = handle.kernel.store
        sessions = store.list_sessions()
        assert len(sessions) == 2
        child_id = next(sid for sid in sessions if sid != handle.session_id)
        _assert_l2_session_requirements(store, handle.session_id)
        _assert_l2_session_requirements(store, child_id)
        assert handle.result(timeout=5).final_output == "parent done"
    finally:
        handle.close()


def test_real_handoff_and_grandchild_publish_l2_requirements(tmp_path) -> None:
    third = Agent("third", "Third.", model="m")
    second = Agent("second", "Second.", model="m", handoffs=[handoff(agent=third)])
    first = Agent("first", "First.", model="m", handoffs=[handoff(agent=second)])
    scripted = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("handoff-1", "transfer_to_second", {"input": "second input"})]),
            LLMResponse("", [ToolCall("handoff-2", "transfer_to_third", {"input": "third input"})]),
            LLMResponse("third done"),
        ]
    )
    provider = FixedModelProvider(scripted, RESOLVED)
    config = RunConfig(workspace=tmp_path, model_provider=provider, max_handoffs=3)

    def make_runtime(agent: Agent) -> Runtime:
        return Runtime(
            agent,
            config,
            RESOLVED,
            provider.llm,
            _unused_heartbeat_store,
            handler_version="handler/h2",
        )

    runtimes = [make_runtime(first), make_runtime(second), make_runtime(third)]
    runtime = runtimes[0]
    with SQLiteStore.standalone(str(tmp_path / "handoff.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        bound = _create_l2_session(raw_store, store, "root", runtime, tmp_path)
        supports = tuple(sorted(set().union(*(set(item.requirements_bound().capabilities) for item in runtimes))))
        raw_store.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=supports,
        )
        push_fixture_input(
            store,
            "root",
            InboxItem("go", "user", {"content": "go"}, feature_level=2, requirements=bound),
        )
        runtime.heartbeat_store = store.heartbeat_store
        runtime.content_reader = fixture_content_reader(store, "root")

        def drain(sid: str, current: Runtime) -> None:
            drive(store, sid, runtime=current, _one_turn=True)
            for stored in store.read_state(sid)[1]:
                record = stored.record
                if record.kind != "op_parked" or record._payload["handle"]["kind"] != "child":
                    continue
                for child in child_handles(record._payload["handle"]):
                    child_runtime = current.child_runtime(store, child["session_id"])
                    drain(child["session_id"], child_runtime)
                    with store.atomic() as tx:
                        child_delivery(store, tx, child["session_id"])
            drive(store, sid, runtime=current, _one_turn=True)

        drain("root", runtime)
        sessions = store.list_sessions()
        assert len(sessions) == 3
        for sid in sessions:
            _assert_l2_session_requirements(store, sid)
        admission_modes = []
        for sid in sessions[1:]:
            created = store.read_state(sid)[1][0]
            hydrated = hydrate_stored_record(created, hydration_for_session_store(store, sid)).record
            admission_modes.append(hydrated.payload["attributes"]["child_admission"]["mode"])
        assert admission_modes == ["handoff", "handoff"]
        assert scripted.steps == []


def test_manual_child_without_trusted_bootstrap_is_rejected_before_creation(tmp_path) -> None:
    @function_tool
    def child_task() -> str:
        raise AssertionError("the host child callback must not execute")

    runtime, _ = runtime_for_steps(
        Agent("parent", "Parent.", model="m", tools=[child_task]),
        RunConfig(workspace=tmp_path),
        [LLMResponse("", [ToolCall("child", "child_task", {})]), LLMResponse("parent done")],
    )
    with SQLiteStore.standalone(str(tmp_path / "manual-child.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        bound = _create_l2_session(raw_store, store, "parent", runtime, tmp_path)
        parent_created = store.read_state("parent")[1][0].record
        runtime.children = {
            "child_task": lambda plan: ChildSession(
                SessionSpec(
                    "manual-child",
                    parent_created.payload["principal"],
                    parent_created.payload["workspace"],
                    parent_session_id=plan.session_id,
                    attributes={"child_admission": {"mode": "custom"}},
                ),
                "child input",
            )
        }
        push_fixture_input(
            store,
            "parent",
            InboxItem("go", "user", {"content": "go"}, feature_level=2, requirements=bound),
        )
        runtime.heartbeat_store = store.heartbeat_store
        runtime.content_reader = fixture_content_reader(store, "parent")

        with pytest.raises(ValueError, match="requires an explicit trusted child bootstrap"):
            drive(store, "parent", runtime=runtime, _one_turn=True)

        assert "manual-child" not in store.list_sessions()
        rows = store.read_state("parent")[1]
        assert not any(row.record.kind in {"op_parked", "turn_ended"} for row in rows)


class _AutoApproval(ApprovalProvider):
    def __init__(self, broker: ApprovalBroker, decision: ApprovalDecision) -> None:
        self.broker = broker
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    def should_request(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        assert self.broker.resolve(request.request_id, self.decision)
        return None


def test_live_approval_provider_uses_trusted_service_ingress(tmp_path) -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        calls.append("effect")
        return "ok"

    broker = ApprovalBroker()
    decision = ApprovalDecision(
        "allow_session",
        reason="trusted approval service",
        metadata={"channel": "live-provider"},
    )
    provider = _AutoApproval(broker, decision)
    llm = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("effect-1", "effect", {})]),
            LLMResponse("finished"),
        ]
    )
    model_provider = FixedModelProvider(llm, RESOLVED)
    handle = Runner.start(
        Agent("approval", "Approval.", model="m", tools=[effect]),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            approval_provider=provider,
            approval_broker=broker,
        ),
    )
    try:
        handle.join(timeout=20)
        assert handle.done()
        store = handle.kernel.store
        rows = store.read_state(handle.session_id)[1]
        park = next(row for row in rows if row.record.kind == "op_parked")
        answer = next(
            row for row in rows if row.record.kind == "input_applied" and row.record.payload["input"]["kind"] == "approval_answer"
        )
        hydrated = hydrate_stored_record(answer, hydration_for_session_store(store, handle.session_id)).record
        item = hydrated.payload["input"]
        assert item["feature_level"] == 1
        assert item.get("requirements") is None
        assert store.read_state(handle.session_id)[1][0].record.feature_level == 2
        assert item["actor"]["subject_kind"] == "service"
        assert item["actor"]["subject_id"] == "approval-provider"
        assert set(item["actor"]["attributes"]) == {"content_id", "mac", "size", "key_id", "labels"}
        assert item["payload"]["request_id"] == park.record.payload["handle"]["request_id"]
        assert item["payload"]["request_digest"] == park.record.payload["handle"]["request_digest"]
        assert item["payload"]["decision"] == "allow_session"

        input_links = _owner_link_values(store, handle.session_id, input_id=item["input_id"])
        assert set(input_links) == {
            APPROVAL_METADATA,
            APPROVAL_REASON,
            APPROVAL_SCOPE,
            INPUT_ACTOR_ATTRIBUTES,
        }
        assert input_links[APPROVAL_SCOPE] == ["effect"]
        assert input_links[APPROVAL_REASON] == decision.reason
        assert input_links[APPROVAL_METADATA] == decision.metadata
        assert input_links[INPUT_ACTOR_ATTRIBUTES] == {"role": "approval-provider"}

        record_links = _owner_link_values(store, handle.session_id, record_seq=answer.seq)
        assert record_links[INPUT_ACTOR_ATTRIBUTES] == {"role": "approval-provider"}
        assert record_links["input.payload.scope"] == ["effect"]
        assert record_links["input.payload.reason"] == decision.reason
        assert record_links["input.payload.metadata"] == decision.metadata
        model_operation_ids = {
            row.record.operation_id
            for row in rows
            if row.record.kind == "op_planned" and row.record.payload["op_kind"] == "model"
        }
        assert (
            len([row for row in rows if row.record.kind == "op_completed" and row.record.operation_id in model_operation_ids])
            == 2
        )
        assert len(calls) == 1
        assert len(provider.requests) == 1
        assert handle.result(timeout=5).final_output == "finished"
    finally:
        handle.close()


def test_approval_answer_cold_recovers_without_repeating_provider_or_model(tmp_path) -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        calls.append("effect")
        return "ok"

    broker = ApprovalBroker()
    provider = _AutoApproval(
        broker,
        ApprovalDecision("allow", reason="cold recovery", metadata={"channel": "cold"}),
    )
    runtime, llm = runtime_for_steps(
        Agent("approval", "Approval.", model="m", tools=[effect]),
        RunConfig(workspace=tmp_path, approval_provider=provider, approval_broker=broker),
        [LLMResponse("", [ToolCall("effect-1", "effect", {})]), LLMResponse("finished")],
    )

    def cut(point: str, record: Record | None) -> None:
        if (
            point == "after_commit"
            and record is not None
            and record.kind == "input_applied"
            and record.payload["input"]["kind"] == "approval_answer"
        ):
            raise _Restart

    with SQLiteStore.standalone(str(tmp_path / "approval-recovery.sqlite")) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        bound = _create_l2_session(raw_store, store, "approval", runtime, tmp_path)
        push_fixture_input(
            store,
            "approval",
            InboxItem("go", "user", {"content": "go"}, feature_level=2, requirements=bound),
        )
        runtime.heartbeat_store = store.heartbeat_store
        runtime.content_reader = fixture_content_reader(store, "approval")
        runtime.hook = cut
        with pytest.raises(_Restart):
            drive(store, "approval", runtime=runtime, _one_turn=True)
        assert len(provider.requests) == 1
        initial_rows = store.read_state("approval")[1]
        initial_model_operation_ids = {
            row.record.operation_id
            for row in initial_rows
            if row.record.kind == "op_planned" and row.record.payload["op_kind"] == "model"
        }
        first_model_ops = [
            row.record.operation_id
            for row in initial_rows
            if row.record.kind == "op_completed" and row.record.operation_id in initial_model_operation_ids
        ]
        assert len(first_model_ops) == 1

        recovered = Runtime(
            runtime.agent,
            runtime.config,
            RESOLVED,
            llm,
            store.heartbeat_store,
            handler_version=runtime.handler_version,
        )
        recovered.heartbeat_store = store.heartbeat_store
        recovered.content_reader = fixture_content_reader(store, "approval")
        drive(store, "approval", runtime=recovered, _one_turn=True)

        rows = store.read_state("approval")[1]
        model_operation_ids = {
            row.record.operation_id
            for row in rows
            if row.record.kind == "op_planned" and row.record.payload["op_kind"] == "model"
        }
        model_ops = [
            row.record.operation_id
            for row in rows
            if row.record.kind == "op_completed" and row.record.operation_id in model_operation_ids
        ]
        assert len(model_ops) == 2
        assert len(set(model_ops)) == 2
        assert model_ops[:1] == first_model_ops
        assert calls == ["effect"]
        assert len(provider.requests) == 1
        assert llm.steps == []
        assert rows[-1].record.kind == "turn_ended"
