"""W3 child routing and worker feature levels share their durable evidence."""

from __future__ import annotations

from contextlib import AbstractContextManager, closing
from dataclasses import replace
from typing import Any, Never, cast

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.llm.base import LLMClient
from vv_agent.model import ScriptedModelProvider, resolve_run_model
from vv_agent.run_config import ToolPolicy
from vv_agent.session.children import ChildSession, create_child
from vv_agent.session.environments import EnvironmentCapabilities, IsolationMode, NetworkMode
from vv_agent.session.records import digest, make_record
from vv_agent.session.runtime import Runtime, TrustedEnvironmentBinding
from vv_agent.session.store import SessionStore
from vv_agent.session.surfaces import SessionDriver

from .helpers import (
    append_fixture_records,
    content_ref,
    create_fixture_session,
    fixture_child_spec,
    fixture_key_provider,
    wrap_fixture_store,
)
from .helpers import (
    record as fixture_record,
)


class _ToolProvider:
    def submit(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError(f"unexpected submit: {args!r} {kwargs!r}")

    def query(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError(f"unexpected query: {args!r} {kwargs!r}")

    def cancel(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError(f"unexpected cancel: {args!r} {kwargs!r}")

    def authenticate(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError(f"unexpected authenticate: {args!r} {kwargs!r}")


def _unexpected_store() -> AbstractContextManager[SessionStore]:
    raise AssertionError("unexpected runtime store open")


def _binding() -> TrustedEnvironmentBinding:
    return TrustedEnvironmentBinding(
        capabilities=EnvironmentCapabilities.create(
            environment_id="environment/ordinary-child-parent",
            generation=3,
            isolation=IsolationMode.CONTAINER,
            network=NetworkMode.NONE,
            enforces_network_policy=True,
            enforces_resource_limits=True,
            persistent=False,
            environment_capabilities=("command.execute",),
            network_policy_ref_digest="1" * 64,
            isolation_policy_ref_digest="2" * 64,
        ),
        descriptor={"provider": "w3-host", "workspace": "/workspace/parent"},
        operational_policy={
            "network_policy_digest": "3" * 64,
            "isolation_policy_digest": "4" * 64,
        },
    )


def _level_three_turn(session_id: str, *, turn_id: str = "parent/w3") -> Any:
    definition = content_ref(f"definition/{session_id}/{turn_id}")
    handler_version = "handler/v3"
    return make_record(
        "turn_started",
        session_id=session_id,
        turn_id=turn_id,
        payload={
            "input_ids": [],
            "definition": definition,
            "definition_digest": digest(definition),
            "handler_version": handler_version,
            "budget": content_ref(f"budget/{session_id}/{turn_id}"),
            "binding": "task",
            "generation": 1,
            "policy_digest": None,
            "requirements": {
                "level": 3,
                "handler_version": handler_version,
                "capabilities": [],
            },
            "tokenizer_binding": None,
            "environment": {
                "descriptor": content_ref(f"environment/{session_id}/{turn_id}"),
                "environment_id": f"environment/{session_id}",
                "generation": 1,
                "isolation": "container",
                "network": "none",
                "enforces_network_policy": True,
                "enforces_resource_limits": True,
                "persistent": False,
                "environment_capabilities": ("command.execute",),
                "capabilities_digest": None,
                "network_policy_ref_digest": None,
                "isolation_policy_ref_digest": None,
                "operational_policy": content_ref(f"operational-policy/{session_id}/{turn_id}"),
            },
        },
        feature_level=3,
    )


def test_for_agent_drops_w3_binding_from_ordinary_child_runtime(tmp_path):
    model_provider = ScriptedModelProvider.new("scripted", "parent", [])
    parent_agent = Agent("parent", "Parent.", model="parent")
    parent_config = RunConfig(workspace=tmp_path / "parent", model_provider=model_provider)
    llm, resolved = resolve_run_model(agent=parent_agent, run_config=parent_config)
    parent_provider = _ToolProvider()
    binding = _binding()

    def resolve_selection(_selection: dict[str, Any]) -> tuple[LLMClient, ResolvedModelConfig]:
        return llm, resolved

    parent_runtime = Runtime(
        parent_agent,
        parent_config,
        resolved,
        llm,
        _unexpected_store,
        providers={"host/provider": parent_provider},
        host_bindings={"host_state": {"tenant": "default"}},
        trusted_environment_binding=binding,
        model_resolver=resolve_selection,
        handler_version="test-handler",
        ttl_ms=120_000,
        heartbeat_seconds=0.5,
        poll_ms=250,
        cancellation_grace=0.5,
        lease_retry_attempts=7,
        lease_retry_base_seconds=0.02,
        lease_retry_cap_seconds=2.0,
        model_timeout=123,
        tool_timeout=234,
    )
    child_config = RunConfig(
        workspace=tmp_path / "child",
        model_provider=model_provider,
        tool_providers={"child/provider": _ToolProvider()},
        tool_policy=ToolPolicy(disallowed_tools=["forbidden"]),
        shared_state={"seed": "child"},
    )
    child_agent = Agent("worker", "Worker.", model="child")

    child_runtime = parent_runtime.for_agent(child_agent, child_config)

    assert parent_runtime.trusted_environment_binding is binding
    assert parent_runtime.turn_environment_binding() == binding
    assert child_runtime.trusted_environment_binding is None
    assert child_runtime.turn_environment_binding() is None

    assert child_runtime.agent is child_agent
    assert child_runtime.resolved.model_id == "child"
    assert child_runtime.llm is llm
    assert child_runtime.model_resolver is parent_runtime.model_resolver
    assert child_runtime.providers["host/provider"] is parent_provider
    assert child_runtime.providers["child/provider"] is child_config.tool_providers["child/provider"]
    assert child_runtime.host_bindings is parent_runtime.host_bindings
    assert child_runtime.children is parent_runtime.children
    assert child_runtime.clock is parent_runtime.clock
    assert child_runtime.lease_keeper is parent_runtime.lease_keeper
    assert child_runtime.invoker is parent_runtime.invoker
    assert child_runtime.lease_retry_jitter is parent_runtime.lease_retry_jitter
    assert child_runtime.lease_retry_sleep is parent_runtime.lease_retry_sleep
    assert child_runtime.ttl_ms == parent_runtime.ttl_ms
    assert child_runtime.heartbeat_seconds == parent_runtime.heartbeat_seconds
    assert child_runtime.poll_ms == parent_runtime.poll_ms
    assert child_runtime.cancellation_grace == parent_runtime.cancellation_grace
    assert child_runtime.lease_retry_attempts == parent_runtime.lease_retry_attempts
    assert child_runtime.lease_retry_base_seconds == parent_runtime.lease_retry_base_seconds
    assert child_runtime.lease_retry_cap_seconds == parent_runtime.lease_retry_cap_seconds
    assert child_runtime.model_timeout == parent_runtime.model_timeout
    assert child_runtime.tool_timeout == parent_runtime.tool_timeout
    assert child_runtime.handler_version == parent_runtime.handler_version
    assert child_runtime.hook is parent_runtime.hook
    assert child_runtime.hook is Runtime.__dataclass_fields__["hook"].default
    assert child_runtime.config.workspace == child_config.workspace
    assert child_runtime.config.tool_policy == child_config.tool_policy
    assert child_runtime.config.shared_state == child_config.shared_state
    assert child_runtime.memory_manager == replace(parent_runtime.memory_manager)


@pytest.mark.persistent_store
def test_level_three_parent_admits_ordinary_child_from_child_ledger(store) -> None:
    handler_version = "1"
    child_requirements = SessionRequirements(2, handler_version, ())
    provider = fixture_key_provider()
    store._rows("UPDATE sk_settings SET write_level=3 WHERE singleton=1")
    store.configure_worker(level=3, handler_versions=(handler_version, "handler/v3"), capabilities=())
    store.configure_reader(level=3)
    content_store = wrap_fixture_store(store, provider)
    create_fixture_session(content_store, "parent", workspace="/tmp/parent")
    append_fixture_records(
        content_store,
        "parent",
        (_level_three_turn("parent"),),
        commit_id="parent-w3",
    )
    assert store._one("SELECT level FROM sk_session WHERE session_id='parent'")[0] == 3

    child_spec = replace(
        fixture_child_spec(content_store, "parent", "child"),
        feature_level=2,
        bootstrap=child_requirements,
    )
    admission = fixture_record("op_planned", sid="parent", tid="parent/admission", oid="parent/op")
    with content_store.atomic() as tx:
        create_child(tx, ChildSession(child_spec, "ordinary child"), admission, 1)

    assert store._one("SELECT level,handler_version FROM sk_session WHERE session_id='child'") == (2, handler_version)

    store.configure_worker(level=2, handler_versions=(handler_version,), capabilities=())
    claimed = store.claim_runnable(
        owner="ordinary-level-two-worker",
        ttl_ms=10_000,
        limit=10,
        work_kinds=("drive",),
        projection_consumers=(),
    )
    assert [(item.work.session_id, item.work.kind) for item in claimed] == [("child", "drive")]
    assert claimed[0].lease is not None
    assert store.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_level_three_worker_reads_and_discovers_level_three_work(store) -> None:
    provider = fixture_key_provider()
    store._rows("UPDATE sk_settings SET write_level=3 WHERE singleton=1")
    store.configure_worker(level=3, handler_versions=("handler/v3",), capabilities=())
    content_store = wrap_fixture_store(store, provider)
    create_fixture_session(content_store, "level-three", workspace="/tmp/level-three")
    append_fixture_records(
        content_store,
        "level-three",
        (_level_three_turn("level-three"),),
        commit_id="level-three-work",
    )
    store.configure_reader(level=2)

    with closing(
        SessionDriver(
            store=store,
            content_key_provider=provider,
            worker_level=3,
            handler_versions=("handler/v3",),
            capabilities=(),
        )
    ) as driver:
        drive_store = cast(Any, driver.store)
        assert drive_store.reader_level == 3
        assert drive_store.worker_level == 3
        assert driver.store.read("level-three", limit=10).head_seq == 2

        # Reader configuration may widen read support, but it never selects or
        # narrows the declared execution level.
        store.configure_reader(level=2)
        assert store.worker_level == 3
        factory = driver._drive_store_factory
        assert factory is not None
        with factory() as drive:
            assert drive.reader_level == 3
            assert drive.worker_level == 3
            claimed = drive.claim_runnable(
                owner="level-three-worker",
                ttl_ms=10_000,
                limit=10,
                work_kinds=("drive",),
                projection_consumers=(),
            )
            assert [(item.work.session_id, item.work.kind) for item in claimed] == [("level-three", "drive")]
            assert claimed[0].lease is not None
            assert drive.release(claimed[0].lease)


def test_for_agent_preserves_custom_runtime_hook(tmp_path):
    model_provider = ScriptedModelProvider.new("scripted", "parent", [])
    parent_agent = Agent("parent", "Parent.", model="parent")
    parent_config = RunConfig(workspace=tmp_path / "parent", model_provider=model_provider)
    llm, resolved = resolve_run_model(agent=parent_agent, run_config=parent_config)
    hook_calls: list[str] = []

    def host_hook(point: str, _record: Any) -> None:
        hook_calls.append(point)

    parent_runtime = Runtime(
        parent_agent,
        parent_config,
        resolved,
        llm,
        _unexpected_store,
        hook=host_hook,
    )
    child_runtime = parent_runtime.for_agent(
        Agent("worker", "Worker.", model="child"),
        RunConfig(workspace=tmp_path / "child"),
    )

    assert child_runtime.hook is host_hook
    child_runtime.hook("after_drive", None)
    assert hook_calls == ["after_drive"]
