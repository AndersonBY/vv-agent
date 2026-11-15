"""The content-reader seam from turn-start references into the session driver."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    ContentLabels,
    ContentRef,
    ContentUnavailable,
)
from vv_agent.kernel._content_store import DurableContentStore
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import MissingContent, MissingContentLink
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
from vv_agent.kernel._record_codec import RecordError
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_reader import DurableContentReader, input_owner, record_owner
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.records import SessionSpec, digest, make_record
from vv_agent.session.runtime import Runtime
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

TENANT = "tenant-a"
SESSION = "session-a"
TURN = f"{SESSION}/turn/start"
MASTER_KEY = b"k" * 32


def _stored_ref(label: str) -> ContentRef:
    body = label.encode()
    return ContentRef(
        content_id=f"content/{label}",
        mac=sha256(body).hexdigest(),
        size=len(body),
        key_id="key/session",
        labels=ContentLabels(),
    )


def _principal() -> dict[str, Any]:
    return {
        "tenant_id": TENANT,
        "subject_id": "user/session-a",
        "subject_kind": "user",
        "attributes": _stored_ref("principal-attributes").to_dict(),
    }


def _definition() -> dict[str, Any]:
    task = AgentTask(
        task_id=TURN,
        model="scripted",
        prompt_bundle=PromptBundle((PromptSection("system", "Work precisely."),)),
        user_prompt="hydrate the definition",
    )
    return {
        "agent_name": "reader-test",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": [],
            "context_length": None,
            "max_output_tokens": None,
        },
        "capabilities": {},
        "task": task.to_dict(),
        "tools": [],
    }


@contextmanager
def _driver(tmp_path: Path, *, reader: bool) -> Iterator[tuple[_Driver, Any, ContentRef]]:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "reader.sqlite")) as root:
        root.install_schema()
        store = cast(SQLiteStore, root.for_tenant(TENANT))
        with store.atomic() as tx:
            tx.create(
                SessionSpec(
                    SESSION,
                    _principal(),
                    _stored_ref("workspace").to_dict(),
                    attributes=_stored_ref("attributes").to_dict(),
                ),
                consumers=(),
            )

        definition = _definition()
        lease = store.acquire(SESSION, owner="reader-test", ttl_ms=15_000)
        assert lease is not None
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            ref = content.put(TENANT, SESSION, canonical_json_bytes(definition), ContentLabels())
            tx.append(
                SESSION,
                lease=lease,
                expected_seq=1,
                commit_id="reader-start",
                records=(
                    make_record(
                        "turn_started",
                        session_id=SESSION,
                        turn_id=TURN,
                        payload={
                            "input_ids": [],
                            "definition": ref.to_dict(),
                            "definition_digest": digest(ref.to_dict()),
                            "handler_version": "1",
                            "budget": _stored_ref("budget").to_dict(),
                            "binding": None,
                            "generation": 1,
                        },
                    ),
                ),
            )
            content.link_record(TENANT, SESSION, 2, ref, "definition")

        content = SQLiteContentStore(root, provider)
        content_reader = DurableContentReader(content, tenant_id=TENANT, session_id=SESSION) if reader else None
        runtime = Runtime(
            Agent("reader-test", "Work precisely."),
            RunConfig(session_memory_enabled=False),
            ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
            ScriptedLLM([]),
            store.heartbeat_store,
            content_reader=content_reader,
        )
        driver = _Driver(store, SESSION, runtime, _Scope(lease, runtime, now_ms=store._now()))
        yield driver, content_reader, ref


def test_durable_reader_validates_owner_before_get_and_supports_both_owner_kinds() -> None:
    calls: list[tuple[Any, ...]] = []

    class Store:
        def assert_link(self, tenant_id, session_id, ref, slot, **owner):
            calls.append(("assert_link", tenant_id, session_id, ref, slot, owner))
            return ref

        def get(self, tenant_id, session_id, ref):
            calls.append(("get", tenant_id, session_id, ref))
            return b"body"

    reader = DurableContentReader(cast(DurableContentStore, Store()), tenant_id=TENANT, session_id=SESSION)
    ref = _stored_ref("definition")

    assert reader.read(TENANT, SESSION, record_owner(2), "definition", ref) == b"body"
    assert reader.read(TENANT, SESSION, input_owner("input-1"), "input", ref) == b"body"
    assert [call[0] for call in calls] == ["assert_link", "get", "assert_link", "get"]
    assert calls[0][-1] == {"record_seq": 2}
    assert calls[2][-1] == {"input_id": "input-1"}


def test_durable_reader_rejects_binding_and_missing_link_without_returning_content() -> None:
    calls: list[str] = []

    class Store:
        def assert_link(self, tenant_id, session_id, ref, slot, **owner):
            calls.append("assert_link")
            raise ContentUnavailable("missing link")

        def get(self, tenant_id, session_id, ref):
            calls.append("get")
            return b"plaintext"

    reader = DurableContentReader(cast(DurableContentStore, Store()), tenant_id=TENANT, session_id=SESSION)
    ref = _stored_ref("definition")

    with pytest.raises(MissingContentLink):
        reader.read("tenant-b", SESSION, record_owner(2), "definition", ref)
    assert calls == []
    with pytest.raises(MissingContentLink):
        reader.read(TENANT, SESSION, record_owner(2), "definition", ref)
    assert calls == ["assert_link"]


def test_durable_reader_maps_missing_content_after_a_valid_link() -> None:
    class Store:
        def assert_link(self, tenant_id, session_id, ref, slot, **owner):
            return ref

        def get(self, tenant_id, session_id, ref):
            raise ContentUnavailable("content gone")

    reader = DurableContentReader(cast(DurableContentStore, Store()), tenant_id=TENANT, session_id=SESSION)
    with pytest.raises(MissingContent):
        reader.read(TENANT, SESSION, record_owner(2), "definition", _stored_ref("definition"))


def test_core_driver_hydrates_turn_task_from_the_same_session_reader(tmp_path: Path) -> None:
    with _driver(tmp_path, reader=True) as (driver, content_reader, ref):
        assert isinstance(driver, CoreDriver)
        assert driver.content_reader is content_reader
        assert driver.hydrator_for(2).binding == (TENANT, SESSION, record_owner(2))
        assert driver.task().user_prompt == "hydrate the definition"
        assert driver._task().prompt_bundle.flatten() == "Work precisely."
        assert driver.shared_state() == {}
        assert driver.state.turns[TURN].start._payload["definition"] == ref.to_dict()

        with pytest.raises(MissingContentLink):
            driver.hydrator_for(3).definition(ref)
        with pytest.raises(ValueError, match="session_id cannot be used"):
            driver.hydrator_for(SESSION)


def test_driver_without_reader_keeps_hydration_unavailable(tmp_path: Path) -> None:
    with _driver(tmp_path, reader=False) as (driver, content_reader, ref):
        assert content_reader is None
        assert driver.content_reader is None
        assert driver.state.turns[TURN].start._payload["definition"] == ref.to_dict()
        with pytest.raises(RecordError, match="content hydration is unavailable"):
            driver.task()
        with pytest.raises(RecordError, match="content hydration is unavailable"):
            driver._task()
        with pytest.raises(RecordError, match="content hydration is unavailable"):
            driver.durable_shared_state()
        with pytest.raises(RecordError, match="content hydration is unavailable"):
            driver.hydrator_for(2)
