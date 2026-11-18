"""S1 digest-domain regressions for clear message and replay identities."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from session.app_server_s1_helpers import bind_development_content, create_s1_session
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._context import message_ids
from vv_agent.kernel._hydration import reference_digest
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.export import export_session
from vv_agent.session.replay import list_events
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import SessionStore
from vv_agent.types import Message, ToolArtifactRef

SESSION = "thread"
PROMPT = "x"
ARTIFACT_PATH = ".vv-agent/artifacts/a/call-x.txt"
ARTIFACT_BYTES = b"x"
APP_SERVER_ATTRIBUTES = {"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}}


def _text_scalars(value: Any) -> Iterable[str]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = json.loads(bytes(value).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            yield bytes(value).decode("utf-8", "ignore")
            return
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _text_scalars(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _text_scalars(item)


def _seed_reference(store: SessionStore, session_id: str) -> ContentRef:
    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("session store must be tenant-bound")
    attributes_ref = ContentRef.from_dict(store.read(session_id, limit=1).records[0].record.payload["attributes"])
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)
        attributes = writer.get_json(attributes_ref)
    return ContentRef.from_dict(attributes["seed"]["messages"][0])


def test_message_ids_retain_content_bound_contract40_identity() -> None:
    messages = [Message("user", PROMPT), Message("user", PROMPT), Message("assistant", "")]

    ids = message_ids(messages)
    assert len(ids) == len(messages)
    assert ids[0].startswith("0/")
    assert ids[1].startswith("1/")
    assert ids[2].startswith("2/")
    assert message_ids([Message("user", "different")] * len(messages)) != ids
    assert sha256(canonical_json_bytes(messages[0].to_dict())).hexdigest() in ids[0]


def test_short_artifact_and_message_do_not_enter_clear_replay_or_skeleton(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    artifact_sha256 = sha256(ARTIFACT_BYTES).hexdigest()
    message = Message(
        "user",
        PROMPT,
        artifact_ref=ToolArtifactRef(ARTIFACT_PATH, "text/plain", "utf-8", len(ARTIFACT_BYTES), artifact_sha256),
    )
    with SQLiteStore.standalone(str(tmp_path / "digest-domains.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        bind_development_content(store)
        create_s1_session(
            store,
            SESSION,
            APP_SERVER_ATTRIBUTES | {"seed": {"messages": [message.to_dict()], "shared_state": {}}},
        )

        page = list_events(store, SESSION, limit=10)
        skeleton = export_session(store, SESSION).to_dict()
        record_body = store.read(SESSION, limit=1).records[0].record.encode()
        clear_values = [page, skeleton, record_body]

        assert {PROMPT, ARTIFACT_PATH, ARTIFACT_BYTES.decode()}.isdisjoint(
            {text for value in clear_values for text in _text_scalars(value)}
        )
        naked_hashes = {
            sha256(canonical_json_bytes(message.to_dict())).hexdigest(),
            sha256(PROMPT.encode()).hexdigest(),
            sha256(ARTIFACT_PATH.encode()).hexdigest(),
            artifact_sha256,
        }
        assert naked_hashes.isdisjoint({text for value in clear_values for text in _text_scalars(value)})
        assert page["throughCursor"]["recordDigest"] == store.read(SESSION, limit=1).records[0].record.digest
        assert page["throughCursor"]["recordDigest"] not in naked_hashes
        assert f'"{PROMPT}"' not in caplog.text
        assert ARTIFACT_PATH not in caplog.text

        attributes_ref = ContentRef.from_dict(store.read(SESSION, limit=1).records[0].record.payload["attributes"])
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=SESSION)
            attributes = writer.get_json(attributes_ref)
            hydrated = Message.from_dict(writer.get_json(ContentRef.from_dict(attributes["seed"]["messages"][0])))
        assert hydrated.artifact_ref is not None
        assert hydrated.artifact_ref.sha256 == artifact_sha256


def test_same_short_message_has_tenant_and_session_scoped_content_identity(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "tenant-session-domains.sqlite")) as root:
        root.install_schema()
        root.configure_input_admission(allow_unsigned=True)
        bind_development_content(root)
        references = []
        clear_record_bodies = []
        for tenant_id, session_id in (("tenant-a", SESSION), ("tenant-b", SESSION), ("tenant-a", "other-thread")):
            store = root.for_tenant(tenant_id)
            create_s1_session(
                store,
                session_id,
                APP_SERVER_ATTRIBUTES | {"seed": {"messages": [Message("user", PROMPT).to_dict()], "shared_state": {}}},
            )
            references.append(_seed_reference(store, session_id))
            clear_record_bodies.append(store.read(session_id, limit=1).records[0].record.encode())

        assert len({ref.content_id for ref in references}) == len(references)
        assert len({ref.key_id for ref in references}) == len(references)
        assert len({ref.mac for ref in references}) == len(references)
        assert len({reference_digest(ref) for ref in references}) == len(references)

        naked_hashes = {
            sha256(PROMPT.encode()).hexdigest(),
            sha256(canonical_json_bytes(Message("user", PROMPT).to_dict())).hexdigest(),
        }
        assert naked_hashes.isdisjoint({text for body in clear_record_bodies for text in _text_scalars(body)})
