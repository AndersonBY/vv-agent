"""S1-A10 evidence for low-entropy plaintext and digest-domain separation."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from support.s1_fixture_values import append_fixture_records

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    Classification,
    ContentBindingMismatch,
    ContentLabels,
    ContentRef,
    SourceKind,
)
from vv_agent.kernel._content_publication import (
    link_declared_record_content,
    make_record_before_publication,
    publish_record_content,
    publish_turn_started,
    resolve_record_content_links,
)
from vv_agent.kernel._content_sql import _content_aad, _content_mac
from vv_agent.kernel._hydration import reference_digest
from vv_agent.kernel._records import make_record
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.export import export_session
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

SESSION = "s1"
TURN = "turn"
INPUT_ID = "input"
TEST_KEY = b"k" * 32

_ENCRYPTED_SQL_COLUMNS = {
    ("sk_content", "nonce"),
    ("sk_content", "ciphertext"),
    ("sk_session_key", "wrapped_key"),
}


def _text_scalars(value: Any) -> Iterable[str]:
    """Yield JSON string leaves, repairing BLOB JSON at the storage boundary."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            yield raw.decode("utf-8", "ignore")
            return
        yield from _text_scalars(decoded)
    elif isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _text_scalars(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _text_scalars(item)


def _clear_sql_values(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for (table,) in tables:
        escaped = table.replace('"', '""')
        columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
        rows = connection.execute(f'SELECT * FROM "{escaped}"').fetchall()
        for row in rows:
            for column in columns:
                column_name = column[1]
                if (table, column_name) in _ENCRYPTED_SQL_COLUMNS:
                    continue
                value = row[column[0]]
                values.extend((table, column_name, text) for text in _text_scalars(value))
    return values


def _assert_not_clear(label: str, values: Iterable[Any], forbidden: set[str]) -> None:
    leaves = {text for value in values for text in _text_scalars(value)}
    leaked = sorted(leaves & forbidden)
    assert not leaked, f"{label} exposed low-entropy plaintext: {leaked}"


def _runtime_path_names(driver: SessionDriver) -> list[str]:
    roots = [
        root
        for root in (
            driver._temporary_store_directory.name if driver._temporary_store_directory is not None else None,
            driver._temporary_content_key_state.name if driver._temporary_content_key_state is not None else None,
        )
        if root is not None
    ]
    return [path.name for root in roots for path in Path(root).rglob("*")]


def _started_record(driver: SessionDriver) -> Any:
    return next(row.record for row in driver.store.read(SESSION).records if row.record.kind == "turn_started")


def test_low_entropy_plaintext_is_absent_from_clear_sql_bytes_export_logs_and_runtime_names(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    single = "x"
    prompt = "hi"
    route_host = "h"
    artifact_path = "/a"
    artifact_name = Path(artifact_path).name
    forbidden = {single, prompt, route_host, artifact_path, artifact_name}

    artifact = tmp_path / "opaque-artifact.bin"
    artifact.write_bytes(b"artifact owner content")

    driver = SessionDriver()
    try:
        attributes = {
            "single": single,
            "prompt": prompt,
            "route": {"host": route_host},
            "artifact_path": artifact_path,
        }
        driver.create(SESSION, artifact_path, attributes)
        cast(SQLiteStore, driver.store).configure_input_admission(allow_unsigned=True)
        driver.push(SESSION, InboxItem(INPUT_ID, "user", {"content": prompt}))

        definition = {
            "single": single,
            "task": {"user_prompt": prompt},
            "route": {"host": route_host},
            "artifact_path": artifact_path,
        }
        budget = {"policy_value": "p"}
        not_published = make_record_before_publication(
            "turn_started",
            session_id=SESSION,
            turn_id=TURN,
            payload={
                "input_ids": [],
                "definition": definition,
                "budget": budget,
                "handler_version": "1",
                "binding": None,
                "generation": 0,
                "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
            },
        )
        append_fixture_records(driver.store, SESSION, (not_published,), commit_id="low-entropy-turn")

        started = _started_record(driver)
        definition_ref = ContentRef.from_dict(started.payload["definition"])
        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id=SESSION)
            assert writer.get_json(definition_ref) == definition

        connection = cast(SQLiteStore, driver._root_store).connection
        clear_sql = _clear_sql_values(connection)
        record_bodies = [
            row[0]
            for row in connection.execute(
                "SELECT body FROM sk_record WHERE tenant_id=? AND session_id=? ORDER BY seq", (driver.tenant_id, SESSION)
            )
        ]
        inbox_bodies = [
            row[0]
            for row in connection.execute(
                "SELECT body FROM sk_inbox WHERE tenant_id=? AND session_id=? ORDER BY input_seq", (driver.tenant_id, SESSION)
            )
        ]
        skeleton = export_session(driver.store, SESSION).to_dict()

        _assert_not_clear("clear SQL projection", (value for _table, _column, value in clear_sql), forbidden)
        _assert_not_clear("record bytes", record_bodies, forbidden)
        _assert_not_clear("inbox bytes", inbox_bodies, forbidden)
        _assert_not_clear("skeleton export", skeleton, forbidden)

        with pytest.raises(ContentBindingMismatch) as mismatch:
            export_session(driver.store, SESSION, expected_head=(999, "0" * 64))
        _assert_not_clear("exception text", (str(mismatch.value), repr(mismatch.value)), forbidden)
        for needle in ('"x"', '"hi"', '"h"', '"/a"', '"a"'):
            assert needle not in caplog.text

        runtime_names = _runtime_path_names(driver)
        assert forbidden.isdisjoint(runtime_names)
        assert definition_ref.content_id not in {artifact_name, f"content_{artifact_name}"}

        payload = started.payload
        assert payload["definition_digest"] == reference_digest(definition_ref)
        assert set(definition_ref.to_dict()) == {"content_id", "mac", "size", "key_id", "labels"}
        skeleton_lines = skeleton["lines"]
        skeleton_started = next(
            line["record"]["payload"]
            for line in skeleton_lines
            if line["line"] == "record" and line["record"]["kind"] == "turn_started"
        )
        assert skeleton_started["definition_digest"] == payload["definition_digest"]
        assert skeleton_started["definition"]["content_id"] == definition_ref.content_id
        assert skeleton_started["definition"]["key_id"] == definition_ref.key_id
        assert skeleton_started["definition"]["mac"] == definition_ref.mac
        assert skeleton_started["definition"]["size"] == definition_ref.size
        assert skeleton_started["definition"]["labels"] == definition_ref.labels.to_dict()

        plaintext_hashes = {
            sha256(canonical_json_bytes(definition)).hexdigest(),
            sha256(prompt.encode()).hexdigest(),
            sha256(artifact_path.encode()).hexdigest(),
        }
        assert payload["definition_digest"] not in plaintext_hashes
        assert definition_ref.mac not in plaintext_hashes
        assert started.digest not in plaintext_hashes
        assert definition_ref.content_id not in plaintext_hashes | {
            single,
            prompt,
            route_host,
            artifact_path,
            artifact_name,
        }
        assert definition_ref.mac != sha256(prompt.encode()).hexdigest()

        content_rows = connection.execute(
            "SELECT key_id,nonce FROM sk_content WHERE tenant_id=? AND session_id=? ORDER BY content_id",
            (driver.tenant_id, SESSION),
        ).fetchall()
        assert len({key_id for key_id, _nonce in content_rows}) == 1
        assert len({nonce for _key_id, nonce in content_rows}) == len(content_rows)
    finally:
        driver.close()


def test_reference_digest_reuses_the_contract_vector_and_separates_plaintext_domains() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures/parity/trusted_storage_s1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    vector = next(
        case for case in fixture["accepted_cases"] if case["name"] == "turn_started_requirements_policy_and_environment"
    )
    ref = ContentRef.from_dict(vector["wire"]["payload"]["definition"])

    expected = vector["wire"]["payload"]["definition_digest"]
    assert reference_digest(ref) == expected

    changed = ContentRef(
        content_id=ref.content_id,
        mac=ref.mac,
        size=ref.size + 1,
        key_id=ref.key_id,
        labels=ref.labels,
    )
    assert reference_digest(changed) != expected


def test_content_hmac_and_aad_have_fixed_jcs_framing_and_tenant_session_boundaries() -> None:
    labels = ContentLabels((SourceKind.USER,), Classification.PUBLIC)
    expected_aad = (
        b'{"content_id":"content/a","domain":"vv-agent/S1/content","framing_version":1,'
        b'"key_id":"key/a","labels":{"classification":"public","source":["user"]},'
        b'"mac":"1111111111111111111111111111111111111111111111111111111111111111",'
        b'"purpose":"content-ciphertext","session_id":"s","size":2,"tenant_id":"t"}'
    )
    aad = _content_aad(
        "t",
        "s",
        "content/a",
        "key/a",
        "1" * 64,
        2,
        labels,
    )
    assert aad == expected_aad
    assert b'"hi"' not in aad

    expected_mac = "dcaf4e557be17171f2f4bde30eb1b3d71eee8182bfb2d7f257b73c6d73b11dbe"
    assert _content_mac(TEST_KEY, "t", "s", b"hi") == expected_mac
    assert _content_mac(TEST_KEY, "t", "s", b"hi") != sha256(b"hi").hexdigest()
    assert _content_mac(TEST_KEY, "other", "s", b"hi") != expected_mac
    assert _content_mac(TEST_KEY, "t", "other", b"hi") != expected_mac
    assert _content_aad("other", "s", "content/a", "key/a", "1" * 64, 2, labels) != expected_aad
    assert _content_aad("t", "other", "content/a", "key/a", "1" * 64, 2, labels) != expected_aad


def _append_turn_with_operational_policy(
    driver: SessionDriver,
    *,
    policy_document: dict[str, Any],
    descriptor: dict[str, Any],
) -> tuple[ContentRef, str, str]:
    head = driver.store.read(SESSION, limit=1)
    lease = driver.store.acquire(SESSION, owner="a10-policy", ttl_ms=60_000)
    assert lease is not None
    try:
        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id=SESSION)
            turn_content = publish_turn_started(
                writer,
                {"task": {"user_prompt": "p"}, "route": {"host": "h"}},
                {},
            )
            descriptor_ref = writer.put_json(descriptor)
            operational_ref = writer.put_json(policy_document)
            network_ref_digest = reference_digest(operational_ref)
            started = make_record(
                "turn_started",
                session_id=SESSION,
                turn_id=TURN,
                payload={
                    "input_ids": [],
                    "definition": turn_content.definition.to_dict(),
                    "definition_digest": turn_content.definition_digest,
                    "handler_version": "1",
                    "budget": turn_content.budget.to_dict(),
                    "binding": None,
                    "generation": 0,
                    "policy_digest": None,
                    "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
                    "tokenizer_binding": None,
                    "environment": {
                        "descriptor": descriptor_ref.to_dict(),
                        "environment_capabilities": ["network.deny"],
                        "capabilities_digest": None,
                        "network_policy_ref_digest": network_ref_digest,
                        "isolation_policy_ref_digest": None,
                        "operational_policy": operational_ref.to_dict(),
                    },
                },
            )
            publication = publish_record_content(writer, (started,))
            receipt = tx.append(
                SESSION,
                lease=lease,
                expected_seq=head.head_seq,
                commit_id="a10-policy-turn",
                records=publication.records,
            )
            links = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, links)
            assert writer.get_json(operational_ref) == policy_document
    finally:
        driver.store.release(lease)
    return operational_ref, network_ref_digest, started.record_id


def test_private_operational_hash_is_only_in_encrypted_reference_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    network_policy = {"allow": [], "mode": "deny"}
    isolation_policy = {"mounts": [], "mode": "strict"}
    remote_network_digest = sha256(canonical_json_bytes(network_policy)).hexdigest()
    remote_isolation_digest = sha256(canonical_json_bytes(isolation_policy)).hexdigest()
    policy_document = {
        "network_policy_digest": remote_network_digest,
        "isolation_policy_digest": remote_isolation_digest,
    }

    driver = SessionDriver()
    try:
        driver.create(SESSION, "/workspace", {"route": {"host": "h"}})
        cast(SQLiteStore, driver.store).configure_input_admission(allow_unsigned=True)
        operational_ref, network_ref_digest, record_id = _append_turn_with_operational_policy(
            driver,
            policy_document=policy_document,
            descriptor={"route": {"host": "h"}},
        )
        assert network_ref_digest == reference_digest(operational_ref)
        assert remote_network_digest != network_ref_digest

        skeleton = export_session(driver.store, SESSION).to_dict()
        encoded = canonical_json_bytes(skeleton)
        connection = cast(SQLiteStore, driver._root_store).connection
        clear_sql = _clear_sql_values(connection)
        record_bodies = [
            row[0]
            for row in connection.execute(
                "SELECT body FROM sk_record WHERE tenant_id=? AND session_id=? ORDER BY seq",
                (driver.tenant_id, SESSION),
            )
        ]
        operational_forbidden = {remote_network_digest, remote_isolation_digest, "deny", "strict", "h", "p"}
        _assert_not_clear(
            "operational clear SQL",
            (value for _table, _column, value in clear_sql),
            operational_forbidden,
        )
        _assert_not_clear("operational record bytes", record_bodies, operational_forbidden)
        _assert_not_clear("operational skeleton export", skeleton, operational_forbidden)
        for needle in ('"deny"', '"strict"', '"h"', '"p"'):
            assert needle not in caplog.text
        assert remote_network_digest not in caplog.text
        assert remote_isolation_digest not in caplog.text
        assert remote_network_digest.encode() not in encoded
        assert remote_isolation_digest.encode() not in encoded
        assert b'"network_policy_digest"' not in encoded
        assert b'"isolation_policy_digest"' not in encoded
        assert b'"network.deny"' in encoded
        assert operational_ref.content_id.encode() in encoded
        assert operational_ref.mac.encode() in encoded
        assert network_ref_digest.encode() in encoded

        started = next(
            line["record"] for line in skeleton["lines"] if line["line"] == "record" and line["record"]["record_id"] == record_id
        )
        assert started["payload"]["environment"]["operational_policy"] == operational_ref.to_dict()
        assert started["payload"]["environment"]["network_policy_ref_digest"] == network_ref_digest
        assert started["payload"]["environment"]["environment_capabilities"] == ["network.deny"]

        with driver.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id=SESSION)
            assert writer.get_json(operational_ref) == policy_document
    finally:
        driver.close()


@pytest.mark.skip(reason="EW-A4 remote attach/start equality is a P3 provider boundary and is not implemented here")
def test_remote_agent_without_session_key_checks_operational_policy_equality() -> None:
    """Keep the unproven remote-provider path explicit instead of fabricating A10 coverage."""
