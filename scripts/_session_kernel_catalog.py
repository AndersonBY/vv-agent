"""Capture only referenced, authenticated S1 content from the test producer."""

from __future__ import annotations

import base64
import hmac
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from vv_agent.kernel._content import ContentRef, WrappedDataKey
from vv_agent.kernel._content_runtime import content_key_provider_for, durable_content_store
from vv_agent.session import content_store
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.sqlite import SQLiteStore

_EXCLUDED = {"session_content_catalog.json", "session_content_capture_s1.json", "trusted_storage_s1.json"}
_REF_FIELDS = {"content_id", "mac", "size", "key_id", "labels"}


def _unique_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate object member in fixture content")
    return result


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"non-finite JSON number in fixture content: {constant}")


def _references(value: Any) -> Iterator[ContentRef]:
    if isinstance(value, dict):
        if set(value) == _REF_FIELDS:
            yield ContentRef.from_dict(value)
        else:
            for child in value.values():
                yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def _curated_refs(output: Path) -> Iterator[ContentRef]:
    for path in sorted(output.glob("*.json*")):
        if path.name in _EXCLUDED:
            continue
        raw = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            yield from _references([json.loads(line) for line in raw.splitlines()])
        elif path.suffix == ".json":
            value = json.loads(raw)
            if not isinstance(value, dict) or value.get("historical_contract_version") != "39.0.0":
                yield from _references(value)


def _fixture_capture_keys(store: SQLiteStore, provider: LocalKeyProvider, fixture_values: ModuleType) -> dict[str, bytes]:
    secrets: Any = content_store.secrets
    if (
        Path(fixture_values.__file__ or "").resolve() != Path(__file__).with_name("s1_fixture_values.py").resolve()
        or type(secrets) is not fixture_values._FixtureSecrets
        or type(store) is not SQLiteStore
        or type(provider) is not LocalKeyProvider
        or provider is not content_key_provider_for(store)
        or provider._master_key != fixture_values._FIXTURE_KEY
        or provider._state_path != Path(f"{store.path}.content-key-state.sqlite3")
    ):
        raise ValueError("content capture requires the active deterministic fixture store")

    # LocalKeyProvider generates three fixture tokens in sequence: ID, nonce, data key.
    replay = fixture_values._FixtureSecrets()
    issued = [replay.token_bytes(32) for _ in range(secrets._counter)]
    return {
        "key_" + base64.urlsafe_b64encode(key_id).decode("ascii").rstrip("="): data_key
        for key_id, data_key in zip(issued[:-2], issued[2:], strict=True)
    }


def capture_content_catalog(
    store: SQLiteStore,
    provider: LocalKeyProvider,
    output: Path,
    base_catalog: dict[str, Any],
    *,
    independent_bytes: Callable[[list[Any]], list[bytes]],
    fixture_values: ModuleType,
) -> dict[str, Any]:
    """Resolve curated refs and their nested refs against the live fixture DB."""
    expected_fixture_keys = _fixture_capture_keys(store, provider, fixture_values)
    base_contents = base_catalog["contents"]
    references: dict[str, ContentRef] = {}

    def add(refs: Iterator[ContentRef]) -> set[str]:
        fresh = set()
        for ref in refs:
            existing = references.get(ref.content_id)
            if existing is not None:
                if existing != ref:
                    raise ValueError(f"conflicting curated content reference: {ref.content_id}")
            else:
                references[ref.content_id] = ref
                fresh.add(ref.content_id)
        return fresh

    links: dict[str, tuple[str, str, int | None, str | None, str]] = {}
    keys: dict[str, tuple[str, str, bytes]] = {}
    contents: dict[str, Any] = {}
    with store.atomic():
        rows = store._rows(
            "SELECT content_id,tenant_id,session_id,record_seq,input_id,slot FROM sk_content_link "
            "ORDER BY content_id,tenant_id,session_id,record_seq,input_id,slot"
        )
        for content_id, tenant, session, record_seq, input_id, slot in rows:
            existing = links.setdefault(content_id, (tenant, session, record_seq, input_id, slot))
            if existing[:2] != (tenant, session):
                raise ValueError(f"ambiguous fixture content scope: {content_id}")

        pending = add(_curated_refs(output))
        while pending:
            captures = []
            next_pending = set()
            for content_id in sorted(pending):
                ref = references[content_id]
                if content_id in base_contents:
                    legacy = base_contents[content_id]
                    if ref.key_id != legacy["key_id"] or content_id in links:
                        raise ValueError(f"conflicting synthetic content identity: {content_id}")
                    next_pending.update(add(_references(legacy["value"])))
                    continue
                if content_id not in links:
                    raise ValueError(f"curated content is not linked in the fixture store: {content_id}")
                tenant_id, session_id, record_seq, input_id, slot = links[content_id]
                if ref.key_id not in keys:
                    row = store._one(
                        "SELECT provider_id,wrapped_key FROM sk_session_key "
                        "WHERE tenant_id=%s AND session_id=%s AND key_id=%s AND state='active'",
                        (tenant_id, session_id, ref.key_id),
                    )
                    if row is None or row[0] != provider.provider_id or ref.key_id not in expected_fixture_keys:
                        raise ValueError(f"unexpected fixture key binding: {ref.key_id}")
                    key = provider.unwrap(tenant_id, session_id, WrappedDataKey(tenant_id, session_id, ref.key_id, bytes(row[1])))
                    if not hmac.compare_digest(key, expected_fixture_keys[ref.key_id]):
                        raise ValueError(f"key was not generated by the deterministic fixture context: {ref.key_id}")
                    keys[ref.key_id] = (tenant_id, session_id, key)
                elif keys[ref.key_id][:2] != (tenant_id, session_id):
                    raise ValueError(f"ambiguous fixture key scope: {ref.key_id}")
                durable = durable_content_store(store.for_tenant(tenant_id))
                durable.assert_link(tenant_id, session_id, ref, slot, record_seq=record_seq, input_id=input_id)
                plaintext = durable.get(tenant_id, session_id, ref)
                framing = {
                    "domain": "vv-agent/S1/content-mac",
                    "framing_version": 1,
                    "purpose": "content-ref",
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                }
                value = json.loads(
                    plaintext.decode("utf-8"), object_pairs_hook=_unique_json_members, parse_constant=_reject_json_constant
                )
                captures.append((ref, plaintext, value, framing))
                next_pending.update(add(_references(value)))

            encoded = independent_bytes([value for _, _, value, _ in captures] + [framing for _, _, _, framing in captures])
            canonical_values, canonical_frames = encoded[: len(captures)], encoded[len(captures) :]
            for (ref, raw, value, framing), canonical, header in zip(captures, canonical_values, canonical_frames, strict=True):
                if canonical != raw or len(raw) != ref.size:
                    raise ValueError(f"fixture content bytes are not canonical: {ref.content_id}")
                expected = hmac.digest(keys[ref.key_id][2], header + b"\0" + raw, "sha256").hex()
                if not hmac.compare_digest(expected, ref.mac):
                    raise ValueError(f"fixture content MAC does not match: {ref.content_id}")
                contents[ref.content_id] = {
                    "tenant_id": framing["tenant_id"],
                    "session_id": framing["session_id"],
                    "key_id": ref.key_id,
                    "value": value,
                    "bytes_base64": base64.b64encode(raw).decode("ascii"),
                }
            pending = next_pending

    return {
        "schema_version": "vv-agent.session-content-capture.v1",
        "status": "deterministic-test-fixture",
        "keys": [
            {"key_id": key_id, "key_base64": base64.b64encode(key).decode("ascii")}
            for key_id, (_, _, key) in sorted(keys.items())
        ],
        "contents": dict(sorted(contents.items())),
    }
