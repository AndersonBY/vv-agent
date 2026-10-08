"""Validated records are immutable; only exact retained bytes bypass storage validation."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session import records
from vv_agent.session.kernel import read_state
from vv_agent.session.records import RecordError, SessionSpec
from vv_agent.session.store import Conflict

from .conftest import open_store
from .helpers import record


def test_record_bytes_and_digest_are_retained_without_payload_aliases(monkeypatch):
    payload = record("turn_started").to_dict()["payload"]
    rec = records.make_record("turn_started", session_id="s", turn_id="t", payload=payload)
    body, checksum = rec.encode(), rec.digest
    payload["definition"]["changed"] = True
    rec.payload["definition"]["changed"] = True
    rec.to_dict()["payload"]["definition"]["changed"] = True

    def unexpected(*args, **kwargs):
        raise AssertionError("a retained record must not encode or validate again")

    monkeypatch.setattr(records, "canonical_json_bytes", unexpected)
    monkeypatch.setattr(records, "validate", unexpected)
    assert rec.encode() is body and rec.digest is checksum
    assert deepcopy(rec) is rec
    assert rec.payload["definition"] == {}


@pytest.mark.persistent_store
@pytest.mark.parametrize("reader", ["read", "read_state", "consumer_batch"])
@pytest.mark.parametrize("tamper", ["schema", "embedded_digest", "storage_digest"])
def test_external_record_bytes_require_full_validation(store, database, reader, tamper):
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=("events",))
    lease = store.acquire("s", owner="writer", ttl_ms=15000)
    assert lease is not None
    read_state(store, "s")  # Keep a warm prefix in the reader before the external append.
    with open_store(database) as other:
        with other.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="external", records=(record("turn_started"),))
        value = record("turn_started").to_dict()
        if tamper == "schema":
            value["payload"]["unexpected"] = 1
        elif tamper == "embedded_digest":
            value["payload"]["definition_digest"] = "0" * 64
        body = canonical_json_bytes(value)
        checksum = "0" * 64 if tamper == "storage_digest" else sha256(body).hexdigest()
        with other.atomic():
            other._rows("UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=2", (body, checksum, "s"))
    with pytest.raises(Conflict if tamper == "storage_digest" else RecordError):
        if reader == "consumer_batch":
            with store.atomic() as tx:
                tx.consumer_batch("s", "events")
        else:
            getattr(store, reader)("s")


@pytest.mark.parametrize("warm", [False, True])
def test_changed_own_record_bytes_do_not_hit_cache(store, warm):
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="writer", ttl_ms=15000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="start", records=(record("turn_started"),))
    if not warm:
        store._fold_cache = None
    value = record("turn_started").to_dict()
    value["payload"]["unexpected"] = True
    body = json.dumps(value).encode()
    with store.atomic():
        store._rows("UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=2", (body, sha256(body).hexdigest(), "s"))
    with pytest.raises(RecordError):
        store.read("s")


# Every existing schema is checked with scalar, collection, missing/extra-field and
# nested mutations; the original jsonschema validator remains the reference.
def mutations(value):
    yield from (None, True, False, 0, 1, -1, 1.0, 2**60, "", "future", "0" * 64 + "\n", [], {}, (), float("nan"))
    if isinstance(value, dict):
        yield value | {"unexpected": None}
        for key, item in value.items():
            yield {k: v for k, v in value.items() if k != key}
            for replacement in mutations(item):
                yield value | {key: replacement}
    elif isinstance(value, list):
        yield tuple(value)
        yield [None]
        for i, item in enumerate(value):
            for replacement in mutations(item):
                yield [*value[:i], replacement, *value[i + 1 :]]


def example(schema):
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema or "oneOf" in schema:
        return example(schema.get("anyOf", schema.get("oneOf"))[0])
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {key: example(value) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [example(schema["items"])]
    if kind == "integer":
        return schema["minimum"]
    if kind == "string":
        return "0" * 64 if "pattern" in schema else "text"
    return True if kind == "boolean" else None


@pytest.mark.parametrize("validator", list(records._VALIDATORS.values()), ids=lambda v: str(list(v.schema["properties"])))
def test_compiled_checks_match_jsonschema(validator):
    value = example(validator.schema)
    check = records._CHECKS[id(validator.schema)]
    for candidate in [value, *mutations(value)]:
        assert check(candidate) == validator.is_valid(candidate), candidate


@pytest.mark.parametrize("variant", records.HANDLE["oneOf"])
def test_compiled_checks_match_all_nested_handles(variant):
    validator = records._StrictValidator(variant)
    check = records._compile_check(variant)
    value = example(variant)
    for candidate in [value, *mutations(value)]:
        assert check(candidate) == validator.is_valid(candidate), candidate


def test_committed_records_remain_isolated_across_receipt_read_and_new_lease(store, monkeypatch):
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="first", ttl_ms=15000)
    assert lease is not None
    with store.atomic() as tx:
        receipt = tx.append("s", lease=lease, expected_seq=1, commit_id="start", records=(record("turn_started"),))
    receipt.records[0].record.payload["definition"]["changed"] = True
    store.release(lease)
    replacement = store.acquire("s", owner="second", ttl_ms=15000)
    assert replacement is not None and replacement.epoch > lease.epoch

    def unexpected(*args, **kwargs):
        raise AssertionError("identical retained bytes must not be decoded and validated again")

    monkeypatch.setattr(records.Record, "parse", unexpected)
    state, rows, _ = read_state(store, "s")
    assert state.turns["t"].start.payload["definition"] == {}
    assert rows[-1].record is receipt.records[0].record


@pytest.mark.parametrize("gap", ["missing", "nonconsecutive"])
def test_driver_tail_rejects_sequence_gap(store, database, gap):
    from vv_agent.session.kernel import _Driver, _Scope
    from vv_agent.session.store import ReadPage, StoredRecord

    from .test_recovery_matrix import runtime

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="driver", ttl_ms=15000)
    assert lease is not None
    rt = runtime(database, [])
    driver = _Driver(store, "s", rt, _Scope(lease, rt))
    if gap == "missing":
        # An advertised head without a row must fail instead of spinning forever.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(store, "read", lambda *a, **kw: ReadPage(2, 0, ()))
            with pytest.raises(Conflict, match="sequence gap"):
                driver.refresh()
    else:
        with pytest.raises(Conflict, match="sequence gap"):
            driver.extend((StoredRecord(record("turn_started"), 3, "gap", lease.epoch, 0),))
