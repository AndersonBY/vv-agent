"""Validated records are immutable; only exact retained bytes bypass storage validation."""

import json
from collections import Counter
from copy import deepcopy
from hashlib import sha256

import pytest
from jsonschema import Draft202012Validator
from jsonschema.validators import extend
from support.s1_fixture_values import (
    clear_fixture_state_cache,
    content_ref,
    create_fixture_session,
    fixture_content_reader,
    read_fixture_state,
    wrap_fixture_store,
)

from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _records as core_records
from vv_agent.kernel import _sql as sql_ledger
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.kernel._record_schema import _CHECKS, _SCHEMAS, HANDLE
from vv_agent.kernel._schema import _compile_check
from vv_agent.kernel._sql import LedgerIntegrityError, _extend_chain_digest
from vv_agent.session import records
from vv_agent.session.kernel import read_state
from vv_agent.session.records import RecordError, digest
from vv_agent.session.store import Conflict, SequenceConflict

from .conftest import open_store
from .helpers import acquire_fixture_lease, append_fixture_records, record
from .test_recovery_matrix import runtime as recovery_runtime


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def turn_record():
    rt = recovery_runtime("unused.sqlite", [])
    rt.model_binding(rt.llm, rt.resolved)
    task = rt.compile("go", "t")
    definition = rt.definition(task)
    payload = record("turn_started").payload | {
        "definition": definition,
        "definition_digest": digest(definition),
        "budget": RunBudgetLimits().to_dict(),
    }
    turn = make_record_before_publication("turn_started", session_id="s", turn_id="t", payload=payload)
    object.__setattr__(turn, "_task_value", task)
    return turn


def rewrite_record(store, row, value, *, digest_override=None):
    body = canonical_json_bytes(value)
    record_digest = sha256(body).hexdigest()
    previous_chain = store._rows(
        "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (store.tenant_id, "s", row.seq - 1),
    )[0][0]
    chain_digest = _extend_chain_digest(previous_chain, row.seq, record_digest)
    commit_id, manifest_body = store._rows(
        "SELECT c.commit_id,c.body FROM sk_commit c JOIN sk_record r "
        "ON r.tenant_id=c.tenant_id AND r.session_id=c.session_id AND r.commit_id=c.commit_id "
        "WHERE r.tenant_id=%s AND r.session_id=%s AND r.seq=%s",
        (store.tenant_id, "s", row.seq),
    )[0]
    manifest = json.loads(manifest_body)
    for item in manifest["records"]:
        if item["record_id"] == row.record.record_id:
            item["digest"] = record_digest
    manifest_body = canonical_json_bytes(manifest)
    commit_digest = sha256(manifest_body).hexdigest()
    with store.atomic():
        store._rows(
            "UPDATE sk_commit SET body=%s,digest=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
            (manifest_body, commit_digest, store.tenant_id, "s", commit_id),
        )
        store._rows(
            "UPDATE sk_record SET body=%s,digest=%s,chain_digest=%s,commit_digest=%s "
            "WHERE tenant_id=%s AND session_id=%s AND seq=%s",
            (
                body,
                digest_override or record_digest,
                chain_digest,
                commit_digest,
                store.tenant_id,
                "s",
                row.seq,
            ),
        )


def test_record_bytes_and_digest_are_retained_without_payload_aliases(monkeypatch):
    payload = record("turn_started").to_dict()["payload"]
    rec = records.make_record("turn_started", session_id="s", turn_id="t", payload=payload)
    body, checksum = rec.encode(), rec.digest
    retained_definition = deepcopy(rec.payload["definition"])
    payload["definition"]["changed"] = True
    rec.payload["definition"]["changed"] = True
    rec.to_dict()["payload"]["definition"]["changed"] = True

    def unexpected(*args, **kwargs):
        raise AssertionError("a retained record must not encode or validate again")

    monkeypatch.setattr(core_records, "canonical_json_bytes", unexpected)
    monkeypatch.setattr(core_records, "validate", unexpected)
    assert rec.encode() is body and rec.digest is checksum
    assert deepcopy(rec) is rec
    assert rec.payload["definition"] == retained_definition


@pytest.mark.sqlite_file
@pytest.mark.parametrize("append_tail", [False, True])
def test_prefix_cache_reads_current_record_metadata(store, database, append_tail):
    create_fixture_session(store, "s")
    append_fixture_records(store, "s", (turn_record(),), commit_id="start")
    _, before, _ = store.read_state("s")
    original = before[1]
    created_ms = original.created_ms + 1

    with open_store(database) as other:
        with other.atomic():
            other._rows(
                "UPDATE sk_record SET created_ms=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
                (created_ms, other.tenant_id, "s", original.seq),
            )
        if append_tail:
            append_fixture_records(wrap_fixture_store(other), "s", (record("usage_observed"),), commit_id="tail")

    _, rows, _ = store.read_state("s")
    assert rows[1].created_ms == created_ms
    assert rows[1].record is original.record
    assert store.read("s", after_seq=1, limit=1).records == (rows[1],)


@pytest.mark.persistent_store
@pytest.mark.parametrize("reader", ["read", "read_state", "consumer_batch"])
@pytest.mark.parametrize("tamper", ["schema", "embedded_digest", "storage_digest"])
def test_external_record_bytes_require_full_validation(store, database, reader, tamper):
    store.configure_input_admission(allow_unsigned=True)
    create_fixture_session(store, "s", consumers=("events",))
    append_fixture_records(store, "s", (turn_record(),), commit_id="external")
    turn = store.read("s").records[-1]
    read_state(store, "s")  # Keep a warm prefix in the reader before the external append.
    with open_store(database) as other:
        other = wrap_fixture_store(other)
        value = turn.record.to_dict()
        if tamper == "schema":
            value["payload"]["unexpected"] = 1
        elif tamper == "embedded_digest":
            value["payload"]["definition_digest"] = "0" * 64
        rewrite_record(other, turn, value, digest_override="0" * 64 if tamper == "storage_digest" else None)
    with pytest.raises(Conflict if tamper == "storage_digest" else RecordError):
        if reader == "consumer_batch":
            with store.atomic() as tx:
                tx.consumer_batch("s", "events")
        else:
            getattr(store, reader)("s")


@pytest.mark.parametrize("reader", ["read", "read_state", "verify_ledger", "consumer_batch"])
def test_warm_prefix_skips_only_record_body_normalization(store, monkeypatch, reader):
    store.configure_input_admission(allow_unsigned=True)
    create_fixture_session(store, "s", consumers=("events",))
    append_fixture_records(store, "s", (turn_record(),), commit_id="warm")
    store.read("s")
    cached = store._root._cached_prefix("s")
    assert cached is not None
    assert all(row.record.encode() == cached.reducer.seen[row.record.record_id] for row in cached.records)

    contexts = []
    record_decodes = 0
    proof_calls = Counter()
    original_canonical = sql_ledger.canonical_json_bytes
    original_loads = sql_ledger.json.loads
    raw = store._root

    def tracked_canonical(value, name="JSON / JCS"):
        contexts.append(name)
        return original_canonical(value, name)

    def tracked_loads(value, *args, **kwargs):
        nonlocal record_decodes
        if isinstance(value, (bytes, bytearray, memoryview)):
            body = bytes(value)
            if all(field in body for field in (b'"record_id"', b'"schema_version"', b'"payload"')):
                record_decodes += 1
        return original_loads(value, *args, **kwargs)

    for name in ("_verify_checkpoints", "_verify_commits", "_verify_head_bounds"):
        original = getattr(raw, name)

        def tracked(*args, _original=original, _name=name, **kwargs):
            proof_calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(raw, name, tracked)
    monkeypatch.setattr(sql_ledger, "canonical_json_bytes", tracked_canonical)
    monkeypatch.setattr(sql_ledger.json, "loads", tracked_loads)

    if reader == "consumer_batch":
        with store.atomic() as tx:
            assert tx.consumer_batch("s", "events") is not None
    else:
        getattr(store, reader)("s")

    assert record_decodes == 0
    assert "stored record body" not in contexts
    assert proof_calls == Counter({"_verify_checkpoints": 1, "_verify_commits": 1, "_verify_head_bounds": 1})


@pytest.mark.persistent_store
def test_authority_pass_reuses_verified_record_metadata(store, monkeypatch):
    store.configure_input_admission(allow_unsigned=True)
    create_fixture_session(store, "s")
    append_fixture_records(store, "s", (turn_record(),), commit_id="single-record")

    raw = store._root
    direct_record_scans = []
    original_rows = raw._rows

    def counted_rows(query, params=()):
        normalized = " ".join(query.upper().split())
        if normalized.startswith("SELECT BODY,SEQ,") and " FROM SK_RECORD " in normalized:
            direct_record_scans.append(query)
        return original_rows(query, params)

    monkeypatch.setattr(raw, "_rows", counted_rows)
    store.read_state("s")

    assert len(direct_record_scans) == 1, direct_record_scans


def test_append_validates_current_locked_context_once(store, monkeypatch):
    create_fixture_session(store, "stable")
    create_fixture_session(store, "changed")
    raw = store._root
    stable_lease = acquire_fixture_lease(store, "stable", owner="stable-worker", ttl_ms=15000)
    changed_lease = acquire_fixture_lease(store, "changed", owner="changed-worker", ttl_ms=15000)
    assert stable_lease is not None and changed_lease is not None

    prefix_calls: Counter[str] = Counter()
    original_prefix = raw._prefix
    counting = True

    def counted_prefix(session_id, row, **kwargs):
        if counting:
            prefix_calls[session_id] += 1
        return original_prefix(session_id, row, **kwargs)

    monkeypatch.setattr(raw, "_prefix", counted_prefix)
    clear_fixture_state_cache(store)
    with store.atomic() as tx:
        tx.append(
            "stable",
            lease=stable_lease,
            expected_seq=1,
            commit_id="stable",
            records=(record("turn_started", sid="stable"),),
        )
    assert prefix_calls["stable"] == 1

    def commit_before_lock():
        nonlocal counting
        counting = False
        try:
            with raw.atomic() as other_tx:
                other_tx.append(
                    "changed",
                    lease=changed_lease,
                    expected_seq=1,
                    commit_id="concurrent",
                    records=(record("turn_started", sid="changed"),),
                )
        finally:
            counting = True
        return changed_lease

    clear_fixture_state_cache(store)
    with store.atomic() as tx:
        tx.bind_lease_guard(commit_before_lock)
        with pytest.raises(SequenceConflict):
            tx.append(
                "changed",
                lease=changed_lease,
                expected_seq=1,
                commit_id="stale",
                records=(record("turn_started", sid="changed"),),
            )
    assert prefix_calls["changed"] == 1


def test_record_body_reuse_is_bound_to_tenant_and_session(store):
    store.configure_input_admission(allow_unsigned=True)
    create_fixture_session(store, "s")
    store.read_state("s")

    raw = store._root
    assert raw._cached_prefix("s") is not None
    assert raw._cached_prefix("other-session") is None
    other_tenant = raw.for_tenant("other-tenant")
    assert other_tenant._cached_prefix("s") is None


def test_changed_noncanonical_body_does_not_use_cached_record(store):
    store.configure_input_admission(allow_unsigned=True)
    create_fixture_session(store, "s")
    append_fixture_records(store, "s", (turn_record(),), commit_id="canonical-body")
    assert store.read("s").head_seq == 2

    body = store._one("SELECT body FROM sk_record WHERE seq=2")[0]
    changed = b" " + body
    with store.atomic():
        store._rows("UPDATE sk_record SET body=%s,digest=%s WHERE seq=2", (changed, sha256(changed).hexdigest()))

    with pytest.raises(LedgerIntegrityError, match="stored record body is not canonical JCS at seq 2"):
        store.read("s")


@pytest.mark.parametrize("warm", [False, True])
def test_changed_own_record_bytes_do_not_hit_cache(store, warm):
    create_fixture_session(store, "s")
    append_fixture_records(store, "s", (turn_record(),), commit_id="start")
    turn = store.read("s").records[-1]
    if not warm:
        clear_fixture_state_cache(store)
    value = turn.record.to_dict()
    value["payload"]["unexpected"] = True
    rewrite_record(store, turn, value)
    with pytest.raises(RecordError):
        store.read("s")


def test_cold_read_validates_record_chain_before_record_schema(store):
    create_fixture_session(store, "s")
    append_fixture_records(store, "s", (turn_record(),), commit_id="start")
    turn = store.read("s").records[-1]
    value = turn.record.to_dict()
    value["payload"]["unexpected"] = True
    rewrite_record(store, turn, value)
    clear_fixture_state_cache(store)

    with store.atomic():
        store._rows("UPDATE sk_record SET chain_digest=%s WHERE seq=2", ("0" * 64,))
    with pytest.raises(LedgerIntegrityError, match="stored record chain digest mismatch at seq 2"):
        store.read("s")

    rewrite_record(store, turn, value)
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
    if kind == "number":
        return schema.get("exclusiveMinimum", schema.get("minimum", 0)) + 0.5
    if kind == "string":
        return "0" * 64 if "pattern" in schema else "text"
    return True if kind == "boolean" else None


StrictValidator = extend(
    Draft202012Validator, type_checker=Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda _, value: type(value) is int)
)


@pytest.mark.parametrize(
    "validator",
    [StrictValidator(schema) for schema in _SCHEMAS],
    ids=lambda v: str(list(v.schema.get("properties", v.schema))),
)
def test_compiled_checks_match_jsonschema(validator):
    value = example(validator.schema)
    check = _CHECKS[id(validator.schema)]
    for candidate in [value, *mutations(value)]:
        valid = validator.is_valid(candidate)
        assert check(candidate) == valid, candidate
        if not valid:
            with pytest.raises(RecordError, match=r"^invalid session schema$"):
                records.validate(candidate, validator.schema, canonical=False)


@pytest.mark.parametrize("variant", HANDLE["oneOf"])
def test_compiled_checks_match_all_nested_handles(variant):
    validator = StrictValidator(variant)
    check = _compile_check(variant)
    value = example(variant)
    for candidate in [value, *mutations(value)]:
        assert check(candidate) == validator.is_valid(candidate), candidate


def test_committed_records_remain_isolated_across_receipt_read_and_new_lease(store, monkeypatch):
    create_fixture_session(store, "s")
    turn = turn_record()
    receipt = append_fixture_records(store, "s", (turn,), commit_id="start")
    receipt.records[0].record.payload["definition"]["changed"] = True
    replacement = store.acquire("s", owner="second", ttl_ms=15000)
    assert replacement is not None and replacement.epoch > 1

    def unexpected(*args, **kwargs):
        raise AssertionError("identical retained bytes must not be decoded and validated again")

    monkeypatch.setattr(records.Record, "parse", unexpected)
    state, rows, _ = read_fixture_state(store, "s")
    assert state.turns["t"].start.payload["definition"] == turn.payload["definition"]
    assert rows[-1].record.encode() == receipt.records[0].record.encode()
    store.release(replacement)


@pytest.mark.parametrize("gap", ["missing", "nonconsecutive"])
def test_driver_tail_rejects_sequence_gap(store, database, gap):
    from vv_agent.session.kernel import _Driver, _Scope
    from vv_agent.session.store import ReadPage, StoredRecord

    create_fixture_session(store, "s")
    lease = acquire_fixture_lease(store, "s", owner="driver", ttl_ms=15000)
    assert lease is not None
    rt = recovery_runtime(database, [], content_reader=fixture_content_reader(store, "s"))
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


@pytest.mark.parametrize(
    "kind,field",
    [("turn_started", "definition"), ("op_planned", "request"), ("op_completed", "result"), ("op_prepared", "request")],
)
def test_record_encoding_reuses_validated_fields_with_identical_jcs(kind, field):
    for index in range(100):
        value = content_ref(f"jcs-{index}")
        payload = example(records.PAYLOADS[kind])
        payload[field], payload[f"{field}_digest"] = value, records.digest(value)
        if kind == "op_planned":
            payload["op_kind"], payload["purpose"] = "model", "primary"
        rec = records.make_record(
            kind,
            session_id="s",
            turn_id="t",
            operation_id="o" if kind.startswith("op_") else None,
            attempt=1 if kind.startswith("op_") else None,
            payload=payload,
        )
        expected = canonical_json_bytes(rec.to_dict())
        assert rec.encode() == expected
        parsed = records.Record.parse(expected)
        assert parsed.encode() == expected and parsed.digest == rec.digest


def test_record_composition_preserves_nested_invalid_json_rejection():
    for value in (float("inf"), float("nan"), 9007199254740992, "\ud800", {1: "invalid key"}):
        with pytest.raises((ValueError, RecordError)):
            payload = record("op_completed").payload | {"usage": {"invalid": value}}
            records.make_record("op_completed", session_id="s", turn_id="t", operation_id="o", attempt=1, payload=payload)
