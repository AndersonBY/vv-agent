from hashlib import sha256

import pytest
from session.helpers import content_ref, item, principal, record
from session.test_recovery_matrix import runtime

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._record_schema import PAYLOADS
from vv_agent.kernel._sql import UnsupportedFeatureLevel, _commit_manifest, _extend_chain_digest
from vv_agent.kernel._store import Conflict
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, Record, SessionSpec, digest, make_record
from vv_agent.session.sqlite import SQLiteStore


def _create(store, session_id: str, *, parent=None, consumers=()):
    parent_session_id, parent_operation_id = parent if parent is not None else (None, None)
    spec = SessionSpec(
        session_id,
        principal(),
        content_ref(f"workspace/{session_id}"),
        parent_session_id=parent_session_id,
        parent_operation_id=parent_operation_id,
        attributes=content_ref(f"attributes/{session_id}"),
    )
    with store.atomic() as tx:
        return tx.create(spec, consumers=consumers)


def _acquire(store, session_id: str = "s"):
    lease = store.acquire(session_id, owner=f"worker/{session_id}", ttl_ms=10_000)
    assert lease is not None
    return lease


def _append(store, lease, records, commit_id: str):
    with store.atomic() as tx:
        return tx.append(
            lease.session_id,
            lease=lease,
            expected_seq=store.read(lease.session_id).head_seq,
            commit_id=commit_id,
            records=tuple(records),
        )


def _level_two_turn(session_id: str):
    return record(
        "turn_started",
        sid=session_id,
        requirements={"level": 2, "handler_version": "1", "capabilities": []},
    )


def _current_kind_record(kind: str) -> Record:
    request = content_ref("request")
    if kind == "op_prepared":
        return make_record(
            kind,
            session_id="s",
            turn_id="t",
            operation_id="o",
            attempt=1,
            payload={
                "request": request,
                "request_digest": digest(request),
                "op_kind": "tool",
                "tool": content_ref("tool"),
                "provider_binding": None,
                "idempotency_key": None,
                "hook_result": None,
                "shared_state": content_ref("shared-state"),
            },
        )
    if kind == "turn_parked":
        return make_record(
            kind,
            session_id="s",
            turn_id="t",
            payload={
                "interaction_id": "interaction",
                "question": content_ref("question"),
                "source_operation_id": "o",
                "source_attempt": 1,
            },
        )
    if kind == "boundary_recorded":
        return make_record(
            kind,
            session_id="s",
            turn_id="t",
            payload={
                "boundary_id": "boundary",
                "stage": "memory_started",
                "source_operation_id": None,
                "source_digest": None,
                "data": {"event": content_ref("event")},
            },
        )
    return record(kind)


@pytest.mark.parametrize("kind", tuple(PAYLOADS))
def test_current_record_kinds_round_trip_through_producer_and_codec(kind):
    encoded = _current_kind_record(kind).encode()
    assert Record.parse(encoded).kind == kind


def test_future_kind_sql_identifier_is_skipped_then_rejected_by_current_codec(tmp_path, monkeypatch):
    path = tmp_path / "future-kind.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        _create(store, "s")
        previous_chain = store._rows(
            "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id='s' AND seq=1",
            (store.tenant_id,),
        )[0][0]
        body = canonical_json_bytes(
            {
                "schema_version": "S1",
                "feature_level": 2,
                "record_id": "future-record",
                "kind": "future_record_kind",
                "session_id": "s",
                "turn_id": None,
                "operation_id": None,
                "attempt": None,
                "payload": {},
            }
        )
        record_digest = sha256(body).hexdigest()
        chain_digest = _extend_chain_digest(previous_chain, 2, record_digest)
        manifest = _commit_manifest(
            head_seq=1,
            head_chain_digest=previous_chain,
            records=(("future-record", record_digest),),
            consume_input_ids=(),
            consumers=(),
        )
        commit_digest = sha256(manifest).hexdigest()
        store._rows(
            "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,kind,body,digest,chain_digest,"
            "schema_version,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
            "VALUES (%s,'s',2,'future-record','future_record_kind',%s,%s,%s,'S1','future-commit',%s,0,0,0)",
            (store.tenant_id, body, record_digest, chain_digest, commit_digest),
        )
        store._rows(
            "INSERT INTO sk_commit(tenant_id,session_id,commit_id,body,digest,record_sequences,head_seq,created_ms) "
            "VALUES (%s,'s','future-commit',%s,%s,%s,2,0)",
            (store.tenant_id, manifest, commit_digest, canonical_json_bytes([["future-record", 2]])),
        )
        store._rows("UPDATE sk_session SET head_seq=2,level=2 WHERE tenant_id=%s AND session_id='s'", (store.tenant_id,))

        decode_calls = []
        decode = store._stored

        def observe_decode(*args, **kwargs):
            decode_calls.append(1)
            return decode(*args, **kwargs)

        monkeypatch.setattr(store, "_stored", observe_decode)
        store.configure_worker(level=1)
        with pytest.raises(UnsupportedFeatureLevel, match="feature level 2 exceeds worker feature level 1"):
            store.read("s")
        assert decode_calls == []

        store.configure_worker(level=2)
        with pytest.raises(RecordError):
            store.read("s")
        assert decode_calls


def test_unknown_kinds_fail_sql_producers_before_writes(tmp_path):
    path = tmp_path / "future-kind-producer.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        _create(store, "s")
        lease = _acquire(store)
        unknown_record = Record("future-record", "future_record_kind", "s", None, None, None, {})
        with pytest.raises(RecordError):
            _append(store, lease, (unknown_record,), "future-record")

        unknown_input = InboxItem("future-input", "future_input_kind", {}, actor=principal())
        with pytest.raises(RecordError), store.atomic() as tx:
            tx.push("s", unknown_input)

        assert store._rows("SELECT count(*) FROM sk_record")[0][0] == 1
        assert store._rows("SELECT count(*) FROM sk_commit")[0][0] == 1
        assert store._rows("SELECT count(*) FROM sk_inbox")[0][0] == 0
        assert store.release(lease)


def test_unknown_kind_direct_sql_is_rejected_by_public_hydration(tmp_path):
    path = tmp_path / "future-input-kind.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        _create(store, "s")
        value = item("future-input").to_dict()
        value["kind"] = "future_input_kind"
        body = canonical_json_bytes(value)
        store._rows(
            "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,available_ms,received_ms) "
            "VALUES (%s,'s','future-input',1,'future_input_kind',%s,%s,0,0)",
            (store.tenant_id, body, sha256(body).hexdigest()),
        )
        store._rows("UPDATE sk_session SET inbox_seq=1 WHERE tenant_id=%s AND session_id='s'", (store.tenant_id,))

        with pytest.raises(RecordError):
            store.peek_inbox("s")


def test_file_heartbeat_and_tenant_facades_share_runtime_configuration(tmp_path):
    path = tmp_path / "runtime-config.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_worker(level=2)
        store.configure_input_admission(allow_unsigned=True)

        tenant = store.for_tenant("tenant-a")
        assert tenant.worker_level == 2
        tenant.configure_worker(level=3)
        assert store.worker_level == 3

        with store.heartbeat_store() as heartbeat:
            assert heartbeat.connection is not store.connection
            assert heartbeat.worker_level == 3
            assert heartbeat._input_admission_policy() == (True, True, None)

        with tenant.heartbeat_store() as heartbeat:
            assert heartbeat.tenant_id == "tenant-a"
            assert heartbeat.worker_level == 3
            assert heartbeat._input_admission_policy() == (True, True, None)


def test_mid_drive_upgrade_gates_decode_paths_and_remains_releasable(tmp_path, monkeypatch):
    path = tmp_path / "read-fences.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_worker(level=1)
        _create(store, "s", consumers=("host",))
        with store.atomic() as tx:
            tx.push("s", item("pending"))
        lease = _acquire(store)
        store._rows(
            "UPDATE sk_session SET level=2 WHERE tenant_id=%s AND session_id='s'",
            (store.tenant_id,),
        )

        def decode_must_not_run(*_args):
            raise AssertionError("decode ran before the worker-level fence")

        monkeypatch.setattr(store, "_stored", decode_must_not_run)
        monkeypatch.setattr(store, "_stored_input", decode_must_not_run)
        mismatch = "session feature level 2 exceeds worker feature level 1"
        with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
            store.read("s")
        with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
            store.read_state("s")
        with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
            store.peek_inbox("s")
        with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
            store.read_input("s", "pending")
        with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
            store.control_records("s", through_seq=1)
        with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
            store.projection_context("s", ())
        with store.atomic() as tx:
            assert tx.consumer_batch("s", "host") is None

        assert store.release(lease)
        store.configure_worker(level=2)
        high = store.acquire("s", owner="upgraded", ttl_ms=10_000)
        assert high is not None
        store.release(high)


def test_drive_surfaces_mid_claim_upgrade_as_capability_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "drive-fence.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        _create(store, "s")
        store.configure_worker(level=1)
        acquire = store.acquire

        def acquire_then_promote(*args, **kwargs):
            lease = acquire(*args, **kwargs)
            if lease is not None:
                store._rows(
                    "UPDATE sk_session SET level=2 WHERE tenant_id=%s AND session_id='s'",
                    (store.tenant_id,),
                )
            return lease

        monkeypatch.setattr(store, "acquire", acquire_then_promote)
        with pytest.raises(UnsupportedFeatureLevel, match="session feature level 2 exceeds worker feature level 1"):
            drive(store, "s", runtime=runtime(store, [], heartbeat_seconds=0.01))
        assert store._rows("SELECT lease_owner FROM sk_session WHERE session_id='s'")[0][0] is None


@pytest.mark.parametrize("purge_child", [False, True])
def test_historical_child_result_level_does_not_depend_on_live_child(tmp_path, purge_child):
    path = tmp_path / f"historical-child-{purge_child}.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store._rows("UPDATE sk_settings SET write_level=2")
        _create(store, "parent", consumers=("host",))
        _create(store, "child", parent=("parent", "parent/op"))
        store.configure_worker(level=2)

        child_lease = _acquire(store, "child")
        high = _level_two_turn("child")
        _append(store, child_lease, (high,), "child-promote")
        store.release(child_lease)

        child_result = InboxItem(
            "child-result",
            "child_result",
            {
                "session_id": "child",
                "turn_id": "t",
                "operation_id": "parent/op",
                "attempt": 1,
                "result": None,
                "status": "completed",
                "terminal_seq": 2,
                "terminal_digest": high.digest,
            },
            actor=principal("child-delivery", "service"),
        )
        with store.atomic() as tx:
            tx.push("parent", child_result)

        applied = record(
            "input_applied",
            sid="parent",
            input=child_result.to_dict(),
            input_digest=digest(child_result.to_dict()),
            disposition="queued",
        )
        parent_lease = _acquire(store, "parent")
        with store.atomic() as tx:
            tx.append(
                "parent",
                lease=parent_lease,
                expected_seq=1,
                commit_id="child-result-applied",
                records=(applied,),
                consume_input_ids=(child_result.input_id,),
            )
        store.release(parent_lease)
        store._fold_cache = None
        store._previous_prefix = None

        if purge_child:
            for table in (
                "sk_content_link",
                "sk_content",
                "sk_inbox",
                "sk_checkpoint",
                "sk_consumer",
                "sk_commit",
                "sk_record",
                "sk_session_key",
                "sk_session",
            ):
                store._rows(
                    f"DELETE FROM {table} WHERE tenant_id=%s AND session_id='child'",
                    (store.tenant_id,),
                )
        else:
            store._rows(
                "UPDATE sk_session SET deletion_state='forgotten' WHERE tenant_id=%s AND session_id='child'",
                (store.tenant_id,),
            )

        _, records, _ = store.read_state("parent")
        assert any(r.record.kind == "input_applied" for r in records)

        missing = InboxItem(
            "missing-child-result",
            "child_result",
            child_result.payload | {"session_id": "missing"},
            actor=principal("child-delivery", "service"),
        )
        with pytest.raises(Conflict, match=r"child result|missing child"), store.atomic() as tx:
            tx.push("parent", missing)
