from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock, Thread
from typing import Any

import pytest
from test_checkpoint_contract import _redis_store

from vv_agent.checkpoint import (
    CheckpointConfig,
    CheckpointError,
    CheckpointExtension,
    EventCursor,
    OperationKind,
    OperationState,
    ResumeObservation,
    ResumePolicy,
    ToolIdempotency,
    canonical_json_bytes,
    compute_event_payload_digest,
    compute_operation_request_digest,
    compute_run_definition_digest,
    validate_checkpoint_extension,
)
from vv_agent.runtime.checkpoint_codec_v2 import (
    checkpoint_v2_from_dict,
    checkpoint_v2_from_json,
    checkpoint_v2_to_dict,
    checkpoint_v2_to_json,
    decode_checkpoint_dict,
    decode_checkpoint_json,
    migrate_terminal_checkpoint_v1,
    validate_extension_state_size,
)
from vv_agent.runtime.state import Checkpoint, CheckpointConflictError, InMemoryStateStore
from vv_agent.runtime.state_v2 import (
    CheckpointStoreV2,
    CheckpointV2,
    EventOutboxEntry,
    OperationJournalEntry,
)
from vv_agent.runtime.stores.redis import RedisStateStore
from vv_agent.runtime.stores.sqlite import SqliteStateStore
from vv_agent.types import AgentResult, AgentStatus, CompletionReason, Message

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _codec_case(name: str) -> dict[str, Any]:
    fixture = _fixture("checkpoint_codec_v2.json")
    return deepcopy(next(case["payload"] for case in fixture["valid_cases"] if case["name"] == name))


def _minimal_checkpoint(*, key: str = "fresh") -> CheckpointV2:
    payload = _codec_case("minimal_running")
    payload["checkpoint_key"] = key
    return checkpoint_v2_from_dict(payload)


def _store(store_kind: str, tmp_path: Path, name: str) -> Any:
    if store_kind == "memory":
        return InMemoryStateStore()
    if store_kind == "sqlite":
        return SqliteStateStore(tmp_path / f"{name}.sqlite3")
    return _redis_store()


def _journal_case(name: str) -> dict[str, Any]:
    fixture = _fixture("operation_journal_v1.json")
    return deepcopy(next(case["entry"] for case in fixture["valid_entries"] if case["name"] == name))


def test_rfc8785_vectors_match_canonical_bytes_and_digests() -> None:
    vectors = (
        ("run_definition_v1.json", ("golden_cases",), "definition"),
        ("operation_journal_v1.json", ("request_digest", "golden_cases"), "request"),
        ("checkpoint_codec_v2.json", ("extension_limits", "canonicalization_vectors"), "entry"),
        ("checkpoint_store_v2.json", ("event_payload_digest", "golden_cases"), "event"),
    )
    for fixture_name, path, value_field in vectors:
        value: Any = _fixture(fixture_name)
        for part in path:
            value = value[part]
        for vector in value:
            canonical = canonical_json_bytes(vector[value_field])
            assert base64.b64encode(canonical).decode("ascii") == vector["canonical_json_base64"]
            assert len(canonical) == vector["canonical_json_utf8_bytes"]
            assert hashlib.sha256(canonical).hexdigest() == vector["sha256"]


def test_run_definition_digest_matches_both_golden_vectors() -> None:
    fixture = _fixture("run_definition_v1.json")
    for case in fixture["golden_cases"]:
        assert compute_run_definition_digest(case["definition"]) == case["sha256"]


def test_checkpoint_v2_round_trip_preserves_unknown_fields_and_uses_jcs() -> None:
    fixture = _fixture("checkpoint_codec_v2.json")
    checkpoint = checkpoint_v2_from_dict(fixture["canonical_checkpoint"])
    encoded = checkpoint_v2_to_dict(checkpoint)

    assert encoded == fixture["canonical_checkpoint"]
    assert encoded["vendor_future"] == {"preserve": True}
    wire = checkpoint_v2_to_json(checkpoint)
    assert wire.encode("utf-8") == canonical_json_bytes(encoded)
    assert checkpoint_v2_from_json(wire) == checkpoint


def test_checkpoint_v2_validates_embedded_definition_schema_and_digest() -> None:
    payload = _codec_case("minimal_running")
    assert checkpoint_v2_from_dict(payload).run_definition_digest == compute_run_definition_digest(
        payload["run_definition"]
    )

    missing_schema = deepcopy(payload)
    missing_schema.pop("run_definition_schema")
    missing_schema.pop("run_definition")
    with pytest.raises(CheckpointError) as error:
        checkpoint_v2_from_dict(missing_schema)
    assert error.value.code == "checkpoint_definition_schema_unsupported"

    mismatch = deepcopy(payload)
    mismatch["run_definition"]["root_input"] = "different"
    with pytest.raises(CheckpointError) as error:
        checkpoint_v2_from_dict(mismatch)
    assert error.value.code == "checkpoint_definition_mismatch"

    unknown = deepcopy(payload)
    unknown["schema_version"] = "vv-agent.checkpoint.v3"
    with pytest.raises(CheckpointError) as error:
        checkpoint_v2_from_dict(unknown)
    assert error.value.code == "checkpoint_schema_unsupported"


def test_checkpoint_v2_invalid_fixture_cases_have_stable_codes() -> None:
    fixture = _fixture("checkpoint_codec_v2.json")
    expected_codes = {
        "unknown_schema": "checkpoint_schema_unsupported",
        "blank_checkpoint_key": "checkpoint_key_invalid",
        "bad_definition_digest": "checkpoint_definition_digest_invalid",
        "zero_resume_attempt": "checkpoint_resume_attempt_invalid",
        "partial_claim": "checkpoint_claim_invalid",
        "claimed_cycle_not_next": "checkpoint_claim_invalid",
        "terminal_with_claim": "checkpoint_status_invalid",
        "journal_cycle_not_active": "checkpoint_journal_cycle_invalid",
        "unknown_required_extension": "checkpoint_required_extension_unavailable",
        "invalid_extension_namespace": "checkpoint_extension_namespace_invalid",
    }
    for case in fixture["invalid_cases"]:
        registered = [] if case["name"] == "unknown_required_extension" else None
        with pytest.raises(CheckpointError) as error:
            checkpoint_v2_from_dict(
                case["payload"],
                registered_extensions=registered,
            )
        assert error.value.code == expected_codes[case["name"]]

    for case in fixture["status_cases"]:
        if "base_valid_case" not in case:
            continue
        payload = _codec_case(case["base_valid_case"])
        payload.update(case["mutation"]["replace"])
        with pytest.raises(CheckpointError) as error:
            checkpoint_v2_from_dict(payload)
        assert error.value.code == case["error_code"]


def test_checkpoint_v2_strict_dual_decoder_and_terminal_v1_migration() -> None:
    migration_cases = _fixture("checkpoint_codec_v2.json")["migration_cases"]
    absent = next(case for case in migration_cases if case["name"] == "absent_discriminator_reads_v1")
    decoded = decode_checkpoint_dict(absent["source"])
    assert isinstance(decoded, Checkpoint)
    assert isinstance(decode_checkpoint_json(json.dumps(absent["source"])), Checkpoint)

    definition = _fixture("run_definition_v1.json")["golden_cases"][0]["definition"]
    migrated = migrate_terminal_checkpoint_v1(
        decoded,
        checkpoint_key="migrated-terminal",
        root_run_id="run-migrated",
        trace_id="trace-migrated",
        run_definition=definition,
    )
    assert migrated.status is AgentStatus.COMPLETED
    assert migrated.terminal_result is not None
    assert migrated.terminal_result.checkpoint_key == "migrated-terminal"
    assert migrated.run_definition_digest == compute_run_definition_digest(definition)

    running = deepcopy(absent["source"])
    running["status"] = "running"
    running["terminal_result"] = None
    running_checkpoint = decode_checkpoint_dict(running)
    assert isinstance(running_checkpoint, Checkpoint)
    with pytest.raises(CheckpointError) as error:
        migrate_terminal_checkpoint_v1(
            running_checkpoint,
            checkpoint_key="running",
            root_run_id="run",
            trace_id="trace",
            run_definition=definition,
        )
    assert error.value.code == "checkpoint_migration_requires_reconciliation"

    old_v2 = next(
        case
        for case in migration_cases
        if case["name"] == "checkpoint_0_5_0_v2_requires_explicit_definition_migration"
    )
    with pytest.raises(CheckpointError) as error:
        decode_checkpoint_dict(old_v2["source_payload"])
    assert error.value.code == "checkpoint_definition_schema_unsupported"

    with pytest.raises(CheckpointError) as error:
        decode_checkpoint_dict({"schema_version": "vv-agent.checkpoint.v9"})
    assert error.value.code == "checkpoint_schema_unsupported"

    with pytest.raises(ValueError, match="invalid"):
        decode_checkpoint_json('{"task_id":"a","task_id":"b"}')


def test_active_v1_claim_cannot_migrate() -> None:
    case = next(
        case
        for case in _fixture("checkpoint_codec_v2.json")["migration_cases"]
        if case["name"] == "active_v1_claim_cannot_migrate"
    )
    source = decode_checkpoint_dict(case["source"])
    assert isinstance(source, Checkpoint)
    definition = _fixture("run_definition_v1.json")["golden_cases"][0]["definition"]
    with pytest.raises(CheckpointError) as error:
        migrate_terminal_checkpoint_v1(
            source,
            checkpoint_key="active",
            root_run_id="run",
            trace_id="trace",
            run_definition=definition,
        )
    assert error.value.code == "checkpoint_migration_active_claim"


def test_extension_size_counts_complete_entry_jcs_bytes() -> None:
    exact = {"version": "1", "required": False, "state": "x" * 65_493}
    over = {"version": "1", "required": False, "state": "x" * 65_494}
    assert len(canonical_json_bytes(exact)) == 65_536
    validate_extension_state_size(
        {"com.example.exact": exact},
        max_extension_state_bytes=262_144,
    )
    with pytest.raises(CheckpointError) as error:
        validate_extension_state_size(
            {"com.example.over": over},
            max_extension_state_bytes=262_144,
        )
    assert error.value.code == "checkpoint_extension_entry_too_large"

    entries: dict[str, dict[str, Any]] = {
        f"com.example.e{index}": {
            "version": "1",
            "required": False,
            "state": "x" * repetitions,
        }
        for index, repetitions in enumerate((65_493, 65_493, 65_493, 65_450, 0))
    }
    validate_extension_state_size(entries, max_extension_state_bytes=262_144)
    entries["com.example.e3"]["state"] += "x"
    with pytest.raises(CheckpointError) as error:
        validate_extension_state_size(entries, max_extension_state_bytes=262_144)
    assert error.value.code == "checkpoint_extension_state_too_large"


def test_extensions_are_validated_by_duck_typing() -> None:
    class DuckExtension:
        namespace = "org.example.future"
        version = "9"
        required = True

        def snapshot(self) -> Any:
            return {"opaque": True}

        def restore(self, state: Any) -> None:
            self.state = state

    extension = DuckExtension()
    assert isinstance(extension, CheckpointExtension)
    validate_checkpoint_extension(extension)
    payload = _codec_case("minimal_running")
    payload["extension_state"] = {
        extension.namespace: {
            "version": extension.version,
            "required": True,
            "state": {"opaque": True},
        }
    }
    assert checkpoint_v2_from_dict(payload, registered_extensions=[extension])
    with pytest.raises(CheckpointError) as error:
        checkpoint_v2_from_dict(payload, registered_extensions=[])
    assert error.value.code == "checkpoint_required_extension_unavailable"


def test_operation_and_event_digest_golden_vectors() -> None:
    journal_fixture = _fixture("operation_journal_v1.json")
    for case in journal_fixture["request_digest"]["golden_cases"]:
        assert compute_operation_request_digest(case["request"]) == case["sha256"]
    model_request = journal_fixture["request_digest"]["golden_cases"][0]["request"]
    model_entry = OperationJournalEntry.from_dict(_journal_case("model_planned"))
    model_entry.verify_request(model_request)
    changed_request = deepcopy(model_request)
    changed_request["request"]["messages"][0]["content"] = "different"
    with pytest.raises(CheckpointError) as error:
        model_entry.verify_request(changed_request)
    assert error.value.code == "checkpoint_journal_integrity_mismatch"

    store_fixture = _fixture("checkpoint_store_v2.json")
    event_case = store_fixture["event_payload_digest"]["golden_cases"][0]
    assert compute_event_payload_digest(event_case["event"]) == event_case["sha256"]
    pending = EventOutboxEntry.pending("evt-1", event_case["event"])
    assert pending.payload_digest == event_case["sha256"]
    pending.verify_payload()


def test_operation_journal_invalid_cases_have_stable_codes() -> None:
    fixture = _fixture("operation_journal_v1.json")
    for case in fixture["valid_entries"]:
        OperationJournalEntry.from_dict(case["entry"])
    for case in fixture["invalid_entries"]:
        with pytest.raises(CheckpointError) as error:
            OperationJournalEntry.from_dict(case["entry"])
        assert error.value.code == case["error_code"], case["name"]


def test_checkpoint_config_validates_store_key_capabilities_and_stable_codes() -> None:
    store = InMemoryStateStore()
    config = CheckpointConfig(
        store=store,
        key=None,
        capability_refs={"runtime_hook:0": {"id": "hook.audit", "version": "1"}},
    )
    assert config.resume_policy is ResumePolicy.NEW
    assert isinstance(store, CheckpointStoreV2)

    invalid_cases: tuple[tuple[Callable[[], CheckpointConfig], str], ...] = (
        (lambda: CheckpointConfig(store=store, key="x" * 513), "checkpoint_key_invalid"),
        (
            lambda: CheckpointConfig(
                store=store,
                key=None,
                resume_policy=ResumePolicy.REQUIRE_EXISTING,
            ),
            "checkpoint_key_required",
        ),
        (
            lambda: CheckpointConfig(
                store=store,
                store_ref={"id": "x", "version": "1"},
            ),
            "checkpoint_store_selection_invalid",
        ),
        (
            lambda: CheckpointConfig(
                store=store,
                capability_refs={"bad slot": {"id": "x", "version": "1"}},
            ),
            "checkpoint_capability_ref_invalid",
        ),
    )
    for factory, code in invalid_cases:
        with pytest.raises(CheckpointError) as error:
            factory()
        assert error.value.code == code


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claim_modes_update_resume_attempt_atomically(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "claim-mode")
    checkpoint = _minimal_checkpoint(key=f"claim-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)

    continued = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-a",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert continued is not None
    assert continued.resume_attempt == 1
    assert continued.revision == 1

    with pytest.raises(CheckpointConflictError):
        store.claim_checkpoint_v2(
            checkpoint.checkpoint_key,
            1,
            claim_token="owner-b",
            lease_expires_at_ms=300,
            now_ms=199,
            claim_mode="recovery",
        )
    rejected = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert rejected is not None
    assert rejected.resume_attempt == 1

    recovered = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-b",
        lease_expires_at_ms=300,
        now_ms=200,
        claim_mode="recovery",
    )
    assert recovered is not None
    assert recovered.resume_attempt == 2
    assert recovered.revision == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_concurrent_recovery_claims_increment_once(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "concurrent-recovery")
    checkpoint = _minimal_checkpoint(key=f"concurrent-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    assert store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="expired-owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    barrier = Barrier(3)
    result_lock = Lock()
    claims: list[CheckpointV2] = []
    conflicts: list[CheckpointConflictError] = []

    def recover(owner: str) -> None:
        barrier.wait()
        try:
            claimed = store.claim_checkpoint_v2(
                checkpoint.checkpoint_key,
                1,
                claim_token=owner,
                lease_expires_at_ms=300,
                now_ms=200,
                claim_mode="recovery",
            )
        except CheckpointConflictError as exc:
            with result_lock:
                conflicts.append(exc)
        else:
            assert claimed is not None
            with result_lock:
                claims.append(claimed)

    workers = [Thread(target=recover, args=(owner,)) for owner in ("owner-a", "owner-b")]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()

    assert len(claims) == 1
    assert len(conflicts) == 1
    persisted = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert persisted is not None
    assert persisted.revision == 2
    assert persisted.resume_attempt == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_progress_and_heartbeat_preserve_claim_and_journal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "progress")
    checkpoint = _minimal_checkpoint(key=f"progress-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.model_call_journal = [OperationJournalEntry.from_dict(_journal_case("model_started"))]
    claimed.shared_state["progress"] = "started"
    assert store.progress_checkpoint_v2(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    assert store.renew_checkpoint_claim_v2(
        checkpoint.checkpoint_key,
        claim_token="owner",
        lease_expires_at_ms=300,
        now_ms=150,
    )

    persisted = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert persisted is not None
    assert persisted.revision == 2
    assert persisted.claim_token == "owner"
    assert persisted.lease_expires_at_ms == 300
    assert persisted.model_call_journal[0].state is OperationState.STARTED
    assert persisted.shared_state["progress"] == "started"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_suspend_preserves_ambiguity_and_recovery_claims_it(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "suspend")
    checkpoint = _minimal_checkpoint(key=f"suspend-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    ambiguous = OperationJournalEntry.from_dict(_journal_case("model_ambiguous"))
    claimed.model_call_journal = [ambiguous]
    claimed.status = AgentStatus.RECONCILIATION_REQUIRED
    assert store.suspend_checkpoint_v2(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    suspended = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert suspended is not None
    assert suspended.status is AgentStatus.RECONCILIATION_REQUIRED
    assert suspended.revision == 2
    assert suspended.resume_attempt == 1
    assert suspended.claim_token is None
    assert suspended.model_call_journal[0].state is OperationState.AMBIGUOUS

    recovery = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="reconciler",
        lease_expires_at_ms=400,
        now_ms=300,
        claim_mode="recovery",
    )
    assert recovery is not None
    assert recovery.status is AgentStatus.RUNNING
    assert recovery.resume_attempt == 2
    assert recovery.model_call_journal[0].state is OperationState.AMBIGUOUS


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cycle_commit_finalize_and_acknowledgement_are_separate(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "terminal")
    checkpoint = _minimal_checkpoint(key=f"terminal-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.model_call_journal = [OperationJournalEntry.from_dict(_journal_case("model_succeeded"))]
    claimed.cycle_index = 1
    assert store.commit_checkpoint_v2(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    committed = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert committed is not None
    assert committed.claim_token is None
    assert committed.model_call_journal == []
    assert committed.revision == 2
    committed.status = AgentStatus.COMPLETED
    committed.terminal_result = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=committed.messages,
        cycles=committed.cycles,
        final_answer="done",
        completion_reason=CompletionReason.NO_TOOL_FINISH,
        checkpoint_key=committed.checkpoint_key,
    )
    assert store.finalize_checkpoint_v2(committed, expected_revision=committed.revision)

    terminal = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.revision == 3
    assert terminal.terminal_result is not None
    assert not store.finalize_checkpoint_v2(committed, expected_revision=terminal.revision)
    assert store.acknowledge_terminal_v2(
        checkpoint.checkpoint_key,
        expected_revision=terminal.revision,
    )
    retained = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.revision == 4
    assert retained.terminal_acknowledged
    assert retained.terminal_result is not None
    assert not store.acknowledge_terminal_v2(
        checkpoint.checkpoint_key,
        expected_revision=retained.revision,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_operator_abort_finalize_preserves_ambiguous_evidence(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "operator-abort")
    checkpoint = checkpoint_v2_from_dict(_codec_case("reconciliation_required_retains_ambiguous_journal"))
    checkpoint.checkpoint_key = f"abort-{store_kind}"
    checkpoint.resume_attempt = 1
    checkpoint.revision = 0
    assert store.create_checkpoint_v2(checkpoint)

    checkpoint.status = AgentStatus.FAILED
    checkpoint.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=checkpoint.messages,
        cycles=checkpoint.cycles,
        error="operator_abort_with_unknown_outcome",
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=checkpoint.checkpoint_key,
        resume_observation=ResumeObservation(
            operation_id=checkpoint.tool_journal[0].operation_id,
            operation_kind=OperationKind.TOOL,
            cycle_index=2,
            risk="unknown external tool outcome",
            idempotency_support=ToolIdempotency.UNKNOWN,
        ),
    )
    assert store.finalize_checkpoint_v2(checkpoint, expected_revision=0)
    terminal = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.status is AgentStatus.FAILED
    assert terminal.tool_journal[0].state is OperationState.AMBIGUOUS
    assert terminal.terminal_result is not None
    assert terminal.terminal_result.resume_observation is not None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_terminal_finalize_clears_claim_and_ordinary_journal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "claimed-terminal")
    checkpoint = _minimal_checkpoint(key=f"claimed-terminal-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-failure",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    failed_entry = _journal_case("model_failed")
    failed_entry["cycle_index"] = 1
    claimed.model_call_journal = [OperationJournalEntry.from_dict(failed_entry)]
    claimed.status = AgentStatus.FAILED
    claimed.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        error="definitive model rejection",
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=claimed.checkpoint_key,
    )

    assert store.finalize_claimed_checkpoint_v2(
        claimed,
        claim_token="owner-failure",
        expected_revision=claimed.revision,
    )
    terminal = store.load_checkpoint_v2(claimed.checkpoint_key)
    assert terminal is not None
    assert terminal.claim_token is None
    assert terminal.claimed_cycle is None
    assert terminal.lease_expires_at_ms is None
    assert terminal.model_call_journal == []
    assert terminal.terminal_result is not None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_operator_abort_preserves_ambiguous_journal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "claimed-abort")
    checkpoint = _minimal_checkpoint(key=f"claimed-abort-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-reconcile",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="recovery",
    )
    assert claimed is not None
    ambiguous_entry = _journal_case("tool_unknown_idempotency")
    ambiguous_entry["cycle_index"] = 1
    ambiguous_entry["state"] = "ambiguous"
    claimed.tool_journal = [OperationJournalEntry.from_dict(ambiguous_entry)]
    observation = ResumeObservation(
        operation_id=claimed.tool_journal[0].operation_id,
        operation_kind=OperationKind.TOOL,
        cycle_index=1,
        risk="unknown_tool_side_effect",
        idempotency_support=ToolIdempotency.UNKNOWN,
    )
    claimed.status = AgentStatus.FAILED
    claimed.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        error="operator_abort_with_unknown_outcome",
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=claimed.checkpoint_key,
        resume_observation=observation,
    )

    assert store.finalize_claimed_checkpoint_v2(
        claimed,
        claim_token="owner-reconcile",
        expected_revision=claimed.revision,
    )
    terminal = store.load_checkpoint_v2(claimed.checkpoint_key)
    assert terminal is not None
    assert terminal.claim_token is None
    assert len(terminal.tool_journal) == 1
    assert terminal.tool_journal[0].state is OperationState.AMBIGUOUS


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_event_delivery_cas_preserves_claim_and_terminal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "event-delivery")
    checkpoint = _minimal_checkpoint(key=f"event-delivery-{store_kind}")
    first = EventOutboxEntry.pending("evt-created", {"type": "checkpoint_created"})
    checkpoint.event_outbox = [first]
    assert store.create_checkpoint_v2(checkpoint)
    first_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 1},
        last_event_id="evt-created",
    )
    assert store.record_event_delivery_v2(
        checkpoint.checkpoint_key,
        event_id=first.event_id,
        payload_digest=first.payload_digest,
        cursor=first_cursor,
        expected_revision=0,
        claim_token=None,
    )

    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-events",
        lease_expires_at_ms=300,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    second = EventOutboxEntry.pending("evt-resumed", {"type": "checkpoint_resumed"})
    claimed.event_outbox.append(second)
    assert store.progress_checkpoint_v2(
        claimed,
        claim_token="owner-events",
        expected_revision=claimed.revision,
    )
    claimed.revision += 1
    second_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 2},
        last_event_id="evt-resumed",
    )
    assert store.record_event_delivery_v2(
        checkpoint.checkpoint_key,
        event_id=second.event_id,
        payload_digest=second.payload_digest,
        cursor=second_cursor,
        expected_revision=claimed.revision,
        claim_token="owner-events",
    )
    delivered = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert delivered is not None
    assert delivered.claim_token == "owner-events"
    assert delivered.event_cursor == second_cursor
    assert delivered.event_outbox[-1].state == "delivered"

    delivered.status = AgentStatus.FAILED
    delivered.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=delivered.messages,
        cycles=delivered.cycles,
        error="terminal",
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=delivered.checkpoint_key,
    )
    terminal_event = EventOutboxEntry.pending("evt-terminal", {"type": "run_failed"})
    delivered.event_outbox.append(terminal_event)
    assert store.finalize_claimed_checkpoint_v2(
        delivered,
        claim_token="owner-events",
        expected_revision=delivered.revision,
    )
    terminal = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert terminal is not None
    terminal_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 3},
        last_event_id="evt-terminal",
    )
    assert store.record_event_delivery_v2(
        checkpoint.checkpoint_key,
        event_id=terminal_event.event_id,
        payload_digest=terminal_event.payload_digest,
        cursor=terminal_cursor,
        expected_revision=terminal.revision,
        claim_token=None,
    )
    retained = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.terminal_result is not None
    assert retained.event_outbox[-1].state == "delivered"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_store_rejects_run_definition_replacement_during_progress(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "definition-immutable")
    checkpoint = _minimal_checkpoint(key=f"definition-{store_kind}")
    assert store.create_checkpoint_v2(checkpoint)
    claimed = store.claim_checkpoint_v2(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.run_definition["root_input"] = "replacement"
    claimed.run_definition_digest = compute_run_definition_digest(claimed.run_definition)
    assert not store.progress_checkpoint_v2(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    retained = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.run_definition["root_input"] == "Summarize the status."


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_store_preserves_unknown_top_level_fields(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "unknown")
    payload = _codec_case("unknown_top_level_is_preserved")
    payload["checkpoint_key"] = f"unknown-{store_kind}"
    checkpoint = checkpoint_v2_from_dict(payload)
    assert store.create_checkpoint_v2(checkpoint)
    loaded = store.load_checkpoint_v2(checkpoint.checkpoint_key)
    assert loaded is not None
    assert checkpoint_v2_to_dict(loaded)["vendor_future"] == {"preserve": True}


def test_sqlite_v2_schema_has_definition_columns_and_separate_namespace(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "schema.sqlite3")
    columns = {
        row[1] for row in store._conn.execute("PRAGMA table_info(checkpoints_v2)").fetchall()
    }
    assert {"run_definition_schema", "run_definition"} <= columns
    checkpoint = _minimal_checkpoint()
    assert store.create_checkpoint_v2(checkpoint)
    assert store.load_checkpoint(checkpoint.checkpoint_key) is None
    assert store.load_checkpoint_v2(checkpoint.checkpoint_key) is not None


def test_redis_v2_key_vectors_match_contract() -> None:
    fixture = _fixture("checkpoint_store_v2.json")
    for vector in fixture["redis_key_vectors"]:
        data_key = RedisStateStore.checkpoint_v2_key(vector["checkpoint_key"])
        assert data_key == vector["v2_data_key"]
        assert RedisStateStore._checkpoint_v2_keys(vector["checkpoint_key"]) == (
            vector["v2_data_key"],
            vector["v2_lease_key"],
        )


def test_cross_runtime_sqlite_v2_probe_from_environment() -> None:
    database = os.environ.get("VV_AGENT_CROSS_RUNTIME_V2_DB")
    if database is None:
        return
    mode = os.environ.get("VV_AGENT_CROSS_RUNTIME_V2_MODE", "read_rust")
    store = SqliteStateStore(database)
    if mode == "write_python":
        checkpoint = _minimal_checkpoint(key="python-wrote-v2")
        checkpoint.messages = [Message(role="user", content="from Python v2")]
        checkpoint.shared_state = {"writer": "python", "format": "checkpoint-v2"}
        assert store.create_checkpoint_v2(checkpoint)
        return
    if mode == "read_rust":
        checkpoint = store.load_checkpoint_v2("rust-wrote-v2")
        assert checkpoint is not None
        assert checkpoint.messages == [Message(role="user", content="from Rust v2")]
        assert checkpoint.shared_state == {"format": "checkpoint-v2", "writer": "rust"}
        assert checkpoint.run_definition_digest == compute_run_definition_digest(
            checkpoint.run_definition
        )
        return
    raise AssertionError(f"unknown cross-runtime v2 mode: {mode}")
