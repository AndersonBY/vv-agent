from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock, Thread
from typing import Any

import pytest

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
from vv_agent.events import (
    CheckpointCreatedEvent,
    CheckpointResumedEvent,
    ModelCallCompletedEvent,
    ModelCallFailedEvent,
    ModelCallStartedEvent,
    RunFailedEvent,
)
from vv_agent.runtime.checkpoint_codec import (
    _strict_json_loads,
    checkpoint_from_dict,
    checkpoint_from_json,
    checkpoint_to_dict,
    checkpoint_to_json,
    validate_extension_state_size,
)
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    CheckpointStore,
    EventOutboxEntry,
    OperationJournalEntry,
)
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.redis import RedisCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    CompletionReason,
    Message,
    ModelCallRecord,
    ModelCallStatus,
    TokenUsage,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"


class _FakeWatchError(Exception):
    pass


class _FakeRedisPipeline:
    def __init__(self, client: _FakeRedisClient) -> None:
        self._client = client
        self._commands: list[tuple[str, str, str | None]] = []
        self._transaction = False

    def __enter__(self) -> _FakeRedisPipeline:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def watch(self, *_keys: str) -> None:
        return None

    def unwatch(self) -> None:
        self._transaction = False
        self._commands.clear()

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def multi(self) -> None:
        self._transaction = True

    def set(self, key: str, value: str) -> None:
        if self._transaction:
            self._commands.append(("set", key, value))
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        if self._transaction:
            self._commands.append(("delete", key, None))
        else:
            self._client.delete(key)

    def execute(self) -> list[object]:
        results: list[object] = []
        for command, key, value in self._commands:
            if command == "set":
                assert value is not None
                results.append(self._client.set(key, value))
            else:
                results.append(self._client.delete(key))
        self._transaction = False
        self._commands.clear()
        return results


class _FakeRedisClient:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self.server_now_ms = 0

    def set(self, key: str, value: str, *, nx: bool = False) -> bool:
        if nx and key in self._values:
            return False
        self._values[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def delete(self, *keys: str) -> int:
        return sum(int(self._values.pop(key, None) is not None) for key in keys)

    def pipeline(self) -> _FakeRedisPipeline:
        return _FakeRedisPipeline(self)

    def scan_iter(self, pattern: str) -> list[str]:
        prefix = pattern.removesuffix("*")
        return [key for key in self._values if key.startswith(prefix)]

    def time(self) -> tuple[int, int]:
        return self.server_now_ms // 1000, (self.server_now_ms % 1000) * 1000


def _redis_store() -> RedisCheckpointStore:
    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = _FakeWatchError
    store._client = _FakeRedisClient()
    return store


def _fixture(name: str) -> dict[str, Any]:
    return _strict_json_loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _codec_case(name: str) -> dict[str, Any]:
    fixture = _fixture("checkpoint_codec.json")
    return deepcopy(next(case["payload"] for case in fixture["valid_cases"] if case["name"] == name))


def _minimal_checkpoint(*, key: str = "fresh") -> Checkpoint:
    payload = _codec_case("minimal_running")
    payload["checkpoint_key"] = key
    return checkpoint_from_dict(payload)


def _checkpoint_created_event(*, event_id: str, checkpoint: Checkpoint) -> dict[str, Any]:
    return CheckpointCreatedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        checkpoint_key=checkpoint.checkpoint_key,
        resume_attempt=checkpoint.resume_attempt,
        cycle_index=checkpoint.cycle_index,
        event_id=event_id,
        created_at=123.0,
    ).to_dict()


def _store(store_kind: str, tmp_path: Path, name: str) -> Any:
    if store_kind == "memory":
        return InMemoryCheckpointStore()
    if store_kind == "sqlite":
        return SqliteCheckpointStore(tmp_path / f"{name}.sqlite3")
    return _redis_store()


def _journal_case(name: str) -> dict[str, Any]:
    fixture = _fixture("operation_journal.json")
    return deepcopy(next(case["entry"] for case in fixture["valid_entries"] if case["name"] == name))


def _model_event_kwargs(checkpoint: Checkpoint, journal: OperationJournalEntry) -> dict[str, Any]:
    assert journal.call_id is not None
    assert journal.model_operation is not None
    assert journal.backend is not None
    assert journal.model is not None
    return {
        "run_id": checkpoint.root_run_id,
        "trace_id": checkpoint.trace_id,
        "call_id": journal.call_id,
        "operation_id": journal.operation_id,
        "attempt": journal.attempt,
        "operation": journal.model_operation,
        "cycle_index": journal.cycle_index,
        "backend": journal.backend,
        "model": journal.model,
    }


def _append_outbox_event(checkpoint: Checkpoint, event: Any) -> None:
    payload = event.to_dict()
    checkpoint.event_outbox.append(EventOutboxEntry.pending(payload["event_id"], payload))


def _attach_model_accounting(
    checkpoint: Checkpoint,
    journal: OperationJournalEntry,
    *,
    status: ModelCallStatus | None = None,
    usage: TokenUsage | None = None,
    error_code: str = "provider_rejected",
) -> None:
    identity = _model_event_kwargs(checkpoint, journal)
    _append_outbox_event(
        checkpoint,
        ModelCallStartedEvent(
            **identity,
            event_id=f"evt-{journal.call_id}-started",
            created_at=100.0,
        ),
    )
    if status is None:
        return

    effective_usage = usage or TokenUsage()
    checkpoint.model_calls.append(
        ModelCallRecord(
            call_id=identity["call_id"],
            operation_id=identity["operation_id"],
            attempt=identity["attempt"],
            operation=identity["operation"],
            cycle_index=identity["cycle_index"],
            backend=identity["backend"],
            model=identity["model"],
            status=status,
            usage=deepcopy(effective_usage),
            error_code=None if status is ModelCallStatus.COMPLETED else error_code,
        )
    )
    if status is ModelCallStatus.COMPLETED:
        terminal_event = ModelCallCompletedEvent(
            **identity,
            usage=deepcopy(effective_usage),
            event_id=f"evt-{journal.call_id}-completed",
            created_at=101.0,
        )
    else:
        terminal_event = ModelCallFailedEvent(
            **identity,
            outcome="ambiguous" if status is ModelCallStatus.AMBIGUOUS else "definitive",
            usage=deepcopy(effective_usage),
            error_code=error_code,
            event_id=f"evt-{journal.call_id}-failed",
            created_at=101.0,
        )
    _append_outbox_event(checkpoint, terminal_event)


def _checkpoint_with_model_journal(
    journal_case: str,
    *,
    status: ModelCallStatus | None = None,
    error_code: str = "provider_rejected",
) -> Checkpoint:
    checkpoint = _minimal_checkpoint(key=f"accounting-{journal_case}")
    checkpoint.claim_token = "owner"
    checkpoint.claimed_cycle = 1
    checkpoint.lease_expires_at_ms = 200
    journal = OperationJournalEntry.from_dict(_journal_case(journal_case))
    checkpoint.model_call_journal = [journal]
    if journal.state is OperationState.AMBIGUOUS:
        checkpoint.status = AgentStatus.RECONCILIATION_REQUIRED
        checkpoint.claim_token = None
        checkpoint.claimed_cycle = None
        checkpoint.lease_expires_at_ms = None
    if journal.state is not OperationState.PLANNED and not (
        journal.state is OperationState.FAILED and status is None
    ):
        _attach_model_accounting(
            checkpoint,
            journal,
            status=status,
            error_code=error_code,
        )
    return checkpoint


def test_rfc8785_vectors_match_canonical_bytes_and_digests() -> None:
    vectors = (
        ("run_definition.json", ("golden_cases",), "definition"),
        ("operation_journal.json", ("request_digest", "golden_cases"), "request"),
        ("checkpoint_codec.json", ("extension_limits", "canonicalization_vectors"), "entry"),
        ("checkpoint_store.json", ("event_payload_digest", "golden_cases"), "event"),
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
    fixture = _fixture("run_definition.json")
    for case in fixture["golden_cases"]:
        assert compute_run_definition_digest(case["definition"]) == case["sha256"]


def test_checkpoint_round_trip_uses_jcs() -> None:
    fixture = _fixture("checkpoint_codec.json")
    checkpoint = checkpoint_from_dict(fixture["canonical_checkpoint"])
    encoded = checkpoint_to_dict(checkpoint)

    assert encoded == fixture["canonical_checkpoint"]
    wire = checkpoint_to_json(checkpoint)
    assert wire.encode("utf-8") == canonical_json_bytes(encoded)
    assert checkpoint_from_json(wire) == checkpoint


def test_checkpoint_round_trip_restores_jcs_large_float_through_codec_and_sqlite(
    tmp_path: Path,
) -> None:
    definition_case = next(
        case for case in _fixture("run_definition.json")["golden_cases"] if case["name"] == "full_unicode_float_and_capabilities"
    )
    checkpoint = _minimal_checkpoint(key="jcs-large-float")
    checkpoint.run_definition = deepcopy(definition_case["definition"])
    checkpoint.run_definition_digest = definition_case["sha256"]

    wire = checkpoint_to_json(checkpoint)
    assert '"large_number":100000000000000000000' in wire
    decoded = checkpoint_from_json(wire)
    large_number = decoded.run_definition["model"]["settings"]["extra_body"]["large_number"]
    assert isinstance(large_number, float)
    assert large_number == 1e20

    store = SqliteCheckpointStore(tmp_path / "jcs-large-float.sqlite3")
    assert store.create_checkpoint(checkpoint)
    restored = store.load_checkpoint(checkpoint.checkpoint_key)
    assert restored is not None
    restored_large_number = restored.run_definition["model"]["settings"]["extra_body"]["large_number"]
    assert isinstance(restored_large_number, float)
    assert restored_large_number == 1e20
    assert restored.run_definition_digest == definition_case["sha256"]


def test_run_definition_rejects_host_integer_above_i_json_safe_range() -> None:
    definition = deepcopy(_fixture("run_definition.json")["golden_cases"][0]["definition"])
    definition["model"]["settings"] = {
        "extra_body": {"count": 9_007_199_254_740_992},
    }

    with pytest.raises(CheckpointError) as error:
        compute_run_definition_digest(definition)

    assert error.value.code == "checkpoint_definition_not_i_json"


def test_checkpoint_validates_embedded_definition_schema_and_digest() -> None:
    payload = _codec_case("minimal_running")
    assert checkpoint_from_dict(payload).run_definition_digest == compute_run_definition_digest(payload["run_definition"])

    missing_definition = deepcopy(payload)
    missing_definition.pop("run_definition")
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(missing_definition)
    assert error.value.code == "checkpoint_missing_field"

    missing_definition_schema = deepcopy(payload)
    missing_definition_schema.pop("run_definition_schema")
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(missing_definition_schema)
    assert error.value.code == "checkpoint_missing_field"

    mismatch = deepcopy(payload)
    mismatch["run_definition"]["root_input"] = "different"
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(mismatch)
    assert error.value.code == "checkpoint_definition_mismatch"

    unknown = deepcopy(payload)
    unknown["schema_version"] = "vv-agent.checkpoint.v4"
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(unknown)
    assert error.value.code == "checkpoint_schema_unsupported"


def test_checkpoint_resume_rejects_definition_mismatch_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(case for case in _fixture("run_definition.json")["golden_cases"] if case["name"] == "minimal")
    stored_definition = deepcopy(case["definition"])
    stored_digest = case["sha256"]
    current_definition = deepcopy(stored_definition)
    current_definition["root_input"] = "Different input"
    checkpoint_key = "definition-mismatch"
    store = InMemoryCheckpointStore()
    checkpoint = Checkpoint(
        checkpoint_key=checkpoint_key,
        task_id="task",
        root_run_id="run",
        trace_id="trace",
        run_definition=deepcopy(stored_definition),
        run_definition_digest=stored_digest,
        resume_attempt=1,
        cycle_index=0,
        status=AgentStatus.RUNNING,
        messages=[],
        cycles=[],
    )
    assert store.create_checkpoint(checkpoint)
    claims = 0
    original_claim = store.claim_checkpoint

    def count_claim(*args: Any, **kwargs: Any) -> Any:
        nonlocal claims
        claims += 1
        return original_claim(*args, **kwargs)

    monkeypatch.setattr(store, "claim_checkpoint", count_claim)
    controller = CheckpointResumeController(
        config=CheckpointConfig(
            store=store,
            key=checkpoint_key,
            resume_policy=ResumePolicy.REQUIRE_EXISTING,
        ),
        task_id="task",
        run_id="run",
        trace_id="trace",
        run_definition=current_definition,
        run_definition_digest=compute_run_definition_digest(current_definition),
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )

    with pytest.raises(CheckpointError) as error:
        controller.admit()

    assert error.value.code == "checkpoint_definition_mismatch"
    assert claims == 0
    retained = store.load_checkpoint(checkpoint_key)
    assert retained is not None
    assert retained.run_definition == stored_definition
    assert retained.run_definition_digest == stored_digest


def test_checkpoint_invalid_fixture_cases_have_stable_codes() -> None:
    fixture = _fixture("checkpoint_codec.json")
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
        "unknown_top_level_is_rejected": "checkpoint_unknown_field",
    }
    for case in fixture["invalid_cases"]:
        registered = [] if case["name"] == "unknown_required_extension" else None
        with pytest.raises(CheckpointError) as error:
            checkpoint_from_dict(
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
            checkpoint_from_dict(payload)
        assert error.value.code == case["error_code"]


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
    assert checkpoint_from_dict(payload, registered_extensions=[extension])
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(payload, registered_extensions=[])
    assert error.value.code == "checkpoint_required_extension_unavailable"


def test_operation_and_event_digest_golden_vectors() -> None:
    journal_fixture = _fixture("operation_journal.json")
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

    store_fixture = _fixture("checkpoint_store.json")
    event_case = store_fixture["event_payload_digest"]["golden_cases"][0]
    assert compute_event_payload_digest(event_case["event"]) == event_case["sha256"]
    pending = EventOutboxEntry.pending("evt-1", event_case["event"])
    assert pending.payload_digest == event_case["sha256"]
    pending.verify_payload()


def test_operation_journal_invalid_cases_have_stable_codes() -> None:
    fixture = _fixture("operation_journal.json")
    for case in fixture["valid_entries"]:
        OperationJournalEntry.from_dict(case["entry"])
    for case in fixture["invalid_entries"]:
        if "base_valid_entry" in case:
            entry = _journal_case(case["base_valid_entry"])
            mutation = case["mutation"]
            if "remove" in mutation:
                entry.pop(mutation["remove"])
            if "replace" in mutation:
                entry.update(mutation["replace"])
        else:
            entry = case["entry"]
        with pytest.raises(CheckpointError) as error:
            OperationJournalEntry.from_dict(entry)
        assert error.value.code == case["error_code"], case["name"]


@pytest.mark.parametrize(
    ("journal_case", "status"),
    [
        ("model_planned", None),
        ("model_started", None),
        ("model_succeeded", ModelCallStatus.COMPLETED),
        ("model_failed", None),
        ("model_failed", ModelCallStatus.FAILED),
        ("model_ambiguous", ModelCallStatus.AMBIGUOUS),
    ],
)
def test_model_journal_accepts_complete_atomic_accounting_states(
    journal_case: str,
    status: ModelCallStatus | None,
) -> None:
    checkpoint_to_dict(_checkpoint_with_model_journal(journal_case, status=status))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("call_id", "different-call"),
        ("operation_id", "different-operation"),
        ("attempt", 2),
        ("operation", "session_memory"),
        ("cycle_index", 2),
        ("backend", "different-backend"),
        ("model", "different-model"),
    ],
)
def test_model_started_event_identity_must_match_journal(
    field: str,
    replacement: Any,
) -> None:
    checkpoint = _checkpoint_with_model_journal("model_started")
    event = deepcopy(checkpoint.event_outbox[0].event)
    event[field] = replacement
    checkpoint.event_outbox[0] = EventOutboxEntry.pending(event["event_id"], event)

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_model_started_journal_requires_atomic_started_event() -> None:
    checkpoint = _checkpoint_with_model_journal("model_started")
    checkpoint.event_outbox.clear()

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


@pytest.mark.parametrize("missing", ["started_event", "terminal_event", "ledger_record"])
def test_terminal_model_journal_requires_complete_atomic_evidence(missing: str) -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_succeeded",
        status=ModelCallStatus.COMPLETED,
    )
    if missing == "started_event":
        checkpoint.event_outbox.pop(0)
    elif missing == "terminal_event":
        checkpoint.event_outbox.pop()
    else:
        checkpoint.model_calls.clear()

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_terminal_model_event_usage_must_match_ledger() -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_succeeded",
        status=ModelCallStatus.COMPLETED,
    )
    event = deepcopy(checkpoint.event_outbox[-1].event)
    event["usage"] = TokenUsage(input_tokens=1, total_tokens=1).to_dict()
    checkpoint.event_outbox[-1] = EventOutboxEntry.pending(event["event_id"], event)

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_failed_model_event_error_must_match_ledger() -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_failed",
        status=ModelCallStatus.FAILED,
    )
    event = deepcopy(checkpoint.event_outbox[-1].event)
    event["error_code"] = "different_error"
    checkpoint.event_outbox[-1] = EventOutboxEntry.pending(event["event_id"], event)

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_terminal_model_event_type_must_match_ledger_status() -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_succeeded",
        status=ModelCallStatus.COMPLETED,
    )
    journal = checkpoint.model_call_journal[0]
    failed_event = ModelCallFailedEvent(
        **_model_event_kwargs(checkpoint, journal),
        outcome="definitive",
        usage=deepcopy(checkpoint.model_calls[0].usage),
        error_code="provider_rejected",
        event_id=f"evt-{journal.call_id}-failed",
        created_at=101.0,
    )
    checkpoint.event_outbox[-1] = EventOutboxEntry.pending(
        failed_event.event_id,
        failed_event.to_dict(),
    )

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_checkpoint_config_validates_store_key_capabilities_and_stable_codes() -> None:
    store = InMemoryCheckpointStore()
    config = CheckpointConfig(
        store=store,
        key=None,
        capability_refs={"runtime_hook:0": {"id": "hook.audit", "version": "1"}},
    )
    assert config.resume_policy is ResumePolicy.NEW
    assert isinstance(store, CheckpointStore)

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
    assert store.create_checkpoint(checkpoint)

    continued = store.claim_checkpoint(
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
        store.claim_checkpoint(
            checkpoint.checkpoint_key,
            1,
            claim_token="owner-b",
            lease_expires_at_ms=300,
            now_ms=199,
            claim_mode="recovery",
        )
    rejected = store.load_checkpoint(checkpoint.checkpoint_key)
    assert rejected is not None
    assert rejected.resume_attempt == 1

    recovered = store.claim_checkpoint(
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
    assert store.create_checkpoint(checkpoint)
    assert store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="expired-owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    barrier = Barrier(3)
    result_lock = Lock()
    claims: list[Checkpoint] = []
    conflicts: list[CheckpointConflictError] = []

    def recover(owner: str) -> None:
        barrier.wait()
        try:
            claimed = store.claim_checkpoint(
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
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
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
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    started = OperationJournalEntry.from_dict(_journal_case("model_started"))
    claimed.model_call_journal = [started]
    _attach_model_accounting(claimed, started)
    claimed.shared_state["progress"] = "started"
    assert store.progress_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    assert store.renew_checkpoint_claim(
        checkpoint.checkpoint_key,
        claim_token="owner",
        lease_expires_at_ms=300,
        now_ms=150,
    )

    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
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
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
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
    _attach_model_accounting(
        claimed,
        ambiguous,
        status=ModelCallStatus.AMBIGUOUS,
        error_code="model_outcome_unknown",
    )
    claimed.status = AgentStatus.RECONCILIATION_REQUIRED
    assert store.suspend_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    suspended = store.load_checkpoint(checkpoint.checkpoint_key)
    assert suspended is not None
    assert suspended.status is AgentStatus.RECONCILIATION_REQUIRED
    assert suspended.revision == 2
    assert suspended.resume_attempt == 1
    assert suspended.claim_token is None
    assert suspended.model_call_journal[0].state is OperationState.AMBIGUOUS

    recovery = store.claim_checkpoint(
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
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    succeeded = OperationJournalEntry.from_dict(_journal_case("model_succeeded"))
    claimed.model_call_journal = [succeeded]
    _attach_model_accounting(
        claimed,
        succeeded,
        status=ModelCallStatus.COMPLETED,
    )
    claimed.cycle_index = 1
    assert store.commit_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    committed = store.load_checkpoint(checkpoint.checkpoint_key)
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
        token_usage=summarize_task_token_usage(committed.model_calls),
        checkpoint_key=committed.checkpoint_key,
    )
    assert store.finalize_checkpoint(committed, expected_revision=committed.revision)

    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.revision == 3
    assert terminal.terminal_result is not None
    assert not store.finalize_checkpoint(committed, expected_revision=terminal.revision)
    assert store.acknowledge_terminal(
        checkpoint.checkpoint_key,
        expected_revision=terminal.revision,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.revision == 4
    assert retained.terminal_acknowledged
    assert retained.terminal_result is not None
    assert not store.acknowledge_terminal(
        checkpoint.checkpoint_key,
        expected_revision=retained.revision,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cycle_commit_rejects_model_journal_without_atomic_accounting(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "invalid-commit-accounting")
    checkpoint = _minimal_checkpoint(key=f"invalid-commit-accounting-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
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

    with pytest.raises(CheckpointError) as error:
        store.commit_checkpoint(
            claimed,
            claim_token="owner",
            expected_revision=claimed.revision,
        )

    assert error.value.code == "checkpoint_status_invalid"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_terminal_rejects_model_journal_without_atomic_accounting(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "invalid-terminal-accounting")
    checkpoint = _minimal_checkpoint(key=f"invalid-terminal-accounting-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.model_call_journal = [OperationJournalEntry.from_dict(_journal_case("model_succeeded"))]
    claimed.status = AgentStatus.FAILED
    claimed.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        error="failed after model dispatch",
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=claimed.checkpoint_key,
    )

    with pytest.raises(CheckpointError) as error:
        store.finalize_claimed_checkpoint(
            claimed,
            claim_token="owner",
            expected_revision=claimed.revision,
        )

    assert error.value.code == "checkpoint_status_invalid"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_operator_abort_finalize_preserves_ambiguous_evidence(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "operator-abort")
    checkpoint = checkpoint_from_dict(_codec_case("reconciliation_required_retains_ambiguous_journal"))
    checkpoint.checkpoint_key = f"abort-{store_kind}"
    checkpoint.resume_attempt = 1
    checkpoint.revision = 0
    assert store.create_checkpoint(checkpoint)

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
    assert store.finalize_checkpoint(checkpoint, expected_revision=0)
    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
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
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
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

    assert store.finalize_claimed_checkpoint(
        claimed,
        claim_token="owner-failure",
        expected_revision=claimed.revision,
    )
    terminal = store.load_checkpoint(claimed.checkpoint_key)
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
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
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

    assert store.finalize_claimed_checkpoint(
        claimed,
        claim_token="owner-reconcile",
        expected_revision=claimed.revision,
    )
    terminal = store.load_checkpoint(claimed.checkpoint_key)
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
    first = EventOutboxEntry.pending(
        "evt-created",
        _checkpoint_created_event(event_id="evt-created", checkpoint=checkpoint),
    )
    checkpoint.event_outbox = [first]
    assert store.create_checkpoint(checkpoint)
    first_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 1},
        last_event_id="evt-created",
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=first.event_id,
        payload_digest=first.payload_digest,
        cursor=first_cursor,
        expected_revision=0,
        claim_token=None,
    )

    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-events",
        lease_expires_at_ms=300,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    second = EventOutboxEntry.pending(
        "evt-resumed",
        CheckpointResumedEvent(
            run_id=claimed.root_run_id,
            trace_id=claimed.trace_id,
            checkpoint_key=claimed.checkpoint_key,
            resume_attempt=claimed.resume_attempt,
            cycle_index=claimed.cycle_index,
            event_id="evt-resumed",
            created_at=124.0,
        ).to_dict(),
    )
    claimed.event_outbox.append(second)
    assert store.progress_checkpoint(
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
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=second.event_id,
        payload_digest=second.payload_digest,
        cursor=second_cursor,
        expected_revision=claimed.revision,
        claim_token="owner-events",
    )
    delivered = store.load_checkpoint(checkpoint.checkpoint_key)
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
    terminal_event = EventOutboxEntry.pending(
        "evt-terminal",
        RunFailedEvent(
            run_id=delivered.root_run_id,
            trace_id=delivered.trace_id,
            error="terminal",
            cycle_index=delivered.cycle_index,
            event_id="evt-terminal",
            created_at=125.0,
        ).to_dict(),
    )
    delivered.event_outbox.append(terminal_event)
    assert store.finalize_claimed_checkpoint(
        delivered,
        claim_token="owner-events",
        expected_revision=delivered.revision,
    )
    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    terminal_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 3},
        last_event_id="evt-terminal",
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=terminal_event.event_id,
        payload_digest=terminal_event.payload_digest,
        cursor=terminal_cursor,
        expected_revision=terminal.revision,
        claim_token=None,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.terminal_result is not None
    assert retained.event_outbox[-1].state == "delivered"


def test_event_outbox_rejects_partial_unknown_and_mismatched_current_events() -> None:
    checkpoint = _minimal_checkpoint(key="strict-outbox-event")
    current = _checkpoint_created_event(event_id="evt-current", checkpoint=checkpoint)

    partial = deepcopy(current)
    partial.pop("run_id")
    with pytest.raises(CheckpointError) as error:
        EventOutboxEntry.pending("evt-current", partial)
    assert error.value.code == "checkpoint_event_invalid"

    unknown = deepcopy(current)
    unknown["unknown_field"] = True
    with pytest.raises(CheckpointError) as error:
        EventOutboxEntry.pending("evt-current", unknown)
    assert error.value.code == "checkpoint_event_invalid"

    with pytest.raises(CheckpointError) as error:
        EventOutboxEntry.pending("evt-other", current)
    assert error.value.code == "event_identity_conflict"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_store_rejects_run_definition_replacement_during_progress(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "definition-immutable")
    checkpoint = _minimal_checkpoint(key=f"definition-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
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
    assert not store.progress_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.run_definition["root_input"] == "Summarize the status."


def test_sqlite_uses_only_the_current_checkpoint_schema(tmp_path: Path) -> None:
    store = SqliteCheckpointStore(tmp_path / "schema.sqlite3")
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(checkpoints)").fetchall()}
    assert {"run_definition_schema", "run_definition"} <= columns
    checkpoint = _minimal_checkpoint()
    assert store.create_checkpoint(checkpoint)
    assert store.load_checkpoint(checkpoint.checkpoint_key) is not None


def test_sqlite_rejects_non_current_checkpoint_table_schema(tmp_path: Path) -> None:
    database = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE checkpoints (task_id TEXT PRIMARY KEY)")

    with pytest.raises(RuntimeError, match="does not match"):
        SqliteCheckpointStore(database)


def test_sqlite_ignores_unrelated_checkpoint_prefixed_tables(tmp_path: Path) -> None:
    database = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE checkpoint_archive (task_id TEXT PRIMARY KEY)")

    store = SqliteCheckpointStore(database)
    assert store._schema_sql("table", "checkpoint_archive") is not None
    assert store._schema_sql("table", "checkpoints") is not None


def test_redis_key_vectors_match_contract() -> None:
    fixture = _fixture("checkpoint_store.json")
    for vector in fixture["redis_key_vectors"]:
        data_key = RedisCheckpointStore.data_key(vector["checkpoint_key"])
        assert data_key == vector["data_key"]
        assert RedisCheckpointStore._keys(vector["checkpoint_key"]) == (
            vector["data_key"],
            vector["lease_key"],
        )


def test_cross_runtime_sqlite_probe_from_environment() -> None:
    database = os.environ.get("VV_AGENT_CROSS_RUNTIME_DB")
    if database is None:
        return
    mode = os.environ.get("VV_AGENT_CROSS_RUNTIME_MODE", "read_rust")
    store = SqliteCheckpointStore(database)
    if mode == "write_python":
        checkpoint = _minimal_checkpoint(key="python-wrote")
        checkpoint.messages = [Message(role="user", content="from Python")]
        checkpoint.shared_state = {"writer": "python", "format": "checkpoint"}
        assert store.create_checkpoint(checkpoint)
        return
    if mode == "read_rust":
        checkpoint = store.load_checkpoint("rust-wrote")
        assert checkpoint is not None
        assert checkpoint.messages == [Message(role="user", content="from Rust")]
        assert checkpoint.shared_state == {"format": "checkpoint", "writer": "rust"}
        assert checkpoint.run_definition_digest == compute_run_definition_digest(checkpoint.run_definition)
        return
    raise AssertionError(f"unknown cross-runtime mode: {mode}")
