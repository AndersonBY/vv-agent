"""Produce contract facts from real kernel scenarios; no synthetic kernel receipts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from scripts.s1_fixture_values import fixture_key_provider, wrap_fixture_store
from tests.session import test_unseen_inputs as cases
from vv_agent._token_counting import count_messages_tokens
from vv_agent.kernel._boundary_hydration import hydrate_and_fold_records
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import APPROVAL_SCOPE, ContentRef
from vv_agent.kernel._content_reader import input_owner
from vv_agent.kernel._content_runtime import durable_content_reader
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.session.context import project_context, protected_start
from vv_agent.session.projection import hydration_for_session_store
from vv_agent.session.sqlite import SQLiteStore


@dataclass(frozen=True)
class UsageObservation:
    turn_id: str
    meter_id: str
    observation: int


@dataclass(frozen=True)
class AppendGroup:
    session_id: str
    record_ids: tuple[str, ...]
    usage: UsageObservation | None


def generate():
    with (
        patch("vv_agent.session.timing.time.monotonic_ns", return_value=100_000_000),
        patch("vv_agent.session.timing.time.time_ns", return_value=1_000_000_000_000),
        patch.object(cases, "HTTP_PORT", 18737),
    ):
        return _generate()


@contextmanager
def _capture_append_groups() -> Iterator[list[AppendGroup]]:
    """Retain successful SQL append groups and the driver usage emitted at each commit."""

    groups: list[AppendGroup] = []
    observations: dict[str, int] = {}
    pending_usage: tuple[str, UsageObservation | None] = ("", None)
    append = SQLSessionTx.append
    commit = CoreDriver.commit

    def observed_commit(self, records, consumed=(), *, guarded=False, prepare=None, _logical_base=None):
        nonlocal pending_usage
        usage = None
        if self.state.active_turn_id is not None:
            meter_id = f"active/{self.state.active_turn_id}/{self.scope.lease.epoch}"
            observations[meter_id] = observations.get(meter_id, 0) + 1
            usage = UsageObservation(
                turn_id=self.state.active_turn_id,
                meter_id=meter_id,
                observation=observations[meter_id],
            )
        pending_usage = (self.sid, usage)
        try:
            return commit(
                self,
                records,
                consumed,
                guarded=guarded,
                prepare=prepare,
                _logical_base=_logical_base,
            )
        finally:
            pending_usage = ("", None)

    def observed_append(self, session_id: str, **kwargs: Any):
        receipt = append(self, session_id, **kwargs)
        usage = pending_usage[1] if pending_usage[0] == session_id else None
        groups.append(
            AppendGroup(
                session_id=session_id,
                record_ids=tuple(record.record_id for record in kwargs["records"]),
                usage=usage,
            )
        )
        return receipt

    with patch.object(CoreDriver, "commit", observed_commit), patch.object(SQLSessionTx, "append", observed_append):
        yield groups


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _todo_state(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _todo_state(item) for key, item in value.items()}
        return {"todo_list": []} if result == {} else result
    if isinstance(value, list):
        return [_todo_state(item) for item in value]
    return value


def _replace_child_results(value: Any, terminals: dict[str, tuple[str, int]]) -> Any:
    if isinstance(value, dict):
        session_id = value.get("session_id")
        if session_id in terminals and "terminal_digest" in value:
            value["terminal_digest"], value["terminal_seq"] = terminals[session_id]
        for key, item in list(value.items()):
            value[key] = _replace_child_results(item, terminals)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _replace_child_results(item, terminals)
        return value
    if isinstance(value, str) and value.startswith("Child completed: "):
        child = json.loads(value.removeprefix("Child completed: "))
        child = _replace_child_results(child, terminals)
        return "Child completed: " + json.dumps(child, ensure_ascii=False, sort_keys=True)
    return value


def _collect_hydrated_content(raw: Any, hydrated: Any, values: dict[bytes, Any]) -> None:
    if isinstance(raw, dict):
        try:
            ref = ContentRef.from_dict(raw)
        except (TypeError, ValueError):
            if isinstance(hydrated, dict):
                for key, item in raw.items():
                    if key in hydrated:
                        _collect_hydrated_content(item, hydrated[key], values)
            return
        values[canonical_json_bytes(ref.to_dict())] = deepcopy(hydrated)
        return
    if isinstance(raw, list) and isinstance(hydrated, list):
        for raw_item, hydrated_item in zip(raw, hydrated, strict=False):
            _collect_hydrated_content(raw_item, hydrated_item, values)


def _approval_scope(
    store,
    session_id: str,
    record: dict[str, Any],
    raw_record,
    hydrated_content: dict[bytes, Any],
) -> None:
    item = record["payload"]["input"]
    raw_item = raw_record._payload["input"]
    if item["kind"] != "approval_answer" or not isinstance(raw_item["payload"].get("scope"), dict):
        return
    payload = item["payload"]
    scope = ContentRef.from_dict(raw_item["payload"]["scope"])
    reader = durable_content_reader(store, tenant_id=store.tenant_id, session_id=session_id)
    body = reader.read(store.tenant_id, session_id, input_owner(raw_item["input_id"]), APPROVAL_SCOPE, scope)
    value = json.loads(body)
    while isinstance(value, dict):
        try:
            key = canonical_json_bytes(ContentRef.from_dict(value).to_dict())
        except (TypeError, ValueError):
            break
        if key not in hydrated_content:
            break
        value = deepcopy(hydrated_content[key])
    payload["scope"] = value


def _prepare_record(store, session_id: str, row, raw_record, hydrated_content: dict[bytes, Any]) -> dict[str, Any]:
    record = deepcopy(row.record.to_dict())
    record.pop("_digest", None)
    record.pop("feature_level", None)
    record["schema_version"] = 1
    payload = record["payload"]
    kind = record["kind"]
    if kind == "session_created":
        payload["principal"] = "test"
        seed = payload["attributes"].get("seed")
        if seed is not None:
            if seed.get("messages") == []:
                seed.pop("messages")
            if seed.get("shared_state") == {}:
                seed.pop("shared_state")
    elif kind == "turn_started":
        for name in ("policy_digest", "tokenizer_binding", "requirements", "environment"):
            payload.pop(name, None)
        definition = payload["definition"]
        if definition.get("model_binding", {}).get("endpoints") == []:
            definition["model_binding"].pop("endpoints")
        definition["task"]["initial_shared_state"] = _todo_state(definition["task"].get("initial_shared_state", {}))
    elif kind == "input_applied":
        payload.pop("input_proof", None)
        item = payload["input"]
        for name in ("actor", "admission_proof", "feature_level"):
            item.pop(name, None)
        item["schema_version"] = 3
        _approval_scope(store, session_id, record, raw_record, hydrated_content)
        if item["kind"] == "approval_answer":
            # Contract 39 recorded the late cancellation answer as applied while retaining
            # the original approval target for audit projection.
            payload["disposition"] = "applied"
            payload["reason"] = None
    elif kind in {"op_planned", "op_prepared", "op_completed"} and payload.get("policy_decision") is None:
        payload.pop("policy_decision", None)
    return record


def _nested_request_digests(value: Any, requests: dict[str, str], operation_id: str | None = None) -> Any:
    if isinstance(value, dict):
        operation_id = value.get("operation_id", operation_id)
        if operation_id in requests and "request_digest" in value:
            value["request_digest"] = requests[operation_id]
        for item in value.values():
            _nested_request_digests(item, requests, operation_id)
        return value
    if isinstance(value, list):
        for item in value:
            _nested_request_digests(item, requests, operation_id)
        return value
    return value


def _request_digests(records: list[dict[str, Any]]) -> dict[str, str]:
    return {
        record["operation_id"]: _digest(record["payload"]["request"])
        for record in records
        if record["kind"] in {"op_planned", "op_prepared"}
    }


def _finish_payload_digests(records: list[dict[str, Any]]) -> None:
    requests = _request_digests(records)
    for record in records:
        payload = record["payload"]
        _nested_request_digests(payload, requests, record.get("operation_id"))
        if record["kind"] in {"op_planned", "op_prepared"}:
            payload["request_digest"] = requests[record["operation_id"]]
        elif record["kind"] == "turn_started":
            payload["definition_digest"] = _digest(payload["definition"])
        elif record["kind"] == "input_applied":
            payload["input_digest"] = _digest(payload["input"])
        elif record["kind"] == "op_completed":
            payload["request_digest"] = requests[record["operation_id"]]
            payload["result_digest"] = _digest(payload["result"])


def _count_historical_tokens(messages: list[dict[str, Any]]) -> int:
    return count_messages_tokens([message for message in messages if not message.get("tool_calls") or message.get("content")])


def _source_messages(store, session_id: str, records, boundary_index: int, terminals) -> list[dict[str, Any]]:
    hydrated = hydrate_and_fold_records(
        records[:boundary_index],
        store.fold_factory,
        hydration_for_session_store(store, session_id),
    )
    messages = [message.to_dict() for message in project_context(hydrated.hydrated_records, hydrated.state)]
    return _replace_child_results(messages, terminals)


def _history_projection(store, session_id: str, records, rows, groups: list[AppendGroup], terminals):
    current = list(rows)
    raw_by_id = {row.record.record_id: row.record for row in records}
    hydrated_content: dict[bytes, Any] = {}
    for row in current:
        _collect_hydrated_content(raw_by_id[row.record.record_id]._payload, row.record._payload, hydrated_content)
    prepared = [_prepare_record(store, session_id, row, raw_by_id[row.record.record_id], hydrated_content) for row in current]
    by_id = {record["record_id"]: record for record in prepared}
    original_ids = {id(record): record["record_id"] for record in prepared}

    historical_state: dict[str, Any] = {"todo_list": []}
    for record in prepared:
        payload = record["payload"]
        if record["kind"] in {"op_planned", "op_prepared"}:
            request = payload.get("request", {})
            metadata = request.get("metadata", {})
            if payload.get("purpose") != "compaction" and "vv_session" in metadata:
                metadata["vv_session"]["shared_state"] = deepcopy(historical_state)
        elif record["kind"] == "op_completed" and payload.get("shared_state") == {}:
            payload["shared_state"] = deepcopy(historical_state)
        if record["kind"] == "boundary_recorded" and "shared_state" in payload["data"]:
            historical_state = deepcopy(payload["data"]["shared_state"])

    for record in prepared:
        _replace_child_results(record["payload"], terminals)

    starts = [
        record for record in prepared if record["kind"] == "boundary_recorded" and record["payload"]["stage"] == "memory_started"
    ]
    finishes = [
        record
        for record in prepared
        if record["kind"] == "boundary_recorded" and record["payload"]["stage"] == "memory_completed"
    ]
    compactions = [record for record in prepared if record["kind"] == "context_compacted"]
    boundary_map: dict[str, str] = {}
    source_map: dict[str, str] = {}
    source_by_boundary: dict[str, list[dict[str, Any]]] = {}
    for index, (start, compaction) in enumerate(zip(starts, compactions, strict=True)):
        current_boundary_id = start["payload"]["boundary_id"]
        boundary_index = next(i for i, row in enumerate(current) if row.record.record_id == start["record_id"])
        source = _source_messages(store, session_id, records, boundary_index, terminals)
        event = start["payload"]["data"]["event"]
        key_parts: list[Any] = [record["turn_id"], source, event["trigger"]]
        requested_by = event.get("metadata", {}).get("requested_by")
        if requested_by is not None:
            key_parts.append(requested_by)
        legacy_boundary_id = _digest(key_parts)
        legacy_source_digest = _digest(source)
        boundary_map[current_boundary_id] = legacy_boundary_id
        source_map[compaction["payload"]["source_digest"]] = legacy_source_digest
        source_by_boundary[legacy_boundary_id] = source
        event["estimated_tokens"] = _count_historical_tokens(source)
        event["message_count"] = len(source)
        if index < len(finishes):
            finish = finishes[index]
            boundary_map[finish["payload"]["boundary_id"]] = legacy_boundary_id
        for field in ("prefix_ids", "tail_ids"):
            ids = []
            for identity in compaction["payload"][field]:
                sequence, _, digest = identity.partition("/")
                message_index = int(sequence)
                if 0 <= message_index < len(source):
                    digest = _digest(source[message_index])
                ids.append(f"{message_index}/{digest}")
            compaction["payload"][field] = ids

    for finish, compaction in zip(finishes, compactions, strict=True):
        event = finish["payload"]["data"]["event"]
        event["summary_tokens"] = _count_historical_tokens(compaction["payload"]["replacement"])

    def replace_historical_ids(value: Any) -> Any:
        if isinstance(value, dict):
            payload = value.get("payload")
            if value.get("kind") == "boundary_recorded" and isinstance(payload, dict):
                stage = payload.get("stage")
                if stage in {"memory_started", "memory_completed"}:
                    event = payload["data"]["event"]
                    event["estimated_tokens"] = (
                        _count_historical_tokens(source_by_boundary[boundary_map[payload["boundary_id"]]])
                        if stage == "memory_started"
                        else event.get("estimated_tokens")
                    )
            for key, item in list(value.items()):
                value[key] = replace_historical_ids(item)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = replace_historical_ids(item)
            return value
        if isinstance(value, str):
            for current_id, legacy_id in boundary_map.items():
                value = value.replace(current_id, legacy_id)
            for current_digest, legacy_digest in source_map.items():
                value = value.replace(current_digest, legacy_digest)
        return value

    for record in prepared:
        replace_historical_ids(record)
        if record["kind"] == "boundary_recorded" and record["payload"]["stage"] == "memory_completed":
            record["payload"]["data"]["event"].pop("estimated_tokens", None)

    current_ids = {row.record.record_id for row in current}
    grouped_ids: set[str] = set()
    session_groups = [group for group in groups if group.session_id == session_id]
    group_record_ids = {item for group in session_groups for item in group.record_ids}
    direct = [record for record in prepared if original_ids[id(record)] not in group_record_ids]
    output: list[dict[str, Any]] = list(direct)
    planned_versions: dict[str, str] = {}
    for group in session_groups:
        if not group.record_ids:
            present = True
        else:
            present = any(identity in current_ids for identity in group.record_ids)
            if not present:
                continue
        context_version = str(len(output))
        if group.usage is not None:
            usage = group.usage
            output.append(
                {
                    "attempt": None,
                    "kind": "usage_observed",
                    "operation_id": None,
                    "payload": {
                        "meter_id": usage.meter_id,
                        "mode": "cumulative",
                        "observation": usage.observation,
                        "source": "kernel-active-scope",
                        "usage": {"elapsed_ms": 0, "host_cost": None},
                    },
                    "record_id": f"usage/{usage.meter_id}/{usage.observation}",
                    "schema_version": 1,
                    "session_id": session_id,
                    "turn_id": usage.turn_id,
                }
            )
        for identity in group.record_ids:
            if identity not in current_ids:
                continue
            record = by_id[identity]
            if record["kind"] == "op_planned":
                operation_id = record["operation_id"]
                if (record.get("attempt") or 1) > 1 and operation_id in planned_versions:
                    record["payload"]["context_version"] = planned_versions[operation_id]
                else:
                    record["payload"]["context_version"] = context_version
                    planned_versions[operation_id] = context_version
            output.append(record)
            grouped_ids.add(identity)
    projected_ids = grouped_ids | {original_ids[id(record)] for record in direct}
    if projected_ids != current_ids:
        missing = sorted(current_ids - projected_ids)
        extra = sorted(projected_ids - current_ids)
        raise AssertionError(f"historical append groups mismatch: missing={missing}, extra={extra}")
    if [record["kind"] for record in output if record["kind"] != "usage_observed"] != [record["kind"] for record in prepared]:
        raise AssertionError("historical append groups reordered logical records")

    _finish_payload_digests(prepared)
    for record in output:
        record.pop("_digest", None)
        record["_digest"] = _digest(record)
    return prepared, output


def _generate():
    result = []
    scenarios: list[tuple[str, Callable[..., dict[str, Any] | None], tuple[Any, ...]]] = [
        ("steer", cases.test_steer_survives_ptl_in_real_primary_request, (1,)),
        ("twelve_steers", cases.test_steer_survives_ptl_in_real_primary_request, (12,)),
        ("user", cases.test_user_reply_survives_ptl, ()),
        ("tool_user_reply", cases.test_tool_user_reply_keeps_whole_batch, ()),
        ("child", cases.test_native_children_survive_ptl, (1,)),
        ("twelve_children", cases.test_native_children_survive_ptl, (12,)),
        ("capacity", cases.test_unseen_capacity_ends_without_summary_or_retry, ()),
        *[
            ("restart_" + cut, cases.test_restart_preserves_unseen_input_once, (cut,))
            for cut in ("planned", "receipt", "accepted")
        ],
    ]
    scenarios += [("provider_callback", cases.test_provider_callback_and_steer_keep_whole_group_after_ptl, (True,))]
    scenarios += [
        ("provider_" + delivery, cases.test_provider_callback_only_protects_projected_content, (delivery,))
        for delivery in ("correction", "audit")
    ]
    scenarios += [
        ("cancelled_" + reply, cases.test_cancelled_interaction_does_not_protect_audit_content, (reply,))
        for reply in ("user", "deny", "timeout")
    ]
    scenarios += [
        ("http_" + mode, cases.http_adapter_primary_coverage, (mode,)) for mode in ("normal", "ptl", "retry", "unknown")
    ]
    scenarios += [
        ("approval_" + decision, cases.test_approval_reason_keeps_original_group_after_ptl, (decision,))
        for decision in ("deny", "timeout")
    ]
    scenarios += [
        (f"snapshot_{reply}_{prepend}", cases.test_normal_callback_after_snapshot_reaches_next_primary, (reply, prepend))
        for reply in ("provider", "user", "deny", "timeout")
        for prepend in (False, True)
    ]
    for name, run, args in scenarios:
        with (
            TemporaryDirectory(prefix="unseen-input-") as directory,
            SQLiteStore.standalone(f"{directory}/session.sqlite") as raw_store,
        ):
            raw_store.install_schema()
            raw_store.connection.create_function("session_now_ms", 0, lambda: 1_000_000)
            store = wrap_fixture_store(raw_store, fixture_key_provider())
            with _capture_append_groups() as groups:
                extra = run(store, Path(raw_store.path), *args)

            states = {}
            for session_id in store.list_sessions():
                _state, records, _ = store.read_state(session_id)
                states[session_id] = hydrate_and_fold_records(
                    records,
                    store.fold_factory,
                    hydration_for_session_store(store, session_id),
                )
                states[session_id] = (records, states[session_id])

            terminals: dict[str, tuple[str, int]] = {}
            for session_id in sorted(set(states) - {"s"}):
                records, hydrated = states[session_id]
                _, child_output = _history_projection(store, session_id, records, hydrated.hydrated_records, groups, {})
                terminal = next(record for record in child_output if record["kind"] == "turn_ended")
                terminals[session_id] = (terminal["_digest"], child_output.index(terminal) + 1)

            records, hydrated = states["s"]
            prepared, output = _history_projection(
                store,
                "s",
                records,
                hydrated.hydrated_records,
                groups,
                terminals,
            )
            plans = [
                record for record in prepared if record["kind"] == "op_planned" and record["payload"]["purpose"] == "primary"
            ]
            compactions = [record["payload"] for record in prepared if record["kind"] == "context_compacted"]
            final_context = _replace_child_results(
                [message.to_dict() for message in project_context(hydrated.hydrated_records, hydrated.state)],
                terminals,
            )
            result.append(
                {
                    **(extra or {}),
                    "case": name,
                    "creation": {key: value for key, value in prepared[0].items() if key != "_digest"},
                    "record_digests": [record["_digest"] for record in output],
                    "last_primary_request": plans[-1]["payload"]["request"] if plans else None,
                    "compactions": compactions,
                    "input_ids": list(hydrated.state.applied_inputs),
                    "terminal_reasons": [record["payload"]["reason"] for record in prepared if record["kind"] == "turn_ended"],
                    "protected_start": protected_start(hydrated.hydrated_records, hydrated.state),
                    "final_context": final_context,
                }
            )
    return {
        "historical_contract_version": "39.0.0",
        "historical_record_schema_version": 1,
        "historical_inbox_schema_version": 3,
        "producer": "scripts/unseen_input_fixtures.py",
        "cases": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + "\n")
