"""Pure summary intent and pending-operation selection."""

from __future__ import annotations

from ._boundary_hydration import hydrate_boundary_data, hydrate_input
from ._content_publication import declared_content_ref, input_content_slots
from ._driver import CoreDriver
from ._records import InboxItem, copy_json


def compaction_request(item: InboxItem) -> dict | None:
    content = item.payload.get("content")
    if item.kind != "user" or not isinstance(content, dict):
        return None
    metadata = content.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("request_compaction") is not True:
        return None
    request = content.get("app_server_compaction")
    return request if isinstance(request, dict) else None


def _logical_input(driver: CoreDriver, item: InboxItem) -> InboxItem:
    if driver.content_reader is None:
        return item
    for slot in input_content_slots(item.kind):
        value = item.payload
        try:
            for part in slot.path:
                value = value[int(part)] if part.isdecimal() else value[part]
        except (IndexError, KeyError):
            continue
        if declared_content_ref(value) is not None:
            return hydrate_input(driver, item)
    return item


def _completed_result(driver: CoreDriver, record) -> dict:
    retained_result = record._payload["result"]
    result = (
        driver.hydrate_record_field(record, "result", retained_result) if driver.content_reader is not None else retained_result
    )
    if type(result) is not dict:
        raise ValueError("completed result must be an object")
    return result


def _boundary_data(driver: CoreDriver, record) -> dict:
    data = record._payload["data"]
    event = data.get("event") if isinstance(data, dict) else None
    if driver.content_reader is None:
        return data
    if isinstance(event, dict) and event.get("version") == "v6":
        return copy_json(data)
    return hydrate_boundary_data(driver, record)


def pending_compaction_requests(driver: CoreDriver) -> list[str]:
    consumed = {
        oid
        for (tid, stage, _), record in driver.state.boundaries.items()
        if tid == driver.state.active_turn_id and stage == "memory_completed"
        for oid in _boundary_data(driver, record)["event"].get("metadata", {}).get("requested_by", [])
    }
    controls = []
    for key, record in driver.state.applied_inputs.items():
        if not (
            record.turn_id == driver.state.active_turn_id
            or (
                driver.state.active_turn_id is not None
                and key in driver.state.turns[driver.state.active_turn_id].start._payload["input_ids"]
            )
        ):
            continue
        item = _logical_input(driver, InboxItem(**record._payload["input"]))
        if (
            record._payload["disposition"] in {"applied", "queued"}
            and compaction_request(item) is not None
            and key not in consumed
        ):
            controls.append(key)
    provider_controls = []
    for oid, op in driver.state.operations.items():
        if (
            op.turn_id != driver.state.active_turn_id
            or op.kind == "model"
            or op.selected_attempt is None
            or op.attempts[op.selected_attempt].context != "normal"
        ):
            continue
        receipt = op.attempts[op.selected_attempt].result
        if receipt is None:
            continue
        result = _completed_result(driver, receipt)
        if (
            result.get("status_code") == "SUCCESS"
            and result.get("metadata", {}).get("request_compaction") is True
            and oid not in consumed
        ):
            provider_controls.append(oid)
    return controls + provider_controls


def summary_pending(driver: CoreDriver, oid: str) -> bool:
    op = driver.state.operations.get(f"{oid}/summary")
    if op is None or op.selected_attempt is not None:
        return False
    attempt = op.attempts[max(op.attempts)]
    return attempt.state in {"planned", "started", "parked"} or bool(
        attempt.unknown and attempt.unknown._payload["retry"] == "retry"
    )
