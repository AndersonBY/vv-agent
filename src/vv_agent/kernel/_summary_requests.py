"""Pure summary intent and pending-operation selection."""

from __future__ import annotations

from ._driver import CoreDriver
from ._records import InboxItem


def compaction_request(item: InboxItem) -> dict | None:
    content = item.payload.get("content")
    if item.kind != "user" or not isinstance(content, dict):
        return None
    metadata = content.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("request_compaction") is not True:
        return None
    request = content.get("app_server_compaction")
    return request if isinstance(request, dict) else None


def pending_compaction_requests(driver: CoreDriver) -> list[str]:
    consumed = {
        oid
        for (tid, stage, _), record in driver.state.boundaries.items()
        if tid == driver.state.active_turn_id and stage == "memory_completed"
        for oid in record._payload["data"]["event"].get("metadata", {}).get("requested_by", [])
    }
    controls = [
        key
        for key, r in driver.state.applied_inputs.items()
        if (
            r.turn_id == driver.state.active_turn_id
            or (
                driver.state.active_turn_id is not None
                and key in driver.state.turns[driver.state.active_turn_id].start._payload["input_ids"]
            )
        )
        and r._payload["disposition"] in {"applied", "queued"}
        and compaction_request(InboxItem(**r._payload["input"])) is not None
        and key not in consumed
    ]
    return controls + [
        oid
        for oid, op in driver.state.operations.items()
        if op.turn_id == driver.state.active_turn_id
        and op.kind != "model"
        and op.selected_attempt is not None
        and op.attempts[op.selected_attempt].context == "normal"
        and (receipt := op.attempts[op.selected_attempt].result) is not None
        and receipt._payload["result"].get("status_code") == "SUCCESS"
        and receipt._payload["result"].get("metadata", {}).get("request_compaction") is True
        and oid not in consumed
    ]


def summary_pending(driver: CoreDriver, oid: str) -> bool:
    op = driver.state.operations.get(f"{oid}/summary")
    if op is None or op.selected_attempt is not None:
        return False
    attempt = op.attempts[max(op.attempts)]
    return attempt.state in {"planned", "started", "parked"} or bool(
        attempt.unknown and attempt.unknown._payload["retry"] == "retry"
    )
