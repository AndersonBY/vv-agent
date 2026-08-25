from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.run_adapter import RunAdapter, TurnResumeError
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadStore
from vv_agent.runtime.checkpoint_codec import checkpoint_from_dict
from vv_agent.runtime.controller import HostInteractionAdmissionContext, HostInteractionRequest
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"


def _checkpoint(key: str) -> Any:
    fixture = json.loads((FIXTURE_DIR / "checkpoint_codec.json").read_text(encoding="utf-8"))
    payload = next(case["payload"] for case in fixture["valid_cases"] if case["name"] == "minimal_running")
    payload = dict(payload)
    payload["checkpoint_key"] = key
    return checkpoint_from_dict(payload)


@pytest.fixture(params=["memory", "sqlite"])
def controller_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        return InMemoryCheckpointStore()
    return SqliteCheckpointStore(tmp_path / "app-server-controller.sqlite3")


def test_turn_action_same_id_replays_and_conflicting_payload_is_zero_write(controller_store: Any) -> None:
    checkpoint_key = "app-server-controller"
    checkpoint = _checkpoint(checkpoint_key)
    assert controller_store.create_checkpoint(checkpoint)
    claimed = controller_store.claim_checkpoint(
        checkpoint_key,
        1,
        claim_token="worker-claim",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    controller_store.produce_host_interaction(
        HostInteractionRequest(
            interaction_id="interaction-1",
            logical_cycle=1,
            operation_id="operation-1",
            tool_call_id="tool-1",
            prompt="Approve the operation",
        ),
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=checkpoint_key,
            claim_token="worker-claim",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )

    thread_store = ThreadStore()
    thread = thread_store.create_thread(agent_key="default")
    turn = thread_store.create_turn(
        thread_id=thread.thread_id,
        input=[{"type": "text", "text": "wait"}],
        status="interrupted",
    )
    thread_store.update_turn(
        turn.turn_id,
        status="interrupted",
        run_id=checkpoint.root_run_id,
        result={"checkpoint": {"key": checkpoint_key}},
    )
    adapter = RunAdapter(
        host=DefaultAppServerHost(),
        store=thread_store,
        state_manager=ThreadStateManager(),
        router=OutgoingRouter(),
    )
    adapter._active_checkpoint_stores[(thread.thread_id, turn.turn_id)] = controller_store
    adapter._active_checkpoint_keys[(thread.thread_id, turn.turn_id)] = checkpoint_key
    public_status = adapter.public_thread_status(thread.thread_id)
    assert public_status["waitReason"] == "host_interaction"
    assert public_status["prompt"] == "Approve the operation"
    action = {
        "kind": "respond",
        "message": {"role": "user", "content": "Approve sk-test-123 at https://example.invalid/run?secret=abc"},
    }

    first = adapter.controller_action(
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        action_id="action-1",
        action=action,
    )
    revision_after_first = controller_store.load_checkpoint(checkpoint_key).revision
    replay = adapter.controller_action(
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        action_id="action-1",
        action=action,
    )
    assert replay == first

    with pytest.raises(TurnResumeError, match="different action payload"):
        adapter.controller_action(
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
            action_id="action-1",
            action={"kind": "respond", "message": {"role": "user", "content": "different"}},
        )
    assert controller_store.load_checkpoint(checkpoint_key).revision == revision_after_first

    with pytest.raises(TurnResumeError, match="public schema"):
        adapter.controller_action(
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
            action_id="action-1",
            action={
                "kind": "respond",
                "message": {"role": "user", "content": "continue"},
                "commandId": "client-controlled",
            },
        )
    assert controller_store.load_checkpoint(checkpoint_key).revision == revision_after_first


def test_public_host_prompt_never_falls_back_to_checkpoint_request(controller_store: Any) -> None:
    checkpoint_key = "app-server-missing-notification"
    checkpoint = _checkpoint(checkpoint_key)
    assert controller_store.create_checkpoint(checkpoint)
    claimed = controller_store.claim_checkpoint(
        checkpoint_key,
        1,
        claim_token="worker-claim",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    outcome = controller_store.produce_host_interaction(
        HostInteractionRequest(
            interaction_id="interaction-missing-notification",
            logical_cycle=1,
            operation_id="operation-missing-notification",
            tool_call_id="tool-missing-notification",
            prompt="Checkpoint-only prompt",
        ),
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=checkpoint_key,
            claim_token="worker-claim",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    thread_store = ThreadStore()
    thread = thread_store.create_thread(agent_key="default")
    turn = thread_store.create_turn(thread_id=thread.thread_id, input=[{"type": "text", "text": "wait"}], status="interrupted")
    thread_store.update_turn(
        turn.turn_id,
        status="interrupted",
        run_id=checkpoint.root_run_id,
        result={"checkpoint": {"key": checkpoint_key}},
    )
    adapter = RunAdapter(
        host=DefaultAppServerHost(),
        store=thread_store,
        state_manager=ThreadStateManager(),
        router=OutgoingRouter(),
    )
    adapter._active_checkpoint_stores[(thread.thread_id, turn.turn_id)] = controller_store
    adapter._active_checkpoint_keys[(thread.thread_id, turn.turn_id)] = checkpoint_key
    if isinstance(controller_store, InMemoryCheckpointStore):
        controller_store._host_interaction_notifications.pop(outcome.notification_id)  # type: ignore[attr-defined]
    else:
        controller_store._conn.execute(  # type: ignore[attr-defined]
            "DELETE FROM host_interaction_notification_outbox WHERE notification_id = ?", (outcome.notification_id,)
        )
        controller_store._conn.commit()  # type: ignore[attr-defined]
    status = adapter.public_thread_status(thread.thread_id)
    assert status["waitReason"] == "host_interaction"
    assert "prompt" not in status


def test_suspended_host_response_replay_keeps_prompt_private(controller_store: Any) -> None:
    checkpoint_key = "app-server-suspended-host"
    checkpoint = _checkpoint(checkpoint_key)
    assert controller_store.create_checkpoint(checkpoint)
    claimed = controller_store.claim_checkpoint(
        checkpoint_key,
        1,
        claim_token="worker-claim",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    outcome = controller_store.produce_host_interaction(
        HostInteractionRequest(
            interaction_id="suspended-interaction",
            logical_cycle=1,
            operation_id="suspended-operation",
            tool_call_id="suspended-tool",
            prompt="Approve sk-private-123 at https://example.invalid/private",
        ),
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=checkpoint_key,
            claim_token="worker-claim",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    del outcome
    thread_store = ThreadStore()
    thread = thread_store.create_thread(agent_key="default")
    turn = thread_store.create_turn(thread_id=thread.thread_id, input=[{"type": "text", "text": "wait"}], status="interrupted")
    thread_store.update_turn(
        turn.turn_id,
        status="interrupted",
        run_id=checkpoint.root_run_id,
        result={"checkpoint": {"key": checkpoint_key}},
    )
    adapter = RunAdapter(
        host=DefaultAppServerHost(),
        store=thread_store,
        state_manager=ThreadStateManager(),
        router=OutgoingRouter(),
    )
    adapter._active_checkpoint_stores[(thread.thread_id, turn.turn_id)] = controller_store
    adapter._active_checkpoint_keys[(thread.thread_id, turn.turn_id)] = checkpoint_key

    suspended = adapter.controller_action(
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        action_id="suspend-1",
        action={"kind": "suspend"},
    )
    assert suspended["waitReason"] == "suspended"
    status = adapter.public_thread_status(thread.thread_id)
    assert status["waitReason"] == "suspended"
    assert "prompt" not in status

    action = {
        "kind": "respond",
        "message": {"role": "user", "content": "Continue sk-response-456 at https://example.invalid/next"},
    }
    response = adapter.controller_action(
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        action_id="respond-1",
        action=action,
    )
    assert response["waitReason"] == "suspended"
    assert "prompt" not in response
    assert (
        adapter.controller_action(
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
            action_id="respond-1",
            action=action,
        )
        == response
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_id", "😀" * 200),
        ("thread_id", "😀" * 200),
        ("turn_id", "😀" * 200),
    ],
)
def test_turn_action_rejects_identity_over_utf8_limit_without_store_access(
    field: str,
    value: str,
) -> None:
    adapter = RunAdapter(
        host=DefaultAppServerHost(),
        store=ThreadStore(),
        state_manager=ThreadStateManager(),
        router=OutgoingRouter(),
    )
    kwargs: dict[str, Any] = {
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "action_id": "action-1",
        "action": {"kind": "cancel"},
    }
    kwargs[field] = value
    with pytest.raises(TurnResumeError, match="UTF-8 byte limit"):
        adapter.controller_action(**kwargs)


def test_turn_action_rejects_message_unknown_fields_before_durable_lookup() -> None:
    adapter = RunAdapter(
        host=DefaultAppServerHost(),
        store=ThreadStore(),
        state_manager=ThreadStateManager(),
        router=OutgoingRouter(),
    )
    with pytest.raises(TurnResumeError, match="respond message"):
        adapter.controller_action(
            thread_id="thread-1",
            turn_id="turn-1",
            action_id="action-1",
            action={"kind": "respond", "message": {"role": "user", "content": "ok", "secret": "x"}},
        )
