"""Smoke the current S1 fixture-author base harness through real producers."""

from __future__ import annotations

import runpy
from pathlib import Path

from vv_agent.events import ModelCallCompletedEvent, ModelCallFailedEvent
from vv_agent.kernel._content import (
    INPUT_ACTOR_ATTRIBUTES,
    INPUT_CONTENT,
    SESSION_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
)
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store, project_records
from vv_agent.session.result import project_result
from vv_agent.types import AgentStatus, CompletionReason

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "session_kernel_fixtures.py"
AUTHOR = runpy.run_path(str(SCRIPT))


def test_fixture_author_native_finish_uses_real_s1_store_and_wire() -> None:
    fixtures = AUTHOR["Fixtures"]()
    session_id = "fixture-author-native-finish"
    answer = "fixture-author-finished"
    try:
        fixtures.admit(
            session_id,
            [AUTHOR["LLMResponse"](answer, raw={"usage": AUTHOR["USAGE"]})],
        )
        fixtures.run(session_id)
        streams = fixtures.collect()
        state = fixtures.state(session_id)
        result = fixtures.result(session_id)

        assert len(streams) == 1
        collected_session_id, rows = streams[0]
        assert collected_session_id == session_id
        assert [row.record.kind for row in rows] == [
            "session_created",
            "input_applied",
            "turn_started",
            "usage_observed",
            "op_planned",
            "usage_observed",
            "op_started",
            "usage_observed",
            "op_completed",
            "usage_observed",
            "turn_ended",
        ]
        assert state.active_turn_id is None
        assert result.status is AgentStatus.COMPLETED
        assert result.completion_reason is CompletionReason.NO_TOOL_FINISH
        assert result.final_output == answer
        assert result.token_usage.total_tokens == 15

        raw_store = fixtures.kernel._root_store
        assert raw_store._one("SELECT storage_major,write_level FROM sk_settings") == ("S1", 1)
        assert raw_store._rows("SELECT DISTINCT schema_version FROM sk_session") == [("S1",)]
        assert raw_store._one(
            "SELECT provider_id,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            ("default", session_id),
        ) == ("local-development-v1", "active")
        assert (
            raw_store._one(
                "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s "
                "AND state='live' AND ciphertext IS NOT NULL",
                ("default", session_id),
            )[0]
            > 0
        )

        bodies = [
            body
            for (body,) in raw_store._rows(
                "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
                ("default", session_id),
            )
        ]
        assert [row.record.encode() for row in rows] == bodies
        assert answer.encode() not in b"".join(bodies)
        assert b"kernel-active-scope" not in b"".join(bodies)

        item = fixtures.inbox[(session_id, "initial")]
        assert declared_content_ref(item.payload["content"]) is not None
        assert item.actor is not None and item.actor["attributes"]["content_id"]
        inbox_body = raw_store._one(
            "SELECT body FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
            ("default", session_id, item.input_id),
        )[0]
        assert item.encode() == inbox_body

        hydration = hydration_for_session_store(fixtures.kernel.store, session_id)
        usage_rows = [row for row in rows if row.record.kind == "usage_observed"]
        usage_payloads = [hydrate_record_payload(row, hydration) for row in usage_rows]
        assert [payload["observation"] for payload in usage_payloads] == [1, 2, 3, 4]
        assert all(payload["source"] == "kernel-active-scope" for payload in usage_payloads)

        links = raw_store._rows(
            "SELECT record_seq,input_id,slot,content_id FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
            ("default", session_id),
        )
        input_slots = {slot for _seq, input_id, slot, _content_id in links if input_id == "initial"}
        assert input_slots == {INPUT_ACTOR_ATTRIBUTES, INPUT_CONTENT}
        created_slots = {slot for seq, _input_id, slot, _content_id in links if seq == 1}
        assert created_slots == {SESSION_ATTRIBUTES, SESSION_PRINCIPAL_ATTRIBUTES, SESSION_WORKSPACE}
        links_by_record: dict[int, set[str]] = {}
        for seq, input_id, slot, _content_id in links:
            if input_id is None and seq is not None:
                links_by_record.setdefault(seq, set()).add(slot)
        for usage in usage_rows:
            assert {"usage", "usage_source"} <= links_by_record[usage.seq]
    finally:
        fixtures.kernel.close()


def test_fixture_author_provider_usage_hydrates_raw_s1_once() -> None:
    fixtures = AUTHOR["Fixtures"]()
    try:
        fixtures.provider_usage()
        assert [(case["case"], case["context"]) for case in fixtures.semantics] == [
            ("late_usage_normal", "normal"),
            ("late_usage_audit", "audit"),
        ]
        streams = dict(fixtures.collect())
        raw_store = fixtures.kernel._root_store
        usage = AUTHOR["USAGE"]
        for session_id, expected_output, expected_context in (
            ("late_usage_normal", "first", "normal"),
            ("late_usage_audit", "second", "audit"),
        ):
            snapshot = fixtures.kernel._read_state(session_id)
            state, rows, _ = snapshot
            result = fixtures.result(session_id)
            assert result.final_output == expected_output
            assert state.applied_inputs["replay"].payload["disposition"] == "noop"

            completions = [row for row in rows if row.record.kind == "op_completed" and row.record.attempt == 1]
            assert len(completions) == 1
            raw_completion_row = completions[0]
            raw_completion = raw_completion_row.record
            assert declared_content_ref(raw_completion.payload["usage"]) is not None
            assert b"prompt_tokens" not in raw_completion.encode()

            hydration = hydration_for_session_store(fixtures.kernel.store, session_id)
            completion = hydrate_record_payload(raw_completion_row, hydration)
            assert completion["context"] == expected_context
            assert completion["usage"] == usage
            assert AUTHOR["independent_bytes"]([completion["usage"]])[0] == AUTHOR["independent_bytes"]([usage])[0]
            events = [
                event
                for event in project_records(rows, hydration=hydration)
                if isinstance(event, ModelCallCompletedEvent) and event.attempt == 1
            ]
            attempt_one_calls = [call for call in result.token_usage.model_calls if call.attempt == 1]
            assert len(events) == len(attempt_one_calls) == 1
            assert events[0].usage == attempt_one_calls[0].usage
            assert attempt_one_calls[0].usage.usage_source.value == "provider_reported"

            db_bodies = [
                body
                for (body,) in raw_store._rows(
                    "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
                    ("default", session_id),
                )
            ]
            assert [row.record.encode() for row in streams.pop(session_id)] == db_bodies
        assert not streams
    finally:
        fixtures.kernel.close()


def test_fixture_author_endpoint_routing_uses_hydrated_s1_plans() -> None:
    fixtures = AUTHOR["Fixtures"]()
    session_id = "endpoint-routing"
    try:
        fixtures.endpoint_routing()

        assert len(fixtures.semantics) == 1
        case = fixtures.semantics[0]
        assert case["case"] == "endpoint_routing"
        assert case["session_id"] == session_id
        assert case["calls"] == ["B", "B"]
        assert case["preferences"] == [None, "B", "B"]
        raw_store = fixtures.kernel._root_store
        _, rows, _ = fixtures.kernel._read_state(session_id)
        hydration = hydration_for_session_store(fixtures.kernel.store, session_id)
        hydrated = [(row, hydrate_record_payload(row, hydration)) for row in rows]

        applied = next(
            payload
            for row, payload in hydrated
            if row.record.kind == "input_applied" and payload["input"]["input_id"] == "select"
        )
        assert applied["disposition"] == "applied"
        assert applied["model_binding"]["model"] == "selected"

        selected_plans = [
            (row, payload)
            for row, payload in hydrated
            if row.record.kind == "op_planned" and payload["op_kind"] == "model" and payload["request"]["model"] == "selected"
        ]
        assert selected_plans
        selected_request = selected_plans[0][1]["request"]
        assert selected_request["metadata"]["vv_session"]["endpoint_order"] == ["A", "B"]
        assert selected_request["metadata"]["vv_session"]["model_binding"]["model"] == "selected"
        retry_request = selected_plans[-1][1]["request"]
        assert retry_request["metadata"]["vv_session"]["endpoint_order"] == ["B"]

        raw_plan = selected_plans[0][0].record
        assert declared_content_ref(raw_plan.payload["request"]) is not None
        unavailable = next(
            (row, payload)
            for row, payload in hydrated
            if row.record.kind == "op_completed" and payload["result"].get("error_code") == "endpoint_unavailable"
        )
        assert case["missing"]["record_id"] == unavailable[0].record.record_id
        assert unavailable[1]["execution_started"] is False
        assert unavailable[1]["usage"] == {}
        assert unavailable[1]["evidence"] == []
        assert not any(
            row.record.kind == "op_unknown"
            and row.record.operation_id == unavailable[0].record.operation_id
            and row.record.attempt == unavailable[0].record.attempt
            for row, _ in hydrated
        )
        terminal = next(
            payload
            for row, payload in hydrated
            if row.record.kind == "turn_ended" and row.record.turn_id == f"{session_id}/turn/followup"
        )
        assert terminal["status"] == "completed" and terminal["result"] == "done"

        raw_bodies = [
            body
            for (body,) in raw_store._rows(
                "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
                ("default", session_id),
            )
        ]
        joined = b"".join(raw_bodies)
        assert b'"model":"selected"' not in joined
        assert b'"endpoint_order":["A","B"]' not in joined
        assert b'"done"' not in joined
    finally:
        fixtures.kernel.close()


def test_fixture_author_model_selection_hydrates_one_s1_snapshot() -> None:
    fixtures = AUTHOR["Fixtures"]()
    session_id = "model_selection"
    try:
        fixtures.model_selection()

        assert len(fixtures.semantics) == 1
        case = fixtures.semantics[0]
        assert case["case"] == "model_selection_mid_session_retry"
        assert case["session_id"] == session_id
        assert case["models"] == ["m", "m", "B"]
        assert case["billable_models"] == ["m", "m", "B"]
        assert case["retry_request_unchanged"] is True
        assert len(fixtures.invalid) == 1
        assert fixtures.invalid[0]["rejection_class"] == "model_selection_input_id_conflict"
        assert fixtures.invalid[0]["session_id"] == session_id

        raw_store = fixtures.kernel._root_store
        snapshot = fixtures.kernel._read_state(session_id)
        state, rows, _ = snapshot
        assert state.active_turn_id is None
        hydration = hydration_for_session_store(fixtures.kernel.store, session_id)
        hydrated = [(row, hydrate_record_payload(row, hydration)) for row in rows]
        plans = [
            (row, payload) for row, payload in hydrated if row.record.kind == "op_planned" and payload["purpose"] == "primary"
        ]
        assert [payload["request"]["model"] for _, payload in plans] == ["m", "m", "B"]
        assert plans[0][1]["request"] == plans[1][1]["request"]
        assert plans[2][1]["request"]["route"] == {"backend_type": "anthropic"}
        assert plans[2][1]["request"]["model_settings"] == {
            "reasoning": {"effort": "high"},
            "max_tokens": 1024,
            "retry": {"max_attempts": 1, "backoff_seconds": 0},
        }
        assert all(declared_content_ref(row.record.payload["request"]) is not None for row, _ in plans)
        assert all(row.record.feature_level == 1 for row, _ in plans)

        selection_row, selection_payload = next(
            (row, payload)
            for row, payload in hydrated
            if row.record.kind == "input_applied" and payload["input"]["input_id"] == "select-B"
        )
        selection = {
            "model": "B",
            "model_settings": {"reasoning": {"effort": "high"}, "max_tokens": 1024},
            "route": {"backend_type": "anthropic"},
        }
        assert selection_row.record.feature_level == 1
        assert selection_payload["input"]["payload"] == {"selection": selection}
        assert selection_payload["model_binding"]["model"] == "B"
        assert declared_content_ref(selection_row.record.payload["input"]["payload"]["selection"]) is not None
        assert declared_content_ref(case["selection_record"]["payload"]["input"]["payload"]["selection"]) is not None

        model_calls = [
            event
            for event in project_records(rows, hydration=hydration)
            if isinstance(event, (ModelCallCompletedEvent, ModelCallFailedEvent))
        ]
        assert [event.model for event in model_calls] == ["m", "m", "B"]
        assert [event.backend for event in model_calls] == ["scripted", "scripted", "anthropic"]
        ledger_calls = project_result(
            fixtures.kernel.store,
            session_id,
            next(reversed(state.turns)),
            runtime=fixtures.runtimes[session_id],
            snapshot=snapshot,
        ).token_usage.model_calls
        assert case["calls"] == [call.to_dict() for call in ledger_calls]

        stored_selection = fixtures.inbox[(session_id, "select-B")]
        assert declared_content_ref(stored_selection.payload["selection"]) is not None
        raw_inbox = raw_store._one(
            "SELECT body FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
            ("default", session_id, "select-B"),
        )[0]
        raw_bodies = [
            body
            for (body,) in raw_store._rows(
                "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
                ("default", session_id),
            )
        ]
        raw_records = b"".join(raw_bodies)
        assert raw_inbox == stored_selection.encode()
        assert [row.record.encode() for row in rows] == raw_bodies
        for raw in (raw_inbox, raw_records):
            assert b'"model":"B"' not in raw
            assert b'"backend_type":"anthropic"' not in raw
            assert b'"effort":"high"' not in raw
            assert b'"B done"' not in raw
    finally:
        fixtures.kernel.close()
