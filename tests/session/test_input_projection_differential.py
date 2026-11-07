"""Fixed-seed differential projection cases, including in-flight snapshots."""

import gzip
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.session.context import project_context, protected_start
from vv_agent.session.records import InboxItem, Record, SessionSpec, digest, make_record
from vv_agent.session.reducer import Fold
from vv_agent.types import Message, ToolCall, ToolExecutionResult

from .fold_history import growing_history
from .helpers import record
from .input_projection_reference import reference


def generate(seed):
    rng = random.Random(seed)
    rows = []
    plans = {}
    counter = 0

    def add(r):
        rows.append(r)

    def steer():
        nonlocal counter
        counter += 1
        item = InboxItem(f"random-{counter}", "steer", {"content": f"INPUT-{seed}-{counter}"}, "t", 1)
        add(record("input_applied", input=item.to_dict(), input_digest=item.digest))

    initial, _ = growing_history(rng.randrange(2, 7), initial_bytes=32)
    # Reused provider call IDs must not alias distinct source groups.
    if seed % 4 == 0:
        changed = []
        for r in initial:
            p = r.payload
            if r.kind == "op_completed" and "/tool/" not in r.operation_id:
                for call in p["result"].get("tool_calls", []):
                    call["id"] = "reused-call-id"
                p["result_digest"] = digest(p["result"])
                r = replace(r, payload=p)
            elif r.kind == "op_planned" and "/tool/" in r.operation_id:
                p["request"]["id"] = "reused-call-id"
                p["request_digest"] = digest(p["request"])
                r = replace(r, payload=p)
            elif r.kind == "op_completed" and "/tool/" in r.operation_id:
                p["result"]["tool_call_id"] = "reused-call-id"
                p["result_digest"] = digest(p["result"])
                r = replace(r, payload=p)
            changed.append(r)
        initial = changed
    add(SessionSpec("s", "test", "/tmp").record())
    for r in initial:
        p = r.payload
        if r.kind == "boundary_recorded":
            for _ in range(rng.randrange(3)):
                steer()
            source = [Message.from_dict(m) for m in reference(rows)[1]]
            updated = list(source)
            if rng.randrange(3) == 0:
                updated = [Message("system", f"HOST-{seed}-{len(rows)}"), *updated]
            if rng.randrange(5) == 0:
                updated += [Message("assistant", f"HOST-APPEND-{seed}-{len(rows)}")]
            p["source_digest"] = digest([m.to_dict() for m in source])
            p["data"]["messages"] = [m.to_dict() for m in updated]
            r = replace(r, payload=p)
        elif r.kind == "op_planned":
            if p["op_kind"] == "model":
                p["request"]["messages"] = [m.to_dict() for m in [Message.from_dict(m) for m in reference(rows)[1]]]
                p["request_digest"] = digest(p["request"])
                p["context_version"] = str(len(rows))
                r = replace(r, payload=p)
            plans[r.operation_id] = r
        elif r.kind == "op_started" and "/tool/" not in r.operation_id:
            add(r)
            for _ in range(rng.randrange(3)):
                steer()
            continue
        elif r.kind == "op_completed":
            p["request_digest"] = plans[r.operation_id].payload["request_digest"]
            r = replace(r, payload=p)
            if "/tool/" in r.operation_id and rng.randrange(2) == 0:
                if rng.randrange(2) == 0:
                    source = [Message.from_dict(m) for m in reference(rows)[1]]
                    replacement = ([Message("system", f"IN-FLIGHT-HOST-{seed}-{len(rows)}")] if rng.randrange(2) else []) + source
                    add(
                        make_record(
                            "boundary_recorded",
                            session_id="s",
                            turn_id="t",
                            payload={
                                "stage": "before_memory",
                                "boundary_id": f"inflight-{len(rows)}",
                                "source_operation_id": None,
                                "source_digest": digest([m.to_dict() for m in source]),
                                "data": {"messages": [m.to_dict() for m in replacement], "shared_state": {}},
                            },
                        )
                    )
                plan = plans[r.operation_id]
                item = InboxItem(
                    f"callback-{len(rows)}",
                    "provider_result",
                    {
                        "operation_id": r.operation_id,
                        "attempt": 1,
                        "request_digest": plan.payload["request_digest"],
                        "provider_binding": plan.payload["provider_binding"],
                        "result": p["result"],
                        "usage": {},
                        "evidence": ["proof"],
                    },
                    "t",
                    1,
                )
                add(record("input_applied", input=item.to_dict(), input_digest=item.digest, target_operation_id=r.operation_id))
        add(r)
        if r.kind == "turn_started":
            item = InboxItem(f"select-{seed}", "model_selection", {"selection": None})
            add(replace(record("input_applied", input=item.to_dict(), input_digest=item.digest), turn_id=None))
            item = InboxItem(f"follow-{seed}", "follow_up", {"content": "queued next turn"}, "t", 1)
            add(record("input_applied", input=item.to_dict(), input_digest=item.digest, disposition="queued"))
            for action in ("suspend", "resume"):
                item = InboxItem(f"{action}-{seed}", "control", {"action": action}, "t", 1)
                add(record("input_applied", input=item.to_dict(), input_digest=item.digest))
        elif r.kind == "op_started" and "/tool/" in r.operation_id and rng.randrange(2) == 0:
            plan = plans[r.operation_id]
            checksum = plan.payload["request_digest"]
            handle = {
                "kind": "provider",
                "provider": plan.payload["provider_binding"],
                "job_id": "job",
                "operation_id": r.operation_id,
                "attempt": 1,
                "request_digest": checksum,
                "evidence": "proof",
                "query_ref": "query",
                "cancel_ref": None,
            }
            item = InboxItem(
                f"evidence-{len(rows)}",
                "provider_evidence",
                {"operation_id": r.operation_id, "attempt": 1, "request_digest": checksum, "handle": handle},
                "t",
                1,
            )
            add(record("input_applied", input=item.to_dict(), input_digest=item.digest, target_operation_id=r.operation_id))
    return rows


def check_prefixes(rows):
    fold = Fold()
    for i, row in enumerate(rows, 1):
        fold.extend([row], consumed_inputs=[InboxItem(**row.payload["input"])] if row.kind == "input_applied" else [])
        expected_start, expected = reference(rows[:i])
        assert protected_start(fold.history, fold.state) == expected_start, (i, row.kind)
        assert fold.state._protection is not None
        assert [m.to_dict() for m in fold.state._protection.messages] == expected, (i, row.kind, "cache")
        assert [m.to_dict() for m in project_context(fold.history, fold.state)] == expected, (i, row.kind, "execution")
        cold = Fold()
        cold.extend(rows[:i], consumed_inputs=[InboxItem(**r.payload["input"]) for r in rows[:i] if r.kind == "input_applied"])
        assert protected_start(cold.history, cold.state) == expected_start, (i, row.kind, "cold start")
        assert [m.to_dict() for m in project_context(cold.history, cold.state)] == expected, (i, row.kind, "cold")


def image_callback_snapshot(native_multimodal, image_url, image_path):
    """A pending image tool, another tool result, then steer/snapshot/callback."""
    initial, _ = growing_history(1, initial_bytes=32)
    rows = [SessionSpec("s", "test", "/tmp").record(), *initial[:-1]]
    definition = rows[1].payload["definition"]
    definition["task"]["native_multimodal"] = native_multimodal
    rows[1] = replace(rows[1], payload=rows[1].payload | {"definition": definition, "definition_digest": digest(definition)})
    other = ToolCall("other-call", "read", {})
    completed = next(i for i, r in enumerate(rows) if r.kind == "op_completed")
    result = rows[completed].payload["result"]
    result["tool_calls"].append(other.to_dict())
    rows[completed] = replace(
        rows[completed], payload=rows[completed].payload | {"result": result, "result_digest": digest(result)}
    )
    parent = rows[completed].operation_id
    other_id = f"{parent}/attempt/1/tool/1"
    other_result = ToolExecutionResult(tool_call_id=other.id, content="other result").to_dict()
    rows += [
        record(
            "op_planned",
            oid=other_id,
            op_kind="tool",
            purpose=None,
            request=other.to_dict(),
            request_digest=digest(other.to_dict()),
            dependencies=[parent],
        ),
        record("op_started", oid=other_id),
        record(
            "op_completed",
            oid=other_id,
            request_digest=digest(other.to_dict()),
            result=other_result,
            result_digest=digest(other_result),
        ),
    ]
    steer = InboxItem("after-image-group", "steer", {"content": "STEER-AFTER-IMAGE-GROUP"}, "t", 1)
    rows.append(record("input_applied", input=steer.to_dict(), input_digest=steer.digest))
    source = reference(rows)[1]
    rows.append(
        make_record(
            "boundary_recorded",
            session_id="s",
            turn_id="t",
            payload={
                "stage": "before_memory",
                "boundary_id": "before-image-callback",
                "source_operation_id": None,
                "source_digest": digest(source),
                "data": {"messages": [Message("system", "HOST").to_dict(), *source], "shared_state": {}},
            },
        )
    )
    completion = initial[-1]
    result = replace(
        ToolExecutionResult.from_dict(completion.payload["result"]), image_url=image_url, image_path=image_path
    ).to_dict()
    completion = replace(completion, payload=completion.payload | {"result": result, "result_digest": digest(result)})
    plan = next(r for r in rows if r.kind == "op_planned" and r.operation_id == completion.operation_id)
    callback = InboxItem(
        "image-callback",
        "provider_result",
        {
            "operation_id": plan.operation_id,
            "attempt": 1,
            "request_digest": plan.payload["request_digest"],
            "provider_binding": plan.payload["provider_binding"],
            "result": result,
            "usage": {},
            "evidence": ["proof"],
        },
        "t",
        1,
    )
    return [
        *rows,
        record("input_applied", input=callback.to_dict(), input_digest=callback.digest, target_operation_id=plan.operation_id),
        completion,
    ]


@pytest.mark.parametrize("native_multimodal", [True, False])
@pytest.mark.parametrize(
    ("image_url", "image_path"),
    [
        ("https://example.invalid/image.png", None),
        ("data:image/png;base64,AAAA", "capture.png"),
        (None, "capture.png"),
    ],
    ids=["url-only", "data-with-path", "path-only"],
)
def test_image_callback_after_snapshot_shifts_steer(native_multimodal, image_url, image_path):
    rows = image_callback_snapshot(native_multimodal, image_url, image_path)
    check_prefixes(rows)
    fold = Fold()
    fold.extend(rows, consumed_inputs=[InboxItem(**r.payload["input"]) for r in rows if r.kind == "input_applied"])
    expected_start, expected = reference(rows)
    protection = fold.state._protection
    assert protection is not None
    assert expected_start == protection.start == 3
    assert [m["role"] for m in expected[3:6]] == ["assistant", "tool", "tool"]
    notices = expected[6:-1]
    assert notices == (
        [Message("user", f"[Image loaded] {image_path}" if image_path else "", image_url=image_url).to_dict()]
        if native_multimodal
        else []
    )
    steer_seq = next(i for i, r in enumerate(rows, 1) if r.kind == "input_applied" and r.payload["input"]["kind"] == "steer")
    before = reference(rows[:-2])[1]
    assert before[-1]["content"] == expected[-1]["content"] == "STEER-AFTER-IMAGE-GROUP"
    assert len(expected) - 1 == len(before) - 1 + int(native_multimodal)
    assert next(pos + protection.input_offset for pos, seq in protection.inputs if seq == steer_seq) == len(expected) - 1


@pytest.mark.parametrize("prepend", [False, True])
def test_review_minimal_prefixes(prepend):
    path = Path(__file__).parents[1] / "fixtures" / "unseen_inputs" / f"minimal-{prepend}.json"
    check_prefixes([Record.parse(s.encode()) for s in json.loads(path.read_text())])


@pytest.mark.parametrize("seed", range(30))
def test_random_record_zero_projection(seed):
    check_prefixes(generate(seed))
    cases = producer_corpus()
    check_prefixes(callback_snapshots(cases[seed % len(cases)], seed))


def test_execution_projection_cannot_mutate_committed_messages():
    rows = generate(1)
    fold = Fold()
    fold.extend(rows, consumed_inputs=[InboxItem(**r.payload["input"]) for r in rows if r.kind == "input_applied"])
    assert fold.state._protection is not None
    expected = [m.to_dict() for m in fold.state._protection.messages]
    exposed = project_context(fold.history, fold.state)
    exposed[0].content = "host mutation"
    assert [m.to_dict() for m in project_context(fold.history, fold.state)] == expected
    for i in range(1, len(rows)):
        assert [m.to_dict() for m in project_context(fold.history[:i], fold.state)] == reference(rows[:i])[1]


def test_replacement_retains_only_surviving_group_operations_without_protected_input():
    rows, _ = growing_history(2, initial_bytes=32)
    fold = Fold()
    fold.extend([SessionSpec("s", "test", "/tmp").record(), *rows])
    protection = fold.state._protection
    assert protection is not None and protection.start is None
    source = protection.messages
    first, second = protection.groups["t/model/1"], protection.groups["t/model/2"]
    replacement = [Message("system", "HOST"), *source[: first[1]], *source[second[1] :]]
    protection.replace(replacement)
    assert "t/model/1" not in protection.groups
    assert protection.groups["t/model/2"][1:] == (first[1] + 1, len(replacement))
    assert protection.start is None


def producer_corpus():
    path = Path(__file__).parents[1] / "fixtures" / "unseen_inputs" / "producer-corpus.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def callback_snapshots(case, seed):
    """Randomly insert committed snapshots before callbacks in real record logs."""
    rng = random.Random(seed)
    rows, versions, kinds = [], {0: 0}, {}
    for old_seq, row in enumerate((Record.parse(s.encode()) for s in case["records"]), 1):
        p = row.payload
        if (
            row.kind == "input_applied"
            and p["disposition"] == "applied"
            and rng.randrange(2) == 0
            and kinds.get(p["target_operation_id"]) in {"tool", "child"}
        ):
            messages = reference(rows)[1]
            rows.append(
                make_record(
                    "boundary_recorded",
                    session_id=row.session_id,
                    turn_id=row.turn_id,
                    payload={
                        "stage": "before_memory",
                        "boundary_id": f"random-callback-{old_seq}",
                        "source_operation_id": None,
                        "source_digest": digest(messages),
                        "data": {"messages": messages, "shared_state": {}},
                    },
                )
            )
        if row.kind == "op_planned":
            kinds[row.operation_id] = p["op_kind"]
            if p["op_kind"] == "model" and p["purpose"] == "primary":
                p["context_version"] = str(versions[int(p["context_version"])])
                row = replace(row, payload=p)
        rows.append(row)
        versions[old_seq] = len(rows)
    return rows


@pytest.mark.parametrize("case", producer_corpus(), ids=lambda case: case["test"].split("::")[-1])
def test_real_producer_record_zero_projection(case):
    check_prefixes([Record.parse(s.encode()) for s in case["records"]])
