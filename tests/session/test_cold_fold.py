"""Cold loads keep their validation and visit each retained snapshot a bounded number of times."""

import json
from contextlib import suppress
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest
from support.s1_fixture_values import append_fixture_records, clear_fixture_state_cache, read_fixture_state

from vv_agent.events import DiagnosticEvent
from vv_agent.session.context import project_context
from vv_agent.session.projection import project_records
from vv_agent.session.records import Record, digest
from vv_agent.session.reducer import Fold, TransitionError
from vv_agent.session.result import project_result
from vv_agent.types import AgentTask, Message

from .fold_history import (
    extend_logical,
    growing_history,
    logical_record,
    logical_session_record,
    parse_logical_record,
    persist_history,
)
from .helpers import create_fixture_session, wrap_fixture_store


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def snapshot_value(obj):
    if isinstance(obj, Record):
        return {"record_id": obj.record_id, "digest": obj._digest or digest(obj.to_dict())}
    if is_dataclass(obj):
        return {f.name: snapshot_value(getattr(obj, f.name)) for f in fields(obj) if f.name != "_protection"}
    if isinstance(obj, dict):
        return [[snapshot_value(k), snapshot_value(v)] for k, v in obj.items()]
    if isinstance(obj, list | tuple):
        return [snapshot_value(x) for x in obj]
    if isinstance(obj, set):
        return sorted(obj)
    return obj


@pytest.mark.parametrize("steps", [1, 4])
def test_cold_snapshot_matches_030_field_by_field(store, steps):
    fixtures = Path(__file__).with_name("fixtures")
    expected = json.loads((fixtures / "fold_030_state.json").read_text())[str(steps)]
    bodies = (fixtures / "fold_030_records.jsonl").read_bytes().splitlines()[: 2 + steps * 7]
    saved = tuple(Record(**json.loads(body)) for body in bodies)
    for body, record in zip(bodies, saved, strict=True):
        object.__setattr__(record, "_digest", sha256(body).hexdigest())
        if record.kind == "turn_started":
            definition = record.payload["definition"]
            object.__setattr__(record, "_task", lambda definition=definition: AgentTask.from_dict(definition["task"]))
    fold = Fold()
    fold.extend(saved, bodies=bodies)
    state = fold.state
    rows = tuple(fold.history)
    for f in fields(state):
        if f.name == "_protection":
            continue
        assert snapshot_value(getattr(state, f.name)) == expected[f.name], f.name
    assert list(saved) == [stored.record for stored in fold.history]
    assert [m.to_dict() for m in project_context(rows, state)] == expected["messages"]


@pytest.mark.parametrize("enabled", [False, True])
def test_fold_message_decodes_grow_with_stored_snapshots(monkeypatch, enabled):
    counts = []
    decode = Message.from_dict
    for steps in (32, 64):
        rows, _ = growing_history(steps, initial_bytes=256)
        calls = []

        def counted(value, calls=calls):
            calls.append(1)
            return decode(value)

        with monkeypatch.context() as patch:
            patch.setattr(Message, "from_dict", counted)
            fold = Fold()
            extend_logical(
                fold,
                (
                    logical_session_record(
                        attributes={"seed": {"unseen_input_protection": enabled}},
                    ),
                    *rows,
                ),
            )
        counts.append(len(calls))
    # Stored full conversations grow quadratically; replaying every obsolete snapshot is cubic.
    assert counts[1] <= 5 * counts[0]
    assert counts[1] < 5 * 64**2


def test_cold_record_shares_canonical_storage_bytes():
    body = logical_session_record(actor="p", workspace="w").encode()
    assert parse_logical_record(body).encode() is body
    spaced = json.dumps(json.loads(body), indent=2).encode()
    assert parse_logical_record(spaced).encode() == body


def test_earlier_snapshots_still_validate_their_source():
    rows, _ = growing_history(4, initial_bytes=256)
    boundary = rows[1]
    bad = replace(boundary, payload=boundary.payload | {"source_digest": "0" * 64})
    fold = Fold()
    with pytest.raises(TransitionError, match="boundary source mismatch"):
        extend_logical(fold, (logical_session_record(), rows[0], bad, *rows[2:]))


@pytest.mark.parametrize("enabled", [False, True])
def test_projected_events_remain_detached_from_records(store, enabled):
    persist_history(store, 2, initial_bytes=256, enabled=enabled)
    _, rows, _ = read_fixture_state(store, "s")
    before = [r.record.encode() for r in rows]
    projected = project_records(rows)
    response = next(e for e in projected if isinstance(e, DiagnosticEvent) and e.code == "cycle_llm_response")
    response.details["tool_calls"][0]["arguments"]["changed"] = True
    result = project_result(store, "s", "t")
    result.raw_result.shared_state["changed"] = True
    assert [r.record.encode() for r in rows] == before
    assert project_result(store, "s", "t").raw_result.shared_state == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_cold_prefix_bounds_fetches_and_shares_only_private_messages(store, monkeypatch, enabled):
    persist_history(store, 40, initial_bytes=256, enabled=enabled)
    clear_fixture_state_cache(store)
    root = store._root
    fetch = root._rows
    sizes = []

    def counted(query, params=()):
        rows = fetch(query, params)
        normalized = " ".join(query.split())
        if "FROM sk_record" in normalized and "body" in normalized:
            sizes.append(len(rows))
        return rows

    monkeypatch.setattr(root, "_rows", counted)
    _, rows, _ = read_fixture_state(store, "s")
    assert sizes
    assert max(sizes) <= 256, sizes
    plans = [r.record for r in rows if r.record.kind == "op_planned" and r.record._payload["op_kind"] == "model"]
    assert plans[0]._payload["request"]["messages"][1] is plans[-1]._payload["request"]["messages"][1]
    plans[-1].payload["request"]["messages"][1]["content"] = "changed"
    assert plans[0].payload["request"]["messages"][1]["content"] == "x" * 256


@pytest.mark.parametrize("enabled", [False, True])
def test_cold_prefix_rejects_a_gap_across_fetch_pages(store, enabled):
    from vv_agent.session.store import Conflict

    persist_history(store, 40, initial_bytes=256, enabled=enabled)
    clear_fixture_state_cache(store)
    with store.atomic():
        store._rows("DELETE FROM sk_record WHERE session_id=%s AND seq=257", ("s",))
    with pytest.raises(Conflict, match="log sequence gap"):
        store.read_state("s")


@pytest.mark.parametrize("enabled", [False, True])
def test_result_projection_does_not_scan_all_tools_for_each_model(store, enabled):
    persist_history(store, 24, initial_bytes=256, enabled=enabled)
    state, rows, watermark = read_fixture_state(store, "s")
    visits = []

    class CountedOperations(dict):
        def values(self):
            for value in super().values():
                visits.append(1)
                yield value

    state.operations = CountedOperations(state.operations)
    result = project_result(store, "s", "t", snapshot=(state, rows, watermark))
    assert len(result.raw_result.cycles) == 24
    assert all(len(c.tool_results) == 1 for c in result.raw_result.cycles)
    assert len(visits) <= 8 * len(state.operations)


@pytest.mark.parametrize("enabled", [False, True])
def test_raw_context_still_includes_the_full_history(store, enabled):
    messages = persist_history(store, 4, initial_bytes=256, enabled=enabled)
    state, rows, _ = read_fixture_state(store, "s")
    assert [m.to_dict() for m in project_context(rows, state, compacted=False)] == [m.to_dict() for m in messages]


@pytest.mark.parametrize("enabled", [False, True])
def test_fold_does_not_visit_or_copy_the_obsolete_record_prefix(enabled):
    class CountedHistory(list):
        visits = 0

        def __getitem__(self, key):
            if isinstance(key, slice):
                self.visits += len(range(*key.indices(len(self))))
            else:
                self.visits += 1
            return super().__getitem__(key)

        def __iter__(self):
            for value in super().__iter__():
                self.visits += 1
                yield value

    rows, _ = growing_history(64, initial_bytes=256)
    fold = Fold()
    history = CountedHistory()
    fold.history = history
    extend_logical(
        fold,
        (
            logical_session_record(attributes={"seed": {"unseen_input_protection": enabled}}),
            *rows,
        ),
    )
    assert history.visits <= 20 * 64


def test_parse_does_not_retain_a_mutable_input_buffer():
    body = logical_session_record(actor="p", workspace="w").encode()
    buffer = bytearray(body)
    parsed = parse_logical_record(buffer)
    buffer[:] = b"changed"
    assert parsed.encode() == body
    assert parsed.payload["principal"] == "p"


def test_cold_prefix_rejects_an_empty_captured_log(store):
    from vv_agent.session.store import Conflict

    create_fixture_session(store, "s", consumers=())
    with store.atomic():
        store._rows("UPDATE sk_session SET head_seq=0 WHERE session_id=%s", ("s",))
    clear_fixture_state_cache(store)
    with pytest.raises(Conflict):
        store.read_state("s")


def test_cached_source_digest_uses_the_normalized_message_bytes():
    from vv_agent.session.records import _MessageCache, digest

    cache = _MessageCache()
    wire = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": ' { "n": 1 } '}}],
        "metadata": {"unicode": "中文😀", "number": 1e-7, "nested": [True, None, 3]},
    }
    cached = cache.parse(cache.share(wire))
    source = [cached, Message("tool", "done", tool_call_id="a")]
    assert cache.source_digest(source) == digest([m.to_dict() for m in source])
    assert cache.source_digest([]) == digest([])
    assert cached.tool_calls is not None
    assert cached.tool_calls[0]["function"]["arguments"] == '{"n":1}'


def test_message_sharing_does_not_merge_json_scalar_types():
    from vv_agent.session.records import _MessageCache

    cache = _MessageCache()
    wires = [{"role": "user", "content": "same", "metadata": {"value": value}} for value in (True, 1, 1.0, "1", None)]
    shared = [cache.share(w) for w in wires]
    assert len({id(w) for w in shared}) == len(wires)
    assert cache.share({"metadata": {"value": True}, "content": "same", "role": "user"}) is shared[0]


def test_record_parser_keeps_jcs_numbers_unicode_and_parser_errors():
    from vv_agent.session.records import RecordError

    attributes = {
        "numbers": [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27, 5e-324, 1.7976931348623157e308, -0.0],
        "unicode": {"😀": "中文", "\ue000": "x"},
    }
    original = logical_session_record(actor="p", workspace="w", attributes=attributes)
    body = original.encode()
    assert parse_logical_record(body).to_dict() == original.to_dict()
    assert parse_logical_record(body).encode() == body
    assert parse_logical_record(b"\xef\xbb\xbf" + body).encode() == body
    with pytest.raises(RecordError, match="duplicate field: schema_version"):
        parse_logical_record(body.replace(b'"schema_version":"S1"', b'"schema_version":"S1","schema_version":"S1"'))


@pytest.mark.parametrize("enabled", [False, True])
def test_message_cache_does_not_encode_a_future_source_early(enabled):
    rows, _ = growing_history(1, initial_bytes=256)
    boundary = rows[1]
    wire = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": '{"x":"\\ud800"}'}}],
    }
    replacement = logical_record(
        "boundary_recorded",
        data={"messages": [wire], "shared_state": {}},
        **{key: value for key, value in boundary.payload.items() if key != "data"},
    )
    fold = Fold()
    if enabled:
        from vv_agent.memory.input_protection import UnseenInputError

        with pytest.raises(UnseenInputError):
            extend_logical(fold, (logical_session_record(), rows[0], replacement))
        return
    extend_logical(
        fold,
        (
            logical_session_record(attributes={"seed": {"unseen_input_protection": enabled}}),
            rows[0],
            replacement,
        ),
    )
    assert fold.state.boundaries[("t", "before_memory", "1")].payload["data"]["messages"] == [wire]


def test_fragment_cache_cannot_hide_invalid_scalars_or_digest_changes():
    from vv_agent.session.records import _MessageCache, digest

    from .fold_history import logical_record

    cache = _MessageCache()
    request = {"messages": [{"role": "user", "content": "same", "metadata": {"value": None}}]}
    original = logical_record("op_planned", request=request, request_digest=digest(request))
    body = original.encode()
    assert parse_logical_record(body, cache=cache).encode() == body
    for replacement in (b"NaN", b"Infinity", b"9007199254740992"):
        bad = body.replace(b'"value":null', b'"value":' + replacement)
        with pytest.raises(ValueError):
            parse_logical_record(bad, cache=cache)
    bad = body.replace(b'"content":"same"', b'"content":"changed"')
    with pytest.raises(ValueError, match="request digest mismatch"):
        parse_logical_record(bad, cache=cache)


def test_fragment_cache_keeps_noncanonical_wire_values_and_digest_bytes():
    from vv_agent.session.records import _MessageCache, digest

    from .fold_history import logical_record

    request = {"messages": [{"role": "user", "content": "😀", "metadata": {"n": 1e-7, "\ue000": 1, "😀": 2}}]}
    original = logical_record("op_planned", request=request, request_digest=digest(request))
    cache = _MessageCache()
    body = original.encode()
    spaced = json.dumps(json.loads(body), ensure_ascii=True, indent=2).encode()
    assert parse_logical_record(spaced, cache=cache).to_dict() == original.to_dict()
    assert parse_logical_record(spaced, cache=cache).encode() == body
    assert parse_logical_record(body, cache=cache).encode() == body


def test_fragment_reuse_preserves_legacy_numeric_normalization():
    from vv_agent.session.records import _MessageCache, digest

    from .fold_history import logical_record

    request = {"messages": [{"role": "user", "content": "same", "metadata": {"n": 1e20}}]}
    value = logical_record("op_planned").to_dict()
    value["payload"].update(request=request, request_digest=digest(request))
    body = json.dumps(value).encode()
    expected = parse_logical_record(body)
    cache = _MessageCache()
    parsed = parse_logical_record(body, cache=cache)
    parsed._share_messages(cache)
    assert parsed.to_dict() == expected.to_dict()
    assert parsed.encode() == expected.encode()


def test_parse_only_shares_plain_immutable_bytes():
    class SpoofedBytes(bytes):
        def __eq__(self, other):
            return True

    body = logical_session_record(actor="p", workspace="w").encode()
    raw = SpoofedBytes(json.dumps(json.loads(body), indent=2).encode())
    parsed = parse_logical_record(raw)
    assert type(parsed.encode()) is bytes
    assert parsed.encode() == body


@pytest.mark.parametrize("enabled", [False, True])
def test_completed_result_does_not_compare_entire_record_payloads(store, monkeypatch, enabled):
    from vv_agent.kernel._budget_values import BudgetUsageSnapshot

    persist_history(store, 2, initial_bytes=256, enabled=enabled)
    read_fixture_state(store, "s")
    append_fixture_records(
        store,
        "s",
        (logical_record("turn_ended", result={}, budget=BudgetUsageSnapshot().to_dict()),),
        commit_id="end",
    )
    snapshot = read_fixture_state(store, "s")

    def unexpected(*args):
        raise AssertionError("terminal is a retained record identity, not a payload comparison")

    monkeypatch.setattr(Record, "__eq__", unexpected)
    result = project_result(store, "s", "t", snapshot=snapshot)
    assert len(result.raw_result.cycles) == 2
    assert result.final_output == {}


def test_fragment_parser_matches_public_validation_for_mutated_messages():
    from copy import deepcopy

    from vv_agent.session.records import _MessageCache, digest

    from .fold_history import logical_record
    from .test_record_validation_cache import mutations

    source = [{"role": "user", "content": "same", "metadata": {"n": 1, "items": [True, None], "fraction": 1e-7}}]
    envelope = logical_record("op_planned").to_dict()
    cache = _MessageCache()
    for messages in mutations(source):
        value = deepcopy(envelope)
        request = {"messages": messages}
        value["payload"]["request"] = request
        with suppress(ValueError, TypeError):
            value["payload"]["request_digest"] = digest(request)
        body = json.dumps(value).encode()
        try:
            expected = parse_logical_record(body)
        except (ValueError, TypeError, KeyError) as error:
            with pytest.raises(type(error)) as caught:
                parse_logical_record(body, cache=cache)
            assert str(caught.value) == str(error)
        else:
            parsed = parse_logical_record(body, cache=cache)
            assert parsed.to_dict() == expected.to_dict()
            assert parsed.encode() == expected.encode()


@pytest.mark.parametrize("kind", ["op_planned", "boundary_recorded"])
def test_fragment_cache_cannot_replace_nan_with_an_interned_null(kind):
    from copy import deepcopy

    from vv_agent.session.records import _MessageCache

    from .fold_history import logical_record

    message = {"role": None, "content": "same"}
    if kind == "op_planned":
        value = logical_record(kind).to_dict()
        value["payload"]["request"] = {"messages": [message]}
    else:
        value = logical_record(
            kind,
            boundary_id="memory",
            stage="before_memory",
            source_operation_id=None,
            source_digest=None,
            data={"messages": [message], "shared_state": {}},
        ).to_dict()
    cache = _MessageCache()
    cache.share(deepcopy(message))
    snapshots = value["payload"]["request"] if kind == "op_planned" else value["payload"]["data"]
    snapshots["messages"][0]["role"] = float("nan")
    body = json.dumps(value).encode()
    with pytest.raises(ValueError) as public:
        parse_logical_record(body)
    with pytest.raises(type(public.value)) as retained:
        parse_logical_record(body, cache=cache)
    assert str(retained.value) == str(public.value)


def test_record_parser_rejects_duplicate_keys_after_unescaping():
    from vv_agent.session.records import RecordError, _MessageCache

    body = logical_session_record(actor="p", workspace="w", attributes={"a": 1}).encode()
    bad = body.replace(b'"a":1', b'"a":1,"\\u0061":2')
    with pytest.raises(RecordError, match="duplicate field: a"):
        parse_logical_record(bad)
    with pytest.raises(RecordError, match="duplicate field: a"):
        Record._parse(bad, _MessageCache())


def test_record_json_numbers_match_stdlib_canonical_encoding():
    import random
    import struct

    from vv_agent.kernel._canonical_json import canonical_json_bytes

    literals = [
        b"1e-400",
        b"1e400",
        b"9007199254740993",
        b"9007199254740993.0",
        b"4.9406564584124654e-324",
        b"2.2250738585072012e-308",
        b"0.84551240822557006",
        b"1.234567890123456789",
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"-0.0",
        b'"\\ud800"',
        b'"\\ud83d\\ude00"',
    ]
    rng = random.Random(17)
    literals.extend(json.dumps(struct.unpack("!d", rng.getrandbits(64).to_bytes(8))[0]).encode() for _ in range(128))
    template = logical_session_record(actor="p", workspace="w", attributes={"probe": None}).encode()
    for literal in literals:
        body = template.replace(b'"probe":null', b'"probe":' + literal)
        value = json.loads(body)
        try:
            expected = canonical_json_bytes(value)
        except (ValueError, TypeError) as error:
            with pytest.raises(ValueError) as caught:
                parse_logical_record(body)
            assert str(caught.value) == str(error)
        else:
            parsed = parse_logical_record(body)
            assert parsed.to_dict() == value
            assert parsed.encode() == expected


def test_192_batch_protection_decodes_at_most_twice_undeclared(monkeypatch):
    from dataclasses import replace

    from vv_agent.session.records import InboxItem

    rows, expected = growing_history(192, initial_bytes=1024, steer=True, usage=8)
    bodies = [row.encode() for row in rows]
    inputs = [InboxItem(**row.payload["input"]) for row in rows if row.kind == "input_applied"]
    counts = []
    decode = Message.from_dict
    for enabled in (False, True):
        creation = logical_session_record()
        if not enabled:
            creation = replace(creation, payload=creation.payload | {"attributes": {}})
        incoming = [parse_logical_record(body) for body in bodies]
        calls = []

        def counted(value, calls=calls):
            calls.append(1)
            return decode(value)

        with monkeypatch.context() as patch:
            patch.setattr(Message, "from_dict", counted)
            fold = Fold()
            extend_logical(fold, (creation, *incoming), consumed_inputs=inputs)
        counts.append(len(calls))
        assert project_context(fold.history, fold.state) == expected
    assert counts[1] <= 2 * counts[0], counts
    assert fold.state._protection is not None
    assert fold.state._protection.messages == expected
