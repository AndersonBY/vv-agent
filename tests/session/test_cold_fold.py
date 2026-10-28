"""Cold loads keep their validation and visit each retained snapshot a bounded number of times."""

import json
from contextlib import suppress
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest

from vv_agent.events import DiagnosticEvent
from vv_agent.session.context import project_context
from vv_agent.session.projection import project_records
from vv_agent.session.records import Record, SessionSpec
from vv_agent.session.reducer import Fold, TransitionError
from vv_agent.session.result import project_result
from vv_agent.types import Message

from .fold_history import growing_history, persist_history


def snapshot_value(obj):
    if isinstance(obj, Record):
        return {"record_id": obj.record_id, "digest": obj.digest}
    if is_dataclass(obj):
        return {f.name: snapshot_value(getattr(obj, f.name)) for f in fields(obj)}
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
    saved = tuple(Record.parse(body) for body in bodies)
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "test", "/tmp"), consumers=())
    lease = store.acquire("s", owner="compatibility", ttl_ms=15000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="history", records=saved[1:])
    store._fold_cache = None
    state, rows, _ = store.read_state("s")
    for f in fields(state):
        assert snapshot_value(getattr(state, f.name)) == expected[f.name], f.name
    assert [r.record.encode() for r in rows] == bodies
    assert [m.to_dict() for m in project_context(rows, state)] == expected["messages"]


def test_fold_message_decodes_grow_with_stored_snapshots(monkeypatch):
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
            fold.extend((SessionSpec("s", "test", "/tmp").record(), *rows))
        counts.append(len(calls))
    # Stored full conversations grow quadratically; replaying every obsolete snapshot is cubic.
    assert counts[1] <= 5 * counts[0]
    assert counts[1] < 5 * 64**2


def test_cold_record_shares_canonical_storage_bytes():
    body = SessionSpec("s", "p", "w").record().encode()
    assert Record.parse(body).encode() is body
    spaced = json.dumps(json.loads(body), indent=2).encode()
    assert Record.parse(spaced).encode() == body


def test_earlier_snapshots_still_validate_their_source():
    rows, _ = growing_history(4, initial_bytes=256)
    boundary = rows[1]
    bad = replace(boundary, payload=boundary.payload | {"source_digest": "0" * 64})
    fold = Fold()
    with pytest.raises(TransitionError, match="boundary source mismatch"):
        fold.extend((SessionSpec("s", "test", "/tmp").record(), rows[0], bad, *rows[2:]))


def test_projected_events_remain_detached_from_records(store):
    persist_history(store, 2, initial_bytes=256)
    _, rows, _ = store.read_state("s")
    before = [r.record.encode() for r in rows]
    projected = project_records(rows)
    response = next(e for e in projected if isinstance(e, DiagnosticEvent) and e.code == "cycle_llm_response")
    response.details["tool_calls"][0]["arguments"]["changed"] = True
    result = project_result(store, "s", "t")
    result.raw_result.shared_state["changed"] = True
    assert [r.record.encode() for r in rows] == before
    assert project_result(store, "s", "t").raw_result.shared_state == {}


def test_cold_prefix_bounds_fetches_and_shares_only_private_messages(store, monkeypatch):
    persist_history(store, 40, initial_bytes=256)
    store._fold_cache = None
    fetch = store._rows
    sizes = []

    def counted(query, params=()):
        rows = fetch(query, params)
        if "SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record" in query:
            sizes.append(len(rows))
        return rows

    monkeypatch.setattr(store, "_rows", counted)
    _, rows, _ = store.read_state("s")
    assert max(sizes) <= 256 and len(sizes) >= 3
    plans = [r.record for r in rows if r.record.kind == "op_planned" and r.record._payload["op_kind"] == "model"]
    assert plans[0]._payload["request"]["messages"][1] is plans[-1]._payload["request"]["messages"][1]
    plans[-1].payload["request"]["messages"][1]["content"] = "changed"
    assert plans[0].payload["request"]["messages"][1]["content"] == "x" * 256


def test_cold_prefix_rejects_a_gap_across_fetch_pages(store):
    from vv_agent.session.store import Conflict

    persist_history(store, 40, initial_bytes=256)
    store._fold_cache = None
    with store.atomic():
        store._rows("DELETE FROM sk_record WHERE session_id=%s AND seq=257", ("s",))
    with pytest.raises(Conflict, match="log sequence gap"):
        store.read_state("s")


def test_result_projection_does_not_scan_all_tools_for_each_model(store):
    persist_history(store, 24, initial_bytes=256)
    state, rows, watermark = store.read_state("s")
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


def test_raw_context_still_includes_the_full_history(store):
    messages = persist_history(store, 4, initial_bytes=256)
    state, rows, _ = store.read_state("s")
    assert [m.to_dict() for m in project_context(rows, state, compacted=False)] == [m.to_dict() for m in messages]


def test_fold_does_not_visit_or_copy_the_obsolete_record_prefix():
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
    fold.extend((SessionSpec("s", "test", "/tmp").record(), *rows))
    assert history.visits <= 20 * 64


def test_parse_does_not_retain_a_mutable_input_buffer():
    body = SessionSpec("s", "p", "w").record().encode()
    buffer = bytearray(body)
    parsed = Record.parse(buffer)  # ty: ignore[invalid-argument-type]
    buffer[:] = b"changed"
    assert parsed.encode() == body
    assert parsed.payload["principal"] == "p"


def test_cold_prefix_rejects_an_empty_captured_log(store):
    from vv_agent.session.store import Conflict

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "test", "/tmp"), consumers=())
        store._rows("UPDATE sk_session SET head_seq=0 WHERE session_id=%s", ("s",))
    store._fold_cache = None
    with pytest.raises(Conflict, match="log sequence gap"):
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


def test_fast_record_parser_keeps_jcs_numbers_unicode_and_legacy_errors():
    from vv_agent.session.records import RecordError

    attributes = {
        "numbers": [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27, 5e-324, 1.7976931348623157e308, -0.0],
        "unicode": {"😀": "中文", "\ue000": "x"},
    }
    original = SessionSpec("s", "p", "w", attributes=attributes).record()
    body = original.encode()
    assert Record.parse(body).to_dict() == original.to_dict()
    assert Record.parse(body).encode() == body
    assert Record.parse(b"\xef\xbb\xbf" + body).encode() == body
    with pytest.raises(RecordError, match="duplicate field: schema_version"):
        Record.parse(body.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'))


def test_message_cache_does_not_encode_a_future_source_early():
    from vv_agent.session.records import make_record

    rows, _ = growing_history(1, initial_bytes=256)
    boundary = rows[1]
    wire = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": '{"x":"\\ud800"}'}}],
    }
    replacement = make_record(
        "boundary_recorded",
        session_id="s",
        turn_id="t",
        payload=boundary.payload | {"data": {"messages": [wire], "shared_state": {}}},
    )
    fold = Fold()
    fold.extend((SessionSpec("s", "test", "/tmp").record(), rows[0], replacement))
    assert fold.state.boundaries[("t", "before_memory", "1")].payload["data"]["messages"] == [wire]


def test_fragment_cache_cannot_hide_invalid_scalars_or_digest_changes():
    from vv_agent.session.records import _MessageCache, digest

    cache = _MessageCache()
    request = {"messages": [{"role": "user", "content": "same", "metadata": {"value": None}}]}
    from .helpers import record

    original = record("op_planned", request=request, request_digest=digest(request))
    body = original.encode()
    assert Record._parse(body, cache).encode() == body
    for replacement in (b"NaN", b"Infinity", b"9007199254740992"):
        bad = body.replace(b'"value":null', b'"value":' + replacement)
        with pytest.raises(ValueError):
            Record._parse(bad, cache)
    bad = body.replace(b'"content":"same"', b'"content":"changed"')
    with pytest.raises(ValueError, match="request digest mismatch"):
        Record._parse(bad, cache)


def test_fragment_cache_keeps_noncanonical_wire_values_and_digest_bytes():
    from vv_agent.session.records import _MessageCache, digest

    from .helpers import record

    request = {"messages": [{"role": "user", "content": "😀", "metadata": {"n": 1e-7, "\ue000": 1, "😀": 2}}]}
    original = record("op_planned", request=request, request_digest=digest(request))
    cache = _MessageCache()
    body = original.encode()
    spaced = json.dumps(json.loads(body), ensure_ascii=True, indent=2).encode()
    assert Record._parse(spaced, cache).to_dict() == original.to_dict()
    assert Record._parse(spaced, cache).encode() == body
    assert Record._parse(body, cache).encode() == body


def test_fragment_reuse_preserves_legacy_numeric_normalization():
    from vv_agent.session.records import _MessageCache, digest

    from .helpers import record

    request = {"messages": [{"role": "user", "content": "same", "metadata": {"n": 1e20}}]}
    value = record("op_planned").to_dict()
    value["payload"].update(request=request, request_digest=digest(request))
    body = json.dumps(value).encode()
    expected = Record.parse(body)
    cache = _MessageCache()
    parsed = Record._parse(body, cache)
    parsed._share_messages(cache)
    assert parsed.to_dict() == expected.to_dict()
    assert parsed.encode() == expected.encode()


def test_parse_only_shares_plain_immutable_bytes():
    class SpoofedBytes(bytes):
        def __eq__(self, other):
            return True

    body = SessionSpec("s", "p", "w").record().encode()
    raw = SpoofedBytes(json.dumps(json.loads(body), indent=2).encode())
    parsed = Record.parse(raw)
    assert type(parsed.encode()) is bytes
    assert parsed.encode() == body


def test_completed_result_does_not_compare_entire_record_payloads(store, monkeypatch):
    from .helpers import record

    persist_history(store, 2, initial_bytes=256)
    _, rows, _ = store.read_state("s")
    lease = store.acquire("s", owner="finish", ttl_ms=15000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=len(rows), commit_id="end", records=(record("turn_ended"),))
    snapshot = store.read_state("s")

    def unexpected(*args):
        raise AssertionError("terminal is a retained record identity, not a payload comparison")

    monkeypatch.setattr(Record, "__eq__", unexpected)
    result = project_result(store, "s", "t", snapshot=snapshot)
    assert len(result.raw_result.cycles) == 2
    assert result.final_output == {}


def test_fragment_parser_matches_public_validation_for_mutated_messages():
    from copy import deepcopy

    from vv_agent.session.records import _MessageCache, digest

    from .helpers import record
    from .test_record_validation_cache import mutations

    source = [{"role": "user", "content": "same", "metadata": {"n": 1, "items": [True, None], "fraction": 1e-7}}]
    envelope = record("op_planned").to_dict()
    cache = _MessageCache()
    for messages in mutations(source):
        value = deepcopy(envelope)
        request = {"messages": messages}
        value["payload"]["request"] = request
        with suppress(ValueError, TypeError):
            value["payload"]["request_digest"] = digest(request)
        body = json.dumps(value).encode()
        try:
            expected = Record.parse(body)
        except (ValueError, TypeError, KeyError) as error:
            with pytest.raises(type(error)) as caught:
                Record._parse(body, cache)
            assert str(caught.value) == str(error)
        else:
            parsed = Record._parse(body, cache)
            assert parsed.to_dict() == expected.to_dict()
            assert parsed.encode() == expected.encode()


def test_fast_parser_rejects_duplicate_keys_after_unescaping():
    from vv_agent.session.records import RecordError, _MessageCache

    body = SessionSpec("s", "p", "w", attributes={"a": 1}).record().encode()
    bad = body.replace(b'"a":1', b'"a":1,"\\u0061":2')
    with pytest.raises(RecordError, match="duplicate field: a"):
        Record.parse(bad)
    with pytest.raises(RecordError, match="duplicate field: a"):
        Record._parse(bad, _MessageCache())


def test_fast_json_decoder_matches_the_original_reader(monkeypatch):
    import random
    import struct

    from vv_agent.session import records

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
        b'{"a":1,"\\u0061":2}',
    ]
    rng = random.Random(17)
    literals.extend(json.dumps(struct.unpack("!d", rng.getrandbits(64).to_bytes(8))[0]).encode() for _ in range(128))
    template = SessionSpec("s", "p", "w", attributes={"probe": None}).record().encode()

    def unsupported(*args, **kwargs):
        raise ValueError("use the original reader")

    def outcome(body):
        try:
            parsed = Record.parse(body)
            return parsed.encode(), parsed.to_dict()
        except (ValueError, TypeError) as error:
            return type(error), str(error)

    for literal in literals:
        body = template.replace(b'"probe":null', b'"probe":' + literal)
        actual = outcome(body)
        with monkeypatch.context() as patch:
            patch.setattr(records.jiter, "from_json", unsupported)
            expected = outcome(body)
        assert actual == expected
