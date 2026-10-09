"""Deterministic coverage sets for the v24 authoring corpus (no synthetic wires)."""

from __future__ import annotations

import json

from vv_agent.session.records import (
    APP_SERVER_ATTRIBUTES,
    BOUNDARY_DATA,
    CHILD_ADMISSION,
    INPUT_PAYLOADS,
    INPUT_SCHEMA,
    PAYLOADS,
    RECORD_SCHEMA,
    REQUEST_SESSION_METADATA,
    SEED,
    TASK_SESSION_METADATA,
)

# Values that distinguish behavior; identifiers, timestamps, text and opaque J do not.
VARIANTS = {
    "type",
    "kind",
    "stage",
    "status",
    "completion_reason",
    "completionReason",
    "phase",
    "purpose",
    "operation",
    "context",
    "disposition",
    "action",
    "mode",
    "dimension",
    "reason",
    "usage_source",
    "cache_status",
    "enforcement_boundary",
    "method",
    "code",
    "role",
    "directive",
    "status_code",
    "background",
    "execution_started",
    "closed",
}
SCENARIO_FILES = {"session_semantics.json": "cases", "session_recovery.json": "cases"}
EVENT_FILES = {
    "run_events.jsonl",
    "runner_events.jsonl",
    "event_store_replay.jsonl",
    "budget_events.jsonl",
    "configured_sub_agent_events.jsonl",
}


def shape_keys(value, path="", *, opaque=False):
    """Field presence/nullability and explicitly listed behavioral variants."""
    if opaque:
        return {path + (":null" if value is None else ":non-null")}
    if isinstance(value, dict):
        keys = set()
        for field, item in value.items():
            target = path + "." + field
            keys.add(target + ":present")
            if field in VARIANTS and isinstance(item, (str, bool, int)):
                keys.add(target + "=" + str(item))
            keys.update(
                shape_keys(
                    item,
                    target,
                    opaque=field in {"metadata", "shared_state", "sharedState", "raw", "arguments", "tool_arguments", "details"},
                )
            )
        return keys
    if isinstance(value, list):
        return {path + (":empty" if not value else ":nonempty")} | set().union(*(shape_keys(item, path + "[]") for item in value))
    return {path + (":null" if value is None else ":non-null")}


def typed_keys(schema, value, path):
    """Follow closed schemas only; opaque J stops at its presence/null boundary."""
    keys = {path + (":null" if value is None else ":non-null")}
    for variant in schema.get("oneOf", schema.get("anyOf", [])):
        kind = variant.get("properties", {}).get("kind", {}).get("const")
        if kind is not None and (not isinstance(value, dict) or value.get("kind") != kind):
            continue
        if variant.get("type") == "null" and value is not None:
            continue
        keys.update(typed_keys(variant, value, path))
    if "const" in schema or "enum" in schema or schema.get("type") == "boolean":
        keys.add(path + "=" + str(value))
    if isinstance(value, dict):
        for field, child in schema.get("properties", {}).items():
            target = path + "." + field
            keys.add(target + (":present" if field in value else ":absent"))
            if field in value:
                keys.update(typed_keys(child, value[field], target))
    if isinstance(value, list) and "items" in schema:
        keys.add(path + (":empty" if not value else ":nonempty"))
        for item in value:
            keys.update(typed_keys(schema["items"], item, path + "[]"))
    return keys


def record_keys(wire, inbox=False):
    kind, p = wire["kind"], wire["payload"]
    path = ("inbox." if inbox else "record.") + kind
    keys = {path} | typed_keys(INPUT_SCHEMA if inbox else RECORD_SCHEMA, wire, path)
    keys.update(typed_keys((INPUT_PAYLOADS if inbox else PAYLOADS)[kind], p, path + ".payload"))
    if not inbox:
        if kind == "boundary_recorded":
            keys.update(typed_keys(BOUNDARY_DATA[p["stage"]], p["data"], path + "." + p["stage"]))
        for field, schema in (("seed", SEED), ("app_server", APP_SERVER_ATTRIBUTES), ("child_admission", CHILD_ADMISSION)):
            attributes = p.get("attributes", {})
            keys.add(path + ".attributes." + field + (":present" if field in attributes else ":absent"))
            if field in attributes:
                keys.update(typed_keys(schema, attributes[field], path + ".attributes." + field))
        definitions = [p["definition"]] if "definition" in p else []
        if "child_admission" in p.get("attributes", {}):
            definitions.append(p["attributes"]["child_admission"]["definition"])
        for definition in definitions:
            keys.update(
                typed_keys(
                    TASK_SESSION_METADATA, definition["task"].get("metadata", {}).get("vv_session", {}), path + ".task.vv_session"
                )
            )
        if "request" in p:
            keys.update(
                typed_keys(
                    REQUEST_SESSION_METADATA, p["request"].get("metadata", {}).get("vv_session", {}), path + ".request.vv_session"
                )
            )
    return keys


def canonical_features(value):
    keys = set()
    if isinstance(value, dict):
        if any(any(ord(char) > 0xFFFF for char in field) for field in value):
            keys.add("jcs:astral-key")
        if any(any(0xE000 <= ord(char) <= 0xFFFF for char in field) for field in value):
            keys.add("jcs:bmp-high-key")
        for item in value.values():
            keys.update(canonical_features(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(canonical_features(item))
    elif isinstance(value, bool):
        keys.add("jcs:boolean=" + str(value))
    elif isinstance(value, int):
        keys.add("jcs:integer")
    elif isinstance(value, float):
        keys.add("jcs:float")
        if value == 0:
            import math

            keys.add("jcs:negative-zero" if math.copysign(1, value) < 0 else "jcs:positive-zero")
        elif abs(value) < 1e-6 or abs(value) >= 1e21:
            keys.add("jcs:exponent")
    elif value is None:
        keys.add("jcs:null")
    return keys


def entry_keys(name, item):
    if name in {"session_codec_vectors.json", "session_records.jsonl", "session_inbox.jsonl", "session_compaction.json"}:
        wire = item.get("wire", item)
        keys = record_keys(wire, item.get("type") == "inbox" or name == "session_inbox.jsonl")
        if name == "session_codec_vectors.json":
            keys.update(canonical_features(wire))
        if name == "session_compaction.json":
            keys.add("scenario=" + item["session_id"])
        return keys
    if name == "session_invalid.json":
        return {"rejection=" + item["rejection_class"] + "@" + item["layer"]}
    if name in SCENARIO_FILES:
        return {"scenario=" + item["case"]} | shape_keys(
            {k: v for k, v in item.items() if k not in {"bytes_base64", "sha256", "record_id"}}
        )
    if name == "runner_trace.jsonl":
        return {"span=" + item["span"]["name"] + "@" + item["method"]} | shape_keys(item)
    if name == "prefix_states":
        return {
            "prefix="
            + item["phase"]
            + ":closed="
            + str(item["closed"])
            + ":active="
            + str(item["active_turn_id"] is not None)
            + ":terminal="
            + str(item["terminal_seq"] > 0)
        }
    if name == "projection_events":
        return {"event=" + item["type"]} | shape_keys(item, item["type"])
    if name == "projection_spans":
        return {"span=" + item["span"]["name"] + "@" + item["method"]} | shape_keys(item)
    if name == "session_projection.json":
        return (
            set().union(*(entry_keys("projection_events", event) for event in item["events"]))
            | set().union(*(entry_keys("projection_spans", span) for span in item["spans"]))
            | {
                "prefix="
                + prefix["phase"]
                + ":closed="
                + str(prefix["closed"])
                + ":active="
                + str(prefix["active_turn_id"] is not None)
                + ":terminal="
                + str(prefix["terminal_seq"] > 0)
                for prefix in item["prefix_states"]
            }
        )
    if name == "app_server_protocol.json":
        request = item.get("request", {})
        keys = {"request=" + request["method"]} if request else set()
        if request:
            keys.add("lifecycle=" + item["connection_id"] + ":" + request["method"])
        for response in item.get("responses", []):
            if "error" in response:
                keys.add("error=" + str(response["error"]["code"]) + ":" + response["error"]["message"])
        for notice in item.get("notifications", []):
            keys.add("notification=" + notice["method"])
        return keys | shape_keys({k: v for k, v in item.items() if k not in {"bytes_base64", "sha256", "record_id"}})
    return shape_keys(item, item["type"] if name in EVENT_FILES else "")


def select(name, items, *, required=()):
    """Greedy set cover; equal gains choose fewer bytes, then original producer order."""
    key_sets = [entry_keys(name, item) for item in items]
    before = set().union(*key_sets)
    remaining = set(before)
    selected = set(required)
    for index in selected:
        remaining.difference_update(key_sets[index])
    sizes = [len(json.dumps(item, ensure_ascii=True, separators=(",", ":"))) for item in items]
    while remaining:
        index = min(
            (i for i in range(len(items)) if key_sets[i] & remaining),
            key=lambda i: (-len(key_sets[i] & remaining), sizes[i], i),
        )
        selected.add(index)
        remaining.difference_update(key_sets[index])
    # Remove redundant greedy choices, retaining explicit scenario/byte obligations.
    for index in sorted(selected, reverse=True):
        if index not in required and set().union(*(key_sets[i] for i in selected - {index})) == before:
            selected.remove(index)
    result = [item for i, item in enumerate(items) if i in selected]
    after = set().union(*(entry_keys(name, item) for item in result))
    assert before == after, (name, before - after)
    return result


def coverage_keys(name, value):
    if isinstance(value, list):
        return set().union(*(entry_keys(name, item) for item in value))
    collection = {
        "session_codec_vectors.json": "vectors",
        "session_invalid.json": "vectors",
        "session_compaction.json": "vectors",
        "session_projection.json": "sessions",
        "app_server_protocol.json": "transcripts",
        **SCENARIO_FILES,
    }.get(name)
    if collection:
        keys = set().union(*(entry_keys(name, item) for item in value[collection]))
        if name == "app_server_protocol.json":
            for family in ("jsonSchema", "typescript"):
                bundles = value["schemas"][family]
                keys.update("schema=" + family + ":" + bundle for bundle in bundles)
        return keys
    return shape_keys(value)


def projection_sources(rows, events, spans, through):
    """Remove redundant planning context only when the real projections stay identical."""
    from vv_agent.session.projection import project_records
    from vv_agent.session.tracing import project_spans

    def reproduces(candidate):
        try:
            actual_events = {event.event_id: event.to_dict() for event in project_records(candidate)} if events else {}
            actual_spans = (
                {(seq, method, span.span_id): span.to_dict() for seq, method, span in project_spans(tuple(candidate))}
                if spans
                else {}
            )
            return all(actual_events.get(event["event_id"]) == event for event in events) and all(
                actual_spans.get((span["seq"], span["method"], span["span"]["span_id"])) == span["span"] for span in spans
            )
        except KeyError:
            return False

    retained = list(rows)
    required = {event["metadata"]["session_seq"] for event in events} | {span["seq"] for span in spans}
    for row in sorted(rows, key=lambda row: (-len(row.record.encode()), row.seq)):
        if row.seq <= through or row.seq in required:
            continue
        candidate = [source for source in retained if source.seq != row.seq]
        if reproduces(candidate):
            retained = candidate
    assert reproduces(retained)
    return retained


def curate(output, streams, semantics, independent, facts):
    """Validate the full producer run first, then retain only covered representatives."""
    from collections import Counter

    def load(path):
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text())
        )

    values = {path.name: load(path) for path in sorted(output.iterdir())}
    before = {name: coverage_keys(name, value) for name, value in values.items()}
    all_vectors = values["session_codec_vectors.json"]["vectors"]
    receipts = {case["receipt_id"] for case in values["session_recovery.json"]["cases"] if "receipt_id" in case}
    vectors = select(
        "session_codec_vectors.json",
        all_vectors,
        required=tuple(i for i, vector in enumerate(all_vectors) if vector["wire"].get("record_id") in receipts),
    )
    values["session_codec_vectors.json"]["vectors"] = vectors
    for kind, name in (("record", "session_records.jsonl"), ("inbox", "session_inbox.jsonl")):
        values[name] = [v["wire"] for v in vectors if v["type"] == kind]

    # Fold/admission negatives retain their exact real source prefixes independently
    # of standalone codec coverage. They are not additional codec vectors.
    sources = {sid: rows for sid, rows in streams}
    supporting = {}
    invalid = values["session_invalid.json"]
    invalid["vectors"] = select("session_invalid.json", invalid["vectors"])
    for vector in invalid["vectors"]:
        if vector["layer"] not in {"fold", "admission"}:
            continue
        ids = set(vector.get("prefix_ids", [])) | {vector.get("audit_record_id")}
        for row in sources[vector["session_id"]]:
            if row.record.record_id in ids:
                supporting[(row.record.session_id, row.record.record_id)] = row.record.to_dict()
    invalid["source_records"] = list(supporting.values())

    projections = values["session_projection.json"]["sessions"]
    events = select("projection_events", [event for item in projections for event in item["events"]])
    spans = select(
        "projection_spans", [span | {"session_id": item["session_id"]} for item in projections for span in item["spans"]]
    )
    prefixes = select(
        "prefix_states", [prefix | {"session_id": item["session_id"]} for item in projections for prefix in item["prefix_states"]]
    )
    selected = []
    source_records = {}
    for item in projections:
        sid = item["session_id"]
        item["events"] = [event for event in events if event["session_id"] == sid]
        for field, candidates in (("spans", spans), ("prefix_states", prefixes)):
            item[field] = [
                {k: v for k, v in candidate.items() if k != "session_id"}
                for candidate in candidates
                if candidate["session_id"] == sid
            ]
        if not any(item[field] for field in ("events", "spans", "prefix_states")):
            continue
        through = max((prefix["seq"] for prefix in item["prefix_states"]), default=0)
        seqs = {event["metadata"]["session_seq"] for event in item["events"]} | {span["seq"] for span in item["spans"]}
        # Fold cases need complete prefixes; event/span cases need their real sources and planning context.
        rows = [
            row
            for row in sources[sid]
            if row.seq <= through
            or row.seq in seqs
            or row.record.kind in {"turn_started", "op_planned", "op_prepared", "op_started", "op_parked"}
        ]
        source_records[sid] = [
            {
                "wire": row.record.to_dict(),
                "seq": row.seq,
                "commit_id": row.commit_id,
                "writer_epoch": row.writer_epoch,
                "created_ms": row.created_ms,
            }
            for row in projection_sources(rows, item["events"], item["spans"], through)
        ]
        selected.append(item)
    for item, raw in zip(
        selected,
        independent([{k: v for k, v in item.items() if k not in {"bytes_base64", "sha256", "record_id"}} for item in selected]),
        strict=True,
    ):
        item.update(facts(raw))
    values["session_projection.json"] = {"sessions": selected, "source_records": source_records}

    for name in EVENT_FILES | {"runner_trace.jsonl"}:
        values[name] = select(name, values[name])
    app = values["app_server_protocol.json"]
    app["transcripts"] = select("app_server_protocol.json", app["transcripts"])
    # Whitespace is not a schema contract. Keep every exported bundle and all types.
    app["schemas"]["jsonSchema"] = {
        name: json.dumps(json.loads(source), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for name, source in app["schemas"]["jsonSchema"].items()
    }

    typescript = app["schemas"]["typescript"]
    source = next(iter(typescript.values()))
    assert all(text == source for text in typescript.values())
    from hashlib import sha256

    app["schemas"]["typescript_source"] = source
    app["schemas"]["typescript"] = {
        name: {"source_ref": "typescript_source", "sha256": sha256(text.encode()).hexdigest()}
        for name, text in typescript.items()
    }
    inventory = values["session_codec_vectors.json"]["coverage"]
    from session_kernel_fixtures import coverage

    values["session_codec_vectors.json"]["coverage"] = coverage(vectors, semantics)
    for group in inventory:
        assert set(inventory[group]) == set(values["session_codec_vectors.json"]["coverage"][group]), group
    report = {}
    for name, value in values.items():
        after = coverage_keys(name, value)
        assert before[name] == after, (name, before[name] - after)
        report[name] = {"before": sorted(before[name]), "after": sorted(after)}
        path = output / name
        if path.suffix == ".jsonl":
            path.write_bytes(b"".join(raw + b"\n" for raw in independent(value)))
        elif name == "session_projection.json":
            # One source row per line keeps real wire objects readable without repeated indentation.
            lines = [
                "{",
                '  "sessions": '
                + json.dumps(value["sessions"], ensure_ascii=True, sort_keys=True, indent=2).replace("\n", "\n  ")
                + ",",
                '  "source_records": {',
            ]
            sources = sorted(value["source_records"].items())
            for index, (sid, rows) in enumerate(sources):
                lines.append("    " + json.dumps(sid) + ": [")
                lines.extend(
                    "      "
                    + json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                    + ("," if i + 1 < len(rows) else "")
                    for i, row in enumerate(rows)
                )
                lines.append("    ]" + ("," if index + 1 < len(sources) else ""))
            path.write_text("\n".join([*lines, "  }", "}", ""]))
        else:
            path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n")
    for name in SCENARIO_FILES:
        cases = Counter(case["case"] for case in values[name]["cases"])
        assert all(count == 1 for count in cases.values()), (name, cases)
    return values, report
