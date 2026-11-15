"""Record values and private fragments have one stdlib-only implementation."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import vv_agent
from vv_agent.kernel import _records as core
from vv_agent.session import records


def content_ref(label: str = "content"):
    payload = label.encode()
    return {
        "content_id": f"content/test/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": "key/test",
        "labels": {"source": [], "classification": "internal"},
    }


def principal():
    return {
        "tenant_id": "default",
        "subject_id": "p",
        "subject_kind": "user",
        "attributes": content_ref("principal"),
    }


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_records_import_and_s1_encode_are_core_only(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
from hashlib import sha256
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {"vv_agent", "vv_agent.kernel"}
                or fullname.startswith("vv_agent.kernel.")
                or fullname.split(".")[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel import _records as r
def ref(label):
    payload = label.encode()
    return {"content_id": "content/" + label, "mac": sha256(payload).hexdigest(), "size": len(payload),
            "key_id": "key/test", "labels": {"source": [], "classification": "internal"}}
principal = {"tenant_id": "default", "subject_id": "p", "subject_kind": "user", "attributes": ref("principal")}
spec = r.SessionSpec("s", principal, ref("workspace"), attributes=ref("attributes")).record()
assert r.Record.parse(spec.encode()).digest == spec.digest
item = r.InboxItem("i", "user", {"content": ref("go")}, actor=principal)
assert r.InboxItem.parse(item.encode()).input_id == "i"
definition = ref("definition")
record = r.make_record("turn_started", session_id="s", turn_id="t", payload={
    "input_ids": [], "definition": definition, "definition_digest": r.digest(definition),
    "handler_version": "1", "budget": ref("budget"), "binding": "task", "generation": 1,
})
try:
    record._task()
except r.RecordError as error:
    assert str(error) == "S1 content hydration is unavailable in the producer/codec phase"
else:
    raise AssertionError("unhydrated S1 content decoded")
assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
           for name in sys.modules if name.startswith("vv_agent"))
"""
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_exports_are_the_actual_core_objects():
    for name in ("Record", "InboxItem", "SessionSpec"):
        assert getattr(vv_agent, name) is getattr(records, name) is getattr(core, name)
        assert getattr(core, name).__module__ == "vv_agent.kernel._records"
    for name in ("_MessageCache", "_RetainedTools", "copy_json", "digest", "record_identity", "make_record"):
        assert getattr(records, name) is getattr(core, name)


def test_private_keys_keep_numeric_null_and_string_types_distinct():
    cache = core._MessageCache()
    values = [None, float("nan"), float("inf"), True, 1, 1.0, "1", 0, -0.0, False]
    wires = [{"role": "user", "content": "same", "metadata": {"value": value}} for value in values]
    shared = [cache.share(wire) for wire in wires]
    assert len({id(wire) for wire in shared}) == len(wires)
    assert math.isnan(shared[1]["metadata"]["value"])
    for wire, retained in zip(wires, shared, strict=True):
        assert cache.share(dict(reversed(list(wire.items())))) is retained


@pytest.mark.parametrize(
    "field,valid,invalid",
    [("tool_calls", [], ()), ("metadata", {"1": "x"}, {1: "x"})],
)
def test_private_keys_do_not_coerce_containers_or_keys(field, valid, invalid):
    cache = core._MessageCache()
    good = {"role": "assistant", "content": "same", field: valid}
    bad = good | {field: invalid}
    assert cache.share(good) is good
    retained = cache.share(bad)
    assert retained is bad
    assert retained[field] is invalid


def test_private_keys_defer_message_argument_validation():
    cache = core._MessageCache()
    wire = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call", "type": "function", "function": {"name": "read", "arguments": '{"x":"\\ud800"}'}}],
    }
    shared = cache.share(wire)
    assert shared is wire
    assert json.loads(shared["tool_calls"][0]["function"]["arguments"]) == {"x": "\ud800"}
    message = cache.parse(shared)
    with pytest.raises(ValueError, match="I-JSON"):
        cache.source_digest([message])


def test_record_freezes_but_inbox_reencodes_mutable_payload():
    source = {"nested": [1]}
    created = core.SessionSpec("s", principal(), content_ref("workspace"), attributes=content_ref("attributes")).record()
    record = replace(created, payload=created.payload | {"attributes": source})
    with pytest.raises(core.RecordError, match="invalid session schema"):
        record.encode()
    record = replace(created, payload=created.payload | {"attributes": content_ref("attributes-2")})
    body = record.encode()
    assert record.encode() is body
    assert record.payload["attributes"] == content_ref("attributes-2")
    assert deepcopy(record) is record
    item = core.InboxItem("i", "user", {"content": content_ref("nested")}, actor=principal())
    first = item.encode()
    item = replace(item, payload={"content": content_ref("nested-2")})
    assert item.encode() != first
    assert core.InboxItem.parse(item.encode()).payload == item.payload


def test_unhydrated_content_never_falls_back_to_plaintext():
    definition = content_ref("definition")
    record = core.make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload={
            "input_ids": [],
            "definition": definition,
            "definition_digest": core.digest(definition),
            "handler_version": "1",
            "budget": content_ref("budget"),
            "binding": "task",
            "generation": 1,
        },
    )
    with pytest.raises(core.RecordError, match="content hydration is unavailable"):
        record.task()
    wrong = record.to_dict()
    wrong["payload"]["definition"] = {"plaintext": "do not accept"}
    with pytest.raises(core.RecordError, match="invalid session schema"):
        core.Record(**wrong).encode()


def test_s1_builder_defaults_are_contract_defaults_only():
    definition = content_ref("definition")
    turn = core.make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload={
            "input_ids": [],
            "definition": definition,
            "definition_digest": core.digest(definition),
            "handler_version": "1",
            "budget": content_ref("budget"),
            "binding": None,
            "generation": 0,
        },
    )
    assert turn.payload["policy_digest"] is None
    assert turn.payload["requirements"] == {"level": 1, "handler_version": "1", "capabilities": []}
    assert turn.payload["tokenizer_binding"] is None
    assert turn.payload["environment"] is None

    request = content_ref("request")
    planned = core.make_record(
        "op_planned",
        session_id="s",
        turn_id="t",
        operation_id="o",
        attempt=1,
        payload={
            "op_kind": "tool",
            "purpose": None,
            "request": request,
            "request_digest": core.digest(request),
            "context_version": "1",
            "dependencies": [],
            "tool": content_ref("tool"),
            "idempotency_key": None,
            "budget_admission": content_ref("budget"),
            "not_before_ms": None,
            "provider_binding": None,
            "consumed_unknowns": [],
        },
    )
    assert planned.payload["policy_decision"] is None


def test_inbox_s1_requires_actor_and_reference_form_without_fallback():
    with pytest.raises(core.RecordError, match="requires a closed Principal"):
        core.InboxItem("i", "user", {"content": content_ref("go")}).encode()
    with pytest.raises(core.RecordError, match="invalid session schema"):
        core.InboxItem("i", "user", {"content": "plaintext"}, actor=principal()).encode()


def test_logical_seed_reference_never_silently_uses_seed_defaults():
    with pytest.raises(core.RecordError, match="content hydration is unavailable"):
        core.inherited_seed_policies(content_ref("attributes"))
