"""The stdlib record codec accepts exactly the canonical contract 40 shapes."""

import base64
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from vv_agent.kernel import _record_codec as codec
from vv_agent.kernel import _record_schema as schemas
from vv_agent.kernel import _records as core_records
from vv_agent.session import records


def ref(label: str = "content"):
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
        "attributes": ref("principal"),
    }


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_record_codec_import_and_rejection_are_core_only(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {"vv_agent", "vv_agent.kernel"}
                or fullname.startswith("vv_agent.kernel.")
                or fullname.split(".")[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel import _record_codec as codec, _record_schema as schemas
assert codec._CHECKS is schemas._CHECKS
assert codec._load(b'{"key":1}') == {"key": 1}
try:
    codec.validate({"CUSTOMER_CONTENT": "private"}, schemas.RECORD_SCHEMA)
except codec.RecordError as error:
    assert isinstance(error, ValueError)
    assert str(error) == "invalid session schema"
else:
    raise AssertionError("invalid envelope accepted")
assert {n for n in sys.modules if n.startswith("vv_agent")} == {
    "vv_agent", "vv_agent.kernel", "vv_agent.kernel._record_codec",
    "vv_agent.kernel._record_schema", "vv_agent.kernel._schema", "vv_agent.kernel._canonical_json"
}
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


def test_outer_records_use_the_actual_core_error_reader_and_validator():
    assert records.RecordError is codec.RecordError
    assert records.validate is codec.validate
    assert core_records._load is codec._load


@pytest.mark.parametrize("body", [b'{"a":}', b"\xff", b'{"a":1,}'])
def test_parser_error_text_and_cause_match_stdlib(body):
    with pytest.raises((ValueError, UnicodeError)) as reference:
        json.loads(body)
    with pytest.raises(codec.RecordError) as actual:
        records.Record.parse(body)
    assert str(actual.value) == str(reference.value)
    assert type(actual.value.__cause__) is type(reference.value)


@pytest.mark.parametrize("parse", [records.Record.parse, records.InboxItem.parse])
def test_duplicate_unescaped_fields_keep_the_specific_error(parse):
    with pytest.raises(codec.RecordError, match=r"^duplicate field: a$") as actual:
        parse(b'{"a":1,"\\u0061":2}')
    assert isinstance(actual.value.__cause__, codec.RecordError)
    assert str(actual.value.__cause__) == "duplicate field: a"


@pytest.mark.parametrize("parse", [records.Record.parse, records.InboxItem.parse])
def test_schema_rejection_never_contains_customer_or_schema_details(parse):
    if parse.__self__ is records.Record:
        value = records.SessionSpec("s", principal(), ref("workspace"), attributes=ref("attributes")).record().to_dict()
        value["payload"]["workspace"] = ["CUSTOMER_CONTENT"]
    else:
        value = records.InboxItem("i", "user", {"content": ref("go")}, actor=principal()).to_dict() | {
            "available_ms": "CUSTOMER_CONTENT"
        }
    with pytest.raises(codec.RecordError, match=r"^invalid session schema$"):
        parse(json.dumps(value).encode())


def test_record_and_inbox_keep_specific_semantic_and_content_ref_errors():
    wire = records.SessionSpec("s", principal(), ref("workspace"), attributes=ref("attributes")).record().to_dict()
    wire["record_id"] = "wrong"
    with pytest.raises(codec.RecordError, match=r"^record_id does not match semantic position$"):
        records.Record.parse(json.dumps(wire).encode())
    with pytest.raises(codec.RecordError, match=r"^invalid session schema$"):
        records.InboxItem("i", "model_selection", {"selection": {"model": "m"}}, actor=principal()).encode()


def test_canonical_rejection_keeps_precedence_and_original_cause():
    with pytest.raises(codec.RecordError, match="I-JSON") as actual:
        codec.validate({"invalid": float("nan")}, schemas.RECORD_SCHEMA)
    assert isinstance(actual.value.__cause__, ValueError)
    assert str(actual.value) == str(actual.value.__cause__)
    with pytest.raises(codec.RecordError, match=r"^invalid session schema$"):
        codec.validate({"invalid": float("nan")}, schemas.RECORD_SCHEMA, canonical=False)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_accepted_bom_inputs_keep_canonical_bytes(encoding):
    expected = records.SessionSpec("s", principal(), ref("workspace"), attributes=ref("attributes")).record().encode()
    parsed = records.Record.parse(expected.decode().encode(encoding))
    assert parsed.encode() == expected


def test_all_canonical_record_and_inbox_vectors_are_real_codec_inputs():
    from support.contract import fixture_root

    corpus = json.loads((fixture_root() / "session_codec_vectors.json").read_text())
    seen = {"record": 0, "inbox": 0}
    for vector in corpus["vectors"]:
        wire = vector.get("wire")
        if wire is None:
            continue
        body = base64.b64decode(vector["bytes_base64"], validate=True)
        codec_type = records.InboxItem if "actor" in wire else records.Record
        parsed = codec_type.parse(body)
        assert parsed.to_dict() == wire
        assert parsed.encode() == body
        seen["inbox" if codec_type is records.InboxItem else "record"] += 1
    assert seen == {"record": 67, "inbox": 18}


def test_all_canonical_codec_rejections_reject_without_historical_fallback():
    from support.contract import fixture_root

    corpus = json.loads((fixture_root() / "session_invalid.json").read_text())
    rejected_layers = 0
    for vector in corpus["vectors"]:
        if vector["layer"] != "codec":
            continue
        body = base64.b64decode(vector["bytes_base64"], validate=True)
        with pytest.raises(codec.RecordError):
            records.Record.parse(body)
        rejected_layers += 1
    assert rejected_layers


def test_trusted_storage_producer_capture_cases_are_enforced_by_the_python_codec():
    from support.contract import fixture_root

    fixture = json.loads((fixture_root() / "trusted_storage_s1.json").read_text())
    assert fixture["status"] == "producer-captured"
    assert fixture["provenance"]["producer_capture"] is True
    assert fixture["revision"] == fixture["provenance"]["source_revision"]
    assert len(fixture["revision"]) == 40
    assert fixture["record_schema_version"] == "S1"
    assert fixture["inbox_schema_version"] == "S1"
    bases = {case["name"]: case["wire"] for case in fixture["accepted_cases"]}
    for case in fixture["accepted_cases"]:
        body = base64.b64decode(case["bytes_base64"], validate=True)
        assert sha256(body).hexdigest() == case["sha256"]
        assert case["canonical_json_base64"] == case["bytes_base64"]
        assert case["canonical_json_utf8_bytes"] == len(body)
        assert case["wire"]["schema_version"] == "S1"
        assert case["wire"]["feature_level"] == 1
        codec_type = records.InboxItem if "actor" in case["wire"] else records.Record
        parsed = codec_type.parse(body)
        assert parsed.to_dict() == case["wire"]
        assert parsed.encode() == body
        assert case["content_refs"]
    for case in fixture["rejected_cases"]:
        wire = json.loads(json.dumps(bases[case["base"]]))
        for patch in case["patch"]:
            target = wire
            for part in patch["path"][:-1]:
                target = target[part]
            if patch.get("delete"):
                del target[patch["path"][-1]]
            else:
                target[patch["path"][-1]] = patch["set"]
        codec_type = records.InboxItem if "actor" in wire else records.Record
        with pytest.raises(codec.RecordError):
            codec_type.parse(json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode())


def test_chain_digest_is_not_part_of_logical_record_identity():
    wire = records.SessionSpec("s", principal(), ref("workspace"), attributes=ref("attributes")).record().to_dict()
    wire["chain_digest"] = "0" * 64
    with pytest.raises(codec.RecordError, match="invalid session schema"):
        records.Record.parse(json.dumps(wire).encode())


@pytest.mark.parametrize("version", [1, 2, 3, True])
def test_removed_record_versions_have_no_decoder(version):
    wire = records.SessionSpec("s", principal(), ref("workspace"), attributes=ref("attributes")).record().to_dict()
    wire["schema_version"] = version
    with pytest.raises(codec.RecordError, match="invalid session schema"):
        records.Record.parse(json.dumps(wire).encode())


@pytest.mark.parametrize("version", [1, 2, 3, True])
def test_removed_inbox_versions_have_no_decoder(version):
    wire = records.InboxItem("i", "user", {"content": ref("go")}, actor=principal()).to_dict()
    wire["schema_version"] = version
    with pytest.raises(codec.RecordError, match="invalid session schema"):
        records.InboxItem.parse(json.dumps(wire).encode())


@pytest.mark.parametrize("field", ["workspace", "attributes"])
def test_creation_content_slots_reject_plaintext(field):
    wire = records.SessionSpec("s", principal(), ref("workspace"), attributes=ref("attributes")).record().to_dict()
    wire["payload"][field] = {"plaintext": "CUSTOMER_CONTENT"}
    with pytest.raises(codec.RecordError, match="invalid session schema"):
        records.Record.parse(json.dumps(wire).encode())
