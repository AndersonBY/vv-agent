"""Real record producers share the pure codec and concise schema diagnostics."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from vv_agent.kernel import _record_codec as codec
from vv_agent.kernel import _record_schema as schemas
from vv_agent.kernel import _records as core_records
from vv_agent.session import records


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
        value = records.SessionSpec("s", "p", "w").record().to_dict()
        value["payload"]["workspace"] = ["CUSTOMER_CONTENT"]
    else:
        value = records.InboxItem("i", "user", {"content": "go"}).to_dict() | {"available_ms": "CUSTOMER_CONTENT"}
    with pytest.raises(codec.RecordError, match=r"^invalid session schema$"):
        parse(json.dumps(value).encode())


def test_record_and_inbox_keep_specific_semantic_and_settings_errors():
    wire = records.SessionSpec("s", "p", "w").record().to_dict()
    wire["record_id"] = "wrong"
    with pytest.raises(codec.RecordError, match=r"^record_id does not match semantic position$"):
        records.Record.parse(json.dumps(wire).encode())
    selection = {"model": "m", "route": {}, "model_settings": {"tool_choice": {"type": "function", "function": {"name": " "}}}}
    with pytest.raises(codec.RecordError, match=r"^named tool_choice requires a non-empty function name$"):
        records.InboxItem("i", "model_selection", {"selection": selection}).encode()


def test_canonical_rejection_keeps_precedence_and_original_cause():
    with pytest.raises(codec.RecordError, match="I-JSON") as actual:
        codec.validate({"invalid": float("nan")}, schemas.RECORD_SCHEMA)
    assert isinstance(actual.value.__cause__, ValueError)
    assert str(actual.value) == str(actual.value.__cause__)
    with pytest.raises(codec.RecordError, match=r"^invalid session schema$"):
        codec.validate({"invalid": float("nan")}, schemas.RECORD_SCHEMA, canonical=False)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_accepted_bom_inputs_keep_canonical_bytes(encoding):
    expected = records.SessionSpec("s", "p", "w").record().encode()
    parsed = records.Record.parse(expected.decode().encode(encoding))
    assert parsed.encode() == expected


def test_real_diagnostic_producer_is_deterministic():
    script = Path(__file__).resolve().parents[1] / "scripts/record_codec_fixtures.py"
    produce = runpy.run_path(str(script))["produce"]
    first, second = produce(), produce()
    assert first == second
    assert first["app_server_error"] == {"code": -32603, "message": "invalid session schema"}
    assert first["provider_reason"] == "provider_result_invalid"
    assert first["model_selection_reason"] == "model_selection_unresolvable"
    assert "CUSTOMER_CONTENT" not in json.dumps(first)


def test_app_server_schema_errors_and_their_chains_do_not_dump_values(monkeypatch):
    errors: list[BaseException] = []
    original = core_records.validate

    def observed(*args, **kwargs):
        try:
            original(*args, **kwargs)
        except codec.RecordError as error:
            errors.append(error)
            raise

    monkeypatch.setattr(core_records, "validate", observed)
    script = Path(__file__).resolve().parents[1] / "scripts/record_codec_fixtures.py"
    error = runpy.run_path(str(script))["app_server_error"]()
    assert error == {"code": -32603, "message": "invalid session schema"}
    assert errors
    seen = set()
    while errors:
        current = errors.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        text = str(current) + repr(current.args) + repr(vars(current))
        for excluded in ("CUSTOMER_CONTENT", "additionalProperties", "Failed validating", "properties"):
            assert excluded not in text
        errors.extend(e for e in (current.__cause__, current.__context__) if e is not None)


def test_real_diagnostic_producer_matches_canonical_fixture():
    from support.contract import fixture_root

    fixture = fixture_root() / "session_record_diagnostics.json"
    assert fixture.is_file(), "Contract 39 fixture required; use VV_AGENT_CONTRACT_SOURCE until immutable adoption"
    script = Path(__file__).resolve().parents[1] / "scripts/record_codec_fixtures.py"
    actual = json.dumps(runpy.run_path(str(script))["produce"](), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    assert actual.encode() == fixture.read_bytes()
