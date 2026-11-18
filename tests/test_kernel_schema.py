"""Authoritative session schemas and their compiler load without the outer SDK."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vv_agent.kernel import _record_schema
from vv_agent.kernel._schema import _compile_check


def test_kernel_schema_import_is_isolated_without_site_packages(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from pathlib import Path

        source = Path(sys.argv[1])
        sys.path.insert(0, str(source))
        before = set(sys.modules)

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "vv_agent" or fullname == "vv_agent.kernel" or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        import vv_agent

        assert "Agent" not in vars(vv_agent)
        assert {"Agent", "AppServer", "Runner", "VvLlmClient", "skills"} <= set(dir(vv_agent))
        try:
            vv_agent.missing_export
        except AttributeError:
            pass
        else:
            raise AssertionError("unknown root exports must raise AttributeError")

        import vv_agent.kernel
        from vv_agent.kernel import _record_schema, _schema

        assert Path(vv_agent.__file__) == source / "vv_agent" / "__init__.py"
        assert Path(vv_agent.kernel.__file__) == source / "vv_agent" / "kernel" / "__init__.py"
        assert Path(_schema.__file__) == source / "vv_agent" / "kernel" / "_schema.py"
        assert Path(_record_schema.__file__) == source / "vv_agent" / "kernel" / "_record_schema.py"
        assert _schema._compile_check.__module__ == "vv_agent.kernel._schema"
        check = _schema._compile_check({"type": "integer", "minimum": 0})
        assert check(0) and not check(True) and not check(1.0)
        assert len(_record_schema.PAYLOADS) == 14
        assert len(_record_schema.INPUT_PAYLOADS) == 9
        assert len(_record_schema.BOUNDARY_DATA) == 7
        assert len(_record_schema.HANDLE["oneOf"]) == 4
        ref = {"content_id": "content/test", "mac": "a" * 64, "size": 1,
               "key_id": "key/test", "labels": {"source": [], "classification": "internal"}}
        principal = {"tenant_id": "default", "subject_id": "p", "subject_kind": "user", "attributes": ref}
        envelope = {
            "schema_version": "S1", "feature_level": 1, "record_id": "s", "kind": "session_created",
            "session_id": "s", "turn_id": None, "operation_id": None, "attempt": None,
            "payload": {"principal": principal, "workspace": ref, "parent_session_id": None,
                        "parent_operation_id": None, "attributes": ref}
        }
        assert _record_schema._CHECKS[id(_record_schema.RECORD_SCHEMA)](envelope)
        assert _record_schema._CHECKS[id(_record_schema.PAYLOADS["session_created"])](envelope["payload"])
        assert not _record_schema._CHECKS[id(_record_schema.RECORD_SCHEMA)](envelope | {"extra": True})
        from vv_agent.kernel._canonical_json import MAX_WIRE_INTEGER, canonical_json_bytes
        from vv_agent.kernel._message_values import COMPACTION_METADATA_KEY, Message, ToolArtifactRef, ToolResultCursor

        artifact = ToolArtifactRef(".vv-agent/artifacts/result.txt", "text/plain", "utf-8", MAX_WIRE_INTEGER, "a" * 64)
        cursor = ToolResultCursor("read_file", "result.txt", MAX_WIRE_INTEGER, "a" * 64)
        evidence = {
            "artifacts": [{"tool_call_id": "call", "tool_name": "tool", "arguments": "{}", "artifact_ref": artifact.to_dict()}],
            "cursors": [{"tool_call_id": "call", "tool_name": "tool", "arguments": "{}", "cursor": cursor.to_dict()}],
        }
        payload = {
            "role": "tool", "content": "result", "artifact_ref": artifact.to_dict(),
            "metadata": {COMPACTION_METADATA_KEY: evidence, "wide": (1 << 64) - 1},
        }
        message = Message.from_dict(payload)
        assert type(message) is Message
        assert type(message.artifact_ref) is ToolArtifactRef
        assert message.to_dict() == payload
        assert ToolResultCursor.from_dict(cursor.to_dict()) == cursor
        try:
            canonical_json_bytes(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("Message integers must not weaken I-JSON digest validation")
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._schema", "vv_agent.kernel._record_schema",
            "vv_agent.kernel._canonical_json", "vv_agent.kernel._message_values", "vv_agent.kernel._message_codec"
        }
        unexpected = {
            name for name in set(sys.modules) - before
            if name.split(".")[0] not in sys.stdlib_module_names
            and name != "vv_agent" and not name.startswith("vv_agent.kernel")
        }
        assert not unexpected, unexpected
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_records_use_the_extracted_compiler():
    from vv_agent.kernel import _record_codec

    assert _record_codec._CHECKS is _record_schema._CHECKS
    assert all(check.__module__ == "vv_agent.kernel._schema" for check in _record_schema._CHECKS.values())


def test_record_schemas_and_checks_have_one_owner():
    from vv_agent.session import records

    for name in (
        "APP_SERVER_ATTRIBUTES",
        "BOUNDARY_DATA",
        "CHILD_ADMISSION",
        "HANDLE",
        "INPUT_PAYLOADS",
        "INPUT_SCHEMA",
        "MODEL_BINDING",
        "MODEL_SELECTION",
        "PAYLOADS",
        "RECORD_SCHEMA",
        "REQUEST_SESSION_METADATA",
        "SEED",
        "TASK_SESSION_METADATA",
        "nullable",
    ):
        assert getattr(records, name) is getattr(_record_schema, name), name
    assert _record_schema._CHECKS.keys() == {id(s) for s in _record_schema._SCHEMAS}
    assert _record_schema.PAYLOADS["input_applied"]["properties"]["input"] is _record_schema.INBOX_ITEM_SCHEMA
    assert _record_schema.INPUT_PAYLOADS["provider_evidence"]["properties"]["handle"] is _record_schema.PROVIDER_HANDLE


def test_contract_41_logical_schema_has_closed_s1_shapes():
    assert set(_record_schema.RECORD_SCHEMA["properties"]) == {
        "schema_version",
        "feature_level",
        "record_id",
        "kind",
        "session_id",
        "turn_id",
        "operation_id",
        "attempt",
        "payload",
    }
    assert "chain_digest" not in _record_schema.RECORD_SCHEMA["properties"]
    assert _record_schema.RECORD_SCHEMA["properties"]["schema_version"] == {"const": "S1"}
    assert _record_schema.RECORD_SCHEMA["properties"]["feature_level"] == {"const": 1, "type": "integer"}
    assert set(_record_schema.INPUT_SCHEMA["properties"]) == {
        "schema_version",
        "feature_level",
        "actor",
        "admission_proof",
        "input_id",
        "kind",
        "target_turn_id",
        "generation",
        "available_ms",
        "payload",
    }
    assert _record_schema.INPUT_SCHEMA["properties"]["schema_version"] == {"const": "S1"}
    assert _record_schema.INPUT_SCHEMA["properties"]["actor"] is _record_schema.PRINCIPAL
    assert _record_schema.INPUT_SCHEMA["properties"]["admission_proof"] is _record_schema.ADMISSION_PROOF
    assert set(_record_schema.INPUT_SCHEMA_LEVEL_2["properties"]) == {
        "schema_version",
        "feature_level",
        "actor",
        "admission_proof",
        "input_id",
        "kind",
        "target_turn_id",
        "generation",
        "available_ms",
        "requirements",
        "payload",
    }
    assert _record_schema.INPUT_SCHEMA_LEVEL_2["properties"]["feature_level"] == {"const": 2, "type": "integer"}
    assert _record_schema.INPUT_SCHEMA_LEVEL_2["properties"]["requirements"] is _record_schema.REQUIREMENTS
    assert _record_schema.INBOX_ITEM_SCHEMA["oneOf"] == [
        _record_schema.INPUT_SCHEMA,
        _record_schema.INPUT_SCHEMA_LEVEL_2,
    ]
    assert _record_schema.RECORD_ITEM_SCHEMA["oneOf"] == [
        _record_schema.RECORD_SCHEMA,
        _record_schema.RECORD_SCHEMA_LEVEL_2,
    ]
    assert _record_schema.RECORD_SCHEMA_LEVEL_2["properties"]["kind"] == {"type": "string", "enum": ["session_created"]}
    assert _record_schema.SESSION_CREATED_LEVEL_2["properties"]["bootstrap"] is _record_schema.REQUIREMENTS
    assert "bootstrap" not in _record_schema.PAYLOADS["session_created"]["properties"]
    assert _record_schema.PRINCIPAL["properties"]["subject_kind"]["enum"] == ["user", "service", "agent"]
    assert set(_record_schema.CONTENT_REF["properties"]) == {"content_id", "mac", "size", "key_id", "labels"}
    for kind in ("op_planned", "op_prepared", "op_completed"):
        assert _record_schema.PAYLOADS[kind]["properties"]["policy_decision"] is _record_schema.POLICY_DECISION
        assert "policy_decision" in _record_schema.PAYLOADS[kind]["required"]
    environment = _record_schema.ENVIRONMENT_BINDING["anyOf"][0]["properties"]
    assert environment["operational_policy"]["anyOf"][0] is _record_schema.CONTENT_REF
    assert environment["operational_policy"]["anyOf"][1] == {"type": "null"}
    assert _record_schema.PAYLOADS["turn_started"]["properties"]["tokenizer_binding"] is _record_schema.TOKENIZER_BINDING
    assert set(_record_schema.USAGE_COUNTS["properties"]) == {"tokens", "budget"}
    assert set(_record_schema.MODEL_CAPABILITY["properties"]) == {
        "backend",
        "model",
        "context_length",
        "max_output_tokens",
    }
    assert set(_record_schema.DEFINITION_MODEL_BINDING["properties"]) == {
        "backend",
        "model",
        "context_length",
        "max_output_tokens",
        "internal",
    }
    assert "internal" not in _record_schema.DEFINITION_MODEL_BINDING["required"]
    assert (
        _record_schema.DEFINITION_MODEL_BINDING["properties"]["internal"]["additionalProperties"]
        is _record_schema.MODEL_CAPABILITY
    )


@pytest.mark.parametrize("value,expected", [(0, True), (1, True), (-1, False), (True, False), (1.0, False), ("1", False)])
def test_schema_integer_is_exact(value, expected):
    assert _compile_check({"type": "integer", "minimum": 0})(value) is expected


def test_schema_preserves_closed_objects_and_union_cardinality():
    check = _compile_check(
        {
            "type": "object",
            "properties": {"kind": {"const": "ready"}, "count": {"type": "integer", "minimum": 0}},
            "required": ["kind"],
            "additionalProperties": False,
        }
    )
    assert check({"kind": "ready"})
    assert check({"kind": "ready", "count": 0})
    assert not check({"kind": "ready", "count": True})
    assert not check({"kind": "ready", "extra": 0})
    assert not check({"count": 0})
    ambiguous = _compile_check({"oneOf": [{"type": "string"}, {"const": "ready"}]})
    assert ambiguous("other")
    assert not ambiguous("ready")


def test_schema_hash_rejects_trailing_newline():
    check = _compile_check({"type": "string", "pattern": "^[0-9a-f]{64}$", "maxLength": 64})
    assert check("a" * 64)
    assert not check("a" * 64 + "\n")


def test_schema_rejects_unsupported_vocabulary():
    with pytest.raises(ValueError, match="unsupported session schema"):
        _compile_check({"type": "string", "maxLength": 1})


@pytest.mark.parametrize(
    "first_import",
    [
        "import vv_agent.session.app_server",
        "from vv_agent import AppServer",
        "from vv_agent.app_server import AppServer",
        "import vv_agent.app_server.host",
        "import vv_agent.app_server.client",
    ],
)
def test_app_server_import_order_preserves_real_exports(tmp_path, first_import):
    source = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
{first_import}

import vv_agent
import vv_agent.app_server
from vv_agent.app_server import *
from vv_agent.app_server.server import AppServer as Server
from vv_agent.session.app_server import SubmissionConflict, _KernelRunAdapter, _KernelThreadStore

assert Path(vv_agent.__file__) == Path(sys.argv[1]) / "vv_agent" / "__init__.py"
assert vv_agent.AppServer is vv_agent.app_server.AppServer is AppServer is Server
assert vars(vv_agent)["AppServer"] is Server
assert vars(vv_agent.app_server)["AppServer"] is Server
assert all(name in globals() for name in vv_agent.app_server.__all__)
assert SubmissionConflict.__module__ == "vv_agent.session.app_server"
assert _KernelRunAdapter.__module__ == "vv_agent.session.app_server"
assert _KernelThreadStore.__module__ == "vv_agent.session.app_server"
for facade in (vv_agent, vv_agent.app_server):
    assert "AppServer" in dir(facade)
    try:
        facade.missing_export
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown exports must raise AttributeError")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
