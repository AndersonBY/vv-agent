"""Authoring fixtures stay outside the v23 snapshot and prove real producer bytes."""

import base64
import json
import re
import runpy
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Event

import pytest
from jsonschema import Draft202012Validator

from vv_agent import Agent, RunConfig, Runner
from vv_agent.approval import ApprovalBroker
from vv_agent.events import ApprovalRequestedEvent, event_from_dict
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem, Record, RecordError
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.store import StoredRecord
from vv_agent.session.surfaces import _SessionKernel
from vv_agent.session.tracing import project_spans
from vv_agent.tools.function import function_tool
from vv_agent.types import AgentResult, LLMResponse, ToolCall

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "session_kernel_fixtures.py"
AUTHOR = runpy.run_path(str(SCRIPT))
REPLACEMENTS = runpy.run_path(str(SCRIPT.with_name("_session_kernel_replacements.py")))
CURATION = runpy.run_path(str(SCRIPT.with_name("_session_kernel_curation.py")))
CHECKS = runpy.run_path(str(SCRIPT.with_name("_session_kernel_checks.py")))
FILES = {
    "session_record.schema.json",
    "session_inbox.schema.json",
    "session_records.jsonl",
    "session_inbox.jsonl",
    "session_codec_vectors.json",
    "session_invalid.json",
    "session_semantics.json",
    "session_recovery.json",
    "session_projection.json",
    "session_compaction.json",
    "app_server_protocol.json",
}
FILES.update(REPLACEMENTS["REPLACE"])


def checked_bytes(vector):
    raw = base64.b64decode(vector["bytes_base64"], validate=True)
    assert sha256(raw).hexdigest() == vector["sha256"]
    return raw


@pytest.fixture(scope="module")
def generated_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("kernel-fixtures")
    directories = [root / "first", root / "second"]
    coverage_runs = []
    reports = []
    for output in directories:
        generated = subprocess.run(
            [sys.executable, str(SCRIPT), "--output", str(output)], capture_output=True, text=True, timeout=600
        )
        assert generated.returncode == 0, generated.stderr
        assert {p.name for p in output.iterdir()} == FILES
        report = json.loads(generated.stdout)
        reports.append(report)
        coverage_runs.append(report["coverage"])
        assert sum(p.stat().st_size for p in output.iterdir()) <= 3_000_000
        assert all(p.stat().st_size <= 512_000 for p in output.iterdir())
    assert coverage_runs[0] == coverage_runs[1]
    assert set(coverage_runs[0]) == FILES
    first, second = directories
    for name in FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    values = {}
    for name in FILES:
        raw = (first / name).read_text()
        values[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
    assert reports[0]["self_checks"] == reports[1]["self_checks"]
    return directories, coverage_runs, values, reports[0]["self_checks"]


def validate_outputs(values):
    return CHECKS["validate_outputs"](values, REPLACEMENTS["BASE"], REPLACEMENTS["KEEP"], REPLACEMENTS["REPLACE"])


def test_session_kernel_fixtures_deterministic_and_revalidated(generated_fixtures):
    directories, coverage_runs, _, _ = generated_fixtures
    first, _second = directories

    def read(name):
        return json.loads((first / name).read_text())

    for name in FILES:
        raw = (first / name).read_text()
        value = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
        keys = coverage_runs[0][name]
        assert keys["before"] == keys["after"] == sorted(CURATION["coverage_keys"](name, value)), name

    replaced = {}
    for name in REPLACEMENTS["REPLACE"]:
        raw = (first / name).read_text()
        for retired in REPLACEMENTS["RETIRED_FIELDS"] | REPLACEMENTS["RETIRED_KINDS"] | REPLACEMENTS["REMOVED_MEMBERS"]:
            assert re.search(r"(?<![A-Za-z0-9_])" + re.escape(retired) + r"(?![A-Za-z0-9_])", raw) is None, (name, retired)
        for symbol in REPLACEMENTS["REMOVED"]:
            # Rendered prompts retain ordinary words such as "Session Memory".
            assert re.search(r'"' + re.escape(symbol) + r'"|vv_agent\.[A-Za-z_.]*\b' + re.escape(symbol) + r"\b", raw) is None, (
                name,
                symbol,
            )
        replaced[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
    assert set(REPLACEMENTS["validate_replacements"](replaced, AUTHOR["independent_bytes"])) == set(REPLACEMENTS["REPLACE"])

    corpus = read("session_codec_vectors.json")
    coverage = corpus["coverage"]
    assert len(coverage["record_kinds"]) == 14
    assert len(coverage["inbox_kinds"]) == 8
    assert len(coverage["boundary_stages"]) == 7
    assert len(coverage["handle_variants"]) == 4
    assert all(coverage["optional_fields"].values())
    assert coverage["optional_fields"]["boundary.output_checked.partial_output"] > 0
    validators = {kind: Draft202012Validator(read(f"session_{kind}.schema.json")) for kind in ("record", "inbox")}
    sessions, records, inputs = defaultdict(list), {}, {}
    independent = AUTHOR["independent_bytes"]([v["wire"] for v in corpus["vectors"]])
    for vector, expected in zip(corpus["vectors"], independent, strict=True):
        raw = checked_bytes(vector)
        assert raw == expected
        validators[vector["type"]].validate(vector["wire"])
        parsed = (Record if vector["type"] == "record" else InboxItem).parse(raw)
        assert parsed.encode() == raw
        if isinstance(parsed, Record):
            assert vector["record_id"] == parsed.record_id == AUTHOR["expected_id"](vector["wire"])
            records[(parsed.session_id, parsed.record_id)] = parsed
            sessions[parsed.session_id].append(
                StoredRecord(parsed, vector["seq"], vector["commit_id"], vector["writer_epoch"], vector["created_ms"])
            )
        else:
            inputs[(vector["session_id"], parsed.input_id)] = parsed
    for kind, name in (("record", "session_records.jsonl"), ("inbox", "session_inbox.jsonl")):
        assert (first / name).read_bytes() == b"".join(checked_bytes(v) + b"\n" for v in corpus["vectors"] if v["type"] == kind)

    for wire in read("session_invalid.json")["source_records"]:
        record = Record(**wire)
        assert record.encode() == AUTHOR["independent_bytes"]([wire])[0]
        records[(record.session_id, record.record_id)] = record
    invalid = read("session_invalid.json")["vectors"]
    classes = {v["rejection_class"] for v in invalid}
    assert {
        "missing_version",
        "stale_version",
        "unknown_version",
        "malformed_version",
        "unknown_kind",
        "unknown_field",
        "unknown_enum",
        "incorrect_nullability",
        "wrong_identity",
        "embedded_digest",
        "lone_surrogate",
        "non_string_key",
        "nonfinite_number",
        "unsafe_integer",
        "non_json_host_object",
        "duplicate_member",
        "nonobject_envelope",
        "cross_record_order",
        "dependencies",
        "request_drift",
        "authorization",
        "stale_generation",
        "terminal_revival",
        "provider_binding_drift",
        "provider_authentication",
        "child_authentication",
        "trailing_newline_hash",
    } <= classes
    for vector in invalid:
        raw = checked_bytes(vector)
        layer = vector["layer"]
        if layer in {"codec", "inbox_codec"}:
            with pytest.raises((RecordError, ValueError, TypeError)):
                (Record if layer == "codec" else InboxItem).parse(raw)
        elif layer == "constructor":
            wire = json.loads(raw)
            key, value = (1, "value") if vector["mutation"] == "non_string_key" else ("host", object())
            wire["payload"]["request"][key] = value
            with pytest.raises((RecordError, ValueError, TypeError)):
                Record(**wire).encode()
        elif layer == "fold":
            sid = vector["session_id"]
            log = [records[(sid, identity)] for identity in vector["prefix_ids"]] + [Record.parse(raw)]
            consumed = [InboxItem(**r.payload["input"]) for r in log if r.kind == "input_applied"]
            with pytest.raises(TransitionError, match=re.escape(vector["reason"])):
                fold(log, consumed_inputs=consumed)
        else:
            assert layer == "admission"
            item = InboxItem.parse(raw)
            audit = records[(vector["session_id"], vector["audit_record_id"])]
            assert audit.payload["disposition"] == "rejected" and audit.payload["reason"] == vector["reason"]
            assert InboxItem(**audit.payload["input"]).encode() == item.encode()

    for name, key in (
        ("session_semantics.json", "cases"),
        ("session_recovery.json", "cases"),
        ("session_projection.json", "sessions"),
        ("app_server_protocol.json", "transcripts"),
    ):
        for vector in read(name)[key]:
            value = {k: v for k, v in vector.items() if k not in {"bytes_base64", "sha256", "record_id"}}
            assert checked_bytes(vector) == AUTHOR["independent_bytes"]([value])[0]
    for projection in read("session_projection.json")["sessions"]:
        rows = tuple(
            StoredRecord(Record(**row["wire"]), row["seq"], row["commit_id"], row["writer_epoch"], row["created_ms"])
            for row in read("session_projection.json")["source_records"][projection["session_id"]]
        )
        expected_events = {e.event_id: e.to_dict() for e in project_records(rows)}
        assert projection["events"] == [expected_events[e["event_id"]] for e in projection["events"]]
        expected_spans = {
            (seq, method, span.span_id): {"seq": seq, "method": method, "span": span.to_dict()}
            for seq, method, span in project_spans(rows)
        }
        assert projection["spans"] == [expected_spans[(s["seq"], s["method"], s["span"]["span_id"])] for s in projection["spans"]]
        for event in projection["events"]:
            assert event["version"] == "v6"
            assert event_from_dict(event, _kernel=True).to_dict() == event
        for prefix in projection["prefix_states"]:
            log = [r.record for r in rows if r.seq <= prefix["seq"]]
            assert [r.seq for r in rows if r.seq <= prefix["seq"]] == list(range(1, prefix["seq"] + 1))
            consumed = [InboxItem(**r.payload["input"]) for r in log if r.kind == "input_applied"]
            state = fold(log, consumed_inputs=consumed)
            assert (state.phase, state.active_turn_id, state.closed, state.terminal_seq) == (
                prefix["phase"],
                prefix["active_turn_id"],
                prefix["closed"],
                prefix["terminal_seq"],
            )
    for recovery in read("session_recovery.json")["cases"]:
        if "receipt_id" in recovery:
            receipt = records[(recovery["session_id"], recovery["receipt_id"])]
            assert receipt.kind == "op_completed"
    for vector in read("session_compaction.json")["vectors"]:
        parsed = Record.parse(checked_bytes(vector))
        assert parsed.record_id == vector["record_id"] == AUTHOR["expected_id"](vector["wire"])
        assert parsed.to_dict() == vector["wire"]

    app = read("app_server_protocol.json")
    from vv_agent.app_server.schema import export_schema_bundles

    exports = export_schema_bundles(_kernel=True)
    schemas = {name: json.loads(source) for name, source in app["schemas"]["jsonSchema"].items()}
    assert schemas == {name: json.loads(source) for name, source in exports["jsonSchema"].items()}
    assert set(app["schemas"]["typescript"]) == set(exports["typescript"])
    for name, reference in app["schemas"]["typescript"].items():
        source = app["schemas"][reference["source_ref"]]
        assert source == exports["typescript"][name]
        assert sha256(source.encode()).hexdigest() == reference["sha256"]
    envelope = Draft202012Validator(schemas["JsonRpcMessage"])
    results = {
        "initialize": "InitializeResponse",
        "thread/start": "ThreadStartResponse",
        "thread/read": "ThreadReadResponse",
        "thread/resume": "ThreadResumeResponse",
        "turn/start": "TurnStartResponse",
        "turn/resume": "TurnResumeResponse",
    }
    for entry in app["transcripts"]:
        request = entry.get("request")
        if "rejected_request" in entry:
            request = json.loads(checked_bytes(entry["rejected_request"]))
        if request is not None:
            errors = list(Draft202012Validator(schemas["ClientRequest"]).iter_errors(request))
            if errors:
                assert entry["responses"][0]["error"]["code"] == -32602
                assert "checkpointKey" in request["params"]
            for response in entry["responses"]:
                envelope.validate(response)
                if "result" in response and request["method"] in results:
                    Draft202012Validator(schemas[results[request["method"]]]).validate(response["result"])
        for notification in entry.get("notifications", []):
            envelope.validate(notification)
    assert app["facts"]["observer_cannot_approve"] and app["facts"]["timeout_at_absolute_deadline"]


def test_host_interaction_wire_references_use_current_owner(generated_fixtures):
    from vv_agent.interaction import HostInteractionRequest
    from vv_agent.runtime.controller import HostInteractionOutcome

    _, _, values, _ = generated_fixtures
    capabilities = {c["python"]: c for domain in values["public_api.json"]["domains"] for c in domain["capabilities"]}
    app = values["app_server_protocol.json"]
    for name, decoder, field in (
        ("HostInteractionRequest", HostInteractionRequest, "request"),
        ("HostInteractionOutcome", HostInteractionOutcome, "outcome"),
    ):
        reference = capabilities["vv_agent.runtime." + name]["wire"]
        assert reference == "fixtures/app_server_protocol.json#/host_interaction_values/" + field
        wire = CHECKS["resolve_pointer"](app, reference.split("#", 1)[1])
        assert decoder.from_dict(wire).to_dict() == wire


def test_terminal_optional_fields_come_from_kernel_schema(generated_fixtures):
    from vv_agent.app_server.schema import export_schema_bundles

    _, _, values, _ = generated_fixtures
    schema = json.loads(export_schema_bundles(_kernel=True)["jsonSchema"]["ServerNotification"])["$defs"]["TurnCompletedParams"]
    optional = values["app_server_observable.json"]["terminal"]["optionalFieldsOmittedWhenAbsent"]
    assert schema["additionalProperties"] is False
    assert set(optional) == set(schema["properties"]) - set(schema["required"])
    assert "waitReason" in optional and "interruption" not in optional


def test_closed_thread_resume_transcripts_cover_execution_and_snapshot(generated_fixtures):
    _, _, values, _ = generated_fixtures
    app = values["app_server_protocol.json"]
    rejected = []
    snapshots = []
    for row in app["transcripts"]:
        request = row.get("request", {})
        if request.get("method") != "thread/resume" or request["params"]["threadId"] != "thread_1":
            continue
        response = row["responses"][0]
        if "error" in response:
            assert response["error"] == {"code": -32602, "message": "Thread is closed"}
            rejected.append(request["params"].get("subscribe", "default"))
        else:
            assert request["params"]["subscribe"] is False
            assert response["result"]["thread"]["status"] == "closed"
            snapshots.append(row)
    assert rejected == [True, "default"]
    assert len(snapshots) == 1


def test_prompt_definition_descriptor_preserves_rendered_bytes(generated_fixtures):
    _, _, values, _ = generated_fixtures
    prompt = values["prompt_bundle.json"]
    descriptor = prompt["run_scope"]["run_definition"]
    assert descriptor == {
        "carrier": "turn_started.definition",
        "field": "task.prompt_bundle",
        "validation_owner": values["run_definition.json"]["top_level_field_policy"]["validation_owner"],
    }
    original = REPLACEMENTS["load"]("prompt_bundle.json")
    assert prompt["scenarios"] == original["scenarios"]
    assert prompt["stable_hash_vectors"] == original["stable_hash_vectors"]


def test_generated_output_self_checks_revalidate_all_files(generated_fixtures):
    _, _, values, report = generated_fixtures
    assert validate_outputs(values) == report
    assert report["references"] > 0 and report["optional_lists"] == 10
    assert {v["value"] for v in report["rejected_versions"]} == {"unsupported"}


@pytest.mark.parametrize(
    "reference",
    [
        "fixtures/missing.json",
        "fixtures/MISSING.json",
        "fixtures/../model_ref.json",
        "fixtures/controller_command.json",
        "memory_local.json#/missing",
        "memory_local.json#summary_compaction",
        "session_records.jsonl#/999999",
        "session_records.jsonl#/01",
        "session_records.jsonl#/0/~2",
    ],
)
def test_generated_output_self_checks_reject_broken_references(generated_fixtures, reference):
    values = deepcopy(generated_fixtures[2])
    values["public_api.json"]["check_reference"] = reference
    with pytest.raises(AssertionError):
        validate_outputs(values)


def test_generated_output_self_checks_resolve_keep_and_escaped_pointers(generated_fixtures):
    values = deepcopy(generated_fixtures[2])
    values["public_api.json"]["a/b~c"] = "fixture-value"
    values["public_api.json"]["check_references"] = [
        "fixtures/model_ref.json#/valid/0",
        "public_api.json#/a~1b~0c",
        "public_api.json#%2Fa~1b~0c",
    ]
    assert validate_outputs(values)["references"] == generated_fixtures[3]["references"] + 3


@pytest.mark.parametrize(
    "version",
    [
        "vv-agent.run-definition.v5",
        "vv-agent.model-call.v1",
        "vv-agent.task-token-usage.v2",
        "vv-agent-public-api-v7",
        "vv-agent.checkpoint.v12",
        "vv-agent.model-call.v999",
        "v2",
    ],
)
def test_generated_output_self_checks_reject_superseded_versions(generated_fixtures, version):
    values = deepcopy(generated_fixtures[2])
    values["prompt_bundle.json"]["run_scope"]["run_definition"]["schema_version"] = version
    with pytest.raises(AssertionError, match="discriminator"):
        validate_outputs(values)


@pytest.mark.parametrize("field,value", [("protocolVersion", "v1"), ("wire_version", "v5"), ("schema_version", 7)])
def test_generated_output_self_checks_reject_stale_protocol_event_inventory(generated_fixtures, field, value):
    values = deepcopy(generated_fixtures[2])
    values["public_api.json"][field] = value
    with pytest.raises(AssertionError, match="discriminator"):
        validate_outputs(values)


@pytest.mark.parametrize(
    "name,pointer",
    [
        ("app_server_observable.json", "/terminal/optionalFieldsOmittedWhenAbsent"),
        ("session_codec.json", "/message_contract/optional_fields"),
        ("bounded_tool_result.json", "/result_contract/optional_fields"),
        ("bounded_tool_result.json", "/result_contract/canonical_writer_normalization/omit_empty_optional_fields"),
        ("prompt_bundle.json", "/section_contract/optional_fields"),
        ("result_public.json", "/agent_result_wire/optional_fields"),
        ("llm_stream_projection.json", "/mappings/assistant_delta/optional_source_fields"),
    ],
)
def test_generated_output_self_checks_reject_unknown_optional_fields(generated_fixtures, name, pointer):
    values = deepcopy(generated_fixtures[2])
    CHECKS["resolve_pointer"](values[name], pointer).append("interruption")
    with pytest.raises(AssertionError, match="unknown optional fields"):
        validate_outputs(values)


def test_generated_output_self_checks_reject_unknown_omission_flag(generated_fixtures):
    values = deepcopy(generated_fixtures[2])
    values["session_codec.json"]["message_contract"]["interruption_omitted_when_absent"] = True
    with pytest.raises(AssertionError, match="unknown omitted field"):
        validate_outputs(values)


def test_private_runner_subscribers_and_observer_failure_keep_committed_result():
    entered, released = Event(), Event()

    def complete(_request):
        entered.set()
        assert released.wait(5)
        return LLMResponse("done")

    def broken_observer(_event):
        raise RuntimeError("observer unavailable")

    kernel = _SessionKernel()
    try:
        provider = ScriptedModelProvider.from_steps("test", "m", [complete])
        handle = Runner.start(
            Agent("fixture", "Answer."),
            "go",
            run_config=RunConfig(model_provider=provider, stream=broken_observer),
            _kernel=kernel,
            _session_id="subscribers",
        )
        assert entered.wait(5)
        first, second = handle.events(), handle.events()
        assert next(first).to_dict() == next(second).to_dict()
        released.set()
        result = handle.result(timeout=5)
        assert result.final_output == "done"
        assert [e.to_dict() for e in first] == [e.to_dict() for e in second]
        assert [e.to_dict() for e in handle.events()] == [e.to_dict() for e in result.events]
        wire = result.raw_result.to_dict()
        assert AgentResult.from_dict(wire, _kernel=True).to_dict() == wire
        with pytest.raises((ValueError, TypeError)):
            AgentResult.from_dict(wire)
    finally:
        released.set()
        kernel.close()


def test_private_shared_state_mutation_does_not_change_the_retained_seed():
    seed = {"nested": {"values": ["seed"]}}

    def before_cycle(_cycle, _messages, shared):
        shared["nested"]["values"].append("local")
        return []

    kernel = _SessionKernel()
    try:
        result = Runner.run_sync(
            Agent("isolation", "Answer."),
            "go",
            _kernel=kernel,
            _session_id="isolation",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
                shared_state=seed,
                before_cycle_messages=before_cycle,
            ),
        )
        state, _, _ = kernel.store.read_state("isolation")
        start = next(iter(state.turns.values())).start
        assert seed == start._task().initial_shared_state == {"nested": {"values": ["seed"]}}
        assert result.raw_result.shared_state == {"nested": {"values": ["seed", "local"]}}
    finally:
        kernel.close()


def test_private_approval_provider_failure_is_a_retained_terminal():
    effects = []

    @function_tool(needs_approval=True)
    def gated() -> str:
        effects.append("ran")
        return "done"

    class FailingApproval:
        def should_request(self, request):
            del request
            return True

        def decide(self, request):
            del request
            raise RuntimeError("approval unavailable")

    broker, kernel = ApprovalBroker(), _SessionKernel()
    try:
        result = Runner.run_sync(
            Agent("approval", "Use gated.", tools=[gated]),
            "go",
            _kernel=kernel,
            _session_id="approval-failure",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("gated", "gated", {})])]),
                approval_provider=FailingApproval(),
                approval_broker=broker,
            ),
        )
        assert result.status.value == "failed" and result.raw_result.error is not None
        assert result.raw_result.error["message"] == "approval unavailable"
        assert not effects
        request = next(e for e in result.events if isinstance(e, ApprovalRequestedEvent))
        assert broker.pending_request(request.request_id) is None
        state, rows, _ = kernel.store.read_state("approval-failure")
        assert state.active_turn_id is None and rows[-1].record.kind == "turn_ended"
        retained = Runner.resume("approval-failure", _kernel=kernel, _turn_id=result.run_id)
        assert retained.to_dict() == result.to_dict()
        assert kernel.store.read_state("approval-failure")[1] == rows
    finally:
        kernel.close()


def test_private_configured_resume_reuses_turn_and_rejects_closed_without_writes():
    kernel = _SessionKernel()
    try:
        provider = ScriptedModelProvider.new("test", "m", [LLMResponse("question"), LLMResponse("answer")])
        runner = Runner.configured(RunConfig(model_provider=provider, no_tool_policy="wait_user"), _kernel=kernel)
        result = runner.run_sync(Agent("fixture", "Answer."), "go", _session_id="resume")
        tid = result.raw_result._kernel_turn_id
        assert tid is not None and result.status.value == "wait_user"
        original = kernel.handles[-1]
        kernel.answer("resume", original.runtime, "reply", "reply")
        resumed = runner.resume("resume", _turn_id=tid)
        assert resumed.raw_result._kernel_turn_id == tid
        assert len(resumed.raw_result.cycles) == 2
        state, rows, _ = kernel.store.read_state("resume")
        assert state.active_turn_id == tid
        with pytest.raises(ValueError, match="turn_id"):
            runner.resume("resume", _turn_id="wrong")
        assert kernel.store.read_state("resume")[1] == rows
        kernel.control("resume", "close", "close", runtime=original.runtime)
        closed = kernel.store.read_state("resume")[1]
        with pytest.raises(ValueError, match="closed"):
            runner.resume("resume", _turn_id=tid)
        assert kernel.store.read_state("resume")[1] == closed
    finally:
        kernel.close()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "go", "name": None},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call", "type": "function", "function": {"name": "echo", "arguments": "[]"}}],
        },
    ],
)
def test_kernel_seed_rejects_invalid_messages_atomically(message):
    kernel = _SessionKernel()
    try:
        with pytest.raises((ValueError, TypeError)):
            kernel.create("invalid-seed", "/fixture", {"seed": {"messages": [message], "shared_state": {}}})
        assert not kernel.store.list_sessions()
    finally:
        kernel.close()


def test_kernel_context_preserves_reasoning_and_removes_empty_assistant():
    from vv_agent.session.context import project_context
    from vv_agent.types import Message

    fixture = json.loads((SCRIPT.parents[1] / "tests/fixtures/parity/assistant_reasoning_history.json").read_text())
    kernel = _SessionKernel()
    try:
        for case in fixture["cases"]:
            sid = case["name"]
            kernel.create(sid, "/fixture", {"seed": {"messages": [Message(**case["message"]).to_dict()], "shared_state": {}}})
            state, rows, _ = kernel.store.read_state(sid)
            messages = project_context(rows, state)
            assert bool(messages) == case["expected"]["retain_in_runtime_history"]
            if messages:
                assert messages[0].content == case["expected"]["visible_content"]
                assert messages[0].reasoning_content == case["expected"]["reasoning_content"]
    finally:
        kernel.close()


@pytest.mark.parametrize("purpose", ["compaction", "session_memory"])
def test_private_memory_route_freezes_separate_provider_and_endpoint(purpose, tmp_path):
    from unittest.mock import patch

    from support import ModelMapProvider

    from vv_agent.config import ResolvedModelConfig
    from vv_agent.llm import ScriptedLLM
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
    from vv_agent.memory import MemoryManager
    from vv_agent.session.kernel import drive
    from vv_agent.types import Message

    primary = ResolvedModelConfig("main", "main-model", "main-model", "main-model", [])
    internal = ResolvedModelConfig("memory", "memory-model", "memory-model", "memory-model", [])
    requests = []

    def response(request):
        requests.append(request)
        return LLMResponse(
            '{"original_user_messages":["request"],"current_work_state":"done"}'
            if purpose == "compaction"
            else '[{"category":"decision","content":"retain goal","importance":8}]',
            raw={"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
        )

    internal_client = VvLlmClient(
        [EndpointTarget("memory-endpoint", "unused", "https://example.invalid")], randomize_endpoints=False
    )
    provider = ModelMapProvider(
        {"main-model": (ScriptedLLM([LLMResponse("done")]), primary), "memory-model": (internal_client, internal)},
        "main-model",
    )
    prefix = "memory_summary" if purpose == "compaction" else "session_memory_extraction"
    kernel = _SessionKernel()
    try:
        config = RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            session_memory_enabled=purpose == "session_memory",
            initial_messages=[Message("user", "request"), Message("assistant", "old facts " * 4000)]
            if purpose == "compaction"
            else None,
            metadata={
                prefix + "_backend": "memory",
                prefix + "_model": "memory-model",
                "session_memory_min_tokens": 1,
                "session_memory_min_text_messages": 1,
                "session_memory_storage_dir": "",
            },
        )
        rt = kernel.runtime(Agent("route", "Be precise.", model="main-model"), config)
        if purpose == "compaction":
            rt.memory_manager = MemoryManager(compact_threshold=1000, keep_recent_messages=1)
        kernel.create("route", "/fixture")
        kernel.push("route", InboxItem("initial", "user", {"content": "go"}))
        with patch.object(VvLlmClient, "complete", side_effect=response):
            drive(kernel.store, "route", runtime=rt, _one_turn=True)
            state, rows, _ = kernel.store.read_state("route")
            assert state.active_turn_id is None
            internal_plans = [r.record for r in rows if r.record.kind == "op_planned" and r.record.payload["purpose"] == purpose]
            assert len(internal_plans) == len(requests) == 1
            plan = internal_plans[0]
            assert plan.payload["request"]["metadata"]["vv_session"]["endpoint_order"] == ["memory-endpoint"]
            assert requests[0].model == "memory-model"
            assert provider.resolved_models == ["main-model", "memory-model"]
            from vv_agent.session.result import project_result

            result = project_result(kernel.store, "route", "route/turn/initial", runtime=rt)
            call = next(c for c in result.token_usage.model_calls if c.model == "memory-model")
            assert call.backend == "memory" and call.usage.total_tokens == 5
            drive(kernel.store, "route", runtime=rt, _one_turn=True)
            assert kernel.store.read_state("route")[1] == rows
            assert len(requests) == 1 and provider.resolved_models == ["main-model", "memory-model"]
        # Repeated definition reads reuse a frozen route, while new selections clear it.
        task = rt.compile("next", "route/turn/next")
        binding = rt._memory_bindings(task)
        assert binding == rt._memory_bindings(task)
        internal_client.endpoint_targets = [EndpointTarget("changed-endpoint", "unused", "https://example.invalid")]
        assert rt._memory_bindings(task)[purpose]["endpoints"] == ["changed-endpoint"]
        internal_client.endpoint_targets = [EndpointTarget("memory-endpoint", "unused", "https://example.invalid")]
        task.metadata[prefix + "_backend"] = "main"
        task.metadata[prefix + "_model"] = "main-model"
        assert purpose not in rt._memory_bindings(task)
        assert rt.model_route(purpose)[0] is rt.llm
        task.metadata[prefix + "_backend"] = "memory"
        task.metadata[prefix + "_model"] = "memory-model"
        assert rt._memory_bindings(task)[purpose]["endpoints"] == ["memory-endpoint"]
        assert rt.model_route(purpose)[0] is internal_client
        assert provider.resolved_models == ["main-model", "memory-model"]
    finally:
        kernel.close()
