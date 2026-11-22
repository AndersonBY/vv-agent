"""Check producer bytes and behavior against the selected canonical contract."""

import base64
import json
import re
import runpy
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator
from support.contract import fixture_root
from support.interactions import displayed_target

from vv_agent import Agent, InteractionAnswer, RunConfig, Runner
from vv_agent.approval import ApprovalBroker
from vv_agent.events import ApprovalRequestedEvent, event_from_dict
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel import _sql as sql_module
from vv_agent.kernel._content import ContentIntegrityError, ContentLinkConflict, ContentRef, ContentUnavailable
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.kernel._content_reader import DurableContentReader
from vv_agent.kernel._content_runtime import SessionContentWriter
from vv_agent.kernel._content_sql import _ContentReadFrame
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._hydration import IntegrityFailure, KeyUnavailable, MissingContentLink
from vv_agent.kernel._sql import LedgerIntegrityError, SQLStore
from vv_agent.kernel._sqlite import _SQLiteContentBackend
from vv_agent.model import ScriptedModelProvider
from vv_agent.session import _recovery_phase
from vv_agent.session import kernel as kernel_module
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.app_server import _KernelRunAdapter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.projection import hydrate_records, hydration_for_session_store, project_records
from vv_agent.session.records import InboxItem, Record, RecordError, digest, make_record
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Lease, StoredRecord
from vv_agent.session.surfaces import (
    SessionDriver,
    _project_record_page,
    _read_snapshot,
    _tenant_bound_sql_store,
)
from vv_agent.session.tracing import project_spans
from vv_agent.session.worker import LocalWorker
from vv_agent.tools.function import function_tool
from vv_agent.types import AgentResult, LLMResponse, ToolCall

from .helpers import fixture_key_provider, read_fixture_state

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "session_kernel_fixtures.py"
AUTHOR = runpy.run_path(str(SCRIPT))
REPLACEMENTS = runpy.run_path(str(SCRIPT.with_name("_session_kernel_replacements.py")))
start_runner = REPLACEMENTS["start_runner"]
CURATION = runpy.run_path(str(SCRIPT.with_name("_session_kernel_curation.py")))
CHECKS = runpy.run_path(str(SCRIPT.with_name("_session_kernel_checks.py")))
FILES = {
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


@pytest.fixture(scope="module")
def session_schemas():
    return CHECKS["session_schema_validators"](REPLACEMENTS["load"])


@pytest.mark.parametrize(("family", "missing_variant"), [("record", 2), ("inbox", 3)])
def test_session_schema_inventory_rejects_missing_variant(family, missing_variant):
    def load(name):
        schema = REPLACEMENTS["load"](name)
        if name == f"session_{family}.schema.json":
            schema["oneOf"] = [
                branch
                for branch in schema["oneOf"]
                if CHECKS["resolve_pointer"](schema, branch["allOf"][0]["$ref"][1:])["properties"]["feature_level"]["const"]
                != missing_variant
            ]
        return schema

    with pytest.raises(AssertionError, match=f"session_{family}.*expected one branch"):
        CHECKS["session_schema_validators"](load)


def test_contract_40_trusted_storage_has_real_python_producer_capture():
    trusted_storage = json.loads((fixture_root() / "trusted_storage_s1.json").read_text())
    assert trusted_storage["status"] == "producer-captured"
    assert trusted_storage["provenance"]["producer_capture"] is True
    assert re.fullmatch(r"[0-9a-f]{40}", trusted_storage["revision"])
    assert trusted_storage["revision"] == trusted_storage["provenance"]["source_revision"]
    assert trusted_storage["record_schema_version"] == "S1"
    assert trusted_storage["inbox_schema_version"] == "S1"
    historical = trusted_storage["provenance"]["historical_capture"]
    assert historical.endswith("session_record_diagnostics.json")


def checked_bytes(vector):
    raw = base64.b64decode(vector["bytes_base64"], validate=True)
    assert sha256(raw).hexdigest() == vector["sha256"]
    return raw


@pytest.fixture(scope="module")
def generated_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("kernel-fixtures")
    directories = [root / "first"]
    coverage_runs = []
    reports = []
    for output in directories:
        generated = subprocess.run(
            [sys.executable, str(SCRIPT), "--output", str(output)], cwd=root, capture_output=True, text=True, timeout=600
        )
        assert generated.returncode == 0, generated.stderr
        assert {p.name for p in output.iterdir()} == FILES
        report = json.loads(generated.stdout)
        reports.append(report)
        coverage_runs.append(report["coverage"])
        # Bound aggregate fixture growth, including contract 32.1.0's 12,000-character artifact preview.
        assert sum(p.stat().st_size for p in output.iterdir()) <= 3_050_000
        assert all(p.stat().st_size <= 512_000 for p in output.iterdir())
    assert set(coverage_runs[0]) == FILES
    first = directories[0]
    second = fixture_root()
    values = {}
    for name in FILES:
        raw = (first / name).read_text()
        values[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
    return [first, second], coverage_runs, values, reports[0]["self_checks"]


def test_generated_fixtures_match_vendored_snapshot(generated_fixtures):
    (produced, vendored), _, _, _ = generated_fixtures
    differences = sorted(name for name in FILES if (produced / name).read_bytes() != (vendored / name).read_bytes())
    assert not differences, differences


def test_app_server_timeout_uses_public_resume_and_retains_durable_decision(monkeypatch: pytest.MonkeyPatch):
    approval_calls = 0
    replay_blocked = Event()
    join_started = Event()
    approval_arguments = _KernelRunAdapter._approval_arguments
    worker_join = LocalWorker.join

    def hold_replayed_approval(adapter, sid, store, state, records, event, wait):
        nonlocal approval_calls
        arguments = approval_arguments(adapter, sid, store, state, records, event, wait)
        if wait["turn_id"] == "thread_3/turn/turn_2":
            approval_calls += 1
            if approval_calls == 2:
                replay_blocked.set()
                assert join_started.wait(timeout=5), "request(40) did not reach its synchronization join"
        return arguments

    def synchronize_join(worker, timeout=None):
        if replay_blocked.is_set():
            join_started.set()
        return worker_join(worker, timeout)

    monkeypatch.setattr(_KernelRunAdapter, "_approval_arguments", hold_replayed_approval)
    monkeypatch.setattr(LocalWorker, "join", synchronize_join)
    fixtures = AUTHOR["Fixtures"]()
    try:
        app = fixtures.app_server()
        resumes = []
        resolved_rows = []
        subscription_position = None
        resolved_position = None
        for position, entry in enumerate(app["transcripts"]):
            request = entry.get("request", {})
            if request.get("id") == 40 and request.get("method") == "thread/resume":
                subscription_position = position
            if request.get("method") == "thread/resume" and request.get("params", {}).get("threadId") == "thread_3":
                resumes.append((request["params"].get("subscribe"), entry["responses"][0]))
            if any(
                notification.get("method") == "approval/resolved"
                and notification.get("params", {}).get("threadId") == "thread_3"
                and notification.get("params", {}).get("turnId") == "thread_3/turn/turn_2"
                for notification in entry.get("notifications", [])
            ):
                resolved_rows.append(entry)
                resolved_position = position

        assert approval_calls >= 2
        assert replay_blocked.is_set()
        assert subscription_position is not None
        assert resolved_position is not None
        subscription_drain = app["transcripts"][subscription_position + 1]
        assert any(notification.get("method") == "approval/request" for notification in subscription_drain["notifications"])
        assert subscription_position < resolved_position
        assert [(subscribe, "error" not in response) for subscribe, response in resumes] == [
            (False, True),
            (True, True),
        ]
        assert any(response["result"]["thread"]["status"] == "approval_wait" for subscribe, response in resumes if subscribe)
        assert len(resolved_rows) == 1
        resolved = [
            notification
            for notification in resolved_rows[0]["notifications"]
            if notification.get("method") == "approval/resolved"
        ]
        assert [(notification["params"]["turnId"], notification["params"]["decision"]) for notification in resolved] == [
            ("thread_3/turn/turn_2", "timeout")
        ]
        assert not any(notification.get("method") == "approval/request" for notification in resolved_rows[0]["notifications"])

        state, records, _ = fixtures.kernel.store.read_state("thread_3")
        assert state.active_turn_id is None
        assert state.waits == {}
        assert state.turns["thread_3/turn/turn_2"].ended
        timeout_answers = [
            row.record
            for row in records
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["kind"] == "approval_answer"
            and row.record.payload["input"].get("target_turn_id") == "thread_3/turn/turn_2"
            and row.record.payload["input"]["payload"].get("decision") == "timeout"
        ]
        assert len(timeout_answers) == 1
    finally:
        fixtures.kernel.close()


def test_approval_answer_optionals_follow_app_server_and_keep_hydrated_prefix_coverage():
    fixtures = AUTHOR["Fixtures"]()
    try:
        fixtures.produce()
        fixtures.approval_answer_optionals()
        item = fixtures.inbox[("approval_optional", "answer")]
        assert {key: item.payload[key] for key in ("policy_digest", "rule_id", "expires_ms")} == {
            "policy_digest": None,
            "rule_id": None,
            "expires_ms": None,
        }

        app = fixtures.app_server()
        resolved = next(
            entry
            for entry in app["transcripts"]
            if any(
                notification.get("method") == "approval/resolved"
                and notification.get("params", {}).get("threadId") == "thread_3"
                and notification.get("params", {}).get("turnId") == "thread_3/turn/turn_2"
                for notification in entry.get("notifications", [])
            )
        )
        assert not any(notification.get("method") == "approval/request" for notification in resolved["notifications"])

        vectors = AUTHOR["valid_vectors"](fixtures.collect(), fixtures.inbox)
        approval_answers = [
            vector["wire"]["payload"]
            for vector in vectors
            if vector["type"] == "inbox" and vector["wire"]["kind"] == "approval_answer"
        ]
        for field in ("policy_digest", "rule_id", "expires_ms"):
            assert sum(field in payload for payload in approval_answers) == 1

        stored = fixtures.kernel.store.read_state("approval_optional")[1]
        hydration = hydration_for_session_store(fixtures.kernel.store, "approval_optional")
        projected_prefixes = 0
        raw_state_failed = False
        for through in range(1, len(stored) + 1):
            logical = [row.record for row in stored[:through]]
            raw_state = fold(
                logical,
                consumed_inputs=[InboxItem(**row.payload["input"]) for row in logical if row.kind == "input_applied"],
            )
            if through == len(stored):
                break
            next_record = stored[through].record
            if next_record.kind not in {"context_compacted", "boundary_recorded"} or next_record.payload["source_digest"] is None:
                continue
            hydrated_prefix = hydrate_records(stored[:through], hydration)
            try:
                AUTHOR["project_context"](hydrated_prefix, raw_state)
            except RecordError as error:
                assert str(error) == "S1 content hydration is unavailable in the producer/codec phase"
                raw_state_failed = True
            _, context = AUTHOR["_hydrated_prefix_projection"](stored[:through], hydration)
            assert sha256(AUTHOR["independent_bytes"]([context])[0]).hexdigest() == next_record.payload["source_digest"]
            projected_prefixes += 1
        assert projected_prefixes
        assert raw_state_failed
    finally:
        fixtures.kernel.close()


def test_fixture_store_explicitly_uses_hash_only_ledger_mode():
    fixtures = AUTHOR["Fixtures"]()
    try:
        assert fixtures._raw_store._ledger_integrity_mode() == "hash-only"
    finally:
        fixtures.kernel.close()


def test_children_fixture_declares_runtime_support_and_reaches_parent_result():
    fixtures = AUTHOR["Fixtures"]()
    try:
        assert fixtures._raw_store.worker_supports.max_level == 1
        fixtures.children()
        assert fixtures._raw_store.worker_supports.max_level == 2
        assert fixtures.result("children").final_output == "parent done"
    finally:
        fixtures.kernel.close()


def test_manager_tool_result_stays_on_projected_primary_cycle():
    fixtures = AUTHOR["Fixtures"]()
    author = REPLACEMENTS["Author"](fixtures, AUTHOR["independent_bytes"], AUTHOR["facts"])
    try:
        fixtures.children()
        result = author.execute(
            "manager_primary_cycle_probe",
            [
                LLMResponse("", [ToolCall("manager", "create_sub_task", {"agent_id": "\u001c", "task_description": "Research"})]),
                LLMResponse("done"),
            ],
            agent=Agent(
                "coordinator",
                "Coordinate.",
                sub_agents={"researcher": REPLACEMENTS["SubAgentConfig"](model="m", description="Research.")},
            ),
        )

        assert len(result.raw_result.cycles) == 2
        primary = result.raw_result.cycles[0]
        assert len(primary.tool_results) == 1
        completed = next(event for event in result.events if event.type == "tool_call_completed")
        assert completed.tool_call_id == primary.tool_calls[0].id
        assert json.loads(primary.tool_results[0].content) == json.loads(
            completed.to_dict()["metadata"]["tool_result"]["preview"]
        )
    finally:
        fixtures.kernel.close()


def test_controls_fixture_reuses_retained_runtime_and_reaches_archive():
    fixtures = AUTHOR["Fixtures"]()
    try:
        fixtures.admit("seed", [LLMResponse("done")])
        fixtures.run("seed")
        fixtures.controls()

        controls = fixtures.runtimes["controls"]
        assert (
            fixtures.kernel.runtime_factory(
                fixtures.kernel.store,
                controls.agent,
                controls.config,
                controls.frozen_task,
                "controls",
            )
            is controls
        )
        assert fixtures.state("controls").archived
        assert fixtures.state("seed").closed
    finally:
        fixtures.kernel.close()


def test_memory_capacity_corpus_reaches_real_turn_terminals(tmp_path):
    fixtures = AUTHOR["Fixtures"]()
    author = REPLACEMENTS["Author"](fixtures, AUTHOR["independent_bytes"], AUTHOR["facts"])
    lifecycle_suite = runpy.run_path(str(Path(__file__).parents[2] / "tests" / "test_memory_lifecycle_contract.py"))
    try:
        provider = ScriptedModelProvider.new("test", "m", [LLMResponse("prime")])
        prime = start_runner(
            fixtures.kernel,
            "capacity-worker-prime",
            Agent("prime", "Prime the fixture worker."),
            "go",
            run_config=RunConfig(model_provider=provider),
        )
        prime.result()
        admitted = fixtures.admit

        def admit_after_coordinator_pass(*args, **kwargs):
            runtime = admitted(*args, **kwargs)
            previous = fixtures.kernel._worker.coordinate_pass
            assert fixtures.kernel._worker.wait_for_coordinate_pass(previous, timeout=5)
            fixtures.kernel._worker.wait_idle(5)
            return runtime

        fixtures.admit = admit_after_coordinator_pass
        author.memory_capacity(lifecycle_suite, tmp_path)

        capacity = author.outputs["memory_lifecycle.json"]["capacity_contract"]
        cases = [*capacity["cases"], *capacity["context_window_resolution"]["cases"]]
        for case in cases:
            sid = "c1c/capacity/" + case["name"]
            state, rows, _ = fixtures.kernel.store.read_state(sid)
            started = [row.record for row in rows if row.record.kind == "turn_started"]
            ended = [row.record for row in rows if row.record.kind == "turn_ended"]

            assert state.active_turn_id is None
            assert len(started) == len(ended) == 1
            assert started[0].turn_id == ended[0].turn_id == f"{sid}/turn/initial"
            assert ended[0].payload["status"] == "completed"
            assert not any(row.record.kind == "turn_parked" for row in rows)
    finally:
        fixtures.kernel.close()


@pytest.mark.parametrize(
    ("producer", "session_id", "input_id"),
    [
        ("endpoint_routing", "endpoint-routing", "select"),
        ("model_selection", "model_selection", "select-B"),
    ],
)
def test_model_selection_fixture_producers_publish_l2_requirements(producer, session_id, input_id, session_schemas):
    fixtures = AUTHOR["Fixtures"]()
    try:
        getattr(fixtures, producer)()

        created = fixtures.kernel.store.read(session_id, limit=1).records[0].record
        stored = fixtures.kernel.store.read_input(session_id, input_id)
        assert stored is not None
        requirements = stored.item.to_dict()["requirements"]

        assert created.feature_level == 2
        assert created.payload["bootstrap"]["level"] == 2
        assert stored.item.feature_level == 2
        assert requirements["level"] == 2
        for family, retained in (("record", created), ("inbox", stored.item)):
            wire = json.loads(retained.encode())
            validator = session_schemas[family]
            validator.validate(wire)
            assert not validator.is_valid(wire | {"feature_level": 1})
            assert not validator.is_valid(wire | {"unexpected": True})
        assert requirements["capabilities"]
        assert all(capability.startswith("model:") for capability in requirements["capabilities"])
        assert fixtures.kernel._runtimes[session_id].requirements.level == 2
        assert fixtures.kernel.store.worker_supports.accepts(
            level=requirements["level"],
            handler_version=requirements["handler_version"],
            capabilities=requirements["capabilities"],
        )
    finally:
        fixtures.kernel.close()


def test_app_server_fixture_orders_model_selection_before_close():
    fixtures = AUTHOR["Fixtures"]()
    try:
        app = fixtures.app_server()
        created = fixtures.kernel.store.read("thread_1", limit=1).records[0].record
        selection = fixtures.kernel.store.read_input("thread_1", "select-app")
        assert created.feature_level == 2
        assert selection is not None
        assert selection.item.feature_level == 2
        assert selection.item.requirements["level"] == 2
        closed_thread = fixtures.state("thread_1")
    finally:
        fixtures.kernel.close()

    transcript = [entry for entry in app["transcripts"] if "request" in entry]
    model_selections = [entry for entry in transcript if entry["request"]["method"] == "thread/model/select"]
    assert [entry["request"]["id"] for entry in model_selections] == [16, 17, 18, 19, 46]
    assert [entry["request"]["params"]["inputId"] for entry in model_selections] == [
        "select-app",
        "select-app",
        "select-app",
        "select-app",
        "select-closed",
    ]
    assert model_selections[0]["responses"][-1]["result"]["accepted"] is True
    assert model_selections[1]["responses"][-1]["result"]["accepted"] is True
    assert [response["code"] for response in (entry["responses"][-1]["error"] for entry in model_selections[2:4])] == [
        -32024,
        -32602,
    ]
    assert model_selections[4]["responses"][-1]["error"] == {
        "code": -32024,
        "message": "submission_conflict",
    }

    positions = {
        entry["request"]["id"]: index for index, entry in enumerate(transcript) if entry["request"].get("id") is not None
    }
    assert positions[19] < positions[13] < positions[14] < positions[46]
    assert closed_thread.closed and closed_thread.archived


def test_store_semantics_prepares_ready_worker_compatible_session():
    fixtures = AUTHOR["Fixtures"]()
    try:
        fixtures.admit(
            "tools",
            [LLMResponse("", [ToolCall("echo", "echo", {"text": "ok"})]), LLMResponse("done")],
        )
        fixtures.run("tools")
        fixtures.kernel._worker.start()
        store = fixtures.kernel.store
        before = tuple(
            (stored.seq, stored.commit_id, stored.writer_epoch, stored.record.record_id)
            for stored in store.read("tools", limit=1024).records
        )
        epoch_before = store._one(
            "SELECT lease_epoch FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, "tools"),
        )[0]

        fixtures.prepare_store_semantics()
        ready = InboxItem("ready", "follow_up", {"content": "store"})
        fixtures.push("tools", ready)

        row = store._one(
            "SELECT next_drive_ms,level,handler_version,required_capabilities "
            "FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, "tools"),
        )
        capabilities = store._session_capabilities(row[3])
        assert row[0] == fixtures.clock
        assert store.worker_supports.accepts(level=row[1], handler_version=row[2], capabilities=capabilities)
        assert store.is_runnable("tools")
        pending = store.peek_inbox("tools")
        assert len(pending) == 1
        assert pending[0].item.input_id == ready.input_id
        assert pending[0].item.kind == ready.kind
        assert pending[0].consumed_seq is None
        assert not fixtures.kernel._worker.running

        lease = store.acquire("tools", owner="store-semantics", ttl_ms=100)
        assert lease is not None
        assert (lease.tenant_id, lease.session_id, lease.epoch) == (store.tenant_id, "tools", epoch_before + 1)
        assert store.release(lease)
        after = tuple(
            (stored.seq, stored.commit_id, stored.writer_epoch, stored.record.record_id)
            for stored in store.read("tools", limit=1024).records
        )
        assert after == before
    finally:
        fixtures.kernel.close()


def test_admission_negatives_capture_published_input_with_rejection_audit():
    fixtures = AUTHOR["Fixtures"]()
    try:
        provider = AUTHOR["Provider"]()
        fixtures.admit(
            "accepted",
            [
                AUTHOR["LLMResponse"]("", [AUTHOR["ToolCall"]("effect", "echo", {"text": "effect"})]),
                AUTHOR["LLMResponse"]("done"),
            ],
            providers={"echo": provider},
        )
        fixtures.run("accepted")
        turn_id = "accepted/turn/initial"
        fixtures.push_provider_evidence("accepted", provider.callback("provider_evidence", turn_id, "evidence"))
        fixtures.push("accepted", provider.callback("provider_result", turn_id, "result"))
        fixtures.run("accepted")
        fixtures.children()

        fixtures.admission_negatives()

        vector = next(v for v in fixtures.invalid if v["rejection_class"] == "provider_authentication")
        raw = checked_bytes(vector)
        item = InboxItem.parse(raw)
        audit = fixtures.state("accepted").applied_inputs[item.input_id]
        assert audit.payload["disposition"] == "rejected"
        assert audit.payload["reason"] == vector["reason"]
        assert audit.payload["input"]["input_id"] == item.input_id
        assert audit.payload["input"]["payload"]["evidence"] == ["forged"]
        assert item.payload["evidence"]["content_id"].startswith("content_")
        assert {v["rejection_class"] for v in fixtures.invalid} == {
            "provider_authentication",
            "provider_binding_drift",
            "stale_generation",
            "child_authentication",
            "authorization",
        }
    finally:
        fixtures.kernel.close()


def test_recovery_cuts_preserve_prefixes_and_single_provider_effects():
    fixtures = AUTHOR["Fixtures"]()
    try:
        fixtures.recovery_cuts()
        cases = {case["case"]: case for case in fixtures.recovery}

        assert set(cases) == {
            *(f"lost_wall_{policy.value}" for policy in AUTHOR["UnavailableMetricPolicy"]),
            "op_planned",
            "op_prepared",
            "op_started",
            "op_completed",
            "boundary_recorded",
            "accepted",
            "parked",
            "tool_a_done_b_pending",
            "boundary_before_commit",
            "boundary_after_commit",
        }
        prefix_kinds = {}
        for name, case in cases.items():
            stored = fixtures.kernel.store.read_state(case["session_id"])[1]
            record_ids = [item.record.record_id for item in stored]
            assert record_ids[: len(case["prefix_ids"])] == case["prefix_ids"]
            assert fixtures.state(case["session_id"]).active_turn_id is None
            prefix_kinds[name] = [item.record.kind for item in stored[: len(case["prefix_ids"])]]

        assert cases["accepted"]["effects"] == cases["parked"]["effects"] == 1
        assert cases["boundary_before_commit"]["callback_calls"] == 3
        assert cases["boundary_after_commit"]["callback_calls"] == 2
        for kind in ("op_planned", "op_prepared", "op_started", "op_completed", "boundary_recorded"):
            assert cases[kind]["remaining_model_steps"] >= 0
            assert prefix_kinds[kind][-1] == kind
        assert prefix_kinds["accepted"][-1] == "op_started"
        assert prefix_kinds["parked"][-1] == "op_parked"
        assert prefix_kinds["boundary_before_commit"][-1] == "op_completed"
        assert prefix_kinds["boundary_after_commit"][-1] == "boundary_recorded"
    finally:
        fixtures.kernel.close()


def validate_outputs(values):
    return CHECKS["validate_outputs"](values, REPLACEMENTS["BASE"], REPLACEMENTS["KEEP"], REPLACEMENTS["REPLACE"])


def test_session_kernel_fixtures_deterministic_and_revalidated(generated_fixtures, session_schemas):
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
        for retired in REPLACEMENTS["RETIRED_FIELDS"] | REPLACEMENTS["RETIRED_KINDS"]:
            assert re.search(r"(?<![A-Za-z0-9_])" + re.escape(retired) + r"(?![A-Za-z0-9_])", raw) is None, (name, retired)
        replaced[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
    assert set(REPLACEMENTS["validate_replacements"](replaced, AUTHOR["independent_bytes"])) == set(REPLACEMENTS["REPLACE"])

    corpus = read("session_codec_vectors.json")
    coverage = corpus["coverage"]
    assert len(coverage["record_kinds"]) == 14
    assert len(coverage["inbox_kinds"]) == 9
    assert len(coverage["boundary_stages"]) == 7
    assert len(coverage["handle_variants"]) == 4
    assert all(coverage["optional_fields"].values())
    assert coverage["optional_fields"]["boundary.output_checked.partial_output"] > 0
    sessions, records, inputs = defaultdict(list), {}, {}
    independent = AUTHOR["independent_bytes"]([v["wire"] for v in corpus["vectors"]])
    for vector, expected in zip(corpus["vectors"], independent, strict=True):
        raw = checked_bytes(vector)
        assert raw == expected
        session_schemas[vector["type"]].validate(vector["wire"])
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
        elif layer == "input_conflict":
            item = InboxItem.parse(raw)
            audit = records[(vector["session_id"], vector["audit_record_id"])].payload
            original = InboxItem(**audit["input"])
            assert audit["disposition"] == "applied"
            assert original.input_id == item.input_id and original.encode() != item.encode()
            assert vector["reason"] == "input_id bytes conflict"
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
        assert projection["spans"]
        source_by_seq = {row.seq: row.record for row in rows}
        for entry in projection["spans"]:
            span = entry["span"]
            source = source_by_seq[entry["seq"]]
            identity = (
                ["tools", source.operation_id, source.attempt]
                if span["name"] == "tool"
                else ["tools", source.turn_id, span["name"]]
            )
            assert span["span_id"] == "sk/span/" + sha256(AUTHOR["independent_bytes"]([identity])[0]).hexdigest()
        for event in projection["events"]:
            assert event["version"] == "v6"
            assert event_from_dict(event).to_dict() == event
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

    exports = export_schema_bundles()
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
        "thread/model/select": "ThreadModelSelectResponse",
        "turn/start": "TurnStartResponse",
        "thread/events/list": "ThreadEventsListResponse",
        "turn/resume": "TurnResumeResponse",
    }
    for entry in app["transcripts"]:
        request = entry.get("request")
        if "rejected_request" in entry:
            request = json.loads(checked_bytes(entry["rejected_request"]))
        if request is not None:
            errors = list(Draft202012Validator(schemas["ClientRequest"]).iter_errors(request))
            if errors:
                if request["method"] == "thread/events/list":
                    assert entry["responses"][0]["error"]["code"] == -32026
                    continue
                assert entry["responses"][0]["error"]["code"] == -32602
                if request["method"] == "thread/start":
                    from vv_agent.app_server.protocol.identity import client_id

                    with pytest.raises(ValueError):
                        client_id(request["params"]["inputId"], "inputId")
                else:
                    bad_fields = {"turn/resume": "checkpointKey", "thread/model/select": "extra"}
                    assert bad_fields[request["method"]] in request["params"]
            for response in entry["responses"]:
                envelope.validate(response)
                if "result" in response and request["method"] in results:
                    name = results[request["method"]]
                    result_schema = schemas.get(name)
                    if result_schema is None:
                        result_schema = {"$ref": f"#/$defs/{name}", "$defs": schemas["ClientRequest"]["$defs"]}
                    Draft202012Validator(result_schema).validate(response["result"])
        for notification in entry.get("notifications", []):
            envelope.validate(notification)
    assert app["facts"]["observer_can_approve"] and app["facts"]["timeout_at_absolute_deadline"]


def test_host_interaction_wire_references_use_current_owner(generated_fixtures):
    from vv_agent.interaction import HostInteractionOutcome, HostInteractionRequest

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
    schema = json.loads(export_schema_bundles()["jsonSchema"]["ServerNotification"])["$defs"]["TurnCompletedParams"]
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

    kernel = SessionDriver()
    try:
        provider = ScriptedModelProvider.from_steps("test", "m", [complete])
        handle = start_runner(
            kernel,
            "subscribers",
            Agent("fixture", "Answer."),
            "go",
            run_config=RunConfig(model_provider=provider, stream=broken_observer),
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
        assert AgentResult.from_dict(wire).to_dict() == wire
        with pytest.raises((ValueError, TypeError)):
            AgentResult.from_dict(wire | {"checkpoint_key": "retired"})
    finally:
        released.set()
        kernel.close()


def test_handle_burst_result_waits_for_the_worker_claim():
    kernel = SessionDriver()
    provider_calls = 0

    def complete(_request):
        nonlocal provider_calls
        provider_calls += 1
        return LLMResponse("draft")

    handle = None
    try:
        provider = ScriptedModelProvider.from_callback("test", "m", complete)
        handle = start_runner(
            kernel,
            "worker-result",
            Agent("fixture", "Continue."),
            "go",
            run_config=RunConfig(model_provider=provider, max_cycles=8, no_tool_policy="continue"),
        )
        first, second = handle.events(), handle.events()
        prefix_a, prefix_b = next(first), next(second)
        observed = []
        original = SessionDriver._handle_done

        def checked(self, sid, requested_turn_id):
            observed.append((provider_calls, self._worker._inflight is None))
            return original(self, sid, requested_turn_id)

        with patch.object(SessionDriver, "_handle_done", checked):
            result = REPLACEMENTS["_result_after_worker"](handle)

        replayed_a = [prefix_a, *first]
        replayed_b = [prefix_b, *second]
        assert len(result.raw_result.cycles) == provider_calls == 8
        assert observed and all(calls == 8 and idle for calls, idle in observed)
        assert replayed_a == replayed_b == list(handle.events()) == result.events
    finally:
        if handle is not None:
            handle.close()
        else:
            kernel.close()


def test_private_shared_state_mutation_does_not_change_the_retained_seed():
    seed = {"nested": {"values": ["seed"]}}

    def before_cycle(_cycle, _messages, shared):
        shared["nested"]["values"].append("local")
        return []

    kernel = SessionDriver()
    try:
        result = start_runner(
            kernel,
            "isolation",
            Agent("isolation", "Answer."),
            "go",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
                shared_state=seed,
                before_cycle_messages=before_cycle,
            ),
        ).result()
        state, _, _ = read_fixture_state(kernel.store, "isolation")
        start = next(iter(state.turns.values())).start
        assert seed == start._task().initial_shared_state == {"nested": {"values": ["seed"]}}
        assert result.raw_result.shared_state == {"nested": {"values": ["seed", "local"]}}
    finally:
        kernel.close()


def test_private_approval_provider_failure_is_a_retained_terminal(tmp_path):
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

    with SQLiteStore.standalone(str(tmp_path / "approval.sqlite")) as store:
        broker = ApprovalBroker()
        store.configure_input_admission(allow_unsigned=True)
        kernel = SessionDriver(store=store, content_key_provider=fixture_key_provider())
        try:
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
            result = start_runner(
                kernel,
                "approval-failure",
                Agent("approval", "Use gated.", tools=[gated]),
                "go",
                run_config=RunConfig(
                    model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("gated", "gated", {})])]),
                    approval_provider=FailingApproval(),
                    approval_broker=broker,
                ),
            ).result()
            assert result.status.value == "failed" and result.raw_result.error is not None
            assert result.raw_result.error["message"] == "approval unavailable"
            assert not effects
            request = next(e for e in result.events if isinstance(e, ApprovalRequestedEvent))
            assert broker.pending_request(request.request_id) is None
            state, rows, _ = kernel.store.read_state("approval-failure")
            assert state.active_turn_id is None and rows[-1].record.kind == "turn_ended"
            retained = Runner.resume("approval-failure", result.run_id)
            assert retained.to_dict() == result.to_dict()
            assert kernel.store.read_state("approval-failure")[1] == rows
        finally:
            kernel.close()


def test_private_configured_resume_reuses_turn_and_rejects_closed_without_writes():
    runtimes = {}

    def runtime_factory(store, agent, config, task, session_id):
        runtime = kernel.runtime(agent, config, task, session_id=session_id, store=store)
        runtimes[session_id] = runtime
        return runtime

    kernel = SessionDriver(runtime_factory=runtime_factory)
    try:
        provider = ScriptedModelProvider.new("test", "m", [LLMResponse("question"), LLMResponse("answer")])
        runner = Runner.configured(RunConfig(model_provider=provider, no_tool_policy="wait_user"))
        with (
            patch("vv_agent.session.surfaces.SessionDriver", return_value=kernel),
            patch("vv_agent.runner.uuid.uuid4", return_value=type("Identity", (), {"hex": "resume"})()),
        ):
            result = runner.run_sync(Agent("fixture", "Answer."), "go")
        tid = result.raw_result.turn_id
        assert tid is not None and result.status.value == "wait_user"
        original_runtime = runtimes["resume"]
        kernel.answer(
            "resume", original_runtime, displayed_target(kernel, "resume"), InteractionAnswer("text", text="reply"), "reply"
        )
        resumed = runner.resume("resume", tid)
        assert resumed.raw_result.turn_id == tid
        assert len(resumed.raw_result.cycles) == 2
        state, rows, _ = kernel.store.read_state("resume")
        assert state.active_turn_id == tid
        with pytest.raises(ValueError, match="turn_id"):
            runner.resume("resume", "wrong")
        assert kernel.store.read_state("resume")[1] == rows
        kernel.control("resume", "close", "close", runtime=original_runtime)
        closed = kernel.store.read_state("resume")[1]
        with pytest.raises(ValueError, match="closed"):
            runner.resume("resume", tid)
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
    kernel = SessionDriver()
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
    kernel = SessionDriver()
    try:
        for case in fixture["cases"]:
            sid = case["name"]
            kernel.create(sid, "/fixture", {"seed": {"messages": [Message(**case["message"]).to_dict()], "shared_state": {}}})
            state, rows, _ = read_fixture_state(kernel.store, sid)
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
    kernel = SessionDriver()
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
        rt = kernel.runtime(Agent("route", "Be precise.", model="main-model"), config, session_id="route", store=kernel.store)
        if purpose == "compaction":
            rt.memory_manager = MemoryManager(compact_threshold=1000, keep_recent_messages=1)
        kernel.create("route", "/fixture", {"seed": {"unseen_input_protection": False}} if purpose == "compaction" else None)
        kernel.push("route", InboxItem("initial", "user", {"content": "go"}))
        with patch.object(VvLlmClient, "complete", side_effect=response):
            drive(kernel.store, "route", runtime=rt, _one_turn=True)
            state, rows, _ = read_fixture_state(kernel.store, "route")
            assert state.active_turn_id is None
            internal_plans = [r.record for r in rows if r.record.kind == "op_planned" and r.record.payload["purpose"] == purpose]
            assert len(internal_plans) == len(requests) == 1
            plan = internal_plans[0]
            assert plan.payload["request"]["metadata"]["vv_session"]["endpoint_order"] == ["memory-endpoint"]
            assert requests[0].model == "memory-model"
            assert provider.resolved_models == ["main-model", *["memory-model"] * 4, "main-model", "main-model"]
            from vv_agent.session.result import project_result

            result = project_result(kernel.store, "route", "route/turn/initial", runtime=rt)
            call = next(c for c in result.token_usage.model_calls if c.model == "memory-model")
            assert call.backend == "memory" and call.usage.total_tokens == 5
            drive(kernel.store, "route", runtime=rt, _one_turn=True)
            assert read_fixture_state(kernel.store, "route")[1] == rows
            assert len(requests) == 1
            assert provider.resolved_models == ["main-model", *["memory-model"] * 4, "main-model", "main-model"]
        # Definitions retain logical identity while route preparation uses current endpoints.
        task = rt.compile("next", "route/turn/next")
        binding = rt._memory_bindings(task)
        assert binding == rt._memory_bindings(task)
        internal_client.endpoint_targets = [EndpointTarget("changed-endpoint", "unused", "https://example.invalid")]
        assert rt._memory_bindings(task) == binding
        assert "endpoints" not in binding[purpose]
        routed_client = rt.model_route(purpose)[0]
        assert isinstance(routed_client, VvLlmClient)
        assert routed_client.endpoint_targets[0].endpoint_id == "changed-endpoint"
        internal_client.endpoint_targets = [EndpointTarget("memory-endpoint", "unused", "https://example.invalid")]
        task.metadata[prefix + "_backend"] = "main"
        task.metadata[prefix + "_model"] = "main-model"
        assert purpose not in rt._memory_bindings(task)
        assert rt.model_route(purpose)[0] is rt.llm
        task.metadata[prefix + "_backend"] = "memory"
        task.metadata[prefix + "_model"] = "memory-model"
        assert rt._memory_bindings(task) == binding
        assert rt.model_route(purpose)[0] is internal_client
        assert provider.resolved_models == [
            "main-model",
            *["memory-model"] * 4,
            *["main-model"] * 2,
            "memory-model",
            "main-model",
            "memory-model",
        ]
    finally:
        kernel.close()


@pytest.mark.parametrize(
    ("tamper", "error", "message"),
    (
        ("commit_body", LedgerIntegrityError, "commit body digest mismatch"),
        ("commit_digest", LedgerIntegrityError, "commit body digest mismatch"),
        ("commit_sequences", LedgerIntegrityError, "record sequences do not match"),
        ("commit_head_seq", LedgerIntegrityError, "head_seq column does not match"),
        ("commit_created_ms", ValueError, "created_ms must be a nonnegative integer"),
        ("inbox", LedgerIntegrityError, "consumed input is missing or unconsumed"),
        ("consumer", LedgerIntegrityError, "creation consumer is missing"),
    ),
)
def test_commit_verification_rejects_tampered_sql_after_warmed_pass(tamper, error, message):
    kernel = SessionDriver()
    try:
        handle = start_runner(
            kernel,
            f"verify-tamper-{tamper}",
            Agent("verify-tamper", "Answer."),
            "go",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new("test", "fixture", [LLMResponse("done"), LLMResponse("done")]),
                max_cycles=2,
                no_tool_policy="continue",
            ),
        )
        handle.result(timeout=30)
        kernel._worker.stop()
        assert kernel._worker.join(5)

        sid = handle.session_id
        raw = _tenant_bound_sql_store(kernel.store)
        assert isinstance(raw, SQLStore)
        head = raw.read(sid, limit=1).head_seq
        raw._verified_records(sid, head)
        commits = raw._rows(
            "SELECT commit_id,body,digest,record_sequences,head_seq,created_ms FROM sk_commit "
            "WHERE tenant_id=%s AND session_id=%s ORDER BY created_ms,commit_id",
            (raw.tenant_id, sid),
        )
        commit_id, _, _, _, stored_head, _ = commits[0]
        if tamper == "commit_body":
            raw._rows(
                "UPDATE sk_commit SET body=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                (b"{}", raw.tenant_id, sid, commit_id),
            )
        elif tamper == "commit_digest":
            raw._rows(
                "UPDATE sk_commit SET digest=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                ("0" * 64, raw.tenant_id, sid, commit_id),
            )
        elif tamper == "commit_sequences":
            commit_id = next(row[0] for row in commits if row[3] != b"[]")
            raw._rows(
                "UPDATE sk_commit SET record_sequences=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                (b"[]", raw.tenant_id, sid, commit_id),
            )
        elif tamper == "commit_head_seq":
            raw._rows(
                "UPDATE sk_commit SET head_seq=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                (stored_head + 1, raw.tenant_id, sid, commit_id),
            )
        elif tamper == "commit_created_ms":
            raw._rows("PRAGMA ignore_check_constraints=ON")
            raw._rows(
                "UPDATE sk_commit SET created_ms=-1 WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                (raw.tenant_id, sid, commit_id),
            )
            raw._rows("PRAGMA ignore_check_constraints=OFF")
        elif tamper == "inbox":
            input_id = next(
                json.loads(row_body)["consume_input_ids"][0]
                for (row_body,) in raw._rows(
                    "SELECT body FROM sk_commit WHERE tenant_id=%s AND session_id=%s ORDER BY created_ms,commit_id",
                    (raw.tenant_id, sid),
                )
                if json.loads(row_body)["consume_input_ids"]
            )
            raw._rows(
                "UPDATE sk_inbox SET consumed_seq=NULL WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                (raw.tenant_id, sid, input_id),
            )
        else:
            consumer = next(
                json.loads(row_body)["consumers"][0]
                for (row_body,) in raw._rows(
                    "SELECT body FROM sk_commit WHERE tenant_id=%s AND session_id=%s ORDER BY created_ms,commit_id",
                    (raw.tenant_id, sid),
                )
                if json.loads(row_body)["consumers"]
            )
            raw._rows(
                "DELETE FROM sk_consumer WHERE tenant_id=%s AND session_id=%s AND consumer=%s",
                (raw.tenant_id, sid, consumer),
            )

        with pytest.raises(error, match=message):
            raw._verified_records(sid, head)
    finally:
        kernel.close()


def test_commit_verification_reads_inbox_and_consumer_inventory_once_per_pass():
    kernel = SessionDriver()
    try:
        handle = start_runner(
            kernel,
            "verify-query-counts",
            Agent("verify-counts", "Answer."),
            "go",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new("test", "fixture", [LLMResponse("done"), LLMResponse("done")]),
                max_cycles=2,
                no_tool_policy="continue",
            ),
        )
        handle.result(timeout=30)
        kernel._worker.stop()
        assert kernel._worker.join(5)

        sid = handle.session_id
        raw = _tenant_bound_sql_store(kernel.store)
        assert isinstance(raw, SQLStore)
        head = raw.read(sid, limit=1).head_seq
        queries = []
        original_rows = raw._rows
        original_canonical = sql_module.canonical_json_bytes
        commit_count = raw._rows(
            "SELECT commit_id FROM sk_commit WHERE tenant_id=%s AND session_id=%s",
            (raw.tenant_id, sid),
        )
        raw._commit_parse_cache.clear()
        canonical_calls = 0

        def counted_rows(query, params=()):
            queries.append(" ".join(query.split()))
            return original_rows(query, params)

        def counted_canonical(*args, **kwargs):
            nonlocal canonical_calls
            canonical_calls += 1
            return original_canonical(*args, **kwargs)

        with patch.object(raw, "_rows", counted_rows), patch.object(sql_module, "canonical_json_bytes", counted_canonical):
            raw._verified_records(sid, head)
            raw._verified_records(sid, head)

        commit_rows_prefix = "SELECT commit_id,body,digest,record_sequences,head_seq,created_ms"
        assert sum(query.startswith(commit_rows_prefix) for query in queries) == 2
        assert sum("FROM sk_inbox" in query and "consumed_seq IS NOT NULL" in query for query in queries) == 2
        assert sum("FROM sk_consumer" in query for query in queries) == 2
        assert not any("FROM sk_inbox" in query and "input_id=%s" in query for query in queries)
        assert not any("FROM sk_consumer" in query and "consumer=%s" in query for query in queries)
        assert canonical_calls == 2 * len(commit_count)
        assert len(raw._commit_parse_cache) == len(commit_count) <= 1024
    finally:
        kernel.close()


def test_projection_280_pages_share_one_verified_snapshot_frame():
    cycles = 280

    def count_messages(messages, **_kwargs):
        return len(messages)

    kernel = SessionDriver()
    try:
        with (
            patch("vv_agent.memory.manager.count_messages_tokens", count_messages),
            patch("vv_agent.session.compaction.count_messages_tokens", count_messages),
            patch("vv_agent.session.reducer.count_messages_tokens", count_messages),
        ):
            handle = start_runner(
                kernel,
                "projection-delivery-280",
                Agent("projection", "Answer."),
                "go",
                run_config=RunConfig(
                    model_provider=ScriptedModelProvider.new("test", "fixture", [LLMResponse("done")]),
                    max_cycles=1,
                ),
            )
            handle.result(timeout=60)
        kernel._worker.stop()
        assert kernel._worker.join(5)

        sid = handle.session_id
        store = kernel.store
        _, seed_rows, _ = store.read_state(sid)
        by_kind = {stored.record.kind: stored for stored in seed_rows}
        raw = _tenant_bound_sql_store(store)
        assert isinstance(raw, SQLStore)
        probe = raw.acquire(sid, owner="projection-delivery-probe", ttl_ms=60_000)
        assert probe is not None
        writer_epoch = probe.epoch + 1
        raw.release(probe)

        records, source_sequences = [], []
        for index in range(1, cycles):
            turn_id = f"{sid}/turn/projection/{index}"
            operation_id = f"{turn_id}/model/primary/1"
            for kind, attempt in (
                ("turn_started", None),
                ("op_planned", 1),
                ("op_started", 1),
                ("op_completed", 1),
                ("turn_ended", None),
            ):
                template = by_kind[kind]
                payload = deepcopy(template.record._payload)
                selected_operation = operation_id if kind.startswith("op_") else None
                if kind == "turn_started":
                    payload["input_ids"] = []
                    payload["generation"] = index
                elif kind == "op_planned":
                    payload["idempotency_key"] = f"{sid}/{turn_id}/model/primary/1"
                elif kind == "op_started":
                    payload["dispatch_id"] = f"{operation_id}/1"
                    payload["epoch"] = writer_epoch
                elif kind == "turn_ended":
                    payload["adopted_results"] = [{"attempt": 1, "operation_id": operation_id}]
                records.append(
                    make_record(
                        kind,
                        session_id=sid,
                        turn_id=turn_id,
                        operation_id=selected_operation,
                        attempt=attempt,
                        payload=payload,
                        feature_level=template.record.feature_level,
                    )
                )
                source_sequences.append(template.seq)

        head = raw.read(sid, limit=1)
        lease = raw.acquire(sid, owner="projection-delivery", ttl_ms=60_000)
        assert lease is not None
        try:
            with raw.atomic() as tx:
                receipt = tx.append(
                    sid,
                    lease=lease,
                    expected_seq=head.head_seq,
                    commit_id="projection-delivery-280",
                    records=tuple(records),
                )
                for (_record_id, new_seq), old_seq in zip(receipt.record_sequences, source_sequences, strict=True):
                    raw._rows(
                        "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,input_id,slot) "
                        "SELECT tenant_id,session_id,content_id,%s,NULL,slot FROM sk_content_link "
                        "WHERE tenant_id=%s AND session_id=%s AND record_seq=%s",
                        (new_seq, raw.tenant_id, sid, old_seq),
                    )
        finally:
            raw.release(lease)

        verify_calls, content_reads = [], []
        original_verify = SQLStore._verified_records
        original_read = DurableContentReader.read

        def counted_verify(self, session_id, through, **kwargs):
            verify_calls.append((session_id, through))
            return original_verify(self, session_id, through, **kwargs)

        def counted_read(self, tenant_id, session_id, owner, slot, ref):
            content_reads.append((owner, slot, ref.content_id))
            return original_read(self, tenant_id, session_id, owner, slot, ref)

        with (
            patch.object(SQLStore, "_verified_records", counted_verify),
            patch.object(DurableContentReader, "read", counted_read),
            _read_snapshot(store),
        ):
            head_seq = raw.projection_prefix(sid).head_seq
            paged_events = []
            cursor, pages = 0, 0
            while cursor < head_seq:
                next_cursor, events, _has_more = _project_record_page(
                    store,
                    sid,
                    after_seq=cursor,
                    through_seq=head_seq,
                )
                assert next_cursor > cursor
                cursor = next_cursor
                paged_events.extend(events)
                pages += 1

        assert len(paged_events) >= 1_100
        assert pages >= 5
        assert verify_calls == [(sid, head_seq)]
        assert sum(through for _session_id, through in verify_calls) == head_seq
        assert len(content_reads) > pages

        _, source_records, _ = store.read_state(sid)
        hydration = hydration_for_session_store(store, sid)
        expected_events = project_records(hydrate_records(source_records, hydration))
        assert [event.to_dict() for event in paged_events] == [event.to_dict() for event in expected_events]

        _, _, cursor, has_more, _, _ = kernel._handle_event_page(
            sid,
            handle.run_id,
            after_seq=0,
        )
        assert has_more
        planned = next(stored for stored in source_records if stored.record.kind == "op_planned")
        request_ref = ContentRef.from_dict(planned.record._payload["request"])
        with pytest.raises(MissingContentLink):
            hydration.read("record:1", "request", request_ref)
        assert kernel.content_key_provider.destroy(store.tenant_id, request_ref.key_id).accepted
        with pytest.raises(KeyUnavailable):
            kernel._handle_event_page(
                sid,
                handle.run_id,
                after_seq=cursor,
            )
    finally:
        kernel.close()


def test_event_pages_isolate_store_arguments_across_shared_driver_state():
    entered_a, entered_b, release = Event(), Event(), Event()
    kernel_a, kernel_b = SessionDriver(), SessionDriver()
    handle_a = handle_b = None

    def complete_a(_request):
        entered_a.set()
        assert release.wait(5)
        return LLMResponse("store-a result")

    def complete_b(_request):
        entered_b.set()
        assert release.wait(5)
        return LLMResponse("store-b result")

    try:
        handle_a = start_runner(
            kernel_a,
            "store-arguments",
            Agent("store-a", "Answer."),
            "go",
            run_config=RunConfig(model_provider=ScriptedModelProvider.from_callback("test", "a", complete_a)),
        )
        handle_b = start_runner(
            kernel_b,
            "store-arguments",
            Agent("store-b", "Answer."),
            "go",
            run_config=RunConfig(model_provider=ScriptedModelProvider.from_callback("test", "b", complete_b)),
        )
        assert entered_a.wait(5) and entered_b.wait(5)
        assert handle_a.run_id == handle_b.run_id

        page_a = kernel_a._handle_event_page(
            handle_a.session_id,
            handle_a.run_id,
            after_seq=0,
            store=kernel_a.store,
        )
        page_b = kernel_a._handle_event_page(
            handle_b.session_id,
            handle_b.run_id,
            after_seq=0,
            store=kernel_b.store,
        )
        assert {event.agent_name for event in page_a[1] if event.agent_name is not None} == {"store-a"}
        assert {event.agent_name for event in page_b[1] if event.agent_name is not None} == {"store-b"}
    finally:
        release.set()
        for kernel, handle in ((kernel_a, handle_a), (kernel_b, handle_b)):
            if handle is None:
                kernel.close()
                continue
            handle.result(timeout=5)
            handle.close()


def _start_logical_driver(
    sessions: SessionDriver,
    store: SQLiteStore,
    session_id: str,
) -> tuple[_Driver, Lease]:
    sessions.create(
        session_id,
        "/workspace",
        {"seed": {"messages": [{"role": "system", "content": "Seed context."}], "shared_state": {}}},
    )
    sessions.push(session_id, InboxItem("input", "user", {"content": "go"}))
    runtime = sessions.runtime(
        Agent("logical-fold", "Work carefully.", model="scripted"),
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
            workspace="/workspace",
        ),
        session_id=session_id,
    )
    driver_store = sessions.store
    lease = driver_store.acquire(session_id, owner="logical-fold", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(driver_store, session_id, runtime, _Scope(lease, runtime, now_ms=driver_store._now()))
    assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
    assert _recovery_phase.run(driver, inputs.start_turn(driver))
    return driver, lease


def _logical_tool_plan(
    driver: _Driver,
    request: dict[str, Any],
    attempt: int,
    *,
    op_kind: str = "tool",
    dependencies: list[str] | None = None,
    operation_id: str | None = None,
) -> Record:
    turn_id = driver.state.active_turn_id
    assert turn_id is not None
    operation_id = operation_id or (f"{turn_id}/model/primary/{attempt}" if op_kind == "model" else "retry-operation")
    return make_record_before_publication(
        "op_planned",
        session_id=driver.sid,
        turn_id=turn_id,
        operation_id=operation_id,
        attempt=attempt,
        payload={
            "op_kind": op_kind,
            "purpose": "primary" if op_kind == "model" else None,
            "request": request,
            "request_digest": digest(request),
            "context_version": "v1",
            "dependencies": dependencies or [],
            "tool": None,
            "idempotency_key": "retry-call" if op_kind == "tool" else operation_id,
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "model" if op_kind == "model" else None,
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )


@pytest.fixture
def projection_frame_bundle(tmp_path: Path):
    session_id = "projection-frame-bundle"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "projection-frame-bundle.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            try:
                other = _Driver(
                    sessions.store,
                    session_id,
                    driver.runtime,
                    _Scope(lease, driver.runtime, now_ms=sessions.store._now()),
                )
                request = {
                    "id": "projection-call",
                    "name": "projection_tool",
                    "model": "scripted",
                    "arguments": {"path": "notes.txt"},
                    "messages": [{"role": "user", "content": f"historical message {index}"} for index in range(8)],
                    "metadata": {"vv_session": {"cycle_index": 1}},
                }
                model_plan = _logical_tool_plan(other, request, attempt=1, op_kind="model")
                assert model_plan.operation_id is not None
                prepared_request = request | {
                    "id": "prepared-projection-call",
                    "messages": [{"role": "user", "content": f"prepared message {index}"} for index in range(8)],
                }
                tool_plan = _logical_tool_plan(
                    other,
                    prepared_request,
                    attempt=1,
                    dependencies=[model_plan.operation_id],
                )
                prepared = make_record_before_publication(
                    "op_prepared",
                    session_id=other.sid,
                    turn_id=tool_plan.turn_id,
                    operation_id=tool_plan.operation_id,
                    attempt=tool_plan.attempt,
                    payload={
                        "request": prepared_request,
                        "request_digest": digest(prepared_request),
                        "op_kind": "tool",
                        "tool": {},
                        "provider_binding": tool_plan._payload["provider_binding"],
                        "idempotency_key": tool_plan._payload["idempotency_key"],
                        "hook_result": None,
                        "shared_state": {},
                        "policy_decision": None,
                    },
                )
                other.commit([model_plan, tool_plan, prepared])
                yield provider, sessions, store, session_id, tuple(store.read_state(session_id)[1])
            finally:
                assert sessions.store.release(lease)
    finally:
        provider.close()


def _canonical_event_bytes(events) -> list[bytes]:
    return [json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode() for event in events]


def test_projection_shallow_hydration_matches_full_event_bytes(projection_frame_bundle):
    _provider, _sessions, store, session_id, records = projection_frame_bundle
    hydration = hydration_for_session_store(store, session_id)
    expected = project_records(hydrate_records(records, hydration))
    projected = project_records(records, hydration=hydration)

    assert _canonical_event_bytes(projected) == _canonical_event_bytes(expected)
    planned = next(stored for stored in records if stored.record.kind == "op_planned")
    assert ContentRef.from_dict(planned.record._payload["request"])


def test_projection_key_destroy_between_pages_fails_closed(projection_frame_bundle):
    provider, _sessions, store, session_id, records = projection_frame_bundle
    hydration = hydration_for_session_store(store, session_id)
    started = next(stored for stored in records if stored.record.kind == "turn_started")
    planned = next(stored for stored in records if stored.record.kind == "op_planned")
    request_ref = ContentRef.from_dict(planned.record._payload["request"])

    assert project_records((started,), hydration=hydration)
    assert provider.destroy(store.tenant_id, request_ref.key_id).accepted
    with pytest.raises(KeyUnavailable):
        project_records((planned,), hydration=hydration)


def test_projection_link_tamper_after_page_fails_closed(projection_frame_bundle):
    _provider, _sessions, store, session_id, records = projection_frame_bundle
    hydration = hydration_for_session_store(store, session_id)
    planned = next(stored for stored in records if stored.record.kind == "op_planned")
    started = next(stored for stored in records if stored.record.kind == "turn_started")
    other_ref = ContentRef.from_dict(started.record._payload["definition"])

    page = (started, planned)
    assert project_records(page, hydration=hydration)
    with store.atomic():
        store._rows(
            "UPDATE sk_content_link SET content_id=? WHERE tenant_id=? AND session_id=? AND record_seq=? AND slot='request'",
            (other_ref.content_id, store.tenant_id, session_id, planned.seq),
        )
    with pytest.raises(MissingContentLink):
        project_records(page, hydration=hydration)


def test_unframed_event_and_span_batches_use_one_disposable_read_frame(projection_frame_bundle):
    provider, _sessions, store, session_id, records = projection_frame_bundle
    hydration = hydration_for_session_store(store, session_id)
    planned = next(stored for stored in records if stored.record.kind == "op_planned")
    owner = f"record:{planned.seq}"
    reads: list[tuple[str, str]] = []
    original_read = DurableContentReader.read

    def counted_read(self, tenant_id, read_session_id, read_owner, slot, ref):
        reads.append((read_owner, slot))
        return original_read(self, tenant_id, read_session_id, read_owner, slot, ref)

    with (
        patch.object(DurableContentReader, "read", counted_read),
        patch.object(provider, "unwrap", wraps=provider.unwrap) as unwrap,
    ):
        events = project_records(records, hydration=hydration)
    assert events
    assert unwrap.call_count == 1
    assert reads.count((owner, "request")) == 1
    assert not any(read_owner == owner and slot == "messages" for read_owner, slot in reads)

    reads.clear()
    with (
        patch.object(DurableContentReader, "read", counted_read),
        patch.object(provider, "unwrap", wraps=provider.unwrap) as unwrap,
    ):
        spans = project_spans(records, hydration=hydration)
    assert spans
    assert unwrap.call_count == 1
    assert reads.count((owner, "request")) == 1
    assert not any(read_owner == owner and slot == "messages" for read_owner, slot in reads)


def test_op_prepared_projection_does_not_hydrate_request_messages(projection_frame_bundle):
    _provider, _sessions, store, session_id, records = projection_frame_bundle
    hydration = hydration_for_session_store(store, session_id)
    prepared = next(stored for stored in records if stored.record.kind == "op_prepared")
    owner = f"record:{prepared.seq}"
    request_ref = ContentRef.from_dict(prepared.record._payload["request"])
    request_container = hydration.read(owner, "request", request_ref)
    assert request_container["messages"]
    assert all(ContentRef.from_dict(message) is not None for message in request_container["messages"])

    expected_events = project_records(hydrate_records(records, hydration))
    reads: list[tuple[str, str]] = []
    original_read = DurableContentReader.read

    def counted_read(self, tenant_id, read_session_id, read_owner, slot, ref):
        reads.append((read_owner, slot))
        return original_read(self, tenant_id, read_session_id, read_owner, slot, ref)

    with patch.object(DurableContentReader, "read", counted_read):
        events = project_records(records, hydration=hydration)
    assert _canonical_event_bytes(events) == _canonical_event_bytes(expected_events)
    assert reads.count((owner, "request")) == 1
    assert not any(read_owner == owner and slot.startswith("request.messages") for read_owner, slot in reads)

    reads.clear()
    with patch.object(DurableContentReader, "read", counted_read):
        spans = project_spans(records, hydration=hydration)
    assert spans
    assert reads.count((owner, "request")) == 1
    assert not any(read_owner == owner and slot.startswith("request.messages") for read_owner, slot in reads)


def test_active_logical_fold_refreshes_same_batch_retry_records_once(tmp_path: Path):
    session_id = "logical-retry-refresh"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "logical-retry-refresh.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            other = _Driver(
                sessions.store,
                session_id,
                driver.runtime,
                _Scope(lease, driver.runtime, now_ms=sessions.store._now()),
            )

            request = {
                "id": "retry-call",
                "name": "retry_probe_tool",
                "arguments": {"text": "one"},
                "messages": [{"role": "user", "content": f"retry history {index}"} for index in range(3)],
            }
            first = _logical_tool_plan(other, request, attempt=1)
            started = other.record(
                "op_started",
                {
                    "dispatch_id": "retry-call/1",
                    "authorization_version": "1",
                    "epoch": lease.epoch,
                    "mode": "sync",
                },
                first,
            )
            unknown = other.record(
                "op_unknown",
                {
                    "reason": {"kind": "worker_lost"},
                    "dispatch_evidence": {"kind": "tool"},
                    "observation": {"attempt": 1},
                    "retry": "retry",
                    "retry_at_ms": None,
                    "duplicate_cost_risk": True,
                    "measurement_missing": True,
                },
                first,
            )
            retry = _logical_tool_plan(other, request, attempt=2)

            with driver._logical_state_scope():
                assert driver._active_logical_fold is not None
                frame_reads = frame_unwraps = 0
                in_frame_read = False
                original_frame_get = _ContentReadFrame.get
                original_unwrap = provider.unwrap

                def counted_frame_get(self, tenant_id, session_id, ref, *, link_validated=False):
                    nonlocal frame_reads, in_frame_read
                    frame_reads += 1
                    in_frame_read = True
                    try:
                        return original_frame_get(
                            self,
                            tenant_id,
                            session_id,
                            ref,
                            link_validated=link_validated,
                        )
                    finally:
                        in_frame_read = False

                def counted_unwrap(tenant_id, session_id, wrapped_key):
                    nonlocal frame_unwraps
                    if in_frame_read:
                        frame_unwraps += 1
                    return original_unwrap(tenant_id, session_id, wrapped_key)

                with (
                    patch.object(_ContentReadFrame, "get", counted_frame_get),
                    patch.object(provider, "unwrap", counted_unwrap),
                ):
                    other.commit([first, started, unknown, retry])
                assert frame_reads > 1
                assert frame_unwraps == 1
                stored_plans = [
                    row
                    for row in store.read_state(session_id)[1]
                    if row.record.kind == "op_planned" and row.record.operation_id == "retry-operation"
                ]
                retry_owner = f"record:{stored_plans[1].seq}"
                source_owner = f"record:{stored_plans[0].seq}"
                reads: list[tuple[str, str]] = []
                original_read = DurableContentReader.read

                def counted_read(self, tenant_id, read_session_id, owner, slot, ref):
                    reads.append((owner, slot))
                    return original_read(self, tenant_id, read_session_id, owner, slot, ref)

                with patch.object(DurableContentReader, "read", counted_read):
                    driver.refresh()

                assert reads.count((retry_owner, "request")) == 1
                assert sum(owner == retry_owner and slot.startswith("request.messages[") for owner, slot in reads) == 3
                assert sum(owner == source_owner and slot.startswith("request.messages[") for owner, slot in reads) == 3
                assert retry.record_id in driver._logical_retry_record_ids
                retry_attempt = driver.state.operations["retry-operation"].attempts[2]
                assert retry_attempt.plan._payload["request"] == request
                raw_retry = next(
                    row.record for row in driver.records if row.record.kind == "op_planned" and row.record.attempt == 2
                )
                assert ContentRef.from_dict(raw_retry._payload["request"])

            assert driver._active_logical_fold is None
            assert driver._logical_prefix is not None
            prefix_fold = driver._logical_prefix.fold
            record_bodies = tuple(stored.record.encode() for stored in driver.records)
            fresh = _Driver(
                sessions.store,
                session_id,
                driver.runtime,
                _Scope(lease, driver.runtime, now_ms=sessions.store._now()),
            )
            cold_state = fresh._logical_view().state

            def unexpected_full_fold(*_args, **_kwargs):
                raise AssertionError("cached logical prefix must not fold the full history again")

            with (
                patch.object(kernel_module, "hydrate_and_fold_records", unexpected_full_fold),
                driver._logical_state_scope(),
            ):
                assert driver._active_logical_fold is prefix_fold
                retry_attempt = driver.state.operations["retry-operation"].attempts[2]
                assert retry_attempt.plan._payload["request"] == request
                assert driver.state == cold_state
            assert tuple(stored.record.encode() for stored in driver.records) == record_bodies

            assert sessions.store.release(lease)
    finally:
        provider.close()


def test_logical_retry_source_rehydrates_when_retry_commits_in_next_batch(tmp_path: Path):
    session_id = "logical-cross-batch-retry"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "logical-cross-batch-retry.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            try:
                request = {
                    "id": "retry-call",
                    "name": "retry_probe_tool",
                    "arguments": {"text": "one"},
                    "messages": [{"role": "user", "content": f"retry history {index}"} for index in range(3)],
                }
                first = _logical_tool_plan(driver, request, attempt=1)
                started = driver.record(
                    "op_started",
                    {
                        "dispatch_id": "retry-call/1",
                        "authorization_version": "1",
                        "epoch": lease.epoch,
                        "mode": "sync",
                    },
                    first,
                )
                unknown = driver.record(
                    "op_unknown",
                    {
                        "reason": {"kind": "worker_lost"},
                        "dispatch_evidence": {"kind": "tool"},
                        "observation": {"attempt": 1},
                        "retry": "retry",
                        "retry_at_ms": None,
                        "duplicate_cost_risk": True,
                        "measurement_missing": True,
                    },
                    first,
                )
                driver.commit([first, started, unknown])
                with driver._logical_state_scope():
                    pass

                retry = _logical_tool_plan(driver, request, attempt=2)
                reads: list[tuple[str, str]] = []
                original_read = DurableContentReader.read

                def counted_read(self, tenant_id, read_session_id, owner, slot, ref):
                    reads.append((owner, slot))
                    return original_read(self, tenant_id, read_session_id, owner, slot, ref)

                with patch.object(DurableContentReader, "read", counted_read), driver._logical_state_scope():
                    reads.clear()
                    driver.commit([retry])
                    assert driver._logical_retry_record_ids == {retry.record_id}
                    assert driver._logical_retry_source_record_ids == {first.record_id}
                    assert driver._logical_deep_record_ids == {first.record_id, retry.record_id}
                    assert driver._cached_logical_plan(retry) is not None

                with (
                    patch.object(CoreDriver, "hydrate_plan", side_effect=AssertionError("retry plan cache miss")),
                    driver._logical_state_scope(),
                ):
                    assert driver._logical_retry_record_ids == {retry.record_id}
                    assert driver._logical_retry_source_record_ids == {first.record_id}
                    assert driver._logical_deep_record_ids == {first.record_id, retry.record_id}
                    assert driver.hydrate_plan(retry)._payload["request"] == request

                stored_plans = [
                    stored
                    for stored in driver.records
                    if stored.record.kind == "op_planned" and stored.record.operation_id == "retry-operation"
                ]
                source_owner = f"record:{stored_plans[0].seq}"
                retry_owner = f"record:{stored_plans[1].seq}"
                assert reads.count((source_owner, "request")) == 1
                assert reads.count((retry_owner, "request")) == 1
                assert sum(owner == source_owner and slot.startswith("request.messages[") for owner, slot in reads) == 3
                assert sum(owner == retry_owner and slot.startswith("request.messages[") for owner, slot in reads) == 3
            finally:
                assert sessions.store.release(lease)
    finally:
        provider.close()


@pytest.mark.parametrize("tamper", [None, "key_destroy", "link", "ciphertext"])
def test_logical_fold_skips_completed_plan_messages_but_hydrates_active_plan(tmp_path: Path, tamper: str | None):
    session_id = "logical-plan-shallow"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "logical-plan-shallow.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            try:
                completed_request = {
                    "id": "completed-call",
                    "name": "probe",
                    "arguments": {"value": "completed"},
                    "messages": [{"role": "user", "content": f"completed history {index}"} for index in range(6)],
                    "metadata": {"vv_session": {"cycle_index": 1}},
                }
                completed_plan = _logical_tool_plan(
                    driver,
                    completed_request,
                    attempt=1,
                    operation_id="completed-operation",
                )
                driver.commit([completed_plan])
                started = driver.record(
                    "op_started",
                    {
                        "dispatch_id": "completed-call/1",
                        "authorization_version": "1",
                        "epoch": lease.epoch,
                        "mode": "sync",
                    },
                    completed_plan,
                )
                driver.commit([started])
                result = {
                    "tool_call_id": "completed-call",
                    "content": "ok",
                    "status_code": "SUCCESS",
                    "directive": "continue",
                }
                completed = driver.record(
                    "op_completed",
                    {
                        "result": result,
                        "result_digest": digest(result),
                        "usage": {},
                        "shared_state": None,
                        "evidence": {"kind": "tool"},
                        "execution_started": True,
                        "context": "normal",
                        "request_digest": driver.stored_record(completed_plan).record._payload["request_digest"],
                        "provider_binding": None,
                    },
                    completed_plan,
                )
                driver.commit([completed])

                active_request = {
                    "id": "active-call",
                    "name": "probe",
                    "arguments": {"value": "active"},
                    "messages": [{"role": "user", "content": f"active history {index}"} for index in range(3)],
                    "metadata": {"vv_session": {"cycle_index": 2}},
                }
                active_plan = _logical_tool_plan(
                    driver,
                    active_request,
                    attempt=1,
                    operation_id="active-operation",
                )
                driver.commit([active_plan])

                records = {stored.record.record_id: stored for stored in driver.records}
                completed_owner = f"record:{records[completed_plan.record_id].seq}"
                active_owner = f"record:{records[active_plan.record_id].seq}"
                reads: list[tuple[str, str]] = []
                original_read = DurableContentReader.read

                def counted_read(self, tenant_id, read_session_id, owner, slot, ref):
                    reads.append((owner, slot))
                    return original_read(self, tenant_id, read_session_id, owner, slot, ref)

                driver._invalidate_logical_view()
                with patch.object(DurableContentReader, "read", counted_read):
                    fold_state = driver._logical_view().state

                completed_slots = [slot for owner, slot in reads if owner == completed_owner]
                active_slots = [slot for owner, slot in reads if owner == active_owner]
                assert "request" in completed_slots
                assert not any(slot.startswith("request.messages[") for slot in completed_slots)
                assert "request" in active_slots
                assert sum(slot.startswith("request.messages[") for slot in active_slots) == 3
                assert fold_state.operations["completed-operation"].state == "completed"
                assert fold_state.operations["active-operation"].state == "planned"

                if tamper is None:
                    return
                stored_completed = records[completed_plan.record_id]
                ref = ContentRef.from_dict(stored_completed.record._payload["request"])
                if tamper == "key_destroy":
                    assert provider.destroy(store.tenant_id, ref.key_id).accepted
                    expected = KeyUnavailable
                elif tamper == "link":
                    turn_started = next(stored for stored in driver.records if stored.record.kind == "turn_started")
                    other_ref = ContentRef.from_dict(turn_started.record._payload["definition"])
                    with store.atomic():
                        store._rows(
                            "UPDATE sk_content_link SET content_id=? "
                            "WHERE tenant_id=? AND session_id=? AND record_seq=? AND slot='request'",
                            (other_ref.content_id, store.tenant_id, session_id, stored_completed.seq),
                        )
                    expected = MissingContentLink
                else:
                    row = store._one(
                        "SELECT ciphertext FROM sk_content WHERE tenant_id=? AND session_id=? AND content_id=?",
                        (store.tenant_id, session_id, ref.content_id),
                    )
                    assert row is not None
                    ciphertext = bytes(row[0])
                    tampered = ciphertext[:-1] + bytes([ciphertext[-1] ^ 1])
                    with store.atomic():
                        store._rows(
                            "UPDATE sk_content SET ciphertext=? WHERE tenant_id=? AND session_id=? AND content_id=?",
                            (tampered, store.tenant_id, session_id, ref.content_id),
                        )
                    expected = IntegrityFailure

                driver._invalidate_logical_view()
                with pytest.raises(expected):
                    driver._logical_view()
            finally:
                assert sessions.store.release(lease)
    finally:
        provider.close()


@pytest.mark.parametrize("tamper", ["key_destroy", "link", "ciphertext"])
def test_logical_content_frame_fails_closed_after_cached_prefix_frame(tmp_path: Path, tamper: str):
    session_id = f"logical-frame-{tamper}"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            try:
                with driver._logical_state_scope():
                    pass
                assert driver._active_logical_fold is None
                assert driver._logical_prefix is not None

                started = next(row for row in driver.records if row.record.kind == "turn_started")
                ref = ContentRef.from_dict(started.record._payload["definition"])
                if tamper == "key_destroy":
                    assert provider.destroy(store.tenant_id, ref.key_id).accepted
                    expected = KeyUnavailable
                elif tamper == "link":
                    other_ref = ContentRef.from_dict(started.record._payload["budget"])
                    with store.atomic():
                        store._rows(
                            "UPDATE sk_content_link SET content_id=? "
                            "WHERE tenant_id=? AND session_id=? AND record_seq=? AND slot=?",
                            (other_ref.content_id, store.tenant_id, session_id, started.seq, "definition"),
                        )
                    expected = MissingContentLink
                else:
                    row = store._one(
                        "SELECT ciphertext FROM sk_content WHERE tenant_id=? AND session_id=? AND content_id=?",
                        (store.tenant_id, session_id, ref.content_id),
                    )
                    assert row is not None
                    ciphertext = bytes(row[0])
                    tampered = ciphertext[:-1] + bytes([ciphertext[-1] ^ 1])
                    with store.atomic():
                        store._rows(
                            "UPDATE sk_content SET ciphertext=? WHERE tenant_id=? AND session_id=? AND content_id=?",
                            (tampered, store.tenant_id, session_id, ref.content_id),
                        )
                    expected = IntegrityFailure

                with pytest.raises(expected), driver._logical_state_scope():
                    pass
            finally:
                assert sessions.store.release(lease)
    finally:
        provider.close()


def test_logical_prefix_resume_reuses_fold_through_one_fresh_frame(tmp_path: Path):
    session_id = "logical-prefix-resume"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "logical-prefix-resume.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            try:
                with driver._logical_state_scope():
                    first_fold = driver._active_logical_fold
                    assert first_fold is not None
                assert driver._active_logical_fold is None
                assert driver._logical_prefix is not None

                frame_calls = 0
                original_frame = DurableContentReader._read_frame

                @contextmanager
                def counted_frame(self, *args, **kwargs):
                    nonlocal frame_calls
                    frame_calls += 1
                    with original_frame(self, *args, **kwargs) as reader:
                        yield reader

                def unexpected_full_fold(*_args, **_kwargs):
                    raise AssertionError("cached logical prefix must not fold the full history again")

                with (
                    patch.object(DurableContentReader, "_read_frame", counted_frame),
                    patch.object(kernel_module, "hydrate_and_fold_records", unexpected_full_fold),
                    driver._logical_state_scope(),
                ):
                    assert driver._active_logical_fold is first_fold
                    assert driver.state.session_id == session_id
                assert frame_calls == 1
            finally:
                assert sessions.store.release(lease)
    finally:
        provider.close()


@pytest.mark.parametrize("cycles", [8, 32])
def test_logical_prefix_scaling_folds_only_each_new_commit(tmp_path: Path, cycles: int):
    session_id = f"logical-prefix-scaling-{cycles}"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)
            calls: list[tuple[int, bool]] = []
            original_fold = kernel_module.hydrate_and_fold_records

            def counted_fold(source_records, *args, **kwargs):
                rows = tuple(source_records)
                calls.append((len(rows), kwargs.get("base_fold") is not None))
                return original_fold(rows, *args, **kwargs)

            try:
                with patch.object(kernel_module, "hydrate_and_fold_records", counted_fold):
                    for index in range(cycles):
                        with driver._logical_state_scope():
                            boundary = driver.boundary_record(
                                "before_memory",
                                f"scaling-{index}",
                                {"messages": [], "shared_state": {}},
                                source_digest=digest([message.to_dict() for message in driver.transcript()]),
                            )
                            driver.commit([boundary])

                assert len(calls) == cycles + 1
                assert calls[0][1] is False
                assert all(uses_base for _size, uses_base in calls[1:])
                assert driver._active_logical_fold is None
                assert driver._logical_prefix is not None
                assert len(driver._logical_prefix.fold.history) == len(driver.records)
            finally:
                assert sessions.store.release(lease)
    finally:
        provider.close()


def test_logical_commit_preserves_fresh_state_ownership_after_replay_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_id = "logical-replay-reload"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / "logical-replay-reload.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            driver, lease = _start_logical_driver(sessions, store, session_id)

            with driver._logical_state_scope():
                boundary = driver.boundary_record(
                    "before_memory",
                    "normal-commit",
                    {"messages": [], "shared_state": {}},
                    source_digest=digest([message.to_dict() for message in driver.transcript()]),
                )
                driver.commit([boundary])
                assert driver._state is driver._logical_projected_state
                assert driver._logical_dirty is False
                assert boundary.record_id in {record.record_id for record in driver.state.boundaries.values()}

                committed = next(row for row in driver.records if row.record.record_id == boundary.record_id)
                replay_records = [row.record for row in driver.records if row.commit_id == committed.commit_id]
                record_count = len(driver.records)
                load_version = getattr(driver, "_load_state_version", 0)
                monkeypatch.setattr(CoreDriver, "refresh", lambda self: self.load_state())
                driver.commit(replay_records)

                assert getattr(driver, "_load_state_version", 0) == load_version + 1
                assert len(driver.records) == record_count
                assert driver._state is driver.fold.state
                assert driver._logical_dirty is True
                assert boundary.record_id in {record.record_id for record in driver.state.boundaries.values()}

            assert sessions.store.release(lease)
    finally:
        provider.close()


@pytest.mark.parametrize("link_count", [8, 32])
def test_content_write_batch_reuses_transaction_key_and_content_queries(tmp_path: Path, link_count: int) -> None:
    session_id = f"content-write-batch-{link_count}"
    provider = LocalKeyProvider(b"k" * 32)
    counts: Counter[str] = Counter()
    original_fetchone = _SQLiteContentBackend.fetchone
    original_fetchall = _SQLiteContentBackend.fetchall
    original_run = _SQLiteContentBackend.run

    def counted_fetchone(self, query, params=()):
        if "sk_content" in query or "sk_session_key" in query or "sk_record" in query:
            counts["fetchone"] += 1
        if "sk_session_key" in query:
            counts["session_key_query"] += 1
        return original_fetchone(self, query, params)

    def counted_fetchall(self, query, params=()):
        if "sk_content" in query or "sk_session_key" in query or "sk_record" in query:
            counts["fetchall"] += 1
        return original_fetchall(self, query, params)

    def counted_run(self, query, params=()):
        if "sk_content" in query or "sk_session_key" in query or "sk_record" in query:
            counts["run"] += 1
        return original_run(self, query, params)

    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            sessions.create(session_id, "/workspace", {})
            head = store.read(session_id, limit=1).head_seq
            initial_content_count = store._one(
                "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
                (store.tenant_id, session_id),
            )[0]
            initial_link_count = store._one(
                "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
                (store.tenant_id, session_id),
            )[0]
            lease = store.acquire(session_id, owner="content-write-batch", ttl_ms=60_000)
            assert lease is not None
            try:
                with (
                    patch.object(provider, "unwrap", wraps=provider.unwrap) as unwrap,
                    patch.object(provider, "restore_data_key", wraps=provider.restore_data_key) as restore,
                    patch.object(_SQLiteContentBackend, "fetchone", counted_fetchone),
                    patch.object(_SQLiteContentBackend, "fetchall", counted_fetchall),
                    patch.object(_SQLiteContentBackend, "run", counted_run),
                    sessions.store.atomic() as tx,
                ):
                    writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                    refs = tuple(writer.put_bytes(b"same value") for _ in range(link_count))
                    assert len(set(refs)) == 1
                    records = tuple(
                        make_record(
                            "usage_observed",
                            session_id=session_id,
                            payload={
                                "meter_id": f"content-write-batch/{index}",
                                "observation": index + 1,
                                "mode": "cumulative",
                                "usage": ref.to_dict(),
                                "source": ref.to_dict(),
                            },
                        )
                        for index, ref in enumerate(refs)
                    )
                    receipt = tx.append(
                        session_id,
                        lease=lease,
                        expected_seq=head,
                        commit_id=f"content-write-batch-{link_count}",
                        records=records,
                    )
                    before_link = counts.copy()
                    writer.link_records(
                        (seq, ref, "provider.result")
                        for (_record_id, seq), ref in zip(receipt.record_sequences, refs, strict=True)
                    )
                    link_counts = counts - before_link
                    memo = writer._write_memo

                assert unwrap.call_count == 1
                assert restore.call_count == 1
                assert counts["session_key_query"] <= 2
                assert link_counts["fetchone"] == 0
                assert link_counts["fetchall"] <= 4
                assert link_counts["run"] <= 2
                assert not memo.session_keys and not memo.data_keys and not memo.content
                assert (
                    store._one(
                        "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
                        (store.tenant_id, session_id),
                    )[0]
                    == initial_content_count + 1
                )
                assert (
                    store._one(
                        "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
                        (store.tenant_id, session_id),
                    )[0]
                    == initial_link_count + link_count
                )
            finally:
                assert store.release(lease)
    finally:
        provider.close()


@pytest.mark.parametrize("rollback", [False, True])
def test_content_write_memo_clears_on_commit_and_rollback(tmp_path: Path, rollback: bool) -> None:
    session_id = f"content-write-memo-{'rollback' if rollback else 'commit'}"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            sessions.create(session_id, "/workspace", {})
            if rollback:
                with pytest.raises(RuntimeError, match="rollback content write"), sessions.store.atomic() as tx:
                    writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                    writer.put_bytes(b"transaction scoped")
                    memo = writer._write_memo
                    assert memo.session_keys and memo.data_keys and memo.content
                    raise RuntimeError("rollback content write")
            else:
                with sessions.store.atomic() as tx:
                    writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                    writer.put_bytes(b"transaction scoped")
                    memo = writer._write_memo
                    assert memo.session_keys and memo.data_keys and memo.content
            assert not memo.session_keys and not memo.data_keys and not memo.content
    finally:
        provider.close()


def test_content_link_batch_rejects_ref_tamper_before_mutation(tmp_path: Path) -> None:
    session_id = "content-write-tamper"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            sessions.create(session_id, "/workspace", {})
            head = store.read(session_id, limit=1).head_seq
            initial_link_count = store._one(
                "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
                (store.tenant_id, session_id),
            )[0]
            lease = store.acquire(session_id, owner="content-write-tamper", ttl_ms=60_000)
            assert lease is not None
            try:
                with sessions.store.atomic() as tx:
                    writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                    ref = writer.put_bytes(b"authenticated content")
                    record = make_record(
                        "usage_observed",
                        session_id=session_id,
                        payload={
                            "meter_id": "content-write-tamper",
                            "observation": 1,
                            "mode": "cumulative",
                            "usage": ref.to_dict(),
                            "source": ref.to_dict(),
                        },
                    )
                    receipt = tx.append(
                        session_id,
                        lease=lease,
                        expected_seq=head,
                        commit_id="content-write-tamper",
                        records=(record,),
                    )
                    seq = receipt.record_sequences[0][1]
                    tampered = replace(ref, mac="0" * 64)
                    with pytest.raises(ContentIntegrityError, match="metadata does not match"):
                        writer.link_records(((seq, tampered, "provider.result"),))
                    assert writer.get_bytes(ref) == b"authenticated content"
            finally:
                assert store.release(lease)
            assert (
                store._one(
                    "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
                    (store.tenant_id, session_id),
                )[0]
                == initial_link_count
            )
    finally:
        provider.close()


def test_content_link_batch_preserves_duplicate_conflict_and_pin_semantics(tmp_path: Path) -> None:
    session_id = "content-write-link-semantics"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            sessions.create(session_id, "/workspace", {})
            head = store.read(session_id, limit=1).head_seq
            lease = store.acquire(session_id, owner="content-write-link-semantics", ttl_ms=60_000)
            assert lease is not None
            try:
                with sessions.store.atomic() as tx:
                    writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                    first = writer.put_bytes(b"first")
                    conflicting = writer.put_bytes(b"conflicting")
                    expired = writer.put_bytes(b"expired pin")
                    record = make_record(
                        "usage_observed",
                        session_id=session_id,
                        payload={
                            "meter_id": "content-write-link-semantics",
                            "observation": 1,
                            "mode": "cumulative",
                            "usage": first.to_dict(),
                            "source": first.to_dict(),
                        },
                    )
                    receipt = tx.append(
                        session_id,
                        lease=lease,
                        expected_seq=head,
                        commit_id="content-write-link-semantics",
                        records=(record,),
                    )
                    seq = receipt.record_sequences[0][1]
                    writer.link_records(((seq, first, "provider.result"), (seq, first, "provider.result")))
                    with pytest.raises(ContentLinkConflict):
                        writer.link_records(((seq, conflicting, "provider.result"),))
                    store._rows(
                        "UPDATE sk_content SET pin_until_ms=0 WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                        (store.tenant_id, session_id, expired.content_id),
                    )
                    with pytest.raises(ContentUnavailable, match="staging pin is expired"):
                        writer.link_records(((seq, expired, "provider.usage"),))
                    links = writer.durable.read_links(store.tenant_id, session_id, record_seq=seq)
                    assert [(link.slot, link.ref) for link in links] == [("provider.result", first)]
                    assert store._one(
                        "SELECT state,pin_until_ms FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                        (store.tenant_id, session_id, first.content_id),
                    ) == ("live", None)
            finally:
                assert store.release(lease)
    finally:
        provider.close()


def test_content_write_rejects_destroyed_session_key(tmp_path: Path) -> None:
    session_id = "content-write-key-destroy"
    provider = LocalKeyProvider(b"k" * 32)
    try:
        with SQLiteStore.standalone(str(tmp_path / f"{session_id}.sqlite")) as store:
            store.install_schema()
            store.configure_input_admission(allow_unsigned=True)
            sessions = SessionDriver(store=store, content_key_provider=provider)
            sessions.create(session_id, "/workspace", {})
            with sessions.store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                ref = writer.put_bytes(b"before key destruction")
            assert provider.destroy(store.tenant_id, ref.key_id).accepted
            with pytest.raises(KeyUnavailable), sessions.store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
                writer.put_bytes(b"after key destruction")
    finally:
        provider.close()
