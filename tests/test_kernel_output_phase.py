"""Real SDK producers and SDK-free replay of the output phase's decision cuts."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

from vv_agent import Agent, GuardrailResult, OutputValidationResult, RunConfig
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.result import project_result
from vv_agent.session.runtime import Runtime
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

CASES = (
    "disabled",
    "guardrail_rewrite",
    "guardrail_block",
    "type_invalid",
    "type_repair",
    "validation_repair",
    "receipt_error",
    "validator_error",
    "validator_contract",
    "invalid_repair",
    "guardrail_mutates_settings",
    "validator_mutates_settings",
    "repairs_disabled",
)


def produce(case: str, *, guardrail_error: Exception | None = None) -> dict[str, Any]:
    trace = []

    def validate(value, context):
        trace.append(["validate", value, context.agent_name, context.output_type is dict])
        if case == "validator_error":
            raise RuntimeError("validator unavailable")
        if case == "validator_contract":
            return True
        if case == "validator_mutates_settings":
            agent.output_repair = repair
            agent.output_repair_model = "late-repair-model"
            agent.output_repair_model_settings = {"temperature": 0.25}
        if value == "valid" or value == {"answer": "valid"}:
            return OutputValidationResult.accept()
        return OutputValidationResult.reject("invalid", "repair needed")

    def repair(request):
        trace.append(["repair", request.invalid_output, request.validation_code, request.model, list(request.tools)])
        if case == "validator_mutates_settings":
            assert request.model_settings == {"temperature": 0.25}
        if case == "receipt_error":
            raise RuntimeError("repair unavailable")
        if case == "invalid_repair":
            return "still invalid"
        return '{"answer":"valid"}' if case == "type_repair" else "valid"

    def guardrail(context, value):
        trace.append(["guardrail", value])
        if guardrail_error is not None:
            raise guardrail_error
        if case == "guardrail_block":
            return GuardrailResult.block("blocked")
        if case == "guardrail_mutates_settings":
            agent.output_type = dict
            agent.output_validation_enabled = True
            agent.output_validator = validate
            return GuardrailResult.rewrite('{"answer":"valid"}')
        return GuardrailResult.rewrite("redacted")

    agent = Agent("output-agent", "Answer.", model="m")
    if case.startswith("guardrail"):
        agent.output_guardrails = [guardrail]
    if case in {"type_invalid", "type_repair"}:
        agent.output_type = dict
    if case not in {"disabled", "guardrail_rewrite", "guardrail_block", "type_invalid", "guardrail_mutates_settings"}:
        agent.output_validation_enabled = True
        agent.output_validator = validate
        agent.output_repair = None if case == "validator_mutates_settings" else repair
    if case == "repairs_disabled":
        agent.output_validation_max_repairs = 0

    settings = {
        "has_guardrails": bool(agent.output_guardrails),
        "has_output_type": agent.output_type is not None,
        "validation_enabled": agent.output_validation_enabled,
        "has_validator": agent.output_validator is not None,
        "has_repair": agent.output_repair is not None,
        "max_repairs": agent.output_validation_max_repairs,
        "repair_model": agent.output_repair_model,
    }

    def respond(request):
        trace.append(["model", request.metadata["purpose"]])
        return LLMResponse("invalid")

    def hook(point, record):
        if point != "after_commit" or record is None:
            return
        if record.kind == "boundary_recorded" and record.payload["stage"] == "output_checked":
            trace.append(["boundary", record.payload["boundary_id"], record.payload["data"]])
        elif record.kind == "op_planned" and record.payload["purpose"] == "output_repair":
            trace.append(["plan_repair", record.payload["request"]["model"], record.payload["request"]["tools"]])
        elif record.kind == "turn_ended":
            trace.append(["terminal", record.payload["status"], record.payload["reason"], record.payload["result"]])

    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "test", "/tmp"), consumers=("events",))
            tx.push("s", InboxItem("initial", "user", {"content": "go"}))
        runtime = Runtime(
            agent,
            RunConfig(no_tool_policy="finish", max_cycles=1),
            ResolvedModelConfig("scripted", "m", "m", "m", []),
            ScriptedLLM([respond]),
            lambda: nullcontext(store),
            hook=hook,
        )
        drive(store, "s", runtime=runtime)
        rows = store.read_state("s")[1]
        records = [row.record for row in rows]
        result = project_result(store, "s", "s/turn/initial", runtime=runtime)
        before_replay = list(trace)
        drive(store, "s", runtime=runtime)
        project_result(store, "s", "s/turn/initial", runtime=runtime)
        assert trace == before_replay
        return {
            "trace": trace,
            "settings": settings,
            "output_records": [
                r.encode().decode()
                for r in records
                if (r.kind == "boundary_recorded" and r.payload["stage"] == "output_checked")
                or (r.kind == "op_planned" and r.payload["purpose"] == "output_repair")
            ],
            "result": {
                "status": result.status.value,
                "output": result.final_output,
                "error": result.raw_result.error,
                "partial": result.partial_output,
            },
            "records": [r.to_dict() for r in records],
        }


@pytest.mark.parametrize("case", CASES)
def test_output_producers_and_terminal_replay(case):
    observed = produce(case)
    callbacks = [item[0] for item in observed["trace"] if item[0] in {"model", "guardrail", "validate", "repair"}]
    assert callbacks.count("model") == 1
    assert callbacks.count("repair") <= 1
    assert len([item for item in observed["trace"] if item[0] == "terminal"]) == 1
    if case == "guardrail_mutates_settings":
        assert callbacks == ["model", "guardrail", "validate"]
        assert observed["result"]["output"] == {"answer": "valid"}
    if case == "validator_mutates_settings":
        assert callbacks == ["model", "validate", "repair", "validate"]
        assert [item for item in observed["trace"] if item[0] == "plan_repair"] == [["plan_repair", "late-repair-model", []]]
    if case == "type_repair":
        assert callbacks == ["model", "repair", "validate"]
        assert observed["result"]["output"] == {"answer": "valid"}
    if case in {"validator_error", "validator_contract", "repairs_disabled"}:
        assert callbacks == ["model", "validate"]
    if case in {
        "guardrail_block",
        "type_invalid",
        "receipt_error",
        "validator_error",
        "validator_contract",
        "invalid_repair",
        "repairs_disabled",
    }:
        assert observed["result"]["status"] == "failed"
    else:
        assert observed["result"]["status"] == "completed"


CORE_SCRIPT = r"""
import encodings.utf_16_be
import importlib.abc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
data = json.loads(Path(sys.argv[2]).read_text())
class OnlyCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, OnlyCore())
from vv_agent.kernel import _output_phase as phase, _event_validation, _message_codec
from vv_agent.kernel._driver_requests import CommitRecords, CreateBoundary, PlanModel
from vv_agent.kernel._records import Record, InboxItem, make_record
from vv_agent.kernel._reducer import Fold
records = [Record(**value) for value in data['records']]
observations = iter(data['observations'])
def forbidden(*args, **kwargs): raise AssertionError('unexpected core effect')
time.monotonic = time.monotonic_ns = time.time = time.sleep = forbidden
def audit(event, args):
    if event in {'open', 'os.listdir', 'os.scandir', 'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen'}:
        forbidden()
sys.addaudithook(audit)
boundaries = [r for r in records if r.kind == 'boundary_recorded' and r._payload['stage'] == 'output_checked']
terminal = next(r for r in records if r.kind == 'turn_ended')
for target in boundaries or [terminal]:
    prefix = records[:records.index(target)]
    fold = Fold(normalize_memory_settings=forbidden, count_messages=forbidden)
    fold.extend(prefix, consumed_inputs=[InboxItem(**r._payload['input']) for r in prefix if r.kind == 'input_applied'])
    steps = phase.prepare_output(fold.state, 'invalid')
    expected = [target] if target is not terminal else []
    repairing = target is not terminal and target._payload['boundary_id'] == 'initial'
    if repairing:
        expected.append(next(r for r in records if r.kind == 'op_planned' and r._payload['purpose'] == 'output_repair'))
    response, proposed, trace = None, [], []
    while True:
        try: request = steps.send(response)
        except StopIteration as done:
            expected_outcome = (None if repairing else ['completed', None, 'invalid'] if target is terminal else
                                [target._payload['data'][key] for key in ('status', 'reason', 'value')])
            assert (list(done.value) if done.value else None) == expected_outcome
            break
        response = None
        trace.append(request.value if isinstance(request, phase.OutputSetting) else type(request).__name__)
        if isinstance(request, phase.OutputSetting):
            response = data['settings'][request.value]
        elif isinstance(request, phase.PostprocessOutput):
            observation = next(observations)
            assert observation['kind'] == 'postprocess' and request.turn_id == 's/turn/initial'
            response = phase.PostprocessedOutput(observation['value'],
                ValueError(observation['error']) if observation['error'] is not None else None, observation['failed'])
        elif isinstance(request, phase.ValidateOutput):
            observation = next(observations)
            assert observation['kind'] == 'validate'
            actual_input = [request.value, str(request.coercion_error) if request.coercion_error else None, request.coerce]
            assert actual_input == observation['input']
            response = (observation['value'], SimpleNamespace(**observation['validation']), observation['repairable'])
        elif isinstance(request, phase.ValidationError):
            observation = next(observations)
            assert observation['kind'] == 'validation_error'
            assert vars(request.validation) == observation['validation']
            response = observation['error']
        elif isinstance(request, phase.SerializeOutput):
            response = phase.serializable_output(request.value)
        elif isinstance(request, CreateBoundary):
            response = make_record('boundary_recorded', session_id='s', turn_id='s/turn/initial', payload={
                'stage': request.stage, 'boundary_id': request.identity, 'source_operation_id': None,
                'source_digest': None, 'data': request.data,
            })
            assert response.encode() == expected[0].encode()
            proposed.append(response)
        elif isinstance(request, PlanModel):
            saved = expected[1]
            assert request.oid == saved.operation_id and request.purpose == 'output_repair'
            expected_request = saved.payload['request']
            metadata = {k: v for k, v in expected_request['metadata'].items() if k != 'vv_session'}
            expected_request = expected_request | {'metadata': metadata}
            assert request.request == expected_request and request.request['tools'] == []
            response = saved
            proposed.append(saved)
        elif isinstance(request, CommitRecords):
            assert [r.encode() for r in request.records] == [r.encode() for r in expected]
        else:
            raise AssertionError(type(request).__name__)
    assert [r.encode() for r in proposed] == [r.encode() for r in expected]
    if target is terminal:
        assert trace == ['has_guardrails', 'has_output_type', 'validation_enabled']
    else:
        assert trace[-2:] == (['PlanModel', 'CommitRecords'] if repairing else ['CreateBoundary', 'CommitRecords'])
    if target is not terminal and target._payload['boundary_id'] == 'final':
        # A durable final boundary wins before any SDK setting or callback is requested.
        fold.extend([target])
        replay = phase.prepare_output(fold.state, 'different')
        try: next(replay)
        except StopIteration as done:
            assert list(done.value) == [target._payload['data'][key] for key in ('status', 'reason', 'value')]
        else: raise AssertionError('replayed final output requested SDK work')
assert next(observations, None) is None
if data['settings']['has_output_type'] and len(boundaries) == 2:
    initial = boundaries[0]
    plan = next(r for r in records if r.kind == 'op_planned' and r._payload['purpose'] == 'output_repair')
    for target in (initial, plan):
        prefix = records[:records.index(target) + 1]
        fold = Fold(normalize_memory_settings=forbidden, count_messages=forbidden)
        fold.extend(prefix, consumed_inputs=[InboxItem(**r._payload['input']) for r in prefix if r.kind == 'input_applied'])
        steps = phase.prepare_output(fold.state, 'invalid')
        assert next(steps) is phase.OutputSetting.HAS_GUARDRAILS
        assert steps.send(False) is phase.OutputSetting.HAS_OUTPUT_TYPE
        if target is initial:
            try: steps.send(True)
            except ValueError as exc: assert str(exc) == 'output repair boundary has no logged operation'
            else: raise AssertionError('orphan output repair boundary accepted')
        else:
            request = steps.send(True)
            assert isinstance(request, CreateBoundary) and request.identity == 'final'
            assert request.data == {'status': 'failed', 'reason': 'output_validation_failed',
                'value': 'output_validation_failed: repair_provider_error', 'partial_output': 'invalid'}
            assert isinstance(steps.send(boundaries[1]), CommitRecords)
            try: steps.send(None)
            except StopIteration as done:
                assert done.value == ('failed', 'output_validation_failed', 'output_validation_failed: repair_provider_error')
            else: raise AssertionError('missing receipt did not finish')
assert all(n in {'vv_agent', 'vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "guardrail_rewrite",
        "guardrail_block",
        "type_invalid",
        "type_repair",
        "validation_repair",
        "receipt_error",
        "validator_error",
        "validator_contract",
        "invalid_repair",
        "repairs_disabled",
    ],
)
@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_decisions_replay_real_producer_observations_without_sdk(case, site_flags, monkeypatch, tmp_path):
    import json
    import subprocess
    import sys
    from dataclasses import asdict
    from pathlib import Path

    from test_token_count_suspension import assert_core_suspended

    from vv_agent.session import output

    observations = []
    postprocess = output.postprocess_output
    validate = output.run_output_validator
    validation_error = output.output_validation_error

    def observed_postprocess(**kwargs):
        assert "prepare_output" in assert_core_suspended()
        value, error = postprocess(**kwargs)
        observations.append(
            {
                "kind": "postprocess",
                "value": value,
                "error": str(error) if error else None,
                "failed": kwargs["raw_result"].status.value == "failed",
            }
        )
        return value, error

    def observed_validate(**kwargs):
        assert "prepare_output" in assert_core_suspended()
        value, validation, repairable = validate(**kwargs)
        observations.append(
            {
                "kind": "validate",
                "value": value,
                "validation": asdict(validation),
                "repairable": repairable,
                "input": [
                    kwargs["candidate"],
                    str(kwargs["output_coercion_error"]) if kwargs["output_coercion_error"] else None,
                    kwargs["coerce"],
                ],
            }
        )
        return value, validation, repairable

    def observed_error(validation):
        assert "prepare_output" in assert_core_suspended()
        error = validation_error(validation)
        observations.append({"kind": "validation_error", "validation": asdict(validation), "error": error})
        return error

    monkeypatch.setattr(output, "postprocess_output", observed_postprocess)
    monkeypatch.setattr(output, "run_output_validator", observed_validate)
    monkeypatch.setattr(output, "output_validation_error", observed_error)
    data = produce(case)
    data["observations"] = observations
    history = tmp_path / "current-producer.json"
    history.write_text(json.dumps(data))
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", CORE_SCRIPT, str(Path(__file__).parents[1]), str(history)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Captured on the committed 4f5b2bb producer before extraction: callback order,
# terminal projection and full canonical output-boundary/repair-plan bytes.
BASELINE_SHA256 = {
    "guardrail_rewrite": "2585675cf2bb4b69180cb6e6283c41c6d59bb5788feb9cf0a5e79b6090594c65",
    "guardrail_block": "b11dcad67f808c854fe2ad3f35b8298cbb36854119794d0f5f7517c076caaec3",
    "type_invalid": "f266a2d13ed912d645ddf28c13b882621aec2141ffd3780eaa8001957519b0dc",
    "type_repair": "fb5cb06b69454d5c13a53daa8487cb28079bfa3502bfe102d2b42a1f42068d08",
    "validation_repair": "e614b3ea7b7573ce758ee1eb24a1efccfe77a152681241dc2fe9f8d8f87c1275",
    "receipt_error": "631b9e0b6c1b015e0e0397c65013f36cb118998262ec2a002ae80c10a0a06fd7",
}


@pytest.mark.parametrize("case", BASELINE_SHA256)
def test_before_after_traces_and_canonical_output_bytes(case):
    import hashlib
    import json

    observed = produce(case)
    projection = {key: observed[key] for key in ("trace", "result", "output_records")}
    body = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(body).hexdigest() == BASELINE_SHA256[case]


def test_sdk_and_session_share_the_actual_serializer():
    from dataclasses import dataclass

    from pydantic import BaseModel

    from vv_agent.kernel import _output_phase as phase
    from vv_agent.result import RunResult
    from vv_agent.session import output

    @dataclass
    class Answer:
        value: str

    class ModelAnswer(BaseModel):
        value: str

    assert output.serializable_output is RunResult._serializable_output is phase.serializable_output
    for serialize in (output.serializable_output, RunResult._serializable_output):
        assert serialize(Answer("答复")) == {"value": "答复"}
        assert serialize(ModelAnswer(value="答复")) == {"value": "答复"}
        assert serialize(Answer) is Answer
        for value in (None, "text", [1], {"nested": Answer("value")}):
            assert serialize(value) is value


@pytest.mark.parametrize("error_type", [LookupError, StopIteration])
def test_real_sdk_guardrail_exceptions_keep_their_identity_and_cause(error_type):
    error = error_type("guardrail failed")
    cause = ValueError("guardrail cause")
    error.__cause__ = cause
    with pytest.raises(error_type) as caught:
        produce("guardrail_rewrite", guardrail_error=error)
    assert caught.value is error and caught.value.__cause__ is cause
