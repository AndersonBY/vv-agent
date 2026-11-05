"""Real kernel receipts for provider rejection versus ambiguous execution."""

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx2 as httpx
import pytest
from openai import APIStatusError

from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.session.result import project_result
from vv_agent.types import LLMResponse, Message

from .test_compaction import SUMMARY, compiled_system, configured, history
from .test_recovery_matrix import records_of, runtime, start


def http_error(status, message="Invalid user message", *, observed=True):
    error = APIStatusError(
        message,
        response=httpx.Response(status, request=httpx.Request("POST", "https://provider.invalid/chat")),
        body={"error": {"code": "invalid_message", "type": "invalid_request_error", "message": message}},
    )
    if observed:
        observe_error(error, status)
    return error


def observe_error(error, status):
    from vv_agent.llm.http_observation import HttpObservation

    # Synthetic adapter evidence; actual SDK retry coverage lives in
    # test_model_http_observation.py and never stamps exceptions this way.
    observation = HttpObservation([status])
    error._vv_http_observation = observation
    source = error
    while source is not None:
        response = getattr(source, "response", None)
        if response is not None:
            response.extensions["vv_agent_http_observation"] = observation
            break
        source = source.__cause__
    return error


def adapter_error(error):
    from vv_agent.llm.errors import ModelRequestRejected, model_request_rejection

    rejection = model_request_rejection(error)
    if rejection is None or isinstance(error, ModelRequestRejected):
        return error
    return ModelRequestRejected(
        rejection.status_code, rejection.message, provider_code=rejection.provider_code, provider_type=rejection.provider_type
    )


def context_http_error(kind, *, transport="httpx2", wrapped=False):
    from importlib import import_module

    from vv_llm.types import classify_exception

    module = import_module(transport)
    status = 413 if kind == "payload" else 400
    detail = {
        "openai": {"code": "context_length_exceeded", "message": "Input exceeds limit"},
        "typed": {"type": "context_length_exceeded", "message": "Input exceeds limit"},
        "anthropic": {"type": "invalid_request_error", "message": "prompt is too long: 210000 tokens"},
        "moonshot": {"type": "invalid_request_error", "message": "maximum context length exceeded"},
        "payload": {"type": "request_too_large", "message": "Payload exceeds limit"},
        "content_string": {"code": "string_above_max_length", "param": "messages[0].content", "message": "String exceeds limit"},
        "tool_string": {"code": "string_above_max_length", "param": "tools[0].function.name", "message": "String exceeds limit"},
    }[kind]
    response = module.Response(status, request=module.Request("POST", "https://provider.invalid"), json={"error": detail})
    if kind == "anthropic":
        from anthropic import APIStatusError as AnthropicStatusError

        error = AnthropicStatusError("Invalid input", response=response, body={"error": detail})
    elif kind in {"openai", "typed"}:
        error = APIStatusError("Invalid input", response=response, body={"error": detail})
    else:
        error = module.HTTPStatusError("Invalid input", request=response.request, response=response)
    observe_error(error, status)
    if wrapped:
        wrapper = classify_exception(error)
        wrapper.__cause__ = error
        return wrapper
    return error


def fail(error, calls):
    def complete(request):
        calls.append(request)
        raise error

    return complete


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_http_rejection_is_one_definitive_failure(store, database, status):
    calls = []
    start(store)
    drive(store, "s", runtime=runtime(database, [fail(http_error(status), calls)] * 2))
    assert len(calls) == 1
    assert not records_of(store, "op_unknown")
    receipt = records_of(store, "op_completed")[0].payload["result"]
    assert receipt["error_code"] == "model_request_rejected"
    assert receipt["rejection"] == {
        "status_code": status,
        "provider_code": "invalid_message",
        "provider_type": "invalid_request_error",
        "message": "Model provider request failed.",
        "retryable": False,
    }
    terminal = records_of(store, "turn_ended")[-1]
    assert terminal.payload["reason"] == "model_request_rejected"
    assert terminal.payload["status"] == "failed"
    assert terminal.payload["unconfirmed_operations"] == []
    assert "Model provider request failed." in terminal.payload["result"]
    events = project_records(read_state(store, "s")[1])
    assert not any(e.to_dict()["type"] == "model_retry_duplicate_risk" for e in events)
    assert project_result(store, "s", terminal.turn_id).error_code == "model_request_rejected"
    failure = next(e for e in events if e.to_dict()["type"] == "run_failed")
    assert f"HTTP {status}" in failure.to_dict()["error"]
    assert "Model provider request failed." in failure.to_dict()["error"]


@pytest.mark.parametrize("kind", ["openai", "typed", "anthropic", "moonshot", "payload", "content_string"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("transport", ["httpx2", "httpx"])
def test_http_context_error_compacts_then_next_primary_succeeds(store, database, kind, wrapped, transport):
    pytest.importorskip(transport)
    calls = []
    start(store)

    def answer(request):
        assert any(m.name == "memory_summary" for m in request.messages)
        return LLMResponse("done")

    rt = configured(
        database, [fail(context_http_error(kind, transport=transport, wrapped=wrapped), calls), LLMResponse(SUMMARY), answer]
    )
    rt.memory_manager.compact_threshold = 999999
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    receipts = records_of(store, "op_completed")
    assert receipts[0].payload["result"]["error_code"] == "prompt_too_long"
    assert not any(r.payload["result"].get("error_code") == "model_request_rejected" for r in receipts)
    assert [p.payload["purpose"] for p in records_of(store, "op_planned")] == ["primary", "compaction", "primary"]
    assert len(records_of(store, "context_compacted")) == 1
    assert not records_of(store, "op_unknown")
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["status"] == "completed"
    assert terminal["unconfirmed_operations"] == []
    assert not any(e.type == "model_retry_duplicate_risk" for e in project_records(read_state(store, "s")[1]))


@pytest.mark.parametrize("success", [True, False])
def test_rate_limit_has_definitive_receipts_and_bounded_retry(store, database, monkeypatch, success):
    monkeypatch.setattr("vv_agent.session.kernel.MODEL_REJECTION_BACKOFF_SECONDS", 0, raising=False)
    calls = []
    rejected = fail(http_error(429, "Rate limited"), calls)
    start(store)
    drive(store, "s", runtime=runtime(database, [rejected, LLMResponse("done") if success else rejected]))
    assert len(calls) == (1 if success else 2)
    assert len(records_of(store, "op_started")) == 2
    assert not records_of(store, "op_unknown")
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["status"] == ("completed" if success else "failed")
    assert terminal["unconfirmed_operations"] == []
    if not success:
        assert terminal["reason"] == "model_request_rejected"
    receipts = records_of(store, "op_completed")
    assert len(receipts) == 2
    assert receipts[0].payload["result"]["rejection"]["retryable"] is True
    assert (
        records_of(store, "op_planned")[0].payload["request_digest"]
        == records_of(store, "op_planned")[1].payload["request_digest"]
    )


@pytest.mark.parametrize(
    "error",
    [http_error(408), http_error(500), TimeoutError("timeout"), ConnectionError("disconnected"), RuntimeError("ordinary")],
)
def test_ambiguous_failures_keep_unknown_retry(store, database, error):
    calls = []
    step = fail(error, calls)
    start(store)
    drive(store, "s", runtime=runtime(database, [step, step]))
    assert len(calls) == 2
    assert len(records_of(store, "op_unknown")) == 2
    assert all(r.payload["duplicate_cost_risk"] for r in records_of(store, "op_unknown"))
    assert records_of(store, "turn_ended")[-1].payload["reason"] == "model_outcome_unknown"


def test_summary_rejection_preserves_history_and_continues(store, database):
    calls, seen = [], []

    def answer(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    start(store)
    drive(store, "s", runtime=configured(database, [fail(http_error(400), calls), answer]))
    assert len(calls) == 1
    assert not records_of(store, "op_unknown")
    assert not records_of(store, "context_compacted")
    assert seen == [compiled_system(), *history()[1:], Message("user", "go")]
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def test_rejection_redacts_before_persistence_and_projection(store, database):
    secret = "sk-super-secret-123456789"
    message = f"Invalid user. Authorization: Bearer {secret}\napi_key={secret}\n" + "x" * 900
    start(store)
    drive(store, "s", runtime=runtime(database, [fail(http_error(400, message), [])]))
    rows = read_state(store, "s")[1]
    saved = json.dumps([r.record.payload for r in rows])
    assert secret not in saved
    assert "Bearer" not in saved
    rejection = records_of(store, "op_completed")[0].payload["result"]["rejection"]
    assert len(rejection["message"]) <= 500
    assert rejection["message"] == "Model provider request failed."


@pytest.mark.parametrize("error", [http_error(400), context_http_error("openai")])
def test_stream_output_then_http_error_remains_unknown(store, database, error):
    calls = []

    class Streaming:
        def complete_with_stream(self, request, callback):
            calls.append(request)
            callback({"event": "assistant_delta", "content_delta": "partial"})
            raise error

    start(store)
    rt = runtime(database, [])
    rt.llm = Streaming()
    rt.config = replace(rt.config, stream=lambda _event: None)
    drive(store, "s", runtime=rt)
    assert len(calls) == 2
    assert len(records_of(store, "op_unknown")) == 2


@pytest.mark.parametrize("status,partial", [(400, False), (401, False), (429, False), (400, True), (500, True)])
@pytest.mark.parametrize("with_callback", [False, True])
def test_real_vv_llm_adapter_preserves_rejection_and_stream_boundary(
    store, database, monkeypatch, status, partial, with_callback
):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

    calls = []

    def create(**_kwargs):
        def complete(**_payload):
            calls.append(1)

            def chunks():
                if partial:
                    yield SimpleNamespace(content="partial")
                raise adapter_error(http_error(status))

            return chunks()

        return SimpleNamespace(create_completion=complete)

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", create)
    monkeypatch.setattr("vv_agent.session.kernel.MODEL_REJECTION_BACKOFF_SECONDS", 0)
    rt = runtime(database, [])
    rt.llm = VvLlmClient([EndpointTarget("a", "unused", "https://provider.invalid")], backend="openai")
    if with_callback:
        rt.config = replace(rt.config, stream=lambda _event: None)
    start(store)
    drive(store, "s", runtime=rt)
    assert len(calls) == (2 if partial or status == 429 else 1)
    assert len(records_of(store, "op_unknown")) == (2 if partial else 0)
    assert records_of(store, "turn_ended")[-1].payload["reason"] == (
        "model_outcome_unknown" if partial else "model_request_rejected"
    )


def test_rate_limit_backoff_survives_restart_and_does_not_repeat_rejected_attempt(store, database, monkeypatch):
    class WorkerDeath(BaseException):
        pass

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.attempt == 2:
            raise WorkerDeath

    calls = []
    start(store)
    now = store._now()
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=runtime(database, [fail(http_error(429), calls)], hook=cut))
    successor = records_of(store, "op_planned")[-1]
    assert successor.attempt == 2 and successor.payload["not_before_ms"] >= now + 2000
    rt = runtime(database, [LLMResponse("done")])
    drive(store, "s", runtime=rt)
    assert len(records_of(store, "op_started")) == 1
    assert not records_of(store, "turn_ended")
    monkeypatch.setattr(store, "clock_sql", str(successor.payload["not_before_ms"] + 1))
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    assert len(records_of(store, "op_started")) == 2
    assert not records_of(store, "op_unknown")
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def test_explicit_temporary_rejection_retries(store, database, monkeypatch):
    from vv_agent.llm.errors import ModelRequestRejected

    monkeypatch.setattr("vv_agent.session.kernel.MODEL_REJECTION_BACKOFF_SECONDS", 0)
    start(store)
    drive(
        store, "s", runtime=runtime(database, [fail(ModelRequestRejected(409, "Busy", retryable=True), []), LLMResponse("done")])
    )
    assert len(records_of(store, "op_completed")) == 2
    assert not records_of(store, "op_unknown")
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


@pytest.mark.parametrize("status", [408, 500])
def test_http_ambiguous_status_never_uses_prompt_too_long_text_heuristic(store, database, status):
    step = fail(http_error(status, "maximum context length exceeded"), [])
    start(store)
    drive(store, "s", runtime=runtime(database, [step, step]))
    assert len(records_of(store, "op_unknown")) == 2
    assert not records_of(store, "op_completed")


@pytest.mark.parametrize("success", [True, False])
def test_summary_rate_limit_can_recover_or_keep_original_history(store, database, monkeypatch, success):
    monkeypatch.setattr("vv_agent.session.kernel.MODEL_REJECTION_BACKOFF_SECONDS", 0)
    rejected = fail(http_error(429), [])
    start(store)
    drive(
        store,
        "s",
        runtime=configured(database, [rejected, LLMResponse(SUMMARY) if success else rejected, LLMResponse("done")]),
    )
    assert not records_of(store, "op_unknown")
    assert len(records_of(store, "context_compacted")) == int(success)
    assert [r.payload["purpose"] for r in records_of(store, "op_planned")] == ["compaction", "compaction", "primary"]
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def test_nonretryable_definitive_result_cannot_authorize_successor(store, database):
    from vv_agent.session.records import InboxItem, make_record
    from vv_agent.session.reducer import TransitionError, fold

    class WorkerDeath(BaseException):
        pass

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_completed":
            raise WorkerDeath

    start(store)
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=runtime(database, [fail(http_error(400), [])], hook=cut))
    log = [r.record for r in read_state(store, "s")[1]]
    plan = records_of(store, "op_planned")[0]
    forged = make_record(
        "op_planned", session_id="s", turn_id=plan.turn_id, operation_id=plan.operation_id, attempt=2, payload=plan.payload
    )
    inputs = [InboxItem(**r.payload["input"]) for r in log if r.kind == "input_applied"]
    with pytest.raises(TransitionError, match="retry not authorized"):
        fold([*log, forged], consumed_inputs=inputs)


@pytest.mark.parametrize("child", [False, True])
def test_json_in_rejection_message_cannot_become_a_summary(store, database, child):
    from .test_summary_requests import call
    from .test_summary_requests import configured as summary_runtime

    rejected = fail(http_error(400, SUMMARY), [])
    rt = (
        summary_runtime(database, [call("child_task"), rejected, LLMResponse("done")], child=True)
        if child
        else configured(database, [rejected, LLMResponse("done")])
    )
    start(store)
    drive(store, "s", runtime=rt)
    assert not records_of(store, "op_unknown")
    assert not records_of(store, "context_compacted")
    if child:
        assert not records_of(store, "op_parked")
        assert any(
            r.payload["result"].get("error_code") == "child_summary_unavailable" for r in records_of(store, "op_completed")
        )
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def test_rejection_does_not_settle_an_earlier_genuinely_unknown_attempt(store, database):
    start(store)
    drive(store, "s", runtime=runtime(database, [fail(TimeoutError("lost"), []), fail(http_error(400), [])]))
    unknowns = records_of(store, "op_unknown")
    assert len(unknowns) == 1 and unknowns[0].payload["duplicate_cost_risk"]
    assert len(records_of(store, "op_completed")) == 1
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["reason"] == "model_request_rejected"
    assert terminal["unconfirmed_operations"] == [unknowns[0].operation_id]


@pytest.mark.parametrize("child", [False, True])
def test_http_context_error_in_summary_keeps_existing_failure_semantics(store, database, child):
    from .test_summary_requests import call
    from .test_summary_requests import configured as summary_runtime

    calls = []
    too_long = fail(context_http_error("openai", wrapped=True), calls)
    rt = (
        summary_runtime(database, [call("child_task"), too_long, LLMResponse("done")], child=True)
        if child
        else configured(database, [too_long, LLMResponse("done")])
    )
    start(store)
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    assert not records_of(store, "op_unknown")
    assert not records_of(store, "context_compacted")
    codes = [r.payload["result"].get("error_code") for r in records_of(store, "op_completed")]
    assert "prompt_too_long" in codes and "model_request_rejected" not in codes
    if child:
        assert not records_of(store, "op_parked")
        assert "child_summary_unavailable" in codes
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


@pytest.mark.parametrize("error", [http_error(400), context_http_error("openai")])
def test_http_errors_in_output_repair_still_use_repair_provider_error(store, database, error):
    from vv_agent.output_validation import OutputValidationResult

    calls = []
    rt = runtime(database, [LLMResponse("invalid")])
    rt.agent.output_validation_enabled = True
    rt.agent.output_validator = lambda value, context: OutputValidationResult.reject("invalid")
    rt.agent.output_repair = fail(error, calls)
    start(store)
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    assert not records_of(store, "op_unknown")
    assert records_of(store, "op_completed")[-1].payload["result"]["error_code"] == "repair_provider_error"
    assert records_of(store, "turn_ended")[-1].payload["reason"] == "output_validation_failed"


def test_http_context_exhaustion_is_compaction_failure_not_rejection(store, database):
    too_long = fail(context_http_error("openai"), [])
    rt = configured(database, [too_long, LLMResponse("invalid"), too_long, LLMResponse("invalid"), too_long, too_long])
    rt.memory_manager.compact_threshold = 999999
    rt.memory_manager.keep_recent_messages = 2
    start(store)
    drive(store, "s", runtime=rt)
    assert records_of(store, "turn_ended")[-1].payload["reason"] == "CompactionExhaustedError"
    assert not records_of(store, "op_unknown")
    assert not any(r.payload["result"].get("error_code") == "model_request_rejected" for r in records_of(store, "op_completed"))


@pytest.mark.parametrize("kind", ["rejected", "context", "timeout"])
def test_session_memory_error_paths_keep_definitive_and_unknown_distinct(store, database, tmp_path, kind):
    error = {"rejected": http_error(400), "context": context_http_error("openai"), "timeout": TimeoutError("lost")}[kind]
    calls = []
    step = fail(error, calls)
    rt = configured(database, [step, step] if kind == "timeout" else [step, LLMResponse("done")])
    rt.memory_manager.compact_threshold = 999999
    rt.config = replace(
        rt.config,
        workspace=tmp_path,
        session_memory_enabled=True,
        metadata={"session_memory_min_tokens": 1, "session_memory_min_text_messages": 1},
    )
    start(store)
    drive(store, "s", runtime=rt)
    assert records_of(store, "op_planned")[0].payload["purpose"] == "session_memory"
    assert len(calls) == (2 if kind == "timeout" else 1)
    terminal = records_of(store, "turn_ended")[-1].payload
    if kind == "context":
        assert terminal["status"] == "completed"
        assert records_of(store, "op_completed")[0].payload["result"]["error_code"] == "prompt_too_long"
    else:
        assert terminal["reason"] == ("model_outcome_unknown" if kind == "timeout" else "model_request_rejected")
    assert len(records_of(store, "op_unknown")) == (2 if kind == "timeout" else 0)
