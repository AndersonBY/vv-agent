"""HTTP status is pending until raw EOF and successful decoder finalization."""

from asyncio import CancelledError
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import Context
from importlib import import_module
from typing import Any, cast
from unittest.mock import patch

import httpx2
import pytest
from httpx2._decoders import GZipDecoder
from session.test_model_rejection import http_error
from session.test_model_response_completion import failed_flush, response_for

from vv_agent.llm import errors
from vv_agent.llm.errors import ModelRequestRejected, is_prompt_too_long_error, model_request_rejection
from vv_agent.llm.http_observation import HttpObservation, ObservedAnthropicHttpClient, ObservedHttpClient


@pytest.mark.parametrize("client_type", [ObservedHttpClient, ObservedAnthropicHttpClient])
@pytest.mark.parametrize("consumer", ["same_context", "new_context", "thread"])
@pytest.mark.parametrize("case", ["pre_raw", "flush", "mid_body", "closed_unread", "partial_close", "raw_only", "complete"])
def test_response_body_completion_without_prefetch_is_required(client_type, consumer, case):
    observation = HttpObservation()
    response, body = response_for(case, 400)

    def consume():
        if case == "closed_unread":
            response.close()
        elif case == "partial_close":
            iterator = response.iter_bytes()
            assert next(iterator)
            response.close()
            iterator.close()
        elif case == "raw_only":
            assert b"".join(response.iter_raw())
        elif case == "complete":
            response.read()
        else:
            with (
                patch.object(GZipDecoder, "flush", failed_flush) if case == "flush" else nullcontext(),
                pytest.raises(httpx2.DecodingError),
            ):
                response.read()

    with (
        client_type(observation, transport=httpx2.MockTransport(lambda r: response), trust_env=False) as client,
        # Test the lower stream observer independently of the bounded pre-reader.
        patch.object(client, "_prefetch", lambda *_: None),
    ):
        client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
        assert not observation.all_rejected
        if consumer == "new_context":
            Context().run(consume)
        elif consumer == "thread":
            with ThreadPoolExecutor(1) as pool:
                pool.submit(consume).result()
        else:
            consume()
        assert observation.all_rejected is (case == "complete")
        # The caught decode error is absent from this graph: observation alone must veto it.
        error = ModelRequestRejected(400, "maximum context length exceeded")
        cast(Any, error)._vv_http_observation = observation
        assert (model_request_rejection(error) is error) is (case == "complete")
        assert is_prompt_too_long_error(error) is (case == "complete")
    assert body.closed
    if case in {"closed_unread", "partial_close", "pre_raw", "flush", "mid_body"}:
        assert observation.outcomes == ["incomplete"]
    assert body.exhausted is (case in {"flush", "raw_only", "complete"})


@pytest.mark.parametrize("case", ["pre_raw", "flush", "mid_body"])
def test_swallowed_decode_failure_cannot_be_erased_by_later_400(case):
    observation = HttpObservation()

    def handle(request):
        return response_for(case, 400)[0] if request.url.path == "/bad" else httpx2.Response(400, json={})

    with (
        ObservedHttpClient(observation, transport=httpx2.MockTransport(handle), trust_env=False) as client,
        patch.object(GZipDecoder, "flush", failed_flush) if case == "flush" else nullcontext(),
    ):
        response = client.send(client.build_request("GET", "https://provider.invalid/bad"), stream=True)
        with (
            patch.object(GZipDecoder, "flush", failed_flush) if case == "flush" else nullcontext(),
            pytest.raises(httpx2.DecodingError),
        ):
            response.read()
        good = client.get("https://provider.invalid/good")
        error = httpx2.HTTPStatusError("Rejected", request=good.request, response=good)
        cast(Any, error)._vv_http_observation = observation
        assert observation.outcomes == ["incomplete", 400]
        assert model_request_rejection(error) is None
        assert not is_prompt_too_long_error(error)


@pytest.mark.parametrize("close", [False, True])
def test_explicit_rejection_cannot_override_pending_response_evidence(close):
    observation = HttpObservation()
    response, _ = response_for("complete", 413)
    with (
        ObservedHttpClient(observation, transport=httpx2.MockTransport(lambda r: response), trust_env=False) as client,
        patch.object(client, "_prefetch", lambda *_: None),
    ):
        client.send(client.build_request("GET", "https://provider.invalid"), stream=True)
        if close:
            response.close()
        error = ModelRequestRejected(413, "request too large")
        # Only the HTTP response carries the observation; the adapter kept its cause.
        error.__cause__ = httpx2.HTTPStatusError("Rejected", request=response.request, response=response)
        assert model_request_rejection(error) is None
        assert not is_prompt_too_long_error(error)


@pytest.mark.parametrize("library", ["httpx2", "httpx"])
@pytest.mark.parametrize("entry", ["explicit", "automatic", "prompt_too_long"])
@pytest.mark.parametrize("edge", ["cause", "context", "args", "group"])
@pytest.mark.parametrize("reverse", [False, True])
def test_actual_decoding_error_vetoes_all_positive_evidence(library, entry, edge, reverse):
    module = pytest.importorskip(library)
    native = module.DecodingError("synthetic decode failure")
    positive = ModelRequestRejected(413, "request too large") if entry == "explicit" else http_error(413)
    outer, inner = (native, positive) if reverse else (positive, native)
    if edge == "group":
        outer = ExceptionGroup("mixed", [outer, inner])
    elif edge == "args":
        outer.args = (*outer.args, inner)
    else:
        setattr(outer, "__" + edge + "__", inner)
    assert model_request_rejection(outer) is None
    assert not is_prompt_too_long_error(outer)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("failure", ["ImportError", "RuntimeError", "missing_export", "invalid_export", "non_exception_type"])
def test_optional_vendor_load_fault_is_isolated(monkeypatch, provider, failure):
    from types import SimpleNamespace

    other = import_module("anthropic" if provider == "openai" else "openai")

    def load(name):
        if name != provider:
            return import_module(name)
        if failure == "missing_export":
            return SimpleNamespace()
        if failure in {"invalid_export", "non_exception_type"}:
            return SimpleNamespace(APIConnectionError=None if failure == "invalid_export" else object)
        raise (ImportError if failure == "ImportError" else RuntimeError)("synthetic vendor initialization failure")

    monkeypatch.setattr(errors, "import_module", load)
    assert errors._provider_connection_types() == (other.APIConnectionError,)
    for error in (ModelRequestRejected(400, "Rejected"), http_error(400)):
        assert model_request_rejection(error) is not None
        assert not is_prompt_too_long_error(error)
    assert is_prompt_too_long_error(ModelRequestRejected(413, "too large"))
    error = http_error(400)
    error.__cause__ = other.APIConnectionError(request=httpx2.Request("POST", "https://provider.invalid"))
    assert model_request_rejection(error) is None
    assert not is_prompt_too_long_error(error)
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://provider.invalid"), json={})
    observation = HttpObservation([400])
    response.extensions["vv_agent_http_observation"] = observation
    status_error = other.BadRequestError("Rejected", response=response, body={})
    status_error._vv_http_observation = observation
    assert model_request_rejection(status_error) is not None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, CancelledError])
def test_optional_vendor_load_does_not_swallow_base_exception(monkeypatch, provider, interrupt):
    def load(name):
        if name == provider:
            raise interrupt()
        return import_module(name)

    monkeypatch.setattr(errors, "import_module", load)
    with pytest.raises(interrupt):
        errors._provider_connection_types()
