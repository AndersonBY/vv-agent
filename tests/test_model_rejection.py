"""Conservative HTTP evidence and adapter-owned rejection receipts."""

from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from session.test_model_rejection import context_http_error, observe_error

from vv_agent.llm.errors import ModelRequestRejected, ModelStreamInterrupted, is_prompt_too_long_error, model_request_rejection


@pytest.mark.parametrize("transport", ["httpx2", "httpx"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("kind", ["openai", "typed", "anthropic", "moonshot", "payload", "content_string"])
def test_real_http_context_errors_are_prompt_too_long(transport, wrapped, kind):
    pytest.importorskip(transport)
    assert is_prompt_too_long_error(context_http_error(kind, transport=transport, wrapped=wrapped))


def test_string_limit_on_unrelated_field_is_not_prompt_too_long():
    error = context_http_error("tool_string")
    assert not is_prompt_too_long_error(error)
    assert model_request_rejection(error) is not None


def test_payload_status_without_error_body_still_requires_smaller_input():
    request = httpx2.Request("POST", "https://provider.invalid")
    error = httpx2.HTTPStatusError("Payload exceeds limit", request=request, response=httpx2.Response(413))
    observe_error(error, 413)
    assert is_prompt_too_long_error(error)


def test_explicit_context_rejection_uses_provider_code():
    assert is_prompt_too_long_error(ModelRequestRejected(400, "Input exceeds limit", provider_code="context_length_exceeded"))


@pytest.mark.parametrize(
    "status,message,too_long",
    [(400, "prompt is too long", True), (413, "Payload exceeds limit", True), (400, "Empty content", False)],
)
def test_unread_http_error_body_does_not_hide_text_or_status(status, message, too_long):
    request = httpx2.Request("POST", "https://provider.invalid")
    response = httpx2.Response(status, request=request, stream=httpx2.ByteStream(b"unread error body"))
    error = observe_error(httpx2.HTTPStatusError(message, request=request, response=response), status)
    try:
        assert is_prompt_too_long_error(error) is too_long
        if not too_long:
            rejection = model_request_rejection(error)
            assert rejection is not None and rejection.message == "Model provider request failed."
        assert not response.is_stream_consumed
    finally:
        response.close()


@pytest.mark.parametrize("kind", ["timeout", "connection", "transport", "stream", "408", "500"])
def test_ambiguity_overrides_prompt_too_long_evidence(kind):
    inner = context_http_error("openai")
    errors = {
        "timeout": TimeoutError("maximum context length"),
        "connection": ConnectionError("prompt is too long"),
        "transport": httpx2.ReadError("too many tokens"),
        "stream": ModelStreamInterrupted("request too large"),
        "408": httpx2.HTTPStatusError(
            "prompt is too long", request=httpx2.Request("POST", "https://provider.invalid"), response=httpx2.Response(408)
        ),
        "500": httpx2.HTTPStatusError(
            "prompt is too long", request=httpx2.Request("POST", "https://provider.invalid"), response=httpx2.Response(500)
        ),
    }
    outer = errors[kind]
    outer.__cause__ = inner
    assert not is_prompt_too_long_error(outer)
    wrapper = RuntimeError("maximum context length")
    wrapper.__cause__ = outer
    assert not is_prompt_too_long_error(wrapper)


def test_explicit_adapter_rejection_is_validated_and_sanitized():
    error = ModelRequestRejected(409, "Busy Authorization: Bearer sk-secret", retryable=True, provider_code="conflict")
    assert model_request_rejection(error) is error
    assert error.retryable is True
    assert error.provider_code == "conflict"
    assert "sk-secret" not in str(error)
    for status in [True, "400", 200, 408, 500]:
        with pytest.raises(ValueError):
            ModelRequestRejected(cast(Any, status), "invalid")


@pytest.mark.parametrize("endpoint_count", [0, 1, 2, 3])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429, 499])
def test_default_retry_classification_uses_frozen_endpoint_count(status, endpoint_count):
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://provider.invalid"))
    error = observe_error(httpx2.HTTPStatusError("Rejected", request=response.request, response=response), status)
    rejection = model_request_rejection(error, endpoint_count=endpoint_count)
    assert rejection is not None
    expected = status == 429 or (status in {401, 403, 404} and endpoint_count > 1)
    assert rejection.retryable is expected
    explicit_default = model_request_rejection(ModelRequestRejected(status, "Rejected"), endpoint_count=endpoint_count)
    assert explicit_default is not None and explicit_default.retryable is expected
    for override in (False, True):
        explicit = ModelRequestRejected(status, "Rejected", retryable=override)
        assert model_request_rejection(explicit, endpoint_count=endpoint_count) is explicit
        assert explicit.retryable is override


def test_only_explicit_cause_chain_is_followed():
    rejected = ModelRequestRejected(400, "invalid")
    outer = RuntimeError("wrapper")
    outer.__cause__ = rejected
    assert model_request_rejection(outer) is rejected
    outer.__cause__ = outer
    assert model_request_rejection(outer) is None
    outer.__cause__ = None
    outer.__context__ = rejected
    assert model_request_rejection(outer) is None
    for error in [TimeoutError("timeout"), ConnectionError("lost"), ModelStreamInterrupted("partial")]:
        error.__cause__ = rejected
        assert model_request_rejection(error) is None


@pytest.mark.parametrize("status,response", [(400, None), ("400", None), (True, None), (400, SimpleNamespace(status_code=400))])
def test_bare_or_forged_status_is_not_http_evidence(status, response):
    error: Any = RuntimeError("invalid")
    error.status_code = status
    error.response = response
    assert model_request_rejection(error) is None


@pytest.mark.parametrize("transport", ["httpx2", "httpx"])
def test_actual_http_status_error(transport):
    module = pytest.importorskip(transport)
    response = module.Response(
        422,
        request=module.Request("POST", "https://user:password@example.invalid/?api_key=secret"),
        json={"error": {"code": "invalid", "type": "request_error", "message": "Empty user message"}},
    )
    error = module.HTTPStatusError("unsafe request dump", request=response.request, response=response)
    observe_error(error, error.response.status_code)
    rejected = model_request_rejection(error)
    assert rejected is not None
    assert rejected.message == "Model provider request failed."
    assert rejected.provider_code == "invalid"
    assert rejected.provider_type == "request_error"


def test_mismatched_status_stays_unknown():
    error: Any = RuntimeError("invalid")
    error.status_code = 400
    error.response = httpx2.Response(500)
    assert model_request_rejection(error) is None


def test_secrets_are_removed_before_message_truncation():
    error = ModelRequestRejected(
        401,
        "API key=supersecret\nAuthorization: Bearer topsecret\nsk-secret-token\n" + "z" * 700,
        provider_code="sk-secret-code",
        provider_type="Authorization: Bearer supersecret",
    )
    assert len(error.message) <= 500
    assert all(secret not in error.message for secret in ["supersecret", "topsecret", "sk-secret-token", "Bearer"])
    assert error.provider_code is None and error.provider_type is None


def test_exception_request_dump_is_never_the_diagnostic():
    from openai import APIStatusError

    response = httpx2.Response(400, request=httpx2.Request("POST", "https://provider.invalid"))
    error = APIStatusError("Authorization: Bearer private; complete request body", response=response, body=None)
    observe_error(error, error.response.status_code)
    rejected = model_request_rejection(error)
    assert rejected is not None and rejected.message == "Model provider request failed."


def test_multiline_request_body_is_omitted():
    error = ModelRequestRejected(400, 'Invalid user. Request body: {\n"messages": ["private content"]\n}')
    assert error.message == "Model provider request failed."


@pytest.mark.parametrize("body", ['{"model":"m", "messages":["private input"]}', '{"headers":{"X-Custom":"private header"}}'])
def test_echoed_request_json_is_omitted(body):
    error = ModelRequestRejected(400, f"Invalid user: {body}")
    assert "private" not in error.message
    assert error.message == "Model provider request failed."


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_status_exception_and_vv_llm_wrapper(provider):
    from importlib import import_module

    from vv_llm.types import classify_exception

    cls = import_module(provider).APIStatusError
    response = httpx2.Response(403, request=httpx2.Request("POST", "https://provider.invalid"))
    error = cls("unsafe dump", response=response, body={"error": {"code": 1234, "type": "denied", "message": "Denied"}})
    wrapper = classify_exception(error)
    # vv-llm middleware/registry raises the classified wrapper from its source.
    wrapper.__cause__ = error
    observe_error(wrapper, 403)
    rejected = model_request_rejection(wrapper)
    assert rejected is not None
    assert rejected.to_dict() == {
        "status_code": 403,
        "provider_code": "1234",
        "provider_type": "denied",
        "message": "Model provider request failed.",
        "retryable": False,
    }
