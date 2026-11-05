"""Mixed evidence and conservative diagnostic boundaries."""

import httpx2
import pytest
from session.test_model_rejection import http_error

from vv_agent.llm.errors import ModelRequestRejected, ModelStreamInterrupted, is_prompt_too_long_error, model_request_rejection


def ambiguous(kind):
    return {
        "timeout": TimeoutError("lost"),
        "transport": httpx2.ReadError("lost"),
        "408": http_error(408),
        "500": http_error(500),
        "stream": ModelStreamInterrupted("partial"),
    }[kind]


@pytest.mark.parametrize("kind", ["timeout", "transport", "408", "500", "stream"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("edge", ["cause", "context", "arg", "group"])
@pytest.mark.parametrize("reverse", [False, True])
def test_whole_exception_graph_vetoes_rejection(kind, explicit, edge, reverse):
    rejected = ModelRequestRejected(400, "Rejected") if explicit else http_error(400)
    uncertain = ambiguous(kind)
    outer, inner = (uncertain, rejected) if reverse else (rejected, uncertain)
    if edge == "group":
        outer = ExceptionGroup("mixed", [outer, inner])
    elif edge == "arg":
        outer.args = (*outer.args, inner)
    else:
        setattr(outer, "__" + edge + "__", inner)
    assert model_request_rejection(outer) is None
    assert not is_prompt_too_long_error(outer)


def test_unobservable_http_response_is_not_rejection():
    assert model_request_rejection(http_error(400, observed=False)) is None


LEAKS = [
    "Invalid API key provided: SYNTHETIC_CREDENTIAL_123456789",
    "x" * 480 + "API key provided: SYNTHETIC_CREDENTIAL_123456789",
    "Invalid request: messages=[SYNTHETIC_PRIVATE_INPUT]",
    "prompt=SYNTHETIC_PRIVATE_INPUT",
    "input: SYNTHETIC_PRIVATE_INPUT",
    "Authorization: Bearer SYNTHETIC_CREDENTIAL_123456789",
    "access_token=SYNTHETIC_CREDENTIAL_123456789",
]


@pytest.mark.parametrize("message", LEAKS)
def test_safe_diagnostics_do_not_keep_even_partial_secret(message):
    rejected = ModelRequestRejected(400, message)
    assert "SYNTHETIC" not in rejected.message
    assert not rejected.message.endswith("SY")
    assert len(rejected.message) <= 500


@pytest.mark.parametrize(
    "code",
    ["api_key:SYNTHETIC_IDENTIFIER_SECRET", "access_token:private", "secret.private", "Authorization", "Bearer", "token=private"],
)
def test_provider_identifiers_reject_credential_markers(code):
    rejected = ModelRequestRejected(400, "Rejected", provider_code=code, provider_type=code)
    assert rejected.provider_code is None and rejected.provider_type is None


def test_observation_of_another_response_cannot_certify_unobserved_http():
    from vv_agent.llm.http_observation import HttpObservation

    error = http_error(400, observed=False)
    error._vv_http_observation = HttpObservation([400])
    assert model_request_rejection(error) is None


def test_unobservable_http_context_error_cannot_bypass_unknown_via_compaction():
    error = http_error(400, "maximum context length exceeded", observed=False)
    assert not is_prompt_too_long_error(error)
    assert model_request_rejection(error) is None
    explicit = ModelRequestRejected(400, "maximum context length exceeded")
    explicit.__cause__ = error
    assert is_prompt_too_long_error(explicit)


@pytest.mark.parametrize("missing", ["openai", "anthropic"])
def test_missing_optional_vendor_skips_only_its_network_types(monkeypatch, missing):
    from importlib import import_module

    from vv_agent.llm import errors

    def load(name):
        if name == missing:
            raise ImportError("synthetic unavailable vendor")
        return import_module(name)

    with monkeypatch.context() as patcher:
        patcher.setattr(errors, "import_module", load)
        available = import_module("anthropic" if missing == "openai" else "openai").APIConnectionError
        assert errors._provider_connection_types() == (available,)
        assert model_request_rejection(http_error(400)) is not None


def test_network_type_name_alone_does_not_veto_rejection():
    APIConnectionError = type("APIConnectionError", (Exception,), {})
    error = ModelRequestRejected(400, "Rejected")
    error.__cause__ = APIConnectionError("synthetic unrelated type")
    assert model_request_rejection(error) is error
