"""Installed vendor network types veto both rejection and compaction in the kernel."""

from importlib import import_module

import httpx2
import pytest

from vv_agent.llm.errors import ModelRequestRejected, is_prompt_too_long_error, model_request_rejection
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore

from .test_model_rejection import fail, http_error
from .test_recovery_matrix import records_of, runtime, start


def native_error(provider, name):
    return getattr(import_module(provider), name)(request=httpx2.Request("POST", "https://provider.invalid"))


def assert_unknown(error):
    assert model_request_rejection(error) is None
    assert not is_prompt_too_long_error(error)
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        calls = []
        step = fail(error, calls)
        drive(store, "s", runtime=runtime(store, [step, step]))
        assert len(calls) == 2
        unknowns = records_of(store, "op_unknown")
        assert len(unknowns) == 2 and all(row.payload["duplicate_cost_risk"] for row in unknowns)
        assert not records_of(store, "op_completed")
        assert not records_of(store, "context_compacted")
        terminal = records_of(store, "turn_ended")[-1].payload
        assert terminal["reason"] == "model_outcome_unknown"
        assert terminal["unconfirmed_operations"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("name", ["APIConnectionError", "APITimeoutError"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("edge", ["cause", "context", "args", "group"])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_network_graph_keeps_unknown(provider, name, explicit, edge, reverse):
    native = native_error(provider, name)
    assert native.__cause__ is None
    rejected = ModelRequestRejected(400, "Rejected") if explicit else http_error(400)
    outer, inner = (native, rejected) if reverse else (rejected, native)
    if edge == "group":
        outer = ExceptionGroup("mixed", [outer, inner])
    elif edge == "args":
        outer.args = (*outer.args, inner)
    else:
        setattr(outer, "__" + edge + "__", inner)
    assert_unknown(outer)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("name", ["APIConnectionError", "APITimeoutError"])
def test_context_text_cannot_settle_native_network_failure(provider, name):
    error = RuntimeError("maximum context length exceeded")
    error.__cause__ = native_error(provider, name)
    assert_unknown(error)
