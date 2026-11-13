"""Real vv-llm -> vendor SDK -> MockTransport, including hidden retries."""

from contextlib import contextmanager
from unittest.mock import patch

import httpx2
import pytest

from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.kernel import drive

from .test_recovery_matrix import records_of, runtime, start


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("case", ["timeout_then_400", "500_then_400", "always_429", "single_400"])
def test_vendor_retries_are_observed(store, database, provider, case):
    with local_vendor_transport(case) as (requests, clients):
        rt = runtime(database, [])
        rt.llm = VvLlmClient([EndpointTarget("a", "synthetic-unused", "https://provider.invalid")], backend=provider)
        start(store)
        drive(store, "s", runtime=rt)
    unknown = case in {"timeout_then_400", "500_then_400"}
    terminal = records_of(store, "turn_ended")[-1].payload
    assert terminal["reason"] == ("model_outcome_unknown" if unknown else "model_request_rejected")
    assert len(requests) == (4 if unknown else 6 if case == "always_429" else 1)
    assert len(records_of(store, "op_planned")) == (2 if unknown or case == "always_429" else 1)
    assert len(records_of(store, "op_unknown")) == (2 if unknown else 0)
    assert bool(terminal["unconfirmed_operations"]) is unknown
    assert all(r.payload["duplicate_cost_risk"] for r in records_of(store, "op_unknown"))
    assert all(client.is_closed for client in clients)


@contextmanager
def local_vendor_transport(case, *, rejection=None):
    """Inject only local HTTP, leaving vv-llm and vendor construction/retries real."""
    requests, clients = [], []

    def handle(request):
        requests.append(request.method)
        position = (len(requests) - 1) % 2
        if case == "timeout_then_400" and position == 0:
            raise httpx2.ReadTimeout("synthetic timeout after submission", request=request)
        status = 500 if case == "500_then_400" and position == 0 else 429 if case == "always_429" else 400
        body = {"error": {"message": "Synthetic rejected", "type": "invalid_request_error"}}
        if rejection is not None:
            status, detail = rejection
            body = {"error": detail}
        return httpx2.Response(status, json=body)

    original = httpx2.Client.__init__

    def local_client(self, **kwargs):
        kwargs["transport"] = httpx2.MockTransport(handle)
        original(self, **kwargs)
        clients.append(self)

    with (
        patch.object(httpx2.Client, "__init__", local_client),
        patch("openai._base_client.SyncAPIClient._sleep_for_retry", lambda *a, **k: None),
        patch("anthropic._base_client.SyncAPIClient._sleep_for_retry", lambda *a, **k: None),
        patch("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0),
    ):
        yield requests, clients
