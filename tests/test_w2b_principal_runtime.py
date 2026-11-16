from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vv_agent import Agent, RunConfig, ToolContext, function_tool
from vv_agent.kernel._content import ContentLabels, ContentRef, Principal
from vv_agent.llm.base import LlmRequest
from vv_agent.model import ScriptedModelProvider
from vv_agent.session._dispatch_context import _session_principal
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall
from vv_agent.workspace import MemoryWorkspaceBackend


def _principal() -> Principal:
    return Principal(
        tenant_id="tenant-w2b-runtime",
        subject_id="subject-w2b-runtime",
        subject_kind="user",
        attributes=ContentRef(
            content_id="principal-attributes-w2b-runtime",
            mac="0" * 64,
            size=1,
            key_id="key-w2b-runtime",
            labels=ContentLabels(),
        ),
    )


def test_trusted_session_principal_reaches_tools_and_model_outside_prompt(tmp_path) -> None:
    attribute_secret = "principal-attribute-w2b-runtime"
    ingress_principal = {
        "tenant_id": "tenant-w2b-runtime",
        "subject_id": "subject-w2b-runtime",
        "subject_kind": "user",
        "attributes": {"role": attribute_secret},
    }
    model_requests: list[LlmRequest] = []
    tool_contexts: list[ToolContext] = []

    @function_tool
    def inspect_principal(context: ToolContext) -> str:
        """Capture the runtime principal without exposing it to the model."""

        tool_contexts.append(context)
        return "captured"

    def complete(request: LlmRequest) -> LLMResponse:
        model_requests.append(request)
        if len(model_requests) == 1:
            return LLMResponse("", [ToolCall("principal-call", "inspect_principal", {})])
        return LLMResponse("done")

    sessions = SessionDriver(tenant_id=ingress_principal["tenant_id"])
    try:
        sessions.create("principal-session", str(tmp_path), principal=ingress_principal)
        handle = sessions.start(
            "principal-session",
            Agent("principal-agent", "Work.", model="w2b", tools=[inspect_principal]),
            RunConfig(
                model_provider=ScriptedModelProvider.from_callback("scripted", "w2b", complete),
                workspace=tmp_path,
            ),
            "go",
        )
        result = handle.result(timeout=10)
        stored_records = tuple(row.record.payload for row in sessions.store.read_state("principal-session")[1])
        stored_model_requests = tuple(
            record["request"] for record in stored_records if record.get("op_kind") == "model" and "request" in record
        )
    finally:
        sessions.close()

    assert result.final_output == "done"
    assert len(model_requests) == 2
    assert len(tool_contexts) == 1
    context = tool_contexts[0]
    assert context.principal is not None
    assert (context.principal.tenant_id, context.principal.subject_id, context.principal.subject_kind) == (
        ingress_principal["tenant_id"],
        ingress_principal["subject_id"],
        ingress_principal["subject_kind"],
    )
    assert all(request.principal == context.principal for request in model_requests)

    hidden_values = (
        context.principal.tenant_id,
        context.principal.subject_id,
        context.principal.attributes.content_id,
        attribute_secret,
    )
    for request in model_requests:
        visible_request = repr((request.messages, request.tools, request.prompt_bundle))
        assert all(value not in visible_request for value in hidden_values)
    for tool_context in tool_contexts:
        assert all(value not in repr(tool_context.arguments) for value in hidden_values)
    assert stored_model_requests
    assert all(value not in repr(stored_model_requests) for value in hidden_values)

    with pytest.raises(AttributeError, match="read-only"):
        context.principal = None
    assert context.principal is not None
    with pytest.raises(FrozenInstanceError):
        context.principal.subject_id = "replacement"
    with pytest.raises(FrozenInstanceError):
        context.principal.attributes.labels.source = ()
    with pytest.raises(AttributeError, match="read-only"):
        model_requests[0].principal = None


def test_principal_defaults_to_unavailable_and_rejects_mutable_values(tmp_path) -> None:
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=MemoryWorkspaceBackend(),
    )
    request = LlmRequest(model="w2b", messages=[])

    assert context.principal is None
    assert request.principal is None
    empty_store = SimpleNamespace(read=lambda *_args, **_kwargs: SimpleNamespace(records=()))
    assert _session_principal(empty_store, "legacy-session") is None
    with pytest.raises(TypeError, match="Principal or None"):
        ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=0,
            workspace_backend=MemoryWorkspaceBackend(),
            principal=_principal().to_dict(),
        )
    with pytest.raises(TypeError, match="Principal or None"):
        LlmRequest(model="w2b", messages=[], principal=_principal().to_dict())
