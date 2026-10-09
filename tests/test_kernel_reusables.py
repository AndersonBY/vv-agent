"""Reusable values have one owner while contracted package exports stay stable."""

import importlib

import pytest


@pytest.mark.parametrize(
    ("old_path", "new_path", "names", "public_path"),
    [
        (
            "vv_agent.checkpoint",
            "vv_agent.canonical_json",
            ("canonical_json_bytes", "canonical_json_sha256", "validate_sha256", "utf16_sort_key", "MAX_WIRE_INTEGER"),
            None,
        ),
        ("vv_agent.checkpoint", "vv_agent.tools.metadata", ("ToolIdempotency",), "vv_agent"),
        ("vv_agent.runtime.controller", "vv_agent.interaction", ("HostInteractionRequest",), "vv_agent.runtime"),
        (
            "vv_agent.deferred",
            "vv_agent.tools.outcomes",
            (
                "ToolCallOutcome",
                "DeferredToolHandle",
                "DeferredHandleError",
                "DeferredResolutionError",
                "DeferredResolutionResultInvalid",
            ),
            None,
        ),
        (
            "vv_agent.deferred",
            "vv_agent.tools.outcomes",
            ("DeferredWireError", "validate_definitive_result", "DEFERRED_HANDLE_SCHEMA", "TOOL_CALL_OUTCOME_SCHEMA"),
            None,
        ),
        ("vv_agent.runtime.cycle_runner", "vv_agent.llm.errors", ("MAX_PTL_RETRIES",), None),
    ],
)
def test_reusable_values_have_no_old_module_exports(
    old_path: str, new_path: str, names: tuple[str, ...], public_path: str | None
) -> None:
    old_module = importlib.import_module(old_path)
    new_module = importlib.import_module(new_path)
    for name in names:
        value = getattr(new_module, name)
        assert not hasattr(old_module, name)
        if public_path is not None:
            assert getattr(importlib.import_module(public_path), name) is value
