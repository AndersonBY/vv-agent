from __future__ import annotations

from typing import Any, cast

import pytest

from vv_agent import ModelSettings, ResponseFormat, RetrySettings, ToolChoice


def test_model_settings_resolve_prefers_non_none_override_values() -> None:
    base = ModelSettings(
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        tool_choice="auto",
        parallel_tool_calls=False,
        reasoning={"effort": "low"},
        response_format={"type": "text"},
        timeout_seconds=30,
        retry=RetrySettings(max_attempts=2, backoff_seconds=0.5),
        extra_headers={"X-Base": "1"},
        extra_body={"base": True},
        extra_args={"base_arg": 1},
    )

    resolved = base.resolve(
        ModelSettings(
            temperature=0.4,
            top_p=None,
            max_tokens=2048,
            tool_choice=None,
            parallel_tool_calls=True,
            reasoning=None,
            response_format={"type": "json_object"},
            timeout_seconds=None,
            retry=None,
            extra_headers={"X-Override": "2"},
            extra_body=None,
            extra_args={"override_arg": 2},
        )
    )

    assert resolved == ModelSettings(
        temperature=0.4,
        top_p=0.9,
        max_tokens=2048,
        tool_choice="auto",
        parallel_tool_calls=True,
        reasoning={"effort": "low"},
        response_format={"type": "json_object"},
        timeout_seconds=30,
        retry=RetrySettings(max_attempts=2, backoff_seconds=0.5),
        extra_headers={"X-Base": "1", "X-Override": "2"},
        extra_body={"base": True},
        extra_args={"base_arg": 1, "override_arg": 2},
    )


def test_model_settings_resolve_accepts_none_override() -> None:
    settings = ModelSettings(temperature=0.1, max_tokens=100)

    assert settings.resolve(None) == settings


def test_model_settings_compact_wire_round_trip() -> None:
    settings = ModelSettings(
        temperature=0.25,
        top_p=0.8,
        max_tokens=512,
        tool_choice="auto",
        parallel_tool_calls=False,
        reasoning={"effort": "high"},
        response_format={"type": "json_object"},
        timeout_seconds=12.5,
        retry=RetrySettings(max_attempts=4, backoff_seconds=0.25),
        extra_body={"provider_option": True},
        extra_args={"request_option": "value"},
    )

    payload = settings.to_dict()

    assert payload == {
        "temperature": 0.25,
        "top_p": 0.8,
        "max_tokens": 512,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": {"effort": "high"},
        "response_format": {"type": "json_object"},
        "timeout_seconds": 12.5,
        "retry": {"max_attempts": 4, "backoff_seconds": 0.25},
        "extra_body": {"provider_option": True},
        "extra_args": {"request_option": "value"},
    }
    assert ModelSettings.from_dict(payload) == settings


def test_model_settings_rejects_unknown_fields_and_invalid_ranges() -> None:
    for payload in (
        {"unknown": True},
        {"max_output_tokens": 256},
        {"retry": {"unknown": True}},
        {"max_tokens": 0},
        {"timeout_seconds": 0},
        {"top_p": 1.1},
    ):
        with pytest.raises((TypeError, ValueError)):
            ModelSettings.from_dict(payload)


def test_empty_retry_and_empty_objects_use_canonical_defaults() -> None:
    settings = ModelSettings.from_dict({"retry": {}, "reasoning": {}})

    assert settings.retry == RetrySettings(max_attempts=3, backoff_seconds=2.0)
    assert settings.reasoning is None
    assert settings.response_format is None
    assert settings.to_dict() == {"retry": {"max_attempts": 3, "backoff_seconds": 2.0}}


def test_response_format_uses_closed_standard_wire() -> None:
    json_schema = {"name": "answer", "schema": {"type": "object"}, "strict": True}

    assert ResponseFormat.text().to_wire() == {"type": "text"}
    assert ResponseFormat.json_object().to_wire() == {"type": "json_object"}
    assert ResponseFormat.json_schema_format(json_schema).to_wire() == {
        "type": "json_schema",
        "json_schema": json_schema,
    }
    assert ModelSettings.from_dict(
        {"response_format": {"type": "json_schema", "json_schema": json_schema}}
    ).response_format == ResponseFormat.json_schema_format(json_schema)

    for invalid in (
        {},
        {"type": "json_schema", "schema": json_schema},
        {"type": "json_schema", "json_schema": []},
        {"type": "json_object", "extra": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            ModelSettings.from_dict({"response_format": invalid})


def test_tool_choice_uses_modes_or_standard_named_tool_wire() -> None:
    named_wire = {"type": "function", "function": {"name": "lookup"}}

    assert ModelSettings(tool_choice="auto").to_dict()["tool_choice"] == "auto"
    assert ModelSettings(tool_choice=ToolChoice.tool("lookup")).to_dict()["tool_choice"] == named_wire
    assert ModelSettings.from_dict({"tool_choice": named_wire}).tool_choice == ToolChoice.tool("lookup")
    for invalid in ("lookup", {"tool": "lookup"}, {"type": "function", "function": {"name": ""}}):
        with pytest.raises((TypeError, ValueError)):
            ModelSettings.from_dict({"tool_choice": invalid})

    with pytest.raises(ValueError, match="Unknown tool_choice mode"):
        ToolChoice(cast(Any, "invalid"))


def test_retry_settings_reject_boolean_backoff() -> None:
    with pytest.raises(ValueError, match="backoff_seconds"):
        RetrySettings(backoff_seconds=True)


def test_model_settings_rejects_non_boolean_and_non_map_values() -> None:
    for payload in (
        {"parallel_tool_calls": 1},
        {"extra_headers": None},
        {"extra_headers": {"x-demo": 1}},
        {"extra_body": []},
        {"extra_args": "query=value"},
    ):
        with pytest.raises((TypeError, ValueError)):
            ModelSettings.from_dict(payload)
