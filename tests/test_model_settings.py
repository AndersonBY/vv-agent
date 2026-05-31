from __future__ import annotations

from vv_agent import ModelSettings, RetrySettings


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
