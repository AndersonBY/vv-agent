from __future__ import annotations

import json
from pathlib import Path

import pytest
from vv_llm.types import BackendType

from vv_agent.config import (
    ConfigError,
    build_vv_llm_settings,
    decode_api_key,
    load_llm_settings_from_file,
    project_resolved_model_limits,
    resolve_model_endpoint,
)


@pytest.fixture
def sample_settings_file(tmp_path: Path) -> Path:
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text(
        """
LLM_SETTINGS = {
    "backends": {
        "moonshot": {
            "default_endpoint": "moonshot-default",
            "models": {
                "kimi-k2.6": {
                    "id": "kimi-k2.6",
                    "endpoints": [
                        {"endpoint_id": "moonshot-default", "model_id": "kimi-k2.6"},
                        {"endpoint_id": "moonshot-backup", "model_id": "kimi-k2.6"},
                    ],
                    "context_length": 256000,
                    "max_output_tokens": 32768,
                    "function_call_available": True,
                    "response_format_available": False,
                    "native_multimodal": True,
                }
            },
        }
    },
    "endpoints": [
        {
            "id": "moonshot-default",
            "api_key": "prefix:sk-test-123456789",
            "api_base": "https://api.moonshot.cn/v1",
        },
        {
            "id": "moonshot-backup",
            "api_key": "prefix:sk-backup-123456789",
            "api_base": "https://api.moonshot.cn/v1",
        }
    ],
}
""",
        encoding="utf-8",
    )
    return settings_file


def test_load_llm_settings_from_file(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    assert "backends" in settings
    assert "moonshot" in settings["backends"]


def test_project_resolved_model_limits_replaces_non_positive_context_only() -> None:
    metadata = {
        "model_context_window": 0,
        "model_max_output_tokens": 2_048,
    }

    project_resolved_model_limits(
        metadata,
        context_length=64_000,
        max_output_tokens=8_192,
    )

    assert metadata == {
        "model_context_window": 64_000,
        "model_max_output_tokens": 2_048,
    }


def test_load_llm_settings_supports_json_and_toml(tmp_path: Path) -> None:
    payload = {
        "backends": {"demo": {"models": {"m": {"id": "m", "endpoints": ["e"]}}}},
        "endpoints": [{"id": "e", "api_key": "sk-test", "api_base": "https://example.invalid/v1"}],
    }
    json_file = tmp_path / "settings.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")
    toml_file = tmp_path / "settings.toml"
    toml_file.write_text(
        """
[[endpoints]]
id = "e"
api_key = "sk-test"
api_base = "https://example.invalid/v1"

[backends.demo.models.m]
id = "m"
endpoints = ["e"]
""",
        encoding="utf-8",
    )

    assert resolve_model_endpoint(load_llm_settings_from_file(json_file), "demo", "m").model_id == "m"
    assert resolve_model_endpoint(load_llm_settings_from_file(toml_file), "demo", "m").model_id == "m"


def test_resolve_model_endpoint(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")
    assert resolved.backend == "moonshot"
    assert resolved.model_id == "kimi-k2.6"
    assert resolved.endpoint.endpoint_id == "moonshot-default"
    assert resolved.endpoint.api_key == "sk-test-123456789"
    assert resolved.context_length == 256_000
    assert resolved.max_output_tokens == 32_768
    assert resolved.function_call_available is True
    assert resolved.response_format_available is False
    assert resolved.native_multimodal is True


def test_resolve_model_endpoint_collects_all_endpoint_options(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")
    assert len(resolved.endpoint_options) == 2
    assert {item.endpoint.endpoint_id for item in resolved.endpoint_options} == {"moonshot-default", "moonshot-backup"}


def test_resolve_model_endpoint_does_not_alias_missing_kimi_k25(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    with pytest.raises(ConfigError, match="kimi-k2\\.5"):
        resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.5")


def test_resolve_model_endpoint_keeps_kimi_k25_and_k26_distinct() -> None:
    settings = {
        "backends": {
            "moonshot": {
                "models": {
                    "kimi-k2.5": {
                        "id": "kimi-k2.5",
                        "endpoints": [{"endpoint_id": "moonshot-default", "model_id": "kimi-k2.5"}],
                    },
                    "kimi-k2.6": {
                        "id": "kimi-k2.6",
                        "endpoints": [{"endpoint_id": "moonshot-default", "model_id": "kimi-k2.6"}],
                    },
                }
            }
        },
        "endpoints": [
            {
                "id": "moonshot-default",
                "api_key": "sk-test-123456789",
                "api_base": "https://api.moonshot.cn/v1",
            }
        ],
    }

    k25 = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.5")
    k26 = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")
    assert k25.selected_model == "kimi-k2.5"
    assert k25.model_id == "kimi-k2.5"
    assert k25.context_length is None
    assert k25.max_output_tokens is None
    assert k25.function_call_available is False
    assert k25.response_format_available is False
    assert k25.native_multimodal is False
    assert k26.selected_model == "kimi-k2.6"
    assert k26.model_id == "kimi-k2.6"


def test_resolve_model_endpoint_uses_exact_model_key(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")
    assert resolved.selected_model == "kimi-k2.6"


def test_decode_api_key_with_compound_format() -> None:
    key = decode_api_key("prefix:sk-live-example")
    assert key == "sk-live-example"


def test_missing_llm_settings_raises(tmp_path: Path) -> None:
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text("A = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_llm_settings_from_file(settings_file)


def test_parse_project_example_settings_file() -> None:
    file_path = Path(__file__).resolve().parents[1] / "local_settings.example.py"
    settings = load_llm_settings_from_file(file_path)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")
    assert resolved.endpoint.api_base.startswith("https://")
    assert resolved.endpoint.api_key.startswith("REPLACE_WITH")


def test_load_llm_settings_supports_nested_llm_settings(tmp_path: Path) -> None:
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text(
        """
LLM_SETTINGS = {
    "LLM_SETTINGS": {
        "backends": {
            "moonshot": {
                "default_endpoint": "moonshot-default",
                "models": {"kimi-k2.6": {"id": "kimi-k2.6", "endpoints": ["moonshot-default"]}},
            }
        },
        "endpoints": [{"id": "moonshot-default", "api_key": "sk-test-123456789", "api_base": "https://api.moonshot.cn/v1"}],
    }
}
""",
        encoding="utf-8",
    )
    settings = load_llm_settings_from_file(settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")
    assert resolved.endpoint.endpoint_id == "moonshot-default"


def test_build_vv_llm_settings_normalizes_provider_aliases_and_keys(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.6")

    vv_settings = build_vv_llm_settings(settings=settings, backend="moonshot", resolved=resolved)
    backend_settings = vv_settings.get_backend(BackendType.Moonshot)
    model_setting = backend_settings.get_model_setting("kimi-k2.6")
    first_endpoint = model_setting.endpoints[0]
    assert isinstance(first_endpoint, dict)

    endpoint = vv_settings.get_endpoint("moonshot-default")
    assert endpoint.api_key == "sk-test-123456789"
    assert first_endpoint["endpoint_id"] == "moonshot-default"
