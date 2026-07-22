from __future__ import annotations

import json
from pathlib import Path

import pytest

from vv_agent.config import (
    ConfigError,
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
    "VERSION": "2",
    "backends": {
        "moonshot": {
            "default_endpoint": "moonshot-default",
            "models": {
                "kimi-k3": {
                    "id": "kimi-k3",
                    "endpoints": [
                        {"endpoint_id": "moonshot-default", "model_id": "kimi-k3"},
                        {"endpoint_id": "moonshot-backup", "model_id": "kimi-k3"},
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
            "api_key": "sk-test-123456789",
            "api_base": "https://api.moonshot.cn/v1",
        },
        {
            "id": "moonshot-backup",
            "api_key": "sk-backup-123456789",
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
        "VERSION": "2",
        "backends": {"demo": {"models": {"m": {"id": "m", "endpoints": ["e"]}}}},
        "endpoints": [{"id": "e", "api_key": "sk-test", "api_base": "https://example.invalid/v1"}],
    }
    json_file = tmp_path / "settings.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")
    toml_file = tmp_path / "settings.toml"
    toml_file.write_text(
        """
VERSION = "2"

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


def test_load_llm_settings_rejects_unknown_extension(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported settings file extension"):
        load_llm_settings_from_file(settings_file)


def test_resolve_model_endpoint(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k3")
    assert resolved.backend == "moonshot"
    assert resolved.model_id == "kimi-k3"
    assert resolved.endpoint.endpoint_id == "moonshot-default"
    assert resolved.endpoint.api_key == "sk-test-123456789"
    assert resolved.context_length == 256_000
    assert resolved.max_output_tokens == 32_768
    assert resolved.function_call_available is True
    assert resolved.response_format_available is False
    assert resolved.native_multimodal is True


def test_resolve_model_endpoint_collects_all_endpoint_options(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k3")
    assert len(resolved.endpoint_options) == 2
    assert {item.endpoint.endpoint_id for item in resolved.endpoint_options} == {"moonshot-default", "moonshot-backup"}


def test_resolve_model_endpoint_rejects_missing_model_key(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    with pytest.raises(ConfigError, match="missing-model"):
        resolve_model_endpoint(settings, backend="moonshot", model="missing-model")


def test_resolve_model_endpoint_uses_exact_model_key(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k3")
    assert resolved.selected_model == "kimi-k3"


def test_missing_llm_settings_raises(tmp_path: Path) -> None:
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text("A = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_llm_settings_from_file(settings_file)


def test_parse_project_example_settings_file() -> None:
    file_path = Path(__file__).resolve().parents[1] / "local_settings.example.py"
    settings = load_llm_settings_from_file(file_path)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k3")
    assert resolved.endpoint.api_base.startswith("https://")
    assert resolved.endpoint.api_key.startswith("REPLACE_WITH")
    assert resolved.context_length == 1_048_576
    assert resolved.max_output_tokens == 131_072


def test_model_settings_contract_fixture_is_enforced() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "parity" / "model_settings.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    canonical = fixture["canonical_settings"]
    resolved = resolve_model_endpoint(canonical, backend="openai", model="contract-model")
    assert resolved.model_id == "contract-model"
    assert resolved.context_length == 1_000_000
    assert resolved.max_output_tokens == 100_000

    for case in fixture["invalid_settings"]:
        with pytest.raises(ConfigError):
            resolve_model_endpoint(case["settings"], backend="openai", model="contract-model")
