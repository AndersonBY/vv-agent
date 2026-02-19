from __future__ import annotations

from pathlib import Path

import pytest
from vv_llm.types import BackendType

from v_agent.config import (
    ConfigError,
    build_vv_llm_settings,
    decode_api_key,
    load_llm_settings_from_file,
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
                "kimi-k2-thinking": {
                    "id": "kimi-k2-thinking",
                    "endpoints": [
                        {"endpoint_id": "moonshot-default", "model_id": "kimi-k2-thinking"},
                        {"endpoint_id": "moonshot-backup", "model_id": "kimi-k2-thinking"},
                    ],
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


def test_resolve_model_endpoint(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2-thinking")
    assert resolved.backend == "moonshot"
    assert resolved.model_id == "kimi-k2-thinking"
    assert resolved.endpoint.endpoint_id == "moonshot-default"
    assert resolved.endpoint.api_key == "sk-test-123456789"


def test_resolve_model_endpoint_collects_all_endpoint_options(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2-thinking")
    assert len(resolved.endpoint_options) == 2
    assert {item.endpoint.endpoint_id for item in resolved.endpoint_options} == {"moonshot-default", "moonshot-backup"}


def test_resolve_model_endpoint_alias(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2.5")
    assert resolved.selected_model == "kimi-k2-thinking"


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
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2-thinking")
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
                "models": {"kimi-k2-thinking": {"id": "kimi-k2-thinking", "endpoints": ["moonshot-default"]}},
            }
        },
        "endpoints": [{"id": "moonshot-default", "api_key": "sk-test-123456789", "api_base": "https://api.moonshot.cn/v1"}],
    }
}
""",
        encoding="utf-8",
    )
    settings = load_llm_settings_from_file(settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2-thinking")
    assert resolved.endpoint.endpoint_id == "moonshot-default"


def test_build_vv_llm_settings_normalizes_provider_aliases_and_keys(sample_settings_file: Path) -> None:
    settings = load_llm_settings_from_file(sample_settings_file)
    resolved = resolve_model_endpoint(settings, backend="moonshot", model="kimi-k2-thinking")

    vv_settings = build_vv_llm_settings(settings=settings, backend="moonshot", resolved=resolved)
    backend_settings = vv_settings.get_backend(BackendType.Moonshot)
    model_setting = backend_settings.get_model_setting("kimi-k2-thinking")
    first_endpoint = model_setting.endpoints[0]
    assert isinstance(first_endpoint, dict)

    endpoint = vv_settings.get_endpoint("moonshot-default")
    assert endpoint.api_key == "sk-test-123456789"
    assert first_endpoint["endpoint_id"] == "moonshot-default"
