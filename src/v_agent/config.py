from __future__ import annotations

import ast
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALIAS_MAP = {
    "kimi-k2.5": "kimi-k2-thinking",
}


@dataclass(slots=True)
class EndpointConfig:
    endpoint_id: str
    api_key: str
    api_base: str
    endpoint_type: str = "default"


@dataclass(slots=True)
class EndpointOption:
    endpoint: EndpointConfig
    model_id: str


@dataclass(slots=True)
class ResolvedModelConfig:
    backend: str
    requested_model: str
    selected_model: str
    model_id: str
    endpoint_options: list[EndpointOption]

    @property
    def endpoint(self) -> EndpointConfig:
        return self.endpoint_options[0].endpoint


class ConfigError(RuntimeError):
    pass


def load_llm_settings_from_file(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        raise ConfigError(f"Settings file not found: {source_path}")

    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    llm_value: ast.expr | None = None

    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "LLM_SETTINGS":
            llm_value = node.value
            break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "LLM_SETTINGS":
                    llm_value = node.value
                    break
            if llm_value is not None:
                break

    if llm_value is None:
        raise ConfigError(f"Cannot find LLM_SETTINGS in {source_path}")

    try:
        settings = ast.literal_eval(llm_value)
    except (ValueError, SyntaxError) as exc:
        raise ConfigError(f"LLM_SETTINGS is not a literal mapping in {source_path}") from exc

    if not isinstance(settings, dict):
        raise ConfigError("LLM_SETTINGS must evaluate to dict")

    # Compatibility: some settings files wrap the actual schema under
    # `{"LLM_SETTINGS": {...}}`.
    nested = settings.get("LLM_SETTINGS")
    if isinstance(nested, dict):
        nested_providers = _get_providers(nested)
        if isinstance(nested_providers, dict) and isinstance(nested.get("endpoints"), list):
            return nested

    return settings


def resolve_model_endpoint(settings: dict[str, Any], backend: str, model: str) -> ResolvedModelConfig:
    providers = _get_providers(settings)
    endpoints = settings.get("endpoints")
    if not isinstance(providers, dict) or not isinstance(endpoints, list):
        raise ConfigError("Invalid LLM_SETTINGS format: missing providers/backends or endpoints")

    provider_config = providers.get(backend)
    if not isinstance(provider_config, dict):
        raise ConfigError(f"Backend {backend!r} not found")

    models = provider_config.get("models")
    if not isinstance(models, dict):
        raise ConfigError(f"Backend {backend!r} has no models")

    selected_model = model if model in models else _ALIAS_MAP.get(model, model)
    model_config = models.get(selected_model)
    if not isinstance(model_config, dict):
        available = ", ".join(sorted(models.keys())[:10])
        raise ConfigError(f"Model {model!r} not found under backend {backend!r}. Available sample: {available}")

    base_model_id = str(model_config.get("id", selected_model))
    endpoint_candidates = model_config.get("endpoints")
    if not endpoint_candidates:
        default_endpoint = provider_config.get("default_endpoint")
        if default_endpoint:
            endpoint_candidates = [default_endpoint]

    if not isinstance(endpoint_candidates, list) or not endpoint_candidates:
        raise ConfigError(f"Model {selected_model!r} has no endpoint candidates")

    endpoint_map = {
        str(item.get("id")): item
        for item in endpoints
        if isinstance(item, dict) and item.get("id")
    }
    if not endpoint_map:
        raise ConfigError("No valid endpoints found in LLM_SETTINGS")

    options: list[EndpointOption] = []
    seen_pairs: set[tuple[str, str]] = set()

    for candidate in endpoint_candidates:
        if isinstance(candidate, str):
            endpoint_id = candidate
            model_id = base_model_id
        elif isinstance(candidate, dict):
            endpoint_id = str(candidate.get("endpoint_id", "")).strip()
            model_id = str(candidate.get("model_id", base_model_id)).strip()
        else:
            raise ConfigError(f"Unsupported endpoint mapping format for model {selected_model!r}")

        if not endpoint_id:
            raise ConfigError(f"Model {selected_model!r} has an endpoint candidate without endpoint_id")

        endpoint_data = endpoint_map.get(endpoint_id)
        if endpoint_data is None:
            raise ConfigError(f"Endpoint {endpoint_id!r} for model {selected_model!r} not found")

        pair = (endpoint_id, model_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        api_key = str(endpoint_data.get("api_key", "")).strip()
        api_base = str(endpoint_data.get("api_base", "")).strip()
        endpoint_type = str(endpoint_data.get("endpoint_type", "default"))
        if not api_key or not api_base:
            raise ConfigError(f"Endpoint {endpoint_id!r} is missing api_key or api_base")

        endpoint = EndpointConfig(
            endpoint_id=endpoint_id,
            api_key=decode_api_key(api_key),
            api_base=api_base,
            endpoint_type=endpoint_type,
        )
        options.append(EndpointOption(endpoint=endpoint, model_id=model_id))

    if not options:
        raise ConfigError(f"Model {selected_model!r} has no usable endpoint options")

    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=selected_model,
        model_id=options[0].model_id,
        endpoint_options=options,
    )


def build_openai_llm_from_local_settings(
    settings_path: str | Path,
    *,
    backend: str,
    model: str,
    timeout_seconds: float = 90.0,
):
    from v_agent.llm.openai_compatible import EndpointTarget, OpenAICompatibleLLM

    settings = load_llm_settings_from_file(settings_path)
    resolved = resolve_model_endpoint(settings, backend=backend, model=model)
    llm = OpenAICompatibleLLM(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id=option.endpoint.endpoint_id,
                api_key=option.endpoint.api_key,
                api_base=option.endpoint.api_base,
                endpoint_type=option.endpoint.endpoint_type,
                model_id=option.model_id,
            )
            for option in resolved.endpoint_options
        ],
        timeout_seconds=timeout_seconds,
    )
    return llm, resolved


def decode_api_key(raw_value: str) -> str:
    raw = raw_value.strip()
    if not raw:
        return raw

    direct = _extract_suffix_key(raw)
    if direct:
        return direct

    if os.getenv("V_AGENT_ENABLE_BASE64_KEY_DECODE") == "1":
        decoded = _maybe_base64_decode(raw)
        if decoded:
            from_decoded = _extract_suffix_key(decoded)
            if from_decoded:
                return from_decoded
            if _looks_like_api_key(decoded):
                return decoded

    return raw


def _get_providers(settings: dict[str, Any]) -> dict[str, Any] | None:
    providers = settings.get("providers")
    if isinstance(providers, dict):
        return providers
    backends = settings.get("backends")
    if isinstance(backends, dict):
        return backends
    return None


def _extract_suffix_key(value: str) -> str | None:
    if ":" not in value:
        return None
    _, suffix = value.split(":", 1)
    suffix = suffix.strip()
    if _looks_like_api_key(suffix):
        return suffix
    return None


def _maybe_base64_decode(value: str) -> str | None:
    padded = value + "=" * ((4 - len(value) % 4) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True)
    except Exception:
        return None
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _looks_like_api_key(value: str) -> bool:
    if not value:
        return False
    if any(ch.isspace() for ch in value):
        return False
    return len(value) >= 10
