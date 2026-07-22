from __future__ import annotations

import ast
import json
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vv_llm.settings import Settings


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
    context_length: int | None = None
    max_output_tokens: int | None = None
    function_call_available: bool = False
    response_format_available: bool = False
    native_multimodal: bool = False

    @property
    def endpoint(self) -> EndpointConfig:
        return self.endpoint_options[0].endpoint


def project_resolved_model_limits(
    metadata: dict[str, Any],
    *,
    context_length: int | None,
    max_output_tokens: int | None,
) -> None:
    current_context = metadata.get("model_context_window")
    has_positive_context = not isinstance(current_context, bool) and isinstance(current_context, int) and current_context > 0
    if (
        not has_positive_context
        and not isinstance(context_length, bool)
        and isinstance(context_length, int)
        and context_length > 0
    ):
        metadata["model_context_window"] = context_length
    if max_output_tokens is not None:
        metadata.setdefault("model_max_output_tokens", max_output_tokens)


class ConfigError(RuntimeError):
    pass


def load_llm_settings_from_file(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        raise ConfigError(f"Settings file not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        try:
            return _require_settings_mapping(json.loads(source), source_path)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON settings file: {source_path}") from exc
    if suffix == ".toml":
        try:
            return _require_settings_mapping(tomllib.loads(source), source_path)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid TOML settings file: {source_path}") from exc

    if suffix != ".py":
        raise ConfigError(f"Unsupported settings file extension: {source_path.suffix or '<none>'}")

    module = ast.parse(source, filename=str(source_path))
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

    return _require_settings_mapping(settings, source_path)


def _require_settings_mapping(settings: Any, source_path: Path | str) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ConfigError(f"LLM_SETTINGS must be an object in {source_path}")
    if settings.get("VERSION") != "2":
        raise ConfigError(f"LLM_SETTINGS.VERSION must be '2' in {source_path}")
    if not isinstance(settings.get("backends"), dict):
        raise ConfigError(f"LLM_SETTINGS.backends must be an object in {source_path}")
    if not isinstance(settings.get("endpoints"), list):
        raise ConfigError(f"LLM_SETTINGS.endpoints must be an array in {source_path}")
    return settings


def resolve_model_endpoint(settings: dict[str, Any], backend: str, model: str) -> ResolvedModelConfig:
    settings = _require_settings_mapping(settings, "runtime settings")
    providers = settings["backends"]
    endpoints = settings.get("endpoints")
    assert isinstance(providers, dict)
    assert isinstance(endpoints, list)

    provider_config = providers.get(backend)
    if not isinstance(provider_config, dict):
        raise ConfigError(f"Backend {backend!r} not found")

    models = provider_config.get("models")
    if not isinstance(models, dict):
        raise ConfigError(f"Backend {backend!r} has no models")

    selected_model = model
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

    endpoint_map = {str(item.get("id")): item for item in endpoints if isinstance(item, dict) and item.get("id")}
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
            api_key=api_key,
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
        context_length=_read_positive_int(model_config.get("context_length")),
        max_output_tokens=_read_positive_int(model_config.get("max_output_tokens")),
        function_call_available=model_config.get("function_call_available") is True,
        response_format_available=model_config.get("response_format_available") is True,
        native_multimodal=model_config.get("native_multimodal") is True,
    )


def build_vv_llm_from_local_settings(
    settings_path: str | Path,
    *,
    backend: str,
    model: str,
    timeout_seconds: float = 90.0,
):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

    settings = load_llm_settings_from_file(settings_path)
    resolved = resolve_model_endpoint(settings, backend=backend, model=model)
    vv_settings = _build_vv_llm_settings(settings)
    llm = VvLlmClient(
        backend=backend,
        selected_model=resolved.selected_model,
        settings=vv_settings,
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


def _build_vv_llm_settings(settings: dict[str, Any]) -> Settings:
    from vv_llm.settings import Settings

    normalized = deepcopy(_require_settings_mapping(settings, "runtime settings"))

    try:
        return Settings(**normalized)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ConfigError(f"Failed to build vv-llm Settings: {exc}") from exc


def _read_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None
