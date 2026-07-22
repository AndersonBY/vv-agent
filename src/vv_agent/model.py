from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from vv_agent.config import (
    ResolvedModelConfig,
    build_vv_llm_from_local_settings,
    load_llm_settings_from_file,
    resolve_model_endpoint,
)
from vv_agent.llm.base import LLMClient, LlmRequest
from vv_agent.llm.scripted import ScriptedLLM, ScriptStep
from vv_agent.model_settings import ModelSettings
from vv_agent.types import LLMResponse


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelRef:
    type: str
    model_name: str = ""
    backend_id: str | None = None
    resolved_config: ResolvedModelConfig | None = None

    def __post_init__(self) -> None:
        if self.type == "named":
            self._require_nonempty("model", self.model_name)
            if self.backend_id is not None or self.resolved_config is not None:
                raise ValueError("named ModelRef cannot include backend or resolved config")
            return
        if self.type == "backend_model":
            self._require_nonempty("backend", self.backend_id)
            self._require_nonempty("model", self.model_name)
            if self.resolved_config is not None:
                raise ValueError("backend_model ModelRef cannot include resolved config")
            return
        if self.type == "resolved" and self.resolved_config is not None:
            return
        raise ValueError(f"Unknown or incomplete ModelRef type: {self.type}")

    @classmethod
    def named(cls, model: str) -> ModelRef:
        return cls(type="named", model_name=model)

    @classmethod
    def backend(cls, backend: str, model: str) -> ModelRef:
        return cls(type="backend_model", backend_id=backend, model_name=model)

    @classmethod
    def resolved(cls, resolved: ResolvedModelConfig) -> ModelRef:
        return cls(
            type="resolved",
            model_name=resolved.selected_model,
            backend_id=resolved.backend,
            resolved_config=resolved,
        )

    @classmethod
    def coerce(cls, value: str | ModelRef | ResolvedModelConfig) -> ModelRef:
        if isinstance(value, cls):
            return value
        if isinstance(value, ResolvedModelConfig):
            return cls.resolved(value)
        if isinstance(value, str):
            return cls.named(value)
        raise TypeError(f"Cannot coerce {type(value).__name__} into ModelRef")

    def to_dict(self) -> dict[str, str]:
        if self.type == "named":
            return {"type": "named", "model": self.model_name}
        if self.type == "backend_model":
            return {"type": "backend_model", "backend": str(self.backend_id), "model": self.model_name}
        raise ValueError("resolved ModelRef is process-local and cannot be serialized")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelRef:
        if not isinstance(payload, dict):
            raise TypeError("ModelRef wire value must be an object")
        ref_type = payload.get("type")
        if not isinstance(ref_type, str):
            raise TypeError("ModelRef type must be a string")
        expected_keys = {
            "named": {"type", "model"},
            "backend_model": {"type", "backend", "model"},
        }.get(ref_type)
        if expected_keys is None or set(payload) != expected_keys:
            raise ValueError("Invalid ModelRef wire shape")
        if ref_type == "named":
            return cls.named(cls._wire_string(payload, "model"))
        return cls.backend(cls._wire_string(payload, "backend"), cls._wire_string(payload, "model"))

    @staticmethod
    def _wire_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str):
            raise TypeError(f"ModelRef {key} must be a string")
        return value

    @staticmethod
    def _require_nonempty(field_name: str, value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ModelRef {field_name} cannot be empty")

    def model(self) -> str:
        return self.model_name

    def backend_name(self) -> str | None:
        return self.backend_id


class ModelProvider(Protocol):
    def resolve(self, model: ModelRef) -> ResolvedModelConfig: ...

    def client(self, resolved: ResolvedModelConfig) -> LLMClient: ...

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings: ...

    def default_model_ref(self) -> ModelRef | None: ...


@dataclass(slots=True)
class _RequestCallbackLLM:
    callback: Callable[[LlmRequest], LLMResponse]

    def complete_request(self, request: LlmRequest, *, stream_callback=None) -> LLMResponse:
        del stream_callback
        return self.callback(request)

    def complete(
        self,
        *,
        model,
        messages,
        tools,
        stream_callback=None,
        model_settings=None,
        request_metadata=None,
    ) -> LLMResponse:
        return self.complete_request(
            LlmRequest(
                model=model,
                messages=list(messages),
                tools=list(tools),
                metadata=dict(request_metadata or {}),
                model_settings=model_settings,
            ),
            stream_callback=stream_callback,
        )


@dataclass(frozen=True, slots=True)
class ScriptedModelProvider:
    backend: str
    default_model: str
    llm: LLMClient
    context_length: int | None = 128_000
    max_output_tokens: int | None = 16_384
    settings: ModelSettings = field(default_factory=ModelSettings)

    @classmethod
    def new(cls, backend: str, default_model: str, responses: list[LLMResponse]) -> ScriptedModelProvider:
        return cls(backend=backend, default_model=default_model, llm=ScriptedLLM(steps=list(responses)))

    @classmethod
    def from_steps(cls, backend: str, default_model: str, steps: list[ScriptStep]) -> ScriptedModelProvider:
        return cls(backend=backend, default_model=default_model, llm=ScriptedLLM(steps=list(steps)))

    @classmethod
    def from_callback(
        cls,
        backend: str,
        default_model: str,
        callback: Callable[[LlmRequest], LLMResponse],
    ) -> ScriptedModelProvider:
        return cls(backend=backend, default_model=default_model, llm=_RequestCallbackLLM(callback))

    def with_default_settings(self, settings: ModelSettings) -> ScriptedModelProvider:
        return replace(self, settings=settings)

    def with_token_limits(
        self,
        context_length: int | None,
        max_output_tokens: int | None,
    ) -> ScriptedModelProvider:
        return replace(self, context_length=context_length, max_output_tokens=max_output_tokens)

    def default_model_ref(self) -> ModelRef:
        return ModelRef.named(self.default_model)

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        if model.type == "resolved" and model.resolved_config is not None:
            return model.resolved_config
        if model.backend_id is not None and model.backend_id != self.backend:
            raise ModelError(f"model backend mismatch: requested `{model.backend_id}`, provider `{self.backend}`")
        model_name = model.model_name or self.default_model
        return ResolvedModelConfig(
            backend=self.backend,
            requested_model=model_name,
            selected_model=model_name,
            model_id=model_name,
            endpoint_options=[],
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
            function_call_available=True,
            response_format_available=True,
        )

    def client(self, resolved: ResolvedModelConfig) -> LLMClient:
        del resolved
        return self.llm

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        del resolved
        return self.settings


@dataclass(frozen=True, slots=True)
class VvLlmModelProvider:
    settings_file: Path
    default_backend: str | None = None
    timeout_seconds: float = 90.0

    @classmethod
    def from_settings_file(cls, path: str | Path) -> VvLlmModelProvider:
        return cls(settings_file=Path(path))

    def with_default_backend(self, backend: str) -> VvLlmModelProvider:
        return replace(self, default_backend=backend)

    def with_timeout_seconds(self, timeout_seconds: float) -> VvLlmModelProvider:
        return replace(self, timeout_seconds=max(float(timeout_seconds), 1.0))

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        if model.type == "resolved" and model.resolved_config is not None:
            return model.resolved_config
        backend = model.backend_id or self.default_backend
        if not backend:
            raise ModelError(f"model provider has no default backend for named model `{model.model_name}`")
        settings = load_llm_settings_from_file(self.settings_file)
        return resolve_model_endpoint(settings, backend=backend, model=model.model_name)

    def client(self, resolved: ResolvedModelConfig) -> LLMClient:
        client, _ = build_vv_llm_from_local_settings(
            self.settings_file,
            backend=resolved.backend,
            model=resolved.selected_model,
            timeout_seconds=self.timeout_seconds,
        )
        return client

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        del resolved
        return ModelSettings(timeout_seconds=self.timeout_seconds)

    def default_model_ref(self) -> ModelRef | None:
        return None
