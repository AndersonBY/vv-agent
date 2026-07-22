from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.base import LLMClient
from vv_agent.model import ModelError, ModelRef
from vv_agent.model_settings import ModelSettings


@dataclass(slots=True)
class FixedModelProvider:
    llm: LLMClient
    resolved: ResolvedModelConfig
    settings: ModelSettings = field(default_factory=ModelSettings)

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        if model.type == "resolved" and model.resolved_config is not None:
            return model.resolved_config
        requested = model.model_name
        accepted = {
            self.resolved.requested_model,
            self.resolved.selected_model,
            self.resolved.model_id,
        }
        if requested not in accepted:
            raise ModelError(f"test provider does not define model `{requested}`")
        if model.backend_id is not None and model.backend_id != self.resolved.backend:
            raise ModelError(f"model backend mismatch: requested `{model.backend_id}`, provider `{self.resolved.backend}`")
        return self.resolved

    def client(self, resolved: ResolvedModelConfig) -> LLMClient:
        if resolved.model_id != self.resolved.model_id or resolved.backend != self.resolved.backend:
            raise ModelError(f"test provider does not define resolved model `{resolved.backend}/{resolved.model_id}`")
        return self.llm

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        del resolved
        return self.settings

    def default_model_ref(self) -> ModelRef:
        return ModelRef.named(self.resolved.model_id)


@dataclass(slots=True)
class FactoryModelProvider:
    factory: Callable[[], LLMClient]
    resolved: ResolvedModelConfig
    settings: ModelSettings = field(default_factory=ModelSettings)

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        if model.type == "resolved" and model.resolved_config is not None:
            return model.resolved_config
        requested = model.model_name
        accepted = {
            self.resolved.requested_model,
            self.resolved.selected_model,
            self.resolved.model_id,
        }
        if requested not in accepted:
            raise ModelError(f"test provider does not define model `{requested}`")
        if model.backend_id is not None and model.backend_id != self.resolved.backend:
            raise ModelError(f"model backend mismatch: requested `{model.backend_id}`, provider `{self.resolved.backend}`")
        return self.resolved

    def client(self, resolved: ResolvedModelConfig) -> LLMClient:
        if resolved.model_id != self.resolved.model_id or resolved.backend != self.resolved.backend:
            raise ModelError(f"test provider does not define resolved model `{resolved.backend}/{resolved.model_id}`")
        return self.factory()

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        del resolved
        return self.settings

    def default_model_ref(self) -> ModelRef:
        return ModelRef.named(self.resolved.model_id)


@dataclass(slots=True)
class ModelMapProvider:
    routes: dict[str, tuple[LLMClient, ResolvedModelConfig]]
    default_model: str
    settings: ModelSettings = field(default_factory=ModelSettings)
    resolved_models: list[str] = field(default_factory=list, init=False)

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        if model.type == "resolved" and model.resolved_config is not None:
            return model.resolved_config
        self.resolved_models.append(model.model_name)
        route = self.routes.get(model.model_name)
        if route is None:
            raise ModelError(f"test provider does not define model `{model.model_name}`")
        resolved = route[1]
        if model.backend_id is not None and model.backend_id != resolved.backend:
            raise ModelError(f"model backend mismatch: requested `{model.backend_id}`, provider `{resolved.backend}`")
        return resolved

    def client(self, resolved: ResolvedModelConfig) -> LLMClient:
        for client, route in self.routes.values():
            if route.backend == resolved.backend and route.model_id == resolved.model_id:
                return client
        raise ModelError(f"test provider does not define resolved model `{resolved.backend}/{resolved.model_id}`")

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        del resolved
        return self.settings

    def default_model_ref(self) -> ModelRef:
        return ModelRef.named(self.default_model)
