from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from vv_agent import (
    Agent,
    ModelRef,
    ModelSettings,
    RunConfig,
    Runner,
    ScriptedModelProvider,
    VvLlmModelProvider,
)
from vv_agent.config import ConfigError, ResolvedModelConfig
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.model import ModelError, resolve_model_selection
from vv_agent.model_settings import RetrySettings
from vv_agent.types import LLMResponse, Message


@pytest.fixture
def vv_provider(tmp_path: Path) -> VvLlmModelProvider:
    settings = {
        "VERSION": "2",
        "backends": {
            "qwen": {
                "models": {
                    "qwen3.6-35b-a3b": {
                        "id": "catalog-id",
                        "endpoints": [{"endpoint_id": "siliconflow", "model_id": "Qwen/Qwen3.6-35B-A3B"}],
                        "context_length": 128_000,
                        "max_output_tokens": 8192,
                    }
                }
            }
        },
        "endpoints": [{"id": "siliconflow", "api_key": "unused", "api_base": "https://example.invalid/v1"}],
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return VvLlmModelProvider.from_settings_file(path).with_default_backend("qwen")


@dataclass
class _WrappedModelProvider:
    provider: VvLlmModelProvider

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        return replace(self.provider.resolve(model), context_length=64_000)

    def client(self, resolved: ResolvedModelConfig):
        return self.provider.client(resolved)

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        return self.provider.default_settings(resolved)

    def default_model_ref(self) -> ModelRef | None:
        return self.provider.default_model_ref()


@pytest.mark.parametrize("explicit_backend", [False, True])
def test_vv_provider_resolves_endpoint_model_id(vv_provider, explicit_backend):
    model_id = "Qwen/Qwen3.6-35B-A3B"
    ref = ModelRef.backend("qwen", model_id) if explicit_backend else ModelRef.named(model_id)
    resolved = vv_provider.resolve(ref)

    assert resolved.selected_model == "qwen3.6-35b-a3b"
    assert resolved.requested_model == model_id
    assert resolved.model_id == model_id
    assert resolved.context_length == 128_000
    assert resolved.max_output_tokens == 8192
    assert resolved.endpoint.endpoint_id == "siliconflow"
    assert vv_provider.resolve(ModelRef.resolved(resolved)) is resolved


@pytest.mark.parametrize("default_backend", ["qwen", None])
@pytest.mark.parametrize("wrapped", [False, True])
def test_vv_provider_runs_session_with_endpoint_model_id(vv_provider, tmp_path, monkeypatch, default_backend, wrapped):
    requests: list[LlmRequest] = []
    resolved_configs: list[ResolvedModelConfig] = []

    def respond(request):
        requests.append(request)
        return LLMResponse(content="done")

    llm = ScriptedModelProvider.from_callback("qwen", "unused", respond).llm

    def client(self, resolved):
        resolved_configs.append(resolved)
        return llm

    monkeypatch.setattr(VvLlmModelProvider, "client", client)
    provider = replace(vv_provider, default_backend=default_backend)
    provider = _WrappedModelProvider(provider) if wrapped else provider
    result = Runner.run_sync(
        Agent("assistant", "Finish.", model=ModelRef.backend("qwen", "qwen3.6-35b-a3b")),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=provider),
    )

    assert result.final_output == "done"
    assert [request.model for request in requests] == ["Qwen/Qwen3.6-35B-A3B"]
    assert len(resolved_configs) >= 2
    assert all(config.selected_model == "qwen3.6-35b-a3b" for config in resolved_configs)
    assert all(config.context_length == (64_000 if wrapped else 128_000) for config in resolved_configs)


def test_vv_provider_model_id_fallback_rejects_unknown_and_wrong_backend(vv_provider):
    for name in ("unknown-model", "catalog-id"):
        with pytest.raises(ConfigError, match="not found under backend 'qwen'"):
            vv_provider.resolve(ModelRef.named(name))
    with pytest.raises(ConfigError, match="Backend 'other' not found"):
        vv_provider.resolve(ModelRef.backend("other", "Qwen/Qwen3.6-35B-A3B"))
    with pytest.raises(ModelError, match="no default backend"):
        replace(vv_provider, default_backend=None).resolve(ModelRef.named("qwen3.6-35b-a3b"))


def test_vv_provider_model_id_fallback_rejects_ambiguity_and_keeps_key_priority(vv_provider):
    settings = json.loads(vv_provider.settings_file.read_text(encoding="utf-8"))
    models = settings["backends"]["qwen"]["models"]
    models["duplicate"] = models["qwen3.6-35b-a3b"]
    vv_provider.settings_file.write_text(json.dumps(settings), encoding="utf-8")

    assert vv_provider.resolve(ModelRef.named("qwen3.6-35b-a3b")).selected_model == "qwen3.6-35b-a3b"
    with pytest.raises(ModelError, match="exactly one"):
        vv_provider.resolve(ModelRef.named("Qwen/Qwen3.6-35B-A3B"))


def test_vv_provider_model_id_fallback_scopes_backend(vv_provider):
    settings = json.loads(vv_provider.settings_file.read_text(encoding="utf-8"))
    settings["backends"]["other"] = settings["backends"]["qwen"]
    vv_provider.settings_file.write_text(json.dumps(settings), encoding="utf-8")

    assert vv_provider.resolve(ModelRef.named("Qwen/Qwen3.6-35B-A3B")).backend == "qwen"
    assert vv_provider.resolve(ModelRef.backend("other", "Qwen/Qwen3.6-35B-A3B")).backend == "other"
    with pytest.raises(ModelError, match="exactly one"):
        replace(vv_provider, default_backend=None).resolve(ModelRef.named("Qwen/Qwen3.6-35B-A3B"))


@pytest.mark.parametrize("wrapped", [False, True])
def test_default_resolver_preserves_model_id_and_rejects_route(vv_provider, monkeypatch, wrapped):
    llm = ScriptedLLM([])
    monkeypatch.setattr(VvLlmModelProvider, "client", lambda self, resolved: llm)
    provider = _WrappedModelProvider(vv_provider) if wrapped else vv_provider
    client, resolved = resolve_model_selection(provider, {"model": "Qwen/Qwen3.6-35B-A3B", "route": {}})

    assert client is llm
    assert resolved.model_id == "Qwen/Qwen3.6-35B-A3B"
    with pytest.raises(ValueError, match="empty route"):
        resolve_model_selection(provider, {"model": resolved.model_id, "route": {"backend": "other"}})
    with pytest.raises((ModelError, ValueError), match=r"exactly one|changed the requested model"):
        resolve_model_selection(provider, {"model": "qwen3.6-35b-a3b", "route": {}})


@pytest.mark.parametrize("entry", [None, {"id": "other-id", "endpoints": ["siliconflow"]}])
def test_vv_provider_keeps_exact_key_precedence_over_endpoint_id(vv_provider, entry):
    model_id = "Qwen/Qwen3.6-35B-A3B"
    settings = json.loads(vv_provider.settings_file.read_text(encoding="utf-8"))
    settings["backends"]["qwen"]["models"][model_id] = entry
    vv_provider.settings_file.write_text(json.dumps(settings), encoding="utf-8")

    if entry is None:
        with pytest.raises(ConfigError, match="not found under backend"):
            vv_provider.resolve(ModelRef.named(model_id))
    else:
        resolved = vv_provider.resolve(ModelRef.named(model_id))
        assert (resolved.selected_model, resolved.model_id) == (model_id, "other-id")


def test_scripted_model_provider_runs_through_the_shared_request_contract(tmp_path: Path) -> None:
    requests: list[LlmRequest] = []

    def respond(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content="done")

    provider = ScriptedModelProvider.from_callback("scripted", "demo-model", respond).with_default_settings(
        ModelSettings(temperature=0.1, max_tokens=100)
    )
    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Finish.",
            model=ModelRef.named("demo-model"),
            model_settings=ModelSettings(temperature=0.2),
        ),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=provider,
            model_settings=ModelSettings(max_tokens=300),
        ),
    )

    assert result.final_output == "done"
    assert len(requests) == 1
    assert requests[0].model == "demo-model"
    assert requests[0].model_settings == ModelSettings(
        temperature=0.2, max_tokens=300, retry=RetrySettings(max_attempts=1, backoff_seconds=0)
    )
    assert requests[0].metadata["trace_id"] == result.trace_id
    assert requests[0].prompt_bundle is not None
    assert requests[0].prompt_bundle.sections


def test_scripted_llm_callback_receives_the_complete_request() -> None:
    requests: list[LlmRequest] = []
    llm = ScriptedLLM(steps=[lambda request: requests.append(request) or LLMResponse(content="done")])
    request = LlmRequest(
        model="demo-model",
        messages=[Message(role="user", content="hello")],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        metadata={"trace_id": "trace-1"},
        model_settings=ModelSettings(temperature=0.2),
    )

    response = llm.complete(request)

    assert response.content == "done"
    assert requests == [request]
    assert requests[0].tools[0]["function"] == {"name": "lookup"}
    assert requests[0].metadata == {"trace_id": "trace-1"}
    assert requests[0].model_settings == ModelSettings(temperature=0.2)


def test_scripted_provider_default_model_is_runner_fallback(tmp_path: Path) -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "provider-default",
        [LLMResponse(content="done")],
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Finish."),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=provider),
    )

    assert result.resolved is not None
    assert result.resolved.model_id == "provider-default"
    assert result.resolved.function_call_available is True
    assert result.resolved.response_format_available is True


def test_model_ref_and_provider_backend_validation_match_rust_contract() -> None:
    model = ModelRef.backend("other", "demo")
    provider = ScriptedModelProvider.new("scripted", "demo", [])

    assert model.model() == "demo"
    assert model.backend_name() == "other"
    with pytest.raises(ModelError, match="backend mismatch"):
        provider.resolve(model)


def test_model_ref_wire_matches_shared_closed_contract() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "parity" / "model_ref.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    refs = [ModelRef.from_dict(payload) for payload in fixture["valid"]]
    assert [ref.to_dict() for ref in refs] == fixture["valid"]
    for payload in fixture["invalid"]:
        with pytest.raises((TypeError, ValueError)):
            ModelRef.from_dict(payload)


def test_resolved_model_ref_cannot_serialize_credentials() -> None:
    resolved = _resolved_for_wire_test()

    with pytest.raises(ValueError, match="process-local"):
        ModelRef.resolved(resolved).to_dict()


def _resolved_for_wire_test() -> ResolvedModelConfig:
    from vv_agent.config import EndpointConfig, EndpointOption

    endpoint = EndpointConfig(
        endpoint_id="private",
        api_key="must-not-serialize",
        api_base="https://example.invalid/v1",
    )
    return ResolvedModelConfig(
        backend="private",
        requested_model="demo",
        selected_model="demo",
        model_id="demo",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="demo")],
    )
