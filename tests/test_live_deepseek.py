from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import Agent, AgentStatus, ModelRef, RunConfig, Runner, VvLlmModelProvider
from vv_agent.memory import SessionMemory
from vv_agent.model import ResolvedModelConfig
from vv_agent.model_settings import ModelSettings
from vv_agent.types import LLMResponse, Message

pytestmark = pytest.mark.live


class _RecordingClient:
    def __init__(self, inner: Any, observations: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._observations = observations

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Any = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        request_summary = {
            "model": model,
            "message_roles": [message.role for message in messages],
            "message_count": len(messages),
            "tool_count": len(tools),
            "request_max_tokens": model_settings.max_tokens if model_settings is not None else None,
        }
        try:
            response = self._inner.complete(
                model=model,
                messages=messages,
                tools=tools,
                stream_callback=stream_callback,
                model_settings=model_settings,
                request_metadata=request_metadata,
            )
        except BaseException as exc:
            self._observations.append({**request_summary, "error_type": type(exc).__name__})
            raise

        raw_usage = response.raw.get("usage")
        self._observations.append(
            {
                **request_summary,
                "response_content_chars": len(response.content or ""),
                "response_json_array_items": _json_array_item_count(response.content or ""),
                "usage": _usage_summary(raw_usage),
            }
        )
        return response


class _RecordingProvider:
    def __init__(self, inner: VvLlmModelProvider) -> None:
        self._inner = inner
        self.observations: list[dict[str, Any]] = []

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        return self._inner.resolve(model)

    def client(self, resolved: ResolvedModelConfig) -> _RecordingClient:
        return _RecordingClient(self._inner.client(resolved), self.observations)

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        return self._inner.default_settings(resolved)

    def default_model_ref(self) -> ModelRef | None:
        return self._inner.default_model_ref()


def _usage_summary(raw_usage: object) -> dict[str, Any] | None:
    if not isinstance(raw_usage, dict):
        return None
    usage = cast(dict[str, Any], raw_usage)
    prompt_details = usage.get("prompt_tokens_details")
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": (
            prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else usage.get("cached_tokens")
        ),
    }


def _json_array_item_count(text: str) -> int | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, list):
            return len(candidate)
    return None


def _live_settings_path() -> Path:
    return Path(
        os.getenv(
            "VV_AGENT_LOCAL_SETTINGS",
            Path(__file__).resolve().parents[1] / "local_settings.py",
        )
    )


def test_deepseek_session_memory_probe_accounts_for_every_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("VV_AGENT_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set VV_AGENT_RUN_LIVE_TESTS=1 to run live integration tests")

    settings_file = _live_settings_path()
    if not settings_file.exists():
        pytest.skip(f"Live settings file not found: {settings_file}")

    backend = os.getenv("VV_AGENT_LIVE_BACKEND", "deepseek")
    model = os.getenv("VV_AGENT_LIVE_MODEL", "deepseek-v4-pro")
    extraction_observations: list[dict[str, Any]] = []
    original_extract = SessionMemory.extract

    def observed_extract(
        memory: SessionMemory,
        messages: list[Message],
        *,
        current_cycle: int,
        current_tokens: int,
    ) -> int:
        merged = original_extract(
            memory,
            messages,
            current_cycle=current_cycle,
            current_tokens=current_tokens,
        )
        extraction_observations.append(
            {
                "merged_entries": merged,
                "persisted_entries": len(memory.state.entries),
                "workspace_configured": memory.workspace is not None,
                "storage_path_available": memory._storage_path() is not None,
            }
        )
        return merged

    monkeypatch.setattr(SessionMemory, "extract", observed_extract)
    provider = _RecordingProvider(
        VvLlmModelProvider.from_settings_file(settings_file)
        .with_default_backend(backend)
        .with_timeout_seconds(300)
    )
    agent = Agent(
        name="deepseek-session-memory-probe",
        instructions="Reply with one short sentence.",
        model=ModelRef.backend(backend, model),
        max_cycles=1,
        no_tool_policy="finish",
    )
    result = Runner.run_sync(
        agent,
        "Remember that the probe marker is cobalt, then acknowledge it.",
        run_config=RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            max_cycles=1,
            no_tool_policy="finish",
            session_memory_enabled=True,
            metadata={
                "session_id": "deepseek-session-memory-probe",
                "session_memory_min_tokens": 1,
                "session_memory_min_text_messages": 1,
            },
        ),
    )

    memory_files = sorted(tmp_path.glob(".memory/session/*/session_memory.json"))
    memory_entries = []
    if memory_files:
        memory_entries = json.loads(memory_files[0].read_text(encoding="utf-8")).get("entries", [])
    probe_summary = {
        "implementation": "python",
        "extractions": extraction_observations,
        "model_calls": provider.observations,
        "session_memory_files": len(memory_files),
        "session_memory_entries": len(memory_entries),
        "reported_model_call_count": len(result.token_usage.model_calls),
        "reported_usage": {
            "input_tokens": result.token_usage.input_tokens,
            "output_tokens": result.token_usage.output_tokens,
            "total_tokens": result.token_usage.total_tokens,
            "cached_input_tokens": result.token_usage.cache_usage.read_input_tokens,
        },
    }
    print(json.dumps(probe_summary, ensure_ascii=False, sort_keys=True))

    assert result.status is AgentStatus.COMPLETED
    assert len(provider.observations) == 2
    assert provider.observations[0]["message_roles"] == ["user"]
    assert memory_files
    assert memory_entries
    assert len(result.token_usage.model_calls) == 2
