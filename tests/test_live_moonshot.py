from __future__ import annotations

import os
from pathlib import Path

import pytest

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.types import Message

pytestmark = pytest.mark.live


def test_moonshot_openai_compatible_smoke() -> None:
    if os.getenv("V_AGENT_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set V_AGENT_RUN_LIVE_TESTS=1 to run live integration tests")

    settings_file = Path(
        os.getenv(
            "V_AGENT_LOCAL_SETTINGS",
            Path(__file__).resolve().parents[1] / "local_settings.py",
        )
    )

    if not settings_file.exists():
        pytest.skip(f"Live settings file not found: {settings_file}")

    backend = os.getenv("V_AGENT_LIVE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_LIVE_MODEL", "kimi-k2.5")

    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)
    response = llm.complete(
        model=resolved.model_id,
        messages=[
            Message(role="system", content="You are a concise assistant."),
            Message(role="user", content="Reply with exactly one word: pong"),
        ],
        tools=[],
    )

    assert resolved.backend == backend
    assert response.content.strip() or response.raw.get("choices")
