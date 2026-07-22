from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.types import CacheUsage, TaskTokenUsage, TokenUsage, UsageSource

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "token_usage.json"


def _contract() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_normalization_matches_canonical_token_usage_cases() -> None:
    for case in _contract()["normalization_cases"]:
        inputs = case["input"]
        usage = normalize_token_usage(
            inputs["raw_usage"],
            usage_source=inputs["usage_source_hint"],
            cache_status=inputs["cache_status_hint"],
        )

        assert usage.to_dict() == case["expected"], case["name"]


def test_aggregation_matches_canonical_cache_observation_cases() -> None:
    for case in _contract()["aggregation_cases"]:
        summary = TaskTokenUsage()
        for cycle_index, observation in enumerate(case["cycles"], start=1):
            summary.add_cycle(
                cycle_index,
                TokenUsage(
                    total_tokens=1,
                    usage_source=UsageSource.PROVIDER_REPORTED,
                    cache_usage=CacheUsage.from_dict(observation),
                ),
            )

        assert summary.cache_usage.to_dict() == case["expected"], case["name"]


def test_explicit_zero_usage_is_observable_and_superseded_wire_is_rejected() -> None:
    explicit_zero = normalize_token_usage(
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    assert explicit_zero.has_usage() is True
    assert explicit_zero.cache_usage.read_input_tokens == 0
    with pytest.raises(ValueError, match="invalid TokenUsage fields"):
        TokenUsage.from_dict({"cached_tokens": 0})


def test_native_cache_write_usage_is_normalized_without_public_aliases() -> None:
    usage = normalize_token_usage(
        {
            "input_tokens": 300,
            "output_tokens": 50,
            "cache_read_input_tokens": 600,
            "cache_write_input_tokens": 100,
        }
    )

    assert usage.input_tokens == 1000
    assert usage.cache_usage.write_input_tokens == 100
    assert usage.cache_usage.uncached_input_tokens == 400
