from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)


def hash_system_prompt_sections(sections: list[object] | None) -> str:
    normalized = []
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        section_dict = cast(dict[str, Any], section)
        text = str(section_dict.get("text") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "id": str(section_dict.get("id") or "").strip(),
                "text": text,
                "stable": bool(section_dict.get("stable", True)),
            }
        )
    if not normalized:
        return ""
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_tool_payload(tools: list[dict[str, Any]] | None) -> str:
    if not tools:
        return ""
    payload = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CacheBreakTracker:
    """Track prompt-cache break causes across repeated requests."""

    _last_system_hash: str = ""
    _last_tool_hash: str = ""
    _total_requests: int = 0
    _cache_breaks: int = 0
    _break_reasons: list[str] = field(default_factory=list)

    def check(self, *, system_hash: str, tool_hash: str = "") -> list[str]:
        reasons: list[str] = []
        if self._last_system_hash and system_hash != self._last_system_hash:
            reasons.append("system_prompt_changed")
        if self._last_tool_hash and tool_hash != self._last_tool_hash:
            reasons.append("tool_schemas_changed")

        self._last_system_hash = system_hash
        self._last_tool_hash = tool_hash
        self._total_requests += 1

        if reasons:
            self._cache_breaks += 1
            self._break_reasons.extend(reasons)
            logger.info(
                "Prompt cache break detected: %s (breaks=%s/%s)",
                reasons,
                self._cache_breaks,
                self._total_requests,
            )
        return reasons

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def cache_breaks(self) -> int:
        return self._cache_breaks

    @property
    def break_reasons(self) -> list[str]:
        return list(self._break_reasons)

    @property
    def cache_hit_rate(self) -> float:
        if self._total_requests <= 0:
            return 1.0
        return 1.0 - (self._cache_breaks / self._total_requests)
