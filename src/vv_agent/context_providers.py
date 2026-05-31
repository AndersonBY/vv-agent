from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ContextFragment:
    id: str
    text: str
    stable: bool = True
    priority: int = 100
    source: str = ""
    cache_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextRequest:
    agent_name: str
    input: str = ""
    model: str | None = None
    trace_id: str | None = None
    session: Any | None = None
    workspace: str | Path | Any | None = None
    context: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_prompt_chars: int | None = None


class ContextProvider(Protocol):
    def fragments(self, request: ContextRequest) -> list[ContextFragment]:
        ...


@dataclass(frozen=True, slots=True)
class ContextSection:
    id: str
    text: str
    stable: bool = True
    priority: int = 100
    source: str = ""
    cache_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "stable": self.stable,
        }
        if self.source:
            payload["source"] = self.source
        if self.cache_hint is not None:
            payload["cache_hint"] = self.cache_hint
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ContextBundle:
    prompt: str
    sections: list[ContextSection]
    sources: dict[str, str]
    stable_hash: str
    total_chars: int
    omitted_section_ids: list[str] = field(default_factory=list)

    def metadata_sections(self) -> list[dict[str, Any]]:
        return [section.to_metadata() for section in self.sections]


def collect_context_fragments(
    request: ContextRequest,
    providers: Iterable[ContextProvider],
) -> list[ContextFragment]:
    fragments: list[ContextFragment] = []
    for provider in providers:
        fragments.extend(provider.fragments(request))
    return fragments


def assemble_context_fragments(
    request: ContextRequest,
    fragments: Iterable[ContextFragment],
) -> ContextBundle:
    ordered_fragments = sorted(
        fragments,
        key=lambda fragment: (
            int(fragment.priority),
            0 if fragment.stable else 1,
            str(fragment.id),
        ),
    )
    sections: list[ContextSection] = []
    prompt_parts: list[str] = []
    omitted_section_ids: list[str] = []
    total_chars = 0

    for fragment in ordered_fragments:
        text = str(fragment.text or "").strip()
        if not text:
            continue
        separator_chars = 2 if prompt_parts else 0
        next_total = total_chars + separator_chars + len(text)
        if request.max_prompt_chars is not None and next_total > max(int(request.max_prompt_chars), 0):
            omitted_section_ids.append(fragment.id)
            continue
        sections.append(
            ContextSection(
                id=fragment.id,
                text=text,
                stable=fragment.stable,
                priority=fragment.priority,
                source=fragment.source,
                cache_hint=fragment.cache_hint,
                metadata=dict(fragment.metadata),
            )
        )
        prompt_parts.append(text)
        total_chars = next_total

    stable_text = "".join(section.text for section in sections if section.stable).encode("utf-8")
    return ContextBundle(
        prompt="\n\n".join(prompt_parts),
        sections=sections,
        sources={section.id: section.source for section in sections if section.source},
        stable_hash=hashlib.sha256(stable_text).hexdigest(),
        total_chars=total_chars,
        omitted_section_ids=omitted_section_ids,
    )


__all__ = [
    "ContextBundle",
    "ContextFragment",
    "ContextProvider",
    "ContextRequest",
    "ContextSection",
    "assemble_context_fragments",
    "collect_context_fragments",
]
