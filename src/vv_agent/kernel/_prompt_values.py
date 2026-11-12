"""Resolved prompt sections and bundles, immutable metadata and stable digests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ._canonical_json import canonical_json_sha256

_ASCII_WHITESPACE = "\t\n\v\f\r "


def _trim_text(value: str) -> str:
    return value.strip(_ASCII_WHITESPACE)


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One resolved, ordered system-prompt section."""

    id: str
    text: str
    stable: bool = True
    source: str | None = None
    cache_hint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("prompt section id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("prompt section text must be a string")
        text = _trim_text(self.text)
        if not text:
            raise ValueError("prompt section text cannot be empty")
        if not isinstance(self.stable, bool):
            raise TypeError("prompt section stable must be a boolean")
        for name in ("source", "cache_hint"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"prompt section {name} must be omitted or a non-empty string")
        if not isinstance(self.metadata, Mapping) or not all(isinstance(key, str) for key in self.metadata):
            raise TypeError("prompt section metadata must be an object with string keys")
        metadata = _thaw_json_value(self.metadata)
        canonical_json_sha256(metadata, "prompt section metadata")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "metadata", _freeze_json_value(metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "text": self.text, "stable": self.stable}
        if self.source is not None:
            payload["source"] = self.source
        if self.cache_hint is not None:
            payload["cache_hint"] = self.cache_hint
        if self.metadata:
            payload["metadata"] = _thaw_json_value(self.metadata)
        return payload

    def __deepcopy__(self, memo: dict[int, Any]) -> PromptSection:
        memo[id(self)] = self
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptSection:
        if not isinstance(payload, dict):
            raise TypeError("PromptSection payload must be an object")
        required = {"id", "text", "stable"}
        allowed = {*required, "source", "cache_hint", "metadata"}
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - allowed)
        if missing or unknown:
            raise ValueError(f"PromptSection fields are invalid: missing={missing}, unknown={unknown}")
        return cls(
            id=payload["id"],
            text=payload["text"],
            stable=payload["stable"],
            source=payload.get("source"),
            cache_hint=payload.get("cache_hint"),
            metadata=payload.get("metadata", {}),
        )


def _stable_hash(sections: tuple[PromptSection, ...]) -> str:
    return canonical_json_sha256(
        [section.to_dict() for section in sections if section.stable],
        "stable prompt sections",
    )


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """The canonical resolved system prompt carried through one run."""

    sections: tuple[PromptSection, ...]
    stable_hash: str = ""

    def __post_init__(self) -> None:
        sections = tuple(self.sections)
        if not sections or not all(isinstance(section, PromptSection) for section in sections):
            raise ValueError("prompt bundle sections must be a non-empty sequence of PromptSection values")
        ids = [section.id for section in sections]
        if len(ids) != len(set(ids)):
            raise ValueError("prompt bundle section ids must be unique")
        expected_hash = _stable_hash(sections)
        if self.stable_hash and self.stable_hash != expected_hash:
            raise ValueError("prompt bundle stable_hash does not match stable sections")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "stable_hash", expected_hash)

    def flatten(self) -> str:
        return "\n\n".join(section.text for section in self.sections)

    def __deepcopy__(self, memo: dict[int, Any]) -> PromptBundle:
        memo[id(self)] = self
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [section.to_dict() for section in self.sections],
            "stable_hash": self.stable_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptBundle:
        if not isinstance(payload, dict):
            raise TypeError("PromptBundle payload must be an object")
        if set(payload) != {"sections", "stable_hash"}:
            raise ValueError("PromptBundle must contain exactly sections and stable_hash")
        raw_sections = payload["sections"]
        if not isinstance(raw_sections, list):
            raise TypeError("PromptBundle sections must be an array")
        stable_hash = payload["stable_hash"]
        if not isinstance(stable_hash, str):
            raise TypeError("PromptBundle stable_hash must be a string")
        return cls(
            sections=tuple(PromptSection.from_dict(section) for section in raw_sections),
            stable_hash=stable_hash,
        )
