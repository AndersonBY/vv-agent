"""Shared requests for boundary construction, model planning and guarded commits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._records import Record


@dataclass(frozen=True, slots=True)
class CreateBoundary:
    stage: str
    identity: str
    data: dict[str, Any]
    source_operation_id: str | None = None


@dataclass(frozen=True, slots=True)
class PlanModel:
    oid: str
    request: dict[str, Any]
    purpose: str


@dataclass(frozen=True, slots=True)
class CommitRecords:
    records: list[Record]
