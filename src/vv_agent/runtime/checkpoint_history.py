"""Immutable checkpoint history and the bounded execution frontier.

Stores append the batch returned by ``compact_checkpoint`` in the same CAS as
the mutated checkpoint. This module performs no I/O.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vv_agent.checkpoint import MAX_WIRE_INTEGER, CheckpointError, canonical_json_sha256, validate_sha256
from vv_agent.types import (
    AgentResult,
    CacheUsage,
    CacheUsageStatus,
    CycleRecord,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    TaskTokenUsage,
    TaskTokenUsageTotals,
)

if TYPE_CHECKING:
    from vv_agent.runtime.state import Checkpoint, CheckpointStore

HISTORY_SCHEMA = "vv-agent.checkpoint-history.v1"
_USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")


def empty_history() -> dict[str, Any]:
    return {
        "sequence": 0,
        "cycle_count": 0,
        "model_call_count": 0,
        "head_digest": None,
        "usage": {**dict.fromkeys(_USAGE_FIELDS, 0), "cache_usage": CacheUsage(source="aggregate").to_dict()},
        "previous_agent_input": None,
    }


def validate_history(history: Any) -> None:
    try:
        _validate_history(history)
    except CheckpointError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointError("invalid checkpoint history", code="checkpoint_history_invalid") from exc


def _validate_history(history: Any) -> None:
    if not isinstance(history, dict) or set(history) != set(empty_history()):
        raise CheckpointError("invalid checkpoint history fields", code="checkpoint_history_invalid")
    for name in ("sequence", "cycle_count", "model_call_count"):
        _integer(history[name], name)
    digest = history["head_digest"]
    if history["sequence"] == 0:
        if history != empty_history():
            raise CheckpointError("empty history must have the initial frontier", code="checkpoint_history_invalid")
        return
    validate_sha256(digest, "checkpoint history head_digest")
    if history["cycle_count"] + history["model_call_count"] == 0:
        raise CheckpointError("history batch cannot be empty", code="checkpoint_history_invalid")
    usage = history["usage"]
    if not isinstance(usage, dict) or set(usage) != {*_USAGE_FIELDS, "cache_usage"}:
        raise CheckpointError("invalid history usage fields", code="checkpoint_history_invalid")
    for name in _USAGE_FIELDS:
        if usage[name] is not None:
            _integer(usage[name], name)
    cache = CacheUsage.from_dict(usage["cache_usage"])
    if cache.source != "aggregate":
        raise CheckpointError("history cache usage must be aggregate", code="checkpoint_history_invalid")
    previous = history["previous_agent_input"]
    if previous is not None:
        if not isinstance(previous, dict) or set(previous) != {"cycle_index", "input_tokens"}:
            raise CheckpointError("invalid previous agent input", code="checkpoint_history_invalid")
        _integer(previous["cycle_index"], "cycle_index", minimum=1)
        if previous["input_tokens"] is not None:
            _integer(previous["input_tokens"], "input_tokens")


def _integer(value: Any, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= MAX_WIRE_INTEGER:
        raise CheckpointError(f"invalid history {name}", code="checkpoint_history_invalid")


def _add_usage(history: dict[str, Any], record: ModelCallRecord) -> None:
    usage = history["usage"]
    first = history["model_call_count"] == 0
    for name in _USAGE_FIELDS:
        value = getattr(record.usage, name)
        usage[name] = None if usage[name] is None or value is None else usage[name] + value
    old = CacheUsage.from_dict(usage["cache_usage"])
    new = record.usage.cache_usage
    status = new.status if first or old.status is new.status else CacheUsageStatus.ACCOUNTING_MISSING
    cache = CacheUsage(status=status, source="aggregate")
    for name in ("read_input_tokens", "write_input_tokens", "uncached_input_tokens"):
        left, right = getattr(old, name), getattr(new, name)
        value = right if first else None if left is None or right is None else left + right
        setattr(cache, name, value if status is CacheUsageStatus.PROVIDER_REPORTED else None)
    usage["cache_usage"] = cache.to_dict()
    history["model_call_count"] += 1
    if record.operation is ModelCallOperation.AGENT_CYCLE and record.status is ModelCallStatus.COMPLETED:
        history["previous_agent_input"] = {"cycle_index": record.cycle_index, "input_tokens": record.usage.input_tokens}


def compact_checkpoint(checkpoint: Checkpoint) -> dict[str, Any] | None:
    """Retire immutable prefixes, retaining the newest completed/active cycle.

    The caller must persist the returned batch atomically with the new frontier.
    Unclosed journals always retain their model accounting evidence.
    """
    from vv_agent.runtime.checkpoint_codec import _cycle_to_dict

    validate_history(checkpoint.history)
    if not checkpoint.cycles:
        return None
    committed = [cycle.index for cycle in checkpoint.cycles if cycle.index <= checkpoint.cycle_index]
    if not committed and checkpoint.terminal_result is None:
        return None
    cutoff = checkpoint.cycles[-1].index if checkpoint.terminal_result is not None else max(committed)
    retired_cycles = [cycle for cycle in checkpoint.cycles if cycle.index < cutoff]
    protected_calls = {entry.call_id for entry in checkpoint.model_call_journal}
    retired_calls = [
        record for record in checkpoint.model_calls if record.cycle_index < cutoff and record.call_id not in protected_calls
    ]
    if not retired_cycles and not retired_calls:
        return None
    batch = {
        "schema_version": HISTORY_SCHEMA,
        "checkpoint_key": checkpoint.checkpoint_key,
        "sequence": checkpoint.history["sequence"] + 1,
        "previous_digest": checkpoint.history["head_digest"],
        "cycles": [_cycle_to_dict(cycle) for cycle in retired_cycles],
        "model_calls": [record.to_dict() for record in retired_calls],
    }
    frontier = deepcopy(checkpoint.history)
    frontier["sequence"] = batch["sequence"]
    frontier["cycle_count"] += len(retired_cycles)
    for record in retired_calls:
        _add_usage(frontier, record)
    frontier["head_digest"] = canonical_json_sha256(batch, "checkpoint history batch")
    retired_ids = {record.call_id for record in retired_calls}
    checkpoint.cycles = [cycle for cycle in checkpoint.cycles if cycle.index >= cutoff]
    checkpoint.model_calls = [record for record in checkpoint.model_calls if record.call_id not in retired_ids]
    checkpoint.history = frontier
    if checkpoint.terminal_result is not None:
        checkpoint.terminal_result.cycles = deepcopy(checkpoint.cycles)
        checkpoint.terminal_result.token_usage = _tail_usage(checkpoint)
    validate_history(frontier)
    return batch


def _tail_usage(checkpoint: Checkpoint) -> TaskTokenUsage:
    from vv_agent.runtime.token_usage import summarize_task_token_usage

    return summarize_task_token_usage(checkpoint.model_calls)


def cumulative_checkpoint_usage(checkpoint: Checkpoint) -> TaskTokenUsageTotals:
    """Constant-size cumulative accounting for execution observers."""
    frontier = deepcopy(checkpoint.history)
    for record in checkpoint.model_calls:
        _add_usage(frontier, record)
    usage = frontier["usage"]
    return TaskTokenUsageTotals(
        **{name: usage[name] for name in _USAGE_FIELDS},
        cache_usage=CacheUsage.from_dict(usage["cache_usage"]),
    )


@dataclass(slots=True)
class CheckpointHistory:
    cycles: list[CycleRecord] = field(default_factory=list)
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    frontier: dict[str, Any] = field(default_factory=empty_history)


def decode_history_batches(checkpoint: Checkpoint, batches: list[dict[str, Any]]) -> CheckpointHistory:
    """Explicit history reads authenticate the entire chain and frontier totals."""
    try:
        return _decode_history_batches(checkpoint, batches)
    except CheckpointError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointError("invalid checkpoint history batch", code="checkpoint_history_invalid") from exc


def _decode_history_batches(checkpoint: Checkpoint, batches: list[dict[str, Any]]) -> CheckpointHistory:
    frontier = empty_history()
    result = CheckpointHistory()
    call_ids: set[str] = set()
    cycle_ids: set[int] = set()
    for batch in batches:
        if not isinstance(batch, dict) or set(batch) != {
            "schema_version",
            "checkpoint_key",
            "sequence",
            "previous_digest",
            "cycles",
            "model_calls",
        }:
            raise CheckpointError("invalid history batch fields", code="checkpoint_history_invalid")
        _integer(batch["sequence"], "sequence", minimum=1)
        if (
            batch["schema_version"] != HISTORY_SCHEMA
            or batch["checkpoint_key"] != checkpoint.checkpoint_key
            or batch["sequence"] != frontier["sequence"] + 1
            or batch["previous_digest"] != frontier["head_digest"]
            or not isinstance(batch["cycles"], list)
            or not isinstance(batch["model_calls"], list)
            or not (batch["cycles"] or batch["model_calls"])
        ):
            raise CheckpointError("history chain integrity mismatch", code="checkpoint_history_invalid")
        for raw in batch["cycles"]:
            cycle = CycleRecord.from_dict(raw)
            if cycle.index in cycle_ids or (result.cycles and cycle.index <= result.cycles[-1].index):
                raise CheckpointError("duplicate or unordered historical cycle", code="checkpoint_history_invalid")
            cycle_ids.add(cycle.index)
            result.cycles.append(cycle)
            frontier["cycle_count"] += 1
        for raw in batch["model_calls"]:
            record = ModelCallRecord.from_dict(raw)
            if record.call_id in call_ids:
                raise CheckpointError("duplicate historical model call", code="checkpoint_history_invalid")
            call_ids.add(record.call_id)
            result.model_calls.append(record)
            _add_usage(frontier, record)
        frontier["sequence"] = batch["sequence"]
        frontier["head_digest"] = canonical_json_sha256(batch, "checkpoint history batch")
    if frontier != checkpoint.history:
        raise CheckpointError("history frontier integrity mismatch", code="checkpoint_history_invalid")
    if cycle_ids.intersection(cycle.index for cycle in checkpoint.cycles) or call_ids.intersection(
        record.call_id for record in checkpoint.model_calls
    ):
        raise CheckpointError("history overlaps active checkpoint records", code="checkpoint_history_invalid")
    result.frontier = deepcopy(frontier)
    return result


def hydrate_checkpoint_result(store: CheckpointStore, checkpoint: Checkpoint, result: AgentResult) -> AgentResult:
    """Materialize a complete public result only at an explicit result boundary."""
    from vv_agent.runtime.token_usage import summarize_task_token_usage

    hydrated = deepcopy(result)
    if checkpoint.history["sequence"] == 0:
        return hydrated
    archive = store.load_checkpoint_history(checkpoint.checkpoint_key)
    if archive.frontier != checkpoint.history:
        raise CheckpointError("checkpoint history changed during result hydration", code="checkpoint_history_changed")
    tail_cycles = [cycle for cycle in hydrated.cycles if cycle.index not in {item.index for item in archive.cycles}]
    hydrated.cycles = archive.cycles + tail_cycles
    calls = archive.model_calls + checkpoint.model_calls
    hydrated.token_usage = summarize_task_token_usage(calls)
    return hydrated
