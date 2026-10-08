"""Explicit process-local shared-state bindings; only names enter the log."""

from typing import Any

from vv_agent.canonical_json import canonical_json_bytes


class MissingHostBinding(ValueError):
    pass


def bind_shared_state(state: dict[str, Any], bindings: dict[str, Any], required: list[str]) -> dict[str, Any]:
    missing = sorted(set(required) - bindings.keys())
    if missing:
        raise MissingHostBinding(f"Missing shared_state host binding: {', '.join(missing)}")
    if state.keys() & bindings.keys():
        raise ValueError("host bindings cannot shadow durable shared_state")
    return state | {name: bindings[name] for name in required}


def durable_shared_state(state: dict[str, Any], bindings: dict[str, Any]) -> dict[str, Any]:
    for name, value in bindings.items():
        if name not in state or state[name] is not value:
            raise ValueError(f"host binding {name!r} cannot be replaced or deleted")
    durable = {name: value for name, value in state.items() if name not in bindings} if bindings else state
    canonical_json_bytes(durable)
    return durable
