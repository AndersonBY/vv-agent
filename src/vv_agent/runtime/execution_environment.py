"""Public W3 protocol for provider-backed execution environments.

The protocol owns lifecycle operations only. Stable command identity remains in
``CommandSpec``, ``LookupBinding`` and ``ProviderLookupResult`` so adapters and
callers share one canonical representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from vv_agent.session.environments import (
    CommandSpec,
    EnvironmentCapabilities,
    LookupBinding,
    ProviderLookupResult,
)

CommandHandle = Mapping[str, Any]
OperationalPolicy = Mapping[str, str]


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """A provider-backed environment with recoverable command executions."""

    capabilities: EnvironmentCapabilities
    provider_id: str

    def start(
        self,
        command: CommandSpec,
        binding: LookupBinding,
        *,
        expected_capabilities: EnvironmentCapabilities | Mapping[str, Any],
        expected_operational_policy: OperationalPolicy,
    ) -> ProviderLookupResult: ...

    def lookup(self, execution_id: str, binding: LookupBinding) -> ProviderLookupResult: ...

    def poll(self, handle: CommandHandle, binding: LookupBinding) -> ProviderLookupResult: ...

    def write_stdin(
        self,
        handle: CommandHandle,
        data: bytes,
        binding: LookupBinding,
    ) -> ProviderLookupResult: ...

    def stop(self, handle: CommandHandle, binding: LookupBinding) -> ProviderLookupResult: ...

    def search(self, *args: object, **kwargs: object) -> object:
        """Run the adapter-defined search operation or raise its capability error."""
        ...


__all__ = [
    "CommandHandle",
    "ExecutionEnvironment",
    "OperationalPolicy",
]
