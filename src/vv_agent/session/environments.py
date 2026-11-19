"""Compatibility exports for kernel-owned W3 environment values."""

from vv_agent.kernel._environment_values import CommandSpec as CommandSpec
from vv_agent.kernel._environment_values import EnvironmentCapabilities as EnvironmentCapabilities
from vv_agent.kernel._environment_values import EnvironmentContractError as EnvironmentContractError
from vv_agent.kernel._environment_values import InvalidCommandSpec as InvalidCommandSpec
from vv_agent.kernel._environment_values import InvalidEnvironmentCapabilities as InvalidEnvironmentCapabilities
from vv_agent.kernel._environment_values import InvalidLookupBinding as InvalidLookupBinding
from vv_agent.kernel._environment_values import InvalidProviderLookupResult as InvalidProviderLookupResult
from vv_agent.kernel._environment_values import IsolationMode as IsolationMode
from vv_agent.kernel._environment_values import LookupBinding as LookupBinding
from vv_agent.kernel._environment_values import LookupStatus as LookupStatus
from vv_agent.kernel._environment_values import NetworkMode as NetworkMode
from vv_agent.kernel._environment_values import ProviderLookupResult as ProviderLookupResult
from vv_agent.kernel._environment_values import StdinMode as StdinMode
from vv_agent.kernel._environment_values import execution_id_for as execution_id_for
from vv_agent.kernel._environment_values import stable_operation_key as stable_operation_key

__all__ = [
    "CommandSpec",
    "EnvironmentCapabilities",
    "EnvironmentContractError",
    "InvalidCommandSpec",
    "InvalidEnvironmentCapabilities",
    "InvalidLookupBinding",
    "InvalidProviderLookupResult",
    "IsolationMode",
    "LookupBinding",
    "LookupStatus",
    "NetworkMode",
    "ProviderLookupResult",
    "StdinMode",
    "execution_id_for",
    "stable_operation_key",
]
