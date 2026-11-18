"""W9.6 export alias retained until the next Python package major release."""

if __name__ == "types":
    import os as _os
    import sys as _sys
    from importlib.machinery import PathFinder as _PathFinder

    _spec = _PathFinder.find_spec("types", [_os.path.dirname(_os.__file__)])
    if _spec is None or _spec.loader is None:
        raise ImportError("standard-library types module unavailable")
    # Keep importlib's original spec so concurrent imports wait for initialization.
    assert __spec__ is not None
    __spec__.origin = __file__ = _spec.origin
    __spec__.loader = __loader__ = _spec.loader
    __spec__.cached = __cached__ = _spec.cached
    _spec.loader.exec_module(_sys.modules["types"])
    del _os, _sys, _PathFinder, _spec
else:
    from vv_agent.core_types import _MESSAGE_FIELDS as _MESSAGE_FIELDS
    from vv_agent.core_types import COMPACTION_METADATA_KEY as COMPACTION_METADATA_KEY
    from vv_agent.core_types import MAX_WIRE_INTEGER as MAX_WIRE_INTEGER
    from vv_agent.core_types import MODEL_CALL_SCHEMA_VERSION as MODEL_CALL_SCHEMA_VERSION
    from vv_agent.core_types import TASK_TOKEN_USAGE_SCHEMA_VERSION as TASK_TOKEN_USAGE_SCHEMA_VERSION
    from vv_agent.core_types import TOKEN_USAGE_SCHEMA_VERSION as TOKEN_USAGE_SCHEMA_VERSION
    from vv_agent.core_types import AgentResult as AgentResult
    from vv_agent.core_types import AgentStatus as AgentStatus
    from vv_agent.core_types import AgentTask as AgentTask
    from vv_agent.core_types import BudgetExhaustion as BudgetExhaustion
    from vv_agent.core_types import BudgetUsageSnapshot as BudgetUsageSnapshot
    from vv_agent.core_types import CacheUsage as CacheUsage
    from vv_agent.core_types import CacheUsageStatus as CacheUsageStatus
    from vv_agent.core_types import CompletionReason as CompletionReason
    from vv_agent.core_types import CycleRecord as CycleRecord
    from vv_agent.core_types import CycleStatus as CycleStatus
    from vv_agent.core_types import LLMResponse as LLMResponse
    from vv_agent.core_types import Message as Message
    from vv_agent.core_types import MicrocompactionPolicy as MicrocompactionPolicy
    from vv_agent.core_types import ModelCallOperation as ModelCallOperation
    from vv_agent.core_types import ModelCallRecord as ModelCallRecord
    from vv_agent.core_types import ModelCallStatus as ModelCallStatus
    from vv_agent.core_types import ModelSettings as ModelSettings
    from vv_agent.core_types import NoToolPolicy as NoToolPolicy
    from vv_agent.core_types import PromptBundle as PromptBundle
    from vv_agent.core_types import Role as Role
    from vv_agent.core_types import SubAgentConfig as SubAgentConfig
    from vv_agent.core_types import SubTaskOutcome as SubTaskOutcome
    from vv_agent.core_types import SubTaskRequest as SubTaskRequest
    from vv_agent.core_types import TaskTokenUsage as TaskTokenUsage
    from vv_agent.core_types import TaskTokenUsageTotals as TaskTokenUsageTotals
    from vv_agent.core_types import TokenUsage as TokenUsage
    from vv_agent.core_types import ToolArtifactRef as ToolArtifactRef
    from vv_agent.core_types import ToolCall as ToolCall
    from vv_agent.core_types import ToolDirective as ToolDirective
    from vv_agent.core_types import ToolExecutionResult as ToolExecutionResult
    from vv_agent.core_types import ToolResultCursor as ToolResultCursor
    from vv_agent.core_types import ToolResultStatus as ToolResultStatus
    from vv_agent.core_types import UsageSource as UsageSource
    from vv_agent.core_types import _agent_result_error_text as _agent_result_error_text
    from vv_agent.core_types import _aggregate_cache_usage as _aggregate_cache_usage
    from vv_agent.core_types import _last_assistant_output as _last_assistant_output
    from vv_agent.core_types import _trim_portable_whitespace as _trim_portable_whitespace
    from vv_agent.core_types import _validate_no_tool_policy as _validate_no_tool_policy
    from vv_agent.core_types import canonical_json_bytes as canonical_json_bytes
    from vv_agent.core_types import normalize_microcompaction_policy as normalize_microcompaction_policy
    from vv_agent.core_types import validate_compaction_metadata as validate_compaction_metadata
