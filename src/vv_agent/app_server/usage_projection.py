from __future__ import annotations

from copy import deepcopy
from typing import Any

from vv_agent.types import ModelCallRecord, TaskTokenUsage, TokenUsage


def token_usage_to_wire(usage: TokenUsage) -> dict[str, Any]:
    cache = usage.cache_usage
    return {
        "schemaVersion": usage.to_dict()["schema_version"],
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
        "totalTokens": usage.total_tokens,
        "reasoningTokens": usage.reasoning_tokens,
        "usageSource": usage.usage_source.value,
        "cacheUsage": {
            "status": cache.status.value,
            "readInputTokens": cache.read_input_tokens,
            "writeInputTokens": cache.write_input_tokens,
            "uncachedInputTokens": cache.uncached_input_tokens,
            "source": cache.source,
        },
        "providerUsage": deepcopy(usage.provider_usage),
    }


def model_call_to_wire(model_call: ModelCallRecord) -> dict[str, Any]:
    return {
        "callId": model_call.call_id,
        "operationId": model_call.operation_id,
        "attempt": model_call.attempt,
        "operation": model_call.operation.value,
        "cycleIndex": model_call.cycle_index,
        "backend": model_call.backend,
        "model": model_call.model,
        "status": model_call.status.value,
        "usage": token_usage_to_wire(model_call.usage),
        "errorCode": model_call.error_code,
    }


def task_token_usage_to_wire(usage: TaskTokenUsage) -> dict[str, Any]:
    cache = usage.cache_usage
    return {
        "schemaVersion": usage.to_dict()["schema_version"],
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
        "totalTokens": usage.total_tokens,
        "reasoningTokens": usage.reasoning_tokens,
        "cacheUsage": {
            "status": cache.status.value,
            "readInputTokens": cache.read_input_tokens,
            "writeInputTokens": cache.write_input_tokens,
            "uncachedInputTokens": cache.uncached_input_tokens,
            "source": cache.source,
        },
        "modelCalls": [model_call_to_wire(model_call) for model_call in usage.model_calls],
    }
