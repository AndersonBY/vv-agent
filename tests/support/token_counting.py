"""The 0.30.1 message-counting algorithm, retained only as a test reference."""

import json
from types import SimpleNamespace
from typing import Any

from vv_llm.chat_clients import utils

from vv_agent.memory import token_utils

PROVIDER_MODELS = ("MiniMax-M2.5", "abab", "moonshot-v1", "kimi-k3", "gemini-3-pro", "claude-sonnet-4", "stepfun", "glm-4-flash")


def provider_counting_settings(model):
    """Mutable endpoint choices let a test compare vv-llm's no-endpoint reference."""
    model_setting = SimpleNamespace(endpoints=[], id=model)
    backend = SimpleNamespace(models={model: model_setting})
    endpoint = SimpleNamespace(
        api_base="https://provider.invalid",
        api_key="unused",
        proxy=None,
        enabled=True,
        is_bedrock=False,
        is_vertex=False,
        endpoint_type="anthropic",
    )
    return backend, model_setting, endpoint


def legacy_count(messages, *, model="", tools=None, native_multimodal=True):
    if not messages:
        return 0
    try:
        kwargs: dict[str, Any] = {"messages": messages, "model": model or "gpt-4o", "native_multimodal": native_multimodal}
        if tools is not None:
            kwargs["tools"] = tools
        result = utils.get_message_token_counts(**kwargs)
        if isinstance(result, int) and result >= 0:
            return result
    except Exception:
        pass
    return token_utils._estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str))
