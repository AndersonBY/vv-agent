"""The 0.30.1 message-counting algorithm, retained only as a test reference."""

import json
from typing import Any

from vv_llm.chat_clients import utils

from vv_agent.memory import token_utils


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
