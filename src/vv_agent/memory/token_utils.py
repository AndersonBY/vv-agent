"""Existing memory counter exports; implementation lives in the SDK-free host leaf."""

from vv_agent._token_counting import _TOKEN_CACHE as _TOKEN_CACHE
from vv_agent._token_counting import _estimate_tokens as _estimate_tokens
from vv_agent._token_counting import count_messages_tokens as count_messages_tokens
from vv_agent._token_counting import count_tokens as count_tokens
from vv_agent._token_counting import incremental_token_counting_status as incremental_token_counting_status
from vv_agent._token_counting import logger as logger
from vv_agent._token_counting import resolve_model_token_limits as resolve_model_token_limits
