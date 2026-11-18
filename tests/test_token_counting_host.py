"""The actual local counter is usable without importing the SDK memory package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r"""
import importlib.abc
import json
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
leaf = sys.argv[2] == 'leaf'
class ForbiddenSDKImport(BaseException): pass
class CounterHostOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'vv_agent' or fullname.startswith('vv_agent.')) and not (
            fullname in {'vv_agent', 'vv_agent.kernel', 'vv_agent._token_counting'}
            or fullname.startswith('vv_agent.kernel.')
        ):
            raise ForbiddenSDKImport(fullname)
if leaf:
    sys.meta_path.insert(0, CounterHostOnly())
class ForbiddenHostIO(BaseException): pass
def audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'subprocess.Popen'}:
        raise ForbiddenHostIO(event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = Path(os.fsdecode(args[0])).name
        if name == 'local_settings.py' or name.startswith('.env') or name == 'credentials':
            raise ForbiddenHostIO(name)
sys.addaudithook(audit)
if leaf:
    from vv_agent import _token_counting as counting
    drain = counting._drain_counts
    assert 'vv_llm' not in sys.modules and 'tiktoken' not in sys.modules
    assert counting.count_tokens('') == counting.count_messages_tokens([]) == 0
    assert 'vv_llm' not in sys.modules and 'tiktoken' not in sys.modules
else:
    from vv_agent.memory import token_utils as counting
    from vv_agent.memory._counting import _drain_counts as drain
from vv_agent.kernel._token_count import TokenCount
assert not counting._TOKEN_CACHE
messages = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': '世界 facts.'}]
calls = []
def profile(frame, event, arg):
    if (event == 'call' and frame.f_globals.get('__name__') == 'vv_llm.chat_clients.utils'
            and frame.f_code.co_name == 'get_token_counts'):
        calls.append([frame.f_locals['text'], frame.f_locals['model'], frame.f_locals['use_token_server_first']])
sys.setprofile(profile)
try:
    first = counting.count_messages_tokens(messages, model='gpt-4o')
    first_calls, first_cache = list(calls), len(counting._TOKEN_CACHE)
    calls.clear()
    repeated = counting.count_messages_tokens(messages, model='gpt-4o')
    repeat_calls, repeat_cache = list(calls), len(counting._TOKEN_CACHE)
    calls.clear()
    unsafe = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': ' leading whitespace'}]
    boundary_fallback = counting.count_messages_tokens(unsafe, model='gpt-4o')
    boundary_calls = list(calls)
finally:
    sys.setprofile(None)
from vv_llm.chat_clients import utils
assert utils.gpt_4o_encoding is not None and not utils._qwen_tokenizer_cache
assert first == repeated == len(utils.gpt_4o_encoding.encode('hello\n世界 facts.'))
assert len(first_calls) == 4 and first_cache == repeat_cache == 2 and repeat_calls == []
assert boundary_fallback == len(utils.gpt_4o_encoding.encode('hello\n leading whitespace'))
assert boundary_calls == [['hello\n leading whitespace', 'gpt-4o', False]]
closed = []
def work():
    try:
        return (yield TokenCount(messages, 'gpt-4o'))
    finally:
        closed.append(True)
assert drain(work(), lambda request: counting.count_messages_tokens(request.messages, model=request.model)) == first
assert closed == [True]
original = utils.get_message_token_counts
custom_calls = []
def custom(**kwargs):
    custom_calls.append(kwargs)
    return 37
utils.get_message_token_counts = custom
try:
    custom_result = counting.count_messages_tokens(messages, model='gpt-4o', tools=[], native_multimodal=False)
    assert custom_result == 37 and custom_calls == [
        {'messages': messages, 'model': 'gpt-4o', 'tools': [], 'native_multimodal': False}]
finally:
    utils.get_message_token_counts = original
logs = []
class Capture(logging.Handler):
    def emit(self, record):
        logs.append([record.name, record.levelname, record.getMessage()])
logger = logging.getLogger('vv_agent.memory.token_utils')
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(Capture())
def unavailable(**kwargs):
    raise OSError('local test counter unavailable')
utils.get_message_token_counts = unavailable
try:
    fallback = counting.count_messages_tokens(messages, model='gpt-4o')
    fallback_cache = len(counting._TOKEN_CACHE)
    fallback_repeat = counting.count_messages_tokens(messages, model='gpt-4o')
    assert fallback == fallback_repeat == counting._estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str))
    assert fallback_cache == len(counting._TOKEN_CACHE) == 4
finally:
    utils.get_message_token_counts = original
assert logs == [['vv_agent.memory.token_utils', 'DEBUG', 'vv-llm message token counting unavailable; using local estimate']] * 2
if leaf:
    assert all(name in {'vv_agent', 'vv_agent.kernel', 'vv_agent._token_counting'} or name.startswith('vv_agent.kernel.')
               for name in sys.modules if name.startswith('vv_agent'))
print(json.dumps({'first': first, 'first_calls': first_calls, 'repeat': repeated, 'repeat_calls': repeat_calls,
                  'cache_entries': first_cache, 'boundary_fallback': boundary_fallback, 'boundary_calls': boundary_calls,
                  'custom': custom_result, 'fallback': fallback, 'fallback_repeat': fallback_repeat,
                  'fallback_cache_entries': fallback_cache, 'logs': logs, 'drain_closed': closed}, ensure_ascii=False))
"""


def run_probe(mode, cwd):
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROBE, str(ROOT), mode],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_real_host_counter_matches_sdk_without_other_sdk_imports(tmp_path):
    assert run_probe("leaf", tmp_path) == run_probe("sdk", tmp_path)


def test_existing_exports_and_sdk_consumers_share_actual_owner():
    import inspect
    import logging
    from typing import get_args, get_type_hints

    from vv_agent import _token_counting as owner
    from vv_agent.kernel._token_count import TokenCount
    from vv_agent.memory import _counting, incremental_token_counting_status, manager, microcompact, session_memory, token_utils
    from vv_agent.session import compaction, context_view, reducer

    for name in (
        "count_tokens",
        "count_messages_tokens",
        "incremental_token_counting_status",
        "resolve_model_token_limits",
        "_estimate_tokens",
        "_TOKEN_CACHE",
        "logger",
    ):
        assert getattr(token_utils, name) is getattr(owner, name)
    assert token_utils is not owner
    assert incremental_token_counting_status is owner.incremental_token_counting_status
    assert (
        manager.count_messages_tokens
        is compaction.count_messages_tokens
        is reducer.count_messages_tokens
        is owner.count_messages_tokens
    )
    assert manager.resolve_model_token_limits is owner.resolve_model_token_limits
    assert session_memory.count_tokens is owner.count_tokens
    assert context_view._estimate_tokens is owner._estimate_tokens
    assert (
        _counting._drain_counts
        is manager._drain_counts
        is microcompact._drain_counts
        is reducer._drain_counts
        is owner._drain_counts
    )
    assert owner.logger is logging.getLogger("vv_agent.memory.token_utils")
    expected_returns = {
        "count_tokens": int,
        "count_messages_tokens": int,
        "incremental_token_counting_status": tuple[bool, str],
        "resolve_model_token_limits": tuple[int | None, int | None],
    }
    for name, returned in expected_returns.items():
        assert get_type_hints(getattr(token_utils, name))["return"] == returned
    assert str(inspect.signature(token_utils.count_messages_tokens)) == (
        "(messages: 'list[dict[str, Any]]', *, model: 'str' = '', tools: 'Any' = None, native_multimodal: 'bool' = True) -> 'int'"
    )
    assert get_args(get_type_hints(_counting._drain_counts)["steps"])[:2] == (TokenCount, int)
