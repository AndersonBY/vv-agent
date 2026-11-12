"""Real prepared encoders share the production algorithm without replay initialization."""

import json
import subprocess
import sys
import textwrap
from functools import wraps
from pathlib import Path

import pytest
from support.token_counting import PROVIDER_MODELS
from vv_llm.chat_clients import utils

from vv_agent.kernel._message_values import Message, ToolResultCursor
from vv_agent.kernel._tool_values import ToolExecutionResult
from vv_agent.memory import token_utils
from vv_agent.memory.prepared_tokens import prepare_token_counter

MODELS = (
    "",
    "deepseek-v4-pro",
    "deepseek-flash",
    "gpt-3.5-turbo",
    "gpt-4o",
    "o1-mini",
    "o3-mini",
    "gpt-5.4",
    "unknown-model",
    "qwen-plus",
    "qwen3.6-27b",
    *PROVIDER_MODELS,
)


def production_messages():
    """Exercise every content/image branch of the actual closed Message projection."""
    messages = []
    for role in ("system", "user", "assistant", "tool"):
        for content in ("", "事实😀 e\u0301\n"):
            for image in (
                None,
                "",
                "https://image.invalid/a.png",
                "data:image/png;base64,AA==",
                "absent.png",
                "file:///absent.png",
                "http://image.invalid/a.png",
            ):
                for calls in ([], [{"id": "call", "type": "function", "function": {"name": "读", "arguments": "{}"}}]):
                    wire = {
                        "role": role,
                        "content": content,
                        "tool_calls": calls,
                        "reasoning_content": "reasoning",
                        "tool_call_id": "call",
                        "name": "name",
                        "metadata": {"content_type": "TXT", "attachments": ["must-not-open.png"]},
                        "artifact_ref": {
                            "path": ".vv-agent/artifacts/task/body.txt",
                            "media_type": "text/plain",
                            "encoding": "utf-8",
                            "size_bytes": 10,
                            "sha256": "0" * 64,
                        },
                    }
                    if image is not None:
                        wire["image_url"] = image
                    message = Message.from_dict(wire)
                    payload = message.to_openai_message()
                    assert "content_type" not in payload and "attachments" not in payload
                    assert "metadata" not in payload and "artifact_ref" not in payload
                    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
                    messages.append(message)
    messages.append(
        ToolExecutionResult(
            "call",
            "恢复😀",
            truncated=True,
            truncation_reason="read_limit",
            original_bytes=20,
            visible_bytes=10,
            cursor=ToolResultCursor("read_file", "恢复.txt", 1, "0" * 64),
        ).to_tool_message()
    )
    return messages


@pytest.fixture(autouse=True)
def clear_count_cache():
    token_utils._TOKEN_CACHE.clear()
    yield
    token_utils._TOKEN_CACHE.clear()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("native_multimodal", [False, True])
def test_prepared_matches_current_production_messages(model, native_multimodal):
    messages = [message.to_openai_message() for message in production_messages()]
    counter = prepare_token_counter(model)
    for tools in (None, [], [{"type": "function", "function": {"name": "读", "parameters": {"type": "object"}}}]):
        assert counter(messages, model=model, tools=tools, native_multimodal=native_multimodal) == (
            token_utils.count_messages_tokens(messages, model=model, tools=tools, native_multimodal=native_multimodal)
        )
    assert counter([], tools=["ignored"]) == 0
    for boundary in ("", "\n", "\r\nleading", " \t", "/", "\u2028", "<|endoftext|>", "old \ud800 中文"):
        payload = [Message("user", text).to_openai_message() for text in ("hello!\n", boundary, "world")]
        assert counter(payload, native_multimodal=native_multimodal) == token_utils.count_messages_tokens(
            payload,
            model=model,
            native_multimodal=native_multimodal,
        )


@pytest.mark.parametrize("first_qwen", ["qwen-plus", "qwen3.6-27b"])
def test_no_initialization_io_after_preparation_in_fresh_process(tmp_path, first_qwen):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import sys
        from pathlib import Path

        root = Path(sys.argv[1])
        sys.path[:0] = [str(root / "src"), str(root / "tests")]
        active = False
        class ForbiddenIO(BaseException):
            pass
        def audit(event, args):
            if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                raise ForbiddenIO(event)
            if event == "open" and isinstance(args[0], (str, bytes)):
                name = str(args[0])
                if (name.endswith("local_settings.py") or "/.env" in name
                        or ("/credentials" in name and "/site-packages/" not in name)):
                    raise ForbiddenIO(name)
            if active and (event == "open" or event in {"os.listdir", "os.scandir", "subprocess.Popen", "socket.__new__"}):
                raise ForbiddenIO(event)
        sys.addaudithook(audit)
        from test_prepared_tokens import MODELS, production_messages
        from vv_agent.memory import token_utils
        from vv_agent.memory.prepared_tokens import prepare_token_counter
        from vv_llm.chat_clients import utils
        from vv_llm.settings import Server
        import tiktoken.registry
        import threading

        import json
        values = production_messages()
        tools = [{"type": "function", "function": {"name": "读", "parameters": {"type": "object"}}}]
        if sys.argv[3] == "baseline":
            utils._get_qwen_tokenizer(sys.argv[2])
            observed = [token_utils.count_messages_tokens(
                [message.to_openai_message() for message in values],
                model=model, tools=tools, native_multimodal=native,
            ) for model in MODELS for native in (False, True)]
            observed.extend(token_utils.count_messages_tokens(
                [{"role": "user", "content": "unseen <|endoftext|> 中文"}], model=model,
            ) for model in MODELS)
            print(json.dumps(observed))
            raise SystemExit(0)

        first = prepare_token_counter(sys.argv[2])
        selected = utils._qwen_tokenizer_cache["qwen"]
        assert first.cache_identity[3].__self__ is selected
        counters = [prepare_token_counter(model) for model in MODELS]
        assert all(c.cache_identity[3].__self__ is selected for c in counters if c.family == "qwen")
        observed = []
        def forbidden(*args, **kwargs):
            raise ForbiddenIO("initializer/metadata/log/provider/media access during prepared counting")
        for name in ("get_gpt_35_encoding", "get_gpt_4o_encoding", "_get_qwen_tokenizer",
                     "get_token_counts", "get_message_token_counts", "format_messages",
                     "format_image_message", "format_text_message", "format_workflow_messages",
                     "Anthropic", "AnthropicVertex", "get_token_with_cache"):
            setattr(utils, name, forbidden)
        utils.gpt_35_encoding = utils.gpt_4o_encoding = None
        utils._qwen_tokenizer_cache.clear()
        tiktoken.registry.ENCODINGS.clear()
        for name in ("_deepseek_versions", "_tiktoken_versions", "incremental_token_counting_status", "version"):
            setattr(token_utils, name, forbidden)
        token_utils.logger.warning = token_utils.logger.debug = forbidden
        threading.Thread.start = forbidden
        type(utils.settings).get_backend = type(utils.settings).get_endpoint = forbidden
        utils.settings.token_server = Server(host="invalid", port=1, url="https://invalid")
        token_utils._TOKEN_CACHE.clear()
        active = True
        try:
            for i, counter in enumerate(counters):
                for native in (False, True):
                    observed.append(counter([message.to_openai_message() for message in values],
                                            model=MODELS[i], tools=tools, native_multimodal=native))
            observed.extend(counter([{"role": "user", "content": "unseen <|endoftext|> 中文"}])
                            for counter in counters)
        finally:
            active = False
        print(json.dumps(observed))
    """)
    outputs = []
    # The fenced child runs first; an independent child computes the reference.
    # No tokenizer or formatter count runs before the prepared child's fence.
    for mode in ("prepared", "baseline"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(root), first_qwen, mode],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 0, mode + result.stdout + result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


def test_transient_prepare_failure_is_not_cached(monkeypatch):
    original = utils.get_gpt_4o_encoding
    with monkeypatch.context() as patch:
        patch.setattr(utils, "get_gpt_4o_encoding", lambda: (_ for _ in ()).throw(OSError("not ready")))
        with pytest.raises(OSError, match="not ready"):
            prepare_token_counter("gpt-4o")
    counter = prepare_token_counter("gpt-4o")
    assert counter.cache_identity[3].__self__ is original()
    assert counter([Message("user", "ready").to_openai_message()]) == token_utils.count_messages_tokens(
        [Message("user", "ready").to_openai_message()],
        model="gpt-4o",
    )


def test_single_count_failure_falls_back_then_recovers(monkeypatch):
    encoding = utils.get_gpt_4o_encoding()
    original = encoding.encode
    messages = [Message("user", "hello " * 200).to_openai_message()]
    expected = token_utils.count_messages_tokens(messages, model="gpt-4o")
    attempts = []

    def flaky(text):
        attempts.append(text)
        if len(attempts) == 1:
            raise ValueError("temporary count error")
        return original(text)

    monkeypatch.setattr(encoding, "encode", flaky)
    counter = prepare_token_counter("gpt-4o")
    assert counter(messages) == token_utils._estimate_tokens(json.dumps(messages, ensure_ascii=False))
    assert counter(messages) == expected
    assert len(attempts) == 3


@pytest.mark.parametrize("name", ["get_token_counts", "get_message_token_counts", "format_messages"])
def test_custom_counter_rejection_is_only_at_new_opt_in_seam(monkeypatch, name):
    original = getattr(utils, name)

    @wraps(original)
    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(utils, name, wrapper)
    with pytest.raises(ValueError, match="prepared equivalent"):
        prepare_token_counter("gpt-4o")
    messages = [Message("user", "legacy").to_openai_message()]
    assert token_utils.count_messages_tokens(messages, model="gpt-4o") >= 0


def test_unverified_versions_keep_whole_count_and_warn_at_preparation(monkeypatch, caplog):
    monkeypatch.setattr(token_utils, "_tiktoken_versions", lambda: ("future", "0.14.0"))
    monkeypatch.setattr(token_utils, "_VERSION_FALLBACK_LOGGED", set())
    first = prepare_token_counter("gpt-4o")
    second = prepare_token_counter("o3-mini")
    assert not first.incremental_status[0] and not second.incremental_status[0]
    assert len(caplog.records) == 1
    caplog.clear()
    monkeypatch.setattr(token_utils, "_tiktoken_versions", lambda: (_ for _ in ()).throw(AssertionError("metadata at count")))
    messages = [Message("user", text).to_openai_message() for text in ("hello!\n", " / unsafe", "world")]
    assert first(messages) == len(utils.get_gpt_4o_encoding().encode("hello!\n\n / unsafe\nworld"))
    assert not caplog.records


def test_model_binding_requires_outer_assembly():
    counter = prepare_token_counter("gpt-4o")
    with pytest.raises(ValueError, match="model does not match"):
        counter([], model="qwen-plus")


def test_bound_cache_keeps_identity_and_digest_only(monkeypatch):
    counter = prepare_token_counter("gpt-4o")
    messages = [Message("user", "identity").to_openai_message()]
    first = counter(messages)
    encoding = utils.get_gpt_4o_encoding()
    original = encoding.encode
    monkeypatch.setattr(encoding, "encode", lambda text: [*original(text), 1])
    changed = prepare_token_counter("gpt-4o")
    assert counter(messages) == first
    assert changed(messages) == first + 1
    assert changed.cache_identity != counter.cache_identity
    monkeypatch.setattr(token_utils, "_TOKEN_CACHE_LIMIT", 4)
    for index in range(8):
        counter([Message("user", f"new {index}").to_openai_message()])
    assert len(token_utils._TOKEN_CACHE) == 4
    assert all(isinstance(n, int) for counts in token_utils._TOKEN_CACHE.values() for n in counts)
