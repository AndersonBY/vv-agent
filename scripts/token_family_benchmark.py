"""Measure local counts and loopback HTTP counting floors on the step-cost workload."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_cost_benchmark import heavy_lock, live_history, text_payload
from vv_llm.chat_clients import utils

from vv_agent.memory import token_utils

MODELS = [
    "deepseek-flash",
    "gpt-3.5-turbo",
    "gpt-4o",
    "o1-mini",
    "o3-mini",
    "gpt-5.4",
    "unknown-model",
    "qwen-plus",
    "qwen3.6-27b",
    "MiniMax-M2",
    "abab",
    "moonshot-v1",
    "kimi-k3",
    "gemini-3-pro",
    "claude-sonnet-4",
    "stepfun",
    "glm-4-plus",
]
REMOTE = ["MiniMax-M2", "abab", "moonshot-v1", "kimi-k3", "gemini-3-pro", "claude-sonnet-4", "stepfun", "glm-4-plus"]


class CounterHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, int]]] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        json.loads(body)
        self.requests.append((self.path, len(body)))
        response = json.dumps(
            {
                "total_tokens": 12345,
                "input_tokens": 12345,
                "segments_num": 12345,
                "data": {"total_tokens": 12345},
                "totalTokens": 12345,
                "usage": {"prompt_tokens": 12345},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        pass


def measure(model, count, steps):
    token_utils._TOKEN_CACHE.clear()
    utils._qwen_tokenizer_cache.clear()
    messages = [m.to_dict() for m in live_history()]
    begin = time.perf_counter()
    count(messages, model=model)
    initial_ms = (time.perf_counter() - begin) * 1000
    samples, tokens = [], []
    request_start = len(CounterHandler.requests)
    for step in range(steps):
        call = f"new-{step}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"Read step {step}.",
                    "tool_calls": [{"id": call, "type": "function", "function": {"name": "payload", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": call, "content": text_payload((30 + step % 3 * 30) * 1024, step + 100)},
            ]
        )
        begin = time.perf_counter()
        tokens.append(count(messages, model=model))
        samples.append((time.perf_counter() - begin) * 1000)
    requests = CounterHandler.requests[request_start:]
    return {
        "initial_ms": initial_ms,
        "step_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "tokens": tokens,
        "requests_per_step": len(requests) / steps,
        "request_bytes": [size for _, size in requests],
        "paths": sorted({path for path, _ in requests}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    results = {
        "versions": {name: version(name) for name in ("vv-llm", "deepseek-tokenizer", "tiktoken", "qwen-tokenizer")},
        "local": {},
        "loopback": {},
    }
    endpoint = SimpleNamespace(
        enabled=True,
        api_key="disposable-loopback",
        api_base="",
        proxy=None,
        is_bedrock=False,
        is_vertex=False,
        endpoint_type="anthropic",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), CounterHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    base = f"http://127.0.0.1:{server.server_port}"
    endpoint.api_base = base
    original_post = utils.httpx2.post

    def loopback_post(*args, **kwargs):
        # MiniMax hardcodes its external URL. Redirect only the transport to loopback.
        if args:
            args = (base + urlsplit(args[0]).path, *args[1:])
        else:
            kwargs["url"] = base + urlsplit(kwargs["url"]).path
        return original_post(*args, **kwargs)

    def settings_for(model, remote=False, global_server=False):
        backend = SimpleNamespace(models=defaultdict(lambda: SimpleNamespace(id=model, endpoints=["local"] if remote else [])))
        return SimpleNamespace(
            token_server=SimpleNamespace(url=base) if global_server else None,
            get_backend=lambda _: backend,
            get_endpoint=lambda _: endpoint,
        )

    try:
        thread.start()
        with heavy_lock(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Exclusive heavy.lock acquired.", flush=True)
            for model in MODELS:
                with patch.object(utils, "settings", settings_for(model)):
                    whole = measure(model, utils.get_message_token_counts, args.steps)
                    incremental = measure(model, token_utils.count_messages_tokens, args.steps)
                    assert whole["tokens"] == incremental["tokens"], model
                    results["local"][model] = {"whole": whole, "current": incremental}
                    print(model, whole["mean_ms"], incremental["mean_ms"], flush=True)
            for model in [*REMOTE, "global-server"]:
                with (
                    patch.object(utils, "settings", settings_for(model, remote=True, global_server=model == "global-server")),
                    patch.object(utils.httpx2, "post", loopback_post),
                ):
                    result = measure(model, token_utils.count_messages_tokens, args.steps)
                    assert result["requests_per_step"] == 1, (model, result)
                    results["loopback"][model] = result
                    print("loopback", model, result["mean_ms"], flush=True)
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
