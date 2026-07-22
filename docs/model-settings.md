# Model Settings

`vv-agent` reads provider configuration from a local Python settings file. The
checked-in template is `local_settings.example.py`; real credentials belong in
`local_settings.py` or another untracked path selected with
`VV_AGENT_LOCAL_SETTINGS`.

## Settings Shape

At minimum, `LLM_SETTINGS` needs provider backends and endpoints:

```python
LLM_SETTINGS = {
    "VERSION": "2",
    "backends": {
        "moonshot": {
            "models": {
                "kimi-k3": {
                    "id": "kimi-k3",
                    "endpoints": [
                        {"endpoint_id": "moonshot-default", "model_id": "kimi-k3"},
                    ],
                    "context_length": 1048576,
                    "max_output_tokens": 131072,
                    "native_multimodal": True,
                    "function_call_available": True,
                    "response_format_available": True,
                }
            }
        }
    },
    "endpoints": [
        {
            "id": "moonshot-default",
            "api_key": "REPLACE_WITH_MOONSHOT_API_KEY",
            "api_base": "https://api.moonshot.cn/v1",
        }
    ],
}
```

`VERSION="2"`, `backends`, and `endpoints` are required. Every other top-level
shape is rejected.

## Exact Model Resolution

Model keys are exact. `resolve_model_endpoint(settings, backend, model)` looks
up the requested model key under `LLM_SETTINGS.backends.<backend>.models`.

Do not rewrite one model key to another. For example,
`kimi-k2.5` and `kimi-k3` are separate model ids. If only `kimi-k3` is
configured, requesting `kimi-k2.5` must raise `ConfigError` instead of silently
using a different model key.

This behavior is covered by `tests/test_config.py`.

## Kimi K3 Request Profile

`kimi-k3` always uses its provider-defined reasoning and sampling profile. The
LLM bridge sends top-level `reasoning_effort="max"`, omits `temperature`,
`top_p`, fixed penalty/count fields, and provider `thinking` controls, and
maps public `max_tokens` to the provider's `max_completion_tokens` field.
These invariants are applied after public `ModelSettings` are merged so
provider-specific `extra_body` values cannot override them. Unrelated
`extra_body` fields continue to pass through.

## Capacity Metadata

Resolved `context_length` and `max_output_tokens` are model capabilities. The
compiler projects them to task metadata as `model_context_window` and
`model_max_output_tokens`. In particular, `model_max_output_tokens` is not a
request default and is not copied to `reserved_output_tokens`.

Memory capacity reserves effective per-request `ModelSettings.max_tokens`
first, then explicit task metadata `reserved_output_tokens`, then `16000` as a
framework fallback. A smaller `model_max_output_tokens` may cap only that
fallback. This keeps model catalog capability separate from a caller-selected
request limit.

`model_context_window` must be positive to override resolved model capacity.
Zero or negative metadata falls through to the resolved model. When capacity
is still unknown, the runtime derives a planning context from the positive
configured compaction threshold (or `250000`) plus the selected output reserve
and the `13000` auto-compaction buffer. The default is therefore `279000` and
does not silently lower the configured threshold. A derived prompt capacity
may still be zero when an explicit positive context is exhausted by the
selected reserve and buffer.

For multi-turn and tool-call requests, every assistant message retains its
complete `reasoning_content`; streamed reasoning deltas are collected through
the end of the provider stream before that message is stored.

## Cache Usage Accounting

`vv-agent` 0.7.2 requires `vv-llm` 0.3.107 or newer. Generic
OpenAI-compatible providers still leave an omitted cache reading unknown, and
an explicit `cached_tokens: 0` remains an observed zero. Moonshot is the one
provider-specific exception: when a cold completion omits both top-level
`cached_tokens` and `prompt_tokens_details`, `vv-llm` projects a zero cache read
from Moonshot's documented response contract. Explicit `null` or malformed
values remain unknown.

The Agent runtime consumes the projected `prompt_tokens_details.cached_tokens`
without rewriting the provider payload. For OpenAI-compatible usage,
uncached input is `prompt_tokens - cached_tokens`; do not map that value onto
Anthropic's `cache_read_input_tokens`, whose base `input_tokens` has a different
meaning.

## Current User-Facing Defaults

| Surface | Default |
| --- | --- |
| CLI `--backend` | `moonshot` |
| CLI `--model` | `kimi-k3` |
| Examples `VV_AGENT_EXAMPLE_BACKEND` | `moonshot` |
| Examples `VV_AGENT_EXAMPLE_MODEL` | `kimi-k3` |
| Live tests `VV_AGENT_LIVE_BACKEND` | `moonshot` |
| Live tests `VV_AGENT_LIVE_MODEL` | `kimi-k3` |

When changing a default, update all user-facing surfaces together: CLI, README,
examples, live-test docs, tests, and `local_settings.example.py`.

## Key Safety

- Do not commit real keys.
- Prefer placeholder values in checked-in templates.
- Do not read key files from sibling projects. Keep this repository's test and
  example settings self-contained.
- Live tests must stay opt-in through `VV_AGENT_RUN_LIVE_TESTS=1`.
