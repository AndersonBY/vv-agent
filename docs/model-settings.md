# Model Settings

`vv-agent` reads provider configuration from a local settings file. The
checked-in template is `local_settings.example.py`. Put real credentials in
`local_settings.py` or in another untracked file, and point
`VV_AGENT_LOCAL_SETTINGS` at it.

## Settings shape

A `.py` settings file must assign `LLM_SETTINGS` a literal mapping. The loader
parses the file and reads that literal; it does not run the file as code.
`.json` and `.toml` files hold the same mapping at the top level.

At minimum, `LLM_SETTINGS` needs backends and endpoints:

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

`VERSION="2"`, a `backends` mapping, and an `endpoints` list are required. A
settings value without them raises `ConfigError`.

## Exact model resolution

Model keys are exact. `resolve_model_endpoint(settings, backend, model)` looks
up the requested key under `LLM_SETTINGS.backends.<backend>.models`.

Never rewrite one model key to another. `kimi-k2.5` and `kimi-k3` are
separate models. If only `kimi-k3` is configured, a request for `kimi-k2.5`
raises `ConfigError` instead of quietly using `kimi-k3`.
`tests/test_config.py` covers this.

`VvLlmModelProvider.resolve()` first uses an exact settings key within the
explicit or default backend. If that key is absent, it accepts an exact
resolved `model_id`, including the first endpoint's override, only when one
configured model matches in that backend. Without a default backend, only a
unique model ID across backends can be resolved by a named reference; settings
keys still require an explicit backend. This lets provider wrappers delegate
the session kernel's model-ID requests while preserving their own capability
overrides. Ambiguous IDs are rejected. `tests/test_model_provider.py` covers
both direct resolution and wrapped-provider session execution.

### Changing the model in a session

A session can switch models through a `model_selection` inbox input. The
selection carries the resolved provider model ID, an optional `ModelSettings` object, and a
`route` reserved for host transport choices such as `backend_type`. Reasoning
effort belongs in the settings, as `model_settings.reasoning.effort`, not in
the route. If the selection omits settings, the new model gets empty settings;
it does not inherit the previous model's settings. Capacity planning then uses
the selected model's context and output limits and the selected `max_tokens`.
See [session-model-selection.md](session-model-selection.md).

For every model request, selected or not, the session kernel overrides
`ModelSettings.retry` to a single transport attempt. The kernel does its own
retries as separate logged attempts, so every call the provider sees is
recorded in the session log.

## Kimi K3 request profile

`kimi-k3` always uses the provider's fixed reasoning and sampling profile. The
model bridge:

- sends top-level `reasoning_effort="max"`;
- omits `temperature`, `top_p`, the fixed penalty and count fields, and the
  provider `thinking` controls;
- sends public `max_tokens` as the provider's `max_completion_tokens` field.

The bridge applies these rules after merging the public `ModelSettings`, so an
`extra_body` value cannot override them. Other `extra_body` fields pass
through unchanged.

## Capacity metadata

A model's `context_length` and `max_output_tokens` describe what the model
can do. The compiler copies them into task metadata as `model_context_window`
and `model_max_output_tokens`. `model_max_output_tokens` is not a request
default, and it is not copied to `reserved_output_tokens`.

Memory planning reserves room for the model's reply. It takes the first value
that is set:

1. the effective `ModelSettings.max_tokens` for the request;
2. explicit task metadata `reserved_output_tokens`;
3. the framework fallback of `16000`, lowered to `model_max_output_tokens` if
   that is smaller.

Only the fallback is capped, so a limit the caller chose is never changed by
the model catalog.

`model_context_window` overrides the model's context only when it is
positive; zero or a negative value falls back to the resolved model. When the
context size is still unknown, the runtime derives one: the compaction
threshold (if positive, otherwise `250000`), plus the output reserve, plus the
`13000` auto-compaction buffer. With defaults this gives `279000`, so the
configured threshold is never silently lowered. If an explicit context is
smaller than the reserve plus the buffer, the remaining prompt capacity can be
zero.

The automatic compaction threshold is the smaller of the configured
`memory_compact_threshold` (default `250000`) and
`max(model_context_window - reserved_output_tokens - autocompact_buffer_tokens, 0)`.
A nonpositive configured threshold uses only the model-derived limit.
The default buffer is `13000`. Compaction starts when the local count
exceeds the threshold; a count equal to the threshold can still proceed.

For the `1048576` context in the settings example, the default `16000`
output reserve leaves a `250000` compaction threshold and `782576` tokens
between that threshold and the prompt capacity. A `128000` context instead
lowers the threshold to `99000`, leaving only the `13000` buffer. If context
size was derived because no real limit was known, that computed margin does
not prove the provider has that capacity.

These counts are local estimates. Different tokenizers, message overhead,
tools and a large new tool result can consume the margin. Hosts that need
more room can lower `memory_compact_threshold` or increase task metadata
`autocompact_buffer_tokens`. See [token counting](token-counting.md).

In multi-turn and tool-call requests, every assistant message keeps its full
`reasoning_content`. Streamed reasoning deltas are collected until the
provider stream ends, and only then is the message stored.

## Catalog capabilities in App Server

model/list uses resolved context_length and max_output_tokens and the catalog's
function_call_available and native_multimodal flags. Optional input_modalities,
reasoning_available, supported_reasoning_efforts, default_reasoning_effort,
prompt_cache_available and continuation_available describe only known provider
capabilities. Missing fields stay null on the wire; no support is inferred from
model names. These fields do not select a model or change request settings.

## Cache usage accounting

`vv-agent` requires `vv-llm` 0.8.5 or newer. For generic OpenAI-compatible
providers, an omitted cache reading stays unknown, and an explicit
`cached_tokens: 0` is an observed zero. Moonshot is the one exception. When a
cold completion omits both top-level `cached_tokens` and
`prompt_tokens_details`, `vv-llm` reports a zero cache read, as Moonshot's
documented response format specifies. An explicit
`null` or a malformed value stays unknown.

The runtime reads `prompt_tokens_details.cached_tokens` as `vv-llm` reports it
and does not rewrite the provider payload. For OpenAI-compatible usage,
uncached input is `prompt_tokens - cached_tokens`. Do not map that value onto
Anthropic's `cache_read_input_tokens`: Anthropic's base `input_tokens` already
means something different.

## Current user-facing defaults

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

## Key safety

- Do not commit real keys.
- Prefer placeholder values in checked-in templates.
- Do not read key files from sibling projects. Keep this repository's test and
  example settings self-contained.
- Live tests must stay opt-in through `VV_AGENT_RUN_LIVE_TESTS=1`.
