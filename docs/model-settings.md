# Model Settings

`vv-agent` reads provider configuration from a local Python settings file. The
checked-in template is `local_settings.example.py`; real credentials belong in
`local_settings.py` or another untracked path selected with
`V_AGENT_LOCAL_SETTINGS`.

## Settings Shape

At minimum, `LLM_SETTINGS` needs provider backends and endpoints:

```python
LLM_SETTINGS = {
    "backends": {
        "moonshot": {
            "models": {
                "kimi-k2.6": {
                    "id": "kimi-k2.6",
                    "endpoints": [
                        {"endpoint_id": "moonshot-default", "model_id": "kimi-k2.6"},
                    ],
                    "context_length": 128000,
                    "max_output_tokens": 16384,
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

`load_llm_settings_from_file()` also accepts templates that wrap the schema under
`{"LLM_SETTINGS": {...}}` when the nested value contains both backends/providers
and endpoints.

## Exact Model Resolution

Model keys are exact. `resolve_model_endpoint(settings, backend, model)` looks
up the requested model key under `LLM_SETTINGS.backends.<backend>.models`.

Do not add aliases between independent provider models. For example,
`kimi-k2.5` and `kimi-k2.6` are separate model ids. If only `kimi-k2.6` is
configured, requesting `kimi-k2.5` must raise `ConfigError` instead of silently
using `kimi-k2.6` or an older `kimi-k2-thinking` key.

This behavior is covered by `tests/test_config.py`.

## Kimi K3 Request Profile

`kimi-k3` always uses its provider-defined reasoning and sampling profile. The
LLM bridge sends top-level `reasoning_effort="max"`, omits `temperature`,
`top_p`, fixed penalty/count fields, and legacy K2.x `thinking` controls, and
maps public `max_tokens` to the provider's `max_completion_tokens` field.
These invariants are applied after public `ModelSettings` are merged so
provider-specific `extra_body` values cannot override them. Unrelated
`extra_body` fields continue to pass through.

For multi-turn and tool-call requests, every assistant message retains its
complete `reasoning_content`; streamed reasoning deltas are collected through
the end of the provider stream before that message is stored.

## Current User-Facing Defaults

| Surface | Default |
| --- | --- |
| CLI `--backend` | `moonshot` |
| CLI `--model` | `kimi-k2.6` |
| Examples `V_AGENT_EXAMPLE_BACKEND` | `moonshot` |
| Examples `V_AGENT_EXAMPLE_MODEL` | `kimi-k2.6` |
| Live tests `V_AGENT_LIVE_BACKEND` | `moonshot` |
| Live tests `V_AGENT_LIVE_MODEL` | `kimi-k2.6` |

When changing a default, update all user-facing surfaces together: CLI, README,
examples, live-test docs, tests, and `local_settings.example.py`.

## Key Safety

- Do not commit real keys.
- Prefer placeholder values in checked-in templates.
- Do not read key files from sibling projects. Keep this repository's test and
  example settings self-contained.
- Live tests must stay opt-in through `V_AGENT_RUN_LIVE_TESTS=1`.
