# Token counting for compaction

Compaction needs a quick estimate of context size before each model call.
`count_messages_tokens` therefore counts on the worker, even if vv-llm has a
global token server or the model has an enabled provider endpoint.

## Local methods

For families whose vv-llm counter can call a provider, compaction uses the
same fallback that vv-llm 0.8.4 uses when no endpoint is available:

| Model name starts with | Local text method | Incremental counting |
| --- | --- | --- |
| `moonshot`, `kimi` | tiktoken `cl100k_base` | With verified dependencies and safe text boundaries |
| `gemini` | tiktoken `cl100k_base` | Same |
| `stepfun` | tiktoken `cl100k_base` | Same |
| `glm` | tiktoken `cl100k_base` | Same |
| `claude` | tiktoken `o200k_base` | Same |
| `abab`, `MiniMax` | `int(len(text) / 1.33)` | Whole text; round once |
| `deepseek` | DeepSeek tokenizer | With verified dependencies and safe text boundaries |
| `qwen` | vv-llm's Qwen tokenizer selection | Whole text |
| Exact `gpt-3.5-turbo` | tiktoken `cl100k_base` | With verified dependencies and safe text boundaries |
| Other names | tiktoken `o200k_base` | Same |

The names select counting methods only. They do not change the model used for
generation. No provider credentials or endpoint settings are needed for these
fallbacks.

The formatter still comes from vv-llm. Text parts are joined with newlines.
Images keep vv-llm's existing fixed-size estimate using the original model
name; they are not downloaded. An explicit `tools` argument is counted as
`str(tools)` separately. MiniMax rounds the combined text once and the tools
text once. Non-text blocks that vv-llm does not count remain uncounted.
If local counting fails, the existing JSON character estimate remains the
last fallback.

## Reusing unchanged text

DeepSeek, cl100k and o200k cache each text part's plain count and its count
with a trailing newline. The cache holds at most 4096 entries per process.
Keys hold digests and counting identities; values hold integers. The cache
retains no message bodies.

The verified pairs are vv-llm `0.8.3` or `0.8.4` with DeepSeek tokenizer
`0.3.0`, or with tiktoken `0.14.0`. Versions are listed individually in
`memory/token_utils.py`. A newer dependency version counts the whole text
locally and logs one warning per encoding per process.

Empty text parts use a whole count. DeepSeek also uses a whole count if a
part starts with CR or LF. Tiktoken does so for any leading whitespace;
o200k also does so for a leading `/`. Those boundaries can combine with the
previous part's newline, so adding separate counts could be wrong.

Hosts can inspect eligibility and the selected local method:

```python
from vv_agent.memory import incremental_token_counting_status

enabled, method = incremental_token_counting_status("kimi-k3")
# With verified dependencies: (True, "verified cl100k_base newline boundaries")
enabled, method = incremental_token_counting_status("MiniMax-M2.5")
# (False, "local int(len(text) / 1.33); whole-conversation counting")
```

The boolean describes eligibility, not whether a particular input can use
the cache. Replaced counting functions disable incremental caching.
An explicitly replaced message counter remains responsible for its own
counting method and I/O.

## How to use the estimate

Local counts equal the specified fallback, not the provider's tokenizer or
billed usage. A different tokenizer can count either more or fewer tokens.
Anthropic's public token-counting guidance warns that tiktoken tends to
undercount Claude, with larger errors on code and non-English text.
That warning does not calibrate `o200k_base` for each Claude model. Hosts
using Claude should allow extra room rather than assume the fallback is
conservative. The guidance is in `anthropics/skills`,
`skills/claude-api/shared/token-counting.md`.
MiniMax's character formula also rounds down: a one-character text counts
as zero. The fixed image estimate and omitted request overhead are further
sources of error. No safety multiplier is applied.

Hosts should configure the real model context limit and leave room for
estimation error and new tool output. [Capacity metadata](model-settings.md#capacity-metadata)
explains the threshold, output reserve and buffer. A smaller threshold
compacts earlier; a larger buffer reduces the model-derived threshold.
Neither setting proves a bound on tokenizer error.

This policy applies to the shared message counter used by compaction and
context planning. `count_tokens` and direct calls to vv-llm retain their own
behavior. Provider usage returned after generation still supplies accounting.
