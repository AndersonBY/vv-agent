# Examples

[中文说明](README_ZH.md)

These scripts are self-contained and runnable from the repository root. Public
SDK examples use `Agent`, `Runner`, `RunConfig`, `ModelSettings`,
`function_tool`, sessions, handoffs, typed events, and tool policy. Low-level
runtime examples are kept separately for backend integration work.

For host-product migrations, prefer provider and executor extension points over
runtime patches: `ApprovalProvider` for UI/rules, `ContextProvider` for product
prompt fragments, `vv_agent.memory.MemoryProvider` for product persistence,
`vv_agent.tools.ToolExecutor` or `FunctionTool` groups for product tools, and
`RunEventStore` for app history. Use `Runner.start()` when the host needs live
typed events plus cancellation or approval control through `RunHandle`.

## Common Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `V_AGENT_LOCAL_SETTINGS` | `local_settings.py` | LLM backend settings file |
| `V_AGENT_EXAMPLE_WORKSPACE` | `./workspace` | Workspace directory |
| `V_AGENT_EXAMPLE_BACKEND` | `moonshot` | LLM backend |
| `V_AGENT_EXAMPLE_MODEL` | `kimi-k2.6` | Model name |
| `V_AGENT_EXAMPLE_PROMPT` | example-specific | Prompt override |
| `V_AGENT_EXAMPLE_TOKEN_BUDGET` | `4000` | Total token limit for example 07 |
| `V_AGENT_EXAMPLE_TOOL_BUDGET` | `12` | Total tool-call limit for example 07 |

## Public SDK

| # | File | Shows |
| --- | --- | --- |
| 01 | `01_quick_start.py` | Minimal `Agent` + `Runner.run_sync` |
| 02 | `02_agent_profiles.py` | Reusable agent profiles and `ModelSettings` |
| 03 | `03_sdk_client.py` | Runner configuration, sessions, custom tool, typed events |
| 04 | `04_session_api.py` | `MemorySession` across multiple runs |
| 05 | `05_ask_user_resume.py` | Tool approval and `WAIT_USER` events |
| 06 | `06_runtime_hooks.py` | Low-level hooks passed through `RunConfig.hooks` |
| 07 | `07_token_budget_guard.py` | Public token and tool-call run budgets |
| 08 | `08_custom_tool.py` | `@function_tool` schema inference and structured output |
| 09 | `09_resource_loader.py` | JSON-backed agent profile loading |
| 10 | `10_read_image.py` | Image metadata through a typed tool |
| 11 | `11_sub_agent_pipeline.py` | `agent.as_tool()` and `handoff()` composition |
| 12 | `12_skill_activation.py` | Skill availability through agent metadata |
| 13 | `13_arxiv_pipeline.py` | Paper-search pipeline with function tools |
| 14 | `14_batch_sub_tasks.py` | Batch-style coordination with agent-as-tool |
| 15 | `15_memory_compact_hook.py` | Memory compaction audit hook |
| 16 | `16_hook_composition.py` | Multiple runtime hooks in one run |
| 17 | `17_error_recovery.py` | Retry wrapper around `Runner.run_sync` |

```bash
uv run python examples/01_quick_start.py
V_AGENT_EXAMPLE_PROFILE=translator uv run python examples/02_agent_profiles.py
V_AGENT_EXAMPLE_SESSION_ID=demo uv run python examples/04_session_api.py
uv run python examples/08_custom_tool.py
uv run python examples/11_sub_agent_pipeline.py
uv run python examples/17_error_recovery.py
```

## Runtime Integration

These examples use lower-level runtime APIs for cancellation, streaming,
threaded execution, checkpointing, Celery dispatch, and workspace backends.

| # | File | Shows |
| --- | --- | --- |
| 18 | `18_cancellation.py` | `CancellationToken` with a running task |
| 19 | `19_streaming.py` | Raw runtime stream callback events |
| 20 | `20_thread_backend.py` | `ThreadBackend` submit/future execution |
| 21 | `21_state_checkpoint.py` | `SqliteStateStore` checkpoints |
| 22 | `22_sdk_advanced.py` | Public SDK with streaming and `ThreadBackend` |
| 23 | `23_celery_backend.py` | `CeleryBackend` distributed cycles |
| 24 | `24_workspace_backends.py` | Local, memory, S3, and custom workspace backends |
| 25 | `25_temporary_tool_injection.py` | Run-scoped tool enablement |

```bash
V_AGENT_EXAMPLE_TIMEOUT=10 uv run python examples/18_cancellation.py
uv run python examples/19_streaming.py
uv run python examples/20_thread_backend.py
V_AGENT_EXAMPLE_DB=./workspace/agent.db uv run python examples/21_state_checkpoint.py
uv run python examples/22_sdk_advanced.py
uv run python examples/24_workspace_backends.py
```

## App Server Integration

These examples use the JSON-RPC App Server boundary for product hosts,
subprocess clients, lifecycle replay, notification filtering, schema export,
and overload handling. Examples 26-28 call a real configured model through
`local_settings.py` and `V_AGENT_EXAMPLE_*` environment variables. Example 29
is a local protocol/schema/backpressure example and does not call a model.

| # | File | Shows |
| --- | --- | --- |
| 26 | `26_app_server_channel_lifecycle.py` | Real model turn through in-process `ChannelTransport`: initialize/initialized, thread, turn, list, archive |
| 27 | `27_app_server_stdio_client.py` | Real model turn through a stdio JSONL subprocess client with the initialized handshake |
| 28 | `28_app_server_notification_opt_out.py` | Real model turn with two clients and per-connection notification opt-out |
| 29 | `29_app_server_schema_backpressure.py` | Typed JSON Schema, self-contained TypeScript, runtime schema/export, and overload handling |

```bash
uv run python examples/26_app_server_channel_lifecycle.py
uv run python examples/27_app_server_stdio_client.py
uv run python examples/28_app_server_notification_opt_out.py
uv run python examples/29_app_server_schema_backpressure.py
```
