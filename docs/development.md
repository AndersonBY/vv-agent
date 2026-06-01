# Development

Run commands from the repository root.

## Environment

```bash
uv sync --dev
```

For real model calls, create a local key file from the checked-in template:

```bash
cp local_settings.example.py local_settings.py
```

Fill in real API keys in `local_settings.py`. Do not commit that file.

## Fast Checks

Use the narrowest check while iterating:

```bash
uv run pytest tests/test_config.py
uv run pytest tests/test_runner.py tests/test_public_agent.py tests/test_function_tool.py
uv run pytest tests/test_runtime.py
uv run pytest tests/test_tools.py
```

Run linting before finishing:

```bash
uv run ruff check
```

Run the full non-live test suite before reporting broad completion:

```bash
uv run pytest
```

Live tests are skipped unless explicitly enabled:

```bash
V_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live
```

Useful live-test environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `V_AGENT_LOCAL_SETTINGS` | `local_settings.py` | Settings file used by live tests and CLI. |
| `V_AGENT_LIVE_BACKEND` | `moonshot` | Provider backend for live smoke tests. |
| `V_AGENT_LIVE_MODEL` | `kimi-k2.6` | Model key for live smoke tests. |

## Test Ownership

| Change area | Primary tests |
| --- | --- |
| Settings/model resolution | `tests/test_config.py` |
| CLI | `tests/test_config.py`, CLI-specific assertions in existing tests |
| Runtime loop and statuses | `tests/test_runtime.py`, `tests/test_cycle_runner.py` |
| Public SDK contract | `tests/test_public_agent.py`, `tests/test_runner.py`, `tests/test_model_settings.py`, `tests/test_function_tool.py`, `tests/test_sessions.py`, `tests/test_run_events.py`, `tests/test_agent_as_tool.py`, `tests/test_handoffs.py`, `tests/test_tool_policy.py`, `tests/test_tool_approval.py`, `tests/test_guardrails.py`, `tests/test_tracing.py`, `tests/test_compiler.py` |
| Live handles and event replay | `tests/test_run_handle_live_stream.py`, `tests/test_event_store.py`, `tests/test_events_v1.py`, `tests/test_session_graph_events.py` |
| Provider contracts | `tests/test_approval_protocol.py`, `tests/test_context_providers.py`, `tests/test_memory_provider.py`, `tests/test_interactive_approval_bridge.py`, `tests/test_interactive_memory_provider_bridge.py` |
| Hooks | `tests/test_runtime_hooks.py` |
| Tools and schemas | `tests/test_tools.py`, `tests/test_tool_schemas.py`, `tests/test_tool_planner.py`, `tests/test_tool_orchestrator.py` |
| Memory and compaction | `tests/test_memory.py`, `tests/test_microcompact.py`, `tests/test_session_memory.py` |
| Execution backends | `tests/test_backends.py`, `tests/test_state_store.py` |
| Workspace backends | `tests/test_workspace_backends.py` |
| Live provider behavior | `tests/test_live_moonshot.py` |

## Change Hygiene

- Keep public API exports in `src/vv_agent/__init__.py` synchronized with new
  public types. New user-facing SDK concepts should be importable from
  `vv_agent`; do not add new public imports under `vv_agent.sdk`.
- Keep the runtime boundary explicit. Host integrations should implement
  `ApprovalProvider`, `ContextProvider`, `MemoryProvider`, `ToolExecutor` or
  `FunctionTool`, and `RunEventStore` instead of patching runner, compiler,
  memory, or tool-dispatch internals.
- Treat typed `RunEvent` objects as the app-state contract. Runtime log payloads
  can remain as compatibility fallbacks, but new host UI behavior should stream
  or replay events through `Runner.start()`, `RunHandle.events()`, and
  `RunEventStore`.
- Update README/examples when user-facing defaults, environment variables, or
  command examples change.
- Keep `local_settings.example.py` as the only checked-in settings template.
- Avoid adding compatibility shims that hide model, backend, or endpoint
  mismatches. Prefer clear errors and tests.
- When a change spans runtime, SDK, and tools, add focused tests at each public
  boundary rather than relying only on one integration test.
