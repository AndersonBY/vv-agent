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
uv run pytest tests/test_sdk_client.py
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
| Hooks | `tests/test_runtime_hooks.py` |
| Tools and schemas | `tests/test_tools.py`, `tests/test_tool_schemas.py`, `tests/test_tool_planner.py` |
| Memory and compaction | `tests/test_memory.py`, `tests/test_microcompact.py`, `tests/test_session_memory.py` |
| SDK client/session | `tests/test_sdk_client.py`, `tests/test_sdk_session.py`, `tests/test_sdk_resources.py` |
| Execution backends | `tests/test_backends.py`, `tests/test_state_store.py` |
| Workspace backends | `tests/test_workspace_backends.py` |
| Live provider behavior | `tests/test_live_moonshot.py`, `tests/test_live_background_command.py` |

## Change Hygiene

- Keep public API exports in `src/vv_agent/__init__.py` synchronized with new
  public types.
- Update README/examples when user-facing defaults, environment variables, or
  command examples change.
- Keep `local_settings.example.py` as the only checked-in settings template.
- Avoid adding compatibility shims that hide model, backend, or endpoint
  mismatches. Prefer clear errors and tests.
- When a change spans runtime, SDK, and tools, add focused tests at each public
  boundary rather than relying only on one integration test.
