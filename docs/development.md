# Development

Run every command from the repository root.

## Environment

Install the package and its development dependencies:

```bash
uv sync --dev
```

CI also installs the `s3` extra (`uv sync --dev --extra s3`).

Real model calls need a local settings file. Create it from the checked-in
template, then fill in your API keys:

```bash
cp local_settings.example.py local_settings.py
```

Never commit `local_settings.py`. `local_settings.example.py` is the only
settings template in the repository.

## Contract snapshot

This repository vendors a copy of the language-neutral contract fixtures in
`tests/fixtures/parity/`. The source of truth is `../vv-agent-contract/`. Never
edit the vendored copy by hand. Check it before you change or release contract
behavior:

```bash
python3 scripts/contract_snapshot.py check
```

[parity-contract.md](parity-contract.md) describes the full workflow for
adopting a contract change. Python is the only required implementation, so a
Python adoption never needs a Rust update.

If a new contract revision changes `builtin_tools.json`, regenerate the
readable Python schema source, then check that generation is reproducible:

```bash
uv run python scripts/generate_builtin_tool_schemas.py
uv run pytest tests/test_builtin_tool_schema_generation.py tests/test_tool_schema_contract.py
```

## Checks

While you iterate, run the narrowest tests that cover your change:

```bash
uv run pytest tests/test_config.py
uv run pytest tests/test_runner.py tests/test_public_agent.py tests/test_function_tool.py
uv run pytest tests/test_runtime.py
uv run pytest tests/test_tools.py
uv run pytest tests/test_app_server_jsonrpc.py tests/test_app_server_initialize.py tests/test_app_server_thread_turn.py
```

Before you finish, run formatting, linting and the type checker:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Before you report a broad change as complete, run the full test suite. Live
tests are skipped by default:

```bash
uv run pytest
```

### App Server changes

Also check the CLI entry points and schema generation:

```bash
uv run pytest tests/test_app_server_*.py
uv run python -m vv_agent --help
uv run python -m vv_agent app-server --help
uv run python -m vv_agent app-server schema --out /tmp/vv-agent-schema
uv run python -m vv_agent app-server generate-ts --out /tmp/vv-agent-schema/typescript
uv run python -m vv_agent debug app-server send-message hello
```

### Tool metadata, policy and telemetry changes

This covers tool capability metadata, denial policy, execution telemetry, and
how they appear in durable events and the App Server. Run the suites that
exercise the real producers. A fixture check alone only proves the bytes match,
not that the code produces them.

```bash
uv run pytest tests/test_tool_orchestrator.py tests/test_runtime_hooks.py tests/test_events_contract.py tests/test_event_validation.py
uv run pytest tests/test_runner.py tests/test_runner_events_producer_parity.py tests/test_runner_trace_contract.py
uv run pytest tests/test_output_validation_contract.py tests/test_run_resume.py
uv run pytest tests/test_app_server_item_mapper.py tests/test_app_server_contract_parity.py
uv run pytest tests/test_run_definition_producer.py
uv run pytest tests/test_configured_sub_agent_parity.py tests/test_sub_task_manager_continuation_recovery.py
uv run pytest tests/test_tool_metadata_contract.py tests/test_function_tool.py tests/test_tool_schema_contract.py tests/test_parity_evidence_manifests.py
```

### Live tests

Live tests call real providers. Enable them explicitly:

```bash
VV_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `VV_AGENT_LOCAL_SETTINGS` | `local_settings.py` | Settings file used by live tests and the CLI. |
| `VV_AGENT_LIVE_BACKEND` | `moonshot` (`deepseek` in `tests/test_live_deepseek.py`) | Provider backend for live smoke tests. |
| `VV_AGENT_LIVE_MODEL` | `kimi-k3` (`deepseek-v4-pro` in `tests/test_live_deepseek.py`) | Model key for live smoke tests. |

### Test isolation

- Tests give default workspaces their own `tmp_path`.
- Scripted smoke processes and examples use temporary workspaces.
- When you run a fixture generator, write its output under `/tmp`, never into
  the vendored snapshot.
- PostgreSQL tests connect through a local Unix socket or
  `VV_AGENT_TEST_POSTGRES_DSN`. They create disposable databases and drop them
  afterwards. Without either, the PostgreSQL cases are skipped.

## Which tests cover what

| Change area | Primary tests |
| --- | --- |
| Snapshot client and support-matrix adoption | `tests/test_contract_snapshot.py` |
| Shared contract and canonical producers | `tests/test_tool_metadata_contract.py`, `tests/test_parity_evidence_manifests.py`, `tests/test_tool_schema_contract.py`, `tests/test_app_server_contract_parity.py`, `tests/test_runner_events_producer_parity.py` |
| Building tool metadata and keeping it out of the model schema | `tests/test_tool_metadata_contract.py`, `tests/test_run_definition_producer.py`, `tests/test_function_tool.py`, `tests/test_tool_schema_contract.py`, `tests/test_parity_evidence_manifests.py` |
| Metadata denial and child delegation | `tests/test_tool_orchestrator.py`, `tests/test_configured_sub_agent_parity.py`, `tests/test_handoffs.py` |
| Typed tool telemetry (planned, started, completed) | `tests/test_tool_orchestrator.py`, `tests/test_runtime_hooks.py`, `tests/test_events_contract.py`, `tests/test_event_validation.py`, `tests/test_runner.py`, `tests/test_runner_events_producer_parity.py`, `tests/test_runner_trace_contract.py` |
| Tool telemetry in the App Server | `tests/test_app_server_item_mapper.py`, `tests/test_app_server_contract_parity.py` |
| Tool metadata in frozen turn definitions | `tests/test_run_definition_producer.py` |
| Optional output validation and tools-free repair | `tests/test_output_validation_contract.py` |
| Settings and model resolution | `tests/test_config.py` |
| CLI | `tests/test_config.py`, `tests/test_cli_contract.py`, and CLI assertions in other tests |
| Runtime loop and statuses | `tests/test_runtime.py`, `tests/test_model_turn.py` |
| Public SDK | `tests/test_public_agent.py`, `tests/test_runner.py`, `tests/test_model_settings.py`, `tests/test_function_tool.py`, `tests/test_run_events.py`, `tests/test_agent_as_tool.py`, `tests/test_handoffs.py`, `tests/test_tool_policy.py`, `tests/test_tool_approval.py`, `tests/test_guardrails.py`, `tests/test_tracing.py`, `tests/test_compiler.py` |
| Live handles and event replay | `tests/test_run_handle_live_stream.py`, `tests/test_event_store.py`, `tests/test_events_contract.py`, `tests/test_session_graph_events.py` |
| App Server protocol, transport, replay, approvals, schema and CLI | `tests/test_app_server_jsonrpc.py`, `tests/test_app_server_initialize.py`, `tests/test_app_server_transport.py`, `tests/test_app_server_request_serialization.py`, `tests/test_app_server_thread_store.py`, `tests/test_app_server_thread_turn.py`, `tests/test_app_server_item_mapper.py`, `tests/test_app_server_approval.py`, `tests/test_app_server_replay.py`, `tests/test_app_server_schema.py`, `tests/test_app_server_cli.py` |
| Host provider interfaces (approval, context, memory) | `tests/test_approval_protocol.py`, `tests/test_context_providers.py`, `tests/test_memory_provider.py`, `tests/test_interactive_approval_bridge.py`, `tests/test_interactive_memory_provider_bridge.py` |
| LLM and tool hooks, after-cycle hooks | `tests/test_runtime_hooks.py`, `tests/test_after_cycle_hooks.py` |
| Tools and schemas | `tests/test_tools.py`, `tests/test_tool_schemas.py`, `tests/test_tool_planner.py`, `tests/test_tool_orchestrator.py` |
| Memory and compaction | `tests/test_memory.py`, `tests/test_microcompact.py`, `tests/test_microcompaction_policy.py`, `tests/test_microcompaction_events.py`, `tests/test_memory_local_contract.py`, `tests/test_session_memory.py` |
| Session execution and persistence | `tests/session/`, `tests/test_run_resume.py` |
| Session model selection | `tests/session/test_model_selection.py`, `tests/test_app_server_model_selection.py` |
| Workspace backends | `tests/test_workspace_backends.py` |
| Live provider behavior | `tests/test_live_*.py` |

[session-kernel-capability-matrix.md](session-kernel-capability-matrix.md)
maps each session kernel capability to its owner module and tests.

## Change hygiene

- Export every new public type from `src/vv_agent/__init__.py`. A user should be
  able to import any SDK concept from `vv_agent`.
- Hosts extend vv-agent through its published extension points, not by patching
  the runner, compiler, memory or tool dispatch. The extension points are:
  - `AppServerHost`, for integration over a JSONL process;
  - `ApprovalProvider` and `ContextProvider`, exported from `vv_agent`;
  - `SessionRunEventStore`, or `JsonlRunEventStore` as a projection sink;
  - `vv_agent.memory.MemoryProvider`;
  - `vv_agent.tools.ToolExecutor` or `FunctionTool`.
- Typed `RunEvent` objects are the only event contract for application state.
  A host UI streams or replays them through `Runner.start()`,
  `RunHandle.events()`, and `SessionRunEventStore` (or the `JsonlRunEventStore`
  projection sink).
- An App Server host renders state from `item/*` notifications and the `items`
  in snapshots.
- When a user-facing default, environment variable or command example changes,
  update the README and examples too.
- Reject model settings, backend keys, endpoint shapes and wire fields that are
  not in the current format. Give a clear error and add a test for it.
- When a change spans the runtime, the SDK and tools, add a focused test at each
  public boundary. One integration test is not enough.
