# Development

Run every command from the repository root.

## Environment

Install the package and its development dependencies:

```bash
uv sync --dev
```

The dev group includes native `httpx` for HTTP error/response test coverage.
The runtime transport remains `httpx2`; the two packages are separate.
Runtime-only installs can still skip the optional native-httpx cases.

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

Before you report a broad change as complete, run the full checks remotely.
Keep correctness tests on their regular temporary filesystem and run the Linux
simulator performance gate separately. Live tests are skipped by default:

```bash
uv run pytest -k "not simulation_smoke"
TMPDIR=/dev/shm timeout 60s uv run pytest tests/sim -k smoke -q -s
```

### Deterministic session simulation

`tests/sim/` schedules the real driver and SQL stores with scripted external
services. See its README for covered invariants, replay reports and limits.
Use a small explicit sample during local development:

```bash
uv run python scripts/session_simulation.py --seeds 5 --workers 2
uv run pytest tests/sim/test_mutations.py -k sqlite -q
```

The PR smoke gate runs 1,000 independent SQLite-file seeds within 60 seconds,
including process startup, tokenizer initialization and process shutdown. Each
world uses one thread; the selected implementation uses the original bounded
batcher with spawned processes and at most `2 * workers` jobs per batch. The CI
smoke step alone sets `TMPDIR=/dev/shm`.
These are real SQLite files on a RAM-backed filesystem, with WAL, FULL
synchronous mode and every invariant check retained. This evidence covers SQL
transactions and worker-process recovery; it does not prove durability through
host power loss.

Do not export that setting for the full suite. Keep disk-directed process and
transaction tests on regular paths and retain PostgreSQL acceptance through a
disposable database over TCP with password authentication. Reports record the
actual temporary directory, filesystem type (or null when unavailable), and
requested `TMPDIR` in `environment.sqlite_temporary_storage`. Full suites and
the million-seed SQLite / 10,000-seed PostgreSQL campaigns run remotely.

### Concurrency-fix seed reference

Classify a concurrency bug fix with a `fix(concurrency): ...` commit subject and
cite at least one reproducing simulation seed on its own line in the commit body:

```text
fix(concurrency): preserve stale lease fencing

Simulation-Seed: 17
```

Ordinary refactors and tests do not need the trailer. CI checks commits in the
selected pull-request or push `base..HEAD` set, including merge and force-update
history. This convention cannot identify a fix that was not classified as
`fix(concurrency)` and does not prove that the cited seed reproduces the bug;
reviewers still own those judgments.

### Kernel storage boundary

Raw connection, cursor and SQL execution APIs belong only in
`src/vv_agent/kernel/_sql.py`, `src/vv_agent/kernel/_sqlite.py`, and
`src/vv_agent/kernel/_postgres.py`. Other kernel code uses `SessionStore` and
`SessionTx` methods such as `append`, `ack`, and lease operations.
`scripts/check_repository_contracts.py` enforces this source boundary.

The check catches direct calls and simple captured aliases to storage APIs.
It does not prove every mutation path: reflection, arbitrary callbacks and
disguised aliases still require review against the storage protocol.

### Operation dispatch boundary

`py:src/vv_agent/kernel/_driver_dispatch.py::dispatch` is the only core producer of an
`InvokeOperation` request. `py:src/vv_agent/session/_driver_scope.py::_invoke` is the only SDK
leaf that calls the runtime invoker. The repository gate scans every Python
file under `src/vv_agent` and requires exactly one construction and one
`.invoker.invoke` call at the located top-level function nodes. A nested or
class method with the same name does not count. Missing files or definitions
are failures; there is no fixture-only or disabled mode.

The check flattens imports, `from ... import ... as ...` aliases, and simple
name-to-name/attribute assignments found anywhere in a module into one lexical
alias map. It rejects additional, wrong-node, and straightforward aliased
calls. The invoker shape is lexical: the resolved callee must end in
`.invoker.invoke`, independent of the receiver's variable name. It does not
follow control flow or assignment order, resolve scope shadowing, infer
parameter or descriptor types, follow re-exports, or analyze computed names,
dynamic `getattr`, arbitrary reflection or callbacks. Scheduler wake, database
persistence, callbacks, provider query/cancel methods and trusted host
admission are different ports and are outside this structural claim. The
result covers these two dispatch leaves; it does not prove that every possible
host side effect is confined.

```bash
uv run python scripts/check_repository_contracts.py
uv run pytest tests/test_kernel_dispatch_boundary.py tests/test_repository_contracts.py tests/test_kernel_boundaries.py
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

### Bounded-result fixture ownership

`scripts/session_kernel_fixtures.py` owns the whole `bounded_tool_result.json`,
including canonical results and `unified_projection`. It calls the real cases
from `scripts/tool_result_artifact_fixtures.py`, whose CLI writes only
`tool_result_artifact.json`. Regenerate with
`uv run python scripts/session_kernel_fixtures.py --output <dir>` outside any
sandbox that blocks the Bash process supervisor, and copy only the files you
mean to change into the contract checkout.

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
