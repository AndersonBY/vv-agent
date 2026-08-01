# Python Contract Integration

`vv-agent` is the Python implementation of the language-neutral contract in
[`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
Normative behavior, fixtures, versioning, and adoption workflow live only in
that repository.

## Pinned Contract

`contract.lock.json` selects contract `6.1.0` at revision
`4efeb24ca0c05d56068314fe75b0c6e5cf78fdc4`. Its immutable release artifact has
SHA-256 `1ac62903ffa9d5b204dfb14fa538ea33796dc8346c304cf4695b9d158f48d56c`.
The current adoption state is not duplicated in this document. Treat
[`vv-agent-contract/support-matrix.json`](https://github.com/AndersonBY/vv-agent-contract/blob/main/support-matrix.json)
as the machine-readable source for the current verified Python and Rust
revisions, verification timestamp, and cross-repository run URL.

The lock records the exact release artifact, artifact digest, vendored fixture
path, and canonical fixture-manifest digest. `tests/fixtures/parity/` is a
generated snapshot, not an editable source.

## Required Workflow

For any shared public, model-visible, runtime, persistence, or wire change:

1. Read this repository's lock and `../vv-agent-contract/AGENTS.md`.
2. Read the central parity, versioning, and change-workflow documents.
3. Change canonical docs and fixtures in `vv-agent-contract` first.
4. Sync both implementation snapshots with `scripts/contract_snapshot.py`.
5. Update real Python and Rust producers, not only fixture parsers.
6. Run both full repository gates and central cross-repository CI.

Never edit a vendored parity fixture or digest directly.

## Snapshot Commands

```bash
python3 scripts/contract_snapshot.py check
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```

After an immutable central release exists:

```bash
python3 scripts/contract_snapshot.py sync \
  --source ../vv-agent-contract \
  --artifact /path/to/vv-agent-contract-6.1.0.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v6.1.0/vv-agent-contract-6.1.0.zip
```

## Python Producer Map

| Contract surface | Python producer and evidence |
| --- | --- |
| Public API | `src/vv_agent/__init__.py`, `tests/test_parity_evidence_manifests.py` |
| Resolved PromptBundle and one-run producer scope | `src/vv_agent/prompt/`, `src/vv_agent/runtime/compiler.py`, `src/vv_agent/runtime/run_definition.py`, `src/vv_agent/llm/`; `tests/test_prompt_builder.py`, `tests/test_context_providers.py`, `tests/test_protocol_types.py`, `tests/test_checkpoint.py`, `tests/test_distributed_checkpoint.py` |
| Canonical 15-tool surface and compact schemas | `src/vv_agent/constants/workspace.py`, `src/vv_agent/tools/registry.py`, `src/vv_agent/tools/executor.py`; `tests/test_tool_schema_contract.py`, `tests/test_builtin_tool_behavior_contract.py` |
| Sparse bounded tool results, artifact recovery, and read cursor | `src/vv_agent/types.py`, `src/vv_agent/workspace/artifacts.py`, `src/vv_agent/tools/handlers/bash.py`, `src/vv_agent/tools/handlers/background.py`, `src/vv_agent/tools/handlers/workspace_io.py`; `tests/test_protocol_types.py`, `tests/test_checkpoint.py`, `tests/test_distributed_checkpoint.py`, `tests/test_bash_tools.py`, `tests/test_workspace_io_parity.py` |
| Tool metadata and policy | `src/vv_agent/tools/metadata.py`, `src/vv_agent/run_config.py`, `src/vv_agent/runtime/tool_planner.py`, `tests/test_tool_metadata_contract.py`, `tests/test_tool_policy.py` |
| Tool execution lifecycle | `src/vv_agent/tools/orchestrator.py`, `src/vv_agent/runtime/tool_call_runner.py`, `tests/test_tool_orchestrator.py`, `tests/test_runtime_hooks.py` |
| Agent, Runner, result, and live control | `src/vv_agent/agent.py`, `src/vv_agent/runner.py`, `src/vv_agent/run_handle.py`, `src/vv_agent/result.py` |
| Typed events | `src/vv_agent/events.py`, `src/vv_agent/event_store.py`, `tests/test_events_contract.py`, `tests/test_event_validation.py`, `tests/test_runner_events_producer_parity.py` |
| LLM stream projection | `src/vv_agent/llm/`, `src/vv_agent/runtime/cycle_runner.py`, `tests/test_llm_interface.py`, `tests/test_runner_events_producer_parity.py` |
| Configured children | `src/vv_agent/runtime/engine.py`, `src/vv_agent/runtime/sub_task_manager.py`, `tests/test_configured_sub_agent_parity.py`, `tests/test_sub_agent_runtime.py` |
| Sessions | `src/vv_agent/sessions/`, `src/vv_agent/interactive.py`, `tests/test_session_store_parity.py`, `tests/test_interactive_lifecycle_contract.py` |
| Memory and compaction | `src/vv_agent/microcompaction.py`, `src/vv_agent/memory/`, `src/vv_agent/runtime/cycle_runner.py`, `tests/test_microcompact.py`, `tests/test_microcompaction_policy.py`, `tests/test_microcompaction_events.py`, `tests/test_memory_lifecycle_contract.py`, `tests/test_memory_provider.py` |
| Model-call ledger, token, and cache usage | `src/vv_agent/types.py`, `src/vv_agent/runtime/model_calls.py`, `src/vv_agent/runtime/token_usage.py`, `src/vv_agent/llm/vv_llm_client.py`, `tests/test_token_usage_contract.py`, `tests/test_runtime_task_serialization.py` |
| Run budgets | `src/vv_agent/budget.py`, `src/vv_agent/runtime/engine.py`, `tests/test_run_budget.py` |
| Durable checkpoint and resume | `src/vv_agent/checkpoint.py`, `src/vv_agent/runtime/checkpoint_codec.py`, `src/vv_agent/runtime/checkpoint_resume.py`, `src/vv_agent/runtime/run_definition.py`, `tests/test_checkpoint.py`, `tests/test_checkpoint_runner.py`, `tests/test_checkpoint_fault_matrix.py` |
| Distributed execution | `src/vv_agent/runtime/backends/distributed.py`, `src/vv_agent/runtime/backends/celery_tasks.py`, `tests/test_distributed_checkpoint.py` |
| App Server | `src/vv_agent/app_server/usage_projection.py`, `src/vv_agent/app_server/item_mapper.py`, `src/vv_agent/app_server/run_adapter.py`, `tests/test_app_server_contract_parity.py`, `tests/test_app_server_item_mapper.py` |
| Output validation | `src/vv_agent/output_validation.py`, `src/vv_agent/runner.py`, `tests/test_output_validation_contract.py` |

A parser-only test cannot prove producer parity. Every declared field must be
consumed by the planner, runtime, adapter, store, or protocol projection that
owns its behavior.

## Current Boundaries

### Events

The public runtime accepts and emits only typed `RunEvent` values. Runtime
producers create semantic lifecycle events directly. Every primary and internal
model dispatch emits `model_call_started` and exactly one terminal
`model_call_completed` or `model_call_failed` event with the same call identity
as the durable ledger. Provider stream payloads remain inside the LLM adapter
and are projected to typed assistant, reasoning, and model-tool-call events at
that boundary; malformed or unknown provider payloads are dropped.

Task-neutral observations use `DiagnosticEvent(level, code, details)`. A
diagnostic cannot replace lifecycle, approval, budget, cancellation, tool, or
terminal state. Child event forwarding preserves the original event identity
and parent/run/trace/session relationships.

RunEvent `v2` is a strict current discriminator. Readers reject missing, stale,
unknown, and malformed fields; there is no alternate event decoder.

### Model Capacity

The configured automatic compaction threshold is `250000`. Resolved model
capacity is projected to `model_context_window` and
`model_max_output_tokens`. Output capability is never copied into request
settings and never creates an implicit `max_tokens` value.

Output reservation order is explicit `ModelSettings.max_tokens`, explicit task
`reserved_output_tokens`, then the `16000` planning fallback. The fallback is
capacity planning only, not a model output limit. Configured children inherit
the same explicit `ModelProvider` and resolve their own model.

### Prompt Bundle And Tools

`PromptBundle` is the only resolved system-prompt representation after a run
starts. `AgentTask`, `LlmRequest`, run definitions, checkpoints, and
distributed envelopes carry it explicitly. Instruction providers, context
providers, and the run clock execute once while compiling a new run; every
cycle reuses the immutable bundle. Resume restores it without calling those
producers again. Providers without section-aware caching receive one
deterministic flattening; Anthropic may use canonical section boundaries for
cache breakpoints. Generic metadata is never a prompt-section transport.

The current model-visible manifest is `vv-agent-builtin-tools-v2` with 15
direct tools. `ToolExposure` has only `direct` and `hidden`; there is no
deferred exposure. The model-visible `compress_memory` tool and its
`memory_notes` state do not exist. Framework-owned automatic compaction remains
internal.

`ToolExecutionResult.status_code` is the only result status. The current values
are `SUCCESS`, `ERROR`, `WAIT_RESPONSE`, `RUNNING`, and `PENDING_COMPRESS`.
`PENDING_COMPRESS` is reserved for framework-internal compaction flow and is
not produced by a built-in model-callable tool. Unknown fields are rejected.

Ordinary results carry their required fields plus only optional fields that are
actually present. A bounded result adds a reason, byte counts, and one recovery
pointer: bash uses an immutable workspace artifact and `read_file` uses a
source-verified cursor. Bash preserves bounded stdout/stderr even when the
process exits non-zero, and terminal background polling reuses the same
artifact instead of writing a second copy. Sparse recovery fields survive
cycle/result, checkpoint, and distributed serialization. Artifact reads must
return through the normal workspace policy; cursors reject changed content,
mismatched paths, and invalid offsets. Large output is never hidden in generic
metadata or automatically replayed into model context.

For a local workspace, `.vv-agent/artifacts/` is a logical recovery namespace,
not a shell-visible directory. The adapter maps it to private storage outside
the shell working directory. Truncated terminal output is streamed into one
exclusive immutable artifact, so the runtime does not materialize the complete
capture in application memory and shell commands cannot mutate recovery bytes.

`ToolMetadata` is the only typed capability declaration and contains
`side_effect`, `idempotency`, `terminal`, `result_retention`,
`capability_tags`, and `cost_dimensions`. Missing retention defaults to
`archive`; `preserve` excludes the result from proactive microcompaction.
Generic host metadata is separate and cannot populate this declaration.
Metadata policy is denial-only across parent and delegated layers.

The executor sequence is `tool_call_planned`, optional approval,
`tool_call_started` immediately before effects may begin, and
`tool_call_completed` after a result exists. Durable journals, not telemetry,
own ambiguity and replay decisions.

### Persistence

Checkpoint records require `vv-agent.checkpoint.v5`; run definitions require
`vv-agent.run-definition.v5`; distributed envelopes require
`vv-agent.distributed-run.v5`. The frozen definition stores `prompt_bundle`,
not a second independently editable flattened system prompt. Readers reject every other shape before claim or
external work. There is no namespace probe, alternate decoder, field synthesis,
or in-place repair.

The checkpoint owns the complete run-level model-call ledger. A started journal
entry and started event become durable together. After dispatch, the terminal
journal state, ledger record, budget observation, and terminal event become
durable together. Journal, started event, terminal event, and ledger must agree
exactly on call id, operation id, attempt, operation, cycle, backend, and model.
A definitive pre-dispatch failure is the only terminal model journal state with
no dispatch evidence.

Distributed worker responses use only the closed
`vv-agent.distributed-worker-response.v3` wire. Python owns the typed value and
strict reader in `runtime/backends/distributed.py`, Celery workers produce it in
`celery_tasks.py`, and the scheduler consumes it in `celery.py`. `pending`,
`committed`, `terminal_candidate`, and `terminal_replay` are the only variants;
the replaced `finished` and terminal boolean combination is rejected. A
candidate accepts reconciliation-required or terminal/interrupted results; a
replay rejects reconciliation-required and must equal the retained durable
result. The scheduler reloads the authoritative checkpoint after every response
or transport failure. Public `AgentResult` readers require the complete current
shape, reject unknown fields, and require absent optional fields to be omitted
rather than encoded as null.

### Model Usage And Memory

`TaskTokenUsage v2` contains the ordered `model_calls` ledger for primary agent
cycles, Session Memory extraction, and memory compaction. Aggregate counts are
null when any dispatched attempt lacks that measurement; an empty ledger has
exact zero token totals. `CycleRecord` does not duplicate usage, and the
low-level `CycleRunner` is not a public export.

Session Memory defaults to disabled. Only the exact public
`session_memory_enabled=true` control enables prompt injection, store access,
workspace writes, or Session Memory model calls. Existing files, supplied
context, seed data, parent configuration, and historical aliases do not enable
it implicitly.

When enabled, a newly compiled run reads persisted entries once and freezes
them into its `PromptBundle`. Extraction during that run may persist new entries
but never rewrites the active bundle; those entries become model-visible only
in a later newly compiled run. Checkpoint resume restores the frozen section
without rereading the store.

`MicrocompactionPolicy` is the public typed control for proactive compaction.
It defaults to trigger/target ratios `0.75`/`0.60`, 3 protected recent cycles,
and a 500-character minimum. Eligible archive results are planned oldest-first
once per cycle and applied until the target is reached. Replacement happens
only after immutable persistence through the effective `WorkspaceBackend`;
failed or short writes stay inline, and existing typed artifacts are reused.
No proactive candidate or no model-visible `read_file` means no micro lifecycle
event.

Run-definition v5 freezes the policy at
`runtime_controls.microcompaction_policy`. Session, checkpoint, distributed,
and host round trips preserve the typed artifact reference while model
projection exposes only this closed marker:

```text
<Tool Result Compact>
tool_name: web_search
artifact_path: .vv-agent/artifacts/<run>/<call>.txt
retrieval_hint: use read_file on artifact_path if needed
excerpt:
<bounded head/tail preview>
</Tool Result Compact>
```

Artifact byte size and SHA-256 remain host-only integrity fields and never
appear in the model-visible marker. SQLite session persistence uses the strict
current schema at `PRAGMA user_version=2`.

### App Server

Model-call events project to `modelCall` items carrying the same seven identity
fields and terminal accounting. Terminal `tokenUsage` recursively camel-cases
the complete task usage object, including `modelCalls` and `cacheUsage`, while
opaque provider-native keys inside `providerUsage` remain unchanged.

## Python Adaptations

The following language-shape differences are allowed only while observable
behavior remains identical:

- Python dataclasses, protocols, decorators, and exceptions map to Rust structs,
  traits, builders, and `Result`.
- Python synchronous entry points may wrap asynchronous internals.
- Python output coercion maps to Rust typed deserialization.
- Celery maps to Apalis through the same envelope, lease, checkpoint, and
  terminal contract.
- Python exposes `DistributedRunHandle`, `DistributedDeliveryOutcome`, and
  `DistributedAdvanceDecision` as the passive handle, transport observation,
  and one-step scheduler decision mapped by the central nonblocking driver
  contract. `CeleryBackend.start()` and `advance()` are enqueue-only; synchronous
  `execute()` remains a separate controller entry point.
- Python settings-file resolution maps to Rust's explicit `ModelProvider`.

## Completion Gate

```bash
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
uv run pytest
uv run ruff check .
uv run ty check
uv build
```

Then run the Rust gate and central cross-repository workflow with exact refs.
Record final revisions and the workflow URL in the central support matrix only
after every gate passes.
