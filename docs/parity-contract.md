# Python Contract Integration

`vv-agent` is the Python implementation of the language-neutral contract in
[`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
Normative behavior, fixtures, versioning, and adoption workflow live only in
that repository.

## Pinned Contract

`contract.lock.json` selects contract `2.0.0` at revision
`8ef7153e9b1f26b90a9fad85bbfcb4642d6462fa`. The central support matrix is
`verified`. Recording run
[`29934473634`](https://github.com/AndersonBY/vv-agent-contract/actions/runs/29934473634)
passed against Python revision
`64743760634fa70c76bf523bf4b51601713ccfb0` and Rust revision
`91a53cd1be9ad560f99c93c4437bc2d830271a09`. The central matrix remains the
authoritative current verification record.

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
  --artifact /path/to/vv-agent-contract-2.0.0.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v2.0.0/vv-agent-contract-2.0.0.zip
```

## Python Producer Map

| Contract surface | Python producer and evidence |
| --- | --- |
| Public API | `src/vv_agent/__init__.py`, `tests/test_parity_evidence_manifests.py` |
| Prompt bundle | `src/vv_agent/prompt/`, `tests/test_prompt_builder.py` |
| Built-in tools | `src/vv_agent/constants/workspace.py`, `src/vv_agent/tools/registry.py`, `tests/test_tool_schema_contract.py`, `tests/test_builtin_tool_behavior_contract.py` |
| Tool metadata and policy | `src/vv_agent/tools/metadata.py`, `src/vv_agent/run_config.py`, `src/vv_agent/runtime/tool_planner.py`, `tests/test_tool_metadata_contract.py`, `tests/test_tool_policy.py` |
| Tool execution lifecycle | `src/vv_agent/tools/orchestrator.py`, `src/vv_agent/runtime/tool_call_runner.py`, `tests/test_tool_orchestrator.py`, `tests/test_runtime_hooks.py` |
| Agent, Runner, result, and live control | `src/vv_agent/agent.py`, `src/vv_agent/runner.py`, `src/vv_agent/run_handle.py`, `src/vv_agent/result.py` |
| Typed events | `src/vv_agent/events.py`, `src/vv_agent/event_store.py`, `tests/test_events_contract.py`, `tests/test_event_validation.py`, `tests/test_runner_events_producer_parity.py` |
| LLM stream projection | `src/vv_agent/llm/`, `src/vv_agent/runtime/cycle_runner.py`, `tests/test_llm_interface.py`, `tests/test_runner_events_producer_parity.py` |
| Configured children | `src/vv_agent/runtime/engine.py`, `src/vv_agent/runtime/sub_task_manager.py`, `tests/test_configured_sub_agent_parity.py`, `tests/test_sub_agent_runtime.py` |
| Sessions | `src/vv_agent/sessions/`, `src/vv_agent/interactive.py`, `tests/test_session_store_parity.py`, `tests/test_interactive_lifecycle_contract.py` |
| Memory and compaction | `src/vv_agent/memory/`, `src/vv_agent/runtime/cycle_runner.py`, `tests/test_memory_lifecycle_contract.py`, `tests/test_memory_provider.py` |
| Token and cache usage | `src/vv_agent/types.py`, `src/vv_agent/runtime/token_usage.py`, `src/vv_agent/llm/vv_llm_client.py`, `tests/test_token_usage_contract.py` |
| Run budgets | `src/vv_agent/budget.py`, `src/vv_agent/runtime/engine.py`, `tests/test_run_budget.py` |
| Durable checkpoint and resume | `src/vv_agent/checkpoint.py`, `src/vv_agent/runtime/checkpoint_codec.py`, `src/vv_agent/runtime/checkpoint_resume.py`, `src/vv_agent/runtime/run_definition.py`, `tests/test_checkpoint.py`, `tests/test_checkpoint_runner.py`, `tests/test_checkpoint_fault_matrix.py` |
| Distributed execution | `src/vv_agent/runtime/backends/distributed.py`, `src/vv_agent/runtime/backends/celery_tasks.py`, `tests/test_distributed_checkpoint.py` |
| App Server | `src/vv_agent/app_server/`, `tests/test_app_server_contract_parity.py`, `tests/test_app_server_item_mapper.py` |
| Output validation | `src/vv_agent/output_validation.py`, `src/vv_agent/runner.py`, `tests/test_output_validation_contract.py` |

A parser-only test cannot prove producer parity. Every declared field must be
consumed by the planner, runtime, adapter, store, or protocol projection that
owns its behavior.

## Current Boundaries

### Events

The public runtime accepts and emits only typed `RunEvent` values. Runtime
producers create semantic lifecycle events directly. Provider stream payloads
remain inside the LLM adapter and are projected to typed assistant, reasoning,
and model-tool-call events at that boundary; malformed or unknown provider
payloads are dropped.

Task-neutral observations use `DiagnosticEvent(level, code, details)`. A
diagnostic cannot replace lifecycle, approval, budget, cancellation, tool, or
terminal state. Child event forwarding preserves the original event identity
and parent/run/trace/session relationships.

RunEvent `v1` is a strict current discriminator. Readers reject missing, stale,
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

### Tools

`ToolExecutionResult.status_code` is the only result status. The current values
are `SUCCESS`, `ERROR`, `WAIT_RESPONSE`, `RUNNING`, and `PENDING_COMPRESS`.
Unknown fields are rejected.

`ToolMetadata` is the only typed capability declaration and contains
`side_effect`, `idempotency`, `terminal`, `capability_tags`, and
`cost_dimensions`. Generic host metadata is separate and cannot populate this
declaration. Metadata policy is denial-only across parent and delegated layers.

The executor sequence is `tool_call_planned`, optional approval,
`tool_call_started` immediately before effects may begin, and
`tool_call_completed` after a result exists. Durable journals, not telemetry,
own ambiguity and replay decisions.

### Persistence

Checkpoint records require `vv-agent.checkpoint.v2`; run definitions require
`vv-agent.run-definition.v1`; distributed envelopes require their one current
discriminator. Readers reject every other shape before claim or external work.
There is no namespace probe, alternate decoder, field synthesis, or in-place
repair.

Distributed worker responses use only the closed
`vv-agent.distributed-worker-response.v1` wire. Python owns the typed value and
strict reader in `runtime/backends/distributed.py`, Celery workers produce it in
`celery_tasks.py`, and the scheduler consumes it in `celery.py`. `pending`,
`committed`, `terminal_candidate`, and `terminal_replay` are the only variants;
the replaced `finished` and terminal boolean combination is rejected. A
candidate accepts reconciliation-required or terminal/interrupted results; a
replay rejects reconciliation-required and must equal the retained durable
result. The scheduler reloads the authoritative checkpoint after every response
or transport failure. Public `AgentResult` readers require all 13 current
fields, reject unknown fields, and require absent optional budget/error fields
to be omitted rather than encoded as null.

## Python Adaptations

The following language-shape differences are allowed only while observable
behavior remains identical:

- Python dataclasses, protocols, decorators, and exceptions map to Rust structs,
  traits, builders, and `Result`.
- Python synchronous entry points may wrap asynchronous internals.
- Python output coercion maps to Rust typed deserialization.
- Celery maps to Apalis through the same envelope, lease, checkpoint, and
  terminal contract.
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
