# Python Contract Integration

`vv-agent` implements the Python side of the canonical contract published by
[`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
The normative behavior and change workflow no longer live in this repository.

## Pinned Contract

`contract.lock.json` is the machine-readable adoption record. It pins:

- semantic contract version;
- exact central Git revision;
- immutable release artifact URL and SHA-256;
- local vendored snapshot path;
- canonical `SHA256SUMS` digest.

`tests/fixtures/parity/` is generated from that release. It is committed for
offline and reproducible tests, but it is not an editable source of truth.

The current lock selects contract `0.10.0` at revision
`4d721573c1b1e0a8bc4a277f0366c52c1a63b4b4`. The central support matrix records
contract `0.10.0` and both implementations as `pending-adoption`; contract
`0.9.0` remains the verified baseline, backed by cross-repository conformance
run [`29769410376`](https://github.com/AndersonBY/vv-agent-contract/actions/runs/29769410376).
This document maps the Python producers; the central matrix remains the
authoritative verification record and must not move to `verified` until both
implementations and cross-repository CI satisfy the central workflow.

## Required Reading

For shared public, model-visible, runtime, persistence, or wire changes, read:

1. `contract.lock.json` in this repository;
2. `../vv-agent-contract/AGENTS.md`;
3. `../vv-agent-contract/docs/parity-contract.md`;
4. `../vv-agent-contract/docs/change-workflow.md`;
5. sibling `../vv-agent-rs/docs/parity-contract.md`.

If the sibling checkout is unavailable, use the exact repository and revision
from the lock. Do not infer the current contract from a floating `main` branch.

## Snapshot Commands

Offline verification of the committed snapshot:

```bash
python3 scripts/contract_snapshot.py check
```

Stronger verification against the sibling canonical checkout:

```bash
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```

Synchronization is allowed only after the canonical version is committed and
its deterministic release zip exists:

```bash
python3 scripts/contract_snapshot.py sync \
  --source ../vv-agent-contract \
  --artifact /path/to/vv-agent-contract-<version>.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip
```

Never repair a contract failure by editing a file under
`tests/fixtures/parity/` or changing only a digest.

## Python Producer Map

| Contract surface | Python producer or evidence |
| --- | --- |
| Public API inventory | `src/vv_agent/__init__.py`, `tests/test_parity_evidence_manifests.py` |
| System prompt | `src/vv_agent/prompt/`, `tests/test_prompt_builder.py` |
| Built-in tool specification | `src/vv_agent/constants/workspace.py`, `src/vv_agent/tools/registry.py`, `tests/test_tool_schema_contract.py` |
| Typed tool metadata and schema isolation | `src/vv_agent/tools/metadata.py`, `src/vv_agent/tools/base.py`, `src/vv_agent/tools/function.py`, `src/vv_agent/tools/executor.py`, `src/vv_agent/tools/registry.py`, `tests/test_tool_metadata_contract.py`, `tests/test_run_definition_producer.py`, `tests/test_function_tool.py`, `tests/test_tool_schema_contract.py`, `tests/test_parity_evidence_manifests.py` |
| Metadata-denial policy and delegation | `src/vv_agent/run_config.py`, `src/vv_agent/types.py`, `src/vv_agent/runner.py`, `src/vv_agent/runtime/compiler.py`, `src/vv_agent/runtime/tool_planner.py`, `src/vv_agent/runtime/engine.py`, `src/vv_agent/runtime/sub_task_manager.py`, `tests/test_tool_orchestrator.py`, `tests/test_configured_sub_agent_parity.py`, `tests/test_handoffs.py` |
| Tool execution telemetry and event wire v1 | `src/vv_agent/tools/orchestrator.py`, `src/vv_agent/runtime/tool_call_runner.py`, `src/vv_agent/events.py`, `tests/test_tool_orchestrator.py`, `tests/test_runtime_hooks.py`, `tests/test_events_v1.py`, `tests/test_run_events_v1_invalid.py`, `tests/test_runner.py`, `tests/test_runner_events_producer_parity.py`, `tests/test_runner_trace_v1.py`, `tests/test_checkpoint_runner_v2.py` |
| Agent, Runner, result, live control | `src/vv_agent/agent.py`, `src/vv_agent/runner.py`, `src/vv_agent/run_handle.py` |
| Optional output validation and repair | `src/vv_agent/output_validation.py`, `src/vv_agent/agent.py`, `src/vv_agent/runner.py`, `tests/test_output_validation_contract.py` |
| Delegation and background tasks | `src/vv_agent/background_task.py`, `src/vv_agent/handoffs.py`, `src/vv_agent/runtime/sub_task_manager.py` |
| Sessions and stores | `src/vv_agent/sessions/`, `src/vv_agent/runtime/stores/`, `tests/test_session_store_parity.py` |
| Events and tracing | `src/vv_agent/events.py`, `src/vv_agent/event_store.py`, `src/vv_agent/tracing.py` |
| Model stream projection | `src/vv_agent/events.py`, `src/vv_agent/runner.py`, `src/vv_agent/runtime/engine.py`, `src/vv_agent/app_server/item_mapper.py`, `tests/test_runner_events_producer_parity.py`, `tests/test_configured_sub_agent_parity.py` |
| Token and cache usage | `src/vv_agent/types.py`, `src/vv_agent/runtime/token_usage.py`, `src/vv_agent/llm/vv_llm_client.py`, `tests/test_token_usage_contract.py` |
| Assistant reasoning history | `src/vv_agent/types.py`, `src/vv_agent/memory/manager.py`, `src/vv_agent/memory/message_sanitizer.py`, `src/vv_agent/runtime/cycle_runner.py`, `tests/test_runner.py`, `tests/test_message_sanitizer.py`, `tests/test_sub_task_status.py` |
| Memory capacity and compaction lifecycle | `src/vv_agent/types.py`, `src/vv_agent/interactive.py`, `src/vv_agent/runtime/compiler.py`, `src/vv_agent/runtime/engine.py`, `src/vv_agent/runtime/cycle_runner.py`, `src/vv_agent/memory/manager.py`, `src/vv_agent/memory/token_utils.py`, `src/vv_agent/events.py`, `tests/test_memory_lifecycle_contract.py`, `tests/test_memory_provider.py`, `tests/test_events_v1.py`, `tests/test_run_events_v1_invalid.py`, `tests/test_compiler.py`, `tests/test_configured_sub_agent_parity.py`, `tests/test_interactive_session_api.py` |
| Run budgets | `src/vv_agent/budget.py`, `src/vv_agent/runtime/engine.py`, `tests/test_run_budget.py` |
| After-cycle lifecycle hooks | `src/vv_agent/runtime/lifecycle.py`, `src/vv_agent/runtime/engine.py`, `src/vv_agent/runtime/run_definition.py`, `src/vv_agent/runtime/backends/distributed.py`, `src/vv_agent/runtime/backends/celery_tasks.py`, `tests/test_after_cycle_hooks.py`, `tests/test_distributed_checkpoint_v2.py` |
| Durable checkpoint/resume v2 | `src/vv_agent/checkpoint.py`, `src/vv_agent/runtime/checkpoint_codec_v2.py`, `src/vv_agent/runtime/checkpoint_resume.py`, `src/vv_agent/runtime/run_definition.py`, `tests/test_run_definition_producer.py`, `tests/test_checkpoint_v2.py`, `tests/test_checkpoint_runner_v2.py`, `tests/test_checkpoint_fault_matrix.py` |
| Distributed runtime | `src/vv_agent/runtime/backends/distributed.py`, `src/vv_agent/runtime/backends/celery_tasks.py`, `src/vv_agent/runtime/checkpoint_codec.py`, `tests/test_distributed_checkpoint_v2.py` |
| App Server | `src/vv_agent/app_server/item_mapper.py`, `src/vv_agent/app_server/`, `tests/test_app_server_item_mapper.py`, `tests/test_app_server_contract_parity.py` |
| Real closure tests | `tests/test_parity_evidence_manifests.py`, `tests/test_runner_events_producer_parity.py` |

A fixture parser or private helper test cannot replace a real public producer
test. A field that is declared but ignored by a planner, executor, provider, or
store remains a contract failure.

## Memory Capacity And Compaction Mapping

The omitted `AgentTask`, `MemoryManager`, and interactive compaction threshold
is `250000`. Explicit values and durable task/checkpoint records retain their
stored value; decoding or resuming an older record does not rewrite it to the
new default.

The compiler records resolved context capability as `model_context_window` and
resolved output capability as `model_max_output_tokens`. The latter is not an
implicit request limit and is never copied into `reserved_output_tokens`.
Runtime reserve resolution uses effective `ModelSettings.max_tokens`, then
explicit task metadata, then the `16000` framework fallback. Only that fallback
may be capped downward by a smaller model output capability. Context resolution
uses explicit task metadata, resolved capability, then `200000`; the derived
prompt capacity subtracts the reserve and `13000` autocompact buffer. A known
derived capacity of zero remains zero.

`CycleRunner` emits the typed trigger and resolved capacity snapshot on every
new `memory_compact_started` event. It emits the strongest applied mode and a
message-content comparison on `memory_compact_completed`. The v1 decoder keeps
all additive fields absent when reading legacy events, while rejecting known
fields with invalid types or unknown enum values. This resolution is mechanical
and does not inspect task category, answer semantics, or semantic progress.

## Tool Metadata And Telemetry Mapping

Python exports `ToolMetadata` and `ToolSideEffect` from `vv_agent` and accepts
`tool_metadata` on `FunctionTool`, `@function_tool`, `ToolSpec`, executors, and
`ToolRegistry.register_tool()`. The actual collection field is
`cost_dimensions` (plural); `metadata.cost_dimension` (singular) is only the
typed denial source reported after an exact cost-dimension match. `side_effect`
is a closed coarse enum, `capability_tags` and `cost_dimensions` are opaque
exact-match labels, and `terminal` only declares that a tool may return
`finish` or `wait_user`. None of these fields grants permission, measures cost,
or ends a run automatically.

The legacy `idempotency` input remains supported. Typed `unknown` may inherit a
non-`unknown` legacy value; conflicting non-`unknown` declarations fail before
model or tool operations. When typed metadata is absent, event metadata stays
absent even though the legacy idempotency remains effective for execution and
the legacy run-definition projection.

`ToolPolicy` adds `denied_side_effects`, `denied_capability_tags`,
`deny_terminal_tools`, and `denied_cost_dimensions`; `SubAgentConfig` exposes
the same four narrowing fields. Lists union and normalize across Agent, Runner,
per-run, and delegated-child layers, while the boolean uses logical OR.
Distributed envelopes serialize the already-effective policy. Planning and
executor dispatch both enforce the denials, which compose with every older
policy check and return `tool_not_allowed`.

The real executor sequence is `tool_call_planned`, optional approval,
`tool_call_started` only at the effect boundary, and `tool_call_completed` once
a result exists. Parse failures have no lifecycle; denials, approval
short-circuits, unknown tools, and durable receipt replay have planned plus
completed without started. Completed writers add `directive`, `error_code`,
`execution_started`, and `duration_ms`; v1 readers preserve missing legacy
fields as absent. Optional typed metadata is carried by all three events but is
omitted when undeclared. Generic metadata cannot fabricate it.

App Server intentionally drops planned events. Started/completed tool items add
camel-case `toolMetadata`, and completed items add `directive`, `errorCode`,
`executionStarted`, and `durationMs` only when present on the event. Checkpoint
v2 freezes `tool_metadata` plus all four policy fields and compares them before
external operations. A 0.7.1 definition is digest-checked first and defaulted
only in an in-memory comparison copy; stored bytes and digest are not rewritten.
Checkpoint v1 and distributed wire v1 writers remain unchanged.

Typed metadata is separate from generic tool metadata and never enters the
model-visible function schema. With no declaration and empty new policy fields,
public schemas, planning, dispatch, event metadata presence, and runtime results
retain the previous behavior.

## Output Validation Mapping

Python registers `output_validator` and optional `output_repair` callbacks on
`Agent`; `output_validation_enabled` remains false unless the host opts in.
`OutputValidationResult` is the typed valid/invalid outcome, while
`OutputValidationContext` exposes only run identity, agent identity, and the
existing `output_type`. A callback receives `(final_output,
validation_context)` in canonical order.

The Runner preserves the existing `output_type` coercion path. When validation
is enabled, an output-type failure can be sent to the one permitted repair, and
the replacement must pass both coercion and the same host validator. Python
maps the canonical empty repair-tool list to the immutable empty tuple `()`.
It does not call the primary model or construct a second tool registry.

The optional validator and at-most-once repair run on the terminal candidate
before session persistence, checkpoint finalization, and terminal-event
emission. Typed rejection returns
`RunResult.error_code == "output_validation_failed"` and commits one failed
terminal; successful repair commits one completed terminal containing the
replacement. Terminal checkpoint replay reuses the validated result without
calling either host callback again. Registered-but-disabled and accepted
validators preserve the prior event and trace observation. Real producer
coverage lives in `tests/test_output_validation_contract.py` and
`tests/test_checkpoint_runner_v2.py`.

## Python Adaptations

The following are API-shape adaptations, not behavioral differences:

- dataclasses, protocols, decorators, and exceptions map to Rust structs,
  traits, builders, and `Result`;
- synchronous convenience wrappers may coexist with async internals;
- Python `output_type` coercion maps to Rust typed deserialization;
- Python exposes synchronous `output_validator` and `output_repair` callbacks
  and an immutable empty repair-tool tuple. These are API-shape adaptations of
  the language-neutral callback lifecycle, not permission to diverge from its
  at-most-once behavior.
- Celery adapters map to Rust Apalis adapters through the same distributed
  envelope, checkpoint, lease, and terminal-state contract;
- Python settings-file model controls map to the equivalent Rust
  `ModelProvider` capability.
- Python exposes copied snapshots through frozen dataclasses and a protocol
  callback; Rust uses immutable structs and a trait object. Both compose
  runner-default hooks before per-run hooks, persist only cumulative denials,
  and resolve distributed `after_cycle_hook_refs` before checkpoint claim.

Add a new adaptation only when both implementations preserve input, output,
safety, persistence, cancellation, and lifecycle semantics.

## Completion Gate

```bash
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
uv run pytest
uv run ruff check .
uv run ty check
```

Then run the Rust gate and the central
`vv-agent-contract/.github/workflows/cross-repository.yml` workflow with exact
contract, Python, and Rust refs. If either implementation is incomplete, keep
the central support matrix at `pending-adoption` or `in-progress`.
