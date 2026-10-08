# Internal session kernel capability gate

Date: 2026-10-09. Scope: `feat/session-kernel-internal`, internal assembly only.
Public defaults retain their current execution path. Interactive, CLI and App Server
accept one private, non-exported `_kernel` owner for internal producer checks.
No top-level exports, default selection, contract lock/fixture change, or Rust work.

**The F3 capability gate is complete.** Inventory: **63 rows — 63 done
(including intentional differences), 0 partial, 0 missing**. `done` means the
bounded behavior named in that row has a paired producer or an explicitly
asserted kernel difference. Public cut-over still requires the canonical C1
artifact and the F3 deletion/extraction work; internal evidence does not mark
contract 24 verified. F2d-4 surface, wire and deletion details are in
[`session-kernel-f2d-surfaces-report.md`](session-kernel-f2d-surfaces-report.md).

Paths in the owner column are relative to `src/vv_agent/`. `P` below means
`tests/session/test_runner_parity.py`. Other test paths are under `tests/`.
`C` means `tests/session/test_capability_parity.py`; `R` means `tests/session/test_recovery_matrix.py`. D means `tests/session/test_delegation_parity.py`. D and C persistence cases run real PostgreSQL, SQLite files and SQLite `:memory:`. A dash is an explicit missing paired producer, not a waiver.

| Capability | Current owner module | Kernel status | Evidence / remaining requirement |
| --- | --- | --- | --- |
| Basic synchronous no-tool completion | `runner.py`, `runtime/engine.py` | done | P `test_basic_runner_parity` |
| FunctionTool and serial multi-tool execution | `tools/function.py`, `tools/orchestrator.py` | done | P `test_basic_runner_parity` |
| ToolRegistry factory / dynamically registered direct executors | `runner.py`, `tools/registry.py`, `runtime/tool_planner.py` | done | P `test_dynamic_enabled_and_registry_factory_parity` |
| FunctionTool `is_enabled` boolean/callback | `runner.py:_tool_is_enabled` | done | P `test_dynamic_enabled_and_registry_factory_parity`, `test_disabled_tool_never_executes` |
| Hidden tools, dynamic schema changes and exposure boundaries | `tools/executor.py`, `tools/registry.py` | done | T `test_hidden_tool_exposure_parity`, `test_dynamic_schema_new_turn_parity`, `test_dynamic_schema_active_turn_rejected`, `test_hook_patch_preserves_hidden_tool_boundary`; hidden invocation is denied, new turns adopt schema changes, active turns reject drift |
| Built-in `read_file` | `tools/handlers/workspace_io.py` | done | P `test_workspace_tool_parity[read_file-arguments0]` |
| Built-in `write_file` | `tools/handlers/workspace_io.py` | done | P `test_workspace_tool_parity[write_file-arguments1]` |
| Built-in `edit_file` including prior-read baseline | `tools/handlers/workspace_io.py` | done | P `test_workspace_tool_parity[edit_file-arguments2]` |
| Built-in `find_files` | `tools/handlers/search.py` | done | P `test_workspace_tool_parity[find_files-arguments3]` |
| Built-in `search_files` | `tools/handlers/search.py` | done | P `test_workspace_tool_parity[search_files-arguments4]` |
| Built-in `file_info` | `tools/handlers/workspace_io.py` | done | P `test_additional_builtin_parity[file_info-arguments0]` |
| Built-in `todo_write` | `tools/handlers/todo.py` | done | P `test_todo_and_skill_state_parity` with fixed clock and caller-owned TODO id |
| Built-in `ask_user` / host interaction response | `tools/handlers/control.py`, `interaction.py` | done (intentional difference) | T `test_user_wait_sdk_lifecycle_parity` (ask_user); SDK ask/resume versus durable park/inbox/rebuild; same kernel turn/operation, identical duplicate is noop, conflict rejected |
| Built-in `bash` | `tools/handlers/bash.py` | done | P `test_additional_builtin_parity[bash-arguments1]` covers foreground output; background lifecycle has separate rows |
| Built-in `check_background_command` | `tools/handlers/background.py` | done | T `test_background_process_restart_owner_parity`, `test_background_forbidden_owner_parity`; running/completed receipts and cross-owner denial |
| Built-in `stop_background_command` | `tools/handlers/background.py` | done | T `test_background_process_restart_owner_parity`, `test_background_unknown_stop_is_not_confirmed_parity`; confirmed stopped versus unconfirmed stopping/unknown |
| Built-in `read_image` | `tools/handlers/image.py` | done | P `test_additional_builtin_parity`, `test_multimodal_model_context_parity` cover URL and local PNG output |
| Built-in `activate_skill` | `tools/handlers/skills.py`, `skills/` | done | P `test_todo_and_skill_state_parity` checks activation and retained active skill state |
| Built-in `create_sub_task` | `tools/handlers/sub_agents.py`, `runtime/sub_task_manager.py` | done (intentional difference) | D `test_configured_child_atomic_sdk_restart_parity`, `test_configured_async_admission_status_and_late_delivery_parity`, `test_configured_child_argument_failure_parity`, `test_blocking_child_user_wait_is_durable_intentional_difference`; configured single/batch, synchronous/async, terminal delivery; a waiting child keeps its parent operation parked |
| Built-in `sub_task_status` | `tools/handlers/sub_task_status.py` | done | D `test_sub_task_status_records_projection_parity`, `test_sub_task_status_message_continuation_and_wait_projection_parity`; owner-scoped record projection, continuation/wait, stable inbox message replay; no second ledger |
| Tool allow/deny predicates and metadata denials | `run_config.py`, `tools/orchestrator.py` | done | T `test_tool_policy_matrix_parity`, `test_frozen_current_policy_dispatch_boundary`, `test_current_policy_predicate_rechecked_after_restart`; allow/deny lists, predicate, side effect/tag/cost/terminal denial and frozen/current dispatch restrictions |
| Approval modes default / always / never / on_request | `tools/orchestrator.py`, `approval.py` | done (intentional difference) | T `test_approval_mode_provider_parity`, `test_approval_broker_restart_session_and_conflicting_answer`, `test_approval_timeout_restart_parity`, `test_provider_decision_receipt_events_restart`; durable answers/session grants and absolute deadlines |
| Input guardrails allow/rewrite/block/require_approval | `runner.py`, `guardrails.py` | done | P `test_guardrail_parity`, `test_blocked_input_does_not_compile_providers` |
| Output guardrails allow/rewrite/block/require_approval | `runner.py`, `guardrails.py` | done | P `test_guardrail_parity` |
| Opt-in output validator accept/reject and one tools-free repair | `runner.py`, `output_validation.py` | done | P `test_output_validation_parity`; repair dispatch retained as an operation |
| Typed output coercion / repair exceptions / repair budget ledger | `runner.py`, `output_validation.py` | done (intentional difference) | C `test_typed_output_repair_ledger_restart_parity`, `test_output_repair_exceptions_parity`, `test_output_coercion_exception_becomes_durable_result`, `test_repair_usage_budget_is_logged_intentional_difference`, `test_uncertain_repair_does_not_retry_on_recovery`, `test_repair_missing_usage_policy_is_durable`; tools-free logged output_repair, typed JSON reconstruction and strict usage ledger |
| Runtime before_llm / after_llm hooks | `runtime/hooks.py` | done | P `test_llm_hook_parity` |
| Runtime before_tool_call / after_tool_call hooks | `runtime/hooks.py` | done | T `test_tool_hooks_state_approval_restart_parity`, `test_hook_patch_preserves_hidden_tool_boundary`; short circuit, serial state, approval and restart after prepared/result commits; recorded hooks are not replayed |
| Runtime before_memory_compact hook | `runtime/hooks.py`, `runtime/cycle_runner.py` | done | C `test_before_memory_hook_replacement_restart_parity`, `test_boundary_validation_rolls_back`; replacement and JSON state committed before compaction, retained decisions not rerun |
| AfterCycleHook continue / steer / deny / stop | `runtime/lifecycle.py`, `runtime/engine.py` | done | C `test_after_cycle_decision_snapshot_restart_parity`, `test_after_cycle_steering_boundary_parity`, `test_after_cycle_wait_and_native_finish_snapshot_restart`; snapshots rebuilt from adopted receipts, decisions replayed before dispatch |
| Context providers and prompt sections | `context_providers.py`, `runtime/compiler.py` | done | P `test_context_provider_parity`; immutable compiled prompt retained by F2a |
| MemoryProvider compact callbacks | `memory/provider.py`, `runtime/cycle_runner.py` | done | C `test_memory_provider_logged_callbacks_restart_parity`; accepted/rejected summary, before/after committed lifecycle and recovery at both boundaries |
| Session memory extraction/save/reload | `memory/session_memory.py`, `runtime/engine.py` | done | C `test_session_memory_logged_extract_save_reload_parity`; session_memory model receipt, structured state and disposable file projection; deleted file rebuilt before next-turn reload |
| Microcompaction, summary and prompt-too-long recovery | `memory/manager.py`, `runtime/cycle_runner.py` | done (intentional difference) | C `test_memory_compaction_runner_producer_parity`, `test_prompt_too_long_logical_cycle_and_tail_parity`; same MemoryManager producer, logical cycle and forced tail; duplicate rejected summary request reuses its logged receipt |
| Budget usage: cycles, tool counts, tokens/cache | `budget.py`, `runtime/token_usage.py` | done | P `test_budget_usage_parity` with measured provider usage |
| Budget limits: total and uncached input tokens | `budget.py`, `runtime/model_calls.py` | done | C `test_token_budget_boundaries_parity`, `test_missing_usage_budget_parity`; zero/equality/overshoot boundaries and STOP/CONTINUE missing accounting |
| Budget limits: total and per-name tool calls | `budget.py`, `runtime/engine.py` | done | C `test_tool_batch_admission_restart_parity`; whole batch atomically reserves names with the model receipt; dispatch checks wall/host and never counts reservations again |
| Budget limits: wall time, host cost, unavailable metrics | `budget.py` | done (intentional difference) | C `test_wall_time_budget_parity`, `test_host_cost_and_unavailable_metrics_parity`, `test_host_meter_failure_latch_restart_parity`, `test_lost_active_interval_is_unavailable_after_restart`; errors, units/currency, decreasing readings and retained unavailable state; strict missing active-interval recovery |
| Tracing processors and run/agent/tool spans | `tracing.py`, `runner.py:_RunTrace` | done (intentional difference) | C `test_trace_delivery_span_parity_and_no_recovery_duplicates`, `test_trace_ack_boundary_and_processor_failure`, `test_trace_processors_cannot_mutate_durable_output`; stable projected spans, separate committed traces cursor; at-most-once telemetry can be lost after acknowledgement |
| Live assistant/reasoning/tool stream deltas | `events.py`, `runtime/cycle_runner.py` | done | C `test_live_stream_deltas_and_durable_final_restart_parity`; volatile sink is optional, definitive content/reasoning/tool calls replay from records |
| Typed lifecycle RunEvents | `events.py`, `runtime/engine.py` | done (intentional difference) | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`, memory/budget/hook suites; stable typed identities and record sequence; child lifecycle belongs to parent admission records |
| Multimodal initial messages and tool image output | `types.py`, `llm/vv_llm_client.py`, `tools/function.py` | done | P `test_multimodal_model_context_parity` compares complete model-visible requests, image notifications and provider tool-call extensions |
| Multiple model endpoints, no stacked retries | `llm/vv_llm_client.py` | done (intentional difference) | C `test_endpoint_routing_preference_logged_attempts_parity`, `test_logged_endpoint_dispatch_rejects_routing_tamper`; actual Runner/VvLlmClient transport producer, preference/randomization, three endpoint attempts, frozen route and one request per attempt |
| Agent/run/provider model settings precedence | `model_settings.py`, `runner.py`, `runtime/compiler.py` | done | P `test_shared_state_and_settings_parity`, `test_provider_default_settings_parity`; transport retry override is intentional |
| Agent.as_tool | `agent.py`, `runner.py` | done (intentional difference) | D `test_sdk_children_never_recursive_runner_or_parent_lease`, `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`, `test_blocking_child_user_wait_is_durable_intentional_difference`; frozen child assembly, terminal-turn result, durable cancellation/wait |
| Configured sub-agents: policy/budget/workspace inheritance | `runtime/sub_task_manager.py`, `runtime/engine.py` | done | D `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`; SubAgentConfig, model binding, prompt/summary, policy/budget/workspace frozen at admission |
| BackgroundAgentTask start/poll/wait/cancel | `background_task.py` | done (intentional difference) | D `test_background_handle_start_poll_wait_cancel_reconstruction`, `test_sdk_parent_cancellation_cascades_after_reconstruction`; tool admission starts a child; reconstructed internal handles project the existing public snapshot type and submit inbox controls |
| Handoff and maximum-handoff enforcement | `handoffs.py`, `runner.py` | done (intentional difference) | D `test_handoff_durable_transfer_and_maximum_parity`, `test_handoff_target_validation_state_and_events_parity`; durable terminal child transfer, inherited state, target guardrails, record-derived count/frozen maximum, stable events |
| Child session atomic admission/delivery/cancellation | `runtime/sub_task_manager.py`, `runtime/backends/` | done (intentional difference) | D `test_configured_child_atomic_sdk_restart_parity`, `test_sdk_completion_delivery_failure_cut_and_replay_parity`, `test_sdk_parent_cancellation_cascades_after_reconstruction`, `test_completion_projection_uses_authenticated_turn_not_later_continuation`; SDK producers plus existing `session/test_children.py` fault cuts |
| shared_state between tools, persistence and reconstruction | `runtime/engine.py`, `tools/base.py` | done (intentional difference) | D `test_shared_state_host_binding_restart_intentional_difference`, `test_host_binding_json_boundary_rejects_invalid_state`; JSON-only durability, explicit process-local object bindings, deterministic missing binding, no pickle |
| Workspace local/memory/S3/streaming backend selection | `workspace/`, `runtime/engine.py` | done | D `test_s3_streaming_workspace_paired_producers`; existing S3 double, real read/write handlers, bounded large-object streaming and Runtime reconstruction with re-supplied backend; existing local/memory P producers |
| Bash background session restart and owner checks | `runtime/background_sessions.py`, `runtime/processes.py` | done | T `test_background_process_restart_owner_parity`; retained handle and unchanged turn owner reattach after rebuilding Runtime and SQL fold; process manager remains alive |
| no_tool_policy finish | `runtime/engine.py` | done | P `test_basic_runner_parity` |
| no_tool_policy continue and max_cycles stop | `runtime/engine.py` | done | P `test_continue_max_cycles_parity` compares one- and two-cycle stops |
| no_tool_policy wait_user | `runtime/engine.py` | done (intentional difference) | T `test_user_wait_sdk_lifecycle_parity` (no_tool); durable turn-level wait with zero fabricated tool operations; reply resumes the same turn |
| tool_use_behavior stop_on_first_tool / stop_at_tool_names | `runtime/tool_call_runner.py` | done | P `test_tool_stop_parity`; T `test_tool_stop_pending_batch_native_finish_parity`, `test_tool_stop_error_result_parity`; pending calls close with the same skipped result and native FINISH is respected |
| Cancellation: cooperative / unknown / descendants | `runtime/cancellation.py`, `run_handle.py` | done (intentional difference) | T `test_cancellation_control_result_event_parity`, `test_cancellation_descendant_control_result_events_parity`; SDK cancellation/exception versus durable confirmed/unknown receipts, targeted descendant controls and terminal events |
| Completed RunResult fields and cycle/tool history | `result.py`, `runner.py` | done (intentional difference) | C typed output/repair/budget/hook suites, `test_wait_result_fields_parity`, `test_completed_result_compaction_flags_are_per_cycle`, `test_endpoint_routing_preference_logged_attempts_parity`; errors, partial output, waits, logical cycles, repair calls and fixed terminal prefix |
| Event store replay and durable consumer acknowledgement | `event_store.py`, `run_handle.py` | done | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`; RunEventStore bridge, SQL cursor rollback/replay, child query and identical append assertion |
| Interactive steer/follow-up/resume/archive/close | `interactive.py`, `sessions/`, `session/interactive.py` | done (intentional difference) | `test_interactive_real_same_turn_user_reply`, `test_interactive_live_steer_and_durable_follow_up`, `test_interactive_child_wait_reply_keeps_child_identity`, `test_kernel_file_facade_rebuild_resume_and_control_identity`, `test_interactive_close_during_model_call_is_idempotent`; records/inbox facade, same-turn/child reply, durable archive/close |
| CLI single-run, stream and persistent sessions | `cli.py`, `session/surfaces.py` | done | `test_cli_real_single_run_and_stream_channels[current/kernel]`, `test_cli_kernel_persistent_session_survives_owner_restart`; SQLite memory/file owners, unchanged default and output channels |
| App Server thread/turn/approval/replay/non-text input | `app_server/server.py`, `session/app_server.py` | done (intentional difference) | Existing thread/turn/approval/replay suites parametrized; `test_kernel_process_restart_retains_calls_approval_image_and_client_cursor`, `test_child_wait_user_is_exposed_and_reply_targets_child`, `test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items`; no second ledger; exact C1 wire items in F2d-4 report |
| App Server model/list, schema and TypeScript export | `app_server/protocol/`, `app_server/schema.py` | done | `test_model_list_forwards_optional_filters_and_emits_canonical_superset`, `test_schema_export_request_returns_json_and_typescript_bundles[current/kernel]`; full result/bundle identity; v23 export remains unchanged |

## Comparison projection

T denotes `tests/session/test_tools_control_parity.py`. Persistent T scenarios run
on PostgreSQL, SQLite files and SQLite `:memory:` using the existing store fixtures;
ordinary policy/exposure/stop scenarios use the paired in-memory P harness.


Every paired P scenario independently creates the scripted provider, Agent and
RunConfig and runs the current public `Runner.run_sync` or `kernel.drive` on a
fresh in-memory SQLite database. It compares final output, ordered tool-call
IDs/content, shared lifecycle event types, and budget usage when enabled. JSON
shared state is compared too. Shared-state restart additionally discards the
SQL prefix cache and constructs a new Runtime after a committed tool result.
Image context tests also compare full model-visible requests, including the canonical system message, initial images, tool image notifications and provider tool-call extension fields. The result projection test compares the existing result fields, raw cycles,
new items, shared state and resolved model on a normal completed tool run.

The comparison deliberately does **not** establish full RunEvent parity:

- Event/run/trace/operation IDs and timestamps are identities from different
  producers and are not compared as literal strings.
- Kernel atomically plans a whole tool batch with its model receipt. Runner
  plans each tool just before executing it. Per-tool planned/started/completed
  ordering and the remaining shared execution sequence are separate checks.
- F2d-2 adds paired agent/cycle/diagnostic/budget/memory/delegation producers and typed replay. The earlier P common-event filter alone does not prove them; C supplies their separate evidence.
- Wall-clock `elapsed_ms` is excluded from numerical usage equality. All other
  usage fields in the tested scenario must match; this is not a wall-budget
  boundary test.
- Input blocking precedes public Runner compilation. Kernel retains a blocked
  turn definition without invoking instruction/context providers, and projects
  only the failed event, with no model dispatch.
- Runner's host repair callback has no v23 model-call ledger discriminator.
  Kernel logs one `output_repair` dispatch for recovery. The repair test asserts
  those two additional model lifecycle events before comparing the remainder.
  F2d-2 closes repair accounting. The existing v23 public enum remains
  agent_cycle; event metadata and the session result ledger preserve output_repair
  explicitly. A new public discriminator belongs to C1.

## Internal persistence and execution changes

All standard registered tools use the existing planner and ToolOrchestrator;
there is no hand-maintained three-tool allowlist. Registered enabled tools keep
their normal exposure rules. Before-LLM patches are frozen in the planned request;
after-LLM and after-tool results are retained in the ordinary operation result.
JSON shared state is stored in the operation receipt's opaque usage object as
`session_shared_state`, outside model-visible ToolExecutionResult metadata.
Unknown effects do not invent a successful state snapshot.

Output repair is a tools-free model-purpose operation in the same driver. It
never calls Runner execution and never retries an unknown host repair callback.
Model requests freeze the existing preferred/randomized endpoint order. Each
logged attempt selects one endpoint; each VvLlmClient call performs one transport
attempt. Durable success restores preference on a later turn. Routing parity and
attempt isolation are proved in C; credentials never enter the frozen binding.

The public Runner is reused only for existing configuration and pure output
helpers. No Runner execution method is called by the internal kernel. Removing
that helper ownership dependency belongs to the cut-over extraction work.

## Short-run benchmark

Run `uv run python scripts/session_kernel_overhead.py --runs 200 --output report.json`.
The benchmark includes SQLite schema creation, session admission, drive, final
state read and connection/thread cleanup. The ten-turn case means ten successive
user turns in one session, compared with ten Runner runs using MemorySession.
The two-tool case includes the model's final response after both tool results.
The children case runs configured create_sub_task, separately drives its child after
releasing the parent lease, commits terminal inbox delivery and resumes the parent;
the Runner reference uses its real configured-child producer.
Start/cancel uses an explicit provider-entry barrier on both paths, submits
cancellation, releases the same cooperative scripted provider and waits for
completion. A timeout or wrong terminal state fails the benchmark.

Each path warms up before at least 200 measured samples. p95 is the nearest-rank
95th percentile. Added p95 is `kernel p95 - Runner p95` for the same scenario;
it is not a paired-difference percentile. Linux current RSS is measured after
GC before/after each path's measured batch; it is not a peak-RSS measurement.
All live thread objects are compared before and after the batch, not only their
count. The process exits nonzero when a single-turn scenario exceeds 50 ms added p95,
the ten-turn scenario exceeds 80 ms added p95 (8 ms per turn amortized), or
any scenario leaves a new kernel thread alive. No daemon-thread timeout is counted as a
confirmed stop.

F2b profiling identified Python character-by-character JCS quoting, repeated log
payload deepcopy/encoding, and duplicate token counting. Its shared path used the stdlib string encoder with strict surrogate rejection, JSON-tree
cloning for detached store reads, validated append bytes for fold/write/digest,
a semantic record-ID commit identity, and one token count per microcompaction
pass. It retains validation, lease/CAS checks, commit replay byte comparisons,
cache invalidation and the same driver. No separate fast executor exists.

## F2b validation results (2026-10-08)

The candidate remains **not ready for F3**: 24 done / 26 partial / 13 missing.
All code/test gates below passed. The capability and short-run performance gates
did not pass. There were no commits or changes to public/default entry points.

| Gate | Result |
| --- | --- |
| `uv run python scripts/contract_snapshot.py check` | PASS: contract 23.0.0, 55 fixture files, unchanged manifest |
| `uv run ruff format --check .` | PASS: 375 files |
| `uv run ruff check .` | PASS |
| `uv run ty check` | PASS |
| `uv run pytest tests/session -q` with local PostgreSQL | PASS: 486 tests, 340.86 seconds |
| `uv run pytest` with dedicated real Redis and local PostgreSQL | PASS: 2,903 passed, 20 skipped, 18 warnings; 494.33 seconds |
| Capability matrix | FAIL: 39 partial/missing rows |
| Short-run added p95 <= 50 ms | FAIL: 3 of 4 scenarios exceed the target |

The 20 full-suite skips are nine inapplicable non-Redis fixture variants of
Redis-only tests, six opt-in remote-model tests, four opt-in cross-runtime probes,
and one unavailable directory-symlink test. The applicable real-Redis variants
ran; PostgreSQL cases were not skipped. Warnings are the existing distributed
tests' multithreaded-fork deprecation warnings. The dedicated Redis instance was
stopped after the suite. No Rust/cargo or production operations ran.

Forty new parametrized cases compare Runner and SQLite, two test kernel endpoint
attempt isolation and state reconstruction, and five check JCS encoding. The
compaction suite retains its full history-preservation checks and now expects
the compiled system prompt, consistent with Runner.

Final benchmark: 200 measured runs per path/scenario after 10 warmups. Values are
milliseconds; ten turns is the aggregate of ten successive user turns. Exact
numbers are in `docs/session-kernel-overhead.json`. RSS values are KiB changes
after GC for the whole measured batch, shown as Runner / kernel.

| Scenario | Runner p50 / p95 | Kernel p50 / p95 | Added p95 | RSS delta (R / K), KiB | Threads after (R / K) |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_tool | 4.38 / 6.69 | 45.39 / 54.03 | 47.34 | +0 / +0 | 1 / 1 |
| two_tools | 6.23 / 8.85 | 94.33 / 121.74 | 112.90 | +0 / +0 | 1 / 1 |
| ten_turns | 39.97 / 47.21 | 998.91 / 1230.80 | 1183.59 | -940 / +16 | 1 / 1 |
| start_cancel | 6.38 / 8.33 | 53.36 / 65.95 | 57.62 | +248 / +72 | 1 / 1 |

All eight measured path/scenario groups reported no newly surviving threads.
The benchmark exited 1 for the three timing failures. No-tool added p95 passed
in this measurement; this is not a claim that arbitrary providers, long-lived
background processes or uncooperative handlers cannot retain threads.


## F2c performance results (2026-10-08)

The short-run timing gate remains **failed**: `two_tools` and `ten_turns` exceed
their limits. The internal/default boundary and the capability matrix above are
unchanged. The benchmark still includes admission, all durable writes, final
state verification, connection cleanup and thread cleanup on the shared driver.

Records retain validated canonical bytes and digests. Store reads reuse only an
exact session/sequence/byte match after checking the stored digest. Committed
receipt records update the driver's fold directly; external tails use bounded
reads. The SQL lease and CAS checks remain in force. The same validated prefix
can rebind to a new lease epoch after its persisted head digest is checked.

`tests/session/test_record_validation_cache.py` adds 59 cases covering retained
bytes, mutation isolation, lease rebinding, missing/nonconsecutive tails, compiler
versus jsonschema equivalence, and independent PG/SQLite writers with tampered
schema, embedded digest or stored digest. The unchanged record/invalid-JSON tests
also pass with the original jsonschema validation path (69 cases) and with the
compiled path (the same 69 plus the 59 new cases). Existing test files are unchanged.

| Gate | Result |
| --- | --- |
| Contract snapshot | PASS: 23.0.0, 55 fixture files, unchanged manifest |
| Ruff format / check and ty | PASS |
| `uv run pytest tests/session -q`, real local PG | PASS: 545 tests, 250.98 seconds |
| Full pytest, real local Redis and PG | PASS: 2,962 passed, 20 skipped, 18 warnings; 415.75 seconds |
| M6, 5k/20k, one sample, `--assert-capacity` | PASS, including 1k/10k catalog scans |
| Short-run timing, 200 runs per path/scenario | FAIL: two of four scenarios |

The full-suite skips retain the F2b breakdown above. All applicable Redis and PG
variants ran. The warnings are the existing distributed multithreaded-fork
warnings; remote-model and cross-runtime probes remain opt-in.

Short runs use 10 warmups and 200 measured samples for each path/scenario.
Values are milliseconds. F2b is the recorded measurement above; F2c is the new
measurement in `session-kernel-overhead-f2c.json`.

| Scenario | F2b added p95 | F2c Runner p50 / p95 | F2c kernel p50 / p95 | F2c added p95 | Limit | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no_tool | 47.34 | 4.08 / 4.81 | 28.31 / 30.08 | 25.27 | 50 | PASS |
| two_tools | 112.90 | 6.22 / 6.93 | 60.69 / 66.69 | 59.76 | 50 | FAIL |
| ten_turns | 1183.59 | 42.95 / 46.62 | 299.51 / 316.03 | 269.41 | 100 | FAIL |
| start_cancel | 57.62 | 5.48 / 6.32 | 29.73 / 33.27 | 26.95 | 50 | PASS |

Ten-turn added p95 is 26.94 ms per turn amortized, above the 10 ms limit. The
benchmark exits 1. All eight groups finish with one live thread and no newly
surviving threads. Kernel RSS deltas after GC are 0 / +16 / 0 / +100 KiB in table
order; Runner deltas are 0 / 0 / -940 / +236 KiB. These are measured batch deltas,
not peak RSS or a guarantee about uncooperative external handlers.

The M6 script retains the original bounded 1 KiB receipt workload and capacity
assertions. Both versions below run on real PostgreSQL with one sample at each
size. The baseline selects the unchanged F2b source at `31e656e`.

| Records | F2b cold drive ms | F2c cold drive ms | F2b steady append ms | F2c steady append ms |
| ---: | ---: | ---: | ---: | ---: |
| 5000 | 1912.77 | 762.24 | 21.98 | 17.43 |
| 20000 | 8344.44 | 3188.77 | 266.36 | 37.93 |

F2b fails the 20k steady-append 50 ms assertion; F2c passes it. Both sizes retain
zero extra provider calls and zero lease-loss failures. F2c full catalog scans
at 1k/10k sessions take 22.09 / 250.72 ms, below the 1-second gate.

The requested cProfile summaries are `/tmp/f2c-profile-before.txt` and
`/tmp/f2c-profile-after.txt` (10 `two_tools` kernel runs after warmup). Total
profiled time falls from 2.293 to 1.426 seconds. Calls below are per run:

| Call | F2b | F2c |
| --- | ---: | ---: |
| `canonical_json_bytes` | 291 | 219 |
| jsonschema `validate` | 253 | 0 |
| Driver `refresh` | 23 | 10 |
| Store `read_state` | 15 | 2 |
| Store `read` | 38 | 12 |

The compiled closed-shape checks replace the generic validator on valid records;
invalid values still go through its diagnostics. Remaining profile costs include
request/definition JCS encoding, tool-schema copies and context preparation.
These measurements do not establish the required short-run acceptance target.

## F2d tools/control comparison boundaries (2026-10-08)

The eleven requested tools/control rows are closed. The other capability rows
remain open as listed above; these tests do not authorize an F3 cut-over.

Intentional differences for review:

- User waits keep the existing kernel turn and interaction operation. SDK
  `Runner.resume` admits another run and carries the earlier WAIT_RESPONSE tool
  message plus a user message. Kernel consumes an identified inbox reply into
  exactly one tool result. Compare the waiting stage, delivered response and
  final output, then separately assert the kernel identity and replay fences.
  An undispatched ask has no external-effect started event.
- `no_tool_policy=wait_user` is a logged `turn_parked` record, not a tool. It
  projects wait/running state events. Its reply appends a user message to the
  existing turn; SDK resumes another run. There is no manufactured call ID.
- Kernel parks approval before calling its provider. Even immediate decisions
  are retained as inbox answers before effects. `allow_session` is recovered
  from the log across Runtime/Broker replacement. Runner retains that grant in
  its process-local broker. Timeout uses the original absolute store deadline
  and the shared orchestrator's exact error result. The deadline includes provider
  decision time: with zero timeout Runner can accept an immediate provider allow,
  while kernel times out before invoking it. This strict expiry difference is
  paired in `test_approval_absolute_deadline_includes_provider_time`. Provider decision reason
  and metadata are retained and projected. If `should_request` returns false,
  kernel still retains its pre-dispatch park/allow decision; Runner omits those
  approval events. Tool effect, result and provider invocation counts agree.
  Broker session flags alone do not authorize kernel effects; only applied log
  answers grant a session allowance. This prevents a late/rejected broker answer
  from leaving an authorization behind.
- Runner's function-tool cancellation currently raises `CancelledError` and
  produces no terminal cancelled event on that path. Kernel commits the targeted
  control, a confirmed cooperative-stop receipt or explicit unknown attempt,
  and a cancelled terminal event. A thread that has not stopped is never called
  stopped. The test compares cancelled control/handle state and dispatched-tool
  count, and asserts the different result/event projections explicitly.
  Descendant cancellation uses the SDK child producer as the reference and
  kernel host child admission as the candidate; public child SDK assembly remains
  the separate open matrix row.

Other normalized comparisons:

- A complete model receipt atomically admits all kernel tool plans. When a tool
  finishes, those unused plans require durable skipped results. Runner emits
  no lifecycle events for its unadmitted skipped calls. Compare all ordered tool
  results/output; compare effects and the skipped lifecycle difference separately.
- Background process receipts normalize generated session IDs, elapsed time and
  the terminal text/ongoing JSON envelope. Status, output and exit code must
  agree. Both paths use the existing process manager and actual shell processes.
  Drive/Runtime reconstruction reuses the retained session handle and stable
  task/workspace owner. Restarting the operating-system worker/process manager
  remains unsupported by its process-local handles; no PID-only adoption is
  claimed. Cross-owner access is rejected before observing/stopping a process.
- Active-turn schema/capability drift is rejected rather than applying a new
  declaration to an old authorization. A new turn sees the changed schema.
  Callable policy predicates are host bindings reevaluated at dispatch; frozen
  serializable denials remain restrictive when current policy loosens.

Before-tool hooks are prepared individually after the preceding tool result,
using its retained JSON state. `op_prepared` freezes the patched call, capability,
provider binding, idempotency key, short-circuit result and hook state before
approval or effects. Recovery reuses it. After-tool hooks and stop behavior are
retained in the definitive result; known results never rerun hooks. An unrecorded
callback interrupted before its commit remains outside exactly-once claims. Hook
renaming retains the target capability from the frozen exposed definition; it
cannot make a hidden tool executable or bypass its normal policy denial.

Approval deadline boundaries use a controlled SQL clock to avoid relying on
PostgreSQL operations finishing inside a millisecond test sleep. Storage, leases,
parked deadlines, inbox consumption and Runtime reconstruction still use the
actual PostgreSQL and SQLite stores.

## F2d-1 validation results (2026-10-08)

The eleven requested rows are closed: 35 done (including four intentional
differences), 16 partial, 12 missing overall. F3 remains blocked.
The detailed Chinese report is `session-kernel-f2d-tools-control-report.md`;
the raw 200-run result is `session-kernel-overhead-f2d-tools-control.json`.

| Gate | Result |
| --- | --- |
| Contract snapshot | PASS: 23.0.0, 55 fixture files, unchanged manifest |
| Ruff format / check and ty | PASS: 381 Python files formatted |
| `uv run pytest tests/session -q`, local PostgreSQL | PASS: 816 tests, no skips, 261.93 seconds |
| Full pytest, real Redis 6399 DB 15 and PostgreSQL | PASS: 3291 passed, 20 skipped, 18 warnings, 415.92 seconds |
| Short-run overhead, 10 warmups / 200 runs per path/scenario | PASS: all targets; no leaked threads |
| Internal/default boundary and diff whitespace | PASS: no public/default/fixture changes; no commits |
| Redis cleanup | PASS: gate-owned server stopped |

The new T suite contains 25 test functions / 222 parameterized cases. The skip
and warning breakdown matches the F2b/F2c breakdown: all applicable Redis and PG
variants ran; live providers and cross-runtime probes remain opt-in.

The following table is the historical F2d-2 single run, superseded by F2d-2b.
Values are milliseconds; added p95 is kernel p95 minus Runner p95.

| Scenario | Runner p50 / p95 | Kernel p50 / p95 | Added p95 | Limit | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| no_tool | 3.40 / 4.07 | 14.07 / 16.11 | 12.04 | 50 | PASS |
| two_tools | 5.25 / 5.84 | 25.06 / 25.94 | 20.10 | 50 | PASS |
| ten_turns | 33.89 / 36.81 | 108.34 / 125.47 | 88.67 | 100 | PASS |
| start_cancel | 4.62 / 5.20 | 14.76 / 16.32 | 11.12 | 50 | PASS |

The first F2d measurement exceeded the ten-turn limit at 104.70 ms added p95.
Profiling identified duplicate definition hashing at each admission; the driver
now uses Runtime's already-computed digest. Record validation still verifies that
digest. Existing cache invalidation/tamper suites passed (108 tests), and the
same candidate then passed the complete gates above. No benchmark workload,
sampling, cleanup, lease/CAS or record-validation boundary was removed.

## F2d-2 validation results (2026-10-08)

The fifteen memory/budget/event/model rows requested for F2d-2 are closed, including
seven intentional differences. Inventory is 50 done / 6 partial / 7 missing;
F3 remains blocked. Paired C producers and reconstruction run on real PostgreSQL,
SQLite files and SQLite `:memory:`. C contains 34 test functions / 358 cases.
The full Chinese evidence, reviewer decisions, remaining inventory, cProfile
before/after entries and capacity tables are in
[session-kernel-f2d-memory-budget-events-report.md](session-kernel-f2d-memory-budget-events-report.md).

Whole tool batches reserve admission names with their model receipt and plans;
dispatch checks dynamic metrics without repeating the reservation. Recorded hook
and compaction decisions are reused. Session memory and output repair are logged
model operations. Stream deltas remain volatile; durable receipts rebuild finals.
Typed event replay uses existing consumer cursors. Tracing acknowledges before
processor delivery to prevent recovery duplicates; this at-most-once telemetry
can be lost after acknowledgement and remains an explicit reviewer decision.

| Gate | Result |
| --- | --- |
| Contract snapshot | PASS: 23.0.0, 55 fixture files, unchanged manifest |
| Ruff format / check and ty | PASS: 386 Python files |
| `uv run pytest tests/session -q`, real local PG | PASS: 1221 passed in 276.52s (0:04:36) |
| Full pytest, real Redis 6400 DB 15 and local PG | PASS: 3696 passed, 20 skipped, 18 warnings in 431.63s (0:07:11) |
| Overhead, 10 warmups / 200 samples per path/scenario | PASS: three independent F2d-2b processes, each single-turn <=50 ms and ten turns <=80 ms; no leaked threads |
| M6 5k/20k, one sample, `--assert-capacity` | PASS: cold drive, steady append and 1k/10k catalog scans |
| Scope / diff / cleanup | PASS: no default wiring, public exports, lock/fixture or Rust changes; HEAD 9c5eafe, no new commits; gate-owned Redis stopped |

The full-suite skip/warning breakdown matches F2d-1. All applicable Redis and PG
variants ran. Shared MemoryProvider callback helpers moved to the retained memory
module without changing Runner behavior; session no longer depends on CycleRunner.

The following table is the historical F2d-2 single run, superseded by F2d-2b.
Values are milliseconds; added p95 is kernel p95 minus Runner p95.

| Scenario | Runner p50 / p95 | Kernel p50 / p95 | Added p95 | Limit | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| no_tool | 3.51 / 4.12 | 13.34 / 14.52 | 10.40 | 50 | PASS |
| two_tools | 5.58 / 6.42 | 26.48 / 35.10 | 28.69 | 50 | PASS |
| ten_turns | 37.94 / 46.51 | 106.48 / 113.14 | 66.63 | 80 | PASS |
| start_cancel | 4.81 / 5.68 | 15.99 / 17.21 | 11.54 | 50 | PASS |

The A-complete pre-optimization candidate measured 99.061 ms ten-turn added p95.
F2d-2's single 66.626 ms result did not establish independent-run headroom:
reviewer reruns measured 84.4, 104.5 and 95.1 ms. F2d-2b's three independent
200-run / 10-warmup processes measure 62.082, 68.990, 53.798 ms,
with a minimum 11.010 ms headroom. The linked report retains added p50
and p95 for every scenario and all three before/after runs, GC pause evidence,
method timing distributions, profile call counts, and the renewed M6 capacity data.
The same driver shares validated schema graphs and JCS bytes, reuses committed
completion receipts and immutable definition JCS fragments, groups record encoding,
copies only state in host snapshots, and folds endpoint preference instead of
scanning the log. Schema validation, embedded and
storage digests, output isolation, lease/CAS/fencing and the original benchmark
workload remain in force. GC is unchanged; the p95 gate is not a max/p99 guarantee.

F2d-3 row mapping, C1 candidates and three independent overhead runs are recorded in
[`session-kernel-f2d-children-report.md`](session-kernel-f2d-children-report.md).

## F2d-4 surface validation results (2026-10-09)

The final four surface rows are closed: **63 done, 0 partial, 0 missing**.
The public default remains unchanged; one private `_kernel` owner selects the
interactive, CLI and App Server producer. The App Server has no second thread
ledger. Restart, retained approval ownership, child WAIT_USER routing, image
replay, cursor replay and archive/close identity are covered in the existing
surface suites. Model/list and the complete schema/TypeScript bundles remain
identical. The explicit C1 wire items and F3 deletion/extraction inventory are in
[`session-kernel-f2d-surfaces-report.md`](session-kernel-f2d-surfaces-report.md).

| Gate | Result |
| --- | --- |
| Contract snapshot | PASS: 23.0.0, 55 fixture files, unchanged lock and manifest |
| Ruff format / check and ty | PASS: 393 Python files formatted |
| `uv run pytest tests/session -q`, real local PG | PASS: 1397 passed in 305.52s |
| Full pytest, real Redis 6400 DB 15 and local PG | PASS: 3944 passed, 20 skipped, 18 warnings in 449.95s |
| Overhead, three independent `--runs 200 --warmup 10` processes | PASS: all six scenarios, no leaked threads |
| M6 5k/20k history, 1k/10k sessions, one sample, `--assert-capacity` | PASS: cold drive 503.28 / 2059.30 ms; steady append 9.70 / 15.44 ms; catalog scan 18.59 / 226.15 ms |
| Scope and cleanup | PASS: default/exports/lock/fixtures/Rust unchanged, no commits; gate-owned Redis stopped |

Added p95 values in milliseconds, from the three independent F2d-4 JSON artifacts:

| Scenario | Run 1 | Run 2 | Run 3 | Limit |
| --- | ---: | ---: | ---: | ---: |
| ten_turns | 56.577 | 59.842 | 58.007 | 80 |
| children | 43.996 | 43.316 | 44.823 | 50 |
| app_server_turn | 25.869 | 24.520 | 26.006 | 50 |

The detailed report retains current/kernel p50 and p95 for every scenario and
run. Live-provider and cross-runtime opt-in probes were not run; applicable
Redis/PG cases ran. M6 is one bounded sample, not a p95 or soak claim. Canonical
C1 adoption and removal of shared dependencies from the old stack remain F3 work.
