# Session kernel capability matrix

Every public entry point (Runner, interactive sessions, the CLI and the App
Server) runs on the same session kernel. This page maps each capability to the
module that owns it and to the tests that prove it. Use it to find where to
change a behavior and which tests to run afterwards.

The "Difference from 0.21.x" column is filled in only where the kernel behaves
differently from the in-process runtime of package 0.21.x (contract 23). These
differences are deliberate. The contract fixtures and public API v11 define the
supported behavior.

## How to read the table

Owner paths are relative to `src/vv_agent/`. A letter before a test name gives
its file under `tests/session/`:

| Letter | File |
| --- | --- |
| P | `test_runner_parity.py` |
| T | `test_tools_control_parity.py` |
| C | `test_capability_parity.py` |
| D | `test_delegation_parity.py` |
| R | `test_recovery_matrix.py` |

Tests without a letter live in the file named in the row. Tests that use a
persistent store run on PostgreSQL, on a SQLite file and on SQLite `:memory:`.
Tests that kill a process run on durable stores only.

Most rows are paired tests. A paired test builds the same run twice, once
through the public Runner and once by driving the kernel directly, then
compares the output, tool calls, projected events, usage and JSON shared state.
Identifiers and clock values are compared only where the contract fixes them.
Recovery tests rebuild the `Runtime` and the store, then check that callbacks,
receipts, authorization, child identity, budgets and same-turn replies were
kept.

## Capabilities

| Capability | Owner module | Tests | Difference from 0.21.x |
| --- | --- | --- | --- |
| Basic synchronous completion without tools | `runner.py`, `session/kernel.py` | P `test_basic_runner_parity` | |
| FunctionTool and serial multi-tool execution | `tools/function.py`, `tools/orchestrator.py` | P `test_basic_runner_parity` | |
| ToolRegistry factory and dynamically registered direct executors | `runner.py`, `tools/registry.py`, `runtime/tool_planner.py` | P `test_dynamic_enabled_and_registry_factory_parity` | |
| FunctionTool `is_enabled` as a boolean or callback | `runtime/compiler.py` (`tool_is_enabled`) | P `test_dynamic_enabled_and_registry_factory_parity`, `test_disabled_tool_never_executes` | |
| Hidden tools, dynamic schema changes and exposure limits | `tools/executor.py`, `tools/registry.py` | T `test_hidden_tool_exposure_parity`, `test_dynamic_schema_new_turn_parity`, `test_dynamic_schema_active_turn_rejected`, `test_hook_patch_preserves_hidden_tool_boundary`. A hidden tool cannot be called. A new turn picks up schema changes; an active turn rejects them. | |
| Built-in `read_file` | `tools/handlers/workspace_io.py` | P `test_workspace_tool_parity[read_file-arguments0]` | |
| Built-in `write_file` | `tools/handlers/workspace_io.py` | P `test_workspace_tool_parity[write_file-arguments1]` | |
| Built-in `edit_file`, including the prior-read baseline | `tools/handlers/workspace_io.py` | P `test_workspace_tool_parity[edit_file-arguments2]` | |
| Built-in `find_files` | `tools/handlers/search.py` | P `test_workspace_tool_parity[find_files-arguments3]` | |
| Built-in `search_files` | `tools/handlers/search.py` | P `test_workspace_tool_parity[search_files-arguments4]` | |
| Built-in `file_info` | `tools/handlers/workspace_io.py` | P `test_additional_builtin_parity[file_info-arguments0]` | |
| Built-in `todo_write` | `tools/handlers/todo.py` | P `test_todo_and_skill_state_parity`, with a fixed clock and a TODO id chosen by the caller | |
| Built-in `ask_user` and host interaction replies | `tools/handlers/control.py`, `interaction.py` | T `test_user_wait_sdk_lifecycle_parity` (ask_user case) | The wait is durable. The turn parks, and the reply arrives as an inbox item that resumes the same turn and operation, even after a rebuild. An identical duplicate reply does nothing; a conflicting one is rejected. |
| Built-in `bash` | `tools/handlers/bash.py` | P `test_additional_builtin_parity[bash-arguments1]` (foreground output; background commands have their own rows) | |
| Built-in `check_background_command` | `tools/handlers/background.py` | T `test_background_process_restart_owner_parity`, `test_background_forbidden_owner_parity`. Covers running and completed receipts and denial to other owners. | |
| Built-in `stop_background_command` | `tools/handlers/background.py` | T `test_background_process_restart_owner_parity`, `test_background_unknown_stop_is_not_confirmed_parity`. A confirmed stop reports stopped; an unconfirmed one reports stopping or unknown. | |
| Built-in `read_image` | `tools/handlers/image.py` | P `test_additional_builtin_parity`, `test_multimodal_model_context_parity` (URL and local PNG) | |
| Built-in `activate_skill` | `tools/handlers/skills.py`, `skills/` | P `test_todo_and_skill_state_parity` (activation and the retained active skill) | |
| Built-in `create_sub_task` | `tools/handlers/sub_agents.py`, `session/delegation.py` | D `test_configured_child_atomic_sdk_restart_parity`, `test_configured_async_admission_status_and_late_delivery_parity`, `test_configured_child_argument_failure_parity`, `test_blocking_child_user_wait_is_durable_intentional_difference`. Covers single and batch, synchronous and async, and delivery of the final result. | While a child waits for the user, the parent operation stays parked. |
| Built-in `sub_task_status` | `tools/handlers/sub_task_status.py` | D `test_sub_task_status_records_projection_parity`, `test_sub_task_status_message_continuation_and_wait_projection_parity`. Status is projected from the owner's records; continuation and wait replay stable inbox messages. There is no second ledger. | |
| Tool allow and deny predicates, metadata denials | `run_config.py`, `tools/orchestrator.py` | T `test_tool_policy_matrix_parity`, `test_frozen_current_policy_dispatch_boundary`, `test_current_policy_predicate_rechecked_after_restart`. Covers allow and deny lists, predicates, denial by side effect, tag, cost or terminal flag, and frozen versus current policy at dispatch. | |
| Approval modes `default`, `always`, `never`, `on_request` | `tools/orchestrator.py`, `approval.py` | T `test_approval_mode_provider_parity`, `test_approval_broker_restart_session_and_conflicting_answer`, `test_approval_timeout_restart_parity`, `test_provider_decision_receipt_events_restart` | Answers and session-wide grants are stored durably, and approval deadlines are absolute times. |
| Input guardrails: allow, rewrite, block, require approval | `runner.py`, `guardrails.py` | P `test_guardrail_parity`, `test_blocked_input_does_not_compile_providers` | |
| Output guardrails: allow, rewrite, block, require approval | `runner.py`, `guardrails.py` | P `test_guardrail_parity` | |
| Opt-in output validator with one tools-free repair | `runner.py`, `output_validation.py` | P `test_output_validation_parity`. The repair call is kept as a logged operation. | |
| Typed output coercion, repair exceptions and the repair budget | `runner.py`, `output_validation.py` | C `test_typed_output_repair_ledger_restart_parity`, `test_output_repair_exceptions_parity`, `test_output_coercion_exception_becomes_durable_result`, `test_repair_usage_budget_is_logged_intentional_difference`, `test_uncertain_repair_does_not_retry_on_recovery`, `test_repair_missing_usage_policy_is_durable` | The repair is a logged, tools-free `output_repair` call, and its token usage is counted against the budget. Typed output is rebuilt from JSON. A repair with an uncertain outcome is not retried on recovery. |
| Runtime `before_llm` and `after_llm` hooks | `runtime/hooks.py` | P `test_llm_hook_parity` | |
| Runtime `before_tool_call` and `after_tool_call` hooks | `runtime/hooks.py` | T `test_tool_hooks_state_approval_restart_parity`, `test_hook_patch_preserves_hidden_tool_boundary`. Covers short circuits, serial state, approval, and restart after the prepared and result commits. Recorded hook decisions are not replayed. | |
| Runtime `before_memory_compact` hook | `runtime/hooks.py`, `session/runtime.py` | C `test_before_memory_hook_replacement_restart_parity`, `test_boundary_validation_rolls_back`. The replacement and JSON state are committed before compaction; retained decisions do not run again. | |
| AfterCycleHook: continue, steer, deny, stop | `runtime/lifecycle.py`, `session/kernel.py` | C `test_after_cycle_decision_snapshot_restart_parity`, `test_after_cycle_steering_boundary_parity`, `test_after_cycle_wait_and_native_finish_snapshot_restart`. Snapshots are rebuilt from adopted receipts, and decisions are replayed before dispatch. | |
| Context providers and prompt sections | `context_providers.py`, `runtime/compiler.py` | P `test_context_provider_parity`. The compiled prompt is frozen when the turn is admitted. | |
| MemoryProvider compaction callbacks | `memory/provider.py`, `session/runtime.py` | C `test_memory_provider_logged_callbacks_restart_parity`. Covers accepted and rejected summaries, the committed before and after steps, and recovery at both points. | |
| Session memory extraction, save and reload | `memory/session_memory.py`, `session/kernel.py` | C `test_session_memory_logged_extract_save_reload_parity`. The `session_memory` model receipt and structured state are logged; the file on disk is a disposable copy that is rebuilt before the next turn reloads it. | |
| Microcompaction, summaries and prompt-too-long recovery | `memory/manager.py`, `session/runtime.py` | C `test_memory_compaction_runner_producer_parity`, `test_prompt_too_long_logical_cycle_and_tail_parity` | The same MemoryManager produces both runs, with the same logical cycle and forced tail. A repeated summary request that was rejected reuses its logged receipt. |
| Budget usage: cycles, tool counts, tokens and cache | `budget.py`, `runtime/token_usage.py` | P `test_budget_usage_parity`, with measured provider usage | |
| Budget limits: total and uncached input tokens | `budget.py`, `session/runtime.py` | C `test_token_budget_boundaries_parity`, `test_missing_usage_budget_parity`. Covers zero, equal and overshoot limits, and STOP and CONTINUE when usage is missing. | |
| Budget limits: total and per-name tool calls | `budget.py`, `session/kernel.py` | C `test_tool_batch_admission_restart_parity`. The whole tool batch reserves its names atomically with the model receipt. Dispatch checks only the wall-time and host-cost limits and never counts reservations again. | |
| Budget limits: wall time, host cost, unavailable metrics | `budget.py` | C `test_wall_time_budget_parity`, `test_host_cost_and_unavailable_metrics_parity`, `test_host_meter_failure_latch_restart_parity`, `test_lost_active_interval_is_unavailable_after_restart`. Covers meter errors, units and currency, decreasing readings and a retained unavailable state. | If the active wall-time interval is lost in a restart, the metric becomes unavailable instead of being guessed. |
| Tracing processors and run, agent and tool spans | `tracing.py`, `session/tracing.py` | C `test_trace_delivery_span_parity_and_no_recovery_duplicates`, `test_trace_ack_boundary_and_processor_failure`, `test_trace_processors_cannot_mutate_durable_output` | Spans are projected from the log with their own committed consumer cursor. Telemetry is delivered at most once and can be lost after it is acknowledged. |
| Live assistant, reasoning and tool stream deltas | `events.py`, `session/runtime.py` | C `test_live_stream_deltas_and_durable_final_restart_parity`. The live sink is optional; final content, reasoning and tool calls replay from records. | |
| Typed lifecycle RunEvents | `events.py`, `session/kernel.py` | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`, plus the memory, budget and hook suites | Events have stable typed identities and follow record order. Child lifecycle events come from the parent's admission records. |
| Multimodal initial messages and image tool output | `types.py`, `llm/vv_llm_client.py`, `tools/function.py` | P `test_multimodal_model_context_parity`. Compares the full model-visible request, image notifications and provider tool-call extensions. | |
| Multiple model endpoints without stacked retries | `llm/vv_llm_client.py` | C `test_endpoint_routing_preference_logged_attempts_parity`, `test_logged_endpoint_dispatch_rejects_routing_tamper`; P `test_endpoint_attempts_do_not_stack` | The kernel logs each endpoint attempt and sends one request per attempt. It does not add transport retries on top. The route is frozen, including endpoint preference and randomization. |
| Model settings precedence: agent, run, provider | `model_settings.py`, `runner.py`, `runtime/compiler.py` | P `test_shared_state_and_settings_parity`, `test_provider_default_settings_parity` | The kernel overrides the transport retry setting (see the endpoint row). |
| Session model selection | `session/kernel.py`, `session/runtime.py`, `session/reducer.py` | `tests/session/test_model_selection.py`: `test_inflight_switch_cross_backend_usage_and_capacity`, `test_persistent_latest_wins_clear_reject_and_idempotency`, `test_selected_endpoints_and_retries_survive_runtime_reconstruction`, `test_recovery_keeps_planned_model_and_late_usage`, `test_process_kill_recovery_uses_original_plan`, `test_selection_is_local_to_one_session_and_default_rejects`, `test_parent_selection_does_not_propagate_to_real_child`. Covers `Runtime.model_resolver`, routing and usage kept per plan, and effective capacity. | |
| App Server `thread/model/select` | `app_server/protocol/thread.py`, `session/app_server.py` | `tests/test_app_server_model_selection.py`: `test_model_selection_method_admission_replay_conflict_and_schema`. Covers admission of the closed request, replay and conflicts. | |
| `Agent.as_tool` | `agent.py`, `runner.py` | D `test_sdk_children_never_recursive_runner_or_parent_lease`, `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`, `test_blocking_child_user_wait_is_durable_intentional_difference` | The tool runs as a child session, never as a recursive Runner that holds the parent's lease. The child setup is frozen, the tool returns the child's final turn result, and cancellation and waits are durable. |
| Configured sub-agents inherit policy, budget and workspace | `session/delegation.py`, `session/kernel.py` | D `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`. `SubAgentConfig`, model binding, prompt, summary, policy, budget and workspace are frozen at admission. | |
| BackgroundAgentTask start, poll, wait, cancel | `background_task.py` | D `test_background_handle_start_poll_wait_cancel_reconstruction`, `test_sdk_parent_cancellation_cascades_after_reconstruction` | Starting the tool admits a child session. A rebuilt handle returns the same public snapshot type and sends its controls through the inbox. |
| Handoff and the maximum handoff count | `handoffs.py`, `runner.py` | D `test_handoff_durable_transfer_and_maximum_parity`, `test_handoff_target_validation_state_and_events_parity` | The transfer to the target is durable and inherits state. The handoff count comes from the records, and the maximum is frozen. Target guardrails apply, and events are stable. |
| Child sessions: atomic admission, delivery and cancellation | `session/delegation.py`, `session/children.py` | D `test_configured_child_atomic_sdk_restart_parity`, `test_sdk_completion_delivery_failure_cut_and_replay_parity`, `test_sdk_parent_cancellation_cascades_after_reconstruction`, `test_completion_projection_uses_authenticated_turn_not_later_continuation`; fault cuts in `tests/session/test_children.py` | Admission, delivery and cancellation commit atomically with the parent's log, and a parent cancellation cascades to children. |
| `shared_state` between tools, persistence and rebuild | `session/kernel.py`, `tools/base.py` | D `test_shared_state_host_binding_restart_intentional_difference`, `test_host_binding_json_boundary_rejects_invalid_state` | Only JSON values are durable; nothing is pickled. Process-local objects need explicit host bindings, and a missing binding fails the same way every time. |
| Workspace backends: local, memory, S3, streaming | `workspace/`, `session/kernel.py` | D `test_s3_streaming_workspace_paired_producers` (S3 test double, real read and write handlers, bounded streaming of large objects, `Runtime` rebuilt with the backend supplied again); local and memory backends are covered by the P workspace tests | |
| Bash background sessions across restarts, owner checks | `runtime/background_sessions.py`, `runtime/processes.py` | T `test_background_process_restart_owner_parity`. The retained handle and turn owner reattach after the `Runtime` and SQL fold are rebuilt; the process manager stays alive. | |
| `no_tool_policy` finish | `session/kernel.py` | P `test_basic_runner_parity` | |
| `no_tool_policy` continue, and stop at `max_cycles` | `session/kernel.py` | P `test_continue_max_cycles_parity` (one-cycle and two-cycle stops) | |
| `no_tool_policy` wait_user | `session/kernel.py` | T `test_user_wait_sdk_lifecycle_parity` (no_tool case) | The wait is a durable turn-level wait with no invented tool operation. The reply resumes the same turn. |
| `tool_use_behavior` `stop_on_first_tool` and `stop_at_tool_names` | `runtime/tool_results.py` | P `test_tool_stop_parity`; T `test_tool_stop_pending_batch_native_finish_parity`, `test_tool_stop_error_result_parity`. Pending calls close with the same skipped result, and a native FINISH is respected. | |
| Cancellation: cooperative, unknown, descendants | `runtime/cancellation.py`, `run_handle.py` | T `test_cancellation_control_result_event_parity`, `test_cancellation_descendant_control_result_events_parity` | Cancellation produces durable receipts that say confirmed or unknown, instead of an SDK exception. Descendants get targeted controls and terminal events. |
| Completed RunResult fields and cycle and tool history | `result.py`, `runner.py` | C typed output, repair, budget and hook suites, `test_wait_result_fields_parity`, `test_completed_result_compaction_flags_are_per_cycle`, `test_endpoint_routing_preference_logged_attempts_parity`. Covers errors, partial output, waits and repair calls. | Results are built from the log. Cycles are logical cycles, and compaction flags are per cycle. |
| Event store replay and durable consumer acknowledgement | `event_store.py`, `run_handle.py` | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`. Covers the `SessionRunEventStore` bridge, SQL cursor rollback and replay, child queries, and identical-append checks. | |
| Interactive steer, follow-up, resume, archive, close | `interactive.py`, `session/surfaces.py` | `tests/test_interactive_session_api.py`: `test_interactive_real_same_turn_user_reply`, `test_interactive_live_steer_and_durable_follow_up`, `test_interactive_child_wait_reply_keeps_child_identity`, `test_interactive_close_during_model_call_is_idempotent`; `tests/test_interactive_session_persistence.py`: `test_creation_seed_hydrates_and_reuses_durable_session_without_duplicates` | The session is a view over records and the inbox. Replies go to the same turn or to the waiting child. Archive and close are durable. |
| CLI single run, streaming and persistent sessions | `cli.py`, `session/surfaces.py` | `tests/test_cli_contract.py`: `test_cli_real_single_run_and_stream_channels`, `test_cli_kernel_persistent_session_survives_owner_restart`. Covers SQLite memory and file owners, the default, and output channels. | |
| App Server threads, turns, approvals, replay, non-text input | `app_server/server.py`, `session/app_server.py` | The thread, turn, approval and replay suites; `test_kernel_process_restart_retains_calls_approval_image_and_client_cursor` (`tests/test_app_server_durable_resume.py`), `test_child_wait_user_is_exposed_and_reply_targets_child` (`tests/test_app_server_thread_turn.py`), `test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items` (`tests/test_app_server_controller_action.py`) | Thread state comes from the session log; the App Server keeps no second ledger. Suspend, reply, resume and terminal controls are inbox items. |
| App Server `model/list`, schema and TypeScript export | `app_server/protocol/`, `app_server/schema.py` | `test_model_list_forwards_optional_filters_and_emits_canonical_superset` (`tests/test_app_server_contract_parity.py`), `test_schema_export_request_returns_json_and_typescript_bundles` (`tests/test_app_server_schema.py`). The exported protocol v2 schema and TypeScript bundle bytes are pinned. | |

## Scheduling, dispatch and durability

These tests cover how sessions get driven, rather than what a single run does.
All of them run on PostgreSQL and SQLite.

| Behavior | Owner module | Tests |
| --- | --- | --- |
| Batches of child sessions: admission, waiting for every sibling, cancel, ordered and background results, invalid batches, crash during delivery | `session/children.py` | `tests/session/test_batch_children.py` |
| `tick` scans: failure isolation across pages, visible error groups, `RuntimeNotReady`, SQL retry backoff, fencing against lost leases and concurrent schedule writers | `session/supervisor.py` | `tests/session/test_supervisor.py` |
| Host dispatch: duplicate, reordered, lost and delayed wake messages; concurrency; lease expiry; races with a holder that is exiting; approvals, children and cancellation over an at-least-once transport | `session/supervisor.py`, `session/kernel.py` | `tests/session/test_dispatch.py` |
| Polling a provider call that returned `Accepted`: first poll at `poll_at_ms`, deferral that unrelated input does not cut short, authenticated results handled at once, the original deadline kept | `session/kernel.py`, `session/providers.py` | `tests/session/test_dispatch.py`, including `test_local_accepted_returns_parked_until_due_drive` |
| Seeded fault repetition | `session/kernel.py`, `session/sql.py` | `tests/session/test_soak.py`; `scripts/session_kernel_soak.py --assert` for release-scale runs |
| Long logs, prefix cache and capacity | `session/sql.py`, `session/kernel.py` | `tests/session/test_capacity.py`; `scripts/session_kernel_benchmark.py --assert-capacity` against real PostgreSQL |
| Provider usage and billing during recovery | `session/kernel.py`, `session/providers.py` | `tests/session/test_provider_usage.py`, `tests/session/test_recovery_matrix.py` |
| Built distributions and optional extras (base, `postgres`, `s3`, `postgres,s3`) in fresh virtual environments | packaging | `scripts/install_matrix.py` |

## Fixtures and public API

`tests/session/test_session_kernel_fixtures.py` generates the session kernel
fixtures from the real producers and compares them byte-for-byte with the
vendored snapshot. `test_public_api_manifest_resolves_real_python_exports` in
`tests/test_parity_evidence_manifests.py` resolves every capability and member
in the public API v11 inventory and fails on extra or missing exports. The
snapshot integrity check runs separately; see
[parity-contract.md](parity-contract.md).

Performance numbers live in
[session-kernel-baseline.md](session-kernel-baseline.md). Host upgrade steps
live in [upgrading.md](upgrading.md).
