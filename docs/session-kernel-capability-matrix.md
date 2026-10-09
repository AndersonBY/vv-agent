# Session kernel capability matrix

All public entrypoints use the same session kernel. Inventory: **63 current
capabilities**, each covered by a real producer test. Differences named below
refer to the transition from the pinned v23 runtime; the current v24 fixtures
and strict public API v8 define the supported behavior.

Paths are relative to `src/vv_agent/`. Test abbreviations under `tests/session/`:
P = `test_runner_parity.py`, T = `test_tools_control_parity.py`,
C = `test_capability_parity.py`, D = `test_delegation_parity.py`,
R = `test_recovery_matrix.py`. Persistent cases use PostgreSQL, SQLite files
and SQLite `:memory:`; process-kill cases use durable stores only.

| Capability | Current owner module | Kernel status | Evidence / remaining requirement |
| --- | --- | --- | --- |
| Basic synchronous no-tool completion | `runner.py`, `session/kernel.py` | done | P `test_basic_runner_parity` |
| FunctionTool and serial multi-tool execution | `tools/function.py`, `tools/orchestrator.py` | done | P `test_basic_runner_parity` |
| ToolRegistry factory / dynamically registered direct executors | `runner.py`, `tools/registry.py`, `runtime/tool_planner.py` | done | P `test_dynamic_enabled_and_registry_factory_parity` |
| FunctionTool `is_enabled` boolean/callback | `runtime/compiler.py:tool_is_enabled` | done | P `test_dynamic_enabled_and_registry_factory_parity`, `test_disabled_tool_never_executes` |
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
| Built-in `create_sub_task` | `tools/handlers/sub_agents.py`, `session/delegation.py` | done (intentional difference) | D `test_configured_child_atomic_sdk_restart_parity`, `test_configured_async_admission_status_and_late_delivery_parity`, `test_configured_child_argument_failure_parity`, `test_blocking_child_user_wait_is_durable_intentional_difference`; configured single/batch, synchronous/async, terminal delivery; a waiting child keeps its parent operation parked |
| Built-in `sub_task_status` | `tools/handlers/sub_task_status.py` | done | D `test_sub_task_status_records_projection_parity`, `test_sub_task_status_message_continuation_and_wait_projection_parity`; owner-scoped record projection, continuation/wait, stable inbox message replay; no second ledger |
| Tool allow/deny predicates and metadata denials | `run_config.py`, `tools/orchestrator.py` | done | T `test_tool_policy_matrix_parity`, `test_frozen_current_policy_dispatch_boundary`, `test_current_policy_predicate_rechecked_after_restart`; allow/deny lists, predicate, side effect/tag/cost/terminal denial and frozen/current dispatch restrictions |
| Approval modes default / always / never / on_request | `tools/orchestrator.py`, `approval.py` | done (intentional difference) | T `test_approval_mode_provider_parity`, `test_approval_broker_restart_session_and_conflicting_answer`, `test_approval_timeout_restart_parity`, `test_provider_decision_receipt_events_restart`; durable answers/session grants and absolute deadlines |
| Input guardrails allow/rewrite/block/require_approval | `runner.py`, `guardrails.py` | done | P `test_guardrail_parity`, `test_blocked_input_does_not_compile_providers` |
| Output guardrails allow/rewrite/block/require_approval | `runner.py`, `guardrails.py` | done | P `test_guardrail_parity` |
| Opt-in output validator accept/reject and one tools-free repair | `runner.py`, `output_validation.py` | done | P `test_output_validation_parity`; repair dispatch retained as an operation |
| Typed output coercion / repair exceptions / repair budget ledger | `runner.py`, `output_validation.py` | done (intentional difference) | C `test_typed_output_repair_ledger_restart_parity`, `test_output_repair_exceptions_parity`, `test_output_coercion_exception_becomes_durable_result`, `test_repair_usage_budget_is_logged_intentional_difference`, `test_uncertain_repair_does_not_retry_on_recovery`, `test_repair_missing_usage_policy_is_durable`; tools-free logged output_repair, typed JSON reconstruction and strict usage ledger |
| Runtime before_llm / after_llm hooks | `runtime/hooks.py` | done | P `test_llm_hook_parity` |
| Runtime before_tool_call / after_tool_call hooks | `runtime/hooks.py` | done | T `test_tool_hooks_state_approval_restart_parity`, `test_hook_patch_preserves_hidden_tool_boundary`; short circuit, serial state, approval and restart after prepared/result commits; recorded hooks are not replayed |
| Runtime before_memory_compact hook | `runtime/hooks.py`, `session/runtime.py` | done | C `test_before_memory_hook_replacement_restart_parity`, `test_boundary_validation_rolls_back`; replacement and JSON state committed before compaction, retained decisions not rerun |
| AfterCycleHook continue / steer / deny / stop | `runtime/lifecycle.py`, `session/kernel.py` | done | C `test_after_cycle_decision_snapshot_restart_parity`, `test_after_cycle_steering_boundary_parity`, `test_after_cycle_wait_and_native_finish_snapshot_restart`; snapshots rebuilt from adopted receipts, decisions replayed before dispatch |
| Context providers and prompt sections | `context_providers.py`, `runtime/compiler.py` | done | P `test_context_provider_parity`; immutable compiled prompt retained at turn admission |
| MemoryProvider compact callbacks | `memory/provider.py`, `session/runtime.py` | done | C `test_memory_provider_logged_callbacks_restart_parity`; accepted/rejected summary, before/after committed lifecycle and recovery at both boundaries |
| Session memory extraction/save/reload | `memory/session_memory.py`, `session/kernel.py` | done | C `test_session_memory_logged_extract_save_reload_parity`; session_memory model receipt, structured state and disposable file projection; deleted file rebuilt before next-turn reload |
| Microcompaction, summary and prompt-too-long recovery | `memory/manager.py`, `session/runtime.py` | done (intentional difference) | C `test_memory_compaction_runner_producer_parity`, `test_prompt_too_long_logical_cycle_and_tail_parity`; same MemoryManager producer, logical cycle and forced tail; duplicate rejected summary request reuses its logged receipt |
| Budget usage: cycles, tool counts, tokens/cache | `budget.py`, `runtime/token_usage.py` | done | P `test_budget_usage_parity` with measured provider usage |
| Budget limits: total and uncached input tokens | `budget.py`, `session/runtime.py` | done | C `test_token_budget_boundaries_parity`, `test_missing_usage_budget_parity`; zero/equality/overshoot boundaries and STOP/CONTINUE missing accounting |
| Budget limits: total and per-name tool calls | `budget.py`, `session/kernel.py` | done | C `test_tool_batch_admission_restart_parity`; whole batch atomically reserves names with the model receipt; dispatch checks wall/host and never counts reservations again |
| Budget limits: wall time, host cost, unavailable metrics | `budget.py` | done (intentional difference) | C `test_wall_time_budget_parity`, `test_host_cost_and_unavailable_metrics_parity`, `test_host_meter_failure_latch_restart_parity`, `test_lost_active_interval_is_unavailable_after_restart`; errors, units/currency, decreasing readings and retained unavailable state; strict missing active-interval recovery |
| Tracing processors and run/agent/tool spans | `tracing.py`, `session/tracing.py` | done (intentional difference) | C `test_trace_delivery_span_parity_and_no_recovery_duplicates`, `test_trace_ack_boundary_and_processor_failure`, `test_trace_processors_cannot_mutate_durable_output`; stable projected spans, separate committed traces cursor; at-most-once telemetry can be lost after acknowledgement |
| Live assistant/reasoning/tool stream deltas | `events.py`, `session/runtime.py` | done | C `test_live_stream_deltas_and_durable_final_restart_parity`; volatile sink is optional, definitive content/reasoning/tool calls replay from records |
| Typed lifecycle RunEvents | `events.py`, `session/kernel.py` | done (intentional difference) | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`, memory/budget/hook suites; stable typed identities and record sequence; child lifecycle belongs to parent admission records |
| Multimodal initial messages and tool image output | `types.py`, `llm/vv_llm_client.py`, `tools/function.py` | done | P `test_multimodal_model_context_parity` compares complete model-visible requests, image notifications and provider tool-call extensions |
| Multiple model endpoints, no stacked retries | `llm/vv_llm_client.py` | done (intentional difference) | C `test_endpoint_routing_preference_logged_attempts_parity`, `test_logged_endpoint_dispatch_rejects_routing_tamper`; actual Runner/VvLlmClient transport producer, preference/randomization, three endpoint attempts, frozen route and one request per attempt |
| Agent/run/provider model settings precedence | `model_settings.py`, `runner.py`, `runtime/compiler.py` | done | P `test_shared_state_and_settings_parity`, `test_provider_default_settings_parity`; transport retry override is intentional |
| Agent.as_tool | `agent.py`, `runner.py` | done (intentional difference) | D `test_sdk_children_never_recursive_runner_or_parent_lease`, `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`, `test_blocking_child_user_wait_is_durable_intentional_difference`; frozen child assembly, terminal-turn result, durable cancellation/wait |
| Configured sub-agents: policy/budget/workspace inheritance | `session/delegation.py`, `session/kernel.py` | done | D `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`; SubAgentConfig, model binding, prompt/summary, policy/budget/workspace frozen at admission |
| BackgroundAgentTask start/poll/wait/cancel | `background_task.py` | done (intentional difference) | D `test_background_handle_start_poll_wait_cancel_reconstruction`, `test_sdk_parent_cancellation_cascades_after_reconstruction`; tool admission starts a child; reconstructed internal handles project the existing public snapshot type and submit inbox controls |
| Handoff and maximum-handoff enforcement | `handoffs.py`, `runner.py` | done (intentional difference) | D `test_handoff_durable_transfer_and_maximum_parity`, `test_handoff_target_validation_state_and_events_parity`; durable terminal child transfer, inherited state, target guardrails, record-derived count/frozen maximum, stable events |
| Child session atomic admission/delivery/cancellation | `session/delegation.py`, `session/children.py` | done (intentional difference) | D `test_configured_child_atomic_sdk_restart_parity`, `test_sdk_completion_delivery_failure_cut_and_replay_parity`, `test_sdk_parent_cancellation_cascades_after_reconstruction`, `test_completion_projection_uses_authenticated_turn_not_later_continuation`; SDK producers plus existing `session/test_children.py` fault cuts |
| shared_state between tools, persistence and reconstruction | `session/kernel.py`, `tools/base.py` | done (intentional difference) | D `test_shared_state_host_binding_restart_intentional_difference`, `test_host_binding_json_boundary_rejects_invalid_state`; JSON-only durability, explicit process-local object bindings, deterministic missing binding, no pickle |
| Workspace local/memory/S3/streaming backend selection | `workspace/`, `session/kernel.py` | done | D `test_s3_streaming_workspace_paired_producers`; existing S3 double, real read/write handlers, bounded large-object streaming and Runtime reconstruction with re-supplied backend; existing local/memory P producers |
| Bash background session restart and owner checks | `runtime/background_sessions.py`, `runtime/processes.py` | done | T `test_background_process_restart_owner_parity`; retained handle and unchanged turn owner reattach after rebuilding Runtime and SQL fold; process manager remains alive |
| no_tool_policy finish | `session/kernel.py` | done | P `test_basic_runner_parity` |
| no_tool_policy continue and max_cycles stop | `session/kernel.py` | done | P `test_continue_max_cycles_parity` compares one- and two-cycle stops |
| no_tool_policy wait_user | `session/kernel.py` | done (intentional difference) | T `test_user_wait_sdk_lifecycle_parity` (no_tool); durable turn-level wait with zero fabricated tool operations; reply resumes the same turn |
| tool_use_behavior stop_on_first_tool / stop_at_tool_names | `runtime/tool_results.py` | done | P `test_tool_stop_parity`; T `test_tool_stop_pending_batch_native_finish_parity`, `test_tool_stop_error_result_parity`; pending calls close with the same skipped result and native FINISH is respected |
| Cancellation: cooperative / unknown / descendants | `runtime/cancellation.py`, `run_handle.py` | done (intentional difference) | T `test_cancellation_control_result_event_parity`, `test_cancellation_descendant_control_result_events_parity`; SDK cancellation/exception versus durable confirmed/unknown receipts, targeted descendant controls and terminal events |
| Completed RunResult fields and cycle/tool history | `result.py`, `runner.py` | done (intentional difference) | C typed output/repair/budget/hook suites, `test_wait_result_fields_parity`, `test_completed_result_compaction_flags_are_per_cycle`, `test_endpoint_routing_preference_logged_attempts_parity`; errors, partial output, waits, logical cycles, repair calls and fixed terminal prefix |
| Event store replay and durable consumer acknowledgement | `event_store.py`, `run_handle.py` | done | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`; SessionRunEventStore bridge, SQL cursor rollback/replay, child query and identical append assertion |
| Interactive steer/follow-up/resume/archive/close | `interactive.py`, `session/surfaces.py` | done (intentional difference) | `test_interactive_real_same_turn_user_reply`, `test_interactive_live_steer_and_durable_follow_up`, `test_interactive_child_wait_reply_keeps_child_identity`, `test_kernel_file_facade_rebuild_resume_and_control_identity`, `test_interactive_close_during_model_call_is_idempotent`; records/inbox facade, same-turn/child reply, durable archive/close |
| CLI single-run, stream and persistent sessions | `cli.py`, `session/surfaces.py` | done | `test_cli_real_single_run_and_stream_channels`, `test_cli_kernel_persistent_session_survives_owner_restart`; SQLite memory/file owners, unchanged default and output channels |
| App Server thread/turn/approval/replay/non-text input | `app_server/server.py`, `session/app_server.py` | done (intentional difference) | Current thread/turn/approval/replay suites; `test_kernel_process_restart_retains_calls_approval_image_and_client_cursor`, `test_child_wait_user_is_exposed_and_reply_targets_child`, `test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items`; no second ledger; current v2 protocol fixtures |
| App Server model/list, schema and TypeScript export | `app_server/protocol/`, `app_server/schema.py` | done | `test_model_list_forwards_optional_filters_and_emits_canonical_superset`, `test_schema_export_request_returns_json_and_typescript_bundles`; full result/bundle identity; current v2 schema/TypeScript bundle bytes are pinned |

## Evidence boundaries

Paired tests independently assemble public Runner and direct kernel producers,
then compare output, tool calls, projected events, usage and JSON shared state.
Identity and clock values are compared only where fixed by the contract. Recovery
cases reconstruct Runtime/store state and assert retained callbacks, receipts,
authorization, child identity, budgets and same-turn replies.

The forty-five generated v24.0.1 fixtures compare byte-for-byte with the vendored
snapshot. Public API v8 resolves every exported capability/member and rejects
extra or missing exports. Snapshot checks establish artifact integrity separately.

Performance methodology and current absolute measurements live in
[session-kernel-baseline.md](session-kernel-baseline.md); host API migration lives
in [migration-v8.md](migration-v8.md).
