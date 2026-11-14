# Session kernel capability matrix

Every public entry point (Runner, interactive sessions, the CLI and the App
Server) runs on the same session kernel. This page maps each capability to the
module that owns it and to the tests that prove it. Use it to find where to
change a behavior and which tests to run afterwards.

The "Difference from 0.21.x" column is filled in only where the kernel behaves
differently from the in-process runtime of package 0.21.x (contract 23). These
differences are deliberate. The contract fixtures and public API v15 define the
supported behavior.

## How to read the table

Owner paths start at the repository root. Each row has a stable capability ID
and explicit pytest function nodes. A node names a function definition, including
all its parameter cases; it does not pin generated parameter IDs. File-only
references in the scheduling and fixture sections are supplementary suite
inventories, not capability declarations.

Tests that use a persistent store run on PostgreSQL, on a SQLite file and on
SQLite `:memory:`. Tests that kill a process run on durable stores only.

Most rows are paired tests. A paired test builds the same run twice, once
through the public Runner and once by driving the kernel directly, then
compares the output, tool calls, projected events, usage and JSON shared state.
Identifiers and clock values are compared only where the contract fixes them.
Recovery tests rebuild the `Runtime` and the store, then check that callbacks,
receipts, authorization, child identity, budgets and same-turn replies were
kept.

## Capabilities

| Capability ID | Capability | Owner module | Tests | Difference from 0.21.x |
| --- | --- | --- | --- | --- |
| `basic-completion` | Basic synchronous completion without tools | `src/vv_agent/runner.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_runner_parity.py::test_basic_runner_parity` |  |
| `serial-tools` | FunctionTool and serial multi-tool execution | `src/vv_agent/tools/function.py`, `src/vv_agent/tools/orchestrator.py` | `tests/session/test_runner_parity.py::test_basic_runner_parity` |  |
| `tool-registry` | ToolRegistry factory and dynamically registered direct executors | `src/vv_agent/runner.py`, `src/vv_agent/tools/registry.py`, `src/vv_agent/runtime/tool_planner.py` | `tests/session/test_runner_parity.py::test_dynamic_enabled_and_registry_factory_parity` |  |
| `tool-enabled` | FunctionTool `is_enabled` as a boolean or callback | `src/vv_agent/runtime/compiler.py` (`py:src/vv_agent/runtime/compiler.py::tool_is_enabled`) | `tests/session/test_runner_parity.py::test_dynamic_enabled_and_registry_factory_parity`, `tests/session/test_runner_parity.py::test_disabled_tool_never_executes` |  |
| `hidden-tools` | Hidden tools, dynamic schema changes and exposure limits | `src/vv_agent/tools/executor.py`, `src/vv_agent/tools/registry.py` | `tests/session/test_tools_control_parity.py::test_hidden_tool_exposure_parity`, `tests/session/test_tools_control_parity.py::test_dynamic_schema_new_turn_parity`, `tests/session/test_tools_control_parity.py::test_dynamic_schema_active_turn_rejected`, `tests/session/test_tools_control_parity.py::test_hook_patch_preserves_hidden_tool_boundary`. A hidden tool cannot be called. A new turn picks up schema changes; an active turn rejects them. |  |
| `read-file` | Built-in `read_file` | `src/vv_agent/tools/handlers/workspace_io.py` | `tests/session/test_runner_parity.py::test_workspace_tool_parity` |  |
| `write-file` | Built-in `write_file` | `src/vv_agent/tools/handlers/workspace_io.py` | `tests/session/test_runner_parity.py::test_workspace_tool_parity` |  |
| `edit-file` | Built-in `edit_file`, including the prior-read baseline | `src/vv_agent/tools/handlers/workspace_io.py` | `tests/session/test_runner_parity.py::test_workspace_tool_parity` |  |
| `find-files` | Built-in `find_files` | `src/vv_agent/tools/handlers/search.py` | `tests/session/test_runner_parity.py::test_workspace_tool_parity` |  |
| `search-files` | Built-in `search_files` | `src/vv_agent/tools/handlers/search.py` | `tests/session/test_runner_parity.py::test_workspace_tool_parity` |  |
| `file-info` | Built-in `file_info` | `src/vv_agent/tools/handlers/workspace_io.py` | `tests/session/test_runner_parity.py::test_additional_builtin_parity` |  |
| `todo-write` | Built-in `todo_write` | `src/vv_agent/tools/handlers/todo.py` | `tests/session/test_runner_parity.py::test_todo_and_skill_state_parity`, with a fixed clock and a TODO id chosen by the caller |  |
| `ask-user` | Built-in `ask_user` and host interaction replies | `src/vv_agent/tools/handlers/control.py`, `src/vv_agent/interaction.py` | `tests/session/test_tools_control_parity.py::test_user_wait_sdk_lifecycle_parity` (ask_user case) | The wait is durable. The turn parks, and the reply arrives as an inbox item that resumes the same turn and operation, even after a rebuild. An identical duplicate reply does nothing; a conflicting one is rejected. |
| `bash` | Built-in `bash` | `src/vv_agent/tools/handlers/bash.py` | `tests/session/test_runner_parity.py::test_additional_builtin_parity` (foreground output; background commands have their own rows) |  |
| `check-background` | Built-in `check_background_command` | `src/vv_agent/tools/handlers/background.py` | `tests/session/test_tools_control_parity.py::test_background_process_restart_owner_parity`, `tests/session/test_tools_control_parity.py::test_background_forbidden_owner_parity`. Covers running and completed receipts and denial to other owners. |  |
| `stop-background` | Built-in `stop_background_command` | `src/vv_agent/tools/handlers/background.py` | `tests/session/test_tools_control_parity.py::test_background_process_restart_owner_parity`, `tests/session/test_tools_control_parity.py::test_background_unknown_stop_is_not_confirmed_parity`. A confirmed stop reports stopped; an unconfirmed one reports stopping or unknown. |  |
| `read-image` | Built-in `read_image` | `src/vv_agent/tools/handlers/image.py` | `tests/session/test_runner_parity.py::test_additional_builtin_parity`, `tests/session/test_runner_parity.py::test_multimodal_model_context_parity` (URL and local PNG) |  |
| `activate-skill` | Built-in `activate_skill` | `src/vv_agent/tools/handlers/skills.py`, `src/vv_agent/skills/` | `tests/session/test_runner_parity.py::test_todo_and_skill_state_parity` (activation and the retained active skill) |  |
| `create-sub-task` | Built-in `create_sub_task` | `src/vv_agent/tools/handlers/sub_agents.py`, `src/vv_agent/session/delegation.py` | `tests/session/test_delegation_parity.py::test_configured_child_atomic_sdk_restart_parity`, `tests/session/test_delegation_parity.py::test_configured_async_admission_status_and_late_delivery_parity`, `tests/session/test_delegation_parity.py::test_configured_child_argument_failure_parity`, `tests/session/test_delegation_parity.py::test_blocking_child_user_wait_is_durable_intentional_difference`. Covers single and batch, synchronous and async, and delivery of the final result. | While a child waits for the user, the parent operation stays parked. |
| `sub-task-status` | Built-in `sub_task_status` | `src/vv_agent/tools/handlers/sub_task_status.py` | `tests/session/test_delegation_parity.py::test_sub_task_status_records_projection_parity`, `tests/session/test_delegation_parity.py::test_sub_task_status_message_continuation_and_wait_projection_parity`. Status is projected from the owner's records; continuation and wait replay stable inbox messages. There is no second ledger. |  |
| `tool-policy` | Tool allow and deny predicates, metadata denials | `src/vv_agent/run_config.py`, `src/vv_agent/tools/orchestrator.py` | `tests/session/test_tools_control_parity.py::test_tool_policy_matrix_parity`, `tests/session/test_tools_control_parity.py::test_frozen_current_policy_dispatch_boundary`, `tests/session/test_tools_control_parity.py::test_current_policy_predicate_rechecked_after_restart`. Covers allow and deny lists, predicates, denial by side effect, tag, cost or terminal flag, and frozen versus current policy at dispatch. |  |
| `approval-modes` | Approval modes `default`, `always`, `never`, `on_request` | `src/vv_agent/tools/orchestrator.py`, `src/vv_agent/approval.py` | `tests/session/test_tools_control_parity.py::test_approval_mode_provider_parity`, `tests/session/test_tools_control_parity.py::test_approval_broker_restart_session_and_conflicting_answer`, `tests/session/test_tools_control_parity.py::test_approval_timeout_restart_parity`, `tests/session/test_tools_control_parity.py::test_provider_decision_receipt_events_restart` | Answers and session-wide grants are stored durably, and approval deadlines are absolute times. |
| `input-guardrails` | Input guardrails: allow, rewrite, block, require approval | `src/vv_agent/runner.py`, `src/vv_agent/guardrails.py` | `tests/session/test_runner_parity.py::test_guardrail_parity`, `tests/session/test_runner_parity.py::test_blocked_input_does_not_compile_providers` |  |
| `output-guardrails` | Output guardrails: allow, rewrite, block, require approval | `src/vv_agent/runner.py`, `src/vv_agent/guardrails.py` | `tests/session/test_runner_parity.py::test_guardrail_parity` |  |
| `output-validator` | Opt-in output validator with one tools-free repair | `src/vv_agent/runner.py`, `src/vv_agent/output_validation.py` | `tests/session/test_runner_parity.py::test_output_validation_parity`. The repair call is kept as a logged operation. |  |
| `typed-output` | Typed output coercion, repair exceptions and the repair budget | `src/vv_agent/runner.py`, `src/vv_agent/output_validation.py` | `tests/session/test_capability_parity.py::test_typed_output_repair_ledger_restart_parity`, `tests/session/test_capability_parity.py::test_output_repair_exceptions_parity`, `tests/session/test_capability_parity.py::test_output_coercion_exception_becomes_durable_result`, `tests/session/test_capability_parity.py::test_repair_usage_budget_is_logged_intentional_difference`, `tests/session/test_capability_parity.py::test_uncertain_repair_does_not_retry_on_recovery`, `tests/session/test_capability_parity.py::test_repair_missing_usage_policy_is_durable` | The repair is a logged, tools-free `output_repair` call, and its token usage is counted against the budget. Typed output is rebuilt from JSON. A repair with an uncertain outcome is not retried on recovery. |
| `llm-hooks` | Runtime `before_llm` and `after_llm` hooks | `src/vv_agent/runtime/hooks.py` | `tests/session/test_runner_parity.py::test_llm_hook_parity` |  |
| `tool-hooks` | Runtime `before_tool_call` and `after_tool_call` hooks | `src/vv_agent/runtime/hooks.py` | `tests/session/test_tools_control_parity.py::test_tool_hooks_state_approval_restart_parity`, `tests/session/test_tools_control_parity.py::test_hook_patch_preserves_hidden_tool_boundary`. Covers short circuits, serial state, approval, and restart after the prepared and result commits. Recorded hook decisions are not replayed. |  |
| `memory-hook` | Runtime `before_memory_compact` hook | `src/vv_agent/runtime/hooks.py`, `src/vv_agent/session/runtime.py` | `tests/session/test_capability_parity.py::test_before_memory_hook_replacement_restart_parity`, `tests/session/test_capability_parity.py::test_boundary_validation_rolls_back`. The replacement and JSON state are committed before compaction; retained decisions do not run again. |  |
| `after-cycle` | AfterCycleHook: continue, steer, deny, stop | `src/vv_agent/runtime/lifecycle.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_capability_parity.py::test_after_cycle_decision_snapshot_restart_parity`, `tests/session/test_capability_parity.py::test_after_cycle_steering_boundary_parity`, `tests/session/test_capability_parity.py::test_after_cycle_wait_and_native_finish_snapshot_restart`. Snapshots are rebuilt from adopted receipts, and decisions are replayed before dispatch. |  |
| `context-providers` | Context providers and prompt sections | `src/vv_agent/context_providers.py`, `src/vv_agent/runtime/compiler.py` | `tests/session/test_runner_parity.py::test_context_provider_parity`. The compiled prompt is frozen when the turn is admitted. |  |
| `memory-provider` | MemoryProvider compaction callbacks | `src/vv_agent/memory/provider.py`, `src/vv_agent/session/runtime.py` | `tests/session/test_capability_parity.py::test_memory_provider_logged_callbacks_restart_parity`. Covers accepted and rejected summaries, the committed before and after steps, and recovery at both points. |  |
| `session-memory` | Session memory extraction, save and reload | `src/vv_agent/memory/session_memory.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_capability_parity.py::test_session_memory_logged_extract_save_reload_parity`. The `session_memory` model receipt and structured state are logged; the file on disk is a disposable copy that is rebuilt before the next turn reloads it. |  |
| `microcompaction` | Microcompaction, summaries and prompt-too-long recovery | `src/vv_agent/memory/manager.py`, `src/vv_agent/session/runtime.py` | `tests/session/test_capability_parity.py::test_memory_compaction_runner_producer_parity`, `tests/session/test_capability_parity.py::test_prompt_too_long_logical_cycle_and_tail_parity` | The same MemoryManager produces both runs, with the same logical cycle and forced tail. A repeated summary request that was rejected reuses its logged receipt. |
| `budget-usage` | Budget usage: cycles, tool counts, tokens and cache | `src/vv_agent/budget.py`, `src/vv_agent/runtime/token_usage.py` | `tests/session/test_runner_parity.py::test_budget_usage_parity`, with measured provider usage |  |
| `token-budget` | Budget limits: total and uncached input tokens | `src/vv_agent/budget.py`, `src/vv_agent/session/runtime.py` | `tests/session/test_capability_parity.py::test_token_budget_boundaries_parity`, `tests/session/test_capability_parity.py::test_missing_usage_budget_parity`. Covers zero, equal and overshoot limits, and STOP and CONTINUE when usage is missing. |  |
| `tool-budget` | Budget limits: total and per-name tool calls | `src/vv_agent/budget.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_capability_parity.py::test_tool_batch_admission_restart_parity`. The whole tool batch reserves its names atomically with the model receipt. Dispatch checks only the wall-time and host-cost limits and never counts reservations again. |  |
| `host-budget` | Budget limits: wall time, host cost, unavailable metrics | `src/vv_agent/budget.py` | `tests/session/test_capability_parity.py::test_wall_time_budget_parity`, `tests/session/test_capability_parity.py::test_host_cost_and_unavailable_metrics_parity`, `tests/session/test_capability_parity.py::test_host_meter_failure_latch_restart_parity`, `tests/session/test_capability_parity.py::test_lost_active_interval_is_unavailable_after_restart`. Covers meter errors, units and currency, decreasing readings and a retained unavailable state. | If the active wall-time interval is lost in a restart, the metric becomes unavailable instead of being guessed. |
| `tracing` | Tracing processors and run, agent and tool spans | `src/vv_agent/tracing.py`, `src/vv_agent/session/tracing.py` | `tests/session/test_capability_parity.py::test_trace_delivery_span_parity_and_no_recovery_duplicates`, `tests/session/test_capability_parity.py::test_trace_ack_boundary_and_processor_failure`, `tests/session/test_capability_parity.py::test_trace_processors_cannot_mutate_durable_output` | Spans are projected from the log with their own committed consumer cursor. Telemetry is delivered at most once and can be lost after it is acknowledged. |
| `stream-deltas` | Live assistant, reasoning and tool stream deltas | `src/vv_agent/events.py`, `src/vv_agent/session/runtime.py` | `tests/session/test_capability_parity.py::test_live_stream_deltas_and_durable_final_restart_parity`. The live sink is optional; final content, reasoning and tool calls replay from records. |  |
| `lifecycle-events` | Typed lifecycle RunEvents | `src/vv_agent/events.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_capability_parity.py::test_lifecycle_events_replay_ack_and_rollback_parity`, `tests/session/test_capability_parity.py::test_delegation_events_and_child_replay_paired_producer`, plus the memory, budget and hook suites | Events have stable typed identities and follow record order. Child lifecycle events come from the parent's admission records. |
| `multimodal` | Multimodal initial messages and image tool output | `src/vv_agent/types.py`, `src/vv_agent/llm/vv_llm_client.py`, `src/vv_agent/tools/function.py` | `tests/session/test_runner_parity.py::test_multimodal_model_context_parity`. Compares the full model-visible request, image notifications and provider tool-call extensions. |  |
| `endpoint-routing` | Multiple model endpoints without stacked retries | `src/vv_agent/llm/vv_llm_client.py` | `tests/session/test_capability_parity.py::test_endpoint_routing_preference_logged_attempts_parity`, `tests/session/test_capability_parity.py::test_logged_endpoint_dispatch_rejects_routing_tamper`; `tests/session/test_runner_parity.py::test_endpoint_attempts_do_not_stack` | The kernel logs each logical endpoint invocation. Vendor SDK retries may send multiple physical HTTP requests within it; the vv-llm bridge observes every send before confirming rejection. The route is frozen, including endpoint preference and randomization. |
| `model-settings` | Model settings precedence: agent, run, provider | `src/vv_agent/model_settings.py`, `src/vv_agent/runner.py`, `src/vv_agent/runtime/compiler.py` | `tests/session/test_runner_parity.py::test_shared_state_and_settings_parity`, `tests/session/test_runner_parity.py::test_provider_default_settings_parity` | The kernel overrides the transport retry setting (see the endpoint row). |
| `model-selection` | Session model selection | `src/vv_agent/session/kernel.py`, `src/vv_agent/session/runtime.py`, `src/vv_agent/kernel/_reducer.py` | `tests/session/test_model_selection.py::test_inflight_switch_cross_backend_usage_and_capacity`, `tests/session/test_model_selection.py::test_persistent_latest_wins_clear_reject_and_idempotency`, `tests/session/test_model_selection.py::test_selected_endpoints_and_retries_survive_runtime_reconstruction`, `tests/session/test_model_selection.py::test_recovery_keeps_planned_model_and_late_usage`, `tests/session/test_model_selection.py::test_process_kill_recovery_uses_original_plan`, `tests/session/test_model_selection.py::test_selection_is_local_to_one_session_and_default_rejects`, `tests/session/test_model_selection.py::test_parent_selection_does_not_propagate_to_real_child`. Covers `Runtime.model_resolver`, routing and usage kept per plan, and effective capacity. |  |
| `app-server-model-selection` | App Server `thread/model/select` | `src/vv_agent/app_server/protocol/thread.py`, `src/vv_agent/session/app_server.py` | `tests/test_app_server_model_selection.py::test_model_selection_method_admission_replay_conflict_and_schema`. Covers admission of the closed request, replay and conflicts. |  |
| `agent-as-tool` | `Agent.as_tool` | `src/vv_agent/agent.py`, `src/vv_agent/runner.py` | `tests/session/test_delegation_parity.py::test_sdk_children_never_recursive_runner_or_parent_lease`, `tests/session/test_delegation_parity.py::test_child_hooks_policy_budget_workspace_inheritance_frozen`, `tests/session/test_delegation_parity.py::test_child_inherited_denial_and_budget_execute_real_producers`, `tests/session/test_delegation_parity.py::test_blocking_child_user_wait_is_durable_intentional_difference` | The tool runs as a child session, never as a recursive Runner that holds the parent's lease. The child setup is frozen, the tool returns the child's final turn result, and cancellation and waits are durable. |
| `child-inheritance` | Configured sub-agents inherit policy, budget and workspace | `src/vv_agent/session/delegation.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_delegation_parity.py::test_child_hooks_policy_budget_workspace_inheritance_frozen`, `tests/session/test_delegation_parity.py::test_child_inherited_denial_and_budget_execute_real_producers`. `SubAgentConfig`, model binding, prompt, summary, policy, budget and workspace are frozen at admission. |  |
| `background-task` | BackgroundAgentTask start, poll, wait, cancel | `src/vv_agent/background_task.py` | `tests/session/test_delegation_parity.py::test_background_handle_start_poll_wait_cancel_reconstruction`, `tests/session/test_delegation_parity.py::test_sdk_parent_cancellation_cascades_after_reconstruction` | Starting the tool admits a child session. A rebuilt handle returns the same public snapshot type and sends its controls through the inbox. |
| `handoff` | Handoff and the maximum handoff count | `src/vv_agent/handoffs.py`, `src/vv_agent/runner.py` | `tests/session/test_delegation_parity.py::test_handoff_durable_transfer_and_maximum_parity`, `tests/session/test_delegation_parity.py::test_handoff_target_validation_state_and_events_parity` | The transfer to the target is durable and inherits state. The handoff count comes from the records, and the maximum is frozen. Target guardrails apply, and events are stable. |
| `child-lifecycle` | Child sessions: atomic admission, delivery and cancellation | `src/vv_agent/session/delegation.py`, `src/vv_agent/session/children.py` | `tests/session/test_delegation_parity.py::test_configured_child_atomic_sdk_restart_parity`, `tests/session/test_delegation_parity.py::test_sdk_completion_delivery_failure_cut_and_replay_parity`, `tests/session/test_delegation_parity.py::test_sdk_parent_cancellation_cascades_after_reconstruction`, `tests/session/test_delegation_parity.py::test_completion_projection_uses_authenticated_turn_not_later_continuation`; fault cuts in `tests/session/test_children.py` | Admission, delivery and cancellation commit atomically with the parent's log, and a parent cancellation cascades to children. |
| `shared-state` | `shared_state` between tools, persistence and rebuild | `src/vv_agent/session/kernel.py`, `src/vv_agent/tools/base.py` | `tests/session/test_delegation_parity.py::test_shared_state_host_binding_restart_intentional_difference`, `tests/session/test_delegation_parity.py::test_host_binding_json_boundary_rejects_invalid_state` | Only JSON values are durable; nothing is pickled. Process-local objects need explicit host bindings, and a missing binding fails the same way every time. |
| `workspace-backends` | Workspace backends: local, memory, S3, streaming | `src/vv_agent/workspace/`, `src/vv_agent/session/kernel.py` | `tests/session/test_delegation_parity.py::test_s3_streaming_workspace_paired_producers` (S3 test double, real read and write handlers, bounded streaming of large objects, `Runtime` rebuilt with the backend supplied again); local and memory backends are covered by the workspace tests |  |
| `background-ownership` | Bash background sessions across restarts, owner checks | `src/vv_agent/runtime/background_sessions.py`, `src/vv_agent/runtime/processes.py` | `tests/session/test_tools_control_parity.py::test_background_process_restart_owner_parity`. The retained handle and turn owner reattach after the `Runtime` and SQL fold are rebuilt; the process manager stays alive. |  |
| `no-tool-finish` | `no_tool_policy` finish | `src/vv_agent/session/kernel.py` | `tests/session/test_runner_parity.py::test_basic_runner_parity` |  |
| `no-tool-continue` | `no_tool_policy` continue, and stop at `max_cycles` | `src/vv_agent/session/kernel.py` | `tests/session/test_runner_parity.py::test_continue_max_cycles_parity` (one-cycle and two-cycle stops) |  |
| `no-tool-wait` | `no_tool_policy` wait_user | `src/vv_agent/session/kernel.py` | `tests/session/test_tools_control_parity.py::test_user_wait_sdk_lifecycle_parity` (no_tool case) | The wait is a durable turn-level wait with no invented tool operation. The reply resumes the same turn. |
| `tool-stop` | `tool_use_behavior` `stop_on_first_tool` and `stop_at_tool_names` | `src/vv_agent/runtime/tool_results.py` | `tests/session/test_runner_parity.py::test_tool_stop_parity`; `tests/session/test_tools_control_parity.py::test_tool_stop_pending_batch_native_finish_parity`, `tests/session/test_tools_control_parity.py::test_tool_stop_error_result_parity`. Pending calls close with the same skipped result, and a native FINISH is respected. |  |
| `cancellation` | Cancellation: cooperative, unknown, descendants | `src/vv_agent/runtime/cancellation.py`, `src/vv_agent/run_handle.py` | `tests/session/test_tools_control_parity.py::test_cancellation_control_result_event_parity`, `tests/session/test_tools_control_parity.py::test_cancellation_descendant_control_result_events_parity` | Cancellation produces durable receipts that say confirmed or unknown, instead of an SDK exception. Descendants get targeted controls and terminal events. |
| `result-fields` | Completed RunResult fields and cycle and tool history | `src/vv_agent/result.py`, `src/vv_agent/runner.py` | typed output, repair, budget and hook suites, `tests/session/test_capability_parity.py::test_wait_result_fields_parity`, `tests/session/test_capability_parity.py::test_completed_result_compaction_flags_are_per_cycle`, `tests/session/test_capability_parity.py::test_endpoint_routing_preference_logged_attempts_parity`. Covers errors, partial output, waits and repair calls. | Results are built from the log. Cycles are logical cycles, and compaction flags are per cycle. |
| `event-store` | Event store replay and durable consumer acknowledgement | `src/vv_agent/event_store.py`, `src/vv_agent/run_handle.py` | `tests/session/test_capability_parity.py::test_lifecycle_events_replay_ack_and_rollback_parity`, `tests/session/test_capability_parity.py::test_delegation_events_and_child_replay_paired_producer`. Covers the `SessionRunEventStore` bridge, SQL cursor rollback and replay, child queries, and identical-append checks. |  |
| `interactive` | Interactive steer, follow-up, resume, archive, close | `src/vv_agent/interactive.py`, `src/vv_agent/session/surfaces.py` | `tests/test_interactive_session_api.py::test_interactive_real_same_turn_user_reply`, `tests/test_interactive_session_api.py::test_interactive_live_steer_and_durable_follow_up`, `tests/test_interactive_session_api.py::test_interactive_child_wait_reply_keeps_child_identity`, `tests/test_interactive_session_api.py::test_interactive_close_during_model_call_is_idempotent`; `tests/test_interactive_session_persistence.py::test_creation_seed_hydrates_and_reuses_durable_session_without_duplicates` | The session is a view over records and the inbox. Replies go to the same turn or to the waiting child. Archive and close are durable. |
| `cli` | CLI single run, streaming and persistent sessions | `src/vv_agent/cli.py`, `src/vv_agent/session/surfaces.py` | `tests/test_cli_contract.py::test_cli_real_single_run_and_stream_channels`, `tests/test_cli_contract.py::test_cli_kernel_persistent_session_survives_owner_restart`. Covers SQLite memory and file owners, the default, and output channels. |  |
| `app-server-turns` | App Server threads, turns, approvals, replay, non-text input | `src/vv_agent/app_server/server.py`, `src/vv_agent/session/app_server.py` | The thread, turn, approval and replay suites; `tests/test_app_server_durable_resume.py::test_kernel_process_restart_retains_calls_approval_image_and_client_cursor`, `tests/test_app_server_thread_turn.py::test_child_wait_user_is_exposed_and_reply_targets_child`, `tests/test_app_server_controller_action.py::test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items` | Thread state comes from the session log; the App Server keeps no second ledger. Suspend, reply, resume and terminal controls are inbox items. |
| `app-server-schema` | App Server `model/list`, schema and TypeScript export | `src/vv_agent/app_server/protocol/`, `src/vv_agent/app_server/schema.py` | `tests/test_app_server_contract_parity.py::test_model_list_forwards_optional_filters_and_emits_canonical_superset`, `tests/test_app_server_schema.py::test_schema_export_request_returns_json_and_typescript_bundles`. The exported protocol v5 schema and TypeScript bundle bytes are pinned. |  |
| `provider-polling` | Polling a provider call that returned `Accepted`: first poll at `poll_at_ms`, deferral that unrelated input does not cut short, authenticated results handled at once, the original deadline kept | `src/vv_agent/session/kernel.py`, `src/vv_agent/session/providers.py` | `tests/session/test_dispatch.py::test_local_accepted_returns_parked_until_due_drive`, `tests/session/test_dispatch.py::test_provider_deadline_caps_idle_deferral`, `tests/session/test_dispatch.py::test_unrelated_input_preserves_provider_deferral`, `tests/session/test_dispatch.py::test_authenticated_provider_result_is_immediate_before_poll_due` | |

## Scheduling, dispatch and durability

These tests cover how sessions get driven, rather than what a single run does.
All of them run on PostgreSQL and SQLite.

| Behavior | Owner module | Tests |
| --- | --- | --- |
| Batches of child sessions: admission, waiting for every sibling, cancel, ordered and background results, invalid batches, crash during delivery | `src/vv_agent/session/children.py` | `tests/session/test_batch_children.py` |
| `tick` scans: failure isolation across pages, visible error groups, `RuntimeNotReady`, SQL retry backoff, fencing against lost leases and concurrent schedule writers | `src/vv_agent/session/supervisor.py` | `tests/session/test_supervisor.py` |
| Host dispatch: duplicate, reordered, lost and delayed wake messages; concurrency; lease expiry; races with a holder that is exiting; approvals, children and cancellation over an at-least-once transport | `src/vv_agent/session/supervisor.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_dispatch.py` |
| Seeded fault repetition | `src/vv_agent/session/kernel.py`, `src/vv_agent/kernel/_sql.py` | `tests/session/test_soak.py`; `scripts/session_kernel_soak.py --assert` for release-scale runs |
| Long logs, prefix cache and capacity | `src/vv_agent/kernel/_sql.py`, `src/vv_agent/session/kernel.py` | `tests/session/test_capacity.py`; `scripts/session_kernel_benchmark.py --assert-capacity` against real PostgreSQL |
| Provider usage and billing during recovery | `src/vv_agent/session/kernel.py`, `src/vv_agent/session/providers.py` | `tests/session/test_provider_usage.py`, `tests/session/test_recovery_matrix.py` |
| Built distributions and optional extras (base, `postgres`, `s3`, `postgres,s3`) in fresh virtual environments | packaging | `scripts/install_matrix.py` |

## Fixtures and public API

`tests/session/test_session_kernel_fixtures.py` generates the session kernel
fixtures from the real producers and compares them byte-for-byte with the
vendored snapshot. `tests/test_parity_evidence_manifests.py::test_public_api_manifest_resolves_real_python_exports`
resolves the exports and verifies the members listed in the public API v15
inventory, including declared signatures. It does not reject unlisted exports.
The snapshot integrity check runs separately; see
[parity-contract.md](parity-contract.md).

Performance numbers live in
[session-kernel-baseline.md](session-kernel-baseline.md). Host upgrade steps
live in [upgrading.md](upgrading.md).

## Mechanical checks

Run `uv run python scripts/check_repository_contracts.py` from the repository
root. It uses only the standard library and reads Python syntax without
importing or running tests. Its focused self-tests are in
`tests/test_repository_contracts.py`.

Capability declarations are rows in a table with `Capability ID` and `Tests`
columns. Each ID is a lowercase kebab-case string. Each test node in that row
must have a direct decorator with that exact ID. Multiple IDs use separate
decorators; module marks, class marks, computed IDs and aliases are unsupported.

```python
@pytest.mark.capability("read-file")
def test_workspace_tool_parity(...):
    ...
```

The checker compares exact ID/node pairs in both directions. Nodes name
module-level test functions or methods under classes whose names start with
`Test`. Method nodes use two double-colon separators:

```text
tests/path.py::TestClass::test_method
```

Nested local functions, imported tests and generated parameter IDs are not definitions in
this syntax. This checker proves static definitions and declared capability
marks, not whether a test proves the prose. Actual collection and execution
remain the responsibility of pytest and CI.

Active Markdown files under `docs/` also have reference checks. Use a single
backtick pair around each reference, entirely on one line:

- Literal paths starting with `src/`, `tests/`, `scripts/`, `docs/` or
  `examples/` must exist, including directory references with a trailing slash.
  These roots always mean this repository; use a Markdown link or an explicit
  sibling-relative path for another repository, even when prose names it.
- Use the repo: prefix before another repository-relative path, such as
  `repo:pyproject.toml`. Paths must stay within the repository.
- Use the py: prefix before a Python file under `src/`, `tests/` or `scripts/`, followed
  by `::` and a lexical class or function definition. For example,
  `py:src/vv_agent/runtime/compiler.py::tool_is_enabled`.
  Methods and nested classes use dotted names. Imports, aliases, variables,
  instance attributes and dynamically generated exports are not resolved.
- Explicit test nodes also check their Python definitions outside capability
  rows. A bare test name in this matrix is rejected; give its full node.

Fenced blocks are examples and are not scanned. Commands, globs, placeholder
paths, bare symbols and paths relative to an unspecified module are not
references in this syntax. Only the historical release notes under
`docs/releases/` are excluded. Templates, examples and similarly named nested
directories remain checked, as do current guides such as `docs/upgrading.md`.
There are no per-reference waivers.
