# Architecture

`vv-agent` is a Python agent runtime extracted from VectorVein's production
runtime. It is organized around a cycle loop: prepare context, call an LLM,
dispatch tool calls, update memory/state, and repeat until an explicit tool
directive or the configured no-tool policy ends or pauses the run.

## Top-Level Flow

```text
Runner / ConfiguredRunner / InteractiveAgentClient / CLI / AppServer
  -> SessionDriver and RunHandle
  -> SessionStore log, inbox, leases and consumer cursors
  -> session.kernel.drive
      -> frozen AgentTask, model operation and tool operation plans
      -> provider receipts and retained hook/memory boundaries
      -> atomic child admission and authenticated terminal delivery
  -> RunResult, RunEvent, tracing and App Server projections
```

Ordinary runs use SQLite `:memory:`. Durable stores are opt-in. Default no-tool
policy is finish; explicit continue and wait_user retain their configured cycle
budget. Waiting is non-terminal and replies resume the same turn. Resource
budgets govern admission and accounting, not semantic task completion.

## Runtime Boundary

`vv-agent` is the framework boundary. It owns the portable agent contract:

- `Agent`, `Runner`, `RunConfig`, `RunHandle`, `RunResult`, and typed
  `RunEvent` objects.
- Prompt assembly, model calls, tool planning, tool dispatch, approval
  interruption, cancellation, memory compaction, and runtime hooks.
- Replayable app history through `SessionRunEventStore`; `JsonlRunEventStore` is an
  optional file projection sink.
- Tool execution through `vv_agent.tools.ToolExecutor` and
  `vv_agent.tools.ToolOrchestrator`, with `FunctionTool` and `@function_tool`
  as the normal public path.

Host products own product concerns outside the framework: product UI, account
and profile resolution, workspace selection, product persistence, browser or IM
integration, and product-specific tools. They should connect those concerns by
implementing providers instead of patching runtime internals:

- `ApprovalProvider` for UI prompts, policy checks, and allow/deny decisions.
- `ContextProvider` for product prompt fragments such as profile, workspace,
  policy, and feature context.
- `vv_agent.memory.MemoryProvider` for product memory search/save and
  compaction lifecycle integration.
- `vv_agent.tools.ToolExecutor` or `FunctionTool` collections for product
  tools.
- `SessionRunEventStore` for app history and parent/child run graph replay.
- `AfterCycleHook` for an optional task-neutral observation/control point after
  a complete cycle. It may steer the next cycle, add tool denials, or stop with
  failure; it cannot expand permissions or manufacture success/waiting states.

The public runtime event boundary is the closed `RunEvent` hierarchy. Runtime
producers create lifecycle events directly, and LLM adapters project only valid
assistant/reasoning deltas and model tool-call start/progress events before the
payload leaves the adapter boundary. Model tool generation uses
`model_tool_call_*`; actual tool execution uses `tool_call_planned`,
`tool_call_started`, and `tool_call_completed`, with parked operation records for
admitted durable external work. Unknown or malformed provider
payloads are dropped. Reasoning remains private telemetry and is not rendered
as App Server answer text.

Token accounting keeps provider truth separate from derived aggregates.
`TokenUsage.usage_source` identifies provider-reported, estimated, or missing
totals. `CacheUsage` distinguishes an explicit zero cache read from missing
accounting and adapter-declared lack of support. `TaskTokenUsage` exposes a
cache total only when every included cycle reports that metric.

## Module Map

| Path | Responsibility |
| --- | --- |
| `src/vv_agent/canonical_json.py` | RFC 8785 encoding, UTF-16 key ordering, canonical SHA-256 digests, and digest validation. |
| `src/vv_agent/interaction.py` | Host-interaction request values, closed wire validation, and request digests. |
| `src/vv_agent/tools/metadata.py` | Tool capability metadata and idempotency declarations. |
| `src/vv_agent/tools/outcomes.py` | Tool-call outcomes, provider outcomes and definitive-result validation; stores own admission and receipts. |
| `src/vv_agent/llm/errors.py` | Provider prompt-too-long classification and its retry limit, shared by model callers and compaction. |
| `src/vv_agent/config.py` | Settings-file loading, provider/backend lookup, endpoint resolution, and `vv-llm` settings construction. |
| `src/vv_agent/cli.py` | Command-line argument parsing and one-shot runtime execution. |
| `src/vv_agent/agent.py` | Public `Agent` definition and agent-as-tool helpers. |
| `src/vv_agent/background_task.py` | Non-blocking background agent task, handle, and snapshot contracts. |
| `src/vv_agent/runner.py` | Public synchronous run and stream entry points. |
| `src/vv_agent/run_handle.py` | Live `Runner.start()` handle for event streaming, cancellation, approvals, and final result retrieval. |
| `src/vv_agent/run_config.py` | Per-run configuration, model provider binding, tool policy, workspace, session, and tracing options. |
| `src/vv_agent/model_settings.py` | Model call parameters and override merging. |
| `src/vv_agent/output_validation.py` | Typed host output-validation result, context, and tools-free repair request contracts. |
| `src/vv_agent/events.py` | Typed run events and dict conversion for UI consumers. |
| `src/vv_agent/event_store.py` | Run event persistence and replay protocol plus JSONL implementation. |
| `src/vv_agent/approval.py` | Approval provider protocol, request/decision objects, and in-process approval broker. |
| `src/vv_agent/context_providers.py` | Context provider protocol and deterministic prompt-fragment assembly. |
| `src/vv_agent/guardrails.py` | Public guardrail result contract and decorators. |
| `src/vv_agent/interactive.py` | Public stateful session/client API for desktop runtimes, interruptions, follow-ups, cancellation, and shared tool state. |
| `src/vv_agent/result.py` | Public `RunResult` wrapper around runtime results. |
| `src/vv_agent/sessions/` | Retired transcript stores awaiting F3b deletion. |
| `src/vv_agent/tracing.py` | Public trace spans and processor protocol. |
| `src/vv_agent/runtime/compiler.py` | Compile layer: `Agent + input + RunConfig -> AgentTask`. Import this submodule directly to avoid runtime package initialization cycles. |
| `src/vv_agent/types.py` | Runtime protocol types: tasks, messages, tool calls, results, statuses, and token usage. |
| `src/vv_agent/llm/` | LLM protocol adapters, scripted test clients, prompt cache behavior, and `vv-llm` client bridge. |
| `src/vv_agent/runtime/` | Shared compiler/hooks/cancellation; retired loops and stores await F3b extraction. |
| `src/vv_agent/tools/` | Tool registry, OpenAI-compatible schemas, dispatcher, and built-in handlers. |
| `src/vv_agent/memory/` | Token counting, history-preserving summary compaction, archive-backed microcompaction, and session memory. |
| `src/vv_agent/prompt/` | System prompt construction and prompt-cache section tracking. |
| `src/vv_agent/workspace/` | Local, memory, and S3-compatible workspace storage backends. |
| `src/vv_agent/skills/` | Skill metadata parsing, validation, normalization, and prompt rendering. |

## Execution and persistence

The session log and inbox are the sole execution ledger. SQL transactions atomically
admit ordered plans, reserve budgets and append receipts. An execution lease fences
writes and external dispatch. Consumers have separate acknowledged prefixes and
need no execution lease. Wake is a hint; scanning discovers due work and consumer
lag. Durable effects can remain unknown and must reconcile against trusted evidence.

Blocking and background children use independently scheduled sessions. Parent
admission freezes identity, prompt, policy, model, JSON state and budget. Parent
closure targets live descendants atomically. Intermediate user waits stay on the
child; the parent adopts only an authenticated terminal for the original handle.

The old checkpoint/controller/backend stack is retained only for F3b extraction.
Current helper dependencies and deletion candidates are listed in
[the F3a report](session-kernel-f3a-report.md). See
[session-kernel.md](session-kernel.md) for store transactions and recovery.

## Tool Boundaries

Tool definitions and behavior are intentionally split:

- `tools/base.py`: schema and execution result types.
- `tools/metadata.py`: typed tool capability declarations, normalization, and
  metadata-policy matching.
- `tools/function.py`: public `FunctionTool` and `function_tool` decorator,
  including signature/dataclass/TypedDict/Pydantic schema inference.
- `tools/outputs.py`: structured public tool output variants.
- `tools/registry.py`: registration and lookup. Planner extra tools retain their
  first registration order and are deduplicated, keeping per-task tool schema
  serialization and digest computation deterministic.
- `tools/orchestrator.py`: policy/approval gates and the planned, started, and
  completed executor lifecycle.
- `tools/dispatcher.py`: argument normalization and handler dispatch.
- `tools/handlers/`: concrete built-in behavior.
- `constants/tool_names.py` and `constants/workspace.py`: stable tool names and
  schemas used by prompts/tests.

Do not bury model-visible behavior in ad hoc handler strings without tests. Tool
schema wording is part of the agent contract.

`ToolMetadata` is an optional closed host declaration carried by
`FunctionTool`, `ToolSpec`, `ToolExecutor`, and `ToolRegistry`. It contains:

- one `side_effect` value from `unknown`, `none`, `read`, `write`, `execute`,
  `network`, or `external`, with no inferred hierarchy;
- the existing tool idempotency classification;
- `terminal`, which says only that the tool may return `finish` or `wait_user`;
- opaque exact-match `capability_tags` and `cost_dimensions`. Cost dimensions
  name possible resource kinds; they are not measurements, prices, or budget
  observations.

Tag and cost-dimension lists trim only tab, LF, CR, and ASCII space, reject
blank or over-128-code-point labels, deduplicate, sort by UTF-16 code units, and
allow at most 32 normalized entries. Generic `FunctionTool.metadata` is a
separate host-owned mapping and is never promoted into `ToolMetadata`. Typed
metadata is not added to `to_openai_schema()` or registry schema exports, so it
does not change the model-visible tool contract. `ToolMetadata.idempotency` is
the only idempotency declaration and reaches execution, events, and durable run
definitions. `terminal=True` never causes a transition by itself; the result
directive and completion policy remain authoritative.

`Agent.as_tool()` compiles a child agent into a callable tool. The child result
is returned as the tool output and the parent agent keeps control. The tool itself
inherits the active parent scope from `ToolContext`, including when registered
as an executor or invoked through `ToolOrchestrator`. All agent-as-tool calls
use the same child-run path; the Runner does not intercept tool metadata to
start a separate child. Missing runtime/provider scope returns the existing
`sub_agents_not_enabled` tool error before starting a run. Child cancellation
is linked to the parent, and child session state remains separate. `handoff()`
compiles to a transfer tool whose result uses a finish directive; the target
agent output becomes the run output and a typed `HandoffEvent` is emitted.

`ToolPolicy` is enforced during schema planning and again at executor dispatch.
In addition to `allowed_tools`, `disallowed_tools`, and `can_use_tool`, it has
`denied_side_effects`, `denied_capability_tags`, `deny_terminal_tools`, and
`denied_cost_dimensions`. List values form a normalized set union across Agent,
configured Runner, and per-run layers; the terminal boolean uses logical OR.
Configured sub-agents, agent-as-tool runs, and handoff targets inherit the
effective parent denials and may only add more. Child admission freezes
that already-effective policy instead of creating another permission layer.

These fields only deny declared capabilities. They use exact matching, return
the existing `tool_not_allowed` error, and report policy sources in
side-effect, terminal, capability-tag, then cost-dimension order. They cannot
add a tool or bypass name, argument, approval, budget, planned-name, or runtime
checks. A tool without typed metadata matches none of the metadata denials.

After tool name and arguments normalize, `ToolOrchestrator` emits
`tool_call_planned` before policy and approval. It emits `tool_call_started`
only immediately before the executor may cause effects, then emits
`tool_call_completed` after a `ToolExecutionResult` exists. Parse failures have
no tool lifecycle. Unknown tools, policy denials, and approval short-circuits
have planned plus completed but no started event. Completed events add the
result directive, nullable error code, `execution_started`, nullable monotonic
`duration_ms`, and the optional declaration. Cancellation or process loss may
leave a started event without completion; retained session operations own ambiguity and recovery.

When no typed declaration exists, metadata-denial fields do not match that
tool. Telemetry observation does not change result, policy, approval,
completion, or event-store failure semantics.

`FunctionTool.needs_approval` and `ToolPolicy.approval="always"` interrupt tool
execution with a wait-user directive before user code runs. The runtime emits
`ApprovalRequestedEvent` directly. `ToolPolicy.approval="never"` skips
that approval gate for trusted runs. `approval="default"` is the unset merge
sentinel, while explicit `approval="on_request"` overrides lower layers and
follows the selected tool's static or dynamic approval declaration. `always`
and `never` do not evaluate dynamic tool approval predicates.

Interrupted results retain session_id and turn_id. Replies enter the inbox;
Runner.resume(session_id, turn_id) drives the same retained turn. No RunState or
checkpoint key is part of the public result.

## Guardrails And Tracing

Input guardrails run inside `Runner` before model/provider resolution. A block
or approval requirement returns a failed run result without calling the model.
Rewrite results replace the user input before the runtime task is compiled.

Output guardrails run after the runtime returns. Rewrite results replace
`RunResult.final_output`; block and approval results convert the public run
status to failed and expose the guardrail message.

Optional output validation is a separate default-off host extension. After the
existing terminal observation is persisted, an explicitly enabled typed
validator may accept the final value or return a coded rejection. A host repair
callback may supply one replacement, which is coerced through `output_type` and
validated again. The repair request has no tools, and the framework does not
infer task or answer semantics. See `output-validation.md`.

Trace processors are read from `RunConfig.tracing["processors"]`. `Runner`
starts a `run` span for each invocation and starts/ends `tool` spans from typed
tool events emitted by the runtime.

## Workspace Boundary

File tools must go through `WorkspaceBackend`. Local filesystem access,
in-memory storage, and S3-compatible storage should keep the same behavior for
read/write/list/grep semantics wherever practical. Path traversal protections
belong at the workspace boundary and are covered by `tests/test_workspace_backends.py`
and `tests/test_tools.py`.

## Invariants

- Model resolution is exact: requested model keys are not aliased to independent
  provider models.
- Runtime terminal states follow no-tool policy, explicit tool outcomes, host
  hooks, cancellation, failure, and resource bounds, not prose heuristics.
- Public SDK code should enter through `Agent`, `Runner`, `RunConfig`,
  `ModelSettings`, tools, sessions, typed `RunEvent` objects, or
  `InteractiveAgentClient` for stateful host-controlled runtimes.
- Long outputs should keep structured data in metadata and model-facing text in
  content.
- Cancellation, streaming, hooks, memory compaction, and execution backends must
  compose without changing public result shapes.
- New public behavior needs tests in the closest `tests/test_*.py` module.

`read_file` and existing-artifact validation scan native Local, Memory, S3,
and discovery-filtered backends in chunks. SHA-256, UTF-8 validation, line
statistics, and page output derive from the same byte stream; no page or
baseline is published before full-source validation finishes. This bounds
extra memory for native backends while retaining full-source I/O on each page.
Custom backends retain their existing `read_bytes` behavior. No new public
workspace capability or cursor wire is introduced.

## History-Preserving Compaction

`MemoryManager` uses one microcompaction planner/application pass. Relative
transcript age protects recent assistant turns even after recompression. The
planner and application share message indices; empty assistant filtering runs
only after a precomputed plan has been applied. Complete tool blocks and images
are never stripped to make room. Invalid blocks leave history unchanged, except
an ordered incomplete final tool block remains intact in the raw tail.

Normal summaries retain at least `keep_recent_messages` raw messages (default
10), moving the cut left across an atomic assistant/tool-results block, including
an incomplete final block. Image notifications and steering messages after tool
results are independent messages. Prefix images become deterministic text
placeholders (`[image omitted from summary input: <content or image>]`) in the
summary request, without `image_url`; raw-tail images remain unchanged. Only an
accepted summary removes original prefix images. Force skips pruning. Emergency uses the same accepted-summary
path with `max(1, floor(keep_recent_messages * (1 - clamp(ratio, 0, 0.95))))`.

The localized prompt carries JCS `Previous Summary` and `Conversation Prefix`
sections without truncation. Callback output is extracted and normalized to the
closed summary 2.0 shape. Empty effective content, failed callbacks, insufficient
summary-route input capacity, unavailable recovery, oversized candidates and
candidates without token reduction cannot replace history. Control-flow errors
propagate. The standalone local-summary helper serves its fixture only; it is
never a fallback authorizing history replacement.

Accepted output contains original system messages, one user `memory_summary`,
and the unchanged tail. `_vv_agent_compaction` metadata merges complete artifact
and cursor records, previous evidence first, with stable whole-record JCS
deduplication. Hashes remain host-only; the `Persisted Artifacts` section exposes
paths and normal read hints. File actions merge by first-seen path without
filesystem access. A final check after `before_llm` requires a visible recovery
tool for summary evidence as well as compacted tool markers. Session Memory
receives `on_compaction` only after acceptance; its token baseline includes the
tail and the frozen system prompt remains unchanged.
