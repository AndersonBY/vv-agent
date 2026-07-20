# Architecture

`vv-agent` is a Python agent runtime extracted from VectorVein's production
runtime. It is organized around a cycle loop: prepare context, call an LLM,
dispatch tool calls, update memory/state, and repeat until an explicit tool
directive or the configured no-tool policy ends or pauses the run.

## Top-Level Flow

```text
Public SDK
  -> Agent / RunConfig / ModelSettings
  -> Runner
  -> RunHandle for live runs
  -> RuntimeTask
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> RunResult / RunEvent / RunEventStore replay

Interactive session SDK
  -> InteractiveAgentDefinition / AgentSessionOptions
  -> InteractiveAgentClient
  -> AgentSession
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> AgentSessionRun / AgentSessionState

CLI / legacy runtime API
  -> config.load_llm_settings_from_file
  -> config.resolve_model_endpoint
  -> llm.VVLlmClient
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> AgentResult
```

The backward-compatible default remains tool-driven: `task_finish` ends the run
and `ask_user` waits for input. Hosts can explicitly set `no_tool_policy` to
`finish` or `wait_user` when a normal assistant response should be terminal.
The runtime applies that declared control without classifying the response text
or inferring task-specific completion.

## Runtime Boundary

`vv-agent` is the framework boundary. It owns the portable agent contract:

- `Agent`, `Runner`, `RunConfig`, `RunHandle`, `RunResult`, and typed
  `RunEvent` objects.
- Prompt assembly, model calls, tool planning, tool dispatch, approval
  interruption, cancellation, memory compaction, and runtime hooks.
- Replayable app history through `RunEventStore`; `JsonlRunEventStore` is the
  built-in file-backed implementation.
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
- `RunEventStore` for app history and parent/child run graph replay.
- `AfterCycleHook` for an optional task-neutral observation/control point after
  a complete cycle. It may steer the next cycle, add tool denials, or stop with
  failure; it cannot expand permissions or manufacture success/waiting states.

Raw runtime logs remain available for compatibility, but typed `RunEvent` is
the primary state contract for host UIs.

Raw model stream callbacks are synchronous at-least-once observers, not a
durable event store. The Runner projects only assistant/reasoning deltas and
model tool-call start/progress into typed events with framework-owned identity
and cycle fields. Model tool generation uses `model_tool_call_*`; actual tool
execution continues to use `tool_call_started` / `tool_call_completed`.
Reasoning remains private telemetry and is not rendered as App Server answer
text. Unknown or malformed raw stream payloads stay raw-only.

Token accounting keeps provider truth separate from compatibility values.
`TokenUsage.usage_source` identifies provider-reported, estimated, or missing
totals. `CacheUsage` distinguishes an explicit zero cache read from missing
accounting and adapter-declared lack of support. `TaskTokenUsage` exposes a
cache total only when every included cycle reports that metric; legacy numeric
fields remain available but do not prove cache-accounting availability.

## Module Map

| Path | Responsibility |
| --- | --- |
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
| `src/vv_agent/sessions/` | Public `Session` protocol plus memory, SQLite, and Redis implementations. |
| `src/vv_agent/tracing.py` | Public trace spans and processor protocol. |
| `src/vv_agent/runtime/compiler.py` | Compile layer: `Agent + input + RunConfig -> RuntimeTask`. Import this submodule directly to avoid runtime package initialization cycles. |
| `src/vv_agent/types.py` | Runtime protocol types: tasks, messages, tool calls, results, statuses, and token usage. |
| `src/vv_agent/llm/` | LLM protocol adapters, scripted test clients, prompt cache behavior, and `vv-llm` client bridge. |
| `src/vv_agent/runtime/` | Core loop, cycle execution, hooks, cancellation, backends, checkpoint stores, and sub-task coordination. |
| `src/vv_agent/tools/` | Tool registry, OpenAI-compatible schemas, dispatcher, and built-in handlers. |
| `src/vv_agent/memory/` | Token counting, compaction, micro-compaction, session memory, and post-compaction file restoration. |
| `src/vv_agent/prompt/` | System prompt construction and prompt-cache section tracking. |
| `src/vv_agent/sdk/` | Internal migration namespace for session/sub-task compatibility. New user code should import from `vv_agent` top level. |
| `src/vv_agent/workspace/` | Local, memory, and S3-compatible workspace storage backends. |
| `src/vv_agent/skills/` | Skill metadata parsing, validation, normalization, and prompt rendering. |

## Execution Backends

- `InlineBackend`: default synchronous cycle execution.
- `ThreadBackend`: non-blocking submission with futures.
- `CeleryBackend`: distributed cycle execution. Distributed mode requires a
  `RuntimeRecipe` and a shared `StateStore`; otherwise it falls back inline.

Checkpoint stores live under `runtime/stores/` and support SQLite and Redis.
Backends must preserve the same `AgentResult` and checkpoint payload shape as
inline execution.

Optional run budgets are evaluated at stable runtime boundaries shared by all
backends. Inline and thread runs keep one evaluator for the active run.
Distributed limits travel in each envelope, while cumulative usage is stored
in the checkpoint so each worker adds only its active monotonic segment. Host
cost remains a worker-local capability and is never reconstructed from an SDK
price table. See `run-budgets.md` for public API and terminal precedence.

Distributed mode sends a versioned `DistributedRunEnvelope` for each cycle.
Workers resolve all referenced capabilities before claiming state, then use a
revision/token lease with heartbeat renewal and CAS commit. The scheduler
accepts a result only after reconciling it with the durable checkpoint;
terminal checkpoints are immutable and replayable until acknowledged. SQLite
uses WAL, a bounded busy timeout, and in-place legacy-column migration.
Before entering the runtime cycle, a worker must complete one successful lease
renewal; initial and renewed lease expiry never extends beyond the job deadline.
Each periodic wait is derived from that renewal's actual deadline-clamped lease,
not only from the configured duration. A renewal result must return before both
the previously known expiry and the new expiry it requested. Response checks
use the conservative maximum of current wall time and request-start wall time
plus monotonic elapsed time, covering wall-clock jumps in either direction.
SQLite refreshes effective time after acquiring its write lock. Redis renewal
uses one atomic script: Redis `TIME` validates both expiries and the original
JSON is the compare-and-set value, so an expired or replaced owner cannot write
a new expiry. The script distinguishes CAS loss from authoritative expiry, so
commit-race suppression can apply only to claim consumption and never to an
expired lease; authoritative expiry takes precedence when both conditions are
observed. Heartbeat renewal uses an independent store connection and remains
active through an explicit commit phase. A durable commit suppresses only an
active-claim rejection from a renewal that started in that commit phase and
returned before its applicable lease expiry. Renewals that started before
commit, expired leases, and other coordination failures remain visible even if
the checkpoint commit later succeeds.
Redis connection I/O and non-renewal optimistic-transaction retries are bounded
so stopping or unwinding a worker cannot wait forever on the heartbeat thread.

This is an at-least-once execution model. Celery revoke during cancellation is
best effort, and an active worker claim may still complete after the scheduler
stops waiting. The cycle idempotency key does not provide an event outbox,
durable cancellation record, or idempotency for external tool side effects.
See `parity-contract.md` for the complete cross-language contract.

## Tool Boundaries

Tool definitions and behavior are intentionally split:

- `tools/base.py`: schema and execution result types.
- `tools/metadata.py`: typed tool capability declarations, normalization, and
  metadata-policy matching.
- `tools/function.py`: public `FunctionTool` and `function_tool` decorator,
  including signature/dataclass/TypedDict/Pydantic schema inference.
- `tools/outputs.py`: structured public tool output variants.
- `tools/registry.py`: registration and lookup.
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
allow at most 32 normalized entries. Generic `FunctionTool.metadata` remains a
separate compatibility mapping and is never promoted into `ToolMetadata`.
Typed metadata is not added to `to_openai_schema()` or registry schema exports,
so it does not change the model-visible tool contract.

The legacy public `idempotency` input remains effective when typed metadata is
absent. A typed `unknown` value may inherit a non-`unknown` legacy value; two
different non-`unknown` declarations fail before model or tool work. When the
typed declaration is present, its normalized effective value reaches the
executor, event metadata, and checkpoint v2. With no typed declaration, the
legacy value still reaches the executor and legacy run-definition field, while
event metadata stays absent. `terminal=True` never causes a transition by
itself; the existing result directive and completion policy remain authoritative.

`Agent.as_tool()` compiles a child agent into a callable tool. The child result
is returned as the tool output and the parent agent keeps control. `handoff()`
compiles to a transfer tool whose result uses a finish directive; the target
agent output becomes the run output and a typed `HandoffEvent` is emitted.

`ToolPolicy` is enforced during schema planning and again at executor dispatch.
In addition to `allowed_tools`, `disallowed_tools`, and `can_use_tool`, it has
`denied_side_effects`, `denied_capability_tags`, `deny_terminal_tools`, and
`denied_cost_dimensions`. List values form a normalized set union across Agent,
configured Runner, and per-run layers; the terminal boolean uses logical OR.
Configured sub-agents, agent-as-tool runs, and handoff targets inherit the
effective parent denials and may only add more. Distributed execution carries
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
leave a started event without completion; checkpoint v2's operation journal,
not telemetry, owns ambiguity and recovery.

With no typed declaration and empty metadata-denial fields, schema planning,
dispatch, completion, and model-visible schemas retain their previous behavior.
Telemetry observation does not change result, policy, approval, completion, or
event-store failure semantics.

`FunctionTool.needs_approval` and `ToolPolicy.approval="always"` interrupt tool
execution with a wait-user directive before user code runs. The runtime log is
converted to `ToolApprovalRequestedEvent`. `ToolPolicy.approval="never"` skips
that approval gate for trusted runs. `approval="default"` is the unset merge
sentinel, while explicit `approval="on_request"` overrides lower layers and
follows the selected tool's static or dynamic approval declaration. `always`
and `never` do not evaluate dynamic tool approval predicates.

Interrupted results expose `RunState` and structured approval snapshots. An
approved result resume executes the captured tool call once. Live
`ApprovalProvider` runs remain active and continue to use `ApprovalBroker` plus
`RunHandle.approve()`. An `allow_session` decision grants only the same tool for
the lifetime of that broker.

Session persistence stores the complete current-turn message delta, including
assistant tool calls and tool results, so the next model request receives an
executable conversation history rather than a reconstructed summary pair.

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
- Runtime terminal states are explicit tool outcomes, not prose heuristics.
- Public SDK code should enter through `Agent`, `Runner`, `RunConfig`,
  `ModelSettings`, tools, sessions, typed `RunEvent` objects, or
  `InteractiveAgentClient` for stateful host-controlled runtimes.
- Long outputs should keep structured data in metadata and model-facing text in
  content.
- Cancellation, streaming, hooks, memory compaction, and execution backends must
  compose without changing public result shapes.
- New public behavior needs tests in the closest `tests/test_*.py` module.
