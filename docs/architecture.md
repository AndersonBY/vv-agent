# Architecture

`vv-agent` is a Python agent runtime extracted from VectorVein's production
runtime. It runs a cycle loop: build the context, call the model, run the tool
calls the model asked for, update memory and state, and repeat. The loop stops
or pauses when a tool returns an explicit directive (`finish` or `wait_user`),
or when the model answers without tools and the no-tool policy says to finish
or wait.

## Top-level flow

```text
Runner / ConfiguredRunner / InteractiveAgentClient / CLI / AppServer
  -> SessionDriver and RunHandle
  -> SessionStore: log, inbox, leases and consumer cursors
  -> session.kernel.drive
      -> frozen AgentTask, planned model operations and tool operations
      -> provider receipts, hook and memory boundaries
      -> child session admission and delivery of the child's final result
  -> RunResult, RunEvent, tracing and App Server projections
```

Every entry point runs through the session kernel. An ordinary `Runner` run
uses an in-memory SQLite store (`:memory:`). Hosts that need durable sessions
supply their own store.

`session/kernel.py` coordinates the drive loop. `kernel/_lease_scope.py` owns
lease rules; `session/_driver_scope.py` supplies synchronization primitives,
heartbeat scheduling and the shared invocation boundary. Public
`HeartbeatStopped` resolves to the same core exception class.
`session/_driver_dispatch.py` owns operation execution and tool finalization.
The driver keeps its synchronous method entry points and delegates to that one
implementation. `session/_driver_inputs.py` owns input acceptance, child replies
and turn admission. `session/_driver_planning.py` builds operation/model plans;
`session/_driver_recovery.py` advances, recovers and closes turns. The driver
keeps one state-owning instance and its synchronous method entry points.

Importing `vv_agent`, its session stores or the App Server does not load model
provider SDKs. The `VvLlmClient` and `EndpointTarget` exports load the `vv-llm`
adapter when accessed. Model execution also loads it when the kernel checks
endpoint bindings. These exports resolve to the same classes as direct imports
from `vv_agent.llm.vv_llm_client`.

Root SDK exports resolve lazily to the original objects in their owning
modules. Importing `vv_agent.kernel` or `vv_agent.kernel._schema` through normal
Python package imports loads only the package facade and the requested core
modules, using the standard library alone. Accessing a root SDK export may load
its outer SDK dependencies.
The App Server package also resolves exports lazily, so importing its host or
session adapter does not initialize the client and server through the package
facade.

Canonical JSON encoding, UTF-16 key ordering, canonical SHA-256 digests and
digest validation are owned by `src/vv_agent/kernel/_canonical_json.py`. The
implementation is shared by its active consumers and supports ordinary package
imports using only the standard library.

The extracted schema core owns the closed record, inbox, handle and boundary
schema definitions and their compiled-check registry in
`src/vv_agent/kernel/_record_schema.py`, using the standard-library-only compiler
in `src/vv_agent/kernel/_schema.py`. Both modules support ordinary package imports
without loading the outer SDK. `session/records.py` consumes these same schema
objects and checks. `kernel/_record_codec.py` owns standard-library object
loading, duplicate-member rejection and schema validation. Closed-schema
failures use the contract 39 diagnostic `invalid session schema`; specific
semantic/parser errors and canonical bytes remain unchanged.
`kernel/_records.py` owns Record, InboxItem, SessionSpec, retained fragments and
message interning with stdlib/core dependencies only. `session/records.py`
exports the same actual objects; no second Record implementation exists.

Message values, artifact references, cursors and compaction evidence validation
are owned by `kernel/_message_values.py`; the strict Message decoder is owned by
`kernel/_message_codec.py`. `types.py` and existing root exports resolve to these
same objects. Both core modules use only the standard library and core canonical
JSON helpers. Message metadata and tool arguments retain their signed-64 through
unsigned-64 integer range; canonical digest encoding and artifact sizes/cursor
offsets retain the narrower I-JSON safe-integer range.

Tool calls, directives, result statuses and the complete `ToolExecutionResult`
value/validator are owned by `kernel/_tool_values.py`. It uses only the standard
library and the existing canonical JSON and Message values. Existing `types.py`
and root bindings refer to the same actual classes and enums; compaction memory
and reducer result validation import them directly from core. Unknown/null field,
metadata, status, recovery-pointer and byte-size rules remain part of this one
implementation. Tool execution and artifact persistence remain outside core.

`kernel/_tool_result_checks.py` owns definitive-result and structured-body checks
without importing records or schemas. `kernel/_provider_outcomes.py` owns
Definitive/Accepted/Unknown values and receipt/handle validation, including
provider/request/attempt identity and evidence checks. Existing outer imports
resolve to these actual objects. FunctionProvider, host-interaction outcomes and
workspace preparation remain in the SDK.

`kernel/_tool_schema.py` emits schema-check requests and owns issue normalization
and rejection results. The explicit host leaf `_jsonschema.py` uses the existing
Draft202012 validator and cache. Core imports do not load this leaf or jsonschema;
SDK and host-side simulation adapters use the same real validator, without
preloading the tools package or substituting a second validation algorithm.

Tool completion-policy application, image notifications and skipped-result
construction live in `kernel/_tool_behavior.py`. `runtime/tool_results.py`
exports the same functions, and context projection imports the image helper
directly from core. Stop policies mutate the existing result directive.

Model settings, tool choice, response format and retry settings are owned by
`kernel/_model_settings.py`. Microcompaction policy and normalization are owned by
`kernel/_microcompaction_policy.py`; tool metadata, its enums, normalization and
metadata-denial rules are owned by `kernel/_tool_metadata.py`. Their existing
outer modules explicitly export the same objects for active public and SDK callers.

Resolved prompt sections/bundles, immutable metadata helpers and stable prompt
digests are owned by `kernel/_prompt_values.py`. `prompt/builder.py` and the prompt
package expose the same resolved values. `SystemPromptBuilder`, template rendering,
workspace/path resolution, Session Memory insertion and current-time capture stay
outside core.

`kernel/_definition.py` owns the single frozen-definition/schema cache, canonical
composition and digest, model identity filtering and historical endpoint
adaptation. Runtime supplies resolved binding facts and the read-only tool
catalog. `kernel/_tool_planner.py` owns tool selection and metadata filtering;
SDK compilation and shell-hint resolution remain outer. Builtin tool names and
the workspace tool list have one core owner shared by SDK exports and the schema
generator.

Budget values and validation are owned by `kernel/_budget_values.py`.
`kernel/_budget.py` owns budget state, reservations, limits and unavailable-metric
latches. It requests clock and host-cost observations; `budget.py` performs those
reads while core evaluation is suspended and preserves the synchronous SDK API.
Task values import the core budget objects without loading the live adapter.

Pure compaction markers, rendering and recovery-envelope functions are owned by
`kernel/_compaction_text.py`. Memory code reuses those actual objects, and the
reducer imports the core renderer. `kernel/_microcompaction.py` owns candidate
planning and Message replacement; its planners and summary acceptance in
`kernel/_summary_compaction.py` yield internal `TokenCount` requests. The core
fold delegates through these generators without invoking the SDK token counter.
The synchronous SDK Fold and MemoryManager drivers call the existing counter
while the core generator chain is suspended, then send the result back. Counts
are requested only at the original decision points, without model pre-scanning,
count-result caching or fold replay. MemoryManager retains its counting hooks,
model callbacks, archive I/O and SessionMemory notification; accepted notifications
reuse the count computed during validation. The public microcompaction planner
remains synchronous, with one authoritative core generator algorithm.
`kernel/_compaction_phase.py` owns compaction scheduling, protected-tail and
PTL decisions, summary receipt adoption and extraction ledger decisions.
`kernel/_microcompaction_apply.py` owns microcompaction application. The SDK
resolves counting, archive, routing and memory callbacks at their existing
points. The phase keeps one settings mapping and derives capacity and threshold
only when each branch needs them. MemoryManager owns defaults and normalization;
SessionMemory owns its extraction, merge and persistence implementation.
Driver plan/commit/close still provide the ledger integration.
Compaction capacity and integer coercion are owned by
`kernel/_compaction_threshold.py`. `MemoryManager._definition_settings()`
provides Runtime with the current settings projection; its constructor remains
the single source of defaults and policy normalization.
The ordinary token counter retains its production model dispatch, incremental
cache, formatting and per-call fallback behavior. CoreFold's synchronous
convenience entry requires a CPU-only counter; the default SDK drives
`extend_steps` outside core. This boundary does not complete W9 or P1; model
execution and other SDK effects remain outside core, and later W9 public
surfaces and staging obligations remain open.

Lease ownership, renewal publication, expiry checks, commit/poll latch ordering,
heartbeat error-budget decisions and control matching live in
`kernel/_lease_scope.py`. The SDK uses those actual methods and supplies the
existing Clock port, synchronization objects and actor identity. Core calls the
injected clock at the original sampling points; it does not read system time or
construct threads, events, locks or thread-local state. SDK constructors, keeper
scheduling, waits/joins, PG timeout setup and the `_invoke` bridge remain in
`session/_driver_scope.py` / `session/timing.py`. Public HeartbeatStopped bindings
resolve to the single core exception class. Invocation uses the real provider
outcomes; no duplicate outcome or cancellation-error type is introduced.

`kernel/_cancellation.py` owns cancellation state, callback ordering and
parent-child propagation. The SDK supplies its existing Event, Lock and child
factory, retaining the public token constructor, child return type and exception
identity used by tool consumers.

ScheduledLeaseKeeper runs the same core heartbeat step and transient-error budget
as the SDK. Recoverable renewal errors retain the last confirmed lease without
publishing readiness; terminal errors publish completion and cancel a scoped
token. Normal stop drains queued renewal work without implying lease loss.
Renewal turns are explicitly scheduled by seeded scenarios. SimWorld has no timed
queue, and a synchronous factory readiness wait does not pump queued retries;
recurring scheduling and factory-wait equivalence are not established.

Execution statuses, tasks and child configuration, usage accounting, cycle/model
call records and final results are owned by `kernel/_execution_values.py`, together
with their existing validation and wire versions. The module imports only stdlib
and other core values. `types.py` remains the explicit existing export surface for
the same actual objects, including the canonical `vv_agent.types.AgentResult`
binding. Private helper exports remain only where SDK or producer callers use them.
`core_types.py` owns that export surface. `types.py` forwards the existing
bindings until the next Python package major release. When found as top-level
`types` from the package working directory, it loads the actual standard-library
module without importing the agent package; see the Python contract mapping.

After-cycle decisions, native outcomes, detached snapshots, validation errors and
durable denial-state helpers are owned by `kernel/_cycle_values.py`. Their
validation and UTF-16-sorted state updates use stdlib and core values only.
`runtime/lifecycle.py` exports those same objects and retains the hook protocol
and `AfterCycleHookManager`, including callback execution and decision composition.
The reducer validates after-cycle and budget boundaries through core values.
`kernel/_cycles.py` owns selected-cycle projection, shared by SDK results and
`kernel/_after_cycle.py`. The after-cycle phase builds snapshots and records
hook decisions; the SDK invokes user hooks and supplies bound shared state.
Before hooks, the SDK also projects retained events to enforce their descriptor
bounds. It does not rebuild the full SDK result merely to read cycles and usage.
`kernel/_output_phase.py` owns output receipt reuse, validation outcomes and the
single logged, tools-free repair plan. The SDK invokes guardrails, coercion and
validators at the original observation points; mutable output settings are read
when their branches need them. Session output and RunResult share one serializer.
Serialization can invoke model_dump or dataclass callbacks; the host handles
that request while the output phase generator is suspended.
The compaction, output and after-cycle phases share ledger request types from
`kernel/_driver_requests.py`; their existing adapters retain the same planning,
boundary and guarded-commit calls.
`kernel/_state.py`, `kernel/_context.py` and `kernel/_reducer.py` own execution
state, aligned context projection, input protection and all record transitions.
The fold receives settings normalization and message-counting callables. It
still recomputes compaction eligibility, prompts, replacements, protected tails
and evidence; the outer adapter supplies values, not an acceptance decision.
`MemoryManager` and replay share the high-level microcompaction planner.

`kernel/_driver.py` owns the driver's retained state, record construction and
lease-fenced commit algorithm. The SDK driver inherits those same methods;
store, fold, lease scope and watermark are not mirrored in another driver.
Explicit clock, fold and observation callbacks preserve transaction timing,
while the SDK owns cost-meter reads, hooks and log emission.
`session/_driver_dispatch.py` and `session/_driver_recovery.py` retain dispatch
and recovery orchestration.
`kernel/_driver_planning.py` owns operation/model plans and selected-model,
fork, then frozen-task/default precedence. The SDK supplies route bindings,
provider membership and callbacks at their original points. Retained fragments,
idempotency keys and consumed unknowns are constructed by the shared planner.
`kernel/_driver_inputs.py` owns inbox ordering, input identity/disposition and
turn admission, using the same CoreDriver and DefinitionCache. Interaction values
and reply validation, summary intent helpers and provider-result identity also
have single core owners. The SDK supplies compilation, authentication and resource
observations; input consumption and guarded commits keep their original ordering.

`kernel/_store.py` owns store protocols, receipts, leases and errors;
`kernel/_sql.py` owns the shared SQL transaction and scheduling rules.
`kernel/_sqlite.py` and `kernel/_postgres.py` own the concrete connections, DDL
and transactions. These two adapters are the explicit storage-I/O exception;
PostgreSQL imports its driver lazily. Core stores require a fold factory, and
SQLite also requires a Clock. Thin `session/` subclasses provide the existing
SDK defaults and public constructors. SQL prefixes, forked folds and the driver
use the same fold factory. The outer context adapter also uses that
factory when a historical prefix must be rebuilt. Replacing a store's factory
clears both validated-prefix caches.

`kernel/_child_store.py` owns child admission, cancellation, terminal delivery,
authenticity checks and ordinary child-result assembly over the store ports.
`session/children.py` exports the same objects; configured-agent and handoff
reconstruction remain in the SDK delegation adapter.

`kernel/_usage.py` owns usage normalization and the model-attempt ledger. SDK
result projection and the simulation's I5 checks consume that same function.
Unselected receipts remain billable, and missing measurements remain distinct
from reported zero usage.

Shared event wire validation lives in `kernel/_event_validation.py`. The outer
event classes retain construction, time and UUID defaults. Retained memory
boundaries use the shared validator before checking their expected event kind,
identities and matching start; cycle-aborted constructor checks retain their
original position after wire validation.

Record task decoding consumes core `AgentTask` and preserves its cached private
value and detached public copies. Its private interning keys remain separate
from canonical wire bytes, and future message validation is still deferred to
its owning boundary. The execution driver and concrete database adapters remain
under `session/`. The seeded simulation currently drives these SDK assembly
paths over the real SQL stores with scripted external services.

The default no-tool policy is `finish`. Choosing `continue` or `wait_user`
does not change the cycle limit. Waiting is not a terminal state: a reply
resumes the same turn. Resource budgets control what may start and how usage
is counted; they do not decide whether the task is done.

## Runtime boundary

`vv-agent` owns the portable agent contract:

- `Agent`, `Runner`, `RunConfig`, `RunHandle`, `RunResult`, and typed
  `RunEvent` objects.
- Prompt assembly, model calls, tool planning and dispatch, approval
  interruptions, cancellation, memory compaction, and runtime hooks.
- Replayable run history through `SessionRunEventStore`. `JsonlRunEventStore`
  is an optional sink that writes the same events to a file.
- Tool execution through `vv_agent.tools.ToolExecutor` and
  `vv_agent.tools.ToolOrchestrator`. Most tools should use `FunctionTool` or
  the `@function_tool` decorator.

The host product owns everything else: UI, accounts and profiles, workspace
selection, product persistence, browser or IM integration, and product tools.
Connect those through providers instead of patching runtime internals:

- `ApprovalProvider` shows approval prompts, applies policy, and returns allow
  or deny decisions.
- `ContextProvider` adds prompt fragments such as profile, workspace, policy,
  and feature context.
- `vv_agent.memory.MemoryProvider` searches and saves product memory and takes
  part in compaction.
- `vv_agent.tools.ToolExecutor` or `FunctionTool` collections add product
  tools.
- `SessionRunEventStore` replays run history, including the parent and child
  run graph.
- `AfterCycleHook` observes each completed cycle. It may add steering for the
  next cycle, deny tools, or stop the run with a failure. It cannot grant
  permissions or report success or waiting.

### Events

`RunEvent` is a closed set of event types. The runtime creates lifecycle
events directly. Model adapters forward only valid assistant text, reasoning
deltas, and `model_tool_call_started` and `model_tool_call_progress` events;
they drop unknown or malformed provider payloads before those payloads leave
the adapter.

The `vv-llm` bridge routes streamed tool arguments by call ID first, then by
the index owned by the most recent named call at that index. This lets
sequential calls reuse an index. A named delta without a call ID always
starts a new call, even when its name and index match an earlier call.
Providers that repeat the name on every delta need call IDs to avoid
splitting one call into several. A continuation without either field is
accepted only when there is one call. Arguments and progress events use the
same resolved call. `tests/test_llm_tool_call_streaming.py` covers this
routing and both argument append and snapshot replacement. Providers must
preserve call IDs or stable indexes for interleaved calls; when an upstream
adapter overwrites indexes, some interleaved streams lose their identity.

Two families of events describe tools. `model_tool_call_*` events report the
model generating a tool call. `tool_call_planned`, `tool_call_started`, and
`tool_call_completed` report the runtime executing it. Reasoning is private
telemetry, and the App Server never renders it as answer text.

### Token usage

Token accounting keeps what the provider reported apart from derived totals.
`TokenUsage.usage_source` says whether a total was `provider_reported`,
`estimated`, or `accounting_missing`. `CacheUsage` tells apart an explicit
zero cache read, missing accounting, and an adapter that does not support cache
accounting. `TaskTokenUsage` exposes a cache total only when every included
cycle reported that metric.

## Module map

| Path | Responsibility |
| --- | --- |
| `src/vv_agent/kernel/_schema.py` | Standard-library-only compiler for the closed session-schema vocabulary. |
| `src/vv_agent/kernel/_record_schema.py` | Authoritative closed session schemas and compiled checks. |
| `src/vv_agent/kernel/_record_codec.py` | Schema rejection, duplicate-checking object loading and concise contract 39 diagnostics. |
| `src/vv_agent/kernel/_records.py` | Record, inbox and session values, canonical identity, retained fragments and private message caching. |
| `src/vv_agent/kernel/_canonical_json.py` | RFC 8785 encoding, UTF-16 key ordering, canonical SHA-256 digests, and digest validation. |
| `src/vv_agent/interaction.py` | Typed question and answer values, wire validation, and request digests. |
| `src/vv_agent/tools/metadata.py` | Tool capability metadata and idempotency declarations. |
| `src/vv_agent/tools/outcomes.py` | Tool-call outcomes, provider outcomes, and validation of definitive results. The session store owns admission and receipts. |
| `src/vv_agent/llm/errors.py` | Whole-exception-graph uncertainty veto, prompt-too-long recovery before definitive provider rejection, and safe error diagnostics. |
| `src/vv_agent/llm/http_observation.py` | Ephemeral physical HTTP outcomes for one logical model call, including vendor SDK retries; no payloads enter observation evidence or persistence. Retained responses close with the client. |
| `src/vv_agent/config.py` | Settings-file loading, backend and endpoint lookup, and `vv-llm` settings construction. |
| `src/vv_agent/cli.py` | Command-line parsing and one-shot runs. |
| `src/vv_agent/agent.py` | Public `Agent` definition and agent-as-tool helpers. |
| `src/vv_agent/handoffs.py` | `Handoff` definition and the `handoff()` helper. |
| `src/vv_agent/background_task.py` | Non-blocking background agent task, handle, and snapshot types. |
| `src/vv_agent/runner.py` | Public synchronous run and stream entry points. |
| `src/vv_agent/run_handle.py` | Live `Runner.start()` handle for streaming events, cancellation, approvals, and the final result. |
| `src/vv_agent/run_config.py` | Per-run configuration: model provider, tool policy, workspace, session, and tracing options. |
| `src/vv_agent/budget.py` | Run budget limits, host cost readings, and budget usage snapshots. |
| `src/vv_agent/model_settings.py` | Model call parameters and override merging. |
| `src/vv_agent/output_validation.py` | Output validation result, context, and tools-free repair request types. |
| `src/vv_agent/events.py` | Typed run events and dict conversion for UI consumers. |
| `src/vv_agent/event_store.py` | Run event persistence and replay protocol, plus the JSONL implementation. |
| `src/vv_agent/approval.py` | Approval provider protocol, request and decision objects, and the in-process approval broker. |
| `src/vv_agent/context_providers.py` | Context provider protocol and deterministic prompt-fragment assembly. |
| `src/vv_agent/guardrails.py` | Guardrail result type and decorators. |
| `src/vv_agent/interactive.py` | Stateful session client for desktop hosts: interruptions, follow-ups, cancellation, and shared tool state. |
| `src/vv_agent/result.py` | Public `RunResult` wrapper around runtime results. |
| `src/vv_agent/tracing.py` | Trace spans and the processor protocol. |
| `src/vv_agent/runtime/compiler.py` | Compiles `Agent + input + RunConfig` into an `AgentTask`. Import this submodule directly to avoid package import cycles. |
| `src/vv_agent/types.py` | Existing exports of the actual kernel-owned value objects. Execution/task/usage/result definitions live in `src/vv_agent/kernel/_execution_values.py`. |
| `src/vv_agent/session/` | Session kernel, SQLite and PostgreSQL stores, `SessionDriver`, child sessions, and projections. See [session-kernel.md](session-kernel.md). |
| `src/vv_agent/app_server/` | App Server protocol, transport, and thread mapping. See [app-server.md](app-server.md). |
| `src/vv_agent/llm/` | Model protocol adapters, scripted test clients, prompt cache behavior, and the `vv-llm` client bridge. |
| `src/vv_agent/runtime/` | Compiler, hooks, lifecycle, cancellation, tool planning and results, token usage, and process management. |
| `src/vv_agent/tools/` | Tool registry, OpenAI-compatible schemas, dispatcher, and built-in handlers. |
| `src/vv_agent/memory/` | Token counting, summary compaction, archive-backed microcompaction, and Session Memory. |
| `src/vv_agent/prompt/` | System prompt construction and prompt-cache section tracking. |
| `src/vv_agent/workspace/` | Local, in-memory, and S3-compatible workspace backends. |
| `src/vv_agent/integrations/` | The `SkillIntegration` extension protocol. |
| `src/vv_agent/skills/` | Skill metadata parsing, validation, normalization, and prompt rendering. |

## Execution and persistence

The session log and inbox are the only record of execution. The log is an
append-only list of records for one session. The inbox holds inputs that
arrived but have not been applied yet: user messages, approval answers,
steering, cancellation, child results.

- Admitting an ordered plan of operations, reserving budget for it, and
  appending a result receipt are each atomic SQL transactions, so a crash
  never leaves half of a change.
- A worker must hold the session's execution lease to append records or start
  external work. A lease is a time-limited claim on a session; it stops two
  workers from driving the same session at once.
- Consumers (event projection, tracing, host notifications) each keep their
  own cursor, the position up to which they have acknowledged records. They do
  not need the execution lease.
- `wake(session_id)` is only a hint. Scanning the store finds due work and
  lagging consumers even if a wake is lost.
- A durable side effect can end in an unknown state, for example when the
  process dies mid-call. It stays unknown until trusted evidence, such as a
  `provider_evidence` input from the host, settles it. The kernel does not
  guess.

Blocking and background child agents run as their own sessions, scheduled
independently. When the parent admits a child, it freezes the child's
identity, prompt, policy, model, JSON state, and budget. Closing the parent
targets all live descendants in the same transaction. If a child waits for the
user, that wait stays on the child. The parent accepts only the child's final
result, and only after checking it belongs to the child it started.

See [session-kernel.md](session-kernel.md) for store transactions and recovery.

## Tool boundaries

Tool definitions and behavior live in separate modules:

- `tools/base.py`: schema and execution result types.
- `tools/metadata.py`: typed tool capability declarations, normalization, and
  policy matching.
- `tools/function.py`: `FunctionTool` and `@function_tool`, including schema
  inference from signatures, dataclasses, `TypedDict`, and Pydantic models.
- `tools/outputs.py`: structured tool output variants.
- `tools/registry.py`: registration and lookup. Extra planner tools keep their
  first registration order and are deduplicated, so the tool schema list and
  its digest are the same on every run.
- `tools/orchestrator.py`: policy and approval gates and the planned, started,
  and completed lifecycle.
- `tools/dispatcher.py`: argument normalization and handler dispatch.
- `tools/handlers/`: built-in tool behavior.
- `constants/tool_names.py` and `constants/workspace.py`: stable tool names and
  schemas used by prompts and tests.

Tool schema wording is part of the agent contract, because the model reads it.
Do not change model-visible behavior through ad hoc handler strings without
tests.

### Tool metadata

`ToolMetadata` is an optional declaration a host attaches to a `FunctionTool`,
`ToolSpec`, `ToolExecutor`, or `ToolRegistry` entry. It has these fields:

- `side_effect`: one of `unknown`, `none`, `read`, `write`, `execute`,
  `network`, or `external`. The values have no implied order; denying `write`
  does not deny `execute`.
- `idempotency`: `supported`, `unsupported`, or `unknown`. This is the only
  idempotency declaration. It reaches execution, events, and durable run
  definitions.
- `terminal`: says the tool may return `finish` or `wait_user`. It does not
  end the run by itself; the result directive and completion policy decide.
- `result_retention`: `archive` (default) or `preserve`. Microcompaction never
  archives results of a `preserve` tool.
- `capability_tags` and `cost_dimensions`: opaque labels matched exactly. Cost
  dimensions name kinds of resources a tool may use. They are not prices or
  measurements.

The runtime normalizes each label list. It trims tabs, line feeds, carriage
returns, and spaces from each label, rejects blank labels and labels longer
than 128 code points, removes duplicates, and sorts by UTF-16 code units. A
list may hold at most 32 labels.

`ToolMetadata` never appears in `to_openai_schema()` or registry schema
exports, so declaring it does not change what the model sees.
`FunctionTool.metadata` is a separate free-form mapping for the host; the
runtime never reads it as `ToolMetadata`.

### Agents as tools and handoffs

`Agent.as_tool()` turns a child agent into a tool. The child's output becomes
the tool result, and the parent agent keeps control. The tool takes the
parent's scope from `ToolContext`, whether it is registered as an executor or
called through `ToolOrchestrator`. All agent-as-tool calls use the same child
run path. If no session runtime is available, the tool returns the
`sub_agents_not_enabled` error without starting a run. Cancelling the parent
cancels the child. The child keeps its own session state.

`handoff()` creates a transfer tool whose result carries a `finish`
directive. The target agent's output becomes the run output, and the runtime
emits `HandoffStartedEvent` and `HandoffCompletedEvent`.

### Tool policy

`kernel/_tool_policy.py` owns ToolPolicy, policy merging, compiler metadata
projection and frozen/current context-policy composition. Compiler and Runtime
use these same implementations; execution context and user callback bindings
remain with their SDK owners.
`kernel/_call_context.py` supplies task/plan-derived call facts at the existing
read points and shares the original preapproval comparisons. SDK context
construction still owns workspace, RunContext and child-manager bindings; the
facts object carries none of those host objects.

`kernel/_tool_admission.py` owns argument parsing and the ordered schema,
policy and preapproval decisions before dispatch. The SDK resolves late reads,
runs the existing JSON Schema primitive and invokes callbacks while that core
generator is suspended. FunctionProvider uses the same admission path for both
preflight and submit; a successful preflight does not bypass submit validation.
Approval transport, events and executor effects remain with the SDK.

`ToolPolicy` is checked twice: when the runtime builds the tool schema list
for the model, and again when the executor dispatches a call. Besides
`allowed_tools`, `disallowed_tools`, and `can_use_tool`, it can deny tools by
their metadata:

- `denied_side_effects`
- `denied_capability_tags`
- `deny_terminal_tools`
- `denied_cost_dimensions`

The Agent, the configured Runner, and the per-run config each may set a
policy. List fields combine as a union across the three layers, and
`deny_terminal_tools` is true if any layer sets it. Sub-agents, agent-as-tool
runs, and handoff targets inherit the parent's effective denials and can only
add more. Child admission freezes that combined policy.

Metadata denials can only remove tools. They match exactly and return the
`tool_not_allowed` error. When several apply, the reported source is the first
match in this order: side effect, terminal, capability tag, cost dimension. A
tool with no `ToolMetadata` matches no metadata denial. Metadata denials never
add a tool or skip the name, argument, approval, budget, or runtime checks.

### Tool call lifecycle

`ToolOrchestrator` emits lifecycle events in this order:

1. After it parses the arguments, it emits `tool_call_planned`. This happens
   before policy and approval checks.
2. Right before the executor may cause effects, it emits `tool_call_started`.
3. Once a `ToolExecutionResult` exists, it emits `tool_call_completed`. The
   event includes the result directive, the error code (or null),
   `execution_started`, `duration_ms` (or null), and the tool's metadata
   declaration if any.

Some calls skip steps. A call whose arguments fail to parse emits no lifecycle
events. Unknown tools, policy denials, and approval interruptions emit
`planned` and `completed` but never `started`. If the process is cancelled or
dies during execution, a `started` event can remain without a `completed`
event; the session log then decides how to recover the operation. Event
observers cannot change the result, policy, approval, completion, or
event-store failure behavior.

### Approval

`FunctionTool.needs_approval` and `ToolPolicy.approval="always"` pause the call
with a `wait_user` directive before any user code runs. The runtime emits
`ApprovalRequestedEvent`. The `approval` field takes four values:

| Value | Behavior |
| --- | --- |
| `default` | Unset. A higher layer decides; with no setting, the tool's own approval declaration applies. |
| `on_request` | Use the tool's static or dynamic approval declaration, overriding lower layers. |
| `always` | Ask for every call. Dynamic approval predicates are not evaluated. |
| `never` | Skip approval for trusted runs. Dynamic approval predicates are not evaluated. |

The first non-`default` value wins, checked in this order: per-run config,
Agent, configured Runner.

An interrupted result keeps its `session_id` and `turn_id`. Replies go into
the inbox, and the same turn continues. See
[runtime-control.md](runtime-control.md#waiting-and-resuming).

## Guardrails and tracing

Input guardrails run when the turn's task is built, before the model is
called. A `block` or `require_approval` outcome returns a failed run without
calling the model. A `rewrite` outcome replaces the user input before the task
is compiled.

Output guardrails run on the final output. A rewrite replaces
`RunResult.final_output`. A block or approval outcome turns the run into a
failure that carries the guardrail message.

Output validation is a separate host extension, off by default. An enabled
validator can accept the final output or reject it with a code, and an
optional repair callback can supply one replacement without tools. See
[output-validation.md](output-validation.md).

Trace processors come from `RunConfig.tracing["processors"]`. The runtime
builds `run`, `agent`, and `tool` spans from the session log and delivers them
through their own consumer cursor. Delivery is at most once: the cursor
advances before processors are called, so a crash can lose spans but never
repeats them.

## Workspace boundary

File tools must go through `WorkspaceBackend`. The local filesystem, in-memory,
and S3-compatible backends should behave the same for read, write, list, and
grep wherever practical. Path traversal protection belongs at the workspace
boundary; `tests/test_workspace_backends.py` and `tests/test_tools.py` cover
it.

`read_file` and the check of existing artifacts read the built-in backends in
chunks. The SHA-256 digest, UTF-8 validation, line counts, and page output all
come from one pass over the bytes. No page is returned until the whole file
has been validated. This keeps memory use bounded, although every page request
still reads the full file. Custom backends without chunk reads are read with their own
`read_bytes`; their streaming memory bound depends on the backend. A line range
may include `start_char` (zero-based Unicode offset in its first line). Each
page reads up to 12,000 characters. A partial page reports the next line/local offset
and a digest-bound cursor. The model receives the next position explicitly.
Partial reads never authorize an overwrite of unseen content.

Model-facing artifact search, reads and file metadata use the same current-task
scope predicate; see [host-integration.md](host-integration.md#artifact-read-authorization)
for trusted identities and host ACL responsibilities.

## Invariants

- Model keys resolve exactly. A requested key is never aliased to a different
  provider model.
- Terminal states come from the no-tool policy, explicit tool outcomes, host
  hooks, cancellation, failures, and resource limits. The runtime never infers
  them from the wording of an answer.
- Public SDK code enters through `Agent`, `Runner`, `RunConfig`,
  `ModelSettings`, tools, sessions, typed `RunEvent` objects, or
  `InteractiveAgentClient` for stateful hosts.
- Long tool outputs keep structured data in metadata and model-facing text in
  content.
- Cancellation, streaming, hooks, memory compaction, and session stores must
  work together without changing public result shapes.
- New public behavior needs tests in the closest `tests/test_*.py` module.

## History-preserving compaction

`MemoryManager` keeps long conversations inside the model's context window in
two ways:

- Microcompaction moves large, older tool results into workspace artifacts and
  leaves a short marker in the transcript. The model can read the full result
  again with `read_file`, so microcompaction runs only when `read_file` is
  available. Recent assistant turns are protected, and tools declared with
  `result_retention="preserve"` are never archived.
- Summary compaction replaces the older part of the conversation with one
  summary message and keeps the most recent messages unchanged.
  `compact(..., force=True)` skips microcompaction and goes straight to a
  summary.

Summary compaction follows these rules:

- It keeps at least `keep_recent_messages` raw messages (default 10). It never
  splits an assistant message from its tool results; the cut moves earlier
  instead.
- Images in the summarized part are sent to the summary model as text
  placeholders such as `[image omitted from summary input: <content or
  image>]`. Images in the kept tail stay as they are.
- The new history is system messages, one user message named
  `memory_summary`, and the unchanged tail. Archived artifacts and file paths
  from earlier summaries carry forward, so a second summary does not lose the
  evidence of the first.
- History changes only when a summary is accepted. A summary that is empty,
  fails, does not reduce the token count, or does not fit the context is
  discarded, and the original history stays. A summary that relies on
  archived evidence is also discarded when `read_file` is not available.
- After a prompt-too-long error, emergency compaction keeps a smaller tail of
  `max(1, floor(keep_recent_messages * (1 - clamp(drop_ratio, 0, 0.95))))`
  messages.
- Session Memory is notified of a compaction only after the summary is
  accepted.

Inside the session kernel, compaction runs as logged operations, and each
accepted summary is a log record. See
[session-kernel.md](session-kernel.md#compaction-through-the-log) for how the
kernel plans, accepts, and replays compaction, and how hosts consume
compaction records.
