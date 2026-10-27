# Session kernel

The session kernel is the only execution path in vv-agent. Runner, configured
Runner, RunHandle, interactive sessions, the CLI, the App Server and delegated
children all run through it. There is no execution selector.

An ordinary run uses an SQLite `:memory:` store that disappears when its owner
closes it. For durable execution a host supplies a `SessionStore`: `SQLiteStore`
on one machine, or `PostgresStore` for workers that share a database. Host setup
(runtime factory, wake, drive, tick, billing) is in
[cloud host integration](host-integration.md). This page is the reference for
how the kernel behaves, for contributors and for hosts that need exact rules.

The package implements contract 31.0.0, pinned in `contract.lock.json`. Current
versions: public API v14, inbox schema 3, record schema 1, RunEvent v6,
model-call v2, task-token-usage v3, strict Message, App Server protocol v4 and
SQLite `user_version` 3. Rust (vv-agent-rs) stays frozen at contract 23.0.0.

## Terms

- **Record**: one immutable log entry, such as `turn_started` or `op_completed`.
  There are 14 record kinds.
- **Inbox item** (input): a message written to a session from outside, such as
  a user reply, steering, a provider result or a control. There are nine kinds.
  The kernel applies one by appending `input_applied` with a disposition such
  as `applied`, `queued` or `rejected`.
- **Turn**: one unit of work started by a user input and closed by `turn_ended`.
  `turn_started` holds a generation number; a targeted input may carry it, and
  a mismatch is rejected.
- **Operation** and **attempt**: an operation is one logical model call, tool
  call or child admission. Each try at running it is a numbered attempt.
- **Lease**: a time-limited claim that lets one worker append execution
  records. Writes are fenced by the lease epoch.
- **Consumer cursor**: a named, durable log position for a projection, such as
  `host`, `app_server`, `traces` or `child_delivery`.
- **Receipt**: the store's answer to a committed append: the new records with
  their sequence numbers and the inbox watermark.

## Sessions and turns

`Runner.resume(session_id, turn_id)` reads the retained turn identity. User and
approval replies continue the same turn; a fresh prompt after the turn ends
admits a new turn. `AgentSession` history and state are read-only projections.
Initial messages and JSON state go in the
[creation-time seed](#reserved-metadata-and-seed). A closed session cannot
reopen.

When a worker loses its lease (`LeaseLost`), the runtime retries with bounded
exponential backoff. `Runtime` defaults allow 5 losses (`lease_retry_attempts`),
starting at 10 ms (`lease_retry_base_seconds`) and capped at 500 ms
(`lease_retry_cap_seconds`). Each delay is sampled between half and all of the
capped value; tests can inject `lease_retry_jitter` and `lease_retry_sleep`.
Recovery reads count toward the same cap. When it runs out, Runner and
interactive callers get `LeaseRetryExhausted` and the App Server sends a typed
error. If a terminal record was already committed, the turn completes without
dispatching again.

## The log is the execution ledger

The log is the only record of what a session did. The reducer derives
operations, attempts, waits, the active turn, controls and the schedule from it.
There is no snapshot table and no second ledger.

The log is the execution ledger for one runtime version. After a record or
inbox schema change, the new runtime rejects old rows, including those of
finished sessions, so hosts keep their own product copy of what they need. See
[session data and upgrades](upgrading.md#session-data-and-upgrades).

| Module | Role |
| --- | --- |
| `records.py` | Record and inbox kinds with closed schemas. |
| `reducer.py` | Folds the log and consumed inputs into execution state. No clock, no persistence. |
| `store.py`, `sql.py` | Store protocols, receipts, leases and errors; the shared SQL rules for appends, CAS, leases, scheduling and acknowledgements. |
| `postgres.py`, `sqlite.py` | The PostgreSQL reference store (psycopg v3, imported lazily; install the `[postgres]` extra or the dev group) and the single-host SQLite store. |
| `context.py`, `compaction.py` | Model-message projection and the MemoryManager adapter. |
| `delegation.py`, `children.py` | Child admission, agent tools, background handles, handoffs and delivery. |
| `bindings.py`, `lifecycle.py`, `memory.py` | Host object bindings, after-cycle decisions, memory callbacks. |
| `projection.py`, `result.py`, `events.py`, `tracing.py` | Typed host projections. |
| `surfaces.py`, `app_server.py` | Public entrypoint handles; App Server threads and timelines. |

### Record validation and identity

Validation is strict. Missing or unknown fields, stale versions, duplicate JSON
keys, invalid identities and wrong embedded digests are rejected, and the
version, envelope, payload, handle and control discriminators are closed. The
store folds new records through the reducer before writing anything.

Logical bytes are RFC 8785 (JCS) from `canonical_json.canonical_json_bytes`,
hashed with SHA-256 once when a record is built. Values returned to hosts
(`payload`, `to_dict()`, task copies, projected messages, replayed hook inputs)
are detached copies. To change a record, build a replacement `Record`.

Storage sequence, receive time and writer epoch are not part of a record's
identity. Record IDs derive from semantic positions. `input_id` and `commit_id`
are stable, session-scoped identities owned by the caller; a retry never makes a
new one. `session/create` is reserved for creation. A repeated record ID inside
one append is rejected; a record that overlaps a persisted one is compared byte
for byte and not inserted again.

Host and provider data is opaque JSON: `attributes`, `definition`, `budget`,
`request`, `tool`, `budget_admission`, `result`, `usage`, `observation`,
`evidence_manifest` and input `content`. It cannot introduce kernel fields. The
adapter that produced it owns its authenticity; an evidence reference is not
cryptographic proof. The reducer still checks operation, attempt, request
digest, provider binding and parked evidence.

### Late results and superseded retries

`op_planned.consumed_unknowns` lists unknown attempts frozen into a later model
request. Each completion must be labeled `normal`, `correction` or `audit` to
match that history:

- A trusted earlier result can supersede a retry that has not been dispatched.
  The retry is marked completed with `superseded=True` and no invented result.
- Once the retry has started, the earlier result is audit-only.
- An unknown already frozen into a later request keeps its single tool message
  and gets a separate correction.
- Cancelled or ended turns accept authentic late receipts as audit only and
  never revive.

The host adapter must also reject inputs from a stale business generation
before it chooses a disposition.

## Joining host transactions

The code pattern is in [host integration](host-integration.md#store-and-transactions).
`join_transaction(connection)` needs a psycopg transaction that is already open
and never commits it. A `SessionTx` cannot outlive that transaction, and `ack`
accepts only a batch the same transaction issued. The cursor stays locked until
the host transaction ends. In `consumer_batch`, `limit` is a soft maximum: a
batch always includes the whole of its last commit. `from_seq` and `through_seq`
are both inclusive.

`PostgresStore(connection).atomic()` opens a transaction, or a savepoint if one
is open. Mutations use savepoints, so a host that catches a validation or
conflict error is left with no partial writes. Receipts stay provisional until
the outer transaction commits. Use one connection per thread.

The heartbeat needs its own connection. `SessionDriver` derives its conninfo
from the original connection, including `ConnectionInfo.password` when set.
Credentials never enter records. A heartbeat failure is kept as the cause of
`LeaseLost` and of `LeaseRetryExhausted`.

`LeasePoll.controls` returns full inbox items with target turn and generation.
A cancel aimed at an older turn does not apply to the current one.

`PostgresStore.standalone(conninfo)` owns an autocommit connection for local
tests. Call `install_schema()` once on an empty database.

## Commit receipts

The kernel must replay a commit that added no rows, and return the original
receipt for a commit whose records all overlapped. For that the store keeps
`sk_commit`, keyed by session and commit ID: a canonical manifest of record and
consumed input IDs, a batch digest, the original sequence mapping and head, the
receive time and, for creation, the consumer set. It holds no record bodies.

Replay compares the manifest and each referenced record's bytes, then returns
the original receipt with `replayed=True`. It neither checks nor renews the old
lease and grants no execution authority. CAS, lease and expected inbox watermark
are admission conditions, not part of commit identity, so a new append still
needs the current epoch, owner, a live lease and head and inbox CAS. Empty
commits pass the same checks and are kept for replay.

## Scheduling and recovery

Each SQL store caches up to two validated log prefixes, so projecting a child
result does not evict the executing parent. A cache entry is bound to session,
head sequence, lease epoch and head digest. A cold read checks every stored
digest, schema and reducer transition; a warm refresh folds only the new tail.
Appends still run both lease checks and CAS. Rollback, conflict or a changed
head digest drops the cache. The cache changes cost, never results.

`read_state` captures head sequence, inbox watermark and epoch with one statement, then loads and folds records up to
that head without a session row lock. Consumed inputs are bounded by both the
captured head and inbox watermark. Snapshot copying also runs without the lock.
Schedule rebuilds prepare history first and recheck the captured position under
the write lock before updating. PostgreSQL appends prepare the retained prefix
and fork its reducer before locking the session; new-record validation, CAS and
fencing remain atomic. SQLite write transactions remain serialized; prepare
large reads before entering a host-owned write transaction.

`rebuild_schedule(session_id)` locks the session and recomputes `phase`,
`next_drive_ms`, `active_turn_id` and `terminal_seq` from the log. Zero means
due now; absolute deadlines come from records. PostgreSQL reads time with
`clock_timestamp()` after taking the session row lock. `drive_retry_at_ms` and
`project_retry_at_ms` are retry gates for execution and projection
(nonnegative, default zero). They are scheduling state, not execution facts.

`list_runnable` returns due execution work, unconsumed input and consumer lag.
It claims nothing. Each branch stops at the page limit, so the merge sorts at
most twice the page size. Page with `after=page[-1].cursor` until empty, then
restart on the next scan. Projection work does not need the execution lease.

- A user or approval wait without a deadline has no due time.
- An approval deadline or response, including a denial, makes the session
  runnable.
- A planned operation blocked on an unresolved dependency does not spin.
- Queued user and follow-up inputs stay due after the active turn ends, even
  once their inbox rows are consumed. A turn input cannot be admitted twice.

After releasing its lease, `drive` rechecks readiness in a fresh transaction
with the same predicate and wakes only for due execution work.
`supervisor.tick` pages through `list_runnable` and either dispatches or drives
inline; per-session failures are raised together after the scan. Tick claims
nothing. The host side of wake, drive and tick is in
[host integration](host-integration.md#workers-drive).

Undispatched operations need an explicit closure before a terminal record, and
every remaining dispatched attempt must be listed as unconfirmed.

Fencing protects database state, not requests already sent. A worker paused
after admission can still send a delayed external request after losing its
lease. Only provider idempotency limits that request's effects.

## Synchronous execution bindings

`kernel.drive(store, session_id, runtime=...)` takes a lease, folds the log,
applies ready inputs, repairs incomplete operations and dispatches work
serially. It folds its own receipts directly instead of rereading the records it
just wrote.

`Runtime` (`session/runtime.py`) supplies the agent, RunConfig, resolved model,
LLM client, tool and provider bindings, and a `heartbeat_store` factory: a
separate connection for files and PostgreSQL, the same locked instance for
memory sessions. Defaults are a 15-second lease (`ttl_ms=15000`) and 250 ms
heartbeats; the heartbeat must be at most one second and under half the TTL.

For a callable runtime, `drive` uses the explicit
`FACTORY_LEASE_TTL_MS = 15000` and `FACTORY_HEARTBEAT_SECONDS = 0.25` preparation
policy. Before calling the factory, it opens `store.heartbeat_store()` and
completes an initial renewal. Renewal continues during the factory; this TTL is
not a build timeout. Every factory exit stops and joins that worker, and passes
the latest Lease to deferral, release or the runtime TTL handoff. A renewal error
or fenced epoch discards the factory result before any driver work. The runtime
heartbeat takes over with the runtime's own TTL and connection factory. Factories
prepare bindings only; they must not dispatch execution effects. Hosts must
bound SQL and callback I/O because joining cannot cancel arbitrary host code.

External calls run on daemon threads, so the driver can close a cancellation
without waiting for a handler that ignores it. A thread counts as stopped only
if its handler confirms cooperative cancellation.

### The frozen definition

Each turn freezes a definition: the compiled AgentTask and PromptBundle, tool
schemas and capabilities, model binding and handler version. Resume does not
rerun instruction or context providers. A changed version, schema, capability or
model binding stops the session with an explicit reason, checked on every driver
step. Dynamic `is_enabled` predicates are evaluated when each new turn compiles.
Dispatch rechecks current authorization and keeps frozen policy denials.
[The capability matrix](session-kernel-capability-matrix.md) lists the producers
for each capability.

The kernel calls the existing pieces rather than reimplementing them:
`AgentCompiler`, `LlmRequest` and `plan_tool_schemas`, `BudgetEvaluator`,
`ToolOrchestrator.run_one`, `FunctionTool.invoke` with `ToolContext` and
`ExecutionContext`, and `MemoryManager`. Tests use `llm/scripted.py:ScriptedLLM`
in place of `VvLlmClient`.

### Model calls and model selection

The kernel freezes `RetrySettings(max_attempts=1)`: one logged attempt is one
provider call, and endpoint fallback is a new logged attempt. The request
freezes the endpoint order from the preference and randomization policy, and
each attempt binds one endpoint from it. The last durable success sets the
preferred endpoint for later turns. Request or endpoint drift is rejected before
dispatch. A custom
client must also make exactly one provider attempt per `complete` call.

To change the model for later cycles, a host pushes a session-scoped
`model_selection` input; editing `Runtime` or task fields does not do it. Once
the current model and tool batch (including retries) closes, the driver resolves
the selection through `Runtime.model_resolver` and applies it. A resolver failure
makes the input `rejected` with `model_selection_unresolvable`, and the previous
choice stays. An accepted selection and its binding live in `input_applied`;
later primary requests retain model, settings, route and binding. A `null`
selection restores the turn's frozen model. Children keep their own selection,
and existing plans, retries and late receipts keep their identity. See
[session model selection](session-model-selection.md).

### Tools, hooks and approval

`FunctionProvider.preflight` runs the real ToolOrchestrator up to its
pre-dispatch callback and stops before the handler; submit runs the same
orchestrator with the original handler. A handler timeout is unknown, since the
thread may still run. Executors marked `policy_managed_by_handler` bypass that
callback, so they are rejected before preflight.

Before-tool hooks run one at a time, just before each preflight. `op_prepared`
keeps the patched call, capability, provider and idempotency binding, any
short-circuit result and the JSON state. After an approval or rebuild the kernel
replays it instead of calling the hook again. Result records keep after-tool
hook output and stop behavior. A native FINISH closes already planned tools with
the shared skipped-result producer.

An approval park binds the exact prepared arguments. ApprovalProvider and
ApprovalBroker are transports only; decisions go through the inbox. Answers are
`approve`, `deny`, `allow_session` or `timeout`, with optional reason and
metadata. Session grants are folded from applied answers, so a new Broker
cannot lose them. Deadlines are absolute store times in the parked record. An
`allow` or `allow_session` after the deadline is rejected, and Broker flags
alone never authorize effects. A duplicate answer is a no-op only if every
decision byte matches; a conflicting one is rejected.

## Authenticated provider usage

A `provider_result` input carries `operation_id`, `attempt`, `request_digest`,
`provider_binding` (nullable), `result`, `usage` and `evidence`. `usage` is a
required object shaped like `op_completed.usage`, `{}` when nothing was
measured. Missing, null or non-object usage and extra fields are rejected.
Authentication must cover usage, result and attempt identity.

The kernel keeps usage bytes unchanged. Late `normal`, `correction` and `audit`
completions are all billable: an audit completion is not adopted, but the call
still happened. A new input for a completed attempt is a no-op only if result
and usage bytes both match; any difference is a conflict. Hosts read usage from
`project_result` or `project_records`, never from `result.raw.usage`; see
[billing](host-integration.md#billing).

## Repair, inputs and provider evidence

### Repairing after a crash

A model result and its dependent tool plans commit together. If a retained
response lacks its plans, the driver rebuilds them without calling the model.
Known results are reused and durable provider waits keep their handle. A
started operation without authentic result or acceptance evidence becomes
unknown only after the inbox watermark is checked again.

- A model operation has at most `max(2, frozen endpoint count)` logged attempts.
  Unknown model records carry `duplicate_model_request_and_cost` and mark the
  measurement missing.
- An uncertain output repair is never retried.
- A tool is retried at most once, only when its frozen metadata declares
  `supported` idempotency. Its key covers session and logical operation, never
  the attempt.

### Providers

A provider implements `submit(plan, context)`, `query(handle)`, `cancel(handle)`
and `authenticate(input, plan)`, returning `Definitive`, `Accepted` or
`Unknown`. The adapter owns evidence authenticity and real idempotency; a job ID
or a model-produced value is not enough.

Each handle is queried at most once per drive, and a failed query leaves the
operation parked. `SessionStore.next_drive_delay_ms` reads `next_drive_ms` under
the held lease in database time, so the first query happens no earlier than
`poll_at_ms` and later ones honor idle deferral, whatever wakes arrive. Commits
that only add inputs or usage keep the schedule when the active turn and reducer
due times are unchanged. Authenticated receipts and
controls apply at once. How in-process runs return on a provider wait is in
[host integration](host-integration.md#provider-waits).

If a turn is cancelled during a dispatched provider wait and the stop cannot be
confirmed, the attempt can become unknown while keeping its acceptance evidence.
Ordinary parked recovery never turns a wait into unknown. A durable abort
control, not token timing, decides whether a terminal is aborted.

### Inputs during a turn

Ready receipts and controls apply before repair.

- **Steering** waits until the current model and tool batch, including retries,
  closes, and reaches the next model request.
- **Follow-up** is recorded as queued and opens a new turn after this one ends.
- A normal end commits against the inbox watermark; an input arriving in that
  window forces a recompute.

A reply to `ask_user` is a `user` input with the active `target_turn_id` and
content `{operation_id, interaction_id, text}`. It produces one tool result in
the same turn, not an extra user message. A no-tool `wait_user` result appends
`turn_parked` with no due time and no tool operation; its reply carries
`interaction_id` and `text` and appends the user message in the same turn. A
tool returning WAIT_USER parks with `turn_parked`, and the reply resumes it
without rerunning the handler. Cancel and suspend still work while parked.

Approval answers bind operation, attempt, request ID, request digest and scope;
the adapter that writes the inbox authenticates the user. Replaying an
identical reply is a no-op. Different bytes for the same interaction are
rejected, even after completion.

`bash`, `check` and `stop` use BackgroundSessionManager. A bash receipt keeps
the process-manager session ID and a rebuilt Runtime keeps the turn owner, so
management tools reattach. A manager or worker restart is outside this
guarantee: a missing handle is never adopted by PID alone, and an unknown or
stopping receipt is not a confirmed stop.

## Budgets

Budget limits are frozen per turn. Counts and usage come from the log, not from
re-admission during replay.

- Tool budgets reserve the whole model batch in the transaction that adopts the
  model receipt and tool plans. A rejected admission has no tool effects.
- Counts use admission names, including reserved calls later skipped by a
  finish or wait. Retries reuse the reservations.
- Token totals include logged internal model calls (compaction, session memory,
  output repair).
- `usage_observed` records per-lease active milliseconds and host cost. Parked
  time is excluded, an interval lost to a crash is marked unavailable, and no
  wall time is inferred from downtime.
- Strict budgets stop on a missing required measurement. Retries cannot reset
  counts or substitute a stale host meter value.

## Projection, events and tracing

`projection.project_records` maps a consistent log prefix to RunEvents with
stable session-scoped IDs, original database timestamps and
`metadata.session_seq`. Child events come from parent records and carry the
child's session and turn. Provider waits project as a typed parked run state.
Projection never dispatches or acknowledges.

`SessionRunEventStore` replays events and validates appends against the
projection. `batch(tx)` turns a consumer batch into events, so a host can write
its projection and acknowledge in one transaction. `consume(sink)` is at least
once; sinks deduplicate by stable event ID. Live assistant, reasoning and tool
deltas are volatile: final content and tool calls rebuild from receipts.

Tracing uses a separate `traces` consumer. `deliver_spans` needs a top-level
transaction and commits its cursor before calling processors, so spans are at
most once and can be lost after acknowledgement. Span output is detached, so
processors cannot change results.

The App Server keeps no thread ledger of its own. Its metadata is the closed
`session_created.attributes.app_server` object (`agent_key`, `cwd`, `metadata`).
It resumes active turns from records after any existing lease is released or
expires. Any authorized connection can control or approve the retained turn. Child waits expose safe session,
turn and interaction identities, and responses go to the child's inbox. Timeline
replay uses bounded record cursors with event offsets. The `app_server` consumer
copies events into the local handle queue and acknowledges that batch; a separate
pump performs host callbacks and transport projection outside the store transaction.
Live notifications are volatile. Each node separately tails lifecycle records with
per-session backoff, releases positions on unsubscribe and recovers from read errors.
Use thread/events/list for durable delivery after a disconnect. Embedded schedulers
call AppServer.wake(thread_id) after commit; the built-in host also makes one
automatic attempt at a blocked holder's lease expiry. Archive and close use
stable control identities: equal bytes replay, different bytes conflict, closed
turns never revive.

## Child sessions and completion delivery

Configured `create_sub_task`, `Agent.as_tool`, `BackgroundAgentTask` and handoff
tools share one child admission path. Adapters assemble definitions and project
results; they never run a nested Runner or a child driver under the parent's
lease. The host schedules each child and its completion consumer independently.

### Admission

`session_created.attributes.child_admission` is closed: mode, selector, frozen
definition and digest, budget, handler version, SubAgentConfig, discovery
filter and handoff count, maximum and metadata. The definition freezes the
child's policy, model binding, prompt, workspace path and initial JSON state.
On recovery the host supplies handlers again and the kernel checks their
version and binding. Workspace credentials never enter records. Configured
children start with their own JSON state; agent tools and handoff children get
a copy of the parent's.

`Runtime.children` maps a tool name to a synchronous callback returning one
`children.ChildSession` or a nonempty sequence. Policy, argument, approval and
budget checks run first. The callback runs inside `store.atomic()`, may join
that host transaction, and must not start an execution loop or cause external
effects. Child creation, initial input, parent dispatch and parent park commit
together; any failure, including `InvalidChildBatch`, rolls all of it back with
the callback's host writes. See
[custom child admission](host-integration.md#custom-child-admission).

The closed child handle holds the child's session, turn and generation, the
background flag, and the parent's session, turn, generation, operation and
attempt. It stays in the fold after completion so late receipts can be checked.
A batch adds `handle.siblings`. A blocking batch completes only when every
member has an authenticated terminal input, and results always project the
terminal turn named in the handle. `Runtime.child_tasks(store, parent_id)`
returns owner-scoped status and handles; `handle.poll`, `snapshot`, `wait` and
`cancel` survive a Runtime rebuild.

### Delivery

`children.child_delivery(store, tx, child_session_id)` reads the
`child_delivery` cursor. For each matching `turn_ended` it pushes one stable
`child_result` input to the parent, with the terminal sequence and record
digest, and advances the cursor in the caller's transaction. Delivery needs
neither a wake nor the child's lease.

The parent checks the child log (creation relationship, handle, terminal and
result bytes) and its own turn, generation, operation and attempt, then
consumes the input and completes the operation in one commit.

- A repeated input ID with different bytes raises `Conflict`. A different input
  ID for an identical completion is a no-op.
- Forged or mismatched inputs are rejected and kept for audit.
- Cancelled, ended or replaced turns never revive; late results and stale generations are
  recorded as audit.
- A blocking operation cannot complete without its terminal input, and a
  successful terminal cannot leave a child wait open (`HasLiveDescendants`).

A blocking child waiting for a user keeps the parent operation parked until the
child reaches a terminal result.

### Cancellation and background children

When a turn's operations close on cancel, the same transaction pushes stable
cancel controls to every live child, background ones included. A child that has
not started applies it before any model call, and each child passes it on to its
descendants. Until terminal evidence arrives, the parent's child wait is listed
as unconfirmed. A handler-version failure keeps that wait unconfirmed, fails the
parent and requests durable cancellation of the child.

With `background=True`, admission completes the parent tool at once with a
child reference. The completion arrives later as a notification input, after the
current model and tool batch closes, and reaches the next request. A
notification after the owning turn ended is audit-only.

### Handoff

Handoff is a terminal child continuation. Count and maximum come from admitted
records, and the source never resumes model execution after the transfer. The
source log stays the durable owner; the result's agent, model, state and output
come from the target, and output validation runs once, on the target.

### Shared-state host bindings

Durable `shared_state` is JSON only. A host passes Python objects through
`Runtime.host_bindings`; only their sorted names are frozen, in
`task.metadata.vv_session.host_binding_names`. Tools and hooks get the original
objects, records keep only JSON, and a bound name cannot shadow a durable key. A
missing binding on rebuild raises `MissingHostBinding` before execution. Nothing
is pickled, and mutations to host objects are not rolled back.

## Compaction through the log

Tools may request compaction and child bindings may require a fresh summary.
See [host summary bindings](host-integration.md#a-compact-tool-and-fresh-child-summaries).
Both use ordinary compaction model operations. Child preparation uses
`metadata.mode=child_summary` and never appends `context_compacted`.


The host supplies `Runtime.memory_manager`; its settings are frozen in the turn
definition. The kernel never calls the legacy summary callback or the Session
Memory extraction loop. It reuses `MemoryManager.plan_summary`,
`accept_summary(plan, text, notify=False)` and `compaction_evidence`.

Before a primary request the driver makes at most one archive-backed prune pass
and commits mode `micro` only after MemoryManager confirms the archive. A
summary is an ordinary model operation (`purpose=compaction`): bound model,
tools disabled, text-only prompt, one transport attempt, normal budget checks.
Its identity covers turn, source context digest, mode and tail target. On
restart a saved `op_completed` is validated again rather than rerun, and an
identical source and tail target reuses its receipt. Invalid output or exhausted
ambiguous attempts keep history and the turn continues; cancellation, lost
ownership and exhausted budgets still stop it.

Acceptance appends `context_compacted`: source digest, ordered prefix and tail
identities, mode, summary operation reference, replacement messages and
evidence manifest. The reducer reruns acceptance against the retained receipt.
MemoryManager verifies archive bytes before commit; the reducer never reads
them. The context projection applies replacements in log order. Raw records,
earlier summaries and evidence are never removed or truncated, and summaries
never become assistant answers.

A prompt-too-long (PTL) error is a definitive logged failure. Recovery uses the
emergency plan with `drop_ratio = 0.2 * consecutive_prompt_too_long_failures`
(clamped by the manager) and allows a new primary operation, up to three
retries with the same logical cycle number. When retries run out, the turn ends
with `CompactionExhaustedError` without dropping history. PTL failures and
summaries do not count against `AgentTask.max_cycles`.

A retry reuses the exact request and context version. Steering waits behind the
frozen operation, and a saved summary is accepted before newly arrived steering
applies.

Session Memory notifications come from a host consumer of `context_compacted`
records; the driver never calls `SessionMemory.on_compaction`, and a raw model
receipt is not a notification. Register the consumer at creation and take
`current_tokens` from the replacement summary plus tail. A host with lasting
notification effects deduplicates by compaction record ID and commits them with
its cursor; file-backed Session Memory needs an idempotent delivery adapter.

## Lifecycle callbacks, session memory and output

`boundary_recorded` stores stage-specific callback, decision, output and budget
data. A decision is committed before the next dispatch and reused after
recovery. A callback interrupted before its commit can run again; a recorded
one never does.

Session-memory extraction is a tools-free model operation with purpose
`session_memory`. Its state is logged before the file is written atomically, so
the file can be rebuilt from records.

Typed output checks and one tools-free repair use the `output_validation`
helpers. The repair is a logged `output_repair` operation whose usage counts
toward budgets. A result is projected from its terminal prefix, so later turns
cannot change it.

## Reserved metadata and seed

Task metadata reserves a closed `vv_session` object with optional
`host_binding_names`, `max_handoffs`, `handoff_targets`, `input_messages`,
`memory_initial_state` and `input_blocked`; a user-supplied one is rejected at
compile time. Request metadata reserves its own closed `vv_session` with
optional `endpoint_order`, `endpoint_id`, `shared_state`, `cycle_index` and
`model_binding`. Other metadata is opaque JSON. Hashes are 64 lowercase hex
characters.

Creation can carry a closed `attributes.seed = {messages, shared_state}`, both
required. Seeded history is visible before the first turn compiles. To reset,
create a session with a new identity; see
[creation-time seed](upgrading.md#creation-time-seed).

## Stores

`PostgresStore` is the correctness reference. A host that manages its own schema
copies the current DDL from `postgres.py` exactly. Custom stores must implement
the [scheduling methods](host-integration.md#if-you-implement-your-own-store).

`SQLiteStore.standalone(path)` owns and closes one connection. Use `":memory:"`
for a process-local session; its heartbeat factory must lend the same store
without closing it, for example `lambda: nullcontext(store)`. File stores use
independent connections to one path. SQLite is single-host only.

SQLite uses STRICT tables, foreign keys, WAL for files, `synchronous=FULL`, a
5000 ms busy timeout and `user_version=3`; any other version is rejected.
Transactions use `BEGIN IMMEDIATE`, lease time is read after taking the write
lock, and nested mutations use savepoints. Host writes and acknowledgements
share `store.atomic()`.

## Transport ownership

SQLite and PostgreSQL are the only kernel stores. vv-agent ships no Redis store
and no Celery or Redis code. A host such as a Celery deployment owns tasks,
queues, connections and runtime factories; it calls `drive`, scans with `tick`
and supplies `Runtime.wake`. Execution authority stays in the store.

## Tests and database isolation

Pure record and reducer tests use no database and do not import psycopg. Store,
recovery, compaction, child and concurrency tests run on PostgreSQL, SQLite
files and SQLite `:memory:`; process-kill tests use the two durable stores.

PostgreSQL tests use `VV_AGENT_TEST_POSTGRES_DSN` if set, otherwise local
`dbname=postgres` over the unix socket. Each test creates and drops its own
`vvsk_test_<uuid>` database, so the role needs CREATEDB. The TCP password test
also creates a temporary login role (CREATEROLE) and connects through
`127.0.0.1`. Without psycopg or a server, PostgreSQL cases skip with a reason;
CI supplies a DSN so they run there.

```bash
uv run pytest tests/session -q
python3 scripts/contract_snapshot.py check
uv run ruff format --check . && uv run ruff check . && uv run ty check
uv run pytest   # full gates need real PostgreSQL
```

`scripts/session_kernel_benchmark.py --assert-capacity` runs on real PostgreSQL
with a bounded 1 KiB receipt history. It requires cold recovery of 5,000 and
20,000 records within 5 and 20 seconds, a steady append median of at most
50 ms, and full catalog pagination within 1 second. Measurements are in
[session-kernel-baseline.md](session-kernel-baseline.md).

## Fixture generation

`scripts/session_kernel_fixtures.py --output /tmp/vv-agent-fixtures` generates 45
fixture files from the real store, kernel, public surfaces and App Server, with
scripted providers and fixed clocks. It never edits the vendored snapshot; it
compares all 45 outputs with it byte for byte, and checks digests, record IDs
and schemas with an independent Node RFC 8785 encoder. The corpus is limited to
3 MB, and 512 KB per file.

## App Server fork seeds

Completed-boundary forks use the existing creation seed and attributes extension
maps; see [App Server](app-server.md#forking-a-thread). app_server_fork stores
params, lineage, effective_model and last_compaction. It is creation evidence,
not a copied execution plan. The inherited model remains the frozen default
across turns and selection clears. Seed shared_state is empty; host process
objects and operational state are never copied. Store versions stay unchanged.
