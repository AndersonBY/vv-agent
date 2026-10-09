# Session kernel

The session kernel is the only execution path for Runner, configured Runner,
RunHandle, interactive sessions, CLI, App Server and delegated children. Ordinary
runs own an SQLite `:memory:` store. Hosts opt into durable execution through
SessionStore, SQLiteStore or PostgresStore. There is no execution selector.

The lock selects contract 24.0.1. Current codecs are RunEvent v6, model-call v2,
task-token-usage v3, strict Message and App Server protocol v2. Public exports
match public_api v8. Rust remains frozen at contract 23.0.0 and is outside this
Python adoption. All execution uses the same kernel and retained log.

Runner.resume(session_id, turn_id) reads the retained identity. User and approval
replies continue the same turn; a fresh prompt after a terminal admits a fresh
turn. AgentSession history and state are read-only projections; initial messages
and JSON state belong to the creation-time seed. A closed session cannot reopen.

LeaseLost recovery uses bounded exponential backoff with jitter. Runtime defaults
are five losses, 10 ms initial delay and a 500 ms delay cap; each delay is sampled
between half and all of the capped delay. Runtime accepts injected sleep/jitter
functions for deterministic tests. Exhaustion raises LeaseRetryExhausted to
Runner and interactive callers and produces typed App Server error responses or
notifications. Recovery reads can also lose their lease and count toward the
same cap. A committed terminal completes its original turn without re-dispatch.

## Modules and logical bytes

- `records.py` defines 14 record variants and eight inbox variants. The
  version, envelope, payload, handles, and control discriminators are closed.
  Missing fields, unknown fields, non-integer/stale versions, duplicate JSON
  object keys, invalid identities, and incorrect embedded digests are rejected.
- `reducer.py` folds the ordered log and the consumed inbox items into operations,
  attempts, waits, active turn, controls, and the scheduling projection. It has
  no clock or persistence access. The store folds before writing anything.
- `store.py` defines the transaction/store protocols, envelopes, receipts,
  leases, pagination cursor, and errors.
- `sql.py` owns the shared SQL transaction rules: identities, commit manifests,
  append validation, lease/CAS checks, scheduling and consumer acknowledgements.
- `postgres.py` implements the reference store on a supplied psycopg v3
  connection. psycopg is lazy-imported and available through `[postgres]` and
  the development group, not the default dependencies.
- `sqlite.py` provides the single-host implementation for local persistent and in-memory sessions.
- `context.py` projects model messages, including accepted compaction replacements;
  `compaction.py` adapts MemoryManager to ordinary logged model operations.
- `delegation.py` assembles configured children, agent tools, background handles and handoffs;
  `bindings.py` separates JSON state from process-local shared-state objects.
- `lifecycle.py` logs after-cycle decisions; `memory.py` binds callbacks and
  session-memory file projections to committed boundaries.
- `projection.py`, `result.py`, `events.py` and `tracing.py` provide typed host
  projections; events and spans acknowledge through existing consumer cursors.
- `surfaces.py` assembles the SQLite owner and host handles; ordinary
  runs use `:memory:` and retained sessions use a SQLite file. Blocking children
  drive after releasing the parent lease; background children drive independently.
- `vv_agent/interactive.py` implements steering, follow-up, user/approval replies,
  archive and close through inbox items, with transcript/result projections.
- `app_server.py` projects threads, turns and timeline items from records. It
  creates no second thread ledger. App Server metadata uses the closed reserved
  `session_created.attributes.app_server` object (`agent_key`, `cwd`, `metadata`).

The App Server resumes active turns from retained records and the original
approval owner. Recovery waits for an existing lease to release or expire. Client
timeline replay uses stable item IDs and `afterItemId`; notification delivery
acknowledges the `app_server` consumer cursor only after transport projection.
Images retain both their wire input and model messages. Child waits expose safe
session/turn/interaction identities; responses target the child's inbox before
the parent adopts its terminal. Archive and close use stable control identities:
equal bytes replay, different bytes conflict, and closed turns never revive.
Host migration is documented in [migration-v8.md](migration-v8.md).

Logical bytes use the existing `canonical_json.canonical_json_bytes` (RFC 8785)
with SHA-256 digests. Producer construction validates and freezes each Record's
canonical bytes and digest once. Parsed JSON is retained privately for read-only
kernel use; already parsed canonical storage bytes are not parsed a second time.
`payload`, `to_dict()`, typed task copies and projected messages detach mutable
values at host boundaries, including tool hooks replayed from a retained model
receipt. Immutable prompt and scalar-only model settings can
be shared. Edit by constructing a replacement Record, never by mutating a returned
payload. Direct dataclass construction is checked by `encode()` before admission.
The closed record schemas compile to checks for their exact keyword vocabulary;
unsupported keywords fail at import, and invalid values use the original
jsonschema validator for diagnostics. Equivalence tests exercise all schema and
handle variants as well as the unchanged invalid-record tests.

The shared JCS encoder uses the stdlib C encoder for safe integers, valid Unicode
strings, arrays and objects with BMP keys and no floats. Other values use the full
encoder, with the same guarded optimization for eligible nested subtrees. Lone
surrogates and unsafe integers retain their rejection boundaries; astral keys
retain UTF-16 ordering. Vendored JCS golden vectors, literal escaping checks and
6,000 generated nested-value comparisons test fast/full byte equivalence.

Storage sequence, receiving time, and writer epoch are excluded. Record IDs derive from semantic positions; input and commit IDs must
be stable caller-owned source/transaction identities. No retry generates an ID.
`input_id` and `commit_id` are scoped by session. `session/create` is reserved
for creation. Repeated record IDs inside one append are rejected; overlap with
previously persisted records is checked byte-for-byte and not reinserted.

Payload fields containing host/provider data (`attributes`, `definition`,
`budget`, `request`, `tool`, `budget_admission`, `result`, `usage`, `observation`,
`evidence_manifest`, and input `content`) are explicit opaque JSON values or
objects. They cannot introduce kernel fields. External authenticity is the
provider/host adapter's responsibility; evidence references in the log are not
cryptographic proof. Operation/attempt, request digest, provider binding and
parked evidence references are checked by the reducer.

`op_planned.consumed_unknowns` records the exact unknown attempts frozen into a
successor model request. The reducer requires completion context `normal`,
`correction`, or `audit` according to that history. A trusted earlier model
result can supersede an undispatched retry; the reducer marks that retry
completed with `superseded=True` and no invented provider result. Once the retry
has started, the earlier result is audit-only. Cancelled/ended turns never
revive. The adapter must additionally reject stale external business generation
before choosing which input disposition to append.

## Joining host transactions

```python
from vv_agent.session.postgres import join_transaction

with connection.transaction():
    tx = join_transaction(connection)
    batch = tx.consumer_batch(session_id, "host")
    if batch is not None:
        apply_host_projection(connection, batch)
        tx.ack(batch)
```

`join_transaction` requires an already-open psycopg transaction. It does not
open or commit a connection. A `SessionTx` cannot escape its original database
transaction, and `ack` accepts only the actual batch issued by that transaction.
The cursor is locked until the host transaction ends. `limit` is a soft maximum:
consumer batches include the entire last commit. `from_seq` is the inclusive
first record; `through_seq` is the inclusive last record.

`PostgresStore(connection).atomic()` opens a transaction, or nests as a savepoint
when the host already has one. Mutation methods use savepoints so catching a
validation/conflict exception inside the host transaction leaves zero partial
writes. Returned receipts are provisional until the outer host transaction
commits. Keep host transactions short. Use one connection per thread; heartbeat
requires its own connection. SessionDriver derives its PostgreSQL heartbeat
conninfo from the original connection, including `ConnectionInfo.password` when
set, for both standalone and caller-owned stores. Credentials are not persisted
in session records. Heartbeat failures retain the original exception as the cause
of LeaseLost, including through LeaseRetryExhausted.
`LeasePoll.controls` preserves complete inbox items,
including their target turn and generation; it is not permission to apply a
stale cancel to the current turn. Django extraction and `in_atomic_block` validation
belong exclusively to the backend adapter.

`PostgresStore.standalone(conninfo)` owns an autocommit connection for local
tests. Call `install_schema()` once on an empty database. There are no migrations
or historical decoders.

## Commit receipts

The four-table design cannot retain an independently replayable commit with no
new record rows, nor the original head and ordered receipt of an all-overlap
commit. This implementation adds **`sk_commit`** keyed by session and commit ID.
It stores a canonical manifest of record IDs and consumed input IDs, a logical
batch digest, original record sequence mappings, original head, and receiving
time. Creation manifests additionally retain the consumer set. It contains no
operation state and no second copy of record bodies. The original four tables
and indexes retain their designed columns and constraints.

Replay compares the manifest and each referenced record's exact canonical
bytes before returning the original receipt with `replayed=True`. It does not
check or renew the old lease and does not grant execution authority. CAS,
lease, and expected inbox watermark are admission conditions, excluded from
logical commit identity. A subsequent new append still needs current epoch,
owner, unexpired lease, and head/inbox CAS. Empty commits are admitted through
those same checks and retained for replay.

## Scheduling and recovery

Each SQL store keeps at most two disposable in-memory validated prefixes, bound to session,
head sequence, lease epoch and the stored head digest. Cold recovery checks stored
byte digests, closed schemas, embedded digests and every reducer transition. A
warm refresh reads and folds only the new tail. Append validates new records
against a fork of that prefix inside the same transaction and retains head/inbox
CAS and both lease checks. Forks copy index containers and clone only operations
or turns that a transition will modify. This avoids allocating an entire old
operation graph for a one-record append. Public state snapshots still detach all
mutable operations, attempts, waits and child handles without copying the fold's
history and record/input identity indexes. Successful model dispatch endpoints are
folded into a scalar preference, including audit receipts, so planning does not
scan the full log to recover that preference. Commit replay and overlap use indexed identity lookups.
Rollback/conflict, head rollback or a changed head digest discard the cache;
an outer host rollback is caught by the next database binding check. A new lease
epoch rebinds the immutable prefix only after its stored head bytes and digest
still match; it never transfers execution authority. Both database lease checks,
head CAS and optional inbox CAS remain mandatory for a new append.
Every fetched row's bytes are checked against its stored digest. Matching session,
sequence and exact retained bytes reuse a validated Record; other bytes still
receive full schema, identity and embedded-digest validation. Native opaque values
(for example tuples) are frozen as the same JSON values a cold reader sees.
Returned execution state is detached, while immutable Records may be shared.
Deleting the cache affects only cost, never authoritative state. No snapshot table
or second ledger exists. `rebuild_schedule(session_id)` locks the session and derives only `phase`, `next_drive_ms`, `active_turn_id`,
and `terminal_seq` again. Zero is the deterministic immediate-due sentinel;
absolute deadlines/not-before values come from records. Reads of current time,
lease deadlines, and receipt times use PostgreSQL `clock_timestamp()` after
acquiring the session row lock.

The second slot prevents child result projection from evicting the executing
parent's prefix. A third session evicts the older slot; rollback or explicit
disposal clears both. Neither slot skips database head/digest binding checks.
Child projection reuses one validated state snapshot and creation record, and
Runtime initializes its tool registry only when planning or dispatching requires
it. Read-only projection still resolves host bindings and the child model/config.

`list_runnable` uses the design's union of due/unconsumed-input execution work
and consumer lag. It does not claim work. Each ordered branch starts at the cursor session using existing primary and
ready-inbox indexes and stops at the page limit before the union is sorted.
The final merge sorts at most twice the page size. Call with `after=page[-1].cursor`
until exhausted, then restart from the beginning on the next scan. Projection
work is independent of the execution lease. User waits and approval waits without
a deadline have no due time. An approval deadline or response, including denial,
makes resolution runnable.
After release, `drive` checks execution readiness in a fresh transaction using
the same SQL predicate as `list_runnable` (including lease availability). It wakes
only due execution work, never consumer lag or idle sessions. A delivery that
finds a held lease returns without waking; the holder's post-release check sees
inputs committed during its exit. Future-available inbox items rely on tick, whose
interval bounds discovery latency after they become due. In-process RunHandle
child scheduling uses explicit `after_drive`/`after_input` hook notifications.
See [cloud host integration](host-integration.md) for transport wiring and recovery.
A planned operation blocked on an unresolved dependency does not spin.
Queued user/follow-up inputs remain due after the active turn ends, even after
their inbox rows are consumed. A turn input cannot be admitted a second time.
Undispatched operations require explicit closure before terminal records; all
remaining dispatched attempts must be named in the unconfirmed list.

Fencing protects database state, not previously issued external requests.
The kernel propagates cancellation and fences every new log write, but a worker
paused after admission can still issue a delayed external request. Only provider
idempotency constrains that request's effects.

## Synchronous execution bindings

`kernel.drive(store, session_id, runtime=...)` acquires a store lease, folds the
log, applies ready inputs, repairs incomplete operations, and serially dispatches
work. Append receipts include the newly committed immutable records with their
store envelopes and the transaction's inbox watermark. After its transaction
commits, the driver applies that delta through the same Fold, without rereading
the committed range. Refresh reads only a bounded new tail and checks sequence
continuity; commit replay or external tails retain the authoritative store path. `Runtime` in `session/runtime.py` supplies the agent, RunConfig, resolved
model, existing LLM client, tool/provider bindings and a factory returning a
heartbeat store (a separate connection for files/PG, the same locked instance
for SQLite memory sessions). It is re-exported by the kernel
module for assembly. The default heartbeat interval is 250 ms with a 15-second
lease; intervals above one second are rejected. External calls use daemon
threads so the synchronous driver can close an unconfirmed cancellation without
waiting for an uncooperative handler. A cancelled thread is never described as
stopped unless the handler confirms cooperative cancellation.

The immutable definition contains the compiled AgentTask/PromptBundle, tool
schemas/capabilities, model binding and handler version. Resume does not rerun
instruction/context providers. Version/schema/capability/model-binding changes
stop with an explicit reason. Each Runtime constructs its registry once, or reuses
the registry supplied by its factory. Bounded definition/digest and schema caches
include task controls, model binding, memory settings, child names, registry
revision, exposure and detached capability declarations. Type-sensitive JSON
fingerprints prevent bool/integer cache-key collisions. Frozen Record tasks reuse a
bounded task fingerprint. Unchanged tool schemas share their validated Record's
private JSON graph and cached JCS bytes across definitions and requests. Reuse
requires exact schema bytes or the same retained schema objects; mutable hook
inputs still detach, and source/digest validation remains mandatory. Persisted
bytes keep the complete schemas, so cold readers use the same wire shape.
The definition's schema/capability/memory/model-binding
JCS fragments are cached separately from the task and rebuilt whenever those
bindings or task tool controls change. Every composed definition digest equals
full JCS encoding; live binding checks still run on every driver step. Failed canonical validation
preserves the last valid definition cache. Dynamic `is_enabled`
predicates are reevaluated when compiling each new turn. Dispatch reevaluates
current authorization and retains frozen policy denials. The internal runtime exposes the policy-filtered built-in planner surface and registered
executors, honoring FunctionTool `is_enabled` and registry exposure. The capability
matrix in `session-kernel-capability-matrix.md` records the current producer evidence.

| Need | Existing implementation called |
| --- | --- |
| Prompt compilation/freeze | `runtime/compiler.py:AgentCompiler.compile`, `types.py:AgentTask.to_dict/from_dict` |
| Request/tool schemas | `llm/base.py:LlmRequest`, `runtime/tool_planner.py:plan_tool_schemas` |
| Model transport | `llm/vv_llm_client.py:VvLlmClient.complete`; tests use `llm/scripted.py:ScriptedLLM.complete` |
| Budget admission | `budget.py:BudgetEvaluator`, `cycle_start`, `model_call_complete`, `preflight_tools` |
| Tool policy, arguments, approval | `tools/orchestrator.py:ToolOrchestrator.run_one` and its existing dispatch callback |
| Handlers and context | `tools/function.py:FunctionTool.invoke`, `tools/base.py:ToolContext`, `runtime/context.py:ExecutionContext`, `runtime/cancellation.py:CancellationToken` |
| Compaction | `memory/manager.py:MemoryManager.plan_microcompaction`, `apply_microcompaction`, `plan_summary`, `accept_summary`, `compaction_evidence` |
| Event types | `events.py:RunEvent` and existing typed lifecycle subclasses |

The kernel builds LlmRequest and uses the configured client with logged
operation admission and persistence. It freezes `RetrySettings(max_attempts=1)` and binds one VvLlmClient endpoint per
logged attempt, following the frozen preferred/randomized order. Custom clients must also perform exactly one
provider attempt per `complete` call. No network/provider
credentials are needed by the scripted recovery tests.

`FunctionProvider.preflight` runs the actual ToolOrchestrator up to its existing
pre-dispatch callback, then stops without executing the handler. Its real submit
uses the same orchestrator and original handler. This avoids duplicating policy,
argument validation or callable approval predicates. Ordinary handler timeout
is unknown because the handler thread may continue. Executors marked
`policy_managed_by_handler` are rejected before preflight because they bypass
the orchestrator dispatch callback; they cannot execute through this wrapper.

Before-tool hooks run serially immediately before each tool's preflight.
`op_prepared` retains the patched call/digest, capability, provider/idempotency
binding, short-circuit result and JSON state. It is replayed after approval or
Runtime reconstruction without calling the hook again. Definitive result records
retain after-tool hooks and stop behavior, including pre-dispatch denials and
short circuits. Native FINISH closes already planned pending tools with the
shared skipped-result producer.

Approval parks bind the exact prepared arguments. The internal approval bridge
uses ApprovalProvider and ApprovalBroker only as transports and pushes their
decisions through the same inbox. Answers allow approve, deny, allow_session and
timeout with optional typed reason/metadata fields. Session grants are folded
from applied answers; a new Broker cannot lose them. Deadlines are absolute
store times retained in the parked record, including provider decision time.
Expired allow/allow_session answers are rejected; Broker session flags alone cannot
authorize effects. Duplicate answers are noop only when all decision bytes agree;
conflicting decisions are rejected.

## Repair, inputs and provider evidence

Model results and all dependent tool plans commit together. A retained response
with missing plans can also reconstruct those plans without another model call.
Known results are reused; planned work is admitted again; durable provider waits
keep their exact handle. A started operation without authentic result/acceptance
evidence becomes unknown only after the inbox watermark is rechecked. Models
have at most max(2, frozen endpoint count) logged attempts; an uncertain output
repair is never retried. Tools retry at most once only when their frozen
metadata declares `supported` idempotency. Their key includes the session and
logical operation, never the attempt. Unknown model records explicitly carry
`duplicate_model_request_and_cost` and missing measurement information.

Providers implement `submit(plan, context)`, `query(handle)`, `cancel(handle)` and
`authenticate(input, plan)`. Outcomes are `Definitive`, `Accepted` or `Unknown`.
The adapter owns evidence authenticity and actual provider idempotency; a job ID
or model-produced value is insufficient. The store/reducer additionally checks
operation, attempt, request digest, provider binding and retained evidence.
Queries are bounded and performed at most once per handle per drive; query
failure preserves the parked operation. `SessionStore.next_drive_delay_ms`
reads the sole SQL `next_drive_ms` schedule under the held lease using database
time. The first query occurs no earlier than `poll_at_ms`; later queries honor idle deferral,
even on duplicate/reordered wakes or unrelated inbox admission. Input/usage-only
commits preserve that schedule when the active turn and reducer due times are
unchanged. Authenticated receipts and controls are applied immediately.
Local handles return when the turn parks on a provider wait. When `poll_at_ms`
is in the future, the in-process run returns parked without querying or leaving
a thread waiting. A later host tick/supervisor drive, wake or `Runner.resume`
at or after the due time performs the query; an authenticated `provider_result`
can continue the turn before that time. Runner, interactive, App Server and
local children use that same handle path.

Ready receipts/controls are applied before repair. Steering stays in the inbox
until the current model/tool batch, including any retry, closes; the next new
model request includes it. Follow-up input is recorded queued and opens a new
turn after the active one ends. Normal end commits use the inbox watermark; an
input arriving in that window forces recomputation. Replies to `ask_user` use a
`user` input with the active `target_turn_id` and a content object containing
`operation_id`, `interaction_id`, and `text`; they produce one tool result in the
same turn, not an additional user message. Approval answers bind the operation,
attempt, request ID, request digest and scope; authentication of the host user
belongs to the inbox-writing adapter. Identical reply replays are noop; different
response bytes for the same interaction are rejected even after completion.

A no-tool `wait_user` result creates `turn_parked` with its source model receipt,
interaction identity and prompt, leaving no due time. It has no tool operation or
invented tool message. A targeted `user` reply contains `interaction_id` and
`text`, clears that wait and appends the user message within the same turn.
Cancel/suspend controls remain effective while parked.

Bash/check/stop reuse the existing BackgroundSessionManager. A retained bash
receipt carries the process-manager session ID; rebuilding a drive/Runtime keeps
the original turn owner, so subsequent management tools reattach through the
same owner-scoped manager. Manager/OS-worker restart is outside this guarantee;
missing handles are never adopted by PID alone. Unknown/stopping receipts are
not confirmed stop receipts.

Late results follow the reducer's original normal/correction/audit rules.
An unknown already frozen into a successor request keeps its single original
tool message and receives a separate correction. Earlier model results cannot
win after a later attempt has started. Cancelled/ended turns accept authentic
late receipts as audit evidence without revival.

Cancellation of an already dispatched provider wait follows this rule: if the turn is cancelled and stopping cannot be confirmed, the
parked attempt can become unknown while retaining its acceptance evidence.
Ordinary parked recovery still cannot manufacture unknown. Undispatched plans
close explicitly without claiming an external effect. Durable abort controls,
not transient token timing, determine an aborted terminal state.

Budget limits stay frozen per turn. Dispatch counts and model usage derive from
the log, not successful re-admission during replay. `usage_observed` contains
per-lease cumulative active milliseconds and host cost observations. Parked time
between drives is excluded; a lost started interval is marked unavailable.
Strict budgets stop on required missing measurements; retries cannot reset
counts or substitute a stale host meter value for a missing current observation.

## Projection and scanning

`projection.project_records` maps a consistent log prefix to existing RunEvent
types with stable session-scoped IDs, original database timestamps and
`metadata.session_seq`. Host-interaction event digests reuse the existing
`HostInteractionRequest` value type. Projection never dispatches or acknowledges.
Consumers can filter the prefix by their durable cursor. Kernel v6 events use
`session_id`. Provider waits use a
typed parked run-state projection.
The App Server adapter projects the same log through protocol v2.

`supervisor.tick` pages through `list_runnable`, calling drive or the supplied
projection callback. It owns no claim mechanism. A host must schedule ticks and
make projection plus cursor acknowledgement atomic using the store transaction.
Wake callbacks are best-effort hints; the scan recovers empty-inbox work and
unprojected terminal records independently.

## Child sessions and completion delivery

The internal Runtime binds configured `create_sub_task`, `Agent.as_tool`,
`BackgroundAgentTask` and handoff tools to the same child admission path.
Adapters only assemble definitions and project results; they never execute a
recursive Runner or a child driver under the parent lease. The host independently
schedules each child and its completion consumer. Public SDK entrypoints use
these adapters; local handles schedule child drives after releasing the parent lease.

`session_created.attributes.child_admission` is closed: mode, selector, frozen
definition/digest, budget, handler version, SubAgentConfig, discovery filter,
handoff count/maximum and handoff metadata. The definition freezes child policy,
model binding, prompt, workspace path and initial JSON state at admission.
Recovery re-supplies the agent/tool/provider/workspace handlers, validates their
version and definition binding, and uses the admitted task. Workspace backends
remain host configuration; S3 clients and credentials never enter the records.
Configured children start with their own JSON state; agent tools and handoff
children inherit a copy of the parent's JSON state.

A batch uses optional closed `handle.siblings` member identities. Every member
has the same parent delivery target and its own initial turn and cursor.
Blocking completion requires an authenticated terminal input for every member.
A disposable fold index supplies completion lookups; it is reconstructed from
`input_applied`, with no second ledger. Result assembly always projects the
terminal turn named in the authenticated handle, even after the child continues.

`Runtime.child_tasks(store, parent_id)` supplies owner-scoped status and handles.
The built-in status adapter uses record projections plus the existing status
formatter, with its own SQL connection on the provider thread. Messages,
continuations, user replies and cancellation use stable inbox IDs; retries
replay identical bytes and conflicting content fails. `handle.poll`, `snapshot`,
`wait` and `cancel` survive Runtime reconstruction. The start operation is the
background tool's atomic child admission; its initial snapshot is durably
running, independent of the child's scheduling race. Public BackgroundAgentTask
handles project that admitted child and its retained operations.

A blocking child waiting for a user keeps the parent operation parked until a
terminal result, rather than returning Runner's intermediate waiting outcome.
Ordinary tools returning WAIT_USER retain their result and park the turn using
`turn_parked`; a reply resumes it without repeating the handler.
Handoff is a terminal child continuation: count and maximum come from admitted
records, and the source never resumes model execution after transfer. The source
log remains the durable owner, while result agent/model/state/output project the
terminal target. Target validation runs once on execution; the source's transient
Runner transfer marker is not a user-facing final output and is not validated.
See [migration-v8.md](migration-v8.md) for current host API migration.

### Shared-state host bindings

Durable shared_state remains JSON-only. A host supplies arbitrary Python objects
explicitly through `Runtime.host_bindings`; only their sorted required names in
`task.metadata.vv_session.host_binding_names` are frozen. Tools and runtime hooks
receive the original references; records, model-visible metadata and result
projections retain only JSON state. Bound references cannot shadow durable keys,
be replaced or be deleted. Reconstructing without a required binding raises
`MissingHostBinding` before execution; the host must re-supply the object.
Nothing pickles or serializes its object representation. These bindings are
process-local and do not promise rollback of mutations to the host object.


`Runtime.children` maps a registered tool name to a synchronous admission
callback returning `children.ChildSession`. The existing tool policy, argument
validation, approval and budget checks run before admission. The callback runs
inside `store.atomic()` and may join that same host transaction; it must not
start a second execution loop or perform external effects. It supplies the child
SessionSpec, content, generation, consumers and blocking/background mode.
The kernel commits child creation, its initial user inbox, parent dispatch and
child park together. Admission failures roll back the entire transaction.
A host store adapter must include its ORM admission writes in that transaction.
PostgreSQL remains the correctness reference; SQLite shares these transaction
operations and runs the applicable child fault matrix.

The closed child handle retains child session/turn/generation, background mode,
and parent session/turn/generation/operation/attempt. It remains in the fold
after completion so replay and late receipts can still be checked. The same
handle is stored in the child's creation attributes. This is a replacement of
the unused prototype child shape, with no compatibility reader.

`children.child_delivery(store, tx, child_session_id)` consumes the reserved
`child_delivery` cursor. Each matching `turn_ended` produces one stable
`child_result` input, containing the original terminal sequence and logical
record digest, and advances the cursor on the caller's transaction. The host
routes that consumer from `list_runnable` alongside its other consumers.
Delivery does not depend on a wake or the child's execution lease. Neither
`drive` nor delivery invokes the child recursively.

The parent reads the child log to verify its creation relationship, exact handle,
terminal kind/turn/sequence/digest and result bytes. It validates the parent
turn/generation and operation/attempt, then consumes the input and completes the
blocking operation in one commit. Repeated input IDs compare canonical bytes;
alias input IDs for an identical completion are no-ops. Forged or mismatched
completion inputs are rejected and retained in the audit log. Same input ID
with changed bytes raises `Conflict`. Cancelled/ended/replaced turns cannot be
revived; authentic late blocking results have audit context, and stale-generation
inputs retain an audit rejection. A blocking child cannot be completed without
its applied terminal input, nor can a successful terminal record leave a child
wait unresolved (`HasLiveDescendants`).

Cancellation pushes stable targeted cancel controls to every live child owned
by the turn, including background children, in the transaction closing the
parent operations. A child not yet started applies cancellation immediately
after admitting its initial turn, before any model dispatch. Each child's own
drive propagates cancellation to its descendants. Until terminal evidence
arrives, a cancelled parent's dispatched child wait is unknown and appears in
its unconfirmed operation list; delivery later records the actual audit result.
A handler-version failure retains the authentic child wait as unconfirmed, ends
the parent with explicit failure, and also requests durable child cancellation.

For `background=True`, admission also completes the parent tool immediately
with a child reference. Completion is a notification input, not a replacement
of that tool result. It becomes a user notification only after the current
model/tool batch closes; a newly applied notification invalidates a previously
frozen completion candidate and reaches the next model request. Notifications
after the owning turn ends are audit-only and never start another turn or
invalidate a newer turn's completion candidate.

`tests/session/test_children.py` exercises these rules against disposable PostgreSQL and SQLite
databases, including process-kill barriers around push/ack, concurrent
delivery and input replay, identity/evidence rejection, atomic admission,
background safe points, cancellation, and late-generation audit. Central contract
24.0.1 adoption is verified at Python 53bdf32; the support matrix records the
exact revision and CI run. Rust stays frozen at contract 23.0.0.


## Compaction through the log

The host supplies `Runtime.memory_manager`; its value settings, including tool
retention declarations, are frozen in the turn definition. Drift stops resume
through the existing definition check. The kernel does not call the legacy
summary callback or Session Memory extraction loop.
Summary inference uses the runtime's bound model and client, with tools disabled,
a text-only prompt, one transport attempt, and the existing budget evaluator.

`MemoryManager.plan_summary` delegates block splitting and prompt rendering to
the existing implementation. Its optional `drop_ratio` also owns the emergency
tail formula used by `emergency_compact`. `accept_summary(plan, text,
notify=False)` uses the existing extraction, normalization, effective-content,
file-path/evidence merging, context-budget and actual-reduction checks.
`compaction_evidence` exposes the same validated artifact/cursor collector for
log manifests. Existing compact/emergency/microcompact signatures and default
Runner behavior are unchanged.

Before a primary request, the driver performs at most one archive-backed prune
pass for that context. It commits mode `micro` only after MemoryManager has
verified persistence. Content-addressed artifact reuse survives a crash after
persistence but before the log commit. If necessary, a summary is then planned
as an ordinary `op_kind=model`, `purpose=compaction` operation. Its stable identity
contains the turn, source context digest, mode and tail target. The
request retains its source log version, rendered prompt, tail target and model
settings. A saved `op_completed` is revalidated on restart; inference is not
repeated. Invalid output, an oversized summary input, or exhausted ambiguous
summary attempts preserve history and allow the normal turn to continue.
Cancellation, lost ownership and exhausted run budgets still stop execution.

Acceptance appends `context_compacted`: source digest, ordered prefix/tail
identities (source position plus message digest), mode, summary operation
reference, complete replacement messages and evidence manifest. The selected
attempt and result remain in the log under that operation. The reducer validates
source and identities, reruns summary acceptance against the retained receipt,
and verifies the exact replacement and manifest. Micro records bind eligible
original tool bodies to recovery markers and artifact hashes. Artifact bytes
are verified by MemoryManager before commit, never by the pure reducer.

The model context projection applies accepted replacements in log order, then
adds subsequent input/results. Raw records remain immutable, including removed
history, previous summaries and receipts. Typed tool artifacts and recovery
cursors reach this projection through `ToolExecutionResult.to_tool_message`.
Successive summaries keep prior evidence; manifests are not truncated. Summary
results do not become assistant answers or tool plans. They retain their own
`MEMORY_COMPACTION` model telemetry and usage accounting in the internal RunEvent projection.

A provider prompt-too-long error is a definitive logged failure rather than an
ambiguous dispatch. Emergency summarization uses MemoryManager's smaller-tail
plan with `drop_ratio = 0.2 * consecutive_prompt_too_long_failures`; the manager
clamps it under the current compaction contract. Each PTL recovery permits a new primary operation, with the existing three-retry
limit. Rejected or unavailable summaries retain history; equal source/tail
targets reuse their retained receipt. PTL exhaustion ends the turn with
`CompactionExhaustedError` without dropping history. PTL failures and summary
operations do not spend `AgentTask.max_cycles`. PTL recovery goes directly to the specified smaller emergency tail.

Retries of an existing operation reuse its exact request and context version;
the reducer rejects a changed version, purpose, request digest or binding.
Steering waits behind the frozen operation and reaches the next request.
Acceptance of a saved summary precedes application of newly arrived steering,
while cancellation and definition/version checks remain authoritative.

Session Memory notifications belong to a host consumer of accepted
`context_compacted` records. The driver and pure fold never call
`SessionMemory.on_compaction` before the append, or during replay. Register that
consumer when creating the session, use `consumer_batch`/`ack`, and derive its
`current_tokens` baseline from the replacement summary plus tail. A host with
persistent notification effects must deduplicate by the compaction record ID
and commit those effects with its cursor; file-backed Session Memory needs an
idempotent host delivery adapter. The raw model receipt alone is not a
compaction notification.

## SQLite file and memory stores

`SQLiteStore.standalone(path)` owns and closes one connection. Use `":memory:"`
for process-local sessions; the database disappears when its owner closes it.
The heartbeat factory must borrow the **same store instance** for memory sessions
(for example `lambda: nullcontext(store)`), without closing it. File stores use
independent connections to the same path. SQLite is single-host only.

SQLite uses STRICT tables, foreign keys, WAL (files), synchronous FULL, a
5000 ms busy timeout and strict `user_version=1`. The record CHECK accepts
`context_compacted`, including summary, emergency and microcompaction records.
There are no historical readers or migrations in this internal module.

The instance serializes its connection with an RLock. Transactions use
`BEGIN IMMEDIATE`; lease time is read after acquiring the write lock. Nested
mutations use savepoints. Host writes and consumer acknowledgements share
`store.atomic()` and its connection; independently opened psycopg host
transactions belong to PostgreSQL only.

## Validation and database isolation

Pure record/reducer tests request no database fixture and do not import psycopg.
Store, recovery, compaction, child and concurrency semantics are parametrized
across PostgreSQL, SQLite files and SQLite `:memory:`. Process-kill tests use both
durable stores: memory sessions cannot survive process death. SQLite file-header
checks run on file stores; external psycopg transaction joins/rollbacks run on PG.

PostgreSQL tests first try `VV_AGENT_TEST_POSTGRES_DSN` when set, then fall back
to local `dbname=postgres` through the unix socket. Each test creates and finally drops
only its own `vvsk_test_<uuid>` database. The role needs CREATEDB. Per-test
databases preserve independent worker/heartbeat connections and process restart
isolation without search_path propagation or shared schema collisions. Connection
options from the supplied DSN are retained when replacing its database name.
The TCP password-authentication regression also creates and drops a temporary
login role, requiring CREATEROLE. It connects through `127.0.0.1`, verifies that
an incorrect password is rejected, and waits for a real heartbeat renewal for
both standalone and caller-owned connections. It skips explicitly only when
local pg_hba rules reject the connection or do not require password authentication.

If psycopg is unavailable, or neither connection is available, PG cases skip
with an explicit reason; central CI supplies a reachable DSN and must run them.
Pure and SQLite tests remain runnable.
No other databases are changed. Teardown waits for test-owned connections to
close and does not terminate server backends.

```bash
uv run pytest tests/session/test_records_reducer.py -q
uv run pytest tests/session -q
# CI: supply VV_AGENT_TEST_POSTGRES_DSN for a disposable test server.
python3 scripts/contract_snapshot.py check
uv run ruff format --check .
uv run ruff check .
uv run ty check
# Full gates require real PostgreSQL.
uv run pytest
```

Capacity regressions exercise cold recovery, long-log cancellation/fencing,
cache disposal, rollback, mutable caller isolation and external tails. Current
absolute measurements are in [session-kernel-baseline.md](session-kernel-baseline.md).

The M6 script uses the same bounded 1 KiB receipt history as the capacity tests,
real PostgreSQL and disposable databases. Its checks require cold recovery at
5k/20k to finish within 5/20 seconds, steady append median <=50 ms and full
catalog pagination <=1 second. The one-sample capacity invocation is:

```bash
uv run python scripts/session_kernel_benchmark.py --sizes 5000 20000 --samples 1 --assert-capacity
```


## Memory, decisions and budgets

`boundary_recorded` retains closed, stage-specific callback, decision, output and
budget data. Before-memory replacements include the exact source context digest
and JSON shared state. After-cycle snapshots use adopted model/tool receipts,
including the parked ask_user WAIT_RESPONSE receipt. Decisions are committed
before another dispatch and reused after recovery. Steer, persisted tool denials,
invalid decisions and non-success stops use the existing lifecycle helpers.
A callback interrupted before its boundary commit can run again; a recorded
callback or decision is never executed again.

MemoryProvider callbacks surround a logged started/completed compaction lifecycle.
Micro and summary work share that lifecycle, with retained archive statistics.
Summary requests, their acceptance and replacements still use MemoryManager and
the ordinary model driver. First prompt-too-long recovery forces compaction with
the configured tail; later retries shrink it. Primary requests retain the logical
cycle number across those retries. Identical rejected summary operations reuse
retained receipts rather than issuing another identical model request.

Session-memory extraction is a tools-free logged model operation with purpose
`session_memory`. Structured state is logged before file projection. Atomic file
replacement and reconstruction from records make the file disposable; projection
runs before compilation/reload of the next turn. Extraction parsing, entry merging
and pruning reuse SessionMemory.

Tool budgets reserve the whole ordered model batch in the same transaction as the
adopted model receipt and tool plans. Rejected admission has no tool effects.
Counts refer to admission names, including reserved calls later skipped by a
finish/wait; per-dispatch wall/host checks do not increment them again. Retry and
recovery reuse those reservations. Total/uncached tokens include logged internal
model calls. Host metrics retain unavailable classifications across reconstruction;
lost active intervals are explicitly unavailable, and strict policy stops instead
of inventing wall time. No wall time is inferred from process downtime.

## Model and result adapters

Endpoint order freezes the existing preference/randomization policy in the model
request; each logged attempt selects exactly one endpoint from that order. Client
and transport retries are set to one, and fallback is represented by another logged
attempt. The last durable model success supplies later-turn preference. Request
and endpoint drift are rejected before dispatch.

Typed output checks and one tools-free repair use output_validation helpers.
Repair is a logged `output_repair` operation; reported usage participates in budget
and result ledgers, and an uncertain repair is never retried automatically.
Candidate/partial output and final decisions survive recovery. Completed results
use their terminal prefix, so later turns cannot change an older result. Per-cycle
compaction flags, waits, errors, budget exhaustion and typed JSON output are
reconstructed from records. Kernel accounting exposes `output_repair` directly,
with model-call v2 and task-token-usage v3; TokenUsage remains v1. Kernel events
use v6 across all entrypoints.

## Events, streams and tracing

Typed RunEvents project agent/cycle/diagnostic/budget/memory/child lifecycle from
records with stable event IDs and `metadata.session_seq`. Child admission/completion
events use parent records and carry child session/turn identities. Existing wait,
approval, skipped-tool and cancellation differences remain explicit. All public entrypoints use this projection.

`SessionRunEventStore` implements replay and validates append against the existing
projection without another event ledger. `batch(tx)` bridges consumer cursors to
projected events; a host can write its projection and acknowledge in the same SQL
transaction. `consume` with an external sink is at least once if a crash occurs
between sink delivery and acknowledgement; stable IDs support deduplication.

Live assistant/reasoning/tool deltas use the existing stream-payload adapter and
remain volatile. Sink loss cannot affect correctness. Definitive content,
reasoning and tool calls rebuild from durable receipts; delta delivery is never a
recovery prerequisite.

Tracing projects stable run/agent/tool spans and uses a separate registered traces
consumer. `deliver_spans` requires a top-level transaction and commits its cursor
before invoking processors, preventing recovery duplicates. Telemetry is therefore
at most once and may be lost after acknowledgement; processor failures remain
isolated. Span output is detached before delivery so processors cannot mutate
retained result data. Host assembly supplies processors explicitly, and all
public entrypoints use these kernel projections.

## Reserved metadata and seed

Kernel task metadata reserves one closed `vv_session` object containing optional
`host_binding_names`, `max_handoffs`, `handoff_targets`, `input_messages`,
`memory_initial_state` and `input_blocked`. User-supplied `vv_session` rejects
at compile time. Request metadata reserves its own closed `vv_session` object
with optional `endpoint_order`, `endpoint_id`, `shared_state` and `cycle_index`.
Other metadata stays opaque JSON. Completion state is `op_completed.shared_state`;
usage contains measurements. Hashes are exactly 64 lowercase hex characters.
Compaction IDs omit the summary segment when summary_operation_id is null.

Creation can carry closed `attributes.seed = {messages, shared_state}`, both
required. History projects immediately, before the first turn compilation.
Messages and JSON state are detached projections after creation. A reset uses a
new durable identity; see [the seed migration](migration-v8.md#creation-time-seed).

## Transport ownership

SQLite and PostgreSQL are the only kernel stores. Celery/Redis remain the .ai
host's runtime framework and transport; vv-agent has no Redis store, Celery/Redis
code, integration module or extra. The host owns tasks, queues, connections and
runtime factories. vv-agent owns transport-independent dispatch semantics and
their SQLite/PostgreSQL tests. Host transport calls `drive`, scans via `tick` and
supplies `Runtime.wake`; durable execution authority remains in the store.
[Cloud host integration](host-integration.md) documents that wiring with Celery
as a host-only example; backend B1 implements the site's tasks.

## Fixture generation

`scripts/session_kernel_fixtures.py --output /tmp/vv-agent-fixtures` generates
forty-five files from real store/kernel/public-surface/App Server producers with
scripted providers and fixed semantic identities/clocks. It never edits the
vendored snapshot. One generation compares all forty-five bytes with v24.0.1.
Independent Node RFC8785 bytes, digests, record IDs, closed schemas, complete
kind/stage/handle/optional-field coverage and source-prefix projections are
revalidated. Deterministic curation preserves each behavioral coverage key.
Seven unchanged fixtures are checked directly. Invalid versions/fields remain
negative inputs to current strict readers. The full corpus is limited to 3 MB
and each output to 512 KB. Producer evidence supplements durable-store
process-kill, concurrency, rollback and authentication tests.
