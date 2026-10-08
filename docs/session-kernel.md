# Internal session kernel

`vv_agent.session` is an internal, non-default synchronous kernel. Runner,
interactive sessions, CLI and App Server still use their existing entry points.
Nothing is exported from the top-level `vv_agent` package, and there is no
user-selectable kernel mode or experimental App Server adapter. The contract
lock and public wire remain at version 23; this module is not a verified public
persistence or wire format. Rust remains frozen and is outside this adoption.

## Modules and logical bytes

- `records.py` defines eleven record variants and eight inbox variants. The
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

Logical bytes use the existing `canonical_json.canonical_json_bytes` (RFC 8785)
with SHA-256 digests. Storage sequence, receiving time, and writer epoch are
excluded. Record IDs derive from semantic positions; input and commit IDs must
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
requires its own connection. `LeasePoll.controls` preserves complete inbox items,
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

Each SQL store keeps one disposable in-memory validated prefix, bound to session,
head sequence, lease epoch and the stored head digest. Cold recovery checks stored
byte digests, closed schemas, embedded digests and every reducer transition. A
warm refresh reads and folds only the new tail. Append validates new records
against a fork of that prefix inside the same transaction and retains head/inbox
CAS and both lease checks. Commit replay and overlap use indexed identity lookups.
Rollback/conflict, epoch changes, head rollback or a changed head digest discard
the cache; an outer host rollback is caught by the next database binding check.
New cached records are decoded from their already-validated write bytes, so
native opaque values (for example tuples) match cold JSON reads. Returned state
and records are detached from the cache. Deleting the cache affects
only cost, never authoritative state. No snapshot table or second ledger exists. `rebuild_schedule(session_id)`
locks the session and derives only `phase`, `next_drive_ms`, `active_turn_id`,
and `terminal_seq` again. Zero is the deterministic immediate-due sentinel;
absolute deadlines/not-before values come from records. Reads of current time,
lease deadlines, and receipt times use PostgreSQL `clock_timestamp()` after
acquiring the session row lock.

`list_runnable` uses the design's union of due/unconsumed-input execution work
and consumer lag. It does not claim work. Each ordered branch starts at the cursor session using existing primary and
ready-inbox indexes and stops at the page limit before the union is sorted.
The final merge sorts at most twice the page size. Call with `after=page[-1].cursor`
until exhausted, then restart from the beginning on the next scan. Projection
work is independent of the execution lease. User/approval-only waits have no
due time; an approval response, including denial, makes resolution runnable.
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
work. `Runtime` in `session/runtime.py` supplies the agent, RunConfig, resolved
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
stop with an explicit reason. Dispatch reevaluates current authorization and
retains frozen policy denials. The internal runtime currently exposes explicitly registered
FunctionTools, `ask_user`, and the policy-filtered built-in `read_file` when workspace
use is enabled; additional managed provider tools require a
session provider adapter. It does not run the old checkpoint controller or
its deferred lifecycle.

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

`CycleRunner._complete_llm` couples request construction to the old coordinator.
The kernel builds the same LlmRequest and uses the same client, replacing only
operation admission/persistence. It freezes `RetrySettings(max_attempts=1)` and
requires exactly one endpoint on VvLlmClient: transport retries or endpoint
fallback must not multiply the two logged attempts. Custom clients must also
perform exactly one provider attempt per `complete` call. No network/provider
credentials are needed by the scripted recovery tests.

`FunctionProvider.preflight` runs the actual ToolOrchestrator up to its existing
pre-dispatch callback, then stops without executing the handler. Its real submit
uses the same orchestrator and original handler. This avoids duplicating policy,
argument validation or callable approval predicates. Ordinary handler timeout
is unknown because the handler thread may continue. Checkpoint-bound deferred
outcomes do not count as trusted session acceptance. Executors marked
`policy_managed_by_handler` are rejected before preflight because they bypass
the orchestrator dispatch callback; they cannot execute through this wrapper.

## Repair, inputs and provider evidence

Model results and all dependent tool plans commit together. A retained response
with missing plans can also reconstruct those plans without another model call.
Known results are reused; planned work is admitted again; durable provider waits
keep their exact handle. A started operation without authentic result/acceptance
evidence becomes unknown only after the inbox watermark is rechecked. Models
have at most two attempts; tools retry at most once only when their frozen
metadata declares `supported` idempotency. Their key includes the session and
logical operation, never the attempt. Unknown model records explicitly carry
`duplicate_model_request_and_cost` and missing measurement information.

Providers implement `submit(plan, context)`, `query(handle)`, `cancel(handle)` and
`authenticate(input, plan)`. Outcomes are `Definitive`, `Accepted` or `Unknown`.
The adapter owns evidence authenticity and actual provider idempotency; a job ID
or model-produced value is insufficient. The store/reducer additionally checks
operation, attempt, request digest, provider binding and retained evidence.
Queries are bounded and performed at most once per handle per drive; query
failure preserves the parked operation. The supervisor's polling cadence bounds
subsequent queries after the immutable initial poll deadline.

Ready receipts/controls are applied before repair. Steering stays in the inbox
until the current model/tool batch, including any retry, closes; the next new
model request includes it. Follow-up input is recorded queued and opens a new
turn after the active one ends. Normal end commits use the inbox watermark; an
input arriving in that window forces recomputation. Replies to `ask_user` use a
`user` input with the active `target_turn_id` and a content object containing
`operation_id`, `interaction_id`, and `text`; they produce one tool result in the
same turn, not an additional user message. Approval answers bind the operation,
attempt, request ID, request digest and scope; authentication of the host user
belongs to the inbox-writing adapter.

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
`HostInteractionRequest` value type; no controller instance is constructed. It is pure; it never dispatches or acknowledges.
Consumers can filter the prefix by their durable cursor. Existing event fields
named `checkpoint_key` carry the experimental session identity; this does not
create or read an old checkpoint. Provider waits use a typed parked run-state
projection because the public deferred event requires the old checkpoint handle.
The formal App Server/product adapter belongs to the later default cut-over.

`supervisor.tick` pages through `list_runnable`, calling drive or the supplied
projection callback. It owns no claim mechanism. A host must schedule ticks and
make projection plus cursor acknowledgement atomic using the store transaction.
Wake callbacks are best-effort hints; the scan recovers empty-inbox work and
unprojected terminal records independently.

## Child sessions and completion delivery

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
background safe points, cancellation, and late-generation audit. Central contract adoption remains pending; Rust stays frozen.


## Compaction through the log

The host supplies `Runtime.memory_manager`; its value settings, including tool
retention declarations, are frozen in the turn definition. Drift stops resume
through the existing definition check. The kernel does not call the legacy
summary callback, checkpoint coordinator, or Session Memory extraction loop.
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
clamps it under contract 23. Each PTL recovery permits a new primary operation, with the existing three-retry
limit. Rejected or unavailable summaries retain history; equal source/tail
targets reuse their retained receipt. PTL exhaustion ends the turn with
`CompactionExhaustedError` without dropping history. PTL failures and summary
operations do not spend `AgentTask.max_cycles`. This
kernel does not reproduce the old CycleRunner's initial forced normal-tail
retry; its PTL path goes directly to the specified smaller emergency tail.

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
# Full gates require both real PostgreSQL and real Redis.
VV_AGENT_TEST_REDIS_URL=redis://127.0.0.1:6395/15 uv run pytest
```

The imported capacity regressions exercise cold 2,000-record recovery, 5,000-record
cancellation/fencing, cache disposal, rollback, mutable caller isolation and
external tails. Full F2 capability completion and the short-run overhead benchmark
(p95 additional overhead <=50 ms against the old default) remain prerequisites
for F3, as do the complete SDK/tool matrix and formal App Server adapter. This
internal promotion does not claim those later gates or contract-24 adoption.
