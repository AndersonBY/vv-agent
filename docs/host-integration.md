# Cloud host integration

This guide is for a host that runs vv-agent sessions on many workers that share
one PostgreSQL database. vv-agent decides what a session does next and records
every step in a durable log. The host owns everything around that: the queue
and workers, database connections, authentication, how a runtime is built, and
the product tables it projects into.

vv-agent ships no Celery or Redis code and no integration extra. The Celery
example below lives in the host. A single-process host can use SQLite instead
of PostgreSQL.

The core relocation to `vv_agent.kernel` does not require host SQL or API
changes. Contract 39 changes a failed closed-session diagnostic to exactly
`invalid session schema`; stored formats and canonical bytes remain unchanged.
Do not parse complete `jsonschema.ValidationError` text or schema dumps.

A few terms used throughout:

- A session's inbox holds inputs waiting to be applied: user messages, replies,
  control commands, provider results. Each input has a stable `input_id`.
  Pushing the same ID with the same bytes again is a no-op; different bytes
  raise a conflict.
- A lease is a time-limited claim on a session. Only the lease holder writes
  to the session's log, and every write is fenced by the lease epoch.
- A consumer cursor is a named position in the session log. A projection
  consumer reads the records after its cursor and then moves it forward (acks).
- A turn is one run of the agent in answer to an input. Each turn has a
  generation number, and replies must name the turn and generation they answer.
- An operation is one model call, tool call or child session inside a turn. A
  receipt is the committed record of an operation's outcome.

## Store and transactions

Use the host's own PostgreSQL database. Bind `PostgresStore(connection)` to a
psycopg v3 connection that the host owns. To write host rows and session rows
atomically, open the transaction yourself and join it:

```python
from vv_agent.session.postgres import join_transaction

with connection.transaction():
    tx = join_transaction(connection)
    batch = tx.consumer_batch(sid, "host")
    if batch is not None:
        project_records_and_bill(connection, batch)
        tx.ack(batch)
```

`join_transaction` requires a transaction that is already open. It never opens
a connection and never commits. A Django host extracts the psycopg connection
and checks that the transaction is open in its own adapter. See
[joining host transactions](session-kernel.md#joining-host-transactions).

Rules for host writes:

- Register consumers when you create the session.
- Commit host projections, billing rows and the consumer `ack` in the same
  transaction. The log is the source of truth and the cursor marks the last
  projected record, so there is no command outbox or reconciliation job.
- Admission writes (`tx.create`, `tx.push`, `tx.append`) can share a
  transaction with host rows in the same way.
- External sinks that cannot join the transaction deduplicate by the stable
  record or event ID.
- Keep transactions short. Never wrap a whole `drive` or a provider call in a
  host transaction.

### Schema

`PostgresStore.install_schema()` runs the current DDL, which is the `DDL`
string owned by `src/vv_agent/kernel/_postgres.py` and exported by
`src/vv_agent/session/postgres.py`. A host that manages its own DDL must copy
that literal from the installed release. SQLite files carry
`PRAGMA user_version` 4 for the S1 store, and the store refuses to open any other version.

The PostgreSQL DDL is the unisolated base schema. For credential-level tenant
isolation, call `install_tenant_rls()` after schema installation and use
`tenant_standalone()` when starting tenant work. Worker and shared-gateway
logins are separate explicit modes; neither satisfies the tenant credential
boundary. See [PostgreSQL tenant RLS](postgres-rls.md).

The store checks the complete S1 catalog fingerprint when it opens or installs
a non-empty database. SQLite uses an equivalent fingerprint and does not
reconfigure the journal on heartbeat connections. Both adapters carry the
worker feature level and input-admission configuration onto independent
heartbeat connections, so a worker can renew a higher-level session without
sharing its foreground connection.

For App Server integrations, return the authenticated identity from
`AppServerHost.resolve_principal(connection_id, method, params)`. The hook is
a trusted host boundary; request parameters remain closed and cannot contain
an actor or principal field. The runtime persists the identity and passes its
immutable form to tools and model adapters outside the prompt.

vv-agent has no schema migrator. Finish old sessions with the release that
created them, then provision the current schema. Any schema change the host
makes itself is the host's responsibility. See
[upgrading](upgrading.md#session-data-and-upgrades) for version history and what
an upgrade means for stored sessions.

## Runtime factory

Host-bound runs use `SessionDriver(..., runtime_factory=...)`. Its callable has
the signature `runtime_factory(store, agent, config, task, session_id)` and must
return a `Runtime`. `SessionDriver` calls it after a worker claim, passing the
exact tenant-bound drive store. The factory must rebuild everything the drive
uses: the exact model and client, tools and providers, workspace, child
factories, host bindings, callbacks and handler version. A callback that keeps a
different store or raw connection is not bound to this drive.

A resumed turn runs on the definition it retained when it started. The factory
must reproduce the retained logical model, tools, handler behavior and host
authorization. If a compatible worker rebuilds a different definition, the
session stops explicitly instead of continuing on different settings.

The worker's trusted support set is separate from persisted requirements.
Declare the worker feature level, every accepted handler version and every
statically reachable model/tool capability in `SessionDriver`. Include support
needed by configured sub-agents. Persisted bootstrap and input requirements are
checks against that declaration; they never add capabilities to a worker's
support set.

There are two supported worker integrations, and both may build callbacks that
write host rows in the drive transaction:

- A direct host queue calls
  `drive(store, session_id, runtime=lambda sid: build_runtime(store))` on the
  worker thread. The closure binds the supplied `store` to the runtime, and the
  rebuilt callbacks use that same store and transaction.
- A managed local worker uses
  `SessionDriver(..., runtime_factory=runtime_factory)`. The factory receives
  the claimed tenant-bound drive store and rebuilds callbacks against it.

In either form, the failure is a callback that captured another store or worker
connection. `SessionDriver` is not required for a host queue to bind callbacks
to its supplied drive store.

`Runtime.heartbeat_store` is a factory that opens an independent connection to
the same session database. Built-in stores implement the needed factories. A
custom store, including a PostgreSQL subclass with host connection management,
must provide the drive connection opener used by direct `drive` or
`SessionDriver(open_store=...)`, plus an equivalent heartbeat-store factory.
Resolve thread-bound connection setup on the worker thread that opens the
store. Never use the foreground drive connection for renewal.

The default lease TTL is 15 seconds (`ttl_ms=15000`) and renewal interval is
250 ms (`heartbeat_seconds=0.25`). The interval must be at most 1 second and
less than half the TTL. Configure the model client for one external attempt per
logical call because the kernel owns retry accounting. Set `wake` to the host
enqueue function or local-worker wake. Bind live stream sinks separately from
durable projection consumers (see
[streaming](#streaming-and-cancellation)).

## Submitting input and waking

1. In a host transaction, write the business row and push an inbox input with a
   stable `input_id` owned by the source (for example a message ID or command
   ID).
2. Commit that transaction.
3. Call `wake(sid)`.

A wake carries only the session ID. It gives the receiver no right to execute;
the worker still has to take the lease. A host transaction must not wait for the
drive it woke. If a wake is lost, the `LocalWorker` scan or a host
[tick](#periodic-tick) finds the session.

`RunHandle` and App Server ingress follow the same boundary. They write inbox
inputs and wake the worker; their cursor reads never turn the caller into the
lease holder.

An input with a future `available_ms` is not applied before that time. The
`LocalWorker` scan or host tick picks it up when it becomes due.

## Workers: drive

`SessionDriver` owns a `LocalWorker` with one coordinator and one single-slot
drive executor. The coordinator reuses its own store for scans, observations
and projections. It claims runnable work with a database lease and filters the
claim by the worker's level, handler versions and capabilities. PostgreSQL
worker claims use `FOR NO KEY UPDATE ... SKIP LOCKED`; the lease epoch and
expiry remain the write-safety fence.

1. A wake hints the coordinator to scan. A wake is not a lease and may be
   dropped.
2. If the drive slot is free, the coordinator submits one claimed session to
   the executor. While a drive is active, the worker still scans observation
   and projection work but does not start a second drive.
3. The executor opens a fresh drive store on its own thread and invokes the
   drive callback with the claimed lease. Once invoked, that callback owns
   defer, release and error handling for the claim.
4. The drive applies inputs and runs operations until nothing is runnable.
   Duplicate, reordered or concurrent deliveries never redispatch committed
   work.
5. The worker records completion or failure and wakes the coordinator. The
   periodic scan recovers due work, expired leases and lost wakes even without
   another hint.

A direct host queue uses its own scanner and transport instead of the
`LocalWorker` coordinator. Each `drive` call receives that queue task's store,
and its runtime closure binds callbacks to the same store on the worker thread.

Lease renewal is not another executor. `ThreadLeaseKeeper` registers heartbeats
with the process-wide scheduler. Each registration opens one independent
heartbeat connection on first renewal, reuses it for the registration lifetime,
and closes it when renewal stops. A heartbeat for runtime-factory preparation
starts at 15 seconds (`FACTORY_LEASE_TTL_MS`) and renews every 250 ms
(`FACTORY_HEARTBEAT_SECONDS`). The runtime's own TTL applies before execution.
A lost lease or changed epoch prevents use of the factory result, and every
append still checks the database fence.

A session is runnable when no live lease holds it, `drive_retry_at_ms <= now`,
and either `next_drive_ms <= now` or an unconsumed inbox input has
`available_ms <= now`. The tick scan and the per-session check use the same SQL
predicate, evaluated in database time.

An input that arrives while the holder is exiting is safe. If its wake finds the
lease still held, the holder's check after release sees the committed input. If
it arrives after that check, its own wake can take the released lease.

An App Server subscription is an observation view, not an execution owner.
Subscriptions select notification recipients and cursor positions; they do not
hold a drive slot or control the coordinator lifecycle. An observer-only
App Server can keep scanning without starting drives. Losing the last local
subscriber drops the live notification position; it does not stop or complete
the durable session.

### Readiness and failure backoff

If the host cannot build the runtime yet, raise
`RuntimeNotReady(retry_after_ms=...)` from the factory. `retry_after_ms` must be
a positive integer. `drive` then sets a retry gate in database time and releases
the lease. It consumes no input, admits no turn and writes no failure. A turn
that was already admitted keeps its frozen definition. Readiness is separate
from input guardrails: a guardrail denial still fails the turn.

If the factory or the driver raises any other exception, `drive` sets the same
gate for `failure_backoff_ms` (default 1000) and re-raises to the worker. This
only happens with the factory form; a drive given a `Runtime` instance raises
without setting a gate, so scheduled workers should always pass the factory.

The retry gate (`drive_retry_at_ms`) delays every drive of that session until it
expires, including drives for ready input, direct wakes and duplicate wakes.
New input does not bypass it, and that includes cancel and suspend commands.
The gate expires on its own, so no new signal is needed. Keep
`retry_after_ms` short, from under a second to a few seconds, and raise again on
the next check if the host is still not ready. The design target for cancelling
a running session with healthy workers is two seconds or less, and long gates
would break it.

### At-least-once delivery and external effects

At-least-once transport is enough for dispatch. It does not make external
effects happen exactly once.

- After a worker dies, committed receipts are reused.
- A model call that was issued but has no committed receipt becomes unknown.
  It may repeat as a logged retry, at most `max(2, frozen endpoint count)`
  attempts in total. The duplicate cost and the missing measurement are
  recorded.
- A tool retries at most once, and only when its frozen metadata declares
  idempotency support. The retry uses the same operation key.
- Other unknown effects are never repeated blindly.
- A paused worker can still send an external request it had already admitted,
  even after losing its lease.

See [recovery semantics](session-kernel.md#repair-inputs-and-provider-evidence).

## Periodic tick

`tick(store, dispatch=enqueue_drive, project=project_to_django)` scans for
runnable sessions and lagging consumers, in cursor order. For each runnable
session it calls `dispatch(sid)`; for each lagging consumer it calls the
`project` callback inline. In dispatch mode it never calls the runtime factory
or drives anything itself. Passing `runtime=callable_sid_to_runtime` instead of
`dispatch` drives each session synchronously inside the scan. A running
`LocalWorker` performs its own scans and does not need an external tick for
execution, though a host may still use `tick` for a separate queue integration.

Hosts that use `tick` must schedule it periodically even with a reliable broker.
Future-available inputs, due provider polls, dropped wakes and expired worker
leases are recovered by a scan. For `LocalWorker`, `scan_interval` supplies that
periodic scan. Queue backlog and execution time come on top of discovery time.

### Host dispatch example

This code lives in the host, with Celery installed there. `store()` owns a
thread-local PostgreSQL connection.

```python
from celery import shared_task

from vv_agent.session.kernel import drive
from vv_agent.session.runtime import RuntimeNotReady
from vv_agent.session.supervisor import tick

TICK_SECONDS = 2


def enqueue_drive(sid):
    drive_session.apply_async(args=(sid,), expires=TICK_SECONDS)


def runtime_factory(sid):
    if not child_inputs_ready(sid):
        raise RuntimeNotReady(retry_after_ms=500)
    # This callback has no drive-store argument. Put host database callbacks on
    # the SessionDriver runtime_factory when they must use the claimed store.
    return build_runtime(sid, wake=enqueue_drive)


@shared_task(name="agent.drive", acks_late=True, reject_on_worker_lost=True,
             ignore_result=True)
def drive_session(sid):
    with store() as session_store:
        drive(session_store, sid, runtime=runtime_factory, failure_backoff_ms=1000)


@shared_task(name="agent.tick", ignore_result=True)
def tick_sessions():
    with store() as session_store:
        tick(session_store, dispatch=enqueue_drive, project=project_to_django,
             page_size=100, failure_backoff_ms=1000)
```

Schedule the tick every two seconds and expire tick messages after two seconds.

Queue settings that go with it:

- Give every wake and every scan dispatch an expiry no longer than the tick
  interval, or coalesce queued messages by session ID. Tick does not claim
  queue messages, so a queued session that no worker has leased yet is still
  runnable and is dispatched again on every scan.
- Bound reserved and prefetched deliveries too, for example prefetch 1 and a
  fixed worker concurrency.
- An expired or lost delivery leaves the SQL work runnable, and the next tick
  finds it without any new input. Leases protect active workers from duplicate
  deliveries.
- Set the broker visibility timeout above the longest drive you allow.
- Give provider and tool calls timeouts, and give workers time limits with
  room for cancellation cleanup. After hard process loss, recovery relies on
  lease expiry plus the tick.

Queue routing, concurrency and prefetch are host choices. Tick needs no Redis
scan, command outbox or reconciliation worker. The beat interval bounds neither
queue latency nor total drive duration.

### Timeouts and errors in tick

In dispatch mode, scan time does not depend on drive duration. In inline mode
the scan is synchronous and its time includes every drive. `tick` itself starts
no thread; `LocalWorker`, when used, owns the coordinator and drive threads. The
kernel does not interrupt host callbacks, so the host must put finite timeouts
on SQL locks and statements, broker publishes and projection I/O. Keep
projections short and transactional.

Tick isolates each item. If a drive or projection fails, tick sets a backoff
for that item (`failure_backoff_ms`, default 1000) and moves on. Projection
backoff (`project_retry_at_ms`) applies only to that one consumer and does not
delay execution or other consumers. After visiting every page, tick raises an
`ExceptionGroup` holding the original exceptions, each with a note naming the
session, work kind and consumer. A failure to store a backoff is reported in
the same group. A failure of the scan query itself, such as a database outage,
aborts the scan, and the next periodic tick retries.

## Waits, children and cancellation

### Waiting for approval, a reply or a child

When a turn waits for approval, a user reply or a child session, the worker
releases the lease and returns. To answer, push the reply into that session's
inbox with the waiting turn, generation and interaction ID it retained, then
wake the session. A wait with no deadline schedules no drive.

Children are scheduled independently through the same transport or the tick.
When a child finishes, the `child_delivery` projection consumer calls
`child_delivery(store, tx, child_sid)`. It pushes the child's verified terminal
result into the parent's inbox and acks the child's cursor in one transaction.
Wake the parent after that transaction commits. A child that is itself waiting
is not finished and sends nothing to the parent.

The `LocalWorker` scans child sessions independently. Child admission records
the new session and its input in the parent transaction; child completion is
delivered by the `child_delivery` projection consumer. `RunHandle` observes the
durable result and does not own a second child driver. An input that arrives
during lease release remains in SQL for the next coordinator scan or explicit
host wake.

### Provider waits

A provider is a host adapter for external jobs that finish later. When a
provider accepts a job, the turn parks with a `poll_at_ms` of now plus
`Runtime.poll_ms` (default 1000 ms).

- Provider queries read `next_drive_ms` under the held lease in database time,
  so duplicate deliveries still respect the first `poll_at_ms` and any later
  deferral.
- Each operation is queried at most once per drive. A query that returns
  `Accepted` again leaves `poll_at_ms` and the log unchanged.
- Before an idle driver releases its lease, it moves an overdue `next_drive_ms`
  to database now plus the poll interval. The new time is capped by the
  earliest future operation deadline, poll time, retry time or inbox
  availability, so deadlines still fire at their original time.
- New input is runnable at once through the inbox check and does not wait for
  the deferred poll, unless a retry gate is active.
- An authenticated `provider_result` input can continue the turn immediately.

Local `RunHandle`, `Runner`, interactive and App Server execution report the
parked turn when a provider wait is recorded. If `poll_at_ms` is in the future,
the worker releases its lease without querying; no drive slot remains occupied.
The `LocalWorker` scan, a host tick, a later wake or `Runner.resume` runs the
query at or after the due time. Broker workers return from an idle `drive` and
rely on their host scan and wakes.

### Custom child admission

`Runtime.children` maps a tool name to a callback that returns one
`ChildSession` or a nonempty sequence of them. The kernel calls the callback
inside the parent's admission transaction. It validates every member and checks
that they share one background flag.

An empty batch, an invalid member or a mix of background and blocking children
raises `session.children.InvalidChildBatch` (code `invalid_child_batch`) before
any child is created or the parent parks. Host rows written through the same
connection roll back too. Do no non-transactional I/O and no external side
effects inside this callback. A Django adapter must open the matching ORM
transaction.

Each child's creation and initial input, the parent's started and parked
records, the background admission result and the callback's host rows all
commit together.

- A blocking batch waits for every original sibling to finish. It
  authenticates each delivery, cancels every live sibling when the parent
  closes, and keeps late or duplicate deliveries as evidence.
- A blocking batch completes with a JSON array of `ToolExecutionResult`
  objects, one per child in admission order, with `metadata.children`. If any
  child failed, the batch result is `ERROR`.
- A background batch's admission result is a JSON array of admitted handles,
  with `metadata.children`.
- A single child keeps the single-result shape. Children configured through
  the SDK use the same admission and delivery path.

#### Transaction callbacks

`prepare`, child `admit` and `before_commit` run in the kernel's database
transaction. Keep them to short checks and database writes. A slow callback
holds business row locks and keeps the transaction open.

Copy files, download or upload data before entering this transaction, or in a
job after commit. Preparation before the transaction needs a stable request
identity and a receipt the host can check. Inside the transaction, recheck
permissions, the request and that receipt, then commit the children and parent
records together. Wake children after commit.

Host business rows must use the same database transaction. A callback must not
commit an independent business transaction or start a child execution loop.
Database rollback does not undo object storage requests or changes to Python
objects.

A `SessionDriver` drive uses the fresh store opened by `LocalWorker`; the
registered `runtime_factory` must rebuild host callbacks against that store.
Direct `drive` uses its supplied store, and `tick` scans the store supplied by
the host. Custom subclasses and per-instance acquire or inbox hooks remain part
of whichever store instance owns the work. Heartbeats use an independent
connection factory.

An admission callback may write a business row and an inbox item in the kernel
transaction. It must not wait for the drive that consumes that input, start an
independent native transaction, or leave work relying on an external side
effect. Wake the target session only after commit.

Lease renewal runs on the process-owned `ThreadLeaseKeeper` thread and uses one
independent heartbeat store and connection for that registration. The drive
callback keeps ownership of its foreground store on the drive thread. Database
owner, epoch and expiry checks fence append, defer and release; the keeper never
shares the foreground connection or grants write authority.

Do not drive or poll while holding a host outer write transaction. It can hold
database rows needed by independent renewal, and leaving a savepoint does not
release the outer transaction's locks. Do not run concurrent native transactions
on the raw connection shared with the runtime; this is unsupported, and the SDK
does not promise to detect it.

A declared provider or foreground-child wait follows the same yield-on-steer
path for a direct host queue and a managed `LocalWorker`. The yield may commit
observed results and applied steering without stopping the underlying work.
`on_yield` renderers may read host state but must not publish files, bill or send
messages. Undeclared waits remain pending until their normal result. See
[yielding an observation](session-kernel.md#yielding-an-observation-to-steering).
Child admission remains `admit(plan)`, or `admit(plan, summary)` for summary
bindings; `Runtime.hook` remains `(point, record)`.

Heartbeat renewal continues during these callbacks. A PostgreSQL heartbeat
allows 250 ms for a lock wait and 1 second for a statement, including on
connections opened by a host's custom factory. A renewal timeout or operational
error retries on the heartbeat interval while the last published Lease has
more time left than both two heartbeat intervals and 1 second. Otherwise the
heartbeat reports `LeaseLost`. A store's `LeaseLost` or a changed owner or epoch
stops renewal immediately. Append refuses an expired Lease even between retries.
Custom connection contexts must also bound connection setup and exit; each
`close()` waits at most 6 seconds and refuses handoff if the thread is still
running.

The final append bounds each SQL lock wait and statement by the remaining
Lease time, with a 1 ms minimum. It restores the host's previous timeout settings
before returning, including when the append rolls back inside an outer
transaction. The final expiry check still rejects writes that outlive the Lease.

Heartbeat renewal is neither a callback
timeout nor permission to run indefinitely. Give network and database operations
bounded timeouts. Once a callback locks the parent session row for writing, it must do no
file or network I/O: another connection cannot renew a row that this transaction
has locked for writing. Do not wrap a whole `drive` in an outer host transaction;
leaving a savepoint does not release the outer transaction's locks.

If the database lease expires or another owner takes over, the final append
raises `LeaseLost` and the transaction rolls back. Do not catch that error and
commit the business results. Admission that rolled back remains eligible for
another drive; external preparation needs its own deduplication and recovery.
SQLite serializes writers, so it cannot renew from another connection during a
long write transaction.

The `vv_agent.session.kernel` logger reports the duration of `prepare`,
`before_commit` and append after the kernel leaves `store.atomic()`, including
failed callbacks. Durations at least half the configured lease TTL produce a
WARNING; shorter phases produce DEBUG. Fields include `session_id`, `turn_id`,
`operation_id`, `request_digest`, `epoch`, `phase`, `callback_ms`, `append_ms`,
`kernel_atomic_ms`, `outcome` and `exception_type`. Arguments, contents and
exception messages are omitted. `kernel_atomic_ms` measures this kernel block,
not an enclosing host transaction. A callback that never returns cannot produce
a completed duration.

### A compact tool and fresh child summaries

A host may offer a `compact` tool through its usual registry. Return intent
from the handler; the kernel makes the paid call after the tool batch:

```python
from vv_agent import ToolExecutionResult, function_tool

@function_tool
def compact() -> ToolExecutionResult:
    return ToolExecutionResult(
        "", "Compaction requested.", metadata={"request_compaction": True}
    )
```

The kernel adds the source operation and context digest to the receipt.
`continue` remains the directive. Finish or wait-user in the same batch takes
precedence. Finish ends the turn without a summary; wait-user defers the intent
until a valid reply resumes that turn. Empty history, rejected output, and exhausted summary failures keep
the context. Requests in one batch share one operation, including when the
threshold also fires. The result acknowledges intent; it does not claim the
summary already succeeded. Memory lifecycle event metadata carries
`requested_by` with the source tool operation IDs. Its trigger category is
`full_threshold`, including forced full compaction below that threshold.

For a child tool that needs current parent history, bind `SummaryChildTool`:

```python
from vv_agent import SummaryChildTool

def admit_child(plan, summary):
    # Write host rows and return ChildSession values on the admission transaction.
    # Include summary in the child's input when it is not None.
    return make_child(plan, parent_summary=summary)

runtime.children["create_sub_task"] = SummaryChildTool(
    requires_summary=lambda plan: bool(
        plan.payload["request"]["arguments"].get("include_main_summary", False)
    ),
    admit=admit_child,
)
```

The predicate must be pure. Change `handler_version` when the binding's behavior changes, as for any
host callback. The callback receives `None` when the predicate is false.
Otherwise it receives accepted summary text, including its memory and evidence
tags. Strip parent-only artifact references in the host when appropriate.
The driver freezes the completed parent history before the current tool batch,
including its recent tail, and logs a tools-free summary operation. It leaves
the parent's context intact. Late input does not change this frozen source.

The callback runs only after selection and acceptance of the saved receipt.
It must not call a model. Invalid or oversized source, invalid output, and
exhausted failures return `child_summary_unavailable` without admitting a child.
Model attempts, usage, endpoint failover and Unknown outcomes follow the normal
kernel rules. A request sent without a durable receipt may be retried with the
existing duplicate-cost warning. Hosts bill durable model receipts by operation
and attempt and retain unknown usage as unavailable.

### Streaming and cancellation

Live token, reasoning and tool-argument deltas can go through a host side
channel such as Redis pub/sub. They are volatile and carry no authority:
losing one cannot affect execution. Durable events, final content and billing
come from record projection and consumer cursors. A reconnecting client replays
those records and events. It should not trust broker results or cached deltas.

To cancel:

1. Authenticate the caller.
2. Push a `control` input with a stable `input_id`, `action="cancel"`, and the
   target turn and generation.
3. Commit, then wake the session.

An active holder sees the cancel through its heartbeat and control polling and
stops at the next boundary. An idle or waiting session processes it on its next
drive or tick. The kernel requests cooperative cancellation, fences later
writes, and records external outcomes as unknown when it cannot confirm they
stopped. Killing or revoking a Celery delivery does not cancel a session.

The [dispatch tests](../tests/session/test_dispatch.py) show these guarantees
with a controlled at-least-once queue against SQLite files and PostgreSQL. They
need no broker.

## Model routing

Resolve the default model and selected models from current host settings.
Endpoint IDs are routing, so adding, removing or renaming one does not change
the session definition. Use `Runtime.definition_digest(turn_started_record)`
when checking a retained definition. Keep checks for logical model identity,
tools and host authorization.

A custom `VvLlmClient.ordered_targets` decides every new model operation's
order, including after a selection. For example, a host can always put its
own endpoints before a shared pool. See [the resolver and ordering rules](session-model-selection.md#resolver-contract).

Treat a completion with `execution_started=false`, empty usage and
`error_code=endpoint_unavailable` as non-chargeable. It has no provider dispatch
to reconcile. A removed endpoint on an already-started attempt still needs the
ordinary receipt and Unknown handling.

## Billing

Bill from canonical per-call usage, keyed by the stable `call_id`. Read it from
`project_result(...).token_usage.model_calls`, or from the typed
`ModelCallCompletedEvent` and `ModelCallFailedEvent` produced by
`project_records`. Each call carries its own `model` and `backend`; bill with
those, never with the session's current model selection. Include late normal
completions and audit completions, because an audit call still cost money at
the provider. Never decode usage from opaque result content.

A recovered provider call arrives as an inbox schema 3 `provider_result`. Along
with the operation, attempt, request digest, provider binding, result and
evidence, it must carry `usage`: a JSON object, `{}` only when no usage was
observed. The host's authentication must cover `usage` too. Replaying the same
result and usage adds no completion and no charge. Different usage for the same
result is a conflict, even after the operation completed.

Keep billing rows in the same transaction as the consumer `ack`, so
deduplication stays with the cursor.

## Ledger integrity

Configure every SQL connection used by a signed deployment with the trusted
ledger signer for writes and the verifier that retains the required public keys
for reads and exports. Signed sessions require checkpoints for every record and
a valid admission proof for every inbox item. Missing signer material, partial
proofs, an unavailable key or a key identifier that does not verify the stored
bytes fails closed. A duplicate input may carry its original proof for replay;
the store compares it with the first admission and rejects a changed proof or
body.

For a rollback-resistant recovery, retain a head anchor outside the session
database and pass it as `expected_head` when reading or appending. Check it
before signing the next record. `export_session` accepts the same anchor, while
`verify_ledger` verifies the complete SQL prefix and returns the observed head.
Without an independent anchor, signature checks still find altered retained
bytes but cannot prove that a signed tail was not deleted or rolled back.

`configure_input_admission(allow_unsigned=True)` selects the local hash-only
mode explicitly. It reports structural consistency only and must not be used to
claim database tamper resistance, historical authenticity or input identity.
Signed evidence cannot be exported under that mode. Export with a verifier that
holds the relevant historical public keys; current writes use the active key
identifier returned by the signer binding.

## What to keep in your own database

The session store is an execution ledger. Its records are readable only by the
vv-agent version that wrote them, and a record or inbox schema change makes
every stored session unreadable, finished ones included. Do not use it as your
product's long-term history.

When a session ends, copy what your product needs into host tables: the
messages to show the user and to seed a follow-up session, shared state, the
final outcome and usage. Write them in the same projection transaction that
acks the consumer cursor. See
[session data and upgrades](upgrading.md#session-data-and-upgrades).

### Encrypted content lifecycle

Bind content reads and writes to the exact tenant, session and content
reference. S1 encryption authenticates those boundaries in the key-wrap AAD,
content AAD and HMAC framing, so moving a ciphertext, reference, key or content
line to another tenant or session fails closed. Content export defaults to the
skeleton. Hydration is an explicit include-content operation: call
`SQLContentStore.export_content_lines(...)` only with the references selected
from an already verified session snapshot. Missing content raises
`ContentUnavailable`; an unavailable or destroyed key raises `KeyUnavailable`.
The skeleton path must not call either read or hydration.

The development `LocalKeyProvider` persists active and destroyed key identity in
a local SQLite state file. Configure its location with the provider's
`state_path` argument or `VV_AGENT_LOCAL_KEY_STATE`; preserve that file across
process restarts. Losing it fails reads closed because wrapped keys no longer
have active lifecycle state. It is still a development provider, not a
production KMS.

`SessionDeletionStore` fences a root and all descendants before deleting their
keys, content and SQL rows. Forget retains the reference-form ledger skeleton and
one complete tombstone per session; purge removes the registered session tables
and retains only those tombstones. Repeating either operation is idempotent.
The SQL content store deletes local ciphertext rows itself. A row with an
external object locator also requires an idempotent cleaner attached with
`set_object_locator_cleaner(...)`; without that capability deletion raises
`ContentCleanupUnavailable`, and any cleaner exception remains a failed,
retryable deletion rather than an assumed remote success. Provider copies, host
tables, logs, exports and backups remain outside this deletion boundary.

## Model selection

A user can switch the model for the next cycle while a session runs. The host
pushes a `model_selection` input in its own transaction, with a stable input ID
for each user command, and wakes the session after commit. No lease is needed.
The runtime needs a `model_resolver` that turns the selection into a client.
Bill each call from its own model and backend, as described in
[billing](#billing).

[Session model selection](session-model-selection.md) owns the details:
payload, when a selection takes effect, the resolver contract and rejection.

## If you implement your own store

`PostgresStore` and `SQLiteStore` implement the `SessionStore` protocol in
`vv_agent/session/store.py`. A custom store must also implement these
scheduling methods with the same semantics.

`next_drive_delay_ms(session_id, *, lease=None) -> int | None` returns `None`
when nothing is scheduled, `0` when the session is due, or the remaining
milliseconds in database time. Provider queries pass their lease, and the store
rejects a lost fence.

`defer_idle_drive(lease, *, poll_ms) -> bool` moves an overdue schedule forward
and returns whether it changed anything.

- Hold the session write lock and read database time after taking it.
- Reject a stale or expired lease with `LeaseLost`.
- Leave a null or future schedule untouched, so a newer future schedule from a
  concurrent commit survives.
- Set the new time to now plus `poll_ms`, capped by the earliest future time in
  `ExecutionState.due_ms` and the earliest future inbox `available_ms`. The
  reducer keeps every operation due time in `due_ms`, so the store does not
  need its own copy of the operation scheduling rules. The folded state's
  `next_drive_ms` is the minimum of that list.

`is_runnable(session_id) -> bool` uses the same lease, due-time and inbox
predicate as `list_runnable`.

`defer_drive(lease, *, retry_after_ms) -> None` locks the session, checks that
the lease owner, epoch and expiry are still current, and advances
`drive_retry_at_ms`. It never changes `next_drive_ms`, input availability, the
log or turn state. The gate applies to every `acquire`, including after the
schedule is rebuilt.

`defer_projection(session_id, consumer, *, retry_after_ms) -> None` advances
only that consumer's `project_retry_at_ms` under its row lock, independently of
the drive gate and of other consumers.

Neither backoff method shortens a retry time that is already later. Both gates
expire by database time.

The SQL schedule changes only when execution changes. A commit that only adds
input or usage, and leaves the turn and the reducer's due times unchanged, keeps
the existing schedule. A commit that changes execution recomputes the schedule
from the log.

## Artifact read authorization

Use the public scope predicate when implementing a host read_file or search_files:

```python
from vv_agent import is_task_artifact_path

allowed = is_task_artifact_path(
    path,
    task_id=host_task_id,
    session_id=current_session_id,
    session_task_id=current_sdk_task_id,
)
```

All identities must come from the admitted runtime or trusted host records. The
model supplies only the path. session_task_id is the SDK turn/task identity,
which can differ from the backend's host task id. Omit it only when those ids are
the same. The predicate accepts one canonical artifact filename in the current
task or current session/task compaction scope. It grants neither account access
nor inherited access from another task, turn or session. Apply your account and
organization ACLs and storage availability checks as well. Do not use raw
workspace read methods as model-facing authorization boundaries.

The built-in read_file accepts start_char with a fixed 12,000-character page. Search columns use the
same zero-based Unicode units; line numbers start at one. Hosts with their own
reader should provide this same mapping, bounded streaming and a next position.
An interrupted or windowed search must say it is incomplete.

Task scopes use the encoding described in [Python mapping](parity-contract.md).
The host must ensure task/session ids do not differ only by case on its chosen
storage. This includes filesystem alias rules such as ignored trailing dots.
Built-in remote object storage is case-sensitive. A scope predicate does not
check a custom filesystem's alias behavior. See [upgrading](upgrading.md) for
old compaction references and frozen turns.
