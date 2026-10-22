# Cloud host integration

This guide is for a host that runs vv-agent sessions on many workers that share
one PostgreSQL database. vv-agent decides what a session does next and records
every step in a durable log. The host owns everything around that: the queue
and workers, database connections, authentication, how a runtime is built, and
the product tables it projects into.

vv-agent ships no Celery or Redis code and no integration extra. The Celery
example below lives in the host. A single-process host can use SQLite instead
of PostgreSQL.

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
string in `vv_agent/session/postgres.py`. A host that manages its own DDL must
copy that literal from the installed release. SQLite files carry
`PRAGMA user_version` 3, and the store refuses to open any other version.

vv-agent has no schema migrator. Finish old sessions with the release that
created them, then provision the current schema. Any schema change the host
makes itself is the host's responsibility. See
[upgrading](upgrading.md#session-data-and-upgrades) for version history and what
an upgrade means for stored sessions.

## Runtime factory

`build_runtime(sid, wake=...)` is a host function. It rebuilds what a session
needs to run: the agent, `RunConfig`, the exact model and client, tools and
providers, workspace, child factories, host bindings and handler version. A
resumed turn runs on the definition it retained when it started. If the rebuilt
binding no longer matches, the session stops explicitly instead of continuing
on different settings.

When you build the `Runtime`:

- Give each driving thread its own store connection. Set `heartbeat_store` to a
  factory that opens an independent connection to the same database; the
  heartbeat thread uses it to renew the lease.
- The lease TTL defaults to 15 seconds (`ttl_ms=15000`) and the heartbeat to
  250 ms (`heartbeat_seconds=0.25`). The heartbeat must be at most 1 second
  and less than half the TTL.
- Configure the model client to make one external attempt per call. The kernel
  logs and retries attempts itself, and client retries would stack on top.
- Set `wake` to the host's enqueue function. The kernel calls it when a session
  still has work after a drive.
- Bind live stream sinks separately from durable projection consumers (see
  [streaming](#streaming-and-cancellation)).

## Submitting input and waking

1. In a host transaction, push an inbox input with a stable `input_id` owned by
   the source (for example a message ID or command ID).
2. After that transaction commits, call `wake(sid)`.

A wake carries only the session ID. It gives the receiver no right to execute;
the worker still has to take the lease. If a wake is lost, the periodic
[tick](#periodic-tick) finds the session.

An input with a future `available_ms` is not applied before that time. The tick
picks it up when it becomes due.

## Workers: drive

A worker calls `drive(store, sid, runtime=runtime_factory)`. Pass the factory
itself, not a built `Runtime`, so preparation runs after the lease is taken and
takes part in scheduled backoff.

1. `drive` tries to take the lease. If another worker holds it, `drive` returns
   at once without waking anyone.
2. It calls the factory under a 15-second preparation lease, then renews the
   lease to the runtime's `ttl_ms`. Keep preparation shorter than 15 seconds
   and give host I/O its own timeouts.
3. It applies inputs and runs operations until nothing is runnable.
   Duplicate, reordered or concurrent deliveries never redispatch committed
   work.
4. It releases the lease and then checks, in a fresh transaction, whether the
   session is still runnable. Only then does it call `wake`. Idle sessions and
   finished turns with no remaining work do not wake themselves.

A session is runnable when no live lease holds it, `drive_retry_at_ms <= now`,
and either `next_drive_ms <= now` or an unconsumed inbox input has
`available_ms <= now`. The tick scan and the per-session check use the same SQL
predicate, evaluated in database time.

An input that arrives while the holder is exiting is safe. If its wake finds the
lease still held, the holder's check after release sees the committed input. If
it arrives after that check, its own wake can take the released lease.

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
or drives anything itself. Passing `runtime=build_runtime` instead of
`dispatch` drives each session synchronously inside the scan.

Run tick periodically even with a reliable broker. Future-available inputs, due
provider polls, dropped wakes and expired worker leases are only found by this
scan. The tick interval bounds how long it takes to discover work after it
becomes due or after a lease expires. Queue backlog and execution time come on
top of that.

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
the scan is synchronous and its time includes every drive. The kernel does not
interrupt host callbacks and starts no background scan threads, so the host
must put finite timeouts on SQL locks and statements, broker publishes and
projection I/O. Keep projections short and transactional.

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

Local `RunHandle` execution schedules children through the `after_drive` and
`after_input` hooks, separately from `wake`.

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

Local `RunHandle`, `Runner`, interactive and App Server execution return when a
turn parks on a provider wait. If `poll_at_ms` is in the future, they return
without querying and leave no thread waiting. The host tick, a later wake or
`Runner.resume` runs the query at or after the due time. Broker workers return
from an idle `drive` and rely on host wakes and ticks.

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
