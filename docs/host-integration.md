# Cloud host integration

vv-agent owns durable execution and dispatch semantics. The host owns transport,
workers, database connections, authentication, runtime assembly and projections.
Celery and Redis remain the .ai site's runtime framework; vv-agent ships neither
their code nor an integration module or extra. SQLite is for a single host;
cloud workers share one authoritative PostgreSQL session store.

## Database and runtime factory

Prefer the host's own PostgreSQL database. Bind `PostgresStore(connection)` to a
caller-owned psycopg v3 connection, and use `join_transaction` for host writes:

```python
from vv_agent.session.postgres import join_transaction

with connection.transaction():
    tx = join_transaction(connection)
    batch = tx.consumer_batch(sid, "host")
    if batch is not None:
        project_records_and_bill(connection, batch)
        tx.ack(batch)
```

Register consumers at session creation. Host projections, billing and their
kernel consumer cursor commit together; admission writes can likewise share
`tx.create`/`tx.push`/`tx.append` with host rows. There is no command outbox or
reconciliation command. The log is authoritative and the cursor marks the last
projected record. External sinks deduplicate by stable record/event identity.
Keep transactions short: never wrap a whole `drive` or provider call in a host
transaction. `join_transaction` requires an already-open transaction and never
commits it. Django connection extraction and transaction validation belong to
the backend adapter. See [transaction details](session-kernel.md#joining-host-transactions).

`build_runtime(sid, wake=...)` reconstructs the session's agent, RunConfig, exact
model/client, tools/providers, workspace, child factories, host bindings and
handler version. Resume uses the retained definition; a changed binding stops
explicitly. Supply one store connection per driving thread and a factory opening
an independent heartbeat connection to the same database. The default lease TTL
is 15 seconds with 250 ms heartbeats. Clients issue one external attempt per
logged dispatch; kernel retries must not stack with client retries. Bind live
stream sinks separately from durable projection consumers.

## Wake, drive and tick

- Commit an inbox input with a stable source-owned `input_id`, then call
  `wake(sid)` after the host transaction commits. A wake carries only a session
  identity, never execution authority. Lost wakes are recovered by `tick`.
- `drive(store, sid, runtime=...)` holds one lease and fences writes by epoch.
  Duplicate, reordered or concurrent deliveries do not redispatch committed
  work. A delivery that finds a held lease returns without waking. The holder
  releases its lease, then checks in a fresh transaction whether execution is
  still runnable. Only then does it wake. Idle sessions and completed turns with
  no remaining work do not self-wake. `next_drive_ms <= now` or an unconsumed inbox item with
  `available_ms <= now` makes execution due, subject to lease availability and
  `drive_retry_at_ms <= now`;
  the scan and per-session check share the same SQL predicate.
- `tick(store, dispatch=enqueue_drive, project=project_to_django)` scans due
  sessions and consumer lag in cursor order, dispatching drive identities and
  calling the projection callback inline. It never calls the runtime factory
  or drives sessions in dispatch mode. Omitting dispatch retains synchronous
  in-process execution through `runtime=build_runtime`. Run it periodically even with a reliable broker.
  Future-available inbox items, due provider polls, dropped wakes and expired
  worker leases rely on this scan. Its interval bounds discovery latency after
  the due time/lease expiry; queue backlog and execution add further latency.

An input arriving as a holder exits is safe: if its wake finds the lease held,
the holder's post-release check sees the committed input; if it arrives after
that check, its own wake can acquire the released lease. Local RunHandle child
scheduling uses explicit `after_drive`/`after_input` hooks, separate from wake.

Provider queries read SQL `next_drive_ms` under the held lease using database
time, so even duplicate deliveries honor the first `poll_at_ms` and later
deferral. Each operation is queried at most once per drive. An `Accepted` query leaves the immutable
`poll_at_ms` and execution log unchanged. Before releasing its lease, an idle
driver defers a stale SQL `next_drive_ms` to database now plus the poll interval,
capped by the earliest future operation deadline, poll, not-before/retry time or
inbox availability. The reducer retains all operation due times in
`ExecutionState.due_ms`; the store uses that list without duplicating operation
scheduling rules. `next_drive_ms` in the folded state remains its minimum.
Deadline handling therefore remains due at its original
time. Outside failure/readiness backoff, new input is immediately runnable
through the inbox predicate and does not wait for the deferred poll.
Input/usage-only commits that leave the turn
and reducer due times unchanged preserve the existing SQL schedule; execution
changes recompute it from the log. Completed `_one_turn` drives return without this idle deferral,
so queued turns remain immediately runnable.

Custom `SessionStore` implementations must implement the public, typed
`next_drive_delay_ms(session_id, *, lease=None) -> int | None`,
`defer_idle_drive(lease, *, poll_ms) -> bool` and
`is_runnable(session_id) -> bool` methods. Deferral returns whether it changed
the schedule, rejects stale/expired leases with `LeaseLost`, and leaves null or
future schedules untouched. Hold the session write lock, read database time
after acquiring it, and fence the update by the lease; a concurrent commit's
newer future schedule must survive. `is_runnable` uses the same lease/due/inbox
predicate as `list_runnable`.

The schedule reader returns null for no schedule, zero when due, or the remaining
database milliseconds. Provider queries pass their lease and reject lost fences.
Local RunHandle, Runner, interactive and App Server execution return when the
turn parks on a provider wait. If `poll_at_ms` is in the future, they return
without querying or leaving a thread waiting. The host tick/supervisor, a later
wake or `Runner.resume` drives the query at or after the due time; an
authenticated `provider_result` can continue the turn immediately. Broker
workers return from an idle `drive` and rely on host wakes/ticks.

Waiting for approval, a user reply or a child releases the lease and returns the
worker. Route approval/user replies to that waiting session's inbox, with its
retained turn/generation and interaction identity, then wake it. Independently
schedule children via the same transport or tick. The `child_delivery` projection
consumer calls `child_delivery(store, tx, child_sid)` to push a verified terminal
input to the parent and acknowledge the child cursor atomically; wake the parent
after that transaction commits. Intermediate child waits are not terminal
notifications. An unresolved wait without a deadline has no scheduled drive.

At-least-once transport is sufficient for dispatch; it does not make external
effects exactly once. After a worker dies, committed receipts are reused.
An issued model call without a committed receipt becomes unknown and may repeat
as a logged retry (at most `max(2, frozen endpoint count)` attempts), with
duplicate cost and missing measurement recorded. Tools retry at most once only
when their frozen metadata declares supported idempotency, using the same
operation key; other unknown effects must not be blindly repeated. A paused
worker can still issue an already-admitted external request even after losing
its lease. See [recovery semantics](session-kernel.md#repair-inputs-and-provider-evidence).

## Host dispatch example

The following lives in the backend with Celery installed there. `store()` owns a
thread-local PostgreSQL connection. Pass the **factory itself** into `drive` so
preparation runs after lease acquisition and participates in scheduled backoff.

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

Schedule tick every two seconds and expire tick messages after two seconds too.
All wake/scan dispatch uses expiry no longer than that interval, or host queue
coalescing by session identity. Tick does not claim queue messages: a queued but
not-yet-leased session remains runnable and may be dispatched on each scan.
Bound reserved/prefetched deliveries as well (for example prefetch 1 and bounded
worker concurrency). Expired/lost deliveries leave SQL work runnable; the next
tick discovers it without a new input. Active workers are protected by leases.
The beat interval does not bound queue latency or total drive duration.

Dispatch mode scan time is independent of drive duration. SQL lock/statement,
broker publish and projection I/O need host-enforced finite timeouts; projections
stay short and transactional. Inline mode is synchronous and its scan time
includes every drive. The kernel does not interrupt arbitrary host callbacks or
create background scan threads. Each item is isolated; after visiting every page,
`tick` raises an `ExceptionGroup` containing original exceptions with session,
kind and consumer notes. Failure to persist a backoff is also reported. A scan
query/database outage still aborts the scan; the next periodic scan retries.

`drive(..., runtime=runtime_factory)` acquires a 15-second preparation lease,
then renews it to the returned Runtime TTL before execution. Keep preparation
shorter than that lease and use host I/O timeouts. RuntimeNotReady has a positive
integer `retry_after_ms`; it sets a database-time retry gate and releases the
lease without consuming input, admitting a turn or writing a failed terminal.
An already-admitted turn and its frozen definition survive deferral. Runtime
factory/driver exceptions defer by `failure_backoff_ms` (default 1000) and
re-raise to the worker. Explicit Runtime-instance drives keep their synchronous
error behavior; use the factory form for scheduled workers. Factory readiness
is separate from input guardrail denial, which still fails the turn.

Stores must implement `defer_drive(lease, *, retry_after_ms) -> None` and
`defer_projection(session_id, consumer, *, retry_after_ms) -> None`. Drive backoff
locks the session, checks the current owner/epoch/unexpired lease, then advances
`drive_retry_at_ms`. It never changes `next_drive_ms`, input availability, log or
turn state; a concurrent newer schedule survives. The gate applies even to ready
inbox and direct/duplicate wakes (`acquire`), including after schedule rebuild.
Projection backoff advances only that consumer's `project_retry_at_ms` under its
row lock, independently of drive and other consumers. Neither primitive shortens
an existing later retry. Backoff expires automatically under database time; a
new signal is unnecessary. New input does not bypass this short retry gate.
The gate delays both control inputs (cancel/suspend) and user input for that
session until it expires. Hosts should keep `RuntimeNotReady.retry_after_ms`
short, from sub-second delays to a few seconds, and re-raise on later readiness
checks instead of returning long delays. The plan's cancellation target of
at most two seconds applies to running sessions under healthy workers.

The PostgreSQL DDL adds exactly those two nonnegative bigint columns with default
zero, one on `sk_session`, one on `sk_consumer`; no table, execution ledger or
index is added. SQLite mirrors them and advances `PRAGMA user_version` to 2,
rejecting old version 1 databases. Record/inbox schema 1, event and App Server
wires are unchanged. Hosts copying DDL must take the current literal from
`session/postgres.py`. The framework has no historical schema migrator: finish
old executions using their pinned artifact, then provision the current schema;
any host-managed schema change is an explicit host responsibility.

Set broker visibility timeout above the host's longest allowed drive. Use
provider/tool timeouts and worker limits with room for cancellation cleanup;
hard process loss leaves recovery to lease TTL plus tick. Queue routing,
concurrency and prefetch remain host choices. Tick needs no Redis scan, command
outbox or reconciliation worker. At-least-once external requests and unknown
outcomes retain the [existing recovery rules](session-kernel.md#repair-inputs-and-provider-evidence).

## Custom child admission

A Runtime.children callback returns one `ChildSession` or a nonempty sequence.
The kernel calls it inside the parent admission transaction, validates all
members and their common background flag, then uses the delegated siblings
path. Empty, invalid-member and mixed-background batches raise
`session.children.InvalidChildBatch` (code `invalid_child_batch`) before creating
children or parking the parent. Host rows written through that same connection
roll back too. No nontransactional I/O or external side effect belongs inside
this callback; Django adapters must open the matching ORM transaction.

Every child creation/initial input, parent started/parked record and background
admission result commits together with callback host rows. Blocking batches wait
for all original sibling terminals, authenticate each delivery, cancel every
live sibling on parent closure and retain late/duplicate evidence. Custom batch
completion content is a JSON array of per-child ToolExecutionResult objects in
admission order, with `metadata.children`; any child error makes the batch ERROR.
Background batch admission content is a JSON array of admitted handles with
`metadata.children`. Singleton results retain their existing shape. Configured
SDK children retain their typed configured-tool projection through the same
admission/delivery machinery.

## Streaming and cancellation

Live token/reasoning/tool deltas can use a host side channel, such as Redis
pub/sub. They are volatile and non-authoritative: losing a delta cannot affect
execution correctness. Durable events, final content and billing come from
record projection and consumer cursors; clients reconnect by replaying those
records/events, not by trusting broker results or cached deltas.

For cancellation, authenticate the caller, push a stable `control` input with
`action="cancel"` and the target turn/generation, commit, then wake the session.
An active holder sees it through heartbeat/control polling and at the next
boundary; an idle or waiting session processes it on drive/tick. The kernel
requests cooperative cancellation, fences subsequent writes and retains unknown
external outcomes when stopping cannot be confirmed. Killing or revoking a
Celery delivery is not the durable cancellation path.

The deterministic [dispatch tests](../tests/session/test_dispatch.py) use a
controlled at-least-once queue against SQLite files and PostgreSQL, including
duplicates, reordering, exit races, abandoned leases, waits, delayed inputs,
cancellation and monotonic projection cursors. No broker is required to test
these kernel guarantees.
