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
  `available_ms <= now` makes execution due, subject to lease availability;
  the scan and per-session check share the same SQL predicate.
- `tick(store, runtime=build_runtime, project=project_to_django)` scans due
  sessions and consumer lag in cursor order, driving work and calling the
  projection callback. Run it periodically even with a reliable broker.
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
time. New input is immediately runnable through the inbox predicate and does
not wait for the deferred poll. Input/usage-only commits that leave the turn
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

## Celery example (host code only)

The following lives in the backend, with Celery installed there. `store()` is a
host context manager owning a thread-local PostgreSQL connection, and
`build_runtime` defaults its wake callback to `drive_session.delay` for ticks.

```python
from celery import shared_task

from vv_agent.session.kernel import drive
from vv_agent.session.supervisor import tick


@shared_task(name="agent.drive", acks_late=True)
def drive_session(sid):
    with store() as session_store:
        drive(session_store, sid, runtime=build_runtime(sid, wake=drive_session.delay))


@shared_task(name="agent.tick")
def tick_sessions():
    with store() as session_store:
        tick(session_store, runtime=build_runtime, project=project_to_django)
```

Schedule `agent.tick` in Celery beat every few seconds. The tick task itself
drives discovered sessions synchronously; size its worker/time budget for the
scan. Queue routing, priorities, concurrency and prefetch are host choices.
`acks_late` acknowledges after return, but process loss can still acknowledge a
task; tick recovery does not depend on broker redelivery.
See [Celery task acknowledgements](https://docs.celeryq.dev/en/stable/userguide/tasks.html#acks-late).

Set the Redis broker's `visibility_timeout` above the expected longest drive
(potentially several turns). Early redelivery while the lease is held safely
returns, but wastes deliveries. Configure the applicable Celery visibility
settings consistently; a long timeout delays broker recovery, while tick still
discovers expired kernel leases. Prefer inbox `available_ms` plus tick for future
work. See [Celery Redis visibility timeout](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#visibility-timeout).

Set provider/tool timeouts first and allow worker time limits enough room for
normal drives and cancellation cleanup. Hard time limits kill the process and
leave recovery to lease TTL plus tick. Graceful deploy restarts allow drives to
finish; forced restarts follow the same recovery path, without an outbox drain
or reconciliation command. See [Celery worker time limits](https://docs.celeryq.dev/en/stable/userguide/workers.html#time-limits).

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
