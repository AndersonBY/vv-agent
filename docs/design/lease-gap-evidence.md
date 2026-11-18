# Lease gap evidence at the P1 baseline

L2, L3 and L4 reproduce lease/control robustness symptoms. The cuts below do
not establish a violation of W1's invariants. They remain W4 backlog items;
P1 does not repair them. L5 is classified here for the W4 implementation contract; the runtime implementation remains a later milestone.

The baseline is `d2d02c238cc8600d30aa9af87366ccba8fedba93` (0.35.0),
contract 38.0.0 at `1f4e6ff891996f1199f02e6b65edb1305ec7d8ae`.
Baseline references below were inspected with `git show d2d02c2:<path>`.
The executable probes use P1's injected timing seams and the real simulator
in `tests/sim/world.py`, with separate SQLite file connections for the host,
worker, heartbeat, competitor and observer. They are robustness checks, not
expected-failure correctness regressions. Seeds are passed to `SimWorld` and
its named choices select timing; labels pin the specific interleaving.

| Gap | Seed and controlled cut | W1 conclusion |
| --- | --- | --- |
| L2 | 2002; `l2/committed-renewal-then-stop`, then `l2/recover-after-db-expiry` | Cleanup/handoff can fail or wait for expiry. No demonstrated I2/I3/I4 failure. |
| L2 handoff | 2002; `l2/formal-driver-consumes-retained-control` | Losing an in-memory poll is not consuming its durable controls. |
| L3 | 3003; `l3/target-matched-switch-turn-resume-control` | Replacement token is spuriously cancelled; durable cancellation still terminates the original turn and its child (I7). |
| L4 | 4004; both wall-clock offsets, then `l4/recover-with-db-time` | Local false expiry or false liveness does not bypass SQL fencing (I2). |
| L5 | Classification only; no seed required | `SQLITE_BUSY`/`SQLITE_LOCKED` during renewal is retryable only while the last confirmed lease retains its safety margin; exhaustion becomes transient `TimeoutError`, and expiry/owner/epoch loss becomes `LeaseLost`. Busy is never treated as a successful renewal. |

## L2: a committed renewal can outlive its retained lease object

At baseline `src/vv_agent/session/kernel.py:123`, `_LeaseHeartbeat.poll`
serializes polls, snapshots the lease, calls `store.renew`, then publishes the
returned lease and deadline only while neither stopped nor lost. If stop wins
after SQL commit, publication is skipped. `drive` closes the factory heartbeat,
reads `build.lease`, and later calls `store.renew(...).lease` when handing off
to the formal scope. That last expression discards controls and database time.

The first probe starts an actual active turn, pushes a targeted cancel, advances
by a seeded choice of 1/2/3 seconds, and stops the build heartbeat after the
real renewal transaction commits but before its receipt is published. The row
has the new expiry; the heartbeat retains its original lease. Time then crosses
the original expiry, remaining inside the renewed interval. Release returns
false, renewal of the stale object raises `LeaseLost`, and another worker cannot
yet acquire. The test advances beyond the persisted expiry and recovers on a
new connection and epoch. The cancel is consumed once and the turn ends
cancelled, without any external call.

This is stranded ownership until the database lease expires, not permission to
write stale records. `src/vv_agent/session/sql.py:141` (`_check_lease`) checks
owner, epoch, persisted expiry and the supplied object's expiry. `release`
uses the same check and returns false on failure. `renew` reads controls but
never changes their consumed marker. The original poll's controls therefore
remain available to `_Driver.apply_input`. The handoff probe runs the real factory path through `drive`, pushes a
cancel after the build keeper has polled, and observes it in the handoff
renewal receipt. The formal driver consumes the retained cancel before dispatch. Every dispatch also polls and compares inbox watermark
before writing `op_started`.

I2 remains fenced, I3 retains its inbox receipt, and I4's recovery scan can claim
after the finite persisted lease expires. A live lease until that deadline is
still a scheduled waiter; immediate takeover is not a W1 promise. This argument
assumes advancing database time and a fairly scheduled compatible worker. It
does not claim a hard cancellation latency bound.

W4 should retain the last committed renewal evidence even when stop wins and
transfer the complete poll, including controls and database-time anchor, across
the factory boundary.

## L3: control delivery and token replacement have different lifetimes

At baseline `src/vv_agent/session/kernel.py:228`, `_Scope.poll` first obtains a
poll and then tests each control's target turn and generation before reading
`self.token` for cancellation. `_Driver.start_turn` installs a fresh token after
committing a new turn. `_Driver.step` also replaces a cancelled token after
suspend/resume when the durable turn is neither cancelled nor suspended. The
matching test and token lookup are not one operation.

The probe first admits a real child session and parks its parent. It then queues
a cancel targeted to the original parent turn plus a follow-up. A Python line
trace pauses `_Scope.poll` after the control target matched and immediately
before the token lookup. At that cut, the driver consumes the original cancel,
queues the child's cancellation in SQL, ends the original turn, and starts the
follow-up with a fresh token. Resuming the poll cancels the fresh token. This
is a deterministic scheduling cut; source lookup only locates the pause, and
no result is inferred from an AST or an implementation string. No timeouts or
thread timing determine the interleaving.

The old token is not cancelled and the replacement token is: the symptom is
real. Nevertheless, the original cancel's `input_applied` receipt has already
set `Turn.cancelled` in `src/vv_agent/session/reducer.py` (`_input`).
`_Driver.step` closes that durable turn, and `_Driver.close` calls
`cancel_children` in the same SQL transaction as its stopping facts. Descendants
receive targeted durable controls independently of whichever token is current.
The probe drives the child to a cancelled terminal, then drives the follow-up
to successful completion after its spurious token is reset. This exercises I7
on a nonempty descendant tree and checks I3 and prefix replay through the real
simulator. It does not prove a bound for a provider that never confirms stopping.

The race can delay interruption of the original external callback or interrupt
a later one. An admitted callback may already be running when cancellation is
requested; contract 38 explicitly separates requested cancel, local interruption
and confirmed external stop. I1/I10 still require the original attempt's unknown
or authentic outcome evidence; a token alone cannot establish completion or
create another attempt. W4 should bind the control match to its token lifetime.

## L4: opaque-store deadline inference is not write authority

At baseline `src/vv_agent/session/kernel.py:110`, `_LeaseHeartbeat.__init__`
uses `time.time()` if no database-time sample was supplied, then derives a local
monotonic deadline. `drive` supplies `_now()` for `SQLStore` instances; an opaque
custom store reaches the fallback. Clock skew and a pause between acquisition,
wall sampling and monotonic sampling can make that local deadline too early or
too late. A delayed matching wall/monotonic sample alone need not be harmful;
it is mismatched clock origins or a pause between those samples that matters.

The two probes omit the constructor's `now_ms` and supply wall time 60 seconds
ahead or behind the real SQLite database clock, while keeping monotonic time
unchanged. There is no substitute storage implementation. An ahead clock rejects
an otherwise valid scope early; the lease can still be released and recovered.
A behind clock admits the local commit latch after the lease expires and a
competitor has acquired the next epoch. The actual SQL append raises `LeaseLost`
and leaves the entire record prefix unchanged. Recovery completes with database
time. The seed chooses the positive offset beyond expiry for the latter cut.

The stock SQLite/PostgreSQL `drive` path supplies database time. More generally,
a conforming custom store must enforce the contract's owner/epoch/expiry fence
inside renew and append, including after validation. Dispatch renews before
its durable start; loss of authority stops it there. A false early local deadline
can abandon the current drive but scanning after expiry can recover it. A false
late deadline cannot authorize an append past the real storage fence. Thus this
fallback alone supplies no I2 counterexample. This reasoning is conditional on
the custom store satisfying its contract; these tests establish real SQLite
behavior, not correctness of arbitrary third-party stores or bounded liveness
under endless clock faults. W4 should use a database-time acquisition/renewal
anchor with a conservatively sampled monotonic start.

## L5: SQLite busy/locked classification

`SQLITE_BUSY` and `SQLITE_LOCKED` from a heartbeat renewal are contention, not
proof that the lease was lost. The settled W4 values are
`safety_margin = max(2 * heartbeat_interval, 1 second)` and
`safety_deadline = lease_expiry - safety_margin`, both on the monotonic lease
clock. Only those SQLite error codes (or the exact `database is locked` /
`database table is locked` messages on Python builds without extended codes)
are retryable while `now < safety_deadline`. Each attempt has a bounded
`heartbeat_attempt_budget`; its retry deadline is
`min(safety_deadline, now + heartbeat_attempt_budget)`. Exhaustion while the
safety deadline is still ahead is `TimeoutError` with the original SQLite error
as its cause. Reaching the safety deadline is `LeaseLost`.

A successful renewal must be confirmed by the database result; a busy response
never extends the lease. Owner/epoch mismatch, a row that is no longer owned,
or actual database-time expiry are `LeaseLost`. Malformed SQLite errors and
connection failures propagate as storage errors; they are not silently turned
into either contention or lease loss. No broad `OperationalError` catch is
allowed.

The current regression names are `test_file_heartbeat_recovers_from_sqlite_lock_without_duplicate_model_call`,
`test_persistent_sqlite_lock_exhausts_retry_budget_as_transient_timeout`, and
`test_sqlite_table_lock_message_is_retried`. W4 must add a real lease-margin
case that proves retry stops at the deadline, an owner/epoch mismatch case that
proves immediate loss, and a concurrent multi-agent write case that proves a
successful renewal is never inferred from a busy response. These tests are
implementation evidence for the settled classification; this design document
does not claim that W4 has shipped.

## Reproduction and limits

Run from the repository root:

```bash
uv run --no-sync pytest -q tests/sim/test_lease_gap_evidence.py
uv run --no-sync ruff check tests/sim/test_lease_gap_evidence.py
uv run --no-sync ruff format --check tests/sim/test_lease_gap_evidence.py
```

The repository artifact guard remains enabled. Tests use disposable SQLite files,
scripted models and production SQL/reducer/driver code. They make no provider
calls. PostgreSQL execution, broad simulation campaigns and full-suite
verification have not been performed for these probes. The P1 seam-based
probes are not claimed to have executed the unmodified baseline; the baseline
classifications above are grounded in its inspected implementation and contract.
I9 and I11 await their planned W2c/W4 implementations; L5 itself is now a settled design classification, pending that implementation evidence.
