# Real-storage deterministic simulator (P1 first slice)

`SimWorld` schedules named participants on one thread. It calls the production
`drive`, heartbeat `poll`, `SQLSessionTx`, reducer, child delivery and result
projection. Factory-bootstrap injection uses the same private `_drive` called
by the unchanged public `drive` facade. Every worker, heartbeat, host, provider, scanner and observer owns a
separate connection. SQLite uses a real WAL file with the production FULL
synchronous setting. No `SimStore` or replacement execution state machine exists.

The injected invoker executes scripted callbacks synchronously. Scheduling can
occur at safe kernel hooks, invocation boundaries and explicit participant steps;
it cannot suspend arbitrary Python instructions or a database transaction. SQL
lock timeouts requeue the blocked participant. This is bounded cooperative
interleaving coverage, not exhaustive instruction-level concurrency exploration.

## Running

From the repository root:

```bash
uv run python scripts/session_simulation.py --seeds 5 --workers 2
uv run python scripts/session_simulation.py --seed 17 --seeds 1 --scenario accepted_lost
uv run pytest tests/sim/test_simulation.py tests/sim/test_mutations.py -k sqlite
VV_AGENT_SIM_SEEDS=5 uv run pytest tests/sim -k simulation_smoke -s
```

The unqualified CLI and smoke gate run 1,000 seeds. The smoke gate asserts at
least 1,000 seeds per minute. The fixed prototype snapshot passed this smoke
requirement on the hardware documented below; **P1 remains incomplete**. Use
explicit small counts for local development. `VV_AGENT_SIM_WORKERS` selects smoke workers;
CLI uses `--workers`. The default is at most five, capped by CPU affinity (CPU count where affinity is
unavailable); the hard limit is eight. `--workers 1` runs serially.

Independent seeds use standard-library processes with the `spawn` start method;
each world stays single-threaded and owns its disposable real database file.
No live SQL connections are inherited. At most twice the worker count of jobs
and result traces are submitted per batch; aggregate memory does not grow with
the seed count. The measured campaign includes identity calculation, process
startup, child imports, initial tokenizer warmup, connection setup, SQL checks,
FULL-durability commits and process shutdown. Reports include Python, platform, CPU count, affinity,
visible memory and worker count. CLI timing also includes its harness imports;
an external stopwatch includes interpreter startup and exit.
No warmup seed is excluded or counted as a campaign seed.

For PostgreSQL set `VV_AGENT_TEST_POSTGRES_DSN` to a local/disposable **admin**
database before running `--backend postgres` or the parametrized tests. The
harness creates and drops only its own `vvsk_sim_*` database. Do not supply a
production DSN. PostgreSQL uses its unchanged `clock_timestamp()` SQL expression
and a real host clock. Lease expiry is an explicit update setting
`lease_until_ms=0`; test schedule advancement changes persisted deadlines rather
than replacing PostgreSQL time. Use TCP/password authentication for acceptance;
no PostgreSQL acceptance is implied by SQLite results.

In a shared checkout, pytest's repository artifact guard can fail when another
writer creates a file. Run these tests from an immutable snapshot with the
shared environment's `python -m pytest` and `PYTHONPATH=<snapshot>/src:<snapshot>`.
Keep the guard intact.

## Checked scope

| Invariant | First-slice evidence and limit |
| --- | --- |
| I1 | Actual model/tool calls counted per operation/attempt; declared tool retry is bounded, keys persist, the scripted external service deduplicates effects; model bound comes from the frozen endpoint order. Recorded results survive worker replacement and delayed duplicate deliveries. |
| I2 | An actual stale-lease append after a competing connection takes ownership must fail. An independent observer checks committed writer epochs/expiry against authority sampled at the actual SQL write. No power-loss durability claim. |
| I3 | Repeated pushes replay their real receipt; durable inbox consumption matches exactly one `input_applied`. |
| I4 | Ready active work loses every broker hint and is recovered solely through bounded SQL scanning. Coverage requires positive discovered work, durable progress and no independently observed runnable work left over. Empty scans and quiescent checks do not claim coverage. Disabling the scanner fails a real ready-work scenario. Arbitrary provider liveness is not covered. |
| I5 | All started model attempts appear once in the result ledger, including unknown attempts with missing measurements; aggregate input/output tokens conserve that ledger. External model observations independently check committed results and cannot supply usage for a rolled-back result. Cumulative `usage_observed` observations are contiguous and retain the latest per-meter value. A metered tool case authenticates usage, rejects forged/conflicting receipts and prevents repeat billing. General host-cost billing is not covered. |
| I6 | Real transactional child delivery is repeated, a second input identity is a no-op, and a forged child identity is rejected. |
| I7 | A cancelled parent terminates its single child via actual control inputs and drive. Deeper descendant trees are not generated yet. |
| I8 | Cold record-prefix replay equals a long-lived incremental Fold and the store cache. Directed recovery tests check every prefix; smoke checks scheduler-step prefixes. |
| I9 | Pending W2c; no hash-chain field or verifier exists in the baseline. |
| I10 | Unsafe unknown does not authorize resend; recovery stays ambiguous until a durable provider receipt or actual child terminal evidence settles it. |
| I11 | Pending W4; compatibility support sets and claim filtering do not exist in the baseline. |

`covered` records the bounded evidence exercised by that scenario. In particular,
I4 needs positive scanner recovery and I5 needs attempt-accounting assertions;
merely calling a check does not establish those conditions.

Scenarios include tools, model unknown/retry exhaustion, declared idempotent and
unsafe tool unknowns, real precommit rollback, postcommit death, lost commit
reply, accepted handles, acceptance before ledger commit, authenticated late
results/redelivery, child delivery, cancellation, `RuntimeNotReady`, competing
claims, scheduled actual renewal, expiry/takeover at three driver cuts, frozen
runtime-definition mismatch, rolled-back model results and metered tool receipts.

Still uncovered: mixed incompatible workers (I11/W4), arbitrary worker
pause/resume during an in-flight external call, generated long lease/heartbeat
interleavings beyond the three directed cuts, delayed input availability,
approvals, grandchildren, model late-result races, and remote process start/query
(W3). Transport reordering currently has one parent session and cannot establish
multi-session ordering coverage. L2-L4 evidence belongs to
`test_lease_gap_evidence.py`, owned separately; L5 classification remains W4 work.

## Mutation evidence and failures

The mutation tests recompile and patch the actual production functions in memory:
append fences, input replay/dedupe, the kernel's retry key, and the kernel's unsafe
unknown retry decision. They retain real SQL and reducer constraints. In
particular, removing input replay reaches the database unique constraint; the
harness rejects that failed duplicate push as an I3 violation rather than
claiming duplicate durable rows were created.

L1 runs the real heartbeat poll and commits a real SQL record at its renewal
boundary. A mutation that holds `commit_lock` throughout that poll prevents the
append from progressing, producing a bounded L1 failure without hanging a test
thread. It is lock-order evidence, not a stress test of OS thread scheduling.

```bash
uv run pytest tests/sim/test_mutations.py -k sqlite -q
uv run python scripts/session_simulation.py --seed 17 --seeds 1 --report /tmp/sim-failure.json
uv run python scripts/session_simulation.py --replay /tmp/sim-failure.json
```

Failures include seed, scenario, named decisions, actual hook events, calls,
backend, code revision, dirty state, source digest, pinned contract version and
revision, and in-memory mutation identities. Campaign identity is computed once
and attached to its success aggregate or failure; successful per-world summaries
do not read Git or hash source. An archive without Git reports null revision and
dirty state, with an exact source digest.

Replay rejects missing/different source digests, contract identities or mutation
identities before executing a seed, and reports every provenance difference.
Git availability differences are reported separately when exact content matches.
The five implementation mutation tests record patched method, bytecode digest
and mutation label in pytest evidence. Ordinary CLI execution cannot reproduce
an in-memory mutant; it rejects that report. Replay also rejects unavailable or
unused recorded choices. PostgreSQL replays scheduling decisions; its real time
and record bytes may differ.

`--minimize-budget N` permits at most N extra disposable reruns after an invariant
failure. Bounded ddmin deletes choices only when a real rerun fails the same
invariant. Missing choices use the first available action. The report retains
both original and reduced decisions and the number of trials; zero budget is
explicitly unminimized. Set its `decisions` field to `minimized_decisions` to replay
the reduced trace. No claim of global minimality is made.

## Bounded measurements

A fixed prototype snapshot, source digest
`09146a2ecf6c8a6494866d1f37d2c63ba426925586b084521417f943a86c1d06`,
passed the SQLite 1,000-seed pytest smoke with seeds 0–999, all 17 scenarios and
14,754 scheduler decisions. Hardware: Python 3.12.12 on WSL/Linux, 20 logical CPUs
and 20-CPU affinity, about 24 GiB visible RAM, five spawned workers. The entire
`uv` command took **53.088440017 seconds** (about **1,130.2 seeds/minute**); the
internal campaign took **52.0506 seconds** (1,152.7 seeds/minute). Complete-command
timing includes startup, worker initialization/warmup and pool shutdown; no
warmup seed was excluded. The command was `uv run --no-sync pytest tests/sim -k smoke -q -s`, run in a frozen source snapshot with the repository-managed environment and five workers.

The same fixed snapshot passed the bounded SQLite suites (57 passed,
40 deselected, 17.11s) and real PostgreSQL 18.6 suites over TCP/password
(39 passed, 58 deselected, 29.72s). Contract: 38.0.0, revision
`1f4e6ff891996f1199f02e6b65edb1305ec7d8ae`.

This is prototype P1 performance evidence for that snapshot and hardware.
**P1 remains incomplete:** final core acceptance and the planned large campaigns
(1,000,000 SQLite seeds and at least 10,000 PostgreSQL seeds) remain pending.
These results do not establish throughput on the two-CPU Colab environment or
large-campaign reliability. I9/W2c and I11/W4 remain later-phase requirements.
