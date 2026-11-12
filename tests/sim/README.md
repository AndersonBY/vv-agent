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
TMPDIR=/dev/shm VV_AGENT_SIM_SEEDS=5 uv run pytest tests/sim -k simulation_smoke -s
```

The unqualified CLI and smoke gate run 1,000 seeds. The smoke gate asserts at
least 1,000 seeds per minute. The fixed prototype snapshot passed this smoke
requirement on the hardware documented below. Use
explicit small counts for local development. `VV_AGENT_SIM_WORKERS` selects smoke workers;
CLI uses `--workers`. The default is at most five, capped by CPU affinity (CPU count where affinity is
unavailable); the hard limit is eight. `--workers 1` runs serially.

The selected performance configuration is the original bounded batcher with
`TMPDIR=/dev/shm`. The Linux CI performance step sets that environment variable
only for the smoke command; the rolling-queue experiment is not retained.
The database is still a real file, with native WAL/FULL settings, separate
participant connections and every oracle enabled. RAM-backed files survive
worker-process exits but are not evidence of host-power-loss durability. Keep
regular-filesystem directed process/transaction tests and PostgreSQL acceptance
separate; do not export `TMPDIR` for the full suite. Full remote checks split into:

```bash
uv run pytest -k "not simulation_smoke"
TMPDIR=/dev/shm timeout 60s uv run pytest tests/sim -k smoke -q -s
```

Independent seeds use standard-library processes with the `spawn` start method;
each world stays single-threaded and owns its disposable real database file.
No live SQL connections are inherited. At most twice the worker count of jobs
and result traces are submitted per batch; aggregate memory does not grow with
the seed count. The measured campaign includes identity calculation, process
startup, child imports, initial tokenizer warmup, connection setup, SQL checks,
FULL-durability commits and process shutdown. Reports include Python, platform, CPU count, affinity,
visible memory and worker count. `environment.sqlite_temporary_storage` records
the actual temporary directory, filesystem type (null if unavailable), and
requested `TMPDIR`, so tmpfs evidence is distinguishable from disk evidence.
CLI timing also includes its harness imports;
an external stopwatch includes interpreter startup and exit.
No warmup seed is excluded or counted as a campaign seed.

For PostgreSQL set `VV_AGENT_TEST_POSTGRES_DSN` to a local/disposable **admin**
database before running `--backend postgres` or the parametrized tests. The
harness creates and drops only its own `vvsk_sim_*` database. Do not supply a
production DSN. PostgreSQL uses its unchanged `clock_timestamp()` SQL expression
and a real host clock. Lease expiry is an explicit update setting
`lease_until_ms=0`; test schedule advancement changes persisted deadlines rather
than replacing PostgreSQL time. The `future_input` case preserves both the inbox body
and `available_ms`; it waits for a real two- or three-second PostgreSQL deadline
with a ten-second bound. SQLite advances its injected clock without a real wait.
Use TCP/password authentication for acceptance;
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
| I3 | Repeated pushes replay their real receipt; durable inbox consumption matches exactly one `input_applied`. Every application must be timestamped at or after its immutable input's `available_ms`; the delayed-input case proves early exclusion and due-time consumption without rewriting the input. |
| I4 | Ready active work loses every broker hint and is recovered solely through bounded SQL scanning. Coverage requires positive discovered work, durable progress and no independently observed runnable work left over. Empty scans and quiescent checks do not claim coverage. Disabling the scanner fails a real ready-work scenario. Arbitrary provider liveness is not covered. |
| I5 | All started model attempts appear once in the result ledger, including unknown attempts with missing measurements; aggregate input/output tokens conserve that ledger. External model observations independently check committed results and cannot supply usage for a rolled-back result. Cumulative `usage_observed` observations are contiguous and retain the latest per-meter value. A metered tool case authenticates usage, rejects forged/conflicting receipts and prevents repeat billing. General host-cost billing is not covered. |
| I6 | Real transactional child delivery is repeated, a second input identity is a no-op, and a forged child identity is rejected. |
| I7 | Root cancellation terminates a nonempty parent → child → grandchild chain. All three sessions have started model work and are parked before cancellation; each descendant receives an actual bound cancel input and exactly one cancelled terminal. Late child deliveries cannot restart them. Wider/deeper trees are not generated. |
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
The original 17 cases remain, with three additional participant cases:

- `approval_reply`: wrong request ID, request digest, scope, target turn and
  generation are rejected without dispatch. A valid reply authorizes exactly
  one effect; duplicate and late identical replies are no-ops.
- `future_input`: a future input is excluded from the ready inbox and early wakes;
  once due, SQL scanning alone resumes it and produces one additional effect.
- `cancel_descendants`: a fixed two-level chain has real active turns and pending
  work before root cancellation reaches both descendants. The grandchild waits
  for approval, so the test cannot pass through already-finished descendants.

Negative checks remove the actual SQL readiness filter, suppress the actual
child-to-grandchild cancellation helper, and submit a false child terminal
receipt through the real driver. The first two fail the corresponding oracle;
the existing reducer rejects the false receipt and its real transaction rolls
back without consuming the child-result input.
Child receipt verification uses the driver's literal `child/<seq>/<digest>`
format and checks it against the session and turn in the retained child handle.

Still uncovered: mixed incompatible workers (I11/W4), arbitrary worker
pause/resume during an in-flight external call, generated long lease/heartbeat
interleavings beyond the three directed cuts, approval denial/expiry variants,
wider/deeper descendant trees, model late-result races, and remote process start/query
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

This is prototype P1 performance evidence for that 17-scenario snapshot and
hardware. It predates the approval, future-input and two-level cancellation
cases; those 17-scenario results do not establish expanded-code acceptance.
The large-campaign requirements are 1,000,000 SQLite seeds and at least 10,000
PostgreSQL seeds. Acceptance status belongs in the fixed-snapshot evidence and
release notes.
These results do not establish throughput on the two-CPU Colab environment or
large-campaign reliability. I9/W2c and I11/W4 remain later-phase requirements.


The expanded participant slice passed all 20 SQLite directed recovery cases
(including every original case) in 6.27s. The nine focused participant,
negative-control and replay checks passed in 5.75s. A bounded cold sample of seeds 0–19
with five workers passed 15 sampled scenarios and 298 decisions in 10.0587s
complete process time (9.9420s campaign/CLI time). Python 3.12.12, WSL/Linux,
20-CPU affinity, about 24 GiB visible RAM; startup resource check: about 8.9 GiB
available and load averages 2.17/1.87/1.68. Evidence:
`/tmp/vva-sim-participants-20.json`; immutable snapshot:
`/tmp/vva-sim-participants-handoff-9cb8igo1`; source digest:
`7bdff222f40e8b8034ea950b6a139fb73d21466fb1d47efa41696b9da3ed56ae`.
Approval was exercised by directed tests but was not selected in this random
sample. No steady-state or 1,000-seed throughput claim follows from this sample.


The accepted 17-scenario configuration is original `ec53fda` bounded batching
with `TMPDIR=/dev/shm`, WAL/FULL and all checks retained. On Colab with Python
3.12.3, eight CPUs, about 50 GiB RAM and eight workers, the same seeds 0–999 /
17 scenarios / 14,754 decisions passed in **26.0672s internally / 27.455982462s
for the full command**. Its report file SHA-256 was
`80ed0a0724db851301b32ee1a3da38cf940a5b0c89ec7dca56fdb60fb867d895`;
source digest `38aa5d9952ee69d9fe46978be68b3a55973d6db0263eddf1a9eab0f43aae8328`.
Evidence: `/content/trusted-p1-results/batch-tmpfs-smoke-status.json` and
`/content/trusted-p1-results/batch-tmpfs-smoke.log`.

Earlier experiments on that VM measured 68.6089s internally for original
batching on `/tmp`; rolling dispatch on `/tmp` took 72.6855s internally /
73.9615s for the full command. Rolling dispatch with `/dev/shm` took 19.9844s
internally / 21.23025s for the full command. The rolling variant remains
experimental evidence only; the simpler original batcher meets the gate and
is the selected implementation.

These fixed 17-scenario measurements do not validate other source snapshots,
final core changes or host-power-loss durability. PostgreSQL and performance
results must identify the exact source snapshot and contract selection.
