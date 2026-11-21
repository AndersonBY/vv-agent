# Real-storage deterministic simulator (P1 first slice)

`SimWorld` schedules named participants on one thread. It enters through the
actual `vv_agent.kernel._drive.drive_core` and drains the real `CoreDriver`,
`_driver_recovery.step/close`, dispatch, planning, input, output, after-cycle,
compaction and outcome generators. SQLite/PostgreSQL stores, `SQLSessionTx`,
Fold, `LeaseOwnership`, child delivery, usage and admission rules remain the
shared core implementations. The assembly imports no Agent, Runtime, ToolRegistry
or `vv_agent.session` module. It supplies only concrete bindings, scripted model
and tool effects, cancellation and host observations.

Memory defaults come from the actual `vv_agent._memory_settings` leaf. Token
counts use the actual `vv_agent._token_counting` leaf and tool schema checks use
the actual `vv_agent._jsonschema` leaf. The assembly does not copy defaults,
prewarm a counter or substitute a prepared terminal. Every worker, heartbeat,
host, provider, scanner and observer owns a separate connection. SQLite uses a
real WAL file with the production FULL synchronous setting. No `SimStore` or
replacement execution state machine exists.

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
TMPDIR=/dev/shm VV_AGENT_SIM_WORKERS=20 uv run --no-sync pytest tests/sim -k simulation_smoke -q -s
uv run pytest tests/sim/test_import_boundary.py -q -s
```

The unqualified CLI and smoke gate run 1,000 seeds. The smoke gate asserts at
least 1,000 seeds per minute. The fixed prototype snapshot passed this smoke
requirement on the hardware documented below. Use
explicit small counts for local development. `VV_AGENT_SIM_WORKERS` selects smoke workers;
CLI uses `--workers`. On Linux and WSL, SQLite uses a warmed `fork` pool when the parent
has no live worker threads or SQLite connections. Other platforms, PostgreSQL, and unsafe
parents use `spawn`. The formal performance gate uses 20 workers on the reference host.
`--workers 1` runs serially.

The smoke gate uses `TMPDIR=/dev/shm` so its real SQLite files and WAL live in
tmpfs. The coordinator keeps at most `2 * workers` futures in flight, consumes
them in seed order and submits one replacement after each result. Set `TMPDIR`
only for the smoke command.
The database is still a real file, with native WAL/FULL settings, separate
participant connections and every oracle enabled. RAM-backed files survive
worker-process exits but are not evidence of host-power-loss durability. Keep
regular-filesystem directed process/transaction tests and PostgreSQL acceptance
separate; do not export `TMPDIR` for the full suite. Full remote checks split into:

```bash
uv run pytest -k "not simulation_smoke"
TMPDIR=/dev/shm VV_AGENT_SIM_WORKERS=20 timeout 60s uv run --no-sync pytest tests/sim -k simulation_smoke -q -s
```

Independent seeds use standard-library processes with a guarded, warmed `fork` start
method on Linux and a `spawn` fallback elsewhere; each world stays single-threaded and
owns its disposable real database file.
No live SQL connections are inherited. At most twice the worker count of jobs
and result traces are retained in the rolling window; aggregate memory does not grow with
the seed count. The measured campaign includes identity calculation, process
startup, child imports, initial tokenizer warmup, connection setup, SQL checks,
FULL-durability commits and process shutdown. Reports include Python, platform, CPU count, affinity,
visible memory and worker count. `environment.sqlite_temporary_storage` records
the actual temporary directory, filesystem type (null if unavailable), and
requested `TMPDIR`, so tmpfs evidence is distinguishable from disk evidence.
CLI timing also includes its harness imports;
an external stopwatch includes interpreter startup and exit.
No warmup seed is excluded or counted as a campaign seed.

`test_import_boundary.py` starts a fresh `-I -S` process, installs its meta-path
guard before simulation imports, and runs all 22 directed scenarios. Only the
kernel package, `_jsonschema`, `_memory_settings`, `_token_counting` and their
real external dependencies are permitted. The test then inspects `sys.modules`
to reject session, Agent, Runtime and tool-registry imports. This is process
isolation, not a loaded-module-cache exemption.

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
| I9 | Every regular SQLite/PostgreSQL world check recomputes record digests, the SQL chain and commit claims through the real store. Signed worlds additionally require and verify every checkpoint and input-admission proof; explicit hash-only worlds reject stray signed evidence. Missing proofs stop the check before another record or input can be trusted. |
| I10 | Unsafe unknown does not authorize resend; recovery stays ambiguous until a durable provider receipt or actual child terminal evidence settles it. |
| I11 | Real SQL claims cover heterogeneous level, handler, tool-schema, tool-metadata and model support sets; extra tool support remains compatible, incompatible workers do not advance lease epochs, no-match work stays pending, and root claims page/fence two tenants with the same session ID. Signed cache verification and claim-time no-content-read are checked. The exact compatible claim runs the real L2 session in core through `runtime.drive(..., lease=...)`; assembly preserves the claim and typed turn requirements. |

`covered` records the bounded evidence exercised by that scenario. In particular,
I4 needs positive scanner recovery and I5 needs attempt-accounting assertions;
merely calling a check does not establish those conditions.

Scenarios include tools, model unknown/retry exhaustion, declared idempotent and
unsafe tool unknowns, real precommit rollback, postcommit death, lost commit
reply, accepted handles, acceptance before ledger commit, authenticated late
results/redelivery, child delivery, cancellation, `RuntimeNotReady`, competing
claims, scheduled actual renewal, expiry/takeover at three driver cuts, frozen
runtime-definition mismatch, rolled-back model results and metered tool receipts.
The original 17 cases remain, with five additional participant cases:

- `approval_reply`: wrong request ID, request digest, scope, target turn and
  generation are rejected without dispatch. A valid reply authorizes exactly
  one effect; duplicate and late identical replies are no-ops.
- `future_input`: a future input is excluded from the ready inbox and early wakes;
  once due, SQL scanning alone resumes it and produces one additional effect.
- `cancel_descendants`: a fixed two-level chain has real active turns and pending
  work before root cancellation reaches both descendants. The grandchild waits
  for approval, so the test cannot pass through already-finished descendants.
- `context_compacted`: explicit low-threshold memory settings flow through the
  actual normalizer, token counter and real summary operation. Two bounded tool
  cycles create a reducible prefix; the accepted `context_compacted` record binds
  its adopted summary receipt, source digest, prefix/tail identities and
  replacement. I9 is active for every world.
- `worker_compatibility`: independent worker connections make real
  `claim_runnable` calls with typed support sets. A compatible worker receives a
  lease that is passed directly to the assembly/core drive path for the real L2
  session. Incompatible support sets and the all-unmatched case are checked on
  independent SQL connections.

Negative checks remove the actual SQL readiness filter, suppress the actual core
child-to-grandchild cancellation helper, and submit a false child terminal
receipt through the real core outcome generator. The first two fail the corresponding oracle;
the existing reducer rejects the false receipt and its real transaction rolls
back without consuming the child-result input.
Child receipt verification uses the driver's literal `child/<seq>/<digest>`
format and checks it against the session and turn in the retained child handle.

Still uncovered: arbitrary worker pause/resume during an in-flight external call, generated long lease/heartbeat
interleavings beyond the three directed cuts, approval denial/expiry variants,
wider/deeper descendant trees, model late-result races, and remote process start/query
(W3). Transport reordering currently has one parent session and cannot establish
multi-session ordering coverage. L2-L4 evidence belongs to
`test_lease_gap_evidence.py`, owned separately; L5 classification remains W4 work.

## Mutation evidence and failures

The mutation tests recompile and patch the actual shared-core functions in memory:
`SQLSessionTx` append fences and input replay/dedupe, `_driver_recovery.step`'s
retry key, `_driver_outcomes.unknown_steps`' retry decision, and
`LeaseOwnership.poll`'s lock ordering. A worker-filter mutation disables the SQL
support predicates and the typed `WorkerSupports.accepts` check, proving I11
rejects an incompatible claim. They retain real SQL and reducer constraints.
The input mutation removes the application replay branch and, only in its
disposable database schema, the `(session_id,input_id)` identity constraint.
The `(session_id,input_seq)` uniqueness remains, so a second push writes a real
duplicate-identity row and fails the explicit I3 receipt check. Production DDL
is unchanged.

L1 runs the real heartbeat poll and commits a real SQL record from its renewal
callback. The probe proves that `poll` does not hold `commit_lock` across the
renewal boundary; it does not construct concurrent database lock waits or a
deadlock. A mutation that holds the latch throughout poll produces a bounded L1
failure without hanging the test thread.

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
These results do not establish large-campaign reliability. The directed I11
worker-compatibility path is now covered; I9/W2c is exercised by every world
check rather than only by directed tests.


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


These fixed 17-scenario measurements do not validate other source snapshots,
final core changes or host-power-loss durability. PostgreSQL and performance
results must identify the exact source snapshot and contract selection.
