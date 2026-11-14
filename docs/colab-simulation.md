# Colab detached simulation

`scripts/colab_campaign.py` owns one P1 SQLite and PostgreSQL campaign on one
fresh Colab CPU high-memory session. It uses short-lived installed CLI commands
and does not use `colab run` for the campaign.

The former `colab run` path failed after a real websocket disconnect. Its
transport failure returned from the long call and the CLI terminated the VM,
so no completion evidence survived. The detached path instead uploads
`scripts/colab_simulation.py` and a supervisor, then executes a short launch
cell. That cell starts the uploaded supervisor with an argv list,
`start_new_session=True`, and no dependency on the cell's `sys.argv` or
`__file__`. The supervisor directly starts and waits for the simulation child.

The remote layout is fixed:

```text
/content/vv-agent-campaigns/<job>/
  colab_simulation.py
  child.stdout.log
  child.stderr.log
  process-status.json
  reports/
    result.json
    sqlite-<start:010d>-<count:010d>.json
    postgres-<start:010d>-<count:010d>.json
```

The local job contains `state.json`, generated supervisor/launch/observer
files, and `evidence/`. Evidence mirrors the producer directory, including
`evidence/reports/result.json`. State records the target session, revisions,
counts, chunk sizes, exact argv, paths, script hash, and launch identity.
Observations are fresh command results and are not substituted from persisted
state.

## Run

The revision arguments must be full pushed Git hashes. Full defaults are
1,000,000 SQLite seeds, 10,000 PostgreSQL seeds, 100,000/1,000 chunk sizes,
seed zero, and eight workers.

```bash
uv run python scripts/colab_campaign.py run \
  --job-dir /tmp/vv-agent-colab/p1-20261114-01 \
  --session vvp1-20261114-01 \
  --vv-agent-revision <40-hex-source-revision> \
  --contract-revision ec154627d2153812d4b81e6863e5bdda0df64c5f
```

For a bounded disconnect test, choose the approved seed and chunk sizes
explicitly:

```bash
uv run python scripts/colab_campaign.py start \
  --job-dir /tmp/vv-agent-colab/five-seed \
  --session vvp1-five-seed \
  --vv-agent-revision <40-hex-source-revision> \
  --contract-revision ec154627d2153812d4b81e6863e5bdda0df64c5f \
  --sqlite-seeds 5 --postgres-seeds 5 \
  --sqlite-chunk-size 2 --postgres-chunk-size 2
```

`start`, `status`, `collect`, `wait`, and `run` all accept the same explicit
job directory. Existing nonempty job directories are rejected, so a second
`start` cannot silently launch another VM.

## Independent seed shards

For distributed P1 evidence, run one controller job per independent VM. The
optional `--seed` is the first seed for that job and defaults to `0`. Each
backend still runs its configured `--sqlite-seeds` or `--postgres-seeds`
count, so a shard covers `[seed, seed + count)` and its campaign result must
carry the same seed and count. Use the same source and contract revisions,
unique job/session directories, and non-overlapping ranges across VMs.

The controller validates a shard from its recorded seed through the configured
total. Combining those independently validated reports is a manual,
human-reviewed step; the controller does not schedule shards, merge their
evidence, or continue one shard from another. Each remote execution still has
a fresh, empty report directory: `colab_simulation.py` refuses resume or replay
when any report already exists. The unknown-job boundary is unchanged, so the
controller never restarts a job automatically.

`scripts/colab_merge.py` performs the separate, read-only aggregation step. It
accepts two or more independently validated shard job directories and writes a
single summary outside those job directories:

```bash
uv run python scripts/colab_merge.py \
  --job-dir /tmp/vv-agent-colab/shard-a \
  --job-dir /tmp/vv-agent-colab/shard-b \
  --output /tmp/vv-agent-colab/p1-merged.json
```

The merger does not start, stop, retry, replay, or modify any shard. It emits
`full_acceptance: true` only when the combined SQLite and PostgreSQL ranges
exactly equal the full P1 totals and every shard and aggregate gate passes.

## Observe And Recover

`status` performs a fresh session check and observer cell. A running process is
accepted only when its live PID, process group, and start ticks match the launch
identity. `wait` collects immutable parts while running and returns success only
after a fresh terminal observation has been followed by successful evidence
collection and VM release.

A non-timeout observation failure or a download failure returns unknown with
exit code `2`; it does not stop the VM, restart the job, or reuse an earlier
live observation. The job and local state remain for a later call:

```bash
uv run python scripts/colab_campaign.py status \
  --job-dir /tmp/vv-agent-colab/five-seed
uv run python scripts/colab_campaign.py collect \
  --job-dir /tmp/vv-agent-colab/five-seed
uv run python scripts/colab_campaign.py wait \
  --job-dir /tmp/vv-agent-colab/five-seed --interval 60
```

The only automatic retry is for a read-only `colab status` or observer CLI
`TimeoutExpired` during `wait`. `wait` prints an unknown retrying record with
the reason, waits the existing interval, and observes the same job again while
it remains below the observation limit. Each timed-out observation stays
unknown and nonterminal, and `status` or `collect` still makes one attempt and
exits `2`. This does not make an unknown execution result safe to repeat: the
retry never restarts, recreates, uploads, launches, or replays any campaign
part.

`wait --max-observations` can stop an automatic loop while the process is still
running. Every observation attempt, including a timed-out one, counts toward
this strict total. That result and a later unknown observation both exit `2`,
allowing a supervisor to resume the same job later. No cron is installed.

## Acceptance

Collected report hashes must match the observer manifest. Missing parts,
non-contiguous ranges, invalid counts, source/contract mismatch, dirty
checkouts, producer errors, duplicate or extra report names, and transfer hash
mismatches fail closed.

Every part must carry the same `code_digest` and
`contract_manifest_sha256`, and `in_memory_mutations` must be empty. The
per-backend scenario aggregate must equal the actual sum from every part. The
campaign coverage must equal the complete union of part coverage.

Both samples and full runs require PostgreSQL temporary role/database cleanup.
Full acceptance additionally requires 21 P1 scenarios, aggregate I1-I8/I10,
ordinary non-tmpfs SQLite storage evidence, exact full totals, and
`full_acceptance: true`. A partial run claiming full acceptance is rejected.
A partial shard with `full_acceptance: false` can be accepted when its complete
configured range is valid, but it does not satisfy the full P1 nightly gate by
itself.
Only after all evidence is retained and validated does `collect` stop an owned
session. A known terminal failure may then be released, but never reported as a
pass.

## Nightly entry

`scripts/colab_nightly.py` is a thin serial entry for one nightly full campaign.
It takes one nonblocking lock for the whole invocation. With no current job it
records the job name before starting the controller. While that job has no
valid terminal `released.json`, every later invocation only resumes the same
job with `wait`; even a successful wait does not start another campaign in that
invocation. A later invocation may start a new full campaign only after the
controller has written a terminal release with `acceptance.accepted`.
`terminal_failed` with `accepted: false` is also a released terminal outcome,
so it allows a new campaign on a later invocation without a same-invocation
retry.

This change does not install a scheduler. Install cron only after the first full
campaign has completed and released. Keep the existing user crontab and add one
entry such as the following, replacing the paths and full public Git hashes.
Include the installed `colab` executable directory in cron's `PATH`; an absolute
Python path alone does not make the Colab CLI available.

```cron
17 2 * * * cd /absolute/path/to/vv-agent && PATH=/absolute/path/to/colab-bin:/usr/local/bin:/usr/bin:/bin /absolute/path/to/venv/bin/python scripts/colab_nightly.py --state-dir /absolute/path/to/colab-nightly-state --vv-agent-revision REPLACE_WITH_FULL_SOURCE_REVISION --contract-revision REPLACE_WITH_FULL_CONTRACT_REVISION
```

The wrapper applies the approved nightly chunk sizes (`10000` SQLite and
`1000` PostgreSQL) and leaves the 1,000,000/10,000 totals and eight workers at
the controller defaults. Controller output is retained in a unique file under
`<state-dir>/logs`, and the entry prints that path. Its nonzero exit status is
the controller's status, including `2` for unknown.

To resume manually, rerun the entry with the same state directory. An
unreleased current job is passed only to `wait`, so changed revision arguments
cannot replace it:

```bash
uv run python scripts/colab_nightly.py \
  --state-dir /absolute/path/to/colab-nightly-state \
  --vv-agent-revision REPLACE_WITH_FULL_SOURCE_REVISION \
  --contract-revision REPLACE_WITH_FULL_CONTRACT_REVISION
```
