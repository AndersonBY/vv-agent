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

## Observe And Recover

`status` performs a fresh session check and observer cell. A running process is
accepted only when its live PID, process group, and start ticks match the launch
identity. `wait` collects immutable parts while running and returns success only
after a fresh terminal observation has been followed by successful evidence
collection and VM release.

An observation failure or a download failure returns unknown with exit code
`2`; it does not stop the VM, restart the job, or reuse an earlier live
observation. The job and local state remain for a later call:

```bash
uv run python scripts/colab_campaign.py status \
  --job-dir /tmp/vv-agent-colab/five-seed
uv run python scripts/colab_campaign.py collect \
  --job-dir /tmp/vv-agent-colab/five-seed
uv run python scripts/colab_campaign.py wait \
  --job-dir /tmp/vv-agent-colab/five-seed --interval 60
```

`wait --max-observations` can stop an automatic loop while the process is still
running. That result and a later unknown observation both exit `2`, allowing a
supervisor to resume the same job later. No cron is installed.

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
Only after all evidence is retained and validated does `collect` stop an owned
session. A known terminal failure may then be released, but never reported as a
pass.
