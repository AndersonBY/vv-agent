# Colab nightly simulation

`scripts/colab_simulation.py` is the P1 nightly entry point. It is uploaded by
the installed Colab CLI, clones the public `AndersonBY/vv-agent` and
`AndersonBY/vv-agent-contract` repositories into a private `/tmp` directory at
the two full Git revisions supplied on the command line, verifies both detached
checkouts are clean, installs or uses `uv`, installs managed Python 3.12, and
runs `uv sync --frozen --dev`.

The existing `scripts/session_simulation.py` runs sequentially for SQLite and
PostgreSQL with at most eight workers. SQLite defaults to 1,000,000 seeds and
PostgreSQL to 10,000. Scenario and pending metadata are loaded once from the
selected source in a child interpreter; the entry point does not import the SDK
or duplicate the scenario list. Child output and its complete JSON report are
streamed to stdout after replacing only that run's exact PostgreSQL password and
DSN. Other seed and digest text is unchanged.

Small counts are sample runs and cannot produce `full_acceptance: true`. Sample
validation checks completed seed counts, actual backend/code/contract identity,
the absence of a report error, child exit status, and source-defined pending
I9/I11. It does not require all scenarios or invariants. Full campaigns also
require every source scenario, I1-I8/I10, verified SQLite `/tmp` storage on a
non-tmpfs filesystem, and exact aggregate counts. PostgreSQL correctly reports
`sqlite_temporary_storage: {verified: false, observations: []}` and is not
required to provide SQLite temporary-file observations.

PostgreSQL uses the Ubuntu package default and records its actual major and
server version. The script creates a uniquely named temporary role and database,
verifies TCP/password authentication, and drops every database owned by that
role before dropping the role. Interrupted or unknown results are not passes.
The entry point does not split, resume, or retry a campaign.

## Run

Both revisions must already be pushed to the public repositories. Pass full
hashes explicitly; do not resolve `main` at run time. The pair below is the
current review example, and both values must be replaced together after runtime
or canonical-contract changes.

```bash
VV_AGENT_REV=8c4db51c70f581d3cec2932c0c096a8bee05ec23
CONTRACT_REV=ec154627d2153812d4b81e6863e5bdda0df64c5f
colab run --high-mem --timeout 0 scripts/colab_simulation.py \
  --vv-agent-revision "$VV_AGENT_REV" \
  --contract-revision "$CONTRACT_REV"
```

Use an already logged-in Colab CLI and retain the complete local log. Each run
gets a unique file, so a later failure does not overwrite earlier evidence:

```bash
set -o pipefail
mkdir -p "$HOME/.local/state/vv-agent"
log=$(mktemp "$HOME/.local/state/vv-agent/colab-nightly-XXXXXXXX.log")
colab run --high-mem --timeout 0 scripts/colab_simulation.py \
  --vv-agent-revision "$VV_AGENT_REV" \
  --contract-revision "$CONTRACT_REV" 2>&1 | tee "$log"
```

The same explicit revisions can be used for a small remote setup check:

```bash
colab run --high-mem --timeout 0 scripts/colab_simulation.py \
  --vv-agent-revision "$VV_AGENT_REV" \
  --contract-revision "$CONTRACT_REV" \
  --sqlite-seeds 3 --postgres-seeds 3
```

`--timeout 0` disables the Colab CLI inactivity timeout. It does not extend the
VM lifetime or guarantee quota. A 1,000,000-seed campaign can take hours, may
exceed 12 hours, and consumes the account's Colab compute units; runtime and
cost depend on the current allocation.

## Nightly serialization

This is a cron example only; this change does not install it. `flock -n`
prevents overlap, `pipefail` propagates the Colab failure, and `mktemp` gives
each attempt a unique log. Replace the two revisions and repository path before
use.

```cron
0 2 * * * /usr/bin/flock -n /tmp/vv-agent-colab-nightly.lock /bin/bash -c 'set -o pipefail; VV_AGENT_REV=8c4db51c70f581d3cec2932c0c096a8bee05ec23; CONTRACT_REV=ec154627d2153812d4b81e6863e5bdda0df64c5f; mkdir -p "$HOME/.local/state/vv-agent"; log=$(mktemp "$HOME/.local/state/vv-agent/colab-nightly-XXXXXXXX.log"); cd /home/makerbi/vectorvein/tmp/wt-vva-trusted-p1 || exit 1; /home/makerbi/.local/bin/colab run --high-mem --timeout 0 scripts/colab_simulation.py --vv-agent-revision "$VV_AGENT_REV" --contract-revision "$CONTRACT_REV" 2>&1 | tee "$log"'
```

The final wrapper must report `full_acceptance: true`, both child processes
must exit zero, and both streamed reports must pass the checks above. Inspect
the preserved seed/replay report after any failure before considering another
run.

The current Colab runtime limit, high-memory entitlement, package network
access, and ability to keep one VM alive for the complete million-seed campaign
have not been verified by this local change. Those conditions require the
supervisor's small remote run and the first full nightly execution.
