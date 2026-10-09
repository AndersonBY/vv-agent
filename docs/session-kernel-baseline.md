# Session kernel performance baseline

Measured on the F4b Python tree on 2026-10-10, contract 24.0.1 / public API v8.
These are local scripted costs, not provider latency or production throughput.
These measurements are retained; the subsequent local-handle return correction
does not rerun performance, release-scale soak or the installation matrix.

## Machine and method

WSL2 x86_64, Linux 6.18.40.1-microsoft-standard-WSL2; Intel Core i9-13900H,
20 logical CPUs, 23.47 GiB guest RAM; Python 3.12.12 and local PostgreSQL 18.6
through its Unix socket. Capacity and short-run harnesses used CPUs 8,9;
PostgreSQL and soak were not pinned. OS/PG buffers stayed warm. CPU frequency
and other host activity were not controlled. Heavy gates ran sequentially,
with performance measured after the full PostgreSQL pytest gate.

## Short-run comparison

Six unchanged scenarios use 10 warmups and 200 samples per path, nearest-rank
p95. Five include admission and cleanup. Ten turns share one retained session;
children include admission, independent execution and terminal delivery;
start/cancel uses a provider-entry barrier. App Server measures turn/start through
turn/completed; construction, initialize, thread/start and cleanup are excluded.

The comparison alternates base 9e43dbf and the current tree three times, using
the same candidate interpreter and CPUs. It uses the `kernel` field on both
trees; the base-only historical `runner` field is not an acceptance metric.
Absolute p50/p95 values are medians of three independent estimates, not pooled
percentiles. Acceptance takes the median of three paired current/base p95 ratios.
The ratio of the absolute median p95 values is also shown; both stay <=1.10.
All six processes exited zero and reported zero leaked kernel threads.

```bash
PYTHONPATH="$TREE/src" PYTHONDONTWRITEBYTECODE=1 taskset -c 8,9 "$PYTHON" \
  scripts/session_kernel_overhead.py --runs 200 --warmup 10 --output /tmp/overhead.json
```

| Scenario | Base p95 ms | Current p50/p95 ms | Median paired p95 ratio | Ratio of median p95 |
| --- | ---: | ---: | ---: | ---: |
| no_tool | 12.517 | 12.017/12.665 | 1.0118 | 1.0118 |
| two_tools | 21.768 | 21.378/23.216 | 1.0202 | 1.0665 |
| ten_turns | 91.913 | 84.183/90.598 | 0.9927 | 0.9857 |
| start_cancel | 15.473 | 14.472/15.634 | 1.0447 | 1.0104 |
| children | 49.549 | 46.345/49.088 | 1.0158 | 0.9907 |
| app_server_turn | 32.636 | 29.496/31.706 | 0.9579 | 0.9715 |

## M6 capacity

```bash
PYTHONDONTWRITEBYTECODE=1 taskset -c 8,9 uv run python \
  scripts/session_kernel_benchmark.py --assert-capacity --output /tmp/capacity.json
```

The default workload uses 100/1000/5000/20000-history sizes, 1000/10000 sessions,
seven measurement samples and three cold samples. Completed tool triples retain
bounded 1 KiB receipts. Cold drive reconstructs a connection/runtime and commits
the terminal; it does not flush OS/PG caches. Steady append advances successive
heads. Pagination scans the whole catalog with 100-item pages. Disposable PG
databases are removed after the run. The fixture-reset transaction is excluded.

| Records | Logical bytes | Append p50/max ms | Steady append p50/max ms | Fold p50/max ms | Read/fold p50/max ms | Cold drive p50/max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 99967 | 13.584/15.045 | 3.620/5.094 | 0.666/0.960 | 18.674/19.666 | 41.106/41.349 |
| 1000 | 864931 | 96.996/98.806 | 4.221/5.698 | 6.642/7.134 | 97.632/155.602 | 121.539/179.224 |
| 5000 | 4274962 | 407.591/473.638 | 6.160/12.042 | 35.091/115.333 | 495.395/523.476 | 604.774/613.932 |
| 20000 | 17091594 | 1875.347/1951.981 | 12.749/17.278 | 147.879/311.702 | 1826.383/2014.412 | 2093.571/2300.601 |

| Sessions | Runnable items | Page size | First page p50/max ms | Late page p50/max ms | Full scan p50/max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1500 | 100 | 1.299/2.460 | 1.173/1.662 | 9.376/10.672 |
| 10000 | 15000 | 100 | 1.277/3.435 | 2.090/7.509 | 205.892/212.080 |

All checks passed: zero cold-drive lease failures, zero provider redispatch;
steady append median <=50 ms; cold drive max <=one millisecond per record for
histories >=2000; full catalog scan max <=1000 ms. Lease TTL is 15000 ms and
heartbeat interval 0.25 s.

## Seeded fault repetition

```bash
uv run python scripts/session_kernel_soak.py --store sqlite --seed 97 --sessions 1024 --assert
VV_AGENT_TEST_POSTGRES_DSN='dbname=postgres' uv run python \
  scripts/session_kernel_soak.py --store postgres --seed 97 --sessions 640 --assert
```

Each store ran seeds 42, 97, 2026, 314159 and 8675309. Every eight parent
sessions cover tools, approval, child, cancellation, delayed inbox, Accepted
polls and worker abandonment before/after receipt commit, plus one child.
A shared at-least-once test transport duplicates, reorders and drops wakes.
Database clocks are injected; lease TTL recovery uses real drive/tick.
Abandonment stops heartbeat and omits release; real process-kill coverage is in
the full recovery suite. Models/providers/broker are scripted.

| Store | Seed | Parents + children | Seconds | Records | Provider submits/polls | Abandoned workers | Dropped wakes | Final queue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqlite | 42 | 1024 + 128 | 177.060 | 24704 | 128/422 | 256 | 1054 | 0 |
| sqlite | 97 | 1024 + 128 | 174.139 | 24704 | 128/462 | 256 | 1081 | 0 |
| sqlite | 2026 | 1024 + 128 | 179.813 | 24704 | 128/445 | 256 | 1083 | 0 |
| sqlite | 314159 | 1024 + 128 | 177.383 | 24704 | 128/432 | 256 | 1059 | 0 |
| sqlite | 8675309 | 1024 + 128 | 179.863 | 24704 | 128/450 | 256 | 1084 | 0 |
| postgres | 42 | 640 + 80 | 250.118 | 15440 | 80/259 | 160 | 657 | 0 |
| postgres | 97 | 640 + 80 | 244.501 | 15440 | 80/290 | 160 | 679 | 0 |
| postgres | 2026 | 640 + 80 | 295.900 | 15440 | 80/287 | 160 | 659 | 0 |
| postgres | 314159 | 640 + 80 | 332.756 | 15440 | 80/268 | 160 | 654 | 0 |
| postgres | 8675309 | 640 + 80 | 328.908 | 15440 | 80/284 | 160 | 695 | 0 |

All ten runs passed expected terminal, unique sequence/record/dispatch identity,
raw log/inbox fold equality, allowed model retry, tool/provider submit uniqueness,
queue-drain and unchanged `1 + floor(T/P)`/poll-interval assertions.
Total: 9360 terminal sessions, 200720 records, 2080 abandoned workers,
1040 Accepted submits and 3599 provider polls. Each SQLite run took 174–180 s;
PG took 245–333 s. The last two PG runs exceeded the approximate five-minute
target by 29/33 seconds; their complete results are retained.
The normal pytest wrapper stays small: eight parents per seed/store.

## Built distribution installation

```bash
VV_AGENT_TEST_POSTGRES_DSN='dbname=postgres' uv run python scripts/install_matrix.py
```

Four fresh venvs outside the source tree installed the built 0.22.0 wheel without
lock constraints. Each resolved 239 API capabilities and the public v8 members,
imported builtin schemas, and completed real Runner turns. Both PG legs ran
against local PostgreSQL, with no PG skip.

| Venv | API/import | SQLite memory/file | PostgreSQL | Removed extras |
| --- | --- | --- | --- | --- |
| base | passed | passed / passed | not applicable | neither adds packages |
| postgres | passed | passed / passed | passed | neither adds packages |
| s3 | passed | passed / passed | not applicable | neither adds packages |
| postgres,s3 | passed | passed / passed | passed | neither adds packages |

Base installed/imported none of psycopg, boto3, Redis or Celery. Installed
optional packages matched their requested extras. The sdist contained
174 members without tests/fixtures, local key files or temporary junk;
the wheel's 152 package files matched source bytes, including package data.
The CI install-matrix job uses Python 3.12 and a PostgreSQL service.

## Repository validation

The F4b full gate passed 3067 tests with seven opt-in/environment skips and no PG
skips; 1572 are session tests. Public API v8 has 182 root exports, extra/missing
zero, and 281 mapped members. All 45 generated fixtures match the immutable
24.0.1 snapshot. Format, lint, type check, snapshot, capacity, soak, install,
stdio v2 smoke, interleaved performance and diff checks are the release-change
evidence. Central verification still names Python 53bdf32; see the
[release notes](releases/0.22.0.md) for the exact ancestry gate and release-record boundary.

Local handles return when an Accepted provider wait parks the turn. A future
`poll_at_ms` means zero queries before return and no waiting thread. A later
host tick/supervisor drive, wake or `Runner.resume` at or after the due time
queries the provider; an authenticated `provider_result` can continue the turn
immediately. `test_local_accepted_returns_parked_until_due_drive` verifies the
return and tick/resume paths on SQLite file and PostgreSQL with an injected clock.
