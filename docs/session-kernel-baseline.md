# Session kernel performance baseline

Measured on the final Python tree on 2026-10-09, contract 24.0.1 / public API v8.
These are local scripted costs, not provider latency or production throughput.

## Machine and method

WSL2 x86_64, Linux 6.18.40.1-microsoft-standard-WSL2; Intel Core i9-13900H,
20 logical CPUs, 23.47 GiB guest RAM; Python 3.12.12 and local PostgreSQL 18.6
through its Unix socket. Harness processes were pinned to CPUs 8,9; PostgreSQL
was not pinned. OS/PG buffers stayed warm. CPU frequency and other host activity
were not controlled. Measurements ran after the full pytest gate.

Short-run workload: six unchanged scenarios, 10 warmups and 200 samples per path,
nearest-rank p95. Five scenarios include schema/session admission and cleanup.
The ten-turn scenario is one retained session with ten successive turns;
children include admission, independent execution and terminal delivery;
start/cancel uses a provider-entry barrier. App Server measures turn/start through
turn/completed with its real adapter; construction, initialize, thread/start and
cleanup are outside that measured interval.
The comparison uses the `kernel` path on both trees; the base-only historical
`runner` field is not an acceptance metric. Absolute values take the median of three
independent p50/p95 estimates. They are not pooled-sample percentiles. All runs
reported zero leaked kernel threads.

Reproduce each side from its repository root with the same managed Python:

```bash
PYTHONPATH="$TREE/src" PYTHONDONTWRITEBYTECODE=1 taskset -c 8,9 "$PYTHON" \
  scripts/session_kernel_overhead.py --runs 200 --warmup 10 --output /tmp/overhead.json
```

Set TREE to the selected checkout and PYTHON to the candidate managed interpreter.
The comparison alternates pre-F3 main 9e43dbf and the current tree three times,
using the same workload and CPUs. Each scenario accepts the median of the three
current/base kernel-p95 ratios only when it is <=1.10. The table uses the second
complete series on identical source: an earlier series exceeded the ratio limit
in three scenarios. Diagnostic step/commit/append counts matched in those cases,
and all six scenario medians passed in the subsequent full repeat. Both series
and the profile remain in external change evidence; no individual pairs were
selected across series. Current absolute values:

| Scenario | p50 ms | p95 ms | Median current/base p95 |
| --- | --- | --- | --- |
| no_tool | 12.253 | 13.212 | 1.0049 |
| two_tools | 20.879 | 22.280 | 0.9567 |
| ten_turns | 83.566 | 88.694 | 1.0009 |
| start_cancel | 14.311 | 15.250 | 1.0031 |
| children | 47.016 | 49.568 | 0.9710 |
| app_server_turn | 30.504 | 33.235 | 1.0052 |

## M6 capacity

```bash
UV_CACHE_DIR=/tmp/f3b-uv-cache PYTHONDONTWRITEBYTECODE=1 taskset -c 8,9 uv run python \
  scripts/session_kernel_benchmark.py --assert-capacity --output /tmp/capacity.json
```

The default workload uses 100/1000/5000/20000-history sizes, 1000/10000 sessions,
seven measurement samples and three cold samples. Completed tool triples retain
bounded 1 KiB receipts. Cold drive reconstructs a connection/runtime and commits
the terminal; it does not flush OS/PG caches. Steady append advances successive
heads. Pagination scans the whole catalog with 100-item pages. Disposable PG
databases are removed after the run.

| Records | Logical bytes | Append p50/max ms | Steady append p50/max ms | Fold p50/max ms | Read/fold p50/max ms | Cold drive p50/max ms |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 99967 | 18.637/21.747 | 6.882/7.662 | 0.689/0.994 | 27.729/40.448 | 63.208/67.834 |
| 1000 | 864931 | 115.638/137.090 | 8.279/9.551 | 7.909/8.827 | 114.482/182.917 | 154.675/238.808 |
| 5000 | 4274962 | 538.312/574.711 | 10.553/19.435 | 44.084/163.915 | 589.969/705.899 | 708.706/713.145 |
| 20000 | 17091594 | 2230.786/2415.351 | 16.428/21.823 | 168.915/376.216 | 2304.174/2456.581 | 2344.897/2619.119 |

| Sessions | Runnable items | Page size | First page p50/max ms | Late page p50/max ms | Full scan p50/max ms |
| --- | --- | --- | --- | --- | --- |
| 1000 | 1500 | 100 | 2.887/4.233 | 1.607/2.583 | 18.583/19.828 |
| 10000 | 15000 | 100 | 1.547/2.660 | 2.621/8.468 | 242.681/248.907 |

All cases passed: zero cold-drive lease failures, zero provider redispatch;
steady append median <=50 ms; cold drive max <=one millisecond per record for
histories >=2000; full catalog scan max <=1000 ms. Lease TTL was 15000 ms and
heartbeat interval 0.25 s. The fixture-reset transaction was excluded from append
measurements. Raw samples are retained with the external change evidence.
