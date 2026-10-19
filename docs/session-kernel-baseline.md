# Session kernel performance baseline

These numbers were measured on 2026-10-10 on package 0.22.0 (contract 24.0.1,
public API v8). Later releases have not been re-measured.

They are local costs with scripted models and providers. They do not include
provider latency and do not predict production throughput.

## Machine and method

- WSL2 x86_64, Linux 6.18.40.1-microsoft-standard-WSL2.
- Intel Core i9-13900H, 20 logical CPUs, 23.47 GiB guest RAM.
- Python 3.12.12 and local PostgreSQL 18.6 over its Unix socket.
- The capacity and short-run harnesses were pinned to CPUs 8 and 9. PostgreSQL
  and the soak runs were not pinned.
- OS and PostgreSQL buffers stayed warm. CPU frequency and other host activity
  were not controlled.
- Heavy jobs ran one after another. Performance was measured after the full
  PostgreSQL test run.

## Short-run comparison

`scripts/session_kernel_overhead.py` times six scenarios, each with 10 warmups
and 200 samples per path, and reports the nearest-rank p95:

- `no_tool`, `two_tools`, `ten_turns`, `start_cancel` and `children` include
  admission and cleanup.
- `ten_turns` runs ten turns in one retained session.
- `children` covers child admission, independent execution and delivery of the
  final result.
- `start_cancel` cancels at a barrier on entry to the provider.
- `app_server_turn` measures from `turn/start` to `turn/completed`. It excludes
  construction, `initialize`, `thread/start` and cleanup.

```bash
PYTHONPATH="$TREE/src" PYTHONDONTWRITEBYTECODE=1 taskset -c 8,9 "$PYTHON" \
  scripts/session_kernel_overhead.py --runs 200 --warmup 10 --output /tmp/overhead.json
```

The comparison ran the base commit `9e43dbf` (an earlier commit on the way to
0.22.0) and the 0.22.0 tree three times each, alternating, with the same
interpreter and CPUs. Both trees report the `kernel` field. The base tree also
reports an older `runner` field, which the comparison ignores.

The absolute p50 and p95 values are medians of three independent estimates,
not percentiles over pooled samples. The acceptance metric is the median of
three paired current-to-base p95 ratios, and it had to stay at or below 1.10.
The last column shows the ratio of the median p95 values for reference; it also
stayed at or below 1.10. All six processes exited with status zero and reported
no leaked kernel threads.

| Scenario | Base p95 ms | Current p50/p95 ms | Median paired p95 ratio | Ratio of median p95 |
| --- | ---: | ---: | ---: | ---: |
| no_tool | 12.517 | 12.017/12.665 | 1.0118 | 1.0118 |
| two_tools | 21.768 | 21.378/23.216 | 1.0202 | 1.0665 |
| ten_turns | 91.913 | 84.183/90.598 | 0.9927 | 0.9857 |
| start_cancel | 15.473 | 14.472/15.634 | 1.0447 | 1.0104 |
| children | 49.549 | 46.345/49.088 | 1.0158 | 0.9907 |
| app_server_turn | 32.636 | 29.496/31.706 | 0.9579 | 0.9715 |

## Capacity benchmark

```bash
PYTHONDONTWRITEBYTECODE=1 taskset -c 8,9 uv run python \
  scripts/session_kernel_benchmark.py --assert-capacity --output /tmp/capacity.json
```

The default workload:

- uses histories of 100, 1000, 5000 and 20000 records, and catalogs of 1000
  and 10000 sessions;
- takes seven measurement samples and three cold samples;
- stores completed tool triples with bounded 1 KiB receipts.

"Cold drive" opens a new connection, rebuilds the runtime and commits the final
record. It does not flush OS or PostgreSQL caches. "Steady append" appends to
successive log heads. "Pagination" scans the whole catalog in pages of 100.
The run creates disposable PostgreSQL databases and removes them afterwards.
The transaction that resets fixtures is not timed.

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

`--assert-capacity` checks these thresholds, and all of them passed:

- no lease failures during cold drives and no provider call dispatched twice;
- steady append median at most 50 ms;
- cold drive maximum at most 1 ms per record for histories of 2000 records or more;
- full catalog scan maximum at most 1000 ms.

The benchmark uses a lease TTL of 15000 ms and a heartbeat interval of 0.25 s.

## Seeded fault repetition

```bash
uv run python scripts/session_kernel_soak.py --store sqlite --seed 97 --sessions 1024 --assert
VV_AGENT_TEST_POSTGRES_DSN='dbname=postgres' uv run python \
  scripts/session_kernel_soak.py --store postgres --seed 97 --sessions 640 --assert
```

Each store ran seeds 42, 97, 2026, 314159 and 8675309. Every group of eight
parent sessions covers tools, approval, a child, cancellation, delayed inbox
items, polls of an `Accepted` provider call, and workers abandoned before and
after the receipt commit. Each group also includes one child session.

A shared test transport delivers wakes at least once and duplicates, reorders
and drops them. Database clocks are injected, and lease TTL recovery uses the
real `drive` and `tick`. An abandoned worker stops its heartbeat and never
releases its lease. Real process kills are covered by the recovery test suite
instead. Models, providers and the approval broker are scripted.

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

All ten runs passed the `--assert` checks:

- every session reached its expected final state;
- sequence numbers, record identities and dispatch identities were unique;
- folding the raw log and inbox gave the same state;
- model retries stayed within the allowed count;
- each tool and provider call was submitted once;
- the queue drained;
- provider polls stayed within `1 + floor(T/P)` for wait time `T` and poll
  interval `P`.

In total the runs finished 9360 sessions with 200720 records, 2080 abandoned
workers, 1040 `Accepted` submits and 3599 provider polls. Each SQLite run took
174 to 180 s and each PostgreSQL run 245 to 333 s. The last two PostgreSQL runs
went 29 and 33 s over the rough five-minute target; their full results are kept
above. The normal pytest wrapper (`tests/session/test_soak.py`) stays small,
with eight parents per seed and store.

## Built distribution installation

```bash
VV_AGENT_TEST_POSTGRES_DSN='dbname=postgres' uv run python scripts/install_matrix.py
```

The script built the 0.22.0 wheel and installed it into four fresh virtual
environments outside the source tree, without lock constraints. In each one it
resolved the 239 API capabilities and the public API v8 members, imported the
built-in tool schemas, and completed real Runner turns. Both PostgreSQL
environments ran against local PostgreSQL; none skipped.

| Venv | API and import | SQLite memory / file | PostgreSQL | Removed `redis` and `celery` extras |
| --- | --- | --- | --- | --- |
| base | passed | passed / passed | not applicable | add no packages |
| postgres | passed | passed / passed | passed | add no packages |
| s3 | passed | passed / passed | not applicable | add no packages |
| postgres,s3 | passed | passed / passed | passed | add no packages |

The base environment installed and imported none of psycopg, boto3, Redis or
Celery. Each environment installed exactly the optional packages its extras
asked for. The sdist had 174 members and no tests, fixtures, local key files or
temporary files. The 152 package files in the wheel matched the source bytes,
including package data. The CI install-matrix job uses Python 3.12 and a
PostgreSQL service.
