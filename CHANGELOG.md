# Changelog


## 0.33.2 — 2026-11-06

Provider HTTP 4xx responses other than 408 now fail definitively instead of
creating an unknown model operation and duplicate-cost warning. Content
rejections stop after one call. 429 and adapter-confirmed temporary rejections
keep failed receipts and retry with durable backoff within the existing limit;
exhaustion ends with `model_request_rejected`. Hosts and users receive the safe
HTTP status, provider code/type and a generic explanation; arbitrary provider prose is omitted.
All physical HTTP sends, including vendor retries, must be observed as 4xx
other than 408 before automatic rejection. Unobservable host adapters need an
explicit assertion; uncertainty anywhere in the exception graph vetoes it.
Output-repair diagnostics use the same safe constructor. Globally stop all
worker/API drivers before upgrading both machines; new-driven sessions cannot
return to 0.33.1. Rollback requires version-owned data isolation and the read-only
`scripts/check_model_rejection_rollback.py` precheck.

A 4xx counts only after its entire raw response body reaches EOF and decodes
successfully; unread/closed bodies and decoder failures remain Unknown, including
413. Every non-408 4xx is pre-read within min(request read timeout, 10 seconds)
and 1 MiB of raw/decoded bytes; exceeded bounds stay Unknown. Buffered bytes and
the remaining stream, or the original read error, reach the vendor on consumption.
Status, headers, vendor retry decisions/counts and backoff remain unchanged,
including Retry-After, x-should-retry and default 409 retries. Each optional
vendor's import/export failure is isolated and is retried on the next lookup.
Pre-reading runs synchronously without observation threads. Before raw iteration,
the request timeout extension is copied and read capped at the observation
deadline. Extra network wait is at most the deadline plus one recv timeout
(at most 20 seconds); replay keeps that shortened per-recv timeout.

Native OpenAI/Anthropic connection and timeout exceptions veto both rejection
and context-length recovery even without an inner HTTP error. The observed
clients retain vendor connection limits and Anthropic TCP keepalive, preserve
proxy/timeout priority, and attribute concurrent failures to their own requests.
Unindexed failures remain uncertain for the whole invocation. Callback exit
explicitly closes streams. `httpx2>=2.7,<3` is now a direct dependency; no other
dependency versions changed. Existing session projection fixtures now reproduce
the safe output-repair diagnostic, alongside the rejection producer cases.

Interrupted streams, 408, 5xx, timeouts, connection loss and cancellation remain
Unknown. A rejected compaction summary keeps history and lets the primary call
continue. Custom adapters can raise `vv_agent.llm.errors.ModelRequestRejected`.
Oversized inputs, including real HTTP 400 context errors and HTTP 413, keep the
existing prompt-too-long compaction/retry path before rejection classification.
Content rejection stops without endpoint failover. 401/403/404 immediately try
the next frozen endpoint within the existing limit, but a single endpoint is
called only once. 429 retains backoff and one single-endpoint retry. Explicit
adapter retryability wins over these defaults; exhaustion reports the last safe
rejection. Tool-name string limits remain rejected.

Requires contract 35.0.0. This SDK patch fixes erroneous outcome classification;
the contract major follows its observable-behavior policy, as in SDK 0.33.1.
No store, protocol or frozen-turn boundary changes. Historical Unknown records
are not rewritten. See [release notes](docs/releases/0.33.2.md) and
[upgrade guide](docs/upgrading.md#0332--2026-11-06).

## 0.33.1 — 2026-11-05

Fix the 0.33.0 search CPU budget design error that stopped ordinary large-file
regex/literal scans before late matches. Each engine invocation now has its own
250 ms ReDoS timeout; enumeration and matching share a 10 s thread CPU limit.
Earlier hits and stop coordinates survive either limit, with distinct messages.
Local, Memory, S3, artifacts, globs/find_files and child exclusions use one rule.
Existing window, page, head and file limits remain unchanged.

Requires contract 34.0.0 under its observable behavior version policy. This is
handler-only: tool schemas, frozen definitions and all store/wire versions stay
unchanged; 0.33.0 turns need no special transition. See the
[release notes](docs/releases/0.33.1.md) and [upgrade guide](docs/upgrading.md).

## 0.33.0 — 2026-11-04

Requires contract 33.0.0. read_file adds character positioning, removes max_chars
and uses fixed 12,000-character pages. Search reads long lines with bounded
windows and interrupts slow matching. Artifact authorization now preserves full,
type-separated task and session identities.

Old executable turns close at the definition guard after the read_file schema
change. Old nonliteral task directories and lossy compaction paths lose model
read authorization. Terminal output retention failures preserve confirmed command
results with retention unavailable and no recovery pointer. Guard closure keeps
unconfirmed provider waits when the turn is not marked cancelled; late results
are audit-only. User cancel/abort retains its existing behavior.

No record, inbox, SQL or App Server schema bump. Read the
[release notes](docs/releases/0.33.0.md) and [upgrade guide](docs/upgrading.md)
before switching workers. Earlier release notes are in [docs/releases/](docs/releases/).
