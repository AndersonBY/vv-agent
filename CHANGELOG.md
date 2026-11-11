# Changelog

## 0.35.0 — 2026-11-08

- Yield callbacks observe read-only host state; verified child terminal results
  and references are supplied, files stay on the host delivery path.
- Provider acceptance recovery preserves and authenticates the original yield
  declaration; later conflicting attachments/queries are rejected.
- Local parked drivers stop without progress. Long receipt candidates include
  content SHA-256 in their immutable identity; structured yield bodies validate.

Hosts can let foreground child and Accepted provider waits yield to new input.
The kernel commits completed observation results and steering together. Children
keep running and later notify as background work; provider handles close without
cancellation. Undeclared waits retain their bytes and behavior. New optional
record fields reject 0.34.1 readers; no SQL migration or dependency changes. See
[release notes](docs/releases/0.35.0.md) and [host usage](docs/session-kernel.md#yielding-a-wait-to-new-input).

## 0.34.1 — 2026-11-07

0.34.0 was tagged but its release workflow failed before publication to PyPI.
0.34.1 includes all of 0.34.0, plus two test corrections: the retention test
waits for the real command to finish, and pathological glob tests use inputs
that exceed the per-call regex CPU budget. Product behavior is unchanged.
See [release notes](docs/releases/0.34.1.md) and the
[0.34.0 upgrade notes](docs/upgrading.md#0340--2026-11-07).

## 0.34.0 — 2026-11-07

New sessions protect inputs until a primary model response commits to history.
Compaction and host hooks retain the same suffix through replay and recovery.
Legacy sessions without the creation declaration retain their frozen behavior.
New declarations are rejected by 0.33.3; children and App Server forks inherit
the choice. Oversized protected inputs fail with an explicit capacity reason.
The vv-llm dependency is now >=0.10.1. See [release notes](docs/releases/0.34.0.md).

Model completion now uses vv-llm >=0.10.1. New sessions declare
`attributes.seed.response_rejection=true`. A content-filter result, or refusal
finish reason without answer text, now ends as `model_request_rejected` with
HTTP 200, safe provider code/type and `retryable=false`. A streamed answer may
already be visible: finish or replace that temporary text when ModelCallFailed
or RunFailed arrives. Rejected partial text is not retained in model history.

OpenAI refusal text is an ordinary answer when the reason is not content_filter.
If content is empty, that text becomes the assistant body, so hosts should render
it as they render any assistant answer. Final evidence is available in ordinary
model receipts at `raw.completion_metadata`. Missing-termination EOF follows
existing Unknown/retry behavior; degraded observation is recorded as unknown
termination and keeps existing completion behavior.

No record, inbox or SQL schema changes. Existing sessions without the response
policy retain 0.33.3 filtering behavior; EOF and refusal-text improvements use
representations 0.33.3 can read and continue. Both seed policies independently
default on only at first creation, inherit on child/fork (absence becomes false),
and preserve creation bytes on transaction replay. 0.33.3 rejects either declared
field, including false. Keep declared sessions and dependent parent/child work on
the new runtime when rolling back; do not strip creation fields to bypass this gate.



## 0.33.3 — 2026-11-06

Fix quota and time-window errors being mistaken for oversized input. Generic
`too many tokens` / `request too large` wording now checks word-bounded quota
signals in the same text source. HTTP 429 needs an explicit context marker;
other 429s retain rejection retry/backoff without compaction. Non-429 explicit
context markers keep priority over quota; `context window exceeded` and
`model context window is` now trigger recovery. Adapter rejection flags share
the same classifier. HTTP 413, content-string param rules, HTTP observations and
uncertainty vetoes keep their behavior. Dependencies and store schemas are unchanged.

The canonical behavior is contract 36.0.0 (a major under its observable-behavior
policy). Local verification selects that contract checkout; the published
35.0.0 lock/snapshot stays until immutable release synchronization. See
[release notes](docs/releases/0.33.3.md) and [upgrading](docs/upgrading.md).

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
