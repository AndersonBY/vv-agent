# Upgrading a host

This guide is for hosts that keep sessions in a durable store (SQLite file or
PostgreSQL) and want to move to a newer vv-agent release. If your runs only use
the default in-memory store, upgrading is an ordinary dependency bump; read the
[version notes](#version-notes) for API changes and skip the rest.

## Session data and upgrades

vv-agent reads exactly one version of each stored shape. There are no decoders
for older versions and no in-place migrations. When a release bumps the record
or inbox schema, the new runtime rejects rows written by the old one with
`RecordError`. That covers finished sessions too, not only running ones.

This follows from what the session store is for. Its log is the execution
ledger: it lets a worker resume a turn after a crash, avoid repeating a model
call that already has a receipt, and prove which inputs were applied. Keeping
that ledger readable forever would mean carrying every old decoder in the
runtime. vv-agent chooses not to.

So do not use the session store as your product's history. Anything your
product needs after a session is done belongs in your own tables:

- the messages you show in the UI, and the messages you use to seed a
  follow-up or retry session;
- the final `shared_state`;
- the final outcome (completed, failed, cancelled, waiting) and its error;
- per-call usage for billing.

The easiest place to write that copy is your projection consumer. It already
runs in a transaction that also advances the consumer cursor, so the copy and
the cursor commit together and a replay writes the same rows again. Record the
vv-agent version that produced the session next to the copy; you will want it
when you upgrade.

vv-agent does not yet ship a versioned export format for this. Hosts own the
shape of their copy.

### When a session no longer needs the store

A session is *quiescent* when all of these hold:

- its last turn has ended, or the session is closed; nothing is active or
  parked waiting for a user, an approval, a child or a provider;
- no operation has an unresolved outcome, including late provider results for
  audit attempts;
- every registered consumer cursor has reached the log head;
- the inbox has no unconsumed item.

A quiescent session whose product copy exists can be left behind by an upgrade.
Nothing in the product should read its kernel rows again.

## Upgrade procedure

1. Read the version notes for every release between yours and the target.
   Look for **Store boundary** entries. A release without one can be deployed
   over existing data.
2. Stop admitting new work on the old version: no new sessions, turns or
   inputs.
3. Let running and parked sessions reach a terminal on the old version, or
   close them explicitly. Parked sessions wait for people or providers, so
   decide how long you are willing to wait and close the rest.
4. Make sure every session you still care about has its product copy. If your
   projection did not write one before, backfill it now, while the old
   runtime can still read the rows.
5. Check that no session is non-quiescent and that every kept session has its
   copy. Do not switch versions until both checks pass.
6. Deploy the new release and provision its schema (see the version notes for
   DDL changes). Old kernel rows can be deleted, or kept for audit and read
   with the old pinned release.

If you cannot drain everything at once, run two worker pools pinned to
different releases and route each session to the release that created it. To
do that, store the vv-agent version in the session's attributes or in your own
session table when you create it. Never let a new-version worker drive an
old-version session.

## Version notes

### 0.35.4 — 2026-11-15

The default runtime lease TTL is now 60 seconds (`Runtime.ttl_ms`). File-backed
SQLite heartbeat connections use a separate configurable busy timeout and a
bounded lock retry window. Hosts that set explicit `ttl_ms` values keep their
existing behavior; no session store or wire migration is required.

See the [release notes](releases/0.35.4.md).

### 0.35.0 — 2026-11-12

Compared with 0.34.1, the SQL core can yield declared child or provider
observations to steering. Undeclared waits keep their existing behavior. A yield
commits the observation completions, applied steering and input consumption
together. L1 changes heartbeat lock ordering; it does not change TTLs, heartbeat
intervals, timeouts, retry classes or lease fences. Admission and Runtime.hook
signatures and the caller-supplied store instance stay the same.

Record schema 1, inbox schema 3 and SQL DDL are unchanged. Existing 0.34.1
sessions without the new fields remain readable. Declared parked operations
and recovered provider acceptance can contain literal-true `yield_on_steer`;
child waits also identify `yieldable_children`. Yielded completions contain
`yielded_by`. An older reader rejects a session containing these new closed
fields, even though the schema versions have not changed. Keep these sessions
and their dependent parent/child work on a reader that supports them. Do not
strip fields to send a session back to an older worker.

New long result candidates include their content hash in archive identity.
Existing committed long receipts keep their original reference and validation
rules. No stored-session rewrite is required.

Known limitations:

- App Server, Interactive and local Runner/RunHandle entry points accept wait
  declarations but treat them as undeclared waits. Steering waits for the normal
  result; these entry points do not provide yieldable waits in 0.35.0.
- A host must not run a whole `drive` or `poll` inside an outer write transaction.
  Savepoints do not release the outer transaction's locks.
- Concurrent native transactions on the same raw connection shared by the host
  and runtime are unsupported. The SDK does not promise to detect them.
- L2–L5 remain unfixed: cleanup evidence and full handoff after stop (L2),
  control-token lifetime (L3), opaque deadline start (L4), and SQLite busy
  classification (L5).

See the [release notes](releases/0.35.0.md) for scope. Canonical contract adoption
and release validation remain separate gates.

### 0.34.1 — 2026-11-07

0.34.1 includes 0.34.0, which was not published to PyPI, plus two test
corrections. It adds no product behavior or store boundary. Follow the
[0.34.0 upgrade notes](#0340--2026-11-07) below.

### 0.34.0 — 2026-11-07

New sessions declare input protection inside the closed creation seed. Python
0.33.3 rejects that declaration when it decodes the creation record, reads from
the beginning, or restores a session. This includes finished sessions. Paged
record reads, inbox reads and raw JSON reads can skip the creation decoder;
check the supported version at each host entry point before using these paths.

Existing sessions without the declaration keep their original compression and
continuation rules. Do not edit their creation records. Repeating transaction
`tx.create` with an omitted choice preserves the existing creation bytes and
rule; only the first creation defaults to true. Explicitly different choices,
other creation fields and consumer sets still conflict.

Children and App Server forks inherit the source choice. New descendants carry
an explicit Boolean, including false inherited from an undeclared parent;
0.33.3 rejects either value. Once an old session creates a child in this new
format, keep the parent's and child's execution, cancellation and result delivery
on the new runtime. An old parent can need to read its new child to cancel it or
receive its result. Checking only the parent's original creation record is not
enough to decide whether rollback is safe.

Host hooks must keep protected messages unchanged and in order. To add product
IDs or other information, include it when producing the tool result, or keep the
original messages and append a separate message. Do not rewrite a protected tool
message in before-memory or before-LLM hooks. `unseen_input_removed` identifies
such a hook/replacement violation; fix the hook instead of blindly retrying.
`unseen_input_capacity_exceeded` means the original protected input cannot fit;
choose a suitable model or explicitly start a new session with host-selected
input. Do not silently summarize the protected text.

A declaration-only seed is not bootstrap history. Hosts that inspect seed before
calling the runtime must distinguish it from the `messages`/`shared_state` pair;
do not clear their configured history when that pair is absent.

There is no SQL or global record-schema migration. The contract is 37.0.0;
publication and immutable lock/snapshot selection are separate release gates.

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


### 0.33.3 — 2026-11-06

A token quota means “wait for more allowance”; shrinking history cannot fix it.
`too many tokens per day` on 429 now follows the existing rejection backoff and
ends with `model_request_rejected` after the permitted attempts. `too many
tokens per minute` on 400 stops after one call. Neither starts compaction.
Actual context-size errors still compact, including the provider phrases
`context window exceeded` and `model context window is ...` outside 429.

No dependencies, stored record/inbox/SQL versions or frozen definitions change.
Stop/drain drivers and API entry points before switching so all workers use the
same classification. Existing receipts retain their recorded outcomes; upgrading
does not settle old Unknown calls or replay quota failures. The 0.33.2 restriction
against handing sessions back to 0.33.1 continues to apply. Contract 36.0.0 is the
new canonical behavior; the release lock/snapshot must be synchronized after its
immutable publication. See [release notes](releases/0.33.3.md).

### 0.33.2 — 2026-11-06

Requires contract 35.0.0. Provider HTTP 4xx rejection (except 408) is now a
confirmed failed attempt, rather than an unresolved external operation. Handle
terminal reason `model_request_rejected` as FAILED; show its safe terminal result
or the failed event error. Structured status/code/type/message is retained in
`op_completed.result.rejection` and `ModelCallFailed.metadata.model_rejection`.
Automatic HTTP rejection requires complete observation of every physical send
through an SDK-owned HTTP client, including vendor retries. The classifier also
checks that the final response was produced by that same observation. A real response alone
is insufficient. Unobservable host adapters must assert rejection explicitly with
`vv_agent.llm.errors.ModelRequestRejected`. Any timeout, transport failure,
408/5xx or incomplete send in that invocation or exception graph retains Unknown.
Arbitrary provider prose is replaced by a generic explanation; status and safe
code/type remain available, including safe output-repair diagnostics.
The uncertainty veto also recognizes installed OpenAI/Anthropic native connection
and timeout exceptions without a low-level cause. A context-length message does
not override that veto. Host adapters must retain the original exception and its
graph/observation; remove quota/context-length text rewrites that discard them.
Apply product wording when mapping the failed result, after SDK classification.
Content-policy failures must not become empty successful responses merely because
their raw payload contains an error code. Test that host path separately.

The observed HTTP clients retain vendor connection limits and Anthropic TCP
keepalive settings. Explicit proxies and per-request timeouts keep their existing
priority; `httpx2>=2.7,<3` is declared directly. Concurrent failures keep their own
request indices, and unindexed failures veto the entire invocation. Callback exit
closes the iterator and retained response streams explicitly.
Recognized oversized input keeps the existing compaction/retry path, including
real HTTP 400 context errors and HTTP 413; it is not `model_request_rejected`.
Content rejection stops without trying the next endpoint. 401/403/404 immediately
advance the frozen endpoint order when multiple endpoints exist, but call a
single endpoint only once. 429 retains backoff and can retry once on a single
endpoint. All retries keep the existing `max(2, frozen endpoint count)` limit;
exhaustion reports the last safe rejection. Explicit adapter `retryable=True`
or `False` overrides these defaults, including the single-endpoint rule.

There is no store boundary: record 1, inbox 3, SQL 3, RunEvent v6, App Server v5,
tool schemas and frozen definitions are unchanged. Existing 0.33.1 turns remain
readable and can continue. Historical Unknown receipts keep their original
meaning; upgrading does not settle or replay them. Earlier upgrade boundaries
still apply. handler_version remains unchanged.

**The .ai upgrade MUST globally stop all workers and API drivers, upgrade both
machines, then restart.** Stop admission, consumers, recovery scans, scheduled
and manual drive/advance entries; let in-flight calls and commits exit, and
prevent old processes from restarting. Start neither new worker until both
machines run 0.33.2. Do not kill an in-flight model call to force the switch;
unconfirmed effects retain Unknown recovery.

**Any session driven by 0.33.2 MUST NOT be handed back to 0.33.1. Rollback MUST
isolate data by version ownership.** Mark ownership in the host at first new
runtime drive, including children, waits, finished histories, retries and
recovery/read paths. Keep those sessions on a 0.33.2 pool or paused. A permanent
rejection receipt committed before turn_ended can be resumed incorrectly as
success by 0.33.1; a rejection retry op_planned makes its reducer fail even on a
finished history. A shared queue or lease does not isolate runtime versions.
Never erase receipts, change attempts, clear Unknown or resubmit to enable rollback.

Before rollback, under the same global driver stop, run the repository's
read-only precheck with the host's existing store binding:

```python
from scripts.check_model_rejection_rollback import check_store

findings = check_store(store, page_size=256)
# Each finding: session_id, turn_id, reasons. No leases or writes are performed.
```

Run the host wrapper from the SDK repository root with `uv run python ...` so
`scripts/` is importable; the helper is a repository script, not an installed SDK
export. It pages all sessions and all history via list_sessions/read without
folding through a reducer. It flags all rejection history conservatively,
including rejection_without_turn_ended and rejection_retry_history. Read or
validation failures propagate and must block rollback. No findings is only
absence of detectable rejection records; it does NOT prove absence of 0.33.2
ownership, which the host must check separately. See [release notes](releases/0.33.2.md).

### 0.33.1 — 2026-11-05

Requires contract 34.0.0. The SDK patch fixes 0.33.0's overly small cumulative
search budget. Each regex engine invocation is limited to 250 ms; enumeration
and all matching share 10 s of thread CPU, retaining nested and in-loop checks.
Earlier content matches and stopping coordinates survive either timeout; the
message distinguishes a slow engine call from total CPU exhaustion.

No store boundary, wire change, tool schema change or frozen-definition change
is introduced. Existing 0.33.0 turns can continue on the corrected handlers;
no old decoder or compatibility branch is needed. When upgrading from 0.32.0
or earlier, the 0.33.0 transition below still applies. See
[release notes](releases/0.33.1.md).

### 0.33.0 — 2026-11-04

Requires contract 33.0.0. Record, inbox, SQL and App Server schema versions are
unchanged. read_file adds start_char, removes max_chars and uses a fixed
12,000-character page. Its schema participates in the
frozen definition digest. An executable turn with the old schema closes as
`handler_schema_or_capability_mismatch` at its next dispatch, even if its next
call is another tool. Paused turns and turn-level waits reach this check after
resumption. Closing the turn can cancel providers and child tasks; outstanding
external effects still need reconciliation.

For a guard closure on a turn that is not marked cancelled, an already parked,
started provider operation keeps its original wait when cancellation is not
definitively confirmed. The terminal record lists it in `unconfirmed_operations`
without writing `op_unknown`. Late authenticated results are audit-only; driving
the session again does not resubmit the work. User cancel/abort keeps its existing
`op_unknown` behavior, including `provider_result_invalid` for invalid Definitive
replies. A closed turn does not prove the external effect stopped.

If saving command output or reading its capture for the first time fails, the
result preserves the confirmed status, exit code and directive. It returns a
bounded preview and
`metadata.retention={"status":"unavailable","reason":"artifact_persist_failed"}`,
without an artifact or cursor recovery pointer. Handle this as unavailable
output, not a reason to run the command again.

Drain running and paused turns on the old runtime before upgrading. If draining
is impossible, users can continue from the conclusion in a new session. New
sessions do not inherit model read authorization for the old session's artifacts.
Do not relax the guard, alter frozen definitions or rewrite old records.

Task and compaction hashes have separate type tags. Only short literal task IDs
keep their 0.32.0 directory segment: 1–64 ASCII characters, the first an ASCII
letter or digit, the rest ASCII letters, digits, `.`, `_` or `-`, and no lowercase
`s-` prefix (exactly `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` without that prefix).
This includes ordinary UUID-like task IDs; `S-task` also remains literal.
Task IDs with the lowercase `s-` prefix, either separator (`/` or `\`), more
than 64 characters, non-ASCII characters or any other nonliteral spelling now
use a hash directory. For example, `s-task`, `task/a`, 65-character IDs and
`任务` change directory segments. Their old objects remain in storage but their
old paths are no longer authorized for model rereads. The lossy sanitizer is
not an authorization rule.

Old compaction paths produced by the lossy,
truncated `session-<sid>-<tid>` encoding are no longer authorized. Their original
text remains in storage, but models cannot reread it through tools. No automatic
object migration or cross-session artifact grant is provided. Fork preserves
immutable evidence references without granting model access to another scope.

The portable content matcher now uses regex VERSION0. For `a\u0301b`, \w+
includes the combining accent, \b finds only the two end boundaries and \W does
not match that accent. Case-insensitive `i` does not match dotless `ı`. These
are observable changes from stdlib re. Reverse matching (`(?r)` or REVERSE flags)
is rejected. Glob and child exclusion patterns share the interruptible request
budget; timeouts are explicit errors. Native ripgrep retains its engine semantics.
See [Python mapping](parity-contract.md) for implementation details and the
[release notes](releases/0.33.0.md) for scope.


### 0.32.0

Requires contract 32.1.1. Long tool results are no longer cut down: the model
sees a 12,000-character head-and-tail preview, and the complete text is
archived under `.vv-agent/artifacts/` with a marker naming both the
`read_file` range read and the `search_files` route back to it.

If your host registers its own tool handlers, check whether you shorten result
content before returning it. A result your handler already cut is not the
complete original, and vv-agent must not claim otherwise. Leave `truncated`
unset unless you are handing over an artifact that holds the full text; when
you do, the receipt's `source_complete` follows your declaration and the kernel
verifies the artifact instead of writing a second copy. Results you do not
shorten need no change.

`metadata.vv_artifact` appears only on results that were bounded. It is
kernel-owned, and the reducer rejects a completion whose `source` does not match
the operation that produced it. Do not write to that key from a host handler.

Contract 32.1.1 is pinned in `contract.lock.json`; run
`scripts/contract_snapshot.py check` after upgrading. Record schema 1, inbox
schema 3, SQLite user_version 3 and RunEvent v6 are unchanged. This is not a
store boundary and existing sessions need no migration.

### 0.31.3

Upgrade vv-llm to 0.8.5 when adopting this release. It is the recommended
version and the new minimum dependency. It preserves provider tool-call indexes,
resolving the [0.31.2 known limit](releases/0.31.2.md) for interleaved calls
sent one piece per chunk without continuation IDs. Providers still need to
send a stable index or call ID.

Local incremental counting now also allows vv-llm 0.8.5 with
deepseek-tokenizer 0.3.0 or tiktoken 0.14.0. This keeps unchanged text from
being tokenized again after the dependency upgrade. The formatter, model
routing and counting paths match 0.8.4; the existing boundary checks still
apply. The previously verified 0.8.3 and 0.8.4 pairs remain listed.

Contract 32.0.0, App Server v5, record schema 1, inbox schema 3 and SQL layouts
are unchanged. This is not a store boundary. Existing sessions need no migration.

### 0.31.2

No host action. Streamed arguments of parallel tool calls are routed by call
ID and index instead of the last opened call; see [0.31.2](releases/0.31.2.md).

### 0.31.1

No host action. `import vv_agent` loads the model SDKs on first model use
instead of at import time; see [0.31.1](releases/0.31.1.md).

### 0.31.0

Regenerate App Server bindings for v5 and public API inventory v15. Hosts can
register business job providers through `RunConfig.tool_providers`; direct
`Runtime.providers` integrations use the same result validation. Rebuild all
named bindings after restart, including during revoked-authority cleanup.
Only an explicit `function` binding selects the synchronous handler.

Provider completions must use the original tool call ID and a definitive
status. Include the original acceptance evidence after a job was parked.
Invalid results keep the job unresolved and report `provider_result_invalid`;
a missing binding reports `missing_host_binding`. Cancel receipts acknowledge
a request, while unconfirmed jobs remain visible after the turn ends.

Record schema 1, inbox schema 3, SQLite user_version 3 and RunEvent v6 remain
unchanged. There is no store boundary. Existing stored bytes still load, but
hosts must rebuild the required providers before driving them.

Completed tool items now carry bounded results and exact receipt references.
The `tool_result` diagnostic no longer contains the full body or arbitrary
metadata. Use `thread/toolResult/read` for byte pages, and implement the required
`AppServerHost.authorize_tool_result_resource` method. A workspace can opt in
to `OutputRetentionBackend`; immutable hosts can name existing versions without
copying them. Without that capability, file/image resources are `not_retained`.
See [tool result reads](app-server-tool-results.md).

Reprojected events change bytes under their existing stable event IDs.
Hosts that store immutable event projections must plan their product projection
version boundary before upgrading; a kernel store schema bump is not required.

User questions now carry a retained typed form in status and snapshot `state`.
Replace `respond.message` with `respond.target` and `respond.answer`. Free text remains
legal for every form, including choices with custom selections disabled; the
model-facing ask_user result is now JSON labeled free_text or selection. Update
product projections and answer parsers accordingly. Existing direct host text
inbox replies remain legal with explicit interaction, turn and generation.
Child-task messages now queue steering instead of inferring a user-wait target.
See [user interactions](app-server-interactions.md) for the exact shapes.

Update output-retention backends for offset/length reads and per-page immutable
version checks. A digest must not be rescanned for every page. Replace private
`vv_agent.session.kernel._HeartbeatStopped` imports with public
`vv_agent.HeartbeatStopped`. Stored records, inbox envelopes and SQL versions
remain unchanged. Older applied text replies and retained questions still load;
only new replies use the labeled result. Reprojected events and snapshots gain
metadata.interaction and state, so hosts own any materialized projection update.

### 0.30.5

PostgreSQL heartbeats now continue while child admission and `before_commit`
callbacks run. Append finishes encoding and history preparation before taking
the latest Lease. It keeps every owner, epoch, expiry, head and inbox check.
Slow callbacks no longer cause a lease loss just by blocking a Python lock.

PostgreSQL heartbeat timeouts and operational errors retry while the last
published Lease has enough time left for another attempt. An expired or fenced
Lease still stops the drive. This also applies during runtime factory setup.
The final append uses the remaining Lease time as its SQL timeout and restores
the host's previous settings before returning from a nested transaction.
Each `close()` waits at most 6 seconds; failure to stop raises `LeaseLost` and
prevents release or runtime handoff. Hosts with custom connection contexts must
bound connection setup and exit too. See
[transaction callbacks](host-integration.md#transaction-callbacks) for exact
limits, callback rules and transaction-duration logging.

Recovery still rolls back failed admission and allows another drive to redo it.
A durable rule that stops an interrupted admission from being repeated is not
included. A real worker crash or database lease loss can still cause that redo.

Contract 31.0.0, record schema 1, inbox schema 3 and SQL layouts stay unchanged.
This release has no store boundary or migration.

### 0.30.4

Compaction now uses local estimates for Moonshot, Kimi, Gemini, Claude,
MiniMax, Stepfun and GLM, even when those models have enabled endpoints.
Each family uses vv-llm's no-endpoint fallback. These checks no longer upload
the conversation to a provider, wait for its retries, or accept its failure
constant as a context size. DeepSeek, other tiktoken models and Qwen keep
their 0.30.3 behavior.

The new counts can differ from provider counts, so a session may compact
earlier or later after upgrading. Review the host's context window, output
reserve and compaction threshold before upgrading. The existing buffer
leaves some room, but is not an error bound for a different tokenizer.
No safety multiplier is applied. See [token counting](token-counting.md) for
the methods and [capacity metadata](model-settings.md#capacity-metadata)
for the threshold calculation.

`incremental_token_counting_status(model)` keeps its `(bool, str)` result.
The text now names the local method, including whole-conversation methods.
The verified dependency pairs below still gate incremental counting.
Explicitly replaced message counters remain in control. Direct vv-llm
calls and `count_tokens` keep their previous behavior.

Contract 31.0.0, record schema 1, inbox schema 3 and SQL layouts stay unchanged.
This release has no store boundary or migration. Saved compaction receipts
remain authoritative; future checks use the new estimates.

### 0.30.3

Compaction counts DeepSeek, local tiktoken models and Qwen on the worker itself,
even when another part of the host configures vv-llm's global token server.
The count matches vv-llm's whole-conversation count with no token server
configured. Tools and multimodal messages keep the same formatting and counts.
Qwen still counts the whole conversation. Unsafe text boundaries and unverified
dependency versions also use a whole count, on the worker.

`incremental_token_counting_status(model)` no longer treats a global token
server as a reason to disable DeepSeek or tiktoken caching. The exact version
pairs listed below and the check for replaced counting functions still apply.
An explicitly replaced message-counting function remains in control.
Claude, Gemini, Moonshot, Kimi, MiniMax, Stepfun and GLM keep their existing
server, provider and fallback behavior.

Hosts do not need to remove their global token server settings. Contract
31.0.0, record schema 1, inbox schema 3 and SQL layouts stay unchanged.
This release has no store boundary or migration.

### 0.30.2

Local DeepSeek and tiktoken counting reuse exact counts for unchanged text instead of
tokenizing the whole conversation at each step. This applies to the explicitly
verified pairs vv-llm 0.8.4 + deepseek-tokenizer 0.3.0 and
vv-llm 0.8.3 + deepseek-tokenizer 0.3.0. The cl100k and o200k paths allow
vv-llm 0.8.3 + tiktoken 0.14.0 and vv-llm 0.8.4 + tiktoken 0.14.0.
Each requires independent newline boundaries. Empty text parts and parts
beginning with CR/LF use the original whole-conversation path. tiktoken also
rejects any leading whitespace, and o200k rejects leading `/`.
In 0.30.2, other dependency versions, token servers, Qwen and families that can count
remotely keep the original whole-conversation path. The local JSON
estimate also reuses character counts and rounds only the final total.

Hosts can check their environment during startup or in their own tests:

```python
from vv_agent.memory import incremental_token_counting_status

enabled, reason = incremental_token_counting_status("deepseek-flash")
assert enabled, reason
# The same check works for "gpt-4o" and "gpt-3.5-turbo".
```

This reports whether the installed dependencies and configuration allow the
incremental path. An individual conversation with unsafe text boundaries still
uses the original exact count. In 0.30.2 a configured global token server also
disables the local path; 0.30.3 removes that restriction.
An unverified version produces one WARNING per model family
per process when counting falls back.

After upgrading vv-llm or a tokenizer, rerun
`uv run pytest tests/test_incremental_tokens.py tests/session/test_step_cost.py`
with real PostgreSQL, review the formatter, model routing and tokenizer
boundaries, then add the exact version pair to the verified set. Versions are
listed individually; a newer version is not assumed safe. The gates are
`_VERIFIED_DEEPSEEK_VERSIONS` and `_VERIFIED_TIKTOKEN_VERSIONS` in
`memory/token_utils.py`. The repository lock
uses vv-llm 0.8.5, deepseek-tokenizer 0.3.0, tiktoken 0.14.0 and
qwen-tokenizer 0.3.0.

Model request encoding reuses private canonical message fragments. Internal
policy reads avoid detached task copies; host callbacks still receive copies.
Routing is still resolved before dispatch, since a host can remove an endpoint
after a plan commits.

Contract 31.0.0, record schema 1, inbox schema 3, SQL layouts and stored bytes
are unchanged. There is no store boundary or migration. Existing sessions from
0.24 through 0.30.1 load to the same state. Both full message snapshots remain
in the log.

### 0.30.1

Cold history loading avoids replaying message snapshots that a later complete
snapshot replaces. Boundary source digests, record validation, sequence checks
and lease fencing remain in place. PostgreSQL reads use binary results to avoid
hex-encoding record bodies. Canonical record bytes share their storage buffer and repeated messages share
private parsed values;
result and event projections copy only the data they return. Validated messages
and canonical fragments are reused while every record, request and source digest
is still checked. The package adds explicit jiter and orjson dependencies; orjson
is only used for private equality keys, never for persisted or protocol bytes.

Contract 31.0.0, record schema 1, inbox schema 3 and SQL layouts are unchanged.
This is not a store boundary. Sessions written by 0.24 through 0.30 keep the same
stored bytes and fold to the same state. No migration or host API change is
needed. Full message snapshots still accumulate in the log, so cold load cost
still grows with the total stored data.

### 0.30.0

Contract 31.0.0 requires renewal throughout runtime factory preparation and
captured-prefix state reads that do not hold the session row while loading,
folding or copying history. The factory starting TTL remains 15 seconds and
renews every 250 ms; preparation may exceed that interval. The returned runtime's
TTL applies before execution. Renewal failure discards the factory result.
Deferral and release use the current lease and every renewal thread is joined.

Hosts can drop a factory-lease workaround such as `RuntimeFactoryStore` and pass
their store and factory directly to `drive`. A custom store must implement
`heartbeat_store()` to open an independent renewal connection to the same
session database. Built-in SQL stores implement it; PostgreSQL subclasses that
use host-managed connections should override it with their host's connection
factory. In-memory SQLite shares its serialized connection. Bound connection
and callback I/O so shutdown can join renewal workers.

Record schema 1, inbox schema 3, SQL layouts, SQLite user_version 3, App Server v4
and the public API inventory v14 are unchanged. This is not a store boundary;
existing 0.27.x sessions remain readable. Hosts pinned to 0.27.x must also review
0.28.0 and 0.29.0 for App Server client changes and fork/compaction routing.
Do not surround history reads with a host-owned session write lock or a SQLite
write transaction: that outer lock remains the host's responsibility.

### 0.29.0

Contract 30.1.0 adds thread/fork. App Server v4, public API inventory v14, record
schema 1, inbox schema 3 and SQL layouts stay unchanged: this is not a store
boundary. Regenerate schema/TypeScript clients to use the new method and optional
lineage in fork context reads. Forks retain a model default in existing creation
attributes; route them to 0.29.0 or newer. Older workers can read these maps but
would ignore that default. The host prepares business workspaces and implements
fork authorization and binding; see [task cloning](app-server-host-integration.md#task-cloning).

### 0.28.0

Contract 30.0.0 selects App Server v4 and public API inventory v14. There is no
store boundary: record schema 1, inbox schema 3, SQL layouts and RunEvent v6 stay
unchanged. Regenerate schema and TypeScript clients. model/list now requires the
capability fields and uses null for unknown values. Use thread/context/read and
thread/compact/start for context UI; model selection remains thread/model/select.
Creation attributes may contain app_server_context, a compiled projection baseline.
Do not route sessions with pending compaction intents to an older runtime, which
would interpret the new content extension as ordinary input.

### 0.27.0

This release adopts contract 29.1.0 and public API inventory v13. Record schema
1, inbox schema 3, SQL layouts, RunEvent v6 and App Server v3 are unchanged.
Existing 0.26 records still load. There is no store boundary.

Hosts can offer a compact tool and bind children that require fresh parent
summaries. See [the binding examples](host-integration.md#a-compact-tool-and-fresh-child-summaries).
Change the host handler version when changing child binding semantics. A 0.26
reader loads the new records but ignores compaction intent and does not enforce
summary-dependent admission. Do not downgrade sessions with pending requests.


### 0.26.0

Contract 29.0.0 separates model identity from endpoint routing. This is not a
store boundary: record schema 1, inbox schema 3, SQLite user_version 3 and the
SQL schemas stay unchanged. Records written by 0.24 and 0.25 remain readable.
Their recorded endpoints remain evidence but no longer cause definition drift.
The binding's required `endpoints` field remains routing evidence, with the
same record and inbox schemas as before.
Existing definitions did not record model limits in their binding, so those
missing fields cannot be checked retroactively. New definitions record both.

Operators can add, remove or rename endpoints without ending live sessions.
New operations use current routing. If an endpoint disappears before its
planned attempt starts, the operation closes with a non-chargeable
`endpoint_unavailable` result; the next primary cycle replans under its normal
limits. Started attempts retain their endpoint evidence and Unknown rules.

Hosts with custom `VvLlmClient.ordered_targets` now get that ordering policy
after model selection too. The resolver must handle the default model as well
as selected models. Remove host checks that compare endpoint lists as identity.
Use `Runtime.definition_digest(turn_started_record)` for a retained definition;
passing only its task cannot reproduce historical definition bytes. Preserve
logical-model, tool, authorization and unknown-outcome checks. See
[session model selection](session-model-selection.md#resolver-contract).

### 0.25.0

Contract 28.0.0 changes the App Server wire to v3. This is **not a store
boundary**: record schema 1, inbox schema 3 and the SQL schemas are unchanged.
Existing 0.24.0 sessions remain readable. Regenerate App Server bindings and
update callers to supply stable inputIds and expected turns. Replace afterItemId
with thread/events/list cursors and replace the old status fields with the rich
status object. Any authorized connection may control or approve a retained
turn. Unsubscribing no longer closes the session. Custom AppServerHost
implementations resolve selected model IDs without submitting provider work;
the frozen empty-route model reuses its original client. Distributed hosts must
implement authorize(connection_id, method, params); the local default allows all.
Client inputId/actionId values contain 1..128 ASCII letters, digits, dot,
underscore, colon or hyphen. Update callers of thread/unsubscribe and turn/action
to their current responses; turn/interrupt no longer accepts reason.
Custom stores also implement the App Server read and admission-lock methods
listed in [host integration](app-server-host-integration.md#custom-stores).

Controller inputs now use controller/ plus the command digest. **actionIds
submitted before this upgrade are not deduplicated afterwards.** Drain those
actions on 0.24.0, or do not retry them across the upgrade. There is no legacy
digest lookup, even though the stored sessions remain readable.

`session.kernel.drive()` now returns bool: true means it acquired a lease
(including a deferred runtime); false means another holder prevented acquisition.
It does not report whether a turn completed. `AppServer.wake(thread_id)` is the
public embedded-host wake entry point and returns None.

### 0.24.0

Contract 27.0.0, public API v11, inbox schema 3.

- Adds per-session model selection: the `model_selection` inbox input,
  `Runtime.model_resolver` and App Server `thread/model/select`. See
  [session model selection](session-model-selection.md). Use it instead of
  changing a frozen task or `Runtime` binding.
- **Store boundary.** Every `input_applied` record embeds the inbox item it
  consumed, and that item must now be inbox schema 3. Sessions written by 0.23.x
  have schema 2 inputs in their log, so 0.24.0 cannot read any of them. Drain
  and copy before switching.
- PostgreSQL: add `'model_selection'` to the `sk_inbox.kind` check constraint.
  Take the current DDL from `src/vv_agent/session/postgres.py`. Widening the
  constraint does not make old rows readable.
- SQLite files use `PRAGMA user_version` 3. Files at version 1 or 2 are
  rejected.

### 0.23.1

Contract 26.0.0, public API v10. No schema or API change; deploy over 0.23.0
data. Token usage now reads cache writes reported as `cache_creation_tokens`
(vv-llm) and as `prompt_tokens_details.cache_write_tokens` (Moonshot kimi-k3,
OpenRouter).

### 0.23.0

Contract 26.0.0, public API v10, inbox schema 2.

- `tick(..., dispatch=enqueue_drive, project=...)` hands session identities to
  your queue instead of driving inline. Give queued drive messages an expiry no
  longer than the tick interval, or coalesce them by session.
- Pass the runtime factory itself to worker `drive`. Raise `RuntimeNotReady`
  when input preparation should be retried without failing the turn.
- `tick` reports per-session failures together as an `ExceptionGroup` after
  the scan; handle it.
- Custom stores must implement `defer_drive` and `defer_projection`.
- `Runtime.children` may return a nonempty sequence of children that are all
  background or all blocking. Host rows written in that callback must use the
  same admission transaction. `InvalidChildBatch` rolls the admission back.
- `provider_result` inputs must carry `usage` (use `{}` when none was
  observed), and authentication must cover it.
- **Store boundary.** Inbox schema 2 rejects sessions written by 0.22.x.
- PostgreSQL: add `drive_retry_at_ms` to `sk_session` and
  `project_retry_at_ms` to `sk_consumer` (nonnegative bigint, default 0).
  SQLite files use `user_version` 2.

See [host integration](host-integration.md) for the full dispatch and retry
rules.

### 0.22.0

Contract 24.0.1, public API v8. Every entry point now runs on the session
kernel. Checkpoints, execution backends, distributed entry points, Redis
stores and old resume APIs are removed; see the tables below. There is no
conversion of a running pre-0.22 execution: finish or close old work with the
old release first.

The `redis` and `celery` extras are gone. Use `[postgres]` for the SQL store
and `[s3]` for workspace objects. Celery tasks, the broker and queue routing
belong to the host; [host integration](host-integration.md) shows an example.

```bash
# Before
pip install 'vv-agent[redis,celery]'
# After, only when this host uses PostgreSQL
pip install 'vv-agent[postgres]'
```

RunEvent v6, model-call v2, task-token-usage v3 and App Server protocol v2
each have one strict shape. Upgrade event readers, billing, UI timeline and
schema clients, and `initialize` negotiation together. The wire changes are in
the [contract 24.0.0 changelog](https://github.com/AndersonBY/vv-agent-contract/blob/v24.0.0/CHANGELOG.md).

## APIs removed in 0.22.0

### Execution and resume

| Removed | Use instead |
| --- | --- |
| `Runner.resume(RunState)`; `RunState`, `from_result`, `approve`, `pending_approval_ids`, `approved_interruption_ids`, `approval_snapshot`; `RunResult.into_state` | `RunHandle.approve` and `Runner.resume(session_id, turn_id)`; `AgentSession.continue_run` for an interactive session |
| `ApprovalSnapshot`, `RunResult.approval_snapshot` | Pending approvals on the live handle or session, and App Server approval requests |
| `Runner.start_distributed`, `start_distributed_compiled`, `finalize_distributed` | The host creates a session and pushes inputs with stable IDs; a worker calls `drive`; consumers project committed records |
| `AgentRuntime`, `ToolCallRunner` | `Runner.run_sync`, `Runner.start`, `Runner.stream_sync`; low-level code uses `session.kernel.drive` with `session.runtime.Runtime` |
| `ExecutionBackend`, `InlineBackend`, `ThreadBackend`, `CeleryBackend.start/advance` | One kernel driver. `RunHandle` runs locally in the background; the host's queue dispatches workers |

```python
# Before
state = result.into_state()
result = Runner.resume(state)

# After: same turn, same frozen budget and definition
result = Runner.resume(result.raw_result.session_id, result.raw_result.turn_id)
```

`Runner.resume` needs the host bindings of the process that started the run.
To recover in a new process, reopen the same durable store and rebuild the
interactive client or App Server with the original providers, tools and host
bindings. User and approval replies continue the original turn. A new prompt
after the turn ends starts a new turn.

### Persistence and dispatch

| Removed | Use instead |
| --- | --- |
| `CheckpointConfig` (including `capability_refs`, `credential_slots`), `CheckpointExtension`, `ReconciliationProvider`, `ResumeObservation`; `RunConfig.checkpoint_config`, `checkpoint_extensions`, `reconciliation_provider` | The frozen turn definition, explicit provider bindings, kernel records and inbox, and trusted provider evidence |
| `Checkpoint`, `CheckpointStore`, `InMemoryCheckpointStore`, `SqliteCheckpointStore`, `RedisCheckpointStore`, `OperationJournalEntry` | `SessionSpec`, `Record`, `InboxItem`, `SessionStore`, `SessionTx`, `SQLiteStore`, `PostgresStore` |
| `RunResult.checkpoint_key`, `resume_observations` | `raw_result.session_id`, `raw_result.turn_id`, and the projected model and tool receipts |
| `RunConfig.execution_backend`, `sub_task_manager` | The kernel runs children itself; the host supplies tool and model providers and, optionally, a durable store |
| `ControllerCommand`, `ControllerCommandReceipt`, `ControllerCommandResolution`; `reap_controller_command_wakes`, `resolve_controller_command` | A `control` `InboxItem` with a stable ID, pushed through the store; `drive` and the scheduling scan pick it up |
| `DistributedBackend.produce_host_interaction`, `claim_and_consume_host_interaction_response` | Typed host interaction values and an authenticated inbox response |
| `DistributedRunEnvelope`, `DistributedRunHandle`, `DistributedAdvanceDecision`, `DistributedDeliveryOutcome`, `DistributedWaitReason`, `RuntimeRecipe`, `CapabilityRef`, `DistributedCapabilityRegistry` | A durable session ID plus a `Runtime` or runtime factory. State stays in SQL, not in worker envelopes |

```python
# Before
config = RunConfig(checkpoint_config=CheckpointConfig(store=checkpoint_store),
                   execution_backend=backend)

# After: a local durable interactive session
from vv_agent import AgentSessionOptions, InteractiveAgentClient, SQLiteStore

with SQLiteStore.standalone("/tmp/agent-sessions.db") as store:
    store.install_schema()  # once, on an empty database
    client = InteractiveAgentClient(options=AgentSessionOptions(
        model_provider=provider, session_store=store,
    ))
    try:
        session = client.create_session(agent=agent, session_id="thread-001")
        result = session.prompt("Continue the work")
    finally:
        client.driver.close()
```

For PostgreSQL, bind `PostgresStore(connection)` to a psycopg connection you
own and use `join_transaction(connection)` so your business writes and the
consumer acknowledgement share one transaction. See
[joining host transactions](session-kernel.md#joining-host-transactions).

### Tool and provider outcomes

| Removed | Use instead |
| --- | --- |
| `DeferredToolHandle`, `ToolContext.defer`, `ToolCallOutcome` | A provider returning `Accepted(handle)`, `Definitive(result)` or `Unknown(reason)` from `vv_agent.session.providers` |
| `AcceptDeferredDecision`, `DeferredResolveDecision`, `DeferredResolutionReceipt`, `DeferredResolutionConflict`, `DeferredResolutionStale`, `DeferredCheckpointClaimed`, `DeferredHandleError`, `CheckpointStore.resolve_deferred` | An authenticated `provider_result` inbox item for the original operation and attempt. Same ID with different bytes raises `Conflict`; a stale lease raises `LeaseLost` |
| `DeferredResolutionError`, `DeferredResolutionResultInvalid` | Invalid provider results raise `ValueError`. The internal definitive-result check keeps its error code as `tools.outcomes.DefinitiveResultInvalid` |

```python
# Before: the handler managed a deferred protocol
return context.defer(handle)

# After: the provider owns submission, query and cancellation
from vv_agent.session.providers import Accepted, Definitive, Unknown

return Accepted(handle={"provider": "jobs", "job_id": admitted_job_id})
# query(handle) returns Definitive(result) only with trusted completion evidence.
# While the job runs, return the same Accepted handle; otherwise Unknown(reason).
```

The handle is your provider's own data, not a kernel wire format. Implement
`Provider.submit`, `query`, `cancel` and `authenticate`, and bind the provider to
the `Runtime`. A timeout or an unconfirmed stop stays unknown; it is never
treated as proof of failure. An ordinary `FunctionTool` still returns a
`ToolExecutionResult` or a supported output value.

### Sessions

| Removed | Use instead |
| --- | --- |
| `Session`, `MemorySession`, `SQLiteSession`, `RedisSession`; `MemorySessionStore`, `SQLiteSessionStore`, `RedisSessionStore`; `RunConfig.session` | `AgentSession` and `InteractiveAgentClient` on a kernel SQLite or PostgreSQL store |
| The transcript-style `SessionStore` (append and commit) | The kernel `SessionStore` protocol (log, inbox, leases, consumers). Same name, different contract |
| Writable `AgentSession.session`, `replace_messages`, `replace_shared_state`, `clear_queues` | A creation-time seed; a fresh session ID to reset; inbox controls for a running session |
| `IdempotentRunEventStore`, the old `RunEventStore` ledger | `SessionRunEventStore` projects history from the log; `JsonlRunEventStore` is an optional sink |
| `session_store_conformance` | Removed. Validate a store against `tests/session/test_store_transactions.py` and the recovery and concurrency suites |

```python
# Before
result = Runner.run_sync(agent, prompt, run_config=RunConfig(session=MemorySession()))

# After
client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider))
try:
    session = client.create_session(agent=agent, session_id="conversation-001")
    result = session.prompt(prompt)
finally:
    client.driver.close()
```

### Creation-time seed

A session's history and shared state can no longer be replaced after it is
created. Pass them when you create it instead. To reset, for example on retry,
create a new session ID; reopening an existing ID must use the same creation
bytes.

```python
# Before
session = client.create_session(agent=agent, session_id=task_id)
session.replace_messages(hydrated)
session.replace_shared_state(previous_shared_state)
session.clear_queues()

# After: hydrated is a list[Message]; shared state must be JSON
session = client.create_session(
    agent=agent,
    session_id=f"{task_id}/retry/{retry_generation}",
    session={
        "messages": [message.to_dict() for message in hydrated],
        "shared_state": previous_shared_state,
    },
)
result = session.prompt(next_prompt)
```

Low-level hosts pass `SessionSpec(..., attributes={"seed": {"messages": [...],
"shared_state": {...}}})`; both members are required. The seed history is in
place before the first turn compiles. After that, `messages` and `shared_state`
are read-only views of committed records. Steering, follow-ups, replies,
archive and close go through the inbox. Closing a session stops further
execution; it does not erase history.

The seed is also how a host continues work across an upgrade: build the new
session's seed from the product copy described in
[session data and upgrades](#session-data-and-upgrades), not from the old
session's kernel rows.

### Events and App Server

`CheckpointCreatedEvent`, `CheckpointResumedEvent`,
`ReconciliationRequiredEvent`, `ReconciliationResolvedEvent`,
`ToolCallDeferredEvent`, `OperationReplayedEvent` and `SessionPersistedEvent`
are removed. Read the current typed `RunEvent`s from session projections.

```python
# Before
send("turn/resume", {"threadId": thread_id, "checkpointKey": key})

# After
send("turn/resume", {"threadId": thread_id, "turnId": turn_id})
```
