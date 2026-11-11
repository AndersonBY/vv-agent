# Python contract integration

`vv-agent` is the Python implementation of a language-neutral contract that
lives in [`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
That repository owns the normative behavior, the fixtures, the versioning rules
and the adoption workflow. This page explains how this repository consumes it.

## Pinned contract

`contract.lock.json` pins one contract release. It records:

- the contract version and Git revision;
- the immutable release artifact URL and its SHA-256 digest;
- the path of the vendored fixtures (`tests/fixtures/parity/`);
- the digest of the canonical fixture manifest.

`tests/fixtures/parity/` is a generated copy of the contract fixtures. Do not
edit a vendored fixture or digest by hand; resync it instead (see below).

This page does not track adoption status. The machine-readable source for that
is [`support-matrix.json`](https://github.com/AndersonBY/vv-agent-contract/blob/main/support-matrix.json)
in the contract repository. It lists the required implementations, the
implementation revision each one was verified at, the verification time, the
central CI run URL, and any frozen baselines. The support matrix uses schema 2,
which names `required_implementations` explicitly and records a
`contract_version` per implementation. The tooling rejects schema 1.

Python is the only required implementation. The Rust implementation
(`vv-agent-rs`) is frozen at contract 23.0.0 and package series 0.21.x. It gets
maintenance fixes only and does not follow new contract releases. A newer
contract version, or a new Python verification run, says nothing about Rust.
Bringing Rust back needs a new maintainer decision and full adoption of the
contract that is current at that time, through the central workflow.

## Required workflow

Follow these steps for any change that affects the public API, what the model
sees, runtime behavior, persisted data, or a wire format:

1. Read `contract.lock.json` in this repository and `../vv-agent-contract/AGENTS.md`.
2. Read the parity, versioning and change-workflow documents in the contract repository.
3. Change the canonical docs and fixtures in `vv-agent-contract` first.
4. Sync the vendored snapshot with `scripts/contract_snapshot.py sync`. Leave the
   frozen Rust lock and fixtures untouched.
5. Update the real Python producers and their behavior tests. Updating only the
   fixture parsers is not enough.
6. Run the full Python test and lint gates, then the central cross-repository CI
   with the contract and Python revisions. Rust adoption and Rust gates are not
   required.

## Snapshot commands

Check the lock and the vendored snapshot:

```bash
python3 scripts/contract_snapshot.py check
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```

The second form also compares the snapshot byte-for-byte with a local checkout
of the contract repository.

Resync the snapshot once an immutable contract release exists:

```bash
python3 scripts/contract_snapshot.py sync \
  --source ../vv-agent-contract \
  --artifact https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip \
  --revision <contract-revision>
```

Before a package release, the release workflow requires that the support
matrix lists Python as verified at a revision contained in the release commit:

```bash
python3 scripts/contract_snapshot.py adoption --implementation python --revision "$(git rev-parse HEAD)"
```

## What the tests prove

Fixture and snapshot checks prove only that the vendored files are intact. The
behavior comes from producer tests: tests that run the real Python code and
compare its output with the contract. They cover the public API, runtime
decisions, events, strict wire readers and recovery under the pinned contract.
Producer and store tests run the execution rules against both SQLite and
PostgreSQL.

`scripts/session_kernel_fixtures.py` regenerates the session kernel fixtures
from the real producers. `tests/session/test_session_kernel_fixtures.py`
compares that output byte-for-byte with the vendored snapshot. The snapshot
integrity check in `contract_snapshot.py` runs separately.

An implementation counts as verified only after its producer tests and the
central gates pass. Passing fixture checks alone is not verification.

## Python producer map

Model rejection uses `llm/errors.py` for the adapter-owned
`ModelRequestRejected` type and conservative HTTP response recognition.
`llm/http_observation.py` observes physical sends across vendor SDK retries;
unobservable host HTTP exceptions stay Unknown without the explicit assertion.
An observed 4xx remains pending until its raw body reaches EOF and decoding
finishes successfully; early close and decoder failure cannot settle a call.
The complete exception graph vetoes uncertainty before positive classification;
installed provider-native connection and timeout classes participate even without
a lower-level HTTP cause. Physical-send failures keep their own indices; an
unindexed failure permanently vetoes that invocation. Vendor network defaults
and host proxy/timeout priority remain intact, and callback exit closes iterators
and response streams explicitly.
`session/kernel.py` records the definitive receipt and terminal,
`session/reducer.py` retains failed receipts across retry, and
`session/projection.py` carries the safe cause to hosts and App Server users.
The kernel checks prompt-too-long recovery first, including real HTTP 4xx
context errors, structured provider codes/types and HTTP 413. Content rejections
stop without endpoint failover. 401/403/404 immediately try the
next frozen endpoint only when multiple endpoints exist; 429 retains backoff and
one retry even with a single endpoint. Explicit adapter retryability overrides
these defaults. Unknown retains its separate retry rules. The normative
classification order, content string limits and summary failure behavior live in the contract's
`docs/model-rejection.md`.
`scripts/model_rejection_fixtures.py` and the matching rejection tests pin this
behavior. Contract 35.0.0 adds no store or frozen-definition boundary.

During development, `VV_AGENT_CONTRACT_SOURCE=/path/to/vv-agent-contract`
selects a local canonical checkout for producer tests without rewriting the
immutable release lock or vendored snapshot. After publication, synchronize
them with the commands above. A local gate is not release or central adoption
evidence.

Paths are relative to `src/vv_agent/` and `tests/`.

| Surface | Producer | Tests |
| --- | --- | --- |
| Public API v15 | package exports, `Runner`, `RunConfig`, interactive sessions, App Server | `test_parity_evidence_manifests.py` |
| Strict wire formats | `events.py`, `types.py`, `message_codec.py`, `app_server/protocol/` | event validation, protocol type and session codec tests |
| Prompt and run definition | `runtime/compiler.py`, `prompt/`, `session/runtime.py` | `test_run_definition_producer.py`, prompt and session fixture tests |
| Execution and recovery | `session/kernel.py`, `session/sql.py`, `session/reducer.py` | `tests/session/`, App Server durable resume tests |
| Events, results and tracing | `session/projection.py`, `session/result.py`, `session/tracing.py` | Runner event, terminal, trace and result tests |
| Child sessions and handoff | `session/delegation.py`, `session/children.py` | configured child, continuation, agent tool and handoff tests |
| Memory and compaction | `session/compaction.py`, `session/memory.py`, `memory/` | memory lifecycle, local compaction and token usage tests |
| Budgets | `budget.py`, `session/kernel.py` | `test_run_budget.py`, budget and session fixtures |
| Model selection | `session/kernel.py`, `session/runtime.py`, `session/app_server.py` | `tests/session/test_model_selection.py`, `test_app_server_model_selection.py` |
| Provider usage | `session/kernel.py`, `session/providers.py` | `tests/session/test_provider_usage.py`, `tests/session/test_recovery_matrix.py` |
| Session supervision and dispatch | `session/supervisor.py`, `session/children.py`, `session/kernel.py` | `tests/session/test_batch_children.py`, `test_supervisor.py`, `test_dispatch.py` |
| App Server v5 | `app_server/`, `session/app_server.py` | schema, protocol, turn, approval, replay and lifecycle tests |
| App Server result reads | `session/tool_results.py`, `session/tool_result_read.py`, `workspace/resources.py` | `session/test_app_server_tool_results.py`, `app_server_tool_results.json` |
| Workspace and tool results | `tools/`, `workspace/`, `runtime/tool_results.py` | workspace parity, Bash, custom tool and metadata tests |

The contract 32.1 bounded-result producer is `session/tool_results.py`, called by
`Kernel.completed` before receipt construction. It applies the existing 12,000
character head/tail limit once, streams long receipt text through
`write_text_chunks_exclusive`, checks new-write byte counts and available file
size metadata, and adds the reserved
`metadata.vv_artifact` entry. Archive failures retain successful tool outcomes
and expose an unavailable expected recovery object. Reused objects and existing
producer artifacts still undergo streaming UTF-8/size/digest validation. New
completion notices include the expected path, source character counts and
read_file/search_files recovery hints within the same 12,000-character budget.
Existing producer previews keep their original marker. Existing producer captures
have `source_complete=false` with `capture_completeness_unverified`; preview
truncation does not prove provider capture loss. `session/reducer.py` binds the
reserved source to the admitted operation and arguments. Record 1 and inbox 3
are unchanged. `scripts/tool_result_artifact_fixtures.py` and
`tests/session/test_tool_result_artifacts.py` cover these real producers,
identity conflicts and process death around archive and receipt persistence.
For local unpublished contract work, `VV_AGENT_CONTRACT_SOURCE` selects canonical
producer fixtures and built-in schema generation from that checkout. It does not
update the release lock or vendored snapshot. Use this variable for the full local
producer gate until the new immutable release exists.

Artifact search is produced by `tools/handlers/search.py`, which routes reserved
explicit paths before discovery options. It uses the public
`is_task_artifact_path(path, *, task_id, session_id=None, session_task_id=None)`
predicate shared with model-facing read_file, file_info and read_image. The
runtime supplies the session id separately from model arguments. The accepted
scopes are the current host task (or SDK task without a host id) and the exact
current session/SDK-task compaction scope. A scope match never replaces host ACLs.
`_session_artifact_scope` carries a session/task tuple inside Python. Hash input
is compact Unicode JSON `["session", sid, tid]` for that tuple and `["task", id]`
for other task ids requiring hashing. Short readable task ids remain literal
unless prefixed s-. Hashes use s- plus 62 lowercase SHA-256 hex characters;
tool-call filename encoding is unchanged.

`workspace/streaming.py` incrementally decodes UTF-8. Search retains a 30,000
character window with 4,096 characters of overlap and a bounded page of match
excerpts. Content rows contain one-based line numbers and zero-based Unicode
columns. offset/head_limit paginate match items, including repeated hits on one
line. Long lines are always scanned and explicitly marked incomplete for
patterns wider than the overlap or sensitive to window boundaries. The same
line and column can be passed as read_file start_line/start_char. A leading BOM
is excluded from both coordinate systems. The existing digest-bound cursor is
still the continuation mechanism across lines.

`workspace/matching.py` uses regex VERSION0 with an interruptible operation
timeout of 250 ms per engine invocation (finditer, match or search over one
line, window or whole text). A finditer timeout covers its iteration rather than
resetting per match. A request shares a separate 10 s thread CPU deadline across
enumeration, globs, child exclusion filters and portable content matching. Time
in storage I/O wait is not charged as thread CPU. The engine's process CPU timer
may interrupt earlier under contention. Each operation receives the smaller of
250 ms and the remaining request budget. Total exhaustion reports
"Search total CPU limit exceeded (10 s); aborted", separately from the single-call
"Regular expression too slow; aborted" notice.
These limits count CPU time. If the host starves the search of CPU, a single
call can take more than 250 ms of wall time; vv-agent does not promise a wall-time
limit for searches.
S3 checks the same deadline within each page and object iteration, including
empty pages. Local checks within directory and file iteration, including empty
directories, before traversal can advance. Ordinary S3 shell globs retain the
platform's fnmatch normcase behavior; the separate recursive glob branch keeps
its existing case-sensitive rules.
Content timeouts preserve earlier hits and the start of the interrupted window;
discovery timeouts are explicit tool errors. Native ripgrep retains its bounded
subprocess and non-backtracking matching. A native listing process is killed
and reaped if its Python glob matcher times out.

VERSION0 uses simple case folding. Scoped/leading inline flags, variable-width
lookbehind, Unicode property escapes and global flags after a prefix are
accepted. REVERSE mode is rejected after compilation, including explicit flags.
Unicode word and dotless-i changes are listed in [upgrading](upgrading.md).
Terminal capture failures preserve the process outcome with metadata.retention
unavailable and no recovery pointer. Failure handlers never reread capture;
UUID exclusive paths preserve partial objects. Head, notice and tail remain
separate while previews shrink, with a strict-decrease assertion and loop bound.

Producers and checks are `scripts/artifact_search_fixtures.py`,
`tests/session/test_artifact_search.py` and
`tests/session/test_artifact_reread_gaps.py`. They cover real kernel dispatch,
compaction scope, line/column readback, pathological regex interruption, Unicode
chunk boundaries, partial edit baselines and memory bounds. Terminal retention
failures use ordinary results with unavailable metadata; real Bash and
kernel tests prove that status, exit code and directive survive and the command
is not repeated. The immutable snapshot is still the published baseline until
release synchronization; local gates do not establish central adoption.

The workspace edit producer is `tools/handlers/workspace_io.py`. It builds the
canonical edit receipt and file metadata. `test_builtin_tool_behavior_contract.py`
checks it against the central success fixture. The workspace tool tests cover
consecutive edits on large files, stale baselines, exact replacement, and
preservation of a UTF-8 BOM and CRLF line endings.

## Current boundaries

- RunEvent v6, model-call v2, task-token-usage v3 and the strict `Message` shape
  each accept exactly one current version. A missing, older, unknown or
  malformed version is rejected. There is no decoder for contract 23 data. See
  [upgrading.md](upgrading.md#session-data-and-upgrades) for what this means for
  stored data.
- Results carry `session_id` and `turn_id`.
- The App Server speaks protocol v5 and reports thread state with one
  `ThreadStatus` enum.
- The session log is the source of truth for execution. Events, results, the
  App Server and tracing are projections built from it. JSONL event stores only
  receive copies.
- Recovery within the same turn reuses the prompt, logical model, limits and usage that
  were frozen when the turn started.
- Durable stores are opt-in through the session API.
- Inbox schema 3 requires provider usage that the provider has authenticated.

## Model selection

Contract 27 adds session model selection. A host pushes
`InboxItem(kind="model_selection", payload={"selection": ...})` with
`SessionTx.push`. The kernel passes the retained request to
`Runtime.model_resolver`, which returns `(LLMClient, ResolvedModelConfig)`.
`Runtime.resolve_model` then checks the logical model identity against the binding
recorded in the log. App Server clients use `AppServerClient.select_model` with
`ThreadModelSelectParams`, which sends `thread/model/select`.

[session-model-selection.md](session-model-selection.md) describes the full
rules.

## App Server A1

Contract 28 replaces the wire with v3: stable submission receipts, controls and
approvals across connections, bounded record cursor replay and rich status.
`SessionDriver.model_resolver` binds AppServerHost.resolve_model; its provider
default is also used by CLI and interactive sessions. Read ranges use indexed
projection dependencies. The SQLite and PostgreSQL DDL, record schema 1 and
inbox schema 3 stay unchanged. This release is not a store boundary.
Embedded hosts use AppServer.wake(thread_id) after commit. The built-in host
retries blocked leases at expiry, deferring to renewed or replacement holders
until their new expiry. Child delivery wakes the parent after commit. Node observers use independent
per-session backoff without holding store locks across projection or transport I/O.
The public API inventory records the wake entry point; Rust remains frozen.

## Endpoint routing

Contract 29 keeps endpoint IDs out of model and definition identity. Python
uses `Runtime.resolve_model` to obtain current primary clients and
`VvLlmClient.ordered_targets` to freeze each operation's routing. Historical
endpoint fields remain readable evidence. Record schema 1 and inbox schema 3
are unchanged. See [model selection](session-model-selection.md#resolver-contract).

## A2 context mapping

Contract 30.0.0 uses App Server v4 and public API inventory v14. Python maps
thread/context/read to session.context_view and shares session.compaction's
memory_settings resolver with execution. thread/compact/start writes the existing
content extension and feeds summary_requests; it adds no inbox kind or store
version. Typed client methods are read_context and compact_thread. The canonical
A2 vectors are reproduced by scripts/app_server_a2_fixtures.py.

### Completed-boundary forks

Contract 30.1.0 adds thread/fork under App Server v4 without a store boundary.
ThreadForkParams maps to AppServerClient.fork_thread. The canonical rules are in
the contract session-kernel document; Python uses the same context_messages and
project_view functions for source cuts and context reads. Creation uses the
existing transaction create receipt. The app_server_fork attribute extension
stores lineage/default model; seed retains its closed messages/shared_state shape.
Real producer and SQL tests are in tests/test_app_server_fork.py and
tests/session/test_app_server_fork.py. No Rust adoption is required.


Public interaction mappings are `AppServerUserInteraction`,
`InteractionAnswerSpec`, `InteractionReplyTarget`, and `InteractionAnswer`.
`HeartbeatStopped` is the public intentional heartbeat-stop signal. The current
App Server v5 snapshot adds `state`, and respond uses target plus answer.
These mappings preserve record 1, inbox 3, SQLite 3 and RunEvent v6.

The model HTTP observer pre-reads every non-408 4xx within a total deadline of
min(request read timeout, 10 seconds) and 1 MiB for raw and decoded bytes. Either
exceeded bound stays Unknown even if the vendor later finishes reading. Successful
content is cached; failures are deferred to body consumption, while bounded
prefixes and the remaining raw stream are replayed. Status, headers and native
vendor retry decisions, counts and backoff remain unchanged. See the contract's
model-rejection rules and `tests/session/test_model_rejection_retry_policy.py`.
The response hook pre-reads synchronously without creating threads. It copies
the request timeout extension before raw iteration and caps read at the
observation deadline. Total extra network wait is at most that deadline plus
one recv timeout (20 seconds maximum); remaining reads retain the shortened
per-recv timeout. `tests/test_model_rejection_lifecycle.py` covers real stalled
HTTP/1.1 sockets, repeated closes, connection-pool recovery and resource counts.

## Context and quota classification

Contract 36.0.0 corrects generic input-size matching and narrows HTTP 429 to an
explicit context allowlist. `llm/errors.py` owns the shared rule used by
`is_prompt_too_long_error` and the original-message flag on ModelRequestRejected.
Kernel primary, compaction, session-memory and summary requests consume that
classification through the existing model boundary; no separate caller heuristic
is added. Provider-field and exception-text sources apply quota vetoes locally,
while uncertainty still vetoes the whole graph. The window phrases are explicit
outside 429; `maximum context length` already covers the numeric model-limit form.

`tests/test_model_rejection.py`, `tests/test_model_rejection_lifecycle.py` and
`scripts/model_rejection_fixtures.py` cover the rules and real provider transports.
The lock now selects 37.0.0 (below).

## Contract 37.0.0 input protection

The lock and vendored snapshot pin the v37.0.0 release (`17d26dd`); local
producer validation can still select a contract checkout with
`VV_AGENT_CONTRACT_SOURCE`. Fold incrementally maintains coverage and protected source
positions in its disposable prefix projection, sharing the message cache;
`session.context.protected_start` reads that same state for execution and replay.
No coverage ledger is persisted. `UnseenInputError.code` maps to
`unseen_input_removed`, and the capacity terminal reason is
`unseen_input_capacity_exceeded` (`UnseenInputCapacityError.code`).
Only first creation defaults on; transaction creation replay with an omitted
choice preserves the original bytes and rule. Children and forks inherit the
source choice. External tool provider callbacks protect their original assistant
group, like blocking child results. The complete inbox classification is in the
canonical input-protection document.

Declared sessions use that same committed-prefix message projection for actual
requests and cold replay, including normal tool callbacks after a before-memory
snapshot. Replacement remaps all verbatim retained operation groups, including
currently covered groups. Native child correction is governed by the generic
non-model receipt branch; normal native-child execution cannot currently reach
it, and provider correction tests do not prove a native-child scenario.

`scripts/unseen_input_fixtures.py` includes local HTTP through the real
`VvLlmClient` and kernel for normal primary, PTL recovery, endpoint retry and
Unknown. Port and clocks are frozen before producing deterministic bytes.
Rust is unsupported for this feature and retains its frozen 23.0.0 baseline.


## Contract 37.0.0 completion evidence

`VvLlmClient._with_completion_metadata` reads final CompletionStream metadata
only after consumption (and metadata on non-stream responses). It maps missing
termination to ModelStreamInterrupted, and fills empty assistant content from
refusal text unless content_filter. `llm.errors.model_response_rejection` creates
the safe HTTP 200 descriptor; the kernel gates it on the independent
`response_rejection` creation seed, before after-LLM hooks. Fully consumed
response rejection is definitive despite earlier deltas; exception-based request
rejection retains the observed-output uncertainty veto. Safe result/event/terminal
projections and bounded Unknown retries use existing paths.
`scripts/model_completion_fixtures.py` and
`tests/session/test_model_completion_evidence.py` prove local HTTP through the
real vv-llm adapter/kernel, independent seed replay and real 0.33.3 SQLite and
PostgreSQL rollback. The closed seed policies share default/replay/inheritance
logic in `session.records`; no record, inbox or SQL version changes.

## Contract 38.0.0 local candidate

Python 0.35.0 adds yieldable waits in the SQL kernel and L1 lease lock ordering.
The canonical rule is `docs/yieldable-waits.md` in the contract repository.
Local App Server, Interactive and Runner retain declarations but wait for real
results; they do not yield. Shared raw connections require exclusive host use,
and concurrent native transactions are unsupported without promised detection.
Admission and hook callback signatures are unchanged.

For the unpublished candidate, run producer and focused regression tests with
`VV_AGENT_CONTRACT_SOURCE=/path/to/the/contract-38-checkout`. This selects the
candidate's fixtures through the existing local-source mechanism. The release
lock and its matching vendored snapshot remain together on the immutable 37.0.0
release until a reviewed contract revision and artifact exist. After publication,
run the snapshot sync command above with that actual revision and artifact, then
rerun producer gates against the synchronized snapshot without the override.
Local validation is not immutable-lock synchronization or central adoption.

`scripts/yieldable_wait_fixtures.py` produces child/provider yielding, recovered
acceptance, terminal-first ordering and model projection from real SQL commits.
`scripts/session_kernel_fixtures.py` also produces the optional closed fields and
codec negatives. The public inventory is generated from current declarations
and signatures; optional provider `on_yield(plan, handle, reason)` is a behavioral
extension consumed by the kernel and exercised by the yield producer. It adds
no connection checkout or borrowed-transaction surface.
