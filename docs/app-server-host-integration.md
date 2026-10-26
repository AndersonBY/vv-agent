# App Server host integration

This guide is for a desktop app, worker, IDE or service that runs the
`vv-agent` App Server as a child process. The protocol itself is in
[App Server](app-server.md).

## Process ownership

- The host owns the process: lifetime, restart policy, stdio pipes, health
  checks and log capture. It also owns authentication, accounts, billing,
  permissions and business task records.
- Pin one `vv-agent` release. Deploy protocol changes together with the
  matching generated schemas or TypeScript bindings.
- Keep reading stdout while work is active. A slow reader can block stdio
  delivery; the host owns pipe timeouts and restart policy. Broken transports
  are disconnected. A blocked notification write holds no session-store lock.

## Host boundary

- `authorize(connection_id, method, params)` runs before every thread read or
  action, including listing and creation. Return false to reject with typed
  `-32027 permission_denied`. Thread listing also checks each returned thread.
  Distributed hosts must implement account and thread permissions here.
  DefaultAppServerHost is a local, single-user host and allows everything.
  For thread/start, authorize against params.inputId: that exact value becomes
  threadId. Validate the requested agentKey, cwd and metadata for that account too.
- `AppServerHost.resolve_agent()` maps an `agent_key` to the current `Agent`
  instructions and tools.
- `build_run_config()` supplies context providers, memory providers, workspace
  storage, approvals, tracing metadata, event persistence and execution policy.
- `resolve_model(request)` resolves the exact provider model ID to a client and config
  without calling a provider. Enforce model and route permissions here. The frozen
  model with an empty route reuses its frozen binding before this hook.
- `list_models()` reads the same canonical model settings that the host's
  scheduler or UI uses.
- Browser, IM, notifications, product memory, release management and other
  product modules stay outside the framework.

## Client lifecycle

1. Start `vv-agent app-server --listen stdio` with explicit settings, backend
   and model arguments.
2. Send `initialize`, read its response, then send the `initialized`
   notification.
3. Use `thread/start` for a new conversation and `thread/resume` for an
   existing durable thread.
4. Use `turn/start` for prompts, `turn/steer` for input to the active turn,
   `turn/followUp` for queued input, and `turn/interrupt` to cancel.
5. Use `thread/model/select` when the user changes model settings. Give each
   user command a stable `inputId`, so a retried request is a replay rather
   than a second change.
6. Read thread/context/read to show the durable context budget. Keep its
   contextVersion when asking thread/compact/start to reject a stale view. Give
   compaction its own stable actionId and follow contextCompaction notifications.
   A replay returns the original durable request identity.
7. Page durable history with `thread/events/list`. Persist its nextCursor and
   retain throughCursor across pages. Reconcile volatile deltas with final items
   by callId or toolCallId; deduplicate durable events by event_id.
8. Route `approval/request` to the host UI or policy service and return one
   canonical decision.

## Reliability

- Treat JSON-RPC error `-32001` as retryable, with bounded exponential backoff.
- Keep the original business job identity across retries.
- Treat parse errors as failures of a single line: record `-32700` and go on
  with later valid JSONL messages.
- Keep stable inputIds for thread/start, turn/start, steer and followUp. A lost
  response is retried with the same body and ID, on any authorized connection.
- Approval answers may come from another node. The retained request binds the
  operation and digest; a connection is only a subscriber.
- Share the session store between API nodes and workers. Resume from execution
  state. In an embedded host call `server.wake(thread_id)` after the outer inbox
  transaction commits, or from the scheduler's due-work scan. This trusted host
  method returns None and tries start-or-attach; it does not accept untrusted wire
  requests or replace authorize. For example:

  ```python
  with store.atomic() as tx:
      tx.push(thread_id, inbox_item)
  server.wake(thread_id)
  ```

  Scan due work as a fallback
  when a wake is lost. Use rich thread/status for machine and human waits.
- The built-in server retries a blocked drive at the observed holder's lease
  expiry. Renewal, progress and replacement epochs keep recovery pending;
  while a live lease exists it waits for the new expiry without acquiring.
  Another failed acquisition after expiry emits error/warning(code=lease_retry_exhausted).
  A foreground child's blocked lease follows the same rule, and committed child
  delivery wakes the parent. Wake and retries select a registered subscriber;
  without one, execution creates no subscription and approvals wait durably for
  approval/resolve from any authorized connection, subject to their retained deadline.
- Each API node observes durable turn records with independent per-session
  polling backoff (50 ms to 1 s), at most 32 due sessions and 256 records per
  session per pass. Transient read errors warn at most once per second per session,
  back off and keep observing. Snapshot record reads do not lock session rows;
  projection, host stream callbacks and transport writes run outside store
  transactions and connection locks. Close its SessionDriver
  before closing a borrowed store. Local pumps flush notifications when joined.
- Unsubscribing does not close the session. Hosts close sessions explicitly
  through the kernel when their business lifecycle ends.
- The last local unsubscribe or disconnect releases its notification position.
  Re-subscription begins at the head, or just before the active turn's start.
  Page past turns with thread/events/list; they are not replayed as live notices.
- thread/list authorizes session IDs before loading metadata and never projects
  full turn results or items. Sorting and archive filtering scan the creation row
  and consumed control facts for each authorized session, then fold state only
  for the requested page. offset/limit bound those folds and the response, not
  the catalog scan; avoid frequent global listing on a large store.

## Custom stores

The SQL stores implement the App Server reads and admission lock. A custom
SessionStore also implements `read_input(session_id, input_id)`, including
consumed inputs, and `projection_context(session_id, records)`, which returns
only the indexed dependencies needed to project those records. These reads
must preserve the page bound. `SessionTx.lock_session(session_id)` serializes
admission with other writers until the outer transaction commits. None of
these methods adds a storage schema or another execution log.
`lease_status(session_id)` returns (head sequence, lease epoch, remaining lease
milliseconds), measured using database time, without a session row lock.
`read()` captures a head and reads immutable records by seq without a write
transaction or row lock; it never reads beyond that captured head.
`control_records(session_id, through_seq=...)` returns consumed control records
through the captured head, without folding history or locking writers, for catalog
archive filtering. It includes rejected controls so callers check disposition.

## What to test

- An integration fixture covers initialize, thread start and resume, turn
  start, progress persistence, approval, cancellation, replay and reconnect.
- A process restart neither duplicates completed items nor loses pending
  approval state.
- Denied and timed-out approvals never run the protected tool.
- A slow or disconnected client does not leave an active turn or an approval
  request stuck in the host UI.

## Context reads and control intent

Context reads use immutable log reads and a stable inbox watermark. Custom stores
accept peek_inbox(..., ready_only=False) to include every accepted pending input,
including inputs whose available_ms is in the future. Execution keeps the default
ready_only=True. This prevents the context version from depending on wall time. They do not
use read_state's PostgreSQL row lock. A busy cut can require a retry. They report
the durable view before future callbacks, rather than running callbacks to guess
what they will add. Thread creation retains a compiled baseline without running
input guardrails or context providers. Compaction-only turns also skip prompt
and final-output guardrails, since they submit no user prompt or final answer. Model
selection still uses thread/model/select.

Manual compaction stores metadata.request_compaction in the existing user-content
JSON extension, with app_server_compaction holding its closed wire params. The
kernel treats that reserved content as control intent and never as a user reply.
The App Server's ordinary turn input cannot create that envelope. A custom host
must authorize compact/start just as it authorizes turn/action, then wake only
after commit. A cancelled turn does not revive to finish compaction.

Custom model listings must supply explicit nulls for unknown capabilities. The
default catalog accepts reasoning_available, supported_reasoning_efforts,
default_reasoning_effort, prompt_cache_available, continuation_available and
input_modalities alongside the existing tool and multimodal fields. Keep provider
secrets outside these fields and outside the wire metadata.

## Task cloning

Prepare the destination workspace, snapshot or worktree in the host. Save the
source thread, forkId, cut selector and opaque workspace object in the task clone
record, then call thread/fork. Bind metadata.workspace in resolve_agent and
build_run_config to the workspace you prepared. Neither callback should infer
files from the fork thread ID. The framework does not interpret workspace fields. When workspace is omitted,
the source binding metadata is retained.
Store the returned threadId and lineage on the new task; retry with the same
params after a lost reply. Use thread/model/select if the clone needs a different
model. Start its first turn with expectedTurnId=null.

Authorize thread/fork against source ownership, creation of the derived
`fork/` + SHA-256(RFC8785([threadId,forkId])) identity, and the requested workspace.
The derived ID is not an authorization token. Keep host binding metadata free of
keys and credentials. The default local host permits all connections.

Artifact storage is separate from business files. LocalWorkspaceBackend maps
reserved artifact paths into a private root keyed by workspace; copying a business
directory alone does not copy that root. S3 and memory backends also expose the
same immutable write interface. Fork reuses readable references, or copies
missing artifact bytes through source and destination bindings at the original
relative paths, checking size and SHA-256. Existing corrupt bytes cause -32602.
No artifact_scope is rewritten: that value names future artifact writes only.
Keep shared artifact references alive for surviving forks. Failed creation can
leave reusable immutable artifacts, but cannot leave a partly created session.

Keep forked sessions on 0.29.0 or newer. Older workers can parse the extension
maps but do not apply the inherited model default. This release has no store
schema boundary and supplies no migration.
