# App Server

The App Server exposes sessions over JSON-RPC 2.0. A thread is a session, a
turn is a retained kernel turn, and an item is a projection of a durable event.
The log and inbox own execution. Connections subscribe to it; they do not own it.
The host supplies authentication, authorize(connection_id, method, params), and
wake scheduling. A refusal returns -32027 permission_denied before dispatch.
DefaultAppServerHost is local and single-user: it allows every connection.

## Startup

```bash
uv run vv-agent app-server --listen stdio \
  --settings settings.json --backend moonshot --model kimi-k3
```

Settings, backend, model and transport must each be provided once. They resolve
before the stdio loop starts. The model timeout defaults to 90 seconds and
approval timeout to 30 seconds. Live provider tests are opt-in.

Every line is one JSON object. Requests and notifications carry
`jsonrpc: "2.0"`. Initialize each new connection before other requests:

```jsonl
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"host"}}}
{"jsonrpc":"2.0","method":"initialized"}
```

The response reports `protocolVersion: "v3"`. Schema exports describe the
current closed wire shapes. Regenerate client bindings when upgrading.
Malformed JSON returns `-32700`; an invalid message returns `-32600`.

## Stable submissions

Choose a globally unique `inputId` for thread creation. The same value becomes
the `threadId`; creation does not list existing sessions. Keep IDs in the host's
business record before submitting, so retries reuse them. Client inputId and
actionId values contain 1..128 ASCII letters, digits, `.`, `_`, `:` or `-`.
Slashes and whitespace are rejected, keeping internal session and inbox IDs private.

```jsonl
{"jsonrpc":"2.0","id":2,"method":"thread/start","params":{"inputId":"thread_1","agentKey":"default","cwd":"./workspace"}}
{"jsonrpc":"2.0","id":3,"method":"turn/start","params":{"threadId":"thread_1","inputId":"prompt_1","expectedTurnId":null,"input":[{"type":"text","text":"Inspect the workspace."}]}}
```

`thread/start` returns threadId, agentKey, cwd, current status, accepted and
replayed. `turn/start`, `turn/steer` and `turn/followUp` return threadId, turnId,
inputId, inputSeq, accepted and replayed. Accepted means the input was committed,
not that execution finished. The stable input ID is stored unchanged in the
inbox. JSON-RPC request IDs and connection IDs never enter its bytes.

Repeat the same ID and body to receive the original admission with
`replayed: true`, even after the turn finishes. A different body returns
`-32024 submission_conflict`. For turn/start, expectedTurnId is null for the
first turn and otherwise the last admitted turn. It also rejects an active turn.
Steer requires the active turn; follow-up requires the last admitted turn and
queues the next turn. Stale expected turns return `-32025 turn_mismatch`.

```jsonl
{"jsonrpc":"2.0","id":4,"method":"turn/steer","params":{"threadId":"thread_1","inputId":"context_1","expectedTurnId":"thread_1/turn/prompt_1","input":[{"type":"text","text":"Check tests too."}]}}
{"jsonrpc":"2.0","id":5,"method":"turn/followUp","params":{"threadId":"thread_1","inputId":"next_1","expectedTurnId":"thread_1/turn/prompt_1","input":[{"type":"text","text":"Write the release note."}]}}
```

## Control and resume

Any authorized connection, including one on another API node, can submit
`turn/action`. The closed actions are cancel, suspend, resume, abort, and respond.
Respond takes `message: {role: "user", content: "..."}`. The server derives a
stable inbox command from threadId, turnId and actionId. Retried actions do not
repeat side effects. Replies keep their retained interaction target.

```jsonl
{"jsonrpc":"2.0","id":6,"method":"turn/action","params":{"threadId":"thread_1","turnId":"thread_1/turn/prompt_1","actionId":"pause_1","action":{"kind":"suspend"}}}
{"jsonrpc":"2.0","id":7,"method":"turn/interrupt","params":{"threadId":"thread_1","expectedTurnId":"thread_1/turn/prompt_1","actionId":"cancel_1"}}
{"jsonrpc":"2.0","id":8,"method":"turn/resume","params":{"threadId":"thread_1","turnId":"thread_1/turn/prompt_1"}}
```

Interrupt submits durable cancel, including when that API node has no handle.
Its result is the turn/action receipt. Resume checks durable execution state,
reuses an available local handle and respects the session lease. It never needs
the original connection. A terminal turn returns its retained result.

`thread/read` returns a full snapshot with thread, turns and items.
`thread/resume` returns the same snapshot and subscribes by default; use
`subscribe: false` for a read-only snapshot. Neither accepts afterItemId.
`thread/unsubscribe` only removes a subscription and returns only threadId and subscribed=false;
leaving a connection never closes a durable session. Closed sessions reject
execution; archived sessions return `-32021`. `thread/list` and `thread/archive`
retain their normal lifecycle roles.

## Bounded replay

Use `thread/events/list` for a long thread. Supply threadId and a limit from 1
to 256, plus optional afterCursor and throughCursor. The result contains events,
nextCursor, throughCursor and more. Each event entry contains its cursor and
the canonical RunEvent wire.

A cursor has threadId, a durable record seq, an event offset within that record,
and recordDigest (SHA-256 of the canonical record bytes). Offset counts events already consumed; zero means before the
first event. This handles several events from one record without losing any.
The returned throughCursor fixes the head captured by the first page. Reuse it
on later pages for a stable replay window.

Each page reads at most limit records and emits at most limit events, plus
indexed plan dependencies and cursor checks. Records that emit no events may
produce an empty page with more=true; continue from nextCursor. Unknown,
cross-thread, stale and malformed cursors return `-32026 invalid_cursor`.

Persist the last cursor and deduplicate by event_id after reconnect. Live deltas
are volatile and have no durable cursor. Reconcile their text or arguments with
the final durable event/item using callId or toolCallId. Item IDs identify events,
not a mutable UI entity: `item_{event_id}`.

## Execution status

`thread/status` and `thread/status/changed` contain exactly:

- threadId, status and kernel phase;
- waitKind: provider, child, user, approval, retry, or null;
- nextDriveAt: Unix milliseconds, or null when unknown;
- pendingInteraction: sessionId, turnId and optional prompt/interactionId/requestId, or null;
- unconfirmedCount: operations in the current turn (the latest turn when terminal) with an attempt that has no definitive completion;
- projectionWatermark: the last record sequence used for this projection.

Statuses distinguish provider_running, child_running, user_wait, approval_wait,
retry_backoff, suspended and terminal. Idle means no turn yet; running means
work is admitted or ready. Archived and closed retain their lifecycle meanings.
A child can expose a pending interaction while the parent is child_running.
No provider handles, job secrets, arguments or digests appear in status.

A turn/completed notification can describe an interrupted execution attempt.
Only a retained turn_ended record makes a turn terminal. Each node tails durable
turn records for lifecycle notifications, including follow-ups. A non-holder
attempts the lease once and schedules a retry at the holder's lease expiry,
including when a foreground child is blocked. Renewal, record progress or a
new holder epoch keep recovery pending; a live lease postpones the check until
its new expiry without another acquire. This also recovers when a replacement
holder dies. An acquisition that still fails after expiry emits
error/warning(code=lease_retry_exhausted) and stops that retry.
Child result delivery wakes its parent after commit. Embedded hosts can call
AppServer.wake(thread_id) after an inbox commit. Wake and retries use a currently
registered subscriber, or execute without subscribing if none exists. Approvals
then remain durably parked for approval/resolve from any authorized connection,
subject to the retained deadline. Notification
observers never acquire execution leases. Read status or replay
records to recover after missing a notification.
Each subscribed session backs off independently from 50 ms to 1 s, including
after transient store failures. The last local unsubscribe drops its position;
the next subscription starts at the head or the active turn's start. Use
thread/events/list for older history.

## Models

`model/list` lists the host's configured models. `thread/model/select` takes
threadId, stable inputId and selection (or null to restore the frozen default).
See [session model selection](session-model-selection.md) for application
boundaries, retained retry plans and billing.

`AppServerHost.resolve_model(request)` returns a client and resolved config
without submitting a provider call. DefaultAppServerHost resolves the exact provider model ID
through its configured ModelProvider and accepts only an empty route. A settings
key is not a model ID: `{"a": {"id": "a-2024"}}` is selected as `a-2024`.
The configured backend is filtered first. Selection and model/list use the resolved
model_id, including a first endpoint's model_id override. Zero or multiple matches
within that effective backend are logged as model_selection_unresolvable.
The frozen model with an empty route reuses its client and config before any
host resolver. Custom hosts enforce model routing permissions in this hook.
CLI and interactive use the same default provider resolver through SessionDriver;
only App Server binds AppServerHost.resolve_model. Bill from each retained
model call's actual model/backend and call ID.

## Approval

The durable approval wait retains request, operation, attempt, scope, request
digest and absolute deadline. The server emits approval/requested before the
approval/request callback. Any authorized connection can use approval/resolve
with threadId, turnId, requestId and decision (allow, allow_session, deny or
timeout), plus optional reason and metadata. The server binds the answer to the
retained wait; callers cannot supply operation or digest fields. A disconnected
subscriber does not change the deadline or remove the wait.

## Items, notifications and exports

Render userMessage, agentMessage, modelCall, toolCall, approval and error items
from item/started and item/completed. Deltas append volatile progress. Planning a
tool call alone emits no item. Tool completion carries executionStarted,
directive, errorCode and durationMs. Reasoning is private telemetry.

Notification opt-out accepts exact method names in initialize's
capabilities.optOutNotificationMethods. It affects only that connection.

```bash
uv run python -m vv_agent app-server schema --out /tmp/vv-agent-schema
uv run python -m vv_agent app-server generate-ts --out /tmp/vv-agent-schema/typescript
```

schema/export returns the same self-contained JSON Schema and TypeScript
bundles. Tests in test_app_server_distributed.py, test_app_server_model_selection.py
and the App Server producer suites pin these behaviors.
