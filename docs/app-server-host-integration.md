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
- Keep reading stdout while work is active. A slow or disconnected reader
  triggers transport backpressure, and the App Server then disconnects that
  transport.

## Host boundary

- `AppServerHost.resolve_agent()` maps an `agent_key` to the current `Agent`
  instructions and tools.
- `build_run_config()` supplies context providers, memory providers, workspace
  storage, approvals, tracing metadata, event persistence and execution policy.
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
6. Render the timeline from `item/*` notifications and replayed snapshot
   `items`. Use thread lifecycle notifications as the source of truth for
   status.
7. Route `approval/request` to the host UI or policy service and return one
   canonical decision.

## Reliability

- Treat JSON-RPC error `-32001` as retryable, with bounded exponential backoff.
- Keep the original business job identity across retries.
- Treat parse errors as failures of a single line: record `-32700` and go on
  with later valid JSONL messages.
- Persist App Server request ids for pending approvals so the right callback
  is resolved.
- After a process restart, rebuild state with `thread/read` or
  `thread/resume`.

## What to test

- An integration fixture covers initialize, thread start and resume, turn
  start, progress persistence, approval, cancellation, replay and reconnect.
- A process restart neither duplicates completed items nor loses pending
  approval state.
- Denied and timed-out approvals never run the protected tool.
- A slow or disconnected client does not leave an active turn or an approval
  request stuck in the host UI.
