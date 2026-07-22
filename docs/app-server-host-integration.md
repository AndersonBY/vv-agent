# App Server Host Integration

This guide describes the current boundary for a desktop app, worker, IDE, or
service that chooses to host the `vv-agent` App Server.

## Process Ownership

- The host owns process lifetime, restart policy, stdio pipes, health checks,
  log capture, authentication, accounts, billing, permissions, and business
  task records.
- Pin one `vv-agent` release and deploy protocol changes atomically with the
  matching generated schemas or TypeScript bindings.
- Keep reading stdout while work is active. A slow or disconnected reader can
  trigger transport backpressure and disconnect the App Server transport.

## Host Boundary

- Implement `AppServerHost.resolve_agent()` to map an `agent_key` to the current
  `Agent` instructions and tools.
- Implement `build_run_config()` for context providers, memory providers,
  workspace storage, approvals, tracing metadata, event persistence, and
  execution policy.
- Implement `list_models()` from the same canonical model settings used by the
  host scheduler or UI.
- Keep browser, IM, notifications, product memory, release management, and
  other product modules outside the framework.

## Client Lifecycle

1. Start `vv-agent app-server --listen stdio` with explicit settings, backend,
   and model arguments.
2. Send `initialize`, read its response, then send the `initialized`
   notification.
3. Use `thread/start` for a new conversation and `thread/resume` for a durable
   existing thread.
4. Use `turn/start` for prompts, `turn/steer` for active-turn input,
   `turn/followUp` for queued input, and `turn/interrupt` for cancellation.
5. Render the timeline from `item/*` notifications and replayed snapshot
   `items`; use thread lifecycle events as the source of truth for status.
6. Route `approval/request` to the host UI or policy service and return one
   canonical decision.

## Reliability

- Treat JSON-RPC error `-32001` as retryable with bounded exponential backoff.
- Keep the original business job identity across retries.
- Treat parse errors as per-line failures; record `-32700` and continue with
  later valid JSONL messages.
- Persist App Server request ids for pending approvals so the exact callback is
  resolved.
- Rebuild state with `thread/read` or `thread/resume` after process restart.

## Verification

- Cover initialize, thread start/resume, turn start, progress persistence,
  approval, cancellation, replay, and reconnect in an integration fixture.
- Confirm process restart does not duplicate completed items or lose pending
  approval state.
- Confirm denied and timed-out approvals never execute the protected tool.
- Confirm a slow or disconnected client does not leave an active turn or
  approval request stuck in the host UI.
