# v-claw App Server Migration Checklist

This checklist moves v-claw from direct interactive runtime wiring toward the
`vv-agent` App Server protocol while preserving a beta fallback path.

## Host Boundary

- [ ] Implement `AppServerHost` in v-claw and keep browser, IM, local memory,
  notification, and release/update code in v-claw.
- [ ] Map each v-claw agent profile to `AgentResolutionRequest.agent_key`,
  `cwd`, and `metadata`, then return a framework `Agent` from
  `resolve_agent()`.
- [ ] Map product tools to `RunConfig` in `build_run_config()`, including
  browser tools, file/workspace tools, notification tools, context providers,
  memory providers, execution backend, and approval timeout policy.
- [ ] Populate `list_models()` from v-claw model settings so the UI does not
  duplicate static model choices.

## Client Protocol

- [ ] Start the App Server process with `vv-agent app-server --listen stdio`
  plus explicit `--settings-file`, `--backend`, and `--model` arguments.
- [ ] Send `initialize`, then send the `initialized` notification after the
  connection is accepted.
- [ ] Use `thread/start` when the user opens a new agent thread, passing the
  selected profile key and workspace path.
- [ ] Use `thread/resume` when reopening a saved conversation or reconnecting
  after a process restart.
- [ ] Use `turn/start` for new user prompts, `turn/steer` for active-turn
  steering, `turn/followUp` for queued next prompts, and `turn/interrupt` for
  cancellation.

## UI And State

- [ ] Route `approval/request` to the existing approval UI and respond with
  `allow`, `deny`, `allow_session`, or `timeout`; `approval/resolve` with
  `allow` or `deny` is the request-only alternative to a response envelope.
- [ ] Reconcile `approval/requested`, `approval/resolved`, and `error/warning`
  notifications into the timeline and error surface.
- [ ] Render the timeline from `item/*` notifications and replayed snapshot
  `items`.
- [ ] Persist thread ids and any product thread metadata needed to reconnect
  with `thread/resume`.
- [ ] Keep the old `InteractiveAgentClient` path as fallback during beta.
- [ ] Remove the raw-log progress mapper only after parity tests pass for
  assistant deltas, tool calls, approval, cancellation, follow-up turns, replay,
  and reconnect.

## v0.5 Lifecycle Events

- [ ] Use `thread/status/changed`, `thread/archived`, and `thread/closed` as
  the source of truth for loaded thread lifecycle.
- [ ] Do not infer lifecycle state from missing deltas, UI task timers, or
  local process state.
- [ ] Use `initialize.params.capabilities.optOutNotificationMethods` to skip
  high-volume notifications such as `item/agentMessage/delta` when the UI does
  not need them.

## Verification

- [ ] Add a JSONL protocol fixture covering initialize, thread start, turn
  start, item notifications, approval, and turn completion.
- [ ] Run v-claw UI parity tests against the App Server path and the fallback
  path.
- [ ] Verify slow or disconnected clients do not leave stuck approval requests
  or active turns in the UI.
