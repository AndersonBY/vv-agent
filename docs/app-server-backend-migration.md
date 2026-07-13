# Backend App Server Migration Checklist

This checklist describes how a backend service or worker can host the
`vv-agent` App Server without moving business logic into the framework.

## Process Ownership

- [ ] Decide App Server process ownership in a worker or dedicated service.
  The owner is responsible for process lifetime, restart policy, stdout/stdin
  pipes, health checks, and log capture.
- [ ] Keep business queues, user accounts, billing, permissions, and task
  records in the backend service.
- [ ] Pin the `vv-agent` version used by workers and document the upgrade
  strategy for protocol schema changes, worker rollout, and rollback.

## Host Provider

- [ ] Inject business user context through `AppServerHost`, not through
  framework globals.
- [ ] Map backend task type or agent key to `AgentResolutionRequest` and return
  a framework `Agent` with the correct tools and instructions.
- [ ] Build `RunConfig` with backend-owned context providers, memory providers,
  workspace storage, approval policy, tracing metadata, and event persistence.
- [ ] Expose model availability through `list_models()` from the same settings
  source used by production scheduling.

## Task And Progress Mapping

- [ ] Create or resume one App Server thread for each durable business task or
  conversation scope.
- [ ] Persist progress from item notifications, especially `item/started`,
  `item/agentMessage/delta`, `item/completed`, and `turn/completed`.
- [ ] Convert `approval/request` into a backend pending task that an operator or
  end user can answer through a matching response or `approval/resolve`.
- [ ] Store server request ids with the pending task so the backend response can
  resolve the exact App Server callback.
- [ ] Use `thread/read` or `thread/resume` after worker restart to rebuild the
  persisted timeline before accepting new user input.

## v0.5 Overload Retry

- [ ] Treat JSON-RPC error code `-32001` as retryable.
- [ ] Use bounded exponential backoff and keep the original business job id so
  retry attempts remain idempotent at the product layer.
- [ ] Continue reading stdout from stdio App Server processes while work is
  active; slow readers can trigger transport backpressure.
- [ ] Treat parse errors as per-line protocol failures. Record the `-32700`
  response and keep the process alive for later valid JSONL messages.

## Verification

- [ ] Add an integration fixture for initialize, thread start, turn start,
  progress persistence, approval response, and replay.
- [ ] Confirm worker restart does not duplicate completed items or lose pending
  approval state.
- [ ] Confirm denied and timed-out approvals are visible in business task state
  and do not execute the protected tool.
