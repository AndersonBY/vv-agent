# Runtime control and resume

Every public entry point runs on the session kernel. A session has a log, an
append-only list of records, and an inbox, a queue of inputs that have arrived
but have not been applied yet. A turn is the work started by one user input;
it ends with one terminal result.

An ordinary `Runner` run creates its own in-memory SQLite store (`:memory:`)
that disappears with the process. Interactive and App Server hosts can supply
a `SessionStore`. Persistent SQLite and PostgreSQL stores keep the same log,
inbox, and projection behavior. See [session-kernel.md](session-kernel.md).

## Defaults and configuration

Settings come from three layers: the per-run `RunConfig`, the configured
Runner's defaults, and the `Agent`. For `max_cycles` and `no_tool_policy`, the
per-run value wins, then the Runner default, then the Agent. The model and the
tool approval mode check the Agent before the Runner default.

Framework defaults:

- 10 cycles per turn (`max_cycles`)
- 10 handoffs (`max_handoffs`)
- no-tool policy `finish`
- Session Memory off (`session_memory_enabled=False`)

When a turn is admitted, the kernel freezes its budget, model, prompt, tool
policy, and cycle allowance. Later configuration changes do not affect a turn
already running. The model's context and output limits also set the memory
capacity; see [model-settings.md](model-settings.md#capacity-metadata).

The `before_cycle_messages` and `interruption_messages` callbacks change the
model request that is logged for a cycle. They do not rewrite the stored
transcript.

### Changing the model during a session

A host can switch the model of a live session by pushing a `model_selection`
input into the inbox. Any process may push it; it does not need the execution
lease. The kernel applies it at the next new primary model call, after the
current batch of operations has closed. The selection then stays in effect for
later turns until another selection replaces it. Every model call records the
model it used, so retries and replays are not affected. See
[session-model-selection.md](session-model-selection.md).

## Waiting and resuming

A run can pause to ask the user a question or to request tool approval. A
pause is not a terminal state. The kernel parks the waiting operation (or the
whole turn) and emits no terminal event. The result has status `wait_user`, a
`wait_reason`, and the `session_id` and `turn_id` of the waiting turn.
Approval waits are identified by the `request_id` in `ApprovalRequestedEvent`.

To continue:

1. Send the reply or approval decision through the object that owns the
   session: `RunHandle.approve()`, `SessionDriver`, `AgentSession.approve()`,
   or the App Server. The reply is written to the inbox.
2. Drive the turn again with `handle.resume()`, `Runner.resume(session_id,
   turn_id)`, `AgentSession.continue_run()`, or the App Server `turn/resume`
   method.

The reply continues the original turn with its remaining budget and cycle
allowance. A new turn starts with fresh counters.

`Runner.resume()` and `handle.resume()` only work in the process that started
the run, because they need the original runtime objects. They raise
`MissingHostBinding` otherwise. Durable hosts should use `AgentSession` or the
App Server.

### Replays and input identity

Asking for the result of a turn that already ended returns the recorded
result. Tools and models are not called again. A result recorded earlier never
absorbs work a child agent did after it.

Every inbox input has an ID. Submitting the same ID with identical bytes is a
no-op and returns the original receipt. Submitting the same ID with different
bytes raises `Conflict` and writes nothing. This lets a host retry a
submission safely after a timeout.

## Session state

`AgentSession.messages` and `AgentSession.shared_state` return detached
copies. Changing them does not change the session.

To start a session with existing history, pass it at creation:
`create_session(..., session={"messages": [...], "shared_state": {...}})`, or
just `shared_state=...`. The seed holds exactly those two keys, and
`shared_state` must be JSON. See [the creation-time seed](upgrading.md#creation-time-seed).

Three ways to send more input:

- `steer()` adds a message to the current turn. The model sees it at its next
  request. A steer sent while the session is idle is applied when the next
  turn starts.
- `follow_up()` queues a new turn that starts after the current one completes.
- `prompt()` sends the input now: it answers a waiting turn, or starts a new
  turn.

A closed session rejects further execution.

## Delegation and background work

`create_sub_task`, configured sub-agents, `Agent.as_tool`,
`BackgroundAgentTask`, and handoffs all start children the same way. Each
child is its own session:

- The parent admits the child in one transaction and freezes its definition.
- The child runs under its own lease.
- If the child waits for the user, that wait stays on the child. The parent
  sees only the child's final result, and accepts it only after checking that
  it belongs to the child it started.

A background child reports `running` as soon as it is admitted, and the
parent keeps going. `BackgroundAgentTaskHandle.poll()` and `wait()` read the
child's stored state. Cancelling a child is written as an input to the
child's inbox, like any other control input.

A handoff ends the source agent's turn with the child's result; the source
model does not run again. The handoff limit is fixed when the turn is
admitted, so changing `max_handoffs` later does not affect a running turn.

## Cancellation

A cancelled run returns a failed result with `completion_reason`
`cancelled`, and emits exactly one `run_cancelled` event. Closing a parent
targets all of its live descendants in the same transaction. After
cancellation, a pending reply or approval cannot start a new side effect.

## Lease loss

The execution lease is a time-limited claim on a session that lets one worker
drive it. If another worker takes the lease, the current driver gets
`LeaseLost`. The runner retries with exponential backoff:

| Setting (`Runtime` field) | Default |
| --- | --- |
| `lease_retry_attempts` | 5 losses, then give up |
| `lease_retry_base_seconds` | 0.01 (10 ms) |
| `lease_retry_cap_seconds` | 0.5 (500 ms) |
| `lease_retry_jitter` | uniform between half and the full delay |
| `lease_retry_sleep` | `time.sleep` |

Tests can inject `lease_retry_jitter` and `lease_retry_sleep` to make retries
deterministic. When the retries run out, the runner raises
`LeaseRetryExhausted` (code `lease_retry_exhausted`), and the App Server
reports that code as a typed error. If the turn's terminal result was already
committed before the lease was lost, the runner returns that result and does
not dispatch anything again.

## Budgets

Budget accounting covers primary model calls, compaction, Session Memory, and
output repair. A model call is atomic, so the kernel checks for overshoot after
the call completes. A tool batch reserves its budget as a whole before any tool
runs. Cancellation and an operation failure that already happened take
priority over a budget stop. See [run-budgets.md](run-budgets.md).

## Events and tracing

`RunEvent` v6 is the only event format. Model-call records use v2, which
includes the `output_repair` operation, and task token usage uses v3. The
canonical definitions live in the
[vv-agent-contract](https://github.com/AndersonBY/vv-agent-contract) release
pinned in `contract.lock.json`.

- Durable events are projected from the stored log and have stable IDs, so a
  replay returns the same events.
- Live assistant, reasoning, and tool deltas are not stored. They appear only
  in live `RunHandle` subscriptions, not in replays.
- A failing event observer does not affect the run.
- Child admission and completion events belong to the parent run and carry
  the child's identity.
- Tracing reads the log through its own acknowledged consumer and delivers
  spans at most once.

Tests for this behavior live in `tests/test_runner_events_producer_parity.py`,
`tests/test_run_handle_live_stream.py`, `tests/test_approval_protocol.py`,
`tests/test_approval_session.py`,
`tests/test_sub_task_manager_continuation_recovery.py`,
`tests/test_run_budget.py`, and `tests/session/`.
