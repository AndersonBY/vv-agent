# Runtime control and resume

All public entrypoints use the session kernel. Ordinary Runner runs own SQLite
`:memory:`; interactive and App Server hosts can supply a SessionStore. Persistent
SQLite and PostgreSQL preserve the same log, inbox and projection semantics.

## Defaults and configuration

Per-run RunConfig overrides configured Runner defaults and Agent defaults. The
framework default is 10 cycles, 10 handoffs and no-tool policy finish. Session
Memory is disabled unless explicitly enabled. Budgets, model/prompt, policy and
cycle allowances freeze at turn admission. Provider limits also configure memory
capacity. Before-cycle and interruption message callbacks alter logged model
requests; they do not replace the raw transcript.

## Waiting and resuming

User and approval waits park an operation or turn and emit no terminal. The
result carries session_id/turn_id and pending wait identities. A reply identifies
the retained wait and is queued through the owning SessionDriver, AgentSession or
App Server. Approval decisions write inbox items; call handle.resume(),
Runner.resume(session_id, turn_id), AgentSession.continue_run() or turn/resume to
drive after a parked handle completes. Replies preserve the original turn and
budget. Fresh turns get fresh counters. RunState resume is retired.

Terminal replay reads the original retained prefix without re-running tools or
models. An old result cannot absorb a child's later continuation. Same-ID
same-byte input replays; changed bytes conflict without writes.

## Session state

AgentSession messages and shared_state are detached, read-only projections.
Creation-time seed contains messages and JSON shared_state. There is no writable
Session, replace_messages, replace_shared_state or clear_queues capability.
Steer targets the current turn; an idle queued steer is consumed after admission.
Follow-up queues a fresh turn after completion. Closed sessions reject execution.

## Delegation and background work

create_sub_task, configured sub-agents, Agent.as_tool, BackgroundAgentTask and
handoff share atomic child admission and authenticated terminal delivery. Children
run under their own leases, freeze their definitions and preserve intermediate
user waits independently. Background admission returns running. Poll and wait read
retained state; cancellation writes inbox items. A handoff adopts the terminal
child result and does not resume the source model. Admission-derived handoff limits
survive changes to live configuration.

## Cancellation, leases and budgets

Cancellation returns a typed failed public result with completion_reason cancelled
and one run_cancelled event. Parent closure atomically targets live descendants.
Pending input/approval cannot permit a new side effect after cancellation.

LeaseLost recovery is bounded: five losses, exponential 10 ms base, 500 ms cap,
and jitter between half/full delay. Runtime knobs and injected sleep/jitter permit
deterministic coverage. Exhaustion raises LeaseRetryExhausted; App Server exposes
its typed error. A terminal already committed is replayed without dispatch.

Budget accounting includes primary, compaction, Session Memory and output repair.
Atomic model completion checks post-operation overshoot; tool batches reserve as a
whole before effects. Cancellation and operation failure retain terminal priority.
See [run-budgets.md](run-budgets.md) and the canonical v24 contract.

## Events and tracing

RunEvent v6 is the only event codec; model-call v2 includes output_repair and task
usage uses v3. Results retain session/turn identities. Durable events project a
retained prefix with stable IDs. Live assistant/reasoning/tool deltas are volatile
and appear in handle subscriptions, not durable result replay. Observer failure is
isolated. Child admission/completion belongs to the parent run and carries child
identity. Tracing uses a separate acknowledged consumer, with at-most-once delivery.

Producer coverage is in test_runner_events_producer_parity.py,
test_run_handle_live_stream.py, test_approval_protocol.py, test_approval_session.py,
test_sub_task_manager_continuation_recovery.py, test_run_budget.py and tests/session.
