# Runtime Control And Resume

This document covers SDK behavior that spans `Agent`, `Runner`, live handles,
interrupted results, approvals, sessions, and typed events.

## Agent Defaults

`Agent.max_cycles` and `Agent.tool_policy` are optional per-agent defaults. The
effective order is:

1. A value supplied by `RunConfig`.
2. The corresponding `Agent` value.
3. The framework default (`10` cycles and no extra tool policy).

This lets an agent carry a reusable safety boundary while a specific run can
override it.

## Per-Run Control Surface

`RunConfig` can replace or extend model selection/settings, workspace and
session state, cycle/handoff bounds, tool policy and registry construction,
execution backend, cancellation, approvals, event storage, hooks, tracing,
context/memory providers, initial messages/shared state, before-cycle and
interruption injection, sub-task management, typed event observers, log preview
length, and LLM request debug dumps.

The cross-language capability manifest is
`tests/fixtures/parity/run_config_controls.json`. Python field coverage is
enforced by `tests/test_run_config_controls_contract.py`; the Rust sibling runs
the same manifest through an end-to-end control wiring test.

Rust packages Python's settings file/backend/builder/timeout controls into a
per-run `ModelProvider`. Rust uses independent `RunHandle` event subscriptions
where Python also permits a direct typed-event observer. These are API-shape
adaptations; the observable capabilities remain equivalent.

## Background Agent Tasks

`Agent.as_background_task()` returns a real `BackgroundAgentTask`, not an
agent-as-tool alias. Starting it delegates to `Runner.start()` and returns
before the child run completes:

```python
from vv_agent import Agent, RunConfig, Runner
from vv_agent.background_task import BackgroundAgentTaskSnapshot

task = Agent(
    name="drafter",
    instructions="Draft the requested report.",
    model="draft-model",
).as_background_task(name="draft_report")

handle = task.start(
    Runner,
    None,
    {"task_description": "Draft the SDK report."},
    run_config=RunConfig(model_provider=provider),
)

current: BackgroundAgentTaskSnapshot = handle.poll()
same_state = handle.snapshot()
completed = handle.wait(timeout=30)
```

`poll()` and `snapshot()` never wait for completion. `wait()` blocks until the
child is terminal or the timeout expires. When the task is invoked by a model,
its tool result contains the task id; `task.get_handle(task_id)` retrieves the
same host-side handle.

## Interrupted Results

A run that stops with `AgentStatus.WAIT_USER` can be converted to `RunState`.
Approval interruptions expose structured `ApprovalSnapshot` records containing
the interruption id, tool call id, tool name, arguments, message, and cycle.

```python
from vv_agent.result import RunState

interrupted = Runner.run_sync(agent, prompt, run_config=config)
state: RunState = interrupted.into_state()
approval = state.approvals[0]
state.approve(approval.interruption_id)

resumed = Runner.resume(state)
# The equivalent handle path is: original_handle.resume(state)
```

For an approved tool interruption, resume executes the captured original call
once without asking the model to recreate it. A live `ApprovalProvider` uses a
different path: the run remains active and the host calls
`RunHandle.approve(request_id, decision)` through the broker. Result resume does
not replace or intercept that live flow.

Approved result resume creates a fresh run ID inside the source trace. If the
tool returns `continue`, the new model loop receives the full configured
`max_cycles`; cycles from the interrupted predecessor do not consume that
budget. Passing new input for an approved tool call is rejected before
cancellation is projected or the approval is claimed. With valid input, an
already-cancelled resume emits one fresh cancelled terminal without claiming
the approval, executing the tool, or running output guardrails. If approved
terminal output fails typed-output validation, its fresh terminal is persisted
before the validation error is returned to the caller.

## Session Approvals

`ApprovalDecision.allow_session()` grants the named tool for the lifetime of
the same `ApprovalBroker`. Later calls to that tool skip the provider; other
tools still request approval. `allow`, `deny`, and `timeout` do not create a
session grant. Reuse an explicit broker in multiple `RunConfig` instances when
the host wants the grant to span multiple runs in one host session.

App Server approval requests, client resolution, notifications, and generated
schemas preserve all four canonical decisions.

`ApprovalResolvedEvent.action` is the only decision field and contains one of
`allow`, `allow_session`, `deny`, or `timeout`. Missing, unknown, or additional
decision fields are rejected.

## Session History

When `RunConfig.session` is set, Runner persists the complete current-turn
delta from `AgentResult.messages`. This includes the user message, assistant
messages with tool calls, and tool result messages. The next run receives that
complete history from the session; Runner does not reconstruct a lossy
`user + final answer` pair.

## Cancellation

`CancellationToken.cancel()` is idempotent and invokes each registered callback
once. A callback registered after cancellation is invoked once immediately.
When a terminal result has already been emitted, a later `RunHandle.cancel()`
returns `False` and the completed result remains authoritative.

## Typed Event Producers

The core Runner emits `run_started` followed by `agent_started`, then
`cycle_started` for every cycle. Every model dispatch admitted at the provider
boundary emits `model_call_started` followed by exactly one terminal
`model_call_completed` or `model_call_failed` event. Agent cycles, Session
Memory, and full memory compaction use this same task-neutral lifecycle.
`assistant_delta` is emitted only when the LLM adapter projects a valid
provider stream delta; the complete `cycle_llm_response` diagnostic is not
projected as a duplicate delta.

Actual tool execution uses a three-event lifecycle after serialized arguments
have been normalized:

1. `tool_call_planned` before policy, approval, or dispatch;
2. `tool_call_started` immediately before the executor may cause effects;
3. `tool_call_completed` after a `ToolExecutionResult` exists.

The corresponding public Python classes are `ToolCallPlannedEvent`,
`ToolCallStartedEvent`, and `ToolCallCompletedEvent`.

Planned and started events contain normalized `arguments` and optional
`tool_metadata`. A completed event contains lowercase `status`, `directive`, a
nullable `error_code`, `execution_started`, nullable monotonic `duration_ms`,
and optional `tool_metadata`. Supported status values are `success`, `error`,
`wait_response`, `running`, and `pending_compress`; directives are `continue`,
`finish`, and `wait_user`. Duration is a non-negative JSON-safe integer floored
to milliseconds, and is null when execution never crossed the started boundary.

Invalid serialized arguments fail before planning and emit none of these
events. Unknown tools, policy denials, and approval short-circuits emit planned
plus completed with `execution_started=False`, no started event, and null
duration. Cancellation or process loss after started may leave no completed
event; these observations do not provide exactly-once execution or replace the
checkpoint v3 operation journal.

When a configured session accepts the current turn through `add_items()`, the
Runner emits `session_persisted`. Run, trace, agent, and session identities are
captured by Runner and cannot be replaced by provider payloads or user metadata.
Tool completion statuses use lowercase wire values such as `success`, `error`,
and `wait_response`.

The event wire version is `v1`, used as a strict discriminator rather than a
decoder selector. Current events require their complete field set; readers
reject missing, stale, unknown, and malformed fields. The public runtime exposes
only typed `RunEvent` objects. Task-neutral internal observations use
`DiagnosticEvent(level, code, details)` and cannot replace authoritative
lifecycle or terminal events.

`tests/fixtures/parity/run_events.jsonl` is the normalized real-producer
fixture shared with Rust. Random event/run ids and timestamps are normalized;
the event order and stable wire fields come from a real Runner execution.
