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
interruption injection, sub-task management, raw runtime observers, log
preview length, and LLM request debug dumps.

The cross-language capability manifest is
`tests/fixtures/parity/run_config_controls_v1.json`. Python field coverage is
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

## Session Approvals

`ApprovalDecision.allow_session()` grants the named tool for the lifetime of
the same `ApprovalBroker`. Later calls to that tool skip the provider; other
tools still request approval. `allow`, `deny`, and `timeout` do not create a
session grant. Reuse an explicit broker in multiple `RunConfig` instances when
the host wants the grant to span multiple runs in one host session.

App Server approval requests, client resolution, notifications, and generated
schemas preserve all four canonical decisions.

`ApprovalResolvedEvent.action` preserves the canonical decision as `allow`,
`allow_session`, `deny`, or `timeout`. Its `approved` field remains available
for compatibility and is derived from the action for newly produced events.
Legacy v1 events containing only `approved` remain readable.

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

## V1 Event Producers

The core Runner emits `run_started` followed by `agent_started`, then
`cycle_started` and `llm_started` for every cycle. `assistant_delta` is emitted
only when the model client invokes the real stream callback; the complete
`cycle_llm_response` diagnostic is not projected as a duplicate delta.

When a configured session accepts the current turn through `add_items()`, the
Runner emits `session_persisted`. Run, trace, agent, and session identities are
captured by Runner and cannot be replaced by model stream payloads or user
metadata. Tool completion statuses use lowercase wire values such as `success`,
`error`, and `wait_response`. `event_from_dict()` accepts legacy
`created_at_ms` input and normalizes it to seconds in `created_at`.

`tests/fixtures/parity/runner_events_v1.jsonl` is the normalized real-producer
fixture shared with the Rust SDK. Random event/run ids and timestamps are
normalized, while implementation-private runtime-log metadata is excluded. The
event order and stable wire fields come from a real Runner execution.
