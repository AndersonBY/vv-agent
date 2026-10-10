# Host migration

Public API v9 selects the local contract 25.0.0 candidate. The execution
replacements introduced by v8 still apply; the additional
[0.23.0 host update](#python-0230--public-api-v9-host-update) below covers scheduling
and storage. Use the current API throughout a host; older runtimes remain in
pinned releases. There is no conversion of a running v23 execution. Finish or
explicitly close old work with its original runtime before switching artifacts.

## Execution and resume

| Retired API | Current replacement |
| --- | --- |
| `Runner.resume(RunState)`; `RunState`, `from_result`, `approve`, `pending_approval_ids`, `approved_interruption_ids`, `approval_snapshot`; `RunResult.into_state` | Retained session/turn identity, `RunHandle.approve` and `Runner.resume(session_id, turn_id)`; `AgentSession.continue_run` for a retained interactive session |
| `ApprovalSnapshot`, `RunResult.approval_snapshot` | Pending approval projection on the live handle/session and App Server approval requests |
| `Runner.start_distributed`, `start_distributed_compiled`, `finalize_distributed` | Host creates a session and pushes stable inbox inputs, worker calls `drive`, consumers project committed records |
| `AgentRuntime`, `ToolCallRunner` | `Runner.run_sync`, `Runner.start`, `Runner.stream_sync`; low-level assembly uses `session.kernel.drive` with `session.runtime.Runtime` |
| `ExecutionBackend`, `InlineBackend`, `ThreadBackend`, `CeleryBackend.start/advance` | One kernel driver; RunHandle supplies local asynchronous execution, host transport owns worker dispatch |

```python
# Before
state = result.into_state()
result = Runner.resume(state)

# After: same admitted turn, same frozen budget/definition
result = Runner.resume(result.raw_result.session_id, result.raw_result.turn_id)
```

Runner resume requires the retained process-local host binding. For recovery in
a new process, reopen the same durable store and reconstruct the interactive
client or App Server with the original providers, tools and host bindings.
User/approval replies continue the original turn. A new prompt after terminal
completion admits a new turn. `Runner.start` remains the non-blocking entrypoint.

## Persistence and host dispatch

| Retired API/configuration | Current replacement |
| --- | --- |
| `CheckpointConfig` (including `capability_refs`, `credential_slots`), `CheckpointExtension`, `ReconciliationProvider`, `ResumeObservation`; `RunConfig.checkpoint_config`, `checkpoint_extensions`, `reconciliation_provider` | Frozen turn definition, explicit provider bindings, kernel records/inbox and trusted provider evidence |
| `Checkpoint`, `CheckpointStore`, `InMemoryCheckpointStore`, `SqliteCheckpointStore`, `RedisCheckpointStore`, `OperationJournalEntry` | `SessionSpec`, `Record`, `InboxItem`, `SessionStore`, `SessionTx`, `SQLiteStore`, `PostgresStore` |
| `RunResult.checkpoint_key`, `resume_observations` | `raw_result.session_id`, `raw_result.turn_id`, projected model/tool operation receipts |
| `RunConfig.execution_backend`, `sub_task_manager` | Kernel-owned execution/children; host supplies tool/model providers and optional durable session store |
| `ControllerCommand`, `ControllerCommandReceipt`, `ControllerCommandResolution`; `reap_controller_command_wakes`, `resolve_controller_command` | Stable `InboxItem` control identity, store `push`, drive/recovery and scheduling scan |
| `DistributedBackend.produce_host_interaction`, `claim_and_consume_host_interaction_response` | Typed host interaction values and authenticated inbox response admission |
| `DistributedRunEnvelope`, `DistributedRunHandle`, `DistributedAdvanceDecision`, `DistributedDeliveryOutcome`, `DistributedWaitReason`, `RuntimeRecipe`, `CapabilityRef`, `DistributedCapabilityRegistry` | Durable session identity plus current Runtime/provider factory; state remains in SQL rather than worker envelopes |

```python
# Before
config = RunConfig(checkpoint_config=CheckpointConfig(store=checkpoint_store),
                   execution_backend=backend)

# After: local durable interactive owner
from vv_agent import AgentSessionOptions, InteractiveAgentClient, SQLiteStore

with SQLiteStore.standalone("/tmp/agent-sessions.db") as store:
    store.install_schema()  # once, on an empty database
    client = InteractiveAgentClient(options=AgentSessionOptions(
        model_provider=provider, session_store=store,
    ))
    try:
        session = client.create_session(agent=agent, session_id="thread-001")
        result = session.prompt("Continue the work")
    finally:
        client.driver.close()
```

For PostgreSQL, use `[postgres]`, a caller-owned psycopg connection and
`PostgresStore(connection)` / `join_transaction(connection)`. Host business
writes and consumer acknowledgement can share one transaction. See
[session-kernel.md](session-kernel.md#joining-host-transactions). Wake is a hint;
scanning due sessions supplies recovery when dispatch delivery is lost.

## Tool and provider outcomes

| Retired API | Current replacement |
| --- | --- |
| `DeferredToolHandle`, `ToolContext.defer`, `ToolCallOutcome` | Provider `Accepted(handle)`, `Definitive(result)` or `Unknown(reason)` in `vv_agent.session.providers` |
| `AcceptDeferredDecision`, `DeferredResolveDecision`, `DeferredResolutionReceipt`, `DeferredResolutionConflict`, `DeferredResolutionStale`, `DeferredCheckpointClaimed`, `DeferredHandleError`, `CheckpointStore.resolve_deferred` | Authenticated `provider_result` inbox item for the original operation/attempt; stable identity/bytes with `Conflict` and `LeaseLost` fencing |
| `DeferredResolutionError`, `DeferredResolutionResultInvalid` | Current provider-result validation raises `ValueError`; conflicting durable inputs raise `Conflict`, stale leases raise `LeaseLost`. The internal definitive-result validator retains its error code/text in `tools.outcomes.DefinitiveResultInvalid` |

```python
# Before: old handler-managed deferred protocol
return context.defer(handle)

# After: provider owns submission/query/cancellation evidence
from vv_agent.session.providers import Accepted, Definitive, Unknown

return Accepted(handle={"provider": "jobs", "job_id": admitted_job_id})
# query(handle) returns Definitive(result) only with trusted completion evidence;
# otherwise return Unknown(reason), or the same Accepted handle while running.
```

The handle above illustrates provider-owned data; it is not a kernel inbox wire
example. Implement `Provider.submit/query/cancel/authenticate` and bind it to
Runtime. Kernel parked handles, results and evidence must pass the current
closed schemas and authenticated original-operation checks. An ordinary
FunctionTool continues to return ToolExecutionResult or supported output values.
Timeout or an unconfirmed stop remains unknown; it does not become failure proof.

## Sessions

| Retired API | Current replacement |
| --- | --- |
| `Session`, `MemorySession`, `SQLiteSession`, `RedisSession`; `MemorySessionStore`, `SQLiteSessionStore`, `RedisSessionStore`; `RunConfig.session` | `AgentSession` / `InteractiveAgentClient` backed by kernel SQLite/PostgreSQL stores |
| Old transcript `SessionStore` append/commit authority | Current kernel `SessionStore` protocol for log/inbox/leases/consumers, a breaking replacement |
| `AgentSession.session` writable access, `replace_messages`, `replace_shared_state`, `clear_queues` | Creation-time seed; use fresh session identity for reset, targeted inbox controls for ongoing execution |
| `IdempotentRunEventStore`, old `RunEventStore` execution ledger | `SessionRunEventStore` projects log history; `JsonlRunEventStore` is an optional sink |
| `session_store_conformance` | The old transcript-only probe is removed. SQL stores implement the current `SessionStore`/`SessionTx` semantics; validate adapters with `tests/session/test_store_transactions.py` and the recovery/concurrency suites |

```python
# Before
result = Runner.run_sync(agent, prompt, run_config=RunConfig(session=MemorySession()))

# After
client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider))
try:
    session = client.create_session(agent=agent, session_id="conversation-001")
    result = session.prompt(prompt)
finally:
    client.driver.close()
```

### Creation-time seed

v-claw previously hydrates SDK history in
`services/agent_service_mixins/sub_task/restore.py:134` with
`replace_messages(hydrated)`, and resets state in `control.py:1042` with
`replace_shared_state(previous_shared_state)`. Pass both values when creating
the session. A retry reset uses a **new durable session identity**; reopening an
existing identity must match its original creation bytes.

```python
# Before
session = client.create_session(agent=agent, session_id=task_id)
session.replace_messages(hydrated)
session.replace_shared_state(previous_shared_state)
session.clear_queues()

# After: hydrated is a list[Message]; shared state must be JSON
session = client.create_session(
    agent=agent,
    session_id=f"{task_id}/retry/{retry_generation}",
    session={
        "messages": [message.to_dict() for message in hydrated],
        "shared_state": previous_shared_state,
    },
)
result = session.prompt(next_prompt)
```

Low-level hosts use `SessionSpec(..., attributes={"seed": {"messages": [...],
"shared_state": {...}}})`. Both seed members are required. Initial history is
projected before first-turn compilation. Later messages/state are detached,
read-only projections of committed records; they cannot be replaced mid-session.
Steering, follow-up, replies, archive and close enter the inbox. Closing a session
prevents subsequent execution and does not erase its history.

## Events and wire versions

`CheckpointCreatedEvent`, `CheckpointResumedEvent`, `ReconciliationRequiredEvent`,
`ReconciliationResolvedEvent`, `ToolCallDeferredEvent`, `OperationReplayedEvent`
and `SessionPersistedEvent` are retired. Consume current typed RunEvents from
session projections; recover with session/turn identities and provider evidence.

RunEvent **v6**, model-call **v2**, task-token-usage **v3** and App Server protocol
**v2** each have one strict current shape. Upgrade event readers, model-call
accounting, UI timeline/schema clients and initialize negotiation together.
For the authoritative wire changes, see the
[contract 24.0.0 CHANGELOG](https://github.com/AndersonBY/vv-agent-contract/blob/v24.0.0/CHANGELOG.md).
This guide does not duplicate that specification. Missing/stale versions and
unknown fields reject. Schema/TypeScript bundle names stay stable.

```python
# Before
send("turn/resume", {"threadId": thread_id, "checkpointKey": key})

# After
send("turn/resume", {"threadId": thread_id, "turnId": turn_id})
```

## Removed extras

`[redis]` and `[celery]` are removed, together with the `celery[redis]` development
dependency. Use `[postgres]` for the SQL reference store and `[s3]` for workspace
objects. No Redis kernel store exists. Celery tasks, broker and queue routing
belong to the host. See [cloud host integration](host-integration.md) for dispatch
semantics and a documentation-only Celery example; backend B1 implements the site's tasks.

```bash
# Before
pip install 'vv-agent[redis,celery]'
# After: only when this host uses PostgreSQL
pip install 'vv-agent[postgres]'
```

## Python 0.23.0 / public API v9 host update

Contract 25.0.0 is a local candidate pending immutable publication and required
central adoption. The v8 execution replacements above still apply; F5 adds
custom child batches and changes the required store/scheduling surface.

- Replace inline Celery tick with `tick(..., dispatch=enqueue_drive, project=...)`.
  Expire or coalesce unleased queued duplicates, including wakes. Pass the runtime
  factory directly to worker `drive`, and raise RuntimeNotReady for input
  preparation that should retry without failing a turn. Handle tick ExceptionGroup.
- Implement lease-fenced `defer_drive` and consumer-local `defer_projection` on
  custom stores. Both gate discovery using database time; execution backoff also
  gates acquisition, including ready inbox and direct duplicate wakes.
- Copy the new PostgreSQL DDL: `drive_retry_at_ms` on sk_session and
  `project_retry_at_ms` on sk_consumer, default zero. SQLite user_version is 2 and
  rejects existing v1 files. No automatic migration of old executing sessions.
- Custom callbacks may return a uniform-background nonempty sequence. Host rows
  must use the same admission transaction. InvalidChildBatch rolls the admission
  back; custom batch results include every child in admission order.

See [host integration](host-integration.md) for the complete dispatch, retry and
transaction contract. Record/inbox/event/App Server wire versions do not change.
