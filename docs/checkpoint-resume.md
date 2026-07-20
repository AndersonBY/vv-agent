# Durable Checkpoint And Resume

The normative cross-language behavior lives in
`vv-agent-contract/docs/checkpoint-resume.md`. This document maps that contract
to the Python implementation and shows the supported public entry point.

## Opt In

Checkpoint v2 is disabled unless a `CheckpointConfig` is attached to the run.
Existing checkpoint v1 codecs and distributed behavior remain available.
Checkpoint v1 and `vv-agent.distributed-run.v1` writers and fixtures do not gain
the checkpoint-v2 tool metadata or metadata-policy fields.

```python
from vv_agent import Agent, CheckpointConfig, RunConfig, Runner
from vv_agent.checkpoint import ResumePolicy
from vv_agent.runtime.stores.sqlite import SqliteStateStore

store = SqliteStateStore(".vv-agent-state/checkpoints.db")
config = RunConfig(
    model="kimi-k2.6",
    workspace="./workspace",
    checkpoint_config=CheckpointConfig(
        key="tenant-7/task-42",
        resume_policy=ResumePolicy.RESUME_IF_PRESENT,
        store=store,
        capability_refs={
            "workspace": {"id": "workspace.tenant-7", "version": "2"},
        },
    ),
)

result = Runner.run_sync(
    Agent(name="worker", instructions="Complete the task carefully."),
    "Process task 42.",
    run_config=config,
)
```

Use the same key and the same immutable run definition to recover after a
process restart. `resume_if_present` creates the checkpoint when absent and
recovers it when present. `require_existing` refuses to create a new record;
`new` refuses to reuse an existing key.

## Frozen Run Definition

Before the first model or tool operation, Runner persists a credential-redacted
RFC 8785 run definition. It includes the compiled prompt, effective model
settings, model-visible tools in request order, tool policies and idempotency,
budgets, output schema, metadata that changes behavior, and extension versions.
Each frozen tool retains the legacy `idempotency` projection and adds
`tool_metadata`, either the normalized typed declaration or null. The effective
policy freezes `denied_side_effects`, `denied_capability_tags`,
`deny_terminal_tools`, and `denied_cost_dimensions` alongside the existing
name, predicate, and approval controls.

Process-local behavior must have a stable `{id, version}` capability reference.
This includes dynamic instructions, context and memory providers, guardrails,
hooks, sessions, workspace backends, predicates, reconciliation providers, and
custom output validators. Credential values are declared with JSON Pointer
slots and replaced with `<credential-redacted>`; credential rotation therefore
does not change the digest.

Recovery preserves the original run ID, trace ID, task ID, prompt, initial
messages, shared state, and session boundary. It does not call dynamic
instructions, input guardrails, context providers, or session reads again.
Current schemas and capability versions are still checked before external work.
Resume also compares the current effective tool declarations and metadata
denials with the stored definition before claim or external operations. Generic
tool metadata is never promoted into the typed declaration.

Definitions written by contract 0.7.1 do not contain these additive nested
fields. Resume first verifies the digest against the stored definition, then
adds null/empty/false defaults only to an in-memory comparison copy. It never
rewrites the stored definition or its original digest. A current non-default
declaration or denial therefore still produces
`checkpoint_definition_mismatch` before external work, while default current
values remain compatible with the exact legacy bytes.

## Operation Journal

Every model request and executable tool call has a stable operation identity.
The durable states are `planned`, `started`, `succeeded`, `failed`, and
`ambiguous`.

- A durable model response or tool receipt is replayed without another external
  call.
- A planned operation may execute normally.
- A started operation without a receipt becomes ambiguous after recovery.
- A model retry requires `retry_with_duplicate_risk` or a reconciliation
  decision and emits an explicit duplicate-cost-risk event.
- A tool retry requires both `retry_idempotent_only` and a tool declaration of
  `ToolIdempotency.SUPPORTED`; the same idempotency key is reused.
- Unknown or unsupported tool idempotency never causes a silent retry.

The typed `tool_call_planned`, `tool_call_started`, and `tool_call_completed`
events are execution observations, not this durable journal. A policy or
approval short-circuit can have planned plus completed telemetry without a
started journal operation; a crash after the started boundary can leave
telemetry incomplete. Journal receipts and reconciliation remain authoritative
for replay, ambiguity, and duplicate-risk decisions.

Without a safe decision, the public result is
`AgentStatus.RECONCILIATION_REQUIRED`. It has no completion reason and does not
run output guardrails, append session messages, or emit a business terminal.
A host can provide a `ReconciliationProvider` to defer, retry, replay a receipt,
record a definitive failure, or abort while retaining the unknown-outcome
evidence.

## Durable Boundaries

Only a complete cycle advances checkpoint messages and cycle records. In-flight
transcript changes are reconstructed from durable operation receipts. Shared
state, budget observations, extension state, and operation journals are saved
at operation progress boundaries.

For a normal terminal, Runner orders work as follows:

1. runtime candidate;
2. output guardrail and typed-output validation;
3. session `add_items_once` using a stable checkpoint commit ID;
4. durable session observation;
5. terminal event staged in the checkpoint outbox;
6. claimed checkpoint finalization;
7. idempotent event delivery and delivery CAS;
8. retained terminal acknowledgement.

Terminal records remain replayable after acknowledgement. Session and event
stores reject reuse of the same identity with different payload bytes.

## Scope And Limits

Checkpoint v2 provides durable resume with explicit ambiguity. It does not make
an arbitrary external API exactly-once, recover a provider response that was
never durably received, make host hooks transactional, or atomically commit an
unrelated state store and event store. Authentication, tenant isolation,
encryption, retention, and checkpoint redaction remain host responsibilities.

Leaving typed tool metadata absent and all four metadata-denial fields at their
defaults preserves the prior run definition comparison, model-visible schema,
and execution behavior. Leaving checkpoint v2 disabled preserves the existing
v1 persistence and distributed wire behavior.

Checkpoint-enabled roots currently reject handoffs. Agent-as-tool and
background children do not inherit the parent checkpoint config, key,
extensions, or reconciliation provider; a host must assign a distinct child
checkpoint explicitly.

Approval resume is a distinct run. Use `Runner.configured(...)` with a distinct
explicit key and `resume_if_present`; the approval claim is bound to that target
key, while the source tool call ID, request digest, and idempotency key are
preserved.

## Verification

```bash
uv run pytest tests/test_checkpoint_v2.py
uv run pytest tests/test_checkpoint_runner_v2.py
uv run pytest tests/test_checkpoint_fault_matrix.py
uv run pytest tests/test_checkpoint_resume_events.py
uv run pytest tests/test_run_definition_producer.py tests/test_distributed_checkpoint_v2.py
```

The fault suite covers F1-F8 deterministic persistence boundaries and a real
SIGKILL canary against SQLite. Full adoption additionally requires the Rust
suite and the central cross-repository workflow against the same contract lock.
