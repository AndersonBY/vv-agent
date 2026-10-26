# App Server runtime mapping

The App Server owns transport connections, subscriptions, protocol formatting
and approval routing. `SessionDriver` and the session kernel own execution.
Product modules stay behind the `AppServerHost` agent and configuration
providers.

| App Server | Kernel owner |
| --- | --- |
| Thread | Session record metadata and the projected `ThreadStatus` |
| Turn | The retained kernel turn identity, plus a process-local `RunHandle` |
| Item | A typed event derived from records and projected to a protocol item |
| Approval | The parked operation keeps the request binding and deadline; any authorized connection may answer |
| Context read | Immutable log/inbox projection and the shared memory-policy resolver |
| Manual compaction | Existing content extension, summary model operations and memory boundaries; an idle request admits a compaction-only turn |
| Model selection | A session-scoped `model_selection` inbox input |
| Live delta | A volatile `RunEvent` observation routed to subscribed transports |
| Read and resume | Retained records and bounded (seq, event offset, recordDigest) cursors; resume drives the retained turn |

`SessionRunEventStore` supplies durable event history. `JsonlRunEventStore` is
an optional sink. `thread_store` and `thread_state` hold projected values and
local handles; they own no queue, transcript or execution ledger. Host controls
enter the inbox.

`thread/model/select` maps `threadId` to the session ID and passes `inputId`
unchanged as `InboxItem.input_id`, with `kind="model_selection"`,
`payload={"selection": ...}`, and null turn and generation. It uses the same
kernel and store and keeps no separate configuration ledger.

Submission IDs are stored unchanged in the existing inbox. Thread creation uses
the host's stable ID as session identity and the atomic creation receipt.
turn/action and interrupt submit existing durable controls. App Server keeps no
execution owner connection and no second ledger. SessionDriver binds the host
model resolver; CLI and interactive paths resolve through their ModelProvider.

Context lifecycle notifications map memory_compact_started and
memory_compact_completed with requested_by identities. A terminal DiagnosticEvent
reports requests that never completed, including cancellation before summary
start. events/list supplies indexed intent and lifecycle dependencies for that
projection. No new record, inbox, SQL or RunEvent version is needed.

## Fork ownership

thread/fork reads an immutable source prefix ending at turn_ended (or creation),
projects model context with the ordinary context reader, and atomically creates
one session. The existing seed carries messages and empty shared state.
attributes.app_server_fork retains the request, lineage, inherited effective
model and last compaction descriptor; app_server_context retains the cut baseline.
That baseline clears the source task's bootstrap messages, initial shared state,
user prompt and reserved execution metadata; only the seed supplies copied history.
No source inbox, attempt or event log is copied. The destination inbox and turns
start empty. Thread metadata exposes lineage and opaque workspace to the host.

session/fork.py owns ID derivation, boundary selection, artifact integrity and
creation replay. The kernel uses the inherited model as its frozen default;
ordinary later selections take precedence. Artifact references use the existing
workspace backend API and layout. Tests in test_app_server_fork.py and
session/test_app_server_fork.py cover processors, cross-node races and all SQL
stores. Host business-file cloning stays outside this mapping.
