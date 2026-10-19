# Session model selection

## Wire and application

Inbox schema 3 adds `model_selection`, with session-scoped null target_turn_id
and generation. Its closed payload is `{selection: null | {model: string,
model_settings?: ModelSettings, route: object}}`. Model must be nonempty; route
is the sole host-opaque routing object. Settings use the existing canonical
ModelSettings fields, including `reasoning: {effort: ...}`: reasoning effort is
a provider request setting, not transport routing. Omitted settings mean empty
settings for the selected model, not inheritance from another model.

Any process calls `tx.push(session_id, InboxItem(...))` inside `store.atomic()` without an
execution lease. The holder consumes selections in inbox order only after the
active operation batch (including retries) closes, and logs `input_applied` with resolved backend, capacity and endpoint binding.
Latest accepted selection persists across session turns; null restores the
frozen turn task model/settings. Input IDs are session-scoped: identical bytes
replay without another effect, different bytes conflict. Selection changes no
existing operation or receipt, including late provider_result, recovery,
projection, retry or replay. Children have independent selection state.

A host-state callback was rejected: its decision would be absent from the log,
unavailable to App Server/other hosts, and potentially different on two workers.

## Resolver and plans

Runtime accepts a public `model_resolver(request) -> (LLMClient,
ResolvedModelConfig)` binding. It is deterministic for a request, performs no
provider submission, and may reject unauthorized or unknown routes. The default
accepts only the runtime's frozen model and empty route. Every primary plan
retains effective model, complete settings and route plus resolved backend and
capacity metadata. Execution resolves from that request alone, verifies its
binding, and uses the selected client; retries reuse the exact plan. The frozen
Runtime/Task definition stays unchanged. Internal summary/extraction routes
retain their existing independent configuration.
Primary endpoint-order validation uses that retained request binding, including after a
cross-backend switch and on retry, rather than the turn's default endpoints.

## Failure and capacity

Before accepting a selection the holder resolves it. A resolver exception makes
the input `rejected` with a stable reason; previous selection remains effective.
A later valid input can be accepted. Resolver failures after planning use existing
normal failed/unknown attempt handling and never rewrite that plan.

Context planning uses the effective model's context/output limits and max_tokens;
explicit compaction policy remains frozen. Model-call events and canonical usage
read model/backend from each retained plan, never current selection. Existing
prompt-too-long failed-attempt receipts and primary logical-cycle accounting
remain in force, with each call accounted once.

## App Server and versions

Add `thread/model/select` with `{threadId, inputId, selection}` and admission
result `{threadId, inputId, accepted: true}`. This feeds the same kernel inbox;
acceptance acknowledges submission, not runtime validation.

Contract 27.0.0 replaces inbox 2 with strict inbox 3 and public API v10 with v11;
record schema remains 1 with current embedded inbox. App Server stays v2 with an
additive method. Python package becomes 0.24.0. Adoption remains pending until
reviewer release, immutable re-pin and central CI; Rust stays frozen at 23.0.0.

SQLite files use user_version 3 with the model_selection kind constraint and
reject prior file versions. PostgreSQL hosts provision the current DDL explicitly.
No historical decoder or automatic database migration is added.
