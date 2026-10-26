# Session model selection

A host can change which model a session uses while the session runs, for
example when a user picks another model or reasoning effort in the UI. The
host submits the choice as a `model_selection` inbox input. The lease holder
applies it at the next cycle boundary and logs the result, so every worker and
every later replay sees the same choice. The selection stays in force across
turns until another selection replaces or clears it. In-flight calls and their
retries never change model.

## Submitting a selection

Push the input in the host's own transaction, with a stable input ID for each
user command:

```python
from vv_agent.session.records import InboxItem

selection = {
    "model": model_name,
    "model_settings": {"reasoning": {"effort": reasoning_effort}},
    "route": {"backend_type": backend_type},
}
with store.atomic() as tx:
    tx.push(session_id, InboxItem(command_id, "model_selection", {"selection": selection}))
```

Then wake the session after the outer transaction commits. Any process can
push a selection; no execution lease is needed.

The payload is closed: `{"selection": null | {"model", "model_settings"?, "route"}}`.

- `model` is a nonempty string: the exact model ID, which must equal the
  `model_id` of the config the resolver returns.
- `model_settings` uses the canonical `ModelSettings` fields. Reasoning effort
  belongs here (`reasoning: {effort: ...}`), because it is a request setting
  for the provider, not routing. Omitted settings mean empty settings for the
  selected model; nothing is inherited from the previous model.
- `route` is the only host-defined routing object. vv-agent passes it to the
  resolver without reading it. Never put keys or credentials in it.
- The input is session-scoped: `target_turn_id` and `generation` must be null.
- `{"selection": None}` clears the selection and restores the model and
  settings frozen in the turn's task.

Input IDs are scoped to the session. Pushing the same ID with identical bytes
is a replay with no further effect. The same ID with different bytes is a
conflict.

## When it takes effect and what it never changes

The lease holder reads selections in inbox order. It applies one only when the
current turn has no open operation batch: no planned, started or parked
operation and no pending retry. It then logs an `input_applied` record that
carries the logical model binding: backend, model, context length and
maximum output tokens.

The next primary model call plans with the selected model, settings and route.
Its context planning uses the selected model's context and output limits and
`max_tokens`. An explicit compaction policy stays as frozen.

A selection never changes work that already exists. An operation that was
already planned keeps its model through retries, late `provider_result`
inputs, recovery, projection and replay. The frozen `Runtime` and task
definition stay unchanged. Internal model routes for summaries and memory
extraction keep their own configuration.

Child sessions have their own selection state. To change a child's model,
push a selection to the child.

## Resolver contract

Bind a resolver on the runtime:

```python
def resolve_request(request):
    # request has "model", "model_settings" and "route", taken from a selection
    # or from a retained plan.
    return llm_client, resolved_config

runtime = Runtime(..., model_resolver=resolve_request)
```

`model_resolver(request)` returns `(LLMClient, ResolvedModelConfig)`.

- It must preserve the logical model for a given request. Endpoint routing may
  change with the host settings.
- It must not submit anything to a provider.
- It resolves the exact model ID; `resolved_config.model_id` must equal
  `request["model"]`, or the selection is rejected.
- It enforces host authorization and may raise for unknown or unauthorized
  models and routes.

The runtime calls the resolver for every primary plan and when preparing its
attempt, including its default model with an empty route. Hosts must resolve
both default and selected requests. Without a resolver, the runtime accepts
only its current default client and model with an empty route.

Every primary call plan retains the effective model, complete settings, route
and logical binding. Identity checks compare backend, model ID, context length
and maximum output tokens. Endpoint IDs describe current routing and do not
participate in those checks.
The binding still carries `endpoints` as routing evidence. Its stored shape
stays unchanged; endpoint order is frozen separately for each operation.

For each new operation, the kernel resolves the current client and freezes
`client.ordered_targets(preferred_endpoint_id)` as `endpoint_order`.
`VvLlmClient.ordered_targets` is the host's single ordering policy, including
after a model selection. The kernel passes a known preference as a hint; it
never promotes an endpoint itself. Unknown preferences are ignored. A host
that always orders A before B keeps that policy even when B succeeded for the
previous model. Internal model purposes use the same hook on their own routes.
An output repair callback owns its own provider calls; the kernel does not
redirect it to the selected primary model.

Retries reuse the exact frozen order, with one endpoint and one transport
attempt per logged attempt. Before dispatch, the runtime prepares the current
client for the frozen endpoint. If that endpoint has disappeared, the kernel
closes the operation with `endpoint_unavailable`, empty usage and
`execution_started=false`. It makes no provider call and adds no duplicate-cost
risk. The next primary cycle plans from current routing, subject to the normal
cycle and budget limits. The Unknown retry limit stays
`max(2, len(endpoint_order))`.

A dispatched attempt keeps its original endpoint ID and evidence after a
routing edit. Crash recovery and late receipts use that evidence. They do not
reinterpret it as an undispatched failure.

## Rejection

Before applying a selection, the holder resolves it. If the resolver raises,
the input is logged as `rejected` with reason `model_selection_unresolvable`
and the previous selection stays in effect. A selection pushed to a closed
session is rejected with reason `session closed`. A later valid selection can
still be accepted.

A successful push only means the input was stored. Check the `input_applied`
record to learn whether the selection was accepted.

If the resolver fails later, when a planned call runs, the attempt fails or
becomes unknown through the normal attempt handling. The plan is never
rewritten.

## Billing

Model-call events and canonical usage take the model and backend from each
call's retained plan, never from the current selection. Bill per call from
`token_usage.model_calls`, or from `ModelCallCompletedEvent` and
`ModelCallFailedEvent`, using each call's `model`, `backend` and stable
`call_id`. Never bill with the session's latest requested model.

Prompt-too-long failed attempts keep their receipts, and primary cycle
accounting is unchanged, so each call is counted once. See
[billing](host-integration.md#billing) for the projection rules.

## App Server method

App Server clients use `thread/model/select`:

```jsonl
{"jsonrpc":"2.0","id":20,"method":"thread/model/select","params":{"threadId":"thread_1","inputId":"select_1","selection":{"model":"kimi-k3","route":{}}}}
{"jsonrpc":"2.0","id":20,"result":{"threadId":"thread_1","inputId":"select_1","accepted":true}}
```

The method pushes the same `model_selection` input into the thread's session;
`inputId` becomes the input ID unchanged. `accepted: true` means the input was
stored. The resolver runs later, when the holder applies the input. Reusing an
`inputId` with a different selection returns `-32024 submission_conflict`.

The built-in App Server binds `AppServerHost.resolve_model`. Its default uses
the configured ModelProvider to resolve an exact provider model ID with an empty
route. For `{"a": {"id": "a-2024"}}`, select `a-2024`; CLI startup still takes
settings key `a`. The default resolver filters the configured backend first and
matches the resolved model_id, including endpoint model_id overrides, as advertised
by model/list. Multiple entries in that effective backend for one selected ID are rejected as
model_selection_unresolvable, as are unknown IDs; there are no aliases.
CLI and interactive use the same default provider resolver through SessionDriver;
only App Server binds the host hook. The default model with an empty route also
passes through that resolver.
Custom hosts supply routing and authorization through the hook. See
[App Server](app-server.md#models).

## Why an inbox input rather than a host callback

An alternative design would let the runtime ask a host callback for the current
model on each cycle. That was rejected. The callback's answer would not be in
the log, App Server and other hosts could not see it, and two workers could get
different answers for the same cycle. An inbox input is logged once, applied at
a defined point, and replays the same way everywhere.

## Fork defaults

thread/fork freezes the source's effective model, settings, route and binding at
the selected ended-turn cut. It does not copy a pending selection. The inherited
choice is the new session's default, so a later null selection restores it.
Selections on either session never alter the other. See
[thread forks](app-server.md#forking-a-thread).
