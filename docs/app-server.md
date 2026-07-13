# App Server

The App Server is the preferred integration boundary for long-running hosts
such as desktop apps, background services, IDE adapters, and workers. It wraps
the normal `Agent`, `Runner`, `RunConfig`, `RunHandle`, and `RunEvent`
contracts behind a bidirectional JSON-RPC protocol.

Use the Python SDK directly for in-process runs. Use the App Server when the
host needs a separate process, stable Thread / Turn / Item lifecycle events,
approval callbacks, replay, and generated schema files.

## Startup

The first transport is JSONL over stdio:

```bash
uv run vv-agent app-server --listen stdio \
  --settings local_settings.py \
  --backend moonshot \
  --model kimi-k2.6 \
  --timeout-seconds 90
```

The listen transport, settings file, backend, and model must each be provided
exactly once, in any order; `--key=value` syntax is also accepted. They are
resolved before the stdio loop starts, so a missing file, backend, model,
endpoint, or key fails at process startup instead of failing the first turn.
`--timeout-seconds` is optional and defaults to 90 seconds. Production turns use
a 30-second approval timeout.

Each inbound and outbound line is one JSON object. Every request, response,
notification, and error uses the standard `"jsonrpc": "2.0"` field. Request
`id` values may be strings or integers.

```jsonl
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"desktop-host","version":"0.1.0"},"capabilities":{"optOutNotificationMethods":[]}}}
{"jsonrpc":"2.0","method":"initialized"}
```

The server responds to `initialize` with the user agent, protocol version, and
capabilities:

```json
{"jsonrpc":"2.0","id":1,"result":{"userAgent":"vv-agent-app-server","protocolVersion":"v1","capabilities":{"modelList":true,"threadLifecycle":true,"notificationOptOut":true,"schemaExport":true,"approvalResolve":true}}}
```

Requests other than `initialize` are rejected until the connection has been
initialized. After receiving the initialize response, clients send
`{"jsonrpc":"2.0","method":"initialized"}` to complete the shared client handshake. Python v1
compatibility is preserved: the server may emit asynchronous notifications as
soon as the initialize response has been sent.

Malformed JSONL input returns error code `-32700` with a null id, and the stdio
server continues reading later lines. A valid JSON value that is not a request,
notification, response, or error object returns `-32600`.

## Threads

A thread is the App Server conversation and replay scope. `thread/start`
creates a thread, resolves the host agent key later during `turn/start`, and
subscribes the current connection to live notifications for that thread.

```jsonl
{"id":2,"method":"thread/start","params":{"agentKey":"default","cwd":"./workspace","metadata":{"profile":"assistant"}}}
```

The server emits `thread/started` and returns the same thread identity:

```json
{"method":"thread/started","params":{"threadId":"thread_1","agentKey":"default","cwd":"./workspace","status":"idle"}}
{"id":2,"result":{"threadId":"thread_1","agentKey":"default","cwd":"./workspace","status":"idle"}}
```

Use `thread/read` to read a persisted snapshot without subscribing another
connection:

```jsonl
{"id":3,"method":"thread/read","params":{"threadId":"thread_1"}}
```

Use `thread/resume` after reconnecting. It subscribes the new connection and
returns the same snapshot shape:

```jsonl
{"id":4,"method":"thread/resume","params":{"threadId":"thread_1"}}
```

Snapshots contain a `thread` object, ordered `turns`, and replayable `items`.

## Thread Lifecycle

Use `thread/list` to list active threads. Archived threads are hidden by
default; pass `includeArchived: true` to include them.

```jsonl
{"id":10,"method":"thread/list","params":{"includeArchived":true}}
```

Use `thread/archive` to mark a thread archived. The server rejects future
`turn/start` requests for archived threads with error code `-32021`.

```jsonl
{"id":11,"method":"thread/archive","params":{"threadId":"thread_1"}}
```

Use `thread/unsubscribe` to remove the current connection from a thread
subscription. If the loaded thread has no subscribers and no active turn, the
server emits `thread/closed`.

```jsonl
{"id":12,"method":"thread/unsubscribe","params":{"threadId":"thread_1"}}
```

`thread/status/changed` reports stable lifecycle changes: `running`, `idle`,
`archived`, and `closed`. Treat these notifications as the source of truth for
loaded-thread state.

## Turns

A turn is one `Runner.start()` execution inside a thread. Start a turn by
passing input items:

```jsonl
{"id":5,"method":"turn/start","params":{"threadId":"thread_1","input":[{"type":"text","text":"Inspect this workspace and summarize the findings."}]}}
```

The server responds when the turn is accepted, then streams lifecycle
notifications:

```json
{"id":5,"result":{"threadId":"thread_1","turnId":"turn_1","status":"running"}}
{"method":"turn/started","params":{"threadId":"thread_1","turnId":"turn_1"}}
```

Use `turn/steer` to queue additional user context for the active turn. Use
`turn/followUp` to queue the next turn after the current one completes. Use
`turn/interrupt` to cancel the active handle. `expectedTurnId` protects hosts
from steering or interrupting a stale turn.

```jsonl
{"id":6,"method":"turn/steer","params":{"threadId":"thread_1","expectedTurnId":"turn_1","input":[{"type":"text","text":"Also check the tests."}]}}
{"id":7,"method":"turn/followUp","params":{"threadId":"thread_1","expectedTurnId":"turn_1","input":[{"type":"text","text":"Write a release note."}]}}
{"id":8,"method":"turn/interrupt","params":{"threadId":"thread_1","expectedTurnId":"turn_1","reason":"User cancelled from UI"}}
```

## Items

Items are UI-ready projections of typed `RunEvent` objects. Hosts should render
timelines from `item/*` notifications and from replayed snapshot `items`, not
from raw runtime logs.

Current item types include:

| Item type | Meaning |
| --- | --- |
| `userMessage` | The accepted user input for the run. |
| `agentMessage` | Assistant text, either as deltas or completed text. |
| `toolCall` | Tool call started and completed states. |
| `approval` | Tool approval request and resolution lifecycle. |
| `error` | Run failure projected into the timeline. |

Important notification methods:

| Method | Use |
| --- | --- |
| `item/started` | Create or mark an item as started. |
| `item/agentMessage/delta` | Append assistant text delta to the active agent message. |
| `item/toolCall/delta` | Report streamed tool-call argument progress. |
| `item/completed` | Mark an item completed and merge its final payload. |
| `approval/requested` | Show a pending tool approval in the client. |
| `approval/resolved` | Reconcile the final approval decision. |
| `error/warning` | Surface a non-protocol runtime or event-stream warning. |
| `thread/status/changed` | Update loaded-thread status such as running, idle, archived, or closed. |
| `thread/archived` | Mark a thread archived in the client. |
| `thread/closed` | Mark a loaded thread closed after the last subscriber leaves. |
| `turn/completed` | Mark the turn terminal and persist final output or error. |

Every item carries `itemId`, `threadId`, `turnId`, `type`, `status`, `payload`,
`createdAt`, and `updatedAt`.

## Notification Opt-Out

`initialize.params.capabilities.optOutNotificationMethods` accepts exact
notification method names. The server suppresses only exact matches for that
connection.

```jsonl
{"id":1,"method":"initialize","params":{"clientInfo":{"name":"desktop-host"},"capabilities":{"optOutNotificationMethods":["item/agentMessage/delta"]}}}
```

## Approval Requests

Tools that require approval are routed as server-to-client requests. The App
Server sends `approval/request` with a request `id` plus approval params:

```json
{"id":"approval_1","method":"approval/request","params":{"requestId":"approval_1","threadId":"thread_1","turnId":"turn_1","toolCallId":"call_1","toolName":"write_file","preview":"Approval required for tool write_file.","arguments":{"path":"notes.md"}}}
```

The client must answer with a normal response whose `id` matches the server
request id:

```jsonl
{"id":"approval_1","result":{"decision":"allow_session","message":"Approved for this tool during the session"}}
```

Clients that prefer request-only control can resolve the same pending callback
with `approval/resolve`. Its `threadId`, `turnId`, and `requestId` must match the
active approval:

```jsonl
{"id":8,"method":"approval/resolve","params":{"threadId":"thread_1","turnId":"turn_1","requestId":"approval_1","decision":"allow_session"}}
{"id":8,"result":{}}
```

The server also emits `approval/requested` and `approval/resolved` notifications
so timeline state does not depend on observing the bidirectional request alone.

Normal response envelopes and `approval/resolve` both support the canonical
decisions `allow`, `allow_session`, `deny`, and `timeout`. The
`approval/resolved` notification preserves the selected decision;
`allow_session` is not collapsed to `allow`. Disconnects and expired requests
resolve as timed-out approval decisions in the runtime.

## Model List

Hosts can expose available models through `model/list`:

```jsonl
{"id":9,"method":"model/list","params":{}}
```

The default host returns the models passed to `DefaultAppServerHost`. Product
hosts should implement `AppServerHost.list_models()` from their normal model
settings.

## Overload Handling

When the server cannot accept more work, it returns JSON-RPC error code
`-32001` with message `Server overloaded; retry later.` Clients should retry
with bounded backoff. Stdio clients must keep reading stdout; if a bounded
transport cannot write outbound messages, the App Server disconnects that
transport and clears pending server-to-client requests.

## Schema Export

Generate JSON Schema files for client bindings and drift checks:

```bash
uv run vv-agent app-server schema --out ./app-server-schema
uv run vv-agent app-server generate-ts --out ./app-server-schema/typescript
```

`generate-json-schema` remains available as a compatibility alias.

The command writes:

- `json/ClientRequest.json`
- `json/ServerNotification.json`
- `json/ServerRequest.json`
- lifecycle schemas `json/AppItem.json`, `json/AppThread.json`, and
  `json/AppTurn.json`
- response schemas including `json/ThreadStartResponse.json` and
  `json/TurnStartResponse.json`
- `json/vv_agent_app_server.schemas.json`

Request params such as `turn/start` are represented inside
`json/ClientRequest.json`; the command does not write a separate
`json/TurnStartParams.json`. Every method variant uses a method discriminator
and typed params. The TypeScript files are self-contained and do not rely on
generated imports.

Initialized clients can also fetch the same JSON Schema and TypeScript bundles
without filesystem access:

```jsonl
{"id":9,"method":"schema/export","params":{}}
```

The debug client prints a complete JSONL flow for a single message:

```bash
uv run vv-agent debug app-server send-message "hello"
```

## Host Provider Boundary

`vv-agent` owns the protocol, transport, thread state, replay store, approval
callback bridge, and `Runner.start()` adapter. Product hosts own product
identity, storage, permissions, tools, UI, billing, and deployment.

Implement `AppServerHost` to keep that boundary explicit:

```python
from vv_agent import Agent, RunConfig
from vv_agent.app_server.host import AgentResolutionRequest, AppServerHost, RunConfigResolutionRequest
from vv_agent.app_server.protocol import ModelListRequest, ModelListResponse

class ProductHost(AppServerHost):
    def resolve_agent(self, request: AgentResolutionRequest) -> Agent:
        return Agent(
            name=request.agent_key,
            instructions="Resolve product instructions here.",
            tools=[],
        )

    def build_run_config(self, request: RunConfigResolutionRequest) -> RunConfig:
        return RunConfig(
            workspace=request.cwd,
            metadata={"thread_id": request.thread_id, **(request.metadata or {})},
        )

    def list_models(self, request: ModelListRequest) -> ModelListResponse:
        return ModelListResponse(models=[])
```

The host should map product profiles, workspaces, model settings, memory
providers, context providers, approval UI, and product tools into `Agent` and
`RunConfig`. The App Server should not import product modules.
