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
uv run vv-agent app-server --listen stdio
```

Each inbound and outbound line is one JSON object. The wire format follows the
project's JSON-RPC envelope but intentionally omits the `"jsonrpc": "2.0"`
field. Request `id` values may be strings or integers.

```jsonl
{"id":1,"method":"initialize","params":{"clientInfo":{"name":"desktop-host","version":"0.1.0"},"capabilities":{"optOutNotificationMethods":[]}}}
{"method":"initialized"}
```

The server responds to `initialize` with the user agent, protocol version, and
capabilities:

```json
{"id":1,"result":{"userAgent":"vv-agent-app-server","protocolVersion":"v1","capabilities":{"modelList":true,"threadLifecycle":true,"notificationOptOut":true}}}
```

Requests other than `initialize` are rejected until the connection has been
initialized.

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
| `item/completed` | Mark an item completed and merge its final payload. |
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
{"id":"server_req_1","method":"approval/request","params":{"requestId":"approval_1","threadId":"thread_1","turnId":"turn_1","toolCallId":"call_1","toolName":"write_file","preview":"Approval required for tool write_file.","arguments":{"path":"notes.md"}}}
```

The client must answer with a normal response whose `id` matches the server
request id:

```jsonl
{"id":"server_req_1","result":{"decision":"allow","message":"Approved from UI"}}
```

Supported decisions are `allow`, `deny`, `allow_session`, and `timeout`.
Disconnects and timeouts resolve as denied or timed-out approval decisions in
the runtime.

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
```

`generate-json-schema` remains available as a compatibility alias.

The command writes:

- `json/ClientRequest.json`
- `json/ServerNotification.json`
- `json/ServerRequest.json`
- `json/vv_agent_app_server.schemas.json`

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
