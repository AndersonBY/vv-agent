# vv-agent

[中文文档](README_ZH.md)

A lightweight agent framework extracted from VectorVein's production runtime. Cycle-based execution with pluggable LLM backends, tool dispatch, memory compression, and distributed scheduling.

## Install

The current stable release is `0.8.0`. It implements the same language-neutral
Contract `3.0.0` behavior as the Rust `vv-agent` crate while keeping a
Python-idiomatic API.

```bash
python -m pip install "vv-agent==0.8.0"
```

Use `vv-agent[celery]`, `vv-agent[redis]`, or `vv-agent[s3]` when those optional
integrations are needed. Contract 3 and repository `HEAD` are forward-only:
current readers accept only the current strict public and wire shapes. Pin an
older package release when an application must retain an older protocol.

### 0.8.0 Highlights

- Every admitted model dispatch is recorded in
  `result.token_usage.model_calls`, including agent cycles, Session Memory,
  full memory compaction, failures, retries, and ambiguous outcomes. Missing
  provider token or cache fields remain unavailable instead of being reported
  as zero.
- Tool arguments are validated as a complete JSON Schema Draft 2020-12 value
  before approval or side effects. Invalid calls return structured
  `invalid_tool_arguments` details without invoking the handler.
- Optional host output validation is disabled by default and can make at most
  one tools-free repair callback before a terminal result is committed.
- Durable execution uses `vv-agent.checkpoint.v3`,
  `vv-agent.run-definition.v2`, `vv-agent.distributed-run.v2`, and
  `vv-agent.distributed-worker-response.v1` for strict recovery and
  distributed-controller boundaries.

See [output validation](docs/output-validation.md) and
[checkpoint/resume](docs/checkpoint-resume.md) for the detailed contracts.

## Architecture

```
Agent / RunConfig / ModelSettings
└── Runner
    └── AgentRuntime
        ├── CycleRunner          # single LLM turn: context -> completion -> tool calls
        ├── ToolCallRunner       # tool dispatch, directive convergence
        ├── RuntimeHookManager   # before/after hooks
        ├── MemoryManager        # automatic history compression
        └── ExecutionBackend     # inline, thread, or Celery scheduling
```

The public SDK entry points are exported from `vv_agent`: `Agent`, `Runner`,
`RunConfig`, `RunHandle`, `ModelSettings`, `function_tool`, `Session`,
typed `RunEvent` objects, `ApprovalProvider`, `ContextProvider`,
`RunEventStore`, and the interactive session API for desktop/runtime
integrations. Extension points that live in package modules include
`vv_agent.memory.MemoryProvider` and `vv_agent.tools.ToolExecutor`.
Lower-level runtime implementation details include `AgentTask`, `AgentResult`,
`Message`, `CycleRecord`, and `ToolCall`.

Task completion is explicit: tool directives are the default, while a declared
no-tool policy can finish or pause on a normal assistant response. No implicit
"last message = answer" heuristic is used.

## Repository Setup

```bash
cp local_settings.example.py local_settings.py
# Fill in your API keys and endpoints in local_settings.py
```

```bash
uv sync --dev
uv run pytest
```

## Quick Start

### CLI

```bash
uv run vv-agent --prompt "Summarize this framework" --backend moonshot --model kimi-k3

# With per-cycle logging
uv run vv-agent --prompt "Summarize this framework" --backend moonshot --model kimi-k3 --verbose
```

CLI flags: `--settings-file`, `--backend`, `--model`, `--verbose`.

### Programmatic SDK

```python
from vv_agent import Agent, RunConfig, Runner, function_tool

@function_tool
def read_order(order_id: str) -> str:
    """Read order information."""
    return "order details"

agent = Agent(
    name="ops",
    instructions="Check facts first, then answer.",
    model="kimi-k3",
    tools=[read_order],
)

result = Runner.run_sync(agent, "Analyze order 123", run_config=RunConfig(
    default_backend="moonshot",
))
print(result.status, result.final_output)
```

For defaults shared by several runs, create a configured Runner instead of
repeating the same `RunConfig`:

```python
runner = Runner.configured(RunConfig(
    model_provider=provider,
    model="kimi-k3",
    workspace="./workspace",
))
result = runner.run_sync(agent, "Analyze order 123")
```

Provider resolution is per-run then Runner. Model resolution is per-run,
Agent, Runner, then the selected provider default. Model settings merge in the
opposite layering direction: provider, Runner, Agent, then per-run, with each
later layer overriding earlier fields.

`Agent.output_type` can coerce JSON final output into `dict`, `list`,
dataclasses, or Pydantic-style models. Decorated tools may accept a leading
`ToolContext` parameter; it is passed at invocation time and omitted from the
tool JSON schema.

### Streaming And Sessions

`RunConfig.workspace` controls the workspace for a run. `RunConfig.session`
accepts `MemorySession`, `SQLiteSession`, or `RedisSession` to persist message
history across runs.

```python
from vv_agent import Agent, MemorySession, RunConfig, Runner

agent = Agent(name="assistant", instructions="Remember context.", model="kimi-k3")
session = MemorySession("thread-001")
config = RunConfig(
    default_backend="moonshot",
    workspace="./workspace/thread-001",
    session=session,
)

Runner.run_sync(agent, "Inspect the project", run_config=config)
for event in Runner.stream_sync(agent, "Continue and report progress", run_config=config):
    if event.type == "assistant_delta":
        print(event.delta, end="")
```

Use `Runner.start()` when the host needs a live handle instead of blocking for
the final result. `RunHandle.events()` yields the same typed `RunEvent` stream
as `Runner.stream_sync()`, `RunHandle.result()` waits for the final
`RunResult`, `RunHandle.cancel()` cancels the run, and `RunHandle.approve()`
resolves pending approval requests. When the handle is attached to an
`AgentSession`, `RunHandle.steer()` queues context for the active run and
`RunHandle.follow_up()` queues the next session turn. Plain one-shot
`Runner.start()` handles do not own session queues, so those methods require an
interactive session controller.

`RunConfig.event_store` can persist every typed event. `JsonlRunEventStore`
stores event dictionaries and replays events by `run_id`, including child runs
whose `parent_run_id` points at the requested run. Typed `RunEvent` is the only
public runtime event boundary; task-neutral observations use
`DiagnosticEvent`.

For a normalized and schema-valid tool call, the execution lifecycle is
`tool_call_planned`, optional approval events, `tool_call_started` immediately
before effects may begin, and `tool_call_completed` after a result exists.
Argument parse failures emit none of these events. Schema validation, policy,
approval, and unknown-tool short-circuits emit planned plus completed without
started; completed events report `directive`, nullable `error_code`,
`execution_started`, and nullable monotonic `duration_ms`. A started event may
remain unmatched after cancellation or process loss, so checkpoint v3's
operation journal remains the recovery authority.

The lower-level `AgentRuntime` API remains available for backend integrations
that need direct cycle-loop control.

Install Redis support with `uv sync --extra redis` or inject a Redis-compatible
client when constructing `RedisSession`.

### App Server

Use the App Server when a desktop app, worker, IDE, or other host process needs
to drive `vv-agent` through a stable protocol instead of embedding the Python
SDK directly. It runs JSONL over stdio, exposes Thread / Turn / Item lifecycle
events, routes tool approval as server-to-client requests, supports
`thread/read` and `thread/resume` replay, and exports typed JSON Schema and
self-contained TypeScript bindings.

```bash
uv run vv-agent app-server --listen stdio --settings local_settings.py --backend moonshot --model kimi-k3
uv run vv-agent app-server schema --out ./app-server-schema
uv run vv-agent app-server generate-ts --out ./app-server-schema/typescript
uv run vv-agent debug app-server send-message "hello"
```

Product hosts implement `AppServerHost` to map product profiles, workspace
context, tools, approval UI, memory, and model settings into framework
`Agent` and `RunConfig` objects. The App Server remains a runtime boundary; it
does not import product UI, account, billing, browser, or IM modules.

See [docs/app-server.md](docs/app-server.md) for protocol details and
[docs/app-server-host-integration.md](docs/app-server-host-integration.md) for
the current host boundary and rollout checks.

### Interactive Sessions

Use `Runner` for one-shot runs, streamed runs, and conversation history managed
by `RunConfig.session`. Use `InteractiveAgentClient` when the host application
needs a stateful, bidirectional runtime session with stable session ids,
runtime listeners, queued steering prompts, follow-up turns, cancellation, and
shared tool state. During a running session, `session.active_run_handle` exposes
the unified `RunHandle` control surface for approval, cancellation, steering,
and follow-up.

Pass an existing `MemorySession`, `SQLiteSession`, or `RedisSession` through
`AgentSessionOptions.session` (or `create_session(session=...)`) to hydrate a
facade from durable history and let `Runner` append each turn to the same
store. When both are provided, the requested `session_id` must match the
backing Session id. Do not also pass that history as initial messages.

```python
from pathlib import Path

from vv_agent import (
    AgentSessionOptions,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    SQLiteSession,
)
from vv_agent.runtime.backends import ThreadBackend

client = InteractiveAgentClient(
    options=AgentSessionOptions(
        settings_file=Path("local_settings.py"),
        default_backend="moonshot",
        workspace=Path("./workspace/thread-001"),
        execution_backend=ThreadBackend(max_workers=4),
        session=SQLiteSession("thread-001", db_path=Path("./sessions.sqlite3")),
    )
)

session = client.create_session(
    session_id="thread-001",
    agent=InteractiveAgentDefinition(
        description="Operate in the user's workspace and report progress.",
        model="kimi-k3",
        no_tool_policy="finish",
    ),
)
unsubscribe = session.subscribe(lambda event, payload: print(event, payload))
try:
    run = session.prompt("Inspect the workspace")
    print(run.result.status, run.result.final_answer)
finally:
    unsubscribe()
```

Interactive sessions are additive to the normal SDK facade; they do not
reintroduce the old 0.1 `AgentSDKClient` or `AgentSDKOptions` names.

### Agent As Tool, Handoff, And Policy

Use `agent.as_tool()` when a child agent should return a result to the parent
agent and let the parent continue. Use `handoff()` when control should transfer
to the target agent and the target output should finish the run.

```python
from vv_agent import Agent, RunConfig, Runner, ToolPolicy, handoff
from vv_agent.constants import TASK_FINISH_TOOL_NAME

researcher = Agent(name="researcher", instructions="Collect facts.", model="kimi-k3")
writer = Agent(
    name="writer",
    instructions="Write from research.",
    model="kimi-k3",
    tools=[researcher.as_tool(name="research", description="Collect facts.")],
)
triage = Agent(
    name="triage",
    instructions="Transfer writing tasks.",
    model="kimi-k3",
    handoffs=[handoff(agent=writer, description="Use for writing.")],
)

result = Runner.run_sync(
    triage,
    "Write a short report.",
    run_config=RunConfig(
        default_backend="moonshot",
        max_handoffs=4,
        tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, "transfer_to_writer"]),
    ),
)
```

A handoff is an outer Runner control transfer, not an agent-as-tool call. The
target Agent resolves its own model and model settings, while the active
session, cancellation token, and mutated shared state continue across the
transition. `max_handoffs` defaults to `10` and limits control transfers
independently from `max_cycles`. Approval resume preserves the same behavior.

No-tool completion is an explicit control, not a task or answer classifier.
Set `Agent(no_tool_policy="finish")` when a normal assistant response should
finish the run without `task_finish`, or override it for one call with
`RunConfig(no_tool_policy="continue" | "wait_user" | "finish")`. Per-run
configuration wins over a configured Runner default, which wins over the
Agent value; omitting every layer uses `continue`. Inspect
`result.completion_reason`, `result.completion_tool_name`,
and `result.partial_output` to distinguish natural completion, tool-driven
completion, waits, cancellation, failure, and max-cycle exhaustion.

`RunConfig.budget_limits` can independently limit total tokens, uncached input
tokens, total or exact-name tool calls, active wall time, and host-metered
cost. Limits are optional and task-neutral. Inspect `result.budget_usage` and
`result.budget_exhaustion`; a budget stop is a typed failed result, not a
successful answer. See [Run Budgets](docs/run-budgets.md).

Tools can request approval with `@function_tool(needs_approval=True)`. By
default the run enters `WAIT_USER` before the tool body is called and emits a
`ApprovalRequestedEvent`. `ToolPolicy(approval="never")` disables that
approval gate for trusted runs. The four policy modes are `default`, `always`,
`never`, and `on_request`: `default` inherits the next configured policy,
whereas explicit `on_request` follows each tool's static or dynamic approval
declaration.

Custom tools may also attach an optional host-visible capability declaration:

```python
from vv_agent import (
    ToolIdempotency,
    ToolMetadata,
    ToolPolicy,
    ToolSideEffect,
    function_tool,
)

@function_tool(
    tool_metadata=ToolMetadata(
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=ToolIdempotency.UNSUPPORTED,
        terminal=False,
        capability_tags=["ticket.write"],
        cost_dimensions=["support_api.request"],
    )
)
def create_ticket(title: str) -> dict[str, str]:
    return {"ticket_id": "TCK-1001", "title": title}

policy = ToolPolicy(
    denied_side_effects=[ToolSideEffect.EXECUTE],
    denied_capability_tags=["filesystem.delete"],
    deny_terminal_tools=True,
    denied_cost_dimensions=["gpu.second"],
)
```

`side_effect` is one coarse declaration with no inferred hierarchy;
`capability_tags` and `cost_dimensions` are opaque exact-match labels, and cost
dimensions are not measurements or prices. `terminal=True` only declares that
a tool may return `finish` or `wait_user`; it never ends a run by itself. The
four new policy fields are cumulative denials across Agent, configured Runner,
per-run, and delegated-child layers, and a matching denial returns
`tool_not_allowed`. They cannot grant a capability or remove an existing name,
argument, approval, budget, or runtime restriction.

Typed metadata is separate from generic `FunctionTool.metadata` and is not
added to the model-visible function schema. `ToolMetadata.idempotency` is the
only idempotency declaration used by execution, telemetry, and checkpointing.

### Guardrails And Tracing

Input guardrails run before the model provider is called. Output guardrails run
after a final output is available. Trace processors receive lightweight run and
tool spans.

```python
from vv_agent import Agent, GuardrailResult, RunConfig, Runner, input_guardrail

@input_guardrail
def reject_empty(ctx, input_text: str) -> GuardrailResult:
    del ctx
    if not input_text.strip():
        return GuardrailResult.block("input is required")
    return GuardrailResult.allow()

agent = Agent(
    name="assistant",
    instructions="Answer carefully.",
    model="kimi-k3",
    input_guardrails=[reject_empty],
)

result = Runner.run_sync(
    agent,
    "Summarize this project.",
    run_config=RunConfig(default_backend="moonshot", tracing={"workflow_name": "summary"}),
)
```

### Shell Runtime Configuration (Windows)

`bash` runtime defaults are a **startup/session configuration**, not tool-call arguments.

- Run defaults: pass `bash_shell`, `windows_shell_priority`, and `bash_env`
  through `RunConfig.metadata`.
- Per-agent defaults: put the same keys in `Agent.metadata`.
- Recommended Windows priority: `["git-bash", "powershell", "cmd"]`
- On Windows, bash-tool child processes default `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` unless already overridden via the parent environment or `bash_env`.
- On Windows, bash-tool child processes are launched with hidden-console flags so GUI hosts can run `bash` / `powershell` commands without flashing a terminal window.
- `Runner.run_sync(...)` and `Runner.stream_sync(...)` both inherit compiled
  shell metadata.
- The `bash` tool schema description includes a runtime shell hint (resolved shell kind + invocation prefix), so the model sees which shell command style is expected before calling the tool.
- The runtime shell hint is frozen per task/session-run to keep tool schemas stable across cycles and preserve LLM prompt cache efficiency.
- Runner/CLI-generated runtime tasks attach structured `system_prompt_sections`
  metadata to the system message when prompt sections are available, so
  Anthropic prompt-cache breakpoints can keep the stable prompt prefix hot while
  treating current time and session-memory blocks as volatile.

```python
from vv_agent import Agent, RunConfig, Runner

agent = Agent(
    name="desktop",
    instructions="Desktop helper",
    model="kimi-k3",
    metadata={"bash_env": {"HTTP_PROXY": "http://127.0.0.1:7890"}},
)
result = Runner.run_sync(
    agent,
    "Check the workspace.",
    run_config=RunConfig(
        default_backend="moonshot",
        metadata={
            "windows_shell_priority": ["git-bash", "powershell", "cmd"],
            "bash_env": {"PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple"},
        },
    ),
)
```

## Execution Backends

The cycle loop is delegated to a pluggable `ExecutionBackend`.

| Backend | Use case |
|---------|----------|
| `InlineBackend` | Default. Synchronous, single-process. |
| `ThreadBackend` | Thread pool. Non-blocking `submit()` returns a `Future`. |
| `CeleryBackend` | Distributed. Each cycle dispatched as an independent Celery task. |

### CeleryBackend

Each cycle is a Celery task. Workers rebuild the `AgentRuntime` from a required
`RuntimeRecipe` and resolve the declared shared `CheckpointStore` capability.

```python
from vv_agent import CheckpointConfig, RunConfig
from vv_agent.runtime.backends.celery import CeleryBackend, RuntimeRecipe, register_cycle_task
from vv_agent.runtime.backends.distributed import (
    CapabilityRef,
    DistributedCapabilities,
    DistributedCapabilityRegistry,
)
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore

checkpoint_ref = CapabilityRef("checkpoint.production", "1")
checkpoint_store = SqliteCheckpointStore(".vv-agent-state/checkpoints.db")
worker_capabilities = DistributedCapabilityRegistry()
worker_capabilities.register("checkpoint_store", checkpoint_ref, checkpoint_store)
register_cycle_task(celery_app, capability_registry=worker_capabilities)

recipe = RuntimeRecipe(
    settings_file="local_settings.py",
    backend="moonshot",
    model="kimi-k3",
    workspace="./workspace",
    capabilities=DistributedCapabilities(checkpoint_store_ref=checkpoint_ref),
)
backend = CeleryBackend(celery_app=celery_app, runtime_recipe=recipe)
run_config = RunConfig(
    execution_backend=backend,
    checkpoint_config=CheckpointConfig(
        key="tenant-7/task-42",
        store=checkpoint_store,
    ),
)
```

Install celery extras: `uv sync --extra celery`.

### Cancellation and Streaming

```python
from vv_agent.events import AssistantDeltaEvent, RunEvent
from vv_agent.runtime import CancellationToken, ExecutionContext

# Cancel from another thread
token = CancellationToken()
ctx = ExecutionContext(cancellation_token=token)
result = runtime.run(task, ctx=ctx)

def on_event(event: RunEvent) -> None:
    if isinstance(event, AssistantDeltaEvent):
        print(event.delta, end="")


# Stream LLM output events, including assistant deltas and tool progress
ctx = ExecutionContext(event_handler=on_event)
result = runtime.run(task, ctx=ctx)
```

### Runtime Log Payloads

`tool_result` runtime events carry full tool output in `content` and any structured tool payload in `metadata` (no implicit truncation of `content`).
`content_preview` and `assistant_preview` are still emitted for UI convenience.

If you need shorter previews for logs/transport, configure an explicit preview limit:

```python
from vv_agent import RunConfig

config = RunConfig(
    log_preview_chars=220,  # optional: enable preview truncation explicitly
)
```

## Workspace Backends

Workspace file I/O is delegated to a pluggable `WorkspaceBackend` protocol. All built-in file tools (`read_file`, `write_file`, `find_files`, etc.) go through this abstraction.

`find_files` includes built-in safety defaults for large workspaces:

- Returns at most `100` paths per call by default (`max_results` can tune this, with hard cap).
- Uses `ripgrep` (`rg`) for fast local traversal when available, with automatic fallback to Python walk.
- `search_files` also uses `rg` for local workspaces (with Python fallback), defaults to smart-case matching (lowercase patterns are case-insensitive; patterns with uppercase stay case-sensitive), and skips hidden/common dependency roots unless explicitly included.
- `search_files` returns model-facing search text in `ToolExecutionResult.content`, while structured files/matches/counts live in `ToolExecutionResult.metadata`.
- Sensitive files such as `.env` and private keys are omitted by default; set `include_sensitive=true` to opt in.
- When listing from workspace root, common dependency/cache roots (for example `node_modules`, `.venv`, `.git`) are summarized instead of expanded.
- You can still inspect those paths explicitly by setting `path` to that directory (or by setting `include_ignored=true`).
- Supports `scan_limit` to stop early on very large trees; when triggered, response sets `count_is_estimate=true`.

| Backend | Use case |
|---------|----------|
| `LocalWorkspaceBackend` | Default. Reads/writes to a local directory with path-escape protection. |
| `MemoryWorkspaceBackend` | Pure in-memory dict storage. Great for testing and sandboxed runs. |
| `S3WorkspaceBackend` | S3-compatible object storage (AWS S3, Aliyun OSS, MinIO, Cloudflare R2). |

```python
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

# Explicit local backend
runtime = AgentRuntime(
    llm_client=llm,
    tool_registry=registry,
    workspace_backend=LocalWorkspaceBackend(Path("./workspace")),
)

# In-memory backend for testing
runtime = AgentRuntime(
    llm_client=llm,
    tool_registry=registry,
    workspace_backend=MemoryWorkspaceBackend(),
)
```

### S3WorkspaceBackend

Install the optional S3 dependency: `uv pip install 'vv-agent[s3]'`.

```python
from vv_agent.workspace import S3WorkspaceBackend

backend = S3WorkspaceBackend(
    bucket="my-bucket",
    prefix="agent-workspace",
    endpoint_url="https://oss-cn-hangzhou.aliyuncs.com",  # or None for AWS
    aws_access_key_id="...",
    aws_secret_access_key="...",
    addressing_style="virtual",  # "path" for MinIO
)
```

### Custom Backend

Implement the `WorkspaceBackend` protocol declared in
`src/vv_agent/workspace/base.py` to plug in any storage backend. A custom
backend provides file enumeration, text/binary reads, writes, metadata,
existence checks, file checks, and directory creation.

```python
from vv_agent.workspace import WorkspaceBackend

class MyBackend(WorkspaceBackend):
    ...
```

## Modules

| Module | Description |
|--------|-------------|
| `vv_agent.runtime.AgentRuntime` | Top-level state machine (completed / wait_user / max_cycles / failed) |
| `vv_agent.runtime.CycleRunner` | Single LLM turn and cycle record construction |
| `vv_agent.runtime.ToolCallRunner` | Tool execution with directive convergence |
| `vv_agent.runtime.RuntimeHookManager` | Hook dispatch (before/after LLM, tool call, memory compact) |
| `vv_agent.runtime.CheckpointStore` | Checkpoint persistence protocol (`InMemoryCheckpointStore` / `SqliteCheckpointStore` / `RedisCheckpointStore`) |
| `vv_agent.memory.MemoryManager` | Context compression when history exceeds threshold |
| `vv_agent.workspace` | Pluggable file storage: `LocalWorkspaceBackend`, `MemoryWorkspaceBackend`, `S3WorkspaceBackend` |
| `vv_agent.tools` | Built-in tools plus `function_tool`, `FunctionTool`, and structured tool outputs |
| `vv_agent` | Public SDK: `Agent`, `Runner`, `RunConfig`, `ModelSettings`, tools, sessions, typed events |
| `vv_agent.app_server` | JSONL App Server protocol, transport, thread state, replay, approval callbacks, schema export, and host provider boundary |
| `vv_agent.skills` | Agent Skills support (`SKILL.md` parsing, validation, unified normalization, prompt rendering with budget management, `activate_skill` tool) |
| `vv_agent.llm.VvLlmClient` | Unified LLM interface via `vv-llm` (endpoint rotation, retry, streaming) |
| `vv_agent.config` | Model/endpoint/key resolution from `local_settings.py` |

## Runtime Boundary

`vv-agent` owns the portable agent runtime: prompt assembly, model calls, tool
planning, tool execution, memory compaction, typed events, cancellation,
approval interruption, and replayable run history. Host products own product
UI, user and workspace resolution, product storage, browser or IM integration,
and the product-specific tools exposed to the model.

Host products should implement providers instead of patching `vv-agent`
internals:

- `AppServerHost` maps product profiles, workspaces, tools, approval UI,
  memory, context, and model settings into App Server `Agent` and `RunConfig`
  objects when the host uses JSONL process integration.
- `ApprovalProvider` decides whether a tool call needs approval and returns the
  allow, deny, session-allow, or timeout decision from product UI or rules.
- `ContextProvider` contributes product prompt fragments such as profile,
  workspace, policy, or feature context before each run is compiled.
- `vv_agent.memory.MemoryProvider` connects product memory stores to memory
  search/save hooks and compaction lifecycle events.
- `vv_agent.tools.ToolExecutor` exposes product tools with schema, approval,
  timeout, error, and execution behavior. `FunctionTool` and `@function_tool`
  cover normal Python functions; custom executors are routed by
  `ToolOrchestrator`.
- `RunEventStore` persists typed `RunEvent` history so app views can replay
  completed runs and parent/child run graphs.

This boundary keeps `Agent`, `Runner`, `RunConfig`, `RunHandle`, and
`RunEvent` stable while allowing each host to keep its own account model,
workspace model, storage backend, and UI workflow outside the framework.

## Memory Compaction

`MemoryManager` measures context size in tokens and compacts history when the
resolved auto-compaction threshold is exceeded.

- Task-level knobs:
  - `memory_compact_threshold` (default `250000`; configured ceiling for full compaction)
  - `memory_threshold_percentage` (warning threshold percentage, default `90`)
- Compile mapping:
  - `AgentCompiler` forwards stable agent/run metadata into `AgentTask`.
  - Resolved model limits are recorded as `model_context_window` and
    `model_max_output_tokens`; output capability is not copied into
    `reserved_output_tokens`.
  - Current durable task/checkpoint records carry the exact configured threshold
    and capacity metadata used by resume.
  - Runtime-only compaction knobs remain metadata-backed until promoted into
    stable public fields.
- Token budget model:
  - Context precedence is explicit `model_context_window`, resolved model
    capability, then a derived planning context. The default is
    `250000 + 16000 + 13000 = 279000`.
  - Output reserve precedence is effective `ModelSettings.max_tokens`, explicit
    `reserved_output_tokens`, then the `16000` framework fallback.
  - Only the framework fallback reserve may be capped downward by a smaller
    `model_max_output_tokens`; capability never overrides an explicit request or
    host reserve.
  - `derived_prompt_capacity = max(model_context_window - reserved_output_tokens - autocompact_buffer_tokens, 0)`
  - `autocompact_threshold = min(memory_compact_threshold, derived_prompt_capacity)`;
    a configured threshold of zero selects the derived capacity, and a known
    derived capacity of zero stays zero.
  - The default autocompact buffer is `13000`; the default microcompact trigger
    is 75% of the effective full threshold.
- Effective-length strategy (backend-aligned):
  - If previous cycle token usage exists:
    - `effective_length = previous_prompt_tokens + token_count(recent_tool_messages)`
  - Otherwise fallback to:
    - `vv_llm.chat_clients.utils.get_message_token_counts(...)`
    - If tokenizer resolution fails, use a local CJK-aware estimate
- Compaction pipeline:
  1. Preemptive microcompact: clear old large tool results when usage crosses `microcompact_trigger_ratio`
  2. Session Memory extraction: persist key facts before full summarization so they survive later compactions
  3. Structural cleanup (stale tool calls, orphan tool messages, assistant-no-tool collapse, old tool result artifactization)
  4. If still over threshold, generate a compressed memory summary that preserves original user messages, file operations, current work state, and resolved errors
  5. If the provider still returns prompt-too-long, retry with forced compaction once, then progressively stronger emergency tail-dropping
  6. After full compaction, re-inject relevant workspace files into `<Post-Compaction File Context>` under a bounded token budget
- Compaction events:
  - New `memory_compact_started` producers include the typed trigger and the
    complete resolved capacity snapshot.
  - New `memory_compact_completed` producers include the strongest actual mode
    (`none`, `micro`, `structural`, `summary`, or `emergency`) and a
    content-aware `changed` flag.
  - Every current event includes the complete typed capacity and result fields;
    missing or unknown fields are rejected.
- Session Memory behavior:
  - Stored in `workspace/.memory/session/<session-or-task-scope>/session_memory.json` by default
  - Scoped to the current session when `metadata.session_id` is present; otherwise scoped to the current `task_id`
  - New sessions/tasks start without inherited Session Memory from previous sessions/tasks
  - Injected into the first system message on every cycle as `<Session Memory>`
  - Extraction reuses the configured memory summary backend/model
  - Full compaction resets transcript tracking but preserves persisted memory entries
  - Sub-tasks disable Session Memory by default to avoid parent/child memory-file contamination

### Runtime metadata keys

Pass these via `Agent.metadata` or `RunConfig.metadata`; the compiler forwards
them into `AgentTask.metadata`:

- `memory_keep_recent_messages`
- `model_context_window`
- `model_max_output_tokens` (resolved model capability; not an implicit request limit)
- `reserved_output_tokens`
- `autocompact_buffer_tokens`
- `microcompact_trigger_ratio`
- `microcompact_keep_recent_cycles`
- `microcompact_min_result_length`
- `microcompact_compactable_tools`
- `include_memory_warning`
- `session_memory_enabled`
- `session_memory_min_tokens`
- `session_memory_max_tokens`
- `session_memory_min_text_messages`
- `session_memory_storage_dir`
- `tool_result_compact_threshold`
- `tool_result_keep_last`
- `tool_result_excerpt_head`
- `tool_result_excerpt_tail`
- `tool_calls_keep_last`
- `assistant_no_tool_keep_last`
- `tool_result_artifact_dir`
- `summary_event_limit`

### Memory summary model selection priority

Priority is strict:

1. `AgentTask.metadata.memory_summary_model`, with optional
   `memory_summary_backend`.
2. The current task model through the run's `ModelProvider`.

## Built-in Tools

`find_files`, `file_info`, `read_file`, `write_file`, `edit_file`, `search_files`, `compress_memory`, `todo_write`, `task_finish`, `ask_user`, `bash`, `read_image`, `create_sub_task`, `sub_task_status`.

Custom tools can be registered via `ToolRegistry.register()`.

The `bash` tool supports two background paths:

- Explicit background: pass `run_in_background=true`, receive a `session_id` immediately, then poll with `check_background_command`.
- Timeout handoff: if a foreground command reaches `timeout`, it is moved into a background session instead of failing immediately. The tool returns a `session_id`, and the session emits terminal background-command events when that process completes, fails, or times out.

## Sub-agents

Use `Agent.as_tool()` when the parent agent should call a child agent and then
continue. Use `handoff()` when the child agent should take over and finish the
run. Use `create_sub_task` and `sub_task_status` when the model needs explicit
background or parallel task management.

Each delegated sub-task runs in a real `AgentSession` whose session id defaults
to the sub-task id. Child `RunEvent` values preserve their run, trace, parent,
task, and session identities so hosts can subscribe, persist, and replay them
without an untyped event translation.

Batch mode in `create_sub_task` dispatches valid sub-task items through the runtime execution backend's `parallel_map`, so synchronous batches run concurrently when the backend supports parallel execution.

Use `sub_task_status` to query runtime sub-task states, inspect
lightweight progress snapshots (`detail_level=snapshot`), or send follow-up
messages to running/completed sub-tasks.

When the parent task cannot make useful progress until background sub-tasks
finish, call `sub_task_status` with `wait_for_completion=true`. The runtime waits
inside that tool call and returns when queried tasks finish or `max_wait_seconds`
is reached, avoiding repeated status-polling cycles in the agent context.

Before a completed sub-task is resumed, the runtime now sanitizes the saved session transcript: empty assistant turns, thinking-only turns, orphaned tool results, and unresolved tail tool calls are removed so the next follow-up prompt resumes from a coherent history.

Sub-task runtime metadata now includes `task_id`, `session_id`, and `browser_scope_key` for each sub-agent run, so session-scoped tools (for example, browser controllers) stay isolated across parallel sub-tasks.

Host apps can interrupt a currently running sub-agent by calling `vv_agent.runtime.engine.steer_sub_agent_session(session_id=..., prompt=...)`.

Configured child runs inherit the same explicit `ModelProvider` as the parent
and resolve their own model. No settings path or backend fallback is rebuilt
inside the child runtime.

## Examples

The `examples/` directory now contains public SDK cookbook scripts plus a small
set of lower-level runtime integration examples. See
[`examples/README.md`](examples/README.md) for the full list.

```bash
uv run python examples/01_quick_start.py
uv run python examples/24_workspace_backends.py
```

## Testing

```bash
uv run pytest                              # unit tests (no network)
uv run ruff check .                        # lint
uv run ty check                            # type check

VV_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live   # integration tests (needs real LLM)
```

Environment variables for live tests:

| Variable | Default | Description |
|----------|---------|-------------|
| `VV_AGENT_LOCAL_SETTINGS` | `local_settings.py` | Settings file path |
| `VV_AGENT_LIVE_BACKEND` | `moonshot` | LLM backend |
| `VV_AGENT_LIVE_MODEL` | `kimi-k3` | Model name |
