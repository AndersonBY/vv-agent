# vv-agent

[中文文档](README_ZH.md)

A lightweight agent framework extracted from VectorVein's production runtime. Cycle-based execution with pluggable LLM backends, tool dispatch, memory compression, and durable session scheduling.

## Install

`contract.lock.json` pins this repository's language-neutral contract. The
schema-2 central support matrix records adoption and verification: Python is the
required implementation; `vv-agent-rs` is frozen at contract `23.0.0` / package
series `0.21.x` for maintenance only. Python adoption does not require a Rust
update. See [the contract workflow](docs/parity-contract.md). This repository
keeps a Python-idiomatic API.

```bash
python -m pip install -e .
```

Install the `postgres` extra for PostgreSQL SessionStore or the `s3` extra
for S3 workspace storage. Repository `HEAD` is forward-only: current readers
accept only the current strict public and wire shapes.

For both extras: `python -m pip install 'vv-agent[postgres,s3]'`.
See the [0.23.0 release notes](docs/releases/0.23.0.md) for breaking changes.

Current HEAD adopts contract v25, public API v9 and one session kernel execution path.
Older runtime behavior is retained in Git tags. See [v8 migration](docs/migration-v8.md)
for API replacements and host seed examples.
See [cloud host integration](docs/host-integration.md) for PostgreSQL transactions,
queue dispatch and a host-owned Celery example.

## Architecture

```text
Runner / InteractiveAgentClient / CLI / AppServer
  -> SessionDriver -> SessionStore (SQLite or PostgreSQL)
  -> session.kernel.drive
  -> retained model/tool operations and child delivery
  -> RunResult / RunEvent / tracing / protocol projections
```

The public SDK entry points are exported from `vv_agent`: `Agent`, `Runner`,
`RunConfig`, `RunHandle`, `ModelSettings`, `function_tool`, `SessionStore`,
`PromptBundle`, `PromptSection`, `ToolExecutionResult`, `ToolArtifactRef`,
`ToolResultCursor`, typed `RunEvent` objects, `ApprovalProvider`,
`ContextProvider`, `SessionRunEventStore`, and the interactive session API for
desktop/runtime integrations. Extension points that live in package modules include
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

Ordinary Runner runs use a fresh SQLite `:memory:` store. `Runner.start()` returns
 a live RunHandle; events(), result(), cancel() and approve() share the same
kernel path. Durable events derive from the session log; assistant deltas are
volatile live observations. A JsonlRunEventStore is an optional projection sink.

```python
from vv_agent import Agent, RunConfig, Runner

agent = Agent("assistant", "Answer briefly.", model="kimi-k3")
handle = Runner.start(agent, "Inspect the project", run_config=RunConfig(default_backend="moonshot"))
for event in handle.events():
    if event.type == "assistant_delta":
        print(event.delta, end="")
print(handle.result().final_output)
```

Use InteractiveAgentClient for multiple turns and SessionStore for durable
retention. Waiting operations retain the same session_id and turn_id. After
answering a parked approval, explicitly resume the handle or retained turn.

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

InteractiveAgentClient owns conversation turns, steering, follow-up, cancellation
and approvals. Its default store is SQLite `:memory:`. Supply SQLiteStore or
PostgresStore through AgentSessionOptions.session_store for durable retention.
Creation-time `session` is a closed seed with messages and shared_state; messages
and shared_state become read-only projections after creation.

```python
from pathlib import Path
from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, SQLiteStore, VvLlmModelProvider

with SQLiteStore.standalone("sessions.sqlite3") as store:
    if not store.connection.execute("PRAGMA user_version").fetchone()[0]:
        store.install_schema()
    client = InteractiveAgentClient(options=AgentSessionOptions(
        model_provider=VvLlmModelProvider(Path("local_settings.py"), default_backend="moonshot"),
        session_store=store,
    ))
    try:
        session = client.create_session(
            agent=Agent("assistant", "Remember prior turns.", model="kimi-k3"),
            session_id="thread-001",
        )
        print(session.prompt("Remember the project codename River.").final_output)
        print(session.prompt("What is the codename?").final_output)
    finally:
        client.driver.close()
```

Runner.resume(session_id, turn_id), AgentSession.continue_run() and App Server
turn/resume drive an explicitly retained turn. Replies preserve turn budgets;
fresh turns reset their counters. Closed sessions reject execution.

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
Agent value; omitting every layer uses `finish`. Inspect
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
only idempotency declaration used by execution, telemetry, and retained operations.

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
- The runtime shell hint is frozen per task/session-run for local LLM requests to keep those request schemas stable across cycles and preserve prompt-cache efficiency. Frozen turn definitions retain the canonical schemas planned for the compiled task, so host-specific hint text does not affect the task-scoped toolset digest.
- Runner/CLI-generated tasks carry one resolved `PromptBundle` explicitly
  through `AgentTask`, each `LlmRequest`, the frozen turn definition and logged model operations. Generic metadata is not a prompt-section transport.
  Anthropic projection may use the canonical sections for cache breakpoints;
  other providers receive the deterministic flattened prompt.

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

## Session Execution

All public entrypoints use SessionDriver and the session kernel. SessionStore
transactions own the log, inbox, leases, child admission and retained receipts.
Ordinary runs use SQLite `:memory:`; durable SQLite/PostgreSQL are opt-in through
the session API. Runner.start() provides non-blocking execution and cancellation.

```python
from vv_agent import Agent, RunConfig, Runner

handle = Runner.start(Agent("assistant", "Answer briefly.", model="kimi-k3"), "Explain session history.",
                      run_config=RunConfig(default_backend="moonshot"))
# A host can call handle.cancel() from its UI or a timer.
print(handle.result().final_output)
```

See [runtime-control.md](docs/runtime-control.md) for waits, explicit resume,
LeaseLost backoff, budgets and event projections.

### Runtime Log Payloads

The `tool_result` diagnostic contains the model-visible `content`, ordinary
metadata, and a bounded `content_preview`; it does not duplicate artifact or
cursor fields. Structured recovery belongs to `ToolExecutionResult` and is
preserved in cycle results and retained operation receipts. A
bounded bash result points to an immutable workspace artifact, while a bounded
`read_file` result points to a source-verified cursor. Hosts must read artifacts
through normal workspace policy; cursors reject changed sources, path
mismatches, and invalid offsets.

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
from pathlib import Path
from vv_agent import RunConfig
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

local_config = RunConfig(workspace_backend=LocalWorkspaceBackend(Path("./workspace")))
memory_config = RunConfig(workspace_backend=MemoryWorkspaceBackend())
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
| `vv_agent.runtime.RuntimeHookManager` | Hook dispatch (before/after LLM, tool call, memory compact) |
| `vv_agent.session.SessionStore` | Durable log, inbox, leases and projections |
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
- `SessionRunEventStore` projects typed `RunEvent` history so app views can replay
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
  - Current durable turn definitions carry the exact configured threshold
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
  - The default autocompact buffer is `13000`. `MicrocompactionPolicy` defaults
    to trigger/target ratios of `0.75`/`0.60`, keeps 3 recent cycles, and only
    considers results longer than 500 characters.
- Effective-length strategy (backend-aligned):
  - If previous cycle token usage exists:
    - `effective_length = previous_prompt_tokens + token_count(recent_tool_messages)`
  - Otherwise fallback to:
    - `vv_llm.chat_clients.utils.get_message_token_counts(...)`
    - If tokenizer resolution fails, use a local CJK-aware estimate
- Compaction pipeline:
  1. One microcompaction pass archives eligible old tool results. Age is relative
     to assistant turns in the current transcript. Failed persistence keeps the
     original result; calls and results stay paired. A planned summary protects its raw tail.
  2. If still over threshold, summarize the previous summary and complete history
     prefix. `memory_keep_recent_messages` defaults to 10 raw messages; a cut inside
     a tool block moves left to retain the assistant and every corresponding result.
  3. Extract and normalize the model response, then replace the prefix only after
     effective-content, context-budget, recovery and token-reduction checks pass.
     Failure retains history; system messages and the raw tail remain unchanged.
  4. Summary metadata preserves artifact/cursor evidence deterministically. The
     model sees paths and retrieval hints; file references never trigger automatic reads.
  5. Prompt-too-long retries force a summary, then re-summarize with smaller tails.
     No successful shrink reaches the existing exhaustion boundary. Enabled Session
     Memory updates its baseline to the accepted summary plus tail.
- Compaction events:
  - New `memory_compact_started` producers include the typed trigger and the
    complete resolved capacity snapshot plus the micro target, candidate count,
    and estimated reclaimable tokens.
  - New `memory_compact_completed` producers include the strongest actual mode
    (`none`, `micro`, `structural`, `summary`, or `emergency`) and a
    content-aware `changed` flag plus archive count, actual reclaimed tokens,
    and artifact failure count.
  - Every current event includes the complete typed capacity and result fields;
    missing or unknown fields are rejected.
- Archive recovery:
  - Every tool defaults to `ToolResultRetention.ARCHIVE`; `PRESERVE` excludes a
    result only from proactive microcompaction.
  - Complete text is persisted through the effective `WorkspaceBackend` under
    the immutable logical `.vv-agent/artifacts/` namespace before replacement.
    A failed or short write leaves the original message inline.
  - Microcompaction is disabled when model-visible `read_file` is unavailable,
    including `use_workspace=False` and explicit tool exclusion.
  - Existing typed artifacts are reused. The model sees only the compact
    marker's tool name, artifact path, retrieval hint, and bounded excerpt.
- Configure proactive compaction with
  `RunConfig(microcompaction_policy=MicrocompactionPolicy(...))`. The policy is
  copied to `AgentTask` and frozen/restored under
  `runtime_controls.microcompaction_policy` in the run definition.
- The model-visible replacement has this closed shape; byte size and SHA-256
  remain only in the host-visible `ToolArtifactRef`:

```text
<Tool Result Compact>
tool_name: web_search
artifact_path: .vv-agent/artifacts/<run>/<call>.txt
retrieval_hint: use read_file on artifact_path if needed
excerpt:
<bounded head/tail preview>
</Tool Result Compact>
```
- Session Memory behavior:
  - Stored in `workspace/.memory/session/<session-or-task-scope>/session_memory.json` by default
  - Scoped to the current session when `metadata.session_id` is present; otherwise scoped to the current `task_id`
  - New sessions/tasks start without inherited Session Memory from previous sessions/tasks
  - Loaded once when a new run is compiled and frozen into its first system
    message as `<Session Memory>`; every cycle reuses the same `PromptBundle`
  - Entries extracted during the active run are persisted but become visible
    only when the next new run is compiled
  - Retained-turn resume reuses the frozen memory section without rereading the
    store or rewriting the active prompt
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
- `include_memory_warning`
- `session_memory_enabled`
- `session_memory_min_tokens`
- `session_memory_max_tokens`
- `session_memory_min_text_messages`
- `session_memory_storage_dir`
- `tool_result_excerpt_head`
- `tool_result_excerpt_tail`
- `summary_event_limit`

### Memory summary model selection priority

Priority is strict:

1. `AgentTask.metadata.memory_summary_model`, with optional
   `memory_summary_backend`.
2. The current task model through the run's `ModelProvider`.

## Built-in Tools

`find_files`, `file_info`, `read_file`, `write_file`, `edit_file`, `search_files`, `todo_write`, `ask_user`, `bash`, `check_background_command`, `stop_background_command`, `read_image`, `create_sub_task`, `sub_task_status`, `activate_skill`.

Custom tools can be registered via `ToolRegistry.register()`.

`bash` waits up to `yield_time_ms` (default 1000, integer 0..10000) for the
command to finish, then returns its current output and a `session_id` while the
process keeps running. Zero requests an immediate handle. Optional
`timeout_seconds` (integer 1..86400) sets one execution deadline from process
start; omission means no execution deadline. Querying does not extend it.

`check_background_command({"session_id":"bg_..."})` reads a current snapshot;
`stop_background_command({"session_id":"bg_..."})` requests process-tree
termination. A successful start or running query is a completed `SUCCESS` /
`continue` management receipt, so the session kernel can call its model
again. The actual process state is in content and metadata. Nonzero observed
exit codes remain errors. Sessions belong to their initiating task and workspace;
a missing local record does not establish whether an external process exited.
Live oversized output includes an immutable artifact for complete recovery.
See [Bash process management](docs/bash-process-management.md) for examples and
termination observations.

## Sub-agents

Use `Agent.as_tool()` when the parent agent should call a child agent and then
continue. Use `handoff()` when the child agent should take over and finish the
run. Use `create_sub_task` and `sub_task_status` when the model needs explicit
background or parallel task management.

Each delegated task is an independently scheduled kernel session. Parent admission
freezes its identity, prompt, policy, model and budget; the parent adopts only its
authenticated terminal. Intermediate user waits stay on the child. Batch children
can drive independently after the parent releases its lease.

Use `sub_task_status` to read owner-scoped progress, wait for completion or admit
a continuation message. Hosts steer a retained child by pushing a targeted
`steer` inbox item through its store or interactive session. Child lifecycle events
carry parent and child identities for subscription and replay.

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
