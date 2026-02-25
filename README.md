# vv-agent

[中文文档](README_ZH.md)

A lightweight agent framework extracted from VectorVein's production runtime. Cycle-based execution with pluggable LLM backends, tool dispatch, memory compression, and distributed scheduling.

## Architecture

```
AgentRuntime
├── CycleRunner          # single LLM turn: context -> completion -> tool calls
├── ToolCallRunner       # tool dispatch, directive convergence (finish/wait_user/continue)
├── RuntimeHookManager   # before/after hooks for LLM, tool calls, memory compaction
├── MemoryManager        # automatic history compression when context exceeds threshold
└── ExecutionBackend     # cycle loop scheduling
    ├── InlineBackend    # synchronous (default)
    ├── ThreadBackend    # thread pool with futures
    └── CeleryBackend    # distributed, per-cycle Celery task dispatch
```

Core types live in `vv_agent.types`: `AgentTask`, `AgentResult`, `Message`, `CycleRecord`, `ToolCall`.

Task completion is tool-driven: the agent calls `task_finish` or `ask_user` to signal terminal states. No implicit "last message = answer" heuristics.

## Setup

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
uv run vv-agent --prompt "Summarize this framework" --backend moonshot --model kimi-k2.5

# With per-cycle logging
uv run vv-agent --prompt "Summarize this framework" --backend moonshot --model kimi-k2.5 --verbose
```

CLI flags: `--settings-file`, `--backend`, `--model`, `--verbose`.

### Programmatic

```python
from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask

llm, resolved = build_openai_llm_from_local_settings("local_settings.py", backend="moonshot", model="kimi-k2.5")
runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry())

result = runtime.run(AgentTask(
    task_id="demo",
    model=resolved.model_id,
    system_prompt="You are a helpful assistant.",
    user_prompt="What is 1+1?",
))
print(result.status, result.final_answer)
```

### SDK

```python
from vv_agent.sdk import AgentSDKClient, AgentSDKOptions

client = AgentSDKClient(options=AgentSDKOptions(
    settings_file="local_settings.py",
    default_backend="moonshot",
    default_model="kimi-k2.5",
))
result = client.run("Explain Python's GIL in one sentence.")
print(result.final_answer)
```

### SDK Workspace Override (Session/Task)

`AgentSDKOptions.workspace` is the SDK default workspace. You can override it per one-shot run, or bind a fixed workspace to a session.

Priority for workspace resolution is:

1. Explicit `workspace` passed to `run(...)` / `query(...)` / `create_session(...)`
2. `AgentSDKOptions.workspace`

```python
from vv_agent.sdk import AgentSDKClient, AgentSDKOptions

client = AgentSDKClient(options=AgentSDKOptions(
    settings_file="local_settings.py",
    default_backend="moonshot",
    default_model="kimi-k2.5",
    workspace="./workspace/default",
))

# One-shot override: this run uses ./workspace/task-a
run = client.run(prompt="Create notes.md", workspace="./workspace/task-a")

# Session override: all turns in this session stay in ./workspace/session-b
session = client.create_session(workspace="./workspace/session-b")
session.prompt("Create todo.md")
session.follow_up("Append one more todo item")
session.continue_run()
```

Notes:

- `AgentSession.workspace` is fixed at session creation time.
- `prompt()/continue_run()/follow_up()` all execute in that same session workspace.
- Top-level SDK helpers `vv_agent.sdk.run(...)` and `vv_agent.sdk.query(...)` also accept `workspace=...`.

## Execution Backends

The cycle loop is delegated to a pluggable `ExecutionBackend`.

| Backend | Use case |
|---------|----------|
| `InlineBackend` | Default. Synchronous, single-process. |
| `ThreadBackend` | Thread pool. Non-blocking `submit()` returns a `Future`. |
| `CeleryBackend` | Distributed. Each cycle dispatched as an independent Celery task. |

### CeleryBackend

Two modes:

- **Inline fallback** (no `RuntimeRecipe`): cycles run in-process, same as `InlineBackend`.
- **Distributed** (with `RuntimeRecipe`): each cycle is a Celery task. Workers rebuild the `AgentRuntime` from the recipe and load state from a shared `StateStore` (SQLite or Redis).

```python
from vv_agent.runtime.backends.celery import CeleryBackend, RuntimeRecipe, register_cycle_task

register_cycle_task(celery_app)

recipe = RuntimeRecipe(
    settings_file="local_settings.py",
    backend="moonshot",
    model="kimi-k2.5",
    workspace="./workspace",
)
backend = CeleryBackend(celery_app=app, state_store=store, runtime_recipe=recipe)
runtime = AgentRuntime(llm_client=llm, tool_registry=registry, execution_backend=backend)
```

Install celery extras: `uv sync --extra celery`.

### Cancellation and Streaming

```python
from vv_agent.runtime import CancellationToken, ExecutionContext

# Cancel from another thread
token = CancellationToken()
ctx = ExecutionContext(cancellation_token=token)
result = runtime.run(task, ctx=ctx)

# Stream LLM output token by token
ctx = ExecutionContext(stream_callback=lambda text: print(text, end=""))
result = runtime.run(task, ctx=ctx)
```

### Runtime Log Payloads

`tool_result` runtime events now carry full tool output in `result`/`content` by default (no implicit truncation).
`content_preview` and `assistant_preview` are still emitted for UI convenience.

If you need shorter previews for logs/transport, configure an explicit preview limit:

```python
from vv_agent.sdk import AgentSDKOptions

options = AgentSDKOptions(
    settings_file="local_settings.py",
    default_backend="moonshot",
    log_preview_chars=220,  # optional: enable preview truncation explicitly
)
```

## Workspace Backends

Workspace file I/O is delegated to a pluggable `WorkspaceBackend` protocol. All built-in file tools (`read_file`, `write_file`, `list_files`, etc.) go through this abstraction.

`list_files` includes built-in safety defaults for large workspaces:

- Returns at most `500` paths per call by default (`max_results` can tune this, with hard cap).
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

Implement the `WorkspaceBackend` protocol (8 methods) to plug in any storage:

```python
from vv_agent.workspace import WorkspaceBackend

class MyBackend:
    def list_files(self, base: str, glob: str) -> list[str]: ...
    def read_text(self, path: str) -> str: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_text(self, path: str, content: str, *, append: bool = False) -> int: ...
    def file_info(self, path: str) -> FileInfo | None: ...
    def exists(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...
    def mkdir(self, path: str) -> None: ...
```

## Modules

| Module | Description |
|--------|-------------|
| `vv_agent.runtime.AgentRuntime` | Top-level state machine (completed / wait_user / max_cycles / failed) |
| `vv_agent.runtime.CycleRunner` | Single LLM turn and cycle record construction |
| `vv_agent.runtime.ToolCallRunner` | Tool execution with directive convergence |
| `vv_agent.runtime.RuntimeHookManager` | Hook dispatch (before/after LLM, tool call, memory compact) |
| `vv_agent.runtime.StateStore` | Checkpoint persistence protocol (`InMemoryStateStore` / `SqliteStateStore` / `RedisStateStore`) |
| `vv_agent.memory.MemoryManager` | Context compression when history exceeds threshold |
| `vv_agent.workspace` | Pluggable file storage: `LocalWorkspaceBackend`, `MemoryWorkspaceBackend`, `S3WorkspaceBackend` |
| `vv_agent.tools` | Built-in tools: workspace I/O, todo, bash, image, sub-agents, skills |
| `vv_agent.sdk` | High-level SDK: `AgentSDKClient`, `AgentSession`, `AgentResourceLoader` |
| `vv_agent.skills` | Agent Skills support (`SKILL.md` parsing, prompt injection, activation) |
| `vv_agent.llm.VVLlmClient` | Unified LLM interface via `vv-llm` (endpoint rotation, retry, streaming) |
| `vv_agent.config` | Model/endpoint/key resolution from `local_settings.py` |

## Memory Compaction

`MemoryManager` compacts history when `AgentTask.memory_compact_threshold` is exceeded.

- Task-level knobs:
  - `memory_compact_threshold` (default `128000`)
  - `memory_threshold_percentage` (warning threshold percentage, default `90`)
- Effective-length strategy (backend-aligned):
  - If previous cycle token usage exists:
    - `effective_length = previous_total_tokens + len(json.dumps(recent_tool_messages))`
  - Otherwise fallback to:
    - `len(json.dumps(messages[2:]))`
- Compaction pipeline:
  1. Structural cleanup (stale tool calls, orphan tool messages, assistant-no-tool collapse, old tool result artifactization)
  2. If still over threshold, generate compressed memory summary

### Runtime metadata keys

Pass these via `AgentTask.metadata`:

- `memory_keep_recent_messages`
- `include_memory_warning`
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

1. `AgentTask.metadata`
   - `memory_summary_backend` / `memory_summary_model`
   - aliases: `compress_memory_summary_backend` / `compress_memory_summary_model`
   - aliases: `memory_compress_backend` / `memory_compress_model`
2. `local_settings.py` constants
   - `DEFAULT_USER_MEMORY_SUMMARIZE_BACKEND` / `DEFAULT_USER_MEMORY_SUMMARIZE_MODEL`
   - aliases: `DEFAULT_MEMORY_SUMMARIZE_BACKEND` / `DEFAULT_MEMORY_SUMMARIZE_MODEL`
   - aliases: `VV_AGENT_MEMORY_SUMMARY_BACKEND` / `VV_AGENT_MEMORY_SUMMARY_MODEL`
3. Fallback
   - runtime `default_backend` + current task `model`

## Built-in Tools

`list_files`, `file_info`, `read_file`, `write_file`, `file_str_replace`, `workspace_grep`, `compress_memory`, `todo_write`, `task_finish`, `ask_user`, `bash`, `read_image`, `create_sub_task`, `batch_sub_tasks`.

Custom tools can be registered via `ToolRegistry.register()`.

## Sub-agents

Configure named sub-agents on `AgentTask.sub_agents`. The parent agent delegates work via `create_sub_task` / `batch_sub_tasks`. Each sub-agent gets its own runtime, model, and tool set.

When a sub-agent uses a different model from the parent, the runtime needs `settings_file` and `default_backend` to resolve the LLM client.

## Examples

24 numbered examples in `examples/`. See [`examples/README.md`](examples/README.md) for the full list.

```bash
uv run python examples/01_quick_start.py
uv run python examples/24_workspace_backends.py
```

## Testing

```bash
uv run pytest                              # unit tests (no network)
uv run ruff check .                        # lint
uv run ty check                            # type check

V_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live   # integration tests (needs real LLM)
```

Environment variables for live tests:

| Variable | Default | Description |
|----------|---------|-------------|
| `V_AGENT_LOCAL_SETTINGS` | `local_settings.py` | Settings file path |
| `V_AGENT_LIVE_BACKEND` | `moonshot` | LLM backend |
| `V_AGENT_LIVE_MODEL` | `kimi-k2.5` | Model name |
| `V_AGENT_ENABLE_BASE64_KEY_DECODE` | - | Set `1` to enable base64 API key decoding |
