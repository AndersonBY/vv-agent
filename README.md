# vv-agent

[中文文档](README_ZH.md)

vv-agent is a lightweight Python agent framework extracted from VectorVein's
production runtime. An agent runs in cycles: each cycle calls the model, runs
the tools it asked for, and decides whether to continue. The framework provides
pluggable LLM backends, tool dispatch, context compaction, and durable sessions
that can resume after a crash.

## Install

```bash
python -m pip install vv-agent
```

Two optional extras add storage backends:

- `postgres` for the PostgreSQL session store (`PostgresStore`).
- `s3` for the S3 workspace backend (`S3WorkspaceBackend`).

```bash
python -m pip install 'vv-agent[postgres,s3]'
```

To work from a checkout, run `python -m pip install -e .` or `uv sync --dev`.

vv-agent 0.27.0 implements version 29.1.0 of the language-neutral
`vv-agent-contract`, pinned in `contract.lock.json`. Python is the maintained
implementation. The Rust port, `vv-agent-rs`, is frozen at contract 23.0.0
(package series 0.21.x) and only gets maintenance fixes. The Python API keeps
Python naming; see [the contract workflow](docs/parity-contract.md).

If you upgrade from an earlier release and keep sessions in SQLite files or
PostgreSQL, read [Upgrading a host](docs/upgrading.md) first. A new release can
make stored sessions unreadable, and that guide explains how to get across.

## Configure a model

Model endpoints and API keys live in a Python settings file. Copy the template
and fill in your keys:

```bash
cp local_settings.example.py local_settings.py
```

The template defines a `moonshot` backend with the `kimi-k3` model. The CLI,
examples and live tests use those two names by default. Never commit
`local_settings.py`. See [Model settings](docs/model-settings.md) for the
settings shape and how a model key is resolved.

## Quick start

### CLI

```bash
uv run vv-agent --prompt "Summarize this framework" --backend moonshot --model kimi-k3

# Print a log line for every cycle
uv run vv-agent --prompt "Summarize this framework" --verbose
```

| Flag | Default |
| --- | --- |
| `--backend` | `moonshot` |
| `--model` | `kimi-k3` |
| `--settings-file` | `$VV_AGENT_LOCAL_SETTINGS`, else `local_settings.py` |
| `--workspace` | `./workspace` |
| `--max-cycles` | `80` |
| `--language` | `zh-CN` (system prompt language; `en-US` also works) |
| `--agent-type`, `--temperature`, `--top-p`, `--max-tokens` | unset |
| `--verbose` | off |

### Python

```python
from vv_agent import Agent, RunConfig, Runner, VvLlmModelProvider, function_tool

provider = VvLlmModelProvider.from_settings_file("local_settings.py").with_default_backend("moonshot")

@function_tool
def read_order(order_id: str) -> str:
    """Read order information."""
    return f"Order {order_id}: 2 items, shipped"

agent = Agent(
    name="ops",
    instructions="Check facts first, then answer.",
    model="kimi-k3",
    tools=[read_order],
)

result = Runner.run_sync(agent, "Analyze order 123", run_config=RunConfig(model_provider=provider))
print(result.status, result.final_output)
```

Every run needs a `ModelProvider`. `VvLlmModelProvider` reads the settings file
and calls the model through `vv-llm`. The examples below reuse `provider`.

By default the run finishes when the model replies without calling a tool. The
framework does not guess whether a reply "looks final"; it follows the declared
no-tool policy, described in [Finishing a run](#finishing-a-run).

### Shared defaults

When several runs share the same settings, configure a Runner once:

```python
from vv_agent import RunConfig, Runner

runner = Runner.configured(RunConfig(
    model_provider=provider,
    model="kimi-k3",
    workspace="./workspace",
))
result = runner.run_sync(agent, "Analyze order 123")
```

Values are resolved in this order:

- Provider: the per-run `RunConfig`, then the Runner default.
- Model: the per-run config, then `Agent.model`, then the Runner default, then
  the provider's default model. The Runner's model is skipped when the per-run
  config brings its own provider.
- `ModelSettings`: merged field by field from provider defaults, then Runner,
  then Agent, then per-run. Later layers win.

Set `Agent.output_type` to coerce the JSON final output into a `dict`, a
`list`, a dataclass or a Pydantic-style model. See
[Output validation](docs/output-validation.md) for validation and repair.

## Streaming and cancellation

`Runner.start()` returns a `RunHandle` right away and runs the agent in the
background. Use `events()` to stream typed `RunEvent` objects, `result()` to
wait for the result, and `cancel()` to stop the run, for example from a UI
button or a timer.

```python
from vv_agent import Agent, RunConfig, Runner

agent = Agent("assistant", "Answer briefly.", model="kimi-k3")
handle = Runner.start(agent, "Inspect the project", run_config=RunConfig(model_provider=provider))
for event in handle.events():
    if event.type == "assistant_delta":
        print(event.delta, end="")
print(handle.result().final_output)
```

Every run is a session in a session store. A plain Runner run uses a fresh
in-memory SQLite store. Durable events are read back from the session log.
Assistant deltas are live only and are not stored. A `JsonlRunEventStore` can
copy events to a file if you want one. See
[Runtime control](docs/runtime-control.md) for waits, resume, budgets and event
projections.

## Multi-turn sessions

`InteractiveAgentClient` manages conversations: turns, steering, follow-up
messages, cancellation and approvals. Its default store is in memory. Pass a
`SQLiteStore` or `PostgresStore` as `AgentSessionOptions.session_store` to keep
sessions across restarts.

```python
from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, SQLiteStore

with SQLiteStore.standalone("sessions.sqlite3") as store:
    if store.connection.execute("PRAGMA user_version").fetchone()[0] == 0:
        store.install_schema()
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, session_store=store))
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

A few rules about sessions:

- `create_session(session=...)` accepts a seed with `messages` and
  `shared_state`, and nothing else. After creation, `session.messages` and
  `session.shared_state` are read-only views. See
  [Creation-time seed](docs/upgrading.md#creation-time-seed).
- A turn that waits for a user reply or an approval keeps its `session_id` and
  `turn_id`. After you answer, drive the same turn again with
  `Runner.resume(session_id, turn_id)`, `AgentSession.continue_run()` or the App
  Server `turn/resume` request.
- A resumed turn keeps its original budget counters. A new turn starts with
  fresh counters.
- A closed session rejects new work.

To run sessions on many workers against one PostgreSQL database, see
[Cloud host integration](docs/host-integration.md).

## App Server

Use the App Server when a desktop app, worker, IDE or another process should
drive vv-agent through a stable protocol instead of importing the Python SDK. It
speaks JSONL over stdio and offers:

- Thread, Turn and Item lifecycle events.
- Tool approvals sent to the client as server-to-client requests.
- `thread/read` and `thread/resume` replay.
- Generated JSON Schema and self-contained TypeScript bindings.

```bash
uv run vv-agent app-server --listen stdio --settings local_settings.py --backend moonshot --model kimi-k3
uv run vv-agent app-server schema --out ./app-server-schema
uv run vv-agent app-server generate-ts --out ./app-server-schema/typescript
uv run vv-agent debug app-server send-message "hello"
```

The App Server takes `--settings`, not `--settings-file`. `--listen`,
`--settings`, `--backend` and `--model` are required; `--timeout-seconds` sets
the provider request timeout.

A product host implements `AppServerHost` to turn its own profiles, workspace
context, tools, approval UI, memory and model settings into `Agent` and
`RunConfig` objects. The App Server never imports product UI, account, billing,
browser or IM code. See [App Server protocol](docs/app-server.md) and
[App Server host integration](docs/app-server-host-integration.md).

## Tools

### Custom tools

Decorate a Python function with `@function_tool` and pass it in
`Agent(tools=[...])`. The decorator builds the JSON schema from the signature.
A tool may take a `ToolContext` as its first parameter; the runtime passes it
in and leaves it out of the schema.

For lower-level tools, register them on a `ToolRegistry` (`register()` or
`register_tool()`) and supply the registry through
`RunConfig.tool_registry_factory`.

### Approvals

A tool declared with `@function_tool(needs_approval=True)` pauses the run
before its body runs. The run enters `WAIT_USER` and emits an
`ApprovalRequestedEvent`. Answer it, then resume the handle:

```python
from vv_agent import Agent, ApprovalDecision, RunConfig, Runner, function_tool

@function_tool(needs_approval=True)
def delete_temp_files() -> str:
    """Delete temporary files."""
    return "deleted 3 files"

agent = Agent("ops", "Clean up when asked.", model="kimi-k3", tools=[delete_temp_files])
handle = Runner.start(agent, "Clean the workspace", run_config=RunConfig(model_provider=provider))
for event in handle.events():
    if event.type == "approval_requested":
        handle.approve(event.request_id, ApprovalDecision.allow())

result = handle.result()
if result.status == "wait_user":
    result = handle.resume()
print(result.final_output)
```

`ToolPolicy(approval=...)` takes `default`, `always`, `never` (for trusted
runs) or `on_request`. An `ApprovalProvider` can answer from product rules or
UI with allow, deny, allow for this session, or timeout. See
[Approval](docs/architecture.md#approval) for how the layers combine.

### Capability metadata and policy

A tool can declare what kind of thing it does. The host sees this metadata; the
model does not.

```python
from vv_agent import ToolIdempotency, ToolMetadata, ToolPolicy, ToolSideEffect, function_tool

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
    """Create a support ticket."""
    return {"ticket_id": "TCK-1001", "title": title}

policy = ToolPolicy(
    denied_side_effects=[ToolSideEffect.EXECUTE],
    denied_capability_tags=["filesystem.delete"],
    deny_terminal_tools=True,
    denied_cost_dimensions=["gpu.second"],
)
```

`side_effect` is one coarse label with no hierarchy. `capability_tags` and
`cost_dimensions` are opaque labels matched exactly. `terminal=True` only says the tool may return
`finish` or `wait_user`. The metadata never reaches the model-visible schema.

The `denied_*` and `deny_terminal_tools` policy fields only remove tools. They
add up across the Agent, the configured Runner, the per-run config and any
delegated child, and a matching call returns `tool_not_allowed`. See
[Tool metadata](docs/architecture.md#tool-metadata) and
[Tool policy](docs/architecture.md#tool-policy).

### Built-in tools

`find_files`, `file_info`, `read_file`, `write_file`, `edit_file`,
`search_files`, `todo_write`, `ask_user`, `bash`, `check_background_command`,
`stop_background_command`, `read_image`, `create_sub_task`, `sub_task_status`,
`activate_skill`.

File search is tuned for large workspaces:

- `find_files` returns at most 100 paths unless you raise `max_results` (up to a
  hard cap). `scan_limit` stops early on huge trees and marks the count with
  `count_is_estimate=true`.
- Both `find_files` and `search_files` use `ripgrep` (`rg`) on local workspaces
  when it is installed, and a Python walk otherwise. `search_files` is
  smart-case: an all-lowercase pattern ignores case. Its structured matches and
  counts are in `ToolExecutionResult.metadata`.
- Hidden directories and dependency or cache roots such as `node_modules`,
  `.venv` and `.git` are skipped, and only summarized in a root listing. Point
  `path` at one, or set `include_ignored=true`, to look inside. Sensitive files
  such as `.env` and private keys need `include_sensitive=true`.

`bash` waits up to `yield_time_ms` (default 1000 ms) for a command. If it is
still running, the tool returns the output so far and a `session_id`, and the
process keeps going. `timeout_seconds` sets an optional deadline from process
start. `check_background_command` and `stop_background_command` take that
`session_id`. See [Bash process management](docs/bash-process-management.md).

When a `bash` or `read_file` result is truncated, `ToolExecutionResult` points
to the rest: an immutable workspace artifact for `bash`, and a cursor for
`read_file`. A cursor rejects a changed file, a different path or an invalid
offset. The `tool_result` diagnostic event carries the model-visible `content`,
status, directive, error code and metadata, but not the artifact or cursor.

## Multi-agent runs

Use `agent.as_tool()` when a child agent should return a result and the parent
should carry on. Use `handoff()` when the target agent should take over and its
output should finish the run.

```python
from vv_agent import Agent, RunConfig, Runner, handoff

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
    run_config=RunConfig(model_provider=provider, max_handoffs=4),
)
```

The model performs a handoff by calling a `transfer_to_<agent name>` tool
(`transfer_to_writer` here). The Runner then switches agents. The target agent
resolves its own model and model settings. The session, the cancellation token
and any changes to shared state carry over. `max_handoffs` defaults to `10` and
is counted separately from `max_cycles`. A run that resumes after an approval
follows the same rules.

The model can also manage background or parallel work itself with
`create_sub_task` and `sub_task_status`. Each child is its own session with its
own lease (a worker's claim to drive a session). The parent fixes the child's
prompt, policy, model and budget when it admits the child, and accepts only the
child's own final result. A child that waits for a user keeps that wait.
`sub_task_status` reads progress, waits for completion or sends a follow-up; a
host steers a child by pushing a targeted `steer` inbox item. A child reuses the
parent's `ModelProvider` and resolves its own model. See
[Delegation and background work](docs/runtime-control.md#delegation-and-background-work).

## Finishing a run

A run ends or pauses in one of two ways:

- A tool returns a `finish` or `wait_user` directive. `ask_user`, for example,
  pauses the run for a reply.
- The model replies without calling a tool, and the no-tool policy decides.

The no-tool policy is `finish` unless you change it. Set it on the agent with
`Agent(no_tool_policy=...)`, or for one call with `RunConfig(no_tool_policy=...)`.
The values are `finish`, `continue` and `wait_user`. A per-run value beats a
configured Runner default, which beats the Agent value.

Read `result.completion_reason` to see why the run stopped: a tool finish, a
no-tool finish, a wait, cancellation, failure, budget exhaustion or running out
of cycles. `result.completion_tool_name` and `result.partial_output` add detail.
Plain runs default to 10 cycles (`max_cycles`); the CLI uses 80.

`RunConfig.budget_limits` takes a `RunBudgetLimits`. It can cap total tokens,
uncached input tokens, all tool calls or calls to one tool, active wall time and
host-metered cost. All limits are optional. They limit resources and never judge
the answer. A run that hits a budget ends as a typed failure, not a success;
inspect `result.budget_usage` and `result.budget_exhaustion`. See
[Run budgets](docs/run-budgets.md).

## Guardrails and tracing

Input guardrails run before the model is called. Output guardrails run once a
final output exists. Trace processors receive lightweight run and tool spans.

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
    run_config=RunConfig(model_provider=provider, tracing={"workflow_name": "summary"}),
)
```

## Workspaces

Built-in file tools read and write through a `WorkspaceBackend`.

| Backend | Use it for |
| --- | --- |
| `LocalWorkspaceBackend` | The default. A local directory, with protection against paths that escape it. |
| `MemoryWorkspaceBackend` | An in-memory dict. Good for tests and sandboxed runs. |
| `S3WorkspaceBackend` | S3-compatible object storage: AWS S3, Aliyun OSS, MinIO, Cloudflare R2. Needs the `s3` extra. |

```python
from pathlib import Path
from vv_agent import RunConfig
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

local_config = RunConfig(workspace_backend=LocalWorkspaceBackend(Path("./workspace")))
memory_config = RunConfig(workspace_backend=MemoryWorkspaceBackend())
```

```python
from vv_agent.workspace import S3WorkspaceBackend

s3_backend = S3WorkspaceBackend(
    bucket="my-bucket",
    prefix="agent-workspace",
    endpoint_url="https://oss-cn-hangzhou.aliyuncs.com",  # None for AWS
    aws_access_key_id="...",
    aws_secret_access_key="...",
    addressing_style="virtual",  # "path" for MinIO
)
```

To plug in other storage, implement the `WorkspaceBackend` protocol in
`src/vv_agent/workspace/base.py`. It covers listing files, text and binary
reads, plain and exclusive (create-only) writes, file info, existence and file
checks, and creating directories.

## Shell settings

The `bash` tool's shell is chosen when the run starts, not per tool call. Set
these keys in `RunConfig.metadata` for one run, or in `Agent.metadata` for an
agent:

- `bash_shell`: the shell to use.
- `windows_shell_priority`: the order to try shells on Windows. We recommend
  `["git-bash", "powershell", "cmd"]`.
- `bash_env`: extra environment variables for commands.

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
        model_provider=provider,
        metadata={
            "windows_shell_priority": ["git-bash", "powershell", "cmd"],
            "bash_env": {"PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple"},
        },
    ),
)
```

On Windows, commands get `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` unless the
parent environment or `bash_env` sets them. They also start without a console
window, so a GUI host does not flash a terminal.

The `bash` tool description tells the model which shell and command prefix it
will get. The hint, like the run's `PromptBundle`, stays fixed for the whole
run, so every cycle sends the same request prefix and the provider's prompt
cache keeps hitting. Anthropic requests can place cache breakpoints at prompt
sections; other providers get the flattened prompt.

## Memory compaction

When the history grows too large for the model, `MemoryManager` compacts it.
It first archives old, large tool results (microcompaction). If that is not
enough, it summarizes the older part of the history and keeps the last 10 raw
messages (`memory_keep_recent_messages`). A failed step leaves the history
unchanged.

- The compaction threshold is the smaller of `memory_compact_threshold`
  (default `250000` tokens) and the model's prompt capacity: its context window
  minus the output reserve minus a `13000`-token buffer. A memory warning
  appears at `memory_threshold_percentage` (default `90`) percent of it. Both
  settings are fields of `AgentTask` and `InteractiveAgentDefinition`; Runner
  runs use the defaults.
  See [Capacity metadata](docs/model-settings.md#capacity-metadata).
- Tune microcompaction with `RunConfig(microcompaction_policy=MicrocompactionPolicy(...))`.
  By default it starts at `0.75` of the threshold, aims for `0.60`, keeps the
  last 3 cycles, and only touches results over 500 characters.
- An archived result is first written in full to `.vv-agent/artifacts/` in the
  workspace; the model then sees a `<Tool Result Compact>` marker with the
  artifact path, a `read_file` hint and a short excerpt. Archiving is off when
  the model cannot call `read_file`, because it could not read the result back.
  A tool declared with `result_retention=ToolResultRetention.PRESERVE` is never
  archived. The artifact's byte size and SHA-256 stay in the host-visible
  `ToolArtifactRef`.
- The summary uses `memory_summary_model` (and optional
  `memory_summary_backend`) from metadata, else the run's own model. It keeps
  artifact and file paths with read hints and never reads those files itself.
- Size comes from the previous cycle's reported prompt tokens plus the new tool
  messages. Without usage data, vv-agent counts tokens with `vv-llm`, then falls
  back to a local estimate that handles CJK text.
- `memory_compact_started` and `memory_compact_completed` events report each
  attempt. The completed event names the strongest step taken: `none`, `micro`,
  `structural`, `summary` or `emergency`.

Session Memory carries notes from one run of a session to the next. It is off
by default; turn it on with `RunConfig(session_memory_enabled=True)`. Notes live
in `workspace/.memory/session/<session-or-task-scope>/session_memory.json`,
scoped by `metadata.session_id` or else the task ID. A run reads them once, at
compile time, into its first system message; notes extracted during the run
show up from the next run on, and a resumed turn reuses its frozen copy.
Extraction uses the summary model unless `session_memory_extraction_model` says
otherwise. Full compaction keeps the stored notes. Sub-tasks keep Session Memory
off by default so parent and child never share a file.

These metadata keys, set in `Agent.metadata` or `RunConfig.metadata`, tune the
details: `model_context_window`, `reserved_output_tokens`,
`autocompact_buffer_tokens`, `memory_keep_recent_messages`,
`memory_summary_model`, `memory_summary_backend`, `include_memory_warning`,
`session_memory_min_tokens`, `session_memory_max_tokens`,
`session_memory_min_text_messages`, `session_memory_storage_dir`,
`session_memory_extraction_model`, `session_memory_extraction_backend`,
`tool_result_excerpt_head`, `tool_result_excerpt_tail` and
`summary_event_limit`.

For the full algorithm, see
[History-preserving compaction](docs/architecture.md#history-preserving-compaction)
and [Compaction through the log](docs/session-kernel.md#compaction-through-the-log).

## How it fits together

```text
Runner / InteractiveAgentClient / CLI / AppServer
  -> SessionDriver -> SessionStore (SQLite or PostgreSQL)
  -> session kernel (vv_agent.session.kernel.drive)
  -> logged model and tool operations, child sessions
  -> RunResult / RunEvent / tracing / App Server items
```

Every entry point goes through the same session kernel. A session store
transaction owns the session's log, its inbox (queued inputs such as replies and
steering messages), its leases (a worker's claim to drive it), child admission,
and the receipts of finished operations. Plain runs use in-memory SQLite.
Durable SQLite and PostgreSQL stores are opt-in through the session API.

vv-agent owns the portable runtime: prompt assembly, model calls, tool planning
and execution, compaction, typed events, cancellation, approvals and replayable
history. Your product owns its UI, users and workspaces, storage, browser or IM
integration, and product tools. Connect the two through providers instead of
patching vv-agent:

| Extension point | What it does |
| --- | --- |
| `AppServerHost` | Builds `Agent` and `RunConfig` for the App Server. |
| `ApprovalProvider` | Decides tool approvals from product rules or UI. |
| `ContextProvider` | Adds product prompt fragments before each run is compiled. |
| `vv_agent.memory.MemoryProvider` | Connects product memory search and save, and compaction events. |
| `vv_agent.tools.ToolExecutor` | Runs product tools with their own schema, approval, timeout and errors. |
| `SessionRunEventStore` | Replays typed `RunEvent` history, including parent and child runs. |

Public names are exported from `vv_agent`. Lower-level runtime types such as
`AgentTask`, `Message` and `ToolCall` live there too, but most code never needs
them. See [Architecture](docs/architecture.md) for the module map and
invariants.

## Examples

`examples/` holds SDK cookbook scripts and a few lower-level integration
examples. See [`examples/README.md`](examples/README.md) for the list.

```bash
uv run python examples/01_quick_start.py
uv run python examples/24_workspace_backends.py
```

## Development

```bash
uv sync --dev
uv run pytest                              # unit tests, no network
uv run ruff check .                        # lint
uv run ty check                            # type check

VV_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live   # calls a real LLM
```

Live tests read these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VV_AGENT_LOCAL_SETTINGS` | `local_settings.py` | Settings file path |
| `VV_AGENT_LIVE_BACKEND` | `moonshot` | LLM backend |
| `VV_AGENT_LIVE_MODEL` | `kimi-k3` | Model name |

See [Development](docs/development.md) for test selection and release steps,
and [the docs index](docs/index.md) for everything else.
