# Architecture

`vv-agent` is a Python agent runtime extracted from VectorVein's production
runtime. It is organized around a cycle loop: prepare context, call an LLM,
dispatch tool calls, update memory/state, and repeat until the agent explicitly
finishes or asks the user for input.

## Top-Level Flow

```text
Public SDK
  -> Agent / RunConfig / ModelSettings
  -> Runner
  -> RunHandle for live runs
  -> RuntimeTask
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> RunResult / RunEvent / RunEventStore replay

Interactive session SDK
  -> InteractiveAgentDefinition / AgentSessionOptions
  -> InteractiveAgentClient
  -> AgentSession
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> AgentSessionRun / AgentSessionState

CLI / legacy runtime API
  -> config.load_llm_settings_from_file
  -> config.resolve_model_endpoint
  -> llm.VVLlmClient
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> AgentResult
```

Task completion is tool-driven. The model must call `task_finish` or `ask_user`
to end the run or wait for user input; the runtime does not infer completion
from the assistant's last message.

## Runtime Boundary

`vv-agent` is the framework boundary. It owns the portable agent contract:

- `Agent`, `Runner`, `RunConfig`, `RunHandle`, `RunResult`, and typed
  `RunEvent` objects.
- Prompt assembly, model calls, tool planning, tool dispatch, approval
  interruption, cancellation, memory compaction, and runtime hooks.
- Replayable app history through `RunEventStore`; `JsonlRunEventStore` is the
  built-in file-backed implementation.
- Tool execution through `vv_agent.tools.ToolExecutor` and
  `vv_agent.tools.ToolOrchestrator`, with `FunctionTool` and `@function_tool`
  as the normal public path.

Host products own product concerns outside the framework: product UI, account
and profile resolution, workspace selection, product persistence, browser or IM
integration, and product-specific tools. They should connect those concerns by
implementing providers instead of patching runtime internals:

- `ApprovalProvider` for UI prompts, policy checks, and allow/deny decisions.
- `ContextProvider` for product prompt fragments such as profile, workspace,
  policy, and feature context.
- `vv_agent.memory.MemoryProvider` for product memory search/save and
  compaction lifecycle integration.
- `vv_agent.tools.ToolExecutor` or `FunctionTool` collections for product
  tools.
- `RunEventStore` for app history and parent/child run graph replay.

Raw runtime logs remain available for compatibility, but typed `RunEvent` is
the primary state contract for host UIs.

## Module Map

| Path | Responsibility |
| --- | --- |
| `src/vv_agent/config.py` | Settings-file loading, provider/backend lookup, endpoint resolution, and `vv-llm` settings construction. |
| `src/vv_agent/cli.py` | Command-line argument parsing and one-shot runtime execution. |
| `src/vv_agent/agent.py` | Public `Agent` definition and agent-as-tool helpers. |
| `src/vv_agent/runner.py` | Public synchronous run and stream entry points. |
| `src/vv_agent/run_handle.py` | Live `Runner.start()` handle for event streaming, cancellation, approvals, and final result retrieval. |
| `src/vv_agent/run_config.py` | Per-run configuration, model provider binding, tool policy, workspace, session, and tracing options. |
| `src/vv_agent/model_settings.py` | Model call parameters and override merging. |
| `src/vv_agent/events.py` | Typed run events and dict conversion for UI consumers. |
| `src/vv_agent/event_store.py` | Run event persistence and replay protocol plus JSONL implementation. |
| `src/vv_agent/approval.py` | Approval provider protocol, request/decision objects, and in-process approval broker. |
| `src/vv_agent/context_providers.py` | Context provider protocol and deterministic prompt-fragment assembly. |
| `src/vv_agent/guardrails.py` | Public guardrail result contract and decorators. |
| `src/vv_agent/interactive.py` | Public stateful session/client API for desktop runtimes, interruptions, follow-ups, cancellation, and shared tool state. |
| `src/vv_agent/result.py` | Public `RunResult` wrapper around runtime results. |
| `src/vv_agent/sessions/` | Public `Session` protocol plus memory, SQLite, and Redis implementations. |
| `src/vv_agent/tracing.py` | Public trace spans and processor protocol. |
| `src/vv_agent/runtime/compiler.py` | Compile layer: `Agent + input + RunConfig -> RuntimeTask`. Import this submodule directly to avoid runtime package initialization cycles. |
| `src/vv_agent/types.py` | Runtime protocol types: tasks, messages, tool calls, results, statuses, and token usage. |
| `src/vv_agent/llm/` | LLM protocol adapters, scripted test clients, prompt cache behavior, and `vv-llm` client bridge. |
| `src/vv_agent/runtime/` | Core loop, cycle execution, hooks, cancellation, backends, checkpoint stores, and sub-task coordination. |
| `src/vv_agent/tools/` | Tool registry, OpenAI-compatible schemas, dispatcher, and built-in handlers. |
| `src/vv_agent/memory/` | Token counting, compaction, micro-compaction, session memory, and post-compaction file restoration. |
| `src/vv_agent/prompt/` | System prompt construction and prompt-cache section tracking. |
| `src/vv_agent/sdk/` | Internal migration namespace for session/sub-task compatibility. New user code should import from `vv_agent` top level. |
| `src/vv_agent/workspace/` | Local, memory, and S3-compatible workspace storage backends. |
| `src/vv_agent/skills/` | Skill metadata parsing, validation, normalization, and prompt rendering. |

## Execution Backends

- `InlineBackend`: default synchronous cycle execution.
- `ThreadBackend`: non-blocking submission with futures.
- `CeleryBackend`: distributed cycle execution. Distributed mode requires a
  `RuntimeRecipe` and a shared `StateStore`; otherwise it falls back inline.

Checkpoint stores live under `runtime/stores/` and support SQLite and Redis.
Backends must preserve the same `AgentResult` and checkpoint payload shape as
inline execution.

## Tool Boundaries

Tool definitions and behavior are intentionally split:

- `tools/base.py`: schema and execution result types.
- `tools/function.py`: public `FunctionTool` and `function_tool` decorator,
  including signature/dataclass/TypedDict/Pydantic schema inference.
- `tools/outputs.py`: structured public tool output variants.
- `tools/registry.py`: registration and lookup.
- `tools/dispatcher.py`: argument normalization and handler dispatch.
- `tools/handlers/`: concrete built-in behavior.
- `constants/tool_names.py` and `constants/workspace.py`: stable tool names and
  schemas used by prompts/tests.

Do not bury model-visible behavior in ad hoc handler strings without tests. Tool
schema wording is part of the agent contract.

`Agent.as_tool()` compiles a child agent into a callable tool. The child result
is returned as the tool output and the parent agent keeps control. `handoff()`
compiles to a transfer tool whose result uses a finish directive; the target
agent output becomes the run output and a typed `HandoffEvent` is emitted.

`ToolPolicy` is enforced when tool schemas are planned: `allowed_tools` filters
the final schema list to an allow-list, and `disallowed_tools` removes matching
tools from defaults and custom function tools.

`FunctionTool.needs_approval` and `ToolPolicy.approval="always"` interrupt tool
execution with a wait-user directive before user code runs. The runtime log is
converted to `ToolApprovalRequestedEvent`. `ToolPolicy.approval="never"` skips
that approval gate for trusted runs.

## Guardrails And Tracing

Input guardrails run inside `Runner` before model/provider resolution. A block
or approval requirement returns a failed run result without calling the model.
Rewrite results replace the user input before the runtime task is compiled.

Output guardrails run after the runtime returns. Rewrite results replace
`RunResult.final_output`; block and approval results convert the public run
status to failed and expose the guardrail message.

Trace processors are read from `RunConfig.tracing["processors"]`. `Runner`
starts a `run` span for each invocation and starts/ends `tool` spans from typed
tool events emitted by the runtime.

## Workspace Boundary

File tools must go through `WorkspaceBackend`. Local filesystem access,
in-memory storage, and S3-compatible storage should keep the same behavior for
read/write/list/grep semantics wherever practical. Path traversal protections
belong at the workspace boundary and are covered by `tests/test_workspace_backends.py`
and `tests/test_tools.py`.

## Invariants

- Model resolution is exact: requested model keys are not aliased to independent
  provider models.
- Runtime terminal states are explicit tool outcomes, not prose heuristics.
- Public SDK code should enter through `Agent`, `Runner`, `RunConfig`,
  `ModelSettings`, tools, sessions, typed `RunEvent` objects, or
  `InteractiveAgentClient` for stateful host-controlled runtimes.
- Long outputs should keep structured data in metadata and model-facing text in
  content.
- Cancellation, streaming, hooks, memory compaction, and execution backends must
  compose without changing public result shapes.
- New public behavior needs tests in the closest `tests/test_*.py` module.
