# Architecture

`vv-agent` is a Python agent runtime extracted from VectorVein's production
runtime. It is organized around a cycle loop: prepare context, call an LLM,
dispatch tool calls, update memory/state, and repeat until the agent explicitly
finishes or asks the user for input.

## Top-Level Flow

```text
CLI / SDK
  -> config.load_llm_settings_from_file
  -> config.resolve_model_endpoint
  -> llm.VVLlmClient
  -> runtime.AgentRuntime
      -> CycleRunner
      -> MemoryManager
      -> ToolPlanner
      -> ToolCallRunner
      -> ExecutionBackend
  -> AgentResult / AgentRun
```

Task completion is tool-driven. The model must call `task_finish` or `ask_user`
to end the run or wait for user input; the runtime does not infer completion
from the assistant's last message.

## Module Map

| Path | Responsibility |
| --- | --- |
| `src/vv_agent/config.py` | Settings-file loading, provider/backend lookup, endpoint resolution, and `vv-llm` settings construction. |
| `src/vv_agent/cli.py` | Command-line argument parsing and one-shot runtime execution. |
| `src/vv_agent/types.py` | Public protocol types: tasks, messages, tool calls, results, statuses, and token usage. |
| `src/vv_agent/llm/` | LLM protocol adapters, scripted test clients, prompt cache behavior, and `vv-llm` client bridge. |
| `src/vv_agent/runtime/` | Core loop, cycle execution, hooks, cancellation, backends, checkpoint stores, and sub-task coordination. |
| `src/vv_agent/tools/` | Tool registry, OpenAI-compatible schemas, dispatcher, and built-in handlers. |
| `src/vv_agent/memory/` | Token counting, compaction, micro-compaction, session memory, and post-compaction file restoration. |
| `src/vv_agent/prompt/` | System prompt construction and prompt-cache section tracking. |
| `src/vv_agent/sdk/` | High-level client, sessions, agent definitions, and resource loading. |
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
- `tools/registry.py`: registration and lookup.
- `tools/dispatcher.py`: argument normalization and handler dispatch.
- `tools/handlers/`: concrete built-in behavior.
- `constants/tool_names.py` and `constants/workspace.py`: stable tool names and
  schemas used by prompts/tests.

Do not bury model-visible behavior in ad hoc handler strings without tests. Tool
schema wording is part of the agent contract.

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
- Long outputs should keep structured data in metadata and model-facing text in
  content.
- Cancellation, streaming, hooks, memory compaction, and execution backends must
  compose without changing public result shapes.
- New public behavior needs tests in the closest `tests/test_*.py` module.
