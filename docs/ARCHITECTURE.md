# v-agent Architecture

## Design Inputs

- Source domain logic: `backend/vector_vein_main/ai_agents/tasks/`
  - prompt composition
  - tool schema constants
  - tool-call status protocol and runtime transitions
- Reference framework: `ref_repos/agent_framework/`
  - modular runtime topology
  - explicit lifecycle states and observability

## Runtime Topology

- `AgentRuntime`: orchestrates full task lifecycle
- `CycleRunner`: executes one LLM cycle
- `ToolCallRunner`: executes and collects tool calls for a cycle
- `ToolPlanner`: decides which tools are visible for current capability set
- `ToolDispatcher`: normalizes tool-call arguments, handles errors, and standardizes result payloads

## State Model

Task-level state:

`pending -> running -> (completed | wait_user | failed | max_cycles)`

Tool-level status code (`ToolResultStatus`):

- `SUCCESS`
- `ERROR`
- `WAIT_RESPONSE`
- `RUNNING`
- `BATCH_RUNNING`
- `PENDING_COMPRESS`

## Tool System

- Tool names and schemas are centralized in `src/v_agent/constants/`.
- Built-in handlers are split by responsibility in `src/v_agent/tools/handlers/`.
- Workflow 语义不做框架内置特殊处理；如需 workflow，按自定义工具注册到 `ToolRegistry`。
- Default registry preloads:
  - control: `_task_finish`, `_ask_user`, `_todo_write`, `_todo_read`
  - workspace: `_read_file`, `_write_file`, `_list_files`, `_workspace_grep`
  - computer: `_bash`, `_check_background_command`, `_read_image`
  - sub-agent delegation: `_create_sub_task`, `_batch_sub_tasks`（由 runtime 内建子任务执行链路驱动）
  - extension stubs: document/skill tools (return standard not-enabled errors by default)

## Prompt Layer

`build_system_prompt(...)` composes:

- `<Agent Definition>`
- `<Environment>` (optional, agent_type=computer)
- `<Tools>`
- `<Current Time>`

Prompt templates include tool-priority governance (prefer specialized tools over shell).

## Memory and LLM

- `MemoryManager` performs context compaction by threshold.
- `OpenAICompatibleLLM` implements:
  - endpoint failover and retries
  - stream/non-stream routing
  - tool-call normalization
  - backend-like settings parsing from `local_settings.py`
