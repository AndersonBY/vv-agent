# v-agent Architecture

## Design Inputs

- Source domain logic: `backend/vector_vein_main/ai_agents/`
  - cycle task orchestration (`agent_cycle.py`, `task_lifecycle.py`)
  - tool dispatch and status transitions (`tool_calls.py`, `tasks/tools/*`)
  - todo-driven finish constraints
  - memory compression strategy
- Reference agent framework: `ref_repos/agent_framework/hive/`
  - explicit runtime topology
  - observable execution stream and stateful lifecycle
  - composable runtime abstractions (runtime / tools / storage)

## Core Runtime State Machine

`pending -> running -> (completed | wait_user | failed | max_cycles)`

Each cycle executes:

1. compact memory (if over threshold)
2. call LLM with current messages + tool schemas
3. append assistant response
4. execute tool calls in order
5. transition by tool directive:
   - `finish` -> `completed`
   - `wait_user` -> `wait_user`
   - otherwise continue

## Tool Philosophy

- `task_finish` and `ask_user` are explicit lifecycle controls.
- `todo_write` / `todo_read` make plan tracking machine-readable.
- `task_finish` enforces todo completeness by default.
- workspace tools are isolated inside a root path to avoid path escape.

## Memory Strategy

- If message chars exceed threshold:
  - keep first system message
  - summarize middle section into `memory_summary`
  - keep recent N messages untouched

This keeps context bounded while preserving short-term execution fidelity.

## LLM Adapter Strategy

- OpenAI-compatible transport to support multiple providers.
- Model/endpoint resolution parsed from standalone project `local_settings.py` (template: `local_settings.example.py`).
- key normalization supports encoded formats used in existing settings.
- Endpoint failover + per-endpoint retry with backoff.
- Stream/non-stream mode auto routing for reasoning-focused model families (inspired by `utilities/llm.py`).
- Tool call ID/名称归一化，避免不同模型的增量差异影响运行时。
