# Migration Guide: v0 -> current

## 1) Tool name changes

Old underscore-prefixed names were replaced by plain names:

- `_task_finish` -> `task_finish`
- `_ask_user` -> `ask_user`
- `_todo_write` -> `todo_write`
- `_todo_read` -> `todo_read`
- `_read_file` -> `read_file`
- `_write_file` -> `write_file`
- `_list_files` -> `list_files`
- `_workspace_grep` -> `workspace_grep`

## 2) TODO protocol changes

`todo_write` now expects full-list payload:

```json
{
  "todos": [
    {"id": "optional", "title": "...", "status": "pending|in_progress|completed", "priority": "low|medium|high"}
  ]
}
```

Rules:

- Omitted previous items are removed.
- At most one `in_progress` item is allowed.
- `todo_read` 不再作为默认内建工具暴露（如需可作为自定义工具注册）。

## 3) Runtime structure changes

Old single-file engine logic is now split:

- `CycleRunner` (LLM cycle)
- `ToolCallRunner` (tool execution)
- `ToolPlanner` (dynamic tool visibility)
- `ToolDispatcher` (argument parsing + protocol mapping)

## 4) Schema source changes

Tool schemas are no longer hardcoded in handlers.
Use constants modules as source of truth:

- `src/vv_agent/constants/workspace.py`
- workflow/document 建议作为自定义工具注册到 `ToolRegistry`（不再内建专项模块）

## 5) Prompt system changes

CLI now uses `build_system_prompt(...)` to generate layered prompt sections:

- `<Agent Definition>`
- `<Environment>`
- `<Tools>`
- `<Current Time>`

## 6) New extension stubs

Skill activation tool interface is present in registry.
By default, handlers return structured `not_enabled` errors until adapters are wired.

## 7) Sub-agent delegation and SDK query

- `create_sub_task` / `batch_sub_tasks` are now built-in and execute configured `AgentTask.sub_agents`.
- `AgentSDKClient` now supports both default single-agent and named-agent one-shot APIs:

```python
run = client.run(prompt="执行任务")
text = client.query(prompt="一句话总结进展")
text = client.query(agent="planner", prompt="一句话总结进展")
```

## 8) Session API, runtime hooks, and resource loader

- 新增 `client.create_session()`：支持状态化多轮执行、`steer/follow_up` 队列、事件订阅。
- 新增 runtime hooks：可在 `before_memory_compact/before_llm/after_llm/before_tool_call/after_tool_call` 拦截或改写流程。
- 新增 `AgentResourceLoader`：自动发现 `workspace/.vv-agent/` 与 `~/.vv-agent/` 的
  `agents.json`、`prompts/*.md`、`skills/`、`hooks/` 资源并注入 SDK。
