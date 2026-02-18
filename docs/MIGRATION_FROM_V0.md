# Migration Guide: v0 -> current

## 1) Tool name changes

Old names (no prefix) were replaced by backend-style names:

- `task_finish` -> `_task_finish`
- `ask_user` -> `_ask_user`
- `todo_write` -> `_todo_write`
- `todo_read` -> `_todo_read`
- `read_file` -> `_read_file`
- `write_file` -> `_write_file`
- `list_files` -> `_list_files`
- `workspace_grep` -> `_workspace_grep`

## 2) TODO protocol changes

`_todo_write` now expects full-list payload:

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

## 3) Runtime structure changes

Old single-file engine logic is now split:

- `CycleRunner` (LLM cycle)
- `ToolCallRunner` (tool execution)
- `ToolPlanner` (dynamic tool visibility)
- `ToolDispatcher` (argument parsing + protocol mapping)

## 4) Schema source changes

Tool schemas are no longer hardcoded in handlers.
Use constants modules as source of truth:

- `src/v_agent/constants/workspace.py`
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

- `_create_sub_task` / `_batch_sub_tasks` are now built-in and execute configured `AgentTask.sub_agents`.
- `AgentSDKClient` now supports one-shot text API:

```python
text = client.query(agent_name="planner", prompt="一句话总结进展")
```
