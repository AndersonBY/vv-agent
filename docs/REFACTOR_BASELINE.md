# v-agent 重构基线记录

## 基线时间

- UTC: 2026-02-17T14:52:49Z
- 分支: `refactor/tool-protocol-runtime`

## 基线校验结果

执行命令：

```bash
uv run ruff check .
uv run ty check
uv run pytest -q
```

结果：

- `ruff`: All checks passed
- `ty`: All checks passed
- `pytest`: 23 passed, 1 skipped

## 当前核心行为（重构前）

- `task_finish`
  - 默认校验 `todo_list` 是否全部完成；若有未完成项，返回错误 `todo_incomplete`。
  - 校验通过后返回 FINISH directive，并由 runtime 结束任务。
- `ask_user`
  - 返回 WAIT_USER directive，由 runtime 进入 `WAIT_USER` 状态。
  - 支持 `question` 和 `options`，但缺少 `selection_type` / `allow_custom_options` 结构化语义。
- `todo_write` / `todo_read`
  - 当前是 `action` 风格（replace/append/set_done），不是 backend 的完整列表覆盖写入协议。
  - todo 条目字段仅 `title/done`，不含 `id/status/priority`。
- workspace 工具
  - 已具备 `read_file` / `write_file` / `list_files` / `workspace_grep`。
  - 缺失 backend 中的 replace、line_replace、bash、background、image、workflow、document 等能力。

## 与 backend 设计差距（已确认）

- 工具名与 schema 不统一：缺少 constants 单一真源，工具描述文案也未对齐 backend 规范。
- 运行协议不完整：`ToolExecutionResult` 仅有 success/error + directive，缺少 `RUNNING/BATCH_RUNNING/PENDING_COMPRESS/WAIT_RESPONSE` 等状态编码。
- runtime 执行模型偏简化：尚未拆分 cycle runner 与 tool-call runner，不支持运行中任务轮询协议。
- prompt 治理不足：当前系统提示词在 CLI 内硬编码，缺少分层 builder 与 capability 注入。
- todo 语义弱：未实现“最多一个 in_progress”约束与完整列表更新模式。

## 回归参考

后续每阶段完成后，至少执行以下命令并记录结果：

```bash
uv run ruff check .
uv run ty check
uv run pytest -q
```
