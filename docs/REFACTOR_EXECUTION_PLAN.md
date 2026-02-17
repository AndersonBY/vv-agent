# v-agent 重构执行计划（可直接按步骤实施）

> 目标：把 `v-agent` 从“简化工具调用器”升级为“工具协议驱动的 Agent 运行时”，对齐 `backend/vector_vein_main/ai_agents/tasks/` 的工具定义、提示词治理、状态机执行逻辑。

---

## 执行进度（持续更新）

| 阶段 | 状态 | 完成时间（UTC） | 备注 |
|---|---|---|---|
| P0 基线冻结 | ✅ 已完成 | 2026-02-17T14:52:49Z | 新建 `docs/REFACTOR_BASELINE.md`，并完成 ruff/ty/pytest 基线验证 |
| P1 协议类型重构 | ✅ 已完成 | 2026-02-17T14:55:07Z | 新增 `ToolResultStatus`/`CycleStatus` 协议枚举并兼容旧 `status` 字段 |
| P2 常量与 schema 中心化 | ✅ 已完成 | 2026-02-17T15:04:41Z | 新增 `constants/` 体系，registry 改为从 constants 读取 backend 风格 schema |
| P3 Prompt Builder | ✅ 已完成 | 2026-02-17T15:09:08Z | 新增分层 `prompt.builder`，生成 `<Agent Definition>/<Environment>/<Tools>/<Current Time>` |
| P4 动态工具规划器 | ✅ 已完成 | 2026-02-17T15:17:28Z | 新增 `runtime/tool_planner.py`，按 capability + memory 阈值动态规划工具集 |
| P5 Dispatcher | ✅ 已完成 | 2026-02-17T15:17:28Z | 新增 `tools/dispatcher.py`，标准化参数解析/错误码/WAIT_RESPONSE 状态映射 |
| P6 工具实现拆分 | ✅ 已完成 | 2026-02-17T15:26:23Z | `builtins.py` 已精简为注册层，workspace/control/todo/search 拆分到 `tools/handlers/` |
| P7 控制工具语义对齐 | ✅ 已完成 | 2026-02-17T15:26:23Z | `todo_write` 升级为完整列表写入语义，加入单 `in_progress` 约束与结构化错误码 |
| P8 runtime 状态机重构 | ✅ 已完成 | 2026-02-17T15:26:23Z | runtime 拆为 `cycle_runner` + `tool_call_runner`，状态迁移路径清晰化 |
| P9 高级工具接入 | ✅ 已完成 | 2026-02-17T15:41:52Z | 接入 `_bash`/`_check_background_command`/`_read_image`，支持后台会话生命周期与图像通知 |
| P10 文档/工作流/skills 扩展 | ✅ 已完成 | 2026-02-17T15:41:52Z | 增加文档/工作流/技能扩展工具骨架，默认返回标准化未启用错误 |
| P11 全量验收与收口 | ✅ 已完成 | 2026-02-17T15:41:52Z | 通过 ruff/ty/pytest/live；补齐 `TOOL_PROTOCOL` 与 `MIGRATION_FROM_V0` 文档 |
| P12 LLM 统一接口对齐 | ✅ 已完成 | 2026-02-17T16:47:41Z | 按 `backend/.../utilities/llm.py` 对齐请求选项和流式 tool call 聚合语义，并通过真实命令验证 |

### 执行日志

- 2026-02-17T14:52:49Z：启动 `refactor/tool-protocol-runtime` 分支并完成基线检查（ruff/ty/pytest 全绿，23 passed, 1 skipped）。
- 2026-02-17T14:55:07Z：完成 P1 协议类型重构，新增 `tests/test_protocol_types.py`，回归结果 `26 passed, 1 skipped`。
- 2026-02-17T15:04:41Z：完成 P2 常量与 schema 中心化，工具名切换到 backend 风格 `_tool_name`，回归结果 `28 passed, 1 skipped`。
- 2026-02-17T15:09:08Z：完成 P3 Prompt Builder，CLI 改为通过 builder 构建系统提示词，回归结果 `31 passed, 1 skipped`。
- 2026-02-17T15:17:28Z：完成 P4 动态工具规划器与 P5 Dispatcher，runtime 已通过 planner + dispatcher 执行工具，回归结果 `39 passed, 1 skipped`。
- 2026-02-17T15:26:23Z：完成 P6/P7/P8，工具 handler 已模块化且控制语义对齐 backend 风格，runtime 主循环拆分，回归结果 `42 passed, 1 skipped`。
- 2026-02-17T15:41:52Z：完成 P9/P10/P11，新增 bash/background/image 与 extension stubs，文档收口；回归 `51 passed, 1 skipped`，live `1 passed`。
- 2026-02-17T16:45:22Z：启动 P12，开始将 `src/v_agent/llm/openai_compatible.py` 请求参数映射与流式工具聚合逻辑对齐 `backend/vector_vein_main/utilities/llm.py`。
- 2026-02-17T16:47:41Z：完成 P12，请求参数与流式聚合逻辑改造落地；回归 `57 passed, 1 skipped`，live `1 passed`，CLI 真实命令 `uv run v-agent --prompt \"请概述一下这个框架的特点\" --backend moonshot --model kimi-k2.5` 返回 `status=completed`。

---

## 0. 文档使用方式

- 这是**执行型清单**：按阶段顺序做，不要并行跨阶段改动。
- 每阶段必须满足本阶段 `DoD`（Definition of Done）才能进入下一阶段。
- 每阶段完成后都执行统一验证命令：

```bash
uv run ruff check .
uv run ty check
uv run pytest -q
```

---

## 1. 参考基线（必须先读）

### 1.1 backend 参考代码（设计来源）

- 工具名常量：`backend/vector_vein_main/ai_agents/tasks/constants/tool_names.py`
- 工作区工具 schema：`backend/vector_vein_main/ai_agents/tasks/constants/workspace.py`
- 文档工具 schema：`backend/vector_vein_main/ai_agents/tasks/constants/document.py`
- 工作流工具 schema：`backend/vector_vein_main/ai_agents/tasks/constants/workflow.py`
- 系统提示词拼装：`backend/vector_vein_main/ai_agents/tasks/prompt.py`
- 动态工具注入：`backend/vector_vein_main/ai_agents/tasks/tools/__init__.py` 中 `_get_available_tools`
- 工具执行与状态处理：`backend/vector_vein_main/ai_agents/tasks/tools/__init__.py` 中 `_process_tool_calls_logic`
- 周期主循环：`backend/vector_vein_main/ai_agents/tasks/agent_cycle.py`
- 工具后处理/轮询：`backend/vector_vein_main/ai_agents/tasks/tool_calls.py`
- TODO 提醒与辅助逻辑：`backend/vector_vein_main/ai_agents/tasks/task_helpers.py`
- LLM 统一接口关键行为：`backend/vector_vein_main/utilities/llm.py`

### 1.2 当前 v-agent 代码（重构目标）

- 类型协议：`v-agent/src/v_agent/types.py`
- runtime：`v-agent/src/v_agent/runtime/engine.py`
- 工具定义（当前集中版）：`v-agent/src/v_agent/tools/builtins.py`
- 工具注册：`v-agent/src/v_agent/tools/registry.py`
- LLM 接口：`v-agent/src/v_agent/llm/openai_compatible.py`
- 配置解析：`v-agent/src/v_agent/config.py`

---

## 2. 阶段总览（强顺序）

| 阶段 | 名称 | 目标 |
|---|---|---|
| P0 | 基线冻结 | 固定当前行为，防止回归 |
| P1 | 协议类型重构 | 建立 backend 风格的状态/结果协议 |
| P2 | 常量与 schema 中心化 | 把工具名和 schema 抽成单一真源 |
| P3 | Prompt Builder | 建立分层系统提示词编排 |
| P4 | 动态工具规划器 | 按 capability 动态注入工具 |
| P5 | Dispatcher | 建立统一工具调用执行器 |
| P6 | 工具实现拆分 | 从 `builtins.py` 拆为 handlers |
| P7 | 控制工具对齐 | 对齐 `task_finish / ask_user / todo_*` 语义 |
| P8 | runtime 状态机重构 | 形成 cycle + tool-call 两阶段执行 |
| P9 | 高级工具接入 | bash/background/read_image 协议化接入 |
| P10 | 文档与扩展接口 | 文档工具/工作流/skills 扩展点 |
| P11 | 全量测试与验收 | 回归+真实联调+文档收口 |
| P12 | LLM 统一接口对齐 | 对齐 backend LLM 参数策略与流式 tool call 聚合 |

---

## 3. 详细执行步骤

## P0 基线冻结

### 目标
- 固定当前可运行状态，作为后续回归对照。

### 步骤
1. 新建分支：
   ```bash
   git checkout -b refactor/tool-protocol-runtime
   ```
2. 记录当前基线命令输出：
   ```bash
   uv run ruff check .
   uv run ty check
   uv run pytest -q
   ```
3. 新建文档：`v-agent/docs/REFACTOR_BASELINE.md`，记录：
   - 当前测试数量
   - 当前核心行为（finish、ask_user、todo、read/write/grep）
   - 当前已知差距（与 backend 的差异）

### DoD
- 基线文档已存在。
- 当前分支测试全绿。

---

## P1 协议类型重构（先做，不碰具体工具逻辑）

### 目标
- 让 runtime/dispatcher 能表达 backend 风格的多状态协议。

### 参考
- `backend/.../tools/__init__.py` 的状态设计（`SUCCESS/ERROR/WAIT_RESPONSE/RUNNING/BATCH_RUNNING/PENDING_COMPRESS`）
- `backend/.../tool_calls.py` 的运行中轮询处理。

### 改动文件
- 修改：`v-agent/src/v_agent/types.py`
- 新增：`v-agent/tests/test_protocol_types.py`

### 具体任务
1. 在 `types.py` 增加：
   - `ToolResultStatus`（枚举）
   - `CycleStatus`（枚举）
   - `ToolExecutionResult` 扩展字段：
     - `status_code: ToolResultStatus`
     - `error_code: str | None`
     - `metadata: dict[str, Any]`
     - `image_url/image_path`（可选）
2. 保持向后兼容：
   - 原 `status: "success"|"error"` 仍可保留一轮（过渡期）。
3. 新增单测：
   - 验证 `to_tool_message()` 行为不变。
   - 验证新枚举值可序列化。

### DoD
- `types.py` 能表达所有目标状态。
- 现有测试不破坏。

---

## P2 工具常量 + schema 中心化

### 目标
- 把工具名和 schema 从 `builtins.py` 中抽离，形成单一真源（backend 同款结构）。

### 参考
- `backend/.../constants/tool_names.py`
- `backend/.../constants/workspace.py`
- `backend/.../constants/document.py`
- `backend/.../constants/workflow.py`

### 改动文件
- 新增：
  - `v-agent/src/v_agent/constants/tool_names.py`
  - `v-agent/src/v_agent/constants/workspace.py`
  - `v-agent/src/v_agent/constants/document.py`
  - `v-agent/src/v_agent/constants/workflow.py`
  - `v-agent/src/v_agent/constants/__init__.py`
- 修改：
  - `v-agent/src/v_agent/tools/registry.py`
  - `v-agent/src/v_agent/tools/builtins.py`（仅保留 handler，不再硬编码 schema）

### 具体任务
1. 复制 backend 的命名策略（保留前缀风格，如 `_task_finish`），统一集中。
2. 将 schema 文案改为“操作手册式描述”，不能简化成一句话。
3. `ToolRegistry.list_openai_schemas()` 改为读取 constants 提供的 schema。

### DoD
- `builtins.py` 不再定义完整 schema 文本。
- schema 能单独导出并单测验证。

---

## P3 Prompt Builder（分层提示词）

### 目标
- 从“固定 system prompt”改为 backend 风格的分层拼装。

### 参考
- `backend/.../prompt.py`：
  - `task_finish_prompt`
  - `ask_user_prompt`
  - `computer_agent_prompt`
  - `get_static_system_prompt`

### 改动文件
- 新增：
  - `v-agent/src/v_agent/prompt/templates.py`
  - `v-agent/src/v_agent/prompt/builder.py`
  - `v-agent/src/v_agent/prompt/__init__.py`
- 修改：
  - `v-agent/src/v_agent/runtime/engine.py`
  - `v-agent/src/v_agent/cli.py`

### 具体任务
1. 实现 `build_system_prompt(...)`：
   - 输出 `<Agent Definition>`, `<Environment>`, `<Tools>`, `<Current Time>`.
2. 注入工具优先级规则（read/write/edit/search 优先工具而非 bash）。
3. 在 runtime 启动时使用 builder，不再硬编码 `cli.py` 的系统提示文本。

### DoD
- prompt 由 builder 生成。
- 有独立测试覆盖模板渲染内容。

---

## P4 动态工具规划器（Tool Planner）

### 目标
- 不再全量暴露工具，按任务能力动态注入。

### 参考
- `_get_available_tools`：`backend/.../tools/__init__.py`

### 改动文件
- 新增：
  - `v-agent/src/v_agent/runtime/tool_planner.py`
  - `v-agent/tests/test_tool_planner.py`
- 修改：
  - `v-agent/src/v_agent/runtime/engine.py`

### 具体任务
1. 定义任务 capability（示例）：
   - `allow_interruption`
   - `use_workspace`
   - `agent_type`
   - `has_sub_agents`
   - `enable_document_tools`
   - `enable_workflow_tools`
   - `memory_usage_percentage`
2. planner 输出当前周期可用工具 schema 列表。
3. 支持阈值注入（如 `compress_memory` 仅高内存占用时注入）。

### DoD
- 不同 capability 下工具集可预测、可测。

---

## P5 Dispatcher（统一工具调用执行器）

### 目标
- 把“解析参数->执行 handler->组装结果->状态迁移信号”标准化。

### 参考
- `backend/.../tools/__init__.py` 中 `_process_tool_calls_logic`

### 改动文件
- 新增：
  - `v-agent/src/v_agent/tools/dispatcher.py`
  - `v-agent/tests/test_dispatcher_protocol.py`
- 修改：
  - `v-agent/src/v_agent/runtime/engine.py`
  - `v-agent/src/v_agent/tools/registry.py`

### 具体任务
1. 处理 arguments 解析错误，返回标准 ERROR payload。
2. 统一填充 `tool_call_id`。
3. 支持状态：
   - `WAIT_RESPONSE`: 触发 runtime wait_user
   - `RUNNING/BATCH_RUNNING`: 交给后续轮询器
4. 为图像类结果预留 message 注入钩子。

### DoD
- runtime 不再直接循环执行 registry，而通过 dispatcher。

---

## P6 工具实现拆分（handlers）

### 目标
- 把 `builtins.py` 大文件拆成按职责模块，便于维护。

### 参考
- workspace/read/write/edit/search 的 backend 各文件实现。

### 改动文件
- 新增目录：`v-agent/src/v_agent/tools/handlers/`
  - `control.py`
  - `todo.py`
  - `workspace_io.py`
  - `workspace_edit.py`
  - `search.py`
  - `common.py`（路径规范化、错误封装）
- 修改：
  - `v-agent/src/v_agent/tools/builtins.py`
  - `v-agent/src/v_agent/tools/__init__.py`

### 具体任务
1. 抽公共 path normalize 与 path escape 防护。
2. 拆 read/write/list/grep/replace/line_replace 到独立 handler。
3. `builtins.py` 仅负责注册，不包含复杂逻辑。

### DoD
- 单文件复杂度显著下降。
- 原有工具测试全部通过。

---

## P7 控制工具语义对齐（高优先）

### 目标
- `task_finish/ask_user/todo_write/todo_read` 行为对齐 backend 语义。

### 参考
- TODO 写入与单 `in_progress` 约束：`backend/.../tools/todo_write.py`
- finish TODO 守卫：`backend/.../tools/__init__.py` 的 `TASK_FINISH_TOOL_NAME` 逻辑
- `ASK_USER_TOOL_SCHEMA`：`backend/.../constants/workspace.py`

### 改动文件
- 修改：
  - `v-agent/src/v_agent/tools/handlers/control.py`
  - `v-agent/src/v_agent/tools/handlers/todo.py`
  - `v-agent/src/v_agent/constants/workspace.py`
- 新增：
  - `v-agent/tests/test_control_semantics.py`

### 具体任务
1. `todo_write` 改为完整列表写入模式（可更新/删除）。
2. 校验最多一个 `in_progress`。
3. `task_finish` 在 todo 未完成时返回结构化错误码（如 `todo_incomplete`）。
4. `ask_user` 返回结构化 options 与 selection 类型元数据。

### DoD
- todo 语义与 backend 一致。
- finish 防误结束能力通过测试。

---

## P8 runtime 状态机重构（核心）

### 目标
- 对齐 backend 的“周期生成 + 工具后处理”双阶段思路。

### 参考
- 周期主循环：`backend/.../agent_cycle.py`
- 工具后处理：`backend/.../tool_calls.py`

### 改动文件
- 新增：
  - `v-agent/src/v_agent/runtime/cycle_runner.py`
  - `v-agent/src/v_agent/runtime/tool_call_runner.py`
- 修改：
  - `v-agent/src/v_agent/runtime/engine.py`
  - `v-agent/src/v_agent/runtime/__init__.py`

### 具体任务
1. 将 `engine.py` 拆分：
   - `cycle_runner` 负责 LLM 调用与 assistant/tool_call 记录。
   - `tool_call_runner` 负责工具执行与状态收敛。
2. 支持 `continue hint` 注入（无工具调用时，提示继续或 finish）。
3. 支持 WAIT_USER/COMPLETED/MAX_CYCLES 明确迁移。

### DoD
- runtime 逻辑清晰分层。
- `test_runtime.py` 覆盖状态迁移路径。

---

## P9 高级工具接入（bash/background/read_image）

### 目标
- 提供 backend 同类协议能力，哪怕先本地化实现。

### 参考
- `run_bash_command.py`
- `check_background_command.py`
- `read_image.py`
- `tool_calls.py` 的图像消息注入

### 改动文件
- 新增：
  - `v-agent/src/v_agent/tools/handlers/bash.py`
  - `v-agent/src/v_agent/tools/handlers/background.py`
  - `v-agent/src/v_agent/tools/handlers/image.py`
  - `v-agent/src/v_agent/runtime/background_sessions.py`
- 新增测试：
  - `v-agent/tests/test_bash_tools.py`
  - `v-agent/tests/test_image_tool.py`

### 具体任务
1. bash 工具加入危险命令拦截与超时控制。
2. 背景任务返回 `RUNNING + session_id`，后续查询状态。
3. read_image 返回 `image_url/image_path`，并支持 runtime 注入消息内容。

### DoD
- 支持后台任务生命周期。
- 图像工具可返回并被 runtime 识别。

---

## P10 文档工具/工作流/skills 扩展接口（先接口，后服务）

### 目标
- 构建可扩展骨架，不强耦合你后端服务。

### 参考
- `document_navigation.py`
- `workflow.py` + `workflow_tools.py`
- `ACTIVATE_SKILL_TOOL_SCHEMA`

### 改动文件
- 新增：
  - `v-agent/src/v_agent/tools/handlers/document.py`
  - `v-agent/src/v_agent/tools/handlers/workflow.py`
  - `v-agent/src/v_agent/tools/handlers/skills.py`
  - `v-agent/src/v_agent/integrations/`（预留适配层）
- 测试：
  - `v-agent/tests/test_extension_points.py`

### 具体任务
1. 先定义接口与 mock 行为，默认 capability 关闭。
2. schema 先可注入，执行器先返回“未启用/未配置”标准错误。
3. 后续可替换成真实后端 adapter。

### DoD
- 扩展点完整，不影响主流程。

---

## P11 全量验收与收口

### 目标
- 通过静态检查、单测、真实模型联调；更新文档。

### 步骤
1. 全量检查：
   ```bash
   uv run ruff check .
   uv run ty check
   uv run pytest -q
   ```
2. 真实联调（你本地真实 key）：
   ```bash
   V_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live -q
   ```
3. 更新文档：
   - `v-agent/README.md`
   - `v-agent/docs/ARCHITECTURE.md`
   - 新增 `v-agent/docs/TOOL_PROTOCOL.md`（状态与 payload 规范）
4. 产出迁移说明：
   - `v-agent/docs/MIGRATION_FROM_V0.md`

### DoD
- 全部检查通过。
- 文档与代码一致。

---

## P12 LLM 统一接口对齐（新增）

### 目标
- 直接对齐 `backend/vector_vein_main/utilities/llm.py` 的统一请求策略，避免在 `v-agent` 维护一套偏离实现。

### 参考
- `backend/vector_vein_main/utilities/llm.py`：
  - `non_stream_response` 中模型参数映射（thinking / reasoning_effort / extra_body）
  - `stream_response` 中流式 tool call 增量聚合与 `last_active_tool_call_id` 处理
- `backend/vector_vein_main/utilities/constants.py`：
  - `TOOL_CALL_INCREMENTAL_BACKENDS`
  - `TOOL_CALL_INCREMENTAL_MODELS`
  - `CLAUDE_THINKING_MODELS`

### 改动文件
- 修改：
  - `v-agent/src/v_agent/llm/openai_compatible.py`
  - `v-agent/tests/test_llm_interface.py`

### 具体任务
1. 新增请求选项解析层（`_RequestOptions` + `_resolve_request_options`）并对齐模型特化规则：
   - deepseek temperature 默认值
   - claude thinking 模型后缀与 thinking 参数
   - o3/o4/gpt-5 `-high` 的 reasoning_effort 映射
   - qwen3 thinking / glm thinking / gemini 2.5 与 gemini 3 extra body 规则
2. 将 stream / non-stream 调用统一走 payload 构建器，避免参数分叉。
3. 对齐流式 tool call 聚合：
   - 支持“name+arguments”与“仅 arguments 片段”混合到达
   - 通过 `last_active_tool_call_id` 归并跨 chunk 参数
   - 兼容 provider extra（如 Gemini 3 的附加字段）保留到 raw。
4. 补齐单测：
   - 请求选项映射测试（claude/gemini/qwen）
   - 无 index 参数片段的 tool call 归并测试
   - 原有 failover/推理内容聚合测试保持通过

### DoD
- `openai_compatible.py` 的参数策略与 backend 参考实现一致（不再是自定义分叉实现）。
- `tests/test_llm_interface.py` 对关键模型和流式工具聚合行为有回归覆盖。
- 通过 `ruff` / `ty` / `pytest`，并能真实执行 CLI 命令。

---

## 4. 提交策略（建议）

建议按阶段拆 10~12 个 commit（每阶段至少 1 个 commit），示例：

1. `refactor: introduce tool/runtime protocol enums`
2. `refactor: centralize tool names and schemas`
3. `feat: add prompt builder and layered templates`
4. `feat: dynamic tool planner by capabilities`
5. `refactor: add dispatcher for tool execution protocol`
6. `refactor: split builtin handlers into modules`
7. `feat: align task_finish and todo semantics`
8. `refactor: split runtime cycle and tool-call runners`
9. `feat: add bash/background/image tool protocol`
10. `docs/test: add parity tests and migration docs`
11. `refactor: align llm request and streaming logic with backend utilities`

---

## 5. 风险与回滚策略

### 主要风险
- 状态机重构导致历史测试失效。
- schema 文案迁移时字段不一致导致 tool call 失败。
- 控制工具语义变化引发“任务结束不了”或“提前结束”。

### 规避
- 严格按阶段提交，禁止跨阶段大改。
- 每阶段全量跑测试。
- 关键控制工具先写语义测试再改实现（TDD）。

### 回滚
- 任何阶段失败，回退到上一个阶段 commit，不带问题进入下一阶段。

---

## 6. 最终验收标准（必须全部满足）

- 工具 schema 来源单一、可追踪、可测试。
- runtime 能表达并处理多状态工具执行协议。
- `task_finish/ask_user/todo_*` 行为与 backend 哲学一致。
- prompt 为分层构建且包含工具使用治理信息。
- 通过静态检查、单测、live 测试。
- 文档齐全：架构、协议、迁移说明、执行记录。
