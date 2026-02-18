# v-agent

`v-agent` 是从 `backend/vector_vein_main/ai_agents/` 的机制抽象出来的轻量 Agent 框架，参考了 `ref_repos/agent_framework/` 与 `backend/vector_vein_main/utilities/llm.py` 的运行思路。

## Agent Philosophy

- **Cycle-first**：每一轮都清晰记录「输入上下文 -> LLM 决策 -> 工具执行 -> 状态迁移」。
- **Tool-driven completion**：任务完成/等待用户由工具显式触发（`_task_finish` / `_ask_user`）。
- **Memory with compression**：上下文超阈值后自动压缩历史，保留最近高价值对话。
- **Portable runtime**：不依赖 Django/Celery，可嵌入 CLI、服务端 worker 或测试环境。

## 核心模块

- `v_agent.runtime.AgentRuntime`: 顶层任务状态机（completed/wait_user/max_cycles/failed）。
- `v_agent.runtime.CycleRunner`: 单轮 LLM 调用与 cycle 记录构建。
- `v_agent.runtime.ToolCallRunner`: 工具调用执行与 directive 收敛。
- `v_agent.runtime.RuntimeHookManager`: 运行时 Hook 分发（before/after llm、tool_call、memory_compact）。
- `v_agent.runtime.BaseRuntimeHook`: 带完整类型提示的 Hook 基类（建议继承后按需覆写）。
- `v_agent.runtime.tool_planner`: 按 capability 动态规划可用工具 schema。
- `v_agent.tools.dispatcher`: 统一处理参数解析/错误码/状态码映射。
- `v_agent.tools.build_default_registry`: 默认工具集（workspace/todo/control/bash/background/image/sub-agent + skill extension stub）。
- `v_agent.sdk.AgentSDKClient`: 代码式 SDK 封装（支持默认单 Agent 直接 `run/query`，也支持命名 profiles；并提供模块级 `v_agent.sdk.run/query` one-shot helper）。
- `v_agent.sdk.AgentSession`: Session-first 会话 API（`prompt/continue_run/steer/follow_up/subscribe`）。
- `v_agent.sdk.AgentResourceLoader`: 自动发现 `.v-agent/` 与 `~/.v-agent/` 的 profiles/prompts/skills/hooks。
- `v_agent.skills`: Agent Skills 标准支持（`SKILL.md` frontmatter 解析/校验、`<available_skills>` prompt XML、标准化激活输入）。
- `v_agent.memory.MemoryManager`: 历史压缩器。
- `v_agent.llm.OpenAICompatibleLLM`: 统一 LLM 接口（端点轮询、重试、流式/非流式聚合、tool call 归一化）。
- `v_agent.config`: 从本地 `local_settings.py` 解析模型+端点+key。

## 配置

先复制示例配置：

```bash
cp local_settings.example.py local_settings.py
```

然后把 `local_settings.py` 里的 API Key、endpoint 按你的环境填好。

> `local_settings.py` 已在 `.gitignore` 中，不会被提交。

## 快速开始

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ty check
```

运行一个真实模型（默认读取当前目录的 `local_settings.py`）:

```bash
uv run v-agent --prompt "请概述一下这个框架的特点" --backend moonshot --model kimi-k2.5
```

查看每轮 LLM/工具执行日志（方便观察运行过程）:

```bash
uv run v-agent --prompt "请概述一下这个框架的特点" --backend moonshot --model kimi-k2.5 --verbose
```

可用参数：

- `--settings-file`: 指定配置文件路径（默认 `local_settings.py`）
- `--backend`: 后端名称（如 `moonshot`）
- `--model`: 模型名称（支持别名 `kimi-k2.5` -> `kimi-k2-thinking`）
- `--verbose`: 输出每轮 cycle 日志（LLM 响应摘要、tool 调用结果、状态迁移）

## 代码式示例（非 CLI）

`examples/` 下提供了多种集成模式，包含：

- `examples/quick_start_programmatic.py`：最小代码式调用
- `examples/agent_profiles.py`：通过 profile 复用不同 Agent 配置
- `examples/sdk_style_client.py`：参考 `claude-agent-sdk-python` 设计风格的 SDK 式封装

说明：
- workflow 不再作为内建特殊能力；如需 workflow，请按自定义工具注册（见 `tests/test_custom_tools.py` 的方式）。
- skills 采用 Agent Skills 规范（https://github.com/agentskills/agentskills），支持 `available_skills` 或 `AgentDefinition.skill_directories`。给定 skills 目录路径时会自动构建系统提示词中的 `<available_skills>`。
- document 工具不再作为内建能力；如需 document 能力，请按自定义工具注册。
- workspace 内建工具固定为：`_list_files`、`_file_info`、`_read_file`、`_write_file`、`_file_str_replace`、`_workspace_grep`、`_compress_memory`、`_todo_write`。
- 子 Agent 已有内建工具支持：`_create_sub_task` / `_batch_sub_tasks`（基于 `AgentTask.sub_agents` 配置）。
- 子 Agent 如需使用与父任务不同的模型/后端，请确保 runtime 提供 `settings_file` + `default_backend`（CLI/SDK 默认会提供）。

快速查看：

```bash
cat examples/README.md
```

## 实时集成测试

默认不会跑真实 LLM。

```bash
V_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live
```

可选环境变量：

- `V_AGENT_LOCAL_SETTINGS`: `local_settings.py` 绝对路径
- `V_AGENT_LIVE_BACKEND`: 默认 `moonshot`
- `V_AGENT_LIVE_MODEL`: 默认 `kimi-k2.5`
- `V_AGENT_ENABLE_BASE64_KEY_DECODE`: 设为 `1` 时启用 base64 API key 解码
