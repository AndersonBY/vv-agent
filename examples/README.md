# Examples

这些示例全部是独立 Python 脚本, 每个文件自包含, 可直接运行.

通过环境变量覆盖默认配置, 无需修改代码.

## 通用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `V_AGENT_LOCAL_SETTINGS` | `local_settings.py` | LLM 后端配置文件 |
| `V_AGENT_EXAMPLE_WORKSPACE` | `./workspace` | 工作目录 |
| `V_AGENT_EXAMPLE_BACKEND` | `moonshot` | LLM 后端 |
| `V_AGENT_EXAMPLE_MODEL` | `kimi-k2.5` | 模型名称 |
| `V_AGENT_EXAMPLE_VERBOSE` | `true` | 是否打印运行时日志 |

---

## Getting Started

| # | 文件 | 说明 |
|---|------|------|
| 01 | `01_quick_start.py` | 最小可用示例: 直接使用 `AgentRuntime` 底层 API |
| 02 | `02_agent_profiles.py` | 多 Agent 配置复用: researcher / translator / computer |
| 03 | `03_sdk_client.py` | SDK 嵌入形态: `AgentSDKClient` + run/query 双模式 |

```bash
# 快速开始
uv run python examples/01_quick_start.py

# 切换 profile
V_AGENT_EXAMPLE_PROFILE=translator uv run python examples/02_agent_profiles.py

# SDK client query 模式
V_AGENT_EXAMPLE_MODE=query uv run python examples/03_sdk_client.py
```

---

## Session & Interaction

| # | 文件 | 说明 |
|---|------|------|
| 04 | `04_session_api.py` | 会话式执行: `create_session` + steer / follow_up |
| 05 | `05_ask_user_resume.py` | WAIT_USER 恢复: `_ask_user` → `continue_run` 闭环 |

```bash
uv run python examples/04_session_api.py

V_AGENT_EXAMPLE_USER_REPLY="请使用口语化风格" uv run python examples/05_ask_user_resume.py
```

---

## Hooks & Guardrails

| # | 文件 | 说明 |
|---|------|------|
| 06 | `06_runtime_hooks.py` | 基础 hook: `before_llm` 注入上下文 + `before_tool_call` 拦截敏感路径 |
| 07 | `07_token_budget_guard.py` | Token 预算保护: 超限自动注入 `_task_finish` |
| 15 | `15_memory_compact_hook.py` | Memory compaction hook: 压缩前审计 + 关键消息保留 |
| 16 | `16_hook_composition.py` | 多 hook 组合: TimingHook + SafetyHook + AuditHook |

```bash
uv run python examples/06_runtime_hooks.py

V_AGENT_EXAMPLE_TOKEN_BUDGET=4000 uv run python examples/07_token_budget_guard.py

V_AGENT_EXAMPLE_PIN_KEYWORDS="priority,critical" uv run python examples/15_memory_compact_hook.py

uv run python examples/16_hook_composition.py
```

---

## Tools & Extensions

| # | 文件 | 说明 |
|---|------|------|
| 08 | `08_custom_tool.py` | 自定义工单工具: `_ticket_store` create/list |
| 09 | `09_resource_loader.py` | 资源自动发现: profiles / prompts / skills / hooks |
| 12 | `12_skill_activation.py` | Skills 目录发现 + Agent 自主激活技能 |

```bash
uv run python examples/08_custom_tool.py

uv run python examples/09_resource_loader.py

V_AGENT_EXAMPLE_SKILLS_DIR=skills uv run python examples/12_skill_activation.py
```

---

## Domain Pipelines

| # | 文件 | 说明 |
|---|------|------|
| 10 | `10_read_image.py` | 图片读取 + Markdown 报告输出 |
| 11 | `11_sub_agent_pipeline.py` | Sub-agent 流水线: research → writer → 最终报告 |
| 13 | `13_arxiv_pipeline.py` | arXiv 论文检索 + PDF 下载 + 图片解释 + 中文翻译 |
| 14 | `14_batch_sub_tasks.py` | `_batch_sub_tasks` 并行多文档处理 |

```bash
uv run python examples/10_read_image.py

uv run python examples/11_sub_agent_pipeline.py

V_AGENT_EXAMPLE_TOKEN_BUDGET=50000 uv run python examples/13_arxiv_pipeline.py

uv run python examples/14_batch_sub_tasks.py
```

---

## Advanced Patterns

| # | 文件 | 说明 |
|---|------|------|
| 17 | `17_error_recovery.py` | 错误恢复: 检测 MAX_CYCLES / FAILED 并自动重试 |

```bash
V_AGENT_EXAMPLE_MAX_RETRIES=2 uv run python examples/17_error_recovery.py
```

---

## Execution Engine (v2)

以下示例展示 v2 执行引擎的新能力: 取消、流式输出、线程后端、状态持久化.

| # | 文件 | 说明 |
|---|------|------|
| 18 | `18_cancellation.py` | `CancellationToken` 超时自动取消运行中的 agent |
| 19 | `19_streaming.py` | `stream_callback` 逐 token 流式输出 LLM 响应 |
| 20 | `20_thread_backend.py` | `ThreadBackend` 非阻塞 submit + Future 模式 |
| 21 | `21_state_checkpoint.py` | `SqliteStateStore` 持久化 checkpoint + 恢复 |
| 22 | `22_sdk_advanced.py` | SDK 层集成: ThreadBackend + 流式输出一站式配置 |
| 23 | `23_celery_backend.py` | `CeleryBackend` 分布式执行 + `celery.group` 并行 |
| 24 | `24_workspace_backends.py` | `WorkspaceBackend` 可插拔文件存储: Local / Memory / S3 / 自定义 |

```bash
# 取消: 10 秒后自动取消
V_AGENT_EXAMPLE_TIMEOUT=10 uv run python examples/18_cancellation.py

# 流式输出
uv run python examples/19_streaming.py

# ThreadBackend 同步 + 非阻塞两种模式
uv run python examples/20_thread_backend.py

# SQLite checkpoint 持久化
V_AGENT_EXAMPLE_DB=./workspace/agent.db uv run python examples/21_state_checkpoint.py

# SDK 一站式: ThreadBackend + streaming
uv run python examples/22_sdk_advanced.py

# CeleryBackend 分布式 (需先启动 Redis + worker, 详见文件头注释)
uv run python examples/23_celery_backend.py

# WorkspaceBackend 可插拔存储 (默认运行全部模式, S3 需配置环境变量)
uv run python examples/24_workspace_backends.py

# 仅运行内存后端模式
V_AGENT_EXAMPLE_WS_MODE=memory uv run python examples/24_workspace_backends.py

# 仅运行 S3 后端模式 (需先配置 .env, 参见 examples/.env.example)
# uv pip install 'vv-agent[s3]'
V_AGENT_EXAMPLE_WS_MODE=s3 uv run python examples/24_workspace_backends.py

# 仅运行自定义后端模式
V_AGENT_EXAMPLE_WS_MODE=custom uv run python examples/24_workspace_backends.py
```
