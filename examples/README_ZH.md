# 示例

[English](README.md)

这些脚本都是自包含示例，需要从仓库根目录运行。公开 SDK 示例使用
`Agent`、`Runner`、`RunConfig`、`ModelSettings`、`function_tool`、Session、
handoff、强类型事件和工具策略。低层 runtime 示例单独保留，用于后端集成场景。

宿主产品迁移时，优先使用 provider 和 executor 扩展点，不要 patch runtime 内部：
`ApprovalProvider` 负责 UI/规则审批，`ContextProvider` 负责产品 prompt 片段，
`vv_agent.memory.MemoryProvider` 负责产品持久化，`vv_agent.tools.ToolExecutor`
或 `FunctionTool` 组合负责产品工具，`RunEventStore` 负责应用历史。宿主需要实时
强类型事件、取消或审批控制时，使用 `Runner.start()` 和 `RunHandle`。

## 通用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `V_AGENT_LOCAL_SETTINGS` | `local_settings.py` | LLM 后端配置文件 |
| `V_AGENT_EXAMPLE_WORKSPACE` | `./workspace` | 工作区目录 |
| `V_AGENT_EXAMPLE_BACKEND` | `moonshot` | LLM 后端 |
| `V_AGENT_EXAMPLE_MODEL` | `kimi-k2.6` | 模型名称 |
| `V_AGENT_EXAMPLE_PROMPT` | 示例内置值 | 覆盖示例 prompt |

## 公开 SDK

| # | 文件 | 展示内容 |
| --- | --- | --- |
| 01 | `01_quick_start.py` | 最小 `Agent` + `Runner.run_sync` |
| 02 | `02_agent_profiles.py` | 可复用 Agent profile 与 `ModelSettings` |
| 03 | `03_sdk_client.py` | Runner 配置、Session、自定义工具、强类型事件 |
| 04 | `04_session_api.py` | 使用 `MemorySession` 跨多次 run 保留上下文 |
| 05 | `05_ask_user_resume.py` | 工具审批与 `WAIT_USER` 事件 |
| 06 | `06_runtime_hooks.py` | 通过 `RunConfig.runtime_hooks` 接入低层 hook |
| 07 | `07_token_budget_guard.py` | 用 input guardrail 做 prompt 预算检查 |
| 08 | `08_custom_tool.py` | `@function_tool` schema 推导与结构化输出 |
| 09 | `09_resource_loader.py` | 从 JSON 资源加载 Agent profile |
| 10 | `10_read_image.py` | 通过 typed tool 读取图片元信息 |
| 11 | `11_sub_agent_pipeline.py` | `agent.as_tool()` 与 `handoff()` 组合 |
| 12 | `12_skill_activation.py` | 通过 Agent metadata 暴露可用 skill |
| 13 | `13_arxiv_pipeline.py` | 用 function tools 组成论文检索流水线 |
| 14 | `14_batch_sub_tasks.py` | 用 agent-as-tool 做批量式协调 |
| 15 | `15_memory_compact_hook.py` | Memory compact 审计 hook |
| 16 | `16_hook_composition.py` | 多个 runtime hook 组合 |
| 17 | `17_error_recovery.py` | `Runner.run_sync` 外层重试封装 |

```bash
uv run python examples/01_quick_start.py
V_AGENT_EXAMPLE_PROFILE=translator uv run python examples/02_agent_profiles.py
V_AGENT_EXAMPLE_SESSION_ID=demo uv run python examples/04_session_api.py
uv run python examples/08_custom_tool.py
uv run python examples/11_sub_agent_pipeline.py
uv run python examples/17_error_recovery.py
```

## Runtime 集成

这些示例使用更低层的 runtime API，展示取消、流式输出、线程后端、checkpoint、
Celery 分发和工作区后端。

| # | 文件 | 展示内容 |
| --- | --- | --- |
| 18 | `18_cancellation.py` | 通过 `CancellationToken` 取消运行中的任务 |
| 19 | `19_streaming.py` | 原始 runtime stream callback 事件 |
| 20 | `20_thread_backend.py` | `ThreadBackend` submit/future 执行 |
| 21 | `21_state_checkpoint.py` | `SqliteStateStore` checkpoint |
| 22 | `22_sdk_advanced.py` | 公开 SDK + streaming + `ThreadBackend` |
| 23 | `23_celery_backend.py` | `CeleryBackend` 分布式 cycle |
| 24 | `24_workspace_backends.py` | Local、memory、S3 与自定义工作区后端 |
| 25 | `25_temporary_tool_injection.py` | run 级别临时启用工具 |

```bash
V_AGENT_EXAMPLE_TIMEOUT=10 uv run python examples/18_cancellation.py
uv run python examples/19_streaming.py
uv run python examples/20_thread_backend.py
V_AGENT_EXAMPLE_DB=./workspace/agent.db uv run python examples/21_state_checkpoint.py
uv run python examples/22_sdk_advanced.py
uv run python examples/24_workspace_backends.py
```

## App Server 集成

这些示例使用 JSON-RPC App Server 边界，展示宿主产品、子进程客户端、生命周期
回放、通知过滤、schema 导出和 overload 处理。示例 26-28 会通过
`local_settings.py` 和 `V_AGENT_EXAMPLE_*` 环境变量调用真实模型；示例 29 是本地
协议、schema 和 backpressure 示例，不调用模型。

| # | 文件 | 展示内容 |
| --- | --- | --- |
| 26 | `26_app_server_channel_lifecycle.py` | 通过进程内 `ChannelTransport` 跑真实模型 turn：initialize、thread、turn、list、archive |
| 27 | `27_app_server_stdio_client.py` | 通过 stdio JSONL 子进程客户端跑真实模型 turn |
| 28 | `28_app_server_notification_opt_out.py` | 两个客户端跑真实模型 turn，并演示连接级通知 opt-out |
| 29 | `29_app_server_schema_backpressure.py` | schema 导出、transport overload 与 request backlog overload |

```bash
uv run python examples/26_app_server_channel_lifecycle.py
uv run python examples/27_app_server_stdio_client.py
uv run python examples/28_app_server_notification_opt_out.py
uv run python examples/29_app_server_schema_backpressure.py
```
