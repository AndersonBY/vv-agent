# Examples

这些示例都使用**代码式 API**（不是 `v-agent` CLI 封装），用于演示如何把 `v-agent` 集成到你自己的 Python 程序。

## 1) Quick Start（最小可用）

文件：`examples/quick_start_programmatic.py`

```bash
uv run python examples/quick_start_programmatic.py \
  --prompt "请概述一下这个框架的特点" \
  --backend moonshot \
  --model kimi-k2.5 \
  --verbose
```

演示点：
- 代码里直接创建 `AgentRuntime`
- 手动构建 `AgentTask`
- 使用 `log_handler` 打印每轮日志

## 2) Agent Profiles（多配置复用）

文件：`examples/agent_profiles.py`

```bash
uv run python examples/agent_profiles.py \
  --profile researcher \
  --prompt "分析 workspace 下这个文档的核心结论"
```

演示点：
- 用 `AgentProfile` 统一管理不同 Agent 配置
- 每个 profile 定义自己的 backend/model/max_cycles/tool capability
- 适合在业务里做“角色化 Agent”切换

## 3) SDK-style Client（参考 claude-agent-sdk 的设计风格）

文件：`examples/sdk_style_client.py`

```bash
uv run python examples/sdk_style_client.py \
  --agent planner \
  --prompt "先拆分任务，再逐步完成并汇报" \
  --mode run
```

只要最终文本时可用 one-shot query 风格：

```bash
uv run python examples/sdk_style_client.py \
  --agent planner \
  --prompt "一句话总结当前任务进展" \
  --mode query
```

演示点：
- `AgentDefinition` + `AgentSDKOptions` + `AgentSDKClient` 的三层抽象
- `client.run_agent(agent_name=..., prompt=...)` 的 SDK 调用体验
- `client.query(agent_name=..., prompt=...)` 的 one-shot 文本查询体验
- 一个进程里管理多个命名 Agent（planner / translator / orchestrator）
- `orchestrator` 示例包含 `sub_agents` 配置输入方式（可触发 `_create_sub_task` / `_batch_sub_tasks`）

## 4) 自定义 workflow 工具

workflow 建议作为自定义工具注册，不做内建特殊分支。可参考：

- `tests/test_custom_tools.py`

## 5) arXiv 论文检索 + 下载 + 图片解释 + 中文翻译（AI Agent Memory）

文件：`examples/arxiv_agent_memory_pipeline.py`

```bash
uv run python examples/arxiv_agent_memory_pipeline.py \
  --settings-file local_settings.py \
  --workspace ./workspace/arxiv_memory_demo \
  --model kimi-k2.5 \
  --verbose
```

演示点：
- 使用 `moonshot` + `kimi-k2.5` 运行一个端到端研究任务
- 自动搜索最近 30 天的 AI Agent Memory 相关 arXiv 论文并下载 PDF
- 抽取第一张图并调用 `_read_image` 做图像解释
- 将论文分段翻译为中文并持续写入结果文件
