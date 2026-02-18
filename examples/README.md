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

## 6) 读取图片并输出 Markdown 报告

文件：`examples/read_image_to_markdown.py`

```bash
uv run python examples/read_image_to_markdown.py \
  --settings-file local_settings.py \
  --workspace ./workspace \
  --image-path test_image.png \
  --output-path artifacts/image_read_report.md \
  --backend moonshot \
  --model kimi-k2.5 \
  --verbose
```

演示点：
- 使用 `native_multimodal=true` 启用 `_read_image`
- 强制先读图，再由模型输出结构化中文 Markdown
- 通过 `_write_file` 将结果写入 `.md` 文件

## 7) Remotion Skill 实测（使用 skills/ 目录自动发现）

文件：`examples/remotion_skill_demo.py`

```bash
uv run python examples/remotion_skill_demo.py \
  --settings-file local_settings.py \
  --workspace ./workspace \
  --skills-dir skills \
  --backend moonshot \
  --model kimi-k2.5 \
  --verbose
```

演示点：
- 给定一个 `skills/` 目录后，会自动发现其中全部 `SKILL.md` 并构建运行时技能包
- 若目录名与 frontmatter 的 `name` 不一致，会自动归一化到 `.v_agent_skill_cache/bundle/<name>/`
- 通过 `AgentDefinition.skill_directories` 自动把可用技能注入系统提示词 `<available_skills>`
- 用户提示词不指定技能名，由 Agent 根据技能描述自行选择并调用 `_activate_skill`
- 最终在 `workspace/artifacts/remotion_demo/` 输出可继续开发的代码与说明文档

