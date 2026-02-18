# Examples

这些示例全部是独立 Python 脚本, 不使用 argparse CLI 参数.

你可以直接运行脚本, 也可以通过环境变量覆盖默认配置.

## 通用环境变量

- `V_AGENT_LOCAL_SETTINGS` (默认 `local_settings.py`)
- `V_AGENT_EXAMPLE_WORKSPACE` (默认 `./workspace`)
- `V_AGENT_EXAMPLE_BACKEND` (默认 `moonshot`)
- `V_AGENT_EXAMPLE_MODEL` (默认 `kimi-k2.5`)
- `V_AGENT_EXAMPLE_PROMPT` (每个示例有自己的默认 prompt)
- `V_AGENT_EXAMPLE_VERBOSE` (默认 `true`)

## 1) Quick Start (最小可用)

文件: `examples/quick_start_programmatic.py`

```bash
uv run python examples/quick_start_programmatic.py
```

可选覆盖示例:

```bash
V_AGENT_EXAMPLE_PROMPT="请总结 workspace 下的主要文件" \
V_AGENT_EXAMPLE_MAX_CYCLES=20 \
uv run python examples/quick_start_programmatic.py
```

## 2) Agent Profiles (多配置复用)

文件: `examples/agent_profiles.py`

```bash
uv run python examples/agent_profiles.py
```

可选覆盖示例:

```bash
V_AGENT_EXAMPLE_PROFILE=translator \
V_AGENT_EXAMPLE_PROMPT="把 workspace/demo.md 翻译成中文" \
uv run python examples/agent_profiles.py
```

## 3) SDK-style Client (接近 SDK 嵌入形态)

文件: `examples/sdk_style_client.py`

这个示例演示两种方式:
- 默认单 Agent：`client.run(prompt=...)` / `client.query(prompt=...)`
- 命名 profile：`client.run(agent=\"translator\", prompt=...)`

```bash
uv run python examples/sdk_style_client.py
```

可选覆盖示例:

```bash
V_AGENT_EXAMPLE_AGENT=orchestrator \
V_AGENT_EXAMPLE_MODE=query \
V_AGENT_EXAMPLE_PROMPT="一句话总结当前任务进度" \
uv run python examples/sdk_style_client.py
```

## 4) arXiv 论文检索 + 下载 + 图片解释 + 中文翻译

文件: `examples/arxiv_agent_memory_pipeline.py`

```bash
uv run python examples/arxiv_agent_memory_pipeline.py
```

可选覆盖示例:

```bash
V_AGENT_EXAMPLE_WORKSPACE=./workspace/arxiv_memory_demo \
V_AGENT_EXAMPLE_MODEL=kimi-k2.5 \
uv run python examples/arxiv_agent_memory_pipeline.py
```

## 5) 读取图片并输出 Markdown 报告

文件: `examples/read_image_to_markdown.py`

```bash
uv run python examples/read_image_to_markdown.py
```

可选覆盖示例:

```bash
V_AGENT_EXAMPLE_IMAGE_PATH=test_image.png \
V_AGENT_EXAMPLE_OUTPUT_PATH=artifacts/image_read_report.md \
uv run python examples/read_image_to_markdown.py
```

## 6) Skills 目录自动发现 + Agent 自主激活技能

文件: `examples/remotion_skill_demo.py`

```bash
uv run python examples/remotion_skill_demo.py
```

可选覆盖示例:

```bash
V_AGENT_EXAMPLE_SKILLS_DIR=skills \
V_AGENT_EXAMPLE_MODEL=kimi-k2.5 \
uv run python examples/remotion_skill_demo.py
```

这个示例会:
- 自动扫描 `skills/` 目录下所有 `SKILL.md`
- 自动构建系统提示词中的 `<available_skills>`
- 由 Agent 根据技能简介自主选择并调用 `_activate_skill`
- 输出 Remotion demo 到 `workspace/artifacts/remotion_demo/`

## 7) Session API（会话式执行 + steer/follow-up）

文件: `examples/session_api_embed.py`

```bash
uv run python examples/session_api_embed.py
```

这个示例会:
- 演示 `client.create_session()` 的会话式调用
- 演示 `session.steer(...)` 和 `session.follow_up(...)`
- 实时订阅并打印 session + runtime 事件

## 8) Runtime Hooks（运行时拦截与上下文注入）

文件: `examples/runtime_hooks_embed.py`

```bash
uv run python examples/runtime_hooks_embed.py
```

这个示例会:
- 使用 `before_llm` 注入额外上下文消息
- 使用 `before_tool_call` 拦截敏感路径写入

## 9) Resource Loader（自动发现 profiles/prompts/skills/hooks）

文件: `examples/resource_loader_embed.py`

```bash
uv run python examples/resource_loader_embed.py
```

这个示例会:
- 自动发现 `workspace/.v-agent/` 与 `~/.v-agent/` 资源
- 自动加载 profile、prompt template、skills 目录和 runtime hooks
