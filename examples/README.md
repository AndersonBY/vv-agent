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
