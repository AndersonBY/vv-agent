# vv-agent

[English](README.md)

vv-agent 是一个轻量的 Python Agent 框架，从向量脉络（VectorVein）的生产运行时中抽取而来。Agent 按轮次（cycle）运行：每一轮调用模型，执行模型要求的工具，再决定是否继续。框架提供可插拔的 LLM 后端、工具分发、上下文压缩，以及崩溃后可以恢复的持久会话。

## 安装

```bash
python -m pip install vv-agent
```

两个可选 extra 提供额外的存储后端：

- `postgres`：PostgreSQL 会话存储（`PostgresStore`）。
- `s3`：S3 工作区后端（`S3WorkspaceBackend`）。

```bash
python -m pip install 'vv-agent[postgres,s3]'
```

从源码开发时，运行 `python -m pip install -e .` 或 `uv sync --dev`。

vv-agent 0.27.0 实现了语言无关的 `vv-agent-contract` 29.1.0 版，具体版本锁定在 `contract.lock.json`。Python 是持续维护的实现。Rust 版 `vv-agent-rs` 冻结在合约 23.0.0（包版本 0.21.x），只做维护修复。Python API 沿用 Python 的命名习惯，详见[合约工作流](docs/parity-contract.md)。

如果你从旧版本升级，并且把会话存在 SQLite 文件或 PostgreSQL 里，请先读[宿主升级指南](docs/upgrading.md)。新版本可能读不了已存储的会话，指南里写了升级前要做哪些准备。

## 配置模型

模型端点和 API Key 写在一个 Python 配置文件里。复制模板，再填入你的 Key：

```bash
cp local_settings.example.py local_settings.py
```

模板里定义了 `moonshot` 后端和 `kimi-k3` 模型。CLI、示例和 live 测试默认都用这两个名字。不要提交 `local_settings.py`。配置结构和模型 key 的解析方式见[模型配置](docs/model-settings.md)。

## 快速开始

### CLI

```bash
uv run vv-agent --prompt "概述一下这个框架" --backend moonshot --model kimi-k3

# 每一轮打印一行日志
uv run vv-agent --prompt "概述一下这个框架" --verbose
```

| 参数 | 默认值 |
| --- | --- |
| `--backend` | `moonshot` |
| `--model` | `kimi-k3` |
| `--settings-file` | `$VV_AGENT_LOCAL_SETTINGS`，未设置时为 `local_settings.py` |
| `--workspace` | `./workspace` |
| `--max-cycles` | `80` |
| `--language` | `zh-CN`（系统提示词语言，也可用 `en-US`） |
| `--agent-type`、`--temperature`、`--top-p`、`--max-tokens` | 不设置 |
| `--verbose` | 关闭 |

### Python

```python
from vv_agent import Agent, RunConfig, Runner, VvLlmModelProvider, function_tool

provider = VvLlmModelProvider.from_settings_file("local_settings.py").with_default_backend("moonshot")

@function_tool
def read_order(order_id: str) -> str:
    """读取订单信息。"""
    return f"订单 {order_id}：2 件商品，已发货"

agent = Agent(
    name="ops",
    instructions="先查证，再回答。",
    model="kimi-k3",
    tools=[read_order],
)

result = Runner.run_sync(agent, "分析订单 123", run_config=RunConfig(model_provider=provider))
print(result.status, result.final_output)
```

每次运行都需要一个 `ModelProvider`。`VvLlmModelProvider` 读取配置文件，通过 `vv-llm` 调用模型。下文的示例都沿用这里的 `provider`。

默认情况下，模型回复时如果没有调用工具，运行就结束。框架不会去猜一段回复“像不像最终答案”，只按声明的 no-tool policy 处理，详见[结束一次运行](#结束一次运行)。

### 共享默认值

多次运行用同一组配置时，先配置一个 Runner：

```python
from vv_agent import RunConfig, Runner

runner = Runner.configured(RunConfig(
    model_provider=provider,
    model="kimi-k3",
    workspace="./workspace",
))
result = runner.run_sync(agent, "分析订单 123")
```

各项取值的顺序：

- Provider：先看本次运行的 `RunConfig`，再看 Runner 默认值。
- 模型：本次运行的配置、`Agent.model`、Runner 默认值、Provider 的默认模型，依次往后找。如果本次运行自带 Provider，就跳过 Runner 的模型。
- `ModelSettings`：按 Provider 默认值、Runner、Agent、本次运行逐层合并，后面的层覆盖前面的字段。

设置 `Agent.output_type` 后，JSON 格式的最终输出会转换成 `dict`、`list`、dataclass 或 Pydantic 风格的 model。校验和修复见[输出校验](docs/output-validation.md)。

## 流式输出与取消

`Runner.start()` 立即返回一个 `RunHandle`，Agent 在后台运行。用 `events()` 逐个读取强类型的 `RunEvent`，用 `result()` 等待结果，用 `cancel()` 停止运行，比如在 UI 按钮或定时器里调用。

```python
from vv_agent import Agent, RunConfig, Runner

agent = Agent("assistant", "简短回答。", model="kimi-k3")
handle = Runner.start(agent, "检查这个项目", run_config=RunConfig(model_provider=provider))
for event in handle.events():
    if event.type == "assistant_delta":
        print(event.delta, end="")
print(handle.result().final_output)
```

每次运行都是会话存储（session store）里的一个会话。普通的 Runner 运行使用一个全新的内存 SQLite 存储。持久事件从会话日志（log）中读出；assistant 增量（delta）只在运行时推送，不落盘。需要文件副本时，可以加一个 `JsonlRunEventStore`。等待、恢复、预算和事件投影见[运行控制](docs/runtime-control.md)。

## 多轮会话

`InteractiveAgentClient` 管理对话：多轮交互、插话引导（steering）、追加消息（follow-up）、取消和审批。它默认把会话存在内存里。要在重启后保留会话，把 `SQLiteStore` 或 `PostgresStore` 传给 `AgentSessionOptions.session_store`。

```python
from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, SQLiteStore

with SQLiteStore.standalone("sessions.sqlite3") as store:
    if store.connection.execute("PRAGMA user_version").fetchone()[0] == 0:
        store.install_schema()
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, session_store=store))
    try:
        session = client.create_session(
            agent=Agent("assistant", "记住之前的对话。", model="kimi-k3"),
            session_id="thread-001",
        )
        print(session.prompt("记住项目代号：River。").final_output)
        print(session.prompt("项目代号是什么？").final_output)
    finally:
        client.driver.close()
```

会话的几条规则：

- `create_session(session=...)` 只接受包含 `messages` 和 `shared_state` 的初始数据（seed）。创建之后，`session.messages` 和 `session.shared_state` 都是只读视图。见[创建时的初始数据](docs/upgrading.md#creation-time-seed)。
- 等待用户回复或审批的轮次（turn）会保留 `session_id` 和 `turn_id`。回复之后，用 `Runner.resume(session_id, turn_id)`、`AgentSession.continue_run()` 或 App Server 的 `turn/resume` 请求继续驱动同一个轮次。
- 恢复的轮次沿用原来的预算计数，新的轮次从零开始计数。
- 已关闭的会话拒绝新的执行。

要让多个 worker 共用一个 PostgreSQL 数据库来运行会话，见[云端宿主集成](docs/host-integration.md)。

## App Server

如果桌面应用、worker、IDE 或其他进程需要通过稳定的协议驱动 vv-agent，而不是直接引入 Python SDK，就用 App Server。它通过 stdio 收发 JSONL，提供：

- Thread、Turn、Item 的生命周期事件。
- 工具审批以 server-to-client 请求的形式发给客户端。
- `thread/read` 和 `thread/resume` 回放。
- 生成 JSON Schema 和自包含的 TypeScript 类型定义。

```bash
uv run vv-agent app-server --listen stdio --settings local_settings.py --backend moonshot --model kimi-k3
uv run vv-agent app-server schema --out ./app-server-schema
uv run vv-agent app-server generate-ts --out ./app-server-schema/typescript
uv run vv-agent debug app-server send-message "hello"
```

App Server 用的参数是 `--settings`，不是 `--settings-file`。`--listen`、`--settings`、`--backend` 和 `--model` 必填；`--timeout-seconds` 设置模型请求的超时时间。

产品宿主实现 `AppServerHost`，把自己的 profile、工作区上下文、工具、审批 UI、memory 和模型设置转换成 `Agent` 与 `RunConfig`。App Server 不会引入产品的 UI、账号、计费、浏览器或 IM 代码。见 [App Server 协议](docs/app-server.md)和 [App Server 宿主集成](docs/app-server-host-integration.md)。

## 工具

### 自定义工具

给 Python 函数加上 `@function_tool`，再放进 `Agent(tools=[...])`。装饰器会根据函数签名生成 JSON schema。工具可以把 `ToolContext` 作为第一个参数；运行时会传入它，但不会写进 schema。

更底层的工具可以注册到 `ToolRegistry`（`register()` 或 `register_tool()`），再通过 `RunConfig.tool_registry_factory` 提供这个 registry。

### 审批

用 `@function_tool(needs_approval=True)` 声明的工具，会在函数体执行前暂停运行。运行进入 `WAIT_USER`，并发出 `ApprovalRequestedEvent`。给出审批结果后，恢复这个 handle：

```python
from vv_agent import Agent, ApprovalDecision, RunConfig, Runner, function_tool

@function_tool(needs_approval=True)
def delete_temp_files() -> str:
    """删除临时文件。"""
    return "已删除 3 个文件"

agent = Agent("ops", "需要时清理工作区。", model="kimi-k3", tools=[delete_temp_files])
handle = Runner.start(agent, "清理工作区", run_config=RunConfig(model_provider=provider))
for event in handle.events():
    if event.type == "approval_requested":
        handle.approve(event.request_id, ApprovalDecision.allow())

result = handle.result()
if result.status == "wait_user":
    result = handle.resume()
print(result.final_output)
```

`ToolPolicy(approval=...)` 可取 `default`、`always`、`never`（用于可信的运行）或 `on_request`。`ApprovalProvider` 可以根据产品规则或 UI 给出结果：允许、拒绝、本会话内允许，或超时。各层如何组合见[审批](docs/architecture.md#approval)。

### 能力声明与策略

工具可以声明自己做的是哪一类事。这些元数据只给宿主看，模型看不到。

```python
from vv_agent import ToolIdempotency, ToolMetadata, ToolPolicy, ToolSideEffect, function_tool

@function_tool(
    tool_metadata=ToolMetadata(
        side_effect=ToolSideEffect.EXTERNAL,
        idempotency=ToolIdempotency.UNSUPPORTED,
        terminal=False,
        capability_tags=["ticket.write"],
        cost_dimensions=["support_api.request"],
    )
)
def create_ticket(title: str) -> dict[str, str]:
    """创建工单。"""
    return {"ticket_id": "TCK-1001", "title": title}

policy = ToolPolicy(
    denied_side_effects=[ToolSideEffect.EXECUTE],
    denied_capability_tags=["filesystem.delete"],
    deny_terminal_tools=True,
    denied_cost_dimensions=["gpu.second"],
)
```

`side_effect` 是一个粗粒度标签，各个值之间没有层级。`capability_tags` 和 `cost_dimensions` 都是按原文精确匹配的标签。`terminal=True` 只表示工具可能返回 `finish` 或 `wait_user`。这份元数据不会进入模型可见的 schema。

`denied_*` 和 `deny_terminal_tools` 这些策略字段只能移除工具。Agent、配置好的 Runner、本次运行和委托出去的子 Agent 各层的拒绝规则会叠加，命中时返回 `tool_not_allowed`。见[工具元数据](docs/architecture.md#tool-metadata)和[工具策略](docs/architecture.md#tool-policy)。

### 内建工具

`find_files`、`file_info`、`read_file`、`write_file`、`edit_file`、`search_files`、`todo_write`、`ask_user`、`bash`、`check_background_command`、`stop_background_command`、`read_image`、`create_sub_task`、`sub_task_status`、`activate_skill`。

文件搜索针对大工作区做了处理：

- `find_files` 默认最多返回 100 个路径，可以用 `max_results` 调大（有硬上限）。目录树很大时，`scan_limit` 会提前停止，并用 `count_is_estimate=true` 标明计数是估算值。
- 本地工作区里，`find_files` 和 `search_files` 在装有 `ripgrep`（`rg`）时用它遍历，否则退回 Python 遍历。`search_files` 默认 smart case：全小写的 pattern 不区分大小写。结构化的命中和计数放在 `ToolExecutionResult.metadata`。
- 隐藏目录和 `node_modules`、`.venv`、`.git` 这类依赖或缓存目录会被跳过，从根目录列出时只显示摘要。把 `path` 指向其中某个目录，或设置 `include_ignored=true`，就能查看里面的内容。`.env`、私钥这类敏感文件需要设置 `include_sensitive=true` 才会返回。

`bash` 最多等待 `yield_time_ms`（默认 1000 毫秒）让命令结束。命令还在运行时，工具返回当前输出和一个 `session_id`，进程继续运行。`timeout_seconds` 可以设置一个从进程启动时算起的截止时间。`check_background_command` 和 `stop_background_command` 接收这个 `session_id`。见 [Bash 进程管理](docs/bash-process-management.md)。

`bash` 或 `read_file` 的结果被截断时，`ToolExecutionResult` 会指向其余部分：`bash` 指向一个不可变的工作区 artifact，`read_file` 指向一个 cursor。文件内容变了、路径不一致或偏移量无效时，cursor 会拒绝读取。`tool_result` 诊断事件包含模型可见的 `content`、状态、指令、错误码和 metadata，但不包含 artifact 或 cursor。

## 多 Agent 运行

子 Agent 返回结果后由父 Agent 继续时，用 `agent.as_tool()`。要让目标 Agent 接管、并由它的输出结束本次运行时，用 `handoff()`。

```python
from vv_agent import Agent, RunConfig, Runner, handoff

researcher = Agent(name="researcher", instructions="负责收集事实。", model="kimi-k3")
writer = Agent(
    name="writer",
    instructions="根据资料写作。",
    model="kimi-k3",
    tools=[researcher.as_tool(name="research", description="收集事实。")],
)
triage = Agent(
    name="triage",
    instructions="把写作任务转交给 writer。",
    model="kimi-k3",
    handoffs=[handoff(agent=writer, description="需要写作时使用。")],
)

result = Runner.run_sync(
    triage,
    "写一份简短报告。",
    run_config=RunConfig(model_provider=provider, max_handoffs=4),
)
```

模型通过调用 `transfer_to_<agent 名称>` 工具（这里是 `transfer_to_writer`）发起 handoff，Runner 随后切换 Agent。目标 Agent 自己解析模型和模型设置。会话、取消令牌（cancellation token）和对共享状态的修改都会延续下去。`max_handoffs` 默认为 `10`，和 `max_cycles` 分开计数。审批后恢复的运行也遵循同样的规则。

模型也可以用 `create_sub_task` 和 `sub_task_status` 自己管理后台或并行任务。每个子任务都是独立的会话，持有自己的租约（lease，即 worker 对会话的驱动权）。父任务接纳子任务时，会固定它的提示词、策略、模型和预算，并且只采纳这个子任务自己的最终结果。子任务等待用户时，等待留在子任务上。`sub_task_status` 可以读取进度、等待完成或发送追加消息；宿主推送一条定向的 `steer` inbox 条目来引导子任务。子任务沿用父运行的 `ModelProvider`，自己解析模型。见[委托与后台任务](docs/runtime-control.md#delegation-and-background-work)。

## 结束一次运行

一次运行会在两种情况下结束或暂停：

- 工具返回 `finish` 或 `wait_user` 指令。比如 `ask_user` 会暂停运行，等待回复。
- 模型回复时没有调用工具，由 no-tool policy 决定怎么处理。

no-tool policy 默认是 `finish`。可以在 Agent 上用 `Agent(no_tool_policy=...)` 设置，也可以用 `RunConfig(no_tool_policy=...)` 只对一次调用生效。可选值是 `finish`、`continue` 和 `wait_user`。本次运行的值优先于 Runner 默认值，Runner 默认值优先于 Agent 上的值。

`result.completion_reason` 说明运行为什么停下：工具结束、无工具结束、等待、取消、失败、预算耗尽或轮次用完。`result.completion_tool_name` 和 `result.partial_output` 提供更多细节。普通运行默认最多 10 轮（`max_cycles`），CLI 默认 80 轮。

`RunConfig.budget_limits` 接收一个 `RunBudgetLimits`，可以限制总 token、未缓存的输入 token、工具调用总数或某个工具的调用次数、实际运行时间，以及宿主计量的成本。每项限制都是可选的。预算只管资源，不评判答案。预算耗尽时，运行以明确类型的失败结束，不会算作成功；用 `result.budget_usage` 和 `result.budget_exhaustion` 查看详情。见[运行预算](docs/run-budgets.md)。

## Guardrail 与 Trace

输入 guardrail 在调用模型之前执行，输出 guardrail 在得到最终输出之后执行。Trace processor 会收到轻量的 run span 和 tool span。

```python
from vv_agent import Agent, GuardrailResult, RunConfig, Runner, input_guardrail

@input_guardrail
def reject_empty(ctx, input_text: str) -> GuardrailResult:
    del ctx
    if not input_text.strip():
        return GuardrailResult.block("input is required")
    return GuardrailResult.allow()

agent = Agent(
    name="assistant",
    instructions="谨慎回答。",
    model="kimi-k3",
    input_guardrails=[reject_empty],
)

result = Runner.run_sync(
    agent,
    "总结这个项目。",
    run_config=RunConfig(model_provider=provider, tracing={"workflow_name": "summary"}),
)
```

## 工作区

内建的文件工具通过 `WorkspaceBackend` 读写文件。

| 后端 | 适用场景 |
| --- | --- |
| `LocalWorkspaceBackend` | 默认后端。读写本地目录，阻止路径逃出该目录。 |
| `MemoryWorkspaceBackend` | 存在内存 dict 里。适合测试和沙箱运行。 |
| `S3WorkspaceBackend` | S3 兼容的对象存储：AWS S3、阿里云 OSS、MinIO、Cloudflare R2。需要 `s3` extra。 |

```python
from pathlib import Path
from vv_agent import RunConfig
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

local_config = RunConfig(workspace_backend=LocalWorkspaceBackend(Path("./workspace")))
memory_config = RunConfig(workspace_backend=MemoryWorkspaceBackend())
```

```python
from vv_agent.workspace import S3WorkspaceBackend

s3_backend = S3WorkspaceBackend(
    bucket="my-bucket",
    prefix="agent-workspace",
    endpoint_url="https://oss-cn-hangzhou.aliyuncs.com",  # AWS 填 None
    aws_access_key_id="...",
    aws_secret_access_key="...",
    addressing_style="virtual",  # MinIO 用 "path"
)
```

要接入其他存储，实现 `src/vv_agent/workspace/base.py` 里的 `WorkspaceBackend` 协议。它包括列出文件、读取文本和二进制、普通写入和排他写入（只能新建）、文件信息、存在性与文件类型检查，以及创建目录。

## Shell 设置

`bash` 工具用哪个 shell，是在运行开始时决定的，不是每次调用工具时传参。在 `RunConfig.metadata` 里设置只对一次运行生效，在 `Agent.metadata` 里设置则对这个 Agent 生效：

- `bash_shell`：使用的 shell。
- `windows_shell_priority`：Windows 上依次尝试的 shell。推荐 `["git-bash", "powershell", "cmd"]`。
- `bash_env`：命令额外使用的环境变量。

```python
from vv_agent import Agent, RunConfig, Runner

agent = Agent(
    name="desktop",
    instructions="桌面助手",
    model="kimi-k3",
    metadata={"bash_env": {"HTTP_PROXY": "http://127.0.0.1:7890"}},
)
result = Runner.run_sync(
    agent,
    "检查工作区。",
    run_config=RunConfig(
        model_provider=provider,
        metadata={
            "windows_shell_priority": ["git-bash", "powershell", "cmd"],
            "bash_env": {"PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple"},
        },
    ),
)
```

在 Windows 上，如果父进程环境或 `bash_env` 没有设置，命令会自动带上 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。命令启动时也不创建控制台窗口，GUI 宿主运行命令时不会闪出终端。

`bash` 工具的描述会告诉模型将使用哪种 shell 和命令前缀。这段提示和本次运行的 `PromptBundle` 一样，在整次运行中保持不变，所以每一轮发出的请求前缀都相同，provider 的 prompt cache 能持续命中。发给 Anthropic 的请求可以按提示词的 section 设置缓存断点，其他 provider 收到的是展平后的提示词。

## 上下文压缩

历史长到模型装不下时，`MemoryManager` 会压缩它。它先归档较旧的大工具结果（microcompaction）；如果还不够，再把较早的历史生成摘要，并保留最近 10 条原始消息（`memory_keep_recent_messages`）。任何一步失败，历史都保持原样。

- 压缩阈值取 `memory_compact_threshold`（默认 `250000` token）和模型的提示词容量中较小的一个。容量等于上下文窗口减去输出预留，再减去 `13000` token 的缓冲。达到阈值的 `memory_threshold_percentage`（默认 `90`）百分比时出现内存预警。这两个设置是 `AgentTask` 和 `InteractiveAgentDefinition` 的字段；Runner 运行使用默认值。见[容量元数据](docs/model-settings.md#capacity-metadata)。
- 用 `RunConfig(microcompaction_policy=MicrocompactionPolicy(...))` 调整 microcompaction。默认在阈值的 `0.75` 处开始，目标降到 `0.60`，保留最近 3 轮，只处理超过 500 个字符的结果。
- 归档时，完整结果先写入工作区的 `.vv-agent/artifacts/`，之后模型看到的是一个 `<Tool Result Compact>` 标记，里面有 artifact 路径、`read_file` 读取提示和一小段摘录。模型无法调用 `read_file` 时不做归档，因为它读不回原结果。声明了 `result_retention=ToolResultRetention.PRESERVE` 的工具不会被归档。artifact 的字节数和 SHA-256 只保存在宿主可见的 `ToolArtifactRef` 里。
- 摘要使用 metadata 里的 `memory_summary_model`（可选配 `memory_summary_backend`），没有设置时使用本次运行的模型。摘要保留 artifact 和文件的路径及读取提示，但不会自己去读这些文件。
- 上下文大小按上一轮报告的 prompt token 数加上新的工具消息来算。没有用量数据时，用 `vv-llm` 计算 token，再不行就用本地估算，估算会考虑中日韩文字。
- 每次压缩都会发出 `memory_compact_started` 和 `memory_compact_completed` 事件。完成事件写明实际执行的最强一步：`none`、`micro`、`structural`、`summary` 或 `emergency`。

Session Memory 把同一会话里上一次运行的笔记带到下一次运行。它默认关闭，用 `RunConfig(session_memory_enabled=True)` 开启。笔记存在 `workspace/.memory/session/<session-or-task-scope>/session_memory.json`，有 `metadata.session_id` 时按会话隔离，否则按任务 ID 隔离。运行在编译时读取一次笔记，写进第一条系统消息；运行中提取的新笔记从下一次运行开始可见，恢复的轮次沿用已经固定下来的那份。提取笔记默认使用摘要模型，可以用 `session_memory_extraction_model` 另行指定。完整压缩不会清掉已存的笔记。子任务默认关闭 Session Memory，避免父子任务共用一个文件。

下面这些 metadata 参数可以在 `Agent.metadata` 或 `RunConfig.metadata` 里设置，用来调整细节：`model_context_window`、`reserved_output_tokens`、`autocompact_buffer_tokens`、`memory_keep_recent_messages`、`memory_summary_model`、`memory_summary_backend`、`include_memory_warning`、`session_memory_min_tokens`、`session_memory_max_tokens`、`session_memory_min_text_messages`、`session_memory_storage_dir`、`session_memory_extraction_model`、`session_memory_extraction_backend`、`tool_result_excerpt_head`、`tool_result_excerpt_tail` 和 `summary_event_limit`。

完整算法见[保留历史的压缩](docs/architecture.md#history-preserving-compaction)和[通过日志完成压缩](docs/session-kernel.md#compaction-through-the-log)。

## 整体结构

```text
Runner / InteractiveAgentClient / CLI / AppServer
  -> SessionDriver -> SessionStore (SQLite or PostgreSQL)
  -> session kernel (vv_agent.session.kernel.drive)
  -> logged model and tool operations, child sessions
  -> RunResult / RunEvent / tracing / App Server items
```

所有入口都走同一个会话内核（session kernel）。会话存储的事务负责会话的日志、inbox（排队的输入，比如回复和引导消息）、租约（某个 worker 对会话的驱动权）、子任务接纳，以及已完成操作的回执。普通运行使用内存 SQLite；持久的 SQLite 和 PostgreSQL 存储需要通过会话 API 主动启用。

vv-agent 负责可移植的运行时：提示词组装、模型调用、工具规划与执行、压缩、强类型事件、取消、审批和可回放的历史。你的产品负责自己的 UI、用户和工作区、存储、浏览器或 IM 集成，以及产品工具。两者通过 provider 对接，不要去改 vv-agent 的内部代码：

| 扩展点 | 作用 |
| --- | --- |
| `AppServerHost` | 为 App Server 构造 `Agent` 和 `RunConfig`。 |
| `ApprovalProvider` | 根据产品规则或 UI 决定工具审批。 |
| `ContextProvider` | 在每次运行编译前加入产品提示词片段。 |
| `vv_agent.memory.MemoryProvider` | 接入产品的记忆检索与保存，以及压缩事件。 |
| `vv_agent.tools.ToolExecutor` | 运行产品工具，自带 schema、审批、超时和错误处理。 |
| `SessionRunEventStore` | 回放强类型的 `RunEvent` 历史，包括父子运行。 |

公开名称从 `vv_agent` 导出。`AgentTask`、`Message`、`ToolCall` 这类底层运行时类型也在其中，但多数代码用不到。模块划分和不变量见[架构](docs/architecture.md)。

## 示例

`examples/` 里有 SDK 用法示例和少量底层集成示例，完整列表见 [`examples/README.md`](examples/README.md)。

```bash
uv run python examples/01_quick_start.py
uv run python examples/24_workspace_backends.py
```

## 开发

```bash
uv sync --dev
uv run pytest                              # 单元测试，不联网
uv run ruff check .                        # lint
uv run ty check                            # 类型检查

VV_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live   # 调用真实 LLM
```

live 测试读取这些环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VV_AGENT_LOCAL_SETTINGS` | `local_settings.py` | 配置文件路径 |
| `VV_AGENT_LIVE_BACKEND` | `moonshot` | LLM 后端 |
| `VV_AGENT_LIVE_MODEL` | `kimi-k3` | 模型名称 |

测试选择和发布流程见[开发指南](docs/development.md)，其他内容见[文档索引](docs/index.md)。
