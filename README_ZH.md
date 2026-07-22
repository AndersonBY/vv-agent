# vv-agent

[English](README.md)

从 VectorVein 生产环境抽象出的轻量 Agent 框架。基于 cycle 的执行模型，支持可插拔 LLM 后端、工具分发、上下文压缩和分布式调度。

## 架构

```
Agent / RunConfig / ModelSettings
└── Runner
    └── AgentRuntime
        ├── CycleRunner          # 单轮 LLM 调用：上下文 -> 补全 -> 工具调用
        ├── ToolCallRunner       # 工具分发与 directive 收敛
        ├── RuntimeHookManager   # before/after 钩子
        ├── MemoryManager        # 上下文超阈值时自动压缩历史
        └── ExecutionBackend     # inline、thread 或 Celery 调度
```

公开 SDK 入口从 `vv_agent` 顶层导出：`Agent`、`Runner`、`RunConfig`、
`RunHandle`、`ModelSettings`、`function_tool`、`Session`、强类型 `RunEvent`、
`ApprovalProvider`、`ContextProvider`、`RunEventStore`，以及面向桌面 runtime 集成的
interactive session API。位于包模块中的扩展点包括 `vv_agent.memory.MemoryProvider`
和 `vv_agent.tools.ToolExecutor`。底层 runtime 实现细节包括 `AgentTask`、
`AgentResult`、`Message`、`CycleRecord` 和 `ToolCall`。

任务完成由工具显式触发：agent 调用 `task_finish` 或 `ask_user` 来标记终态，不做"最后一条消息即答案"的隐式推断。

## 配置

```bash
cp local_settings.example.py local_settings.py
# 在 local_settings.py 中填入 API Key 和 endpoint
```

```bash
uv sync --dev
uv run pytest
```

## 快速开始

### CLI

```bash
uv run vv-agent --prompt "概述一下这个框架" --backend moonshot --model kimi-k3

# 带每轮日志
uv run vv-agent --prompt "概述一下这个框架" --backend moonshot --model kimi-k3 --verbose
```

参数：`--settings-file`、`--backend`、`--model`、`--verbose`。

### 代码调用 SDK

```python
from vv_agent import Agent, RunConfig, Runner, function_tool

@function_tool
def read_order(order_id: str) -> str:
    """读取订单信息。"""
    return "订单详情"

agent = Agent(
    name="ops",
    instructions="先查证，再回答。",
    model="kimi-k3",
    tools=[read_order],
)

result = Runner.run_sync(agent, "分析订单 123", run_config=RunConfig(
    default_backend="moonshot",
))
print(result.status, result.final_output)
```

多次运行需要复用同一组默认值时，使用 configured Runner，避免重复传入
`RunConfig`：

```python
runner = Runner.configured(RunConfig(
    model_provider=provider,
    model="kimi-k3",
    workspace="./workspace",
))
result = runner.run_sync(agent, "分析订单 123")
```

Provider 优先级为 per-run、Runner；Model 优先级为 per-run、Agent、Runner、
当前 Provider 默认模型；ModelSettings 则按 Provider、Runner、Agent、per-run
逐层合并，后面的层覆盖前面的字段。

`Agent.output_type` 可以把 JSON 终态输出转换为 `dict`、`list`、dataclass
或 Pydantic 风格 model。`@function_tool` 包装的函数也可以把 `ToolContext`
作为第一个参数；运行时会在调用时传入它，但不会把它暴露到工具 JSON schema。

### 流式输出与 Session

`RunConfig.workspace` 控制本次运行的工作区。`RunConfig.session` 可传入
`MemorySession`、`SQLiteSession` 或 `RedisSession`，用于跨多次运行保留消息历史。

```python
from vv_agent import Agent, MemorySession, RunConfig, Runner

agent = Agent(name="assistant", instructions="记住上下文。", model="kimi-k3")
session = MemorySession("thread-001")
config = RunConfig(
    default_backend="moonshot",
    workspace="./workspace/thread-001",
    session=session,
)

Runner.run_sync(agent, "先分析项目", run_config=config)
for event in Runner.stream_sync(agent, "继续刚才的话题并汇报进度", run_config=config):
    if event.type == "assistant_delta":
        print(event.delta, end="")
```

宿主需要活跃运行句柄而不是阻塞等待结果时，使用 `Runner.start()`。
`RunHandle.events()` 会产生与 `Runner.stream_sync()` 相同的强类型 `RunEvent`
流，`RunHandle.result()` 等待最终 `RunResult`，`RunHandle.cancel()` 取消运行，
`RunHandle.approve()` 处理待审批请求。当 handle 挂接到 `AgentSession` 时，
`RunHandle.steer()` 会为当前运行排入 steering 上下文，`RunHandle.follow_up()`
会排入下一个 session turn。普通一次性 `Runner.start()` handle 不拥有 session
队列，因此这些方法需要交互式 session controller。

`RunConfig.event_store` 可以持久化每个强类型事件。`JsonlRunEventStore` 会保存事件
字典，并按 `run_id` 回放事件，包括 `parent_run_id` 指向该 run 的子 run。公开的
runtime 事件入口只有强类型 `RunEvent`；任务无关的内部观测统一使用
`DiagnosticEvent`。

一个参数已规范化的工具调用依次发出 `tool_call_planned`、可选审批事件、在可能产生
副作用前紧邻发出的 `tool_call_started`，以及结果形成后的
`tool_call_completed`。参数解析失败不会发出这些事件；策略拒绝、审批短路和未知工具
只发出 planned 与 completed，不发 started。completed 事件包含 `directive`、可空的
`error_code`、`execution_started` 和可空的单调时钟 `duration_ms`。取消或进程退出可能
留下没有 completed 的 started 事件，因此恢复时仍以 checkpoint v2 operation journal
为准。

需要直接控制 cycle loop 的后端集成仍可使用底层 `AgentRuntime` API。

Redis 支持可通过 `uv sync --extra redis` 安装，也可以在构造 `RedisSession`
时注入 Redis 兼容 client。

### App Server

桌面应用、worker、IDE 或其他宿主进程如果需要通过稳定协议驱动 `vv-agent`，而不是
直接嵌入 Python SDK，可以使用 App Server。它通过 stdio 传输 JSONL，暴露
Thread / Turn / Item 生命周期事件，把工具审批路由成 server-to-client request，
支持 `thread/read` 与 `thread/resume` 回放，并可导出 typed JSON Schema 与自包含的
TypeScript 客户端类型。

```bash
uv run vv-agent app-server --listen stdio --settings local_settings.py --backend moonshot --model kimi-k3
uv run vv-agent app-server schema --out ./app-server-schema
uv run vv-agent app-server generate-ts --out ./app-server-schema/typescript
uv run vv-agent debug app-server send-message "hello"
```

宿主产品实现 `AppServerHost`，把产品 profile、工作区上下文、工具、审批 UI、memory
和模型设置映射成框架层的 `Agent` 与 `RunConfig`。App Server 是 runtime 边界，
不引入产品 UI、账号、计费、浏览器或 IM 模块。

协议细节见 [docs/app-server.md](docs/app-server.md)，当前宿主边界与上线检查见
[docs/app-server-host-integration.md](docs/app-server-host-integration.md)。

### Interactive Session

普通一次性运行、流式运行，以及由 `RunConfig.session` 管理历史的会话，优先使用
`Runner`。宿主应用需要稳定 `session_id`、运行时监听、运行中 steering、follow-up、
取消和共享工具状态时，使用 `InteractiveAgentClient`。session 运行期间，
`session.active_run_handle` 会暴露统一的 `RunHandle` 控制面，可用于审批、取消、
steering 和 follow-up。

需要持久化历史时，可通过 `AgentSessionOptions.session`（或
`create_session(session=...)`）注入已有的 `MemorySession`、`SQLiteSession` 或
`RedisSession`。facade 会在创建时恢复完整历史，后续每轮由 `Runner` 写回同一个
Session；不要再把同一份历史作为 initial messages 重复传入。同时提供
`session_id` 时，它必须与 backing Session 的 id 一致。

```python
from pathlib import Path

from vv_agent import (
    AgentSessionOptions,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    SQLiteSession,
)
from vv_agent.runtime.backends import ThreadBackend

client = InteractiveAgentClient(
    options=AgentSessionOptions(
        settings_file=Path("local_settings.py"),
        default_backend="moonshot",
        workspace=Path("./workspace/thread-001"),
        execution_backend=ThreadBackend(max_workers=4),
        session=SQLiteSession("thread-001", db_path=Path("./sessions.sqlite3")),
    )
)

session = client.create_session(
    session_id="thread-001",
    agent=InteractiveAgentDefinition(
        description="在用户工作区内操作并汇报进度。",
        model="kimi-k3",
        no_tool_policy="finish",
    ),
)
unsubscribe = session.subscribe(lambda event, payload: print(event, payload))
try:
    run = session.prompt("检查工作区")
    print(run.result.status, run.result.final_answer)
finally:
    unsubscribe()
```

Interactive session 的公开入口是 `InteractiveAgentClient`、
`AgentSessionOptions` 与 `AgentSession`。

### Agent as Tool、Handoff 与工具策略

子 Agent 的结果要回到父 Agent 并让父 Agent 继续时，使用 `agent.as_tool()`。
需要把控制权转交给目标 Agent，并用目标 Agent 的输出结束本次运行时，使用
`handoff()`。

```python
from vv_agent import Agent, RunConfig, Runner, ToolPolicy, handoff
from vv_agent.constants import TASK_FINISH_TOOL_NAME

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
    run_config=RunConfig(
        default_backend="moonshot",
        max_handoffs=4,
        tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, "transfer_to_writer"]),
    ),
)
```

Handoff 是 Runner 外层的控制权转移，不是 agent-as-tool 调用。目标 Agent 会重新解析
自己的 model 和 model settings，同时沿用当前 session、cancellation token，并继承源
Agent 已修改的 shared state。`max_handoffs` 默认值为 `10`，它独立于
`max_cycles` 限制控制转移次数；审批恢复也保持相同语义。

`RunConfig.budget_limits` 可以分别限制总 token、未缓存输入 token、工具总调用数、
指定工具调用数、活跃运行时间和宿主计量成本。预算只控制资源，不判断任务内容或答案
质量。可通过 `result.budget_usage` 和 `result.budget_exhaustion` 查看结果；预算耗尽是
类型明确的失败终态，不会伪装成任务成功。详见 [运行预算](docs/run-budgets.md)。

工具可通过 `@function_tool(needs_approval=True)` 请求审批。默认情况下，
运行会在工具函数真正执行前进入 `WAIT_USER`，并发出
`ApprovalRequestedEvent`。可信运行可用 `ToolPolicy(approval="never")`
关闭这道审批门。四种 policy mode 为 `default`、`always`、`never` 和
`on_request`：`default` 继续继承下一层配置，显式 `on_request` 则按每个工具的
静态或动态审批声明决定是否请求审批。

自定义工具还可以附加可选的、仅宿主可见的能力声明：

```python
from vv_agent import (
    ToolIdempotency,
    ToolMetadata,
    ToolPolicy,
    ToolSideEffect,
    function_tool,
)

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
    return {"ticket_id": "TCK-1001", "title": title}

policy = ToolPolicy(
    denied_side_effects=[ToolSideEffect.EXECUTE],
    denied_capability_tags=["filesystem.delete"],
    deny_terminal_tools=True,
    denied_cost_dimensions=["gpu.second"],
)
```

`side_effect` 只是一个粗粒度声明，不存在自动推导的层级；`capability_tags` 与
`cost_dimensions` 都是精确匹配的不透明标签，cost dimension 也不是用量或价格。
`terminal=True` 只表示工具可能返回 `finish` 或 `wait_user`，不会自行结束运行。
四个新策略字段会在 Agent、configured Runner、per-run 和委托子 Agent 各层累积拒绝；
命中时返回 `tool_not_allowed`。它们不能授予能力，也不能移除已有的名称、参数、审批、
预算或 runtime 限制。

Typed metadata 与通用的 `FunctionTool.metadata` 相互独立，也不会进入模型可见的函数
schema。`ToolMetadata.idempotency` 是执行、事件与 checkpoint 唯一使用的幂等性声明。

### Guardrails 与 Trace

Input guardrail 会在调用模型前执行。Output guardrail 会在得到最终输出后执行。
Trace processor 会收到轻量级 run span 和 tool span。

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
    run_config=RunConfig(default_backend="moonshot", tracing={"workflow_name": "summary"}),
)
```

### Windows Shell 运行时配置

`bash` 运行时默认配置属于**启动/会话配置**，不是工具参数。

- Run 级默认：通过 `RunConfig.metadata` 传入 `bash_shell`、`windows_shell_priority`、`bash_env`
- Agent 级默认：通过 `Agent.metadata` 传入同名字段
- Windows 推荐优先级：`["git-bash", "powershell", "cmd"]`
- 在 Windows 上，`bash` 工具启动的子进程会默认注入 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8`；若父进程环境或 `bash_env` 已显式设置，则以显式值为准。
- 在 Windows 上，`bash` 工具启动子进程时还会附带隐藏控制台窗口的启动参数，方便 GUI 宿主调用 `bash` / `powershell` 时不再闪出额外终端窗口。
- `Runner.run_sync(...)` 与 `Runner.stream_sync(...)` 都会继承编译后的 shell 元数据。
- `bash` 工具 schema 的 description 会注入运行时 shell 提示（解析后的 shell 类型与调用前缀），模型在调用前即可知道应使用哪种命令风格。
- 该运行时 shell 提示会在单个 task/session-run 内固化，确保跨 cycles 的 tool schema 文本稳定，保护 LLM prompt cache 命中率。
- SDK/CLI 自动生成的任务现在还会把 `system_prompt_sections` 元数据挂到 system message 上，Anthropic prompt cache 可以据此把稳定前缀长期缓存，同时把当前时间、Session Memory 这类易变段落视为 volatile。

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
        default_backend="moonshot",
        metadata={
            "windows_shell_priority": ["git-bash", "powershell", "cmd"],
            "bash_env": {"PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple"},
        },
    ),
)
```

## 执行后端

cycle 循环由可插拔的 `ExecutionBackend` 调度。

| 后端 | 场景 |
|------|------|
| `InlineBackend` | 默认。同步，单进程。 |
| `ThreadBackend` | 线程池。`submit()` 返回 `Future`，非阻塞。 |
| `CeleryBackend` | 分布式。每轮 cycle 作为独立 Celery task 分发到 worker。 |

### CeleryBackend

每轮 cycle 都作为 Celery task 执行。Worker 从必填的 `RuntimeRecipe` 重建
`AgentRuntime`，并解析其中声明的共享 `CheckpointStore` 能力。

```python
from vv_agent import CheckpointConfig, RunConfig
from vv_agent.runtime.backends.celery import CeleryBackend, RuntimeRecipe, register_cycle_task
from vv_agent.runtime.backends.distributed import (
    CapabilityRef,
    DistributedCapabilities,
    DistributedCapabilityRegistry,
)
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore

checkpoint_ref = CapabilityRef("checkpoint.production", "1")
checkpoint_store = SqliteCheckpointStore(".vv-agent-state/checkpoints.db")
worker_capabilities = DistributedCapabilityRegistry()
worker_capabilities.register("checkpoint_store", checkpoint_ref, checkpoint_store)
register_cycle_task(celery_app, capability_registry=worker_capabilities)

recipe = RuntimeRecipe(
    settings_file="local_settings.py",
    backend="moonshot",
    model="kimi-k3",
    workspace="./workspace",
    capabilities=DistributedCapabilities(checkpoint_store_ref=checkpoint_ref),
)
backend = CeleryBackend(celery_app=celery_app, runtime_recipe=recipe)
run_config = RunConfig(
    execution_backend=backend,
    checkpoint_config=CheckpointConfig(
        key="tenant-7/task-42",
        store=checkpoint_store,
    ),
)
```

安装 celery 依赖：`uv sync --extra celery`。

### 取消与流式输出

```python
from vv_agent.events import AssistantDeltaEvent, RunEvent
from vv_agent.runtime import CancellationToken, ExecutionContext

# 从另一个线程取消
token = CancellationToken()
ctx = ExecutionContext(cancellation_token=token)
result = runtime.run(task, ctx=ctx)

def on_event(event: RunEvent) -> None:
    if isinstance(event, AssistantDeltaEvent):
        print(event.delta, end="")


# 流式输出 LLM 事件，包括 assistant delta 和工具参数进度
ctx = ExecutionContext(event_handler=on_event)
result = runtime.run(task, ctx=ctx)
```

### Runtime 日志载荷

`tool_result` 事件现在会把完整工具文本放在 `content`，把结构化工具载荷放在 `metadata`，且不会隐式截断 `content`。
同时保留 `content_preview`、`assistant_preview` 供前端轻量展示。

如果你希望限制预览长度（例如节省传输），可以显式配置：

```python
from vv_agent import RunConfig

config = RunConfig(
    log_preview_chars=220,  # 可选：显式开启预览截断
)
```

## 工作区存储后端

工作区文件 I/O 通过可插拔的 `WorkspaceBackend` 协议分发。所有内建文件工具（`read_file`、`write_file`、`find_files` 等）均经过此抽象层。

- 本地 `find_files` 优先使用 `rg` 加速遍历，必要时自动回退到 Python。
- `search_files` 在本地工作区同样优先使用 `rg`，默认采用 smart case：
  小写 pattern 默认不区分大小写；pattern 中包含大写字母时默认区分大小写。
- `search_files` 的 `ToolExecutionResult.content` 只保留给模型的文本结果，结构化的命中/计数数据放在 `ToolExecutionResult.metadata`。
- `search_files` / `find_files` 默认会跳过隐藏目录和常见依赖/缓存根目录，除非显式开启包含选项。

| 后端 | 场景 |
|------|------|
| `LocalWorkspaceBackend` | 默认。读写本地目录，带路径逃逸保护。 |
| `MemoryWorkspaceBackend` | 纯内存 dict 存储。适合测试和沙箱运行。 |
| `S3WorkspaceBackend` | S3 兼容对象存储（AWS S3、阿里云 OSS、MinIO、Cloudflare R2）。 |

```python
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

# 显式指定本地后端
runtime = AgentRuntime(
    llm_client=llm,
    tool_registry=registry,
    workspace_backend=LocalWorkspaceBackend(Path("./workspace")),
)

# 内存后端，适合测试
runtime = AgentRuntime(
    llm_client=llm,
    tool_registry=registry,
    workspace_backend=MemoryWorkspaceBackend(),
)
```

### S3WorkspaceBackend

安装可选 S3 依赖：`uv pip install 'vv-agent[s3]'`。

```python
from vv_agent.workspace import S3WorkspaceBackend

backend = S3WorkspaceBackend(
    bucket="my-bucket",
    prefix="agent-workspace",
    endpoint_url="https://oss-cn-hangzhou.aliyuncs.com",  # AWS 留 None
    aws_access_key_id="...",
    aws_secret_access_key="...",
    addressing_style="virtual",  # MinIO 用 "path"
)
```

### 自定义后端

实现 `WorkspaceBackend` 协议（8 个方法）即可接入任意存储：

```python
from vv_agent.workspace import WorkspaceBackend

class MyBackend:
    def list_files(self, base: str, glob: str) -> list[str]: ...
    def read_text(self, path: str) -> str: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_text(self, path: str, content: str, *, append: bool = False) -> int: ...
    def file_info(self, path: str) -> FileInfo | None: ...
    def exists(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...
    def mkdir(self, path: str) -> None: ...
```

## 模块一览

| 模块 | 说明 |
|------|------|
| `vv_agent.runtime.AgentRuntime` | 顶层状态机（completed / wait_user / max_cycles / failed） |
| `vv_agent.runtime.CycleRunner` | 单轮 LLM 调用与 cycle 记录构建 |
| `vv_agent.runtime.ToolCallRunner` | 工具执行与 directive 收敛 |
| `vv_agent.runtime.RuntimeHookManager` | Hook 分发（before/after LLM、工具调用、上下文压缩） |
| `vv_agent.runtime.CheckpointStore` | Checkpoint 持久化协议（`InMemoryCheckpointStore` / `SqliteCheckpointStore` / `RedisCheckpointStore`） |
| `vv_agent.memory.MemoryManager` | 历史超阈值时自动压缩 |
| `vv_agent.workspace` | 可插拔文件存储：`LocalWorkspaceBackend`、`MemoryWorkspaceBackend`、`S3WorkspaceBackend` |
| `vv_agent.tools` | 内建工具，以及 `function_tool`、`FunctionTool`、结构化工具输出 |
| `vv_agent` | 公开 SDK：`Agent`、`Runner`、`RunConfig`、`ModelSettings`、tools、sessions、typed events |
| `vv_agent.app_server` | JSONL App Server 协议、transport、thread state、回放、审批回调、schema 导出和宿主 provider 边界 |
| `vv_agent.skills` | Agent Skills 支持（`SKILL.md` 解析、校验、统一 normalize、带预算管理的 prompt 渲染、`activate_skill` 工具） |
| `vv_agent.llm.VvLlmClient` | 统一 LLM 接口，基于 `vv-llm`（端点轮询、重试、流式） |
| `vv_agent.config` | 从 `local_settings.py` 解析模型/端点/Key |

## Runtime 边界

`vv-agent` 负责可移植的 agent runtime：prompt 组装、模型调用、工具规划、工具执行、
memory 压缩、强类型事件、取消、审批中断，以及可回放的运行历史。宿主产品负责产品
UI、用户和工作区解析、产品存储、浏览器或 IM 集成，以及暴露给模型的产品工具。

宿主产品应实现 provider，而不是 patch `vv-agent` 内部：

- `AppServerHost` 在宿主使用 JSONL 进程集成时，把产品 profile、工作区、工具、审批
  UI、memory、context 和模型设置映射成 App Server 使用的 `Agent` 与 `RunConfig`。
- `ApprovalProvider` 根据产品 UI 或规则决定工具调用是否需要审批，并返回允许、拒绝、
  本 session 允许或超时决策。
- `ContextProvider` 在每次 run 编译前注入产品 prompt 片段，例如 profile、workspace、
  policy 或功能上下文。
- `vv_agent.memory.MemoryProvider` 把产品 memory 存储接入 memory search/save hook
  和压缩生命周期事件。
- `vv_agent.tools.ToolExecutor` 暴露产品工具的 schema、审批、超时、错误和执行行为。
  普通 Python 函数使用 `FunctionTool` 或 `@function_tool`；自定义 executor 由
  `ToolOrchestrator` 路由。
- `RunEventStore` 持久化强类型 `RunEvent` 历史，让应用视图可以回放已完成 run 和父子
  run 图。

这个边界让 `Agent`、`Runner`、`RunConfig`、`RunHandle` 和 `RunEvent` 保持稳定，
同时允许每个宿主把账号模型、工作区模型、存储后端和 UI 流程留在框架之外。

## Memory 压缩与配置

`MemoryManager` 按 token 计算上下文大小，并在超过解析后的自动压缩阈值时触发压缩。

- 任务级参数：
  - `memory_compact_threshold`（默认 `250000`，作为完整压缩流程的配置上限）
  - `memory_threshold_percentage`（内存预警百分比，默认 `90`）
- 编译映射：
  - `AgentCompiler` 将稳定的 Agent / Run 元数据转发到 `AgentTask`。
  - 解析出的模型容量分别记录为 `model_context_window` 和
    `model_max_output_tokens`；输出 capability 不会自动复制到
    `reserved_output_tokens`。
  - 既有 durable task / checkpoint 在解码或恢复时保留原有阈值和元数据，
    不按新默认值回写。
- token 预算模型：
  - Context 优先级：显式 `model_context_window`、解析出的模型 capability、
    最后使用动态规划上下文。默认值为 `250000 + 16000 + 13000 = 279000`。
  - 输出预留优先级：有效的 `ModelSettings.max_tokens`、显式
    `reserved_output_tokens`、最后使用框架默认 `16000`。
  - 只有框架默认预留可以被更小的 `model_max_output_tokens` 向下裁剪；
    capability 不会覆盖显式请求限制或宿主预留。
  - `derived_prompt_capacity = max(model_context_window - reserved_output_tokens - autocompact_buffer_tokens, 0)`
  - `autocompact_threshold = min(memory_compact_threshold, derived_prompt_capacity)`；
    配置阈值为零时使用 derived capacity，已知 derived capacity 为零时保持零。
  - 默认 autocompact buffer 为 `13000`，默认 microcompact trigger 是有效完整
    压缩阈值的 75%。
- 有效长度策略（与 backend 对齐）：
  - 如果有上一轮 token 用量：
    - `effective_length = previous_prompt_tokens + token_count(recent_tool_messages)`
  - 否则兜底：
    - `vv_llm.chat_clients.utils.get_message_token_counts(...)`
    - 如果 tokenizer 不可用，再用本地 CJK 感知估算
- 压缩流程：
  1. 预防性 microcompact：当使用量超过 `microcompact_trigger_ratio` 时，先清理旧的大 tool result
  2. Session Memory 提取：在全量摘要前抽取并持久化关键事实，避免后续压缩丢失
  3. 结构化清理（陈旧 tool_calls、孤儿 tool 消息、assistant 无工具消息折叠、旧 tool 结果 artifact 化）
  4. 若仍超阈值，再生成增强版压缩记忆总结，显式保留用户原始消息、文件操作、当前工作状态和错误修复信息
  5. 如果 provider 仍返回 prompt-too-long，再执行一次强制压缩，之后逐步加大 emergency tail-dropping 重试
  6. 全量压缩后，会在预算内自动恢复相关工作区文件内容到 `<Post-Compaction File Context>`
- 压缩事件：
  - 新的 `memory_compact_started` producer 会携带 typed trigger 和完整的容量解析快照。
  - 新的 `memory_compact_completed` producer 会携带实际执行的最强 mode
    （`none`、`micro`、`structural`、`summary` 或 `emergency`）以及基于消息内容比较的
    `changed`。
  - 当前事件必须包含完整的强类型容量与结果字段；缺失或未知字段会被拒绝。
- Session Memory 行为：
  - 默认存放在 `workspace/.memory/session/<session-or-task-scope>/session_memory.json`
  - 若 `metadata.session_id` 存在，则按当前 session 隔离；否则按当前 `task_id` 隔离
  - 新 session / 新 task 不会继承上一轮 session / task 的 Session Memory
  - 每个 cycle 都会以 `<Session Memory>` 的形式注入到第一条 system message
  - 提取阶段复用现有的 memory summary backend/model 选择逻辑
  - 全量压缩后只重置 transcript 跟踪索引，不清空已持久化记忆
  - 子任务默认关闭 Session Memory，避免父子任务共享同一记忆文件

### Runtime metadata 参数

通过 `Agent.metadata` 或 `RunConfig.metadata` 传入；compiler 会转发到 `AgentTask.metadata`：

- `memory_keep_recent_messages`
- `model_context_window`
- `model_max_output_tokens`（解析出的模型 capability，不代表隐式请求限制）
- `reserved_output_tokens`
- `autocompact_buffer_tokens`
- `microcompact_trigger_ratio`
- `microcompact_keep_recent_cycles`
- `microcompact_min_result_length`
- `microcompact_compactable_tools`
- `include_memory_warning`
- `session_memory_enabled`
- `session_memory_min_tokens`
- `session_memory_max_tokens`
- `session_memory_min_text_messages`
- `session_memory_storage_dir`
- `tool_result_compact_threshold`
- `tool_result_keep_last`
- `tool_result_excerpt_head`
- `tool_result_excerpt_tail`
- `tool_calls_keep_last`
- `assistant_no_tool_keep_last`
- `tool_result_artifact_dir`
- `summary_event_limit`

### 记忆总结模型选择优先级

优先级严格如下：

1. `AgentTask.metadata.memory_summary_model`，可选
   `memory_summary_backend`。
2. 通过本次运行的 `ModelProvider` 使用当前任务模型。

## 内建工具

`find_files`、`file_info`、`read_file`、`write_file`、`edit_file`、`search_files`、`compress_memory`、`todo_write`、`task_finish`、`ask_user`、`bash`、`read_image`、`create_sub_task`、`sub_task_status`。

通过 `ToolRegistry.register()` 注册自定义工具。

`bash` 工具支持两种后台路径：

- 显式后台：传 `run_in_background=true`，立即返回 `session_id`，后续用 `check_background_command` 轮询。
- 超时转后台：前台命令如果达到 `timeout` 仍未结束，不会直接中断报错，而是自动转入后台 session，并返回 `session_id` 与提示信息；后续同样用 `check_background_command` 查询。
- 主动终态通知：后台命令完成、失败或超时后，会触发 session 级事件；如果当前 session 正在运行，系统会自动向 Agent 注入一条 steering 提醒。

## 子 Agent

使用 `Agent.as_tool()` 时，子 Agent 的结果会作为工具结果回到父 Agent，父 Agent 继续控制流程。使用 `handoff()` 时，控制权转交给目标 Agent，并由目标 Agent 的输出结束本次运行。模型需要显式管理后台或并行任务时，使用 `create_sub_task` 与 `sub_task_status`。

每个子任务都会创建真实 `AgentSession`（默认 `session_id == task_id`）。子任务的 `RunEvent` 会原样保留 run、trace、parent、task 与 session 标识，宿主应用无需经过非强类型转换即可独立订阅、持久化与回放。

`create_sub_task` 的批量模式现在会通过 runtime 执行后端的 `parallel_map` 分发有效子任务；当后端支持并行时，同步批量任务会并发执行。

使用 `sub_task_status` 可以查询 runtime 子任务状态、查看轻量级进度快照（`detail_level=snapshot`），或向运行中/已完成的子任务追加消息。

已完成的子任务在续传前会先清洗保存下来的会话 transcript：空 assistant、只有 thinking 的 assistant、孤儿 tool result、以及未完成的尾部 tool call 都会被移除，避免把无效历史再次注入下一轮上下文。

每个子任务的 runtime metadata 现在会写入 `task_id`、`session_id` 和 `browser_scope_key`，确保浏览器这类会话级工具在并行子任务间保持隔离。

宿主应用可以通过 `vv_agent.runtime.engine.steer_sub_agent_session(session_id=..., prompt=...)` 向正在运行的子任务定向插话。

子 Agent 继承父运行显式提供的同一个 `ModelProvider`，并独立解析自己的模型；子运行不会重新读取 settings 路径或构造 backend fallback。

## 示例

`examples/` 下保留公开 SDK cookbook 和低层 runtime 集成示例。完整列表见 [`examples/README.md`](examples/README.md)。

```bash
uv run python examples/01_quick_start.py
uv run python examples/24_workspace_backends.py
```

## 测试

```bash
uv run pytest                              # 单元测试（无网络）
uv run ruff check .                        # lint
uv run ty check                            # 类型检查

VV_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live   # 集成测试（需要真实 LLM）
```

集成测试环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VV_AGENT_LOCAL_SETTINGS` | `local_settings.py` | 配置文件路径 |
| `VV_AGENT_LIVE_BACKEND` | `moonshot` | LLM 后端 |
| `VV_AGENT_LIVE_MODEL` | `kimi-k3` | 模型名称 |
