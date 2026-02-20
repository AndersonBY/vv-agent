# v-agent

[English](README.md)

从 VectorVein 生产环境抽象出的轻量 Agent 框架。基于 cycle 的执行模型，支持可插拔 LLM 后端、工具分发、上下文压缩和分布式调度。

## 架构

```
AgentRuntime
├── CycleRunner          # 单轮 LLM 调用：上下文 -> 补全 -> 工具调用
├── ToolCallRunner       # 工具分发，directive 收敛（finish/wait_user/continue）
├── RuntimeHookManager   # before/after 钩子：LLM、工具调用、上下文压缩
├── MemoryManager        # 上下文超阈值时自动压缩历史
└── ExecutionBackend     # cycle 循环调度
    ├── InlineBackend    # 同步执行（默认）
    ├── ThreadBackend    # 线程池 + Future
    └── CeleryBackend    # 分布式，每轮 cycle 作为独立 Celery task
```

核心类型定义在 `v_agent.types`：`AgentTask`、`AgentResult`、`Message`、`CycleRecord`、`ToolCall`。

任务完成由工具显式触发：agent 调用 `_task_finish` 或 `_ask_user` 来标记终态，不做"最后一条消息即答案"的隐式推断。

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
uv run v-agent --prompt "概述一下这个框架" --backend moonshot --model kimi-k2.5

# 带每轮日志
uv run v-agent --prompt "概述一下这个框架" --backend moonshot --model kimi-k2.5 --verbose
```

参数：`--settings-file`、`--backend`、`--model`、`--verbose`。

### 代码调用

```python
from v_agent.config import build_openai_llm_from_local_settings
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask

llm, resolved = build_openai_llm_from_local_settings("local_settings.py", backend="moonshot", model="kimi-k2.5")
runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry())

result = runtime.run(AgentTask(
    task_id="demo",
    model=resolved.model_id,
    system_prompt="你是一个有用的助手。",
    user_prompt="1+1 等于几？",
))
print(result.status, result.final_answer)
```

### SDK

```python
from v_agent.sdk import AgentSDKClient, AgentSDKOptions

client = AgentSDKClient(options=AgentSDKOptions(
    settings_file="local_settings.py",
    default_backend="moonshot",
    default_model="kimi-k2.5",
))
result = client.run("用一句话解释 Python 的 GIL。")
print(result.final_answer)
```

## 执行后端

cycle 循环由可插拔的 `ExecutionBackend` 调度。

| 后端 | 场景 |
|------|------|
| `InlineBackend` | 默认。同步，单进程。 |
| `ThreadBackend` | 线程池。`submit()` 返回 `Future`，非阻塞。 |
| `CeleryBackend` | 分布式。每轮 cycle 作为独立 Celery task 分发到 worker。 |

### CeleryBackend

两种模式：

- **Inline 回退**（不传 `RuntimeRecipe`）：cycle 在本地执行，行为与 `InlineBackend` 一致。
- **分布式**（传入 `RuntimeRecipe`）：每轮 cycle 是一个 Celery task。Worker 从 recipe 重建 `AgentRuntime`，从共享 `StateStore`（SQLite 或 Redis）加载状态。

```python
from v_agent.runtime.backends.celery import CeleryBackend, RuntimeRecipe, register_cycle_task

register_cycle_task(celery_app)

recipe = RuntimeRecipe(
    settings_file="local_settings.py",
    backend="moonshot",
    model="kimi-k2.5",
    workspace="./workspace",
)
backend = CeleryBackend(celery_app=app, state_store=store, runtime_recipe=recipe)
runtime = AgentRuntime(llm_client=llm, tool_registry=registry, execution_backend=backend)
```

安装 celery 依赖：`uv sync --extra celery`。

### 取消与流式输出

```python
from v_agent.runtime import CancellationToken, ExecutionContext

# 从另一个线程取消
token = CancellationToken()
ctx = ExecutionContext(cancellation_token=token)
result = runtime.run(task, ctx=ctx)

# 逐 token 流式输出
ctx = ExecutionContext(stream_callback=lambda text: print(text, end=""))
result = runtime.run(task, ctx=ctx)
```

## 工作区存储后端

工作区文件 I/O 通过可插拔的 `WorkspaceBackend` 协议分发。所有内建文件工具（`_read_file`、`_write_file`、`_list_files` 等）均经过此抽象层。

| 后端 | 场景 |
|------|------|
| `LocalWorkspaceBackend` | 默认。读写本地目录，带路径逃逸保护。 |
| `MemoryWorkspaceBackend` | 纯内存 dict 存储。适合测试和沙箱运行。 |
| `S3WorkspaceBackend` | S3 兼容对象存储（AWS S3、阿里云 OSS、MinIO、Cloudflare R2）。 |

```python
from v_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

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

安装可选 S3 依赖：`uv pip install 'v-agent[s3]'`。

```python
from v_agent.workspace import S3WorkspaceBackend

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
from v_agent.workspace import WorkspaceBackend

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
| `v_agent.runtime.AgentRuntime` | 顶层状态机（completed / wait_user / max_cycles / failed） |
| `v_agent.runtime.CycleRunner` | 单轮 LLM 调用与 cycle 记录构建 |
| `v_agent.runtime.ToolCallRunner` | 工具执行与 directive 收敛 |
| `v_agent.runtime.RuntimeHookManager` | Hook 分发（before/after LLM、工具调用、上下文压缩） |
| `v_agent.runtime.StateStore` | Checkpoint 持久化协议（`InMemoryStateStore` / `SqliteStateStore` / `RedisStateStore`） |
| `v_agent.memory.MemoryManager` | 历史超阈值时自动压缩 |
| `v_agent.workspace` | 可插拔文件存储：`LocalWorkspaceBackend`、`MemoryWorkspaceBackend`、`S3WorkspaceBackend` |
| `v_agent.tools` | 内建工具：workspace I/O、todo、bash、image、sub-agent、skills |
| `v_agent.sdk` | 高层 SDK：`AgentSDKClient`、`AgentSession`、`AgentResourceLoader` |
| `v_agent.skills` | Agent Skills 支持（`SKILL.md` 解析、prompt 注入、激活） |
| `v_agent.llm.VVLlmClient` | 统一 LLM 接口，基于 `vv-llm`（端点轮询、重试、流式） |
| `v_agent.config` | 从 `local_settings.py` 解析模型/端点/Key |

## 内建工具

`_list_files`、`_file_info`、`_read_file`、`_write_file`、`_file_str_replace`、`_workspace_grep`、`_compress_memory`、`_todo_write`、`_task_finish`、`_ask_user`、`_bash`、`_read_image`、`_create_sub_task`、`_batch_sub_tasks`。

通过 `ToolRegistry.register()` 注册自定义工具。

## 子 Agent

在 `AgentTask.sub_agents` 上配置命名子 Agent。父 Agent 通过 `_create_sub_task` / `_batch_sub_tasks` 委派任务。每个子 Agent 有独立的 runtime、模型和工具集。

子 Agent 使用与父任务不同的模型时，runtime 需要提供 `settings_file` 和 `default_backend` 来解析 LLM 客户端。

## 示例

`examples/` 下有 24 个编号示例。完整列表见 [`examples/README.md`](examples/README.md)。

```bash
uv run python examples/01_quick_start.py
uv run python examples/24_workspace_backends.py
```

## 测试

```bash
uv run pytest                              # 单元测试（无网络）
uv run ruff check .                        # lint
uv run ty check                            # 类型检查

V_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live   # 集成测试（需要真实 LLM）
```

集成测试环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `V_AGENT_LOCAL_SETTINGS` | `local_settings.py` | 配置文件路径 |
| `V_AGENT_LIVE_BACKEND` | `moonshot` | LLM 后端 |
| `V_AGENT_LIVE_MODEL` | `kimi-k2.5` | 模型名称 |
| `V_AGENT_ENABLE_BASE64_KEY_DECODE` | - | 设为 `1` 启用 base64 API key 解码 |
