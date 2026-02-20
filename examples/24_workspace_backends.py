#!/usr/bin/env python3
"""Workspace backends: 可插拔文件存储, 支持本地/内存/S3/自定义后端.

演示四种使用方式:
1. 默认行为 — 不传 workspace_backend, 自动使用 LocalWorkspaceBackend
2. MemoryWorkspaceBackend — 纯内存, 适合测试/沙箱
3. S3WorkspaceBackend — 对接 S3 兼容存储 (AWS / MinIO / OSS / R2)
4. 自定义后端 — 实现 WorkspaceBackend Protocol, 对接任意存储

S3 模式需要配置环境变量, 参见 examples/.env.example
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask
from vv_agent.workspace import (
    FileInfo,
    LocalWorkspaceBackend,
    MemoryWorkspaceBackend,
    WorkspaceBackend,
)


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_wait_user",
        "cycle_failed",
    }:
        print(f"  [{event}] {payload}", flush=True)


# ---------------------------------------------------------------------------
# 自定义后端示例: PrefixedBackend
# 在真实场景中可替换为 S3Backend / OSSBackend / RedisBackend 等
# ---------------------------------------------------------------------------

class PrefixedBackend:
    """在所有写入内容前自动添加前缀的演示后端.

    内部委托给任意 WorkspaceBackend, 展示装饰器模式.
    """

    def __init__(self, inner: WorkspaceBackend, prefix: str = "[TAGGED] ") -> None:
        self._inner = inner
        self._prefix = prefix

    def list_files(self, base: str, glob: str) -> list[str]:
        return self._inner.list_files(base, glob)

    def read_text(self, path: str) -> str:
        return self._inner.read_text(path)

    def read_bytes(self, path: str) -> bytes:
        return self._inner.read_bytes(path)

    def write_text(
        self, path: str, content: str, *, append: bool = False,
    ) -> int:
        tagged = f"{self._prefix}{content}" if not append else content
        return self._inner.write_text(path, tagged, append=append)

    def file_info(self, path: str) -> FileInfo | None:
        return self._inner.file_info(path)

    def exists(self, path: str) -> bool:
        return self._inner.exists(path)

    def is_file(self, path: str) -> bool:
        return self._inner.is_file(path)

    def mkdir(self, path: str) -> None:
        self._inner.mkdir(path)


# ---------------------------------------------------------------------------
# Helper: 构建 runtime + task 并执行
# ---------------------------------------------------------------------------

def _build_and_run(
    *,
    label: str,
    workspace_backend: WorkspaceBackend | None,
    workspace: Path,
    llm_client: Any,
    model_id: str,
    verbose: bool,
    prompt: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"[demo] {label}")
    print(f"{'='*60}")

    runtime = AgentRuntime(
        llm_client=llm_client,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler if verbose else None,
        workspace_backend=workspace_backend,
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent. Use workspace tools to complete tasks.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    task = AgentTask(
        task_id=f"ws_backend_{uuid.uuid4().hex[:8]}",
        model=model_id,
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_cycles=5,
    )

    result = runtime.run(task, workspace=workspace)
    print(f"\n  状态: {result.status.value}")
    print(f"  回答: {result.final_answer}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    """从 .env 文件加载环境变量 (不覆盖已有值)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    # 自动加载 examples/.env (不覆盖已有环境变量)
    _load_dotenv(Path(__file__).parent / ".env")

    settings_file = Path(
        os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"),
    )
    backend_name = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
    workspace = Path(
        os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace"),
    ).resolve()
    verbose = os.getenv(
        "V_AGENT_EXAMPLE_VERBOSE", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    mode = os.getenv("V_AGENT_EXAMPLE_WS_MODE", "all").strip().lower()

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_openai_llm_from_local_settings(
        settings_file, backend=backend_name, model=model,
    )
    common = dict(
        workspace=workspace,
        llm_client=llm,
        model_id=resolved.model_id,
        verbose=verbose,
    )

    # --- 方式 1: 默认 (不传 workspace_backend, 自动 LocalWorkspaceBackend) ---
    if mode in {"all", "default"}:
        _build_and_run(
            label="方式 1: 默认 — 自动使用 LocalWorkspaceBackend",
            workspace_backend=None,
            prompt=(
                "在 workspace 中创建 hello.txt 写入 'Hello from default backend', "
                "然后读取并输出内容。"
            ),
            **common,
        )

    # --- 方式 2: MemoryWorkspaceBackend (纯内存, 不落盘) ---
    if mode in {"all", "memory"}:
        _build_and_run(
            label="方式 2: MemoryWorkspaceBackend — 纯内存, 不落盘",
            workspace_backend=MemoryWorkspaceBackend(),
            prompt=(
                "在 workspace 中创建 memo.txt 写入 'Hello from memory backend', "
                "然后读取并输出内容。"
            ),
            **common,
        )
        # 验证: 文件不会出现在磁盘上
        if not (workspace / "memo.txt").exists():
            print("  ✓ 验证通过: memo.txt 未落盘 (纯内存)")

    # --- 方式 3: S3WorkspaceBackend (S3 兼容存储) ---
    if mode in {"all", "s3"}:
        s3_bucket = os.getenv("S3_BUCKET", "")
        if not s3_bucket:
            if mode == "s3":
                print(
                    "[跳过] S3 模式需要设置 S3_BUCKET 环境变量.\n"
                    "       参见 examples/.env.example",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("\n[跳过] 方式 3: S3 — 未设置 S3_BUCKET, 跳过")
        else:
            from vv_agent.workspace import S3WorkspaceBackend

            s3_backend = S3WorkspaceBackend(
                bucket=s3_bucket,
                prefix=os.getenv("S3_PREFIX", ""),
                endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
                region_name=os.getenv("S3_REGION") or None,
                aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID") or None,
                aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY") or None,
                addressing_style=os.getenv("S3_ADDRESSING_STYLE", "virtual"),
            )
            _build_and_run(
                label="方式 3: S3WorkspaceBackend — S3 兼容存储",
                workspace_backend=s3_backend,
                prompt=(
                    "在 workspace 中创建 s3_test.txt 写入 'Hello from S3 backend', "
                    "然后读取并输出内容。"
                ),
                **common,
            )

    # --- 方式 4: 自定义 PrefixedBackend (装饰器模式) ---
    if mode in {"all", "custom"}:
        inner = LocalWorkspaceBackend(workspace)
        prefixed = PrefixedBackend(inner, prefix="[AUTO-TAG] ")
        _build_and_run(
            label="方式 4: PrefixedBackend — 自定义装饰器后端",
            workspace_backend=prefixed,
            prompt=(
                "在 workspace 中创建 tagged.txt 写入 'custom backend works', "
                "然后读取并输出内容。"
            ),
            **common,
        )
        # 验证: 文件内容带前缀
        tagged = workspace / "tagged.txt"
        if tagged.exists():
            content = tagged.read_text(encoding="utf-8")
            print(f"  ✓ 磁盘内容: {content!r}")

    print("\n[demo] 完成!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
