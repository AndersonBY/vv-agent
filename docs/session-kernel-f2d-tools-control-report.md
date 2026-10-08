# F2d-1 tools/control 验收报告

日期：2026-10-08。工作树：`/home/makerbi/vectorvein/tmp/wt-f2-vv-agent`，分支 `feat/session-kernel-internal`，基线 `1530da7`。

指定 11 行全部关闭，其中 4 行为 `done (intentional difference)`。整体矩阵仍为 63 行：35 done、16 partial、12 missing；F3 仍被阻塞。

实现仅涉及内部 session 模块及其测试、文档。复用同一 driver、ToolOrchestrator、ApprovalProvider/Broker、runtime hooks 和 BackgroundSessionManager。公共 Runner、默认 wiring、顶层 exports、契约锁和 vendored fixtures 未修改；无提交、无 Rust/cargo 工作。

## 关闭行及证据

下列测试均位于 `tests/session/test_tools_control_parity.py`：25 个测试函数、222 个参数化用例。普通曝光/策略/停止场景复用 Runner-vs-SQLite `:memory:` harness；持久化场景额外运行真实 PostgreSQL、SQLite 文件和 `:memory:`。

| 矩阵行 | 状态 | 测试名称 |
| --- | --- | --- |
| 隐藏工具、动态 schema、曝光边界 | done | `test_hidden_tool_exposure_parity`, `test_dynamic_schema_new_turn_parity`, `test_dynamic_schema_active_turn_rejected`, `test_hook_patch_preserves_hidden_tool_boundary` |
| ask_user / SDK host interaction 回复 | done (intentional difference) | `test_user_wait_sdk_lifecycle_parity`（ask_user 分支） |
| check_background_command | done | `test_background_process_restart_owner_parity`, `test_background_forbidden_owner_parity` |
| stop_background_command | done | `test_background_process_restart_owner_parity`, `test_background_unknown_stop_is_not_confirmed_parity` |
| bash 后台会话 drive 重启 / owner | done | `test_background_process_restart_owner_parity` |
| 工具允许/拒绝、predicate、metadata、冻结/当前策略 | done | `test_tool_policy_matrix_parity`, `test_frozen_current_policy_dispatch_boundary`, `test_current_policy_predicate_rechecked_after_restart` |
| 审批模式、Provider/Broker、allow_session、timeout | done (intentional difference) | `test_approval_mode_provider_parity`, `test_approval_broker_restart_session_and_conflicting_answer`, `test_approval_timeout_restart_parity`, `test_provider_decision_receipt_events_restart`, `test_approval_provider_metadata_and_request_boundary_parity`, `test_expired_approval_answer_cannot_authorize_effect`, `test_rejected_broker_session_grant_cannot_authorize_next_tool`, `test_approval_absolute_deadline_includes_provider_time`, `test_approval_answer_closed_optional_fields` |
| before/after_tool_call hooks、short circuit、状态、重启 | done | `test_tool_hooks_state_approval_restart_parity`, `test_hook_patch_preserves_hidden_tool_boundary` |
| tool_use_behavior、pending batch、native FINISH | done | `test_tool_stop_pending_batch_native_finish_parity`, `test_tool_stop_error_result_parity` |
| no_tool_policy=wait_user | done (intentional difference) | `test_user_wait_sdk_lifecycle_parity`（no_tool 分支） |
| 取消 cooperative / unknown / descendants | done (intentional difference) | `test_cancellation_control_result_event_parity`, `test_cancellation_descendant_control_result_events_parity` |

## 有意差异：提交 reviewer 决策

1. `ask_user`：Runner SDK resume 开启新的 run，并保留先前 WAIT_RESPONSE 工具消息及新 user 消息；kernel 的 inbox 回复继续原 turn/op，只生成一个 definitive 工具结果。相同回复 noop，冲突回复拒绝，晚到重复不创建新 turn。
2. `no_tool_policy=wait_user`：kernel 写入真实 `turn_parked`，不制造工具调用；回复成为同一 turn 的 user 消息。Runner SDK resume 另开 run。
3. 审批：kernel 先 durable park，再经 inbox 记录 Provider/Broker 决定，`allow_session` 只从已应用的日志答复恢复。Broker 内存中的 session flag 无权授权。绝对 deadline 从 park 开始，覆盖 provider 决策处理阶段；零期限下 Runner 可以接受立即 allow，kernel 会先超时。已过期 allow/allow_session 答复拒绝，正常超时结果复用 Runner 的错误 producer。
4. 取消：Runner 函数工具路径当前抛出 CancelledError，且该路径无终止 run_cancelled 事件；kernel 提交 control、确认的 cooperative-stop receipt 或明确 unknown，并产生 cancelled 终止事件。未确认停止的工作不会被声明已停止。

## 归一化比较与保证边界

- FINISH/WAIT_USER 后的 pending batch：Runner 对未 admission 的 skipped 工具无 lifecycle；kernel 必须为已计划工作保留 skipped result。比较所有有序结果、最终输出和实际 effect 数，分别检查 lifecycle 差异。完整 Typed RunEvents 仍是 partial。
- 后台进程比较去除生成的 session ID、elapsed time 及终止文本/运行中 JSON envelope；status、output、exit code 必须一致。真实 shell 进程在 drive/Runtime 与 SQL fold 重建后由原 session/owner 重新附着。
- drive/Runtime 重建要求原进程管理器仍存活。OS worker/进程管理器重启不在保证内，不通过 PID 猜测接管；跨 owner 的读取、停止均拒绝。
- `op_prepared` 保存 before hook 改写的调用、冻结 capability、provider/idempotency binding、short-circuit 和 JSON 状态；已记录 before/after hook 结果恢复时不重跑。中断发生在回调提交前时，仍无 exactly-once 保证。
- 静态冻结拒绝在当前策略放宽后仍生效；可调用 predicate 是 host binding，在实际 dispatch 重新求值。hook 改写工具名不能扩大冻结曝光边界。
- timeout 边界测试显式推进 SQL 时钟，避免依赖 PG 在毫秒 sleep 内完成。数据、锁、park、inbox 和恢复仍走真实 SQL store。

## 剩余缺口

本轮闭合范围之外仍有 28 行未完成；下列 inventory 与 capability matrix 一致。完整 SDK/默认入口切换仍需这些能力和 F3 门禁。

| 状态 | 未闭合能力 |
| --- | --- |
| partial | Built-in `create_sub_task` |
| missing | Built-in `sub_task_status` |
| partial | Typed output coercion / repair exceptions / repair budget ledger |
| missing | Runtime before_memory_compact hook |
| missing | AfterCycleHook continue / steer / deny / stop |
| missing | MemoryProvider compact callbacks |
| partial | Session memory extraction/save/reload |
| partial | Microcompaction, summary and prompt-too-long recovery |
| partial | Budget limits: total and uncached input tokens |
| partial | Budget limits: total and per-name tool calls |
| partial | Budget limits: wall time, host cost, unavailable metrics |
| missing | Tracing processors and run/agent/tool spans |
| missing | Live assistant/reasoning/tool stream deltas |
| partial | Typed lifecycle RunEvents |
| partial | Multiple model endpoints, no stacked retries |
| missing | Agent.as_tool |
| missing | Configured sub-agents: policy/budget/workspace inheritance |
| missing | BackgroundAgentTask start/poll/wait/cancel |
| missing | Handoff and maximum-handoff enforcement |
| partial | Child session atomic admission/delivery/cancellation |
| partial | shared_state between tools, persistence and reconstruction |
| partial | Workspace local/memory/S3/streaming backend selection |
| partial | Completed RunResult fields and cycle/tool history |
| partial | Event store replay and durable consumer acknowledgement |
| partial | Interactive steer/follow-up/resume/archive/close |
| missing | CLI single-run, stream and persistent sessions |
| missing | App Server thread/turn/approval/replay/non-text input |
| partial | App Server model/list, schema and TypeScript export |

## 性能门禁

命令：`uv run python scripts/session_kernel_overhead.py --runs 200 --output docs/session-kernel-overhead-f2d-tools-control.json`。每条路径/场景 10 次 warmup、200 次测量；下面单位均为 ms。added p95 是两条路径 p95 之差。测量包含 schema、admission、日志写入、最终状态检查及 connection/thread cleanup。

| 场景 | Runner p50 / p95 | kernel p50 / p95 | added p95 | 目标 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| no_tool | 3.40 / 4.07 | 14.07 / 16.11 | 12.04 | 50 | PASS |
| two_tools | 5.25 / 5.84 | 25.06 / 25.94 | 20.10 | 50 | PASS |
| ten_turns | 33.89 / 36.81 | 108.34 / 125.47 | 88.67 | 100 | PASS |
| start_cancel | 4.62 / 5.20 | 14.76 / 16.32 | 11.12 | 50 | PASS |

四个场景结束后 Runner/kernel 均仅剩 1 个原始线程，新增线程为 0。

首轮十轮 added p95 为 104.70 ms，未达标。profile 后删除了 admission 对定义摘要的重复计算，复用 Runtime 已计算的摘要；record 仍验证嵌入摘要，lease/CAS 和执行路径保持一致。缓存失效、篡改及 hot-path 相关 108 个用例通过；上表为改动后的 200 次实测。

## 最终门禁

| 门禁 | 结果 |
| --- | --- |
| `uv run python scripts/contract_snapshot.py check` | PASS：契约 23.0.0，55 fixture files，manifest 未变 |
| `uv run ruff format --check .` | PASS：381 files |
| `uv run ruff check` | PASS |
| `uv run ty check` | PASS |
| `uv run pytest tests/session -q`（本地 PG） | PASS：816 passed，0 skipped，261.93 秒 |
| `VV_AGENT_TEST_REDIS_URL=redis://127.0.0.1:6399/15 uv run pytest`（真实 Redis + PG） | PASS：3291 passed，20 skipped，18 warnings，415.92 秒 |
| `uv run python scripts/session_kernel_overhead.py --runs 200` | PASS：4/4 场景达标，无新存活线程 |
| `git diff --check` / public/default/fixture 边界 | PASS：仅内部 session、测试和文档；HEAD 仍为 1530da7 |
| 测试 Redis 6399 清理 | PASS：已 shutdown nosave，端口不再响应 |

全量测试的 20 个 skip 为 6 个未开启的 live provider 用例、4 个跨 runtime/store 用例、9 个不适用于所选非 Redis backend 的参数变体和 1 个当前环境不支持的 symlink 用例。没有因缺少 Redis 或 PostgreSQL 而跳过适用场景。18 个 warning 均来自现有 distributed checkpoint 的多线程 fork DeprecationWarning。

命令均从 repo root 使用 repo-managed uv 环境运行；本环境的 `UV_CACHE_DIR` 指向 `/tmp/f2d-uv-cache`。Redis 由本次门禁启动并在退出时关闭；PG 各用例使用独立 disposable database。

未运行 opt-in 真实模型或跨语言探针，未运行 Rust/cargo。OS worker/进程管理器重启、未提交 hook 的 exactly-once，以及上面的 28 行仍不在本轮完成保证内。
