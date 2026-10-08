# F2d-2 memory / budgets / events / model 验收报告

日期：2026-10-08。工作树：`/home/makerbi/vectorvein/tmp/wt-f2-vv-agent`，分支 `feat/session-kernel-internal`，基线 `e9194f4`。

指定 15 行全部关闭，其中 7 行为 `done (intentional difference)`。整体矩阵为 63 行：50 done、6 partial、7 missing；F3 仍被阻塞。

内部组装继续使用同一个 driver。新增 memory、lifecycle、events、tracing 适配器；为了避免依赖计划退役的 CycleRunner，将既有 MemoryProvider 回调和错误处理原样提取到保留的 `memory/provider.py`，Runner 与 kernel 复用。没有切换默认入口、增加公共 exports、修改 contract lock 或 vendored fixtures；没有提交、push、部署或 Rust/cargo 工作。

## 关闭行及配对证据

`C` 为 `tests/session/test_capability_parity.py`，34 个测试函数、358 个参数化用例；`R` 为 `tests/session/test_recovery_matrix.py`。C 的持久化场景均运行真实 PostgreSQL、SQLite 文件和 SQLite `:memory:`。配对两侧分别使用公共 Runner 与 kernel 的真实 producer，脚本模型只替代网络调用。端点场景调用真实 VvLlmClient，模拟 transport 的失败与响应，核对每次请求的路由。纯校验和污染隔离测试是配对用例之外的补充。

| 矩阵行 | 状态 | 测试名称 |
| --- | --- | --- |
| before_memory_compact hook | done | C `test_before_memory_hook_replacement_restart_parity`, `test_boundary_validation_rolls_back` |
| AfterCycleHook continue / steer / deny / stop | done | C `test_after_cycle_decision_snapshot_restart_parity`, `test_after_cycle_steering_boundary_parity`, `test_after_cycle_wait_and_native_finish_snapshot_restart` |
| MemoryProvider before / after_compact | done | C `test_memory_provider_logged_callbacks_restart_parity` |
| Session memory extraction / save / reload | done | C `test_session_memory_logged_extract_save_reload_parity` |
| Microcompaction / summary / prompt-too-long | done (intentional difference) | C `test_memory_compaction_runner_producer_parity`, `test_prompt_too_long_logical_cycle_and_tail_parity` |
| Total / uncached input token budgets | done | C `test_token_budget_boundaries_parity`, `test_missing_usage_budget_parity` |
| Total / per-name tool budgets | done | C `test_tool_batch_admission_restart_parity` |
| Wall time / host cost / unavailable metrics | done (intentional difference) | C `test_wall_time_budget_parity`, `test_host_cost_and_unavailable_metrics_parity`, `test_host_meter_failure_latch_restart_parity`, `test_lost_active_interval_is_unavailable_after_restart`; R `test_budget_counts_and_elapsed_survive_recovery` |
| Tracing processors / run-agent-tool spans | done (intentional difference) | C `test_trace_delivery_span_parity_and_no_recovery_duplicates`, `test_trace_ack_boundary_and_processor_failure`, `test_span_ids_are_scoped_by_session`, `test_trace_processors_cannot_mutate_durable_output` |
| Live assistant / reasoning / tool deltas | done | C `test_live_stream_deltas_and_durable_final_restart_parity` |
| Typed lifecycle RunEvents | done (intentional difference) | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer`, `test_event_observers_cannot_mutate_durable_receipts`，以及 memory / budget / hook 配对用例 |
| Multiple model endpoints | done (intentional difference) | C `test_endpoint_routing_preference_logged_attempts_parity`, `test_logged_endpoint_dispatch_rejects_routing_tamper` |
| Typed output / repair exceptions / ledger | done (intentional difference) | C `test_typed_output_repair_ledger_restart_parity`, `test_output_repair_exceptions_parity`, `test_output_coercion_exception_becomes_durable_result`, `test_repair_usage_budget_is_logged_intentional_difference`, `test_repair_missing_usage_policy_is_durable`, `test_uncertain_repair_does_not_retry_on_recovery` |
| Completed RunResult fields | done (intentional difference) | C `test_wait_result_fields_parity`, `test_completed_result_compaction_flags_are_per_cycle`，以及 typed output / endpoint / budget / hook 配对用例 |
| RunEventStore replay / durable ACK | done | C `test_lifecycle_events_replay_ack_and_rollback_parity`, `test_delegation_events_and_child_replay_paired_producer` |

Token 用例覆盖零额度、usage 前一单位、相等、超额和足够额度：total usage 15，uncached input usage 6；缺失 total/cache 明细分别覆盖 STOP / CONTINUE。工具预算的两调用 batch 覆盖 0/1/2/3 额度及总数/每名限制，并在 admission commit 后重建，验证 effect 为 0 或 2，预算不重复计数。AfterCycleHook 覆盖等待、native FINISH、invalid/exception、max-cycle steering 禁止边界，并比较重建 snapshot 的 cycles、native outcome 和 token usage。

## 重启边界与预算语义

- `boundary_recorded` 为闭合、按 stage 校验的内部记录。before-memory 的替换 context、source digest 与 JSON shared state 在压缩前提交；after-cycle 的 snapshot 来自 adopted model/tool receipts，决定在下一次 dispatch 前提交。恢复不会重新运行已记录回调/决定。未知字段、source/state/budget 错误整体 rollback。
- Micro 与 summary 共用 logged started/completed lifecycle。MemoryProvider 的 metadata、错误处理及 archive stats 绑定该 lifecycle。第一轮 prompt-too-long 使用配置的 tail，后续 emergency 缩小 tail；逻辑 cycle 不因这些恢复请求增加。
- Session memory 是 tools-free logged model operation，purpose=`session_memory`。structured state 先入日志，再 atomic replace 文件。文件是可删除、可从日志重建的 projection；下一 turn 编译之前恢复文件并 reload。
- 工具预算采用**完整 batch admission**：model receipt、按 admission name 计数的 reservation 与所有 tool plans 在同一 commit 中保存。任何总数/每名限制不足时，整个 batch 不执行。已保留但后来被 FINISH/wait 跳过的调用仍计入 admission；重试/恢复不重新保留。每次 dispatch 继续检查 wall/host 等动态限制。Runner 的 whole-batch preflight 与 kernel admission 的配对结果一致。
- Total/uncached token usage 包含 primary、compaction、session_memory、output_repair 等内部调用。reported usage 进入账本；missing usage 遵循 STOP/CONTINUE。host meter 错误、unit/currency 不一致、下降读数及 unavailable latch 跨 Runtime 重建保留。
- Live deltas 为 volatile sink，不入日志。sink 丢失或异常不会影响最终 content、reasoning、tool calls；最终内容只从 durable receipts 重建。
- Typed events 和 spans 都有稳定身份，包含 session 维度。事件 observer 与 tracing processor 获得 detached 数据，不能修改保留的日志结果。
- `SessionRunEventStore` 不维护第二本 event ledger；replay 是 record projection，`batch(tx)` 使用 consumer cursor，宿主写 projection 与 ACK 可同事务 rollback。外部 sink 在 delivery 与 ACK 间崩溃仍可能重复，宿主可按稳定 event ID 去重。
- 完成结果采用本 turn 的 terminal prefix，后续 turn 不污染旧结果；memory_compacted 按 cycle 计算。等待、errors、partial output、budget exhaustion、typed output 与 model repair ledger 均从记录重建。

## 有意差异：提交 reviewer / C1 决策

1. **Typed output 错误**：Runner 某些 coercion 路径直接抛 ValueError；dataclass repair 的结果序列化路径可抛 TypeError。kernel 把失败变成 durable failed result，并正确重建合法 dataclass 输出。错误/partial output 行为按相应配对测试逐项比较。
2. **Repair 计账**：kernel 将 tools-free callback 作为 logged `output_repair` model operation；reported usage 或 accounting_missing 都进入 token budget 和结果 ledger。Runner 不记录 repair model-call 账本。strict missing usage 可以使 repair 后结果失败；unknown repair 不自动重试。当前 v23 公共 enum 无 repair 项，公共投影沿用 AGENT_CYCLE，准确 purpose 在事件 metadata 与 `RunResult.metadata.session_model_calls` 中保存；新的公共 discriminator 由 C1 决定。
3. **压缩请求复用**：相同 source/mode/tail 的 rejected summary 复用已记录 receipt；Runner 可以再次请求相同 summary。接受/拒绝、tail 与真实 MemoryManager 输出另行配对，不通过重复昂贵请求模拟旧行为。
4. **Wall time unavailable**：worker 丢失 active interval 时明确标为 accounting_missing，不从进程 downtime 猜测 elapsed；strict policy 停止。重建无法提供旧进程的真实 monotonic interval。
5. **Tracing at-most-once**：独立 traces cursor 必须先 commit ACK，再调用非事务 processor。恢复不重复 spans；ACK 后崩溃、processor failure 可能丢 telemetry。拒绝 ambient host transaction，避免未提交 ACK 下先发出 spans。此交付取舍须由 reviewer 明确接受。
6. **Typed event 身份和顺序**：child admission/completion lifecycle 属于 parent log/run，携带 child session/turn 身份；Runner 的 configured-child lifecycle 属于 child run。记录投影顺序与即时发射顺序、whole-batch planned/skipped lifecycle 有明确差异。全量子代理 SDK 仍是独立未完成行，本轮仅关闭 typed projection/replay producer。
7. **Endpoint attempts / RunResult ledger**：请求冻结 preferred/randomized order，每 logged attempt 只向一个 endpoint 发一次请求，最多覆盖冻结列表（不足两个 endpoint 时保留两次 kernel attempt 上限）；Runner 的内部 retry/fallback 被拆成独立账目，没有叠加 transport retries。成功 endpoint 的 preference 从 durable success 恢复。严格 missing-usage policy 可以阻止 fallback。结果账本如实保留这些 attempts 和 repair，因而不同于 Runner 的合并 call 账本。

上述变化未写入只读的树外替换计划；这里是该计划“评审记录”与 C1 契约 24 的待决补充。本轮不发布公共契约，也不执行 F3 切换。

## 剩余缺口

本轮指定 15 行没有遗留。矩阵还有以下 13 行，完整 SDK / 默认入口切换仍被它们及 F3 门禁阻塞。

| 状态 | 未闭合能力 |
| --- | --- |
| partial | Built-in create_sub_task configured-agent adapter |
| partial | Child session atomic admission/delivery/cancellation 的完整 SDK parity |
| partial | shared_state arbitrary Python objects 的 host reconstruction bindings |
| partial | Workspace S3 / streaming backend 配对 |
| partial | Interactive steer/follow-up/resume/archive/close facade |
| partial | App Server model/list、schema、TypeScript cut-over producer |
| missing | Built-in sub_task_status |
| missing | Agent.as_tool |
| missing | Configured sub-agents inheritance |
| missing | BackgroundAgentTask start/poll/wait/cancel |
| missing | Handoff / maximum-handoff enforcement |
| missing | CLI single-run / stream / persistent sessions |
| missing | App Server thread/turn/approval/replay/non-text input |

## Per-turn headroom 与 cProfile

A 完整实现后的优化前候选实测 ten_turns added p95 为 99.061 ms（reviewer F2d-1 重测为 98.7 ms）。本轮目标收紧为十轮 80 ms、单轮 50 ms。前后都使用原场景、同一个 driver、10 次 warmup / 200 次测量，包含 SQLite schema、admission、durable writes、final state read 和 connection/thread cleanup。added p95 定义为 kernel p95 减 Runner p95，不是逐次差值的 p95。下面单位均为 ms；Runner 的主机时序差异如实列出。

命令：`uv run python scripts/session_kernel_overhead.py --runs 200 --output docs/session-kernel-overhead-f2d2.json`。

| 场景 | 优化前 added p95 | 最终 Runner p50 / p95 | 最终 kernel p50 / p95 | 最终 added p95 | 目标 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no_tool | 33.746 | 3.506 / 4.118 | 13.339 / 14.519 | 10.401 | 50 | PASS |
| two_tools | 41.258 | 5.582 / 6.416 | 26.478 / 35.102 | 28.686 | 50 | PASS |
| ten_turns | 99.061 | 37.945 / 46.510 | 106.476 / 113.137 | 66.626 | 80 | PASS |
| start_cancel | 19.307 | 4.812 / 5.679 | 15.989 / 17.214 | 11.535 | 50 | PASS |

最终十轮 headroom 为 13.374 ms；摊销 added p95 为 6.663 ms/turn。四场景每条路径结束时都仅剩 1 个原始线程，leaked_threads 为空。RSS 是 GC 后整批 current RSS 差值，不能当 peak RSS。

| 场景 | Runner / kernel RSS delta (KiB) | Runner / kernel threads after |
| --- | ---: | ---: |
| no_tool | +32 / +676 | 1 / 1 |
| two_tools | +0 / +244 | 1 / 1 |
| ten_turns | -940 / +0 | 1 / 1 |
| start_cancel | +244 / +88 | 1 / 1 |

Profiling 使用同样的 ten_turns kernel 场景，5 次 warmup 后测 30 次。完整 cumulative top 25 位于 `session-kernel-profile-f2d2-before.txt` 和 `session-kernel-profile-f2d2-after.txt`；机器可读热路径条目位于 `session-kernel-profile-f2d2-summary.json`。profile total_tt 7.566 → 6.924 秒，function calls 9,007,681 → 7,915,021。累计时间嵌套重叠，不能相加。下表列出涉及优化的主要条目。

| cProfile 条目 | 前调用数 → 后调用数 | 前累计秒 | 后累计秒 |
| --- | ---: | ---: | ---: |
| `session/kernel.py:step` | 1,800 → 1,800 | 6.813 | 6.143 |
| `canonical_json.py:canonical_json_bytes` | 18,780 → 42,180 | 2.169 | 1.705 |
| `session/records.py:make_record` | 3,330 → 3,330 | 1.847 | 1.494 |
| `session/records.py:encode` | 28,020 → 28,020 | 1.736 | 1.383 |
| `session/sql.py:append` | 1,800 → 1,800 | 1.221 | 1.069 |
| `canonical_json.py:_stdlib_compatible` | 886,410 → 625,110 | 1.108 | 0.788 |
| `canonical_json.py:_jcs_encode` | 54,900 → 33,600 | 1.080 | 0.719 |
| `session/records.py:_validate` | 3,390 → 3,390 | 0.753 | 0.792 |
| `session/runtime.py:_definition` | 1,500 → 1,200 | 0.695 | 0.657 |
| `session/kernel.py:task` | 2,700 → 1,800 | 0.384 | 0.162 |
| `session/records.py:task` | 2,700 → 1,800 | 0.378 | 0.158 |

优化复用 embedded-digest 检查已经生成的 nested JCS bytes，组合闭合 ASCII-key record envelope，避免再次编码大 definition/request/result。内部只读 definition 检查使用 retained task；host callbacks 仍拿 detached copies。tool context 从冻结 model request 取得 schemas，减少重复 definition preparation；无压缩时推迟 source digest。更多 canonical_json_bytes 调用是小片段编码，累计耗时下降。Record._validate、lease/CAS/fencing 没有删除或放宽，也没有新增执行路径。

`test_record_encoding_reuses_validated_fields_with_identical_jcs` 比较完整 JCS 字节（4 variants × 100 vectors，包含浮点、Unicode/astral key 和 escaping），`test_record_composition_preserves_nested_invalid_json_rejection` 保留嵌套非法 JSON 拒绝；既有 mutation-isolation、篡改、cache invalidation 和跨 writer suites 继续运行。

## M6 capacity

命令：`uv run python scripts/session_kernel_benchmark.py --sizes 5000 20000 --samples 1 --assert-capacity --output docs/session-kernel-capacity-f2d2.json`。真实 PostgreSQL，每个 size 一次测量，保留原有 bounded 1 KiB receipt workload 和断言。单位为 ms。

| Records | Cold drive | Steady append | Full fold | 额外 provider calls | Lease failures |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5000 | 582.823 | 9.298 | 36.036 | 0 | 0 |
| 20000 | 2289.341 | 16.770 | 301.343 | 0 | 0 |

| Catalog sessions | Runnable items | Full pagination ms | 目标 ms | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 1000 | 1500 | 16.420 | 1000 | PASS |
| 10000 | 15000 | 208.137 | 1000 | PASS |

两种规模的 cold drive、steady append ≤50 ms、完整 catalog pagination ≤1 秒均通过 `--assert-capacity`。这是请求的一次容量采样，不能当长期生产负载分布。

## 最终门禁

| 门禁 | 结果 |
| --- | --- |
| `uv run python scripts/contract_snapshot.py check` | PASS：契约 23.0.0，55 fixture files，manifest 未变 |
| `uv run ruff format --check .` | PASS：386 files |
| `uv run ruff check` | PASS |
| `uv run ty check` | PASS |
| `uv run pytest tests/session -q`（本地 PG） | PASS：1187 passed in 281.48s (0:04:41) |
| `VV_AGENT_TEST_REDIS_URL=redis://127.0.0.1:6400/15 uv run pytest` | PASS：3662 passed, 20 skipped, 18 warnings in 427.71s (0:07:07) |
| Overhead，10 warmups / 200 runs | PASS：4/4，single-turn ≤50 ms，ten_turns ≤80 ms，无新存活线程 |
| M6 5000 / 20000，`--samples 1 --assert-capacity` | PASS：cold drive / append / catalog 扫描均通过 |
| `git diff --check` / 公共与默认入口边界 | PASS：无 exports/default wiring/contract lock/fixture/local_settings/Rust 变更，HEAD e9194f4，无提交 |
| 测试 Redis 6400 清理 | PASS：核验 gate-owned PID 后 shutdown nosave，端口不再响应 |

全量 20 个 skip 为 6 个未开启 live provider 用例、4 个跨 runtime/store 用例、9 个不适用所选非 Redis backend 的参数变体、1 个环境 symlink 用例。没有因缺少 Redis 或 PG 而跳过适用用例。18 个 warning 来自既有 distributed checkpoint 的多线程 fork DeprecationWarning。

命令从 repo root 使用 repo-managed uv 环境执行；`UV_CACHE_DIR=/tmp/f2d2-uv-cache`。Redis 用 `redis-server --port 6400 --daemonize yes --save ""` 启动，额外绑定 loopback 并保存 gate PID，等待 PONG 后运行全量测试，结束核验 PID 后关闭；PG fixture 为每个用例建立/删除独立 disposable database。

未运行 opt-in 真实模型、跨语言探针或 Rust/cargo。未提交回调仍为至少一次；tracing ACK 后可能丢 telemetry，外部 event sink 可重复交付；上述 13 行缺口和 F3 切换仍未完成。
