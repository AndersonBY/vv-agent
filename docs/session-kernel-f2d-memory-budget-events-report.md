# F2d-2 memory / budgets / events / model 验收报告

日期：2026-10-08。工作树：`/home/makerbi/vectorvein/tmp/wt-f2-vv-agent`，分支 `feat/session-kernel-internal`，F2d-2 基线 `e9194f4`；F2d-2b 复验基线为已提交 `9c5eafe`。

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

## Per-turn headroom 与 F2d-2b 复验

F2d-2 的单次 ten_turns added p95 66.626 ms 不能建立独立复跑的 headroom。Reviewer 的三次独立结果为 84.4、104.5、95.1 ms，均超过 80 ms；其中后两次 added p50 为 70.1、71.0 ms，two_tools added p95 范围为 20.7–34.3 ms。

F2d-2b 从已提交基线 `9c5eafe` 开始。本机再次采集了三次改前和三次改后完整数据；改前这组三次碰巧通过，但最窄余量仅 2.520 ms，不能替代 reviewer 的失败证据。两组均逐次启动独立进程，命令完全相同：`uv run python scripts/session_kernel_overhead.py --runs 200 --warmup 10`。最终三次与 pytest、M6 和诊断插桩分开串行执行。脚本没有改动；四个场景、正常 driver、SQLite schema/admission/durable writes/final read/connection/thread cleanup 均在原计时范围内，没有禁用或 freeze GC。

added p50/p95 分别为 kernel 分位数减 Runner 对应分位数，不是逐次差值的分位数。下面单位均为 ms，单轮目标 50 ms，十轮目标 80 ms。每个单元格为 **added p50 / added p95**。

| 场景 | 改前 1 | 改前 2 | 改前 3 | 改后 1 | 改后 2 | 改后 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_tool | 9.801 / 10.479 | 9.802 / 10.099 | 9.722 / 9.597 | 9.818 / 10.344 | 9.560 / 9.984 | 9.089 / 9.538 |
| two_tools | 20.181 / 22.920 | 20.480 / 21.677 | 19.369 / 20.629 | 17.059 / 19.254 | 16.932 / 17.697 | 17.159 / 17.928 |
| ten_turns | 65.247 / 77.480 | 64.873 / 73.143 | 64.933 / 74.340 | 59.006 / 62.082 | 54.138 / 68.990 | 53.115 / 53.798 |
| start_cancel | 10.565 / 10.944 | 10.947 / 11.944 | 11.141 / 11.890 | 10.572 / 11.342 | 10.312 / 11.311 | 10.073 / 10.110 |

仅做 fragment/receipt 优化但尚未共享 schema 对象图的中间候选也完整跑了三次。其 ten_turns added p95 为 69.048、68.158、86.894 ms，第三次失败，不能作为验收通过。下表保留四场景 added p50 / added p95（ms）：

| 场景 | 中间候选 1 | 中间候选 2 | 中间候选 3 |
| --- | ---: | ---: | ---: |
| no_tool | 9.376 / 9.494 | 9.181 / 9.450 | 9.794 / 10.900 |
| two_tools | 17.848 / 18.544 | 17.988 / 18.525 | 18.012 / 19.639 |
| ten_turns | 58.271 / 69.048 | 60.036 / 68.158 | 62.605 / 86.894 |
| start_cancel | 10.300 / 10.824 | 10.846 / 13.104 | 10.780 / 11.836 |

原始分位数如下；R 为 Runner，K 为 kernel，每组数值均为 p50 / p95（ms）。

| 场景 | 轮次 | 改前 R | 改前 K | 改后 R | 改后 K |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_tool | 1 | 3.176 / 3.513 | 12.976 / 13.992 | 3.415 / 3.916 | 13.233 / 14.261 |
| no_tool | 2 | 3.412 / 4.039 | 13.214 / 14.138 | 3.225 / 3.699 | 12.785 / 13.684 |
| no_tool | 3 | 3.420 / 5.224 | 13.142 / 14.821 | 3.251 / 3.596 | 12.340 / 13.134 |
| two_tools | 1 | 5.265 / 5.859 | 25.446 / 28.779 | 5.170 / 5.840 | 22.230 / 25.094 |
| two_tools | 2 | 5.389 / 6.309 | 25.869 / 27.986 | 5.166 / 5.655 | 22.098 / 23.352 |
| two_tools | 3 | 5.142 / 5.772 | 24.511 / 26.402 | 5.229 / 5.705 | 22.388 / 23.633 |
| ten_turns | 1 | 35.922 / 39.078 | 101.169 / 116.558 | 34.171 / 36.373 | 93.176 / 98.455 |
| ten_turns | 2 | 36.219 / 39.542 | 101.093 / 112.685 | 33.799 / 35.077 | 87.937 / 104.067 |
| ten_turns | 3 | 34.990 / 36.872 | 99.923 / 111.212 | 34.663 / 37.018 | 87.778 / 90.816 |
| start_cancel | 1 | 4.388 / 4.887 | 14.953 / 15.831 | 4.302 / 4.787 | 14.873 / 16.129 |
| start_cancel | 2 | 4.558 / 5.219 | 15.504 / 17.163 | 4.252 / 4.735 | 14.564 / 16.046 |
| start_cancel | 3 | 4.387 / 4.942 | 15.529 / 16.832 | 4.295 / 4.847 | 14.367 / 14.958 |

最终三次十轮 headroom 分别为 17.918, 11.010, 26.202 ms；最小余量 11.010 ms。kernel 的十轮 p95−p50 从改前 15.389, 11.592, 11.290 ms 变为改后 5.278, 16.130, 3.038 ms。所有三次的四个场景均通过；Runner/kernel 结束后均为 1 个原始线程，leaked_threads 均为空。

仅将最后一次完整结果写入既有 `session-kernel-overhead-f2d2.json`。前三次改前、三次改后和 profile 的关键数字保留在本报告中；删除 `session-kernel-overhead-f2d2-before.json` 及 `session-kernel-profile-f2d2-*.txt/json`，没有添加 docs 下的逐轮或 profile 文件。最后一轮 RSS 数据如下；它是整批结束并 GC 后的 current RSS 差值，不是峰值。

| 场景 | Runner / kernel RSS delta (KiB) | Runner / kernel threads after |
| --- | ---: | ---: |
| no_tool | +0 / +4 | 1 / 1 |
| two_tools | +0 / +212 | 1 / 1 |
| ten_turns | -940 / +616 | 1 / 1 |
| start_cancel | +240 / +56 | 1 / 1 |

### 根因与诊断证据

诊断在两个独立进程中执行相同工作负载：10 次 warmup，200 次 ten_turns 的 `gc.callbacks` 采样，随后 30 次 cProfile，再 100 次方法计时。GC 阈值保持 `(700, 10, 10)`；插桩数字只用于定位，不是上面的验收数字。

1. **GC 确实制造大尖峰，但不是全部 p95 的解释。** 改前 200 个样本发生 1468/133/7 次 generation 0/1/2 collection；gen2 暂停 42.224–50.435 ms，全部 collected=0。含 gen2 的 7 个样本 p50 为 158.664 ms，其余 193 个样本 p50/p95 为 109.620/122.237 ms。最终同样 200 次采样为 770/69/1 次 collection，gen2 仅一次，暂停 44.226 ms；没有改变 GC 配置。
2. **重复编码是可消除的常态成本。** 每个已完成 primary model 在下一步检查完成 effects 时，原实现再次构造 `op_completed`、校验/编码 result，然后丢弃 `[0]`。现在直接复用 durable receipt 的 effects 检查；30 次 ten_turns 的 make_record 从 3330 次降至 3030 次。Record payload 从逐字段 key/value 调用改为连续字段块 JCS，冻结 task 指纹与 definition 的固定 schema/capability/memory/model-binding 字节分别缓存，变化仍触发重算。
3. **Schema 对象图被反复编码、复制并长期保留，是减少 GC 频率的关键。** 未过门槛候选在十轮中保存了 20 份相同 schema 的独立根对象，摘要只有 1 个，但实际包含 2240 个独立 dict/list。最终 20 处引用共享 1 个已验证 Record 的只读 JSON 图，只剩 112 个独立容器；缓存其 JCS 字节并复用于 definition/request 的完整 digest 和 envelope。Tracemalloc 的 JSON decoder 存活分配由 955 KiB / 14194 allocations 降到 410 KiB / 6461 allocations；cold tokenizer/import 的整次 peak 约 43.2 MB，基本不变，不能声称总峰值同比下降。Schema 只有在字节完全匹配，或 request 的元素仍是来源 Record 的同一只读对象时才复用。Hooks 修改得到独立副本；源 Record、字段 digest、schema/capability/model-binding drift 和正常 lease/CAS 路径仍被检查。
4. **快照有小而确定的额外复制。** 原 `Fold.snapshot()` 先 fork 整个 fold，再丢弃 copied history/seen/consumed 索引。现在只 detach ExecutionState；100 次 ten_turns 的 fork 调用 7100 → 6000，state/wait/child handle 的输出隔离继续保留。
5. **SQL 和线程是保留的常态成本。** SQL append 调用数在两组方法采样中均为 6000，耗时分布见下表；没有通过减少提交或 lease/CAS 校验换取收益。独立线程计时中 100 次十轮 `_invoke` 共 850.285 ms（含 callback）、heartbeat start/join 共 267.648/155.971 ms、external join 共 8.301 ms。没有改变线程/超时/取消机制，也没有证据支持把 join 本身作为主因。
6. **Endpoint O(records) 扫描有真实代码证据，但不是本基准的来源。** ScriptedLLM 不进入 VvLlmClient 分支。该历史反向扫描另行改为 fold 内的 scalar preference；普通成功与 audit 成功都与原逻辑一致，失败、未 dispatch 和无 endpoint 不替换 preference；fork/snapshot/冷重建及真实端点配对测试覆盖恢复。

GC 采样中的 kernel ten_turns p50/p95 为 109.865/126.093 → 92.992/108.625 ms。原中间候选另有一个不含 gen2 的 220.960 ms 样本，不能归因；没有将它丢弃。最终的 max 为 135.504 ms。验收针对 p95，不建立 max/p99 SLO。

100 次十轮方法计时的每次调用分布如下。不同步骤和阶段混在同一方法中；累计时间包含嵌套调用，不能相加。

| 方法 | 改前调用数 / p50 / p95 ms | 改后调用数 / p50 / p95 ms |
| --- | ---: | ---: |
| `_Driver.step` | 6000 / 1.593 / 2.806 | 6000 / 1.250 / 2.671 |
| `Runtime._definition` | 4000 / 0.099 / 0.614 | 4000 / 0.070 / 0.309 |
| `_Driver.plan` | 1000 / 0.637 / 0.837 | 1000 / 0.378 / 0.553 |
| `SQLSessionTx.append` | 6000 / 0.302 / 0.454 | 6000 / 0.302 / 0.450 |
| `Fold.fork` | 7100 / 0.020 / 0.027 | 6000 / 0.020 / 0.028 |
| `Fold.snapshot` | 1100 / 0.081 / 0.123 | 1100 / 0.071 / 0.106 |
| `_Driver.load_state` | 1000 / 0.323 / 0.489 | 1000 / 0.312 / 0.465 |

cProfile 30 次十轮的 total_tt 为 5.956 → 5.122 秒，function calls 为 7,915,304 → 6,493,214。它受插桩和线程交织影响；调用数用于识别重复工作，绝对累计时间不替代验收。

| cProfile 条目 | 改前调用数 → 改后调用数 | 改前累计秒 | 改后累计秒 |
| --- | ---: | ---: | ---: |
| `canonical_json.py:canonical_json_bytes` | 42,180 → 22,140 | 1.492 | 0.844 |
| `session/records.py:make_record` | 3,330 → 3,030 | 1.293 | 0.963 |
| `session/runtime.py:_definition` | 1,200 → 1,200 | 0.581 | 0.319 |
| `session/records.py:_record_bytes` | 3,390 → 3,090 | 0.274 | 0.182 |
| `session/records.py:_check_digest` | 1,500 → 1,200 | 0.500 | 0.281 |

`test_definition_composition_matches_full_jcs_across_turns_and_nested_values` 比较组成的 definition digest 与完整 JCS（浮点、Unicode/astral keys、布尔/数字、嵌套 task/tools 名称）；既有 4 variants × 100 vectors 的 record 字节比较及非法嵌套 JSON 拒绝继续通过。新增 frozen task 不重复序列化且 schema/capability/memory/children/model binding 改变会失效的测试，snapshot 索引不复制与 mutation-isolation 测试，receipt 构造计数及 endpoint 冷重建/opaque receipt 测试；新增 schema sharing、host mutation isolation、source full-validation、embedded digest mismatch 与 UTF-16/JCS 等价测试。closed schema、embedded/storage digests、lease/CAS/fencing 与宿主输出隔离均未放宽。

## M6 capacity

命令：`uv run python scripts/session_kernel_benchmark.py --sizes 5000 20000 --samples 1 --assert-capacity --output docs/session-kernel-capacity-f2d2.json`。真实 PostgreSQL，原有 bounded 1 KiB receipt workload，每个 size 一次采样。单位为 ms。

| Records | Cold drive | Steady append | Full fold | 额外 provider calls | Lease failures |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5000 | 580.316 | 11.423 | 41.942 | 0 | 0 |
| 20000 | 2326.007 | 17.993 | 326.863 | 0 | 0 |

| Catalog sessions | Runnable items | Full pagination ms | 目标 ms | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 1000 | 1500 | 20.462 | 1000 | PASS |
| 10000 | 15000 | 237.248 | 1000 | PASS |

`--assert-capacity` 全部通过；没有追加 provider 调用或 lease failure。这是一次容量采样，不是持续生产负载分布。

## 最终门禁（F2d-2b）

| 门禁 | 结果 |
| --- | --- |
| `uv run python scripts/contract_snapshot.py check` | PASS：契约 23.0.0，55 fixture files，manifest 未变 |
| `uv run ruff format --check .` | PASS：386 files |
| `uv run ruff check` | PASS |
| `uv run ty check` | PASS |
| `uv run pytest tests/session -q`（本地 PG） | PASS：1221 passed in 276.52s (0:04:36) |
| `VV_AGENT_TEST_REDIS_URL=redis://127.0.0.1:6400/15 uv run pytest` | PASS：3696 passed, 20 skipped, 18 warnings in 431.63s (0:07:11) |
| 三次独立 overhead，`--runs 200 --warmup 10` | PASS：每次四场景均满足 single-turn ≤50 ms、ten_turns ≤80 ms，无新存活线程 |
| M6 5000 / 20000，`--samples 1 --assert-capacity` | PASS：cold drive / steady append / catalog 全部断言通过 |
| `git diff --check` / 公共与默认入口边界 | PASS：HEAD `9c5eafe`，仅当前树内部 product/tests/docs 改动，无提交 |
| 测试 Redis 6400 清理 | PASS：核对 gate-owned process ID 后 `shutdown nosave`，端口不再响应 |

命令从 repo root 使用 repo-managed uv 环境执行；`UV_CACHE_DIR=/tmp/f2d2b-uv-cache`。Redis 用 `redis-server --port 6400 --daemonize yes --save ""` 启动，额外绑定 loopback、目录/日志/PID 指向 `/tmp`；只关闭这一实例。PG fixtures 建立并删除独立 disposable databases。

全量 20 个 skip 为 6 个未开启 live provider 用例、4 个跨 runtime/store 用例、9 个不适用所选非 Redis backend 的参数变体、1 个环境 symlink 用例。没有因缺少适用的 Redis 或 PG 而跳过测试。18 个 warning 来自既有 distributed checkpoint 的多线程 fork DeprecationWarning。

未运行 opt-in 真实模型、跨语言探针或 Rust/cargo；没有提交、push 或部署。gen2 GC 和未归因的极端延迟仍存在，三次 p95 验收不代表 max/p99 或未来 F3 的 headroom 保证。上述 13 行缺口与 F3 切换仍未完成。
