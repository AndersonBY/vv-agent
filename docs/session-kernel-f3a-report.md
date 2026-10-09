# F3a：v24 adoption 与单一 session kernel

F3a 本地范围已完成。唯一保留失败是严格的 45-file fixture byte comparison，差异完全属于 reviewer 已裁定的一个 producer evidence defect。v24.0.1 尚待发布及同步；当前 lock 和 vendored snapshot 保持 v24.0.0，不宣称中央 verified adoption。F3b 仍须迁出共享依赖并删除旧栈，随后才能形成原子 F3 PR。

工作树为 `feat/session-kernel-f3`，HEAD 为 `9e43dbf1e6bc2fd0b2502141d0183e1845aa533f`。本轮没有提交；Rust、中央 contract 工作树未修改。性能基线源码与 lock 未修改，最终 Git 状态 clean，辅助命令新生成的 .venv 已移除。

## 1. Adopt v24

`contract.lock.json` 选择 24.0.0，实际 checkout revision 为 `a7a6df885a1397c3ca7e8dccfd36215ee17dafc7`，artifact SHA256 为 `5288fc7cbd2776e6c847a4c9b78998ffe1361aae37f65dd7bd779101dd76d1b2`。任务中的 `889aa2b` 是较早的短 revision；sync 使用实际完整 revision。`contract_snapshot.py check` 通过：53 files、52 manifest entries。所有 vendored 变动来自 snapshot sync，没有手改 fixture。

## 2. Flip：全部入口使用同一 kernel

| 入口 | 当前 producer / 生命周期 |
| --- | --- |
| Runner run/run_sync/start/stream_sync/compiled path | SessionDriver、kernel、RunHandle；普通运行拥有 SQLite `:memory:` |
| Runner / ConfiguredRunner resume | 显式 `session_id, turn_id`；读取保留的 turn、绑定原 host runtime |
| ConfiguredRunner | 合并 defaults 后进入同一个 Runner facade |
| RunHandle | kernel drive、事件/trace 投影、实时 delta、取消/审批 inbox；已提交 terminal 不重复 admission |
| AgentSession / InteractiveAgentClient | retained session/turn、creation-time seed；消息和状态只读投影；支持 caller 提供 SessionStore/SQLiteStore/PostgresStore |
| CLI | 编译后进入同一 kernel compiled path |
| AppServer / MessageProcessor / RunAdapter / ThreadStore | thread=session、turn=kernel turn；JSON-RPC v2；thread/turn/item/status/replay/approval 都由保留记录投影 |
| create_sub_task / configured children | 原子 child admission、retained child handles、terminal delivery 和 status；无旧 registry executor |
| Agent.as_tool / BackgroundAgentTask / handoff | kernel child adapters；handoff、异步 admission 和后续 continuation 使用保留身份 |

已移除 `_kernel` 参数、`hasattr(...kernel)` selector、`_SessionKernel` 间接类与旧 Runner loop/recovery/backend 分支。RunEvent 只接受 v6，model-call 只接受 v2（含 `output_repair`），task-token-usage 只接受 v3，Message 使用严格 kernel codec；旧版本拒绝。AgentResult/RunResult 记录 session/turn identity，退休 checkpoint_key/resume_observations。RunHandle 的 v8 签名拒绝非空旧 state/token/payload。AgentSession 的 replace_messages、replace_shared_state、clear_queues 和可写 session 已退休，使用闭合 `{messages, shared_state}` creation seed。App Server 统一 ThreadStatus，并拒绝 closed thread 的 execution resume。

取消会在同一事务将 `parent-cancel/{sid}/{tid}/{operation}/{attempt}` 写入全部保留 child handles；实时 model delta 同时进入 RunHandle 和既有 stream observer。ChildTasks.get 在一次 snapshot 内读取 outcome，并先读取 pending inbox，防止 follow-up admission 与旧 terminal 读取交错导致 wait 提前返回。

## 3. Public API v8

vendored `public_api.json` v8 的真实 export、member 和 signature resolver 全通过。根 `__all__` 共 182 个名字（180 个直接 domain exports 加 AgentTask、skills）；extra/missing 均为零。§6 retired root symbols 已移除；drive/tick/child_delivery 仅模块限定。RunState-based public resume 退休，没有 wrapper 或 model alias。

## 4. 既有测试迁移及真实 producer evidence

修改 89 个既有测试/support 模块，新增一个 kernel assembly support 模块，删除 17 个退休测试模块。root surface fixture 只运行一次 kernel path，原比较驱动只保留 kernel 行。没有并行旧 suite，没有为 migrated behavior 新增 skip/xfail；退休 checkpoint/store/backend/dispatch/deferred 测试驱动已删除。

保留的 events/terminal/transcript/trace、usage/budget/output validation、children/handoff、memory/compaction、run definition/prompt、App Server protocol/thread/turn/approval/replay/lifecycle/schema 和 public API 均由既有模块断言当前 producer 与 vendored v24。memory/hooks 测试通过 SessionDriver 实际执行，scripted callbacks 仅是模型 doubles；不再调用 CycleRunner。parallel batches 和 manager status 使用真实 retained children；旧 ThreadBackend parallel_map、私有 registry outcome 注入、私有 pending 状态和 checkpoint control 断言已退休。仍保留共享纯 helper 的测试，F3b 提取后应继续保留。

before_llm hook 若移除 read_file，而上下文仍包含 microcompact marker 或 artifact/cursor recovery evidence，kernel 在模型调用前 durable close 为 `microcompaction_recovery_unavailable`；该分支保留实际 producer coverage。child admission/status race 有三个 store backend 的确定性用例。

45-file 测试只运行 ONE generation，随后严格逐字节比较 vendored snapshot；所有 generator self-checks、schemas、coverage 和 real-producer provenance 校验仍运行。没有 fixture 忽略列表、skip 或 xfail。

测试迁移完整清单：

- `tests/conftest.py`
- `tests/session/test_capability_parity.py`
- `tests/session/test_delegation_parity.py`
- `tests/session/test_internal_boundary.py`
- `tests/session/test_recovery_matrix.py`
- `tests/session/test_runner_parity.py`
- `tests/session/test_session_kernel_fixtures.py`
- `tests/session/test_tools_control_parity.py`
- `tests/support/compaction.py`
- `tests/test_after_cycle_hooks.py`
- `tests/test_agent_as_tool.py`
- `tests/test_agent_as_tool_scope.py`
- `tests/test_agent_defaults.py`
- `tests/test_app_server_approval.py`
- `tests/test_app_server_client.py`
- `tests/test_app_server_contract_parity.py`
- `tests/test_app_server_controller_action.py`
- `tests/test_app_server_durable_resume.py`
- `tests/test_app_server_initialize.py`
- `tests/test_app_server_replay.py`
- `tests/test_app_server_schema.py`
- `tests/test_app_server_thread_lifecycle.py`
- `tests/test_app_server_thread_store.py`
- `tests/test_app_server_thread_turn.py`
- `tests/test_app_server_transport.py`
- `tests/test_approval_protocol.py`
- `tests/test_approval_session.py`
- `tests/test_approval_tool_policy_contract.py`
- `tests/test_bash_process_management.py`
- `tests/test_cancellation.py`
- `tests/test_cli_contract.py`
- `tests/test_compiler.py`
- `tests/test_completion_policy_contract.py`
- `tests/test_configured_sub_agent_manager_tool_envelope.py`
- `tests/test_configured_sub_agent_parity.py`
- `tests/test_custom_tools.py`
- `tests/test_cycle_runner.py`
- `tests/test_event_store.py`
- `tests/test_event_store_replay_contract.py`
- `tests/test_event_validation.py`
- `tests/test_events_contract.py`
- `tests/test_guardrails.py`
- `tests/test_handoffs.py`
- `tests/test_interactive_approval_bridge.py`
- `tests/test_interactive_lifecycle_contract.py`
- `tests/test_interactive_memory_provider_bridge.py`
- `tests/test_interactive_session_api.py`
- `tests/test_interactive_session_persistence.py`
- `tests/test_interactive_session_run_handle_bridge.py`
- `tests/test_kernel_reusables.py`
- `tests/test_live_edit_file.py`
- `tests/test_live_search_tools.py`
- `tests/test_live_sub_task_wait.py`
- `tests/test_memory.py`
- `tests/test_memory_lifecycle_contract.py`
- `tests/test_memory_local_contract.py`
- `tests/test_memory_provider.py`
- `tests/test_message_sanitizer.py`
- `tests/test_microcompaction_events.py`
- `tests/test_microcompaction_policy.py`
- `tests/test_model_provider.py`
- `tests/test_output_validation_contract.py`
- `tests/test_parallel_subtasks.py`
- `tests/test_parity_evidence_manifests.py`
- `tests/test_protocol_types.py`
- `tests/test_result_public_contract.py`
- `tests/test_run_budget.py`
- `tests/test_run_config_controls_contract.py`
- `tests/test_run_definition_producer.py`
- `tests/test_run_handle_live_stream.py`
- `tests/test_run_resume.py`
- `tests/test_runner.py`
- `tests/test_runner_defaults_contract.py`
- `tests/test_runner_events_producer_parity.py`
- `tests/test_runner_terminal_contract.py`
- `tests/test_runner_trace_contract.py`
- `tests/test_runtime.py`
- `tests/test_runtime_hooks.py`
- `tests/test_session_graph_events.py`
- `tests/test_streaming.py`
- `tests/test_sub_agent_runtime.py`
- `tests/test_sub_task_manager_continuation_recovery.py`
- `tests/test_sub_task_status.py`
- `tests/test_token_usage_contract.py`
- `tests/test_tool_approval.py`
- `tests/test_tool_metadata_contract.py`
- `tests/test_tool_orchestrator.py`
- `tests/test_tool_planner.py`
- `tests/test_tracing.py`

新增 support：`tests/support/kernel_runtime.py`。它只继承旧 engine 的纯 memory/sub-agent construction helpers，执行始终使用 SessionDriver；这项构造依赖是 F3b 必须先提取的内容。

退休测试模块：

- `tests/test_backends.py`
- `tests/test_checkpoint.py`
- `tests/test_checkpoint_fault_matrix.py`
- `tests/test_checkpoint_history_growth.py`
- `tests/test_checkpoint_history_runner.py`
- `tests/test_checkpoint_history_stores.py`
- `tests/test_checkpoint_reconciliation.py`
- `tests/test_checkpoint_resume_events.py`
- `tests/test_checkpoint_runner.py`
- `tests/test_deferred_tools.py`
- `tests/test_dispatch_outbox.py`
- `tests/test_distributed_checkpoint.py`
- `tests/test_runtime_checkpoint_early_stop.py`
- `tests/test_runtime_checkpoint_history.py`
- `tests/test_runtime_controller.py`
- `tests/test_session_store_parity.py`
- `tests/test_sessions.py`

README、当前 architecture/runtime/App Server/预算/输出验证文档及 examples 同步到当前 API。03/04/18/19/20/21/22/24 共八个 examples 通过 scripted-provider smoke；23 Celery backend example 退休。

## Fixture evidence correction：一个冲突，三个文件

reviewer 已确认 App Server 的 producer defect：task metadata 已记录 provider window/output 为 128000/16384，但 MemoryManager 仍使用默认 279000/null。修正位于 `src/vv_agent/session/surfaces.py:SessionDriver.runtime`，将已解析 provider limits 传给真实压缩器。符合 v24 frozen definition 与 definition_digest 规范；24.0.1 只修正 evidence，不改变 wire shape。

| 字段 | vendored v24.0.0 | 当前 producer |
| --- | ---: | ---: |
| model_context_window | 279000 | 128000 |
| model_max_output_tokens | null | 16384 |
| reserved_output_tokens | 16000 | 16000 |
| autocompact_buffer_tokens | 13000 | 13000 |
| 有效压缩阈值 | 250000 | 99000 |

| 不同文件 | pointer / record | 差异原因 |
| --- | --- | --- |
| session_codec_vectors.json | `/vectors/45/wire`；`turn/thread_1/turn/turn_1/started` | memory limits、definition_digest；派生 bytes_base64 与 sha256 随之变化 |
| session_records.jsonl | 第 46 行；同一 thread_1 record | 同一 corrected definition 和 digest |
| session_projection.json | `/source_records/thread_2/2/wire`、`/source_records/thread_3/0/wire`；两个对应 turn_started records | 相同 App Server memory limit 修正及派生 digest |

thread_1 definition digest 从 `f8344da8305b0da821ad995470c4b2c300184e5af078b1805404fddeb1375cb9` 变为 `193a76ae91aa609060075409373e33d6eba1ca6ccb8d6692e294e988e1b935e7`。list 格式的 `session-kernel-f3a-fixture-conflict.json` 保留全部文件、pointer/record、前后值、根因、src 位置和 normative basis；没有发现新增冲突。

最终完整候选为 `/tmp/f3a-candidate-fixtures/`，第二次独立生成位于 `/tmp/f3a-candidate-fixtures-second/`。两次均 45 文件，目录 diff 为空；42 文件与 vendored 逐字节相同，仅上表三个不同。`/tmp/f3a-candidate-determinism.diff` 为空，generator JSON 报告保存在 `/tmp/f3a-candidate-1.log` 与 `-2.log`。候选没有写回 vendored snapshot。

## 5. LeaseLost backoff / cap / typed failure

`RunHandle._drive_tree` 每次失租先检查 retained terminal；已提交 terminal 直接完成，不创建新 turn。其他失租使用有抖动的指数退避，默认最多 5 次 drive，base=0.01s、单次 cap=0.5s，jitter 区间 `[delay/2, delay]`。attempt/delay/cap 是 Runtime 配置；初始化拒绝 bool、非整数 attempts、非正数、NaN/Infinity 和 base>cap，并向 child Runtime 传播。sleep/jitter 可注入。

耗尽抛 `LeaseRetryExhausted(LeaseLost)`，携带 session/turn/attempts 和 code `lease_retry_exhausted`。Runner、interactive 收到 typed exception；App Server 发明确 `error/warning` 通知并完成 failed attempt。确定性 tests 覆盖三种公共 caller、退避值及封顶、暂时失租成功、已提交 terminal、恢复读再次失租、非法配置和 caller store ownership；测试 sleep 被注入 recorder，没有实际长等待。

## 6. Perf continuity：六次交错独立进程

`session_kernel_overhead.py` 现在只测 absolute kernel p50/p95，保留六个 workload、ScriptedLLM、200 measured runs、10 warmups、GC 与 timing 语义；`--json` 是 `--output` 的同义参数，`rows[].kernel` absolute layout 与基线兼容。App Server 测量区间从 turn/start 到 turn/completed，构造、initialize/thread/start 和 cleanup 在区间外。

正式测量在最终 pytest 之后、M6 之前执行，顺序 base1 → F3a1 → base2 → F3a2 → base3 → F3a3。base 使用只读 `/home/makerbi/vectorvein/tmp/wt-f3-base`、同一 HEAD 9e43dbf；两侧同一 Python 环境，probe 确认从各自 tree/src 导入，关闭 bytecode 写入；两侧测量进程均 taskset 到 CPU 8、9，以固定线程调度与 cache locality，workload/runs/warmup 不变。每组测量均没有残留线程。

下表是全部六次 kernel absolute p95，单位 ms；验收阈值为同 scenario F3a/base≤1.10。

| 场景 | base1 | F3a1 | base2 | F3a2 | base3 | F3a3 | 三对倍率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_tool | 16.577 | 15.196 | 16.239 | 15.682 | 18.898 | 13.608 | 0.917 / 0.966 / 0.720 |
| two_tools | 26.089 | 26.646 | 29.683 | 26.276 | 33.521 | 25.086 | 1.021 / 0.885 / 0.748 |
| ten_turns | 94.680 | 105.676 | 113.351 | 106.118 | 96.864 | 106.228 | 1.116 / 0.936 / 1.097 |
| start_cancel | 18.492 | 18.316 | 16.950 | 17.767 | 16.224 | 17.590 | 0.990 / 1.048 / 1.084 |
| children | 81.759 | 60.592 | 57.761 | 57.899 | 53.283 | 58.426 | 0.741 / 1.002 / 1.097 |
| app_server_turn | 40.517 | 38.503 | 39.415 | 38.529 | 34.029 | 38.326 | 0.950 / 0.978 / 1.126 |

基线旧脚本第 1 次退出码为 1，原因是其旧 kernel-minus-Runner added-p95 门槛超限；六个 workload 正常完成且无线程泄漏。该旧门槛不是 F3a 的 absolute kernel A/B 验收条件，原值和 exit status 均保留。

逐场景采用三次独立 p95 的中位值作 A/B 比较，六个场景均通过 ≤1.10× 门槛，最大中位倍率为 1.096。最大单对倍率为 1.126，所有单对原值见表，不将噪声单对记为全部通过。中位倍率：no_tool=0.917；two_tools=0.885；ten_turns=1.096；start_cancel=1.048；children=1.012；app_server_turn=0.977。

完整 absolute p50/p95、RSS、threads 和各次 exit status 位于 `/tmp/f3a-perf-ab.json` 及 `/tmp/f3a-perf-{base,f3a}-{1,2,3}.json`。补迁移前一轮保留在 `/tmp/f3a-perf-before-final-pytest/`，start_cancel 第1对为 1.249×。最终 pytest 后的未固定 CPU 轮保留在 `/tmp/f3a-perf-unpinned-post-pytest/`，App Server 的三轮 p95 中位倍率为 1.215×，未过门槛。该轮中后续测量的各场景延迟整体上升；单 CPU 3 轮保留在 `/tmp/f3a-perf-cpu3-post-pytest/`，ten_turns 的三轮 p95 中位倍率为 1.112×，未过门槛。预热后的十轮 profile 显示两侧 step/commit/SQLite execute 次数完全相同，总函数调用数差约 0.03%；没有发现新增重复执行。最终采用固定两侧 CPU 8、9 affinity 后重新完成的六次交错测量，没有修改 producer 或 benchmark workload 来绕过差异。

## 门禁

| 门禁 | 最终证据 | 结果 |
| --- | --- | --- |
| ruff format --check | 383 files already formatted | PASS |
| ruff check | 全仓 All checks passed | PASS |
| ty check | 全仓 All checks passed | PASS |
| contract_snapshot.py check | 24.0.0；53 files / 52 entries | PASS |
| 45-file ONE generation + strict byte comparison | 42 matches；3 corrected-evidence differences；无豁免 | 唯一允许失败 |
| tests/session -q + local PG | 1481 passed / 1 known failure；446.89s | 按 amended rule 通过 |
| full pytest + Redis 6400/15 + local PG | 2963 passed / 1 known failure / 7 existing skips；546.06s | 按 amended rule 通过 |
| 补迁移窄检查 | 84 passed | PASS |
| CLI contract / App Server CLI | 43 passed | PASS |
| stdio App Server smoke | v2 → thread/start → turn/start → completed；exit 0 | PASS |
| examples scripted smoke | 8 current examples passed | PASS |
| base/F3a interleaved ×3 | 6 scenario median p95 ratios ≤1.10×；max=1.096 | PASS |
| M6 capacity | PG history100/1k/5k/20k；sessions1k/10k；取消≤2s | PASS |
| git diff --check | 无 whitespace errors | PASS |

full pytest 的 7 个 skips 均为既有环境门禁：六个真实 provider opt-in 和一个 directory symlink unavailable。独立 tests/session 没有 skips。唯一 failure 的精确名称为 `tests/session/test_session_kernel_fixtures.py::test_generated_fixtures_match_vendored_snapshot`，其三个差异文件与冲突 JSON 完全一致。

CLI 采用既有 CLI contract 与 App Server CLI tests（43 passed）；stdio smoke 使用 scripted model，实际 initialize v2 → thread/start → turn/start → turn/completed，输出 `smoke done`，退出 0，stderr 为空。原始消息为 `/tmp/f3a-app-server-smoke.jsonl`。

Redis 使用专用 127.0.0.1:6400/15，full pytest 启动前 PONG；PG 使用本地 admin 连接和 disposable test databases。结束后关闭该专用 Redis，并确认端口不再接受连接。

## M6 容量

最终 PostgreSQL benchmark 独立执行 `scripts/session_kernel_benchmark.py --assert-capacity`，默认 7 samples、最多 3 cold samples、默认 lease 15000ms/heartbeat 0.25s。测量包含实际 durable transaction；cold 使用新连接、新 Runtime，但 OS/PG buffers 仍 warm。历史由保留模型回执及有界 1KiB tool receipts 组成，是容量下界。

| M6 场景 | steady append median / ms | cold drive max / ms | gate |
| --- | ---: | ---: | --- |
| 100 records | 4.935 | 48.195 | PASS |
| 1000 records | 5.660 | 142.603 | PASS |
| 5000 records | 6.890 | 669.173 | PASS |
| 20000 records | 15.664 | 2278.142 | PASS |
| 1000 sessions full scan | — | 10.635 | ≤1000ms PASS |
| 10000 sessions full scan | — | 247.885 | ≤1000ms PASS |

无 LeaseLost，没有 provider redispatch，capacity assertions 全通过。原始数据为 `/tmp/f3a-m6.json`。

5000-record 长日志取消的四个实际测量（PG/SQLite × cooperative/noncooperative）：

| 用例 | latency / s |
| --- | ---: |
| test_long_log_cancel_5000[postgres-False] | 0.222 |
| test_long_log_cancel_5000[postgres-True] | 0.222 |
| test_long_log_cancel_5000[sqlite-False] | 0.249 |
| test_long_log_cancel_5000[sqlite-True] | 0.240 |

全部 ≤2s；long-log zombie/fencing、2000-record cold drive 及 cache ownership/invalidation 的断言同样通过。

## F3b：旧执行路径模块与先提取内容

以下 32 模块合计 27146 物理行，旧执行路径已无法从公共入口到达。它们仍有共享值、纯 helper 或 type-only consumers；行数不是可直接整文件删除的证明。

| 路径（src/vv_agent/） | 物理行数 |
| --- | ---: |
| `checkpoint.py` | 1245 |
| `deferred.py` | 232 |
| `sessions/__init__.py` | 16 |
| `sessions/base.py` | 380 |
| `sessions/memory.py` | 73 |
| `sessions/redis.py` | 135 |
| `sessions/sqlite.py` | 316 |
| `runtime/checkpoint_codec.py` | 386 |
| `runtime/checkpoint_history.py` | 258 |
| `runtime/checkpoint_resume.py` | 2616 |
| `runtime/controller.py` | 967 |
| `runtime/dispatch_outbox.py` | 374 |
| `runtime/state.py` | 2839 |
| `runtime/run_definition.py` | 323 |
| `runtime/sub_task_manager.py` | 966 |
| `runtime/sub_task_identity.py` | 40 |
| `runtime/model_calls.py` | 482 |
| `runtime/backends/__init__.py` | 28 |
| `runtime/backends/base.py` | 30 |
| `runtime/backends/celery.py` | 1063 |
| `runtime/backends/celery_tasks.py` | 1014 |
| `runtime/backends/distributed.py` | 1842 |
| `runtime/backends/inline.py` | 117 |
| `runtime/backends/thread.py` | 132 |
| `runtime/stores/__init__.py` | 5 |
| `runtime/stores/controller_store.py` | 1737 |
| `runtime/stores/memory.py` | 655 |
| `runtime/stores/redis.py` | 2458 |
| `runtime/stores/sqlite.py` | 2700 |
| `runtime/engine.py` | 2555 |
| `runtime/cycle_runner.py` | 610 |
| `runtime/tool_call_runner.py` | 552 |

先提取或替换：

1. Runner 当前配置/精确模型解析、guardrails、输出验证/coercion、child prompt、消息差量与 tracing helpers；保留校验及错误语义，删除旧 executor 依赖。
2. interactive lifecycle/subscriptions/value objects、`_SubTaskTurnSnapshot`；替换旧 Session 类型引用和退休 queue fields。
3. App Server input/approval/terminal formatting、严格 controller validation、ThreadRecord/TurnRecord/ThreadSnapshot；替换 processor 的 CheckpointError 引用。
4. tools/outcomes.py 的当前 provider outcomes、definitive result 与 host interaction normalization；迁移 registry/executor/orchestrator consumers 后删除 Deferred* 遗留类型。
5. SubTask identity 纯 normalizer、ChildTasks.tool_manager 的 SubTaskManager type-only cast；替换旧身份 ContextVar。
6. tests/support/kernel_runtime.py 的纯 memory/sub-agent construction helpers；MemoryManager control error 的 ModelCallBudgetExhausted 等共享类型；distributed 模块中的纯 toolset/tool schema digest helpers 及其现有 registry/planner tests。保留这些当前能力，删除旧 backend capability resolution。
7. 明确 kernel 跨模块私有接口：LLM endpoint selection、budget accounting 和 memory provider hooks；提供窄的当前接口后删除旧 owner。
8. 归档或删除旧 checkpoint 使用文档、F1/F2 分轮证据，保留一份当前性能/容量基准；清理 test manifest 中退休 surface mapping scaffolding。

ToolCallRunner 的 image notification、skipped result、tool-use behavior 三个纯 helper 已迁到 `runtime/tool_results.py`。compiler、context、hooks、lifecycle、cancellation、token_usage、tool_planner、process manager 和 memory 是当前共享能力，不能整删 runtime 目录。

## 24.0.1 同步（reviewer 追记，2026-10-09）

契约 v24.0.1（revision `9ce5dd26cb689eee94ab005f18e5c927f676a8d4`，artifact SHA256 `15cde7bb23ca1d51640bea2ef048500dd1487914703129edea9837ddcf8cd4c0`）已发布并经 `contract_snapshot.py sync` 同步。三份修正 fixture 与当前 producer 输出逐字节一致，上文所述的 fixture byte comparison 保留失败随之消除。
