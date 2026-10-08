# F2d-3 children / delegation / host bindings / workspace 验收报告

日期：2026-10-09。工作树：`/home/makerbi/vectorvein/tmp/wt-f2-vv-agent`，分支 `feat/session-kernel-internal`，基线 `2e87598`（F2d-2 与 F2d-2b）。

指定九行已关闭，其中六行为 `done (intentional difference)`。整体矩阵为 63 行：59 done、2 partial、2 missing；F3 仍被交互、CLI 与 App Server 的四行阻塞。

新增内部 `delegation.py`、`bindings.py`，复用 `create_child`、`child_outcome`、`verify_completion` 与 completion inbox。四种委派入口共享 admission / independently scheduled child / terminal projection，差别仅为定义组装和结果格式。没有递归 Runner、共享父 lease 的嵌套 driver、第二本子任务账本或对象 pickle。Runner 的配置/纯结果 helper 仍被复用，执行入口与默认 wiring 未切换。

没有公共 exports、contract lock / vendored fixture 变更，没有修改 `local_settings.py`；无 Rust/cargo、提交、push、发布或部署。非 session 源码改动仅为本报告列出的内部 helper 清理及其既有调用者更新。

## 关闭行与配对证据

`D` 为 `tests/session/test_delegation_parity.py`，19 个测试函数、159 个参数化用例。所有持久化场景使用真实 PostgreSQL 18.6、SQLite 文件和 SQLite `:memory:`；引用公共 Runner、Agent.as_tool、BackgroundAgentTask.start 和 handoff 的真实 producer，脚本模型仅替代网络。S3 使用仓库既有 `_FakeS3Client` / `_make_s3_backend`，无需真实凭据或网络。纯 schema / ownership 校验为补充证据。

| 矩阵行 | 状态 | D 测试名称 |
| --- | --- | --- |
| Child session atomic admission / delivery / cancellation | done (intentional difference) | `test_configured_child_atomic_sdk_restart_parity`, `test_sdk_completion_delivery_failure_cut_and_replay_parity`, `test_sdk_parent_cancellation_cascades_after_reconstruction`, `test_completion_projection_uses_authenticated_turn_not_later_continuation` |
| Built-in create_sub_task with configured agents | done (intentional difference) | `test_configured_child_atomic_sdk_restart_parity`, `test_configured_async_admission_status_and_late_delivery_parity`, `test_configured_child_argument_failure_parity`, `test_blocking_child_user_wait_is_durable_intentional_difference` |
| Built-in sub_task_status | done | `test_sub_task_status_records_projection_parity`, `test_sub_task_status_message_continuation_and_wait_projection_parity` |
| Configured sub-agents inheritance | done | `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers` |
| Agent.as_tool | done (intentional difference) | `test_sdk_children_never_recursive_runner_or_parent_lease`, `test_child_hooks_policy_budget_workspace_inheritance_frozen`, `test_child_inherited_denial_and_budget_execute_real_producers`, `test_blocking_child_user_wait_is_durable_intentional_difference` |
| BackgroundAgentTask start / poll / wait / cancel | done (intentional difference) | `test_background_handle_start_poll_wait_cancel_reconstruction`, `test_sdk_parent_cancellation_cascades_after_reconstruction` |
| Handoff / maximum-handoff enforcement | done (intentional difference) | `test_handoff_durable_transfer_and_maximum_parity`, `test_handoff_target_validation_state_and_events_parity` |
| shared_state arbitrary host objects | done (intentional difference) | `test_shared_state_host_binding_restart_intentional_difference`, `test_host_binding_json_boundary_rejects_invalid_state` |
| Workspace S3 / streaming backend | done | `test_s3_streaming_workspace_paired_producers` |

新配对同时检查同步单个 / 批量、异步单个 / 批量的工具 envelope，错误参数、未知代理和 discovery regex；配置子代理初始 JSON state、Agent.as_tool state 继承、system/user prompt、主任务摘要与 output requirements；真实 denied child tool 与不足的 child batch budget；后台 running / zero-timeout / completed / cancelled 句柄；handoff 0/1/2 上限、重建时提高配置上限仍不能越过 admission 的原上限、target guardrail 阻断 / 输出改写 / state 修改与事件 metadata。

配置子代理还冻结并配对验证 parent task 的 shell、shell priority、environment、outside-workspace policy、language 与 skills metadata；重建不采用宿主后来改写的这些值。

事件、任务和 session 的动态身份及时间戳按生产者分别生成，不作为字面字符串比较；结果内容、状态、错误、完成原因及相应预算字段逐项比较。既有 child lifecycle 的 parent-log 身份差异沿用先前评审结论。`sub_task_status` 的每次投影由记录重建；后台 start 指模型工具的原子 child admission，句柄取回使用内部 `Runtime.child_tasks(...).handle(id)`。本轮不改公共 BackgroundAgentTask 的进程内 handle registry 或 start wiring。

## Durable 边界

- 子创建、初始 inbox、父 op_started、op_parked，以及异步 admission 的 op_completed 同事务提交。配置与 child task 在 admission 前编译并冻结；重建直接使用该 task，SubAgentConfig 的后续修改不改变已接收的任务。child principal 继承父 session。
- 批量 `siblings` 各自持有 child session/turn/generation，与同一个 parent operation/attempt 关联。每个成员必须有认证 terminal input；全部满足才完成 blocking parent operation。fold 的 disposable completion index 来自 input_applied，fork/snapshot 与冷重建均不丢失或共享可变容器。没有 child completion 的普通会话使用 None，不在每次 fork/snapshot 新建空 map；真实 child receipt 后才创建索引。既有 hot_path_cache 用例与 D 的三库冷重建用例验证这两条路径。
- delivery 的 before_parent_inbox / after_parent_inbox / after_child_ack 三个失败点整体 rollback，重建后可重新投递。相同 input ID / canonical bytes 重放，不同 bytes 冲突；别名重复、伪造 terminal 或错误 parent/generation 的既有矩阵继续通过。
- 父取消向同步、批量、异步、Agent.as_tool、background children 写 targeted cancel inbox。子 driver 自己传播给 descendants。晚到 completion 保留 audit 结果，不能恢复已结束父 turn 或重复 terminal。
- 父结果来自认证 handle 所指向的原始 terminal turn；status / public snapshot 投影最新 active / completed turn。因此子任务在父投递消费前续跑，也不能替换已认证的原始结果。原 child_delivery cursor 只投递初始 turn 一次。
- 状态消息、续跑和用户回复使用稳定 inbox ID；相同 ID / 内容重放原输入，内容冲突拒绝。状态工具的 provider thread 使用独立 PostgreSQL 连接，避免与父 driver 的事务嵌套。等待只轮询记录，host 独立调度子 session。
- arbitrary host objects 只由 `Runtime.host_bindings` 提供。tool/hook 获得原引用，提交状态剔除这些对象并通过 JSON codec。缺失绑定在恢复前确定性失败；不允许与 JSON key 重叠、替换或删除对象绑定。对象自身的变更不具备事务 rollback 保证。
- S3 配对包含 80,000 行 Unicode 大对象、真实 read_file / write_file、最大 64 KiB source read、source stream close、读取后 Runtime 重建以及最终远端对象 bytes 对比。S3 backend/client 是宿主重新提供的配置，不序列化客户端或凭据；小型 write_file 的既有 post-write read_bytes 行为保持原样。

## 有意差异与 C1 候选

先前替换计划最终“评审记录”接受的取消、child event 身份、memory / budget / endpoint 等差异继续沿用，不重新请求决策。以下为本轮内部语义与字段的新增待决项，本轮不发布契约 24 或执行 F3。

1. **Blocking child WAIT_USER**：Runner 可以向父返回中间等待结果（configured 路径为 sub_task_wait_user），继续父模型并完成父任务。kernel 保持原 child wait，宿主向子 session 发回复后才采用 terminal delivery，避免重启后把非终态作为已完成子任务。普通工具的 WAIT_USER receipt 使用既有 turn_parked 保留并恢复，不重复 handler。
2. **后台 admission 与句柄**：kernel 的 start receipt 固定为 running，避免 Runner 的异步线程可能已完成 / 仍运行的 snapshot race 进入不可重复的 admission。poll/wait 读取当前 records，cancel 写 inbox；重建重新取句柄。公共 Runner 句柄仍为进程内对象。本轮适配的是该句柄行为与既有 BackgroundAgentTaskSnapshot 类型，未替换公共 start 入口。
3. **Durable handoff**：transfer 为同一 child 机制上的 terminal continuation。父/source 保留日志及运行身份，目标输出、agent/model 与 JSON state 从目标终态投影；source 不再执行模型，也不把 Runner 的临时 transfer marker 作为用户 final output 验证。目标的输入/输出验证在执行时各自生效。计数和最大值来自 admission records；达到上限是 durable failed result，Runner 可以直接抛 RuntimeError。source-marker output guardrail 的差异由配对用例明确检查，须由 C1 reviewer 决策。
4. **Host binding**：Runner 的 Python shared_state 可含任意对象；kernel 持久化只保留 JSON 和 required binding names，缺失时抛 MissingHostBinding，不静默返回 None。结果 projection 也仅有 JSON。恢复宿主重新提供引用，这不是旧对象的序列化/跨进程续存承诺。
5. **SDK 取消**：Runner 的 Agent.as_tool 路径可直接抛 CancelledError；kernel 写 durable cancelled terminal，公开结果投影为 FAILED / CANCELLED，并以 inbox / audit 追踪子结果。这是既有取消差异在完整委派 producer 上的补证。

没有新增 record kind 或 inbox kind（仍为 14 / 8）。新增内部字段如下，全部属于 C1 候选；没有旧 shape reader、迁移或兼容 alias：

| 字段 | 闭合形状 / 校验 |
| --- | --- |
| session_created.attributes.child_admission | mode（configured / agent_as_tool / background_task / handoff）、selector、definition / SHA-256 definition_digest、budget、handler_version、sub_config、exclude_files_pattern、handoff_count、max_handoffs、handoff_metadata；顶层拒绝额外字段，SubAgentConfig 字段闭合；definition、budget 和 metadata 使用既有显式 JSON descriptor / host-data 形状 |
| op_parked.handle.siblings（可选） | session_id / turn_id / generation 数组；成员形状闭合、身份唯一，每个成员的原始 handle 与 child creation 关联验证 |
| op_parked.delegation（可选） | mode / agent_name / handoff_count / max_handoffs / metadata；闭合 control 字段，metadata 为明确的 JSON extension 对象；供纯 records event projection |
| task.metadata.session_host_binding_names | 显式 process-local binding 的非空字符串名列表，编译排序；对象不进入日志，恢复缺失 / JSON key 重叠拒绝 |
| task.metadata.session_max_handoffs / session_handoff_targets | handoff source 的冻结上限与 tool-to-target-name 映射；child admission 保存继承计数/上限，重建不得使用新的更高配置覆盖它 |

`ExecutionState.child_completions` 是 disposable 派生索引，没有新增持久化字段。没有修改 contract lock / snapshot；本轮测试证明内部 producer 行为，不是 canonical contract adoption 的替代。

## F2d-2 清理

| 原依赖 | 处理 |
| --- | --- |
| VvLlmClient._preferred_endpoint_id / _ordered_targets | kernel 改用 ordered_targets(preferred_endpoint_id)，不再读取/写入 client 私有 preference，也不为排序复制 client；Runner 的无参路径保持原行为 |
| BudgetEvaluator._model_call_start | 改为窄 model_call_start 方法，cycle_start / Runner / kernel 调用者同步更新，计账逻辑未改 |
| 跨模块 memory/provider 私有 helper | before / after helper 改为 call_before_memory_providers / call_after_memory_providers；cycle_runner 和 session 复用，原有错误处理未变 |
| kernel.py inline imports | 提升到模块 import；kernel.py 无 inline import |

Runtime 的既有 Runner 配置/纯 helper 依赖以及避免循环依赖的局部 imports 仍存在，属于切换阶段的 extraction 范围。本轮没有变更 Runner 执行行为。

## 剩余缺口

| 状态 | 行 | 后续范围 |
| --- | --- | --- |
| partial | Interactive steer / follow-up / resume / archive / close facade | F2d-4 / F3 |
| missing | CLI single-run / stream / persistent sessions | F2d-4 / F3 |
| missing | App Server thread / turn / approval / replay / non-text input | F2d-4 / F3 |
| partial | App Server model/list、schema、TypeScript export | F2d-4 / F3 |

## 三次独立 overhead

三次分别启动独立进程，命令为 `UV_CACHE_DIR=/tmp/f2d3-uv-cache uv run python scripts/session_kernel_overhead.py --runs 200 --warmup 10`，与 pytest 和 M6 分开串行执行。保留正常 GC、SQLite schema/admission、durable writes、drive、final read 和 connection/thread cleanup；没有移出计时范围的工作。children 场景为父模型调用 create_sub_task，释放父 lease 后独立驱动一个 child，提交 completion delivery，再恢复父模型。

p50/p95 为 kernel 分位数减 Runner 相应分位数，不是逐次差值分位数；p95 使用 nearest-rank。单轮 / children 门槛 50 ms，ten_turns 门槛 80 ms。下面单位为 ms，单元格为 added p50 / added p95。

| 场景 | 第 1 轮 added p50 / p95 | 第 2 轮 | 第 3 轮 | 门槛 p95 |
| --- | ---: | ---: | ---: | ---: |
| no_tool | 9.480 / 9.963 | 9.542 / 9.819 | 9.610 / 9.859 | 50 |
| two_tools | 17.281 / 17.942 | 17.886 / 18.285 | 17.771 / 18.529 | 50 |
| ten_turns | 56.057 / 61.253 | 54.755 / 57.550 | 54.117 / 54.632 | 80 |
| start_cancel | 9.737 / 10.027 | 10.412 / 10.760 | 10.347 / 11.198 | 50 |
| children | 42.179 / 43.985 | 42.892 / 45.046 | 42.127 / 44.424 | 50 |

| 轮次 / 场景 | Runner p50 / p95 (ms) | Kernel p50 / p95 (ms) | Runner / Kernel RSS Δ (bytes) | 结束线程 Runner / Kernel | 新增存活线程 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 / no_tool | 3.191 / 3.541 | 12.671 / 13.504 | 53248 / 839680 | 1 / 1 | 0 |
| 1 / two_tools | 5.188 / 5.729 | 22.469 / 23.671 | 8192 / 110592 | 1 / 1 | 0 |
| 1 / ten_turns | 33.255 / 35.425 | 89.312 / 96.678 | -962560 / 16384 | 1 / 1 | 0 |
| 1 / start_cancel | 4.415 / 5.144 | 14.151 / 15.172 | 253952 / 32768 | 1 / 1 | 0 |
| 1 / children | 7.421 / 8.350 | 49.601 / 52.334 | 0 / 0 | 1 / 1 | 0 |
| 2 / no_tool | 3.461 / 3.990 | 13.003 / 13.808 | 8192 / 802816 | 1 / 1 | 0 |
| 2 / two_tools | 5.640 / 6.345 | 23.526 / 24.630 | 8192 / 57344 | 1 / 1 | 0 |
| 2 / ten_turns | 36.476 / 38.144 | 91.231 / 95.694 | -962560 / 4096 | 1 / 1 | 0 |
| 2 / start_cancel | 4.511 / 5.267 | 14.923 / 16.027 | 249856 / 57344 | 1 / 1 | 0 |
| 2 / children | 7.335 / 8.323 | 50.227 / 53.368 | 0 / 0 | 1 / 1 | 0 |
| 3 / no_tool | 3.355 / 3.973 | 12.965 / 13.832 | 8192 / 806912 | 1 / 1 | 0 |
| 3 / two_tools | 5.516 / 6.123 | 23.287 / 24.653 | 8192 / 102400 | 1 / 1 | 0 |
| 3 / ten_turns | 35.106 / 37.715 | 89.223 / 92.346 | -962560 / 16384 | 1 / 1 | 0 |
| 3 / start_cancel | 4.318 / 4.849 | 14.665 / 16.047 | 237568 / 61440 | 1 / 1 | 0 |
| 3 / children | 7.295 / 7.916 | 49.422 / 52.340 | 0 / 0 | 1 / 1 | 0 |

三次的 Runner / kernel 结束线程均为 1，新增存活线程均为 0。children 最小 p95 余量 4.954 ms；ten_turns 最小余量 18.747 ms，added p50 为 56.057 / 54.755 / 54.117 ms。RSS 是整批结束并 GC 后 current RSS 差值，不是峰值。原始最后一轮结果保存在 [session-kernel-overhead-f2d3.json](session-kernel-overhead-f2d3.json)；本报告保留三轮关键数字。这是本机 p95 门禁，不建立 p99/max 或生产 provider latency 承诺。

第一版 children 完整 200-run 测量 added p95 为 63.663 ms，未通过。profile 发现重复 child creation / state 读取、重复 SDK result projection、仅做投影却构造工具 registry，以及父子 session 互相驱逐单项 prefix。最终路径复用同一 validated snapshot / creation，projection Runtime 延迟初始化 registry，child admission 的 definition canonical bytes 经 digest 校验后复用到嵌套 envelope 编码。store 保留最多两个已验证 prefix，第三个 session 驱逐较旧项；普通单 session 路径不新增 map 或 per-op 对象图。两个 prefix 每次仍绑定数据库 head / digest / epoch，rollback 与显式 disposal 同时失效。

`test_two_session_prefixes_reuse_validated_bytes_with_bounded_eviction` 和 `test_displaced_parent_prefix_still_checks_rollback_and_database_bytes`（三库，共 15 用例）验证复用、有限驱逐、回滚、同 head 替换、schema / stored-digest 篡改和冷重建。D 的原始 terminal 投影测试也断言恢复父 driver 只构造父 registry，不为已完成的 child 再构造 registry。没有新增持久化缓存或削弱 lease/CAS/closed schema / digest / input authenticity 检查。

## M6 capacity

真实 PostgreSQL，原有 bounded 1 KiB receipt workload。命令为 `UV_CACHE_DIR=/tmp/f2d3-uv-cache uv run python scripts/session_kernel_benchmark.py --sizes 5000 20000 --samples 1 --assert-capacity --output docs/session-kernel-capacity-f2d3.json`。

| records | logical bytes | committed append (ms) | steady append (ms) | full fold (ms) | cold read/fold (ms) | cold drive (ms) | provider 增量 / lease failure |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5000 | 4241706 | 6.900 | 8.907 | 37.641 | 414.139 | 536.210 | 0 / 0 |
| 20000 | 16958334 | 12.526 | 16.907 | 316.295 | 1981.159 | 2209.252 | 0 / 0 |

| sessions | runnable items | page size | first page (ms) | late page (ms) | full pagination (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1500 | 100 | 2.426 | 1.465 | 19.299 |
| 10000 | 15000 | 100 | 2.487 | 1.445 | 214.308 |

所有 assert-capacity 条件通过：cold drive ≤ records 数值对应的 ms 限额、steady append ≤50 ms、全分页 ≤1000 ms；无 lease failure，没有补发 provider 调用。lease TTL 15000 ms、heartbeat 0.25 s 保持原设置。原始结果为 [session-kernel-capacity-f2d3.json](session-kernel-capacity-f2d3.json)。

一次容量采样，不是持续生产负载分布。

## 最终门禁

| 命令 / 门禁 | 最终结果 |
| --- | --- |
| python3 scripts/contract_snapshot.py check | PASS：23.0.0 / ad2d4974545f987e237aed421cc4f65680e9a8dc，55 fixtures，manifest digest 未改 |
| uv run ruff format --check . | PASS：389 Python files |
| uv run ruff check | PASS |
| uv run ty check | PASS |
| uv run pytest tests/session -q，真实本地 PG | PASS：1396 passed，301.45 s，无 skips |
| VV_AGENT_TEST_REDIS_URL=redis://127.0.0.1:6400/15 uv run pytest | PASS：3871 passed / 20 skipped / 18 warnings，432.45 s；真实 Redis 8.10.1，真实 PG 18.6 |
| overhead --runs 200 --warmup 10 | PASS：三次连续独立进程，5 场景均满足 50 / 80 ms 门槛；无新增存活线程 |
| M6 --sizes 5000 20000 --samples 1 --assert-capacity | PASS：两种 history size、两种 catalog size，退出码 0 |
| git diff --check / scope | PASS：HEAD 2e87598，无 commits；default wiring、public exports、contract lock / fixtures、local_settings.py 和 Rust 未改 |
| Redis cleanup | PASS：只关闭本轮 PID，6400 端口已关闭 |

20 个 skips：6 个 opt-in live provider、4 个 cross-runtime / cross-language 探针、9 个仅适用于真实 Redis 的检查在 memory / SQLite / fake Redis 参数分支跳过，以及 1 个目录 symlink 能力检查。上述 Redis 检查的 real_redis 分支实际执行通过。18 个 warnings 均来自既有 distributed checkpoint 多线程 fork 的 DeprecationWarning。本轮没有要求真实 S3 或 provider 凭据。

优化后目标回归为 358 passed（77.68 s），覆盖 child / cache / capacity / validation；最终完整 session 和 full gates 均在性能验收相同源码上重新执行。

最终检查没有提交或变更默认入口；F3 仍被剩余四行及 canonical C1 决策阻塞。
