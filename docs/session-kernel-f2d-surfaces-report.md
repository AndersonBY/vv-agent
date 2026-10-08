# F2d-4 interactive / CLI / App Server 验收报告

日期：2026-10-09。工作树：`tmp/wt-f2-vv-agent`，分支 `feat/session-kernel-internal`，基线 `a4325c215f69a21867ea7f72f4855d44a343d182`（F2d-3）。

四个入口能力行已实现。内部选择统一为非导出的 `_SessionKernel` owner，通过私有 `_kernel` 参数交给 interactive、CLI 或 App Server；没有环境开关。默认仍执行现有路径，普通 kernel 运行使用 SQLite `:memory:`，持久会话使用同一 SQLiteStore 的文件模式。contract 23 lock、55 个 vendored fixture、公共 exports 和模型默认值均不变；本轮没有 Rust、提交、推送或发布。

## 能力关闭与既有测试

根 `tests/conftest.py::surface` 将既有行为测试参数化为 `{current, kernel}`；没有平行入口套件。旧 checkpoint 专属驱动仍验证旧默认路径；kernel 的恢复与控制差异在同一既有测试模块内显式断言。协议值对象与 mapper 的独立黄金向量继续验证固定 payload，不伪装成两次运行入口。

| 矩阵行 | 状态 | 真实 producer / 既有测试中的证据 |
| --- | --- | --- |
| Interactive steer/follow-up/resume/archive/close | done (intentional difference) | `test_interactive_real_same_turn_user_reply`、`test_interactive_live_steer_and_durable_follow_up`、`test_interactive_child_wait_reply_keeps_child_identity`、`test_kernel_file_facade_rebuild_resume_and_control_identity`、`test_interactive_close_during_model_call_is_idempotent`；既有 session API 与 context-provider bridge 选择两条路径 |
| CLI single-run、stream、persistent sessions | done | `test_cli_real_single_run_and_stream_channels` 的两条 producer 保持 stdout JSON / stderr 事件通道；`test_cli_kernel_persistent_session_survives_owner_restart` 重建 SQLite 文件 owner，模型历史为 first → first+second；现有参数、默认模型、进程退出码 fixture 检查保留 |
| App Server thread/turn/approval/replay/non-text input | done (intentional difference) | `test_app_server_thread_turn.py`、`test_app_server_approval.py`、`test_app_server_replay.py` 的真实行为双路径；`test_kernel_process_restart_retains_calls_approval_image_and_client_cursor[model/tool/approval]`、`test_child_wait_user_is_exposed_and_reply_targets_child`、`test_kernel_controller_suspend_reply_resume_and_terminal_are_inbox_items[cancel/abort]`；thread lifecycle 的 archive/close 双路径 |
| App Server model/list、schema、TypeScript export | done | `test_model_list_forwards_optional_filters_and_emits_canonical_superset`、`test_schema_export_request_returns_json_and_typescript_bundles` 双路径比较原完整结果；JSON/TS bundle 无改动，现有生成文件与自包含 TS 检查保留 |

App Server thread = kernel session，turn/run 身份 = kernel turn。`_KernelThreadStore` 不初始化旧 ThreadStore：thread、turn、输入、items、token usage 和归档状态全部从 records/inbox 投影。数据库只有 `sk_session`、`sk_record`、`sk_inbox`、`sk_consumer`、`sk_commit` 五张 kernel 表。process-local runtime、泵线程、订阅者和后台 handle 仅为宿主绑定，不保留执行事实或第二本 thread ledger。

通知从 `SessionRunEventStore.consume()` 的 sink 投影后 ACK `app_server` consumer。handle 连续消费所有批次，既有 `test_thread_read_replays_emitted_items[kernel]` 将批次限制为一条 record，仍要求 live items = record replay 且 cursor 到达 head。客户端 `afterItemId` 由稳定 item ID 映射记录投影，重启后只返回 cursor 后的 items；这不是独立持久事件表。传输写入成功不代表客户端已经消费，客户端仍需用 item/event 身份去重和重放。

## 必覆盖场景

| 场景 | 证明 |
| --- | --- |
| App Server 进程在 turn 中退出 | 独立子进程 `os._exit(91)`，分别在模型回执与整批 tool plans 已提交、tool receipt 已提交、实际 approval/request 已发出后退出；重建 runtime / thread，等待旧 lease 释放或自然过期 |
| 恢复不重复模型/工具调用 | 三个恢复 case 的调用文件严格为 `model, tool, model`；同一 turn 只有一个 durable `turn_ended`。未获得 receipt 的未知副作用仍遵循既有 recovery fence，不能据此声称任意外部调用 exactly-once |
| 重启后回答审批 | 原 request ID 保持；observer 首先恢复 thread 时，请求仍发给 retained owner，observer 的 approval/resolve 被 INVALID_PARAMS 拒绝；原 owner 回复后工具仅执行一次；审批决定经 `approval_answer` inbox 应用 |
| active owner / terminal replay | `test_resume_during_active_turn_subscribes_before_later_notifications[kernel]` 对 active turn 的 turn/resume 仅返回响应；terminal replay 仅返回 retained result；handle 数与模型调用不增加 |
| child WAIT_USER | App Server 和 interactive 暴露 child session/turn/interaction，parent/child 保持 parked；回复进入 child inbox，child terminal 后 parent 才完成；同源相同 reply 重放，不同 payload 冲突 |
| 图片与非文本输入 | 恢复用原 `app_server_observable.json#input.valid`（text+image）；turn snapshot 保存原对象，模型重建请求仍有图片消息；既有 input 校验与真实 turn fixture producer 双路径运行 |
| archive/close 身份与幂等 | 文件 facade 重建后相同 id+bytes 返回 replay；同 id 改成另一 action 抛 Conflict；活动模型中的 close 只有一个 cancelled terminal，重复 close 返回 false |
| 连续与后台子会话 | `test_interactive_sequential_children_complete_in_one_prompt` 驱动连续两个 child；`test_interactive_background_child_runs_independently` 在 child 阻塞时 parent 已返回，owner join 后 child terminal 可从记录读取 |

## intentional differences

沿用计划最终评审已接受的 F2d-1/2/3 决定，不重新定义：ask_user 在原 turn/op 内继续、取消持久 terminal、RunEvent 至少一次、未提交的 callback 可重跑、tracing ACK-first、JSON shared_state 与宿主 bindings、child WAIT_USER 保留在 child、后台 process manager 的原 owner 限制、已保留的 delegation/预算/模型 attempt 差异。本轮将这些行为接到入口，而非用旧 checkpoint 或新 turn 模拟恢复。

本轮入口差异：

1. 等待中的 `turn/completed:interrupted` 是宿主当前尝试的投影，不是 durable turn terminal；回复后同一 turn 可再次产生通知，最终只有一条 `turn_ended`。ask_user 的 tool item 尚未 definitive，kernel 暴露 interaction，待回答才发布 definitive tool receipt。
2. App Server 重启保留 active turn 和原审批 owner，不把 running thread 重置为 idle。恢复尊重现有 lease；owner 尚未注册时 observer 不能接管其审批。重复 active resume 和已完成 turn 的 resume 不重新执行。
3. closed session 为 durable 边界，thread/resume 不复活 closed thread。archive/close 与用户回答采用稳定 inbox 身份，相同 bytes 重放，不同 bytes 冲突。
4. App Server 的后续 turn 使用同一 kernel session 的完整记录历史；旧 App Server 默认没有 transcript session。此项是 model-visible 变化，列入 C1，不能称两条路径的后续模型请求完全相同。
5. turn/action 为 inbox admission。新 action 的即时 receipt 为 accepted/running，随后状态读投影反映 suspended/interrupted/terminal；不暴露旧 controller receipt、revision、lease 或 command digest。action ID 仍采用原黄金向量的 thread/turn 作用域长度前缀 JCS 算法。

## C1 记录与公开 API 项

| 项 | 当前内部表示 / C1 工作 |
| --- | --- |
| thread 元数据唯一 home | 闭合保留字段 `session_created.attributes.app_server = {agent_key, cwd, metadata}`，额外字段拒绝；metadata 自身为已有 opaque JSON map |
| 输入与多模态冻结 | surface content `{text, messages, app_server:{input, metadata, owner}}`；模型消息冻结在 `task.metadata.session_input_messages`；C1 定义保留 namespace，明确 owner 绑定/恢复责任 |
| 会话/API 退休 | F3 将旧 transcript store、RunState/checkpoint、distributed、deferred 的公开恢复面替换为 session/turn 引用；私有 `_kernel` selector 同时退休，不能成为长期用户配置 |
| App Server 跨 turn 模型上下文 | 同一 session 的完整历史；更新当前 App Server model-visible 规则与 producer fixture |

纯 action ID 算法移到已有 `interaction.py`，旧 controller 仍引用同一函数，现有 identity 黄金向量与默认 wire 不变；kernel 不导入 retired controller。此移动不新增协议差异。

## C1 wire 项：与 contract 23 分开记录

下面 fixture 路径均相对于 `tests/fixtures/parity/app_server_observable.json`；`schema/export.result.jsonSchema.<name>` 是 JSON 字符串，后续 schema 字段指解析后的内容，`typescript` 的键带 `.ts`。lock / fixture / JSON schema / TypeScript 在 F2d-4 全部保持 v23；内部分支的差异必须在 C1 正式新版本收敛，不能宣称已 verified adoption。

| 精确 fixture / field | kernel 分支行为 / C1 需要替换 |
| --- | --- |
| `durableResume.requestFields`；`durableResume.protocolCases[*].request.params.checkpointKey`；`schema/export.result.jsonSchema.TurnResumeParams.{required,properties.checkpointKey}` 与 `typescript[TurnResumeParams.ts]` 的 `checkpointKey` | 请求只接受 `threadId,turnId`；checkpointKey 不接受。INVALID_PARAMS code 不变，错误 message 改为需要这两个字段；v23 导出暂不宣告这个私有请求形状 |
| `durableResume.responseFields`、`durableResume.checkpointSummary`、`durableResume.interruptionSummary`；`durableResume.protocolCases[*].{response.result,notifications[*].params}.{checkpoint,interruption}`；`durableResume.projectionCases[*].{checkpoint,interruption}` | retained result 来自 records；不产生 checkpoint/interruption summary。live claim 与 terminal replay 仍零额外调用；旧 deferred/reconciliation checkpoint 专属 cases 在 C1 替换为 parked/unknown record producers |
| `restart.staleRunningThreadStatus` | v23 为 idle；kernel records 中未结束 turn 保持 running，parked/suspended 的 turn snapshot 为 interrupted |
| `schema/export.result.jsonSchema.ClientRequest.$defs.ThreadResumeParams.properties.afterItemId`（现无该字段）；`typescript[ClientRequest.ts]` 的 `ThreadResumeParams` | kernel thread/resume 支持 `afterItemId` 的 client cursor；旧默认仍按原行为忽略该额外字段。thread/read 的现有 cursor 行为保持 |
| `controllerAdmission.hostPromptProjection.{fields,readSurface.payloadFields,unknown_fields}`；`schema/export.result.jsonSchema.ServerNotification` 的 thread/status/changed params | interaction 通知增加 `interactionId,sessionId,childTurnId,prompt`；状态投影增加 `interactions[]`，每项仅 `sessionId,turnId,prompt?,interactionId?`。不公开 operationId、requestDigest、工具参数、lease 或 handle |
| `terminal.threadStatusAfterTurn`；`ordering.turnTerminal`；`controllerAdmission.hostPromptProjection.readSurface.deliverySemantics` | 普通完成仍为 idle→turn/completed；parked attempt 在原 terminal-order 后追加 interrupted 状态及安全 interactions。读取状态与 child wait 保持 parked 事实；至少一次重投影按稳定 item/event ID 去重 |
| `controllerAdmission.askUserTerminalSemanticsUnchanged`；`terminal.agentStatusProjection[name=wait_user_is_interrupted_without_error]`；`toolLifecycle.executed.startedNotifications/completedNotifications`；`liveReplay.item` | interrupted status、wait_user reason 与无 error 保持；ask_user 未 definitive 时不提前产生 started/completed tool item 或问题文本的 agentMessage item，改用 interaction 状态；回答后同 turn/op 产生 receipt。旧 fixture 的 askUserTerminalSemanticsUnchanged=true 必须替换 |
| `controllerAdmission.{responseFields,publicReceiptFields}` 中 `status,waitReason`；`controllerAdmission.frameworkReceipt.appServerFields`；`controllerAdmission.serverDerivedCommand.internalFenceSource`、`controllerAdmission.internalDerivations` | 新动作即时 admitted/running，不同步返回旧 controller 的 applied status/waitReason。稳定身份黄金算法不变；fence 来自 retained turn/generation，旧 checkpoint revision/outbox derivations 退休 |
| `schema/export.result.jsonSchema.AppThread.properties.status`；`restart.staleRunningThreadStatus` 与 thread/resume 的 closed thread projection | closed 不再恢复成 idle；之后 turn/start 返回 INVALID_PARAMS（-32602），message 为 `Thread is closed`。archive 的现有 THREAD_ARCHIVED code 不变 |
| `modelLifecycle.identityFields`、`modelLifecycle.{startedNotifications,completedNotifications,failedNotifications}[*].params.payload.{callId,operationId}`；`terminal.tokenUsageProjection.value.modelCalls[*].{callId,operationId}`；`liveReplay.item.{itemId,turnId}`；`durableResume.protocolCases[*].response.result.runId` | 身份来自 session/turn/op/attempt 的记录位置，turnId=runId；值与 v23 独立生成身份不同。字段类型与 tool/model item payload shape 保持；测试将跨 producer 的身份视为不同身份，不当作字面相等 |
| `approval.{decisions,caseSensitive,timeoutDecision,disconnectDecision}`；非 fixture 指定的 `approval/requested` 与 `approval/request` 相对顺序 | 四个 decision、归属检查与 timeout 形状保持；kernel 先提交/投影 approval/requested，再通过 owner transport 发 approval/request。该相对顺序不在 v23 ordering 数组中，C1 明确记录 |

model/list 结果、schema/export JSON/TS 内容均完全不受 selector 影响。上表是内部运行行为与 v23 导出的显式差异，不能以“导出未改”代替 C1 wire adoption。

## F3 删除清单与先提取的依赖

此节仅盘点，本轮不删除。物理行数含空行、注释；符号行数含 decorators，来自当前源码 AST。整文件行数不是可直接删除的净行数：共享能力、窄值对象和下面的实际 imports 必须先迁出，行为测试必须换 producer，而非删掉能力。vendored fixtures 只能由中央 C1 release/sync 替换，本轮不修改。

### 整模块旧实现：F3 迁出共享符号后删除

| 路径 | 当前物理行数 |
| --- | ---: |
| `src/vv_agent/checkpoint.py` | 1245 |
| `src/vv_agent/deferred.py` | 232 |
| `src/vv_agent/sessions/__init__.py` | 16 |
| `src/vv_agent/sessions/base.py` | 380 |
| `src/vv_agent/sessions/memory.py` | 73 |
| `src/vv_agent/sessions/redis.py` | 135 |
| `src/vv_agent/sessions/sqlite.py` | 316 |
| `src/vv_agent/runtime/checkpoint_codec.py` | 386 |
| `src/vv_agent/runtime/checkpoint_history.py` | 258 |
| `src/vv_agent/runtime/checkpoint_resume.py` | 2637 |
| `src/vv_agent/runtime/controller.py` | 967 |
| `src/vv_agent/runtime/dispatch_outbox.py` | 374 |
| `src/vv_agent/runtime/state.py` | 2848 |
| `src/vv_agent/runtime/run_definition.py` | 454 |
| `src/vv_agent/runtime/sub_task_manager.py` | 966 |
| `src/vv_agent/runtime/sub_task_identity.py` | 40 |
| `src/vv_agent/runtime/model_calls.py` | 482 |
| `src/vv_agent/runtime/backends/__init__.py` | 28 |
| `src/vv_agent/runtime/backends/base.py` | 30 |
| `src/vv_agent/runtime/backends/celery.py` | 1066 |
| `src/vv_agent/runtime/backends/celery_tasks.py` | 1014 |
| `src/vv_agent/runtime/backends/distributed.py` | 1842 |
| `src/vv_agent/runtime/backends/inline.py` | 117 |
| `src/vv_agent/runtime/backends/thread.py` | 132 |
| `src/vv_agent/runtime/stores/__init__.py` | 5 |
| `src/vv_agent/runtime/stores/controller_store.py` | 1739 |
| `src/vv_agent/runtime/stores/memory.py` | 655 |
| `src/vv_agent/runtime/stores/redis.py` | 2458 |
| `src/vv_agent/runtime/stores/sqlite.py` | 2700 |

### 保留门面或共享能力：只删除旧实现部分

| 路径 | 当前物理行数 |
| --- | ---: |
| `src/vv_agent/runner.py` | 3221 |
| `src/vv_agent/run_handle.py` | 351 |
| `src/vv_agent/background_task.py` | 234 |
| `src/vv_agent/interactive.py` | 1332 |
| `src/vv_agent/result.py` | 287 |
| `src/vv_agent/run_config.py` | 238 |
| `src/vv_agent/events.py` | 3867 |
| `src/vv_agent/__init__.py` | 414 |
| `src/vv_agent/tools/outcomes.py` | 272 |
| `src/vv_agent/tools/function.py` | 550 |
| `src/vv_agent/tools/executor.py` | 239 |
| `src/vv_agent/tools/registry.py` | 182 |
| `src/vv_agent/tools/orchestrator.py` | 720 |
| `src/vv_agent/tools/dispatcher.py` | 93 |
| `src/vv_agent/runtime/engine.py` | 3211 |
| `src/vv_agent/runtime/cycle_runner.py` | 610 |
| `src/vv_agent/runtime/tool_call_runner.py` | 536 |
| `src/vv_agent/runtime/__init__.py` | 94 |
| `src/vv_agent/app_server/run_adapter.py` | 923 |
| `src/vv_agent/app_server/thread_store.py` | 489 |
| `src/vv_agent/app_server/thread_state.py` | 171 |
| `src/vv_agent/app_server/protocol/turn.py` | 233 |
| `src/vv_agent/app_server/protocol/__init__.py` | 77 |
| `src/vv_agent/app_server/processor.py` | 950 |
| `src/vv_agent/app_server/server.py` | 66 |
| `src/vv_agent/app_server/schema.py` | 860 |
| `src/vv_agent/cli.py` | 494 |
| `src/vv_agent/session/providers.py` | 117 |

### 退休的旧 checkpoint / store / backend 测试驱动

| 路径 | 当前物理行数 |
| --- | ---: |
| `tests/test_backends.py` | 231 |
| `tests/test_checkpoint.py` | 5098 |
| `tests/test_checkpoint_fault_matrix.py` | 755 |
| `tests/test_checkpoint_history_growth.py` | 240 |
| `tests/test_checkpoint_history_runner.py` | 155 |
| `tests/test_checkpoint_history_stores.py` | 376 |
| `tests/test_checkpoint_reconciliation.py` | 423 |
| `tests/test_checkpoint_resume_events.py` | 112 |
| `tests/test_checkpoint_runner.py` | 2600 |
| `tests/test_deferred_tools.py` | 1529 |
| `tests/test_dispatch_outbox.py` | 416 |
| `tests/test_distributed_checkpoint.py` | 3566 |
| `tests/test_run_definition_producer.py` | 411 |
| `tests/test_run_resume.py` | 902 |
| `tests/test_runtime_checkpoint_early_stop.py` | 315 |
| `tests/test_runtime_checkpoint_history.py` | 141 |
| `tests/test_runtime_controller.py` | 2164 |
| `tests/test_session_store_parity.py` | 436 |
| `tests/test_sessions.py` | 130 |

### 行为测试保留：迁移 producer，删除旧驱动 / 旧形状分支

| 路径 | 当前物理行数 |
| --- | ---: |
| `tests/test_app_server_contract_parity.py` | 1020 |
| `tests/test_app_server_controller_action.py` | 362 |
| `tests/test_app_server_durable_resume.py` | 704 |
| `tests/test_app_server_item_mapper.py` | 370 |
| `tests/test_app_server_schema.py` | 198 |
| `tests/test_app_server_thread_store.py` | 288 |
| `tests/test_bash_process_management.py` | 928 |
| `tests/test_compiler.py` | 333 |
| `tests/test_configured_sub_agent_manager_tool_envelope.py` | 499 |
| `tests/test_configured_sub_agent_parity.py` | 3751 |
| `tests/test_cycle_runner.py` | 343 |
| `tests/test_events_contract.py` | 555 |
| `tests/test_interactive_lifecycle_contract.py` | 236 |
| `tests/test_interactive_session_api.py` | 702 |
| `tests/test_interactive_session_persistence.py` | 339 |
| `tests/test_interactive_session_run_handle_bridge.py` | 299 |
| `tests/test_kernel_reusables.py` | 49 |
| `tests/test_memory.py` | 520 |
| `tests/test_microcompaction_policy.py` | 212 |
| `tests/test_parallel_subtasks.py` | 283 |
| `tests/test_parity_evidence_manifests.py` | 2087 |
| `tests/test_protocol_types.py` | 249 |
| `tests/test_run_config_controls_contract.py` | 103 |
| `tests/test_run_handle_live_stream.py` | 468 |
| `tests/test_runner.py` | 683 |
| `tests/test_runner_events_producer_parity.py` | 308 |
| `tests/test_runner_terminal_contract.py` | 291 |
| `tests/test_sub_task_manager_continuation_recovery.py` | 1184 |
| `tests/test_sub_task_status.py` | 309 |
| `tests/test_tool_metadata_contract.py` | 389 |
| `tests/test_tool_orchestrator.py` | 263 |
| `tests/test_tracing.py` | 223 |

### 本轮比较驱动：保留 kernel 行，移除 current selector 分支

以下文件保留能力断言；F3 删除 `surface` fixture 的 current 参数及对应旧路径断言。`tests/conftest.py` 保留 kernel owner 的 setup/teardown，移除双工厂选择；overhead 脚本的旧 Runner/RunAdapter 比较改成统一 kernel 基准，历史基线由 Git tag 保存。新增 kernel surface 模块保留，仅取消私有 selector 传递与旧基类依赖。

| 路径 | 当前物理行数 |
| --- | ---: |
| `tests/conftest.py` | 33 |
| `tests/test_app_server_initialize.py` | 165 |
| `tests/test_app_server_approval.py` | 477 |
| `tests/test_app_server_replay.py` | 244 |
| `tests/test_app_server_thread_lifecycle.py` | 247 |
| `tests/test_app_server_thread_turn.py` | 486 |
| `tests/test_cli_contract.py` | 307 |
| `tests/test_interactive_context_provider_bridge.py` | 87 |
| `scripts/session_kernel_overhead.py` | 222 |

### 失效的旧实现文档：删除

| 路径 | 当前物理行数 |
| --- | ---: |
| `docs/checkpoint-resume.md` | 375 |

### 仍需保留的文档：删除旧路径说明并重写当前入口

| 路径 | 当前物理行数 |
| --- | ---: |
| `docs/architecture.md` | 422 |
| `docs/runtime-control.md` | 210 |
| `docs/app-server.md` | 338 |
| `docs/app-server-runtime-mapping.md` | 28 |
| `docs/app-server-host-integration.md` | 61 |
| `docs/parity-contract.md` | 467 |
| `docs/development.md` | 151 |
| `docs/index.md` | 45 |
| `docs/model-settings.md` | 129 |
| `docs/bash-process-management.md` | 112 |
| `docs/output-validation.md` | 98 |
| `docs/run-budgets.md` | 99 |
| `docs/session-kernel.md` | 735 |
| `docs/session-kernel-capability-matrix.md` | 503 |

### 按轮证据 JSON：F3 合并成一份当前基准后删除

| 路径 | 当前物理行数 |
| --- | ---: |
| `docs/session-kernel-capacity-f2d2.json` | 158 |
| `docs/session-kernel-capacity-f2d3.json` | 158 |
| `docs/session-kernel-capacity-f2d4.json` | 158 |
| `docs/session-kernel-overhead-f2d4-1.json` | 123 |
| `docs/session-kernel-overhead-f2d4-2.json` | 123 |
| `docs/session-kernel-overhead-f2d4-3.json` | 123 |
| `docs/session-kernel-overhead-f2c.json` | 85 |
| `docs/session-kernel-overhead-f2d-tools-control.json` | 85 |
| `docs/session-kernel-overhead-f2d2.json` | 85 |
| `docs/session-kernel-overhead-f2d3.json` | 104 |
| `docs/session-kernel-overhead.json` | 80 |

### 历史验收 Markdown：归档或合并，不能作为当前默认路径说明

| 路径 | 当前物理行数 |
| --- | ---: |
| `docs/session-kernel-f2d-children-report.md` | 161 |
| `docs/session-kernel-f2d-memory-budget-events-report.md` | 201 |
| `docs/session-kernel-f2d-tools-control-report.md` | 111 |

### 中央 C1 退休或替换的 v23 fixtures（仅盘点）

| 路径 | 当前物理行数 |
| --- | ---: |
| `tests/fixtures/parity/checkpoint_codec.json` | 4008 |
| `tests/fixtures/parity/checkpoint_config.json` | 512 |
| `tests/fixtures/parity/checkpoint_resume.json` | 2249 |
| `tests/fixtures/parity/checkpoint_sqlite_canonical.sql` | 227 |
| `tests/fixtures/parity/checkpoint_store.json` | 2672 |
| `tests/fixtures/parity/controller_command.json` | 3723 |
| `tests/fixtures/parity/deferred_tool.json` | 1446 |
| `tests/fixtures/parity/distributed_run_driver.json` | 309 |
| `tests/fixtures/parity/distributed_run_envelope.json` | 726 |
| `tests/fixtures/parity/distributed_worker_response.json` | 885 |
| `tests/fixtures/parity/run_definition.json` | 1389 |
| `tests/fixtures/parity/session_codec.json` | 679 |
| `tests/fixtures/parity/session_items.jsonl` | 4 |
| `tests/fixtures/parity/session_sqlite_canonical.sql` | 23 |


### 关键部分删除与提取：逐符号行数

| 模块 | 符号 | 行数 | F3 动作 |
| --- | --- | ---: | --- |
| `src/vv_agent/runner.py` | `Runner.start_distributed` | 18 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner.start_distributed_compiled` | 22 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._start_distributed` | 38 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner.finalize_distributed` | 33 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._resume_state` | 42 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._resume_approved_tool_call` | 331 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._cancelled_approval_resume_result` | 51 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._approval_snapshot_matches` | 25 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._execute_checkpoint_approved_tool` | 93 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._run` | 167 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._inherit_handoff_metadata_denials` | 15 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._emit_chain_event` | 17 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._notify_observer` | 8 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._run_single_agent` | 65 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._run_single_agent_inner` | 683 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._apply_optional_output_validation` | 64 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._terminal_event` | 73 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._normalize_completion_observation` | 24 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._build_tool_registry` | 129 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._execute_function_tool` | 30 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._invoke_function_tool` | 28 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._tool_run_config_from_context` | 52 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._policy_denial_result` | 59 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._approval_result` | 84 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._brokered_approval_result` | 135 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._approval_error_result` | 33 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._extract_handoff` | 31 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._agent_tool_parent_config` | 23 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._run_child_agent` | 15 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._session_initial_messages` | 6 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._session_items_for_persistence` | 13 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._normalize_input` | 3 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._initial_budget_usage` | 10 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._metadata_with_initial_budget_usage` | 11 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._resolve_workspace` | 5 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._resolve_trace_id` | 5 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._resolve_event_session_id` | 8 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._workflow_name` | 5 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._trace_processors` | 11 | 删除旧 loop/resume/distributed helper |
| `src/vv_agent/runner.py` | `Runner._effective_run_config` | 92 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._apply_input_guardrails` | 13 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._apply_output_guardrails` | 13 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._postprocess_output` | 43 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._run_output_validator` | 32 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._output_validation_error` | 6 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._mark_output_validation_failure` | 16 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._replace_raw_result_output` | 13 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._coerce_output_type` | 30 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._resolve_model` | 12 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._provider_default_settings` | 6 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._tool_is_enabled` | 9 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._child_run_config` | 23 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._child_agent_prompt` | 17 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner._new_session_items` | 10 | 先提取共享 helper |
| `src/vv_agent/runner.py` | `Runner.configured` | 3 | 保留薄 API，重写为 kernel |
| `src/vv_agent/runner.py` | `Runner.run_sync` | 3 | 保留薄 API，重写为 kernel |
| `src/vv_agent/runner.py` | `Runner.stream_sync` | 5 | 保留薄 API，重写为 kernel |
| `src/vv_agent/runner.py` | `Runner.start` | 8 | 保留薄 API，重写为 kernel |
| `src/vv_agent/runner.py` | `Runner._run_compiled_sync` | 15 | 保留薄 API，重写为 kernel |
| `src/vv_agent/runner.py` | `Runner._start_compiled` | 16 | 保留薄 API，重写为 kernel |
| `src/vv_agent/runner.py` | `Runner.resume` | 3 | 保留薄 API，重写为 kernel |
| `src/vv_agent/tools/outcomes.py` | `DeferredWireError` | 6 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/tools/outcomes.py` | `DeferredHandleError` | 2 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/tools/outcomes.py` | `DeferredResolutionError` | 2 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/tools/outcomes.py` | `DeferredResolutionResultInvalid` | 8 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/tools/outcomes.py` | `_non_empty` | 4 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/tools/outcomes.py` | `DeferredToolHandle` | 80 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/tools/outcomes.py` | `ToolCallOutcome` | 109 | 删除旧 deferred/outcome ABI；先迁移共享 definitive 判断 |
| `src/vv_agent/result.py` | `ApprovalSnapshot` | 20 | 删除旧快照/闭包恢复 |
| `src/vv_agent/result.py` | `_PendingToolApproval` | 15 | 删除旧快照/闭包恢复 |
| `src/vv_agent/result.py` | `_RunResumeContext` | 20 | 删除旧快照/闭包恢复 |
| `src/vv_agent/result.py` | `_ApprovalConsumption` | 16 | 删除旧快照/闭包恢复 |
| `src/vv_agent/result.py` | `RunState` | 37 | 删除旧快照/闭包恢复 |
| `src/vv_agent/result.py` | `_approval_snapshots` | 27 | 删除旧快照/闭包恢复 |
| `src/vv_agent/events.py` | `ToolCallDeferredEvent` | 108 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/events.py` | `SessionPersistedEvent` | 30 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/events.py` | `CheckpointCreatedEvent` | 43 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/events.py` | `CheckpointResumedEvent` | 5 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/events.py` | `OperationReplayedEvent` | 56 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/events.py` | `ReconciliationRequiredEvent` | 67 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/events.py` | `ReconciliationResolvedEvent` | 60 | 删除/替换旧 checkpoint/transcript 事件及 decoder 分支 |
| `src/vv_agent/app_server/protocol/turn.py` | `TurnResumeParams` | 12 | C1 新 wire 替换 |
| `src/vv_agent/app_server/protocol/turn.py` | `CheckpointSummary` | 16 | C1 新 wire 替换 |
| `src/vv_agent/app_server/protocol/turn.py` | `InterruptionSummary` | 18 | C1 新 wire 替换 |
| `src/vv_agent/app_server/protocol/turn.py` | `TurnResumeResponse` | 36 | C1 新 wire 替换 |
| `src/vv_agent/run_handle.py` | `RunHandle._start_worker` | 70 | 删除旧独立 worker/恢复/快照内容；保留薄 handle 能力 |
| `src/vv_agent/run_handle.py` | `RunHandle.resume` | 16 | 删除旧独立 worker/恢复/快照内容；保留薄 handle 能力 |
| `src/vv_agent/run_handle.py` | `RunHandle.state` | 48 | 删除旧独立 worker/恢复/快照内容；保留薄 handle 能力 |
| `src/vv_agent/run_handle.py` | `RunHandle._mark_terminal_event` | 31 | 删除旧独立 worker/恢复/快照内容；保留薄 handle 能力 |
| `src/vv_agent/app_server/thread_store.py` | `ThreadStore` | 405 | 删除第二 thread ledger；保留/迁出三个值对象 |
| `src/vv_agent/runtime/engine.py` | `AgentRuntime._run_active` | 281 | 删除旧 runtime 调度；提取仍需保留的纯组装 |
| `src/vv_agent/runtime/engine.py` | `AgentRuntime._build_cycle_executor` | 543 | 删除旧 runtime 调度；提取仍需保留的纯组装 |
| `src/vv_agent/runtime/engine.py` | `AgentRuntime._run_sub_task` | 645 | 删除旧 runtime 调度；提取仍需保留的纯组装 |
| `src/vv_agent/runtime/engine.py` | `AgentRuntime._build_sub_agent_task` | 131 | 删除旧 runtime 调度；提取仍需保留的纯组装 |
| `src/vv_agent/runtime/tool_call_runner.py` | `ToolCallRunner._apply_tool_use_behavior` | 20 | kernel 正在复用，先提取 |
| `src/vv_agent/runtime/tool_call_runner.py` | `ToolCallRunner._build_image_notification` | 15 | kernel 正在复用，先提取 |
| `src/vv_agent/runtime/tool_call_runner.py` | `ToolCallRunner._build_skipped_result` | 20 | kernel 正在复用，先提取 |
| `src/vv_agent/interactive.py` | `AgentSession.steer` | 8 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession.follow_up` | 8 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession.clear_queues` | 6 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession.continue_run` | 8 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession.replace_messages` | 11 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession._persist_custom_run_delta` | 14 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession._drain_next_queued_prompt` | 8 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession._before_cycle_messages` | 8 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/interactive.py` | `AgentSession._interruption_messages` | 7 | 删除旧 transcript/内存队列实现；当前接口由 C1 定义保留或退休 |
| `src/vv_agent/background_task.py` | `BackgroundAgentTask.start` | 19 | 删除进程 registry 与 Runner.start wiring；保留 kernel handle facade |
| `src/vv_agent/background_task.py` | `BackgroundAgentTask.get_handle` | 6 | 删除进程 registry 与 Runner.start wiring；保留 kernel handle facade |


### kernel 当前仍依赖、F3 必须先提取或替换的旧代码

| 当前 consumer | 活跃依赖 | F3 先做什么 |
| --- | --- | --- |
| `session/runtime.py`, `surfaces.py` | `Runner._effective_run_config/_resolve_model/_tool_is_enabled/_apply_input_guardrails` | 提取配置、精确模型解析、启用条件、guardrail helpers；不得保留 Runner loop 作为后备 executor |
| `session/output.py`, `result.py` | `Runner._postprocess_output/_run_output_validator/_output_validation_error/_coerce_output_type/_new_session_items` 与调用的 output guardrail / raw-output helpers | 迁出输出验证/typed coercion/消息差量；保留错误和校验边界；重写 API facade |
| `session/delegation.py` | `Runner._child_agent_prompt`；`SubTaskManager` 的 type-only `tool_manager()` annotation | 提取父摘要组装；删除旧 manager 类型 cast，以 kernel child manager 的实际接口代替 |
| `session/context.py`, `kernel.py`, `lifecycle.py` | `ToolCallRunner._build_image_notification/_build_skipped_result/_apply_tool_use_behavior` | 提取三个纯 helper 后删除 ToolCallRunner 执行循环；图像、skip 和 FINISH 的 producer 不退休 |
| `session/interactive.py` | 继承 `AgentSession` 的 subscriptions、事件、锁、run lifecycle、background process watchers；`AgentSessionRun/InteractiveAgentDefinition` 值对象 | 提取共有 facade/value/event 层，保留被复用的生命周期；移除旧 transcript、Python pending queue 和 Runner 执行分支。旧可变 transcript API（如 replace_messages）不由只读 kernel transcript 支持，C1 明确退休或定义记录操作后才能切公开默认 |
| `session/app_server.py` | 继承 `RunAdapter` 的 constructor、`_prompt_from_input/_with_app_server_controls/_notify_subscribers/_complete_turn/_turn_status/_result_error_text/_validate_public_controller_action*` | 提取 transport/approval/input/terminal formatting 与严格校验；移除 checkpoint summary/binding/controller cache 和旧泵线程。`_complete_turn` 的旧 ledger writes 在 kernel store 为投影校验/noop，F3 删除这些调用 |
| `session/app_server.py` | `ThreadRecord/TurnRecord/ThreadSnapshot`、`ThreadStore` 基类 | 保留/迁出三个值对象；删除基类依赖和旧 ledger/codec/schema；将 thread_state 只保留 subscriptions / live handle，移除旧 steering/follow-up queues 与持久状态覆盖 |
| `session/providers.py` 经 tools registry/executor/function/orchestrator/dispatcher | `ToolCallOutcome` 的 completed/deferred/host-interaction 分支 | C1 改成唯一 Provider outcome；删 ToolCallOutcome / DeferredToolHandle / Deferred* errors 前迁移 executor 类型、返回值归一化、definitive/ambiguous 校验与 host-interaction adapter。不能整删仍活跃的 tool registry/dispatcher |
| `session/delegation.py` 经 `tools/handlers/sub_agents.py` | `runtime/sub_task_identity.py` 的 normalize_identity_string 与 assigned_sub_task_identity scope | 迁出规范化；删除旧 ContextVar 分配机制，使用实际 child admission 身份；保留相同工具 envelope producer |
| `session/runtime.py`, `kernel.py`, `lifecycle.py`, `providers.py`, `result.py` | compiler/context/hooks/lifecycle/cancellation/token_usage/tool_planner，process manager 和 memory modules | 这些是共享能力，不是待删的旧 executor；保留模块、删其已失效的 checkpoint imports/annotation，禁止把整个 runtime/ 目录删掉 |
| 既有默认 package imports | `__init__.py`, `runtime/__init__.py`, RunConfig checkpoint 字段、RunResult.into_state、protocol/schema 中旧恢复类型 | 原子切换时移除旧 exports/decoders/字段/类型引用；同步中央 C1 artifact 和 lock；不保留旧 schema selector 或兼容 reader |

`runtime/controller.py` 的纯 command ID helper 已迁到 `interaction.py`；kernel 不再直接依赖该 retired 模块。保留的 `OperationAmbiguousEvent`、`ModelRetryDuplicateRiskEvent` 是 kernel 当前真实 producer，不能随 checkpoint 事件一并删除。`run_handle.py`、`background_task.py`、`interactive.py`、App Server 的行为门面保留，其旧实现部分退休。普通模型、工具、审批、预算、workspace、memory 与 tracing 测试不因旧执行栈退休而删除。


## 开销、M6 容量与门禁

三次测量分别使用独立进程，连续执行 `uv run python scripts/session_kernel_overhead.py --runs 200 --warmup 10 --output <file>`；pytest 与 M6 不并行。既有五个场景的 workload、计时与 GC 设置未变。新增 `app_server_turn` 对两条真实 App Server producer 计时：in-process JSON-RPC `turn/start` 入站至 `turn/completed` 出站；initialize、thread/start、owner 构造与 cleanup 在区间外。使用相同 ScriptedLLM，断言最终输出及所有脚本调用耗尽。

原始数据：[run 1](session-kernel-overhead-f2d4-1.json)、[run 2](session-kernel-overhead-f2d4-2.json)、[run 3](session-kernel-overhead-f2d4-3.json)。单位均为 ms；added p50/p95 是 kernel 与 current 各自分位数之差，不是逐样本差值的分位数。`current` 在前五行使用 Runner，App Server 行使用原 RunAdapter。所有 36 个 path/scenario 测量组均只剩一个主线程，无新增存活线程。

| 轮次 | 场景 | current p50 / p95 | kernel p50 / p95 | added p50 | added p95 | p95 门槛 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | no_tool | 3.274 / 3.844 | 13.641 / 15.055 | 10.367 | 11.211 | 50 | PASS |
| 1 | two_tools | 5.563 / 6.154 | 23.856 / 24.833 | 18.293 | 18.679 | 50 | PASS |
| 1 | ten_turns | 36.642 / 38.100 | 91.418 / 94.677 | 54.777 | 56.577 | 80 | PASS |
| 1 | start_cancel | 4.383 / 4.779 | 14.654 / 15.640 | 10.271 | 10.861 | 50 | PASS |
| 1 | children | 7.377 / 7.868 | 49.750 / 51.864 | 42.373 | 43.996 | 50 | PASS |
| 1 | app_server_turn | 7.335 / 8.950 | 32.079 / 34.818 | 24.744 | 25.869 | 50 | PASS |
| 2 | no_tool | 3.325 / 3.764 | 12.662 / 13.346 | 9.337 | 9.582 | 50 | PASS |
| 2 | two_tools | 5.326 / 5.915 | 23.079 / 24.343 | 17.754 | 18.429 | 50 | PASS |
| 2 | ten_turns | 35.342 / 37.518 | 89.937 / 97.361 | 54.595 | 59.842 | 80 | PASS |
| 2 | start_cancel | 4.412 / 4.895 | 14.823 / 16.069 | 10.411 | 11.174 | 50 | PASS |
| 2 | children | 7.580 / 8.416 | 49.559 / 51.733 | 41.979 | 43.316 | 50 | PASS |
| 2 | app_server_turn | 7.316 / 8.704 | 31.314 / 33.224 | 23.997 | 24.520 | 50 | PASS |
| 3 | no_tool | 3.427 / 3.918 | 13.132 / 13.976 | 9.705 | 10.058 | 50 | PASS |
| 3 | two_tools | 5.364 / 5.868 | 23.730 / 25.102 | 18.366 | 19.234 | 50 | PASS |
| 3 | ten_turns | 35.087 / 38.632 | 91.912 / 96.639 | 56.825 | 58.007 | 80 | PASS |
| 3 | start_cancel | 4.442 / 4.982 | 14.857 / 16.152 | 10.415 | 11.170 | 50 | PASS |
| 3 | children | 7.751 / 8.501 | 51.156 / 53.323 | 43.404 | 44.823 | 50 | PASS |
| 3 | app_server_turn | 7.491 / 8.841 | 32.438 / 34.847 | 24.947 | 26.006 | 50 | PASS |

三个进程的 `ten_turns` added p95 为 56.577 / 59.842 / 58.007 ms，最小余量 20.158 ms；`children` 为 43.996 / 43.316 / 44.823 ms，最小余量 5.177 ms；App Server 为 25.869 / 24.520 / 26.006 ms，最小余量 23.994 ms。三轮 App Server kernel 的 batch RSS 增量约 7.3 / 7.2 / 7.1 MiB，current 约 2.8 / 2.7 / 2.8 MiB；RSS 是 GC 后批次增量，不能作为峰值或长期泄漏结论。p95 通过不表示 max/p99 或慢 provider 延迟上界。

M6 命令：`uv run python scripts/session_kernel_benchmark.py --sizes 5000 20000 --samples 1 --assert-capacity --output docs/session-kernel-capacity-f2d4.json`。真实本地 PostgreSQL 18.6；Python 3.12.12 / WSL2 x86_64。数据见 [capacity JSON](session-kernel-capacity-f2d4.json)。workload 仍为同一 active turn、已保留的 final model receipt 与 bounded 1 KiB tool receipts；cold drive 重建 connection/Runtime，但 OS/PG buffer 仍可能热。每个规模一个样本，不声称 p95、长期压测或物理冷启动。

| records | logical bytes | append ms | steady append ms | full fold ms | read+fold ms | cold drive ms | 新增 provider 调用 | lease failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5000 | 4241706 | 7.189 | 9.697 | 38.453 | 427.851 | 503.276 | 0 | 0 |
| 20000 | 16958334 | 12.522 | 15.440 | 291.875 | 1843.380 | 2059.302 | 0 | 0 |

| sessions | runnable items | page size | first page ms | late page ms | full catalog scan ms | full scan 门槛 ms | 结果 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1500 | 100 | 2.511 | 1.734 | 18.588 | 1000 | PASS |
| 10000 | 15000 | 100 | 2.309 | 1.488 | 226.151 | 1000 | PASS |

M6 的原 capacity assertions 全部通过：cold drive 不超过规模对应的 5000 / 20000 ms 且没有 LeaseLost，steady append median ≤50 ms，1k/10k catalog 全分页扫描 ≤1000 ms；新增 provider 调用为零。lease TTL 15000 ms、heartbeat 0.25 s 均未放宽。

| 门禁 / 命令 | 结果 |
| --- | --- |
| `python3 scripts/contract_snapshot.py check` | PASS：contract 23.0.0 / revision `ad2d4974545f987e237aed421cc4f65680e9a8dc`；55 fixture files、54 manifest entries；digest `0e4c98ac3d22c959e2b1dd5969f55b56491c0b2eab1dbacaf38f68298bcf7d98` |
| `uv run ruff format --check .` | PASS：393 files already formatted |
| `uv run ruff check` | PASS |
| `uv run ty check` | PASS |
| `uv run pytest tests/session -q` | PASS：1397 passed，305.52 s；真实本地 PG，无 skips |
| `VV_AGENT_TEST_REDIS_URL=redis://127.0.0.1:6400/15 uv run pytest` | PASS：3944 passed、20 skipped、18 warnings，449.95 s；真实 Redis / PG |
| `uv run pytest tests/test_app_server_initialize.py -q` | PASS：最终入口驱动复核 16 passed，0.09 s；通过 AppServer 工厂实际选择两条路径 |
| 三个独立 overhead 进程，各 `--runs 200 --warmup 10` | PASS：上述全部 18 行，所有原门槛及 App Server ≤50 ms |
| M6 `--sizes 5000 20000 --samples 1 --assert-capacity` | PASS：上述 history / runnable 两表 |
| `git diff --check` 与范围检查 | PASS：HEAD 仍为基线 a4325c2；公开默认、exports、lock、fixtures、local_settings.py 和 Rust 未改；无提交 |
| Redis 清理 | PASS：本轮专用 6400 Redis 已 `shutdown nosave`，端口释放 |

全套的 20 个 skip：4 个跨 runtime/语言 opt-in probes、9 个 FakeRedis 不提供真实 CAS/命令类型执行的变体、6 个 live-provider opt-in cases、1 个目录 symlink 环境不可用。对应适用的真实 Redis 变体已执行。18 warnings 是旧 distributed 测试在多线程进程调用 fork 的 DeprecationWarning；本轮没有隐藏这些结果，也未运行真实 provider live 测试。

F3 可以按“正式 C1 artifact + 切换公开默认 + 删除旧栈”推进；能力矩阵没有剩余 partial/missing 行。公开 wire 差异需先随 C1 规范与 fixture 收敛，旧代码的活跃依赖需先提取；本轮内部测试与性能通过不等同于新 contract verified adoption。
