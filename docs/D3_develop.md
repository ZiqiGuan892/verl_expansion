# D3 开发记录：CE 成员注册与 target-only bootstrap

## 1. 阶段目标

D3 在 D2 的 `RUNTIME_READY` 基础上完成 borrower 的首次参数同步：

```text
borrowed runtime RUNTIME_READY
    -> Trainer 查询 replica 投影
    -> CE Manager register_replica（pending）
    -> Trainer 在参数快照边界读取 current_param_version
    -> CE Manager 建立 actor + 目标 borrowed workers 的临时通信组
    -> prepare/build_topology/init_process_group
    -> actor send + borrowed receive/ServerAdapter.update_weights
    -> finalize
    -> last_synced_versions[replica_rank] = snapshot_version
    -> borrowed 进入普通全成员同步集合
```

D3 不实现 LB `commit_ready`、请求接流、sleep/wake、reclaim/destroy、GS 全局账本或
完整失败回收。`WEIGHTS_READY` 只表示 CE Worker 和 server adapter 已完成本次参数加载，
不等于 LB `READY`。

## 2. 修改文件与原因

### 2.1 `src/multi_task_scheduler/checkpoint/checkpoint_engine_manager.py`

`MultiTaskCheckpointEngineManager` 仍继承原生 `CheckpointEngineManager`，不复制后端传输协议。
新增字段：

| 字段 | 含义 |
| --- | --- |
| `sync_gate` | 串行化普通全成员同步、注册、注销和 target-only bootstrap 的异步锁 |
| `sync_state` | `IDLE` 表示可接受操作，`SYNCING` 表示有通信事务，`BLOCKED` 表示上次事务失败，需要外部处理后才能继续 |
| `inflight_replicas` | 当前事务开始时固定的 replica 快照，避免操作期间成员列表变化 |
| `pending_bootstrap` | `replica_rank -> snapshot_version/None`；已注册但尚未完成首次 bootstrap 的成员，不参加普通全成员同步 |
| `last_synced_versions` | `replica_rank -> version`；只有 CE receive、server adapter 加载和 finalize 全部成功后才写入 |

新增方法：

| 方法 | 功能 |
| --- | --- |
| `register_replica(replica)` | 校验 rank、Worker 数和重复句柄，将 borrowed replica 加入本地 CE registry，并标记 pending；同一实例重复注册幂等 |
| `unregister_replica(replica_or_rank)` | 在 gate 内从后续同步快照删除 replica，同时清理 pending 和版本记录；不销毁 server/Worker |
| `update_weights(global_steps)` | 在 gate 内暂时使用有效成员快照调用原生全量同步；pending 成员被过滤，成功后更新所有参与者的版本 |
| `bootstrap_replica(replica, snapshot_version)` | 只为目标 replica 创建临时 `RayWorkerGroup`，复用原生拓扑、权重传输和 finalize 流程 |

target-only bootstrap 的具体调用顺序为：

1. 对目标 replica abort 当前请求并释放 KV cache；新建 runtime 尚未接流，因此不会影响其他 replica。
2. 使用目标 Worker handles 构造临时 `RayWorkerGroup`。
3. 调用原生 `build_process_group(target_group)`，由 backend 根据 actor world size 和目标 world size 生成拓扑。
4. 同时执行 actor `update_weights(global_steps=V, mode=backend)` 与目标 group `update_weights(global_steps=V)`。
5. 对 actor 和目标 group 执行 `finalize()`，再恢复目标 KV cache 和生成。
6. 只有上述步骤全部完成才删除 `pending_bootstrap`、写入 `last_synced_versions` 和 `serving_version`。

异常时 manager 保持 `BLOCKED`，目标仍停留在 pending，不会进入普通同步有效集合；如果通信组已经建立，执行 best-effort `finalize`，并尝试恢复 KV cache/生成，原始异常继续向 Trainer 传播。

### 2.2 `src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py`

新增本地投影方法：

| 方法 | 功能 |
| --- | --- |
| `get_replica_for_ce(replica_rank)` | 从 manager 的 `borrowed_operations` 找到 `RUNTIME_READY` 的 borrowed replica，并只在任务内返回给 Trainer |
| `register_borrowed_replica_for_ce(replica_rank)` | 将 borrowed replica 加入本任务的 `rollout_replicas` 投影，但不修改 LB server 列表或提交接流 |
| `mark_replica_serving_version(replica_rank, version)` | 将 CE 确认的版本写回 manager 所有的 replica 对象 |
| `cleanup_d3_runtime(replica_rank)` | D3 测试结束后只清理本次创建的 borrowed Actor，不声称 claims 已归还 |

这些方法不持有 GS，也不建立第二份全局 `bundle_leases` 表。

### 2.3 `src/multi_task_scheduler/integration/verl/experimental_fully_async/rollouter.py`

新增 Rollouter 到本地 manager 的薄转发：

- `run_d3_runtime_smoke(scenario)`：复用 D2 正式创建入口，保留 runtime 并登记 CE 投影；
- `get_borrowed_replica_for_ce(replica_rank)`：向 Trainer 提供当前 manager-owned replica；
- `mark_replica_serving_version(...)`：转发版本确认；
- `cleanup_d3_runtime(...)`：转发测试清理。

Rollouter 不解析 PG、不创建 CE 通信域、不持有 GS 句柄。

### 2.4 `src/multi_task_scheduler/integration/verl/experimental_fully_async/trainer.py`

新增 Trainer 级任务接口：

| 方法 | 参数/返回值 | 功能 |
| --- | --- | --- |
| `register_replica(replica_rank: int) -> dict` | 任务内 rank / 注册回执 | 从 Rollouter 查询 replica，再调用 CE manager 注册 |
| `bootstrap_replica(replica_rank: int) -> dict` | 任务内 rank / `WEIGHTS_READY` 回执 | 在 `parameter_snapshot_gate` 内读取一次当前版本，并调用 target-only bootstrap |
| `unregister_replica(replica_rank: int) -> dict` | 任务内 rank / 注销回执 | 从 CE 有效集合移除，不做物理销毁 |

Trainer 还覆盖了两个原生边界：

- `load_checkpoint()` 完成原生 checkpoint 加载后，若打开 D3 测试开关，则创建并 bootstrap 一个 borrowed replica。这样 resume 场景使用的就是已加载版本；
- `_fit_update_weights()` 使用同一 `parameter_snapshot_gate` 串行化普通同步，并在首次普通同步后输出 `D3_NORMAL_SYNC_RESULT`，验证 borrowed 已进入全成员同步。

正常入口仍由原生 `main_ppo` 调用；D3 只是通过配置开关插入测试路径。

### 2.5 `src/multi_task_scheduler/checkpoint/hccl_checkpoint_engine.py`

目标服务器使用 Ascend/NPU。当前 `vllm-ascend` 的 `PyHcclCommunicator` 将销毁函数放在
底层 HCCL library wrapper 上，而原生 verl 的 `HCCLCheckpointEngine.finalize()` 直接调用
communicator 的 `destroyComm`。因此新增 `MultiTaskHCCLCheckpointEngine`，只覆盖
`finalize()`：兼容旧的 `destroyComm`，并在当前 API 下调用
`pyhccl.hccl.hcclCommDestroy(pyhccl.comm)`；prepare、topology、send、receive 和
ServerAdapter 参数流转全部继承原生实现。

`D3_test.sh` 通过 `checkpoint_engine.custom_backend_module` 在所有相关 Worker 进程导入
该模块，并将 backend 设为 `multitask_hccl`。这样不修改原生 `nccl`/HCCL registry，也
不会依赖 import 顺序覆盖原生实现。

## 3. 测试设计

### 3.1 本地单元测试

新增 `tests/unit/test_checkpoint_membership.py`，不导入真实 Ray/vLLM，使用 fake actor group 验证：

- 注册幂等以及不同 Worker 句柄冲突；
- pending replica 不进入普通全成员同步；
- target-only 的调用顺序包含 topology、actor/target update 和 finalize；
- bootstrap 成功后版本表更新，下一次普通同步包含 borrower；
- 注销会清理 pending 和版本记录。

运行命令：

```bash
python -m pytest -q -p no:cacheprovider \
  tests/unit/test_checkpoint_membership.py \
  tests/unit/test_borrowed_contract.py \
  tests/unit/test_borrowed_runtime.py \
  tests/unit/test_wiring.py
```

### 3.2 真实环境一键测试

新增 `D3_test.sh`。服务器从原生 `verl` 目录执行：

```bash
bash ../D3_test.sh
```

脚本复用 `multi_task_run.sh` 的模型、数据、Ascend、Python 路径和训练配置。由于当前
Ascend 运行时的 `PyHcclCommunicator` 没有原生 HCCL 后端调用的 `destroyComm` 方法，
D3 脚本选择插件内的 `multitask_hccl` 后端：它继承原生 HCCL 的传输逻辑，只替换
通信域销毁适配，并通过 `custom_backend_module` 在训练 Worker、CE Worker 和 Trainer
侧注册同一个后端。

脚本只追加：

```text
actor_rollout_ref.rollout.checkpoint_engine.backend=multitask_hccl
actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=multi_task_scheduler.checkpoint.hccl_checkpoint_engine
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.multitask_hccl.rebuild_group=true
+multitask.d3_bootstrap_test.enabled=true
+multitask.d3_bootstrap_test.scenario=split
+multitask.d3_bootstrap_test.cleanup_after_test=true
```

脚本默认执行一个 `split` 场景，也支持：

```bash
D3_RUNTIME_SCENARIOS=basic,split bash ../D3_test.sh
D3_RUNTIME_SCENARIOS=cross_pg bash ../D3_test.sh
```

每个场景必须同时出现：

```text
D3_BOOTSTRAP_RESULT {"state": "WEIGHTS_READY", ...}
D3_NORMAL_SYNC_RESULT {"state": "FULL_SYNC_READY", ...}
```

此外要求 main_ppo 和 `tee` 均退出为 0。异常文本仍会保留在日志中供定位，不能替代训练完成回执作为通过或失败依据。
日志写入 `${VERL_REPO_DIR}/logs/d3_runtime/`。

本次开发环境已执行以下不依赖 Ray/GPU 的验证：

```text
37 passed in 0.40s
py_compile: passed
git diff --check: passed（仅提示 Windows 换行转换）
```

本机没有可用的 Linux Bash、Ray、vLLM 和 NPU 运行环境，因此没有在本机伪造 `D3_test.sh`
的通过结果。服务器上应从与 `multi_task_run.sh` 相同的目录执行该脚本；只有日志中的两个
D3 marker 和训练进程退出码均满足检查，才能将 D3 记为真实环境通过。

如果 bootstrap 在 D3 测试 hook 中失败，Trainer 会先注销已注册成员，再请求 Rollouter
清理本次测试创建的 borrowed Actor；清理失败不会覆盖原始 bootstrap 异常。该路径只服务
于测试 hook，不能替代后续阶段的生产 reclaim/destroy 恢复流程。

### 3.3 真实验收观察点

需要从日志和 Ray 运行信息确认：

1. borrowed CE Worker handles 与 donor Worker 不同；
2. target-only bootstrap 只使用 borrower Worker，不 abort 或 release 其他 native replica；
3. bootstrap 返回版本等于 Trainer 的 `current_param_version`；
4. 首次普通全成员同步成功，borrower 的 `last_synced_versions` 更新；
5. target-only 与全成员通信域都完成 `finalize`，没有遗留的同步异常；
6. 测试清理只杀死本次 borrowed Actor，不删除 donor PG，也不报告 claims 已归还。

一键脚本只验证参数传输和 CE membership；它不把 HTTP 请求成功或 LB 路由成功作为 D3 通过条件。

## 4. 当前限制与后续边界

- 当前实现依赖分布式 checkpoint backend；`naive` backend 会明确拒绝 target-only bootstrap。
- `rebuild_group=true` 必须在 CE Worker 创建前配置，不能在 actor 已启动后动态修改；D3 脚本显式设置 `multitask_hccl` 选项。
- `BLOCKED` 只表示本地 CE 事务失败，当前没有自动恢复通信域或重建 actor 的流程；完整错误恢复留待后续阶段。
- `unregister_replica()` 不执行 server、engine、CE Worker 或 PG 的物理销毁；这些动作仍属于 lifecycle 阶段。
- D3 不调用 LB `commit_ready()`，所以 bootstrap 成功后 replica 仍不会接收业务请求。接流和在途请求处理属于 D4/生命周期实现。
- D3 的 Ascend HCCL 适配只为通信域销毁接口提供插件层兼容；若目标环境的
  `vllm-ascend` API 不同，应调整该插件后端，不要修改原生 `verl/checkpoint_engine/hccl_checkpoint_engine.py`。

## 5. D3 borrowed 创建的显存交接修复

### 5.1 失败证据

在 1 节点 8 卡服务器上运行 `D3_RUNTIME_SCENARIOS=basic bash ../D3_test.sh` 时，native
TP=4 已经在目标 4 张 NPU 上运行。borrowed Engine 的 Worker 初始化失败日志为：

```text
Free memory on device (3.05/29.49 GiB) on startup is less than
desired GPU memory utilization (0.3, 8.85 GiB)
```

这发生在 vllm-ascend 的 `Worker.init_device()`，不是 CE 注册、HCCL finalize 或 LB 接流。
原生 standalone `vLLMHttpServer.sleep()` 对该模式是 no-op，因此不能直接用原生
`replica.sleep()` 释放 donor 显存。

### 5.2 插件修改

`MultiTaskvLLMHttpServer` 增加 `sleep_for_runtime_test()` 和
`wake_for_runtime_test()`。两者只在 `node_rank == 0` 调用 vLLM Engine 的
`sleep(level=1)`/`wake_up(tags=["weights", "kv_cache"])`，并校验测试脚本显式打开
`enable_sleep_mode`、`free_cache_engine`。level 1 保留 CPU 权重、丢弃 KV，适合在同一
批物理卡上启动第二个 Engine；它不覆盖原生 `sleep()`，也不改变生产生命周期接口。

`MultiTaskLLMServerManager.run_d2_runtime_smoke()` 在成功场景创建 borrowed 前，按 claims
找到当前任务的 native donors，并对 donor 的所有 server 调用该测试接口。borrowed 创建
完成后，测试清理阶段先让 borrowed 进入 sleep，再销毁其本次测试 Actor，最后恢复 donor。

D3 bootstrap 后不会立即恢复 donor。Trainer 将已经 sleep 的 donor rank 写入 CE Manager
的暂时排除集合，普通同步只构造 actor worker + borrowed worker 的 HCCL 拓扑。这样 donor
CE Worker 不会和 borrowed CE Worker 在同一物理 NPU 上重复加入 HCCL communicator，避免
`hcclCommInitRank(...): HCCL error: parameter error`。普通同步完成后，测试清理 borrowed
并恢复 donor 的服务显存，再清除 CE 排除集合。

### 5.3 验证

`D2_runtime_test.sh`、`D3_test.sh` 增加：

```text
actor_rollout_ref.rollout.enable_sleep_mode=true
actor_rollout_ref.rollout.free_cache_engine=true
```

运行：

```bash
D3_RUNTIME_SCENARIOS=basic bash ../D3_test.sh
```

应依次看到 `RUNTIME_TEST_DONORS_SLEEPING`、`D3_BOOTSTRAP_RESULT`、
`DONORS_SLEEPING_BORROWER_ONLY_EFFECTIVE` 和 `D3_NORMAL_SYNC_RESULT`。如果再次出现
`Free memory on device ... less than desired GPU memory utilization`，需保存完整 Worker
日志，确认目标 vllm-ascend 版本确实实现了 level-1 sleep 的显存释放。

普通同步结束后，donor 不在本轮 CE 拓扑中，因此其 server 不会通过参数同步自动更新
`global_steps`。D3 清理阶段在唤醒 donor 后显式调用 `set_global_steps(current_param_version)`；
否则下一次生成返回的版本字段为 `None`，会在 `detach_utils.py` 的 batch 组装阶段触发
`None - None`。

### 5.4 本次调整后的生命周期边界

本次只保留两项必要的 CE 步骤：

1. donor 进入借用窗口时，`MultiTaskCheckpointEngineManager.suspend_replicas_for_sync()`
   将 donor rank 从 effective set 排除，避免 donor CE Worker 和 borrowed CE Worker 在同一
   物理设备上共同加入下一次 HCCL 通信域；
2. 借用窗口结束且 donor 已经完成恢复与追平后，
   `resume_replicas_for_sync()` 清除排除标记，使 donor 能够参加后续同步。

Trainer 对这两个 CE 操作提供了 `suspend_donors_for_borrow(replica_ranks)` 和
`resume_donors_after_borrow(replica_ranks)` 两个窄接口，供后续 lifecycle 编排调用；它们
只转发 CE effective-set 变更，不持有或操作 server、LB、engine 句柄。

真实的请求摘流、in-flight 请求处理、LB 路由提交、vLLM server sleep/wake、旧通信域
finalize/重建、最新参数的 target-only 同步以及 reclaim/destroy 均由后续 lifecycle 实现负责，
当前代码只在相应位置保留 `TODO(lifecycle)`。`sleep_for_runtime_test()`、
`wake_for_runtime_test()` 和 `set_global_steps()` 仅是 D3 显存/版本字段测试夹具，不能作为生产
生命周期接口或真实参数同步的替代品。

生产顺序必须遵守：

```text
CE suspend + 旧域 finalize
  -> drain/LB remove
  -> server sleep
  -> borrowed 创建与使用
  -> borrowed drain/remove/destroy
  -> server wake
  -> target-only CE sync 到当前 actor version + finalize
  -> CE resume
  -> LB READY
```

### 5.5 严格训练完成与 CE 逐参数校验

`MultiTaskFullyAsyncTrainer.fit()` 在原生 fit 正常返回后比较
`progress_bar.n` 与 Rollouter 计算出的 `total_train_steps`，只有两者相等才输出：

```text
MULTITASK_TRAINING_COMPLETE {"state": "COMPLETED", "completed": true, ...}
```

`[ASYNC MAIN] One component completed successfully` 只表示一个 Ray component 返回，
不能作为训练完成依据；没有上述回执时测试必须失败。

启用 `+multitask.parameter_validation.enabled=true` 后，
`MultiTaskCheckpointEngineWorker` 在接收每一个 named tensor 时记录 name、shape、dtype、
numel 和 SHA-256。`MultiTaskCheckpointEngineManager.validate_parameter_sync()` 在每次
bootstrap 或普通同步收尾时比较所有 CE Worker 的完整 manifest，并校验冻结的
`global_steps`，成功后输出 `CE_PARAMETER_VALIDATION`。该开关默认关闭，因为逐参数 hash
会增加同步开销；D0/D3/D4 验收脚本显式打开。`delta_flush` 暂不支持完整 manifest 校验，
开启严格校验时会明确失败，不能伪报成功。
