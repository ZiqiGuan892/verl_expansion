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

此外要求 main_ppo 和 `tee` 均退出为 0，并拒绝包含 `Traceback` 或 `AssertionError` 的日志。
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

### 3.4 EngineCore 清理与重复运行

借用的 HTTP server 与 donor 共享物理 NPU，D2/D3 测试结束时不能只依赖 `ray.kill`。
插件 `MultiTaskvLLMHttpServer.shutdown_engine()` 会先调用 vLLM `AsyncLLM.shutdown()`，
兼容旧版的 `shutdown_background_loop()`；如果 EngineCore 在初始化异常时尚未挂到
`self.engine`，则只清理当前 HTTP actor 的递归子进程。随后
`MultiTaskvLLMReplica._cleanup_runtime()` 才执行 Ray actor kill。

这样可以在连续 smoke run 之间释放 EngineCore 的显存、IPC 和设备上下文。该方法是
best-effort 清理，receipt 会保留清理错误；它不宣称 lease 已归还，也不执行 donor PG
删除。若完整日志仍显示 EngineCore 启动失败，应继续查看通用包装异常之前的
`WorkerProc`/`torch_npu`/`ACL`/`HCCL`/`OOM` 根因行。

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
