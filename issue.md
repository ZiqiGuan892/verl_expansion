# D1 启动问题记录

本文记录 D1 联调期间出现的两个启动问题、证据、根因和最小修复。两项修复都只修改
`verl-multi-task`，没有修改外层原生 `verl`。

## 1. `DEFAULT_ROUTING_CACHE_SIZE` 导入路径错误

### 现象

启用 `experimental_fully_async_standalone` 后，在加载扩展的
`MultiTaskLLMServerManager` 或扩展 LB 时出现：

```text
Traceback (most recent call last):
  File ".../multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py", line 17, in <module>
    from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE
ImportError: cannot import name 'DEFAULT_ROUTING_CACHE_SIZE' from 'verl.workers.rollout.llm_server'
```

### 根因

扩展代码按照另一版 verl 的模块边界，从 `verl.workers.rollout.llm_server` 导入
`DEFAULT_ROUTING_CACHE_SIZE`。当前配套的 verl 代码将该常量和
`GlobalRequestLoadBalancer` 放在 `verl/workers/rollout/router.py`；`llm_server.py`
只从 router 使用/转发路由类，并不保证导出该常量。因此这是插件导入路径错误，不是
LB 逻辑或 Ray 资源创建错误。

### 最小修复

修改两个扩展文件的导入路径：

```text
src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py
src/multi_task_scheduler/rollout/load_balancer.py
```

将：

```python
from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE
```

以及：

```python
from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer
```

分别改为从 `verl.workers.rollout.router` 导入。没有复制常量，也没有增加兼容回退逻辑，
避免掩盖实际版本/API 不一致。

## 2. TaskRunner 注册 GS 超时

### 现象

启动日志在 `MultiTaskFullyAsyncTaskRunner.run()` 处失败：

```text
ray.exceptions.GetTimeoutError: Get timed out: some object(s) not ready.

  File ".../multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py", line 48, in run
    ray.get(self.group_scheduler.attach_task.remote(task_id, context.current_actor), timeout=30)
```

当时的前置日志已经显示 profile 已解析、GroupScheduler 已创建；因此失败边界是
TaskRunner 等待 `GroupScheduler.attach_task` 返回，而不是 profile 选择或 vLLM 启动。

### 根因

Ascend/MindSpeed worker 在启动和导入设备运行时期间需要较长时间。原代码为 GS 注册 RPC
固定设置 30 秒等待，合法的慢启动被误判为 RPC 超时。这个问题不需要改变 GS 的数据结构、
Actor 校验、调度逻辑或重试语义。

### 最小修复

只修改：

```text
src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py
```

将 TaskRunner 启动阶段的等待上限从 30 秒改为 120 秒：

```python
ray.get(self.group_scheduler.attach_task.remote(task_id, context.current_actor), timeout=120)
```

退出清理阶段的 `detach_task` 仍使用原来的 30 秒。这样只扩大启动等待预算，不引入轮询、
环境变量、重复 RPC、ActorHandle 兼容分支或额外日志；超时之后仍按原有异常传播和清理逻辑
处理。

## 3. D2 borrowed runtime 无法读取 PlacementGroup 名称

### 现象

在服务器执行：

```bash
D2_RUNTIME_SCENARIOS=split bash ../D2_runtime_test.sh
```

native replica 和 native HTTP server 已经启动，但构造 D2 测试 spec 时失败：

```text
RuntimeError: native placement group has no globally discoverable name
  ... llm_server_manager.py, _snapshot_native_claims
```

### 根因

`ray.util.placement_group()` 返回的 `PlacementGroup` 是 ID/bundle 句柄，不能假设
存在公开的 `.name` 或 `._name` 属性。原实现从这两个属性读取名称，因此把实际存在的
native PG 误判为不可查找。PG 创建时的名称由 Ray placement-group table 保存。

### 最小修复

只修改插件文件：

```text
src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py
```

`_snapshot_native_claims()` 现在对每个 PG 调用：

```python
pg_info = ray.util.placement_group_table(placement_group)
pg_name = pg_info["name"]
pg_id = placement_group.id.hex()
```

再把 `pg_name`、`pg_id` 和 `bundle_index` 写入测试 claims。borrowed 创建阶段仍使用
`ray.util.get_placement_group(pg_name)` 查找已有 PG，不创建或删除 donor PG。若 Ray table
确实没有名称，则继续明确失败，因为该 PG 不满足跨任务借用所需的可发现条件。

### 验证

- 增加了无 `.name` 属性句柄、单 PG、多 PG 和缺失名称的单元测试。
- D1/D2/wiring 目标测试结果：`27 passed`。
- 修复后服务器的 `split` 场景继续进入 borrowed runtime 创建和主训练流程。

## 4. D2 smoke 测试训练 batch 过小

### 现象

PlacementGroup 问题修复后，D2 hook 已执行完成，但主训练阶段失败：

```text
AssertionError: number of items:[1] < k_partitions:[4]
```

失败位置为：

```text
verl/utils/seqlen_balancing.py, get_seqlen_balanced_partitions
```

### 根因

旧版 `D2_runtime_test.sh` 为缩短 smoke run，同时设置了：

```text
ppo_mini_batch_size=1
rollout.n=1
rollout.total_rollout_steps=1
```

因此只产生 1 个 rollout sample，最终只有 1 条序列，无法在 4 个训练 data-parallel
rank 之间进行均衡分片。该错误发生在 borrowed runtime 创建之后，与 PG、CE Worker、
HTTP server 或 vLLM engine 创建无关。

### 最小修复

`D2_runtime_test.sh` 现在显式向 `multi_task_run.sh` 追加 Hydra overrides，避免旧环境变量
或启动脚本默认值覆盖测试配置：

```text
actor_rollout_ref.actor.ppo_mini_batch_size=2
actor_rollout_ref.rollout.n=2
async_training.require_batches=1
async_training.trigger_parameter_sync_step=1
trainer.total_training_steps=1
trainer.total_epochs=1
rollout.total_rollout_steps=2
```

此时：

```text
required_samples = 2
最终序列数 = 2 × rollout.n(2) = 4
训练 data-parallel 数 = 4
```

脚本还增加了 batch 可被训练 DP 整除、rollout 步数足够的启动前检查。

### 验证

修复后的服务器 `split` 日志满足：

- `D2 runtime 场景通过：split`；
- pipeline exit status 为 `0`；
- 训练 rank 成功保存 model、optimizer 和 extra state；
- 输出 `Training completed or interrupted`，没有 `AssertionError` 或 `Traceback`。

本问题的训练 batch 修复不修改原生 verl 的 batch 平衡逻辑，只修改插件仓库的 D2 测试脚本。

## 5. D3 target-only bootstrap 在 Ascend finalize 阶段失败

### 现象

执行：

```bash
D3_RUNTIME_SCENARIOS=split bash ../D3_test.sh
```

borrowed runtime 已创建，D3 bootstrap 也已进入通信域 finalize，但任务失败：

```text
ray.exceptions.RayTaskError(AttributeError): ... HCCLCheckpointEngine.finalize
AttributeError: 'PyHcclCommunicator' object has no attribute 'destroyComm'
```

失败调用链为：

```text
MultiTaskCheckpointEngineManager.bootstrap_replica
  -> execute_checkpoint_engine(["finalize"])
  -> verl.checkpoint_engine.hccl_checkpoint_engine.HCCLCheckpointEngine.finalize
  -> self.pyhccl.destroyComm(self.pyhccl.comm)
```

### 根因

在当前 Ascend/vllm-ascend 组合中，`PyHcclCommunicator` 将底层 HCCL library wrapper
保存在 `pyhccl.hccl`，销毁接口是 `pyhccl.hccl.hcclCommDestroy(pyhccl.comm)`；
`destroyComm` 并不是 communicator 的方法。D3 设置 `rebuild_group=true` 后才会执行
这个原生 finalize 销毁分支，因此 D2 的 runtime 创建测试没有暴露该问题。

### 最小修复

只在插件仓库新增：

```text
src/multi_task_scheduler/checkpoint/hccl_checkpoint_engine.py
```

`MultiTaskHCCLCheckpointEngine` 继承原生 `HCCLCheckpointEngine`，只重写 `finalize()`：

1. 在拥有 communicator 时同步对应 NPU；
2. 优先兼容旧版 `destroyComm`，否则调用当前 vllm-ascend 的
   `pyhccl.hccl.hcclCommDestroy(pyhccl.comm)`；
3. 销毁成功后清理 communicator、rank、world size 和 buffer；
4. 销毁失败时保留原句柄和 buffer，并继续抛出原始异常。

`D3_test.sh` 改用 `multitask_hccl`，并通过
`checkpoint_engine.custom_backend_module` 让 actor、CE Worker、Trainer 使用同一个插件
后端；原生 `nccl` registry 和外层 `verl` 文件均未修改。

### 验证

- 新增 HCCL finalize 单元测试，覆盖当前 API、旧 API、无 communicator、关闭重建、销毁失败和注册继承；
- D3 CE/borrowed 目标测试结果：`37 passed`；
- 目标服务器需要重新执行 `D3_RUNTIME_SCENARIOS=split bash ../D3_test.sh`，确认出现
  `WEIGHTS_READY` 和 `FULL_SYNC_READY`。插件单元测试不能替代真实 HCCL 通信验证。

## 6. D3 borrowed Engine 启动时每卡显存不足

### 现象

`D3_RUNTIME_SCENARIOS=basic bash ../D3_test.sh` 在创建 borrowed replica 时失败。EngineCore
包装异常之前的 Worker 日志给出了实际原因：

```text
ValueError: Free memory on device (3.05/29.49 GiB) on startup is less than
desired GPU memory utilization (0.3, 8.85 GiB). Decrease GPU memory utilization
or reduce GPU memory used by other processes.
```

4 个 Worker 都在 `vllm_ascend.worker.Worker.init_device()` 处失败，随后
`WorkerProc initialization failed` 向上包装成 `Engine core initialization failed`，所以
`create_borrowed_replica()` 返回 `RUNTIME_CREATION_FAILED`。

### 根因

basic 场景的 native replica 是 TP=4，并且已经在同一批 4 张 NPU 上启动。原生
`vLLMReplica.sleep()` 在 `STANDALONE` 模式下只记录日志，不调用 Engine 的 sleep；因此
创建 borrowed replica 时，native 权重和 KV cache 仍占用显存。Ray 的 fractional GPU/CPU
资源配额只影响调度，不提供显存隔离，第二个 TP=4 Engine 只能看到每卡约 3 GiB 的空闲空间。

### 最小修复

只在插件仓库中增加 D3/D2 runtime smoke 使用的显存交接接口：

```text
src/multi_task_scheduler/rollout/http_server.py
```

`MultiTaskvLLMHttpServer.sleep_for_runtime_test()` 在 server 的主节点直接调用
`self.engine.sleep(level=1)`。level 1 将权重转移到 CPU 并丢弃 KV cache，释放 donor 的
设备显存；脚本显式设置 `enable_sleep_mode=true` 和 `free_cache_engine=true`。创建
borrowed 前，`MultiTaskLLMServerManager` 对本任务 smoke 使用的 native donors 调用该
接口，之后才调用原有 `create_borrowed_replica()`，不改变原生 PG、Worker 或
`launch_servers()` 的创建流程。

D3 bootstrap 完成后，测试钩子保持 donor asleep，并把 donor rank 从 CE 的暂时有效集合中
排除；普通同步只使用 actor worker 和 borrowed worker。这样不会让 donor CE Worker 与
borrowed CE Worker 在相同物理 NPU 上重复初始化 HCCL communicator。普通同步结束后，测试
清理 borrowed、恢复 donor，再清除 CE 排除集合。该显存交接是 D2/D3 的一次性测试夹具，
跨任务生产调度仍需由 donor 所属 TaskRunner 执行 sleep/wake。

### 回退范围与限制

上一轮试图在 `_cleanup_runtime()` 中扫描并回收 vLLM EngineCore 子进程的改动已回退；本次
修复没有引入 psutil、子进程扫描或修改原生 verl。若 EngineCore 在异常启动后留下子进程，
仍应先结束本次训练进程再重试。当前日志已证明本次失败点是显存启动阈值，不是上述清理告警。

### 验证

真实服务器上重新执行：

```bash
D3_RUNTIME_SCENARIOS=basic bash ../D3_test.sh
```

日志中应先出现 `RUNTIME_TEST_DONORS_SLEEPING`，再出现 borrowed 的
`D3_BOOTSTRAP_RESULT`。若仍在 `Worker.init_device()` 报显存不足，应保留该 Worker 的
完整日志，并检查 level-1 sleep 是否在目标 vllm-ascend 版本实际释放了设备显存。

## 7. D3 脚本无法覆盖旧版配置中的 `enable_sleep_mode`

### 现象

服务器运行 D3 脚本时，在 Hydra 组合配置阶段失败：

```text
omegaconf.errors.ConfigAttributeError: Key 'enable_sleep_mode' is not in struct
hydra.errors.ConfigCompositionException: Could not override
'actor_rollout_ref.rollout.enable_sleep_mode'.
To append to your config use +actor_rollout_ref.rollout.enable_sleep_mode=true
```

此时训练尚未创建 Ray Actor，也没有进入 vLLM 或 borrowed replica 创建流程。

### 根因与修复

服务器使用的 verl 配置 schema 没有声明 `enable_sleep_mode`，而 D3/D2 脚本使用普通
覆盖语法，Hydra 的 struct 校验因此拒绝该字段。将两个脚本中的 override 改为：

```text
+actor_rollout_ref.rollout.enable_sleep_mode=true
```

这样会在旧 schema 中追加该测试字段；已有的 `free_cache_engine` 字段继续使用普通覆盖。
MSC 的 `Profile "" not found` 日志是可选存储配置探测告警，不是本次 Hydra 失败的原因。

## 8. D3 普通同步中 HCCL communicator 参数错误

### 现象

Hydra 配置修复后，D3 在第一次普通参数同步阶段失败：

```text
MultiTaskCheckpointEngineManager.update_weights
  -> CheckpointEngineManager.build_process_group
  -> HCCLCheckpointEngine.init_process_group
  -> PyHcclCommunicator.hcclCommInitRank
RuntimeError: HCCL error: parameter error
```

### 根因

basic 场景的 native replica 和 borrowed replica 都是 TP=4，并且使用同一批 4 张物理
NPU。target-only bootstrap 完成后，borrowed 从 `pending_bootstrap` 移出；原实现的普通
同步会把 native 与 borrowed 的全部 CE Worker 合并到同一个 HCCL communicator。于是同一
物理 device 被分配给多个 HCCL rank。即使 native vLLM server 已经 sleep，native CE Worker
仍会占据该拓扑位置，HCCL 初始化会返回 `parameter error`。这不是 EngineCore OOM，也不
是 `rebuild_group` 可以单独解决的问题。

### 修复

`MultiTaskCheckpointEngineManager` 新增 `suspended_replica_ranks` 和两个方法：

```text
suspend_replicas_for_sync(ranks)
resume_replicas_for_sync(ranks)
```

D3 bootstrap 完成后，Trainer 暂时排除已经 sleep 的 donor rank，普通 CE 同步只包含
actor workers 和 borrowed workers；同步完成、borrowed 清理并恢复 donor 后，再解除排除。
`D3_test.sh` 检查 `DONORS_SLEEPING_BORROWER_ONLY_EFFECTIVE` 标记。

### 验证

本地目标测试：`38 passed`。服务器必须使用包含该修改的插件代码重新运行：

```bash
D3_RUNTIME_SCENARIOS=basic bash ../D3_test.sh
```

若日志仍显示 native 与 borrowed workers 同时进入 `build_process_group`，说明服务器没有
加载最新插件代码或脚本仍使用旧版本。

## 9. donor 唤醒后 rollout 参数版本为空

### 现象

HCCL 拓扑修复后，D3 可以完成 bootstrap 和普通同步，但第一次采样组 batch 时失败：

```text
TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'

verl/experimental/fully_async_policy/detach_utils.py:153
param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
```

### 根因

为避免 donor 与 borrowed CE Worker 在相同 NPU 上重复进入 HCCL，D3 普通同步只更新
borrowed。donor 随后被唤醒继续生成，但它被排除在本轮 CE 同步之外，vLLM HTTP server
的 `global_steps` 仍然是初始化值 `None`。生成结果携带空参数版本，batch 组装时执行
`None - None` 失败。

### D3 测试路径修复

`cleanup_d3_runtime()` 增加可选的 `global_steps` 参数。Trainer 清理 borrowed、唤醒 donor
后，显式调用继承自原生 server 的 `set_global_steps(global_steps)`，使 donor 生成结果
带上当前参数版本。该操作只补齐 D3 测试交接路径中的版本元数据，不更新 donor 的模型权重，
不能替代真实的 target-only 参数同步；不修改原生 verl 的 batch 逻辑。

## 10. donor sleep/wake 与 CE membership 的职责边界

### 背景

前述 D3 修复曾经容易被理解为已经实现了完整的 donor sleep/wake 生命周期。当前设计已经
明确收窄：sleep 和 wake 由后续 lifecycle 开发负责，本插件当前只实现 CE effective-set 的
必要切换。

### 当前实现

`MultiTaskCheckpointEngineManager` 提供：

```text
suspend_replicas_for_sync(replica_ranks)
resume_replicas_for_sync(replica_ranks)
```

Trainer 通过以下窄接口调用它们：

```text
suspend_donors_for_borrow(replica_ranks)
resume_donors_after_borrow(replica_ranks)
```

前者把 donor rank 从后续 CE effective snapshot 排除，避免 donor CE Worker 与 borrowed CE
Worker 在同一物理设备上同时进入 HCCL 通信域；后者在 donor 已完成恢复和追平后重新允许其
参加同步。这些方法不创建/销毁 Worker，不操作 vLLM engine，不修改 LB 路由，也不负责旧
通信域 finalize。

`sleep_for_runtime_test()`、`wake_for_runtime_test()` 和 `set_global_steps()` 仍然只用于
D2/D3 smoke：前两者用于测试显存交接，后者只恢复测试结果中的版本字段。它们不是生产
sleep/wake 接口，也不能保证 donor 的权重已经追平。

### 后续 lifecycle 实现必须补齐的顺序

```text
CE suspend + 旧域 finalize
  -> drain/abort 请求并从 LB 摘流
  -> donor server sleep
  -> borrowed 创建/使用
  -> borrowed drain/remove/destroy
  -> donor server wake
  -> target-only CE 参数同步到当前 actor version + finalize
  -> CE resume
  -> LB READY
```

上述真实 server 操作、in-flight 请求处理、通信域清理/重建、参数追平和 reclaim/destroy
均在当前代码中保留为 `TODO(lifecycle)`，由其他开发者实现。D3 smoke 当前先使用测试
sleep 释放显存，再由 Trainer 更新 CE 投影；该顺序仅用于测试，不是生产事务顺序。

## 11. S5、S7、S8、S9 批量回归失败（2026-09-25）

### S5：第二个 shared_bundle borrower 创建失败，错误文本为空

日志节选：

```text
RuntimeError: shared_bundle create 1 failed: {
  'replica_rank': 2, 'state': 'FAILED',
  'error': {'code': 'RUNTIME_CREATION_FAILED', 'message': ''},
  'cleanup': {'kill_requested': [], 'errors': [], 'release_confirmed': False}}
```

代码中确认存在一个调度阻塞条件：`_create_workers_from_claims()` 在创建 CE Actor 前，
先在首个 claim 的 PG/bundle 上运行原生 `get_master_addr_port`。它是普通 Ray task，
默认申请 1 个逻辑 CPU。S5 bundle 有 2 CPU，donor CE 占 1 CPU，borrower A 占 0.5 CPU，
即使 A 的 engine 已 sleep，其 CE Actor 的资源预约仍保留，只剩 0.5 CPU。此时 borrower B
的端口任务不能运行，尽管 B 的 CE 本身只需要 0.5 CPU。

这个条件可以解释超时前没有创建 CE Actor、`kill_requested=[]` 和空 `TimeoutError`
消息，但原日志没有 traceback，不能仅凭空文本断言本次一定是该异常。
Ray 的默认任务资源规则见 [Ray Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)。

修复位于 `src/multi_task_scheduler/rollout/replica.py`：仅为该短暂端口任务设置
`num_cpus=0`，保留原来的 PG/bundle 调度约束；CE Actor 的 `0.25 NPU + 0.5 CPU`
申请不变。donor + A + B 总计 `1 NPU + 2 CPU`，不需要为本场景增大 M，也不销毁 donor。

同时记录 `creation_stage`，超时包含阶段、lease、时间上限；Manager 在创建失败时记录
异常 `type/message/stage/cause/traceback` 并输出完整 traceback，避免下次仍只有空文本。
`release_confirmed`、`released` 不因记录异常或发出 kill 请求而改成 true。

### S7：merge_world_size 的 NPU 可见设备顺序错误

日志节选：

```text
torch_npu.npu._lazy_init() -> torch_npu._C._npu_init()
RuntimeError: ... aclInit, error code is 107001
[Error]: Invalid device ID.
value 0 for parameter userDevId is invalid. Expected value: [0, 0).
```

日志能说明进程无法初始化有效设备，不能单独证明显存耗尽或上一场景有进程残留。
代码中发现独立且可复现的设备排序缺陷：两个 donor 的 local_rank 都从 0 开始，旧
`_reindex_test_claims()` 按 donor local_rank 排序。例如 donor A 的设备为 `[4,5]`、
donor B 为 `[6,7]`，合并后会得到 `[4,6,5,7]`。继承的原生 `launch_servers()` 按 CE
Worker 顺序拼接设备列表，HTTP Server 将其写入 `ASCEND_RT_VISIBLE_DEVICES`。
CANN 文档要求此变量内的设备 ID 升序排列，见
[ASCEND_RT_VISIBLE_DEVICES](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0028.html)。

修复：

- 测试 spec builder 按节点分组、按真实 Ray accelerator ID 的**数值**排序，再生成
  borrower rank/local_rank；保留每项 claim 的 PG/bundle/device/donor 归属。
- snapshot 的 `local_gpu_index` 使用实际设备索引，避免误把 donor local_rank 当设备号。
- borrowed placement 校验和实际 CE Worker 校验都检查 NPU 设备顺序。生产 GS spec
  不被偷偷重排；不满足顺序要求时明确失败。
- 打印 `BORROWED_WORKER_PLACEMENT`，包含 rank/world_size、node/device、PID 和设备可见性
  环境变量。原生 HTTP Server 仍负责设置并打印最终 engine 的可见设备列表。

不能只在 HTTP Server 内排序设备字符串而保持 CE ranks 不变，否则 CE 与 engine 的 rank
设备对应关系可能错位。该修复消除已确认的排序缺陷；本次 ACL 错误是否完全由它引起，
仍需在服务器复测。如果仍失败，应对照新映射日志与 EngineCore 原始 traceback继续定位，
不能自动将原因归为残留进程。

### S8 / S9：测试场景名误传给 placement builder，以及 Worker 数量误判

日志节选：

```text
prepare_d4_runtime_smoke -> _build_d2_test_spec(scenario)
ValueError: unknown D2 runtime scenario: idempotent
ValueError: unknown D2 runtime scenario: concurrent_idempotent
```

`idempotent` / `concurrent_idempotent` 是命令重试方式，不是两种新的 placement 算法。
`rollouter.py::prepare_d4_runtime_smoke()` 将这两者映射为 `basic` 布局；TaskRunner 仍按
原场景分别提交串行两次 create 或并发两次 create，不把场景降级成只创建一次。

进一步发现 TaskRunner 原断言 `worker_count == server_count == 1` 也会使默认 TP=4
的单个 replica 失败。修为 `worker_count == spec.world_size`，
`server_count == claims 中不同 node_id 的数量`，并继续核对同 rank、同 server。
回执附带预期数量；若幂等断言失败，已经注册的 borrowed CE 也会在测试 finally 中注销。

注意：`MULTITASK_TRAINING_COMPLETE` 只证明 native 训练步骤完成。D4 smoke 在 native
训练返回后执行，之后的创建/幂等验证仍可能失败，不能据此前一个标记判定整个 S 通过。

### 验证与服务器复测

本地无 Ray/NPU 回归：`test_borrowed_runtime.py`、`test_borrowed_contract.py`、
`test_d4_lifecycle.py`、`test_checkpoint_membership.py`、`test_hccl_checkpoint_engine.py`，
共 **60 passed**。新增用例覆盖剩余 0.5 CPU 时的端口任务请求、空异常诊断、合并设备排序、
保留 PG/bundle 归属、S8/S9 映射和 TP=4 完整 smoke 控制逻辑（RPC 使用替身），并验证
错误 Worker 数量仍会失败。本地结果不等价于真实 NPU 运行通过。

在服务器原启动目录、更新插件文件后执行：

```bash
# 每次一个场景，可分别观察控制台和该场景日志
D0_D4_SCENARIOS=S5 bash ../D0_D4_comprehensive_test.sh
D0_D4_SCENARIOS=S7 bash ../D0_D4_comprehensive_test.sh
D0_D4_SCENARIOS=S8 bash ../D0_D4_comprehensive_test.sh
D0_D4_SCENARIOS=S9 bash ../D0_D4_comprehensive_test.sh

# 或由现有 batch 严格串行执行这四项，逐项保存结果
D0_D4_BATCH_SCENARIOS=S5,S7,S8,S9 bash ../D0_D4_batch_test.sh
```

这些场景目前仍属阶段验证。修复后阶段链路成功时，严格综合脚本仍可能报告
`INCOMPLETE`（缺少完整 borrowed generate/sync/生命周期证据），不能将其与本次
`FAIL` 混为一谈，也不修改通过标准来掩盖错误。
