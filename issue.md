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

D3 bootstrap 完成后，测试钩子释放 borrowed 的 KV cache，再恢复 donor；这样参数同步
仍能验证 borrowed Worker，而 donor 可以继续服务主训练循环。该显存交接是 D2/D3 的
一次性测试夹具，跨任务生产调度仍需由 donor 所属 TaskRunner 执行 sleep/wake。

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
