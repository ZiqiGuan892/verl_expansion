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
