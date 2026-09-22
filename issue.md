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

