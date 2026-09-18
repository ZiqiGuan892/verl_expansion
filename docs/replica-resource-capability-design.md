# VERL 资源共享能力扩展设计：Replica 生命周期

## 1. 目标与范围

本文只设计 5.2 节的任务侧执行能力，不实现 GlobalScheduler 的调度策略，也不修改 verl 原代码。扩展以 `verl-multi-task` 插件包的形式接入，在任务内部为上层编排提供以下能力：

1. 根据 GS 授权的 node ID、GPU ID 创建 borrowed replica；
2. 销毁 borrowed replica；
3. 休眠 native 或 borrowed replica；
4. 唤醒已休眠 replica；
5. 执行 borrowed replica 的强制回收；
6. 在 replica 集合变化后调整 Checkpoint Engine 通信拓扑。

本层不决定 donor、borrower、资源数量和调度时机，不直接修改 GS 全局资源账本，不把 donor 的 ResourcePool、Placement Group 或 CE Worker 交给 borrower。

## 2. 总体架构

```text
GlobalScheduler
      │ 只发送已授权的操作命令和物理租约
      ▼
MultiTaskTaskRunner
      ├── MultiTaskFullyAsyncRollouter
      │     └── MultiTaskLLMServerManager
      │           ├── native replicas
      │           ├── borrowed replicas
      │           └── MultiTaskGlobalRequestLoadBalancer
      └── MultiTaskFullyAsyncTrainer
            └── MultiTaskCheckpointEngineManager
                  └── communication topology
```

TaskRunner 是 GS 进入任务内部的唯一控制入口。Rollouter Manager 负责 server/replica 生命周期，Trainer 内的 CE Manager 负责有效 replica 集合和同步通信拓扑。GS 不直接调用这两个普通对象。

### 2.1 Replica 分类

| 类型 | 创建者 | 资源来源 | 回收方式 | 是否允许销毁 |
|---|---|---|---|---|
| native replica | 任务启动流程 | 本任务 ResourcePool/PG | 休眠后可唤醒 | 不作为借卡步骤销毁 |
| borrowed replica | borrower Manager | GS 授权的 node/GPU slot | 中断、摘流、移除后休眠或销毁 | 允许 |

native replica 的对象、worker handle、PG 和资源来源证明始终由 donor 持有。borrowed replica 由 borrower 创建并持有自己的 server 对象及生命周期引用；GS 只保存资源租约和状态摘要。

## 3. 核心数据模型

### 3.1 `ReplicaPlacement`

```python
@dataclass(frozen=True)
class ReplicaPlacement:
    node_id: str
    gpu_ids: tuple[int, ...]
    lease_id: str
    lease_epoch: int
```

约束：

- `node_id` 必须是 GS 授权的真实节点；
- `gpu_ids` 非空、非负且不能重复；
- `lease_id + lease_epoch` 用于防止旧命令误操作新租约；
- 对外传递时只包含可序列化信息，不包含 Ray ActorHandle、PG handle 或 CUDA IPC 对象。

### 3.2 `ReplicaDescriptor`

```python
@dataclass(frozen=True)
class ReplicaDescriptor:
    replica_id: str
    owner_task_id: str
    kind: Literal["native", "borrowed"]
    placement: ReplicaPlacement | None
    serving_version: int | None
    lifecycle_epoch: int
```

Descriptor 是跨组件回执中的摘要。真实 replica 对象只能存在于 Manager/Trainer 所属进程，不能通过 GS 传递。

### 3.3 `ReplicaOperation`

```python
@dataclass(frozen=True)
class ReplicaOperation:
    operation_id: str
    operation: Literal["ADD", "WAKE", "SLEEP", "RECLAIM", "DESTROY"]
    replica_id: str
    lease_id: str | None
    lease_epoch: int | None
```

所有操作必须携带唯一 `operation_id`。组件保存最近完成的操作结果：相同 operation 重试返回原结果；不同 operation 试图操作旧对象时返回 `STALE_OPERATION`。

## 4. Manager 组件设计

### 4.1 `MultiTaskLLMServerManager`

Manager 维护三组集合：

```text
native_replicas       本任务初始创建的副本，生命周期永久归 donor
borrowed_replicas     当前借入的副本及其 lease
active_replicas       当前允许接收推理请求的副本
```

建议接口：

```python
async def prepare_borrowed_replica(
    operation: ReplicaOperation,
    placement: ReplicaPlacement,
) -> PreparedReplica

async def activate_replica(operation: ReplicaOperation) -> ActiveReplica
async def sleep_replica(operation: ReplicaOperation) -> OperationReceipt
async def wake_replica(operation: ReplicaOperation) -> OperationReceipt
async def reclaim_borrowed_replica(operation: ReplicaOperation) -> OperationReceipt
async def destroy_borrowed_replica(operation: ReplicaOperation) -> OperationReceipt
```

Manager 不暴露具体 actor 创建细节。通过 `ReplicaFactory` 插件适配不同 verl/vLLM 版本：

```python
async def create(
    *,
    replica_id: str,
    placement: ReplicaPlacement,
    config,
) -> PreparedReplica
```

Factory 必须同时完成：

1. 使用 node affinity 将 actor 放到指定 node；
2. 使用显式 GPU 绑定和 local rank 建立 borrower server；
3. 创建 borrower 自己的 HTTP server、推理后端和同步接收端；
4. 返回 server handles、CE endpoints 和健康状态；
5. 禁止调用会新建 ResourcePool/PG 的 `init_standalone()`。

### 4.2 `BorrowedReplica`

```text
PREPARED
   │ bootstrap + CE topology 成功
   ▼
ACTIVE
   │ stop admission
   ▼
DRAINING
   ├── sleep → SLEEPING
   └── destroy → DESTROYED
```

状态规则：

- `PREPARED` 不得被 LB 路由；
- `ACTIVE` 才能加入 effective replica 和 LB；
- `DRAINING` 禁止新请求，但允许处理 abort/续推；
- `SLEEPING` 只允许执行 wake；
- `DESTROYED` 是终态；
- 状态转移和 operation 校验必须在 Manager 内串行化。

## 5. 五类能力的实现方案

### 5.1 Replica 创建

创建分为 Prepare、Bootstrap、Activate 三阶段，不能合并成一个“创建成功”标志。

```text
GS → TaskRunner: ADD(operation, placement)
TaskRunner → Rollouter: prepare
Rollouter → Manager: ReplicaFactory.create
Manager → Ray: node affinity + GPU binding 创建 server/receiver
Ray → Manager: handles + health
Manager → Trainer: bootstrap endpoint
Trainer → CE: 加入临时同步集合并同步 serving version
Trainer → CE: 重建通信拓扑
Rollouter → LB: commit READY
Manager: active_replicas.add
TaskRunner → GS: ACTIVE receipt
```

创建阶段失败时，副本保持不可路由。已创建的 server 按 `abort → sleep` 清理；如果清理失败，返回 `QUARANTINED`，不能向 GS 报告 GPU 已释放。

### 5.2 Replica 销毁

销毁只允许 borrowed replica 使用，且不等同于回收：

```text
确认 kind=borrowed
→ LB 摘流
→ 等待/中断 in-flight
→ 从 CE effective set 移除
→ 关闭 CE 旧连接并重建拓扑
→ 停止 server/backend actor
→ 删除 Manager 引用
→ 向 GS 回报 DESTROYED
```

native replica 调用销毁接口必须直接拒绝。native 资源归还采用 sleep，保留 PG 和 replica 对象，后续由 donor 自己 wake。

### 5.3 Replica 休眠

休眠分为 native sleep 和 borrowed sleep：

- native：从 LB 和 CE active 视图摘除，调用原生 `sleep()` 释放权重/KV cache，但不删除 worker、PG 和 Manager 引用；
- borrowed：先摘流和 abort，再从 CE/Manager 移除，调用 `sleep()`；如果 GS 明确要求释放 actor，改走 destroy。

休眠完成的必要条件是：LB 不再路由、in-flight 已清零或已生成 partial-rollout 回执、server 确认 sleep 成功。只有这些条件全部满足，才能上报 slot 可租借。

### 5.4 Replica 唤醒

唤醒不是单一的 `wake_up()` RPC，而是一个由 TaskRunner 串行协调的事务。整个过程中 replica 都不能被 LB 路由；只有 CE、权重版本、server 健康状态和 LB 路由提交全部成功后，才返回 `ACTIVE`。

#### 5.4.1 公共时序和函数责任

```mermaid
sequenceDiagram
    participant GS as GlobalScheduler
    participant TR as MultiTaskTaskRunner
    participant M as MultiTaskLLMServerManager
    participant R as Replica(native/borrowed)
    participant LB as MultiTaskGlobalRequestLoadBalancer
    participant T as MultiTaskCheckpointEngineManager
    participant S as Supervisor/Runtime

    GS->>TR: WAKE(operation_id, replica_id, lease_epoch)
    TR->>M: wake_replica(operation)
    M->>M: acquire_operation_lock(); validate_state()
    M->>LB: mark_waking(replica_id)
    M->>R: validate_ownership_and_lease()
    R->>S: wake_runtime() / wake_up()
    S-->>R: runtime_ready + health_snapshot
    M->>TR: acquire_replica_sync_gate(operation_id)
    TR->>T: prepare_candidate(replica)
    T->>T: add_candidate + rebuild_topology(topology_epoch)
    T->>R: bootstrap_if_version_mismatch(serving_version)
    R-->>T: BootstrapReceipt
    T->>T: commit_effective(replica, topology_epoch)
    M->>LB: commit_routable(replica_id, routing_epoch)
    M->>M: active_replicas.add(); recompute_capacity()
    TR-->>GS: ACTIVE(receipt)
    TR->>M: release_replica_sync_gate(operation_id)
```

各步骤的组件和函数职责如下：

| 步骤 | 调用方 → 被调用方 | 函数/接口 | 原生 verl 可复用部分 | 插件新增或重构部分 |
|---|---|---|---|---|
| 1. 接收命令 | GS → TaskRunner | `handle_replica_operation(WAKE)` | 无；GS 不属于 verl 原生组件 | 新增 operation adapter、operation ID/lease epoch 校验 |
| 2. 校验状态 | TaskRunner → LLMServerManager | `wake_replica(operation)`、`validate_state()` | `RolloutReplica` 的状态字段可复用 | 新增状态机、幂等表和 operation lock |
| 3. 防止接流 | Manager → LB | `mark_waking(replica_id)` | 原生 LB 的 server 表可复用 | 新增 `WAKING`/`DRAINED` 门控；不能直接调用 `add_servers()` |
| 4. 校验资源 | Manager → Replica | `validate_ownership_and_lease()` | native 可复用本地 replica/PG 存在性检查 | borrowed 新增 lease epoch、物理 GPU 映射和 donor 独立性检查 |
| 5. 恢复运行时 | Manager → Replica/Runtime | native `wake_up()`；borrowed `wake_runtime()` | `RolloutReplica.wake_up()` 和 server `wake_up.remote()` 在 backend 支持时可复用 | borrowed 新增 Supervisor RPC、进程存活检查和外部 endpoint 恢复 |
| 6. 后端健康检查 | Replica → server/engine | `health()`、`get_server_address()` | vLLM 的地址查询可复用 | 新增权重存在性、显存、CUDA_VISIBLE_DEVICES、进程和端口检查 |
| 7. 获取同步锁 | TaskRunner → Trainer | `acquire_replica_sync_gate()` | 原生 update_weights 流程可作为被保护操作 | 新增跨 Rollouter/Trainer 的任务级 gate，不能只用本地 asyncio lock |
| 8. 准备 CE 候选 | Trainer → CE Manager | `prepare_candidate(replica)` | `CheckpointEngineManager.add_replicas()` 可用于列表登记 | 重构为 candidate/effective 两阶段，不能登记后立即接流 |
| 9. 重建拓扑 | CE Manager → backend | `build_process_group(rollout)` | 原生 `prepare()`、`build_topology()`、`init_process_group()` 可复用 | 新增 topology adapter、旧 group 清理、epoch 和 backend 能力检查 |
| 10. 追平权重 | CE Manager → Replica CE endpoint | `bootstrap_if_version_mismatch(version)` | Worker 的 `update_weights()`/ServerAdapter 可复用底层传输 | 新增只针对唤醒 replica 的 bootstrap；原生 `update_weights()` 不能直接表达 partial membership |
| 11. 提交 CE | CE Manager → Trainer projection | `commit_effective(replica, epoch)` | `self.replicas` 列表可复用 | 新增 effective snapshot、epoch 校验和失败回滚 |
| 12. 提交 LB | Manager → LB | `commit_routable(replica_id, routing_epoch)` | 原生 `add_servers()` 可作为最终动作 | 新增 READY/version/epoch 原子门控 |
| 13. 恢复并发额度 | Manager → Rollouter | `active_replicas.add()`、`recompute_capacity()` | replica 的 `max_concurrency` 属性可复用 | 新增 active projection 和容量重新计算 |
| 14. 返回回执 | TaskRunner → GS | `OperationReceipt(ACTIVE)` | 无 | 新增全步骤 receipt、耗时、版本和失败阶段 |

`wake_up()`、`add_replicas()`、`build_process_group()`、`update_weights()` 和 `add_servers()` 都只能作为底层动作复用，不能把任何一个原生调用单独当作“唤醒完成”。

#### 5.4.2 native replica 唤醒路径

native replica 的运行时对象和 PG 始终由 donor 任务持有，因此唤醒不创建新的 replica ID、ResourcePool、Placement Group 或 CE Worker：

```text
GS → TaskRunner.handle_replica_operation(WAKE)
→ MultiTaskLLMServerManager.wake_replica()
→ NativeReplica.validate_state_and_pg()
→ MultiTaskGlobalRequestLoadBalancer.mark_waking()
→ NativeReplica.wake_up()
   └─ RolloutReplica.wake_up()
      └─ server.wake_up.remote()
→ NativeReplica.health()
→ TaskRunner.acquire_replica_sync_gate()
→ MultiTaskCheckpointEngineManager.prepare_candidate(native)
→ CheckpointEngineManager.add_replicas([native])  # 仅列表登记
→ CheckpointTopologyAdapter.rebuild(topology_epoch)
→ 如 serving_version 不一致：Native CE endpoint.bootstrap_weights()
→ MultiTaskCheckpointEngineManager.commit_effective(native)
→ LB.commit_routable() → 原生 add_servers()
→ active_replicas.add(native); recompute_capacity()
→ 返回 ACTIVE
```

native 的 `wake_up()` 可以复用原生实现，但当前 vLLM STANDALONE server 的 `wake_up()` 可能是空操作。因此 `NativeReplica.health()` 必须确认权重和 KV cache 真的恢复；若 backend 只恢复 KV cache 或仍缺少权重，必须继续走 CE bootstrap 或返回 `UNSUPPORTED`，不能直接提交 READY。

#### 5.4.3 borrowed replica 唤醒路径

borrowed replica 由 borrower 持有独立 server、CE Worker 和运行时进程。唤醒前必须确认 lease 仍然有效，并验证 donor 没有重新占用该物理 slot：

```text
GS → TaskRunner.handle_replica_operation(WAKE)
→ MultiTaskLLMServerManager.wake_replica()
→ BorrowedReplica.validate_lease(expected_epoch)
   └─ LeaseResourceBinder.verify(node_id, physical_gpu_ids, lease_id)
   └─ Supervisor.assert_slot_owned_by(replica_id)
→ MultiTaskGlobalRequestLoadBalancer.mark_waking()
→ BorrowedReplica.wake_runtime()
   └─ Supervisor.wake_process() 或恢复 SLEEPING runtime
   └─ 恢复 CUDA context、server endpoint 和本地控制通道
→ BorrowedReplica.health()
→ TaskRunner.acquire_replica_sync_gate()
→ MultiTaskCheckpointEngineManager.prepare_candidate(borrowed)
   └─ 使用 borrower 自有 CE Worker handles
   └─ 禁止混入 donor workers
→ CheckpointTopologyAdapter.rebuild(topology_epoch)
→ BorrowedReplica.bootstrap_weights(serving_version)
   └─ 仅更新当前 borrowed endpoint
→ MultiTaskCheckpointEngineManager.commit_effective(borrowed)
→ LB.commit_routable() → 原生 add_servers()
→ active_replicas.add(borrowed); recompute_capacity()
→ 返回 ACTIVE
```

如果 borrowed runtime 在休眠时保留了完整且版本正确的权重，`bootstrap_weights()` 可以降级为 `verify_serving_version()`；如果进程曾被销毁、权重被卸载或版本落后，则必须先通过临时 CE membership 完成 bootstrap，再提交 effective set。borrowed 唤醒绝不能调用原生 `init_standalone()`，否则会重新申请 Ray PG/GPU；应调用 `BorrowedRuntimeFactory.start_or_wake()`。

#### 5.4.4 两类 replica 的复用与新增点

| 能力 | native replica | borrowed replica | 处理方式 |
|---|---|---|---|
| 状态校验 | 检查本地对象和 PG | 检查 lease、slot 和 Supervisor | 公共 `validate_state()`，策略不同 |
| server 唤醒 | 原生 `RolloutReplica.wake_up()` | `Supervisor.wake_process()`/endpoint wake | 公共 `wake()`，两个 RuntimeAdapter |
| 权重/KV 恢复 | 原生 server wake，必要时 bootstrap | 独立 runtime wake，必要时 bootstrap | 复用底层 CE/ServerAdapter，新增版本检查 |
| CE Worker | 原有 native Worker handles | borrower 新建 Worker handles | endpoint 协议一致，句柄绝不共享 |
| CE 加入 | native worker 加入候选拓扑 | borrowed worker 加入候选拓扑 | 公共 `prepare_candidate/rebuild/commit` |
| LB 提交 | 原生 `add_servers()` 最终提交 | 同一 `add_servers()` 最终提交 | 外包 `commit_routable()`，统一 READY 门控 |
| PG 处理 | 保留原 PG，不重新创建 | 不创建新的 PG，不触碰 donor PG | ResourceBinder 策略不同 |
| lease | 无跨任务 lease | 必须校验 lease epoch 和物理 slot | borrowed 专有前置条件 |
| 失败后状态 | 回到 SLEEPING 或 QUARANTINED | 回到 SLEEPING、QUARANTINED 或释放 lease | 公共回滚协议，错误码不同 |

#### 5.4.5 唤醒失败和回滚

任何步骤失败，都必须执行逆向清理，且不能部分进入 active：

```text
runtime wake 失败
→ 保持 SLEEPING 或标记 QUARANTINED

CE prepare/rebuild/bootstrap 失败
→ 不提交 effective set
→ 清理候选 topology/临时 group
→ borrowed runtime 回到 SLEEPING；native 保留原有 PG 但不接流

LB commit 失败
→ 保持 CE effective set 不可路由或回滚 CE membership
→ 清理 routing candidate
→ 返回 RETRYABLE/QUARANTINED

全部成功
→ active_replicas.add()
→ 恢复并发额度
→ 返回 ACTIVE
```

在返回 `ACTIVE` 前，receipt 至少要包含 `replica_id`、`kind`、`lease_epoch`（borrowed）、`serving_version`、`topology_epoch`、`routing_epoch`、server health、实际 GPU 映射和各阶段耗时。这样 GS 能区分“命令已接受”“运行时已唤醒”“已完成 CE bootstrap”和“已经可以接流”。

### 5.5 Borrowed Replica 强制回收

强制回收必须与销毁区分：

```text
停止向目标 replica 发新请求
→ abort_all_requests
→ 由 partial-rollout 机制在其他 READY replica 上续推
→ 等待 release_server / in-flight 计数归零
→ 从 LB 移除
→ 从 CE effective set 移除
→ 清理通信连接
→ sleep 或 destroy
→ 回报资源释放
```

回收只针对 borrowed replica。任何仍在途的请求都必须有续推结果或明确失败记录，不能因 server actor 被销毁而静默丢失样本。

## 6. Checkpoint Engine 通信拓扑

`MultiTaskCheckpointEngineManager` 增加独立的 effective replica projection：

```text
native replicas + ACTIVE borrowed replicas
```

每次集合变化生成递增 `topology_epoch`，并执行：

```text
获取 replica-sync gate
→ 冻结当前 effective snapshot
→ add/remove replica endpoint
→ 关闭旧 process-group / connection
→ 使用原生 backend 重建 topology
→ 验证所有 endpoint 已加入目标 epoch
→ 释放 gate
```

bootstrap 只同步新 replica 到当前 serving version，不更新已有 replica。原生周期同步仍按 verl 原有触发条件执行。拓扑重建期间禁止 LB 把新 replica 标记为 READY。

## 7. 一致性与异常处理

必须满足以下不变量：

1. 一个 GPU slot 同时只能存在一个有效 lease；
2. 未完成 bootstrap 的 replica 永远不能接流；
3. LB、CE、Manager 三个视图都成功更新后，操作才算完成；
4. “已发送命令”不等于“资源已交接”；必须等待执行回执；
5. timeout 不得推断 GPU 已释放，失败资源进入隔离状态；
6. 重试使用同一 operation ID，不能创建第二个副本；
7. 原生参数同步不能被半完成的 add/remove 永久阻塞；
8. GS 不可达时，任务只能完成本地安全回滚，不能自行把 slot 转租给其他任务。

## 8. 插件化接入方式

不修改 verl 原文件，插件只提供：

- `MultiTaskFullyAsyncRollouter` 和 `MultiTaskLLMServerManager`；
- `MultiTaskCheckpointEngineManager`；
- `ReplicaFactory`、`BorrowedReplica` 和 lifecycle controller；
- 与 GS 通信的 TaskRunner command adapter。

原生入口通过已有 runtime profile 选择插件 ActorClass；profile 关闭时完全走原生 verl 路径。不同 vLLM/verl 版本只需要替换 Factory 和 CE backend adapter，不改变上层生命周期协议。

## 9. 分阶段实施计划

| 阶段 | 交付内容 | 验证方式 |
|---|---|---|
| P0 | 数据模型、状态机、幂等和错误码 | 纯 Python unit |
| P1 | Manager native sleep/wake 与 borrowed prepare | mocked Ray |
| P2 | Node affinity、GPU 绑定、borrowed server/receiver factory | 单机真实 Ray |
| P3 | CE effective set、通信组重建、旧连接清理 | CPU/多进程 backend |
| P4 | abort、partial-rollout、强制回收 | vLLM async 集成测试 |
| P5 | 多节点 GPU、跨 job lease、故障隔离 | 真实集群验收 |

在 P4 之前只能声称“控制协议和任务侧执行链可用”，不能声称多任务 GPU 资源共享已经完成。

## 10. 测试矩阵

- 状态迁移：合法迁移、非法迁移、重复操作、过期 operation；
- 资源边界：重复 GPU、错误 node、过期 lease、native 销毁拒绝；
- 创建回滚：server 创建失败、CE bootstrap 失败、拓扑重建失败；
- 回收顺序：摘流先于 abort，CE/LB 移除先于销毁，释放回执晚于实际释放；
- 请求完整性：abort 后 partial rollout 能续推，不能丢失样本；
- 拓扑一致性：epoch 单调、旧连接清理、有效集合快照稳定；
- 运行验收：真实 Ray actor、node affinity、GPU 可见性、vLLM server、CE 同步和跨任务交接。

## 11. 架构理想设想与当前 verl 实际能力的冲突

本节记录理想架构与当前代码事实之间的差异。后续实现必须以这些事实为边界，不能把协议级设计直接当作已有 verl API。

### 11.1 Ray 的资源所有权与跨任务借卡冲突

**理想设想：** donor 将空闲 GPU 交给 GS，borrower 根据 node ID/GPU ID 创建 Ray replica；donor 的 replica 保留但处于休眠状态。

**当前事实：** verl 通过 `ResourcePool → PlacementGroup → RayWorkerGroup` 预留 GPU。replica 进入 sleep 只会释放模型显存或 KV cache，不会释放 PG 的 GPU 资源。Ray 仍认为这些 GPU 属于原任务，因此 borrower 如果以 `num_gpus > 0` 创建 actor，不会自动获得 donor 的 GPU。

**解决方案：** borrowed replica 不申请新的 Ray GPU 资源，采用“已有 Ray 资源锚点 + 插件启动 GPU 进程”的方式：

1. donor 保留 native worker/PG；
2. donor server 执行真正的权重/KV cache sleep；
3. borrower 在目标 node 上创建不声明 GPU 的 launcher/server/receiver actor；
4. launcher 根据 GS 租约设置物理 GPU 映射和 local rank；
5. GS 保证 donor 与 borrower 在同一时刻不会使用同一 slot。

该方案绕过 Ray 的二次 GPU 预留，但必须增加进程级 GPU 隔离、退出清理和可见性校验。NodeAffinity 只能解决节点选择，不能单独证明 GPU 选择正确。

### 11.2 GPU ID 与 Ray accelerator ID 不一致

**理想设想：** GS 给出 `gpu_ids=[0, 1]`，replica 直接使用这些 ID。

**当前事实：** verl 原生通过 worker actor 的 `get_accelerator_ids()` 获取 Ray 实际分配的 accelerator ID，再拼成 `CUDA_VISIBLE_DEVICES`。用户配置的物理 GPU ID、Ray accelerator ID 和进程内 local rank 不是同一个抽象。

**解决方案：** Placement 只能作为 GS 的物理租约；创建时增加 `ReplicaLaunchSpec`，记录：

```text
node_id
physical_gpu_ids
ray_worker_ids 或 anchor 信息
local_rank 映射
TP/DP/PP world size
master 地址和端口
```

Factory 必须在启动前读取节点实际 GPU 清单，核对租约，再生成进程环境。任何映射不一致都返回失败，不允许仅凭字符串设置 `CUDA_VISIBLE_DEVICES` 后报告成功。

### 11.3 standalone 的 sleep/wake 与理想流程冲突

**理想设想：** native STANDALONE replica sleep 后释放权重和 KV cache，wake 后恢复并继续服务。

**当前事实：** 当前 vLLM server 在 `STANDALONE` 模式下的 `sleep()` 和 `wake_up()` 是跳过操作；现有 `release_kv_cache()` 主要服务参数同步，通常会重新唤醒权重，并不等价于完整资源休眠。

**解决方案：** 插件 server 必须显式实现 standalone 生命周期：

- `sleep(level=2)`：释放权重和 KV cache；
- `wake_up(tags=["weights", "kv_cache"])`：恢复权重和 KV cache；
- 检查 `free_cache_engine`、MTP、LoRA、NPU 等配置是否支持对应 sleep level；
- 对不支持完整休眠的后端返回 `UNSUPPORTED`，不能伪装成资源已释放。

此能力依赖具体 vLLM 版本，必须由 `VLLMSleepAdapter` 隔离，不能假设所有 rollout backend 都有相同语义。

### 11.4 “借用 donor CE Worker”与“borrower 自有同步接收端”冲突

**理想架构的不同章节存在两种假设：** 一处要求 borrowed replica 拥有自己的同步接收端，另一处又提出不创建新的 `CheckpointEngineWorker`、临时关联 donor Worker。

**当前事实：** CE Worker 保存自己的 checkpoint engine、backend 状态、rank、world size、ServerAdapter 和通信连接。一个 Worker 不能安全地同时属于 donor 和 borrower 的两个通信组；即使两边使用相同数值的 rank/world size，也不改变 Worker 对具体 server 端点和进程状态的单一绑定。

**解决方案：** 统一采用 borrower 自有 CE Worker：

1. borrowed replica 创建自己的 receiver actor；
2. receiver 只绑定 borrower 的 ServerAdapter/endpoint；
3. donor CE Worker 永不跨任务转移；
4. GS 只传递 slot lease 和 operation receipt，不传递 CE Worker handle。

这样增加了 Worker factory，但所有权、故障隔离和任务退出语义清晰。

### 11.5 CE Worker 能否在 donor 与 borrower 之间共用：核心结论

本节针对“donor 和 borrower 复用同一个物理 GPU slot，因此 rank/world_size/group 理论上相同，是否可以共用一个 CE Worker”进行代码级判断。结论适用于当前 verl 的 `CheckpointEngineWorker`、`CheckpointEngineManager` 和 vLLM `ServerAdapter`，不是对一般通信框架的抽象猜测。

#### 11.5.1 先区分四个不同概念

| 概念 | 共享 GPU slot 是否意味着相同 | 对共用 CE Worker 的含义 |
|---|---|---|
| 物理 GPU | 是，同一租约期间指向同一张卡 | 只说明两个进程可能落在同一设备上，不说明进程状态可共享 |
| 数值 `rank` | 可能相同 | 仅是某个通信组内的编号；不同通信组可以有相同数值 |
| 数值 `world_size` | 可能相同 | 仅是某个通信组的成员数；不能标识成员身份或端点 |
| process group | 若真相同，则必须共享同一通信成员集合和生命周期 | 同一 group 中同一 rank 只能对应一个通信参与者；两个独立 replica 不能各自占用同一个 rank |

因此，“同一 GPU slot”与“同一个 CE Worker”之间没有逻辑蕴含关系。实际可行的设计通常是：两个独立进程在同一物理 GPU 上分别使用相同的数值 rank/world_size，但各自使用不同的 group/rendezvous 标识；如果强行使用同一个 group，则它们不再是两个独立的 CE 成员。

#### 11.5.2 当前实现中 CE Worker 的实际绑定关系

`verl/checkpoint_engine/base.py` 中的 `CheckpointEngineWorker` 在构造时只创建一套 `checkpoint_engine` 和一套 `server_adapter`。其 `update_weights()` 逻辑是：从该 Worker 的 engine 取出一条权重流，然后调用这一个 adapter 的 `update_weights()`。原生接口没有 `replica_id`、`task_id` 或“本次把同一权重发送给多个 borrower server”的参数。

对 vLLM 后端，`ServerAdapter` 还在 Worker 进程内固定了以下状态：

- 根据进程环境和 `replica_rank` 计算 `rollout_rank`、`node_rank`；
- 根据 Ray job ID、replica rank 和 local rank 生成唯一的 ZMQ/IPC handle；
- 通过 server 名称解析并缓存一个 `server_handle`；
- 权重更新、权重版本和 KV cache 操作都发给这个 server 端点。

所以 CE Worker 与 replica 的关系是“运行时强耦合、管理对象上可抽象分离”：Manager 可以用 descriptor 管理 replica，但一个原生 Worker 实例不能在同一时刻代表两个不同的 server/engine。把 `ActorHandle` 放入另一个任务的列表，不会复制这些进程内状态。

#### 11.5.3 `rank/world_size/group` 相同为什么仍然不能共用

有三个独立的阻断点：

1. **同一个 group 不允许两个独立端点拥有同一个 rank。** `CheckpointEngineManager.build_process_group()` 会按 worker 列表位置为每个参与者生成 rank/world size 并执行 collective 初始化。若 donor Worker handle 被重复放入列表，列表长度会被当作 rollout world size；同一个 actor 可能被并发赋予两个列表位置，而不是得到“一个 rank 的两个别名”，结果是 rank 冲突、collective 次序不一致或死锁。
2. **不同 group 才能让两个进程各自使用相同数值 rank。** 这时它们已经是两个独立通信组，donor 的 group 不能被 borrower 复用；borrower 必须有自己的 Worker 或自己的进程内 CE endpoint。把 group 名称、store、master 地址和连接 epoch 也做成相同，反而会让两个生命周期互相破坏。
3. **通信组相同也不能解决 ServerAdapter 的单端点问题。** donor Worker 的 adapter 仍指向 donor 的 server、IPC 路径和 Ray job 上下文。borrower 创建了新的 server 后，共用 Worker 不会自动把权重流切换或复制到新 server；在更新过程中强行修改 adapter 还会与 in-flight update、KV cache 和 server 生命周期竞争。

#### 11.5.4 四种可能的“共享”分别意味着什么

| 方案 | 是否能满足两个独立 replica 同时工作 | 判断 |
|---|---:|---|
| borrower 新建 server/engine，但复用 donor CE Worker | 否 | 一个 Worker 只有一个 ServerAdapter/端点；原生 update path 无法定向 borrower |
| donor 停止后，把同一个 Worker 的 adapter/group 改绑给 borrower | 只能串行转移，不能同时共享 | 需要销毁并重建通信组、重绑 adapter、刷新 rank/端点和所有权；这是 Worker 所有权转移，不是 replica 共用，且 donor 不能继续使用 |
| 一个 Worker/engine 后面挂一个“多租户 fan-out adapter”，同时服务两个 server | 理论上可做 | 这会把两个 replica 合并成一个共享服务端点，需要重新设计权重广播、版本、LB、配额、故障和回收语义；不再是当前架构中的 donor/borrower 独立 replica，原生 CE API 不能直接复用 |
| borrower 创建自己的 CE Worker、ServerAdapter、server/engine 和 group，仅复用物理 slot | 可以 | 符合资源租约模型；通过新的 WorkerGroup 和新的 topology epoch 完成 bootstrap/同步，推荐采用 |

这里的“不能”是针对当前目标“borrowed replica 内容独立创建、可单独接流、可单独休眠/唤醒/回收”。如果产品目标改成“多个任务共享一个推理服务”，应另立共享服务架构，不能把它描述为 borrowed replica 复用 donor CE Worker。

#### 11.5.5 对 `CheckpointEngineManager` worker 列表的直接影响

原生同步流程会遍历：

```python
for replica in self.replicas:
    workers.extend(replica.workers)
```

这里的 `workers` 是要被临时包装成 rollout `RayWorkerGroup` 的 CE Worker actor handles。把 donor 的 handle 再加入 borrower replica 会产生两个问题：

- 若列表中保留重复 handle，`world_size`、rank 分配和 backend topology 会把同一 actor 当作两个参与者；Ray 调用也不能让一个 actor 进程同时安全执行两个 rank 的 collective；
- 若为了避免重复而只保留一次 handle，CE 视图中仍只有一个 replica 成员，无法表达两个 server 的独立 bootstrap、版本、LB 状态和回收状态。

因此，`replica.workers` 的列表拼接只能接收 borrower 自己创建的 Worker handles。它不是把一个已有 Worker “挂载”到另一个 replica 的通用扩缩接口。

#### 11.5.6 最终设计决策

1. **禁止**在 borrowed replica 中复用 donor 的 `CheckpointEngineWorker`、`ServerAdapter`、checkpoint engine 实例、ResourcePool/Placement Group 或 process group。
2. **必须**由 `ReplicaFactory` 为 borrowed replica 创建独立的 CE Worker 集合；Worker 数量按该 replica 的 TP/DP/PP 拓扑确定，并创建独立 server/engine、IPC 端点和通信组。
3. **允许共享的只有** GS 授权的物理 GPU slot，以及只读模型文件、配置模板等不携带运行时状态的内容。两个独立进程可在同一物理 GPU 上拥有数值相同但 group 标识不同的 rank/world size；这需要 Supervisor 和 lease ledger 保证不发生并发显存超卖。
4. **如果 backend 不支持独立 topology epoch 或运行期建组/销毁**，该 backend 暂不支持 borrowed replica；不能用“共用 donor Worker”绕过限制。
5. 文档中原“临时关联已有 `MultiTaskCheckpointEngineWorker`”的表述废弃，统一改为：

   > `MultiTaskLLMServerManager` 为 borrowed replica 创建独立的 `MultiTaskCheckpointEngineWorker`、`ServerAdapter`、vLLM server/engine 和通信拓扑。borrowed replica 只复用 GS 授权的物理 GPU slot，不复用 donor 的 Worker、ResourcePool、Placement Group、ServerAdapter 或 checkpoint process group。

这一定义使“资源共享”落在物理 slot 租约层，而不是把任务运行时对象跨任务借用；也是在不侵入 verl 原代码的前提下保持故障隔离和生命周期可证明的最小方案。

### 11.6 静态 CE process group 与动态 effective replica 冲突

**理想设想：** 通过增删 replica 列表并重建通信组即可动态加入或移除借用副本。

**当前事实：** 原生 `CheckpointEngineManager.add_replicas()`/`remove_replicas()` 只修改列表。真正同步时，verl 会收集所有 worker 执行 `prepare()`，根据 backend 计算 rank/world size，再初始化 process group。NCCL/HCCL/NIXL 的连接、rank 和 world size 可能是固定的。

**解决方案：** 增加 backend-specific `CheckpointTopologyAdapter`，每次拓扑变化都执行：

```text
停止新请求
→ 完成或中断旧同步
→ prepare 新成员
→ 销毁旧 process group/连接
→ 生成新 rank/world size
→ 初始化新 process group
→ 验证 topology_epoch
→ 允许 LB 接流
```

通信组名称必须包含 `task_id + topology_epoch + backend`，避免不同任务或 epoch 共用默认 group name。若某个 backend 不支持运行期重建，则该 backend 只能暂不支持 borrowed replica。

### 11.7 LB 立即 remove 与 in-flight 请求清理冲突

**理想设想：** 从 LB 移除 replica 后 abort 请求，等待请求续推。

**当前事实：** 原生 router 的 `remove_servers()` 会直接删除 server 和对应 in-flight 计数。如果请求尚未完成，计数和 sticky 映射会丢失，无法可靠判断 server 是否已经排空。

**解决方案：** 插件 LB 增加显式状态：

```text
READY → DRAINING → ABORTING → DRAINED → REMOVED
```

`DRAINING` 禁止新请求，但保留旧请求计数；`ABORTING` 调用 server 的 abort；只有 `inflight == 0` 且 partial-rollout 回执已生成后，才执行原生 `remove_servers()`。这样摘流、回收和样本续推的责任不会混在一个 remove 操作中。

### 11.8 理想的统一 replica-sync gate 与当前训练流程冲突

**理想设想：** bootstrap、CE 成员变化、LB 提交和原生参数同步共用一把 gate。

**当前事实：** Fully Async Trainer 的参数同步是训练流程内部方法，没有公开的动态 replica 事务接口；Trainer 和 Rollouter 又是不同 Ray Actor，简单增加一个本地 asyncio lock 无法形成跨 Actor 的互斥。

**解决方案：** 在 TaskRunner 内增加任务级事务协调器，由它串行化操作，并通过单向调用避免锁循环：

```text
TaskRunner acquire operation gate
→ 调 Rollouter prepare/drain
→ 调 Trainer bootstrap/topology update
→ 调 Rollouter LB commit
→ TaskRunner release gate
```

原生 `_fit_update_weights()` 进入同一 gate 前必须先完成当前同步；任何跨 Actor 调用不得在持有 Trainer 内部锁时反向等待 Rollouter。

### 11.9 destroy 与 PG 清理冲突

**理想设想：** destroy borrowed replica 即可释放其 GPU。

**当前事实：** `RolloutReplica` 没有统一 destroy API；Ray actor、worker actor 和 PG 的生命周期可能不同。杀掉 actor 不代表 PG 已释放，也不代表 vLLM 子进程、端口和通信组已清理。

**解决方案：** 将销毁限定为 borrower-owned、无独立 PG 所有权的 actor/process：

```text
LB DRAINED
→ CE remove + topology rebuild
→ 关闭 server/receiver/backend
→ kill actor 或回收子进程
→ 清理端口、临时文件和 CUDA IPC
→ 校验进程退出
→ 回报 DESTROYED
```

native replica 不走 destroy，只走 sleep/wake。若 borrowed replica 使用了独立 PG，则必须先实现 PG 的显式释放，否则只能标记为 QUARANTINED，不能向 GS 报告 slot 可用。

## 12. 修订后的可行性判断

在不修改 verl 原文件的前提下，以下能力可以复用现有实现：

- replica 的 `abort_all_requests()`、`resume_generation()`；
- vLLM 的部分 KV cache/权重 sleep/wake 原语；
- router 的基础 add/remove server；
- CE Manager 的 worker prepare、update 和 finalize 流程。

以下能力不能仅靠少量子类完成：

- 物理 GPU slot 的安全借用；
- standalone 完整权重休眠与唤醒；
- borrowed CE Worker 创建与 endpoint 管理；
- NCCL/HCCL/NIXL 动态拓扑重建；
- 带 in-flight 保证的 LB draining；
- actor、子进程、PG 和通信组的完整销毁。

因此推荐的插件化最小实现顺序是：

1. 先实现单节点、固定 TP、vLLM STANDALONE 的 borrowed process factory；
2. 再实现 standalone sleep/wake adapter；
3. 再实现 borrower 自有 CE Worker 和单一 backend 的拓扑重建；
4. 最后接入 LB draining、partial rollout 和跨任务回收。

只有完成第 4 步并通过真实 GPU 验收后，才能宣称实现了架构文档中完整的 replica 资源共享能力。

## 13. GPU slot 复用的进程模型与启动开销

### 13.1 不是从 donor Actor 直接 fork

“以 donor Actor 为资源锚点”不等于“在 donor Actor 内 `fork()` borrower”。donor Actor、native worker 或 vLLM server 通常已经初始化 CUDA、PyTorch allocator、NCCL、ZMQ 和后台线程；在此状态下 fork 会继承不安全的 CUDA/runtime 状态，可能出现死锁、NCCL hang、失效文件描述符或 CUDA 初始化错误。vLLM 的 multiprocessing 文档也指出，fork 与使用线程的依赖存在兼容性问题，CLI 路径通常选择 spawn。

正确的进程关系是：

```text
donor native Actor/PG：继续由 Ray 持有资源锚点
NodeSlotSupervisor：独立 CPU actor/节点进程，不初始化 CUDA
borrower process：由 Supervisor 使用 spawn/subprocess 启动
```

borrower 进程只复用 donor 暂时释放出的物理 GPU slot，不继承 donor 的 Python 对象、CUDA context、NCCL group、ServerAdapter 或 CE Worker。它需要独立完成 CUDA context、vLLM engine、推理通信和 CE receiver 的初始化。

### 13.2 启动开销确实较大

每次首次创建 borrowed replica 可能包含以下开销：

```text
进程启动与 Python import
→ CUDA context 初始化
→ vLLM engine 初始化和模型权重加载
→ 多 GPU/NCCL 或其他 backend 建组
→ HTTP server 和端口建立
→ CE receiver prepare/init
→ bootstrap 当前 serving version
→ LB/CE 提交
```

其中模型加载、vLLM engine 初始化和多 GPU 通信组建立通常远大于普通 Ray RPC。因此，如果每个短暂空泡都销毁并重新创建进程，启动成本可能抵消资源共享收益，甚至降低吞吐。

### 13.3 降低开销的生命周期策略

第一版不应采用“每次借用都创建、每次归还都销毁”的策略，而应区分冷启动和热切换：

1. **进程复用：** borrowed process 首次创建后由 Supervisor 保留；归还时执行 abort、摘流、CE 移除和权重/KV cache sleep，保留进程、端口和本地控制通道。
2. **唤醒复用：** 下一次同一 borrower 获得兼容 lease 时，优先 wake 已存在进程；只有 node、GPU 数量、并行拓扑或模型配置不兼容时才销毁并重新创建。
3. **预热池：** 在调度收益足以覆盖启动成本时，Supervisor 可预先启动少量处于 SLEEPING 状态的进程；预热进程不得加入 LB 或 CE effective set，也不得向 GS 报告为已占用的 active slot。
4. **批量操作：** 将同一 donor 的多个 GPU slot 作为完整 replica 批量创建、批量 bootstrap 和批量回收，避免逐卡重复建立通信组。
5. **调度阈值：** GS 只有在预计借用时长超过 `startup_cost + bootstrap_cost + teardown_margin` 时才下发 ADD；短于阈值的空泡只上报，不触发创建。
6. **拓扑缓存：** 对相同 task、并行度和 backend 复用端口/拓扑模板，但每次实际加入 CE 时仍必须创建新的 topology epoch 并验证成员。

### 13.4 回收时优先 sleep，销毁是例外

borrowed process 的推荐状态为：

```text
STOPPED
  └── spawn + initialize + bootstrap → READY
READY
  └── drain + abort + CE/LB remove + sleep → SLEEPING
SLEEPING
  └── wake + bootstrap/校验 + CE/LB commit → READY
SLEEPING
  └── lease 过期、配置不兼容或故障 → destroy → STOPPED
```

这样，模型进程只在第一次使用或配置发生变化时承担冷启动成本。sleep 仍必须由后端确认权重/KV cache 已释放；进程存活本身不代表 GPU slot 可以安全交给 donor。

### 13.5 启动成本必须进入调度回执和观测

Supervisor 和任务侧需要记录：

```text
process_start_ms
cuda_init_ms
engine_load_ms
topology_init_ms
bootstrap_ms
lb_commit_ms
sleep_ms
wake_ms
destroy_ms
```

GS 的调度策略应使用这些实际数据判断是否值得借用。第一版可以使用静态阈值，但不能假设所有模型、GPU、backend 和拓扑的启动时间相同。

### 13.6 方案取舍

该进程模型牺牲了“直接复用 donor Python 进程状态”的便利，换取 CUDA、vLLM、NCCL 和任务所有权隔离。它仍然绕过 Ray 对 GPU slot 的二次分配，因此必须由 GS 维护物理 slot lease，并由 Supervisor 负责进程退出、孤儿检测和 GPU 可见性校验。

如果运行环境要求 Ray 完整管理 GPU 隔离，或者不允许任务绕过 Ray 启动 GPU 进程，则该资源共享方案不可用，应改为 Ray 原生 fractional GPU 调度或固定资源切分，而不能继续使用本节的 slot 复用模型。

## 14. CE 与 LB 动态扩缩接口复用方案

### 14.1 原生接口盘点

当前 verl 已经提供部分动态扩缩接口，但这些接口只覆盖“成员登记”和“路由登记”，不构成完整的动态 replica 事务。

| 组件 | 原生接口 | 能否直接复用 | 直接复用的边界 |
|---|---|---|---|
| `CheckpointEngineManager` | `add_replicas(replicas)` | 可以复用列表登记 | 只执行 `self.replicas.extend()`，不做 bootstrap、加锁或拓扑重建 |
| `CheckpointEngineManager` | `remove_replicas(replicas)` | 可以复用列表移除 | 只从列表删除，不清理连接，不等待同步，不验证 replica 状态 |
| `CheckpointEngineManager` | `build_process_group(rollout)` | 可以复用底层建组流程 | 调用方必须提供完整 `RayWorkerGroup`，并确保 backend 支持新的 rank/world size |
| `CheckpointEngineManager` | `update_weights()` | 可以复用同步主流程 | 它会按当前 `self.replicas` 重建临时 worker group；不理解 lease、topology epoch 或 partial ADD |
| `CheckpointEngineManager` | `sleep_replicas()` | 部分复用 | 只调用 replica 的原生 `sleep()`；STANDALONE vLLM 当前可能是空操作 |
| `CheckpointEngineManager` | `wake_up_replicas()` | 部分复用 | 只调用 replica 的原生 `wake_up()`；不负责 bootstrap、CE 加入或 LB 提交 |
| `GlobalRequestLoadBalancer` | `add_servers(servers)` | 可以复用 | 原子加入 server 和初始化计数，但不校验 server READY/权重版本 |
| `GlobalRequestLoadBalancer` | `remove_servers(server_ids)` | 只能在排空后复用 | 会立即删除 server 和 in-flight 计数，不提供 DRAINING 状态 |
| `GlobalRequestLoadBalancer` | `get_inflight_count()` | 可以复用 | 可用于排空检查，但不能阻止新的 acquire，必须配合插件状态门控 |
| `GlobalRequestLoadBalancer` | `clear_sticky_cache()` | 可以复用 | 用于重新分布请求，不等于摘流或回收 |

### 14.2 推荐的复用原则

不修改 verl 原实现，插件只在原生接口外包一层状态机和事务协调：

```text
插件控制器：校验 operation/lease、获取任务级 gate、维护状态
原生 CE：执行 worker prepare、process group、update_weights、finalize
原生 LB：执行 READY server 的 add/remove 和请求计数
插件适配器：补齐 DRAINING、bootstrap、拓扑 epoch 和回滚
```

原生方法的调用顺序不能被直接暴露给 GS。GS 只能调用 TaskRunner 的高层命令，例如 `prepare_add`、`commit_add`、`reclaim`；TaskRunner 再把命令拆成 CE、LB 和 Manager 操作。

### 14.3 ADD 的最小插件编排

```text
1. 创建 borrower server/CE worker，状态 PREPARED
2. 获取任务级 replica-sync gate
3. 用原生 CE worker prepare + build_process_group 建立新拓扑
4. 用原生 CE update_weights 对新 replica 做 bootstrap
5. 将新 replica 加入 effective replica projection
6. 将 server 以 READY 状态交给插件 LB
7. 插件 LB 内部调用原生 add_servers()
8. 更新 Manager active 列表和拓扑 epoch
9. 释放 gate，返回 ACTIVE receipt
```

这里不能先调用原生 `add_servers()` 再做 CE bootstrap，否则新 server 可能在权重未完成时收到请求。原生 `add_replicas()` 只能作为 CE 成员登记的内部步骤，不能作为 ADD 完成标志。

### 14.4 RECLAIM 的最小插件编排

```text
1. 插件 LB 将 server 标记 DRAINING，禁止新的 acquire
2. 调用原生 server abort 或等待 drain
3. 使用 get_inflight_count() 等待计数归零
4. 调用原生 remove_servers()
5. 获取 replica-sync gate
6. 调用原生 CE remove_replicas()
7. 销毁旧通信组并按 backend adapter 重建拓扑
8. 对 borrowed server 执行 sleep 或 destroy
9. 从 Manager borrowed projection 删除
10. 释放 gate，返回 RELEASED receipt
```

如果必须强制回收仍有请求的 server，步骤 2 必须先生成 partial-rollout 续推回执；不能直接调用原生 `remove_servers()`。如果 CE 拓扑重建失败，server 保持不可路由，任务不得向 GS 报告资源已释放。

### 14.5 native replica 的归还

native replica 不执行原生 `remove_replicas()`，因为它仍属于 donor 的基线集合。donor 只执行：

```text
LB DRAINING
→ abort/排空
→ 原生 LB remove_servers()
→ native server 真正 sleep
→ 向 GS 报告 slot 可借用
```

归还时：

```text
GS 确认 borrowed slot 已释放
→ native server wake
→ 使用当前 serving version bootstrap/校验
→ 原生 CE add/rebuild 或恢复固定拓扑
→ 原生 LB add_servers()
→ native replica 恢复 READY
```

### 14.6 为什么不能直接把原生接口暴露给 GS

原生接口缺少以下跨组件约束：

- CE membership 与 LB route 的原子提交；
- in-flight 请求排空和 partial rollout 续推；
- borrowed/native 所有权区分；
- lease epoch 和重复命令校验；
- backend-specific process group 销毁和重建；
- bootstrap 到当前 serving version；
- 失败回滚和资源隔离。

因此本方案的结论是：**CE 和 LB 的基础操作可以直接复用，动态扩缩的完整语义不能直接复用。** 插件应复用原生底层动作，在其上增加生命周期状态、事务 gate、拓扑适配器和回执协议。这样既不侵入 verl 原代码，也避免把几个列表 API 误认为已经具备跨任务资源共享能力。

## 15. native 与 borrowed replica 的兼容性边界

### 15.1 原始架构的实际含义

原始架构把 replica 分为 native 和 borrowed 两类，并在 Manager、CE effective set、LB 路由和生命周期流程中把它们都称为 replica。这表达了一个正确的目标：**上层应当能够把两类实例视为同一种可服务副本进行编排**。

但原始架构并没有做到“只在名称上有区别、类方法和内部逻辑完全相同”。文档本身已经规定了不同的创建和所有权语义：

- native replica 由任务启动流程通过自己的 ResourcePool/Placement Group 创建，保留原有 worker 和 PG；
- borrowed replica 根据 GS 给出的 node/GPU 租约创建，不能调用会重新申请 PG/GPU 的 `init_standalone()`，也不能复用 donor 的 Worker；
- native 资源归还通常是 sleep/wake，borrowed 还需要 lease 校验、强制回收和 destroy；
- HYBRID 与 STANDALONE 的参数同步接收路径不同，borrowed 还可能使用独立 CE receiver 或 external process。

所以，理想架构目前表达的是“同一抽象角色 + 不同资源提供者”，不是两个完全相同的具体类。若强行让两者内部逻辑完全一致，要么重新申请 Ray 资源，破坏借卡语义；要么把 donor 的运行时对象跨任务复用，破坏前面已经确认的 Worker、通信组和生命周期隔离。

### 15.2 必须统一的部分：协议和可观察行为

为保证兼容，native 与 borrowed 应实现同一个 `Replica` 能力协议。上层 Manager、LB 和训练流程只能依赖这些公共操作和状态，不得通过 `kind` 分支访问具体 Worker/Server 字段：

```text
describe() → placement、replica_id、kind、serving_version、capabilities
prepare()  → 创建或恢复运行时对象，但尚不可路由
activate() → 完成 CE/LB 提交并进入 READY
drain()    → 禁止新请求，保留 in-flight 事实
sleep()    → 释放后端允许释放的权重/KV/cache 资源
wake()     → 恢复运行时并追平 serving version
remove()   → 从 CE/LB 有效集合移除
destroy()  → 结束本副本拥有的进程和连接
```

两类 replica 必须共享以下可观察语义：

1. 相同的状态机和状态转移校验，例如 `PREPARED → ACTIVE → DRAINING → SLEEPING`；
2. 相同的 operation ID、lease epoch、幂等回执和错误码规则；
3. 相同的 READY 条件：权重版本、通信拓扑、server 健康状态和 LB 路由全部完成后才可接流；
4. 相同的 drain、partial-rollout、CE 移除和资源释放顺序；
5. 相同的观测字段，使 GS 不需要知道实例是通过 PG 还是 slot supervisor 创建的。

这才是“兼容”的必要条件。公共接口应当表达能力和后置条件，而不是暴露 native 的 PG handle 或 borrowed 的进程句柄。

### 15.3 可以不同的部分：策略、资源绑定和实现细节

公共协议下面应使用策略/适配器隔离实现差异：

| 层次 | native replica | borrowed replica | 是否需要一致 |
|---|---|---|---|
| 资源绑定 | 任务自有 PG/worker slot | GS lease + Supervisor 绑定物理 GPU | 语义一致，句柄不同 |
| 创建 | 原生 worker group 或既有 server | 新进程、新 CE Worker、独立 server/engine | 结果一致，步骤不同 |
| CE endpoint | donor 自有 Worker 集合 | borrower 自有 Worker 集合 | endpoint 协议一致 |
| 休眠 | 保留 PG，释放后端显存 | 保留或销毁 borrower 进程，视 lease 决定 | 状态和回执一致 |
| 销毁 | 通常禁止作为捐赠归还动作 | lease 结束时允许 | 能力由策略声明 |
| 参数同步 | 原生 topology | 新 topology epoch/独立 group | 版本和完成条件一致 |

因此建议的对象结构是：

```text
Replica (公共协议、状态机、回执)
├── NativeReplica     + NativeResourceBinder + NativeLifecycleAdapter
└── BorrowedReplica   + LeaseResourceBinder  + BorrowedLifecycleAdapter
```

`NativeReplica` 和 `BorrowedReplica` 可以是不同具体类，但 Manager 只持有 `Replica` 协议；`ResourceBinder` 决定如何获得 slot，`LifecycleAdapter` 决定如何调用后端，`CheckpointEndpoint` 决定如何接入 CE。这样既保持类方法兼容，也不制造虚假的内部同构。

### 15.4 对原始理想表述的修订

原始架构中“native、borrowed replica 的销毁、休眠和唤醒能力”应理解为公共能力模型，而不是要求所有实例都必须执行完全相同的底层动作。更准确的表述是：

> native 与 borrowed replica 对外提供一致的 Replica 生命周期协议和可观察行为；二者在资源绑定、运行时对象创建、CE endpoint、所有权策略和可用操作上允许不同，由适配器隐藏差异。任何不支持某项操作的类型必须返回明确的 `UNSUPPORTED` 或 `POLICY_DENIED`，不能静默改变语义。

同理，`effective_replicas = native + borrowed` 表示两类实例都满足同一接流和参数版本契约，不表示它们共享相同的 Ray actor、PG、CE Worker 或 process group。

### 15.5 最终判断

回答“理想架构是否已经做到两种 replica 只有名称不同”：**没有，当前文档的理想设想只在抽象生命周期和上层编排层面要求统一，在资源创建、所有权、CE 接收端和销毁策略上明确存在差异；其中‘borrowed 不创建独立 CE Worker、临时使用 donor Worker’还是与该统一目标和当前 verl 实现冲突的错误设想，已在本设计中废弃。**

实现时应坚持“同一协议、不同适配器、独立运行时对象”的原则。这样 native replica 可以继续走 verl 原生路径，borrowed replica 可以通过插件路径创建，而 Manager、LB 和训练流程仍能以同一套生命周期逻辑处理二者。

### 15.6 若要求具体类也统一：采用策略注入，而不是复制两套实现

如果项目把“类方法和逻辑完全一致”作为硬性工程约束，可以不定义两个包含重复逻辑的 `NativeReplica`/`BorrowedReplica` 实现，而采用一个具体的 `MultiTaskReplica`：

```text
MultiTaskReplica
├── ResourceBinder
│   ├── NativeResourceBinder   # 使用已有 PG/worker handles
│   └── LeaseResourceBinder    # 使用 GS lease + Supervisor 启动独立进程
├── RuntimeFactory
│   ├── NativeRuntimeFactory   # 连接已有 server/CE endpoint
│   └── BorrowedRuntimeFactory # 创建独立 server/CE Worker/engine
└── LifecyclePolicy
    ├── NativePolicy           # donor 归还走 sleep，destroy 受策略限制
    └── BorrowedPolicy         # 支持 lease reclaim/destroy
```

`MultiTaskReplica.prepare/activate/drain/sleep/wake/remove/destroy` 的事务、状态机、回执和异常处理全部只有一份；`kind` 只用于选择注入的 binder/factory/policy 和声明 capability，不能让调用方绕过公共协议。这样可以满足“上层类方法完全一致”，同时保留 native 与 borrowed 必须存在的资源实现差异。

需要强调的是，统一具体类不等于共享运行时对象：`NativeRuntimeFactory` 和 `BorrowedRuntimeFactory` 仍必须返回各自独立的 server、CE Worker、ServerAdapter、IPC 端点和 process group。若把统一类误解为共用 donor 的这些对象，仍会触发 11.5 节所述的 rank、端点和所有权冲突。

### 15.7 是否直接继承 native 具体类

继承可以作为代码复用手段，但不能作为兼容性设计本身。当前 verl 的继承层次是 `vLLMReplica → RolloutReplica`；其中以下 native 实现带有明确的资源假设：

- `RolloutReplica.init_standalone()` 会创建新的 ResourcePool、Placement Group 和带 GPU 资源的 `RayWorkerGroup`，borrowed replica 必须覆盖该方法；
- `vLLMReplica.launch_servers()` 假设 `self.workers` 是 Ray Actor handles，会调用 `__ray_call__` 获取 Ray accelerator ID，并据此创建 Ray `vLLMHttpServer`，外部 Supervisor/子进程模式不能直接使用；
- `sleep()`、`abort_all_requests()`、`release_kv_cache()` 等方法假设 `self.servers` 中的对象支持 `.remote()`，borrowed 的进程控制通道若不是 Ray actor，也必须覆盖或提供兼容 facade；
- CE Manager 要求 `replica.workers` 是 borrower 自己创建的 CE Worker handles，继承 native 类不会自动产生这些 handles。

因此有三种选择：

1. **推荐：** 两类都继承 `RolloutReplica`（或插件定义的 `Replica` 协议），共享状态机/mixin，分别实现资源绑定和运行时工厂。这样依赖最少，避免把 `init_standalone()` 等 native 假设带入 borrowed。
2. **可接受：** `BorrowedVLLMReplica(vLLMReplica)`，仅在 vLLM 后端且能提供与 native 相同的 server/worker facade 时使用；必须覆盖 `init_*`、`launch_servers`、所有 server 控制方法、destroy/lease 校验，并为 borrowed 创建独立 CE Worker。父类方法只能在逐项审查后复用。
3. **不推荐：** 直接继承 native 类而不覆盖资源创建和 server 控制方法。这会重新申请 PG/GPU，或把外部进程误当 Ray actor，造成资源越权和运行时错误。

所以，“borrowed replica 必须实现 native 对外暴露的接口”可以通过继承来帮助满足类型兼容，但不能推出“borrowed 可以直接复用 native 的初始化、Worker、ServerAdapter 或通信组”。继承关系应服务于公共协议，资源和运行时实现仍必须保持独立。

### 15.8 `BorrowedReplica` 详细类设计

本节给出插件侧 `BorrowedReplica` 的字段、方法和与 native replica 的兼容边界。设计目标是：上层 Manager、CE Manager、LB 和 TaskRunner 只依赖公共 Replica 协议；borrowed 的资源租约、进程启动和所有权细节封装在类内部。

#### 15.8.1 类结构和字段

推荐让 `BorrowedReplica` 实现插件定义的 `Replica` 协议，并在 vLLM 后端可选地继承 `RolloutReplica` 以复用配置解析和公共属性。下面的字段是逻辑字段，不要求全部暴露为公共属性：

```python
class BorrowedReplica(RolloutReplica):
    # ---------- identity and ownership ----------
    replica_id: str
    owner_task_id: str                 # borrower task
    donor_task_id: str | None
    kind: Literal["borrowed"]
    replica_rank: int
    lifecycle_epoch: int

    # ---------- immutable runtime configuration ----------
    config: RolloutConfig
    model_config: HFModelConfig
    backend: str                       # vllm / sglang / trtllm
    rollout_mode: RolloutMode
    tp_size: int
    dp_size: int
    pp_size: int
    world_size: int
    nnodes: int
    gpus_per_replica_node: int

    # ---------- physical resource lease ----------
    slot_lease: ReplicaPlacement       # node_id, physical_gpu_ids, lease_id, epoch
    lease_expire_at: float | None
    lease_state: Literal["ACTIVE", "REVOKING", "EXPIRED", "RELEASED"]
    resource_binder: LeaseResourceBinder
    supervisor: NodeSlotSupervisor
    cuda_visible_devices: tuple[str, ...]
    local_rank_map: dict[int, int]

    # ---------- independently owned runtime objects ----------
    workers: list[CEWorkerHandle]       # borrowed-owned CE Workers only
    servers: list[ServerEndpoint]       # borrowed-owned server/process endpoints
    _server_address: str | None    # 对外通过 server_address property 暴露
    _server_handle: ServerEndpoint | None
    process_handles: list[ProcessHandle]
    runtime_factory: BorrowedRuntimeFactory
    server_adapter: BorrowedServerAdapter

    # ---------- CE and serving state ----------
    checkpoint_endpoint: CheckpointEndpoint
    topology_epoch: int | None
    serving_version: int | None
    routing_epoch: int | None
    ce_membership: Literal["ABSENT", "PREPARED", "ACTIVE", "REMOVING"]
    runtime_state: Literal["STOPPED", "PREPARED", "READY", "DRAINING", "SLEEPING", "QUARANTINED", "DESTROYED"]
    capabilities: frozenset[str]

    # ---------- transaction and observability ----------
    operation_lock: asyncio.Lock
    inflight_snapshot: int
    completed_operations: dict[str, OperationReceipt]
    last_error: ReplicaError | None
    metrics: ReplicaLifecycleMetrics
```

字段约束如下：

- `slot_lease` 是唯一的 GPU 使用授权；类不能根据 `node_id/gpu_ids` 自行猜测或扩大资源范围；
- `workers` 只能保存本 borrowed replica 自己创建的 CE Worker handles，不能填入 donor handles；
- `servers` 可以是 Ray actor facade，也可以是 Supervisor 管理的外部进程 endpoint，但必须实现统一的 `ServerEndpoint` 协议；
- `topology_epoch`、`routing_epoch` 和 `serving_version` 分别表示 CE 成员版本、LB 路由版本和模型权重版本，不能用一个字段代替；
- `operation_lock` 只保护本 replica 内部状态，跨 Rollouter/Trainer 的互斥仍由 TaskRunner 的 replica-sync gate 负责；
- `capabilities` 必须显式声明 `full_sleep`、`partial_rollout`、`dynamic_topology`、`destroy` 等能力，调用方不能假定所有 backend 都支持。

#### 15.8.2 公共接口：native 与 borrowed 必须一致

下表中的接口应由 native 和 borrowed 都提供，Manager、CE Manager、LB 和 TaskRunner 只能调用这些接口。返回值应统一为 `OperationReceipt`、`HealthSnapshot` 或协议定义的 endpoint，不返回某一类专有的 PG/actor 内部对象。

| 接口 | 作用 | borrowed 的实现要求 |
|---|---|---|
| `describe()` | 返回身份、拓扑、版本、能力和资源摘要 | 包含 lease 摘要和 `kind=borrowed`，不暴露 donor Worker |
| `prepare()` | 创建或恢复运行时对象，进入 `PREPARED` | 校验 lease，启动独立 server/CE Worker/process group，但暂不接流 |
| `activate()` | 完成 CE bootstrap、拓扑提交和 LB 注册，进入 `READY` | 只在 serving version、CE、server health 全部满足后成功 |
| `drain()` | 禁止新请求并等待或中断 in-flight 请求 | 保留 in-flight 事实，不能直接丢弃计数 |
| `abort_all_requests()` | 中断本 replica 上的请求并返回请求标识 | 为 partial rollout 提供可续推的请求/输出数据 |
| `resume_generation()` | 在请求续推条件满足后恢复生成 | 只能在 server READY 且 LB/CE 视图一致后调用 |
| `remove_from_lb()` | 从路由有效集合摘除 | 内部先进入 DRAINING，排空后才调用原生 remove |
| `remove_from_ce()` | 从 CE effective set 移除 | 更新 topology epoch，清理本 replica 的 CE 连接 |
| `sleep()` | 释放后端允许释放的权重/KV/cache | 先检查 lease 和 drain 状态；不代表自动释放 PG |
| `wake()` | 恢复运行时并追平 serving version | 不重新申请 Ray PG；必要时由 Supervisor 恢复进程 |
| `release_kv_cache()` / `resume_kv_cache()` | 参数同步前后的 KV cache 操作 | 只在 backend capability 声明支持时执行 |
| `health()` | 返回进程、server、CE、版本和显存状态 | 必须能检测孤儿进程、错误 GPU 映射和端口失效 |
| `destroy()` | 终止本 replica 拥有的 server、CE Worker、进程和连接 | 校验已 DRAINED 且 lease 不再使用；native 可按策略返回 `POLICY_DENIED` |

#### 15.8.3 borrowed 专有接口

以下接口不要求 native 暴露为调度入口，但属于 `BorrowedReplica` 的内部能力或 TaskRunner 事务接口：

```python
async def create_from_lease(
    self, placement: ReplicaPlacement, operation: ReplicaOperation
) -> PreparedReplica

async def validate_lease(self, expected_epoch: int) -> None
async def build_launch_spec(self) -> ReplicaLaunchSpec
async def start_runtime(self) -> RuntimeReceipt
async def bootstrap_weights(self, target_version: int) -> BootstrapReceipt
async def prepare_ce_membership(self, topology_epoch: int) -> CEMembershipReceipt
async def commit_routable(self, routing_epoch: int) -> OperationReceipt
async def begin_reclaim(self, operation: ReplicaOperation) -> ReclaimReceipt
async def cleanup_runtime(self, *, destroy_process: bool) -> CleanupReceipt
async def release_slot(self) -> SlotReleaseReceipt
```

这些接口的调用顺序固定为：

```text
validate_lease
→ build_launch_spec
→ start_runtime
→ prepare_ce_membership
→ bootstrap_weights
→ commit_routable
```

回收顺序固定为：

```text
drain
→ abort/partial-rollout
→ remove_from_lb
→ remove_from_ce
→ cleanup_runtime(destroy_process=False or True)
→ release_slot
```

`create_from_lease()`、`cleanup_runtime()` 和 `release_slot()` 不应暴露给普通请求路由代码，只能由 Replica Manager/TaskRunner 在 operation gate 内调用。

#### 15.8.4 CE 接口设计

borrowed replica 对 CE 暴露的是 endpoint 协议，而不是 donor 的 Worker 对象：

```python
class CheckpointEndpoint(Protocol):
    @property
    def workers(self) -> Sequence[CEWorkerHandle]: ...

    async def prepare(self) -> CEMetadata: ...
    async def init_process_group(self, topology: CETopology) -> None: ...
    async def bootstrap(self, target_version: int) -> BootstrapReceipt: ...
    async def remove(self, topology_epoch: int) -> None: ...
    async def finalize(self) -> None: ...
```

borrowed endpoint 的实现必须保证：

1. `workers` 数量等于 borrowed replica 的 TP/DP/PP world size；
2. 每个 Worker 是 borrower 自己创建的 actor/process endpoint；
3. `init_process_group()` 使用独立的 `task_id + replica_id + topology_epoch` 命名空间；
4. `bootstrap()` 只使新 replica 追平目标 serving version，不改变 donor 或其他 replica 的状态；
5. `remove()` 返回后，旧 group、IPC socket、RDMA registration 和临时 buffer 已关闭或进入可验证的隔离状态。

CE Manager 通过 `replica.checkpoint_endpoint` 或 `replica.workers` 获取这些接口；不能因为 native 和 borrowed 都有 `workers` 字段，就把 donor handles 混入 borrowed 列表。

#### 15.8.5 LB 接口设计

两类 replica 都向 LB 提供同样的路由接口，但 borrowed 必须把 lease 和生命周期纳入状态检查：

```python
async def register_route(self, endpoint: ServerEndpoint, routing_epoch: int) -> None
async def acquire(self, request_id: str) -> RouteLease
async def release(self, request_id: str) -> None
async def begin_drain(self) -> DrainReceipt
async def inflight_count(self) -> int
async def commit_remove(self) -> None
```

`acquire()` 只有在 `runtime_state == READY`、`lease_state == ACTIVE`、`ce_membership == ACTIVE` 且权重版本有效时才允许成功。`begin_drain()` 后仍要保留旧请求计数，直到 partial rollout 或请求完成；`commit_remove()` 才能调用原生 `remove_servers()`。

#### 15.8.6 与 native replica 的共用和区别

| 维度 | 共用部分 | native replica | borrowed replica |
|---|---|---|---|
| 类型协议 | `describe/prepare/activate/drain/sleep/wake/remove/destroy` | 实现同一协议 | 实现同一协议 |
| 状态机 | `PREPARED → READY → DRAINING → SLEEPING/DESTROYED` | 同一状态语义 | 同一状态语义，增加 lease 校验 |
| 配置 | rollout/model/TP/DP/PP/backend | 使用任务启动配置 | 使用 borrower 配置和 lease 拓扑 |
| replica 标识 | `replica_id/rank/world_size/version` | 任务初始分配 | borrower 新分配，不能复用 donor replica ID |
| 资源绑定 | 对外都表现为 placement | 自有 ResourcePool/PG | GS slot lease + Supervisor，不能新建 PG 抢卡 |
| CE Worker | 都通过 `CheckpointEndpoint` 提供 workers | 任务自有 Worker | borrower 新建 Worker，禁止 donor handle |
| Server/engine | 都提供 ServerEndpoint 和控制方法 | 原生 Ray/vLLM runtime | borrower 自有 Ray facade 或外部进程 runtime |
| process group | 都需报告 topology/version | 原生 topology | 独立 topology epoch/group namespace |
| LB | 都可注册、摘流、排空、移除 | 通常长期存在 | 受 lease 过期和 reclaim 控制 |
| 参数同步 | 都必须追平 serving version | 原生同步路径 | bootstrap + borrowed topology 路径 |
| sleep/wake | 状态和回执语义一致 | 通常保留 PG | 保留或恢复进程，不能把进程存活当作资源释放 |
| destroy | 接口存在，能力由 policy 决定 | 捐赠归还时通常 `POLICY_DENIED` | lease 结束后允许，必须清理进程和端口 |
| 故障处理 | 健康检查、隔离、幂等 | 任务退出负责清理 | Supervisor 负责孤儿检测和 slot 回收 |

#### 15.8.7 兼容性和生命周期不变量

`BorrowedReplica` 必须满足以下不变量，才能被放入 `effective_replicas`：

1. `kind == borrowed` 且 `slot_lease` 未过期，物理 GPU 映射与启动环境一致；
2. `workers` 和 `servers` 全部由 borrower 创建并可独立销毁；
3. CE、LB、Manager 的 replica ID 和 topology/routing epoch 一致；
4. 当前 serving version 已完成 bootstrap，且 health 状态为 READY；
5. donor 退出、sleep 或 destroy 不会通过对象所有权连带杀死 borrowed runtime；
6. borrowed reclaim 完成前，所有 in-flight 请求都有完成、续推或明确失败回执；
7. 任一阶段失败都只能返回 `PREPARED`、`QUARANTINED` 或 `DESTROYED`，不能返回 ACTIVE。

最终，native 和 borrowed 的兼容性由公共协议保证；`BorrowedReplica` 的独立 Worker、server、进程和通信组由资源租约保证。二者可以共享一套 Manager 和生命周期算法，但不能共享 donor 的运行时对象。
