# D2 开发记录：按 claims 创建 borrowed runtime

## 1. 阶段结论

D2 的目标是把 D1 的“创建请求记录”推进为真实 runtime 创建：

```text
GS 下发 placement claims
  -> MultiTaskLLMServerManager 分配本任务 replica_rank
  -> MultiTaskvLLMReplica 解析已存在的 PlacementGroup/bundle
  -> 每个 claim 创建一个独立的 CE Worker actor
  -> 复用 verl 原生 vLLMReplica.launch_servers()
  -> HTTP server 启动 vLLM engine
  -> 校验 endpoint、node/device 映射
  -> 返回 RUNTIME_READY
```

本阶段没有实现 CE 参数同步、LB 接流、sleep、wake、reclaim 或 destroy 的完整编排。创建成功只表示 borrower 的 Worker、HTTP server 和 engine 已经启动并且实际落点通过校验；它不会自动加入全局 LB，也不会被加入 donor 的 Worker 或通信域。

D2 选择“复用 donor PG 的指定 bundle、创建新的 actor”的方案。borrower 不创建 PG，也不复用 donor CE Worker。这样支持非连续 bundle、多个 PG 拼接以及与 donor 不同的 world size，同时保持 donor actor 的所有权不变。

## 2. 本阶段修改的文件

### 2.1 `src/multi_task_scheduler/rollout/replica.py`

`MultiTaskvLLMReplica` 仍然继承原生 `vLLMReplica`。新增逻辑全部位于扩展类中，原生 `init_standalone()` 和 `launch_servers()` 均保持可用；D2 的 borrowed 路径只新增 `init_from_lease()`，不改变 native PG 创建。

新增或扩展的成员：

| 成员 | 用途 |
| --- | --- |
| `operation_id` | 当前 GS 创建操作的幂等标识，用于生成 actor 名称和故障定位 |
| `worker_group` | 对新建 CE Worker handles 的 `RayWorkerGroup.from_detached()` 包装；不代表 donor 的 WorkerGroup |
| `created_actor_names` | 本次新建的 Worker actor 名称；启动失败时用于诊断，清理只针对本 replica 创建的 handles |
| `expected_device_map` | rank 到 GS 声明的 node/GPU 映射 |
| `actual_device_map` | rank 到 Worker 运行时通过 Ray context 读取的 node/GPU 映射 |
| `node_layout` | node 到 node rank、全局 ranks、GPU 标识和 local GPU index 的布局 |

新增方法：

| 方法 | 参数/返回值 | 作用 |
| --- | --- | --- |
| `validate_placement(spec)` | `dict -> dict[str, PlacementGroup]` | 用 `ray.util.get_placement_group()` 按 `pg_name` 或 `pg_id` 查找已有 PG，并检查 bundle 下标；找不到立即失败，不创建新 PG |
| `_validate_claim_layout(claims, world_size)` | `list[dict], int -> tuple[dict, dict]` | 检查 rank 顺序必须按 `node_rank/local_rank` 分组，生成 server 分组所需的 `node_layout` 和期望设备表 |
| `_configure_borrowed_parallelism(spec)` | `dict -> None` | 按 spec 的并行配置复制 rollout config，计算 borrower 的 world size 和均匀节点拓扑；不会修改 native replica 的 config |
| `_create_workers_from_claims(groups, spec)` | `dict, dict -> None` | 每个 claim 使用明确的 PG、bundle、CPU fraction、GPU fraction、rank/world/master 环境创建一个新的 `MultiTaskCheckpointEngineWorker` actor，并包装为 detached WorkerGroup |
| `_cleanup_runtime()` | `() -> dict` | 只 `ray.kill()` 当前 replica 已创建的 server/worker handles，并返回 kill 请求和错误；绝不删除 donor PG 或 donor actor |
| `init_from_lease(spec)` | `dict -> dict` | 完成 placement 解析、Worker 创建、server/engine 启动和设备校验，成功返回可序列化 runtime metadata；失败清理并抛出异常 |
| `runtime_metadata()` | `() -> dict` | 返回 rank、lease、world size、endpoint、拓扑和实际设备映射，不返回 Ray handles |

每个 CE Worker actor 的调度策略是：

```python
PlacementGroupSchedulingStrategy(pg, claim["bundle_index"])
```

CPU 使用 claim 的 `cpu_request`，加速器使用 `num_gpus=gpu_fraction`（目标 verl 0.9 的 Ray worker API）；同一个 bundle 是否能容纳多个 Worker 由 PG 创建时的 CPU/GPU 容量和 GS 的 claims 共同决定；修改 `max_colocate_count` 不会扩大一个已经存在的 PG。Ray 资源调度失败时，创建流程进入 FAILED，不能把“提交了 actor 请求”当成成功。

Worker 环境显式设置 `WORLD_SIZE`、`RANK`、`LOCAL_RANK`、`RAY_LOCAL_WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`。master 地址和端口通过原生 `get_master_addr_port()` 在第一个 claim 的 PG bundle 上获取，其他 PG 只复用这个通信入口。

server 创建没有另写一套 vLLM 启动器，而是先按 borrower 的 node/local rank 排序 Worker，再调用继承自原生 `vLLMReplica` 的 `launch_servers()`。该原生方法会：

1. 从 Worker actor 的 Ray runtime context 读取 node id 和 accelerator id；
2. 按每节点固定的 Worker 数切分 Worker 列表；
3. 使用 `NodeAffinitySchedulingStrategy` 在对应节点创建 `MultiTaskvLLMHttpServer`；
4. 将 `cuda_visible_devices`、rollout config、model config、Worker handles 和并行拓扑传给 HTTP server；
5. 调用每个 server 的 `launch_server()`，最后读取第一个 server 的 endpoint。

这里的 HTTP server 是 borrower 自己创建的 server actor，内部 engine 也由该 actor 自己初始化。donor 的 HTTP server、engine、CE Worker 和通信域均不复用。

### 2.2 `src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py`

`create_borrowed_replica()` 从 D1 的“记录 FAILED”改为真实异步创建：

1. 在锁内做 D1 spec 校验、lease/operation 幂等检查和 `replica_rank` 分配；
2. 将记录写为 `CREATING`，冻结规范化后的 claims；
3. 释放锁，在锁外构造 `MultiTaskvLLMReplica` 并调用 `await init_from_lease()`，避免慢速 engine 启动阻塞其他 lease；
4. 失败时把记录更新为 `FAILED`，只返回错误 receipt，不发布任何 server；
5. 成功时保存本地 replica、worker/server handles、actor 名称和 runtime metadata，状态更新为 `RUNTIME_READY`；
6. receipt 只包含字符串、数字和映射，不向 GS 或其他任务泄露 ActorHandle/PG handle。

manager 仍只拥有本任务的 `borrowed_operations`。全局 lease、bundle 是否可借、跨任务冲突和归还授权由 GS 决定，manager 不维护第二份全局表。

### 2.3 测试文件

| 文件 | 内容 |
| --- | --- |
| `tests/unit/test_borrowed_contract.py` | 将 D1 的创建断言更新为 fake runtime receipt；仍验证 idempotency、rank 分配和预留 reclaim 行为 |
| `tests/unit/test_borrowed_runtime.py` | 不启动 Ray/vLLM，使用真实扩展类方法验证 claims 的 node/rank 分组、交错布局拒绝、异构 world size 的 config 隔离，以及不可见 named PG 的明确失败 |

单元测试不宣称 GPU runtime 成功。真实 actor、engine、非连续 bundle 和跨 PG 场景必须由服务器上的 Ray/GPU 验收执行。

## 3. 完整创建流程与组件调用关系

```mermaid
sequenceDiagram
    participant GS as GlobalScheduler
    participant TR as TaskRunner
    participant M as MultiTaskLLMServerManager
    participant R as MultiTaskvLLMReplica
    participant PG as Existing PGs
    participant W as New CE Workers
    participant S as Borrowed HTTP Servers
    participant E as vLLM Engines

    GS->>TR: create spec with claims
    TR->>M: create_borrowed_replica(spec)
    M->>M: validate, deduplicate, allocate rank
    M->>R: construct borrowed replica
    R->>PG: get_placement_group(pg_name or pg_id)
    PG-->>R: existing PG handles
    R->>W: create one worker per claim on bundle
    W-->>R: actor handles
    R->>W: read runtime node/device and validate
    R->>S: inherited launch_servers()
    S->>E: initialize engine with borrower topology
    E-->>S: serving endpoint
    S-->>R: server address
    R-->>M: runtime metadata
    M-->>TR: RUNTIME_READY receipt
    TR-->>GS: serializable result only
```

此阶段没有 `CE register`、`bootstrap` 或 `LB commit READY` 箭头。D2 创建出的 endpoint 只保存在本任务 manager 的 operation record 中，直到后续阶段完成参数同步后才能接收业务请求。

## 4. 关键前提与失败语义

### 4.1 PG 必须可被 borrower 查找

GS 只下发 claims 时不会传递 Python PG handle。Ray 的公开 API 只能通过可见的 placement group 名称查找已有 PG，因此本实现将 claim 的 `pg_id` 作为默认 `pg_name` 使用，也接受显式 `pg_name`。

实际部署必须满足：

1. donor 创建 PG 时使用全局唯一、可查找的名称；
2. donor 和 borrower 在同一可见 Ray 集群及 namespace，或部署方提供等价的跨 job 查找机制；
3. GS 在 claim 中携带实际名称，不能只携带 donor 本地 Python 对象的 repr。

如果 `get_placement_group()` 查不到 PG，D2 返回明确的 runtime creation failure。实现不会猜测 PG、重新创建 PG 或退回使用 donor Worker。

### 4.2 world size 与模型并行约束

borrower rank 是本任务新分配的连续 rank，不能沿用 donor rank。manager 会把 `parallelism` 归一化到 claims 的 world size；未显式提供时使用 `TP=borrower_world_size, DP=PP=1`，显式提供时要求 `TP*DP*PP==world_size`。实现复制 rollout config 后启动 engine。vLLM 或模型如果不支持这个并行度，会在正常 engine 启动阶段失败，并按失败路径杀掉新建 actors。

因此“支持异构 world size”表示 placement、Worker rank 和 server 拓扑不再强制等于 donor；它仍受模型、vLLM backend 和设备显存的真实并行约束。D2 不通过修改 donor config 绕过这些限制。

### 4.3 设备身份校验

Worker 创建后从每个 actor 的 `ray.get_runtime_context()` 读取 node id 和 accelerator id，并与 claim 的 `node_id/gpu_uuid` 比较；若 claim 同时带 `local_gpu_index`，Ray 返回本地索引也作为同一设备的兼容表示。任何一个 rank 不匹配都进入 FAILED 并清理本次新建 handles。NPU/CUDA 环境仍应在 GS claim 生成侧统一 GPU 标识格式，索引兼容只用于适配 Ray 常见的 accelerator-id 表示，不能跨 node 使用索引推断设备归属。

### 4.4 失败清理边界

以下资源属于 borrower 本次创建，可以清理：

- 已创建的 CE Worker actors；
- 已创建的 HTTP server actors；
- 这些 actor 触发的 engine 子进程（由 actor 退出路径负责）。

以下资源不属于本次创建，不能清理：

- donor 的 PlacementGroup；
- donor 的 bundle 预留；
- donor CE Worker、HTTP server 或通信域。

完整 `reclaim()`/`destroy()`、PG claim 归还以及 engine shutdown 在 D2 仍是预留接口，不能把 D2 的失败清理误认为生产回收实现。

## 5. 验证步骤

### 5.1 本地可执行的静态/单元检查

服务器上可以直接从原生 `verl` 目录执行：

```bash
bash ../D2_test.sh
```

脚本会沿用 `multi_task_run.sh` 的 `VERL_REPO_DIR`、`VERL_SOURCE_ROOT`、
`VERL_MULTI_TASK_ROOT`、Ascend 环境和 vLLM 预检配置，默认执行 D1 契约、
D2 runtime 规划、wiring 以及 native parent 适配测试。每次运行的完整输出
保存在 `${VERL_REPO_DIR}/logs/d2_test_<timestamp>.log`。如果只验证不依赖
Ray/vLLM 的 D2 隔离测试，可使用：

```bash
D2_INCLUDE_NATIVE=0 bash ../D2_test.sh
```

在可用的 Linux Python 环境中，从插件仓根执行：

```bash
export VERL_SOURCE_ROOT=/absolute/path/to/compatible-verl
export PYTHONPATH="$PWD/src:$VERL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RAY_USAGE_STATS_ENABLED=0

python -m pytest -q -p no:cacheprovider \
  tests/unit/test_borrowed_contract.py \
  tests/unit/test_borrowed_runtime.py \
  tests/native_unit/test_native_adapters.py
```

通过标准：D1 契约、D2 纯布局和 native 父类适配测试全部通过；不得因为没有 Ray 而把 GPU 测试标为成功。

### 5.2 真实 Ray/GPU 验收

使用 D0 已能跑通的 Ascend/CUDA 启动环境，在 GS 预先产生一份**全局可查找**且确实有余量的 donor PG，然后由 borrower TaskRunner 调用 manager 的 `create_borrowed_replica(spec)`。每个场景至少保存：

- PG name、Ray namespace、bundle index 和 claim JSON；
- 每个 rank 的 actor ID、node ID、accelerator ID、CPU/GPU fraction；
- HTTP server actor、engine PID、endpoint 和启动耗时；
- `runtime_metadata()` 与 manager receipt；
- 失败场景中的 actor 清理结果及 donor PG 存活状态。

必须依次验证：

1. 一个 PG 的连续 claims；
2. 一个 donor 4 卡拆成两个 world size=2 的 borrower；
3. 两个 donor PG 各取 2 卡组成 world size=4 的 borrower；
4. 同一个 PG 的非连续 bundle；
5. 同 bundle 的多个 fractional Worker（PG 资源和 GS claims 均足够）；
6. 跨节点均匀布局和不均匀布局拒绝；
7. 缺失 PG、设备不匹配、engine 启动异常时不返回 READY，且 donor PG 未被删除。

本阶段暂不以业务请求成功作为通过条件，因为 LB 和参数 bootstrap 归 D3/D4；只确认 runtime 能启动、实际设备正确、endpoint 已建立且失败边界可定位。

## 6. 当前验证记录与限制

- 代码修改范围仅在 `verl-multi-task` 仓库；外层原生 `verl` 未修改。
- `uv run ... py_compile` 已通过。
- D2 相关隔离测试、D1 契约测试和 wiring 测试已通过：`22 passed`。这些测试没有启动 Ray、GPU 或 vLLM engine。
- 全量 `tests/unit` 在当前环境收集失败，因为本地虚拟环境没有 `omegaconf`；原生适配测试收集还需要 `ray`。这两项不能记录为通过，应在服务器的完整 verl 环境中执行。
- D2 代码依赖与实际运行环境匹配的 verl/Ray/vLLM/vLLM-Ascend 版本。导入路径、vLLM engine 启动失败和设备标识格式不一致，都应先记录为环境/兼容性问题，不应通过修改 donor 资源归属来规避。
- D2 完成后仍需用户以真实 Ray/GPU 日志验收，才能进入 D3 的 CE 通信域和 bootstrap 开发。
