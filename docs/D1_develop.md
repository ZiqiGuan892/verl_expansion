# D1 开发记录：创建契约、replica 身份与幂等操作

## 1. 阶段目标与边界

D0 已由用户完成真实环境验收并确认通过。本阶段依据 [verl 能力扩展最新版](verl能力扩展最新版.md) 的 D1 设计，实现 borrowed replica 创建请求的本地契约层：

1. 校验并归一化 GS 下发的 `spec`，把旧字段 `selected_slots` 统一转换为 `claims`。
2. 校验 `world_size`、claim rank、碎片化 PG/bundle 描述、租约、过期时间、容量份额和均匀节点布局。
3. 为 borrower 任务分配只增不复用的 `replica_rank`，并避开 native、hybrid、已失败和已退休的编号。
4. 用 manager 内的 `borrowed_operations` 和短临界区锁保证同一 lease 的创建请求幂等。
5. 预留 `reclaim`、`destroy` 等生命周期接口，但明确返回未实现，不伪造 `RUNTIME_READY`、`DESTROYED` 或 `released=True`。

本阶段**不**解析 Ray PlacementGroup，不查询 Ray bundle，不创建 CE Worker、HTTP server、vLLM engine，不建立通信域，也不改变 GS 的全局租约表。真实 borrowed runtime 属于 D2。

### D1 运行时回归修复：GS 注册 RPC 超时

真实 Ascend 启动日志曾在 `MultiTaskFullyAsyncTaskRunner.run()` 的
`GroupScheduler.attach_task` 处报 `ray.exceptions.GetTimeoutError`。profile 解析、根 Actor
创建和 GS 发现已进入执行；失败边界是 TaskRunner 等待 GS 注册 RPC 返回。原实现固定等待
30 秒；日志显示 GS 进程此时仍在执行 Ascend/MindSpeed 初始化，启动等待预算可能不足。

修复位于 `integration/verl/experimental_fully_async/task_runner.py`：

仅将 `ray.get(self.group_scheduler.attach_task.remote(...), timeout=30)` 改为
`timeout=120`。退出时 `detach_task` 的 30 秒等待和原有异常处理保持不变。

按 MVP 原则撤回此前添加的超时轮询、环境变量配置、Actor ID 转换、ActorHandle 兼容处理
和额外日志；`scheduler/group_scheduler.py` 恢复原有严格类型检查，无最终代码改动。

两次故障（CACHE_SIZE 导入路径、GS 注册超时）的 traceback、证据、修复差异和服务器验证命令
统一记录在 [issue.md](../issue.md)。本地检查不代表真实 Ascend 训练已通过；服务器需重新运行
`multi_task_run.sh`，确认通过 GS 注册并进入 Trainer/Rollouter 初始化。若 120 秒后仍超时，
需检查 GS 日志、Actor 状态和资源，而不能仅凭超时认定 ActorHandle 不兼容。

## 2. 修改文件总览

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py` | 扩展 `MultiTaskLLMServerManager` 的 D1 状态、spec 校验、rank 分配、幂等记录、创建/回收预留入口 | manager 是本任务本地 runtime 和操作记录的所有者；它不能把 ActorHandle 或 PG handle 传给 GS |
| `src/multi_task_scheduler/rollout/replica.py` | 扩展 `MultiTaskvLLMReplica` 的 allocation/lease/claim/runtime 元数据；增加显式未实现的 `init_from_lease`、`destroy`、`reclaim` | borrowed 必须继续兼容 `vLLMReplica` 的外部对象契约，D1 只先建立字段和生命周期边界 |
| `tests/unit/test_borrowed_contract.py` | 新增隔离单元测试 | 不导入 Ray、vLLM、NPU；直接执行扩展类真实方法体，验证 D1 的纯状态逻辑 |
| `D1_test.sh` | 新增服务器一键测试入口 | 对齐用户已跑通的 `multi_task_run.sh`：固定 `verl/multi_task_verl` 路径、加载相同 Ascend 环境、设置相同 `PYTHONPATH`，执行 D1 契约测试并保存日志；兼容旧 Bash，不依赖 `pipefail` |
| `docs/develop_step.md` | 将 D1 标记为代码开发完成，链接本记录 | 保持阶段门禁和开发日志与代码状态一致 |
| `Agent.md` | 更新交接门禁为 D1，记录当前新增文件及 D2 禁止提前开发 | 防止下一位 Agent 将 D1 代码误认为 D2 runtime 已完成 |

外层 `D:\verl` 原生仓库没有修改。本阶段没有创建新的组件类、全局资源表、SlotSupervisor、Coordinator 或 ReplicaFactory。

## 3. 逐步开发过程

### D1-1：确定 manager 的本地状态

在 `MultiTaskLLMServerManager.__init__` 中保留原生父类的三参数构造方式。这一点很重要：配套的 `FullyAsyncLLMServerManager` 不接受较新版本 `LLMServerManager` 的 `start_rank` 和 `load_balancer_cls` 参数，因此不能把新版本签名直接传给父类。

父类初始化完成后，扩展以下字段：

```python
self.start_rank: int
self.max_colocate_count: int
self.next_replica_rank: int
self.retired_replica_ranks: set[int]
self.borrowed_operations: dict[str, dict]
self.replica_operation_lock: asyncio.Lock
```

- `start_rank` 和 `next_replica_rank` 用于任务内 rank 分配。
- `retired_replica_ranks` 记录已完成生命周期的编号；编号永不复用，避免 CE、server 或日志仍持有旧 rank 时产生歧义。
- `borrowed_operations` 的 key 是 borrower 主 `lease_id`，value 是一次借用生命周期的本地状态，不是持久审计日志，也不是 GS 的全局租约表。
- `replica_operation_lock` 只保护字典查重和 rank 分配，不包住后续 Ray/engine 耗时操作。D1 没有耗时 runtime，所以创建记录随后立即变为明确失败状态。

`_read_max_colocate_count()` 从 rollout 配置读取 M；缺省为 10。它只校验本地配置是正整数，不会修改已经创建的 PG。D2 才会用 M 参与 Worker 放置。

### D1-2：实现 `_validate_create_spec`

`_validate_create_spec(spec)` 在进入锁前执行，返回深拷贝后的规范字典，绝不修改调用方对象，且不访问 Ray。

它执行以下检查：

| 检查 | 规则 | 目的 |
| --- | --- | --- |
| 顶层身份 | `operation_id`、borrower `lease_id`、`borrower_task_id`、`borrower_replica_id` 必须是非空字符串 | 防止不同任务或不同借用混用同一记录 |
| 期限 | `expires_at` 是未来 Unix 时间戳 | 过期授权在创建前直接失败 |
| 规模 | `world_size > 0` 且等于 claims 数量 | 允许 donor/borrower world size 不同，同时保证 borrower rank 数确定 |
| 输入字段 | 优先读取 `claims`，否则读取 `selected_slots` | 兼容当前 placement contract，归一化后只保存 claims |
| claim 身份 | `claim_id` 不重复；PG、bundle、node、GPU、donor task 和 source lease 完整 | 每笔容量声明可独立追踪；同一 bundle 可出现多个不同 claim |
| rank | 缺失时使用 selected list 的顺序；最终必须是 `0..world_size-1` | borrower 使用自己的连续 rank，不沿用 donor rank |
| 份额 | `0 < gpu_fraction <= 1`，`0 < cpu_request <= M` | 防止单笔 claim 本身非法 |
| bundle 累计份额 | 同一 `(pg_id, bundle_index)` 的 GPU 累计不超过 1、CPU 累计不超过 M | 支持碎片化 bundle 和多个 Worker 共用 bundle，同时阻止明显超额请求 |
| 节点布局 | `node_rank` 连续；每个 node 的 Worker 数相同；`local_rank` 从 0 连续 | 保证 D2 可以复用原生按节点启动 HTTP server 的假设 |
| source lease | 顶层 `lease_ids` 覆盖每个 claim 的 `lease_id` | 一个 borrowed replica 可由多个 donor/source lease 拼成 |

函数只做静态契约校验。PG 是否存在、bundle 是否仍有实际空间、设备 UUID 是否匹配，由 D2 的 runtime 创建阶段负责。

### D1-3：实现稳定 rank 分配

`_used_replica_ranks()` 汇总以下编号：

- 原生 `rollout_replicas`；
- fully async 的 `hybrid_replicas` 和 `alive_replicas`；
- `borrowed_operations` 中已经登记过的 rank，包括失败记录；
- `retired_replica_ranks`。

`_allocate_replica_rank_locked()` 只能在 `replica_operation_lock` 内调用：

1. 如果 spec 指定了 rank，检查该编号未被占用，然后推进 `next_replica_rank`。
2. 如果 spec 未指定 rank，从 `next_replica_rank` 开始递增，跳过所有已使用编号。
3. 分配后立即推进游标，失败不回退，也不把编号重新放回可用集合。

因此 `replica_rank` 是任务内稳定身份，不是列表下标；borrowed replica 创建失败或后续销毁后，该编号仍不能被新的 replica 复用。

### D1-4：实现创建幂等记录

`create_borrowed_replica(spec)` 的 D1 行为如下：

1. 在锁外调用 `_validate_create_spec`。
2. 在锁内按 borrower 主 `lease_id` 查找已有记录。
3. 如果已有记录，比较冻结的 request spec。比较时忽略新的 `operation_id`，因为它只表示调用方尝试；lease 加 placement 才是幂等身份。
4. 输入相同则返回第一次操作的 receipt，不重新分配 rank；输入不同则拒绝，避免一个 lease 覆盖两个 placement。
5. 新 lease 还会检查 `operation_id` 是否已被另一条记录使用；相同 operation ID 不能指向两个 lease。
6. 新请求分配 rank，保存 claims、source leases、请求快照和本地 runtime 引用槽位。
7. D1 不启动 runtime，因此把记录转为 `FAILED`，返回错误码 `RUNTIME_CREATION_NOT_IMPLEMENTED`。

返回值只包含可序列化元数据：

```python
{
    "operation_id": str,
    "lease_id": str,
    "lease_ids": list[str],
    "replica_rank": int,
    "state": "FAILED",
    "released": False,
    "error": {"code": str, "message": str},
}
```

它不包含 replica、Worker、server、PG 或 GS 句柄。D2 会把同一个记录的状态推进到 `RUNTIME_READY`，但不能改变 D1 已定义的幂等键和回执结构。

### D1-5：保留生命周期边界

manager 的 `reclaim_replica(lease_id)` 只校验 borrower lease 是否有本地记录，并返回 `LIFECYCLE_NOT_IMPLEMENTED`；它不把 claims 标记成全局空闲，也不清理任何 Actor。

`MultiTaskvLLMReplica` 新增的字段均在调用父类后保持 native-compatible：

```python
allocation_kind: str                 # native 或 borrowed
lease_id: str | None                 # borrower 主 lease；native 为 None
source_lease_ids: list[str]          # donor 授权的 source leases
donor_task_ids: list[str]
donor_replica_ranks: list[int]
runtime_state: str                   # 初始 CREATING
owns_resource_pool: bool             # borrowed 必须为 False
max_colocate_count: int | None
claims: list[dict]
serving_version: int | None
```

- `init_from_lease()` 当前直接抛出 `NotImplementedError` 并置为 `FAILED`，不能被误认为已经创建 runtime。
- `destroy()` 和 `reclaim()` 当前只返回 `released=False` 的可序列化回执。
- `reclaim()` 会拒绝 native replica 和错误的 borrower lease。
- 原生 `init_standalone()`、`launch_servers()`、`sleep()`、`wake_up()` 等业务方法没有被复制或改写。

## 4. 测试设计与执行

### 4.1 新增单元测试

`tests/unit/test_borrowed_contract.py` 使用 AST 提取扩展类的真实方法体，并替换原生父类。这样可以验证 manager/replica 的 D1 代码，同时不导入服务器才有的 Ray、vLLM、torch_npu 或 NPU runtime。

覆盖内容：

1. claims 归一化不修改输入，rank 连续，source lease 汇总正确。
2. world size、过期 lease、重复 claim、GPU fraction、非均匀节点布局被拒绝。
3. 同一 lease 的重复创建只分配一个 rank；新的 operation ID 不会重复创建。
4. 并发重复创建只产生一个本地 operation 记录。
5. 创建失败不伪造 runtime 成功，reclaim 不伪造 released。
6. rank 退休后不复用，错误 owner 不能退休该 rank。
7. native/borrowed replica 元数据隔离，错误 lease 被拒绝。

### 4.2 服务器上的验证命令

将 `D1_test.sh` 放到服务器上 `multi_task_run.sh` 所在目录。2026-09-22 起，脚本直接沿用用户已跑通版本的路径配置，不再自动猜测插件目录名：

```text
VERL_REPO_DIR/
├── multi_task_run.sh
├── D1_test.sh
└── verl/
    ├── verl/                # 原生 Python 包
    └── multi_task_verl/      # 插件仓库，包含 src/ 和 tests/
```

在 D0 已通过的环境中，按训练入口相同的调用方式执行：

```bash
cd /absolute/path/to/VERL_REPO_DIR/verl
bash ../D1_test.sh
# 可选：使用指定解释器，遇到首个失败即退出，并输出详细测试名。
MT_PYTHON=/absolute/path/to/python3 bash ../D1_test.sh -x -vv
```

脚本默认使用 `python3`，与训练入口一致；该解释器需要已经安装 pytest。`VERL_REPO_DIR`、`VERL_SOURCE_ROOT`、`VERL_MULTI_TASK_ROOT`、`ASCEND_TOOLKIT_ENV`、`ASCEND_ATB_ENV` 均可按训练入口的方式覆盖。若直接从插件仓内运行脚本，必须显式指定 `VERL_REPO_DIR` 为上图最外层目录。

| 配置 | 与已跑通 `multi_task_run.sh` 的关系 |
| --- | --- |
| 目录与 Python 搜索路径 | 相同的 `BASH_SOURCE[0]` 定位、三个根目录默认值及两次 `PYTHONPATH` 设置；先进入原生源码根 |
| Ascend 环境 | 相同的 Toolkit/ATB 脚本路径、加载顺序和 NPU/HCCL/vLLM 环境变量默认值 |
| 缓存和日志 | `HF_DATASETS_CACHE=${VERL_REPO_DIR}/cache`；`LOG_DIR=${VERL_REPO_DIR}/logs`，D1 可用 `D1_LOG_DIR` 单独覆盖 |
| 执行入口 | D1 改为进入插件根执行 pytest；追加参数为 pytest 参数，不接收 Hydra 配置 |
| 模型、数据和训练配置 | D1 契约测试不消费这些参数，因此无需模型、数据集、训练资源或 runtime profile 配置 |
| 必要测试设置 | 禁用 pytest 第三方插件自动加载和缓存，检查测试目标存在，使用 `PIPESTATUS` 传播测试/日志错误以兼容旧 Bash |

需要手工运行或扩大测试范围时，可执行：

```bash
cd /absolute/path/to/VERL_REPO_DIR/verl/multi_task_verl
export PYTHONPATH="$PWD/src:/absolute/path/to/VERL_REPO_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RAY_USAGE_STATS_ENABLED=0
export MT_PYTHON=/absolute/path/to/the/d0/python

"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/unit/test_borrowed_contract.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/unit
```

D1 一键入口默认只运行 `tests/unit/test_borrowed_contract.py`，可通过 `D1_TEST_TARGET` 指定其他测试路径或 pytest node ID。默认日志为 `${VERL_REPO_DIR}/logs/d1_test_<时间戳>.log`。加载 Ascend 环境只为保持与服务器入口一致，不表示本测试会使用 NPU；默认 D1 测试不会启动 GPU、Ray Actor 或 vLLM server。`tests/native_unit` 仍需在配套 verl/vLLM 环境中单独运行；它验证父类关系，不代表 D1 runtime 创建成功。

### 4.3 本工作区执行结果

已完成以下静态核对：

- 新增测试文件、D1 文档和源文件均位于 `D:\verl\verl-multi-task`。
- 未修改外层 `D:\verl\verl` 原生仓库。
- D1 代码未出现 PlacementGroup 创建、Ray worker 创建、HTTP server 启动或 engine 启动调用。
- 由于当前 Windows 工作区没有可用 Python，执行 `uv run --no-project --no-cache python -m pytest -q -p no:cacheprovider tests/unit/test_borrowed_contract.py` 失败在 uv 发现 Python 阶段：`C:\Users\10764\AppData\Roaming\uv\python` 无访问权限。因此不能把本地测试记为通过。

服务器执行测试后，应将完整命令、Python/verl/vLLM 版本、测试输出和失败栈追加到本节。只有测试在 D0 已验收环境中通过，并由用户确认，才可以进入 D2。

## 5. D1 交付结论与 D2 门禁

D1 的代码开发已完成：输入契约、claim 归一化、rank 身份、幂等操作记录和生命周期预留接口均已落在现有扩展类中；没有引入新的业务组件，也没有触碰外层 verl。

D1 尚未宣称真实测试通过。下一阶段 D2 允许实现的内容仅包括：读取已授权的 PG/bundle claim、创建独立 CE Worker、创建 HTTP/vLLM runtime、读取实际设备并校验。D2 不得重新设计 D1 的 lease、claim、rank 和 receipt 契约，也不得通过 `init_standalone()` 为 borrowed replica 新建 PG。
