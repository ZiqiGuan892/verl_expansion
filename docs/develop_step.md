# Replica 能力扩展：逐步开发与验证计划

本文将 [verl_expansion.md](verl_expansion.md) 拆成可以逐项实现、逐项验证的开发步骤。开发基础是本仓 `src/multi_task_scheduler/`，不是重新实现一套调度系统，也不是修改外层 verl 原生类。

**本文同时记录开发状态。** 未在“开发日志”中明确标记为已验收的步骤，不得作为后续步骤的通过依据；每一步通过用户确认后再接入依赖它的下一步。

## 1. 开发范围与共同约束

### 1.1 本轮交付目标

- native replica：安全退出服务、真实 sleep、归还后 wake、销毁。
- borrowed replica：在 donor 原有 PG/bundles 上创建独立 CE Worker、HTTP server 和 vLLM engine；加载 borrower 权重后服务；reclaim 时销毁自身运行时并确认归还。
- CE：动态注册/注销完整 replica，复用原生参数传输与通信域建立、finalize 流程。
- LB：摘流、等待请求结束、清理路由、提交 READY。
- TaskRunner：提供上述能力的任务级入口；GS 只调用 TaskRunner，并只接收结果元数据。

本轮不实现公平性算法、自动空泡识别、完整请求跨 replica 迁移、borrowed 休眠缓存或生产级故障恢复。测试中可以通过显式命令驱动一次借用，以验证能力；这不等于实现完整全局调度策略。

### 1.2 延续设计文档的选择

| 项目 | 本轮约束 |
| --- | --- |
| 代码位置 | 只在本仓扩展类、测试及示例中开发；不新增外层 verl 修改 |
| 插件入口 | 延续已有 `experimental_fully_async_standalone` profile 和原生训练入口；不创建伴生训练入口 |
| replica 类 | native、borrowed 均用已有 `MultiTaskvLLMReplica(vLLMReplica)`，以来源字段和初始化入口区分 |
| CE Worker | 保留现有 `MultiTaskCheckpointEngineWorker` 空子类即可；本轮不增加 Worker 通信或通用清理方法 |
| GPU 放置 | 用原生 `SubRayResourcePool` 包装 donor 的已有 PG，再用原生 `RayWorkerGroup` 创建新 CE Actors |
| 拓扑范围 | 完整 donor replica 的 bundles、相同 world size 和节点布局；不支持任意切卡、异构 world size 或多 donor 拼接 |
| 首个通信后端 | 全量 NCCL；在所有参与 backend 实例创建前配置 `rebuild_group=True` 与任务独立的 `group_name` |
| 回收语义 | borrowed 执行 destroy，donor 保留原 runtime；超时和 lease 过期不等于资源已归还 |
| 请求处理 | 首版以自然排空为必需路径；强制 abort 只有在 AgentLoop 结果处理闭环验证后才能开放 |
| 类的数量 | 不引入 SlotSupervisor、ReplicaFactory、Coordinator 或新的业务数据类；使用现有类与普通字典 |

相同 world size 不等于复用 donor 的 rank 环境、权重、通信域或 CE Worker。borrower 的所有运行时对象仍独立创建。不同 TP/DP/PP 组合是否支持，必须通过本轮布局及真实 engine 验收，不能仅凭乘积相等就放行。

[development-plan.md](development-plan.md) 记录此前 P1 接线阶段。本文件是用户新授权的能力开发阶段，旧计划中“不开发租约、成员管理等”的阶段范围不作为本轮禁止项；此前的验收记录也不自动证明新功能通过。

## 2. 如何保证每一步都能验证

### 2.1 每一步的完成规则

每一步均交付一小块行为及对应验证，顺序为：明确输入和失败条件 → 实现最小能力 → 测试成功、失败和重复调用 → 记录结果 → 再接下游。不要先把所有类和空方法写完，再集中调试。

1. 测试优先检查外部可观察结果，例如 Actor 数量、GPU UUID、实际权重版本、请求完成状态和资源释放；不只检查“某个方法调用过”。
2. 依赖尚未开发时，可以在单元测试中使用明确标注的替身，但真实环境验收必须调用生产扩展代码。不得用另一套测试实现替代生产创建/回收逻辑。
3. GPU、跨 job、进程退出等验收未通过，该步只能记为“局部验证通过/真实环境待验收”。不得将依赖这项结论的后续集成记为完成；可以继续独立组件的单元开发。
4. 超时用于暴露故障，不能作为清理成功依据。失败现场保留精确资源标识；测试清理只操作本次创建的 Actor、PG 和进程。
5. 每步只运行当前功能及受影响的回归项。阶段验收时再运行组合测试，不反复执行无关完整训练。

### 2.2 验证层次

| 标记 | 环境及证明内容 | 不能证明的内容 |
| --- | --- | --- |
| U | 无 GPU 的单元测试；字典校验、状态转换、并发交错和失败传播，可使用替身 | 原生类初始化、Ray 放置、CUDA、NCCL、vLLM 能否运行 |
| N | 真实 verl 父类和选定依赖；方法契约、序列化和原生委托 | 未启动的 GPU 运行时行为 |
| R | 真实 CPU Ray Actor/RPC；并发入口、跨进程投影和句柄边界 | GPU 配额绑定、显存释放、通信域正确性 |
| G | Linux、真实 Ray/GPU/verl/vLLM；设备绑定、权重传输、生成、休眠和进程清理 | 没有实际执行的跨 job、多卡或跨机场景 |

`tests/unit/`、`tests/native_unit/`、`tests/integration/` 已存在；计划新增 `tests/gpu/`。GPU 用例使用新注册的 `gpu_integration` marker，不能被默认 `tests/unit/` 无意启动。显式选择的 GPU 验收若缺配置、依赖或设备，应明确报错，不静默 skip 后报告通过。

### 2.3 命令约定与记录

本文件所有测试命令均从 **verl-multi-task 仓根**执行。环境管理使用 `uv` 和项目虚拟环境；GPU 测试沿用已经能运行原生训练的环境，不在没有 GPU 的开发机安装整套训练依赖。

Linux 验收环境准备如下；路径须替换为实际已存在的解释器和配置：

```bash
export MT_VERL_SOURCE_ROOT=/absolute/path/to/verl
export PYTHONPATH="$PWD/src:$MT_VERL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RAY_USAGE_STATS_ENABLED=0
export MT_PYTHON="$PWD/.venv/bin/python"

# 现有测试，可以在相应依赖具备时立即执行。
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/unit
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/native_unit/test_native_adapters.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/integration/test_group_scheduler.py
```

Windows 开发机使用实际虚拟环境的 `Scripts/python.exe` 和 PowerShell 环境变量语法；G 层在 Linux GPU 环境执行。单元层的原生源码对比仍要求 `MT_VERL_SOURCE_ROOT` 的 Git 对象包含既有固定基线。

下面各步骤列出的新测试路径是**待创建的验收文件**。创建后使用统一命令：

```bash
"$MT_PYTHON" -m pytest -q -p no:cacheprovider <该步骤列出的测试文件>
```

GPU 测试 fixture 统一读取 `MT_GPU_TEST_CONFIG` 指向的本地配置，至少包含隔离 Ray 集群地址、共同 namespace、已验证的 donor/borrower 原生配置路径、模型路径、超时和输出目录。fixture 负责验证条件和记录资源标识，不承担业务调度。CPU Ray 测试与 GPU 集群测试分开运行。

每一步记录：两个仓库的 commit/工作区状态、修改文件、命令、测试层次、通过/失败/未执行项、日志路径及下一步是否可开始。真实环境还记录 Ray/vLLM/PyTorch/CUDA 版本、每 rank 的 node/PG/bundle/GPU UUID、Actor ID、engine PID、权重版本及显存。不得直接沿用旧文档的测试数量作为本次结果。

## 3. 步骤总览与依赖

开发顺序优先处理容易推翻方案的底层条件，再组装管理接口。因此它不同于运行时的 `sleep → create → reclaim/destroy → wake` 顺序。

| 步骤 | 可独立交付的能力 | 主要依赖 | 必需验证 |
| --- | --- | --- | --- |
| D00 | 建立当前版本和原生运行基线 | 无 | U；进入 GPU 开发前补齐 N/G 基线 |
| D01 | replica 字段、placement 契约和纯校验 | D00 | U/N |
| D02 | 跨 job 共享 PG 与名称隔离可行性 | D01 | R/G |
| D03 | GS 句柄边界与训练期间命令可达性 | D00 | U/N/R |
| D04 | HTTP/engine shutdown 和 replica destroy | D01 | U/N/G |
| D05 | standalone native 的真实 sleep/wake | D04 | U/G |
| D06 | donor placement 导出与 lease 占用 | D01、D02、D05 | U/R；真实 donor 导出 G |
| D07 | 在租借 bundles 上创建独立 CE Actors | D02、D04、D06 | U/N/G |
| D08 | 启动 borrowed HTTP/engine 并完整回滚 | D07 | U/G |
| D09 | manager 的幂等创建、取消和本地归属 | D08 | U/R/G |
| D10 | CE 成员管理与同步 gate | D03、D09 | U/N/R |
| D11 | borrowed bootstrap 与真实通信域重建 | D10 | G |
| D12 | LB 摘流、READY 和请求收尾 | D01 | U/N/R；接 engine 后 G |
| D13 | TaskRunner 五类能力入口闭环 | D03—D12 | U/R/G |
| D14 | 跨 job、多卡/跨机与失败验收 | D13 | G |

D03、D12 可以在底层 GPU 验证期间独立开发；依赖未完成时只验收它们自身的行为。D02、D04、D05、D11 是关键阻断点：真实验证失败时，先修复当前步骤或修订局部设计，不能继续叠加上层功能掩盖问题。

## 4. 详细开发步骤

### D00：建立基线与最小验收环境

**目标：** 后续失败能够区分为既有环境问题或本次改动。

**开发与准备：**

1. 记录本仓及外层 verl 的版本、已有工作区修改；保留用户的未提交文件，不重置仓库。
2. 运行第 2.3 节的已有测试，按 U/N/R 层分别记录；缺依赖的层保留未执行状态。
3. 在 GPU 环境固定一份已能工作的 native standalone 配置和小模型；确认启用/关闭 profile 均能初始化、生成，并完成至少一次后续参数同步。
4. 为 GPU 用例建立最小 fixture 和配置说明。模型、集群、设备数量均由测试配置提供，不在测试中下载或假造 GPU。
5. 固定首轮可验收拓扑和显存预算。运行 NCCL 发送端时还需真实训练侧设备，不能把“1 张共享推理卡”误写为完整测试只需 1 张卡。

**验证：** 现有三组测试及原生入口训练；新增 fixture 的配置错误用例，缺模型/错误集群应在创建资源前失败。

**通过标准：** 有本次执行的分层记录和可复现的原生配置。N/G 基线未通过前，不把后续原生运行失败归因于新接口。

### D01：补齐 replica 字段和 placement 输入契约

**修改位置：** `rollout/replica.py`；manager 的输入检查位置；新增 `tests/unit/test_replica_contract.py`。下文生产源码路径均相对 `src/multi_task_scheduler/`。

**开发内容：**

1. 在 `MultiTaskvLLMReplica.__init__()` 增加设计中的来源、lease、状态、PG 所有权、placement、创建资源记录和 serving version 字段；继续透传全部原生构造参数。
2. native 默认使用自己的资源；borrowed 只有在合法的 `init_from_lease(spec)` 路径中设置 `owns_resource_pool=False`，不改变父类计算并行规模的逻辑。
3. 实现 `validate_placement(spec)` 的纯校验部分：必需字段、完整 bundles、world size、节点布局、设备映射、租约标识/期限和 `max_colocate_count`。动态 PG 存活性留给 D02/D06。
4. 固定普通字典契约：placement 不含 ActorHandle/PG 对象；创建返回 `RUNTIME_READY` 元数据；回收成功必须同时有 `DESTROYED` 与 `released=True`。`RUNTIME_READY` 不等于 LB 可服务。
5. 固定标识规则：`lease_id` 去重；任务内用 `replica_rank` 定位；重试不得给同一创建分配第二组名称。过期时间使用上层统一时间语义，不跨进程比较各自的 monotonic 时间。

**验证：**

- U：合法 placement、缺字段、重复 bundle/GPU、world size 或节点数不符、已过期 lease、混入运行时句柄均有结果；失败时没有 Ray 创建调用。
- N：真实父类的构造签名和并行规模不变；对象在创建前及持有真实句柄后均可按原生投影方式序列化。锁和 asyncio Task 放在 manager 内，不能误放进需传给 Trainer 的 replica 对象。

**通过标准：** 不需要 GPU 即可判断哪些输入可进入创建；native 原有入口不受影响；本步不提供返回假成功的创建实现。

### D02：先验证跨任务 PG 复用与名称隔离

**修改位置：** replica 的 PG 解析私有函数；必要时在现有 manager 初始化入口增加任务级 rank 分配适配；新增 `tests/integration/test_placement_discovery.py`、`tests/gpu/test_shared_pg_binding.py`。

**开发内容：**

1. donor 导出 PG name/ID/namespace；borrower 在约定共同 namespace 解析，并核对 ID、状态和 bundle 布局。PG 名字相同但 ID 已变化时拒绝。
2. 使用两个独立 Ray jobs 做最小试验：donor 保留 PG 和 `GPU=0.5` 的 Actor，borrower 在同一个 bundle 新建独立 `GPU=0.5` Actor，读取真实 GPU UUID。
3. 提前解决共同 namespace 中的命名冲突。为两个任务的 native 和 borrowed 分配不重叠的 runtime replica rank；必要时在扩展 manager 的原生初始化调用点传入任务 rank 起点。CE adapter、server 命名和本地索引必须一致，不能只改 server 的 suffix。
4. 测量 bundle 的 CPU/GPU 余量；确认 actor 创建超时可以被定位、取消和清理。第三个占用者的试验只在隔离资源上执行，并及时清理待调度 actor。

**验证：**

- R：命名 PG 跨 job 可发现，错误 namespace/ID/已删除 PG 被拒绝；R 层可用 CPU bundles，但结论只限发现与所有权。
- G：两个独立 Actor ID、相同 PG ID/bundle/node/GPU UUID；没有多创建 GPU PG；第二个 Actor 销毁后配额可复用且 donor 不受影响。
- 名称检查同时包含两个任务的 native replicas；只让 borrowed 名称不冲突不算通过。

**通过标准：** 两个真实 jobs 的结果证明方案可放置。此时还没有证明原生 CE/Gloo/vLLM 可以运行；这分别由 D07/D08 验证。失败时停在本步，不改用手写 CUDA_VISIBLE_DEVICES 绕过 Ray。

### D03：修正 GS 句柄边界，验证管理命令可达

**修改位置：** `integration/verl/experimental_fully_async/{task_runner,rollouter,llm_server_manager}.py`、`rollout/load_balancer.py`；更新既有 wiring 测试，新增 `tests/integration/test_task_runner_control.py`。

**开发内容：**

1. 移除 Rollouter、LLMServerManager、LB 的 `group_scheduler` 参数、字段和下传；TaskRunner 保留 GS，GS 保留 TaskRunner。更新真实构造链与相应用例。
2. 增加任务级命令入口的参数检查、组件就绪检查和关闭状态检查。尚未完成的操作明确报错，不能返回成功。
3. 针对长时同步 `run()` 选择并验证现有 TaskRunner 的执行方式。优先评估 Ray 同步 Actor 的独立控制 concurrency group；不要只加一个 `async def`，将原生阻塞调用带进单个事件循环。
4. 若命令入口最终采用同步方法，则业务签名为 `execute_replica_operation(operation: str, request: dict) -> dict`，调用方仍通过 `.remote()` 获取结果；同步/异步选择和原因同步回设计文档。
5. 初始化/退出和控制线程共享的组件状态采用适合该并发模型的保护；不得跨线程共用 `asyncio.Lock`，不得持有本地状态锁等待跨 Actor RPC。

**验证：**

- U/N：构造链无 GS 下传；未就绪、退出中、非法命令均明确失败，原生异常仍向上传播。
- R：用真实扩展 TaskRunner 的控制实现，在训练等待路径持续占用时提交管理命令，命令可在测试期限内进入并返回；并发初始化/退出不会读到半初始化组件。
- 可以用测试事件替代训练计算，但必须说明这只证明控制可达性。D13 再用真实训练验收。

**通过标准：** 管理入口确实可执行，GS 只持有任务边界句柄；不能只靠配置了 `max_concurrency` 判定通过。

### D04：先实现 shutdown/destroy，再允许 borrowed 创建

**修改位置：** `rollout/http_server.py`、`rollout/replica.py`；新增 `tests/unit/test_replica_cleanup.py`、`tests/gpu/test_runtime_shutdown.py`。

**开发内容：**

1. 对固定 vLLM 版本确认 HTTP 主节点的 engine client、服务任务以及 headless 分支的关闭入口，新增 `MultiTaskvLLMHttpServer.shutdown()`。
2. 主节点关闭准入、HTTP 和 engine；其他节点处理自身 headless 运行时。原生 headless 使用后台线程，取消 asyncio 外壳不代表线程或子进程已退出，必须实现并验证实际关闭和等待。
3. `replica.destroy()` 支持空对象、部分 CE、部分 server 和完整 runtime；顺序为关闭 server/engine → 确认子进程退出 → 终止 server/CE Actors → 确认资源释放。它不主动查找其他 Actor 内的 CE Manager/LB。
4. 对已进入同步/服务的对象，destroy 要求外部已完成退出确认；对未发布的创建失败对象，可以直接回滚。borrowed 永远不删除 donor PG。
5. 记录本次确切 Actor 名称/ID和可核验的进程信息。清理失败保留引用及错误，重复 destroy 继续清理，不伪报成功。

**验证：**

- U：各创建阶段失败、重复 destroy、shutdown 异常、Actor 已退出；不删除 donor PG，不按宽泛名称前缀清理。
- G：先用原生创建的单节点 standalone runtime 验证关闭，生成请求已结束后关闭服务；确认 endpoint 不可用、Actor DEAD、engine/worker PID 消失，GPU 内存回到测得的合理基线。
- 多节点 headless 必须另测；没有跨机环境时仅标记单节点通过，不开放跨机借用。

**通过标准：** 有“进程和资源实际退出”的证据。只调用 `ray.kill()` 或只看到 GPU 计数下降不能通过。

### D05：实现 standalone native 的真实 sleep/wake

**修改位置：** `rollout/http_server.py`，必要的 replica 状态检查；新增 `tests/unit/test_server_sleep_wake.py`、`tests/gpu/test_native_sleep_wake.py`。

**开发内容：**

1. 在扩展 HTTP server 中补齐 STANDALONE 分支，调用真实 vLLM engine sleep/wake；其他已支持模式保留原生语义。遵守 `node_rank==0` 的 engine 控制边界，不能在每个 headless server 上访问不存在的 `self.engine`。
2. 检查 sleep 已启用、`free_cache_engine` 等前置配置。能力未启用或引擎不支持时返回明确错误，不能沿用原生 skip 后报告 SLEEPING。
3. 固定首个支持的 sleep level，分别说明权重是否保留及 wake 后是否必须重新加载；不同时承诺所有 level。保留 `wake_up(tags=None)` 原生参数契约。
4. 测试中先保证无请求/无同步，再直接调用底层 sleep/wake。生产级 LB、AgentLoop、CE 前置边界到 D13 才接入，本步不对外授予可借 lease。

**验证：**

- U：正确分支、节点控制、禁用配置、重复调用、引擎异常；不误把 `release_kv_cache()` 当作捐卡 sleep。
- G：睡眠前后 donor server/engine 身份保持；逐卡显存释放足以容纳计划的 borrower 启动和同步峰值；wake 后重新完成必要的权重恢复和生成。
- 多次循环检查显存/CPU offload 内存无持续增长；level 2 等丢弃权重场景在参数重新加载前不得验收为 READY。

**通过标准：** donor 确实释放了可借显存并能恢复；borrowed sleep/wake 不在本步实现。

### D06：实现 donor placement 导出与 lease 占用

**修改位置：** `integration/verl/experimental_fully_async/llm_server_manager.py`；新增 `tests/unit/test_bundle_leases.py`，扩充 D02/D05 的真实资源用例。

**开发内容：**

1. 增加 `bundle_leases` 和管理状态的短临界区；实现 `export_borrowable_placement(replica_id, lease_id) -> dict`。
2. 导出前要求本地 donor 已确认停止请求、退出 CE、进入 SLEEPING，且 bundles 没有其他有效 lease；读取真实 PG/设备元数据而非根据 GPU 序号猜测。
3. 同一 lease 重试返回同一 placement；其他 lease 竞争相同 bundles 失败。核对 Ray 余量只是前置检查，不当作新的资源预留或精确显存隔离。
4. `confirm_lease_released(lease_id)` 只在任务边界已经核验 borrower 的实际释放结果后解除占用；lease 过期、RPC 超时和 unknown lease 不触发盲目归还。
5. 本地方法不持有 GS；placement 经 TaskRunner 交给上层。测试可以显式设置已验证的前置状态，生产入口到 D13 接通。

**验证：** 未休眠/仍在同步时拒绝；重复导出一致；并发借用只有一个成功；过期不解锁；真实导出字典不含运行时对象且能在另一个 job 解析到相同 PG/GPU。

**通过标准：** 产生一份可以被 D07 使用且所有权清晰的 placement；donor 不会因 lease 过期自动 wake。

### D07：只打通 borrowed 的 CE Actor 创建和绑定

**修改位置：** `rollout/replica.py`；新增 `tests/unit/test_borrowed_placement.py`、`tests/gpu/test_borrowed_ce_workers.py`。

**开发内容：**

1. 将 `init_from_lease()` 的资源创建部分组织为已有 replica 内的私有辅助方法：解析/核验 PG → 构造 `SubRayResourcePool` → 构造原生 `RayWorkerGroup`。不新增 Factory/资源池类。
2. 使用完整 donor bundles 和原生 STANDALONE 语义；不调用 `init_standalone()` 申请新 PG，也不直接改为 `init_colocated()`。
3. CE class selector 保留当前空子类，复用全部原生构造和方法。借用 borrower 自己的配置，rank/master 环境由原生 WorkerGroup 产生，不拷贝 donor 环境。
4. `validate_runtime()` 的 worker 阶段检查 Actor 数量、初始化完成、PG/bundle/node、GPU UUID 和原生 rank/world size。CE 本地 rank、传输域 rank、engine rank分别记录，不假定相等。
5. 在 WorkerGroup 构造前登记精确创建标识；处理“部分 Actor 已创建，但构造尚未返回”的失败。同步构造不得无限阻塞取消入口，必须有可验证的超时、取消与残留追踪。

**验证：**

- U：没有新建或删除 PG；构造失败后精确清理；输入不合法不创建 Actor。
- G：新 CE Actor 使用原生构造完成 Gloo 初始化；与 donor Actor ID 不同，但每 rank 物理 GPU 一致；donor 不加入 borrower Gloo 组。
- 在 donor 本轮 CE 同步 finalize 且 sleep 后，测量 CE 初始化额外显存；任何 rank 失败都能清理本组而保留 donor。

**通过标准：** borrowed CE 确实在目标卡上独立初始化。本步尚未启动 vLLM，不返回公开的 `RUNTIME_READY`；测试直接验收私有创建阶段。

### D08：启动 borrowed HTTP/engine，形成可清理的 runtime

**修改位置：** `rollout/replica.py`、确有必要的 `rollout/http_server.py` 适配；新增 `tests/unit/test_borrowed_runtime.py`、`tests/gpu/test_borrowed_runtime.py`。

**开发内容：**

1. 接上继承的 `launch_servers()`：从新 CE 查询设备，按节点创建本任务 HTTP/headless servers，再由 server 启动 vLLM MP engine。
2. 核对 CE adapter 与 server 使用相同 runtime replica rank，job/IPC 寻址属于 borrower；支持范围内优先原生创建，不复制整个 `launch_servers()`。
3. 完成 endpoint、server 健康、engine 就绪及实际 GPU 映射检查后，才让 `init_from_lease()` 返回并进入 `RUNTIME_READY`。
4. 在每个耗时阶段结束以及最终发布前检查取消/lease 状态；失败统一调用 D04 的精确回滚。此时禁止自动加入 LB 或宣告已加载当前训练权重。

**验证：**

- G：donor sleep 后 borrowed engine 能在同一卡启动；拥有独立 server/engine/PID；原生接口可以生成测试请求。该生成只证明 engine 工作，不证明权重已 bootstrap。
- 在首个 server 创建、后续节点启动、engine 初始化和健康检查阶段注入失败；Actor/子进程清空、配额可复用、donor PG 保留。
- 连续执行创建 → 销毁 → 再创建，确认命名、端口及 IPC 没有阻塞第二次使用。

**通过标准：** 一个真正可运行但尚未发布的 borrowed runtime；失败不会留下占卡残留，也不会污染 donor。

### D09：实现 manager 的幂等创建与取消

**修改位置：** `integration/verl/experimental_fully_async/{llm_server_manager,rollouter}.py`；新增 `tests/unit/test_borrowed_operations.py`、`tests/integration/test_replica_projection.py`。

**开发内容：**

1. 增加 `borrowed_operations`，保存 lease 对应的输入摘要、创建状态、取消标记、本地 runtime 引用和结果；本地引用不能随结果发给 GS。
2. `create_borrowed_replica(spec)` 在短锁内去重和登记，锁外调用 replica 初始化，再次验证后提交；同 lease 不同 placement 应报冲突。
3. 同时收到相同创建时共享同一创建结果；创建中回收先标记取消，再由创建者收尾。不能删除记录后让旧创建协程把 replica 再次发布。
4. 原生 replica 列表、server 索引和本地操作表各有明确用途，避免新建另一份全量 registry。已建 runtime 与可路由 active 集合分开提交。
5. Rollouter 增加薄转发入口；Trainer 可获取本任务新 replica 的投影，但反序列化后的对象不作为原对象身份使用。

**验证：** 并发同 lease 只创建一组 Actors；不同输入冲突；创建中取消清理完成且不发布；清理失败保持不可归还。R 层验证投影可序列化、引用属于 borrower，任务级结果只有元数据；G 层复用 D08 检查真实 Actor 数量。

**通过标准：** 上层可以重复调用而不产生第二份 runtime；下一步 CE 注册有稳定可获取的对象投影。

### D10：实现 CE 成员管理与同步互斥

**修改位置：** `checkpoint/checkpoint_engine_manager.py`、`integration/verl/experimental_fully_async/trainer.py`；新增 `tests/unit/test_ce_membership.py`、`tests/native_unit/test_ce_delegation.py`。

**开发内容：**

1. 新增 `sync_gate`、同步状态、在途成员快照和已确认版本；`update_weights()` 在入口取得 gate，直到完整原生同步返回才释放，保留原生返回指标及 `auto_await` 语义。
2. 实现 `register_replica()`/`unregister_replica()`，在同一 gate 下复用原生 `add_replicas/remove_replicas`。按稳定标识查找本地投影，移除本地已保存对象；禁止用新反序列化对象做身份相等判断。
3. 失败时进入 BLOCKED 并保留参与者信息；不尝试在仍可能传输时杀 Worker，也不将再次调用 finalize 视为通用恢复。空成员集合跳过建组，返回兼容结果。
4. Trainer 增加注册/注销入口；所有可能与生命周期冲突的批量 wake/sleep/参数更新必须受同一成员边界约束。内部同步调用不能再次取得自身已经持有的非重入锁。
5. 在创建训练端、native 接收端和 borrowed 接收端 backend 之前，准备共享能力所需 NCCL 配置；核验实际实例的 `rebuild_group/group_name`。只改 CE Manager 配置不生效；默认关闭 profile 的配置保持原生。
6. 不扩展 CE Worker，不新增 `close_runtime/close_transfer`。原生 backend 完成本轮 finalize；销毁时由 server shutdown 和 CE Actor 退出负责各自长期资源。

**验证：**

- U：通过可控制事件分别暂停在 prepare、传输、finalize、恢复生成阶段，期间注册/注销不能完成；同步成功后注销，再次同步快照不含该 Worker。
- U：重复注册/注销、序列化副本、空集合、同步异常、取消；版本只在完整成功后提交。
- N/R：原生同步方法实际被复用；async gate 与 Actor 事件循环一致，不跨线程/跨循环使用；测试返回值和调用契约。

**通过标准：** CE 只看到完整成员快照，sleep/destroy 可取得“目标已退出同步”的确认。单元测试通过尚不代表 NCCL 动态成员可用，必须继续 D11。

### D11：验证 borrowed 权重 bootstrap 和通信域重建

**修改位置：** Trainer 的现有权重同步边界及必要的薄入口、CE Manager；新增 `tests/gpu/test_ce_dynamic_membership.py`。

**开发内容：**

1. 在 D08 的 `RUNTIME_READY` 上注册 borrowed，复用原生 `prepare → build_topology → init_process_group → update_weights → finalize`。
2. 首版使用全成员同步，不开发 target-only bootstrap。优先在 Trainer 原有安全同步点处理新成员；若必须在当前窗口立即 bootstrap，新增的薄入口也必须等待训练侧权重不再修改，并锁定同一权重版本。
3. 区分 CE gate 与训练优化器安全边界：CE gate 不能单独防止 optimizer 更新参数。不得由 TaskRunner 在任意时刻直接并发调用 `_fit_update_weights()`；该方法还包含原生训练/陈旧度控制行为。
4. 成功后把明确的 serving version 返回任务内调用方，仍不自行发布 LB；失败维持不可服务。等待安全同步点超出 lease 时，取消并回滚，不使用磁盘初始权重冒充当前版本。
5. 回收先等当前同步完整退出再注销；在 `rebuild_group=True` 下上一轮已经 finalize，不重复销毁。下一轮自然按剩余成员建新域。

**验证：**

- G：借用前、加入后、移除后各执行一次真实同步；核对参与 Actor、传输 rank/world size 和 NCCL 组生命周期；donor Workers 不在 borrower 同步域。
- 用可区分的 donor/borrower 参数内容或受控权重变化，验证实际 engine 加载正确；单有 `global_steps` 标签不足以证明权重到达。证据可使用测试专用参数校验读数及固定输入的预期输出变化。
- borrowed 删除后 borrower 的 native replicas 仍能同步和生成；donor 不被本次参数更新改变。
- 重复注册/同步/移除检查通信资源和显存无持续增长；如果 backend 残留连接妨碍循环，停在此步定位后端资源，不先加入一套通用 Worker 清理层。

**通过标准：** 真实动态通信域和正确权重加载均通过；bootstrap 不增加训练 step、不打乱优化器/参数版本，也不会唤醒已捐出的 donor。

### D12：实现 LB 摘流、READY 和请求收尾

**修改位置：** `rollout/load_balancer.py`、现有 Rollouter/manager 的请求入口；新增 `tests/unit/test_lb_lifecycle.py`、`tests/integration/test_request_drain.py`。

**开发内容：**

1. 增加 `draining_servers`，实现 `begin_drain()`：停止新 acquire，保留已 acquire 请求的计数；正常选路、sticky 和 full-determinism 分支都必须排除 draining 目标。
2. `commit_remove()` 在请求已结束后清理目标 sticky 缓存、server 映射、计数和 drain 标记，复用原生 `remove_servers()`；迟到/重复 release 不得污染其他实例或形成负计数。
3. `commit_ready()` 只由已核验 runtime/serving version 的调用方触发；运行时就绪但未同步不得服务。无可路由 server 时沿用或明确处理等待/失败语义，不能返回休眠目标。
4. Rollouter 同时处理 AgentLoop 的在途工作、已 acquire 尚未提交的请求和客户端本地 server 引用。不能仅以 engine 的请求数为零就判断排空。
5. 首版自然排空：已有工作正常完成，后续轮次重新获取可用 server；若原生客户端缓存导致请求绕过 LB，需在本仓适配现有创建/客户端扩展点并测试，不能忽略。
6. 若开放强制 abort，则必须另外验证原生 partial rollout 或明确的重试结果处理，确保样本不被静默丢失/重复。闭环尚未验证时拒绝强制回收或保持 DRAINING。

**验证：**

- U/N：摘流前后 acquire、sticky 命中、确定性路由、无 server、重复移除、迟到 release；原生未摘流目标路由规则不变。
- R/G：在 acquire 后、提交前、生成中和 AgentLoop 下一轮分别触发 drain；没有新工作进入目标，已分发工作可完成或明确返回待处理结果，计数与实际完成一致。

**通过标准：** `DRAINING` 只表示退出准入并等待收尾；达到明确的排空条件后才允许 CE 退出和底层清理。它本身不是“已经回收”。

### D13：接通 TaskRunner 的五类能力入口

**修改位置：** 现有 TaskRunner、Trainer、Rollouter、LLMServerManager；新增 `tests/unit/test_replica_operations.py`、`tests/gpu/test_replica_capability_cycle.py`。

**开发内容：** 在同一任务内串行处理冲突生命周期命令，用已有对象编排有限步骤，不引入独立事务类。GS 只通过 `execute_replica_operation()` 调用；本步测试由明确脚本下发命令，不实现 GS 自动策略。

| 命令 | 必须等待完成的调用链与条件 |
| --- | --- |
| sleep | TaskRunner → Rollouter/manager → LB `begin_drain` 与 AgentLoop 收尾 → TaskRunner → Trainer `unregister_replica` 等待旧同步完成 → Rollouter/replica `sleep` → 实际释放确认 → manager 提交 SLEEPING/导出 placement → TaskRunner 返回元数据 |
| create | TaskRunner → Rollouter/manager `create_borrowed_replica` → replica `init_from_lease` → RUNTIME_READY → Trainer 注册并在安全边界 bootstrap → 返回明确版本 → manager/LB `commit_ready` → READY |
| reclaim | TaskRunner → Rollouter 摘流与请求收尾 → Trainer 注销并确认无在途同步 → Rollouter/manager `reclaim_replica` → replica `reclaim/destroy` → 实际释放确认 → 返回 released 元数据 |
| destroy | 先判断是否已发布；已发布的对象执行请求和 CE 退出，未发布的部分对象直接精确回滚；借用对象不删除 PG，native 删除自有 PG 前检查未归还 leases |
| wake | donor TaskRunner 核验归还结果并解除对应 lease → Rollouter/replica `wake_up` → Trainer 注册及必要的权重追平 → 确认当前 serving version → LB `commit_ready` → READY |

sleep/reclaim 开始即在任务本地标记目标状态，阻止冲突命令。物理 sleep/destroy 必须在 CE gate 内的注销确认之后；注销后其余 replica 可继续同步，不能再访问该目标。不同 Actor 的锁不被描述成跨 Actor 原子事务。

创建/回收操作耗时超过普通 RPC 期限时，可先返回带 lease 的处理中元数据，后续读取同一操作记录；不得将客户端超时转成重新创建，也不得把处理中当最终成功。结果中可以包含错误及清理状态，但不包含下层运行时句柄。

**验证：**

- U/R：核对调用顺序和失败截断；CE 注销未成功不 sleep/kill，bootstrap 未成功不 READY，shutdown 未成功不确认归还。
- G：一个完整循环 `donor sleep → borrowed create/同步/生成 → reclaim/destroy → donor wake/同步/生成`，通过生产扩展入口执行。
- 实际训练 `run()/fit()` 持续运行时仍能提交命令；donor 借出期间训练侧更新只作用于其他有效 replicas，不自动唤醒目标。
- reclaim 后 donor 的 CE/server/engine 保留原身份；borrowed 的相应对象已消失；两任务各自产生正确版本的推理结果。

**通过标准：** 五类能力有可调用、可重复、可观察的入口和完成条件；没有实际资源释放就没有成功回收结果。

### D14：验收支持范围与失败边界

**修改位置：** 必要的最小缺陷修复、`tests/gpu/test_replica_capability_failures.py`、能力说明和运行示例；不趁验收扩展调度算法。

**验证矩阵：**

| 场景 | 必须观察到的结果 |
| --- | --- |
| 同集群两个独立 jobs | 命名 PG 可解析；只有元数据跨任务；CE、IPC、HTTP 名称和请求归属隔离 |
| 单节点单卡、单节点多卡 | 逐 rank 对应预期 bundle/GPU，engine 并行配置正确，生命周期循环成功 |
| 多节点同构布局 | 原生 CE 分组及 HTTP/headless 启停均正确；远端进程退出也有证据 |
| 非同构 world size/部分 bundles | 创建前明确拒绝，不留 Actor |
| 同 lease 并发 create/reclaim | 不重复创建，取消能收尾，旧创建不会在回收后重新发布 |
| 部分 CE/server 创建失败 | 只清理本次对象，donor PG 和其他任务不受影响 |
| 参数同步任一阶段失败 | CE BLOCKED，不提前释放资源、不发布 READY；保留可诊断的参与者和错误 |
| lease 过期或 RPC 超时 | 不自动唤醒 donor；通过实际清理结果决定能否解除占用 |
| donor 正常退出且有有效 lease | 等待已借资源处理完成后才清理自有 PG |
| donor/PG 意外消失 | borrowed 停止准入并报告失效；不声称具有自动重建/继续服务能力 |
| 无在飞与有在飞回收 | 自然排空正确；强制 abort 若尚未验证则明确拒绝，不静默丢样本 |
| 连续借用/归还循环 | Actor、进程、通信资源和显存没有持续增长，下一轮能重新创建 |
| 不借卡及关闭 profile | 原生训练、参数同步和插件禁用路径保持可运行 |

首次实现可先交付已通过的单节点支持范围，但必须在运行校验中拒绝尚未验收的跨机路径；不能仅把未测范围写入文档而仍允许上线创建。

**性能记录：** 分别测量 donor drain/sleep、CE 初始化、engine 冷启动、bootstrap、有效生成、borrowed shutdown/回收和 donor wake，记录逐卡峰值显存。借用窗口不足以覆盖成本时如实说明，不把“功能正确”写成“必然提高吞吐”。

**通过标准：** 每个声称支持的场景有本次真实证据；未支持项在入口明确拒绝。只修复触发的缺陷并重跑受影响用例，最后执行一次完整能力循环及原生回归。

## 5. 功能与验收步骤对照

| 最终能力 | 底层实现 | 任务入口及验收 | 不能省略的证据 |
| --- | --- | --- | --- |
| borrowed 创建 | D01、D02、D06—D09 | D11、D13、D14 | 同 GPU/独立 runtime/正确 borrower 权重/READY 后实际生成 |
| 销毁 | D04 | D08 回滚、D13、D14 | engine/子进程/CE Actor 实际退出；borrowed 不删 donor PG |
| native sleep | D05 | D10、D12、D13 | 请求已收尾、CE 已退出、显存释放且不被自动唤醒 |
| native wake | D05 | D11、D13 | borrowed 真正释放、原实例身份保留、权重追平后服务 |
| borrowed reclaim | D04、D09、D10、D12 | D13、D14 | 摘流、同步退出、销毁、归还确认全部完成 |
| CE 动态通信域 | D10 | D11、D14 | 加入前/加入后/移除后三轮真实传输正确，无旧成员残留影响 |

## 6. 开发记录模板与阶段交付

每次完成一个步骤后，在实际开发记录中填写以下项目，不提前勾选：

```text
步骤：Dxx
状态：未开始 / 开发中 / 局部验证通过 / 已验收 / 阻塞
代码及配置改动：
测试命令与环境：
U / N / R / G 结果：
真实证据与日志位置：
失败场景及资源清理结果：
未验证范围：
允许继续的依赖步骤：
```

阶段交付按以下边界判断：

- **底层可行：** D02、D04、D05 通过真实环境验证，说明共享位置、清理和 donor 休眠具备基础条件。
- **borrowed 可创建：** D06—D09 通过，说明独立 CE/server/engine 能创建和回滚；尚不能宣称支持当前训练权重服务。
- **borrowed 可服务：** D10—D12 通过，说明通信、权重、路由与请求退出具备闭环条件。
- **生命周期可交付：** D13 及 D14 的目标支持矩阵通过，才能对外声明本轮创建、销毁、sleep、wake、reclaim 能力完成。

本轮按此计划推进能力开发；非同构资源布局、borrowed sleep/wake 缓存、强制回收的跨 replica 续推，以及更复杂的通信后端，在对应前置能力完成后另行拆分步骤。

## 7. 开发日志

### D00：建立基线与最小验收环境

**状态：开发完成，真实环境待验收。** 本步没有修改任何运行时业务类，也没有开始 D01 的 replica 字段或 placement 校验。原因是 D00 的交付物是可复现的验收入口；若先修改业务类，后续失败无法区分环境基线问题和新代码问题。

**已修改文件与目的：**

| 文件 | 修改内容 | 目的 |
| --- | --- | --- |
| `pyproject.toml` | 注册 `gpu_integration` pytest marker | 让 GPU 验收不会被默认 unit 测试隐式启动，并在 `--strict-markers` 下有明确归属 |
| `tests/gpu/__init__.py` | 新增 GPU 验收测试包说明 | 建立独立的 GPU 验收层，不导入业务运行时 |
| `tests/gpu/conftest.py` | 新增 `MT_GPU_TEST_CONFIG` JSON 配置加载、路径/资源字段校验和 `nvidia-smi` 可见 GPU 检查 | 在创建 Ray、Actor 或 replica 前失败，避免缺配置、缺模型、GPU 不足被误报为功能成功；fixture 不启动 Ray、不创建资源 |
| `tests/gpu/test_baseline_environment.py` | 新增 D0 配置完整性及验收记录可序列化测试 | 验证 native 配置、verl 源码、模型路径、GPU 数量和输出目录满足后续真实验收前提 |
| `examples/experimental_fully_async/gpu_test_config.example.json` | 新增真实环境配置模板 | 规定 D0 需要由操作者填写的 Ray namespace、原生配置、模型、节点/GPU 数和结果目录 |
| `docs/develop_step.md` | 增加开发日志和本步结果 | 记录本步边界、验证命令、未执行项和继续条件 |

**本步刻意没有做的工作：** 没有实现 `init_from_lease()`、lease、PG 解析、sleep/wake、shutdown、CE 成员变更、LB 摘流或 TaskRunner 生命周期入口；这些属于 D01 及后续步骤，当前不能提前开发。

**已执行的静态检查：**

- 检查 D00 新增文件存在、JSON 模板格式正确、pytest marker 已注册。
- 检查 `develop_step.md` 的 D00—D14 标题连续且代码围栏闭合。
- 检查当前工作区，未修改外层 verl 原生源码。

**本机运行结果：**

- `python -m pytest -q -p no:cacheprovider tests/unit`：未执行成功；当前 Windows 开发环境没有可调用的 `python` 命令，且仓内 `.venv` 不存在。
- `python -m pytest -q -p no:cacheprovider tests/gpu/test_baseline_environment.py`：同样未执行成功；缺少 Python 解释器。由于 fixture 要求显式 `MT_GPU_TEST_CONFIG`，没有配置时也应主动失败，不能记录为 skip 或通过。
- `uv run --no-project --no-cache python --version`：未执行成功；本机没有可用的 uv managed Python 安装，且默认 managed Python 目录返回权限错误。没有通过安装依赖绕过该环境问题。
- 真实 native standalone 训练、GPU 可见性、Ray 连接及后续参数同步：尚未执行。

**真实环境验证步骤：** 在已经能够运行原生 verl experimental Fully Async 的 Linux GPU 环境中执行：

```bash
cd /absolute/path/to/verl-multi-task
export MT_VERL_SOURCE_ROOT=/absolute/path/to/verl
export PYTHONPATH="$PWD/src:$MT_VERL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RAY_USAGE_STATS_ENABLED=0
export MT_PYTHON="$PWD/.venv/bin/python"
export MT_GPU_TEST_CONFIG=/absolute/path/to/d0-gpu-test-config.json

"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/unit
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/native_unit/test_native_adapters.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/integration/test_group_scheduler.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/gpu/test_baseline_environment.py
```

然后使用 `examples/experimental_fully_async/README.md` 中的同一份 native 配置，在关闭 profile 和启用 `experimental_fully_async_standalone` 两种情况下分别运行一个最小训练任务。必须记录：

1. 两个仓库的 commit、Python/Ray/verl/vLLM/PyTorch/CUDA 版本和完整配置路径；
2. GPU baseline 测试输出、`nvidia-smi -L`、Ray namespace/地址以及 driver/worker 实际导入的源码路径；
3. native 初始化、至少一次生成和一次后续参数同步的日志；
4. 任务结束后的 Ray Actor、engine 进程和 GPU 显存状态。

**D00 通过标准：** U 层测试和配置检查通过；真实 native 配置在启用/关闭 profile 下均能完成初始化、生成及至少一次后续参数同步；结果和版本证据已归档，并由用户明确确认 D00 通过。当前仅完成开发和静态检查，因缺少 Python、GPU 和真实训练环境，D01 不得开始。
