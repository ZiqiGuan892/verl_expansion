# verl 能力扩展：开发步骤与验收计划

本文依据 [verl能力扩展最新版.md](verl能力扩展最新版.md)，将开发流程收敛为 **5 个阶段：D0 基线、D1 创建契约、D2 runtime 创建、D3 CE 同步、D4 创建链路接通**。每阶段包含内部验证点，完成一个验证点再接下一部分，不把所有代码写完后集中调试。

本版取代旧计划的 D00—D14。D0 与历史 D00 是同一步；交接文件中的“D01 尚未获准开始”，对应本版 D1 的进入门禁。旧计划中的全量生命周期实现不再是本轮交付目标。

**当前状态：D0、D1 已由用户确认验收通过；D2 的 split、fragmented、missing_pg、cross_pg 真实场景已通过；D3 代码、单元测试和真实 main_ppo 测试入口已完成，详细记录在 [D3_develop.md](D3_develop.md)，等待服务器运行 `D3_test.sh` 留存 CE bootstrap 证据。D4 未开始。**

## 1. 本轮范围与实现边界

本轮交付 borrowed replica 的可验证创建能力：

`GS placement metadata → TaskRunner → Rollouter/LLMServerManager → 独立 CE Worker、HTTP server、engine → Trainer/CE 注册与 target-only bootstrap → LB READY`

| 项目 | 执行口径 |
| --- | --- |
| 开发位置 | 基于本仓 `src/multi_task_scheduler/` 的现有扩展类开发；沿用 `experimental_fully_async_standalone` profile，不修改外层 verl 原生类实现 |
| replica 类型 | native、borrowed 均使用 `MultiTaskvLLMReplica(vLLMReplica)`；native 尽量委托原生方法，borrowed 新增 `init_from_lease(spec)` |
| 资源放置 | 按 claim 显式指定 `(PG, bundle_index)` 创建新 CE Worker；支持跨 PG、非连续 bundle、异构 donor/borrower world_size；不新建 borrowed PG |
| 多 Worker 共用 bundle | 配置 `max_colocate_count=M`，native 建 PG 时确定 CPU 容量；borrowed 按 claim 申请 CPU/GPU fraction。已有 PG 不因修改 M 而扩容 |
| 拓扑限制 | 重建 borrower 的 rank、端口和通信环境；继承 `launch_servers()` 时保持每节点 Worker 数一致；不能从 donor 的 rank 推导 borrower 的 rank |
| GS 边界 | 只有 TaskRunner 持有 GS 句柄；全局 claim 授权和容量归属由 GS 决定。Manager 只维护本任务创建状态、校验输入与实际落点，不建立第二份全局租约表 |
| CE Worker | 直接复用原生实现，或保留已接线的空子类；不新增 Worker 通信协议或通用销毁方法 |
| 首次权重 | 在同一 rollout 窗口最早安全快照点执行 target-only bootstrap；不等待下一次周期同步，也不对全部 replica 执行一次同步来代替 bootstrap |
| 生命周期范围 | replica/manager 的 sleep、wake、reclaim、destroy，以及 server shutdown 只保留必要状态和接口；不开发完整摘流、续推、物理清理或 donor 恢复流程 |
| 新组件 | 不引入 SlotSupervisor、ReplicaFactory、Coordinator、新资源池或业务数据类；协议使用普通字典 |
| 首个同步后端 | 按最新版设计验收全量 NCCL 路径；训练端与接收端在 backend 初始化前配置 `rebuild_group=True`。其他后端、Ascend/NPU 不能据此自动算作通过 |

最新版中的创建示例使用 `M=4`，类定义给出默认值 `M=10`；实现按类定义提供配置默认值，示例通过显式配置覆盖。验收记录实际 M 和 PG 的 CPU/GPU 容量，不能仅看配置推断剩余资源。

设计中“锁定 bundle/累计 claim”在本地只表示本次请求及本任务记录的检查，不能代替 GS 的跨任务授权。创建前提是 donor 的设备已允许借用且具有足够运行空间；本轮不通过实现完整 donor sleep/reclaim 来生成这个前提。隔离环境可预先准备符合前提的 PG/claims，但必须注明这只是创建能力验收。

## 2. 五个阶段与验收门禁

| 阶段 | 本阶段交付 | 主要修改位置 | 必须拿到的验收证据 |
| --- | --- | --- | --- |
| D0：运行基线 | 原生与插件路径均可启动的基线 | 已有测试配置、fixture、开发日志 | 同一原生配置在 profile 关闭/开启时均完成初始化、生成和参数同步 |
| D1：创建契约与本地管理 | spec 校验、身份分配、幂等记录和预留接口 | `llm_server_manager.py`、`rollout/replica.py` | 非法输入无副作用；重复请求不产生重复 rank/runtime；返回值不泄漏 handles |
| D2：borrowed runtime 创建 | 显式 PG/bundle → CE Workers → HTTP server/engine，达到 RUNTIME_READY；CE 注册、bootstrap、LB 接流留空 | `rollout/replica.py`、manager、Rollouter 测试 hook、D2 runtime 脚本 | 真实放置正确；拆分、拼接、碎片化、多 claim 和失败清理可验证；borrower 不创建或删除 donor PG |
| D3：CE 注册与首次同步 | target-only bootstrap、版本确认、后续全成员同步与成员注销 | `checkpoint_engine_manager.py`、`trainer.py` | 新 replica 真正加载 borrower 权重；已有服务不被 bootstrap 中断；通信域完成 finalize |
| D4：任务入口与接流 | TaskRunner 编排创建、Rollouter 薄转发、LB READY 和最终回执 | `task_runner.py`、`rollouter.py`、manager、`load_balancer.py` | 训练期间可执行创建；新 replica 在确认权重后收到并完成请求；GS 只接收元数据 |

顺序固定为 **D0 → D1 → D2 → D3 → D4**。不并行开发尚未获用户确认的后续阶段，也不新增单独的“最终测试阶段”；对应功能的真实验证在所属阶段完成，D4 只补齐端到端连接与回归。

## 3. 各阶段具体开发与验证

### D0：确认运行基线，沿用已有工作

**目标：** 确认环境、原生版本和插件接线可用，避免把基线问题归因于新能力。

本阶段已有交付物为 `gpu_integration` marker、`tests/gpu/conftest.py`、`tests/gpu/test_baseline_environment.py` 和 GPU 配置模板，不重复创建。先补齐文末历史日志中尚未执行的真实环境验收。

**验证：**

1. 记录外层 verl 与插件仓版本、工作区状态及 driver/worker 实际导入路径；确认使用的原生版本包含必要 profile 接线。
2. 运行已有 U/N/R 测试及 GPU 配置检查。
3. 用同一最小 native 配置分别关闭/开启 profile，完成初始化、一次生成及一次后续参数同步；保留 Actor、engine 和设备状态记录。

**通过条件：** 上述结果可复现且用户确认 D0 通过。配置 JSON 可读、静态接线正确或 mocked 测试通过，均不能替代真实训练基线。

### D1：创建契约、replica 身份与本地操作记录

**本阶段实现记录：** [D1_develop.md](D1_develop.md)。代码开发已完成；本地 Windows 环境未安装可用 Python，真实测试命令需在 D0 已验收的服务器环境执行。D2 的 PG/Worker/server 创建不得提前实现。

**目标：** 先把“接收什么输入、如何识别一次创建、失败返回什么”做正确，不启动新的 GPU runtime。

**修改范围：** `integration/verl/experimental_fully_async/llm_server_manager.py`、`rollout/replica.py` 及对应单元测试。新增字段和私有辅助方法均落在现有类内。

| 顺序 | 开发内容 | 完成后立即验证 |
| --- | --- | --- |
| 输入归一化 | 在 manager 用 `_validate_create_spec(spec)` 归一化 `selected_slots → claims`，兼容设计中的 world_size 字段；只保存一份规范输入。明确主 lease、source leases、claim、PG namespace、rank 和并行配置的含义 | 检查缺字段、过期 lease、错误任务、重复 claim、rank 不完整、world_size 不匹配、不均匀节点布局、非法 CPU/GPU fraction；在创建 Actor 前明确失败 |
| 身份及幂等 | 初始化 `next_replica_rank`、`borrowed_operations`、短临界区锁和设计要求的生命周期字段；短锁内按 lease 查重，仅新请求调用 `_allocate_replica_rank_locked()` 并登记 CREATING；耗时操作在锁外 | native rank 后继续分配；失败不回退计数、不复用 rank；同 lease 相同输入复用记录，冲突输入拒绝；并发重复请求只分配一次身份 |
| 对象及返回协议 | replica 保留父类必需字段，增加 allocation_kind、lease/source leases、claims、runtime_state、serving_version；operation 保存冻结的请求、局部 runtime 引用、错误和回执 | 按稳定 replica_rank 查找，不能将 rank 当列表下标；跨组件返回 metadata receipt，GS 返回值不含 ActorHandle/PG handle |
| 生命周期预留 | 在本阶段涉及的 manager/replica 中声明 reclaim/destroy 等必要入口；sleep/wake 保留原生契约，borrowed 未实现路径明确失败 | 预留接口返回“不支持”或抛出明确异常，不返回虚假的成功、DESTROYED 或 released=True；不添加完整生命周期逻辑 |

`_validate_create_spec()` 在进入本地操作锁前完成纯输入检查；`_allocate_replica_rank_locked()` 仅在锁内确认该 lease 尚无创建记录后调用。重复请求先匹配已冻结的输入，不因新的 operation_id 再创建一次 runtime。动态变化的 PG 存活、设备及实际配额核验留到 D2。

**验收文件：** 已新增 `tests/unit/test_borrowed_contract.py`，并沿用已有 `tests/native_unit/test_native_adapters.py` 的父类兼容检查。D1 的隔离测试只验证契约、幂等和错误传播，生产入口在 D2 实现前不能假报 RUNTIME_READY。

**通过条件：** 输入、并发重复请求、身份和错误回执验证通过；不新增全局 bundle_leases 表，不启动 GPU Actor；用户确认后进入 D2。

### D2：按 claims 创建独立 borrowed runtime

**本阶段实现记录：** [D2_develop.md](D2_develop.md)。当前已完成创建代码、隔离单元测试和 main_ppo 真实测试入口；真实 Ray/GPU 矩阵由 `D2_runtime_test.sh` 执行。以下表格保留阶段级开发与验收门禁，具体方法、字段、时序和失败语义见 D2 日志。

**目标：** 由 manager 调用真实 `init_from_lease(spec)`，得到包含独立 CE Workers、HTTP server 和 engine 的 RUNTIME_READY replica。此时不向 LB 发布新 server。

**修改范围：** `rollout/replica.py`、`llm_server_manager.py`、现有 profile/config 校验和真实环境测试。HTTP 启动尽量复用 `http_server.py` 的父类；CE Worker 不增加新行为。

| 顺序 | 开发内容 | 完成后立即验证 |
| --- | --- | --- |
| native 配额与 PG 可达性 | 保持 native `init_standalone()` 和原生资源池不变；borrowed 只解析已有 PG，核对 namespace、bundle 和 donor 可达性；名称加入任务/lease/rank 区分 | 两个真实 Ray job 解析同一 donor PG；profile/native 创建仍正确；borrowed 不新建 PG |
| CE Workers | 实现 `validate_placement()` 与 `_create_workers_from_claims()`：逐 claim 使用新的 `RayClassWithInitArgs`，明确 PG/bundle、CPU/GPU 请求；生成 borrower 独立 rank/world/master 环境，再用 `RayWorkerGroup.from_detached()` 包装新 handles | 先只启动 CE Workers，读取实际 node/device 和 rank；确认数量、落点、资源份额及命名正确，borrower/donor handles 不相同 |
| server 与 engine | 按 borrower node_rank/local_rank 排序 workers，设置 world_size、nnodes、gpus_per_replica_node；调用继承的 `launch_servers()`，由原生 NodeAffinity/server 启动链创建 HTTP/headless engine；实现 `validate_runtime()` | 核对 server 节点、可见设备、engine PID、endpoint 和健康状态；失败不能发布地址为可服务路由 |
| 提交结果及失败记录 | manager 在返回前重新检查 lease/取消状态，保存本地 replica 和实际映射，返回 RUNTIME_READY；逐次记录已创建资源，异常时保留精确句柄/Actor 名称与错误 | PG 消失、放置超时、设备不符、部分 Worker/engine 启动失败时，不进入 READY、不误删 donor PG、不谎报 claims 已归还 |

**真实环境矩阵：**

| 场景 | 必须观察到的结果 |
| --- | --- |
| 单 PG 基础创建 | borrower 使用指定 bundles；PG 总数不因 borrower 新建而增加 |
| 一拆二：donor 4 → borrower 2 + 2 | 两个新 replica 各有自己的 world_size=2、CE Workers、server/engine 和名称；使用各自指定的 claims |
| 二合一：donor 2 + 2 → borrower 4 | 一个新 replica 的 4 个 Worker 来自两个 PG；rank 连续重新编号，不沿用 donor rank |
| 碎片化 | 显式选择非连续 index，例如 PG-A 的 1、3 与 PG-B 的 0、2；实际放置逐条匹配 |
| 同 bundle 多 CE Worker | 选择 M>2 且资源足够的 PG，验证同 bundle 的多个独立 claim（含 donor 常驻占用）能创建至少 3 个 CE Worker；超出授权/配额时拒绝或有界失败 |
| 跨节点均匀布局 | 使用每节点相同 Worker 数的 spec，验证 server 分组与跨节点 engine 启动；不均匀布局必须在启动前拒绝 |

多 CE Worker 的 fractional 资源分配只证明 Ray 放置能力，不证明显存隔离，也不能推出同一 vLLM 并行组的多个 rank 可重复使用一张物理卡。共卡 engine 必须另有足够显存、独立端口/IPC，并以真实运行结果确认；不能把同卡多个 CE Worker 当成多张物理卡。

**验收文件：** 已新增 `tests/unit/test_borrowed_runtime.py` 和 `D2_runtime_test.sh`，扩充 D1 校验用例及已有 native 适配测试。真实脚本使用 native 初始化后读取的 donor PG；不能把仍在运行的 donor engine 部分 rank 直接拿走。

**通过条件：** 对矩阵分别记录实际 Actor/device/PG 映射和成功或失败证据；`D2_runtime_test.sh` 至少完成一个成功场景和一个失败场景；CE 注册、bootstrap、LB 接流不属于本阶段。创建失败只要求可定位、不可接流和释放状态真实；完整自动回收不属于 D2。用户确认后进入 D3。

### D3：CE 成员管理、target-only bootstrap 与通信域验证

**本阶段实现记录：** [D3_develop.md](D3_develop.md)。D3 已完成本地实现和无 GPU 单元验证；真实服务器验收使用 `D3_test.sh`，在用户执行前不将 D3 标记为验收通过。

**目标：** RUNTIME_READY replica 在当前 rollout 窗口加载一次稳定的 borrower 权重，获得可验证的 serving version，后续可参加普通同步。

**修改范围：** `checkpoint/checkpoint_engine_manager.py`、`integration/verl/experimental_fully_async/trainer.py`，以及必要的 replica 投影查询。传输协议、Worker 和 ServerAdapter 继续复用原生实现。

| 顺序 | 开发内容 | 完成后立即验证 |
| --- | --- | --- |
| 成员管理 | CE manager 增加 sync_gate、sync_state、pending_bootstrap、last_synced_versions、inflight_replicas；实现 register/unregister；Trainer 按 replica_rank 从 Rollouter 取得当前投影后委托 CE | 重复注册幂等；同 rank 不同 runtime 拒绝；首次注册不写确认版本；pending 不进入普通同步；注销删除版本而不销毁 runtime |
| 稳定参数边界 | Trainer 实现 `bootstrap_replica(replica_rank)`，在 parameter_snapshot_gate 内读取一次 current_param_version；实际写参数的训练路径也遵守该边界，再传冻结的 snapshot_version 给 CE | 正在执行的 optimizer 操作完成前不能读取不稳定权重；不能只给版本读取加锁而允许参数继续被修改；统一先快照 gate、后 CE gate 的锁顺序，验证无死锁 |
| 目标同步 | CE `bootstrap_replica(replica, snapshot_version)` 仅固定 borrower 训练 workers 和目标接收 workers，复用 prepare/build_topology/init_process_group、发送/接收、ServerAdapter 加载、finalize | 已有 native/borrowed 接收端不加入本次目标组，不被全量 abort；engine 加载及 finalize 成功后才写确认版本、清除 pending，并返回结果 |
| 普通同步及异常 | update_weights 在整个同步期间持有 CE gate，固定已 bootstrap 成员快照；成员变化后按实际拓扑建组；失败进入 BLOCKED，保留失败快照并使不确定的版本确认失效 | bootstrap → 全成员同步 → 注销目标 → 下一轮同步连续成功；注册/注销不能与进行中的同步交错；finalize 失败不能接流 |

训练端、已有 native 接收端和新 borrowed 接收端的 backend 均须在初始化前应用 `rebuild_group=True`，按任务/操作隔离通信域名称。不能等新成员加入时才更改其中一端。正常 finalize 已成功时，unregister 不重复清理；backend 或 Actor 长期资源的故障恢复不在本阶段扩展为完整 destroy 流程。

**验收文件：** 已新增 `tests/unit/test_checkpoint_membership.py` 和 `D3_test.sh`。本地隔离测试覆盖注册幂等、pending 过滤、target-only 调用顺序、版本记录、普通同步和注销；真实验证必须执行 `D3_test.sh`，并结合 CE/engine 日志确认实际加载结果，不能只断言版本字典被赋值。

**通过条件：** 冻结版本与实际加载一致，target-only 与后续普通同步都完成，旧成员服务未被 bootstrap 全量中断，失败不产生 READY。服务器日志必须同时出现 `D3_BOOTSTRAP_RESULT` 的 `WEIGHTS_READY` 和 `D3_NORMAL_SYNC_RESULT` 的 `FULL_SYNC_READY`；记录从 RUNTIME_READY 到权重确认的耗时，证明可在测试 rollout 窗口内完成；用户确认后进入 D4。

### D4：TaskRunner 入口、LB 接流与端到端验收

**目标：** 将已分别验证的 runtime 和 CE 能力接成完整的创建调用链。既有睡眠、回收、销毁入口仍保持预留状态。

**修改范围：** `task_runner.py`、`rollouter.py`、`llm_server_manager.py`、`rollout/load_balancer.py` 及已有 GS 的任务注册/命令边界；不实现全局公平调度算法。

| 顺序 | 开发内容 | 完成后立即验证 |
| --- | --- | --- |
| 句柄和并发边界 | 清除 Rollouter、manager、LB 的 GS 构造参数和成员，只由 TaskRunner 注册/注销 GS；实现训练 run 期间仍可执行管理方法的并发入口 | 真实 CPU Ray 验证长时间 run 不阻塞管理请求；组件未初始化时明确拒绝；训练异常/退出仍注销；非 TaskRunner 组件不持有 GS |
| 创建调用链 | TaskRunner 的 execute_replica_operation(create, request) 调 Rollouter 薄转发 → manager 返回 RUNTIME_READY → Trainer register → bootstrap → Rollouter/manager 提交 READY | Rollouter 不解析 PG、不直接操作 CE；中间回执与最终成功明确区分；同 lease 的重复命令不会重复创建或重复启动并发 bootstrap |
| LB 提交 | 在现有 Rollouter/manager 内补齐任务内部的 READY 提交入口，将 CE 确认结果显式写回本地 serving_version；校验 lease、健康和版本后调用 LB commit_ready，仅发布主 server，再更新可服务并发额度 | bootstrap 前不可获取新 server；bootstrap 后请求能到达新 engine；重复 READY 不重置已有计数/粘性路由，也不把 headless server 加入 HTTP 路由 |
| 结果及失败 | TaskRunner 向 GS 返回可序列化 receipt；GS 不获取 replica/CE/server/PG handles；在 runtime、bootstrap、LB 各边界注入失败或 lease 失效 | 任一前置失败不发布 READY、不增加并发；LB 提交结果不确定时核对路由实际状态再重试/报告，不能仅凭 RPC 超时宣告成功或失败；未确认清理时 released=False |

本阶段补齐 TaskRunner/Rollouter/server 中设计要求的非创建预留接口；不借此实现 begin_drain、请求迁移、完整 commit_remove 或 sleep/reclaim/destroy 编排。CE unregister 已在 D3 验证，但它不代表物理资源已经回收。

**验收文件：** 计划新增 `tests/integration/test_replica_command_entry.py`、`tests/gpu/test_borrowed_create_e2e.py`，按需要扩充已有 wiring/LB 单元测试。

**真实验收：** 在隔离 Ray 集群中保持 donor PG owner 存活，由 GS 测试入口向 borrower TaskRunner 下发明确 spec；观察完整创建、当前窗口 bootstrap、新 server 接收请求、随后一次普通同步。复用 D2 已验证的 placement 场景，不重复实现测试专用创建器。对重复命令、租约失效和局部失败分别留存证据；结束后核对并清理本次测试拥有的资源。

**通过条件：** GS → TaskRunner → runtime → CE → LB 的创建链路可重复运行，普通训练和已有 replica 服务保持正确，故障不被误报成功。用户确认 D4 后，本轮“创建能力及生命周期预留接口”完成；完整 sleep/wake/reclaim/destroy 需另行设计、授权和验收。

## 4. 验证环境、命令与证据

### 验证分层

| 层次 | 能证明什么 | 不能代替什么 |
| --- | --- | --- |
| U：单元测试 | 输入、幂等、状态、错误传播；替身必须明确标注 | 真实 Ray、设备放置与 engine 行为 |
| N：真实父类适配 | 当前 verl API、原生委托、序列化兼容 | 未实际启动的 GPU runtime |
| R：CPU Ray | 跨进程调用、管理入口并发、句柄边界 | GPU 份额、显存和通信域 |
| G：真实 GPU 运行 | PG/bundle 放置、engine、权重同步和生成 | 未执行的多卡、跨节点或 NPU 场景 |

D3—D4 表内新增测试路径仍是**待开发的验收文件**；D2 已有 `D2_runtime_test.sh`，但必须在目标服务器执行后才能形成真实 runtime 证据。已有 `tests/gpu/` 和 marker 来自 D0。显式选择的真实验收缺配置或硬件时，记录环境阻塞，不能以 skip 或 mocked 结果算作通过。

### 命令约定

从 `verl-multi-task` 仓根执行，使用 `uv` 管理项目虚拟环境。真实验收沿用已能运行原生训练的 Linux 环境；以下路径必须换成实际路径：

```bash
export MT_VERL_SOURCE_ROOT=/absolute/path/to/compatible-verl
export PYTHONPATH="$PWD/src:$MT_VERL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RAY_USAGE_STATS_ENABLED=0
export MT_PYTHON="$PWD/.venv/bin/python"
export MT_GPU_TEST_CONFIG=/absolute/path/to/gpu-test-config.json

# 已存在的基线测试，按所需依赖和测试层次分别执行。
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/unit
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/native_unit/test_native_adapters.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/integration/test_group_scheduler.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/gpu/test_baseline_environment.py

# 以下是调用模板；相应阶段实现并创建文件后，替换为实际验收文件。
"$MT_PYTHON" -m pytest -q -p no:cacheprovider <当前阶段的测试文件>
```

GPU 配置填写 namespace、Ray 地址、原生配置和模型路径、节点/卡数、超时、日志目录；D2 起按场景补充 PG/claim 与 borrower 拓扑。配置和测试准备不能代替生产实现，尤其不能在测试中另写创建器绕过待验收类。

每个内部验证点只运行相关测试，通过后接下一部分；阶段结束再做受影响的 native 回归。不安装无关依赖，不为了文档修订启动训练。

### 真实环境证据与测试清理

记录两个仓库版本和工作区状态、完整命令/配置、依赖版本、实际导入路径，以及每 rank 的 PG/bundle、node/device UUID、Actor ID、engine PID、参数版本与耗时。失败需标出停在哪个边界、哪些资源仍存在。

完整生命周期尚未实现，测试环境必须有独立的清理步骤和清理确认记录。测试夹具只清理自己创建且确认归属的 Actor/engine；PG 由其测试 owner 最后清理。不能只看到 Ray Actor 退出就假设 engine 子进程与显存已释放，也不能把测试人工清理记为生产 reclaim/destroy 已完成。

## 5. 开发日志与当前进度

每个阶段只维护一条主日志，内部验证结果追加在该条下面。记录“修改了什么、为什么、怎么验证”，不拆成十几个新的 D 步骤。

**后续日志模板：**

- 阶段及状态：未开始 / 开发中 / 局部验证通过 / 真实环境待验收 / 用户已确认通过。
- 修改文件与目的：逐文件说明行为变化、复用原生部分及影响范围。
- 实际验证：环境、完整命令、U/N/R/G 层次、结果和证据路径；未执行项单列。
- 失败与限制：已知问题、保留资源、恢复/测试清理结果。
- 用户确认：确认时间或原话；没有确认则不得进入下一阶段。

**本次计划修订：** 以最新版设计替换旧 D00—D14，合并为 D0—D4；加入异构 world_size、碎片化 bundle、多 claim 和 target-only bootstrap 的阶段验收；将完整生命周期实现移出本轮。仅修改本文，不修改代码、不执行运行时测试、不将任何阶段标记为新通过。

以下 D00 日志原文保留，其中旧编号、环境描述和检查结果均为当时记录，不代表本次重新验证；D00 即本版 D0。

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

```
"multitask.runtime.profile=experimental_fully_async_standalone"
```
