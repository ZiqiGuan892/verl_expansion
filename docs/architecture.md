# 实体与初始化：完整接线，新增业务为空

状态：P1 接线代码已由用户确认；以下 AS-IS 描述当前源码，不表示完整原生运行已通过。
核对日期：2026-09-08。

源码目录为 `/Users/nyp/Documents/multi_task_verl/verl` 与同级 `verl-multi-task`。
配套 verl HEAD 是 `a9ebd0bb2354068229620b7e7a7aab2987edf864`，describe 为 `v0.9.0-5-ga9ebd0bb`。
该提交与先前配套提交 `2625ae0b` 的文件内容完全一致，以下代码行号不变。
原生对比基线是 `adc7eefa16dad75c5f7b878823d5a76eac90c7b3`；接线提交只修改 main 与主 YAML。
伴生代码/测试/已有文档提交依次为 `3aa443d`、`9a2b785`、`66e56dc`。
以下原生路径相对 verl 仓根，伴生路径相对 `src/multi_task_scheduler/`，行号对应上述源码。

当前范围是 experimental Fully Async + 纯 STANDALONE + vLLM 非 PD。
PD（Prefill/Decode disaggregation）指预填充与解码分离；当前 profile 明确拒绝该拓扑。
V1 separate_async、HYBRID rollout 和资源共享业务不属于本次接线范围。

## 1. 实体、所有者和创建点

所有 MultiTask 类都是伴生扩展，不是 verl 原有类。ActorClass 是 Ray 创建描述；创建者调用
ActorClass 的 `.remote()` 后才创建 Ray Actor，并取得 ActorHandle（远程对象句柄）。
CE 指 Checkpoint Engine 参数同步组件；LB 指 Load Balancer 请求负载均衡器。

| 实体 | 运行类型与引用所有者 | 伴生定义/创建点 |
|---|---|---|
| GroupScheduler | 全局 Ray Actor；GS 保存 TaskRunner 句柄；TaskRunner 发现并持有 GS 句柄 | `scheduler/group_scheduler.py:9`；`scheduler/discovery.py:21` 创建或获取 |
| MultiTaskFullyAsyncTaskRunner | driver 创建并持有的 Ray Actor | `integration/verl/experimental_fully_async/task_runner.py:34`；原生 `verl/trainer/main_ppo.py:91`、`:93` 创建 |
| MultiTaskFullyAsyncTrainer | TaskRunner 创建并持有的 Ray Actor | `integration/verl/experimental_fully_async/trainer.py:26`；`task_runner.py:87` 创建 |
| MultiTaskFullyAsyncRollouter | TaskRunner 创建的 Ray Actor；Trainer 也持有其句柄 | `integration/verl/experimental_fully_async/rollouter.py:20`；`task_runner.py:60` 创建 |
| MultiTaskLLMServerManager | Rollouter 内普通对象 | `integration/verl/experimental_fully_async/llm_server_manager.py:12`；`rollouter.py:34` 调用继承的 create |
| MultiTaskCheckpointEngineManager | Trainer 内普通对象 | `checkpoint/checkpoint_engine_manager.py:6`；`integration/verl/experimental_fully_async/trainer.py:34` 构造 |
| MultiTaskGlobalRequestLoadBalancer | Manager 创建的 Ray Actor；Manager/client 持有其句柄 | `rollout/load_balancer.py:6`；`integration/verl/experimental_fully_async/llm_server_manager.py:25` 包装并创建 |
| MultiTaskvLLMReplica | Manager 内普通对象；Trainer 经 RPC 取得序列化副本及其中的 ActorHandle | `rollout/replica.py:13`；原生 `verl/workers/rollout/llm_server.py:557` 构造 |
| MultiTaskCheckpointEngineWorker | Replica 通过 RayWorkerGroup 创建的 Ray Actor | `checkpoint/checkpoint_engine_worker.py:6`；`rollout/replica.py:20` 选择类；原生 `verl/workers/rollout/replica.py:217` 创建 WorkerGroup |
| MultiTaskvLLMHttpServer | Replica 创建的 Ray Actor；LB 接收 head server 句柄 | `rollout/http_server.py:6`；`rollout/replica.py:18` 选择类；原生 `verl/workers/rollout/vllm_rollout/vllm_async_server.py:1168`、`:1176` 创建 |

表中简称 `task_runner.py`、`rollouter.py` 均指伴生 `integration/verl/experimental_fully_async/` 下的文件。
Replica、Manager 和 CE Manager 都不是 Ray Actor；TaskRunner 不直接持有后两个普通对象。

原生 FullyAsyncAgentLoopManager、FullyAsyncLLMServerClient、MessageQueue、MessageQueueClient、WorkerDict 和
ServerAdapter 继续参与原生流程，伴生仓不为改名添加包装。
Rollouter 创建 AgentLoop/Client 的调用位于伴生 `rollouter.py:39`、`:41`；
TaskRunner 创建 MessageQueue/Client 的调用仍位于原生
`verl/experimental/fully_async_policy/fully_async_main.py:97`、`:98`。
CE Worker 仍在原生 `verl/checkpoint_engine/base.py:323`、`:324` 构造 ServerAdapter。

## 2. 初始化调用流程

1. 原生 main 加载 upstream Hydra primary，再执行设备设置、rollout 节点/GPU 字段映射和 reward 配置迁移。
   main 随后读取 `multitask.runtime.profile`，关闭时保留原生 FullyAsyncTaskRunner。
   证据：`verl/experimental/fully_async_policy/fully_async_main.py:222`、`:233`、`:235`、`:237`、`:238`。
2. 启用时，main 调用伴生 `resolve_runtime_profile()`。解析器验证纯 STANDALONE 前提，并返回
   MultiTaskFullyAsyncTaskRunner ActorClass；解析器不初始化 Ray、不发现 GS、不创建 Actor。
   证据：原生 `fully_async_main.py:243`；伴生 `integration/verl/runtime_profile.py:28`、`:87`。
3. 原生 `run_ppo()` 初始化 Ray，创建所选 TaskRunner，并调用其 `run.remote(config)`。
   证据：`verl/trainer/main_ppo.py:75`、`:91`、`:93`、`:94`。
4. 扩展 TaskRunner.run 创建或获取 GS，并把自己的 ActorHandle 登记到 GS；TaskRunner 随后调用
   `super().run(config)`。构造函数本身只初始化 `group_scheduler=None`。
   证据：伴生 `task_runner.py:38`、`:44`、`:48`、`:49`。
5. 原生 `_initialize_components()` 调用扩展 `_create_trainer()`。
   扩展方法保留 Role.Rollout 过滤、资源池参数和 `init_workers.remote()`，只更换 Trainer ActorClass。
   证据：原生 `fully_async_main.py:78`；伴生 `task_runner.py:78`、`:87`、`:96`。
6. 原生 TaskRunner 检查 HYBRID WorkerGroup 注入条件；profile 要求的两个 false 值使该条件不成立。
   TaskRunner 随后调用扩展 `_create_rollouter()`，传入 GS 句柄，并保留原生 Worker 初始化和样本额度设置。
   证据：原生 `fully_async_main.py:159`、`:170`、`:184`；伴生 `task_runner.py:57`、`:65`、`:72`、`:73`。
7. Rollouter 在原生时机调用 MultiTaskLLMServerManager.create。继承的 create 使用 `cls(...)` 构造
   扩展 Manager，再依次初始化 Replica 与 LB。
   证据：伴生 `rollouter.py:28`、`:34`；原生 `verl/workers/rollout/llm_server.py:511`、`:513`、`:514`、`:515`。
8. Manager 在父类构造前设置扩展 Replica 类型。原生初始化据此创建 Replica，并调用继承的
   `init_standalone()`；Replica 通过既有选择方法提供 CE Worker ActorClass，并通过 `server_class` 提供 HTTP Server ActorClass。
   原生 RayWorkerGroup 和 launch_servers 继续完成资源池、PG、bundle、节点和 GPU 绑定。
   证据：伴生 `llm_server_manager.py:18`、`rollout/replica.py:18`、`:20`；
   原生 `verl/workers/rollout/llm_server.py:557`、`:577`，`verl/workers/rollout/replica.py:217`、`:225`、`:226`。
9. Manager 将 head server 地址/句柄交给扩展 LB，并显式保留原生 `full_determinism` 参数。
   Rollouter 随后创建原生 FullyAsyncAgentLoopManager 与 FullyAsyncLLMServerClient。
   证据：伴生 `llm_server_manager.py:25`、`:28`，`rollouter.py:39`；
   原生 head server 视图来自 `verl/workers/rollout/llm_server.py:579`、`:580`。
10. 原生 TaskRunner 将 Rollouter 句柄交给 Trainer。Trainer 通过 Rollouter.get_replicas 取得 Replica 列表，
    并在 Trainer 进程内构造 MultiTaskCheckpointEngineManager。
    证据：原生 `fully_async_main.py:87`、`verl/experimental/fully_async_policy/fully_async_trainer.py:346`、`:350`；
    伴生 `trainer.py:30`、`:32`、`:34`。
11. 原生 TaskRunner 连接 MessageQueue、恢复 checkpoint、触发初始参数同步和可选验证，再启动 Trainer/Rollouter.fit。
    扩展没有改写这些方法。证据：原生 `fully_async_main.py:97`、`:102`、`:106`、`:110`、`:112`、`:190`、`:191`。

以上流程描述代码中的调用顺序。真实集群是否能完成所有构造、序列化和 GPU 初始化，仍属于
[开发计划](development-plan.md) 的 T2—T6 待验项。

## 3. 原生行为与扩展边界

三个 controller 分别保留 TaskRunner 1 CPU、Trainer 10 CPU、Rollouter 10 CPU 与 max_concurrency=100；
声明位于伴生 `task_runner.py:34`、`trainer.py:26`、`rollouter.py:20`。
`integration/verl/ray_actor.py:4` 隔离 `__ray_actor_class__` 解包机制；
该实现依赖 Ray 的类表示，不能据此宣称兼容所有 Ray/verl 版本。

Manager 继承原生资源初始化，Replica 继承原生放置与服务启动。
CE Manager、CE Worker、HTTP Server 只声明原生子类，没有新增业务方法。
LB 继承原生获取/释放 server 和路由缓存行为；Manager 的覆写只保留扩展 LB 所需的原生构造参数和 GS 句柄。

当前 GS 只维护 TaskRunner 句柄字典。原生资源池仍管理资源，Manager 仍持有原生 Replica/server 列表，
CE 仍通过 Replica/Worker 句柄同步权重；GS 不维护 node/GPU/lease 资源视图。
`GroupScheduler.schedule()` 返回空列表（`scheduler/group_scheduler.py:43`），没有调度策略或真实资源动作。

## 4. GS 生命周期与退出边界

TaskRunner、Rollouter、Manager、LB 依次持有同一个 GS 句柄；Trainer/CE 不持有 GS 句柄。

> **设计调整（尚未修改代码）：** 上句描述当前实现，但目标边界已明确为 GS 只与各任务的 TaskRunner 通信、双方互持句柄。后续应移除 Rollouter、Manager、LB 的 GS 参数与字段；推理侧调用经 TaskRunner → Rollouter 转发，CE 操作经 TaskRunner → Trainer 转发，结果沿原链路返回 TaskRunner 后回复 GS。本文仍按当前源码记录，扩展设计见 [verl_expansion.md 第 2.3 节](verl_expansion.md#23-gs-句柄边界)。

发现函数使用固定 namespace `verl-multi-task` 和 name `verl-multi-task-group-scheduler`，
通过 `get_if_exists=True` 获取同名 detached Actor，并核对实现标识。
证据：`scheduler/discovery.py:3`、`:21`、`:27`。

TaskRunner.run 的 finally 调用 GS.detach_task，移除 GS 对该任务的引用。
TaskRunner 只记录解绑异常，不用清理异常替换原生异常；任务结束不会销毁 GS。
证据：`integration/verl/experimental_fully_async/task_runner.py:50`、`:52`、`:55`。

该 finally 不提供进程崩溃、强制 kill 或 GS 不可达后的自动恢复保证。
当前代码没有心跳清理，GS 可能保留遗留句柄；集群操作者负责最终清理不再使用的 detached GS。
当前接入没有新增 TaskRunner control concurrency group，也没有中转通信 Actor。
