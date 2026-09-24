# D4 开发记录：TaskRunner 创建入口与 LB READY

## 1. 目标和边界

D0–D4 的统一综合验收方案见 [comprehensive_acceptance_test.md](comprehensive_acceptance_test.md)。本文件只记录 D4 的实现细节和阶段脚本，不能替代综合验收。

D4 把 D2 的 borrowed runtime 创建和 D3 的 CE bootstrap 接到一个任务级命令入口：

```text
TaskRunner.execute_replica_operation(create, spec)
  -> Rollouter.create_borrowed_replica(spec)
  -> Trainer.register_replica(replica_rank)
  -> Trainer.bootstrap_replica(replica_rank)
  -> Rollouter.commit_replica_ready(replica_rank)
  -> MultiTaskGlobalRequestLoadBalancer.commit_ready()
  -> LB_READY receipt
```

本阶段只实现创建、首次参数同步和 READY 发布。生产级 donor sleep/wake、请求 drain/abort、
LB `begin_drain/commit_remove`、通信域恢复、reclaim 和 destroy 仍由后续 lifecycle 开发；
D4 的清理仅是测试辅助清理，不代表资源已经具备生产回收语义。

## 2. 修改内容

### 2.1 `MultiTaskGlobalRequestLoadBalancer`

文件：`src/multi_task_scheduler/rollout/load_balancer.py`

新增 `commit_ready(servers: dict[str, ActorHandle]) -> dict`：

1. 校验 server ID 和 handle；
2. 对已存在的同 ID/同 ActorHandle 请求幂等返回，保留原有 inflight 计数；
3. 对同 ID/不同 handle 的旧 runtime 替换请求抛错；
4. 只把新主 server 加入 `_servers` 和 `_inflight_requests`；
5. 返回 `READY`、server ID 和新增列表。

该方法没有实现摘流、请求排空或删除。原生 `add_servers()` 没有重复 READY 保护，因此不能
直接作为 D4 的提交接口。

### 2.2 `MultiTaskLLMServerManager`

文件：`src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py`

新增或扩展：

- `ready_replica_ranks`：本地已提交 LB READY 的任务内 rank 集合；
- `mark_replica_serving_version()`：同时记录 replica 和 operation 的确认版本；
- `commit_replica_ready(replica_rank)`：确认 `RUNTIME_READY`、serving version、主 server
  endpoint 后调用 LB 的 `commit_ready()`，再更新本地地址/句柄投影；
- `probe_replica_ready(replica_rank)`：D4 测试检查 HTTP server endpoint 和 LB server ID；
- `sleep_d4_test_donors(spec)`、`cleanup_d4_runtime(replica_rank)`：仅用于 main_ppo D4
  smoke 的显存释放和测试清理，生产 sleep/wake 与 commit_remove 仍是 TODO；
- `get_replica_for_ce()`：允许 `RUNTIME_READY` 和 `LB_READY` 状态被 Trainer 查询。

`commit_replica_ready()` 采用短状态转换：`RUNTIME_READY → READY_COMMITTING → LB_READY`。
LB RPC 失败时恢复到 `RUNTIME_READY` 并保留错误，不返回已发布的假回执。

### 2.3 `MultiTaskFullyAsyncRollouter`

文件：`src/multi_task_scheduler/integration/verl/experimental_fully_async/rollouter.py`

新增薄转发方法：

- `create_borrowed_replica(spec)`；
- `commit_replica_ready(replica_rank)`；
- `prepare_d4_runtime_smoke(scenario)`；
- `probe_replica_ready(replica_rank)`；
- `cleanup_d4_runtime(replica_rank)`。

Rollouter 不解析 placement、不持有 CE Manager、不持有 GS 句柄，只把命令转发给本任务
`MultiTaskLLMServerManager`。

### 2.4 `MultiTaskFullyAsyncTaskRunner`

文件：`src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py`

新增：

- Ray Actor `max_concurrency=8`，使训练入口运行时仍能接受管理命令；
- `_replica_operation_lock`，串行化同一 TaskRunner 的 replica command chain；
- `execute_replica_operation(operation, request)`：唯一任务级创建编排入口；
- `_maybe_run_d4_runtime_smoke(config)`：由 main_ppo 显式配置启用的真实 smoke。

创建命令的返回值只包含 metadata、版本、状态和错误，不包含 PG、Worker、server 或 GS
句柄。`reclaim`、`destroy` 等非创建操作返回 `LIFECYCLE_NOT_IMPLEMENTED` 且
`released=False`，避免把预留接口误报为完成。

D4 smoke 在 native training 返回后执行，这是为了让 Trainer 已经完成 checkpoint loading、
拥有稳定参数版本，再调用同一个 TaskRunner 命令入口完成 borrowed bootstrap。生产 GS 命令
可以在任务运行期间调用同一入口；main_ppo smoke 不声称验证训练中并发调度。

### 2.5 测试文件和脚本

- `tests/unit/test_d4_lifecycle.py`：覆盖 LB READY 幂等、不同 handle 冲突、TaskRunner 调用顺序
  和不支持的生命周期回执；不启动 Ray/GPU；
- `D4_test.sh`：沿用 `multi_task_run.sh` 的真实环境、模型和数据配置，通过 main_ppo 执行
  `basic/split/fragmented/cross_pg` 任一或多个场景；检查 `LB_READY`、endpoint probe、
  cleanup、严格训练完成回执以及 CE source-to-receiver 参数校验。

## 3. 一键真实测试

在服务器上与 `multi_task_run.sh` 相同的目录运行：

```bash
# 默认运行 basic
bash ../D4_test.sh

# 运行四个创建拓扑场景
D4_RUNTIME_SCENARIOS=basic,split,fragmented,cross_pg bash ../D4_test.sh
```

脚本会在 `${VERL_REPO_DIR}/logs/d4_runtime/` 保存每个场景日志。通过条件：

1. main_ppo 和 `tee` 的退出码均为 0；
2. 出现 `D4_RUNTIME_RESULT` 且 JSON 状态为 `LB_READY`；
3. `probe` 报告 endpoint 可查询且 server 已在 LB；
4. 出现 `D4_RUNTIME_CLEANUP`；
5. 非 `shared_bundle` 场景出现 `CE_PARAMETER_VALIDATION` 且包含
   `source_state=SOURCE_TO_RECEIVER_VALIDATED`；
6. 出现 `MULTITASK_TRAINING_COMPLETE` 且 `completed_steps == target_steps`。

`Traceback`、`AssertionError` 等文本只保留给故障定位；`LIFECYCLE_NOT_IMPLEMENTED` 仍表示
当前场景要求的生命周期回执没有实现，属于操作级失败。

脚本中的 `sleep_for_runtime_test()` 仅用于释放同卡 donor 的测试显存，`cleanup_d4_runtime()`
使用测试 teardown 删除路由并杀死本次创建 Actor；它们不能证明生产 sleep/wake、drain 或
claim 归还已经完成。

测试 fixture 在 donor 逐个休眠或 READY 后检查失败时执行 best-effort 唤醒/清理；如果清理
本身失败，TaskRunner 会记录 warning，脚本会因缺少 `D4_RUNTIME_CLEANUP` 或进程异常而失败，
不能把残留资源当成通过。

## 4. 分层验证

### U：本地无 GPU

```bash
uv run --no-project --no-cache python -m pytest -q -p no:cacheprovider \
  tests/unit/test_d4_lifecycle.py \
  tests/unit/test_checkpoint_membership.py \
  tests/unit/test_borrowed_runtime.py \
  tests/unit/test_borrowed_contract.py \
  tests/unit/test_wiring.py \
  tests/unit/test_hccl_checkpoint_engine.py
```

本机执行结果（不包含真实 Ray/GPU）：

```text
41 passed in 0.44s
```

这组用例同时覆盖 D4 新增的 READY 幂等和调用顺序，以及 D1-D3 的回归用例。
完整 `tests/unit` 收集时，本机环境还缺少 `omegaconf`，因此 `test_entry.py` 和
`test_runtime_profile.py` 无法收集；这属于测试环境依赖缺失，不把它记为 D4 通过。

### G：真实服务器

```bash
D4_RUNTIME_SCENARIOS=basic bash ../D4_test.sh
D4_RUNTIME_SCENARIOS=split,fragmented,cross_pg bash ../D4_test.sh
```

每个场景需要留存 native/borrowed 的 replica rank、server ID、endpoint、参数版本、Ray
Actor PID、实际 node/device 映射以及 cleanup 结果。真实 GPU 测试未执行前，不能将 D4
标记为验收通过。

## 5. 已知限制和后续 TODO

- D4 smoke 在原生训练完成后运行，未验证训练循环中 GS 命令与 rollout/optimizer 并发；
- `commit_ready()` 已实现，`begin_drain()`、`get_drain_status()`、`commit_remove()` 未实现；
- donor 的 CE suspend/resume 仍是窄接口，真实 sleep/wake 和 target-only wake sync 由其他
  lifecycle 开发负责；
- D4 cleanup 直接使用测试 teardown，不代表 `released=True` 或全局 claims 已归还；
- GS 当前只保存 TaskRunner handles，尚未实现真实 selected_slots 账务和跨任务调度策略。
- `D4_test.sh` 自身使用旧 Bash 可用语法，但它复用的 `multi_task_run.sh` 必须保持服务器上
  已验证的 Bash 版本兼容性；脚本不会替换该既有启动入口。
