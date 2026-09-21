# verl-multi-task Agent 交接说明

本文是当前工作区的交接入口。下一位 Agent 必须先阅读本文件、[`AGENTS.md`](AGENTS.md) 和 [`docs/develop_step.md`](docs/develop_step.md)，再开始任何工作。

## 当前任务边界

当前用户授权的是“按 D 步骤逐步开发 replica 能力”。D0 已由用户确认验收通过；当前正在执行 **D1**，D2 及后续步骤仍然禁止提前开发。

因此当前硬门禁是：

1. D0 已完成真实环境验收并由用户确认通过。
2. D1 只实现创建契约、输入归一化、任务内 rank 分配、幂等记录和生命周期预留接口。
3. 不得开始 D2 的 PG 解析、Worker/server/engine 创建、borrowed runtime 或真实 GPU 资源操作。
4. D1 完成后仍需用户明确确认，才能进入 D2；每一步继续按 `develop_step.md` 的验证和日志要求执行。

## 当前状态

D0 的开发工作和真实环境验收已经完成，用户已确认通过；D1 的本地代码开发已完成，测试命令和环境限制记录在 `docs/D1_develop.md`。

D0 修改了：

- `pyproject.toml`：注册 `gpu_integration` marker。
- `tests/gpu/conftest.py`：读取 `MT_GPU_TEST_CONFIG`，检查配置路径和本机可见 GPU；不启动 Ray，不创建 Actor。
- `tests/gpu/test_baseline_environment.py`：检查 D0 基线配置和结果记录可序列化。
- `examples/experimental_fully_async/gpu_test_config.example.json`：真实 GPU 验收配置模板。
- `docs/develop_step.md`：开发步骤和 D0 开发日志。
- `src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py`：D1 创建契约、rank 分配、幂等记录和预留回执。
- `src/multi_task_scheduler/rollout/replica.py`：D1 replica 元数据和生命周期预留接口。
- `tests/unit/test_borrowed_contract.py`：D1 隔离单元测试。

D1 没有创建 Ray Actor、PlacementGroup、CE Worker、HTTP server 或 vLLM engine；D2 的实际借卡 runtime 仍未实现。

当前本机验证结果：

- JSON 模板、文档结构、代码围栏和 marker 注册检查通过。
- `python -m pytest ...` 未执行成功：当前环境没有可调用的 Python，仓内没有 `.venv`。
- `uv run ... python --version` 未执行成功：没有可用的 uv managed Python，默认目录还有权限错误。
- 没有真实 GPU、Ray、verl/vLLM 环境，因此 native standalone 初始化、生成和参数同步尚未验证。

结论必须写作：**D0 已由用户确认通过；D1 代码开发完成，等待 D1 验证和用户确认；D2 禁止开始。**

## 必读文档及用途

| 文件 | 用途 |
| --- | --- |
| `AGENTS.md` | 仓库级开发规则和当前门禁 |
| `Agent.md` | 本交接说明，面向所有后续 Agent |
| `Claude.md` | Claude 类 Agent 的简化入口，内容不得绕过本文件和 `AGENTS.md` |
| `docs/develop_step.md` | D00—D14 的唯一开发顺序、依赖、验证标准和开发日志 |
| `docs/verl_expansion.md` | verl 原生创建路径和能力扩展设计，描述的是设计目标，不表示功能已实现 |
| `docs/development-plan.md` | 之前 P1 接线阶段的历史计划，不能当作本轮 D0 通过证明 |
| `docs/architecture.md` | 之前的实体接线和组件说明，可能包含历史阶段信息，使用前要和当前代码核对 |

## 插件化接线复现步骤

下面记录 `verl` 的入口接线和 `verl-multi-task` 的下游替换链。它是复现当前插件化方案的操作说明，不等于已经实现了 replica 借卡、回收或调度能力。

### 1. 只在原生 verl 修改两个接线点

配套的原生接线提交 `a9ebd0bb` 只修改以下两个文件，共新增 15 行、删除 1 行：

1. `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`
   增加可选配置：

   ```yaml
   multitask:
     runtime:
       profile: null
   ```

   `null` 或缺失字段保留原生路径；启用时使用
   `experimental_fully_async_standalone`。不要在此处配置逐类 Python FQN，组件组合由 profile 统一选择。

2. `verl/experimental/fully_async_policy/fully_async_main.py`
   在 `main(config)` 调用 `run_ppo` 前：

   - 默认设置 `task_runner_class = FullyAsyncTaskRunner`；
   - 用 `OmegaConf.select` 检查 `multitask` 和 `multitask.runtime` 必须是 mapping 或 null；
   - 只有 `multitask.runtime.profile` 非 null 时，才延迟导入
     `multi_task_scheduler.integration.verl.runtime_profile.resolve_runtime_profile`；
   - 将解析结果作为 `run_ppo(config, task_runner_class=task_runner_class)` 的第二个参数。

   不要修改原生 `run_ppo`：它本来就接收 `task_runner_class`，负责初始化 Ray、创建该 ActorClass 并调用 `runner.run.remote(config)`。

### 2. 在插件仓库实现 profile 解析

在 `src/multi_task_scheduler/integration/verl/runtime_profile.py` 中保持依赖轻量：

1. `_select(config, path)` 读取配置并在父节点类型错误时抛出 `ProfileConfigurationError`；
2. `validate_runtime_profile(config)` 校验 profile 名称、standalone/vLLM 非 PD、checkpoint backend、并行规模和原生训练前提；
3. `resolve_runtime_profile(config)` 在 profile 关闭时返回 `None`，在启用时导入并返回 `MultiTaskFullyAsyncTaskRunner`；
4. 启用 profile 后的导入错误必须向上抛出，不能静默退回原生类冒充插件成功。

profile 解析器不能在 import 阶段启动 Ray、发现 GS 或创建 Actor。Ray 初始化由原生 `run_ppo` 完成。

### 3. 让扩展 TaskRunner 进入原生创建链

`MultiTaskFullyAsyncTaskRunner` 继承原生 `FullyAsyncTaskRunner` 的真实 Python 类。原生类已被 `@ray.remote` 包装时，先通过 `integration/verl/ray_actor.py::unwrap_native_actor_class` 读取 `__ray_actor_class__`，再对扩展类重新使用 `@ray.remote`。

扩展 TaskRunner 的复现要点：

- `run(config)` 获取或创建 GS，保存本 TaskRunner 的 GS 句柄，然后调用 `super().run(config)`；退出时解绑自己的句柄；
- `_create_trainer(config)` 只替换 ActorClass 为 `MultiTaskFullyAsyncTrainer`，其余 tokenizer、role mapping、resource pool 和 native 初始化参数保持一致；
- `_create_rollouter(config)` 只替换 ActorClass 为 `MultiTaskFullyAsyncRollouter`，并将 GS 句柄传给它；
- 不把 GS 句柄传给 Trainer、LLMServerManager、LB、Replica 或 CE Worker。GS 的直接通信边界是 TaskRunner。

### 4. 在各创建点选择扩展子类

不要复制整套训练流程；在原生方法的最小创建点重写：

| 创建点 | 插件类 | 关键做法 |
| --- | --- | --- |
| Trainer 的 `_setup_checkpoint_manager` | `MultiTaskCheckpointEngineManager` | 继承原生 `CheckpointEngineManager`，用扩展类实例替换同一创建位置；参数同步逻辑继续继承原生实现 |
| Rollouter 的 `_init_async_rollout_manager` | `MultiTaskLLMServerManager` | 继承原生 Manager，传入原生 worker group 和 GS 的间接上下文；AgentLoopManager 仍使用原生类 |
| LLM Manager 的 replica 创建 | `MultiTaskvLLMReplica` | 设置 `rollout_replica_class`，让原生 `_initialize_llm_servers` 的数量计算、placement 和启动流程继续执行 |
| LLM Manager 的 LB 创建 | `MultiTaskGlobalRequestLoadBalancer` | 只替换 Ray ActorClass，保留原生 server 地址/handle 和路由参数 |
| Replica 的 server 创建 | `MultiTaskvLLMHttpServer` | 继承原生 HTTP Server，并由 Replica 按原生方式包装为 Ray Actor |
| Replica 的 CE Worker 创建 | `MultiTaskCheckpointEngineWorker` | 继承原生 Worker；当前只作为类型选择点，不添加借卡或通信域清理逻辑 |

关键依赖 Python 的动态分派：父类初始化代码调用 `self._method()` 时，会执行子类重写的方法。因此只重写创建点即可复用原生训练、队列、请求路由、参数同步和资源初始化。

### 5. 启动和验证插件选择

在带有上述两处原生接线的 verl 根目录运行，保证 driver 和所有 Ray 节点都能导入同一份插件源码：

```bash
export PYTHONPATH="$PWD/verl-multi-task/src:$PWD"
python -m verl.experimental.fully_async_policy.fully_async_main \
  multitask.runtime.profile=experimental_fully_async_standalone \
  <其余原生模型、数据、trainer 和 rollout 参数>
```

验证顺序：先用 `profile=null` 跑原生基线，再使用同一配置启用 profile；从初始化日志和 Actor 类型确认 `MultiTaskFullyAsyncTaskRunner`、扩展 Trainer、Rollouter、LLM Manager、Replica、HTTP Server、LB 和 CE Manager/Worker 被创建。静态检查或 CPU Ray 测试不能证明 vLLM、CUDA、NCCL 或跨节点导入成功。

### 6. 版本和工作区注意事项

插件不是独立可运行的训练入口，也不会自动安装或定位 verl；它依赖外部 verl 的 Python 包和相匹配的原生 API。当前工作区外层 `D:\\verl` 检出的是 `main`（版本文件为 `0.10.0.dev`），而上述接线来自保存的 `origin/multi_task_dev` 的 `0.9.0` 兼容提交链。若要复现该接线，必须使用包含 `a9ebd0bb` 的原生分支，或先人工完成并审查到当前 verl 版本的适配；不能只复制 `verl-multi-task` 就声称插件已激活。

## 设计不变量

- 只在 `verl-multi-task` 中扩展，不修改外层 `verl` 原生类实现。
- GS 只与每个 TaskRunner 互持句柄；其他组件不持有 GS 句柄。
- 不引入 `SlotSupervisor`、`ReplicaFactory`、独立 `Coordinator` 或另一套 replica registry。
- `MultiTaskvLLMReplica` 复用 native replica 的继承结构；borrowed 不复用 donor 的 CE Worker、HTTP server、engine 或通信域。
- `MultiTaskCheckpointEngineWorker` 当前保持空子类或直接复用原生 Worker，不增加通用 `close_runtime()`/`close_transfer()`。
- CE Manager 才负责本任务 replica 投影、同步 gate 和成员变更；通信域继续复用原生 backend 流程。
- placement、lease 和操作结果只能传递元数据，不能把 ActorHandle、server handle、CE handle 或 PG 对象交给 GS。
- 不把 `RUNTIME_READY` 当成权重已经同步，也不把 `DRAINING` 当成已经回收。
- 超时、lease 过期和 RPC 失败都不能直接声明资源已释放；必须有实际清理确认。
- Mock、AST、静态检查和 CPU Ray 测试不能证明 GPU、CUDA、NCCL、vLLM 或跨 job 行为。

## 工作流程

每个 D 步骤必须按以下顺序执行：

1. 读取该步骤的目标、依赖、修改范围和通过标准。
2. 检查 `git status --short`，保留用户现有未提交文件；使用 `apply_patch` 做最小修改。
3. 只实现当前步骤，不预先添加后续步骤的空接口或业务逻辑。
4. 先运行当前步骤的 U/N/R 测试，再执行需要真实 GPU 的 G 验收。
5. 分别记录通过、失败、未执行和环境阻塞，不把未执行写成通过。
6. 在 `docs/develop_step.md` 的“开发日志”中记录修改文件、目的、命令、结果、证据和下一步门禁。
7. 等待用户明确确认本步骤通过后，才能进入下一个 D 步骤。

## D0 真实验收要求

真实 Linux GPU 环境中，先复制并填写 `examples/experimental_fully_async/gpu_test_config.example.json`，再设置：

```bash
export MT_VERL_SOURCE_ROOT=/absolute/path/to/verl
export PYTHONPATH="$PWD/src:$MT_VERL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MT_PYTHON="$PWD/.venv/bin/python"
export MT_GPU_TEST_CONFIG=/absolute/path/to/d0-gpu-test-config.json
```

然后运行：

```bash
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/unit
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/native_unit/test_native_adapters.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/integration/test_group_scheduler.py
"$MT_PYTHON" -m pytest -q -p no:cacheprovider tests/gpu/test_baseline_environment.py
```

还必须用同一份 native 配置分别验证关闭 profile 和启用 `experimental_fully_async_standalone`：完成初始化、至少一次生成和一次后续参数同步，并记录 Ray/verl/vLLM/PyTorch/CUDA 版本、实际源码路径、GPU、Actor、engine PID 和显存状态。

在 D1 测试证据交给用户并得到明确确认前，任何 Agent 都只能修复 D1 本身的问题，不能进入 D2。D0 的真实验收已由用户确认通过。

## 当前工作区和外部传输

- 当前仓库为 `D:\verl\verl-multi-task`，外层 `D:\verl` 是另一个仓库。
- 当前存在用户未提交的文档、测试和 `pyproject.toml` 修改，不能使用 reset、clean 或批量删除来“整理”工作区。
- GitHub 私有仓库 `verl_test` 的创建和推送目前没有完成；不能在交接时声称代码已经上传。
- 不要上传 `.git` 之外的本地虚拟环境、模型、缓存、GPU 配置中的私密路径或凭据。若用户之后再次要求上传，先检查忽略规则和待上传文件清单。
