# Claude Agent Instructions

开始工作前必须读取：

1. [`AGENTS.md`](AGENTS.md)；
2. [`Agent.md`](Agent.md)；
3. [`docs/develop_step.md`](docs/develop_step.md)。

当前只允许 D0。D0 的状态是“开发完成，真实环境待验收”，所以不要实现 D01 或任何后续业务代码。特别禁止提前修改：

- `MultiTaskvLLMReplica` 的 placement/lease/borrowed 创建；
- PG 解析、SubRayResourcePool、CE Worker 创建；
- sleep/wake、shutdown、destroy/reclaim；
- CE Manager 成员管理和同步 gate；
- LB 摘流、READY 和 AgentLoop 请求处理；
- TaskRunner 生命周期命令。

复现插件化接线时，先读 `Agent.md` 的“插件化接线复现步骤”。核心是：原生 verl 只在
`fully_async_ppo_trainer.yaml` 增加 `multitask.runtime.profile` 默认配置，并在
`fully_async_main.py` 根据该 profile 选择 `task_runner_class`；`run_ppo` 本身不修改。
启用后由 `runtime_profile.py` 返回扩展 TaskRunner，再通过子类重写 Trainer、Rollouter、
LLMServerManager 的创建点选择扩展 CE Manager、Replica、HTTP Server、LB 和 CE Worker。
profile 关闭时不导入插件；启用时导入/校验失败必须报错，不能静默退回原生类。

D0 只包含 GPU 验收基础设施：`gpu_integration` marker、`tests/gpu/`、GPU 配置模板和开发日志。当前本机缺少 Python 虚拟环境、GPU、Ray 和真实 verl/vLLM 运行环境，不能声称 D0 已通过。

修改任何文件前运行 `git status --short`，不要覆盖用户未提交内容。只使用 `apply_patch` 做最小修改，不修改外层 verl 原生代码。每个步骤完成后，必须把修改文件、目的、测试命令、真实环境结果、未验证范围和下一步门禁写入 `docs/develop_step.md`，并等待用户确认后才能继续。

设计边界保持不变：GS 只与 TaskRunner 通信；borrowed 不复用 donor 的 CE/server/engine；`MultiTaskCheckpointEngineWorker` 不增加通用清理行为；测试替身、AST 检查和 CPU Ray 不能替代 GPU/NCCL/vLLM 验证。

GitHub 仓库 `verl_test` 尚未创建或推送，不要假设外部传输已经完成。
