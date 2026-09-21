# Claude Agent Instructions

开始工作前必须读取：

1. [`AGENTS.md`](AGENTS.md)；
2. [`Agent.md`](Agent.md)；
3. [`docs/develop_step.md`](docs/develop_step.md)。

当前 D0 已由用户确认验收通过，正在执行 D1。D1 只允许实现创建契约、claim 归一化、任务内 rank、幂等记录和生命周期预留接口；D2 及后续业务代码仍禁止提前实现。特别禁止提前修改：

- `MultiTaskvLLMReplica` 的实际 placement/lease/borrowed runtime 创建；
- PG 解析、SubRayResourcePool、CE Worker 创建；
- sleep/wake、shutdown 以及 destroy/reclaim 的实际清理；
- CE Manager 成员管理和同步 gate；
- LB 摘流、READY 和 AgentLoop 请求处理；
- TaskRunner 生命周期命令。

复现插件化接线时，先读 `Agent.md` 的“插件化接线复现步骤”。核心是：原生 verl 只在
`fully_async_ppo_trainer.yaml` 增加 `multitask.runtime.profile` 默认配置，并在
`fully_async_main.py` 根据该 profile 选择 `task_runner_class`；`run_ppo` 本身不修改。
启用后由 `runtime_profile.py` 返回扩展 TaskRunner，再通过子类重写 Trainer、Rollouter、
LLMServerManager 的创建点选择扩展 CE Manager、Replica、HTTP Server、LB 和 CE Worker。
profile 关闭时不导入插件；启用时导入/校验失败必须报错，不能静默退回原生类。

D0 包含 GPU 验收基础设施，已由用户确认通过。D1 的实现记录、测试设计和本机环境限制见 [`docs/D1_develop.md`](docs/D1_develop.md)；当前本机仍缺少可用 Python，因此不能把未执行的 D1 测试声称为通过。

修改任何文件前运行 `git status --short`，不要覆盖用户未提交内容。只使用 `apply_patch` 做最小修改，不修改外层 verl 原生代码。每个步骤完成后，必须把修改文件、目的、测试命令、真实环境结果、未验证范围和下一步门禁写入 `docs/develop_step.md`，并等待用户确认后才能继续。

设计边界保持不变：GS 只与 TaskRunner 通信；borrowed 不复用 donor 的 CE/server/engine；`MultiTaskCheckpointEngineWorker` 不增加通用清理行为；测试替身、AST 检查和 CPU Ray 不能替代 GPU/NCCL/vLLM 验证。

GitHub 仓库 `verl_test` 尚未创建或推送，不要假设外部传输已经完成。
