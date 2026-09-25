# D0–D4 真实 E2E 验收开发记录

日期：2026-09-25。状态：同机正例测试夹具与严格回执判定已实现，本机未运行真实 NPU
验收。本文记录实现和验证方法；场景完整门槛见
[综合验收测试设计](comprehensive_acceptance_test.md)。

## 1. 本次解决的问题

旧 `D4_test.sh` 在训练结束后创建 borrowed runtime，检查 endpoint/LB，再进行测试清理。
这能验证 D4 创建链路，但没有证明 borrowed 实际生产 token、参与训练期普通参数同步、
使用新版本再次生成。因此旧综合脚本将 S1–S5、S7–S9 固定记为 `INCOMPLETE`。

新实现通过 `+multitask.e2e_test.enabled=true` 在原生 fit 前创建 borrowed runtime，
保留它直到真实训练及普通 CE 同步结束，再核验新版本生成和测试清理。原训练入口、
profile、模型、optimizer、CE 传输和 LB 保持使用既有实现，没有新建训练入口。

`D4_test.sh` 默认仍执行旧 smoke；`D4_E2E_TEST=1` 才切换 E2E，同时关闭旧
`multitask.d4_runtime_test`。综合脚本自动为 S1–S5、S7–S9、S16 打开此开关。
这些正例现在按真实回执返回 `PASS` 或 `FAIL`，没有硬编码 `PASS`。
E2E 默认执行两个真实 training step，旧 smoke 默认仍为一步；可用
`D4_TOTAL_TRAINING_STEPS` 调整，脚本按训练步数计算所需 rollout prompts。

## 2. 修改位置与职责

| 文件 | 本次职责 |
| --- | --- |
| `integration/verl/experimental_fully_async/task_runner.py` | 在 opt-in E2E 模式包装继承的原生 fit；与旧 D2/D3/D4 smoke 互斥；复用已有 create 命令 |
| `integration/verl/experimental_fully_async/rollouter.py` | 测试 action 的薄转发；runtime、LB 和 donor 句柄继续保留在所属组件 |
| `integration/verl/experimental_fully_async/trainer.py` | 原生普通同步成功后采集真实 effective rank、确认版本和 manifest；提供测试状态及恢复入口 |
| `rollout/http_server.py` | 默认关闭的真实生成审计；启用后记录完成请求、token、版本、失败和 inflight，不伪造输出 |
| `testing/e2e_runtime.py` | 构造同任务真实 donor 拓扑、编排前后静止窗口、调用原生 fit、恢复 donor 并汇总最终回执 |
| `testing/e2e_generation.py` | 通过原生 `FullyAsyncLLMServerClient` 真实生成，核对实际 acquired server、token 和版本，恢复测试前路由 |
| `testing/e2e_cleanup.py` | 捕获测试所属 Actor/engine 进程身份，执行有限范围清理并独立核验 RPC、进程和端口；失败保留诊断 |
| `testing/e2e_verdict.py` | 标准库校验器；校验单一完整回执及其内部对应关系；CLI 不依赖 Ray/verl 导入 |
| `D4_test.sh` | 保留默认 smoke；添加 E2E 开关、pressure 场景、真实进程/tee 状态及回执校验 |
| `D0_D4_comprehensive_test.sh` | 将可执行正例映射到 E2E；保留负例现状，明确 S6/S15 阻塞原因 |

表中 Python 路径均相对于 `src/multi_task_scheduler/`。相应单元测试位于
`tests/unit/test_e2e_runtime.py`、`test_e2e_generation.py`、`test_e2e_cleanup.py`、
`test_e2e_verdict.py`；
替身和合成回执只验证测试设施本身，不能作为设备验收结果。

## 3. 真实执行顺序

1. 原生初始化完成后，从 native Worker 快照构造测试 spec，并核对 donor 拓扑。
2. 在静止窗口先暂停 donor CE membership，再移除原 native LB 路由，随后使用已有
   测试 sleep helper 释放 donor engine 的借用空间。
3. 通过现有 TaskRunner create 入口创建独立 borrowed Worker/server/engine，完成
   target-only bootstrap 与 LB READY。记录首次参数版本和实际 placement。
4. 逐 borrowed rank 通过原生客户端生成，检查非空真实 token、实际 server、首次版本。
   探测结束恢复原有路由，并确认 LB inflight 归零。
5. 清零并启用训练期 HTTP 审计，委托原生 fit。只有原生 optimizer 后普通 CE 同步成功、
   finalize 完成、effective rank 的 `last_synced_versions` 一致，才记录普通同步回执。
6. 训练完成后确认 `completed_steps==target_steps` 且参数版本推进；收集每个 active
   borrowed rank 的非空训练请求审计，再按最终版本真实生成。
7. 注销 borrowed CE、删除测试路由、清理测试所属 runtime，核验 Actor RPC、Actor
   进程、已捕获 engine 子进程和 HTTP 端口。核对 donor PG/Worker/server 保留。
8. 唤醒 donor，执行真实最新版本 CE 同步，恢复 donor 路由并实际生成。全部事实通过
   校验后，输出唯一 `D0_D4_E2E_RESULT`。

`normal_syncs[].origin=optimizer_loop` 才能证明训练后的普通同步。S5 准备阶段的
`test_explicit_normal_sync` 可记录真实同步，但不能替代 B 在训练后的新版本同步。
版本来自真实同步确认及生成输出，不以手工写版本号代替权重传输。

## 4. 场景差异

| 综合场景 | D4 场景名 | 本次真实 E2E 要求 |
| --- | --- | --- |
| S1 | `basic` | 单 PG borrowed 完成前后生成、训练、普通同步和清理 |
| S2 | `split` | 一个四卡 donor 拆为两份独立 `[2,2]` borrowed；两者同时保留并各自提供完整证据 |
| S3 | `cross_pg` | 两个 TP=2 donor 的全部四个 claims 合并为一个 TP=4 borrower；双 PG 与实际四 Worker 一致 |
| S4 | `fragmented` | 非连续 bundle claims 与实际 node/device 对应，完成完整顺序 |
| S5 | `shared_bundle` | 同 bundle 两个 TP=1 runtime 串行激活 A→B→A；详见下文 |
| S7 | `merge_world_size` | donor `[2,2]` 与 borrower `[4]`，完整生成及普通同步 |
| S8 | `idempotent` | 同 spec/lease 串行 create 两次，仍为同 rank/server 和一套 runtime，再执行完整顺序 |
| S9 | `concurrent_idempotent` | 两个线程并发相同 create，仍为同 rank/server 和一套 runtime，再执行完整顺序 |
| S16 | `pressure` | 训练前后分别并发启动至少四个真实请求；检查独立 ID、server、token、版本和恢复 |

S5 在 A 首次 bootstrap/生成/当前版本普通同步后，将 A 从 CE 和 LB 暂停，再创建 B。
B 完成首次生成、真实训练、optimizer 后普通同步和最终生成后暂停；A 唤醒并重新
bootstrap 到最终参数版本，再真实生成。回执中的 `activation_order=[A,B,A]`、
`training_replica_ranks=[B]` 必须对应此过程。A/B 都必须有前后生成和清理证据；
训练审计及训练后普通同步只要求 active B。两个同设备 runtime 不同时进入 HCCL
effective set，不把同时接流或显存隔离作为本测试已证明的能力。

S6 需要多节点真实环境和跨节点 fixture，仍为 `BLOCKED`。S15 需要独立 donor/borrower
Task、GS 授权和公平性负载，仍为 `BLOCKED`。S12–S14 的可控 engine/CE/LB 故障注入
保持原状态；S10/S11 继续使用已有 D2 negative 路径。

## 5. 回执与失败判定

成功日志必须含恰好一条 `D0_D4_E2E_RESULT ` 后的一行 JSON，`schema_version=1`。
校验器同时检查选定场景、真实训练步数和最终版本、拓扑、逐 rank bootstrap、普通同步
映射及 source/receiver manifest、逐请求 token/版本、训练审计和清理/恢复事实。
完整字段说明见综合验收文档第 7 节。

缺少任一训练期 borrowed rank、新版本同步不是 optimizer 路径、manifest Worker 数
对不上、请求来自其他 server、token 为空、请求 ID 被复用、版本未推进、清理或 donor
恢复未确认，都会失败。重复 marker、拼接多个运行、损坏 JSON 或实际进程非零退出也
不能通过。`state=PASSED`、`LB_READY`、没有 traceback 等单独标记不构成成功证据。

脚本通过 `PIPESTATUS` 分别检查训练进程和 `tee`，兼容现有 Bash 用法。回执校验器采用
绝对源码路径调用；输出 `E2E PASS/FAIL`，不会再输出一份最终 marker。

## 6. 验证命令与当前结果

在插件仓库、已按项目要求由 uv 准备的环境中运行本地测试：

```bash
uv run --no-sync python -m pytest -q -p no:cacheprovider \
  tests/unit/test_e2e_runtime.py \
  tests/unit/test_e2e_generation.py \
  tests/unit/test_e2e_cleanup.py \
  tests/unit/test_e2e_verdict.py
bash -n D4_test.sh D0_D4_comprehensive_test.sh
```

已执行的整合回归命令（Windows 仓内 `.venv`）：

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.venv\Scripts\python.exe -m pytest tests/unit --ignore=tests/unit/test_entry.py -q
& 'C:/Program Files/Git/bin/bash.exe' -n D4_test.sh D0_D4_comprehensive_test.sh D0_D4_batch_test.sh
```

整合回归结果为 **252 passed in 1.38s**；三个脚本语法检查退出码 0，修改范围的 `compileall`
及 `git diff --check` 通过。新增测试覆盖真实路径的替身编排、生成失败及路由恢复、
有归属的进程清理和独立观察、单条完整回执、错版本/漏 rank/不完整 manifest、清理和
donor 恢复失败。现有 `test_wiring.py` 的父类替身补齐 config/manager 字段，以验证当前
继承接线；这不是改变生产训练逻辑。

首次全目录运行因未配置 native source root 而失败；整合命令明确排除了
`test_entry.py` 的 16 项测试。它们依赖 `MT_VERL_SOURCE_ROOT` 及指定旧原生接线提交
的一致性，本轮没有准备并验证该 checkout。已有测试依赖 `omegaconf` 通过 uv 安装。
上述通过结果属于本地单元/替身层，不包含真实 Ray/NPU 训练，也不是 native entry 验收。

目标服务器沿用现有模型、数据、Python、profile 和源码布局，从已能运行
`multi_task_run.sh` 的目录执行：

```bash
# 每次一个独立场景，自动开启 E2E
D0_D4_SCENARIOS=S1 bash ../D0_D4_comprehensive_test.sh
D0_D4_SCENARIOS=S5 bash ../D0_D4_comprehensive_test.sh
D0_D4_SCENARIOS=S16 bash ../D0_D4_comprehensive_test.sh

# 同机正例子集，逐项串行；保留每项日志和汇总
D0_D4_BATCH_SCENARIOS=S1,S2,S3,S4,S5,S7,S8,S9,S16 \
  bash ../D0_D4_batch_test.sh

# 直接复用 D4 的同一 E2E 模式
D4_E2E_TEST=1 D4_RUNTIME_SCENARIOS=split bash ../D4_test.sh

# 默认仍为旧 smoke
D4_RUNTIME_SCENARIOS=basic bash ../D4_test.sh
```

服务器脚本输出的 `environment.json`、场景日志和 `summary.json` 是本次实际运行证据。
单场景或子集 `PASS` 只适用于已执行范围。完整 S0–S16 还包含阻塞场景，不能据同机
正例子集推导完整验收已通过。本机没有 NPU/兼容 native runtime，未执行以上设备命令。

## 7. 测试清理和生产边界

测试夹具只在自有的静止窗口改动已有 CE/LB 投影，使用已存在的测试 sleep/wake helper。
清理只针对已确认归属的 borrowed Actor 及捕获 PID、启动时间的 HTTP engine 子进程；
不执行全局杀进程，不清除 donor PG，不返回 GS claim 归还成功。

Actor 退出、子进程执行结束、HTTP 端口关闭分别核验；RPC 超时、节点不可达和端口探测
超时不算释放证据。Linux 已退出但未被父进程回收的 zombie 单列
`unreaped_processes`，不伪报为 PID 消失；进程表回收与执行资源释放的语义分开记录。
donor 最新参数同步和恢复后的真实生成进一步验证资源可重新使用。本轮没有
`npu-smi` 逐卡显存回到基线的量化审计，不把进程退出观察表述成显存数值检查。

这里的恢复和 teardown 是 opt-in 验收设施，不能据此宣布生产 sleep/wake、drain、
reclaim、destroy、跨 Task 生命周期或公平性策略已实现。
