#!/usr/bin/env bash
#
# D2 真实 borrowed replica 创建验收入口。
#
# 该脚本通过 multi_task_run.sh 进入 verl.experimental.fully_async_policy
# 的 main_ppo 主入口，在 native replica 初始化完成后启用插件内的 D2 测试
# hook。每次只运行一个 scenario，成功创建后清理 borrowed Actor，避免当前
# 尚未实现 reclaim/destroy 时多个 vLLM Engine 累积占用显存。
#
# 默认场景：
#   split         一个 native replica 拆成 world_size=2 的 borrowed replica
#   fragmented   同一个 PG 选取非连续 bundle
#   missing_pg    不存在的 PG，预期返回失败且不删除 donor PG
#
# 可选场景：basic、cross_pg、duplicate_device、expired。
# 多个场景用逗号分隔，例如：
#   D2_RUNTIME_SCENARIOS=split,fragmented,missing_pg bash ../D2_runtime_test.sh
#
# borrowed 只验证到 RUNTIME_READY，不执行其 CE 注册、bootstrap 或 LB 接流。
# 测试 hook 返回后主入口仍会执行 native rollout/训练，所以训练 batch 必须合法。
set -eu
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VERL_REPO_DIR="${VERL_REPO_DIR:-${SCRIPT_DIR}}"
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}"
export VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT:-${VERL_SOURCE_ROOT}/multi_task_verl}"

[ -f "${SCRIPT_DIR}/multi_task_run.sh" ] || {
    echo "未找到 multi_task_run.sh：${SCRIPT_DIR}/multi_task_run.sh" >&2
    exit 1
}

D2_RUNTIME_SCENARIOS="${D2_RUNTIME_SCENARIOS:-split,fragmented,missing_pg}"
D2_RUNTIME_LOG_DIR="${D2_RUNTIME_LOG_DIR:-${VERL_REPO_DIR}/logs/d2_runtime}"
mkdir -p "${D2_RUNTIME_LOG_DIR}"

# 缩短运行步数，但保留 multi_task_run.sh 已验证的 batch=8、rollout.n=2。
# 不能把两者都缩为 1：4 个训练 DP rank 至少需要 4 条且可均分的序列。
# 具体模型、数据和 Ascend/vLLM 环境仍由 multi_task_run.sh 的已有变量控制。
export TRAIN_TOTAL_EPOCHS="${D2_TRAIN_TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${D2_TOTAL_TRAINING_STEPS:-1}"
export RESPONSES_PER_PROMPT="${D2_RESPONSES_PER_PROMPT:-2}"
export RESPONSES_PER_PROMPT_VAL="${D2_RESPONSES_PER_PROMPT_VAL:-1}"
export PPO_MINI_BATCH_SIZE="${D2_PPO_MINI_BATCH_SIZE:-8}"
export ASYNC_TRIGGER_SYNC_STEP="${D2_ASYNC_TRIGGER_SYNC_STEP:-1}"
export ASYNC_REQUIRE_BATCHES="${D2_ASYNC_REQUIRE_BATCHES:-1}"
export LOG_DIR="${D2_RUNTIME_LOG_DIR}"

# 此检查对应 multi_task_run.sh 的 FSDP2 配置：训练卡数就是 DP 数。
export TRAIN_NODES="${TRAIN_NODES:-1}"
export TRAIN_NPUS_PER_NODE="${TRAIN_NPUS_PER_NODE:-4}"
for count in "${PPO_MINI_BATCH_SIZE}" "${RESPONSES_PER_PROMPT}" "${ASYNC_REQUIRE_BATCHES}" \
    "${ASYNC_TRIGGER_SYNC_STEP}" "${TOTAL_TRAINING_STEPS}" "${TRAIN_NODES}" "${TRAIN_NPUS_PER_NODE}"; do
    if ! [ "${count}" -gt 0 ] 2>/dev/null; then
        echo "D2 的 batch、响应数、步数和训练卡数必须是正整数，实际值：${count}" >&2
        exit 1
    fi
done
D2_TRAIN_DP_SIZE=$((TRAIN_NODES * TRAIN_NPUS_PER_NODE))
D2_MINIBATCH_SEQUENCES=$((PPO_MINI_BATCH_SIZE * RESPONSES_PER_PROMPT))
if [ "${D2_MINIBATCH_SEQUENCES}" -lt "${D2_TRAIN_DP_SIZE}" ] || \
    [ "$((D2_MINIBATCH_SEQUENCES % D2_TRAIN_DP_SIZE))" -ne 0 ]; then
    echo "D2 batch 配置错误：mini_batch*n=${D2_MINIBATCH_SEQUENCES} 必须 >= DP=${D2_TRAIN_DP_SIZE} 且能被整除。" >&2
    exit 1
fi

# FullyAsync 队列按 prompt 计数，一条 RolloutSample 含 n 条响应。
# 每次训练收集 mini_batch*require_batches 个 prompt；同步周期还包含 trigger 个训练步。
D2_REQUIRED_PROMPTS=$((PPO_MINI_BATCH_SIZE * ASYNC_REQUIRE_BATCHES * ASYNC_TRIGGER_SYNC_STEP * TOTAL_TRAINING_STEPS))
export TOTAL_ROLLOUT_STEPS="${D2_TOTAL_ROLLOUT_STEPS:-${D2_REQUIRED_PROMPTS}}"
if ! [ "${TOTAL_ROLLOUT_STEPS}" -ge "${D2_REQUIRED_PROMPTS}" ] 2>/dev/null; then
    echo "D2_TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS} 不足，至少需要 ${D2_REQUIRED_PROMPTS} 个 prompt。" >&2
    exit 1
fi

case "${D2_RUNTIME_SCENARIOS}" in
    *[!a-zA-Z0-9_,]*)
        echo "D2_RUNTIME_SCENARIOS 只允许使用字母、数字、下划线和逗号。" >&2
        exit 1
        ;;
esac

for scenario in $(printf '%s' "${D2_RUNTIME_SCENARIOS}" | tr ',' ' '); do
    [ -n "${scenario}" ] || continue
    case "${scenario}" in
        basic|split|fragmented|cross_pg|missing_pg|duplicate_device|expired)
            ;;
        *)
            echo "未知 D2 runtime 场景：${scenario}" >&2
            exit 1
            ;;
    esac

    # cross_pg 需要至少两个 native replica。1 节点 4 张 rollout NPU、TP=2
    # 时会产生两个 world_size=2 的 native PG；其他场景沿用默认 TP=4。
    if [ "${scenario}" = "cross_pg" ]; then
        export ROLLOUT_TP="${D2_CROSS_PG_ROLLOUT_TP:-2}"
    else
        export ROLLOUT_TP="${D2_ROLLOUT_TP:-4}"
    fi

    log_file="${D2_RUNTIME_LOG_DIR}/d2_${scenario}_$(date +%Y%m%d%H%M%S).log"
    echo "开始 D2 runtime 场景：${scenario}"
    echo "日志：${log_file}"

    set +e
    bash "${SCRIPT_DIR}/multi_task_run.sh" \
        "+multitask.d2_runtime_test.enabled=true" \
        "+multitask.d2_runtime_test.scenario=${scenario}" \
        "+multitask.d2_runtime_test.cleanup_after_test=true" \
        2>&1 | tee "${log_file}"
    command_statuses=( "${PIPESTATUS[@]}" )
    set -e

    # multi_task_run.sh 的默认日志管道可能只返回 tee 的状态；以 hook
    # 输出的明确结果作为 borrowed 创建是否完成的判据。
    if [ "${command_statuses[0]}" -ne 0 ]; then
        echo "场景 ${scenario} 的 main_ppo 进程失败，exit=${command_statuses[0]}；日志：${log_file}" >&2
        exit "${command_statuses[0]}"
    fi
    if [ "${command_statuses[1]}" -ne 0 ]; then
        echo "场景 ${scenario} 的日志写入失败，tee exit=${command_statuses[1]}；日志：${log_file}" >&2
        exit "${command_statuses[1]}"
    fi
    if ! grep -Fq "D2_RUNTIME_RESULT" "${log_file}"; then
        echo "场景 ${scenario} 没有产生 D2_RUNTIME_RESULT；不能判定 borrowed 创建成功。" >&2
        exit 1
    fi
    echo "D2 runtime 场景通过：${scenario}；日志：${log_file}"
done

echo "D2 真实 borrowed replica 创建场景全部通过。"
echo "CE 注册、bootstrap、LB 接流和 reclaim/destroy 未在本阶段执行。"
