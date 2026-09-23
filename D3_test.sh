#!/usr/bin/env bash
#
# D3 target-only CE bootstrap 验收入口。
#
# 每个场景通过 multi_task_run.sh 进入真实 main_ppo：先由 native rollout
# 创建 donor，再创建 borrowed runtime，随后在 Trainer 侧注册 CE、执行一次
# target-only bootstrap，并让 main_ppo 的第一次普通参数同步再次包含该 replica。
#
# 兼容服务器上的旧 Bash：不使用 pipefail，通过 PIPESTATUS 显式传播训练和
# tee 的退出码。本脚本沿用 multi_task_run.sh 的 Ascend NPU 环境，选用
# multitask_hccl（继承原生 HCCL，仅修正 vllm-ascend 的通信域销毁接口）。
# 显式打开 rebuild_group，保证 finalize 后可以重建全成员通信域。
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

D3_RUNTIME_SCENARIOS="${D3_RUNTIME_SCENARIOS:-split}"
D3_RUNTIME_LOG_DIR="${D3_RUNTIME_LOG_DIR:-${VERL_REPO_DIR}/logs/d3_runtime}"
mkdir -p "${D3_RUNTIME_LOG_DIR}"

# 与 D2 smoke 保持同样的最小合法训练规模：4 张训练卡需要 4 条序列。
export TRAIN_TOTAL_EPOCHS="${D3_TRAIN_TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${D3_TOTAL_TRAINING_STEPS:-1}"
export RESPONSES_PER_PROMPT="${D3_RESPONSES_PER_PROMPT:-2}"
export RESPONSES_PER_PROMPT_VAL="${D3_RESPONSES_PER_PROMPT_VAL:-1}"
export PPO_MINI_BATCH_SIZE="${D3_PPO_MINI_BATCH_SIZE:-2}"
export ASYNC_TRIGGER_SYNC_STEP="${D3_ASYNC_TRIGGER_SYNC_STEP:-1}"
export ASYNC_REQUIRE_BATCHES="${D3_ASYNC_REQUIRE_BATCHES:-1}"
export LOG_DIR="${D3_RUNTIME_LOG_DIR}"

export TRAIN_NODES="${TRAIN_NODES:-1}"
export TRAIN_NPUS_PER_NODE="${TRAIN_NPUS_PER_NODE:-4}"
D3_TRAIN_DP_SIZE=$((TRAIN_NODES * TRAIN_NPUS_PER_NODE))
D3_MINIBATCH_SEQUENCES=$((PPO_MINI_BATCH_SIZE * RESPONSES_PER_PROMPT))
if [ "${D3_MINIBATCH_SEQUENCES}" -lt "${D3_TRAIN_DP_SIZE}" ] || \
    [ "$((D3_MINIBATCH_SEQUENCES % D3_TRAIN_DP_SIZE))" -ne 0 ]; then
    echo "D3 batch 配置错误：mini_batch*n=${D3_MINIBATCH_SEQUENCES} 必须 >= DP=${D3_TRAIN_DP_SIZE} 且能被整除。" >&2
    exit 1
fi

D3_REQUIRED_PROMPTS=$((PPO_MINI_BATCH_SIZE * ASYNC_REQUIRE_BATCHES * ASYNC_TRIGGER_SYNC_STEP * TOTAL_TRAINING_STEPS))
export TOTAL_ROLLOUT_STEPS="${D3_TOTAL_ROLLOUT_STEPS:-${D3_REQUIRED_PROMPTS}}"
if ! [ "${TOTAL_ROLLOUT_STEPS}" -ge "${D3_REQUIRED_PROMPTS}" ] 2>/dev/null; then
    echo "D3_TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS} 不足，至少需要 ${D3_REQUIRED_PROMPTS} 个 prompt。" >&2
    exit 1
fi

case "${D3_RUNTIME_SCENARIOS}" in
    *[!a-zA-Z0-9_,]*)
        echo "D3_RUNTIME_SCENARIOS 只允许使用字母、数字、下划线和逗号。" >&2
        exit 1
        ;;
esac

for scenario in $(printf '%s' "${D3_RUNTIME_SCENARIOS}" | tr ',' ' '); do
    [ -n "${scenario}" ] || continue
    case "${scenario}" in
        basic|split|fragmented|cross_pg)
            ;;
        *)
            echo "D3 不支持场景：${scenario}；可选 basic、split、fragmented、cross_pg。" >&2
            exit 1
            ;;
    esac

    # cross_pg 需要两个 native PG；TP=2 会在 4 张 rollout NPU 上创建两个 replica。
    if [ "${scenario}" = "cross_pg" ]; then
        export ROLLOUT_TP="${D3_CROSS_PG_ROLLOUT_TP:-2}"
    else
        export ROLLOUT_TP="${D3_ROLLOUT_TP:-4}"
    fi

    log_file="${D3_RUNTIME_LOG_DIR}/d3_${scenario}_$(date +%Y%m%d%H%M%S).log"
    echo "开始 D3 bootstrap 场景：${scenario}"
    echo "日志：${log_file}"

    set +e
    bash "${SCRIPT_DIR}/multi_task_run.sh" \
        "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
        "actor_rollout_ref.rollout.n=${RESPONSES_PER_PROMPT}" \
        "async_training.require_batches=${ASYNC_REQUIRE_BATCHES}" \
        "async_training.trigger_parameter_sync_step=${ASYNC_TRIGGER_SYNC_STEP}" \
        "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
        "trainer.total_epochs=${TRAIN_TOTAL_EPOCHS}" \
        "rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}" \
        "actor_rollout_ref.rollout.enable_sleep_mode=true" \
        "actor_rollout_ref.rollout.free_cache_engine=true" \
        "actor_rollout_ref.rollout.checkpoint_engine.backend=multitask_hccl" \
        "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=multi_task_scheduler.checkpoint.hccl_checkpoint_engine" \
        "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.multitask_hccl.rebuild_group=true" \
        "+multitask.d3_bootstrap_test.enabled=true" \
        "+multitask.d3_bootstrap_test.scenario=${scenario}" \
        "+multitask.d3_bootstrap_test.cleanup_after_test=true" \
        2>&1 | tee "${log_file}"
    command_statuses=( "${PIPESTATUS[@]}" )
    set -e

    if [ "${command_statuses[0]}" -ne 0 ]; then
        echo "场景 ${scenario} 的 main_ppo 进程失败，exit=${command_statuses[0]}；日志：${log_file}" >&2
        exit "${command_statuses[0]}"
    fi
    if [ "${command_statuses[1]}" -ne 0 ]; then
        echo "场景 ${scenario} 的日志写入失败，tee exit=${command_statuses[1]}；日志：${log_file}" >&2
        exit "${command_statuses[1]}"
    fi
    if ! grep -Fq "D3_BOOTSTRAP_RESULT" "${log_file}" || \
        ! grep -Fq "WEIGHTS_READY" "${log_file}"; then
        echo "场景 ${scenario} 没有产生 WEIGHTS_READY bootstrap 结果；日志：${log_file}" >&2
        exit 1
    fi
    if ! grep -Fq "D3_NORMAL_SYNC_RESULT" "${log_file}" || \
        ! grep -Fq "FULL_SYNC_READY" "${log_file}"; then
        echo "场景 ${scenario} 没有通过 bootstrap 后的普通全成员同步；日志：${log_file}" >&2
        exit 1
    fi
    if ! grep -Fq "DONORS_RESTORED_BORROWER_KV_RELEASED" "${log_file}"; then
        echo "场景 ${scenario} 未完成 donor 恢复和 borrowed KV 释放；日志：${log_file}" >&2
        exit 1
    fi
    if grep -Eq "Traceback|AssertionError" "${log_file}"; then
        echo "场景 ${scenario} 日志包含未处理异常；日志：${log_file}" >&2
        exit 1
    fi
    echo "D3 bootstrap 场景通过：${scenario}；日志：${log_file}"
done

echo "D3 target-only bootstrap 与后续普通同步场景全部通过。"
echo "D3 仍不实现 LB 接流、sleep/wake、reclaim/destroy 的完整生命周期。"
