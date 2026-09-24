#!/usr/bin/env bash
#
# D4 真实 main_ppo 端到端入口：
# native training -> TaskRunner create command -> borrowed runtime -> CE
# target-only bootstrap -> LB READY -> endpoint probe -> test cleanup。
#
# D4 的生产 sleep/wake、drain、commit_remove、reclaim 和 destroy 仍未实现；
# 本脚本只验证 D4 创建调用链和 READY 发布边界。
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

D4_RUNTIME_SCENARIOS="${D4_RUNTIME_SCENARIOS:-basic}"
D4_RUNTIME_LOG_DIR="${D4_RUNTIME_LOG_DIR:-${VERL_REPO_DIR}/logs/d4_runtime}"
mkdir -p "${D4_RUNTIME_LOG_DIR}"

# 保持与 D3 的最小合法训练规模一致：4 张训练 NPU 需要 4 条序列。
export TRAIN_TOTAL_EPOCHS="${D4_TRAIN_TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${D4_TOTAL_TRAINING_STEPS:-1}"
export RESPONSES_PER_PROMPT="${D4_RESPONSES_PER_PROMPT:-2}"
export RESPONSES_PER_PROMPT_VAL="${D4_RESPONSES_PER_PROMPT_VAL:-1}"
export PPO_MINI_BATCH_SIZE="${D4_PPO_MINI_BATCH_SIZE:-2}"
export ASYNC_TRIGGER_SYNC_STEP="${D4_ASYNC_TRIGGER_SYNC_STEP:-1}"
export ASYNC_REQUIRE_BATCHES="${D4_ASYNC_REQUIRE_BATCHES:-1}"
export LOG_DIR="${D4_RUNTIME_LOG_DIR}"
export TRAIN_NODES="${TRAIN_NODES:-1}"
export TRAIN_NPUS_PER_NODE="${TRAIN_NPUS_PER_NODE:-4}"

D4_DP_SIZE=$((TRAIN_NODES * TRAIN_NPUS_PER_NODE))
D4_MINIBATCH_SEQUENCES=$((PPO_MINI_BATCH_SIZE * RESPONSES_PER_PROMPT))
if [ "${D4_MINIBATCH_SEQUENCES}" -lt "${D4_DP_SIZE}" ] || \
    [ "$((D4_MINIBATCH_SEQUENCES % D4_DP_SIZE))" -ne 0 ]; then
    echo "D4 batch 配置错误：mini_batch*n=${D4_MINIBATCH_SEQUENCES} 必须 >= DP=${D4_DP_SIZE} 且能被整除。" >&2
    exit 1
fi

D4_REQUIRED_PROMPTS=$((PPO_MINI_BATCH_SIZE * ASYNC_REQUIRE_BATCHES * ASYNC_TRIGGER_SYNC_STEP * TOTAL_TRAINING_STEPS))
export TOTAL_ROLLOUT_STEPS="${D4_TOTAL_ROLLOUT_STEPS:-${D4_REQUIRED_PROMPTS}}"
if ! [ "${TOTAL_ROLLOUT_STEPS}" -ge "${D4_REQUIRED_PROMPTS}" ] 2>/dev/null; then
    echo "D4_TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS} 不足，至少需要 ${D4_REQUIRED_PROMPTS} 个 prompt。" >&2
    exit 1
fi

case "${D4_RUNTIME_SCENARIOS}" in
    *[!a-zA-Z0-9_,]*)
        echo "D4_RUNTIME_SCENARIOS 只允许使用字母、数字、下划线和逗号。" >&2
        exit 1
        ;;
esac

for scenario in $(printf '%s' "${D4_RUNTIME_SCENARIOS}" | tr ',' ' '); do
    [ -n "${scenario}" ] || continue
    case "${scenario}" in
        basic|split|fragmented|cross_pg|shared_bundle|merge_world_size|idempotent|concurrent_idempotent)
            ;;
        *)
            echo "D4 不支持场景：${scenario}；可选 basic、split、fragmented、cross_pg、shared_bundle、merge_world_size、idempotent、concurrent_idempotent。" >&2
            exit 1
            ;;
    esac

    if [ "${scenario}" = "cross_pg" ] || [ "${scenario}" = "merge_world_size" ]; then
        export ROLLOUT_TP="${D4_CROSS_PG_ROLLOUT_TP:-2}"
    else
        export ROLLOUT_TP="${D4_ROLLOUT_TP:-4}"
    fi

    log_file="${D4_RUNTIME_LOG_DIR}/d4_${scenario}_$(date +%Y%m%d%H%M%S).log"
    echo "开始 D4 command-chain 场景：${scenario}"
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
        "+actor_rollout_ref.rollout.enable_sleep_mode=true" \
        "actor_rollout_ref.rollout.free_cache_engine=true" \
        "actor_rollout_ref.rollout.checkpoint_engine.backend=multitask_hccl" \
        "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=multi_task_scheduler.checkpoint.hccl_checkpoint_engine" \
        "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.multitask_hccl.rebuild_group=true" \
        "+multitask.d4_runtime_test.enabled=true" \
        "+multitask.d4_runtime_test.scenario=${scenario}" \
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
    if [ "${scenario}" = "shared_bundle" ]; then
        # S5 只验证同一 bundle 上多个 fractional CE Worker 的串行创建和
        # 回收。它不会把两个 active Worker 同时加入 HCCL effective set，
        # 因而不应伪造 LB_READY 证据。
        if ! grep -Fq "D4_SHARED_BUNDLE_RESULT" "${log_file}" || \
            ! grep -Fq '"state": "PLACEMENT_READY"' "${log_file}" || \
            ! grep -Fq "D4_SHARED_BUNDLE_CLEANUP" "${log_file}"; then
            echo "场景 ${scenario} 缺少 shared-bundle placement 或 cleanup 证据；日志：${log_file}" >&2
            exit 1
        fi
    else
        if [ "${scenario}" = "idempotent" ]; then
            if ! grep -Fq "D4_IDEMPOTENCY_RESULT" "${log_file}" || \
                ! grep -Fq '"state": "LB_READY"' "${log_file}"; then
                echo "场景 ${scenario} 缺少 LB_READY 或重复 create 的幂等性证据；日志：${log_file}" >&2
                exit 1
            fi
        elif [ "${scenario}" = "concurrent_idempotent" ]; then
            if ! grep -Fq "D4_CONCURRENCY_RESULT" "${log_file}" || \
                ! grep -Fq '"state": "LB_READY"' "${log_file}"; then
                echo "场景 ${scenario} 缺少 LB_READY 或并发 create 的幂等性证据；日志：${log_file}" >&2
                exit 1
            fi
        elif ! grep -Fq "D4_RUNTIME_RESULT" "${log_file}" || \
            ! grep -Fq '"state": "LB_READY"' "${log_file}"; then
            echo "场景 ${scenario} 没有达到 LB_READY；日志：${log_file}" >&2
            exit 1
        fi
        if ! grep -Fq "D4_RUNTIME_CLEANUP" "${log_file}"; then
            echo "场景 ${scenario} 没有完成 D4 测试清理；日志：${log_file}" >&2
            exit 1
        fi
    fi
    if grep -Eq "Traceback|AssertionError|LIFECYCLE_NOT_IMPLEMENTED" "${log_file}"; then
        echo "场景 ${scenario} 日志包含未处理异常或错误生命周期回执；日志：${log_file}" >&2
        exit 1
    fi
    echo "D4 command-chain 场景通过：${scenario}；日志：${log_file}"
done

echo "D4 TaskRunner -> runtime -> CE bootstrap -> LB READY 场景全部通过。"
echo "D4 不实现生产 sleep/wake、drain、commit_remove、reclaim/destroy。"
