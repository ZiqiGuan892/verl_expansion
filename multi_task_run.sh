#!/usr/bin/env bash
#
# Multi-task Fully Async 启动入口。
# 复用 async_run.sh 的模型、数据、Ascend、资源和异步参数，
# 只额外启用唯一的 Multi-task runtime profile。
#
set -eu


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VERL_REPO_DIR="${VERL_REPO_DIR:-${SCRIPT_DIR}}"
# 服务器布局为：VERL_REPO_PATH/{verl,multi_task_verl,multi_task_run.sh}。
# 插件的 src 和原生源码根都必须进入 Python 搜索路径。
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}"
export VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT:-${VERL_REPO_DIR}/verl/multi_task_verl}"
# 不把 VERL_REPO_DIR 本身加入 PYTHONPATH，避免仓根下的 vllm/、torch/ 等
# 源码目录遮蔽已安装的 vLLM/torch 包。原生 Python 包位于 VERL_SOURCE_ROOT。
# 清理调用者继承的 PYTHONPATH，避免其中残留 VERL_REPO_DIR/vllm 或旧 vLLM
# 源码路径；Python 会自动保留当前解释器的 site-packages。
export PYTHONPATH="${VERL_MULTI_TASK_ROOT}/src:${VERL_SOURCE_ROOT}"
# 将两个根目录传给 async_run.sh，避免它按旧的嵌套布局重新推导。

echo $VERL_MULTI_TASK_ROOT
echo $PYTHONPATH

#!/usr/bin/env bash
#
# 基于 run.sh 的 Fully Async 启动脚本。
# 普通模型、数据和算法参数与 run.sh 保持一致；仅替换 Fully Async 必需的
# 入口、配置文件、4+4 独立资源和 async_training 参数。
#
# 默认：1 节点 8 张 Ascend NPU，训练 4 张、rollout 4 张。
#
# 服务器 Bash 版本可能不支持 pipefail；该入口只使用基础错误退出选项。
set -eu
set -x


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 服务器布局：VERL_REPO_DIR/ 下放脚本；其 verl/ 下放原生仓库与插件。
# VERL_REPO_DIR/verl/verl/{trainer,experimental,...} 是 Python 包，
# VERL_REPO_DIR/verl/verl_multi_task/src/ 是插件的 Python 源码目录。
export VERL_REPO_DIR="${VERL_REPO_DIR:-${SCRIPT_DIR}}"
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}"
# 与用户从第一个 verl/ 目录执行 bash ../async_run.sh 时的工作目录一致。
cd "${VERL_SOURCE_ROOT}"

# Driver 和 Ray 子进程都必须能导入 verl 及 multi_task_scheduler。
export PYTHONPATH="${VERL_MULTI_TASK_ROOT}/src:${VERL_SOURCE_ROOT}"
export HF_DATASETS_CACHE="${VERL_REPO_DIR}/cache"
# ------------------------------ Ascend 环境 ------------------------------

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

[[ -f "${ASCEND_TOOLKIT_ENV}" ]] || { echo "未找到 ${ASCEND_TOOLKIT_ENV}" >&2; exit 1; }
[[ -f "${ASCEND_ATB_ENV}" ]] || { echo "未找到 ${ASCEND_ATB_ENV}" >&2; exit 1; }

# set_env.sh 可能在 bash + nounset 下引用未定义变量。
set +e +u
source "${ASCEND_TOOLKIT_ENV}"
source "${ASCEND_ATB_ENV}"
set -eu

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# 训练命令固定使用的解释器；服务器若有多个 Python，可显式设置 PYTHON_BIN。
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 在启动 Ray/训练前确认该解释器实际加载的是完整 vLLM 包。此前的
# “cannot import name 'LLM' ... (unknown location)”通常表示 vllm 被
# 仓内同名目录遮蔽，或 vLLM 安装在另一个 Python 环境中。
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("vllm")
print("vllm_preflight_python:", sys.executable)
print("vllm_preflight_spec_origin:", None if spec is None else spec.origin)
print(
    "vllm_preflight_search_locations:",
    None if spec is None or spec.submodule_search_locations is None else list(spec.submodule_search_locations),
)
if spec is None:
    raise SystemExit("未找到 vllm；请在当前 PYTHON_BIN 环境安装匹配的 vllm/vllm-ascend")

import vllm

print("vllm_preflight_file:", getattr(vllm, "__file__", None))
print("vllm_preflight_version:", getattr(vllm, "__version__", None))
if not hasattr(vllm, "LLM"):
    raise SystemExit(
        "当前解释器加载的 vllm 没有 LLM；请检查 PYTHONPATH 是否遮蔽 site-packages，"
        "以及 vllm/vllm-ascend 是否安装在同一 Python 环境"
    )
print("vllm_preflight_LLM: available")
PY

# ------------------------------ run.sh 的共同任务配置 ------------------------------

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
TASK_NAME="${TASK_NAME:-gsm8k}"
DATASET_NAME="${DATASET_NAME:-${TASK_NAME}}"

case "${TASK_NAME}" in
    gsm8k)
        TRAIN_FILE="${TRAIN_FILE:-/workspace/l00839669/datasets/gsm8k/train.parquet}"
        TEST_FILE="${TEST_FILE:-/workspace/l00839669/datasets/gsm8k/test.parquet}"
        ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${VERL_REPO_DIR}/Qwen3-0.6B}"
        APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-True}"
        ;;
    geo3k)
        TRAIN_FILE="${TRAIN_FILE:-/workspace/l00839669/datasets/geo3k/train.parquet}"
        TEST_FILE="${TEST_FILE:-/workspace/l00839669/datasets/geo3k/test.parquet}"
        ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/l00839669/models/Qwen3-VL-30B-A3B-Instruct}"
        APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-False}"
        ;;
    dapo)
        TRAIN_FILE="${TRAIN_FILE:-/workspace/l00839669/datasets/dapo-math-17k/dapo-math-17k.parquet}"
        TEST_FILE="${TEST_FILE:-/workspace/l00839669/datasets/aime24/aime-2024.parquet}"
        ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/l00839669/models/Qwen3-8B-Base}"
        APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-False}"
        ;;
    *)
        echo "TASK_NAME=${TASK_NAME} 不支持，可选值：gsm8k、geo3k、dapo。" >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-${ACTOR_MODEL_PATH}}"
DATA_PROMPT_KEY="${DATA_PROMPT_KEY:-prompt}"
[[ -d "${MODEL_PATH}" ]] || { echo "模型目录不存在: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${TRAIN_FILE}" ]] || { echo "训练数据不存在: ${TRAIN_FILE}" >&2; exit 1; }
[[ -f "${TEST_FILE}" ]] || { echo "验证数据不存在: ${TEST_FILE}" >&2; exit 1; }

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-2}"
RESPONSES_PER_PROMPT_VAL="${RESPONSES_PER_PROMPT_VAL:-1}"

ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
LOSS_MODE="${LOSS_MODE:-vanilla}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_COEF="${KL_COEF:-0.001}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ROLLOUT_IS="${ROLLOUT_IS:-sequence}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
ROLLOUT_IS_BATCH_NORMALIZE="${ROLLOUT_IS_BATCH_NORMALIZE:-true}"

# ------------------------------ Fully Async 必要配置 ------------------------------

TRAIN_NODES="${TRAIN_NODES:-1}"
ROLLOUT_NODES="${ROLLOUT_NODES:-1}"
TRAIN_NPUS_PER_NODE="${TRAIN_NPUS_PER_NODE:-4}"
ROLLOUT_NPUS_PER_NODE="${ROLLOUT_NPUS_PER_NODE:-4}"
TOTAL_REQUESTED_NPUS=$((TRAIN_NODES * TRAIN_NPUS_PER_NODE + ROLLOUT_NODES * ROLLOUT_NPUS_PER_NODE))
[[ "${TOTAL_REQUESTED_NPUS}" -le 8 ]] || {
    echo "请求 ${TOTAL_REQUESTED_NPUS} 张 NPU，超过单节点 8 张 NPU。" >&2
    exit 1
}

# 4 张训练 NPU 无法沿用 run.sh 的 Megatron TP=4、PP=2 八卡拓扑，
# 因此异步脚本使用 FSDP2；这是资源规模变化带来的必要配置。
ROLLOUT_TP="${ROLLOUT_TP:-4}"
ROLLOUT_DP="${ROLLOUT_DP:-1}"
ROLLOUT_EP="${ROLLOUT_EP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}"
ROLLOUT_MAX_MODEL_LENGTH="${ROLLOUT_MAX_MODEL_LENGTH:-${MAX_PROMPT_LENGTH}}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LENGTH}}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

TRAIN_TOTAL_EPOCHS="${TRAIN_TOTAL_EPOCHS:-15}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-100}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-5}"
TEST_FREQUENCY="${TEST_FREQUENCY:-5}"
RESUME_MODE="${RESUME_MODE:-disable}"
ASYNC_STALENESS_THRESHOLD="${ASYNC_STALENESS_THRESHOLD:-0.1}"
ASYNC_TRIGGER_SYNC_STEP="${ASYNC_TRIGGER_SYNC_STEP:-4}"
ASYNC_REQUIRE_BATCHES="${ASYNC_REQUIRE_BATCHES:-1}"
ASYNC_PARTIAL_ROLLOUT="${ASYNC_PARTIAL_ROLLOUT:-True}"

PROJECT_NAME="${PROJECT_NAME:-transfer_queue_test_new_trainer}"
MODEL_NAME_ONLY="${MODEL_ID##*/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME_ONLY}-${ADV_ESTIMATOR}-fsdp2-vllm-async}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${VERL_REPO_DIR}/checkpoint/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${VERL_REPO_DIR}/logs}"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d%H%M%S).log"

# ------------------------------ Fully Async 命令 ------------------------------

CMD=(
    "${PYTHON_BIN}" -m verl.experimental.fully_async_policy.fully_async_main
    --config-path=config
    --config-name=fully_async_ppo_trainer.yaml
    multitask.runtime.profile=experimental_fully_async_standalone

    algorithm.adv_estimator="${ADV_ESTIMATOR}"
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}"
    algorithm.kl_ctrl.kl_coef="${KL_COEF}"
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}"
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}"
    algorithm.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE}"
    algorithm.rollout_correction.bypass_mode=True
    algorithm.gamma=1.0
    algorithm.lam=0.95

    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key="${DATA_PROMPT_KEY}"
    data.return_raw_chat=True
    data.train_batch_size=0
    data.gen_batch_size=1
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=False
    data.truncation=error

    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.hybrid_engine=False

    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}"
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True

    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.n="${RESPONSES_PER_PROMPT}"
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n="${RESPONSES_PER_PROMPT_VAL}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LENGTH}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl

    trainer.device=npu
    trainer.nnodes="${TRAIN_NODES}"
    trainer.n_gpus_per_node="${TRAIN_NPUS_PER_NODE}"
    trainer.logger=[console]
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.val_only=False
    trainer.log_val_generations=100
    trainer.save_freq="${SAVE_FREQUENCY}"
    trainer.test_freq="${TEST_FREQUENCY}"
    trainer.total_epochs="${TRAIN_TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.resume_mode="${RESUME_MODE}"

    rollout.nnodes="${ROLLOUT_NODES}"
    rollout.n_gpus_per_node="${ROLLOUT_NPUS_PER_NODE}"
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}"

    async_training.staleness_threshold="${ASYNC_STALENESS_THRESHOLD}"
    async_training.trigger_parameter_sync_step="${ASYNC_TRIGGER_SYNC_STEP}"
    async_training.require_batches="${ASYNC_REQUIRE_BATCHES}"
    async_training.partial_rollout="${ASYNC_PARTIAL_ROLLOUT}"
    async_training.use_trainer_do_validate=False
    async_training.use_dynamic_resource_scheduling=False

    ray_kwargs.timeline_json_file="${VERL_REPO_DIR}/ray_timeline.json"
)

# 额外参数用于注入 profile 或单次 Hydra 覆盖。
if [[ $# -gt 0 ]]; then
    CMD+=( "$@" )
fi

echo "VERL_REPO_DIR=${VERL_REPO_DIR}"
echo "VERL_SOURCE_ROOT=${VERL_SOURCE_ROOT}"
echo "VERL_MULTI_TASK_ROOT=${VERL_MULTI_TASK_ROOT}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "TEST_FILE=${TEST_FILE}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "TRAIN_NPUS_PER_NODE=${TRAIN_NPUS_PER_NODE}"
echo "ROLLOUT_NPUS_PER_NODE=${ROLLOUT_NPUS_PER_NODE}"
echo "TOTAL_REQUESTED_NPUS=${TOTAL_REQUESTED_NPUS}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
