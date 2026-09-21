set -x

log_dir="./logs"
mkdir -p $log_dir
timestamp=$(date +"%Y%m%d%H%M%S")

export HCCL_OP_EXPANSION_MODE="AIV"
# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=vanilla

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# rollout correction
rollout_is="sequence"                     # Self-normalized sequence-level IS
rollout_is_threshold=2.0                  # Upper threshold for IS weights
rollout_is_batch_normalize="true"         # Self-normalization (mean=1.0)

# ===================================== Data/Model =====================================
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

TASK_NAME=${TASK_NAME:-gsm8k}
if [[ $TASK_NAME == "gsm8k" ]]; then
    train_files=/workspace/l00839669/datasets/gsm8k/train.parquet
    test_files=/workspace/l00839669/datasets/gsm8k/train.parquet
    actor_model_path=/workspace/n00873601/multi_rl_task_verl/Qwen3-0.6B
    apply_rope_fusion=True
elif [[ $TASK_NAME == "geo3k" ]]; then
    train_files=/workspace/l00839669/datasets/geo3k/train.parquet
    test_files=/workspace/l00839669/datasets/geo3k/test.parquet
    actor_model_path=/workspace/l00839669/models/Qwen3-VL-30B-A3B-Instruct
    apply_rope_fusion=False
elif [[ $TASK_NAME == "dapo" ]]; then
    train_files=/workspace/l00839669/datasets/dapo-math-17k/dapo-math-17k.parquet
    test_files=/workspace/l00839669/datasets/aime24/aime-2024.parquet
    actor_model_path=/workspace/l00839669/models/Qwen3-8B-Base
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 8))
else
    echo "TASK_NAME $TASK_NAME not supported"
    exit 1
fi

critic_model_path=$actor_model_path

max_prompt_length=${max_prompt_length:-$((1024 * 1))}
max_response_length=${max_response_length:-$((1024 * 2))}
train_batch_size=32
ppo_mini_batch_size=8
n_resp_per_prompt=2
n_resp_per_prompt_val=1

MODEL_NAME_ONLY=${MODEL_ID##*/}
log_file="${log_dir}/${MODEL_NAME_ONLY}_${DATASET_NAME}_transferqueue_NEW_TRAINER_longrun_${timestamp}.log"

n_gpus_training=8
# ===================================== Training =====================================
backend=${BACKEND:-megatron} # fsdp, fsdp2, megatron

actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length)))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))

USP_SIZE=2
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$USP_SIZE"

TP_SIZE=4
CP_SIZE=1
PP_SIZE=2
VPP_SIZE=null
EP_SIZE=1
ETP_SIZE=1
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=$apply_rope_fusion \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

CIRITC_CONFIG="
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_name=vllm
infer_tp=4
infer_dp=1
infer_ep=1

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# wandb
project_name=transfer_queue_test_new_trainer
experiment_name=${MODEL_NAME_ONLY}-$adv_estimator-$backend-$rollout_name
default_local_dir=./checkpoint/$project_name/$experiment_name


GLOBAL_PROFILER="
    +global_profiler.tool=npu \
    +global_profiler.steps=[1] \
    +global_profiler.profile_continuous_step=false"

python3 -m verl.trainer.main_ppo \
    --config-path=/workspace/n00873601/multi_rl_task_verl/verl/verl/trainer/config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    algorithm.rollout_correction.rollout_is=$rollout_is \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=${n_gpus_training} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=3 \
    trainer.resume_mode=disable \
    ray_kwargs.timeline_json_file=./ray_timeline.json \
    $ACTOR_CONFIG \
    $CIRITC_CONFIG \
    $ROLLOUT_CONFIG \
    $GLOBAL_PROFILER \
    $@ 2>&1 | tee "$log_file"