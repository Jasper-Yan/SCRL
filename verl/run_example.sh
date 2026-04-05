#!/bin/bash

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export WANDB_API_KEY=""
# ------------------------------------------------------------
# Experiment Configuration
# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AIME25-TTT"
BACKBONE="Qwen2.5-3B"  # "Qwen3-4B" or "Llama-3.1-8B-Instruct"
ADVANTAGE="grpo"

K=3
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * $K))
if [ "$K" -gt 8 ]; then
  N=4
else
  N=16  # Number of rollouts for validation and majority voting
fi

EPISODE=80
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=32  # Total rollouts per prompt
N_SAMPLES_PER_PROMPT=16  # Samples used for training (downsampling)
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

DATA_LOCAL_DIR="path/to/SCRL/verl/data"
BACKBONE_PATH="path/to/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="SCRL-Len@${K}k"

# Pseudo-positive label thresholds
TAU_POS=0.375            # Minimum proportion for pseudo-positive
TAU_MARG=0.125            # Margin between top-2 answers
TAU_LOW_MIN=0.125        # Minimum threshold for low proportion

# Reward configuration
REWARD_MODE="continuous"     # "binary" or "continuous"
A_PLUS=1.0              # Positive reward scaling
A_MINUS=1.0             # Negative reward scaling
LAMBDA_H=0.1            # Entropy penalty weight (for continuous mode)

WANDB_PROJECT="SCRL-${N_VOTES_PER_PROMPT}-${N_SAMPLES_PER_PROMPT}"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${REWARD_MODE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${REWARD_MODE}"
# ------------------------------------------------------------

python -m verl.trainer.main_ppo \
--config-name='ppo_trainer_scrl.yaml'\
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/utils/reward_score/scrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  custom_val_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_val_reward_function.name=reward_func \
  scrl.tau_pos=$TAU_POS \
  scrl.tau_marg=$TAU_MARG \
  scrl.tau_low_min=$TAU_LOW_MIN \
  scrl.reward_mode=$REWARD_MODE \
  scrl.a_plus=$A_PLUS \
  scrl.a_minus=$A_MINUS \
  scrl.lambda_h=$LAMBDA_H \
  scrl.enable=True \
  scrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  scrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.val_before_train=True \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=1000 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@" 2>&1 | tee scrl_${N_VOTES_PER_PROMPT}-${N_SAMPLES_PER_PROMPT}_${BACKBONE}_${TASK}_Len@${K}k_${TIME_TAG}.log

echo "Output directory: $OUTPUT_DIR"