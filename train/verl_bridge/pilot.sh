#!/usr/bin/env bash
# GRPO pilot: Nemotron-3-Nano + LoRA on Cloud-OpsBench train, G=4. Run from the repo root.
#
#   STAGE=smoke bash train/verl_bridge/pilot.sh    # 2 tasks x 2 samples, 1 step: does it run at all?
#   STAGE=pilot bash train/verl_bridge/pilot.sh    # 8 tasks x 4 samples, 5 steps: time/step, signal
#   STAGE=full  bash train/verl_bridge/pilot.sh    # the real run (~100 steps)
#
# Every key marked VERIFY: check it exists in YOUR verl version
#   (python -m verl.trainer.main_ppo --cfg job | grep <key>)  before paying for GPUs.
set -euo pipefail

STAGE=${STAGE:-smoke}
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
DATA=${DATA:-data/verl/cloudops-boutique}
NGPUS=${NGPUS:-4}

case $STAGE in
  smoke) BATCH=2; N=2; STEPS=1 ;;
  pilot) BATCH=8; N=4; STEPS=5 ;;
  full)  BATCH=16; N=4; STEPS=100 ;;
esac

export PYTHONPATH=$PWD:${PYTHONPATH:-}        # so verl's workers can import src/ and train/
export BROWNOUT_TOOL_PARSER=${BROWNOUT_TOOL_PARSER:-qwen3_coder}   # hermes for Qwen3-8B
export BROWNOUT_MAX_TURNS=20
export BROWNOUT_TOKENS_PER_TURN=4096
export BROWNOUT_MODEL_ID=${BROWNOUT_MODEL_ID:-nemotron-nano-lora-$STAGE}
export BROWNOUT_SAVE_DIR=${BROWNOUT_SAVE_DIR:-runs/train-$STAGE}    # python -m src.report runs/train-$STAGE
# CLOUDOPSBENCH_ROOT must point at the Cloud-OpsBench clone on this machine (.env works)

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA/train.parquet \
  data.val_files=$DATA/val.parquet \
  data.train_batch_size=$BATCH \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=train/verl_bridge/agent_loop.yaml \
  trainer.n_gpus_per_node=$NGPUS \
  trainer.nnodes=1 \
  trainer.total_training_steps=$STEPS \
  trainer.test_freq=-1 \
  trainer.save_freq=$STEPS \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=brownout \
  trainer.experiment_name=cloudops-grpo-$STAGE \
  "$@"
# VERIFY: lora_* / target_modules (all-linear may also hit the routed experts: list the
#   attention + Mamba in/out projections + shared expert explicitly once names are known),
#   rollout.agent.agent_loop_config_path, use_dynamic_bsz.
# Dynamic sampling (drop groups whose 4 rewards are equal) lives in verl's DAPO recipe
#   (algorithm.filter_groups.*). Add it after the pilot shows how many groups are flat.
