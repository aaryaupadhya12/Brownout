#!/usr/bin/env bash
# GRPO + LoRA on Cloud-OpsBench train. Run from the repo root. On Modal, modal_app.py sets
# MODEL/NGPUS/... (Qwen3-4B, 1 H100); the Nemotron defaults below are for a 4-GPU box.
# Every setting and every error it took to get here: train/verl_bridge/DEBUGGING.md
#
#   STAGE=smoke bash train/verl_bridge/pilot.sh    # 2 tasks x 2 samples, 1 step: does it run at all?
#   STAGE=pilot bash train/verl_bridge/pilot.sh    # 8 tasks x 4 samples, 5 steps: time/step, signal
#   STAGE=full  bash train/verl_bridge/pilot.sh    # the real run: 8 x 4, 50 steps, val every 10
#
# Every key marked VERIFY: check it exists in YOUR verl version
#   (python -m verl.trainer.main_ppo --cfg job | grep <key>)  before paying for GPUs.
set -euo pipefail

STAGE=${STAGE:-smoke}
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
DATA=${DATA:-data/verl/cloudops-boutique}
NGPUS=${NGPUS:-4}
TP=${TP:-$NGPUS}          # SGLang tensor parallel; must divide NGPUS (verl's default is 2)

# BATCH tasks/step x N samples. MINI: tasks per optimizer update (BATCH/MINI updates per step;
#   MINI=BATCH means 1 update, and then ppo_kl / pg_clipfrac are always 0).
# RESUME: smoke/pilot always start fresh (a leftover checkpoint would make verl skip training);
#   full resumes from its last checkpoint after a crash/preemption.
# VAL0 / TEST_FREQ: validation (53 val tasks, ~5 min) before training / every N steps (-1 = never).
# SAVE_FREQ: checkpoint every N steps (each ~30 s; also what RESUME=auto restarts from).
case $STAGE in
  smoke) BATCH=2; N=2; STEPS=1;  MINI=2; RESUME=disable; VAL0=False; TEST_FREQ=-1; SAVE_FREQ=1 ;;
  pilot) BATCH=8; N=4; STEPS=5;  MINI=4; RESUME=disable; VAL0=False; TEST_FREQ=-1; SAVE_FREQ=5 ;;
  full)  BATCH=8; N=4; STEPS=50; MINI=4; RESUME=auto;    VAL0=True;  TEST_FREQ=10; SAVE_FREQ=10 ;;
  # eval: validation only, no training. VAL_FILE picks the split. RESUME=resume_path plus
  #   trainer.resume_from_path=<checkpoint> scores a checkpoint; disable = the base model
  eval)  BATCH=8; N=4; STEPS=1;  MINI=4; RESUME=${RESUME:-disable}; VAL0=True; TEST_FREQ=-1; SAVE_FREQ=-1 ;;
esac
VAL_ONLY=$([ "$STAGE" = eval ] && echo True || echo False)

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
  data.val_files=${VAL_FILE:-$DATA/val.parquet} \
  data.train_batch_size=$BATCH \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.5} \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=train/verl_bridge/agent_loop.yaml \
  trainer.n_gpus_per_node=$NGPUS \
  trainer.nnodes=1 \
  trainer.total_training_steps=$STEPS \
  trainer.val_before_train=$VAL0 \
  trainer.val_only=$VAL_ONLY \
  trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.resume_mode=$RESUME \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=brownout \
  trainer.experiment_name=${EXPERIMENT:-cloudops-grpo-$STAGE} \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS:-32} \
  "$@"
# VERIFY: lora_* / target_modules (all-linear may also hit the routed experts: list the
#   attention + Mamba in/out projections + shared expert explicitly once names are known),
#   rollout.agent.agent_loop_config_path, use_dynamic_bsz.
# ray_kwargs.ray_init.num_cpus: Ray's CPU count is only a scheduling budget, not real cores.
#   verl's actors (TaskRunner, workers, SGLang, 8 TransferQueue storage units, ...) each claim
#   some; if Ray counts fewer than they need it waits forever, silently. 32 avoids that.
# Dynamic sampling (drop groups whose 4 rewards are equal) lives in verl's DAPO recipe
#   (algorithm.filter_groups.*). Add it after the pilot shows how many groups are flat.
