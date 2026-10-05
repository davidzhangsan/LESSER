#!/usr/bin/env bash
# Selection under corrupted rewards (paper Figure 5, Finding 3; Appendix "Selection under corrupted rewards").
#
# Qwen3-8B-Base, 512 DAPO candidates of which 256 have coin-flip rewards (pool dapo_noisy50_sub512),
# 43 AMC22 query problems, 64 responses per candidate and per query, one selection of 128 candidates at the
# base model (k = 4), no training (GRADALIGN_STOP_AFTER_SELECT=1). A corrupted candidate's reward for each
# response is a hash-seeded fair coin (GRADALIGN_CORRUPT_SEED=0), identical in the rollout judge and in the
# verl reward.
#
# usage: rl/run_corrupted.sh METHOD [extra dynamic_selection.py arguments]
#   METHOD  lesser  samples the rollouts, scores them with LESSER (--mode prod, projected 64 x 128 feature,
#                   GRADALIGN_PROD_SKETCH=1), selects 128. Run first.
#           full    full policy-gradient GradAlign (--mode sim) on the rollouts of the lesser run (DONOR).
# AccGreedy and the Random expectation are computed offline from the saved rewards by
# `python rl/figures.py corrupted` (see rl/README.md for copying a run's outputs into data/rl/corrupted/).
#
# Settings of the paper runs (defaults below). The paper's LESSER run scored the rollouts of another run of
# the same setting (copied in as by DONOR); the reward counts are identical.
#   lesser: NUM_GPUS=2 analysis ranks, scoring mini-batch 4 (48 of the 512 candidates were scored with
#           mini-batch 1; see rl/README.md, "Run settings"), 2 vLLM engines.
#   full:   NUM_GPUS=4 or 8 analysis ranks (mini-batch 1, so the rank count does not change the feature),
#           FSDP CPU offload off (GRADALIGN_ANALYSIS_NO_CPU_OFFLOAD=1) and NCCL_P2P_LEVEL=SYS for 80 GB
#           A100s; the 8B full gradient runs out of memory on two 80 GB ranks.
# Environment: see rl/_common.sh (GRADALIGN_DIR required; RUN_ROOT, LOG_DIR, DRY_RUN optional).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[ $# -ge 1 ] || lesser_die "usage: rl/run_corrupted.sh {lesser|full} [args]"
METHOD="$1"; shift
case "$METHOD" in
  lesser) MODE=prod; ARM=head; MB="${MINIBATCH_SIZE:-4}"; export NUM_GPUS="${NUM_GPUS:-2}"; export GRADALIGN_PROD_SKETCH=1 ;;
  full)   MODE=sim;  ARM=full; MB="${MINIBATCH_SIZE:-1}"; export NUM_GPUS="${NUM_GPUS:-4}"
          export GRADALIGN_ANALYSIS_NO_CPU_OFFLOAD="${GRADALIGN_ANALYSIS_NO_CPU_OFFLOAD:-1}" NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}" ;;
  *) lesser_die "unknown METHOD $METHOD" ;;
esac

lesser_setup
lesser_check_pools dapo_noisy50_sub512 amc22 amc23

export GRADALIGN_STOP_AFTER_SELECT=1                             # selection only, no GRPO training
export GRADALIGN_CORRUPT_SEED="${GRADALIGN_CORRUPT_SEED:-0}"     # coin-flip rewards of corrupted candidates
export GRADALIGN_ROLLOUT_N="${GRADALIGN_ROLLOUT_N:-128}" GRADALIGN_USE_KL="${GRADALIGN_USE_KL:-0}"  # unused without training
export GRADALIGN_MAX_RETRIES="${GRADALIGN_MAX_RETRIES:-3}" GRADALIGN_MIN_AVAIL_GB="${GRADALIGN_MIN_AVAIL_GB:-0}"
export GRADALIGN_REAP_ANY_ORPHAN="${GRADALIGN_REAP_ANY_ORPHAN:-1}"
export GRADALIGN_VLLM_UTIL="${GRADALIGN_VLLM_UTIL:-0.6}"
export GRADALIGN_INFER_CONCURRENCY="${GRADALIGN_INFER_CONCURRENCY:-2}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

PREFIX="corrupted-$ARM"
EXP_NAME="${PREFIX}_qwen3-8b-base_dapo_noisy50_sub512_dapo_noisy50_sub512_amc22"
EXP="$RUN_ROOT/$EXP_NAME"
if [ "$METHOD" = full ] && [ "${DONOR:-}" != none ]; then
  DONOR="${DONOR:-$RUN_ROOT/corrupted-head_qwen3-8b-base_dapo_noisy50_sub512_dapo_noisy50_sub512_amc22}"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "would copy round-0 rollouts from $DONOR"; else lesser_copy_donor "$DONOR" "$EXP"; fi
fi
lesser_check_shards "$EXP" "$NUM_GPUS" 512

lesser_run_pipeline "$PREFIX" \
  --prefix "$PREFIX" --model qwen3-8b-base --train_dataset dapo_noisy50_sub512 --val_dataset amc22 \
  --seed 42 --mode "$MODE" --num_gpus "$NUM_GPUS" --ckpt_root "$RUN_ROOT" \
  --verl_val_set amc23 --reward_path noisy_math_reward.py \
  --chunk_size 512 --k 4 --num_selections 1 --train_batch_size 128 --iters_per_select 1 \
  --minibatch_size "$MB" --max_tokens 3072 --n_samples_train 64 --n_samples_val 64 "$@"
echo "Outputs for rl/figures.py: $EXP/global_step_0/train_split/accuracy_by_problem.jsonl,"
echo "  $EXP/global_step_0/train_split/part_0/similarity_results_cosine_real.jsonl, $EXP/selected/iter_0_${MODE}_128/train.jsonl"
