#!/usr/bin/env bash
# Online RL on the Countdown mixture (paper Figure 6, Finding 4; Appendix "Targeted reinforcement learning").
#
# One run = one method and one seed: Qwen2.5-Math-1.5B-Instruct, pool cdmix4 (4,000 Countdown, 20,000
# WebInstruct, 10,000 DAPO), five rounds; each round scores a 5,120-problem chunk with 16 responses per
# candidate and per query (200 countdown4_val queries), selects 256 problems (k = 20), and trains 10 GRPO
# steps (128 problems x 128 responses per step, AdamW 1e-6, no KL, responses up to 3,072 tokens).
# verl evaluates the policy every 5 steps on the 500 held-out countdown4_test problems (16 responses each);
# the scalar val-core/countdown/reward/mean@16 is the curve in Figure 6.
#
# usage: rl/run_countdown.sh METHOD SEED [extra dynamic_selection.py arguments]
#   METHOD  lesser   LESSER: projected output-layer features, 64 x 128 two-sided sketch
#                    (--mode prod, GRADALIGN_PROD_SKETCH=1)
#           full     full policy-gradient GradAlign (--mode sim)
#           dynamic  dynamic sampling, pass rate in [1/16, 15/16] (--mode acc)
#           random   uniform selection (--mode rand)
#   SEED    42 or 43 (chunk order and selection RNG)
#
# Matched design: all methods of one seed score the same round-0 responses. Run `lesser` first; the
# other methods copy its round-0 rollouts (chunks, responses and rewards, never the tokenized shards).
# DONOR=<experiment dir with global_step_0> sets the donor, also for `lesser` (the paper's LESSER and
# full-gradient runs took round 0 from another run of the same seed this way); DONOR=none samples round 0 anew.
# Re-running the same command resumes (chunks, rollouts, scores, selections and the verl checkpoints, saved
# every 10 steps, are reused).
#
# The defaults are the settings of the LESSER and full-gradient runs, each on 2 x A100 80GB:
#   NUM_GPUS=2                     analysis (scoring) ranks. Changes which responses share a scoring
#                                  mini-batch of 2 (the released analyzer shards the length-sorted responses
#                                  round-robin across ranks), so keep it fixed within a comparison.
#   GRADALIGN_TRAIN_GPUS=2         verl training GPUs (FSDP), GRADALIGN_INFER_CONCURRENCY=2 vLLM engines,
#   GRADALIGN_VLLM_UTIL=0.45       vLLM memory fraction. Throughput only.
# Those runs also set HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 with a Hugging Face cache that holds the
# model; set them the same way when the cache is complete.
# Environment: see rl/_common.sh (GRADALIGN_DIR required; RUN_ROOT, LOG_DIR, DRY_RUN optional).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[ $# -ge 2 ] || lesser_die "usage: rl/run_countdown.sh {lesser|full|dynamic|random} SEED [args]"
METHOD="$1"; SEED="$2"; shift 2
[[ "$SEED" =~ ^[0-9]+$ ]] || lesser_die "SEED must be an integer"
EXTRA=()
case "$METHOD" in
  lesser)  MODE=prod; ARM=prod; export GRADALIGN_PROD_SKETCH=1 ;;   # projected 64 x 128 feature
  full)    MODE=sim;  ARM=full ;;
  dynamic) MODE=acc;  ARM=acc;  EXTRA=(--acc_low 0.0625 --acc_high 0.9375) ;;
  random)  MODE=rand; ARM=rand; export GRADALIGN_SKIP_ROLLOUTS_FOR_RAND="${GRADALIGN_SKIP_ROLLOUTS_FOR_RAND:-1}" ;;
  *) lesser_die "unknown METHOD $METHOD" ;;
esac

lesser_setup
lesser_check_pools cdmix4 countdown4_val countdown4_test

# Paper settings that differ from the released code defaults, and the knobs the paper runs set.
export GRADALIGN_ROLLOUT_N="${GRADALIGN_ROLLOUT_N:-128}"        # GRPO group size (code default 32)
export GRADALIGN_USE_KL="${GRADALIGN_USE_KL:-0}"                # KL penalty off (code default on)
export GRADALIGN_MAX_RETRIES="${GRADALIGN_MAX_RETRIES:-3}"      # retry a failed stage; stages resume from saved state
export GRADALIGN_MIN_AVAIL_GB="${GRADALIGN_MIN_AVAIL_GB:-0}"    # no host-RAM launch gate
export GRADALIGN_REAP_ANY_ORPHAN="${GRADALIGN_REAP_ANY_ORPHAN:-1}"
# Throughput only (same computation up to rounding): sequence packing, fused kernels, dynamic batching.
# These need flash-attn; set all three to 0 without it.
export GRADALIGN_REMOVE_PADDING="${GRADALIGN_REMOVE_PADDING:-1}" GRADALIGN_FUSED_KERNELS="${GRADALIGN_FUSED_KERNELS:-1}"
export GRADALIGN_DYNBSZ="${GRADALIGN_DYNBSZ:-1}"
export GRADALIGN_PPO_MAX_TOKEN_LEN="${GRADALIGN_PPO_MAX_TOKEN_LEN:-8192}" GRADALIGN_LOGPROB_MAX_TOKEN_LEN="${GRADALIGN_LOGPROB_MAX_TOKEN_LEN:-24576}"
export GRADALIGN_VLLM_UTIL="${GRADALIGN_VLLM_UTIL:-0.45}"
export GRADALIGN_TRAIN_GPUS="${GRADALIGN_TRAIN_GPUS:-2}"
export GRADALIGN_INFER_CONCURRENCY="${GRADALIGN_INFER_CONCURRENCY:-2}"
export NUM_GPUS="${NUM_GPUS:-2}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

PREFIX="countdown-$ARM-s$SEED"
EXP_NAME="${PREFIX}_qwen2.5-1.5b-math_cdmix4_cdmix4_countdown4_val"
EXP="$RUN_ROOT/$EXP_NAME"
if [ "$METHOD" = lesser ]; then
  DONOR="${DONOR:-none}"
else
  DONOR="${DONOR:-$RUN_ROOT/countdown-prod-s${SEED}_qwen2.5-1.5b-math_cdmix4_cdmix4_countdown4_val}"
fi
if [ "$DONOR" != none ]; then
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "would copy round-0 rollouts from $DONOR"; else lesser_copy_donor "$DONOR" "$EXP"; fi
fi
lesser_check_shards "$EXP" "$NUM_GPUS" 5120

lesser_run_pipeline "$PREFIX" \
  --prefix "$PREFIX" --model qwen2.5-1.5b-math --train_dataset cdmix4 --val_dataset countdown4_val \
  --seed "$SEED" --mode "$MODE" --num_gpus "$NUM_GPUS" --ckpt_root "$RUN_ROOT" \
  --verl_val_set countdown4_test --reward_path countdown_mixed_reward.py \
  --chunk_size 5120 --k 20 --num_selections "${ROUNDS:-5}" --train_batch_size 128 --iters_per_select 10 \
  --minibatch_size 2 --max_tokens 3072 --n_samples_train 16 --n_samples_val 16 \
  ${EXTRA[@]+"${EXTRA[@]}"} "$@"
echo "TensorBoard events: $GRADALIGN_DIR/automated/tensorboard_log/Dynamic-restart/$EXP_NAME/"
echo "Selections: $EXP/selected/iter_<round>_${MODE}_256/train.jsonl"
