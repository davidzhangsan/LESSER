# Shared helpers for the RL launch recipes (sourced by run_countdown.sh and run_corrupted.sh; not
# executable on its own).
#
# Required environment:
#   GRADALIGN_DIR   upstream GradAlign at commit 7e25218 with third_party/GradAlign/lesser_gradalign.patch
#                   applied, and its Python environment in $GRADALIGN_DIR/.venv (see rl/README.md).
# Optional environment:
#   GRADALIGN_VENV  virtual environment to activate (default: $GRADALIGN_DIR/.venv).
#   RUN_ROOT        where rollouts, scores, selections and checkpoints go (default:
#                   $GRADALIGN_DIR/data_local/chkpt). The released trainer always writes checkpoints to
#                   $GRADALIGN_DIR/data_local/chkpt/<experiment>, while the orchestrator merges them from
#                   --ckpt_root; a separate RUN_ROOT is therefore linked from data_local/chkpt.
#   LOG_DIR         where the driver log goes (default: $RUN_ROOT/logs).
#   DRY_RUN=1       print the pipeline command and environment instead of running it.

LESSER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LESSER_DATA="$LESSER_ROOT/data/rl"

lesser_die() { echo "error: $*" >&2; exit 1; }

# Check the GradAlign checkout and activate its environment; resolve and create RUN_ROOT and LOG_DIR.
lesser_setup() {
  [ -n "${GRADALIGN_DIR:-}" ] || lesser_die "set GRADALIGN_DIR to the patched GradAlign checkout (see rl/README.md)"
  GRADALIGN_DIR="$(cd "$GRADALIGN_DIR" && pwd)" || lesser_die "GRADALIGN_DIR does not exist"
  [ -f "$GRADALIGN_DIR/select/parallel/prod_feature.py" ] \
    || lesser_die "$GRADALIGN_DIR lacks select/parallel/prod_feature.py: apply third_party/GradAlign/lesser_gradalign.patch"
  local venv="${GRADALIGN_VENV:-$GRADALIGN_DIR/.venv}"
  [ -f "$venv/bin/activate" ] || lesser_die "no virtual environment at $venv (set GRADALIGN_VENV)"
  if [ "${DRY_RUN:-0}" != 1 ]; then
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
  fi
  local link="$GRADALIGN_DIR/data_local/chkpt"
  RUN_ROOT="${RUN_ROOT:-$link}"
  mkdir -p "$RUN_ROOT"
  RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
  if [ "$RUN_ROOT" != "$link" ]; then
    if [ -e "$link" ] || [ -L "$link" ]; then
      [ "$(cd "$link" && pwd -P)" = "$(cd "$RUN_ROOT" && pwd -P)" ] \
        || lesser_die "$link exists and is not RUN_ROOT; remove it or link it to $RUN_ROOT"
    else
      mkdir -p "$GRADALIGN_DIR/data_local" && ln -s "$RUN_ROOT" "$link"
    fi
  fi
  LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
  mkdir -p "$LOG_DIR"
}

# Check that the problem pools exist in $GRADALIGN_DIR/data_local/data/<name>/ (built by rl/build_pools.py)
# and match data/rl/pools/SHA256SUMS. A missing or different file is an error; nothing is rebuilt here.
lesser_check_pools() {
  local dir="$GRADALIGN_DIR/data_local/data" name f
  for name in "$@"; do
    for f in train.jsonl train.parquet; do
      [ -f "$dir/$name/$f" ] && continue
      if [ "${DRY_RUN:-0}" = 1 ]; then echo "note: $dir/$name/$f is missing"; continue; fi
      lesser_die "missing $dir/$name/$f; build the pools first: python $LESSER_ROOT/rl/build_pools.py --out $dir"
    done
  done
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "python $LESSER_ROOT/rl/build_pools.py --check $dir --only $*"
    return 0
  fi
  python "$LESSER_ROOT/rl/build_pools.py" --check "$dir" --only "$@" \
    || lesser_die "the pools in $dir differ from the released ones (see above)"
}

# Matched design: copy the round-0 rollouts (candidate and query responses and their rewards) and the
# candidate chunks of a donor experiment, so that every method of one seed scores the same round-0
# responses. The donor's tokenized shards (data/data_<rank>.npz) are deliberately NOT copied: the
# released analyzer only checks that data_0 ... data_<ranks-1> exist, so shards prepared for more ranks
# than this run uses would silently drop responses. Each run re-tokenizes for its own rank count.
lesser_copy_donor() {
  local donor="$1" exp="$2" f
  [ -d "$donor/global_step_0" ] || lesser_die "donor $donor has no global_step_0"
  [ -d "$exp/global_step_0" ] && { echo "round 0 already present in $exp; donor not copied"; return 0; }
  [ -d "$donor/chunks" ] || lesser_die "donor $donor has no chunks/"
  for f in train_split/part_0/responses.json train_split/part_0/responses_sorted.json train_split/accuracy_by_problem.jsonl \
           val_responses/responses.json val_responses/responses_sorted.json; do
    [ -f "$donor/global_step_0/$f" ] || lesser_die "donor round 0 is incomplete: missing $f"
  done
  mkdir -p "$exp/global_step_0/train_split/part_0" "$exp/global_step_0/val_responses"
  cp -r "$donor/chunks" "$exp/"
  for f in responses.json responses_sorted.json metadata.json; do
    if [ -f "$donor/global_step_0/train_split/part_0/$f" ]; then cp "$donor/global_step_0/train_split/part_0/$f" "$exp/global_step_0/train_split/part_0/"; fi
    if [ -f "$donor/global_step_0/val_responses/$f" ]; then cp "$donor/global_step_0/val_responses/$f" "$exp/global_step_0/val_responses/"; fi
  done
  cp "$donor/global_step_0/train_split/accuracy_by_problem.jsonl" "$exp/global_step_0/train_split/"
  echo "copied round-0 rollouts from donor $donor"
}

# Refuse to resume a round whose scoring is incomplete with tokenized shards prepared for a different
# number of analysis ranks (the released shard check would load only part of the responses).
lesser_check_shards() {
  local exp="$1" ranks="$2" chunk="$3" step sim rows d n
  for step in "$exp"/global_step_*; do
    [ -d "$step" ] || continue
    sim="$step/train_split/part_0/similarity_results_cosine_real.jsonl"
    rows=0; [ -f "$sim" ] && rows=$(wc -l < "$sim")
    [ "$rows" -ge "$chunk" ] && continue
    for d in "$step/train_split/part_0/data" "$step/val_responses/data"; do
      [ -d "$d" ] || continue
      n=$(find "$d" -maxdepth 1 -name 'data_*.npz' | wc -l)
      [ "$n" -eq "$ranks" ] || lesser_die "$d holds shards for $n analysis ranks but NUM_GPUS=$ranks; \
rerun with NUM_GPUS=$n, or delete $d and the partial $sim"
    done
  done
}

# Run the released orchestrator from $GRADALIGN_DIR/automated, logging to $LOG_DIR/<prefix>.log.
lesser_run_pipeline() {
  local prefix="$1"; shift
  local log="$LOG_DIR/$prefix.log"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_$(echo "$prefix" | md5sum | cut -c1-6)}"  # short path: AF_UNIX socket limit
  if [ "${DRY_RUN:-0}" = 1 ]; then
    env | grep -E '^(GRADALIGN_[A-Z_]+|NUM_GPUS|NCCL_P2P_LEVEL|WANDB_MODE|RAY_TMPDIR)=' | grep -v '^GRADALIGN_DIR=' | sort
    echo "cd $GRADALIGN_DIR/automated && python dynamic_selection.py $*"
    return 0
  fi
  mkdir -p "$RAY_TMPDIR"
  echo "=== $(date -Is) start prefix=$prefix host=$(hostname) args: $*" >> "$log"
  local rc=0
  (cd "$GRADALIGN_DIR/automated" && python dynamic_selection.py "$@") 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
  echo "=== $(date -Is) exit $rc prefix=$prefix" >> "$log"
  return "$rc"
}
