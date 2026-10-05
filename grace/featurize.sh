#!/usr/bin/env bash
# Compute the GRACE feature banks behind the LESSER paper's GRACE results (tier 3, GPU).
#
# For each setting of Table 12 this runs the patched GRACE pipeline (third_party/GRACE) to build
# the prompt subset, generate teacher responses, and write one full-gradient bank
# (Gradients_<teacher>.pkl) and one LESSER bank (Prod_<teacher>.pkl) per teacher into
# $FEATURES_DIR/<setting>/. grace/score.py then scores the banks.
#
# Usage: bash grace/featurize.sh <setting> <stage> [teacher_short_name ...]
#   setting  gsm8k_llama1b  GSM8K, Llama-3.2-1B student, 14 teachers (Table 12 row 1, App. F.4)
#            gsm8k_olmo1b   GSM8K, OLMo-2-1B student, 10 teachers (row 2; reuses the row-1 responses)
#            math_llama3b   MATH, Llama-3.2-3B student, 10 teachers (row 3)
#   stage    subset         write the 512-prompt subset and check its sha256
#            generate       generate teacher responses with vLLM
#            features       tokenize and write both feature banks
#   Teacher names restrict the stage to those teachers (default: all teachers of the setting;
#   gsm8k_olmo1b always processes its ten teachers and skips finished ones).
#
# Environment:
#   GRACE_DIR     GRACE checkout at 64fc99a with third_party/GRACE/lesser_grace.patch applied (required)
#   FEATURES_DIR  output root (default: <lesser repo>/artifacts/grace/features)
#   GEN_PY        python with vLLM, for the generate stage (default: python)
#   FEAT_PY       python with torch/transformers/traker/fast_jl (default: $GRACE_DIR/.venv/bin/python,
#                 built by $GRACE_DIR/scripts/setup_venv.sh)
#   GPU           CUDA device index (default: 0)
#   NUM_SMS       SM count pinned for fast_jl (default: 128, the value behind the stored banks)
#
# Hardware used for the paper banks: RTX 4090 24 GB for everything except the MATH full-gradient
# banks (RTX PRO 6000 96 GB; peak 49.4 GiB). Generation of the four largest GSM8K teachers used a
# 96 GB GPU and vLLM 0.16.0. The pipeline sets no sampling seed, and batching, hardware, and the
# vLLM version can change sampled responses; the stored banks were computed from stored
# generations, which are not distributed.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '9,26p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi
SETTING="$1"; STAGE="$2"; shift 2
REQUESTED=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO/data/grace"
: "${GRACE_DIR:?set GRACE_DIR to the patched GRACE checkout}"
GRACE_DIR="$(cd "$GRACE_DIR" && pwd)"
FEATURES_DIR="${FEATURES_DIR:-$REPO/artifacts/grace/features}"
GEN_PY="${GEN_PY:-python}"
FEAT_PY="${FEAT_PY:-$GRACE_DIR/.venv/bin/python}"
GPU="${GPU:-0}"
NUM_SMS="${NUM_SMS:-128}"
export GEN_PY FEAT_PY NUM_SMS

ROSTER="$DATA_DIR/scores_${SETTING}.json"
[[ -f "$ROSTER" ]] || { echo "unknown setting '$SETTING' (no $ROSTER)" >&2; exit 1; }
[[ -f "$GRACE_DIR/scripts/run_teacher.sh" ]] || {
    echo "$GRACE_DIR has no scripts/run_teacher.sh; apply third_party/GRACE/lesser_grace.patch first" >&2; exit 1; }
command -v "$FEAT_PY" > /dev/null || { echo "FEAT_PY=$FEAT_PY not found" >&2; exit 1; }

# Teacher roster of the setting as "hf_id<TAB>short<TAB>vllm_extra_args" lines, restricted to the
# requested teachers (an unknown name is an error).
roster() {
    "$FEAT_PY" - "$ROSTER" ${REQUESTED[@]+"${REQUESTED[@]}"} <<'EOF'
import json, sys
teachers = json.load(open(sys.argv[1]))["teachers"]
wanted = sys.argv[2:]
unknown = set(wanted) - {t["short"] for t in teachers}
if unknown:
    sys.exit(f"not in the roster: {sorted(unknown)}")
for t in teachers:
    if not wanted or t["short"] in wanted:
        print(t["hf_id"], t["short"], t.get("generation", {}).get("vllm_extra_args", ""), sep="\t")
EOF
}

# Check a generated prompt subset against the sha256 recorded in the roster.
check_subset() {
    "$FEAT_PY" - "$ROSTER" "$GRACE_DIR" <<'EOF'
import hashlib, json, os, sys
spec = json.load(open(sys.argv[1]))["prompt_subset"]
path = os.path.join(sys.argv[2], spec["file"])
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
if digest != spec["sha256"]:
    sys.exit(f"{path}: sha256 {digest} differs from the paper subset {spec['sha256']}")
print(f"{spec['file']}: sha256 matches the paper subset")
EOF
}

ROSTER_LINES="$(roster)"  # a failing roster() stops the script here (set -e)
mapfile -t TEACHERS <<< "$ROSTER_LINES"
[[ -n "$ROSTER_LINES" ]] || { echo "no teachers selected" >&2; exit 1; }
SHORTS=(); for line in "${TEACHERS[@]}"; do SHORTS+=("$(cut -f2 <<< "$line")"); done
OUT="$FEATURES_DIR/$SETTING"
cd "$GRACE_DIR"

case "$SETTING:$STAGE" in
gsm8k_llama1b:subset)
    "$FEAT_PY" scripts/generate_subset.py
    check_subset ;;
gsm8k_llama1b:generate)
    for line in "${TEACHERS[@]}"; do
        IFS=$'\t' read -r hf short extra <<< "$line"
        GEN_EXTRA_ARGS="$extra" bash scripts/run_teacher.sh "$hf" "$short" "$GPU" gen
    done ;;
gsm8k_llama1b:features)
    mkdir -p "$OUT"
    for line in "${TEACHERS[@]}"; do
        IFS=$'\t' read -r hf short extra <<< "$line"
        bash scripts/run_teacher.sh "$hf" "$short" "$GPU" post
        cp "grace_data/gsm8k_synthetic/Gradients_${short}.pkl" "grace_data/gsm8k_synthetic/Prod_${short}.pkl" "$OUT/"
    done ;;
gsm8k_olmo1b:subset|gsm8k_olmo1b:generate)
    echo "gsm8k_olmo1b scores the GSM8K responses of gsm8k_llama1b; run that setting's $STAGE stage." ;;
gsm8k_olmo1b:features)
    [[ ${#REQUESTED[@]} -eq 0 ]] || { echo "gsm8k_olmo1b features always cover its ten teachers; drop the names" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$FEAT_PY" scripts/featurize_olmo_student.py --out-dir "$OUT" --num-sms "$NUM_SMS" ;;
math_llama3b:subset)
    "$FEAT_PY" scripts/math_subset_512.py
    check_subset ;;
math_llama3b:generate)
    SPECS=(); for line in "${TEACHERS[@]}"; do SPECS+=("$(cut -f1 <<< "$line")=$(cut -f2 <<< "$line")"); done
    bash scripts/math_gen_queue.sh "$GPU" "${SPECS[@]}" ;;
math_llama3b:features)
    bash scripts/math_tokenize_all.sh "${SHORTS[@]}"
    export CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$FEAT_PY" scripts/featurize_math_prod.py --out-dir "$OUT" --teachers "${SHORTS[@]}" --num-sms "$NUM_SMS"
    # Full gradients of the fp32 3B student need a GPU with at least ~50 GiB.
    "$FEAT_PY" scripts/featurize_math_full.py --data-dir grace_data/math_synthetic --out-dir "$OUT" \
        --device cuda:0 --teachers "${SHORTS[@]}" --num-sms "$NUM_SMS" ;;
*)
    echo "unknown stage '$STAGE' for setting '$SETTING'" >&2; exit 1 ;;
esac

if [[ "$STAGE" == features ]]; then
    echo "banks in $OUT; score them with:"
    echo "  python $REPO/grace/score.py grace --setting $SETTING --features-dir $OUT --grace-repo $GRACE_DIR --out <out.json> --check"
fi
