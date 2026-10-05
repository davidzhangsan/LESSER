# GradAlign with LESSER features

The RL experiments of the paper run the released GradAlign pipeline (Yang et al., 2026) and replace only
the selection feature. GradAlign has no license file, so this directory ships a patch, not the code.

- Upstream: https://github.com/StigLidu/GradAlign, commit `7e25218` ("Update README.md", 2026-03-04).
- `lesser_gradalign.patch` turns that commit into the code that the recipes of `rl/` run
  (`automated/dynamic_selection.py` with `--mode prod`, `sim`, `acc` or `rand`). It contains only changed
  files that this pipeline executes, plus the README that `pip install -e ./verl` needs: 18 files, 805
  insertions, 82 deletions; sha256 `0f26d65022ea380d86474dc707757330cfda706c769384d23d3e6fcf38ddc65d`.
- `environment.txt` lists the Python packages of the virtual environment that ran the experiments.

```bash
git clone https://github.com/StigLidu/GradAlign.git && cd GradAlign
git checkout 7e25218
git apply /path/to/lesser/third_party/GradAlign/lesser_gradalign.patch
```

The patch applies cleanly (`git apply --check`) to a fresh checkout of `7e25218`. Every file the pipeline
executes and the patch does not list is unchanged from `7e25218`. Changes for running on a single node are
tagged `env: single-node adaptation` in the code.

## What the patch contains

The patch changes the orchestrator and its stages (`automated/`), the rollout judge
(`select/inference_ray_batch.py`), the scoring stage (`select/parallel/`) and the verl training code (`verl/`).
Two related pieces are not part of it:

- options of `automated/mix.py` that build the corrupted-reward pool; `rl/build_pools.py` reimplements that step
  and rebuilds all pools from the upstream datasets;
- `verl/verl/utils/reward_score/noisy_math_reward.py`, the verl training reward for the corrupted-reward pool,
  which `rl/run_corrupted.sh` names but never loads, because those runs stop after the first selection.

## Changes and what they do

### The LESSER feature (the intervention)

- `select/parallel/prod_feature.py` (new). For a problem with sampled responses, the released scorer's loss
  is `L = 8 * sum_t (-A_i(t) * ratio_t * mask_t) / sum(mask)` per mini-batch. Its gradient with respect to the
  output layer is `G = sum_t w_t (softmax(z_t) - e_{y_t}) h_t^T` with `w_t = 8 A_i(t) mask_t / sum(mask)`,
  where `h_t` is the final hidden state and `z_t` the logits. The module computes `G` from one forward pass,
  with the released mini-batching, per-mini-batch normalization, the extra one-position mask shift and the
  zero-advantage rule, and returns the flattened feature, all-reduce summed across ranks.
  - `GRADALIGN_PROD_SKETCH=1`, which both recipes set for LESSER: the two-sided Rademacher
    projection `R_v^T G R_h` with `64 x 128` entries (one CPU generator seeded 0 draws the vocabulary map
    first), compared by cosine. This equals `lesser.projection.RL_PAPER` bit for bit.
  - Without the variable the module returns the exact `V x d` block (`151,936 x 1,536` for
    Qwen2.5-Math-1.5B); the recipes do not use this mode.
- `select/parallel/grpo_grad_analyze_parallel.py`: `mode='prod'` computes the validation feature and one cosine
  per candidate without any backward pass; zero-advantage groups score 0 as in the released path. An FSDP
  warm-up forward and backward precedes the first no-grad forward (FSDP lazy initialization fails otherwise).
  A single-process run sorts its items like the multi-process gather.
- `automated/dynamic_selection.py`, `automated/run_parallel_analysis.py`,
  `select/parallel/analyze_grpo_parallel.py`, `select/parallel/launch_parallel_analysis.py`: `--mode prod`
  is passed through; LESSER writes its score into the `similarity` field and is selected exactly like the
  released `sim` mode (top `chunk_size / k` problems).

### Needed for the paper settings

- `select/inference_ray_batch.py`: the rollout judge scores Countdown answers with the repository's own
  `countdown_judge.py` (the generic string comparison cannot check expressions). Problems whose data source
  is `*_corrupted` get a fair-coin reward per response, hashed from `GRADALIGN_CORRUPT_SEED`, the group and the
  sample id; only the corrupted-reward recipe uses this branch.
- `verl/verl/utils/reward_score/countdown_mixed_reward.py` (new): training reward for the Countdown mixture
  (Countdown judge for Countdown, the released last-boxed rule for the rest; an expression the judge cannot
  evaluate scores 0 instead of crashing verl).
- `automated/launch_verl_training.py`: `GRADALIGN_ROLLOUT_N` (GRPO group size; code default 32, paper 128)
  and `GRADALIGN_USE_KL` (code default on, paper off).
- `automated/config.py`: model keys `qwen2.5-1.5b-math` (Qwen/Qwen2.5-Math-1.5B-Instruct) and `qwen3-8b-base`
  (Qwen/Qwen3-8B-Base).
- `automated/dynamic_selection.py`: `GRADALIGN_STOP_AFTER_SELECT=1` ends a run after its first selection
  (corrupted-reward recipe); the analysis-completion check counts one score row per problem (the released
  check expected one per response, so a resumed run always rescored).
- `verl/README.md` (new): a stub, because `verl/setup.py` reads it during `pip install -e ./verl`.

### Running outside the GradAlign authors' 8-GPU nodes

These changes affect where and how fast the pipeline runs, not what it computes, with one exception noted
below (the number of analysis ranks).

- Paths: data, responses and checkpoints live under the checkout (`data_local/`); verl config and reward files
  come from the vendored `verl/` instead of `~/data_selection/...`; Hugging Face hub ids are accepted as model
  paths.
- Parallelism: analysis ranks `--num_gpus` (released: 8, hard-coded), training GPUs `GRADALIGN_TRAIN_GPUS`
  (released: 8), rollout tensor parallel 1, FSDP instead of Megatron, no `x8` scaling of the per-GPU
  micro-batch, a non-FSDP accelerate config for a single analysis process (FSDP with one process crashes under
  torch 2.7.1). The exception: the analysis rank count changes which responses share a scoring mini-batch
  and therefore the features of both scorers slightly; see `rl/README.md`.
- Memory and throughput switches (some defaults differ from the release, e.g. log-prob micro-batch 8 instead
  of 40 and eager vLLM; none changes the computation beyond floating-point rounding): `GRADALIGN_VLLM_UTIL`,
  `GRADALIGN_INFER_CONCURRENCY`, `GRADALIGN_BATCHED_TOKENS`, `GRADALIGN_PPO_MICRO`,
  `GRADALIGN_LOGPROB_MICRO`, `GRADALIGN_FSDP_OFFLOAD`, `GRADALIGN_ENFORCE_EAGER`, `GRADALIGN_DYNBSZ`,
  `GRADALIGN_REMOVE_PADDING`, `GRADALIGN_FUSED_KERNELS`, `GRADALIGN_PPO_MAX_TOKEN_LEN`,
  `GRADALIGN_LOGPROB_MAX_TOKEN_LEN`, `GRADALIGN_RAY_CPUS`, `GRADALIGN_PREPARE_WORKERS`,
  `GRADALIGN_ANALYSIS_CPU_OFFLOAD`, `GRADALIGN_ANALYSIS_NO_CPU_OFFLOAD`; vLLM swap space 0; no validation
  before training (the curve starts at step 5). `GRADALIGN_PPO_MINI` sets the PPO mini-batch (default 32 as
  released; other values change the optimization).
- Checkpoints: verl saves every 10 steps, the end of a Countdown round (`GRADALIGN_SAVE_FREQ`; released: 2),
  with a process-group timeout of 10,800 s (`GRADALIGN_NCCL_TIMEOUT`; verl's default of 600 s can expire
  during saves to slow storage). Neither default changes the training computation.
- Robustness: the released stage cleanup killed every Python process on the machine; it now waits for the
  GPUs to drain and terminates only orphaned stage processes started from this checkout
  (`GRADALIGN_REAP_ANY_ORPHAN=1` drops the virtual-environment check for dedicated containers). Failed stages
  can be retried (`GRADALIGN_MAX_RETRIES`, default 0); a host-memory gate delays verl launches
  (`GRADALIGN_MIN_AVAIL_GB`, default 32 GiB, 0 disables); `GRADALIGN_SKIP_ROLLOUTS_FOR_RAND=1` skips the rollouts
  that the random selector never reads.
- verl: attention falls back to SDPA when flash-attn is missing (`VERL_ATTN_IMPL` overrides); the policy
  entropy (logged only; entropy coefficient 0) is computed in chunks of 512 positions (a full-length fp32
  softmax needs about 70 GB; largest difference 7.6e-6); the hard-coded `Gemma3DecoderLayer` FSDP wrap class
  is removed; a ray >= 2.5x import rename is handled.

## Environment

The experiments ran with Python 3.11.16, torch 2.7.1 (CUDA 12.6), vllm 0.10.0, ray 2.47.1,
transformers 4.53.2, tokenizers 0.21.4, accelerate 1.8.1, flash-attn 2.8.3 and the vendored verl installed
with `pip install -e ./verl`; `environment.txt` has the full list. transformers 5.x breaks vLLM 0.10's
tokenizer handling. One site-packages edit was needed: vLLM 0.10 installs uvloop's event-loop policy, whose
`get_event_loop()` refuses to create a loop, and ray 2.47.1's `ray/_common/utils.py:get_or_create_event_loop`
relies on that. Replace its final `return asyncio.get_event_loop_policy().get_event_loop()` branch (Python
3.10 and later) with:

```python
            try:
                return asyncio.get_event_loop_policy().get_event_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop_policy().new_event_loop()
                asyncio.set_event_loop(loop)
                return loop
```

Models: `Qwen/Qwen2.5-Math-1.5B-Instruct` at revision `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35` and
`Qwen/Qwen3-8B-Base` at revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
