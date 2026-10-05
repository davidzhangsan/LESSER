# RL experiments (GradAlign with LESSER features)

This directory reproduces the RL results of the paper; the RL cost numbers are reproduced by the cost
component (below). LESSER replaces GradAlign's full policy-gradient feature with the output-layer gradient
of the same loss; the rest of the released GradAlign pipeline is unchanged. The Countdown recipe projects
the feature to 64 x 128 entries with two random matrices (`GRADALIGN_PROD_SKETCH=1`, identical to
`lesser.projection.RL_PAPER`). Two tiers:

- Tier 1 (CPU, about a minute): recompute the numbers and redraw Figures 5 and 6 from the logged data in
  `data/rl`. The regenerated figures carry the same data payload and the same drawn page content as the
  paper figures.
- Tier 2 (GPUs): rerun the experiments with the patched GradAlign (`third_party/GradAlign`).

| file | purpose |
|---|---|
| `figures.py` | tier 1: Figures 5 and 6 and the RL numbers of the text and Appendix F.4; `verify` compares figures |
| `run_countdown.sh` | tier 2: one online-RL run on the Countdown mixture (Figure 6) |
| `run_corrupted.sh` | tier 2: one selection under corrupted rewards (Figure 5) |
| `build_pools.py` | tier 2: rebuilds the six problem pools of the runs from the upstream datasets and verifies them |
| `_common.sh` | shared helpers of the two recipes (environment checks, pool check, round-0 donor copy) |

The RL cost results (FLOPs and stored feature size per scored token, and the scoring wall clock of the cost
table) belong to the cost component: `python -m cost.rl_flops` and `python -m cost.timing_rl.parse_logs`
(see `cost/README.md`; the timed job ran the scoring stage of this patched GradAlign).

## Paper results and how to reproduce them

| paper result | value | tier 1 | tier 2 |
|---|---|---|---|
| Figure 5, Finding 3, Appendix "Selection under corrupted rewards": corrupted share of the 128 selected problems | LESSER 29.7% (38), full gradients 27.3% (35), AccGreedy 86.7% (111), Random 50% (64 expected) | `figures.py corrupted` | `run_corrupted.sh lesser`, then `run_corrupted.sh full` |
| Figure 6, Finding 4: held-out Countdown accuracy, two-seed means | step 50: LESSER 32.46, full gradients 32.46 (dynamic sampling 31.66, Random 29.79) | `figures.py training` | `run_countdown.sh {lesser,full,dynamic,random} {42,43}` |
| Finding 4: LESSER reaches Random's final accuracy after 15 steps, 70% fewer GRPO steps | first evaluation at or above 29.79: LESSER 15, full 15, dynamic sampling 30, Random 50 | `figures.py training` | as above |
| Experiments, RL paragraph: Countdown share of the pool | 4,000 of 34,000 = 11.8% | `figures.py training` | `build_pools.py --only cdmix4` |
| Appendix F.4, analysis section, introduction: agreement in the first round | 1,835 of 5,120 problems with nonzero advantage; Spearman 0.69; 69% shared at 256; Jaccard 0.53 vs 0.075 for two random selections | `figures.py agreement` | not included, see below |
| cost table (b), appendix RL cost paragraphs, introduction | 3.11e9 vs 9.26e9 FLOPs per scored token (3.0x); 32.8 kB vs 6.2 GB per feature; 18.1 vs 63.3 minutes of scoring per round (3.5x) | `python -m cost.rl_flops`, `python -m cost.timing_rl.parse_logs` | see `cost/README.md` |

Both recipes run LESSER with the projected feature (64 x 128 two-sided sketch, `GRADALIGN_PROD_SKETCH=1`),
as do the runs behind `data/rl` (see "Run settings" below).

## Tier 1: regenerate from logged data

Requirements: Python 3.10 or later with numpy, reportlab, tensorboard and pypdf
(`pip install -e ".[figures]"` from the repository root).

```bash
python rl/figures.py all            # all RL numbers; figures and JSON manifests in outputs/rl/
```

`all` prints the numbers of the table above, writes `outputs/rl/figure5_corrupted_rewards.pdf`,
`outputs/rl/figure6_countdown_training.pdf`, their JSON manifests, `appendix_f4_rl_agreement.json` and
`rl_numbers.json`, and exits non-zero unless both figure payloads equal the paper's and the 21 RL numbers
of the table above (except cost) are reproduced at their printed precision (`data/rl/expected/`).
Single parts: `corrupted`, `training`, `agreement`.

To compare with the figure files of the paper sources (their metadata was stripped for anonymity, so only the
drawn page content is compared):

```bash
python rl/figures.py verify outputs/rl/figure5_corrupted_rewards.pdf <paper figure 5>.pdf
python rl/figures.py verify outputs/rl/figure6_countdown_training.pdf <paper figure 6>.pdf
```

The reference payloads cover every evaluation, the seed-43 replicates, the per-round Countdown shares and the
SHA-256 of every input file. TensorBoard names its files `events.out.tfevents.<time>.<host>.<pid>.<n>`; the
copies in `data/rl` carry the host name `node`, and payloads are compared with every host name replaced by
`node` (`data/rl/MANIFEST.md`).

## Tier 2: rerun the experiments

### Setup

1. GradAlign with the LESSER patch (`third_party/GradAlign/README.md` lists every change):

   ```bash
   git clone https://github.com/StigLidu/GradAlign.git && cd GradAlign && git checkout 7e25218
   git apply /path/to/lesser/third_party/GradAlign/lesser_gradalign.patch
   ```

2. A virtual environment at `GradAlign/.venv` with the versions of `third_party/GradAlign/environment.txt`
   (Python 3.11, torch 2.7.1 for CUDA 12.6, vllm 0.10.0, ray 2.47.1, transformers 4.53.2, tokenizers 0.21.4,
   flash-attn 2.8.3, and `pip install -e ./verl`), plus the small ray fix described in that README.
   Containers need a CUDA devel image (vLLM compiles a helper at engine start) and a large `/dev/shm`.
3. The models at the revisions the paper used (the recipes load them from the Hugging Face cache):

   ```bash
   huggingface-cli download Qwen/Qwen2.5-Math-1.5B-Instruct --revision aafeb0fc6f22cbf0eaeed126eff8be45b0360a35
   huggingface-cli download Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4
   ```

4. The problem pools, rebuilt from the upstream datasets (see "Problem pools" below; about 430 MB of
   downloads, then under a minute on a CPU):

   ```bash
   python rl/build_pools.py --out $GRADALIGN_DIR/data_local/data     # run with the GradAlign environment
   ```

5. Environment for the recipes: `GRADALIGN_DIR=/path/to/GradAlign` (required); `RUN_ROOT` for rollouts,
   scores, selections and checkpoints (default `$GRADALIGN_DIR/data_local/chkpt`; a different directory is
   linked from there because the released trainer always writes there); `LOG_DIR` (default
   `$RUN_ROOT/logs`). `DRY_RUN=1` prints the pipeline command and environment without running anything. Each
   recipe first checks the pools it needs (`build_pools.py --check`) and stops if one is missing or differs.

Every recipe resumes when rerun with the same arguments (chunks, rollouts, scores, selections and the verl
checkpoints are reused). verl saves a checkpoint every 10 steps, at the end of each round
(`GRADALIGN_SAVE_FREQ`), with a process-group timeout of 10,800 s (`GRADALIGN_NCCL_TIMEOUT`), so a restart
repeats the training steps of the interrupted round.

### Figure 6: online RL on Countdown

Qwen2.5-Math-1.5B-Instruct on the 34,000-problem mixture; five rounds, each scoring a 5,120-problem chunk
with 16 responses per candidate and per query (200 `countdown4_val` problems), selecting 256 problems and
training 10 GRPO steps (128 problems x 128 responses, AdamW 1e-6, no KL, at most 3,072 response tokens).
verl evaluates on the 500 held-out `countdown4_test` problems every 5 steps.

```bash
export GRADALIGN_DIR=/path/to/GradAlign RUN_ROOT=/big/disk/rl_runs
for SEED in 42 43; do
  rl/run_countdown.sh lesser  $SEED          # projected LESSER; samples the shared round-0 rollouts, trains 50 steps
  rl/run_countdown.sh full    $SEED          # the other methods copy the lesser run's round-0 rollouts
  rl/run_countdown.sh dynamic $SEED
  rl/run_countdown.sh random  $SEED
done
```

The methods of one seed can run in parallel once the `lesser` run has finished its round-0 rollouts
(`$RUN_ROOT/countdown-prod-s<SEED>_.../global_step_0/train_split/accuracy_by_problem.jsonl` exists).
`DONOR=<experiment dir>` makes any method, `lesser` included, copy round 0 from another run of the same
seed instead (see "Run settings"). One run takes about two days on two A100 80GB; rollouts (about 2 hours
per round) and training (about 6 hours per round) take most of it. Defaults: `NUM_GPUS=2` analysis ranks, `GRADALIGN_TRAIN_GPUS=2`,
`GRADALIGN_INFER_CONCURRENCY=2` vLLM engines. More than 7 training GPUs need `GRADALIGN_RAY_CPUS=16`.

To draw Figure 6 from new runs, collect the TensorBoard files and selections into the layout of `data/rl`:

```bash
NEW=my_rl; S=42
for ARM in prod full acc rand; do
  EXP=countdown-$ARM-s${S}_qwen2.5-1.5b-math_cdmix4_cdmix4_countdown4_val
  mkdir -p $NEW/events/seed$S/$ARM && cp $GRADALIGN_DIR/automated/tensorboard_log/Dynamic-restart/$EXP/events.out.tfevents.* $NEW/events/seed$S/$ARM/
  for d in $RUN_ROOT/$EXP/selected/iter_*_256; do mkdir -p $NEW/selections/$ARM/$(basename $d); cp $d/train.jsonl $NEW/selections/$ARM/$(basename $d)/; done
done   # repeat the events part for seed 43 (arms prod full acc rand)
python rl/figures.py training --events-root $NEW/events --selections-root $NEW/selections
```

The recipe's LESSER run is collected as `prod`.

### Figure 5: selection under corrupted rewards

Qwen3-8B-Base, the 512 DAPO candidates of the pool `dapo_noisy50_sub512` (256 with coin-flip rewards),
43 AMC22 queries, 64 responses per candidate and per query, one selection of 128 at the base model, no
training.

```bash
rl/run_corrupted.sh lesser   # rollouts and LESSER scores; 2 x A100 80GB, about 4 to 5 hours
rl/run_corrupted.sh full     # full-gradient scores on the same rollouts; about 4 hours on NUM_GPUS=8 x A100 80GB
```

The 8B full gradient needs four or more 80 GB ranks; the recipe sets `NCCL_P2P_LEVEL=SYS` (without it, NCCL
can fall back to shared-memory copies between GPUs, which made each backward pass about 13 times slower)
and turns FSDP CPU offload off. To draw Figure 5 from these runs, copy
`global_step_0/train_split/accuracy_by_problem.jsonl`, `global_step_0/train_split/part_0/similarity_results_cosine_real.jsonl`
and `selected/iter_0_{prod,sim}_128/train.jsonl` of the two experiments into `<dir>/{head,full}/`, the pool
`$GRADALIGN_DIR/data_local/data/dapo_noisy50_sub512/train.jsonl` into `<dir>/subpool/train.jsonl`, and run
`python rl/figures.py corrupted --corrupted-root <dir>`.

### Appendix F.4: agreement audit

The audit recomputed, at the base policy and in FP32, both cosines of every nonzero-advantage problem of the
shared seed-42 first round from its saved rollouts, on RTX 4090s (about 4 minutes per problem with the
one-step utility). The audit script is not part of the patch; its six result files are in
`data/rl/countdown/audit`, and `figures.py agreement` recomputes the reported numbers from them.

### Running in containers

The paper runs were container jobs: image `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel`, 2 A100 80GB, 16 CPUs,
192 GiB of memory, a large `/dev/shm`, the GradAlign checkout and `RUN_ROOT` on shared storage, and
`HF_HUB_OFFLINE=1` with a pre-filled Hugging Face cache. An interrupted job resumes from its last
completed stage and verl checkpoint when resubmitted with the same command. A run resumed on a different
number of training GPUs cannot load its verl checkpoint (the FSDP shards are per rank); resubmit with the same
`GRADALIGN_TRAIN_GPUS`.

## Problem pools

The runs read six problem pools from `$GRADALIGN_DIR/data_local/data/<name>/train.{jsonl,parquet}`. The pools
are not shipped: `build_pools.py` rebuilds them from four upstream Hugging Face datasets at pinned revisions,
with the procedures that built the originals, and checks every file against `data/rl/pools/SHA256SUMS`.

| pool | problems | role | procedure |
|---|---|---|---|
| `cdmix4` | 34,000: 4,000 Countdown, 20,000 WebInstruct, 10,000 DAPO | Figure 6 candidates | GradAlign's `prepare_data.py`, then the released `mix.py` |
| `countdown4_val` | 200 Countdown | Figure 6 selection queries | four-number Countdown problems shuffled with seed 42, after the 4,000 of `cdmix4` |
| `countdown4_test` | 500 Countdown | Figure 6 held-out evaluation | remaining distinct four-number problems, shuffled with seed 0 |
| `dapo_noisy50_sub512` | 512 DAPO, 256 with coin-flip rewards | Figure 5 candidates | deduplicated DAPO with a seeded half marked corrupted (the procedure of the original runs, reimplemented in `build_pools.py`), first 5,120-problem chunk (seed 42), 256 clean and 256 corrupted drawn with seed 0 |
| `amc22` | 43 AMC 12 problems of 2022 | Figure 5 selection queries | `prepare_data.py` |
| `amc23` | 40 AMC 12 problems of 2023 | verl validation set of the Figure 5 runs, never evaluated because they stop after selection | `prepare_data.py` |

- The released `mix.py` shuffles without a seed, so the order of `cdmix4` is shipped as
  `data/rl/pools/cdmix4_order.json`.
- Every JSONL file must be byte-identical. A Parquet file is byte-identical when it is written with the
  library versions of the original, recorded in `data/rl/pools/pools.json`: the GradAlign environment for
  four pools, pyarrow 21.0.0 with datasets 4.1.1 for `countdown4_test` and `amc23`. With other versions only
  the Parquet encoding can differ; the script then compares the content (columns, rows and values) with a
  recorded digest and prints a note. `--require-identical-bytes` turns that case into an error.
- `--check DIR` verifies existing files without building; the recipes run it before every run. It rejects a
  file with one changed byte and a Parquet file with one changed value.
- In the GradAlign environment, starting from an empty Hugging Face cache, the script reproduces all 12 JSONL
  files and the Parquet files of `cdmix4`, `countdown4_val`, `dapo_noisy50_sub512` and `amc22` byte for byte,
  and the `countdown4_test` and `amc23` Parquet files by content; with pyarrow 21.0.0 and datasets 4.1.1 those
  two are byte-identical as well.

Upstream data and licenses:

| key | dataset | revision | file | license |
|---|---|---|---|---|
| `countdown` | `Jiayi-Pan/Countdown-Tasks-3to4` | `408f70d177020686d34a56bba5952feb45aaaee4` | `data/train-00000-of-00001.parquet` | none stated on the dataset card |
| `webinstruct` | `TIGER-Lab/WebInstruct-verified-unfiltered` | `ac48a536b31d3cabb346e1139fb6c69ae631abd7` | `data/train-00000-of-00001.parquet` | Apache-2.0 |
| `dapo` | `BytedTsinghua-SIA/DAPO-Math-17k` | `65877096c24ffa7abc4e4fa5edb95cf3413a5674` | `data/dapo-math-17k.parquet` | Apache-2.0 |
| `amc` | `AI-MO/aimo-validation-amc` | `69d78a4a2c840e82d69af6bc742bda09005f6316` | `data/train-00000-of-00001.parquet` | Apache-2.0; the problems are from the AMC 12 competitions |

`--source-file KEY=PATH` uses a local copy of an upstream file (checked against its pinned SHA-256), and
`HF_HUB_OFFLINE=1` builds from an existing Hugging Face cache. The problems in `data/rl` (the selections in
`data/rl/countdown/selections` and the candidates in `data/rl/corrupted`) are rows of these datasets and
remain under their licenses.

## Run settings

All runs used the patched code of `third_party/GradAlign` and the arguments of the recipes. The analysis rank
count and the scoring mini-batch differ between run types:

| run | analysis ranks (`NUM_GPUS`) | scoring mini-batch |
|---|---|---|
| Countdown, LESSER and full gradients, seeds 42 and 43 | 2 | 2 |
| Countdown, dynamic sampling and Random | no scoring | no scoring |
| Corrupted rewards, LESSER | 2 | 4 (48 of the 512 candidates were scored with mini-batch 1) |
| Corrupted rewards, full gradients | 4 or 8 | 1 |

Why the analysis rank count matters: the released analyzer sorts a round's responses by problem and length,
deals them round-robin to the ranks, and normalizes the loss per mini-batch of two consecutive responses on
each rank. Changing the rank count changes which responses share a mini-batch and therefore the per-response
weights of both features (not with mini-batch 1). Keep it fixed within a comparison.

- Round 0 is shared through `DONOR`: all runs of one seed score the same round-0 rollouts, and the
  corrupted-reward full-gradient run scores the rollouts of the LESSER run. In the paper runs, the Countdown
  LESSER and full-gradient runs of each seed and the corrupted-reward LESSER run also took their round 0 from a
  donor run of the same setting. Round 0 is copied as responses, never as tokenized shards: the released shard
  check in `launch_parallel_analysis.py` only tests that `data_0` to `data_{n-1}` exist, so shards made for
  another rank count would silently drop responses. The recipes never copy tokenized shards and refuse to
  resume with mismatched shards.
- The released `rand` selector draws 256 problems uniformly from the whole 34,000-problem pool file, not from
  the round's chunk; the `rand` and `acc` draws are not seeded.
