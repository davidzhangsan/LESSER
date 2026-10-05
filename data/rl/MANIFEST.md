# data/rl: inputs of the RL results

Logged outputs of the paper's RL runs and the metadata that `rl/build_pools.py` needs to rebuild the problem
pools the runs read. `python rl/figures.py all`
recomputes every RL number of the paper from these files (see `rl/README.md`). `SHA256SUMS` lists every file
below; `pools/SHA256SUMS` lists the pool files that `rl/build_pools.py` rebuilds. Total size about 11 MB.
Rollouts, policies and caches are not included.

## Figure 6: online RL on Countdown

- `countdown/events/seed{42,43}/{prod,full,acc,rand}/events.out.tfevents.*`: verl TensorBoard files of the
  eight runs (LESSER with the projected feature, full gradients, dynamic sampling, Random; seeds 42 and 43).
  Every file of a run is kept, including those of resumed segments; the held-out curve is the scalar `val-core/countdown/reward/mean@16`
  (500 `countdown4_test` problems, 16 responses each, every 5 steps). A step logged twice after a resume keeps
  the later evaluation (the payload lists the three superseded Random evaluations). TensorBoard names a file
  `events.out.tfevents.<time>.<host>.<pid>.<n>`; here the host name is written as `node` (time stamp, process
  id and counter are kept), and `rl/figures.py` compares payloads with every host name replaced by `node`.
- `countdown/selections/seed42/{prod,full,acc,rand}/iter_<r>_<mode>_256/train.jsonl`: the 256 problems each
  seed-42 run selected at the start of round r (step 10r). The figure payload records their Countdown share.

## Appendix F.4: agreement of the two features in the first round

- `countdown/round0_seed42/accuracy_by_problem.jsonl`: pass rate of each of the 5,120 round-0 candidates
  (16 responses; shared by all seed-42 runs). 1,835 have a nonzero advantage.
- `countdown/audit/{subset,remaining}_shard{0,1,2}of3.json`: base-policy audit of those 1,835 problems
  (FP32, one RTX 4090 per shard; the audit script is not distributed): per problem the
  full-gradient cosine `cos_full`, the output-layer cosine `cos_head`, per-layer terms and a one-step utility;
  a 615-problem subset and its 1,220-problem complement, three shards each. The paths in
  `provenance.train_responses_dir`, `provenance.val_responses_dir` and `provenance.command` are relative to the
  audit's working directory.
- `countdown/audit/{subset,remaining}_groups.json`: problem ids of the two parts.

## Figure 5: selection under corrupted rewards

Same layout as the run's output directory:

- `corrupted/subpool/train.jsonl`: the 512 DAPO candidates, 256 with `extra_info.corrupted = true` (identical
  to the pool file `dapo_noisy50_sub512/train.jsonl` that `rl/build_pools.py` rebuilds).
- `corrupted/{head,full}/accuracy_by_problem.jsonl`: reward counts over 64 responses per candidate; both runs
  score the same rollouts (LESSER with the projected feature = `head`, full gradients = `full`).
- `corrupted/{head,full}/similarity_results_cosine_real.jsonl`: the two scores; each selection is the top 128.
- `corrupted/head/selected/iter_0_prod_128/train.jsonl`, `corrupted/full/selected/iter_0_sim_128/train.jsonl`:
  the 128 selected candidates.

## Problem pools (`pools/`)

The pools themselves are not included; `rl/build_pools.py` rebuilds them from four upstream Hugging Face
datasets at pinned revisions (sources, procedures, licenses and the verification record are in
`rl/README.md`, section "Problem pools"). This directory holds its metadata:

- `SHA256SUMS`: SHA-256 of the twelve pool files the runs read from GradAlign's `data_local/data/<name>/`
  (`train.jsonl` and `train.parquet` of `cdmix4`, `countdown4_val`, `countdown4_test`,
  `dapo_noisy50_sub512`, `amc22`, `amc23`).
- `pools.json`: the pinned upstream files (repository, revision, file, size, SHA-256) and their licenses; the
  SHA-256 of the four intermediate files the pools derive from (the Countdown training split, the
  `prepare_data.py` outputs of WebInstruct and DAPO, and the corrupted DAPO pool); and per pool file its size,
  a content digest of each Parquet file and the writer and library versions of the original.
- `cdmix4_order.json`: the order of the 34,000 problems of `cdmix4` (a permutation of the concatenated
  4,000 Countdown, 20,000 WebInstruct and 10,000 DAPO problems). The released `automated/mix.py` shuffles
  without a seed, so this order cannot be recomputed.

## Reference payloads (`expected/`)

`expected/figure5_payload.json` and `expected/figure6_payload.json`: the data payloads embedded in the paper
figures before their metadata was stripped, normalized by `rl/figures.py normalize_payload` (producer,
command and output locations removed; input paths reduced to file names or their last components; host names
in event file names replaced by `node`). `rl/figures.py all` compares the regenerated figures with them.
`expected/paper_numbers.json`: 21 RL numbers as printed in the paper text, which `rl/figures.py all` compares
at their printed precision (the RL cost numbers are checked by the cost component, `cost/`).
