# GRACE patch for the LESSER experiments

The GRACE results of the LESSER paper (Table 12 and Appendices A.3, A.4, F.4) run the released
GRACE code of Panigrahi et al. (*In Good GRACES*, ICLR 2026) with one change: the student's
gradient features can be replaced by LESSER features, the exact output-layer gradient computed in a
forward pass. The upstream repository has no license file, so this directory ships a patch against
a pinned upstream commit instead of a copy of the code. How the patched pipeline is driven, and what
the paper runs did, is described in [`grace/README.md`](../../grace/README.md).

## Pin, apply, set up

- Upstream: <https://github.com/abhishekpanigrahi1996/GRACE>
- Commit: `64fc99a10049f79abe05e2be638a178b448985d0` (2026-06-07, "modified") on upstream `main`.
- Patch: `lesser_grace.patch`, sha256
  `5998044b855811456c546c36f10ca8673f81138ea6f1874219fe194a1fbc4309`.

```bash
git clone https://github.com/abhishekpanigrahi1996/GRACE.git
cd GRACE
git checkout 64fc99a10049f79abe05e2be638a178b448985d0
git apply --check /path/to/lesser/third_party/GRACE/lesser_grace.patch
git apply /path/to/lesser/third_party/GRACE/lesser_grace.patch
bash scripts/setup_venv.sh
```

`scripts/setup_venv.sh` creates `.venv` with the versions of the paper runs (Python 3.11,
torch 2.6.0+cu124, transformers 4.51.1, datasets 4.1.1, traker 0.3.2) and builds fast-jl 0.1.3
from source against a CUDA 12.4 toolchain installed into a conda prefix. Set `CONDA_BASE`,
`PYTHON311`, and `FAST_JL_CUDA_ARCH` (default `8.9` for the RTX 4090 of the paper runs; `8.0` for
A100, `9.0` for H100) as needed. Teacher generation uses a separate environment with vLLM; the paper
runs used vLLM 0.11.0, and 0.16.0 for four GSM8K teachers.

## Contents of the patch

Modified upstream files:

| file | change |
|---|---|
| `GRACE/gradient_computation.py` | `--feature prod` computes the LESSER feature per response: `G = R^T H / n_sup` over the supervised positions (`r_t = softmax(z_t) - e_{y_t}`, `h_t` the final hidden state), from one forward pass without backward, flattened and projected with the same TRAK `CudaProjector` type as the full gradients (Rademacher, seed 0, `--proj-dim`), divided by `sqrt(V*d)`. `--feature both` writes full-gradient and LESSER features for the same rows in two passes per response. Also: time-chunked accumulation of `G`, gradient checkpointing for responses longer than 2,560 tokens, feature sanity checks before saving, and `--num-sms` (below). |
| `data_generation/generate_responses.py` | optional vLLM caps `--max_model_len` and `--max_num_seqs` for 24 GB GPUs |
| `.gitignore` | ignores the venv, the CUDA toolchain prefix, logs, and generated subsets |

New files:

| file | role | paper result |
|---|---|---|
| `scripts/setup_venv.sh` | environment recipe | all |
| `scripts/sanity_prod_gate.py` | checks that `G` equals autograd of the mean cross-entropy with respect to an untied copy of `lm_head.weight` (relative Frobenius error below 1e-3 on five GSM8K examples, fp32) | all |
| `scripts/memory_probe_both.py` | memory check of `--feature both` on 24 GB GPUs, and check that gradient checkpointing changes a projected full gradient by less than 1e-4 in relative L2 | all |
| `scripts/generate_subset.py` | 512-prompt GSM8K subset (seed 42, the selection rule of upstream `tokenize_data.py`) | Table 12 rows 1, 2; F.4 |
| `scripts/run_teacher.sh` | per-teacher driver for GSM8K with the Llama-3.2-1B student: generation, tokenization with the 4-of-16 response subsample, both feature types, GRACE scores | Table 12 row 1; F.4 |
| `scripts/featurize_olmo_student.py` | both feature types with the OLMo-2-1B student on the row-1 responses of the ten original teachers | Table 12 row 2 |
| `scripts/math_subset_512.py`, `scripts/math_gen_queue.sh`, `scripts/math_tokenize_all.sh` | 512-prompt MATH subset, generation, tokenization | Table 12 row 3 |
| `scripts/featurize_math_prod.py`, `scripts/featurize_math_full.py` | LESSER and full-gradient features with the Llama-3.2-3B student | Table 12 row 3 |

Upstream files used unchanged: `GRACE/GRACE_computation.py` (the GRACE scorer, sha256
`db76b9702ef318895b141bb9eb7140ddb824523baa2dcacb86989a944a941458`, which `grace/score.py`
checks), `data_generation/tokenize_data.py`, `data/gsm8k/train.jsonl`, `data/math/train.jsonl`.

The patch contains only the code that the paper results use. The GRACE scoring and analysis
steps are done by `grace/score.py` and `grace/tables.py` of this release.

## The fast_jl SM count

trak's `CudaProjector` passes the GPU's streaming-multiprocessor count to fast_jl, and the
Rademacher signs fast_jl generates depend on it. The same seed therefore gives a different projection
on a GPU with a different SM count. All stored features were computed with 128, the SM count of the
RTX 4090: natively for every bank except the MATH full-gradient banks, which ran on an RTX PRO 6000
(188 SMs) with the count pinned to 128. The patch adds `--num-sms` to
`gradient_computation.py` (default: the GPU's own count, which is the upstream behavior) and to the
three `featurize_*.py` scripts (default 128); `run_teacher.sh` passes `NUM_SMS`, default 128. With
the count pinned, the timing run of Appendix A.4 on an A100 (108 SMs) reproduced stored full-gradient and
LESSER rows of all three settings with cosine similarity at least 0.99997 (field `check` of
`data/cost/timing_grace/timing.json` in this release).

## Notes on the patched code

- The patched code performs the computation of the paper runs. The student is loaded as float32
  explicitly; under transformers 4.51.1 this gives the same weights as the default `--dtype auto`,
  and it protects against newer transformers releases that load the bf16 checkpoint dtype by
  default.
- `gradient_computation.py` also offers `--row-scale` (default `none`) and an `ablation` feature
  mode. No paper result uses them; with `--feature both|prod|full_grad` and `--row-scale none` the
  computation is the plain full-gradient and LESSER feature extraction described above.
- In `prod` mode the full-gradient projector is not built.
