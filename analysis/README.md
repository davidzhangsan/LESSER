# Analysis: Section 5 and Appendices E-F

This directory reproduces every analysis result of the paper: the per-example audit (Section 5.1,
Appendix F.1), the batch-gradient alignment (Section 5.2, Figure 7a, Appendix F.2), the training
trajectories (Section 5.3, Figure 7b, Appendix F.3) and the selection-overlap table (Appendix E).

Every reported number and figure is regenerated on CPU from the small inputs in `data/analysis`
(39 MB) and the selections in `data/sft/selections`. The GPU steps that produced those inputs are
included as well. Their raw outputs are too large to distribute; each pipeline below ends with the CPU
command that reduces them to the shipped inputs, and the shipped files record the sha256 of the raw files
they were built from.

## Setup

- Python 3.10 or newer with `numpy` and `torch`; figures need `reportlab` and `pypdf`:
  `pip install -e ".[figures]"` from the repository root. Run every command from the repository root.
- GPU steps also need `transformers` and `datasets` (`pip install -e ".[sft]"`) and the base models. Pool
  examples and queries are encoded by `sft/nayak_data.py`, the SFT component's copy of the encoders of
  [targeted-instruction-selection](https://github.com/dcml-lab/targeted-instruction-selection). Data sets:
  `Harvard-DCML/tulu-v2-197K-processed` (pool) and `Harvard-DCML/targeted-query-set-processed` (queries).
- The analysis reads the LESSER, LESS, RDS+ and Random selections from `data/sft/selections` (see below);
  their sha256 are recorded in `data/analysis/MANIFEST.json`.

## Check everything at once

```bash
python -m analysis.verify --paper-dir PATH/TO/PAPER_SOURCE
```

This regenerates all results into `outputs/analysis/` and prints one PASS/FAIL line per check (the
rounded numbers printed in the paper, the data stored in the paper's figure PDFs, and the original
analysis outputs the paper numbers were read from). With `--paper-dir`, it also renders the regenerated
and the paper figures at 200 dpi and compares pixels. Without it, all other checks still run. With the
shipped data, 43 of 43 checks pass (39 without `--paper-dir`), in about 5 seconds.

## Results and commands

All commands are CPU-only and read `data/analysis` (and `data/sft/selections`); outputs go to `outputs/analysis/`.

| Paper result | Value in the paper | Command | Regenerated |
|---|---|---|---|
| Sec. 5.1, App. F.1: Spearman correlation of output-layer and full-gradient cosine rankings | median 0.52 over 20 pairs; per-model medians 0.36-0.72 | `python -m analysis.layer_curve` | 0.5224; 0.3569 (Llama-3.2-3B) to 0.7242 (Llama-2-7B) |
| App. F.1: median correlation with half of the decoder blocks / all blocks | 0.65 / 1.00 | same | 0.6459 / 0.9999 |
| App. F.1 figure | | same, writes `figures/layer_curve.pdf` | stored curves identical; pixels identical |
| App. F.1: candidates per pair | 1,903-3,000 | same | 1,903-3,000 |
| App. F.1: corr(H, B); pairs with Delta_B > 0; median Delta_B / Delta_H (top 10% under the output layer) | 0.36; 19 of 20; 0.82 | `python -m analysis.hb_decomposition` | 0.3554; 19; 0.8167 |
| Sec. 5.2, Fig. 7a, App. F.2: centered alignment with LESS at batch size 128, medians | LESSER 0.29, RDS+ 0.03, Random about 0, LESS-LESS 0.53 | `python -m analysis.batch_alignment read` | 0.2873, 0.0273, -0.0064, 0.5274 |
| App. F.2: pairs with LESSER-LESS > 0 | 17 of 20 | same | 17 |
| App. F.2: raw cosine between random batches of 128 | median 0.81 | same | 0.8058 |
| App. F.2: median over pairs of (LESSER-LESS) / (LESS-LESS), batch sizes 64-512 | 0.58-0.66 | same | 0.576-0.658 (0.633 at 128) |
| Fig. 7a | | `python -m analysis.batch_alignment figure` | stored medians equal to 1e-16; pixels identical |
| Sec. 5.3, Fig. 7b: median share of the largest query-loss decrease reached by step 10 | LESSER 84%, LESS 82%, Random 35% | `python -m analysis.trajectories stats` | 0.8402, 0.8244, 0.3475 (20, 20, 19 pairs) |
| Fig. 7b | | `python -m analysis.trajectories figure` | stored medians identical; pixels identical |
| App. F.3 figure | | same | curves identical to the plotted curves; pixels identical |
| App. E table `tab:selection-overlap`; Sec. 5.1 Llama-2-7B mean Jaccard | e.g. Llama-2-7B TyDiQA 491 / 0.052; 0.037 | `python -m analysis.overlap --check-tex PAPER/contents/_appendix/additional_experiments.tex` | all four rows identical to the paper source |

## Pipelines, including the GPU steps

The CPU commands above start from `data/analysis`. The steps below regenerate that directory from raw
outputs. Every script documents its protocol in its docstring; `--help` lists the options.

**Selections (all analyses).** Five k = 5,000 lists per model-task pair (`analysis/selections.py`):
`lesser`, the released `less` and `rds` selections of Nayak et al. (2026), and `randA`, the paper's
Random subset (one list per model: `numpy.random.default_rng(0).permutation` for Llama-2-7B,
`numpy.random.RandomState(42).permutation` for the other models), are read from `data/sft/selections`;
`randB`, a second uniform pool sample, is stored in `data/analysis/selections/k5000/`.

**Appendix F.1, per-example audit** (`audit.py`, `layer_curve.py`, `hb_decomposition.py`).

```bash
# GPU, one pair per job: fp32 gradients of all parameters for 3,000 seeded candidates (seed 0; 7 for
# Llama-3.2-3B, as in the paper). The GPU holds the fp32 model and its gradient; G_Q and the per-block
# gradients live on --storage-device (default: the GPU; use cpu for the 7B models on 80 GB cards).
python -m analysis.audit compute --model llama3 --task gsm8k --model-path SNAPSHOT \
    --out runs/audit/llama3_gsm8k.json
python -m analysis.audit compact --pattern 'runs/audit/{model}_{task}.json'   # -> data/analysis/audit
# a pattern with glob characters, e.g. 'runs/audit/{model}_{task}_*_*.json', merges shards of one permutation
python -m analysis.layer_curve && python -m analysis.hb_decomposition
```

The paper's audit records were computed with an earlier implementation of the same audit (its sha256 is
recorded as `runner_sha256` in `data/analysis/audit/index.json`). `audit.py compute` is a focused port of its
all-parameter audit that stores the same quantities with the same arithmetic. It leaves out two quantities
that no reported number uses: the finite SGD step on each candidate and the matched RDS+ embedding.
`data/analysis/audit` was built from those records with `audit.py compact`; `audit/index.json` lists the
source files and their sha256. OLMo-3-7B was audited in shards of the seed-0 permutation (1,903-2,944
candidates per task after merging). `reference/audit_port_cpu_check.json` compares `compute` on CPU with the
paper's records for three Llama-3.2-3B GSM8K candidates (see the docstring of `audit.py`).

**Appendix F.2, batch-gradient alignment** (`batch_alignment.py`, shared CountSketch in `sketch.py`).

```bash
python -m analysis.batch_alignment batches      # CPU: the 1,024-example samples (stored in data/analysis)
# GPU, one pair per job: 7 samples x 16 chunks of 64 examples, bf16 base model; about 1 GB of sketches
python -m analysis.batch_alignment compute --model llama2 --task gsm8k --model-path SNAPSHOT \
    --out-dir runs/setgrad_llama2
python -m analysis.batch_alignment gram --sketch-root llama2=runs/setgrad_llama2 llama3=runs/setgrad_llama3 \
    qwen=runs/setgrad_qwen olmo3=runs/setgrad_olmo3   # -> data/analysis/batch_alignment/gram
python -m analysis.batch_alignment read && python -m analysis.batch_alignment figure
```

Every alignment is a function of the Gram matrix of the chunk sketches, so `data/analysis` stores the
Gram matrices (1.7 MB) with the exact chunk norms and dot products with the query gradient, which replace
their sketched values.

**Appendix F.3, training trajectories** (`trajectories.py`, trainer `train_sft_probe.py`).

```bash
# GPU, 120 runs: arms lesser and less with seeds 0 and 1; randA and randB with seed 0
python -m analysis.trajectories train --model llama2 --task gsm8k --arm lesser --seed 0 \
    --model-path SNAPSHOT --runs-dir runs/traj
python -m analysis.trajectories pack-loss --runs-dir runs/traj   # -> data/analysis/trajectories/loss
python -m analysis.trajectories stats && python -m analysis.trajectories figure
```

The recipe is the paper's SFT recipe: full fine-tuning in bf16, batch 128 (1 x 128 accumulation), two
epochs of 5,000 examples = 80 AdamW steps, learning rate 2e-5 with 3 linear warmup steps, no weight
decay. `train_sft_probe.py` follows `training/train_sft.py` of targeted-instruction-selection with two
differences: "[PAD]" is added only when the tokenizer has no pad token (as in the paper's SFT runs), and
the LoRA branch is omitted. Its query-loss callback scores the dev queries before training and after
every optimizer step.

**Appendix E, overlap.** `python -m analysis.overlap` reads the LESSER and LESS selections.

## Contents of `data/analysis`

| Path | Size | Contents |
|---|---|---|
| `audit/` | 31 MB | per-candidate audit records of the 20 pairs (`<model>_<task>.npz`) and `index.json` (source files, sha256, audit settings) |
| `batch_alignment/gram/` | 1.7 MB | Gram matrices of the chunk-gradient sketches; each file lists the sha256 of its sketch files |
| `batch_alignment/batches/` | 1.1 MB | index lists of the seven 1,024-example samples per pair |
| `selections/k5000/` | 0.7 MB | the 20 `randB` pool samples; the other selections are read from `data/sft/selections` |
| `trajectories/loss/` | 3.4 MB | the 120 `loss_trajectory.json` files, gzipped byte for byte; `index.json` has their sha256 |
| `reference/` | 0.3 MB | numbers as printed in the paper and the curves plotted in the Appendix F.3 figure, the plotted data stored in the paper's figure PDFs, the original analysis outputs, and the CPU check of the audit port |
| `MANIFEST.json` | | sha256, size and provenance of every file, and the sha256 of the `data/sft` selection files read here; `python -m analysis.verify manifest` checks both |

## Notes

- `reference/paper_figures` holds the plotted data stored in the paper's figure PDFs and the sha256 of each
  figure's inputs; `python -m analysis.verify payloads --paper-dir PAPER` extracts them again.
- The original analysis computed the F.1 rank correlations and the H/B split from a recomputation of the
  audit (a position decomposition of the same candidates). The release computes them from the audit records
  themselves; per-pair values differ by at most 4e-5 and every reported number is unchanged.
- The original analysis read the batch alignments with the same estimator as the release; the release
  reproduces them to 1e-12 or better.
- The Appendix F.3 figure stores no curves in its PDF, only the names and sha256 of the four files it was
  drawn from. The plotted curves (per pair, the mean query loss of the LESSER, LESS and Random runs at steps
  0-80) were extracted from those four files into `reference/paper_numbers.json`; the curves regenerated
  from `trajectories/loss` equal them exactly. The sha256 of the four files, one per model:
  - Llama-2-7B: `d496f4bd85d962e3b0f4a42126b27142984039ed4741f31cb0bb8395c1c5fd90`
  - Llama-3.2-3B: `6b2de4eddfa429fd6d64816f787d164e122d3def372db49eb71a2f174c2b597a`
  - Qwen3-4B-Base: `781f136d6fdeb7e52b76d4f2d1040c0395cfbf22cc4e6553f420ed928492c533`
  - OLMo3-7B: `ca2f05487e8ab50dbb8c9ba44f7a1811f82f9bdfbbd7036f8a9f69fb499549f3`
- The GPU steps encode with `sft/nayak_data.py`. It reproduces the encoding sha256 of all 12,000 audited
  candidates (3,000 per model) and all 736 query encodings (184 per model) stored in the paper's audit
  records, and matches the encoder used for the paper's runs on 4,000 random pool examples.
