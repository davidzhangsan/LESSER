# LESSER: SFT experiments

Code and released inputs for every SFT result of the paper: LESSER feature extraction, round-robin selection,
full fine-tuning and evaluation, the similarity-bin diagnostic, and regeneration of the SFT tables and figures.
All commands run from the repository root as modules (`python -m sft.<name>`; running a file directly fails on the
package-relative imports).

| Stage | Module | Hardware | Inputs | Outputs |
|---|---|---|---|---|
| 1. Features | `sft/extract.py` | GPU (self-test: CPU) | base model, Tulu pool, query sets | `pool_prod_*.pt`, `val_<task>_prod.pt` |
| 2. Selection | `sft/select.py` | CPU | features | `<task>_k<k>.json` |
| 3. Fine-tune + evaluate | `sft/train_eval.py` (+ `sft/mmlu_pro_5shot.py`) | GPU | a selection | one result record |
| 4. Similarity bins | `sft/bins.py` | CPU (build), GPU (cells) | features | bins, bin-cell records |
| 5. Tables and statistics | `sft/paper_tables.py` | CPU | `data/sft` | Tables 3, 9--11, 13 and quoted numbers |
| 6. Figures and bin statistics | `sft/paper_figures.py` | CPU | `data/sft` | Figures 3, 4, 8--10 and quoted numbers |

Shared code: `sft/common.py` (models, tasks, paths), `sft/results.py` (readers of the released records),
`sft/nayak_data.py` (Nayak et al.'s tokenization and loss masking), `sft/plot.py` (the figures' PDF renderer).

## Setup

- Stages 2, 5, 6 need `torch`, `numpy`, `scipy`, `reportlab`, `pypdf` (`pip install -e .[figures]`).
- Stages 1, 3, 4 also need `transformers`, `datasets` and, for stage 3, Nayak et al.'s repository with our patch plus
  vLLM and lm-eval: see `third_party/targeted-instruction-selection/README.md`. Models, the pool
  (`Harvard-DCML/tulu-v2-197K-processed`, train split, 197,196 examples) and the query sets
  (`Harvard-DCML/targeted-query-set-processed`, dev split: 9 TyDiQA, 70 MMLU-Pro, 8 GSM8K, 16 Codex, 81 BBH) come from
  the Hugging Face Hub. OLMo3-7B needs transformers >= 4.57.
- Pool features are large (about 6.5 GB per model, 197,196 x 8,192 float32) and are not distributed. They are needed
  only to rebuild selections and bins; `sft/extract.py` writes them to `data/sft/features/<model>/` (git-ignored),
  which `sft/select.py` and `sft/bins.py` read by default (or `--features-dir`, or `$LESSER_ARTIFACTS/sft/features`).

## Paper results and commands

Stages 5 and 6 compare their output with the paper and exit nonzero on any difference. By default they compare with the
paper's tables, figure values and quoted numbers recorded in `data/sft/reference/`; with `--paper-dir $PAPER` (the
paper's LaTeX source directory) they compare with the sources and figure PDFs themselves and check that the recorded
reference is current (`--write-reference` rewrites it). The recorded reference corresponds to the paper source at
revision df9fc85. Status is the outcome of running the command on the released data, with and without `--paper-dir`.

| Paper result | Command | Status |
|---|---|---|
| Tables 9--11: LESSER, LESS, RDS+, Random; 4 models x 5 tasks x 3 budgets, 3 seeds | `python -m sft.paper_tables [--paper-dir $PAPER]` | all three tables identical line by line |
| Mean absolute gap of LESSER to LESS / RDS+ / Random over 60 cells: 1.3 / 2.2 / 2.6 (Introduction, Figure 3 caption) | same | identical (computed from the printed values: 1.333 / 2.153 / 2.628; unrounded 1.335 / 2.153 / 2.625) |
| Appendix C.1 ranges: TyDiQA above Random by 0.5 to 9.6; within 0.9 of LESS on Qwen3/OLMo3; LESS ahead by 3.1 to 4.4 (k=1,000) and 0.8 to 1.2 (k=10,000) on the Llama models; MMLU-Pro within 1.3 | same | identical |
| Section 4: Llama-2 per-task comparison of LESSER and LESS; LESSER above Random in 38 of 60 cells | same (printed) | consistent (mean LESSER - LESS over budgets: TyDiQA -2.2, MMLU-Pro +0.2, GSM8K -0.1, Codex +4.3, BBH -1.7) |
| Table 3 (Appendix A.1): released value / our rerun; LESS within 0.9, Random up to 3.0 | same | identical (Llama-3.2-3B BBH Random: three-seed mean 47.149, printed 47.1) |
| Table 13 (Appendix E): LESSER vs LESS overlap at k=5,000; Llama-2 mean Jaccard 0.037 (Section 5) | same | identical |
| Figure 3: Llama-2-7B budget curves | `python -m sft.paper_figures [--paper-dir $PAPER]` | PDF drawing operators byte-identical; plotted means and SDs identical |
| Figure 4 and Figures 8--10: query loss per similarity bin | same | PDF drawing operators byte-identical; plotted losses identical |
| Bin statistics: Llama-2 mean Spearman 0.86; 19 of 20 positive; 0.714 / 0.887 / 0.058 for LESSER / LESS / RDS+; closest bin lowest in 18; OLMo3/MMLU-Pro -0.030 | same | identical |
| LESSER selections behind all of the above (`data/sft/selections/<model>/lesser`) | `python -m sft.select --model M --out-dir OUT --compare-to data/sft/selections/M/lesser` | Llama-3.2-3B and Qwen3-4B-Base (CPU, stored features): all 15 sets identical; order identical at k=1,000 except 2 positions (Qwen3 GSM8K), and at most 36 swapped positions per longer list, all between candidates whose scores tie or differ by at most 3e-7 (the paper scored on a GPU). Llama-2-7B and OLMo3-7B: not re-verified (the paper's pool features are not distributed) |
| LESSER similarity bins (`data/sft/bins`) | `python -m sft.bins build --model M --out-dir OUT` then `python -m sft.bins compare --model M --built OUT` | Llama-3.2-3B and Qwen3-4B-Base: all 100 bins identical, in order (numpy 1.26); see "Reproducibility" for other numpy versions. Llama-2-7B and OLMo3-7B: not re-verified (pool features not distributed) |
| LESSER features | `python -m sft.extract pool --model M --out-dir OUT` and `... queries ...` | not re-run on a GPU. `self-test` passes. `compare --device cpu --compute-dtype float32` against the stored features: cosine 0.99957--0.99996 over 86 examples (pool examples and queries of Llama-3.2-3B and Qwen3-4B-Base; queries of Llama-2-7B and OLMo3-7B); examples without response tokens are zero, as stored |
| Per-cell scores (LESSER cells, five-shot MMLU-Pro reruns, Table 3 reruns) | `python -m sft.train_eval --model M --method {lesser,less,rds,random} --task T --budget K --seed S` | not re-run (GPU); `--dry-run` prints the exact commands |
| Table 3 reruns | Llama-3.2-3B: `... --model llama-3.2-3b --method {less,random-draw} --task T --budget 1000 --seed S --gsm-chat-stop blank_line`; Qwen3/OLMo3: `... --method less --task tydiqa --budget 1000 --seed 0` | not re-run (GPU) |
| Bin cells (Figures 4, 8--10) | `python -m sft.bins cell --model M --task T --bin B` | not re-run (GPU) |

Per-model feature extraction for the pool, as in the paper (one GPU; shard with `--start/--end` across GPUs):

```bash
python -m sft.extract pool    --model llama-2-7b --out-dir data/sft/features --device cuda:0
python -m sft.extract queries --model llama-2-7b --out-dir data/sft/features --device cuda:0
python -m sft.select --model llama-2-7b --out-dir outputs/sft/selections/llama-2-7b
python -m sft.extract self-test          # CPU, tiny random Llama, no downloads
```

## Released data (`data/sft`)

Selections, similarity bins, per-cell result records and Nayak et al.'s released results. Layout, provenance and
licenses are described in `data/sft/README.md`; every file is listed with its sha256 in `data/sft/MANIFEST.json`.

## Protocol summary

- Features: base model (no warmup), bf16 forward, fp32 feature arithmetic; the output-layer gradient of the summed
  response-token loss, projected with `lesser.SFT_PAPER` (128 x 64 = 8,192 entries) and L2-normalized. Pool and query
  features share the projection. Examples whose response is truncated away get zero features and are never selected
  (2 of 197,196 pool examples for Llama-3.2-3B, 1 for Qwen3-4B-Base).
- Selection: cosine similarity; round-robin over the task's queries until k examples; k in {1,000, 5,000, 10,000}.
- Training (Appendix A): full fine-tuning, 2 epochs, learning rate 2e-5, linear schedule, 3% warmup, weight decay 0,
  effective batch 128 (1 x 128 accumulation), bf16, maximum length 2,048; seeds 0, 1, 2.
- Evaluation: TyDiQA F1, GSM8K exact match, Codex pass@10, BBH exact match (Nayak et al.'s harness, vLLM); MMLU-Pro
  five-shot accuracy with lm-eval `mmlu_pro` on 12,032 questions for all four methods (retrained with the released
  selections; Random with the fixed subsets above).
- Bins: round-robin order of the whole pool, 10 equal consecutive bins, the first 500 of each; one training run per bin
  (seed 0); loss = mean over dev queries of the per-query mean token cross-entropy (`construct_test_sample`).

## Reproducibility

- Selections are reproducible exactly from the stored features (checked on CPU for Llama-3.2-3B and Qwen3-4B-Base).
  The within-selection order can differ at exact or near-exact float32 ties, because the paper scored on a GPU.
- The extractor reproduces the stored features up to floating-point differences: on a CPU (bf16 weights, fp32
  arithmetic) the cosine to the stored GPU bf16 features is 0.99957--0.99996 for all four models.
- Re-extracting features is not bit-identical across hardware and software (bf16 forward passes). A full re-extraction
  of the Llama-2-7B pool on another GPU type, combined with the paper's query features, gives selections that share
  95--98% of their examples with the released ones (Jaccard 0.949--0.984 over the 15 lists). Downstream numbers were
  obtained with the released selections.
- Llama-3.2-3B: the paper's tokenizer adds a `[PAD]` token and resizes the tied embedding and readout, which appends one
  row drawn from a normal distribution centred on the mean embedding with 1e-9 times the embedding covariance
  (transformers' mean resizing); its logit enters the softmax. The paper's run did not fix that draw;
  `sft/extract.py` seeds it (`--resize-seed`, default 0). The tokenizer itself equals the one the paper used
  (identical ids and masks on all 197,196 pool examples and all queries).
- Bins depend on the order of the whole pool, which is sensitive to float32 near-ties in the scores. `sft/bins.py`
  scores exactly as the paper's builder (torch row normalization, one numpy float32 product): with numpy 1.26
  (OpenBLAS) it reproduces all released Llama-3.2-3B and Qwen3-4B-Base bins exactly; with numpy 2.x, 0--6 of the 500
  examples of a bin differ.
- Nayak et al.'s code paths that differ between the paper's runs (pad-token handling, GSM8K stop sequence) are
  switches of `sft/train_eval.py`; see `third_party/targeted-instruction-selection/README.md`.
