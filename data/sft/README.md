# SFT data

Small inputs of the SFT experiments: selections, similarity bins and per-cell result records, plus Nayak et al.'s
released results. They are enough to regenerate every SFT table and figure of the paper (`sft/README.md`). The pool
and query feature files are not distributed; `python -m sft.extract` writes them to `features/<model>/` (git-ignored).

`MANIFEST.json` lists every data file with its sha256, size, source (for third-party files) and a note.

## Layout

| Path | Content |
|---|---|
| `selections/<model>/lesser/<task>_k<k>.json` | LESSER selections (pool indices in selection order; smaller budgets are prefixes) |
| `selections/<model>/{less,rds}/<task>_k<k>.json` | Nayak et al.'s released LESS and RDS+ selections: the `index` column of `Harvard-DCML/tis-subset-datasets-<model>`, configs `{less,rds}_rr_<task>_10000`, first k rows |
| `selections/<model>/random/mmlu_pro_k<k>.json` | the fixed uniform-random subsets of the five-shot MMLU-Pro reruns: `numpy.random.RandomState(42).permutation(197196)[:k]`, and for Llama-2-7B `numpy.random.default_rng(0).permutation(197196)[:k]` (Nayak et al.'s `selection/random.py --seed 0`). Nayak et al. do not release their Random subsets |
| `bins/<model>/<task>/bin<b>.json` | LESSER similarity bins (500 indices each); the Llama-2-7B lists were recovered from the cells' training files by exact match (pool rows are unique) |
| `results/lesser/<model>/<task>_k<k>_s<seed>.json` | LESSER cells, TyDiQA / GSM8K / Codex / BBH |
| `results/mmlu_pro_5shot/<model>/<method>_k<k>_s<seed>.json` | five-shot MMLU-Pro reruns of all four methods |
| `results/bins_ce/<model>/<task>_bin<b>.json` | mean dev-query cross-entropy after training on each LESSER bin |
| `results/replication/<model>/...` | Table 3 (Appendix A.1) reruns |
| `nayak/<model>/{budget_true_metric_budget,binning_ce_loss_quantile}.csv` | Nayak et al.'s released scores and bin losses |
| `reference/paper_tables.json`, `reference/paper_figures.json` | the paper's SFT tables as printed and the values embedded in its SFT figure PDFs (with a digest of each figure's drawing operators), extracted from the paper sources; `sft/paper_tables.py` and `sft/paper_figures.py` compare against them when the paper sources are not given |

Pool indices refer to `Harvard-DCML/tulu-v2-197K-processed` (train split, default order, 197,196 examples).

## Result records

The result records are the records written by the paper's runs. Their `model` and `method` fields use internal
tokens that `sft/results.py` maps to the names used here (for example `baseprod` = LESSER, `llama` = Llama-3.2-3B).
`env` names the kind of machine a cell ran on (`workstation` or `cluster`). Only the fields that identify a cell and
its score are read by the code; the others are kept as provenance.

## Licenses

| Files | License |
|---|---|
| `nayak/` | Apache License 2.0. Unmodified copies of `assets/plot_data/quantile_budget/<model>/` of https://github.com/dcml-lab/targeted-instruction-selection at commit `8ff397ad1c6e649465948a7ce847cb003140224c`, copyright the authors of that repository; license text in `third_party/targeted-instruction-selection/LICENSE`. |
| `selections/*/less/`, `selections/*/rds/` | No license declared by the source. The lists are the `index` column of Nayak et al.'s Hugging Face datasets `Harvard-DCML/tis-subset-datasets-*`, whose dataset cards declare no license. They contain only integer row indices into the Tulu V2 pool (the pool itself is derived from `allenai/tulu-v2-sft-mixture`, ODC-BY), no text. The code that produces them is Apache License 2.0 (repository above). |
| everything else | MIT License of this repository (`LICENSE` at the repository root). |
