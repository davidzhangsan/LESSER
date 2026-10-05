# LESSER: Post-Training Data Selection with Output-Layer Gradients

LESSER replaces the full-parameter gradient features of gradient-based data selectors with the gradient
of the output layer, which a forward pass already provides:

    g_p = sum_t alpha_t (softmax(W h_t) - e_{y_t}) h_t^T

where h_t is the final hidden state that predicts token y_t and W is the output projection. The feature is
projected with two small random matrices and plugged into an existing selector unchanged: LESS-style
round-robin selection for SFT, GradAlign for online RL, and GRACE for distillation teacher selection.

This repository contains the `lesser` package and everything needed to reproduce each result in the paper.

## Quick start

```bash
pip install -e ".[sft,figures]"          # Python >= 3.10, PyTorch >= 2.1
python verify_all.py                     # regenerate every paper table, figure and number (CPU, < 1 minute)
```

`verify_all.py` runs the unit tests, regenerates each result from the released data in `data/`, and compares it
with the values reported in the paper; it exits non-zero on any unexplained difference. Add `--verbose` to print
each component's full output, and `--paper-dir <paper LaTeX sources>` to also compare against the LaTeX sources and
figure PDFs. Run all commands from the repository root.

## Using LESSER on your own data

```python
import torch
from lesser import SFT_PAPER, TwoSidedProjection, l2_normalize, select, token_factors

W = model.get_output_embeddings().weight                     # (vocabulary, hidden)
proj = TwoSidedProjection(W.shape[0], W.shape[1], SFT_PAPER, device=W.device)

@torch.no_grad()
def lesser_feature(input_ids, labels):                       # labels: -100 on prompt tokens
    hidden = model(input_ids=input_ids[None], output_hidden_states=True).hidden_states[-1][0]
    h, r = token_factors(hidden, W, labels)                  # per-token factors of Proposition 1
    return l2_normalize(proj.project(h, r))                  # 8,192-dimensional feature

subsets = select(pool_features, query_features, budgets=(1000, 5000, 10000))   # round-robin
```

`lesser.checks.check_exactness` compares the forward-only gradient with autograd on a readout copy of `W`.
`RL_PAPER` reproduces the projection used in the RL experiments; `sft/extract.py` is the full SFT
extraction pipeline (masking, sharding, resuming, quality checks).

## Repository layout

| Path | Contents |
|---|---|
| `lesser/` | Output-layer gradient (Proposition 1), two-sided projection presets, cosine scoring and round-robin selection, exactness check |
| `sft/` | SFT pipeline: feature extraction, selection, fine-tuning and evaluation, similarity bins, Tables 9–11 and Figures 3–4 |
| `rl/` | GradAlign runs on Countdown and under corrupted rewards, Figures 5–6 |
| `grace/` | GRACE teacher ranking with LESSER features, Table 12 |
| `analysis/` | Section 5 and Appendices E–F: per-example audit, batch-gradient alignment, training trajectories, selection overlap |
| `cost/` | FLOP accounting and timing measurements: Table 1, Appendix A.4 |
| `third_party/` | Pinned upstream repositories and the patches applied to them |
| `data/` | Released selections, per-run results, logs and audit outputs used to regenerate every number |

## Paper results

Each component README maps its results to commands and describes how to rerun the underlying
experiments on GPUs.

| Result | Component | Regenerate from released data |
|---|---|---|
| Section 4 SFT results, Figure 3, Tables 9–11, Appendix C.1 | [`sft/`](sft/README.md) | `python -m sft.paper_tables`, `python -m sft.paper_figures` |
| Figure 4 and Appendix C bin-loss figures | [`sft/`](sft/README.md) | `python -m sft.paper_figures` |
| Appendix A.1 replication table, Appendix E overlap table | [`sft/`](sft/README.md) | `python -m sft.paper_tables` |
| Figure 5 (corrupted rewards), Figure 6 (Countdown), Appendix F.4 RL agreement | [`rl/`](rl/README.md) | `python rl/figures.py all` |
| Table 12 and Appendix F.4 GRACE statistics | [`grace/`](grace/README.md) | `python grace/tables.py` |
| Table 1, Appendix A.4 cost tables | [`cost/`](cost/README.md) | `python -m cost.table1` |
| Section 5, Figure 7, Appendices E–F | [`analysis/`](analysis/README.md) | `python -m analysis.verify` |

## Data

`data/<component>/` holds the small inputs that regenerate every number: selections, per-run results, logs and
measurement outputs, each with a manifest of sha256 checksums. Large intermediate artifacts (pool feature matrices,
gradient sketches, trained policies, feature banks) are not distributed; they are needed only to rerun intermediate
stages on GPUs, and the component READMEs describe the stages that produce them. The problem pools of the RL runs are not
shipped either: `python rl/build_pools.py --out <dir>` rebuilds them from pinned upstream datasets and checks every
file against `data/rl/pools/SHA256SUMS`.

## Third-party code

`third_party/` pins targeted-instruction-selection (Apache-2.0), GradAlign and GRACE at the upstream
commits the experiments used, and provides the patches that add LESSER features to them. GradAlign and
GRACE are distributed as patches because their upstream repositories carry no license.

## Reproducibility notes

* Selections regenerated from stored features match the released ones exactly as sets; a few within-set
  positions can swap on different hardware because of exact or near-exact score ties.
* Re-extracting features on other hardware gives cosine similarity above 0.999 to the stored features.
* The similarity bins reproduce exactly under numpy < 2; numpy 2 breaks ties differently.
* The checks compare with the paper source at revision df9fc85.

## License

The code is released under the MIT License (`LICENSE`). Third-party code in `third_party/` and rows of
third-party datasets in `data/` keep their own licenses, which each component's README lists, including the
sources that declare no license.
