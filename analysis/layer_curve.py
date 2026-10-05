"""Per-example rankings as decoder blocks are added to the output layer (Section 5.1, Appendix F.1).

Reported results (20 model-task pairs; audit records of ``audit.py``; candidates of at most 1,024 tokens
for Llama-2-7B and OLMo-3-7B, leaving 1,903-3,000 per pair):
  * Output-layer and full-gradient cosine scores have median Spearman correlation 0.52; per-model
    medians range from 0.36 (Llama-3.2-3B) to 0.72 (Llama-2-7B).
  * The median correlation with the full-gradient ranking rises to 0.65 with half of the decoder blocks
    and to 1.00 with all of them (the Appendix F.1 figure).

Score with j blocks added (from the last block toward the first): the cosine of the candidate's gradient
with G_Q over the included parameters,
    (S_head + sum of the j blocks' dots) / sqrt(|G^out|^2 + sum of the j blocks' squared norms),
ranked against the full-gradient cosine full_dot / full_norm. The ``inner`` curve ranks the unnormalized
inner products instead (a check on the per-block records). The median curve interpolates each pair's
curve linearly on 41 fractions of the blocks added.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit, common
from analysis.common import MODEL_KEYS, MODELS, PAIRS

MAX_TOKENS = {"llama2": 1024, "olmo3": 1024}   # the 7B audits keep candidates of at most 1,024 tokens
FRACTIONS = [i / 40 for i in range(41)]


def filtered_pair(model, task, audit_dir=audit.AUDIT_DIR) -> dict:
    """Audit arrays of one pair restricted to the analysed candidates."""
    d = audit.load_pair(model, task, audit_dir)
    keep = d["sequence_length"] <= MAX_TOKENS.get(model, 10 ** 9)
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == keep.shape[0] else v)
            for k, v in d.items()}


def curves(d) -> dict:
    """Spearman correlation with the full-gradient score for j = 0..L blocks added, late to early."""
    L = d["dots_per_layer"].shape[1]
    H, h2 = d["s_head"], d["norm_head"] ** 2
    D, N2 = d["dots_per_layer"], d["norms_per_layer"] ** 2
    fd, fc = d["full_dot"], d["full_dot"] / d["full_norm"]
    inner, cosine = [], []
    for j in range(L + 1):
        blocks = list(range(L - j, L))
        num = H + D[:, blocks].sum(1)
        den = np.sqrt(h2 + N2[:, blocks].sum(1))
        inner.append(common.spearman(num, fd))
        cosine.append(common.spearman(num / den, fc))
    return dict(n=int(len(H)), n_layers=int(L), inner=inner, cosine=cosine)


def interpolate(row, fraction, key="cosine"):
    position = fraction * row["n_layers"]
    lower = int(position)
    upper = min(lower + 1, row["n_layers"])
    weight = position - lower
    return row[key][lower] * (1 - weight) + row[key][upper] * weight


def median_curve(cells, key="cosine"):
    return [common.median(interpolate(row, f, key) for row in cells.values()) for f in FRACTIONS]


def compute(audit_dir=audit.AUDIT_DIR) -> dict:
    cells = {common.pair_key(m, t): curves(filtered_pair(m, t, audit_dir)) for m, t in PAIRS}
    med = median_curve(cells)
    first = {k: c["cosine"][0] for k, c in cells.items()}
    summary = {
        "n_candidates": {k: c["n"] for k, c in cells.items()},
        "output_layer_rank_corr": first,
        "output_layer_rank_corr_median": common.median(first.values()),
        "output_layer_rank_corr_model_median": {
            m: common.median(v for k, v in first.items() if k.startswith(m + "_")) for m in MODEL_KEYS},
        "rankcorr_median_at": {f"{f:.2f}": med[int(round(f * 40))] for f in (0, .25, .5, .75, 1)},
        "inner_median_at": {f"{f:.2f}": median_curve(cells, "inner")[int(round(f * 40))] for f in (0, .5, 1)},
    }
    return dict(cells=cells, summary=summary)


def figure(result, source_path, out):
    """Appendix F.1 layer curve: thin lines are pairs colored by model, the black line their median."""
    from analysis import vector_figures as vf

    layers = result["cells"]
    for key, row in layers.items():
        if len(row["cosine"]) != row["n_layers"] + 1 or not all(math.isfinite(v) and 0 <= v <= 1.06 for v in row["cosine"]):
            raise ValueError(f"invalid layer curve: {key}")
    medians = median_curve(layers)
    payload = {"generator": "analysis/layer_curve.py", "renderer": "reportlab-vector",
               "source": {"file": Path(source_path).name, "sha256": common.sha256_file(source_path)},
               "curves": layers, "rankcorr_median_at": {f"{f:.2f}": medians[int(f * 40)] for f in (0, .25, .5, .75, 1)}}
    entries = [(MODELS[m].short, MODELS[m].color, 1.3) for m in MODEL_KEYS] + [("Median", "#1A1A1A", 1.45)]
    legend_width = 14 + max(vf.string_width(label, "Helvetica", 8) for label, _, _ in entries)
    left, bottom = 30, 32
    right, top = left + 181.8, bottom + 76
    legend_x = right + 12
    width, height = legend_x + legend_width + 3, top + 6
    f = vf.Fig(out, width, height, payload)
    step, median_gap = 11.5, 4
    center = (bottom + top) / 2 + (step * (len(entries) - 1) + median_gap) / 2
    for label, color, weight in entries:
        if label == "Median":
            center -= median_gap
        f.line(legend_x, center, legend_x + 10, center, color, weight)
        f.text(legend_x + 14, center - 2.8, label, 8)
        center -= step
    xmap = lambda fraction: left + fraction * (right - left)
    ymap = lambda value: bottom + value / 1.06 * (top - bottom)
    for tick in (0, .5, 1):
        f.line(left, ymap(tick), right, ymap(tick))
        f.text(left - 5, ymap(tick) - 2.6, f"{tick:g}", 7.5, "right", color=vf.MUTED)
    for fraction, label in zip((0, .25, .5, .75, 1), ("0%", "25%", "50%", "75%", "100%")):
        f.line(xmap(fraction), bottom, xmap(fraction), bottom - 2.5, vf.RULE)
        f.text(xmap(fraction), bottom - 12, label, 7.5, "center", color=vf.MUTED)
    for key, row in layers.items():
        f.polyline([(xmap(i / row["n_layers"]), ymap(v)) for i, v in enumerate(row["cosine"])],
                   MODELS[key.split("_")[0]].color, .55, alpha=.6)
    f.polyline([(xmap(fr), ymap(v)) for fr, v in zip(FRACTIONS, medians)], "#1A1A1A", 1.45)
    f.vtext(8.2, (bottom + top) / 2, "Rank correlation", 8)
    f.text((left + right) / 2, 4.5, "Decoder blocks added, late to early", 8, "center")
    f.save()
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", default=str(audit.AUDIT_DIR))
    ap.add_argument("--out-dir", default=str(common.DEFAULT_OUT / "per_example"))
    ap.add_argument("--figure", default=str(common.DEFAULT_OUT / "figures" / "layer_curve.pdf"))
    args = ap.parse_args(argv)
    result = compute(args.audit_dir)
    out = common.write_json(result, Path(args.out_dir) / "layer_curve.json")
    figure(result, out, args.figure)
    s = result["summary"]
    print(f"output layer vs full gradient, median Spearman {s['output_layer_rank_corr_median']:.4f}; per model "
          + ", ".join(f"{MODELS[m].name} {v:.4f}" for m, v in s["output_layer_rank_corr_model_median"].items()))
    print("median rank correlation at 0/25/50/75/100% of blocks added:",
          " ".join(f"{v:.4f}" for v in s["rankcorr_median_at"].values()))
    print("wrote", out, "and", args.figure)


if __name__ == "__main__":
    main()
