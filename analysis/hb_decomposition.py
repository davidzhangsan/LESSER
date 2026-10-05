"""Output-layer and remaining-network contributions to the per-example score (Appendix F.1).

With G_p = [G_p^out; G_p^body] and G_Q likewise, the first-order score splits as
    G_Q^T G_p / |G_p| = H(p) + B(p),   H = <G_Q^out, G_p^out> / |G_p|,   B = full score - H,
so B is the remainder: the contribution of every parameter other than the output-layer readout. For models
with tied input and output matrices (Llama-3.2-3B, Qwen3-4B-Base), B also holds the input-embedding part of
the shared matrix's gradient and its cross terms with the readout part.

Reported results (same 20 pairs and candidates as ``layer_curve.py``):
  * corr(H, B) over candidates has median 0.36 across pairs;
  * for the top 10% of candidates under the output-layer cosine score (T), with
    Delta_X = mean_T X - mean_all X: Delta_B > 0 in 19 of 20 pairs, and the median of Delta_B / Delta_H is 0.82.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit, common
from analysis.common import PAIRS
from analysis.layer_curve import filtered_pair

TOP_FRACTION = 0.1


def decompose(d) -> dict:
    H = d["s_head"] / d["full_norm"]
    B = (d["full_dot"] - d["s_head"]) / d["full_norm"]
    k = int(round(TOP_FRACTION * len(H)))
    top = np.argsort(-(d["s_head"] / d["norm_head"]), kind="stable")[:k]            # output-layer cosine picks
    top_full = np.argsort(-(d["full_dot"] / d["full_norm"]), kind="stable")[:k]      # full-gradient picks (reference)
    return dict(n=int(len(H)), k=k, corr_HB=float(np.corrcoef(H, B)[0, 1]),
                dH=float(H[top].mean() - H.mean()), dB=float(B[top].mean() - B.mean()),
                full_picks_dH=float(H[top_full].mean() - H.mean()), full_picks_dB=float(B[top_full].mean() - B.mean()),
                B_positive_share_top=float((B[top] > 0).mean()), B_positive_share_all=float((B > 0).mean()))


def compute(audit_dir=audit.AUDIT_DIR) -> dict:
    cells = {common.pair_key(m, t): decompose(filtered_pair(m, t, audit_dir)) for m, t in PAIRS}
    for c in cells.values():
        c["ratio_dB_dH"] = c["dB"] / c["dH"]
    summary = dict(corr_HB_median=common.median(c["corr_HB"] for c in cells.values()),
                   dB_positive=sum(c["dB"] > 0 for c in cells.values()), n_pairs=len(cells),
                   ratio_dB_dH_median=common.median(c["ratio_dB_dH"] for c in cells.values()))
    return dict(cells=cells, summary=summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", default=str(audit.AUDIT_DIR))
    ap.add_argument("--out-dir", default=str(common.DEFAULT_OUT / "per_example"))
    args = ap.parse_args(argv)
    result = compute(args.audit_dir)
    out = common.write_json(result, Path(args.out_dir) / "hb_decomposition.json")
    s = result["summary"]
    print(f"corr(H, B) median {s['corr_HB_median']:.4f}; Delta_B > 0 in {s['dB_positive']} of {s['n_pairs']}; "
          f"median Delta_B / Delta_H {s['ratio_dB_dH_median']:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
