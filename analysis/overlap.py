"""Overlap of the LESSER and LESS selections at k = 5,000 (Appendix E, Table "selection-overlap"; Section 5.1).

For each of the 20 model-task pairs: the number of examples shared by the saved LESSER selection and the
released LESS selection, and their Jaccard similarity |S_LESSER & S_LESS| / |S_LESSER | S_LESS|, over the
197,196-example Tulu pool. Reported: Llama-2-7B mean Jaccard 0.037 (quoted in Section 5.1); per-model means
0.032-0.045; e.g. Llama-2-7B TyDiQA 491 shared / 0.052.

``--check-tex`` compares the generated rows with the table in the paper source.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import common, selections
from analysis.common import MODEL_KEYS, MODELS, TASK_KEYS


def compute() -> dict:
    cells = {}
    for m in MODEL_KEYS:
        for t in TASK_KEYS:
            a = set(selections.load(m, t, "lesser"))
            b = set(selections.load(m, t, "less"))
            cells[common.pair_key(m, t)] = dict(shared=len(a & b), jaccard=len(a & b) / len(a | b))
    means = {m: sum(cells[common.pair_key(m, t)]["jaccard"] for t in TASK_KEYS) / len(TASK_KEYS) for m in MODEL_KEYS}
    return dict(cells=cells, mean_jaccard=means)


def latex_rows(result) -> list:
    rows = []
    for m in MODEL_KEYS:
        parts = []
        for t in TASK_KEYS:
            c = result["cells"][common.pair_key(m, t)]
            parts.append(f"${c['shared']:,}$ / ${c['jaccard']:.3f}$".replace(",", "{,}"))
        rows.append(f"{MODELS[m].name} & " + " & ".join(parts) + f" & ${result['mean_jaccard'][m]:.3f}$ \\\\")
    return rows


def check_tex(rows, tex_path) -> int:
    """Number of generated rows found verbatim in the paper's overlap table (raises if any is missing)."""
    tex = Path(tex_path).read_text()
    start = tex.index("\\label{tab:selection-overlap}")
    body = tex[start:tex.index("\\end{tabular}", start)]
    table = [re.sub(r"\s+", " ", line.strip()) for line in body.splitlines()]
    missing = [r for r in rows if re.sub(r"\s+", " ", r) not in table]
    if missing:
        raise ValueError("rows not in the paper table:\n" + "\n".join(missing))
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(common.DEFAULT_OUT / "overlap"))
    ap.add_argument("--check-tex", help="paper source file with the overlap table (additional_experiments.tex)")
    args = ap.parse_args(argv)
    result = compute()
    rows = latex_rows(result)
    common.write_json(result, Path(args.out_dir) / "selection_overlap.json")
    (Path(args.out_dir) / "selection_overlap_rows.tex").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    if args.check_tex:
        print(f"{check_tex(rows, args.check_tex)} of {len(rows)} rows match the paper table")


if __name__ == "__main__":
    main()
