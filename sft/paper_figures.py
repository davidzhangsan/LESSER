"""Regenerate the paper's SFT figures and the similarity-bin statistics from the released records in ``data/sft``.

Produces (paper numbering):

* Figure 3 (``sft_budget_<model>.pdf``, Llama-2-7B in the paper): score against subset size for LESSER, LESS, RDS+ and
  Random on the five tasks; mean and sample standard deviation over three seeds.
* Figure 4 and Figures 8--10 (``bin_loss_<model>.pdf``): query loss after training on the top 500 examples of each of
  ten similarity bins, for LESSER, LESS and RDS+.
* The bin statistics of Section 4 and Appendix C.2: mean Spearman correlation between bin index and loss (Llama-2:
  0.86; all 20 pairs: 0.714 LESSER / 0.887 LESS / 0.058 RDS+), positive correlations (19 of 20), closest bin lowest
  (18 of 20), and the OLMo3 / MMLU-Pro exception (-0.030).

The statistics are compared with the values printed in the paper, and the plotted values of every figure with the
values embedded in the paper's figure PDFs, as recorded in ``data/sft/reference/paper_figures.json`` (the drawing
operators are compared too, for information: they depend on the ReportLab version). With ``--paper-dir`` pointing at
the paper's LaTeX sources, figures are compared with the paper's PDFs themselves, drawing operators byte for byte,
and the recorded reference is checked to be current (``--write-reference`` rewrites it). The exit status is nonzero
on any difference.

Run: python -m sft.paper_figures [--out-dir outputs/sft/figures] [--paper-dir PATH [--write-reference]]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .common import (BUDGETS, DATA_DIR, DEFAULT_OUT, MODEL_ORDER, MODELS, N_BINS, TASK_LABEL, TASKS, display_path,
                     read_json, write_json)
from .plot import PAPER_METHOD_COLORS, paper_main_vector_figure, pdf_page_content, pdf_payload
from .results import BIN_METHODS, bin_curves, downstream_cells

# The paper's figure files and the model token used inside their embedded payloads.
# Paper figure PDFs, identified by the end of their file name in figures/ of the paper source.
PAPER_BUDGET_FIGURE = {"llama-2-7b": "sft_budget_llama2.pdf"}
PAPER_BIN_FIGURES = {"llama-2-7b": "bin_loss_llama2.pdf", "llama-3.2-3b": "bin_loss_llama3.2-3b.pdf",
                     "qwen3-4b-base": "bin_loss_qwen3-4b-base.pdf", "olmo3-7b": "bin_loss_olmo3-7b.pdf"}


def paper_figure_path(paper: Path, ending: str) -> Path:
    """The one PDF in figures/ of the paper source whose file name ends with ``ending``."""
    found = sorted(q for q in (Path(paper) / "figures").glob("*.pdf") if q.name.endswith(ending))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one paper figure ending in {ending}, found {[q.name for q in found]}")
    return found[0]
PAYLOAD_MODEL = {m: MODELS[m].nayak_dir for m in MODELS}
PAYLOAD_METHOD = {"lesser": "PROD", "less": "LESS", "rds": "RDS+", "random": "Random"}
PAYLOAD_BIN_METHOD = {"lesser": "PROD", "less": "LESS (RR)", "rds": "RDS+ (RR)"}
LABEL = {"lesser": "LESSER", "less": "LESS", "rds": "RDS+", "random": "Random"}


def budget_figure(cells: dict, model: str, output: Path) -> dict:
    methods = [{"key": m, "label": LABEL[m], "color": PAPER_METHOD_COLORS[LABEL[m]]}
               for m in ("lesser", "less", "rds", "random")]
    panels = []
    for task in TASKS:
        series = {}
        for method in methods:
            values = [cells[model, task, b, method["key"]] for b in BUDGETS]
            series[method["key"]] = {"x": list(BUDGETS), "y": [v["mean"] for v in values],
                                     "sd": [v["std"] for v in values]}
        panels.append({"title": TASK_LABEL[task], "xlim": (400, 11100), "series": series})
    payload = {"generator": "sft/paper_figures.py",
               "cells": [{"model": model, "task": t, "budget": b, "method": m, "mean": cells[model, t, b, m]["mean"],
                          "std": cells[model, t, b, m]["std"], "seeds": cells[model, t, b, m]["seeds"],
                          "sources": cells[model, t, b, m]["sources"]}
                         for t in TASKS for b in BUDGETS for m in ("lesser", "less", "rds", "random")]}
    paper_main_vector_figure(panels, methods, output, ylabel="Test metric (%)", xlabel="Subset size",
                             xticks=list(zip(BUDGETS, ("1k", "5k", "10k"))), metadata=payload, shared_xlabel=True)
    return payload


def bin_figure(curves: dict, model: str, output: Path) -> dict:
    methods = [{"key": m, "label": LABEL[m], "color": PAPER_METHOD_COLORS[LABEL[m]]} for m in BIN_METHODS]
    panels = [{"title": TASK_LABEL[t], "xlim": (-.35, 9.35),
               "series": {m: {"x": list(range(N_BINS)), "y": curves[model, t, m], "sd": [0] * N_BINS}
                          for m in BIN_METHODS}}
              for t in TASKS]
    payload = {"generator": "sft/paper_figures.py", "methods": list(BIN_METHODS),
               "cells": [{"model": model, "task": t, "method": m, "loss": curves[model, t, m]}
                         for t in TASKS for m in BIN_METHODS]}
    paper_main_vector_figure(panels, methods, output, ylabel="Query CE loss", xlabel="Query similarity",
                             xticks=[(0, "High"), (9, "Low")], metadata=payload, endpoint_labels=True,
                             shared_xlabel=True)
    return payload


def bin_statistics(curves: dict) -> dict:
    """Spearman(bin index, loss) per model--task pair; positive = higher similarity gives lower loss."""
    out = {}
    for method in BIN_METHODS:
        pairs = []
        for model in MODEL_ORDER:
            for task in TASKS:
                values = np.asarray(curves[model, task, method])
                pairs.append({"model": model, "task": task,
                              "spearman": float(spearmanr(list(range(N_BINS)), values).statistic),
                              "lowest_loss_bins": np.flatnonzero(values == values.min()).tolist()})
        out[method] = {
            "mean_spearman": float(np.mean([p["spearman"] for p in pairs])),
            "mean_spearman_by_model": {m: float(np.mean([p["spearman"] for p in pairs if p["model"] == m]))
                                       for m in MODEL_ORDER},
            "positive_pairs": sum(p["spearman"] > 0 for p in pairs),
            "closest_bin_lowest_pairs": sum(0 in p["lowest_loss_bins"] for p in pairs),
            "pairs": pairs}
    return out


def claims(stats: dict) -> list[tuple[str, str, str, str]]:
    s = stats["lesser"]
    olmo_mmlu = next(p["spearman"] for p in s["pairs"] if p["model"] == "olmo3-7b" and p["task"] == "mmlu_pro")
    negative = [f"{p['model']}/{p['task']}" for p in s["pairs"] if p["spearman"] <= 0]
    return [
        ("LESSER mean Spearman, Llama-2-7B", "experiments.tex", "0.86",
         f"{s['mean_spearman_by_model']['llama-2-7b']:.2f}"),
        ("LESSER positive pairs", "experiments.tex; _appendix/additional_experiments.tex", "19 of 20",
         f"{s['positive_pairs']} of 20"),
        ("mean Spearman LESSER", "_appendix/additional_experiments.tex", "0.714", f"{s['mean_spearman']:.3f}"),
        ("mean Spearman LESS", "_appendix/additional_experiments.tex", "0.887",
         f"{stats['less']['mean_spearman']:.3f}"),
        ("mean Spearman RDS+", "_appendix/additional_experiments.tex", "0.058",
         f"{stats['rds']['mean_spearman']:.3f}"),
        ("LESSER closest bin lowest", "_appendix/additional_experiments.tex", "18", f"{s['closest_bin_lowest_pairs']}"),
        ("the negative LESSER pair", "_appendix/additional_experiments.tex", "OLMo3/MMLU-Pro",
         "OLMo3/MMLU-Pro" if negative == ["olmo3-7b/mmlu_pro"] else ",".join(negative)),
        ("its Spearman", "_appendix/additional_experiments.tex", "-0.030", f"{olmo_mmlu:.3f}"),
    ]


def _same_numbers(a, b) -> bool:
    return all(math.isclose(x, y, rel_tol=0, abs_tol=0) for x, y in zip(a, b)) and len(a) == len(b)


def compare_budget_payload(ours: dict, paper: dict, model: str) -> list[str]:
    diffs = []
    ref = {(c["task"], c["budget"], c["method"]): c for c in paper["cells"]}
    for c in ours["cells"]:
        p = ref.get((c["task"], c["budget"], PAYLOAD_METHOD[c["method"]]))
        if p is None or p["model"] != PAYLOAD_MODEL[model]:
            diffs.append(f"budget figure: no paper value for {c['task']}/k{c['budget']}/{c['method']}")
        elif not (p["mean"] == c["mean"] and p["std"] == c["std"] and p["seeds"] == c["seeds"]):
            diffs.append(f"budget figure {c['task']}/k{c['budget']}/{c['method']}: paper {p['mean']}+-{p['std']}, "
                         f"regenerated {c['mean']}+-{c['std']}")
    if len(ref) != len(ours["cells"]):
        diffs.append(f"budget figure: {len(ref)} paper cells vs {len(ours['cells'])} regenerated")
    return diffs


def compare_bin_payload(ours: dict, paper: dict, model: str) -> list[str]:
    diffs = []
    ref = {(c["task"], c["method"]): c for c in paper["cells"]}
    for c in ours["cells"]:
        p = ref.get((c["task"], PAYLOAD_BIN_METHOD[c["method"]]))
        if p is None or p["model"] != PAYLOAD_MODEL[model]:
            diffs.append(f"bin figure {model}: no paper curve for {c['task']}/{c['method']}")
        elif not _same_numbers(p["loss"], c["loss"]):
            diffs.append(f"bin figure {model} {c['task']}/{c['method']}: paper {p['loss']} regenerated {c['loss']}")
    if len(ref) != len(ours["cells"]):
        diffs.append(f"bin figure {model}: {len(ref)} paper curves vs {len(ours['cells'])} regenerated")
    return diffs



REFERENCE = Path("reference") / "paper_figures.json"
PLOTTED_FIELDS = {"budget": ("model", "task", "budget", "method", "mean", "std", "seeds"),
                  "bins": ("model", "task", "method", "loss")}


def paper_figure_reference(paper: Path) -> dict:
    """Plotted values embedded in the paper's figure PDFs and the digest of their drawing operators."""
    ref = {"_note": "Values embedded in the paper's SFT figure PDFs (Figures 3, 4, 8-10) and the sha256 of"
                    " each page's drawing operators, written by `python -m sft.paper_figures --paper-dir PAPER"
                    " --write-reference`.", "figures": {}}
    for kind, files in (("budget", PAPER_BUDGET_FIGURE), ("bins", PAPER_BIN_FIGURES)):
        for model, rel in files.items():
            pdf = paper_figure_path(paper, rel)
            payload = pdf_payload(pdf)
            ref["figures"][f"{kind}/{model}"] = {
                "paper_file": rel,
                "drawing_sha256": hashlib.sha256(pdf_page_content(pdf)).hexdigest(),
                "cells": [{k: c[k] for k in PLOTTED_FIELDS[kind]} for c in payload["cells"]]}
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "figures")
    ap.add_argument("--budget-models", nargs="+", default=["llama-2-7b"], choices=MODEL_ORDER,
                    help="models to draw budget curves for (the paper shows Llama-2-7B)")
    ap.add_argument("--paper-dir", type=Path, help="paper LaTeX sources; compare with its figure PDFs instead of the "
                                                   "recorded reference in data/sft/reference")
    ap.add_argument("--write-reference", action="store_true",
                    help="with --paper-dir: rewrite data/sft/reference/paper_figures.json from the paper's PDFs")
    args = ap.parse_args(argv)
    if args.write_reference and not args.paper_dir:
        ap.error("--write-reference needs --paper-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = downstream_cells(args.data_dir)
    curves = bin_curves(args.data_dir)
    outputs = {}
    for model in args.budget_models:
        path = args.out_dir / f"sft_budget_{model}.pdf"
        outputs[path] = ("budget", model, budget_figure(cells, model, path))
    for model in MODEL_ORDER:
        path = args.out_dir / f"bin_loss_{model}.pdf"
        outputs[path] = ("bins", model, bin_figure(curves, model, path))
    stats = bin_statistics(curves)
    claim_rows = claims(stats)
    (args.out_dir / "bin_statistics.json").write_text(json.dumps(
        {"statistics": stats,
         "claims": [dict(zip(("quantity", "location", "paper", "regenerated"), c)) for c in claim_rows],
         "curves": [{"model": k[0], "task": k[1], "method": k[2], "loss": v} for k, v in sorted(curves.items())]},
        indent=1) + "\n")
    for what, where, claimed, derived in claim_rows:
        print(f"{what:36s} paper {claimed:15s} regenerated {derived}")
    for path in outputs:
        print(f"wrote {display_path(path)}")

    diffs = [f"{what}: paper {claimed}, regenerated {derived}"
             for what, where, claimed, derived in claim_rows if claimed != derived]
    reference_path = args.data_dir / REFERENCE
    if args.paper_dir:
        source = f"the paper's figures in {display_path(args.paper_dir)}"
        if args.write_reference:
            write_json(paper_figure_reference(args.paper_dir), reference_path, indent=1)
            print(f"wrote {display_path(reference_path)}")
        elif read_json(reference_path) != paper_figure_reference(args.paper_dir):
            diffs.append(f"{display_path(reference_path)} does not match the paper's figures"
                         " (rerun with --write-reference)")
        for what, where, claimed, derived in claim_rows:
            texts = [(args.paper_dir / "contents" / f).read_text().replace("$", "") for f in where.split("; ")]
            if not any(claimed in t for t in texts):
                diffs.append(f"{what}: the claimed value {claimed} was not found in {where}")
    else:
        source = f"the paper values recorded in {display_path(reference_path)}"
        recorded = read_json(reference_path)["figures"]
    for path, (kind, model, payload) in outputs.items():
        ref_rel = (PAPER_BUDGET_FIGURE if kind == "budget" else PAPER_BIN_FIGURES).get(model)
        if ref_rel is None:
            print(f"(no paper figure for {kind}/{model}; not compared)")
            continue
        compare = compare_budget_payload if kind == "budget" else compare_bin_payload
        drawing = hashlib.sha256(pdf_page_content(path)).hexdigest()
        if args.paper_dir:
            ref = paper_figure_path(args.paper_dir, ref_rel)
            same_drawing = drawing == hashlib.sha256(pdf_page_content(ref)).hexdigest()
            if not same_drawing:
                diffs.append(f"{path.name}: drawing operators differ from {ref_rel}")
            payload_diffs = compare(payload, pdf_payload(ref), model)
        else:
            entry = recorded[f"{kind}/{model}"]
            same_drawing = drawing == entry["drawing_sha256"]   # informational: depends on the ReportLab version
            payload_diffs = compare(payload, {"cells": entry["cells"]}, model)
        diffs += payload_diffs
        print(f"{path.name} vs {ref_rel}: plotted values {'identical' if not payload_diffs else 'DIFFERENT'}, "
              f"drawing operators {'identical' if same_drawing else 'different'}")
    for d in diffs:
        print("DIFFERENCE:", d)
    print(f"paper comparison: {'IDENTICAL' if not diffs else f'{len(diffs)} difference(s)'} (with {source})")
    return 1 if diffs else 0

if __name__ == "__main__":
    sys.exit(main())
