"""Regenerate the paper's SFT tables and the numbers quoted from them, from the released records in ``data/sft``.

Produces (paper numbering):

* Tables 9--11 (``main_results_k{1000,5000,10000}.tex``): held-out scores of LESSER, LESS, RDS+ and Random for four
  models, five tasks and three budgets; mean and sample standard deviation over three seeds; bold marks the largest
  rounded mean.
* The mean absolute score gaps quoted in the introduction and the Figure 3 caption (1.3 / 2.2 / 2.6 over 60 cells).
* The ranges quoted in Appendix C.1 and the Llama-2 comparisons of Section 4.
* The pipeline replication table of Appendix A.1 (``baseline_replication.tex``) and its two summary numbers.
* The LESSER--LESS selection-overlap table of Appendix E (``selection_overlap.tex``) and the Llama-2 mean Jaccard of
  Section 5.

Every quoted number is compared with the value printed in the paper. The regenerated tables are compared line by line
with the paper's tables as recorded in ``data/sft/reference/paper_tables.json``, or, with ``--paper-dir`` pointing at
the paper's LaTeX sources, with the sources themselves (which also checks that the recorded reference is current;
``--write-reference`` rewrites it from the sources). The exit status is nonzero on any difference; lines listed in
``KNOWN_PAPER_ERRATA`` (none at present) are reported without failing.

Run: python -m sft.paper_tables [--out-dir outputs/sft/tables] [--paper-dir PATH [--write-reference]]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .common import (unwrap_cc, BUDGETS, DATA_DIR, DEFAULT_OUT, MODEL_ORDER, MODELS, TASK_LABEL, TASKS, display_path, read_json,
                     read_selection, selection_path, write_json)
from .results import downstream_cells, nayak_budget_rows, replication_cells

COLUMNS = (("lesser", r"\LESSER{}"), ("less", r"\LESS{}"), ("rds", r"\RDS{}"), ("random", "Random"))
CAPTIONS = {
    1000: r"Held-out performance at $k=1{,}000$, mean $\pm$ sample standard deviation over three seeds; each task uses "
          r"its target metric on a $0$--$100$ scale, higher is better.",
    5000: r"Held-out performance at $k=5{,}000$.",
    10000: r"Held-out performance at $k=10{,}000$.",
}


def shown(value: float) -> float:
    """The value as printed in the tables (one decimal)."""
    return float(f"{value:.1f}")


def fmt_cell(cell: dict, bold: bool) -> str:
    body = f"{cell['mean']:.1f}" + (rf"{{\scriptstyle\pm}}{cell['std']:.1f}" if cell["std"] is not None else "")
    return rf"$\mathbf{{{body}}}$" if bold else f"${body}$"


def main_table(cells: dict, budget: int) -> str:
    lines = [r"\begin{table*}[!htbp]", r"\centering", rf"\caption{{{CAPTIONS[budget]}}}",
             rf"\label{{tab:main-results-k{budget}}}", r"\small\begin{tabular}{l|l|r|r|r|r}",
             "Model & Task & " + " & ".join(h for _, h in COLUMNS) + r" \\", r"\midrule"]
    for mi, model in enumerate(MODEL_ORDER):
        for task in TASKS:
            row = {m: cells[model, task, budget, m] for m, _ in COLUMNS}
            best = max(round(c["mean"], 1) for c in row.values())
            values = [fmt_cell(row[m], round(row[m]["mean"], 1) == best) for m, _ in COLUMNS]
            lines.append(f"{MODELS[model].label} & {TASK_LABEL[task]} & " + " & ".join(values) + r" \\")
        if mi < len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def grid_statistics(cells: dict) -> dict:
    """Numbers quoted from Tables 9--11, computed from the printed (one-decimal) values as the paper did."""
    keys = [(m, t, b) for b in BUDGETS for m in MODEL_ORDER for t in TASKS]
    v = {(m, t, b, meth): shown(cells[m, t, b, meth]["mean"]) for (m, t, b) in keys for meth, _ in COLUMNS}
    raw = {(m, t, b, meth): cells[m, t, b, meth]["mean"] for (m, t, b) in keys for meth, _ in COLUMNS}
    out = {"n_cells": len(keys)}
    for tag, table in (("printed", v), ("unrounded", raw)):
        out[f"mean_abs_gap_{tag}"] = {
            other: statistics.mean(abs(table[k + ("lesser",)] - table[k + (other,)]) for k in keys)
            for other in ("less", "rds", "random")}
    out["lesser_above_random_cells"] = sum(v[k + ("lesser",)] > v[k + ("random",)] for k in keys)
    ty = [(m, b) for m in MODEL_ORDER for b in BUDGETS]
    d = [v[m, "tydiqa", b, "lesser"] - v[m, "tydiqa", b, "random"] for m, b in ty]
    out["tydiqa_lesser_minus_random"] = {"min": min(d), "max": max(d), "all_positive": all(x > 0 for x in d)}
    d = [v[m, "tydiqa", b, "less"] - v[m, "tydiqa", b, "lesser"]
         for m in ("qwen3-4b-base", "olmo3-7b") for b in BUDGETS]
    out["tydiqa_qwen3_olmo3_largest_less_lead"] = max(d)
    for b in (1000, 10000):
        d = [v[m, "tydiqa", b, "less"] - v[m, "tydiqa", b, "lesser"] for m in ("llama-3.2-3b", "llama-2-7b")]
        out[f"tydiqa_llama_less_lead_k{b}"] = {"min": min(d), "max": max(d)}
    spread = [max(v[m, "mmlu_pro", b, meth] for meth, _ in COLUMNS)
              - min(v[m, "mmlu_pro", b, meth] for meth, _ in COLUMNS)
              for m in MODEL_ORDER for b in (5000, 10000)]
    out["mmlu_pro_spread_k5000_k10000_max"] = max(spread)
    out["llama2_lesser_minus_less_by_task"] = {
        t: statistics.mean(raw["llama-2-7b", t, b, "lesser"] - raw["llama-2-7b", t, b, "less"] for b in BUDGETS)
        for t in TASKS}
    return out


def replication_table(data_dir: Path) -> tuple[str, dict]:
    """Appendix A.1: released value / our rerun. Three-seed means for Llama-3.2-3B; seed 0 for the single reruns."""
    reruns = replication_cells(data_dir)
    rows, less_gaps, random_gaps = [], [], []
    for model, task in [("llama-3.2-3b", t) for t in ("tydiqa", "gsm8k", "codex", "bbh")] + \
                       [("qwen3-4b-base", "tydiqa"), ("olmo3-7b", "tydiqa")]:
        nayak = nayak_budget_rows(model, data_dir)
        entries = []
        for method in ("less", "random"):
            ours = reruns.get((model, task, method))
            if ours is None:
                entries.append("--")
                continue
            seeds = range(len(ours))
            reported = statistics.mean(nayak[task, 1000, method, s] for s in seeds)
            rerun = statistics.mean(ours)
            (less_gaps if method == "less" else random_gaps).append(abs(shown(reported) - shown(rerun)))
            entries.append(f"${reported:.1f}$ / ${rerun:.1f}$")
        rows.append(f"{MODELS[model].label} & {TASK_LABEL[task]} & " + " & ".join(entries) + r" \\")
    body = [r"\small\begin{tabular}{l|l|r|r}", r"Model & Task & \LESS{} & Random \\", r"\midrule",
            *rows[:4], r"\midrule", *rows[4:], r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(body), {"less_max_abs_diff": max(less_gaps), "random_max_abs_diff": max(random_gaps)}


def overlap_table(data_dir: Path, budget: int = 5000) -> tuple[str, dict]:
    """Appendix E: shared examples and Jaccard similarity of the LESSER and released LESS selections."""
    order = ("llama-2-7b", "llama-3.2-3b", "qwen3-4b-base", "olmo3-7b")
    rows, means = [], {}
    for model in order:
        entries, jaccards = [], []
        for task in TASKS:
            a = set(read_selection(selection_path(model, "lesser", task, budget, data_dir), budget))
            b = set(read_selection(selection_path(model, "less", task, budget, data_dir), budget))
            shared, jac = len(a & b), len(a & b) / len(a | b)
            jaccards.append(jac)
            entries.append(f"${shared:,}$".replace(",", "{,}") + f" / ${jac:.3f}$")
        means[model] = statistics.mean(jaccards)
        rows.append(f"{MODELS[model].label} & " + " & ".join(entries) + f" & ${means[model]:.3f}$" + r" \\")
    body = [r"{\begin{tabular}{l|r|r|r|r|r|r}",
            "Model & " + " & ".join(TASK_LABEL[t] for t in TASKS) + r" & Mean Jaccard \\", r"\midrule",
            *rows, r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(body), {"mean_jaccard": means}


# Numbers the paper quotes, with where they appear; each is compared with the regenerated value.
def claims(stats: dict, rep: dict, ov: dict) -> list[tuple[str, str, str, str]]:
    g = stats["mean_abs_gap_printed"]
    ty = stats["tydiqa_lesser_minus_random"]
    k1, k10 = stats["tydiqa_llama_less_lead_k1000"], stats["tydiqa_llama_less_lead_k10000"]
    where = "introduction.tex; experiments.tex (Fig. 3 caption)"
    return [
        ("mean |gap| to LESS, 60 cells", where, "1.3", f"{g['less']:.1f}"),
        ("mean |gap| to RDS+, 60 cells", where, "2.2", f"{g['rds']:.1f}"),
        ("mean |gap| to Random, 60 cells", where, "2.6", f"{g['random']:.1f}"),
        ("TyDiQA LESSER above Random, range", "_appendix/additional_experiments.tex (C.1)", "0.5 to 9.6",
         f"{ty['min']:.1f} to {ty['max']:.1f}" + ("" if ty["all_positive"] else " (not all positive)")),
        ("TyDiQA Qwen3/OLMo3: largest LESS lead", "_appendix/additional_experiments.tex (C.1)", "0.9",
         f"{stats['tydiqa_qwen3_olmo3_largest_less_lead']:.1f}"),
        ("TyDiQA Llama LESS lead at k=1,000", "_appendix/additional_experiments.tex (C.1)", "3.1 to 4.4",
         f"{k1['min']:.1f} to {k1['max']:.1f}"),
        ("TyDiQA Llama LESS lead at k=10,000", "_appendix/additional_experiments.tex (C.1)", "0.8 to 1.2",
         f"{k10['min']:.1f} to {k10['max']:.1f}"),
        ("MMLU-Pro spread at k=5,000 and 10,000", "_appendix/additional_experiments.tex (C.1)", "1.3",
         f"{stats['mmlu_pro_spread_k5000_k10000_max']:.1f}"),
        ("LESS reruns agree within", "_appendix/setup_details.tex (A.1)", "0.9", f"{rep['less_max_abs_diff']:.1f}"),
        ("Random reruns differ by up to", "_appendix/setup_details.tex (A.1)", "3.0",
         f"{rep['random_max_abs_diff']:.1f}"),
        ("Llama-2 mean Jaccard LESSER vs LESS, k=5,000", "analysis_new.tex", "0.037",
         f"{ov['mean_jaccard']['llama-2-7b']:.3f}"),
    ]


def active_lines(text: str) -> list[str]:
    """Non-comment lines, with the paper's text-color macro \\cc{...} unwrapped."""
    return [unwrap_cc(line.rstrip()) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("%")]


# Paper-side errors to report without failing (table lines mapped to an explanation). None for the paper revision
# that the recorded reference was written from.
KNOWN_PAPER_ERRATA: dict = {}


def compare_with_paper(paper: Path, tables: dict, rep_body: str, ov_body: str, claim_rows) -> list[str]:
    """Return a list of differences between the regenerated artifacts and the paper sources (empty = identical).

    Lines listed in KNOWN_PAPER_ERRATA are reported by main() as known paper-side errors, not differences."""
    diffs = []
    for budget, text in tables.items():
        want = active_lines((paper / "contents/tables" / f"main_results_k{budget}.tex").read_text())
        got = active_lines(text)
        if want != got:
            diffs.append(f"main_results_k{budget}.tex differs: "
                         + "; ".join(f"paper {a!r} vs regenerated {b!r}" for a, b in zip(want, got) if a != b)[:2000]
                         + ("" if len(want) == len(got) else f" (line counts {len(want)} vs {len(got)})"))
    for name, rel, body in (("replication", "contents/_appendix/setup_details.tex", rep_body),
                            ("overlap", "contents/_appendix/additional_experiments.tex", ov_body)):
        source = unwrap_cc((paper / rel).read_text())
        for line in active_lines(body):
            if line not in source and line not in KNOWN_PAPER_ERRATA:
                diffs.append(f"{name} table line not in {rel}: {line!r}")
    for what, where, claimed, derived in claim_rows:
        texts = [unwrap_cc((paper / "contents" / f.split(" ")[0]).read_text()).replace("$", "") for f in where.split("; ")]
        if not any(claimed in t for t in texts):
            diffs.append(f"{what}: the claimed value {claimed} was not found in {where}")
    return diffs



REFERENCE = Path("reference") / "paper_tables.json"
PAPER_TABLE_FILES = {b: f"contents/tables/main_results_k{b}.tex" for b in BUDGETS}
PAPER_TABULARS = {"baseline_replication": ("contents/_appendix/setup_details.tex", "tab:baseline-replication"),
                  "selection_overlap": ("contents/_appendix/additional_experiments.tex", "tab:selection-overlap")}


def tabular_after_label(text: str, label: str) -> list[str]:
    """Active lines of the tabular environment that follows ``\\label{label}``."""
    lines = active_lines(text)
    i = next(i for i, line in enumerate(lines) if f"\\label{{{label}}}" in line)
    j = next(j for j in range(i, len(lines)) if "\\begin{tabular}" in lines[j])
    k = next(k for k in range(j, len(lines)) if lines[k].startswith("\\end{tabular}"))
    return lines[j:k + 1]


def paper_reference(paper: Path) -> dict:
    """The paper's SFT tables as printed, read from its LaTeX sources."""
    ref = {"_note": "Active lines of the SFT tables in the paper's LaTeX sources (Tables 9-11, Table 3 and"
                    " Table 13), written by `python -m sft.paper_tables --paper-dir PAPER --write-reference`."}
    for b, rel in PAPER_TABLE_FILES.items():
        ref[f"main_results_k{b}"] = active_lines((paper / rel).read_text())
    for name, (rel, label) in PAPER_TABULARS.items():
        ref[name] = tabular_after_label((paper / rel).read_text(), label)
    return ref


def compare_with_reference(ref: dict, tables: dict, rep_body: str, ov_body: str) -> list[str]:
    """Differences between the regenerated tables and the recorded paper tables (KNOWN_PAPER_ERRATA excepted)."""
    diffs = []
    for budget, text in tables.items():
        want, got = ref[f"main_results_k{budget}"], active_lines(text)
        if want != got:
            diffs.append(f"main_results_k{budget}: " + "; ".join(
                f"paper {a!r} vs regenerated {b!r}" for a, b in zip(want, got) if a != b)[:2000]
                + ("" if len(want) == len(got) else f" (line counts {len(want)} vs {len(got)})"))
    for name, body in (("baseline_replication", rep_body), ("selection_overlap", ov_body)):
        want, got = ref[name], active_lines(body)
        if len(want) != len(got):
            diffs.append(f"{name}: {len(got)} lines, paper {len(want)}")
        for line in got:
            if line not in want and line not in KNOWN_PAPER_ERRATA:
                diffs.append(f"{name} table line not in the paper: {line!r}")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "tables")
    ap.add_argument("--paper-dir", type=Path, help="paper LaTeX sources; compare with them instead of the recorded "
                                                   "reference in data/sft/reference")
    ap.add_argument("--write-reference", action="store_true",
                    help="with --paper-dir: rewrite data/sft/reference/paper_tables.json from the paper sources")
    args = ap.parse_args(argv)
    if args.write_reference and not args.paper_dir:
        ap.error("--write-reference needs --paper-dir")

    cells = downstream_cells(args.data_dir)
    tables = {b: main_table(cells, b) for b in BUDGETS}
    stats = grid_statistics(cells)
    rep_body, rep = replication_table(args.data_dir)
    ov_body, ov = overlap_table(args.data_dir)
    claim_rows = claims(stats, rep, ov)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for b, text in tables.items():
        (args.out_dir / f"main_results_k{b}.tex").write_text(text)
    (args.out_dir / "baseline_replication.tex").write_text(rep_body)
    (args.out_dir / "selection_overlap.tex").write_text(ov_body)
    (args.out_dir / "sft_statistics.json").write_text(json.dumps(
        {"grid": stats, "replication": rep, "overlap": ov,
         "claims": [dict(zip(("quantity", "location", "paper", "regenerated"), c)) for c in claim_rows],
         "cells": [{"model": k[0], "task": k[1], "budget": k[2], "method": k[3], **v}
                   for k, v in sorted(cells.items())]},
        indent=1) + "\n")
    for what, where, claimed, derived in claim_rows:
        print(f"{what:45s} paper {claimed:12s} regenerated {derived}")
    print(f"LESSER above Random in {stats['lesser_above_random_cells']} of {stats['n_cells']} cells; "
          f"unrounded mean |gaps| " + ", ".join(f"{k} {v:.3f}" for k, v in stats["mean_abs_gap_unrounded"].items()))
    print("Llama-2 mean (LESSER - LESS) over budgets: "
          + ", ".join(f"{TASK_LABEL[t]} {d:+.2f}" for t, d in stats["llama2_lesser_minus_less_by_task"].items()))
    print(f"wrote {display_path(args.out_dir)}")
    diffs = [f"{what}: paper {claimed}, regenerated {derived}"
             for what, where, claimed, derived in claim_rows if claimed != derived]
    reference_path = args.data_dir / REFERENCE
    if args.paper_dir:
        source = f"the paper sources in {display_path(args.paper_dir)}"
        diffs += compare_with_paper(args.paper_dir, tables, rep_body, ov_body, claim_rows)
        if args.write_reference:
            write_json(paper_reference(args.paper_dir), reference_path, indent=1)
            print(f"wrote {display_path(reference_path)}")
        elif read_json(reference_path) != paper_reference(args.paper_dir):
            diffs.append(f"{display_path(reference_path)} does not match the paper sources"
                         " (rerun with --write-reference)")
    else:
        source = f"the paper tables recorded in {display_path(reference_path)}"
        diffs += compare_with_reference(read_json(reference_path), tables, rep_body, ov_body)
    for line, note in KNOWN_PAPER_ERRATA.items():
        if line in rep_body:
            print(f"KNOWN paper-side error: {note}")
    for d in diffs:
        print("DIFFERENCE:", d)
    print(f"paper comparison: {'IDENTICAL' if not diffs else f'{len(diffs)} difference(s)'} (with {source})")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
