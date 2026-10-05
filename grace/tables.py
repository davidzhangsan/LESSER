#!/usr/bin/env python
"""Regenerate the GRACE results of the LESSER paper from data/grace (tier 1: CPU, seconds).

* Table 12 (Appendix "GRACE teacher rankings"): per setting, the number of teachers, the Spearman
  correlation between the full-gradient and LESSER GRACE rankings over all teachers, and the
  top-ranked teacher (lowest GRACE score).
* Main text, "Distillation teacher selection": the range of the three correlations and whether
  LESSER picks the full-gradient top teacher in every setting (also the claim in the introduction).
* Appendix F.4, GRACE paragraph: across the 14 GSM8K / Llama-3.2-1B teachers, the Spearman
  correlation between the full-gradient and LESSER values of each teacher-level statistic, and the
  mean Jaccard similarity of the 64 most central responses under the two feature types.

Each number is compared with the value printed in the paper; any difference is listed and the
script exits with status 1. ``--data-dir`` also accepts the output directory of grace/score.py.

Usage: python grace/tables.py [--data-dir data/grace]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO, "data", "grace")

# Values as printed in the paper (iclr-data-selection @ 24f4b57).
PAPER_TABLE12 = {  # setting: (task, student, teachers, Spearman, top-ranked teacher)
    "gsm8k_llama1b": ("GSM8K", "Llama-3.2-1B", 14, "0.938", "Qwen2.5-1.5B-Instruct"),
    "gsm8k_olmo1b": ("GSM8K", "OLMo-2-1B", 10, "0.964", "Qwen2.5-3B-Instruct"),
    "math_llama3b": ("MATH", "Llama-3.2-3B", 10, "0.976", "Qwen2.5-3B-Instruct"),
}
PAPER_MAIN_TEXT = {"spearman_range": ("0.94", "0.98"), "same_top_teacher": True}
PAPER_F4 = {"teachers": 14, "responses": 2048, "spectral_entropy": "0.94", "effective_rank": "0.93",
            "others_range": ("0.75", "0.98"), "central_jaccard": "0.12"}
F4_SETTING = "gsm8k_llama1b"
F4_HEADLINE = ("spectral_entropy", "effective_rank")


def average_ranks(values) -> np.ndarray:
    """Ranks 1..n with ties sharing their average rank (the convention of scipy.stats.rankdata)."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("need two equally long vectors with at least 3 entries")
    rx, ry = average_ranks(x), average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def read_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


class Checker:
    def __init__(self):
        self.failures = []

    def __call__(self, label: str, ours, paper) -> None:
        ok = ours == paper
        if not ok:
            self.failures.append(f"{label}: regenerated {ours!r}, paper {paper!r}")
        print(f"  [{'ok' if ok else 'MISMATCH'}] {label}: {ours}" + ("" if ok else f" (paper: {paper})"))


def table12(data_dir: str, check: Checker) -> dict:
    print("Table 12: teacher-ranking agreement between LESSER and full-gradient GRACE")
    rows = {}
    for setting, (task, student, n_paper, rho_paper, top_paper) in PAPER_TABLE12.items():
        d = read_json(os.path.join(data_dir, f"scores_{setting}.json"))
        names = [t["short"] for t in d["teachers"]]
        full = [t["full"] for t in d["teachers"]]
        lesser = [t["lesser"] for t in d["teachers"]]
        rho = spearman(full, lesser)
        top_full, top_lesser = names[int(np.argmin(full))], names[int(np.argmin(lesser))]
        rows[setting] = {"rho": rho, "top_full": top_full, "top_lesser": top_lesser}
        print(f"{d['task']} & {d['student_label']} & ${len(names)}$ & ${rho:.3f}$ & {top_full} \\\\")
        check(f"{setting} task/student", (d["task"], d["student_label"]), (task, student))
        check(f"{setting} teachers", len(names), n_paper)
        check(f"{setting} Spearman (rho = {rho:.6f})", f"{rho:.3f}", rho_paper)
        check(f"{setting} top-ranked teacher, full gradients", top_full, top_paper)
        check(f"{setting} top-ranked teacher, LESSER", top_lesser, top_paper)
    return rows


def main_text(rows: dict, check: Checker) -> None:
    print("Main text: range of the Table 12 correlations and top-teacher agreement")
    rhos = [r["rho"] for r in rows.values()]
    check("Spearman range", (f"{min(rhos):.2f}", f"{max(rhos):.2f}"), PAPER_MAIN_TEXT["spearman_range"])
    check("same top-ranked teacher in every setting",
          all(r["top_full"] == r["top_lesser"] for r in rows.values()), PAPER_MAIN_TEXT["same_top_teacher"])


def appendix_f4(data_dir: str, check: Checker) -> None:
    print("Appendix F.4: GRACE teacher-level statistics, full gradients vs LESSER")
    d = read_json(os.path.join(data_dir, f"bank_stats_{F4_SETTING}.json"))
    roster = read_json(os.path.join(data_dir, f"scores_{F4_SETTING}.json"))
    teachers = d["teachers"]
    check("teachers", len(teachers), PAPER_F4["teachers"])
    check("teacher roster matches Table 12 row 1", [t["short"] for t in teachers],
          [t["short"] for t in roster["teachers"]])
    check("responses per teacher", roster["prompts"] * roster["responses_per_prompt"], PAPER_F4["responses"])
    rho = {}
    for name in d["statistics"]:
        rho[name] = spearman([t["full"][name] for t in teachers], [t["lesser"][name] for t in teachers])
        print(f"    Spearman({name}) = {rho[name]:.4f}")
    check("spectral entropy Spearman", f"{rho['spectral_entropy']:.2f}", PAPER_F4["spectral_entropy"])
    check("effective rank Spearman", f"{rho['effective_rank']:.2f}", PAPER_F4["effective_rank"])
    others = [v for k, v in rho.items() if k not in F4_HEADLINE]
    check(f"remaining {len(others)} statistics, Spearman range",
          (f"{min(others):.2f}", f"{max(others):.2f}"), PAPER_F4["others_range"])
    key = next(k for k in teachers[0]["full"] if k.startswith("central_"))
    jac = [len(set(t["full"][key]) & set(t["lesser"][key])) / len(set(t["full"][key]) | set(t["lesser"][key]))
           for t in teachers]
    print(f"    mean Jaccard of the {key.split('_')[1]} most central responses = {np.mean(jac):.4f}")
    check(f"{key} mean Jaccard", f"{np.mean(jac):.2f}", PAPER_F4["central_jaccard"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="directory with scores_*.json and bank_stats_*.json (default: data/grace)")
    args = ap.parse_args()
    check = Checker()
    rows = table12(args.data_dir, check)
    main_text(rows, check)
    appendix_f4(args.data_dir, check)
    if check.failures:
        print(f"\n{len(check.failures)} value(s) differ from the paper:")
        for f in check.failures:
            print("  " + f)
        sys.exit(1)
    print("\nAll GRACE values match the paper.")


if __name__ == "__main__":
    main()
