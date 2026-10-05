"""Readers for the released result records in ``data/sft`` (the inputs of every SFT table and figure).

Sources, all validated field by field (a missing or unexpected record raises):

* ``nayak/<model>/budget_true_metric_budget.csv``: Nayak et al.'s released per-seed scores of Random, RDS+ and LESS
  (TyDiQA, GSM8K, Codex, BBH).
* ``results/lesser/<model>/<task>_k<k>_s<seed>.json``: our LESSER cells on the same four tasks.
* ``results/mmlu_pro_5shot/<model>/<method>_k<k>_s<seed>.json``: five-shot MMLU-Pro reruns of all four methods, which
  replace every MMLU-Pro cell in the paper.
* ``results/bins_ce/<model>/<task>_bin<b>.json`` and ``nayak/<model>/binning_ce_loss_quantile.csv``: query loss after
  training on each similarity bin (LESSER cells; LESS and RDS+ released by Nayak et al.).
* ``results/replication/...``: reruns of released selections in our pipeline (Appendix A.1).

Means and sample standard deviations use :mod:`statistics` over seeds 0, 1, 2 in that order, as the paper did.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from .common import (BUDGETS, DATA_DIR, MODEL_ORDER, MODELS, N_BINS, NAYAK_METHOD, SEEDS, TASKS, read_json)

DOWNSTREAM_METHODS = ("lesser", "less", "rds", "random")
LEGACY_METHOD = {"lesser": "baseprod"}          # method token inside the released LESSER cell records
FIVESHOT_TAG = {"lesser": "head", "less": "less", "rds": "rds", "random": "random"}   # tags inside five-shot records
MMLU_PRO_QUESTIONS = 12032
BIN_METHODS = ("lesser", "less", "rds")


def _rel(path: Path, data_dir: Path) -> str:
    try:
        return str(Path(path).relative_to(data_dir))
    except ValueError:
        return str(path)


def nayak_budget_rows(model: str, data_dir: Path = DATA_DIR) -> dict:
    """{(task, budget, method, seed): score} for Random, RDS+ and LESS from Nayak et al.'s released CSV."""
    path = Path(data_dir) / "nayak" / MODELS[model].nayak_dir / "budget_true_metric_budget.csv"
    by_name = {v: k for k, v in NAYAK_METHOD.items()}
    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"dataset", "method", "num_samples", "seed", "true_metric"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames}")
        for r in reader:
            method = by_name.get(r["method"])
            if method is None or int(r["num_samples"]) not in BUDGETS:
                continue
            key = (r["dataset"], int(r["num_samples"]), method, int(r["seed"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate row {key}")
            rows[key] = float(r["true_metric"])
    return rows


def lesser_cell(model: str, task: str, budget: int, seed: int, data_dir: Path = DATA_DIR) -> tuple[float, str]:
    path = Path(data_dir) / "results" / "lesser" / model / f"{task}_k{budget}_s{seed}.json"
    rec = read_json(path)
    want = {"model": MODELS[model].legacy_token, "method": LEGACY_METHOD["lesser"], "task": task,
            "budget": budget, "seed": seed, "base_model": MODELS[model].hf_id}
    for key, value in want.items():
        if rec.get(key) != value:
            raise ValueError(f"{path}: {key}={rec.get(key)!r}, expected {value!r}")
    return float(rec["metric_value"]), _rel(path, data_dir)


def fiveshot_mmlu_pro(model: str, method: str, budget: int, seed: int, data_dir: Path = DATA_DIR) -> tuple[float, str]:
    """Five-shot MMLU-Pro accuracy (0--100) of one rerun."""
    path = Path(data_dir) / "results" / "mmlu_pro_5shot" / model / f"{method}_k{budget}_s{seed}.json"
    rec = read_json(path)
    tag = FIVESHOT_TAG[method]
    if model == "llama-3.2-3b" and budget == 1000:   # these records use shorter tags
        want_tag = f"{tag}_5shot" if seed == 0 else f"{tag}_5shot_s{seed}"
    else:
        want_tag = f"{tag}_k{budget}_5shot_s{seed}"
    if rec.get("tag") != want_tag or rec.get("num_fewshot") != 5 or rec.get("n") != MMLU_PRO_QUESTIONS:
        raise ValueError(f"{path}: unexpected tag/shots/size "
                         f"{rec.get('tag')}, {rec.get('num_fewshot')}, {rec.get('n')}")
    return 100 * float(rec["accuracy"]), _rel(path, data_dir)


def _summarize(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None}


def downstream_cells(data_dir: Path = DATA_DIR, models=MODEL_ORDER) -> dict:
    """{(model, task, budget, method): {mean, std, seeds, values, sources}} for the paper's SFT grid.

    MMLU-Pro cells of all four methods come from the five-shot reruns; the other tasks use Nayak et al.'s released
    baselines and our LESSER cells. Every cell must have seeds 0, 1, 2.
    """
    data_dir = Path(data_dir)
    cells = {}
    for model in models:
        nayak = nayak_budget_rows(model, data_dir)
        for task in TASKS:
            for budget in BUDGETS:
                for method in DOWNSTREAM_METHODS:
                    values, sources = [], []
                    for seed in SEEDS:
                        if task == "mmlu_pro":
                            value, source = fiveshot_mmlu_pro(model, method, budget, seed, data_dir)
                        elif method == "lesser":
                            value, source = lesser_cell(model, task, budget, seed, data_dir)
                        else:
                            key = (task, budget, method, seed)
                            if key not in nayak:
                                raise KeyError(f"{model}: Nayak et al. CSV lacks {key}")
                            value = nayak[key]
                            source = f"nayak/{MODELS[model].nayak_dir}/budget_true_metric_budget.csv"
                        if not (math.isfinite(value) and 0 <= value <= 100):
                            raise ValueError(f"{model}/{task}/k{budget}/{method}/s{seed}: invalid score {value}")
                        values.append(value)
                        sources.append(source)
                    cells[model, task, budget, method] = {**_summarize(values), "seeds": list(SEEDS),
                                                          "values": values, "sources": sources}
    return cells


def _validate_curve(values, context) -> list[float]:
    if len(values) != N_BINS or not all(math.isfinite(v) and v >= 0 for v in values) or max(values) == min(values):
        raise ValueError(f"{context}: expected ten finite, non-negative, non-constant losses, got {values}")
    return [float(v) for v in values]


def lesser_bin_ce(model: str, task: str, b: int, data_dir: Path = DATA_DIR) -> float:
    """Mean dev-query cross-entropy after training on LESSER bin ``b`` (one run, seed 0)."""
    path = Path(data_dir) / "results" / "bins_ce" / model / f"{task}_bin{b}.json"
    rec = read_json(path)
    if "ce_loss" in rec:   # records written by the bin-cell runners
        want = {"task": task, "bin": b, "model": MODELS[model].hf_id,
                "convention": "mean over dev queries of construct_test_sample CE"}
        for key, value in want.items():
            if rec.get(key) != value:
                raise ValueError(f"{path}: {key}={rec.get(key)!r}, expected {value!r}")
        return float(rec["ce_loss"])
    if set(rec) == {"dev"}:   # Qwen3 cells were scored by Nayak et al.'s evaluation/ce_loss.py, same convention
        if model != "qwen3-4b-base":
            raise ValueError(f"{path}: 'dev'-only records are expected only for Qwen3-4B-Base")
        return float(rec["dev"])
    raise ValueError(f"{path}: unrecognized bin-cell record {sorted(rec)}")


def bin_curves(data_dir: Path = DATA_DIR, models=MODEL_ORDER) -> dict:
    """{(model, task, method): [loss for bins 0..9]} for LESSER (ours) and LESS / RDS+ (Nayak et al.)."""
    data_dir = Path(data_dir)
    curves = {}
    for model in models:
        for task in TASKS:
            curves[model, task, "lesser"] = _validate_curve(
                [lesser_bin_ce(model, task, b, data_dir) for b in range(N_BINS)], f"{model}/{task}/lesser")
        path = data_dir / "nayak" / MODELS[model].nayak_dir / "binning_ce_loss_quantile.csv"
        rows = {}
        by_name = {v: k for k, v in NAYAK_METHOD.items()}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not {"dataset", "method", "bin", "dev"} <= set(reader.fieldnames or ()):
                raise ValueError(f"{path}: unexpected columns {reader.fieldnames}")
            for r in reader:
                if r["method"] == "EMBED (RR)":
                    continue   # released but not shown in the paper
                method = by_name.get(r["method"])
                key = (r["dataset"], method, int(r["bin"]))
                if key[0] not in TASKS or method not in ("less", "rds") or not 0 <= key[2] < N_BINS:
                    raise ValueError(f"{path}: unexpected row {r}")
                if key in rows:
                    raise ValueError(f"{path}: duplicate bin {key}")
                rows[key] = float(r["dev"])
        for task in TASKS:
            for method in ("less", "rds"):
                curves[model, task, method] = _validate_curve(
                    [rows[task, method, b] for b in range(N_BINS)], f"{model}/{task}/{method}")
    return curves


def replication_cells(data_dir: Path = DATA_DIR) -> dict:
    """{(model, task, method): [scores by seed]} of our reruns of released selections (Appendix A.1)."""
    data_dir = Path(data_dir) / "results" / "replication"
    out = {}
    for method in ("less", "random"):
        for task in ("tydiqa", "gsm8k", "codex", "bbh"):
            values = []
            for seed in SEEDS:
                path = data_dir / "llama-3.2-3b" / f"{method}_{task}_k1000_s{seed}.json"
                rec = read_json(path)
                want = {"model": "meta-llama/Llama-3.2-3B", "method": method, "task": task, "seed": seed,
                        "budget": 1000}
                if any(rec.get(k) != v for k, v in want.items()):
                    raise ValueError(f"{path}: unexpected fields")
                values.append(float(rec["score"]))
            out["llama-3.2-3b", task, method] = values
    for model in ("qwen3-4b-base", "olmo3-7b"):
        path = data_dir / model / "less_tydiqa_k1000_s0.json"
        rec = read_json(path)
        want = {"model": MODELS[model].legacy_token, "method": "less", "task": "tydiqa", "budget": 1000, "seed": 0}
        if any(rec.get(k) != v for k, v in want.items()):
            raise ValueError(f"{path}: unexpected fields")
        out[model, "tydiqa", "less"] = [float(rec["metric_value"])]
    return out
