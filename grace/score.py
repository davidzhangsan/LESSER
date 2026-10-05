#!/usr/bin/env python
"""GRACE scores and teacher-bank statistics from stored feature banks (tier-2 regeneration).

This script computes the per-teacher numbers behind the GRACE results of the LESSER paper.

* ``grace``: the GRACE score of every teacher in one setting, once with full-gradient features
  and once with LESSER (output-layer gradient) features. The scorer is the released
  ``GRACE/GRACE_computation.py::grace`` from a GRACE checkout, called with the defaults of the
  released pipeline (``scripts/grace.sh``). Paper: Table 12 and Appendix A.3.
* ``bank-stats``: the teacher-level statistics of Appendix F.4 (spectral entropy, effective rank,
  six other aggregate statistics, and the 64 most central responses) for both feature types.

Both subcommands read the teacher roster from ``data/grace/scores_<setting>.json`` and write a
JSON file with the schema of the matching reference file in ``data/grace``, so
``grace/tables.py --data-dir`` accepts the output in place of ``data/grace``. With ``--check``
the output is compared with the reference and any mismatch raises.

Feature banks are the pickles written by the patched GRACE pipeline (third_party/GRACE):
``Gradients_<teacher>.pkl`` (full gradients) and ``Prod_<teacher>.pkl`` (LESSER). Each holds
2,048 float32 vectors of length 512 in prompt-major order, four responses per prompt.
The stored banks are not distributed; bank-stats records the sha256 of each bank it reads.

Usage:
  python grace/score.py grace --setting gsm8k_llama1b --features-dir DIR --grace-repo GRACE_DIR \
      --out OUT.json [--check]
  python grace/score.py bank-stats --features-dir DIR --out OUT.json [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO, "data", "grace")
SETTINGS = ("gsm8k_llama1b", "gsm8k_olmo1b", "math_llama3b")
FEATURE_FILES = {"full": "Gradients_{short}.pkl", "lesser": "Prod_{short}.pkl"}
N_ROWS, PROJ_DIM = 2048, 512

# The scorer must be the released one: sha256 of GRACE/GRACE_computation.py at upstream 64fc99a.
# lesser_grace.patch does not modify this file.
UPSTREAM_SCORER_SHA256 = "db76b9702ef318895b141bb9eb7140ddb824523baa2dcacb86989a944a941458"
# Keyword arguments of the released pipeline (scripts/grace.sh and GRACE_computation.main).
GRACE_KWARGS = {"dim": 512, "n_gen_per_prompt": 4, "test_fraction": 0.1, "n_splits": 10,
                "smooth_coeff": 1e-3, "normalize_train": True, "normalize_test": False}

# Appendix F.4 statistics. All are computed on row-normalized features; the eigenvalues are those
# of the second-moment matrix X^T X / n, which sum to 1.
STATISTICS = {
    "spectral_entropy": "Shannon entropy (natural log) of the eigenvalues",
    "effective_rank": "1 / sum of squared eigenvalues",
    "sum_sqrt_eigenvalues": "sum of the square roots of the eigenvalues",
    "top_eigenvalue_share": "largest eigenvalue",
    "mean_pairwise_cosine": "mean cosine similarity over all pairs of responses",
    "p95_pairwise_cosine": "95th percentile of the pairwise cosine similarities",
    "frac_pairs_cosine_gt_0.8": "fraction of pairs with cosine similarity above 0.8",
    "skew_pairwise_cosine": "skewness (population) of the pairwise cosine similarities",
}
N_CENTRAL = 64


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: dict) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    print(f"wrote {path}")


def load_bank(path: str) -> np.ndarray:
    """Load one feature bank as float64 (2048, 512) and reject malformed or degenerate banks."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        X = np.asarray(pickle.load(f), dtype=np.float64)
    if X.shape != (N_ROWS, PROJ_DIM):
        raise ValueError(f"{path}: shape {X.shape}, expected {(N_ROWS, PROJ_DIM)}")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite values")
    if X.std() <= 0:
        raise ValueError(f"{path}: zero variance")
    n_unique = len(np.unique(X, axis=0))
    if n_unique <= N_ROWS // 2:
        raise ValueError(f"{path}: only {n_unique}/{N_ROWS} unique rows")
    if (np.linalg.norm(X, axis=1) <= 0).any():
        raise ValueError(f"{path}: zero-norm row")
    return X


def load_grace_scorer(grace_repo: str):
    """Import grace() from GRACE/GRACE_computation.py of a GRACE checkout, after checking that
    the file is byte-identical to the released scorer."""
    path = os.path.join(grace_repo, "GRACE", "GRACE_computation.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} (pass the GRACE checkout with --grace-repo)")
    digest = sha256_file(path)
    if digest != UPSTREAM_SCORER_SHA256:
        raise ValueError(f"{path} differs from the released scorer (sha256 {digest})")
    spec = importlib.util.spec_from_file_location("GRACE_computation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grace, digest


def reference_path(data_dir: str, kind: str, setting: str) -> str:
    return os.path.join(data_dir, f"{kind}_{setting}.json")


# ----------------------------------------------------------------------------- grace scores

def score_setting(ref: dict, features_dir: str, grace_fn) -> list[dict]:
    if ref["grace"] != GRACE_KWARGS:
        raise ValueError(f"reference GRACE settings {ref['grace']} differ from {GRACE_KWARGS}")
    rows = []
    for t in ref["teachers"]:
        row = {"short": t["short"], "hf_id": t["hf_id"]}
        inputs = {}
        for kind, pattern in FEATURE_FILES.items():
            path = os.path.join(features_dir, pattern.format(short=t["short"]))
            row[kind] = float(grace_fn(gradients=load_bank(path), **GRACE_KWARGS))
            inputs[kind] = {"file": os.path.basename(path), "sha256": sha256_file(path)}
        row["features"] = inputs
        rows.append(row)
        print(f"{t['short']:28s} full {row['full']:.10f}  lesser {row['lesser']:.10f}", flush=True)
    return rows


def check_scores(new: dict, ref: dict) -> None:
    """Scores must agree to the reference precision (10 decimals for row 1, full precision
    otherwise; a relative 1e-9 absorbs BLAS/LAPACK rounding differences)."""
    decimals = ref.get("score_decimals")
    rounding = 0.0 if decimals is None else 0.5 * 10.0 ** (-decimals) * (1 + 1e-6)
    ref_by_short = {t["short"]: t for t in ref["teachers"]}
    if [t["short"] for t in new["teachers"]] != [t["short"] for t in ref["teachers"]]:
        raise AssertionError("teacher roster differs from the reference")
    worst = 0.0
    for t in new["teachers"]:
        for kind in FEATURE_FILES:
            a, b = t[kind], ref_by_short[t["short"]][kind]
            tol = rounding + 1e-9 * abs(b)
            worst = max(worst, abs(a - b))
            if abs(a - b) > tol:
                raise AssertionError(f"{t['short']} {kind}: {a!r} vs reference {b!r} (tolerance {tol:.1e})")
    print(f"check passed: {len(new['teachers'])} teachers x 2 feature types, max |diff| {worst:.2e}")


# ------------------------------------------------------------------------- F.4 statistics

def bank_statistics(X: np.ndarray) -> dict:
    """Appendix F.4 statistics of one bank (rows are normalized here)."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    lam = np.clip(np.linalg.eigvalsh((X.T @ X) / n), 0.0, None)
    if abs(lam.sum() - 1.0) > 1e-8:
        raise ValueError(f"eigenvalues sum to {lam.sum()}, expected 1")
    lam_pos = lam[lam > 1e-12]
    cos = (X @ X.T)[np.triu_indices(n, k=1)]
    centered = cos - cos.mean()
    center = X.mean(axis=0)
    center /= np.linalg.norm(center)
    central = np.argsort(-(X @ center))[:N_CENTRAL]
    stats = {
        "spectral_entropy": float(-(lam_pos * np.log(lam_pos)).sum()),
        "effective_rank": float(1.0 / np.sum(lam_pos ** 2)),
        "sum_sqrt_eigenvalues": float(np.sqrt(lam_pos).sum()),
        "top_eigenvalue_share": float(lam.max()),
        "mean_pairwise_cosine": float(cos.mean()),
        "p95_pairwise_cosine": float(np.quantile(cos, 0.95)),
        "frac_pairs_cosine_gt_0.8": float((cos > 0.8).mean()),
        "skew_pairwise_cosine": float((centered ** 3).mean() / (centered ** 2).mean() ** 1.5),
    }
    stats[f"central_{N_CENTRAL}"] = [int(i) for i in central]
    return stats


def stats_setting(roster: dict, features_dir: str) -> list[dict]:
    rows = []
    for t in roster["teachers"]:
        row = {"short": t["short"]}
        for kind, pattern in FEATURE_FILES.items():
            path = os.path.join(features_dir, pattern.format(short=t["short"]))
            row[kind] = bank_statistics(load_bank(path))
            row[kind]["features_sha256"] = sha256_file(path)
        rows.append(row)
        print(f"{t['short']:28s} spectral entropy full {row['full']['spectral_entropy']:.4f} "
              f"lesser {row['lesser']['spectral_entropy']:.4f}", flush=True)
    return rows


def check_stats(new: dict, ref: dict) -> None:
    if [t["short"] for t in new["teachers"]] != [t["short"] for t in ref["teachers"]]:
        raise AssertionError("teacher roster differs from the reference")
    worst = 0.0
    for t, r in zip(new["teachers"], ref["teachers"]):
        for kind in FEATURE_FILES:
            for name in STATISTICS:
                a, b = t[kind][name], r[kind][name]
                rel = abs(a - b) / max(abs(b), 1e-300)
                worst = max(worst, rel)
                if rel > 1e-9:
                    raise AssertionError(f"{t['short']} {kind} {name}: {a!r} vs reference {b!r}")
            key = f"central_{N_CENTRAL}"
            if set(t[kind][key]) != set(r[kind][key]):
                raise AssertionError(f"{t['short']} {kind}: central responses differ from the reference")
    print(f"check passed: {len(new['teachers'])} teachers x 2 feature types, max relative diff {worst:.2e}")


# ----------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grace", help="GRACE scores of one setting (Table 12)")
    g.add_argument("--setting", required=True, choices=SETTINGS)
    g.add_argument("--grace-repo", required=True, help="GRACE checkout that holds GRACE/GRACE_computation.py")
    s = sub.add_parser("bank-stats", help="teacher-bank statistics (Appendix F.4)")
    s.add_argument("--setting", default="gsm8k_llama1b", choices=SETTINGS)
    for p in (g, s):
        p.add_argument("--features-dir", required=True, help="directory with Gradients_*.pkl and Prod_*.pkl")
        p.add_argument("--out", required=True, help="output JSON path")
        p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="reference files (default: data/grace)")
        p.add_argument("--check", action="store_true", help="compare the output with the reference file")
    args = ap.parse_args()

    roster = read_json(reference_path(args.data_dir, "scores", args.setting))
    if args.command == "grace":
        grace_fn, scorer_sha = load_grace_scorer(args.grace_repo)
        out = {k: v for k, v in roster.items() if k not in ("teachers", "source")}
        out["score_decimals"] = None  # full precision
        out["computed"] = {"by": "grace/score.py grace", "scorer_sha256": scorer_sha}
        out["teachers"] = score_setting(roster, args.features_dir, grace_fn)
        write_json(args.out, out)
        if args.check:
            check_scores(out, roster)
    else:
        out = {
            "description": "Teacher-level statistics of the student's feature banks under full-gradient "
                           "and LESSER features; LESSER paper Appendix F.4.",
            "setting": args.setting, "task": roster["task"], "student": roster["student"],
            "statistics": STATISTICS,
            "central": f"central_{N_CENTRAL}: indices of the {N_CENTRAL} responses whose normalized "
                       "features have the largest cosine similarity to the normalized mean feature",
            "computed": {"by": "grace/score.py bank-stats"},
            "teachers": stats_setting(roster, args.features_dir),
        }
        write_json(args.out, out)
        if args.check:
            check_stats(out, read_json(reference_path(args.data_dir, "bank_stats", args.setting)))


if __name__ == "__main__":
    main()
