"""The k = 5,000 selections analysed in Section 5 and Appendices E-F (index lists into the Tulu pool).

Arms per model-task pair, as returned by ``load(model, task, arm)``:
  lesser  LESSER round-robin selection (output-layer gradients at the base model);
  less    released LESS round-robin selection of Nayak et al. (2026);
  rds     released RDS+ round-robin selection of Nayak et al. (2026);
  randA   the paper's Random subset, one list for every task of a model:
          ``numpy.random.default_rng(0).permutation(197196)[:5000]`` for Llama-2-7B (Nayak et al.'s
          ``selection/random.py --seed 0``) and ``numpy.random.RandomState(42).permutation(197196)[:5000]``
          for the other three models;
  randB   a second uniform sample of the pool, ``random.Random(1000 * task index + 99).sample(range(197196), 5000)``.

The first four are read from the SFT component's data (``data/sft/selections/<model>/<method>/<task>_k5000.json``;
randA is its ``random/mmlu_pro_k5000.json``, which holds the task-independent Random subset). Only randB is
stored here, in ``data/analysis/selections/k5000/<model>_<task>_randB.json``. Each list holds 5,000 unique
indices into ``Harvard-DCML/tulu-v2-197K-processed`` (train split, default order), in selection order.

``build`` rewrites the randB files from their rule.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import common
from analysis.common import MODELS, PAIRS, POOL_SIZE, TASK_KEYS

K = 5000
ARMS = ("lesser", "less", "rds", "randA", "randB")
SELECTION_DIR = common.DATA_DIR / "selections" / "k5000"
SFT_DATA_DIR = common.REPO_ROOT / "data" / "sft"
SFT_METHOD = {"lesser": "lesser", "less": "less", "rds": "rds"}


def selection_file(model, task, arm) -> Path:
    """The file holding one list; lesser, less, rds and randA come from data/sft."""
    from sft.common import selection_path
    common.check_model(model)
    common.check_task(task)
    if arm in SFT_METHOD:
        return selection_path(MODELS[model].sft_key, SFT_METHOD[arm], task, K, SFT_DATA_DIR)
    if arm == "randA":   # task-independent; data/sft ships it under the MMLU-Pro name
        return selection_path(MODELS[model].sft_key, "random", "mmlu_pro", K, SFT_DATA_DIR)
    if arm == "randB":
        return SELECTION_DIR / f"{model}_{task}_randB.json"
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def external_files() -> list:
    """The data/sft files this component reads (their sha256 are recorded in data/analysis/MANIFEST.json)."""
    return sorted({selection_file(m, t, arm) for m, t in PAIRS for arm in ("lesser", "less", "rds", "randA")})


def validate(indices, what) -> list:
    indices = [int(i) for i in indices]
    if len(indices) != K or len(set(indices)) != K or not all(0 <= i < POOL_SIZE for i in indices):
        raise ValueError(f"{what}: expected {K} unique pool indices")
    return indices


def load(model, task, arm) -> list:
    return validate(common.load_json(selection_file(model, task, arm)), f"{model} {task} {arm}")


def random_b(task) -> list:
    return random.Random(1000 * TASK_KEYS.index(task) + 99).sample(range(POOL_SIZE), K)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="write the randB files")
    p.add_argument("--out-dir", default=str(SELECTION_DIR))
    args = ap.parse_args(argv)
    for m, t in PAIRS:
        common.write_json(random_b(t), Path(args.out_dir) / f"{m}_{t}_randB.json", indent=None)
    print(f"wrote {len(PAIRS)} randB files to {args.out_dir}")


if __name__ == "__main__":
    main()
