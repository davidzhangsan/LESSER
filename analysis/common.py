"""Shared constants and helpers for the analysis scripts.

The four models and five tasks give the 20 model-task pairs used throughout Section 5 and
Appendices E-F. Keys (``llama2``, ``llama3``, ``qwen``, ``olmo3``) match the file names under
``data/analysis``.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "analysis"
DEFAULT_OUT = REPO_ROOT / "outputs" / "analysis"

POOL_DATASET = "Harvard-DCML/tulu-v2-197K-processed"
QUERY_DATASET = "Harvard-DCML/targeted-query-set-processed"
POOL_SIZE = 197_196


@dataclass(frozen=True)
class Model:
    key: str
    name: str      # name used in tables
    short: str     # name used in figure legends
    hf_id: str
    color: str     # model color shared by all analysis figures
    sft_key: str   # the model's key in sft/ and data/sft


MODELS = {
    "llama2": Model("llama2", "Llama-2-7B", "Llama-2", "meta-llama/Llama-2-7b-hf", "#819DCB", "llama-2-7b"),
    "llama3": Model("llama3", "Llama-3.2-3B", "Llama-3.2", "meta-llama/Llama-3.2-3B", "#C47A27", "llama-3.2-3b"),
    "qwen": Model("qwen", "Qwen3-4B-Base", "Qwen3", "Qwen/Qwen3-4B-Base", "#8A63B6", "qwen3-4b-base"),
    "olmo3": Model("olmo3", "OLMo3-7B", "OLMo3", "allenai/Olmo-3-1025-7B", "#506477", "olmo3-7b"),
}
MODEL_KEYS = tuple(MODELS)
TASKS = {"tydiqa": "TyDiQA", "mmlu_pro": "MMLU-Pro", "gsm8k": "GSM8K", "codex": "Codex", "bbh": "BBH"}
TASK_KEYS = tuple(TASKS)
PAIRS = tuple((m, t) for m in MODEL_KEYS for t in TASK_KEYS)


def pair_key(model: str, task: str) -> str:
    return f"{model}_{task}"


def check_model(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_KEYS}")
    return model


def check_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASK_KEYS}")
    return task


def require(path) -> Path:
    """Return ``path`` as a Path, raising if it does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"required input is missing: {path}")
    return path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(require(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(require(path)) as handle:
        return json.load(handle)


def write_json(obj, path, indent=1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=indent)
    os.replace(tmp, path)
    return path


def spearman(x, y) -> float:
    """Spearman correlation as the Pearson correlation of ordinal ranks (ties ranked by position).

    This is the estimator used for every rank correlation in Appendix F.1; with continuous scores
    it agrees with average-rank Spearman to rounding.
    """
    rx = np.argsort(np.argsort(np.asarray(x))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def median(values) -> float:
    """statistics.median of floats (the median used for every reported number)."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("median of an empty sequence")
    return float(statistics.median(values))
