"""Constants, model registry, and paths shared by the SFT scripts.

Paths default to this repository (``data/sft``); large feature files resolve from ``--features-dir``, else
``$LESSER_ARTIFACTS/sft/features``, else ``data/sft/features``. Nothing falls back silently: a missing input raises.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "sft"
DEFAULT_OUT = REPO_ROOT / "outputs" / "sft"

# Candidate pool and query sets (Nayak et al., 2026)
POOL_DATASET = "Harvard-DCML/tulu-v2-197K-processed"   # train split, default order
QUERY_DATASET = "Harvard-DCML/targeted-query-set-processed"   # one config per task, split "dev"
N_POOL = 197_196
MAX_SEQ_LENGTH = 2048

TASKS = ("tydiqa", "mmlu_pro", "gsm8k", "codex", "bbh")
TASK_LABEL = {"tydiqa": "TyDiQA", "mmlu_pro": "MMLU-Pro", "gsm8k": "GSM8K", "codex": "Codex", "bbh": "BBH"}
N_QUERIES = {"tydiqa": 9, "mmlu_pro": 70, "gsm8k": 8, "codex": 16, "bbh": 81}
BUDGETS = (1000, 5000, 10000)
SEEDS = (0, 1, 2)
N_BINS, BIN_SIZE = 10, 500

# Methods as named in this release, in the paper's table order, and their names elsewhere.
METHODS = ("lesser", "less", "rds", "random")
METHOD_LABEL = {"lesser": "LESSER", "less": "LESS", "rds": "RDS+", "random": "Random"}
NAYAK_METHOD = {"less": "LESS (RR)", "rds": "RDS+ (RR)", "random": "Random"}   # method column of Nayak's CSVs


@dataclass(frozen=True)
class ModelSpec:
    key: str            # release identifier
    label: str          # name used in the paper
    hf_id: str          # Hugging Face model
    vocab: int          # rows of the readout matrix W used for features
    hidden: int         # hidden size d
    gradient_checkpointing: bool   # the 7B models were fine-tuned with gradient checkpointing
    nayak_dir: str      # model directory in Nayak et al.'s released result tables
    legacy_token: str   # model token in the released per-cell result records


MODELS = {m.key: m for m in (
    ModelSpec("llama-3.2-3b", "Llama-3.2-3B", "meta-llama/Llama-3.2-3B", 128257, 3072, False, "llama3.2-3b", "llama"),
    ModelSpec("llama-2-7b", "Llama-2-7B", "meta-llama/Llama-2-7b-hf", 32000, 4096, True, "llama2-7b", "llama2-7b"),
    ModelSpec("qwen3-4b-base", "Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", 151936, 2560, False, "qwen3-4b-base", "qwen3-4b"),
    ModelSpec("olmo3-7b", "OLMo3-7B", "allenai/Olmo-3-1025-7B", 100278, 4096, True, "olmo3-7b", "olmo3-7b"),
)}
MODEL_ORDER = tuple(MODELS)   # the paper's table order


def model_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choices: {', '.join(MODELS)}")
    return MODELS[key]


def features_dir(arg: str | Path | None = None) -> Path:
    """Directory holding ``<model>/pool_prod_<start>_<end>.pt`` and ``<model>/val_<task>_prod.pt``."""
    if arg is not None:
        return Path(arg)
    if os.environ.get("LESSER_ARTIFACTS"):
        return Path(os.environ["LESSER_ARTIFACTS"]) / "sft" / "features"
    return DATA_DIR / "features"


def selection_path(model: str, method: str, task: str, budget: int, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / "selections" / model / method / f"{task}_k{budget}.json"


def bin_path(model: str, task: str, b: int, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / "bins" / model / task / f"bin{b}.json"


def display_path(path: str | Path) -> str:
    """A path relative to the repository root when it lies inside it (for messages)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_selection(path: str | Path, expected_size: int | None = None) -> list[int]:
    """A selection file: a JSON list of distinct pool indices, in selection order."""
    idx = read_json(path)
    if not isinstance(idx, list) or not all(isinstance(i, int) for i in idx):
        raise ValueError(f"{path}: expected a JSON list of integers")
    if len(set(idx)) != len(idx) or (idx and (min(idx) < 0 or max(idx) >= N_POOL)):
        raise ValueError(f"{path}: duplicate or out-of-range pool indices")
    if expected_size is not None and len(idx) != expected_size:
        raise ValueError(f"{path}: {len(idx)} indices, expected {expected_size}")
    return idx


def write_json(obj, path: str | Path, indent: int | None = None) -> None:
    """Atomic write (temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + ("\n" if indent else ""))
    tmp.replace(path)


def unwrap_cc(text: str) -> str:
    """Remove the paper's text-color macro \\cc{...}, keeping its argument."""
    out, i = [], 0
    while (j := text.find("\\cc{", i)) >= 0:
        out.append(text[i:j]); k, depth = j + 4, 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0); k += 1
        out.append(text[j + 4:k - 1]); i = k
    return "".join(out) + text[i:]
