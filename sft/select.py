"""Build LESSER selections from stored features, and check them against the released ones.

Paper results: the LESSER selections behind Tables 9--11, Figure 3 and the overlap table of Appendix E
(``data/sft/selections/<model>/lesser/<task>_k<k>.json``). Each task's query features are compared with every pool
feature by cosine similarity, and the round-robin rule of ``lesser.select`` picks one example per query in turn; the
three budgets are prefixes of one ranking.

Inputs (from ``sft/extract.py``), in ``<features-dir>/<model>/``:
  pool_prod_<start>_<end>.pt   one or more pool shards; together they must tile [0, 197196) exactly
  val_<task>_prod.pt           query features of each task

Shards are ordered by their start index and must not overlap or leave gaps, so a merged file and its parts cannot be
counted twice. A missing query file raises.

Run:
  python -m sft.select --model llama-3.2-3b [--features-dir DIR] --out-dir outputs/sft/selections/llama-3.2-3b
  python -m sft.select --model llama-3.2-3b --out-dir ... --compare-to data/sft/selections/llama-3.2-3b/lesser
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

from lesser import select as select_round_robin

from .common import BUDGETS, N_POOL, N_QUERIES, TASKS, features_dir, model_spec, read_selection, write_json

SHARD_RE = re.compile(r"^pool_prod_(\d+)_(\d+)\.pt$")
FEATURE_DIM = 8192


def pool_shards(model_dir: Path, explicit: list[Path] | None = None) -> list[Path]:
    """Pool shard files ordered by start index; they must tile [0, N_POOL) with no overlap or gap."""
    paths = [Path(p) for p in explicit] if explicit else sorted(Path(model_dir).glob("pool_prod_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no pool_prod_<start>_<end>.pt files in {model_dir}")
    spans = []
    for p in paths:
        m = SHARD_RE.match(p.name)
        if not m:
            raise ValueError(f"{p}: shard names must be pool_prod_<start>_<end>.pt")
        spans.append((int(m.group(1)), int(m.group(2)), p))
    spans.sort()
    cursor = 0
    for start, end, p in spans:
        if start != cursor:
            raise ValueError(f"pool shards do not tile the pool: {p.name} starts at {start}, expected {cursor} "
                             f"(overlapping or missing shards; pass --pool-shards explicitly)")
        cursor = end
    if cursor != N_POOL:
        raise ValueError(f"pool shards end at {cursor}, expected {N_POOL}")
    return [p for _, _, p in spans]


def load_pool(paths: list[Path]) -> torch.Tensor:
    parts = []
    for p in paths:
        x = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != FEATURE_DIM or x.dtype != torch.float32:
            raise ValueError(f"{p}: expected a float32 tensor with {FEATURE_DIM} columns")
        m = SHARD_RE.match(p.name)
        if x.shape[0] != int(m.group(2)) - int(m.group(1)):
            raise ValueError(f"{p}: {x.shape[0]} rows do not match its name")
        parts.append(x)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def load_queries(model_dir: Path, task: str) -> torch.Tensor:
    path = Path(model_dir) / f"val_{task}_prod.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing query features {path}")
    q = torch.load(path, map_location="cpu", weights_only=True)
    if q.shape != (N_QUERIES[task], FEATURE_DIM):
        raise ValueError(f"{path}: shape {tuple(q.shape)}, expected ({N_QUERIES[task]}, {FEATURE_DIM})")
    return q


def build(pool: torch.Tensor, model_dir: Path, tasks, budgets, out_dir: Path, device: str = "cpu") -> None:
    for task in tasks:
        picks = select_round_robin(pool, load_queries(model_dir, task), budgets=budgets, device=device)
        for k, idx in picks.items():
            if len(set(idx)) != k:
                raise RuntimeError(f"{task} k={k}: duplicate picks")
            write_json(idx, Path(out_dir) / f"{task}_k{k}.json")
        print(f"[select] {task}: wrote budgets {list(budgets)}", flush=True)


def compare(out_dir: Path, reference_dir: Path, tasks, budgets) -> bool:
    """Print order/set agreement of new selections with reference ones; True if every set is identical."""
    all_same = True
    for task in tasks:
        for k in budgets:
            ours = read_selection(Path(out_dir) / f"{task}_k{k}.json", k)
            ref = read_selection(Path(reference_dir) / f"{task}_k{k}.json", k)
            same_set = set(ours) == set(ref)
            first = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), None)
            jac = len(set(ours) & set(ref)) / len(set(ours) | set(ref))
            all_same &= same_set
            print(f"[compare] {task:8s} k={k:<6d} same set: {same_set}  same order: {first is None}"
                  + ("" if first is None else f" (first difference at position {first})") + f"  Jaccard {jac:.4f}")
    return all_same


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--features-dir", type=Path, help="default: $LESSER_ARTIFACTS/sft/features, else data/sft/features")
    ap.add_argument("--pool-shards", type=Path, nargs="+", help="explicit pool shard files (still checked for tiling)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--compare-to", type=Path, help="reference selections, e.g. data/sft/selections/<model>/lesser")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    model_spec(args.model)
    model_dir = features_dir(args.features_dir) / args.model
    shards = pool_shards(model_dir, args.pool_shards)
    print(f"[select] {args.model}: pool shards {[p.name for p in shards]}", flush=True)
    pool = load_pool(shards)
    build(pool, model_dir, args.tasks, sorted(args.budgets), args.out_dir, args.device)
    if args.compare_to:
        return 0 if compare(args.out_dir, args.compare_to, args.tasks, sorted(args.budgets)) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
