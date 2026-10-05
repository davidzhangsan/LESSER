"""Similarity-bin diagnostic: Figure 4, Figures 8--10 and the bin statistics of Section 4 and Appendix C.2.

Protocol of Nayak et al.'s ``quantile/convert_to_dist_quant.py`` applied to LESSER features:

  build    order the whole pool by the round-robin rule over a task's queries (``lesser.round_robin`` with k = pool
           size), split the order into 10 equal consecutive bins (bin 0 = most similar; the last bin also takes the
           remainder), and keep the first 500 examples of each bin -> ``<out>/<task>/bin<b>.json``.
  cell     fine-tune the base model on one bin with the downstream recipe and seed 0 (``sft/train_eval.py``) and record
           the mean dev-query cross-entropy -> ``<out>/<model>/<task>_bin<b>.json`` (GPU).
  compare  overlap of built bins with the released ones in ``data/sft/bins/<model>``.

Scores are computed as in the paper's builder: rows L2-normalized with torch, then one float32 matrix product with
numpy. The full-pool order is sensitive to float32 near-ties: with numpy 1.26 (OpenBLAS) the released Llama-3.2-3B
and Qwen3-4B-Base bins are reproduced exactly; with numpy 2.x, 0--6 of the 500 examples of a bin differ.
Figures and statistics are drawn from the released cells by ``sft/paper_figures.py``.

Run:
  python -m sft.bins build   --model llama-3.2-3b [--features-dir DIR] --out-dir outputs/sft/bins/llama-3.2-3b
  python -m sft.bins compare --model llama-3.2-3b --built outputs/sft/bins/llama-3.2-3b
  python -m sft.bins cell    --model llama-3.2-3b --task tydiqa --bin 0 [train_eval options]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lesser import round_robin

from .common import BIN_SIZE, DATA_DIR, N_BINS, TASKS, bin_path, features_dir, model_spec, read_selection, write_json
from .select import load_pool, load_queries, pool_shards


def _normalize_rows(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    return (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)).numpy()


def pool_order(pool: torch.Tensor, queries: torch.Tensor) -> list[int]:
    """Round-robin order of the whole pool (every index once)."""
    sim = _normalize_rows(queries) @ _normalize_rows(pool).T          # (n_queries, n_pool), float32
    order = round_robin(sim, sim.shape[1])
    if len(order) != sim.shape[1] or len(set(order)) != sim.shape[1]:
        raise RuntimeError("the round-robin order is not a permutation of the pool")
    return order


def split_bins(order: list[int], n_bins: int = N_BINS, per_bin: int = BIN_SIZE) -> list[list[int]]:
    size = len(order) // n_bins
    return [order[b * size:(len(order) if b == n_bins - 1 else (b + 1) * size)][:per_bin] for b in range(n_bins)]


def build(model: str, feat_dir: Path, tasks, out_dir: Path) -> None:
    model_dir = feat_dir / model
    pool = load_pool(pool_shards(model_dir))
    for task in tasks:
        bins = split_bins(pool_order(pool, load_queries(model_dir, task)))
        for b, idx in enumerate(bins):
            write_json(idx, Path(out_dir) / task / f"bin{b}.json")
        print(f"[bins] {model}/{task}: wrote {len(bins)} bins of {BIN_SIZE}", flush=True)


def compare(model: str, built: Path, tasks, data_dir: Path = DATA_DIR) -> bool:
    same_all = True
    for task in tasks:
        marks = []
        for b in range(N_BINS):
            ours = read_selection(Path(built) / task / f"bin{b}.json", BIN_SIZE)
            ref = read_selection(bin_path(model, task, b, data_dir), BIN_SIZE)
            same_all &= ours == ref
            marks.append("identical" if ours == ref else
                         "same set" if set(ours) == set(ref) else f"{len(set(ours) & set(ref))}/{BIN_SIZE} shared")
        print(f"[compare] {model}/{task}: " + "; ".join(f"bin {b}: {m}" for b, m in enumerate(marks)), flush=True)
    return same_all


def run_bin_cell(args) -> None:
    """Train on one released (or --bins-dir) bin and write a bin-cell record (mean dev-query cross-entropy)."""
    from .train_eval import run_cell
    spec = model_spec(args.model)
    sel = (Path(args.bins_dir) / args.task / f"bin{args.bin}.json") if args.bins_dir else \
        bin_path(args.model, args.task, args.bin, args.data_dir)
    read_selection(sel, BIN_SIZE)
    args.method, args.budget, args.seed, args.selection = f"bin{args.bin}", BIN_SIZE, 0, sel
    if args.pad_token_mode is None:   # as in the paper's bin cells (third_party/targeted-instruction-selection)
        args.pad_token_mode = "if_missing" if args.model == "qwen3-4b-base" else "always"
    args.dev_ce, args.no_eval = True, True
    record = run_cell(args)
    if record is None:   # dry run
        return
    write_json({"task": args.task, "bin": args.bin, "ce_loss": record["dev_ce"], "n_dev": record["n_dev"],
                "per_query": record["per_query"], "model": spec.hf_id, "selection": str(sel),
                "convention": record["convention"]},
               Path(args.out_dir) / "bins_ce" / args.model / f"{args.task}_bin{args.bin}.json", indent=1)


def main(argv=None) -> int:
    from .train_eval import add_arguments
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--features-dir", type=Path)
    b.add_argument("--out-dir", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--built", type=Path, required=True)
    c.add_argument("--data-dir", type=Path, default=DATA_DIR)
    for p in (b, c):
        p.add_argument("--model", required=True)
        p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    cell = sub.add_parser("cell")
    cell.add_argument("--model", required=True)
    cell.add_argument("--task", required=True, choices=TASKS)
    cell.add_argument("--bin", type=int, required=True, choices=range(N_BINS))
    cell.add_argument("--bins-dir", type=Path, help="bins built by `build` (default: the released data/sft/bins)")
    add_arguments(cell)
    args = ap.parse_args(argv)
    model_spec(args.model)
    if args.cmd == "build":
        build(args.model, features_dir(args.features_dir), args.tasks, args.out_dir)
    elif args.cmd == "compare":
        return 0 if compare(args.model, args.built, args.tasks, args.data_dir) else 1
    else:
        run_bin_cell(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
