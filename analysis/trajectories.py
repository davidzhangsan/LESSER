"""SFT training trajectories on the k = 5,000 selections (Section 5.3, Figure 7b, Appendix F.3).

Reported results (20 model-task pairs, 80 AdamW steps at batch size 128 on k = 5,000 examples, two
training seeds for LESSER and LESS, one run on each of two random selections):
  * Figure 7b / Section 5.3: by step 10, the median share of the largest query-loss decrease over the 80
    steps is 84% for LESSER, 82% for LESS and 35% for Random. Each method's curve is the mean of its two
    runs; the share at step t is (L_0 - L_t) / max_s (L_0 - L_s). Pairs whose loss never falls below L_0 are
    left out (Random on one pair), so the medians are over 20, 20 and 19 pairs.
  * Appendix F.3 figure: query-loss curves of every pair (log step axis; step 1 equals step 0 because the
    first update has learning rate 0 under the 3-step warmup).

Runs: ``<model>_<task>_<arm>_s<seed>_n5000`` with arms lesser (s0, s1), less (s0, s1), randA (s0) and randB (s0),
120 runs in all; see ``analysis/selections.py``.

Subcommands:
  train      GPU  one run with the recipe (``train_sft_probe.py``), recording the query loss at every step.
  pack-loss  CPU  run directories -> data/analysis/trajectories/loss/*.json.gz (the raw loss files, gzipped).
  stats      CPU  loss files -> loss-share medians (the numbers above).
  figure     CPU  Figure 7b and the Appendix F.3 per-pair figure.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import common, selections
from analysis.common import MODEL_KEYS, MODELS, PAIRS, TASK_KEYS, TASKS

RUNS = ("lesser_s0", "lesser_s1", "less_s0", "less_s1", "randA_s0", "randB_s0")
TRAIN_ARMS = ("lesser", "less", "randA", "randB")
METHOD_RUNS = {"LESSER": ("lesser_s0", "lesser_s1"), "LESS": ("less_s0", "less_s1"), "Random": ("randA_s0", "randB_s0")}
N_STEPS = 80
LOSS_DIR = common.DATA_DIR / "trajectories" / "loss"
RECIPE = ["--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "128", "--num_train_epochs", "2",
          "--learning_rate", "2e-5", "--warmup_ratio", "0.03", "--lr_scheduler_type", "linear", "--weight_decay", "0.0",
          "--logging_steps", "1", "--bf16", "--report_to", "none", "--save_strategy", "no"]


def run_name(model, task, run, n=5000):
    arm, seed = run.rsplit("_s", 1)
    return f"{model}_{task}_{arm}_s{seed}_n{n}"


# ---------------------------------------------------------------------------------------------- train (GPU)
def write_train_file(model, task, arm, path, n=5000):
    """The selected examples, in selection order, as the JSON-lines file the recipe trains on."""
    from datasets import load_dataset
    idx = selections.load(model, task, arm)[:n]
    pool = load_dataset(common.POOL_DATASET, split="train")
    if len(pool) != common.POOL_SIZE:
        raise ValueError(f"expected {common.POOL_SIZE} pool rows")
    sub = pool.select([int(i) for i in idx])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in sub:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")
    return len(sub)


def train(args):
    run_dir = Path(args.runs_dir) / run_name(args.model, args.task, f"{args.arm}_s{args.seed}", args.n)
    loss_out = run_dir / "loss_trajectory.json"
    if loss_out.exists() and common.load_json(loss_out).get("complete"):
        print(f"{run_dir} is complete")
        return
    rows = write_train_file(args.model, args.task, args.arm, run_dir / "train.jsonl", args.n)
    print(f"train.jsonl rows {rows}", flush=True)
    cmd = [sys.executable, "-u", "-m", "analysis.train_sft_probe", "--model_name", args.model_path,
           "--output_dir", str(run_dir / "model"), "--seed", str(args.seed), "--train_dataset_path", str(run_dir / "train.jsonl"),
           "--num_samples", str(args.n), "--run_name", f"traj_{args.model}_{args.task}_{args.arm}_s{args.seed}",
           "--dev_task", args.task, "--loss_out", str(loss_out), *RECIPE]
    if args.gradient_checkpointing:
        cmd += ["--gradient_checkpointing", "true"]
    cmd += shlex.split(args.extra_args or "")
    with open(run_dir / "train.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=common.REPO_ROOT)
    shutil.rmtree(run_dir / "model", ignore_errors=True)
    if rc != 0:
        raise RuntimeError(f"training failed with exit code {rc}; see {run_dir / 'train.log'}")


# ---------------------------------------------------------------------------------------------- loss files
def pack_loss(runs_dir, out_dir=LOSS_DIR):
    """Gzip every run's loss_trajectory.json (byte-for-byte, fixed gzip mtime) and index their sha256."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for m, t in PAIRS:
        for run in RUNS:
            name = run_name(m, t, run)
            src = common.require(Path(runs_dir) / name / "loss_trajectory.json")
            raw = src.read_bytes()
            check_loss(json.loads(raw), name)
            with open(out_dir / f"{name}.json.gz", "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                gz.write(raw)
            index[name] = common.sha256_file(src)
    common.write_json(index, out_dir / "index.json")
    return index


def check_loss(d, name):
    if d.get("complete") is not True or [r["step"] for r in d["rows"]] != list(range(N_STEPS + 1)):
        raise ValueError(f"{name}: expected a complete loss trajectory with steps 0..{N_STEPS}")
    return d


def load_loss(model, task, run, loss_dir=LOSS_DIR):
    name = run_name(model, task, run)
    with gzip.open(common.require(Path(loss_dir) / f"{name}.json.gz"), "rb") as f:
        return check_loss(json.loads(f.read()), name)


def loss_curves(loss_dir=LOSS_DIR) -> dict:
    """{pair: {run: [mean query loss at steps 0..80]}}."""
    return {common.pair_key(m, t): {run: [r["mean_query_loss"] for r in load_loss(m, t, run, loss_dir)["rows"]] for run in RUNS}
            for m, t in PAIRS}


def method_curves(curves) -> dict:
    """{pair: {method: mean curve of its runs}}."""
    return {k: {name: [sum(v) / len(v) for v in zip(*(c[r] for r in runs))] for name, runs in METHOD_RUNS.items()}
            for k, c in curves.items()}


def loss_share_medians(curves):
    """Per step t: median over pairs of (L0 - Lt) / max_s (L0 - Ls), pairs without any decrease left out."""
    mc = method_curves(curves)
    out, counts = {}, {}
    for name in METHOD_RUNS:
        rows = []
        for cs in mc.values():
            c = cs[name]
            best = c[0] - min(c)
            if best > 0:
                rows.append([(c[0] - x) / best for x in c])
        out[name] = [common.median(r[s] for r in rows) for s in range(N_STEPS + 1)]
        counts[name] = len(rows)
    return out, counts


def stats(loss_dir=LOSS_DIR) -> dict:
    curves = loss_curves(loss_dir)
    share, counts = loss_share_medians(curves)
    return dict(loss_share_median={k: {str(s): v[s] for s in (2, 3, 5, 10, 80)} for k, v in share.items()},
                loss_share_pairs=counts, loss_share_median_all_steps=share)


# ---------------------------------------------------------------------------------------------- figures
def figure_share(share, counts, source, out):
    """Figure 7b: median share of the largest query-loss reduction reached by each step (log step axis)."""
    from analysis import vector_figures as vf
    payload = {"generator": "analysis/trajectories.py", "figure": "7b loss only", "loss_share_pairs": counts,
               "loss_share_median": {k: {s: v[s] for s in (2, 3, 5, 10, 80)} for k, v in share.items()},
               "source": {"file": Path(source).name, "sha256": common.sha256_file(source)}}
    f = vf.Fig(out, vf.W7, vf.H7, payload)
    left, right, bottom, top = 36, vf.W7 - 9, 32, vf.H7 - 22
    f.text((left + right) / 2, vf.H7 - 13, "(b) During training", vf.FT, "center")
    xm = lambda s: left + math.log(s) / math.log(80) * (right - left)
    lo, hi = -0.8, 1.05
    ym = lambda v: bottom + (v - lo) / (hi - lo) * (top - bottom)
    for tv, lab in ((-0.5, "-50%"), (0, "0%"), (0.5, "50%"), (1.0, "100%")):
        f.line(left, ym(tv), right, ym(tv), vf.RULE if tv == 0 else vf.GRID, .6 if tv == 0 else .45)
        f.text(left - 3, ym(tv) - 2.6, lab, vf.FY, "right", color=vf.MUTED)
    f.axes(left, right, bottom, top)
    f.line(xm(10), bottom, xm(10), top, vf.RULE, .7, (2, 2))
    styles = [("LESSER", vf.METHOD_COLORS["LESSER"], ()), ("LESS", vf.METHOD_COLORS["LESS"], (4, 2.4)),
              ("Random", vf.METHOD_COLORS["Random"], (1, 1.7))]
    for name, color, dash in styles:
        v = share[name]
        if any(not lo <= x <= hi for x in v[1:]):
            raise ValueError(f"{name} median share outside the axis")
        f.polyline([(xm(s), ym(v[s])) for s in range(1, N_STEPS + 1)], color, 1.3 if name != "Random" else 1.1, dash)
    f.vtext(10, (bottom + top) / 2, "Share of query-loss reduction", vf.FL)
    lw = max(vf.string_width(n, "Helvetica", vf.FLEG) for n, _, _ in styles)
    lgx = right - 3 - lw - 16
    for n, (name, color, dash) in enumerate(styles):
        ly = ym(0) - 11 - n * 10
        f.line(lgx, ly + 2.6, lgx + 12, ly + 2.6, color, 1.4, dash)
        f.text(lgx + 16, ly, name, vf.FLEG)
    for s in (1, 2, 5, 10, 20, 40, 80):
        f.line(xm(s), bottom, xm(s), bottom - 2, vf.RULE)
        f.text(xm(s), bottom - 11.5, str(s), vf.FX, "center", color=vf.MUTED)
    f.text((left + right) / 2, 5, "Training step (log scale)", vf.FL, "center")
    f.save()
    return payload


def figure_curves(curves, source, out):
    """Appendix F.3: query loss of every pair; mean of each method's runs, steps 1..80 on a log axis."""
    from analysis import vector_figures as vf
    mc = method_curves(curves)
    for k, cs in mc.items():
        for name, c in cs.items():
            if abs(c[1] - c[0]) > 1e-9:
                raise ValueError(f"{k} {name}: step 1 differs from step 0; the log axis would hide a change")
    payload = {"generator": "analysis/trajectories.py", "figure": "query loss, log step axis", "curves": mc,
               "source": {"file": Path(source).name, "sha256": common.sha256_file(source)}}
    styles = [("LESSER", vf.METHOD_COLORS["LESSER"], ()), ("LESS", vf.METHOD_COLORS["LESS"], (4, 2.4)),
              ("Random", vf.METHOD_COLORS["Random"], (1, 1.7))]
    width, height = 396, 300
    f = vf.Fig(out, width, height, payload)
    x = 104
    for name, color, dash in styles:
        f.line(x, height - 8, x + 13, height - 8, color, 1.2, dash)
        f.text(x + 17, height - 10.7, name, 8.5)
        x += 17 + vf.string_width(name, "Helvetica", 8.5) + 14
    f.line(x, height - 12, x, height - 3, vf.RULE, .7, (2, 2))
    f.text(x + 4, height - 10.7, "step 10", 8, color=vf.MUTED)
    left, right, gap_x, gap_y, top0, bottom0 = 56, width - 4, 16, 17, height - 30, 24
    pw = (right - left - 4 * gap_x) / 5
    ph = (top0 - bottom0 - 3 * gap_y) / 4
    lx = lambda s: math.log(s) / math.log(80)
    for ri, m in enumerate(MODEL_KEYS):
        yb = top0 - (ri + 1) * ph - ri * gap_y
        f.vtext(8, yb + ph / 2, MODELS[m].short, 7.8, bold=True)
        for ci, t in enumerate(TASK_KEYS):
            x0 = left + ci * (pw + gap_x)
            cs = mc[common.pair_key(m, t)]
            vals = [v for c in cs.values() for v in c[1:]]
            lo, hi = min(vals), max(vals)
            pad = .08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            tk, dg = vf.ticks_for(lo, hi, 3)
            xm = lambda s, x0=x0: x0 + lx(s) * pw
            ym = lambda v, yb=yb, lo=lo, hi=hi: yb + (v - lo) / (hi - lo) * ph
            if ri == 0:
                f.text(x0 + pw / 2, top0 + 5, TASKS[t], 8, "center", bold=True)
            for tv in tk:
                f.line(x0, ym(tv), x0 + pw, ym(tv), vf.GRID, .4)
                f.text(x0 - 2.5, ym(tv) - 2.4, f"{tv:.{dg}f}", 6.4, "right", color=vf.MUTED)
            f.line(x0, yb, x0 + pw, yb, vf.RULE, .5)
            for s in (1, 10, 80):
                f.line(xm(s), yb, xm(s), yb - 2, vf.RULE)
                if ri == 3:
                    f.text(xm(s), yb - 9, str(s), 6.6, "center", color=vf.MUTED)
            f.line(xm(10), yb, xm(10), yb + ph, vf.RULE, .6, (2, 2))
            for name, color, dash in styles:
                f.polyline([(xm(s), ym(cs[name][s])) for s in range(1, N_STEPS + 1)], color,
                           1.0 if name != "LESSER" else 1.15, dash)
    f.text(left + (right - left) / 2, 3, "Training step (log scale)", 8.2, "center")
    f.vtext(22, bottom0 + (top0 - bottom0) / 2, "Query CE loss", 8)
    f.save()
    return payload


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train", help="GPU: one recipe run with the query loss at every step")
    p.add_argument("--model", choices=MODEL_KEYS, required=True)
    p.add_argument("--task", choices=TASK_KEYS, required=True)
    p.add_argument("--arm", choices=TRAIN_ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model-path", required=True, help="local snapshot directory or hub id of the base model")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--gradient-checkpointing", action="store_true", help="same gradients, less memory")
    p.add_argument("--extra-args", help="further TrainingArguments flags, e.g. '--use_cpu true' for a CPU smoke test")
    p = sub.add_parser("pack-loss", help="CPU: run directories -> data/analysis/trajectories/loss")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out-dir", default=str(LOSS_DIR))
    p = sub.add_parser("stats", help="CPU: loss-share medians")
    p.add_argument("--loss-dir", default=str(LOSS_DIR))
    p.add_argument("--out-dir", default=str(common.DEFAULT_OUT / "trajectories"))
    p = sub.add_parser("figure", help="CPU: Figure 7b and the Appendix F.3 per-pair figure")
    p.add_argument("--loss-dir", default=str(LOSS_DIR))
    p.add_argument("--out-dir", default=str(common.DEFAULT_OUT / "trajectories"))
    p.add_argument("--fig-dir", default=str(common.DEFAULT_OUT / "figures"))
    args = ap.parse_args(argv)

    if args.cmd == "train":
        train(args)
    elif args.cmd == "pack-loss":
        print(f"packed {len(pack_loss(args.runs_dir, args.out_dir))} loss files")
    elif args.cmd == "stats":
        s = stats(args.loss_dir)
        out = common.write_json(s, Path(args.out_dir) / "loss_share.json")
        print(json.dumps({k: {"10": v["10"]} for k, v in s["loss_share_median"].items()}), s["loss_share_pairs"])
        print("wrote", out)
    elif args.cmd == "figure":
        curves = loss_curves(args.loss_dir)
        src = common.write_json(curves, Path(args.out_dir) / "loss_curves.json")
        share, counts = loss_share_medians(curves)
        figure_share(share, counts, src, Path(args.fig_dir) / "fig7b_loss_share.pdf")
        figure_curves(curves, src, Path(args.fig_dir) / "query_loss_curves.pdf")
        print("wrote", Path(args.fig_dir) / "fig7b_loss_share.pdf", "and", Path(args.fig_dir) / "query_loss_curves.pdf")


if __name__ == "__main__":
    main()
