"""Batch-gradient alignment between selections at the base model (Section 5.2, Figure 7a, Appendix F.2).

Reported results (20 model-task pairs, k = 5,000 selections, full-model gradients at the base model):
  * Figure 7a / F.2: centered alignment with LESS batches at batch size 128. Medians: LESSER 0.29,
    RDS+ 0.03, Random about 0 (-0.006), and 0.53 between two disjoint LESS batches. LESSER-LESS is
    positive in 17 of 20 pairs.
  * F.2: the raw (uncentered) cosine between two random batches of 128 has median 0.81.
  * F.2: across batch sizes 64-512, the median over pairs of (LESSER-LESS) / (LESS-LESS) is
    0.58-0.66 (0.63 at 128).

Estimator (Eq. of Appendix F.2). For batches S of one sample and S' of another,
    avg (G_S - mu1)^T (G_S' - mu2) / sqrt( avg (G_S - mu1)^T (G_S - mu2) * avg (G_S' - mu1)^T (G_S' - mu2) ),
where mu1 and mu2 are expected random-batch gradients estimated from random samples that are not part of
the comparison, with no random example in both: when neither sample is random, two of the four random
samples give mu1 and the other two mu2 (three splits, both orders); when one sample is random, one of
the other three random samples gives mu1 and another mu2 (all ordered choices). A batch gradient is the
gradient of the recipe's token-averaged loss; batches of n examples are token-weighted unions of n / 64
consecutive 64-example chunks, so every batch size comes from the same chunk gradients.

Pipeline (the GPU step writes about 1 GB of sketches per pair; ``gram`` reduces them to the shipped Gram matrices):
  batches  CPU  1,024-example samples of the LESSER, LESS, Random, RDS+ selections and three uniform
                pool samples (index lists; data/analysis/batch_alignment/batches/<model>.json).
  compute  GPU  per sample, 16 chunks of 64: exact squared norm, exact dot with the query gradient G_Q,
                and a CountSketch (m = 2^21, seed 20260923) of the chunk gradient; bf16 base model.
  gram     CPU  sketch files of one pair -> Gram matrix of all chunk sketches and G_Q (+ exact values).
  read     CPU  Gram matrices -> alignments (writes b128_comparison.json, batch_sizes.json, raw_cosine.json).
  figure   CPU  Figure 7a from b128_comparison.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import common, selections, sketch
from analysis.common import MODEL_KEYS, PAIRS, POOL_SIZE, TASK_KEYS

SELECTIONS = ("lesser", "less", "rds")           # samples of k = 5,000 selections
RANDOM = ("rand1", "rand2", "rand3", "rand4")    # rand1: sample of the paper's Random subset; rand2-4: uniform pool samples
SETS = SELECTIONS + RANDOM
SAMPLE_SIZE, CHUNK, N_CHUNKS = 1024, 64, 16
BATCH_SIZES = (64, 128, 256, 512)
# sample -> (k = 5,000 selection file arm, seed offset); the seed is 1000 * task index + offset
SAMPLE_SOURCES = {"lesser": ("lesser", 0), "less": ("less", 1), "rand1": ("randA", 2), "rds": ("rds", 3)}
GRAM_DIR = common.DATA_DIR / "batch_alignment" / "gram"
BATCHES_DIR = common.DATA_DIR / "batch_alignment" / "batches"


# ---------------------------------------------------------------------------------------------- batches
def build_batches(model: str) -> dict:
    """Index lists of the seven 1,024-example samples of every task, in the order used in the paper."""
    out = {}
    for ti, task in enumerate(TASK_KEYS):
        for sample, (arm, offset) in SAMPLE_SOURCES.items():
            sel = selections.load(model, task, arm)
            out[f"{task}_{sample}"] = random.Random(1000 * ti + offset).sample(sel, SAMPLE_SIZE)
        for j in (2, 3, 4):
            out[f"{task}_rand{j}"] = random.Random(1000 * ti + 10 + j).sample(range(POOL_SIZE), SAMPLE_SIZE)
    return out


# ---------------------------------------------------------------------------------------------- compute (GPU)
def compute(args):
    """Per sample: 16 chunk gradients of 64 examples at the base weights, exact norms and dots with G_Q, sketches.

    Recipe loss: the token-weighted mean cross-entropy over the chunk's supervised tokens, i.e. each example's
    mean loss weighted by its share of the chunk's tokens. The query gradient G_Q is the mean over queries of
    each query's response-token mean loss. Model in bf16 (the training precision); products in fp32/fp64.
    Each finished sample is saved to <out_dir>/<task>/<sample>.pt and skipped on restart.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sft.nayak_data import construct_test_sample, encode_with_messages_format
    batches = common.load_json(args.batches)
    tdir = Path(args.out_dir) / args.task
    tdir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in (args.samples or SETS) if not (tdir / f"{s}.pt").exists()]
    print(f"[{args.task}] {len(todo)} samples to compute: {todo}", flush=True)
    if not todo and (tdir / "q.pt").exists():
        return
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                                 device_map={"": torch.device(args.device)})
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = model.device
    names = [n for n, _ in model.named_parameters()]
    params = dict(model.named_parameters())

    def tensorize(row):
        return {k: torch.as_tensor(row[k], dtype=torch.long)[None] for k in ("input_ids", "attention_mask", "labels")}

    t0 = time.time()
    dq = load_dataset(common.QUERY_DATASET, args.task, split="dev")
    dq = dq.map(lambda x: construct_test_sample(sample=x, tokenizer=tok, max_length=2048))
    queries = [tensorize(dq[i]) for i in range(len(dq))]
    gq = {n: torch.zeros(p.shape, dtype=torch.float32) for n, p in params.items()}
    for q in queries:
        model.zero_grad(set_to_none=True)
        (model(**{k: v.to(dev) for k, v in q.items()}).loss / len(queries)).backward()
        for n, p in params.items():
            if p.grad is not None:
                gq[n] += p.grad.detach().float().cpu()
    model.zero_grad(set_to_none=True)
    if not (tdir / "q.pt").exists():
        sq = torch.zeros(args.m, dtype=torch.float64, device=dev)
        for n in names:
            sketch.sketch_add(sq, n, gq[n], args.seed)
        qnorm = math.sqrt(sum(float(gq[n].double().pow(2).sum()) for n in names))
        torch.save(dict(sketch=sq.float().cpu(), norm=qnorm), tdir / "q.pt")
        del sq
    print(f"[{args.task}] query gradient done in {time.time() - t0:.0f}s ({len(queries)} queries)", flush=True)

    pool = load_dataset(common.POOL_DATASET, split="train")
    if len(pool) != POOL_SIZE:
        raise ValueError(f"expected {POOL_SIZE} pool rows, found {len(pool)}")
    block = 1 << 24
    for key in todo:
        t1 = time.time()
        items = batches[f"{args.task}_{key}"]
        if len(items) != SAMPLE_SIZE or len(set(items)) != len(items):
            raise ValueError(f"{args.task}_{key}: expected {SAMPLE_SIZE} unique indices")
        sketches, dots, norms2, ntoks = [], [], [], []
        for c in range(len(items) // args.chunk):
            model.zero_grad(set_to_none=True)
            encs = [tensorize(encode_with_messages_format(pool[int(i)], tok, max_seq_length=2048))
                    for i in items[c * args.chunk:(c + 1) * args.chunk]]
            nt = [int((e["labels"][0, 1:] != -100).sum()) for e in encs]
            total = sum(nt)
            for enc, n_i in zip(encs, nt):
                if n_i == 0:
                    continue
                (model(**{k: v.to(dev) for k, v in enc.items()}).loss * n_i / total).backward()
            sk = torch.zeros(args.m, dtype=torch.float64, device=dev)
            dot = nn2 = 0.0
            for n in names:
                g = params[n].grad
                if g is None:
                    continue
                gf, qf = g.detach().reshape(-1), gq[n].reshape(-1)
                for s0 in range(0, gf.numel(), block):
                    x = gf[s0:s0 + block].float()
                    qq = qf[s0:s0 + block].to(dev)
                    dot += float((x.double() * qq.double()).sum())
                    nn2 += float(x.double().pow(2).sum())
                    sketch.sketch_add(sk, n, x, args.seed, offset=s0)
            if not math.sqrt(nn2) < args.grad_norm_ceiling:
                raise RuntimeError(f"{args.task} {key} chunk {c}: gradient norm {math.sqrt(nn2):.1f} above the ceiling")
            sketches.append(sk.float().cpu())
            dots.append(dot)
            norms2.append(nn2)
            ntoks.append(total)
        model.zero_grad(set_to_none=True)
        torch.save(dict(set=key, chunk=args.chunk, sketches=torch.stack(sketches), g_dot_q=dots, norm2=norms2,
                        ntok=ntoks, seconds=time.time() - t1, m=args.m, seed=args.seed), tdir / f"{key}.pt")
        print(f"[{args.task}] {key}: {len(sketches)} chunks, {time.time() - t1:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- gram (CPU)
def build_gram(sketch_dir) -> dict:
    """Gram matrix of the 7 x 16 chunk sketches and the query sketch of one pair, with the exact values."""
    import torch

    sketch_dir = Path(sketch_dir)
    q = torch.load(common.require(sketch_dir / "q.pt"), map_location="cpu", weights_only=False)
    rows, norm2, dots, ntok, sources = [], [], [], [], {}
    for s in SETS:
        path = common.require(sketch_dir / f"{s}.pt")
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d["sketches"].shape[0] != N_CHUNKS or d["chunk"] != CHUNK:
            raise ValueError(f"{path}: expected {N_CHUNKS} chunks of {CHUNK}")
        rows.append(d["sketches"].double())
        norm2 += list(d["norm2"])
        dots += list(d["g_dot_q"])
        ntok += list(d["ntok"])
        sources[f"{s}.pt"] = common.sha256_file(path)
    sources["q.pt"] = common.sha256_file(sketch_dir / "q.pt")
    X = torch.cat(rows + [q["sketch"].double()[None]])
    K = (X @ X.T).numpy()
    return dict(sets=np.array(SETS), chunks=N_CHUNKS, K_sketch=K, norm2=np.array(norm2, dtype=float),
                g_dot_q=np.array(dots, dtype=float), ntok=np.array(ntok, dtype=np.int64),
                q_norm=float(q["norm"]), m=int(X.shape[1]), sources=json.dumps(sources, sort_keys=True))


def load_gram(path):
    """Gram with exact chunk norms and chunk . G_Q substituted for their sketched values.

    Returns (K, index of (sample, chunk) -> row, row of G_Q, token counts, max relative sketch error of chunk . G_Q).
    """
    z = np.load(common.require(path), allow_pickle=False)
    sets = [str(s) for s in z["sets"]]
    if tuple(sets) != SETS or int(z["chunks"]) != N_CHUNKS:
        raise ValueError(f"{path}: unexpected layout {sets} x {int(z['chunks'])}")
    K = z["K_sketch"].copy()
    idx = {(s, c): i for i, (s, c) in enumerate((s, c) for s in sets for c in range(N_CHUNKS))}
    qi = K.shape[0] - 1
    qn = float(z["q_norm"])
    ntok, err = {}, []
    for (key, i) in idx.items():
        err.append(abs(K[i, qi] - z["g_dot_q"][i]) / math.sqrt(z["norm2"][i] * qn ** 2))
        K[i, i] = z["norm2"][i]
        K[i, qi] = K[qi, i] = z["g_dot_q"][i]
        ntok[key] = int(z["ntok"][i])
    K[qi, qi] = qn ** 2
    return K, idx, qi, ntok, max(err), z["K_sketch"]


# ---------------------------------------------------------------------------------------------- estimator
class Vec:
    """Linear combinations of chunk gradients as coefficient vectors over the Gram's rows."""

    def __init__(self, K, idx, ntok):
        self.K, self.idx, self.ntok = K, idx, ntok

    def mean(self, members):
        """Token-weighted mean of the listed chunks: the recipe's batch gradient of their union."""
        w = np.zeros(self.K.shape[0])
        tot = sum(self.ntok[k] for k in members)
        for k in members:
            w[self.idx[k]] += self.ntok[k] / tot
        return w

    def dot(self, a, b):
        return float(a @ self.K @ b)

    def cos(self, a, b):
        return self.dot(a, b) / math.sqrt(self.dot(a, a) * self.dot(b, b))


def groups(sample, n):
    """Disjoint batches of n examples: groups of n / 64 consecutive chunks."""
    j = n // CHUNK
    return [[(sample, c) for c in range(g * j, (g + 1) * j)] for g in range(N_CHUNKS // j)]


def centerings(V, rsets, n):
    """Pairs (mu1, mu2) of expected random-batch gradients from disjoint random samples.

    Each mu is the equal-weight average of that sample's batches of the same size n. Four samples: halves
    (three splits, both orders); otherwise one sample per side (all ordered choices).
    """
    gm = lambda sets: np.mean([V.mean(g) for s in sets for g in groups(s, n)], axis=0)
    if len(rsets) == 4:
        out = []
        for a, b in ((0, 1), (0, 2), (0, 3)):
            rest = [i for i in range(4) if i not in (a, b)]
            h1, h2 = gm([rsets[a], rsets[b]]), gm([rsets[i] for i in rest])
            out += [(h1, h2), (h2, h1)]
        return out
    return [(gm([a]), gm([b])) for a, b in itertools.permutations(rsets, 2)]


def cdot(V, x, y, cen):
    """Unbiased estimate of (x - mu)^T (y - mu)."""
    return float(np.mean([V.dot(x - r1, y - r2) for r1, r2 in cen]))


def centered_alignment(V, G, a, b, n):
    """Ratio of averages over all batch pairs of samples a and b (Eq. of Appendix F.2)."""
    cen = centerings(V, [k for k in RANDOM if k not in (a, b)], n)
    num = np.mean([cdot(V, x, y, cen) for x in G[a] for y in G[b]])
    na = np.mean([cdot(V, x, x, cen) for x in G[a]])
    nb = np.mean([cdot(V, y, y, cen) for y in G[b]])
    return float(num / math.sqrt(max(na, 1e-30) * max(nb, 1e-30)))


def self_alignment(V, G, a, n):
    """Centered alignment between two disjoint batches of the same sample (the LESS-LESS reference)."""
    cen = centerings(V, list(RANDOM), n)
    xs = G[a]
    num = np.mean([cdot(V, x, y, cen) for i, x in enumerate(xs) for j, y in enumerate(xs) if i < j])
    return float(num / np.mean([cdot(V, x, x, cen) for x in xs]))


def raw_cosines(K_sketch, idx, ntok, n=128):
    """Uncentered cosine between batches of n (sketched norms, as in the paper's raw-cosine check)."""
    V = Vec(K_sketch, idx, ntok)
    G = {s: [V.mean(g) for g in groups(s, n)] for s in SETS}

    def cos_raw(a, b):
        vals = [V.cos(x, y) for i, x in enumerate(G[a]) for j, y in enumerate(G[b]) if a != b or i != j]
        return float(np.mean(vals))

    return dict(rand_rand=float(np.mean([cos_raw(a, b) for a, b in itertools.combinations(RANDOM, 2)])),
                less_less=cos_raw("less", "less"), lesser_less=cos_raw("lesser", "less"),
                rds_less=cos_raw("rds", "less"), rand_less=float(np.mean([cos_raw(r, "less") for r in RANDOM])))


def read_pair(gram_path):
    K, idx, qi, ntok, err, K_sketch = load_gram(gram_path)
    V = Vec(K, idx, ntok)
    sizes = {}
    for n in BATCH_SIZES:
        G = {s: [V.mean(g) for g in groups(s, n)] for s in SETS}
        sizes[n] = dict(lesser_less=centered_alignment(V, G, "lesser", "less", n),
                        rds_less=centered_alignment(V, G, "rds", "less", n),
                        rand_less=[centered_alignment(V, G, k, "less", n) for k in RANDOM],
                        less_less=self_alignment(V, G, "less", n),
                        lesser_lesser=self_alignment(V, G, "lesser", n))
    return sizes, raw_cosines(K_sketch, idx, ntok), err


def read(gram_dir, out_dir, verbose=False) -> dict:
    """All pairs: writes b128_comparison.json, batch_sizes.json, raw_cosine.json and summary.json; returns the summary."""
    out_dir = Path(out_dir)
    b128, per_size, raw = [], [], {}
    for m, t in PAIRS:
        sizes, rc, err = read_pair(Path(gram_dir) / f"{m}_{t}.npz")
        s = sizes[128]
        b128.append(dict(model=m, task=t, lesser_less=s["lesser_less"], rds_less=s["rds_less"], rand_less=s["rand_less"],
                         less_less=s["less_less"]))
        per_size.append(dict(model=m, task=t, sketch_error=err, **{f"{k}_{n}": v[k] for n, v in sizes.items()
                                                                   for k in ("lesser_less", "less_less", "lesser_lesser", "rds_less")}))
        raw[f"{m}_{t}"] = rc
        if verbose:
            print(f"{m:7s} {t:9s} LESSER {s['lesser_less']:+.3f} | LESS-LESS {s['less_less']:+.3f} | RDS+ {s['rds_less']:+.3f} | "
                  f"random {np.mean(s['rand_less']):+.3f} | sketch error {err:.4f}", flush=True)
    common.write_json(b128, out_dir / "b128_comparison.json")
    common.write_json(per_size, out_dir / "batch_sizes.json")
    raw_summary = dict(cells=raw, median={k: common.median(r[k] for r in raw.values()) for k in next(iter(raw.values()))},
                       note="uncentered cosine between batches of 128 at the base model (sketched norms)")
    common.write_json(raw_summary, out_dir / "raw_cosine.json")
    summary = summarize(b128, per_size, raw_summary)
    common.write_json(summary, out_dir / "summary.json")
    return summary


def summarize(b128, per_size, raw_summary) -> dict:
    """The numbers quoted in Section 5.2 and Appendix F.2."""
    med = lambda key: common.median(v for r in b128 for v in (r[key] if key == "rand_less" else [r[key]]))
    return {
        "b128_median": {k: med(k) for k in ("lesser_less", "rds_less", "rand_less", "less_less")},
        "b128_lesser_less_positive": sum(r["lesser_less"] > 0 for r in b128),
        "raw_cosine_random_random_median": raw_summary["median"]["rand_rand"],
        "ratio_lesser_less_over_less_less_median": {
            str(n): common.median(r[f"lesser_less_{n}"] / r[f"less_less_{n}"] for r in per_size) for n in BATCH_SIZES},
        "n_pairs": len(b128),
    }


# ---------------------------------------------------------------------------------------------- figure 7a
def figure(b128_path, out) -> dict:
    """Figure 7a: centered alignment with LESS batches at batch size 128; dots are pairs, bars medians."""
    from analysis import vector_figures as vf

    src = common.require(b128_path)
    rows = common.load_json(src)
    if len(rows) != 20 or len({(r["model"], r["task"]) for r in rows}) != 20:
        raise ValueError("need the 20 model-task pairs once each")
    cats = [("less_less", "LESS", vf.METHOD_COLORS["LESS"], "other batch"),
            ("lesser_less", "LESSER", vf.METHOD_COLORS["LESSER"], None),
            ("rds_less", "RDS+", vf.METHOD_COLORS["RDS+"], None),
            ("rand_less", "Random", vf.METHOD_COLORS["Random"], None)]
    vals = {k: [v for r in rows for v in (r[k] if k == "rand_less" else [r[k]])] for k, *_ in cats}
    meds = {k: common.median(v) for k, v in vals.items()}
    payload = {"generator": "analysis/batch_alignment.py", "figure": "7a", "medians": meds,
               "source": {"file": src.name, "sha256": common.sha256_file(src)}}
    W7, H7 = vf.W7, vf.H7
    f = vf.Fig(out, W7, H7, payload)
    left, right, bottom, top = 30, W7 - 3, 32, H7 - 22
    f.text((left + right) / 2, H7 - 13, "(a) At the base model", vf.FT, "center")
    lo, hi = -0.55, 0.95
    ym = lambda v: bottom + (v - lo) / (hi - lo) * (top - bottom)
    for tv in (-0.5, 0, 0.5):
        f.line(left, ym(tv), right, ym(tv), vf.RULE if tv == 0 else vf.GRID, .6 if tv == 0 else .45)
        f.text(left - 3, ym(tv) - 2.6, f"{tv:g}", vf.FY, "right", color=vf.MUTED)
    f.axes(left, right, bottom, top)
    step = (right - left) / len(cats)
    for ci, (k, label, color, note) in enumerate(cats):
        cx = left + (ci + .5) * step
        rng = random.Random(ci)
        for v in vals[k]:
            if not lo <= v <= hi:
                raise ValueError(f"{k} value {v} outside the axis")
            f.pdf.saveState()
            f.pdf.setFillAlpha(.8 if k != "rand_less" else .5)
            f.marker(cx + (rng.random() - .5) * 16, ym(v), color, r=1.8 if k != "rand_less" else 1.3)
            f.pdf.restoreState()
        f.line(cx - 13, ym(meds[k]), cx + 13, ym(meds[k]), vf.INK, 1.4)
        f.line(cx, bottom, cx, bottom - 2, vf.RULE)
        f.text(cx, bottom - 11.5, label, vf.FX, "center", color=vf.MUTED)
        if note:
            f.text(cx, bottom - 21, note, 7, "center", color=vf.MUTED)
    f.vtext(9, (bottom + top) / 2, "Centered similarity", vf.FL)
    f.save()
    return payload


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("batches", help="rebuild the 1,024-example samples (index lists)")
    p.add_argument("--model", choices=MODEL_KEYS, nargs="+", default=list(MODEL_KEYS))
    p.add_argument("--out-dir", default=str(BATCHES_DIR))
    p = sub.add_parser("compute", help="GPU: chunk gradients, exact values and sketches for one pair")
    p.add_argument("--model", choices=MODEL_KEYS, required=True)
    p.add_argument("--task", choices=TASK_KEYS, required=True)
    p.add_argument("--model-path", required=True, help="local snapshot directory or hub id of the base model")
    p.add_argument("--batches", help="index lists (default data/analysis/batch_alignment/batches/<model>.json)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--m", type=int, default=sketch.BATCH_M)
    p.add_argument("--seed", type=int, default=sketch.BATCH_SEED)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--grad-norm-ceiling", type=float, default=500.0)
    p.add_argument("--samples", nargs="+", choices=SETS, help="compute only these samples (default: all seven)")
    p = sub.add_parser("gram", help="CPU: sketch files -> Gram matrices")
    p.add_argument("--sketch-root", required=True, nargs="+",
                   help="per model MODEL=DIR with DIR/<task>/{lesser,less,rds,rand1..4,q}.pt")
    p.add_argument("--out-dir", default=str(GRAM_DIR))
    p = sub.add_parser("read", help="CPU: Gram matrices -> alignments and summary")
    p.add_argument("--gram-dir", default=str(GRAM_DIR))
    p.add_argument("--out-dir", default=str(common.DEFAULT_OUT / "batch_alignment"))
    p = sub.add_parser("figure", help="CPU: Figure 7a")
    p.add_argument("--b128", default=str(common.DEFAULT_OUT / "batch_alignment" / "b128_comparison.json"))
    p.add_argument("--out", default=str(common.DEFAULT_OUT / "figures" / "fig7a_alignment_base.pdf"))
    args = ap.parse_args(argv)

    if args.cmd == "batches":
        for m in args.model:
            path = common.write_json(build_batches(m), Path(args.out_dir) / f"{m}.json", indent=None)
            print("wrote", path)
    elif args.cmd == "compute":
        args.batches = args.batches or str(BATCHES_DIR / f"{args.model}.json")
        compute(args)
    elif args.cmd == "gram":
        roots = dict(item.split("=", 1) for item in args.sketch_root)
        for m, root in roots.items():
            common.check_model(m)
            for t in TASK_KEYS:
                g = build_gram(Path(root) / t)
                out = Path(args.out_dir) / f"{m}_{t}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, **g)
                print("wrote", out, flush=True)
    elif args.cmd == "read":
        print(json.dumps(read(args.gram_dir, args.out_dir, verbose=True), indent=1))
    elif args.cmd == "figure":
        print("medians", json.dumps(figure(args.b128, args.out)["medians"]), "->", args.out)


if __name__ == "__main__":
    main()
