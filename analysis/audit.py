"""Per-example gradient audit at the base model (Appendix F.1): compute step and the compact records.

For each model-task pair the audit samples candidates from the pool and computes, at the base weights,
fp32 gradients of all native parameters for every candidate and for the query loss L_Q (the mean over
queries of each query's response-token mean loss). Per candidate it stores:
  * S_head = <G_p^out, G_Q^out> and |G_p^out| for the output layer. The output-layer gradient is the
    readout part sum_t (dL/dlogits_t) h_t^T, captured separately from the actual parameter gradient, so
    for tied models (Llama-3.2-3B, Qwen3-4B-Base) it excludes the input-embedding use of the shared matrix;
  * the dot product with G_Q and the norm of every decoder block's gradient (attention, MLP and norms);
  * the same for the remaining global blocks (embedding / shared input-output matrix, final norm, head);
  * full_dot = <G_p, G_Q> and full_norm = |G_p| over every unique parameter (native weight tying).
Products are fp32 with fp64 reductions over 2^20-entry chunks, TF32 is off, dropout is zero, and the
actual attention masks are used; the model is in train mode only so that gradient checkpointing applies.

Paper protocol: 3,000 candidates per pair from a seeded permutation of the pool (seed 0; seed 7 for
Llama-3.2-3B), OLMo-3-7B audited in shards (1,903-2,944 candidates per task); the analysis keeps
candidates of at most 1,024 tokens for the two 7B models (see ``layer_curve.py``).

Relation to the paper's records. The records behind the paper were produced by an earlier implementation
of this audit (its sha256 is the ``runner_sha256`` field of ``data/analysis/audit/index.json``), which also
measured a finite SGD step on every candidate (loss drop / eta at eta = 2e-5 with exact parameter
restoration) and matched RDS+ embeddings. No reported number uses those two quantities, so ``compute``
below omits them; everything it stores is computed with the same arithmetic as that implementation.

Subcommands:
  compute  GPU  audit one pair (resumable JSON with the field names of the paper's records).
  compact  CPU  audit JSONs -> data/analysis/audit/<model>_<task>.npz plus index.json (provenance).
  compare  CPU  a ``compute`` output against the paper's records of the same candidates.

Port check (``data/analysis/reference/audit_port_cpu_check.json``). ``compute`` on CPU for the first three
Llama-3.2-3B GSM8K candidates gives the same candidates, token counts and encoding sha256 as the paper's GPU
records. Norms agree to 2.5e-4 relative, S_head and full_dot to 7.4e-4,
and the small per-block dot products to 8.5e-3; G_Q norms agree to 5e-5. These are the sizes of fp32 kernel
differences between CPU (torch 2.5.1) and GPU (torch 2.9.1, CUDA 12.8) backward passes.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import common
from analysis.common import MODEL_KEYS, PAIRS, POOL_SIZE, TASK_KEYS

AUDIT_DIR = common.DATA_DIR / "audit"
AUDIT_SEEDS = {"llama2": 0, "llama3": 7, "qwen": 0, "olmo3": 0}
IP_CHUNK = 1 << 20


# ---------------------------------------------------------------------------------------------- arithmetic
def inner_product(left, right) -> float:
    """fp32 products with chunked fp64 sums (the audit convention)."""
    import torch
    if left.shape != right.shape or left.device != right.device:
        raise ValueError("inner-product tensors must have identical shape and device")
    a, b = left.detach().reshape(-1), right.detach().reshape(-1)
    total = 0.0
    for s in range(0, a.numel(), IP_CHUNK):
        value = float((a[s:s + IP_CHUNK].float() * b[s:s + IP_CHUNK].float()).sum(dtype=torch.float64))
        if not math.isfinite(value):
            raise ValueError("nonfinite inner product")
        total += value
    return total


def norm(value) -> float:
    return math.sqrt(inner_product(value, value))


def encoding_hash(encoded) -> str:
    """sha256 of the encoded ids, attention mask and labels of one example."""
    digest = hashlib.sha256()
    for key in ("input_ids", "attention_mask", "labels"):
        value = encoded[key].detach().cpu().contiguous().reshape(-1)
        digest.update(f"{key}:{value.dtype}:{value.numel()}:".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def candidate_indices(seed: int, skip: int, n: int) -> list:
    """Candidates [skip, n) of the seeded permutation of the pool."""
    import torch
    if not 0 <= skip < n <= POOL_SIZE:
        raise ValueError("expected 0 <= skip < n <= pool size")
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(POOL_SIZE, generator=g)[skip:n].tolist()


# ---------------------------------------------------------------------------------------------- compute (GPU)
def parameter_layout(model):
    """(name, parameter, block) for every unique parameter; block = decoder index or a global block name."""
    import torch
    aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(name)
    inp, out = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    layout = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.dtype != torch.float32:
            raise ValueError(f"the audit needs trainable fp32 parameters: {name}")
        layers = {int(m.group(1)) for a in aliases[id(p)] if (m := re.search(r"layers\.(\d+)\.", a))}
        if len(layers) > 1:
            raise ValueError(f"parameter shared between decoder blocks: {name}")
        if p is inp and p is out:
            block = "shared_input_output"
        elif p is inp:
            block = "embedding"
        elif p is out:
            block = "output_head"
        elif layers:
            block = layers.pop()
        elif name in {"model.norm.weight", "model.norm.bias"}:
            block = "final_norm"
        else:
            block = f"other:{name}"
        layout.append((name, p, block))
    return layout


def block_gradients(layout, device):
    """Flat fp32 gradient per block (parameters concatenated in layout order); clears parameter grads."""
    import torch
    groups = {}
    for name, p, block in layout:
        if p.grad is None or p.grad.dtype != torch.float32 or p.grad.shape != p.shape:
            raise RuntimeError(f"missing or invalid gradient: {name}")
        groups.setdefault(block, []).append(p)
    out = {}
    for block, members in groups.items():
        out[block] = torch.cat([p.grad.detach().reshape(-1) for p in members]).to(device)
        for p in members:
            p.grad = None
    return out


def forward_backward(model, batch):
    """Loss backward with the output-only readout gradient captured from the lm_head call; returns it on CPU."""
    import torch
    captured = {}

    def forward_hook(_module, inputs, output):
        if len(inputs) != 1 or inputs[0].ndim != 3 or inputs[0].shape[0] != 1 or not output.requires_grad:
            raise ValueError("expected one differentiable output-layer call on one sequence")
        hidden = inputs[0].detach()

        def grad_hook(grad):
            if "head" in captured:
                raise RuntimeError("output-layer gradient captured twice")
            with torch.no_grad():
                captured["head"] = (grad.detach().reshape(-1, grad.shape[-1]).float().T
                                    @ hidden.reshape(-1, hidden.shape[-1]).to(device=grad.device, dtype=torch.float32)).cpu()
            return None

        output.register_hook(grad_hook)

    handle = model.get_output_embeddings().register_forward_hook(forward_hook)
    try:
        loss = model(**batch).loss
    finally:
        handle.remove()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite loss")
    loss.backward()
    if "head" not in captured:
        raise RuntimeError("output-layer gradient was not captured")
    return captured["head"], float(loss)


def compute(args):
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sft.nayak_data import construct_test_sample, encode_with_messages_format
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    storage = torch.device(args.storage_device or args.device)
    settings = dict(task=args.task, audit_seed=args.seed, audit_skip=args.skip, audit_n=args.n)
    results = []
    if out_path.exists():
        prior = common.load_json(out_path)
        if any(prior.get(k) != v for k, v in settings.items()):
            raise ValueError(f"{out_path} exists with different settings")
        if prior.get("complete"):
            print(f"{out_path} is complete", flush=True)
            return
        results = prior["results"]

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
                                                 device_map={"": args.device})
    for p in model.parameters():
        p.requires_grad_(True)
    tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
    layout = parameter_layout(model)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()  # dropout is zero; train mode only engages gradient checkpointing
    torch.random.manual_seed(0)
    device = next(model.parameters()).device

    # query gradient: mean over queries, accumulated in fp32 on the CPU in query order
    dev = load_dataset(common.QUERY_DATASET, args.task, split="dev")
    if not {"prompts", "labels"}.issubset(dev.column_names) or not len(dev):
        raise ValueError("query split must contain prompts/labels records")
    dev = dev.map(lambda x: construct_test_sample(sample=x, tokenizer=tok, max_length=2048))
    dev.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    nq = len(dev)
    gq_head, gq_blocks = None, {}
    t0 = time.time()
    for qi in range(nq):
        if not bool((dev[qi]["labels"][1:] != -100).any()):
            raise ValueError(f"query {qi} has no supervised tokens")
        batch = {k: dev[qi][k].unsqueeze(0).to(device) for k in ("input_ids", "attention_mask", "labels")}
        model.zero_grad(set_to_none=True)
        head, _ = forward_backward(model, batch)
        gq_head = head / nq if gq_head is None else gq_head.add_(head / nq)
        for block, g in block_gradients(layout, "cpu").items():
            gq_blocks[block] = g / nq if block not in gq_blocks else gq_blocks[block].add_(g / nq)
        print(f"[audit] query {qi + 1}/{nq}", flush=True)
    model.zero_grad(set_to_none=True)
    gq_blocks = {b: v.to(storage) for b, v in gq_blocks.items()}
    layers = [b for b in gq_blocks if isinstance(b, int)]
    globals_ = [b for b in gq_blocks if isinstance(b, str)]
    header = dict(settings, model_path=str(args.model_path), n_queries=nq, input_output_tied=tied,
                  gq_norm_head=norm(gq_head), gq_norms_per_layer={str(b): norm(gq_blocks[b]) for b in layers},
                  gq_norms_global={b: norm(gq_blocks[b]) for b in globals_},
                  protocol="all native unique parameters; native weight tying; output-only head gradient captured "
                           "separately; response-only loss; dev-mean queries; actual attention masks; fp32 products "
                           "with fp64 reductions; no SGD step or RDS+ capture (see analysis/audit.py)",
                  provenance=dict(torch=torch.__version__, transformers=transformers.__version__, device=str(device),
                                  storage=str(storage), dtype="float32", tf32=False, command=sys.argv,
                                  seconds_query_gradient=time.time() - t0))
    header["gq_norm_full"] = math.sqrt(sum(v ** 2 for v in header["gq_norms_per_layer"].values())
                                       + sum(v ** 2 for v in header["gq_norms_global"].values()))

    def save(complete):
        common.write_json(dict(header, complete=complete, n_done=len(results), results=results), out_path, indent=None)

    pool = load_dataset(common.POOL_DATASET, split="train")
    if len(pool) != POOL_SIZE or "messages" not in pool.column_names:
        raise ValueError(f"expected {POOL_SIZE} pool rows with messages")
    idxs = candidate_indices(args.seed, args.skip, args.n)
    if [r["pool_index"] for r in results] != idxs[:len(results)]:
        raise ValueError("saved rows do not follow the candidate order")
    for i in idxs[len(results):]:
        t1 = time.time()
        enc = encode_with_messages_format(pool[i], tok, max_seq_length=2048)
        batch = {"input_ids": enc["input_ids"].unsqueeze(0).to(device),
                 "attention_mask": enc["attention_mask"].unsqueeze(0).to(device),
                 "labels": enc["labels"].unsqueeze(0).to(device)}
        if not bool((batch["labels"][:, 1:] != -100).any()):
            raise ValueError(f"pool item {i} has no supervised tokens after truncation")
        model.zero_grad(set_to_none=True)
        head, _ = forward_backward(model, batch)
        blocks = block_gradients(layout, storage)
        dot_head = inner_product(head, gq_head)
        dots = {b: inner_product(blocks[b], gq_blocks[b]) for b in layers}
        norms = {b: norm(blocks[b]) for b in layers}
        gdots = {b: inner_product(blocks[b], gq_blocks[b]) for b in globals_}
        gnorms = {b: norm(blocks[b]) for b in globals_}
        del blocks
        full_dot = sum(dots.values()) + sum(gdots.values())
        full_norm = math.sqrt(sum(v ** 2 for v in norms.values()) + sum(v ** 2 for v in gnorms.values()))
        body = sum(dots.values())
        results.append({
            "pool_index": i,
            "preds": {"S_head": dot_head, "S_head_all": dot_head + body, "S_body_all": body, "S_full": full_dot},
            "dots_per_layer": {str(b): v for b, v in dots.items()},
            "norm_head": norm(head),
            "norms_per_layer": {str(b): v for b, v in norms.items()},
            "sequence_length": int(batch["input_ids"].shape[1]),
            "response_tokens": int((batch["labels"][:, 1:] != -100).sum()),
            "encoding_sha256": encoding_hash(enc),
            "full_dot": full_dot, "full_norm": full_norm, "dots_global": gdots, "norms_global": gnorms,
        })
        if len(results) % args.save_every == 0:
            save(False)
        print(f"[audit] {len(results)}/{len(idxs)} item {i} ({time.time() - t1:.1f}s) S_head={dot_head:.4f} "
              f"full_dot={full_dot:.4f}", flush=True)
    save(True)


# ---------------------------------------------------------------------------------------------- compact records
def audit_rows(pattern, model, task):
    """Audit rows of one pair in the order used by the analysis, and their source files.

    ``pattern`` names the outputs of ``compute`` for a pair, e.g. "runs/audit/{model}_{task}.json". A pattern
    with glob characters (e.g. "runs/audit/{model}_{task}_*_*.json") may match several shards of one
    candidate permutation; they are merged keeping the first row of every pool index in sorted file order.
    """
    spec = pattern.format(model=model, task=task)
    files = sorted(Path(p) for p in glob.glob(spec)) if any(c in spec for c in "*?[") else [Path(spec)]
    if not files:
        raise FileNotFoundError(f"no audit file matches {spec}")
    if len(files) > 1:
        seen, rows = set(), []
        for f in files:
            d = common.load_json(f)
            if d.get("complete") is not True:
                raise ValueError(f"{f}: shard is not complete")
            for r in d["results"]:
                if r["pool_index"] not in seen:
                    seen.add(r["pool_index"])
                    rows.append(r)
        return rows, files
    d = common.load_json(files[0])
    if d.get("complete") is not True:
        raise ValueError(f"{files[0]}: audit is not complete")
    return d["results"], files


def merged_slices(names):
    """[skip, n) ranges of the shard files a merged audit file was built from (file names end in _<skip>_<n>.json)."""
    if not names:
        return None
    out = []
    for name in names:
        m = re.search(r"_(\d+)_(\d+)\.json$", name)
        if m is None:
            raise ValueError(f"cannot read the candidate range of {name}")
        out.append([int(m.group(1)), int(m.group(2))])
    return sorted(out)


def compact_pair(pattern, model, task):
    rows, files = audit_rows(pattern, model, task)
    L = len(rows[0]["dots_per_layer"])
    gnames = list(rows[0]["dots_global"])
    for r in rows:
        if len(r["dots_per_layer"]) != L or len(r["norms_per_layer"]) != L or list(r["dots_global"]) != gnames:
            raise ValueError(f"{model} {task}: inconsistent row layout at pool index {r['pool_index']}")
    arrays = dict(
        pool_index=np.array([r["pool_index"] for r in rows], dtype=np.int64),
        sequence_length=np.array([r["sequence_length"] for r in rows], dtype=np.int64),
        response_tokens=np.array([r["response_tokens"] for r in rows], dtype=np.int64),
        s_head=np.array([r["preds"]["S_head"] for r in rows]),
        norm_head=np.array([r["norm_head"] for r in rows]),
        full_dot=np.array([r["full_dot"] for r in rows]),
        full_norm=np.array([r["full_norm"] for r in rows]),
        dots_per_layer=np.array([[r["dots_per_layer"][str(l)] for l in range(L)] for r in rows]),
        norms_per_layer=np.array([[r["norms_per_layer"][str(l)] for l in range(L)] for r in rows]),
        global_names=np.array(gnames),
        dots_global=np.array([[r["dots_global"][g] for g in gnames] for r in rows]),
        norms_global=np.array([[r["norms_global"][g] for g in gnames] for r in rows]),
    )
    first = common.load_json(files[0])
    prov = first.get("provenance", {})
    if len(files) > 1:       # shards of one permutation
        slices = sorted([common.load_json(f)["audit_skip"], common.load_json(f)["audit_n"]] for f in files)
    elif first.get("merged_from"):   # one file merged from shards
        slices = merged_slices(first["merged_from"])
    else:
        slices = [[first["audit_skip"], first["audit_n"]]]
    meta = dict(n=len(rows), n_layers=L, global_blocks=gnames, audit_seed=first["audit_seed"], candidate_slices=slices,
                n_queries=prov.get("query_count", first.get("n_queries")), L0=first.get("L0"),
                gq_norm_head=first["gq_norm_head"], gq_norm_full=first["gq_norm_full"],
                runner_sha256=prov.get("runner_sha256"), torch=prov.get("torch"), transformers=prov.get("transformers"),
                sources={Path(f).name: common.sha256_file(f) for f in files})
    return arrays, meta


def compare(computed_path, model, task, audit_dir=AUDIT_DIR) -> dict:
    """Relative differences between a ``compute`` output and the paper's records of the same candidates."""
    d = common.load_json(computed_path)
    ref = load_pair(model, task, audit_dir)
    pos = {int(i): k for k, i in enumerate(ref["pool_index"])}
    rows = [r for r in d["results"] if r["pool_index"] in pos]
    if not rows:
        raise ValueError("no candidate of the computed file is in the paper's records")
    ix = [pos[r["pool_index"]] for r in rows]
    rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - b) / np.abs(b)))
    same = lambda key, field: all(r[key] == int(ref[field][i]) for r, i in zip(rows, ix))
    return dict(
        n_compared=len(rows), pool_index_equal=True, sequence_length_equal=same("sequence_length", "sequence_length"),
        response_tokens_equal=same("response_tokens", "response_tokens"),
        max_rel_diff=dict(
            s_head=rel([r["preds"]["S_head"] for r in rows], ref["s_head"][ix]),
            norm_head=rel([r["norm_head"] for r in rows], ref["norm_head"][ix]),
            full_dot=rel([r["full_dot"] for r in rows], ref["full_dot"][ix]),
            full_norm=rel([r["full_norm"] for r in rows], ref["full_norm"][ix]),
            block_norms=rel([[r["norms_per_layer"][str(l)] for l in range(ref["norms_per_layer"].shape[1])] for r in rows],
                            ref["norms_per_layer"][ix]),
            block_dots=rel([[r["dots_per_layer"][str(l)] for l in range(ref["dots_per_layer"].shape[1])] for r in rows],
                           ref["dots_per_layer"][ix])),
        computed=dict(file=Path(computed_path).name, settings={k: d.get(k) for k in ("task", "audit_seed", "audit_skip", "audit_n")},
                      provenance={k: (d.get("provenance") or {}).get(k) for k in ("torch", "transformers", "device", "storage", "dtype")}))


def load_pair(model, task, audit_dir=AUDIT_DIR) -> dict:
    """Compact audit records of one pair as numpy arrays (see ``compact_pair``)."""
    z = np.load(common.require(Path(audit_dir) / f"{model}_{task}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compute", help="GPU: audit one model-task pair")
    p.add_argument("--model", choices=MODEL_KEYS, required=True)
    p.add_argument("--task", choices=TASK_KEYS, required=True)
    p.add_argument("--model-path", required=True, help="local snapshot directory or hub id of the base model")
    p.add_argument("--out", required=True, help="output JSON (resumed when it exists)")
    p.add_argument("--seed", type=int, help="candidate permutation seed (paper: 0, Llama-3.2-3B 7)")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--storage-device", help="where block gradients and G_Q live for the dot products (default --device)")
    p.add_argument("--save-every", type=int, default=8)
    p = sub.add_parser("compact", help="CPU: audit JSONs -> data/analysis/audit")
    p.add_argument("--pattern", required=True,
                   help="audit JSONs of a pair, e.g. 'runs/audit/{model}_{task}.json'; glob characters merge shards")
    p.add_argument("--out-dir", default=str(AUDIT_DIR))
    p = sub.add_parser("compare", help="CPU: compare a compute output with the paper's records of the same pair")
    p.add_argument("--computed", required=True)
    p.add_argument("--model", choices=MODEL_KEYS, required=True)
    p.add_argument("--task", choices=TASK_KEYS, required=True)
    p.add_argument("--audit-dir", default=str(AUDIT_DIR))
    p.add_argument("--out", help="write the comparison as JSON")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        result = compare(args.computed, args.model, args.task, args.audit_dir)
        if args.out:
            common.write_json(result, args.out)
        print(json.dumps({k: v for k, v in result.items() if k != "computed"}, indent=1))
        return
    if args.cmd == "compute":
        if args.seed is None:
            args.seed = AUDIT_SEEDS[args.model]
        compute(args)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        index = {}
        for m, t in PAIRS:
            arrays, meta = compact_pair(args.pattern, m, t)
            np.savez_compressed(out_dir / f"{m}_{t}.npz", **arrays)
            index[f"{m}_{t}"] = meta
            print(f"{m}_{t}: {meta['n']} candidates, {meta['n_layers']} blocks", flush=True)
        common.write_json(index, out_dir / "index.json")


if __name__ == "__main__":
    main()
