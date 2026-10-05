"""LESSER pool and query features for the SFT experiments (one forward pass per example, no backpropagation).

Paper results: the features behind every LESSER SFT selection (Tables 9--11, Figures 3, 4, 8--10, Appendix E).
For an example with supervised response tokens t, the feature is

    phi = normalize( vec( P_h^T (sum_t r_t h_t^T)^T P_v ) ),   r_t = softmax(W h_t) - e_{y_t},

where h_t is the post-final-norm hidden state that predicts token y_t, W the base model's readout matrix, and
P_h (d x 128), P_v (|V| x 64) the Rademacher maps of ``lesser.SFT_PAPER`` (8,192 entries). The model runs in bf16 and
all feature arithmetic in fp32 (``lesser.token_factors``, ``lesser.TwoSidedProjection``). Tokenization and loss masking
are Nayak et al.'s (``sft/nayak_data.py``, maximum length 2,048): pool examples supervise assistant tokens, query
examples the label. This reproduces the extractor and per-model base-model loaders behind the paper's
features; ``python -m sft.extract self-test`` checks the path on a tiny random model.

Model notes:
  * Llama-3.2-3B: as in the paper, the tokenizer is the base tokenizer plus a ``[PAD]`` token (128,257 entries; this
    equals the tokenizer saved by LESS warmup training that the paper loaded, with identical ids and masks on the whole
    pool), and the embeddings and tied readout are resized to match. Resizing appends one mean-initialized row whose
    logit enters the softmax; the paper's run drew it without a fixed seed, so we seed it (``--resize-seed``).
    Re-extracted Llama-3.2-3B features are therefore not bit-identical to the stored ones; the released selections
    were verified from the stored features.
  * OLMo3-7B needs transformers >= 4.57.
  * Features are fp32; examples whose response is truncated away have zero features (reported in the metadata).

Outputs in ``<out-dir>/<model>/``: ``pool_prod_<start>_<end>.pt`` (+ ``pool_meta_<start>_<end>.json``) and
``val_<task>_prod.pt`` (+ ``val_meta.json``). Pool extraction is sharded by index range and resumable: finished
chunks are kept in ``pool_prod_<start>_<end>.parts/`` and skipped on restart; the shard file is written only after
quality checks pass (finite values, nonzero column spread, >50% unique rows, unit norm or exactly zero per row).

Run (GPU):
  python -m sft.extract pool    --model llama-2-7b --start 0 --end 197196 --out-dir OUT [--device cuda:0]
  python -m sft.extract queries --model llama-2-7b --out-dir OUT
  python -m sft.extract compare --model llama-2-7b --features-dir DIR --indices 0 1 2 --tasks tydiqa   (vs stored;
                                add --device cpu --compute-dtype float32 to check on a CPU)
  python -m sft.extract self-test                                                                     (CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

from lesser import SFT_PAPER, TwoSidedProjection, l2_normalize, output_layer_gradient, token_factors
from lesser.checks import autograd_readout_gradient, check_exactness

from .common import (MAX_SEQ_LENGTH, N_POOL, N_QUERIES, POOL_DATASET, QUERY_DATASET, TASKS, features_dir, model_spec,
                     write_json)
from .nayak_data import construct_test_sample, encode_with_messages_format


# ----------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------
def load_model(key: str, device: str, dtype=torch.bfloat16, resize_seed: int = 0):
    """Base model and tokenizer exactly as used for the paper's features (see the module docstring)."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = model_spec(key)
    tok = AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"   # renamed in transformers 4.56
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, device_map={"": device}, **{dtype_kw: dtype})
    if key == "llama-3.2-3b":
        if tok.pad_token is not None:
            raise RuntimeError("expected the Llama-3.2-3B base tokenizer to have no pad token")
        tok.add_special_tokens({"pad_token": "[PAD]"})
        torch.manual_seed(resize_seed)   # the new row is a random draw around the mean embedding
        model.resize_token_embeddings(len(tok), mean_resizing=True)
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token    # never used: examples are processed one at a time, without padding
    if tok.eos_token is None:
        raise RuntimeError(f"{spec.hf_id}: tokenizer has no end-of-sequence token")
    W = model.get_output_embeddings().weight
    if tuple(W.shape) != (spec.vocab, spec.hidden):
        raise RuntimeError(f"{spec.hf_id}: readout {tuple(W.shape)}, expected {(spec.vocab, spec.hidden)}")
    if len(tok) > W.shape[0]:
        raise RuntimeError(f"{spec.hf_id}: tokenizer ({len(tok)}) larger than the readout ({W.shape[0]})")
    model.eval()
    return model, tok


class FeatureExtractor:
    """Per-example LESSER features for one model; projection maps are shared by pool and queries."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.W = model.get_output_embeddings().weight.detach().float()   # (V, d), fp32 once
        self.proj = TwoSidedProjection(self.W.shape[0], self.W.shape[1], SFT_PAPER, device=device)
        self.readout_checked = False

    @torch.no_grad()
    def __call__(self, item: dict) -> tuple[torch.Tensor, int]:
        ids = item["input_ids"].to(self.device)[None]
        mask = item["attention_mask"].to(self.device)[None]
        labels = item["labels"].to(self.device)
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        h_all = out.hidden_states[-1][0].float()          # post-final-norm states, the readout input
        if labels.shape[0] != h_all.shape[0]:
            raise RuntimeError(f"label length {labels.shape[0]} != sequence length {h_all.shape[0]}")
        if not self.readout_checked:                       # once per run: hidden_states[-1] feeds the readout
            ref = out.logits[0].float()
            err = float((h_all @ self.W.T - ref).norm() / ref.norm())
            if err > 2e-2:
                raise RuntimeError(f"hidden_states[-1] is not the readout input (relative logit error {err:.3g})")
            self.readout_checked = True
        h, r = token_factors(h_all, self.W, labels)         # next-token shift; supervised positions only
        return l2_normalize(self.proj.project(h, r)).cpu(), int(h.shape[0])


# ----------------------------------------------------------------------------------------------------------------
# Quality checks
# ----------------------------------------------------------------------------------------------------------------
def quality_report(X: torch.Tensor, n_resp: list[int], name: str) -> dict:
    """Raise unless features are finite, spread, mostly unique, and unit norm (zero for examples without response)."""
    if X.shape[0] != len(n_resp):
        raise ValueError(f"{name}: {X.shape[0]} rows but {len(n_resp)} response counts")
    if not torch.isfinite(X).all():
        raise ValueError(f"{name}: non-finite features")
    norms = X.norm(dim=1)
    empty = torch.tensor([n == 0 for n in n_resp])
    if empty.any() and float(norms[empty].abs().max()) != 0.0:
        raise ValueError(f"{name}: an example without supervised tokens has a nonzero feature")
    if (~empty).any() and float((norms[~empty] - 1).abs().max()) > 1e-3:
        raise ValueError(f"{name}: features are not unit norm")
    report = {"rows": int(X.shape[0]), "zero_rows": torch.nonzero(empty).flatten().tolist()}
    if X.shape[0] > 1:
        report["min_column_std"] = float(X.std(dim=0).min())
        rows = X.contiguous().numpy()
        report["unique_row_fraction"] = len({hashlib.sha1(row.tobytes()).digest() for row in rows}) / X.shape[0]
        if report["min_column_std"] <= 0 or report["unique_row_fraction"] <= 0.5:
            raise ValueError(f"{name}: degenerate features {report}")
    return report


# ----------------------------------------------------------------------------------------------------------------
# Pool (sharded, resumable) and queries
# ----------------------------------------------------------------------------------------------------------------
def extract_pool(extract: FeatureExtractor, tokenizer, rows, start: int, end: int, out_dir: Path,
                 chunk: int = 2000, max_seq_length: int = MAX_SEQ_LENGTH, provenance: dict | None = None) -> Path:
    """Features of pool rows [start, end) -> ``pool_prod_<start>_<end>.pt``; completed chunks survive restarts."""
    if not 0 <= start < end <= len(rows):
        raise ValueError(f"bad range [{start}, {end}) for a pool of {len(rows)}")
    out_dir = Path(out_dir)
    final = out_dir / f"pool_prod_{start}_{end}.pt"
    if final.exists():
        raise FileExistsError(f"{final} exists")
    parts = out_dir / f"pool_prod_{start}_{end}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for c0 in range(start, end, chunk):
        c1 = min(c0 + chunk, end)
        feat_path, meta_path = parts / f"part_{c0}_{c1}.pt", parts / f"part_{c0}_{c1}.json"
        if feat_path.exists() and meta_path.exists():   # finished before a restart
            continue
        feats, n_resp = [], []
        for j in range(c0, c1):
            f, n = extract(encode_with_messages_format({"messages": rows[j]["messages"]}, tokenizer, max_seq_length))
            feats.append(f)
            n_resp.append(n)
        tmp = feat_path.with_name(feat_path.name + ".tmp")
        torch.save(torch.stack(feats), tmp)
        tmp.replace(feat_path)
        write_json({"start": c0, "end": c1, "n_resp": n_resp}, meta_path)
        done += c1 - c0
        print(f"[pool] rows {c0}-{c1} done ({done / max(time.time() - t0, 1e-9):.1f} examples/s)", flush=True)
    feats, n_resp = [], []
    for c0 in range(start, end, chunk):
        c1 = min(c0 + chunk, end)
        x = torch.load(parts / f"part_{c0}_{c1}.pt", map_location="cpu", weights_only=True)
        meta = json.loads((parts / f"part_{c0}_{c1}.json").read_text())
        if x.shape[0] != c1 - c0 or meta["start"] != c0 or len(meta["n_resp"]) != c1 - c0:
            raise RuntimeError(f"corrupt chunk {c0}-{c1} in {parts}")
        feats.append(x)
        n_resp += meta["n_resp"]
    X = torch.cat(feats)
    report = quality_report(X, n_resp, final.name)
    tmp = final.with_name(final.name + ".tmp")
    torch.save(X, tmp)
    tmp.replace(final)
    write_json({"start": start, "end": end, "n_resp": n_resp, "quality": report, **(provenance or {})},
               out_dir / f"pool_meta_{start}_{end}.json")
    for p in sorted(parts.iterdir()):
        p.unlink()
    parts.rmdir()
    print(f"[pool] wrote {final} {tuple(X.shape)}; zero rows (no response tokens): {report['zero_rows']}", flush=True)
    return final


def extract_queries(extract: FeatureExtractor, tokenizer, queries: dict, out_dir: Path,
                    max_seq_length: int = MAX_SEQ_LENGTH, provenance: dict | None = None) -> None:
    """``queries``: task -> list of {prompts, labels}. Writes ``val_<task>_prod.pt`` and ``val_meta.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for task, rows in queries.items():
        feats, n_resp = zip(*(extract(construct_test_sample(tokenizer, r, max_seq_length)) for r in rows))
        X = torch.stack(feats)
        meta[task] = {"n_resp": list(n_resp), "quality": quality_report(X, list(n_resp), task)}
        torch.save(X, out_dir / f"val_{task}_prod.pt")
        print(f"[queries] {task}: {tuple(X.shape)}", flush=True)
    write_json({"tasks": meta, **(provenance or {})}, out_dir / "val_meta.json", indent=1)


def load_pool_rows():
    from datasets import load_dataset
    ds = load_dataset(POOL_DATASET, split="train")
    if len(ds) != N_POOL:
        raise RuntimeError(f"{POOL_DATASET}: {len(ds)} rows, expected {N_POOL}")
    return ds


def load_query_rows(task: str):
    from datasets import load_dataset
    ds = load_dataset(QUERY_DATASET, task, split="dev")
    if len(ds) != N_QUERIES[task]:
        raise RuntimeError(f"{QUERY_DATASET}/{task}: {len(ds)} queries, expected {N_QUERIES[task]}")
    return [ds[i] for i in range(len(ds))]


def provenance(args, model) -> dict:
    import transformers
    return {"model": args.model, "hf_id": model_spec(args.model).hf_id, "dtype": "bfloat16", "device": args.device,
            "projection": "lesser.SFT_PAPER", "max_seq_length": MAX_SEQ_LENGTH, "torch": torch.__version__,
            "transformers": transformers.__version__,
            **({"resize_seed": args.resize_seed} if args.model == "llama-3.2-3b" else {})}


# ----------------------------------------------------------------------------------------------------------------
# Self-test on a tiny random Llama (CPU, no downloads)
# ----------------------------------------------------------------------------------------------------------------
def _toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>"]
    words = ("you are helpful what is two plus three five the answer name a color blue red green translate hello "
             "bonjour count to four one").split()
    vocab = {w: i for i, w in enumerate(specials + words)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    core.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", vocab["<s>"])])
    return PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   additional_special_tokens=specials[3:])


_TOY_POOL = [
    {"messages": [{"role": "user", "content": "what is two plus three"}, {"role": "assistant", "content": "five"}]},
    {"messages": [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "name a color"},
                  {"role": "assistant", "content": "blue"}, {"role": "user", "content": "name a color"},
                  {"role": "assistant", "content": "red green"}]},
    {"messages": [{"role": "user", "content": "translate hello"}, {"role": "assistant", "content": "bonjour"}]},
    {"messages": [{"role": "user", "content": "count to four " * 14}, {"role": "assistant", "content": "one two"}]},
    {"messages": [{"role": "user", "content": "count to four"},
                  {"role": "assistant", "content": "one two three four"}]},
]
_TOY_QUERIES = {"toy": [
    {"prompts": "<|user|>\nwhat is two plus three\n<|assistant|>\n", "labels": "the answer is five"},
    {"prompts": "<|user|>\nname a color\n<|assistant|>\n", "labels": "green"}]}


def self_test(verbose: bool = True) -> None:
    """Run the extraction path on a tiny random Llama and check it against independent computations."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tok = _toy_tokenizer()
    max_len = 40   # the fourth pool example exceeds it, so its response is truncated away (a zero feature)
    # 1. masking: supervised tokens are exactly the assistant contents and their end-of-sequence tokens
    enc = encode_with_messages_format(_TOY_POOL[1], tok, max_len)
    supervised = tok.convert_ids_to_tokens(enc["input_ids"][enc["labels"] != -100].tolist())
    assert supervised == ["blue", "</s>", "red", "green", "</s>"], supervised
    q = construct_test_sample(tok, _TOY_QUERIES["toy"][0], max_len)
    assert tok.convert_ids_to_tokens(q["input_ids"][q["labels"] != -100].tolist()) == "the answer is five </s>".split()

    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    extract = FeatureExtractor(model, "cpu")
    # 2. per-example checks against autograd and an explicit projection of the exact gradient
    for ex in _TOY_POOL[:3] + [_TOY_POOL[4]]:
        item = encode_with_messages_format(ex, tok, max_len)
        feat, n = extract(item)
        with torch.no_grad():
            out = model(input_ids=item["input_ids"][None], output_hidden_states=True)
        h_all = out.hidden_states[-1][0]
        assert torch.allclose(h_all @ model.lm_head.weight.T, out.logits[0], atol=1e-5), "readout input"
        check_exactness(h_all, model.lm_head.weight, item["labels"], rtol=1e-5)
        G = autograd_readout_gradient(h_all, model.lm_head.weight, item["labels"])        # (V, d), independent
        explicit = (extract.proj.P_h.T @ G.T @ extract.proj.P_v).reshape(-1)             # (128, 64) layout
        assert torch.allclose(feat, l2_normalize(explicit), atol=1e-5), "projection of the exact gradient"
        h, r = token_factors(h_all, model.lm_head.weight, item["labels"])
        assert torch.allclose(output_layer_gradient(h, r), G, atol=1e-5)
        assert n == int((item["labels"][1:] != -100).sum())
    # 3. sharded + interrupted + resumed pool extraction equals one pass; zero feature for the truncated example
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        one = extract_pool(extract, tok, _TOY_POOL, 0, 5, tmp / "one", chunk=5, max_seq_length=max_len)
        # simulate an interruption: finish only the first chunk of a 2-chunk run, then restart
        run = tmp / "two"
        first = FeatureExtractor(model, "cpu")
        (run / "pool_prod_0_5.parts").mkdir(parents=True)
        feats = [first(encode_with_messages_format(_TOY_POOL[j], tok, max_len)) for j in range(2)]
        torch.save(torch.stack([f for f, _ in feats]), run / "pool_prod_0_5.parts" / "part_0_2.pt")
        write_json({"start": 0, "end": 2, "n_resp": [n for _, n in feats]},
                   run / "pool_prod_0_5.parts" / "part_0_2.json")
        resumed = extract_pool(extract, tok, _TOY_POOL, 0, 5, run, chunk=2, max_seq_length=max_len)
        a, b = (torch.load(p, weights_only=True) for p in (one, resumed))
        assert torch.equal(a, b), "resumed extraction differs"
        meta = json.loads((run / "pool_meta_0_5.json").read_text())
        assert meta["n_resp"][3] == 0 and meta["quality"]["zero_rows"] == [3] and float(a[3].abs().max()) == 0.0
        extract_queries(extract, tok, _TOY_QUERIES, tmp / "q", max_seq_length=max_len)
        dim = SFT_PAPER.k_hidden * SFT_PAPER.k_vocab
        assert torch.load(tmp / "q" / "val_toy_prod.pt", weights_only=True).shape == (2, dim)
    # 4. the bf16 path runs and stays close to fp32
    f32, _ = extract(encode_with_messages_format(_TOY_POOL[0], tok, max_len))
    f16, _ = FeatureExtractor(model.to(torch.bfloat16), "cpu")(encode_with_messages_format(_TOY_POOL[0], tok, max_len))
    cos = float(f32 @ f16)
    assert cos > 0.99, cos
    if verbose:
        print(f"[self-test] passed: masking, readout input, autograd exactness, projection, sharding/resume, "
              f"zero-response rows, queries; bf16 vs fp32 cosine {cos:.5f}")


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "queries", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--resize-seed", type=int, default=0, help="Llama-3.2-3B only: seed of the added [PAD] row")
    sub.choices["pool"].add_argument("--start", type=int, default=0)
    sub.choices["pool"].add_argument("--end", type=int, default=N_POOL)
    sub.choices["pool"].add_argument("--chunk", type=int, default=2000, help="examples per resumable chunk")
    for name in ("pool", "queries"):
        sub.choices[name].add_argument("--out-dir", type=Path, required=True, help="features go to <out-dir>/<model>/")
    sub.choices["queries"].add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    c = sub.choices["compare"]
    c.add_argument("--features-dir", type=Path, help="stored features; default $LESSER_ARTIFACTS/sft/features")
    c.add_argument("--indices", type=int, nargs="*", default=[], help="pool indices to re-extract")
    c.add_argument("--tasks", nargs="*", default=[], choices=TASKS, help="query sets to re-extract")
    c.add_argument("--compute-dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="float32 casts the bf16-loaded model up for CPU checks (CPU bf16 matmuls are very slow)")
    sub.add_parser("self-test")
    args = ap.parse_args(argv)

    if args.cmd == "self-test":
        self_test()
        return 0
    model, tok = load_model(args.model, args.device, resize_seed=args.resize_seed)
    if getattr(args, "compute_dtype", "bfloat16") == "float32":
        model = model.float()   # same bf16-rounded weights, fp32 arithmetic
    extract = FeatureExtractor(model, args.device)
    if args.cmd == "pool":
        extract_pool(extract, tok, load_pool_rows(), args.start, args.end, args.out_dir / args.model, args.chunk,
                     provenance=provenance(args, model))
    elif args.cmd == "queries":
        extract_queries(extract, tok, {t: load_query_rows(t) for t in args.tasks}, args.out_dir / args.model,
                        provenance=provenance(args, model))
    else:   # compare re-extracted features with stored ones (cosine per example)
        from .select import load_pool, pool_shards
        stored_dir = features_dir(args.features_dir) / args.model
        report = {}

        def record(key, feature, n, stored):
            stored = stored.float()
            if n == 0:   # no supervised tokens: the feature must be zero, as stored
                report[key] = ("zero" if float(stored.norm()) == 0 and float(feature.norm()) == 0 else "MISMATCH", n)
            else:
                report[key] = (float(feature @ stored), n)

        if args.indices:
            rows, pool = load_pool_rows(), load_pool(pool_shards(stored_dir))
            for j in args.indices:
                record(f"pool[{j}]", *extract(encode_with_messages_format({"messages": rows[j]["messages"]}, tok)),
                       pool[j])
        for task in args.tasks:
            stored = torch.load(stored_dir / f"val_{task}_prod.pt", map_location="cpu", weights_only=True)
            for i, r in enumerate(load_query_rows(task)):
                record(f"{task}[{i}]", *extract(construct_test_sample(tok, r)), stored[i])
        for key, (cos, n) in report.items():
            shown = cos if isinstance(cos, str) else f"{cos:.6f}"
            print(f"[compare] {key:16s} response tokens {n:5d}  cosine to stored {shown}")
        cosines = [c for c, _ in report.values() if not isinstance(c, str)]
        if any(c == "MISMATCH" for c, _ in report.values()):
            raise SystemExit("[compare] an example without response tokens has a nonzero feature")
        print(f"[compare] minimum cosine {min(cosines):.6f} over {len(cosines)} examples with response tokens"
              f" ({len(report) - len(cosines)} without)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
