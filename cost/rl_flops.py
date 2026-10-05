"""RL scoring FLOPs per scored token and stored feature size per problem: LESSER against full gradients.

Paper: Appendix A.4 "RL scoring FLOPs and storage" (19,857,408 projection FLOPs per scored token,
2P = 3.09e9, 3.11e9 in total, 6P = 9.26e9, 3.0x) and Table 7 (FLOPs per scored token; stored
feature 32.8 kB against 6.2 GB); the feature sizes of Table 1(b); the 3.0x in the abstract,
Section 1, Section 4 "Cost analysis" and Figure 1.

Per scored token, for Qwen2.5-Math-1.5B-Instruct:

    LESSER, projected   2P + 2(|V|m + dn) + 2mn
                        forward pass, projection of the output residual (|V| -> m) and of the
                        hidden state (d -> n), and the m x n outer product
    LESSER, exact       2P + 2|V|d
                        forward pass and the |V| x d outer product (the exact feature; not in the
                        compiled paper)
    full gradients      6P
                        forward-backward pass

Stored feature per problem, FP32: m n values (LESSER), |V| d (exact), P (full gradient); kB, MB
and GB are decimal units. Softmax, normalization, dot products and data transfer are not counted.

P is derived from the Qwen2.5-1.5B architecture and checked against the parameter count the RL
scorer logged in data/cost/timing_rl/; |V|, d and both feature sizes are checked against the
feature construction the scorer logged; m and n come from lesser.RL_PAPER.

    python -m cost.rl_flops [--paper-dir PAPER_SOURCE]
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from cost._common import Claim, Row, add_common_args, finish, fixed, grouped, row_claims, sci
from cost.timing_rl.parse_logs import load as load_rl_timing
from lesser.projection import RL_PAPER

# Qwen2.5-Math-1.5B-Instruct, from its Hugging Face config (tied input and output embeddings,
# grouped-query attention with biases on q, k and v).
QWEN25_MATH_1_5B = dict(vocab=151_936, hidden=1_536, layers=28, intermediate=8_960, heads=12, kv_heads=2)
FP32_BYTES = 4


def qwen2_params(vocab: int, hidden: int, layers: int, intermediate: int, heads: int, kv_heads: int) -> int:
    """Qwen2 parameter count with tied embeddings: the embedding matrix once; per layer q (with
    bias), k and v (kv_heads x head_dim outputs, with bias), o (no bias), the gated MLP and two
    RMSNorm weight vectors; and the final RMSNorm."""
    kv = kv_heads * (hidden // heads)
    attention = (hidden * hidden + hidden) + 2 * (hidden * kv + kv) + hidden * hidden
    per_layer = attention + 3 * hidden * intermediate + 2 * hidden
    return vocab * hidden + layers * per_layer + hidden


def compute() -> dict:
    """FLOPs per scored token and bytes per stored feature, as exact integers."""
    P = qwen2_params(**QWEN25_MATH_1_5B)
    V, d = QWEN25_MATH_1_5B["vocab"], QWEN25_MATH_1_5B["hidden"]
    m, n = RL_PAPER.k_vocab, RL_PAPER.k_hidden
    projection = 2 * (V * m + d * n) + 2 * m * n
    flops = dict(lesser=2 * P + projection, exact=2 * P + 2 * V * d, full=6 * P)
    storage = dict(lesser=m * n * FP32_BYTES, exact=V * d * FP32_BYTES, full=P * FP32_BYTES)
    return dict(P=P, V=V, d=d, m=m, n=n, forward=2 * P, projection=projection, flops=flops, storage=storage,
                ratio=Fraction(flops["full"], flops["lesser"]), ratio_exact=Fraction(flops["full"], flops["exact"]))


def cross_check(res: dict, timing: dict) -> None:
    """Raise unless the RL scorer logs record the same model size and feature sizes."""
    logs = timing["logs"]
    if timing["params"] != res["P"]:
        raise ValueError(f"scorer logged {timing['params']:,} parameters, the architecture gives {res['P']:,}")
    expected = dict(prodsk=res["m"] * res["n"], prod=res["V"] * res["d"])
    for v, dim in expected.items():
        feat = logs[v]["feature"]
        if feat is None or (feat["vocab"], feat["hidden"], feat["dim"]) != (res["V"], res["d"], dim):
            raise ValueError(f"{v}.log: feature construction {feat} differs from |V|={res['V']}, d={res['d']}, dim={dim}")


def load(data_dir: Path, timing: dict | None = None) -> dict:
    """Compute the RL FLOP quantities and cross-check them against the RL timing logs."""
    res = compute()
    cross_check(res, timing if timing is not None else load_rl_timing(data_dir))
    return res


def summary(res: dict) -> str:
    f, s = res["flops"], res["storage"]
    return "\n".join([
        "RL scoring FLOPs and storage, Qwen2.5-Math-1.5B-Instruct (Appendix A.4, Table 7)",
        f"  P = {res['P']:,}   |V| = {res['V']:,}   d = {res['d']:,}   (m, n) = ({res['m']}, {res['n']})"
        "   (P, |V|, d and feature sizes match the scorer logs)",
        f"  projection per scored token 2(|V|m+dn)+2mn = {res['projection']:,};  forward 2P = {sci(res['forward'], 3)}",
        f"  per scored token: LESSER projected {sci(f['lesser'], 3)} | exact {sci(f['exact'], 3)} | full gradients {sci(f['full'], 3)}",
        f"  full / LESSER projected = {fixed(res['ratio'], 2)};  full / exact = {fixed(res['ratio_exact'], 2)}"
        "   (exact: not in the compiled paper)",
        f"  stored feature per problem (FP32): LESSER {s['lesser']:,} B = {fixed(Fraction(s['lesser'], 10**3), 1)} kB | exact"
        f" {s['exact']:,} B = {fixed(Fraction(s['exact'], 10**6), 0)} MB | full {s['full']:,} B"
        f" = {fixed(Fraction(s['full'], 10**9), 1)} GB",
    ])


def checks(res: dict) -> tuple[list[Claim], list[Row]]:
    f, s = res["flops"], res["storage"]
    lesser_kb, full_gb = fixed(Fraction(s["lesser"], 10**3), 1), fixed(Fraction(s["full"], 10**9), 1)
    rows = [
        Row("Table 7", r"\LESSER{} & $<0>$ & $<1>$~kB \\", (
            Claim("Table 7: LESSER FLOPs per scored token", sci(f["lesser"], 2), "3.11e9", "Table 7"),
            Claim("Table 7: LESSER stored feature (kB)", lesser_kb, "32.8", "Table 7"))),
        Row("Table 7", r"Full gradients & $<0>$ & $<1>$~GB \\", (
            Claim("Table 7: full-gradient FLOPs per scored token", sci(f["full"], 2), "9.26e9", "Table 7"),
            Claim("Table 7: full-gradient stored feature (GB)", full_gb, "6.2", "Table 7"))),
    ]
    inline = [
        Claim("vocabulary size |V|", grouped(res["V"]), "151,936", "App. A.4", (r"For vocabulary size $|V|=<0>$",)),
        Claim("hidden dimension d", grouped(res["d"]), "1,536", "App. A.4", (r"hidden dimension $d=<0>$",)),
        Claim("projection dimension m", str(res["m"]), "64", "App. A.4", (r"projection dimensions $m=<0>$",)),
        Claim("projection dimension n", str(res["n"]), "128", "App. A.4", (r"and $n=<0>$, the projection costs",)),
        Claim("projection FLOPs per scored token", grouped(res["projection"]), "19,857,408", "App. A.4",
              (r"2(|V|m+dn)+2mn = <0>",)),
        Claim("forward pass 2P per scored token", sci(res["forward"], 2), "3.09e9", "App. A.4",
              (r"the forward pass $2P=<0>$",)),
        Claim("model parameters P", grouped(res["P"]), "1,543,714,304", "App. A.4", (r"for $P=<0>$ parameters",)),
        Claim("LESSER FLOPs per scored token", sci(f["lesser"], 2), "3.11e9", "App. A.4", (r"parameters, $<0>$ in total",)),
        Claim("full-gradient FLOPs per scored token 6P", sci(f["full"], 2), "9.26e9", "App. A.4", (r"6P = <0>",)),
        Claim("full / LESSER scoring FLOPs (x)", fixed(res["ratio"], 1), "3.0",
              "Abstract; Sec. 1; Sec. 4 Cost analysis; App. A.4; Fig. 1 is checked by cost.table1",
              (r"FLOPs per scored token, $<0>\times$ the cost of \LESSER{}", r"and $<0>\times$ for RL benchmarks",
               r"and $<0>\times$ fewer FLOPs on RL", r"This reduces scoring FLOPs by $<0>\times$.")),
    ]
    return inline + row_claims(rows), rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="RL scoring FLOPs and storage (paper Appendix A.4, Table 7).")
    add_common_args(ap)
    args = ap.parse_args(argv)
    res = load(args.data_dir)
    print(summary(res))
    claims, rows = checks(res)
    finish(claims, rows, "RL scoring FLOPs and storage", args.paper_dir)


if __name__ == "__main__":
    main()
