"""SFT feature-extraction FLOPs: LESSER against released four-checkpoint LESS on Llama-2-7B.

Paper: Appendix A.4, "SFT extraction FLOPs" (Table 5, the 9.7x ratio and the approximately 12x
estimate under the 6PT convention) and "SFT extraction FLOPs relative to downstream training"
(Table 6, "at least 97.0%", 7.7--9.5x). The 9.7x also appears in the abstract, Section 1,
Section 4 "Cost analysis" and Figure 1; Section 4 also states "97% to 99.7%".

Counting, as in Appendix A.4:

    F_LESSER = 2PT + 2T_r(|V|m + dn + mn)
        one forward pass over the pool plus the two-sided projection of the output-layer gradient
    F_LESS   = C(4PT + 6 P_LoRA T + 2 D P_LoRA N)
        per checkpoint: forward 2PT, backward of the activation gradients through the frozen model
        2PT, LoRA weight-gradient terms 6 P_LoRA T, and the CUDA projection of each example's LoRA
        gradient to D dimensions, 2 D P_LoRA per example
    24PT
        the common convention (2PT per forward pass, 6PT per forward-backward pass) for four
        checkpoints, which gives the approximately 12x estimate

Extraction plus fine-tuning for one target and one training run (Table 6) is F + 6 P T_train with
T_train = epochs * k * T / N: two epochs, and selected examples as long as the pool mean. The LESSER
side uses F_LESSER, as in the computation behind the paper; the paper text writes it as 2PT_pool,
and the projection term (0.01%) changes no printed digit.

P and P_LoRA are derived from the Llama-2-7B architecture and LESS's LoRA configuration; |V|, d and
the budgets k come from sft.common, the projection sizes m and n from lesser.SFT_PAPER, and N, T and
T_r from data/cost/pool_token_count.json (cost/pool_token_count.py regenerates it). The script also reproduces
data/cost/sft_flops.json, the computation the paper numbers were taken from (integers
exactly, ratios to a relative 1e-12), and fails if it does not.

    python -m cost.sft_flops [--paper-dir PAPER_SOURCE]
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from cost._common import (POOL_EXAMPLES, Claim, Row, add_common_args, finish, fixed, grouped, require,
                          row_claims, sci, tex_number)
from lesser.projection import SFT_PAPER
from sft.common import BUDGETS, MODELS

# Llama-2-7B: vocabulary and hidden size from the SFT model registry; layers and MLP width from the Hugging Face
# config of meta-llama/Llama-2-7b-hf (untied embedding and output matrices).
_LLAMA2 = MODELS["llama-2-7b"]
LLAMA2_7B = dict(vocab=_LLAMA2.vocab, hidden=_LLAMA2.hidden, layers=32, intermediate=11_008)
# LESS as released (dcml-lab/targeted-instruction-selection, commit 8ff397a): LoRA rank 128 on the q, k, v
# and o projections of every layer; Adam-preconditioned LoRA gradients projected to 8,192 dimensions at each
# of four warmup checkpoints.
LORA_RANK, LORA_MODULES = 128, 4
LESS_PROJ_DIM, LESS_CHECKPOINTS = 8_192, 4
TRAIN_EPOCHS = 2
TOKENIZER = _LLAMA2.hf_id
REFERENCE = "sft_flops.json"


def llama_params(vocab: int, hidden: int, layers: int, intermediate: int) -> int:
    """Llama-2 parameter count: untied embedding and output matrices; per layer the q, k, v and o
    projections (hidden x hidden each), the gated MLP (3 x hidden x intermediate) and two RMSNorm
    weight vectors; and the final RMSNorm."""
    per_layer = 4 * hidden * hidden + 3 * hidden * intermediate + 2 * hidden
    return 2 * vocab * hidden + layers * per_layer + hidden


def lora_params(rank: int, hidden: int, layers: int, modules: int) -> int:
    """LoRA A (rank x hidden) and B (hidden x rank) on ``modules`` square projections per layer."""
    return modules * layers * rank * 2 * hidden


def load_counts(data_dir: Path) -> dict:
    """Pool token counts under the Llama-2 tokenizer from pool_token_count.json."""
    counts = json.loads(require(Path(data_dir) / "pool_token_count.json").read_text())
    if TOKENIZER not in counts:
        raise KeyError(f"pool_token_count.json has no entry for {TOKENIZER}")
    c = counts[TOKENIZER]
    if c["examples"] != POOL_EXAMPLES:
        raise ValueError(f"pool_token_count.json: {c['examples']} examples, expected {POOL_EXAMPLES}")
    return c


def compute(counts: dict) -> dict:
    """All SFT extraction FLOP quantities, as exact integers and fractions."""
    P = llama_params(**LLAMA2_7B)
    P_lora = lora_params(LORA_RANK, LLAMA2_7B["hidden"], LLAMA2_7B["layers"], LORA_MODULES)
    N, T, T_r = counts["examples"], counts["tokens"], counts["response_tokens"]
    V, d = LLAMA2_7B["vocab"], LLAMA2_7B["hidden"]
    m, n = SFT_PAPER.k_vocab, SFT_PAPER.k_hidden
    D, C = LESS_PROJ_DIM, LESS_CHECKPOINTS

    lesser = 2 * P * T + 2 * T_r * (V * m + d * n + m * n)
    per_ckpt = dict(forward=2 * P * T, activation_backward=2 * P * T, lora_terms=6 * P_lora * T,
                    projection=2 * D * P_lora * N)
    less = C * sum(per_ckpt.values())
    six_pt = C * 6 * P * T
    shares = {}
    for k in BUDGETS:
        train = Fraction(6 * P * TRAIN_EPOCHS * k * T, N)
        shares[k] = dict(lesser=lesser / (lesser + train), less=less / (less + train),
                         ratio=(less + train) / (lesser + train))
    return dict(P=P, P_lora=P_lora, N=N, T=T, T_r=T_r, V=V, d=d, m=m, n=n, D=D, C=C,
                lesser=lesser, less=less, less_per_ckpt=per_ckpt, six_pt=six_pt,
                ratio=Fraction(less, lesser),
                ratio_without_projection=Fraction(less - C * per_ckpt["projection"], lesser),
                ratio_six_pt=Fraction(six_pt, lesser),
                projection_share=Fraction(C * per_ckpt["projection"], less), shares=shares)


def as_reference(res: dict) -> dict:
    """The quantities in the schema of data/cost/sft_flops.json."""
    out = dict(P=res["P"], T=res["T"], T_r=res["T_r"], N=res["N"], lora_params=res["P_lora"],
               lesser_pool_flops=res["lesser"], less_released_4ckpt_flops=res["less"],
               less_6PT_convention_flops=res["six_pt"],
               ratio_released_with_projection=float(res["ratio"]),
               ratio_released_without_projection=float(res["ratio_without_projection"]),
               ratio_6PT_convention=float(res["ratio_six_pt"]),
               projection_share_of_less=float(res["projection_share"]))
    for k, s in res["shares"].items():
        out[f"share_k{k}"] = dict(lesser=float(s["lesser"]), less_released=float(s["less"]),
                                  subtotal_ratio=float(s["ratio"]))
    return out


def compare_reference(res: dict, path: Path) -> None:
    """Raise unless ``res`` reproduces the recorded computation behind the paper."""
    ref, new = json.loads(require(path).read_text()), as_reference(res)

    def walk(a, b, key):
        if isinstance(b, dict):
            if set(a) != set(b):
                raise ValueError(f"{path.name}:{key}: keys {sorted(a)} differ from {sorted(b)}")
            for k in b:
                walk(a[k], b[k], f"{key}.{k}")
        elif isinstance(b, int):
            if a != b:
                raise ValueError(f"{path.name}:{key}: recomputed {a}, recorded {b}")
        elif abs(a - b) > 1e-12 * abs(b):
            raise ValueError(f"{path.name}:{key}: recomputed {a!r}, recorded {b!r}")

    walk(new, ref, "")


def summary(res: dict) -> str:
    r = res
    per = r["less_per_ckpt"]
    lines = [
        "SFT extraction FLOPs, Llama-2-7B (Appendix A.4, Tables 5 and 6)",
        f"  P = {r['P']:,}   P_LoRA = {r['P_lora']:,}   N = {r['N']:,}   T = {r['T']:,}   T_r = {r['T_r']:,}",
        f"  |V| = {r['V']:,}   d = {r['d']:,}   D = {r['D']:,}   (m, n) = ({r['m']}, {r['n']})   C = {r['C']}",
        f"  F_LESSER = 2PT + 2T_r(|V|m+dn+mn)          = {sci(r['lesser'], 3)}",
        f"  F_LESS   = C(4PT + 6 P_LoRA T + 2 D P_LoRA N) = {sci(r['less'], 3)}"
        f"   (per checkpoint: forward {sci(per['forward'], 3)}, activation backward"
        f" {sci(per['activation_backward'], 3)}, LoRA terms {sci(per['lora_terms'], 3)},"
        f" projection {sci(per['projection'], 3)})",
        f"  F_LESS / F_LESSER = {fixed(r['ratio'], 2)}   (without LESS's projection {fixed(r['ratio_without_projection'], 2)};"
        f" projection {fixed(100 * r['projection_share'], 1)}% of F_LESS)",
        f"  24PT / F_LESSER   = {fixed(r['ratio_six_pt'], 2)}   (6PT forward-backward convention, four checkpoints)",
        f"  extraction + fine-tuning, {TRAIN_EPOCHS} epochs (Table 6):",
        "         k   LESSER share   LESS share   LESS / LESSER",
    ]
    for k, s in r["shares"].items():
        lines.append(f"    {k:>6,}   {fixed(100 * s['lesser'], 1):>11}%   {fixed(100 * s['less'], 1):>9}%"
                     f"   {fixed(s['ratio'], 2):>12}x")
    return "\n".join(lines)


def checks(res: dict) -> tuple[list[Claim], list[Row]]:
    """Paper claims regenerated from ``res``: inline numbers plus the rows of Tables 5 and 6."""
    r = res

    def t5(name, got, paper):
        return Claim(f"Table 5: {name}", got, paper, "Table 5")

    def t6(k, name, got, paper):
        return Claim(f"Table 6, k={k:,}: {name}", got, paper, "Table 6")

    rows = [
        Row("Table 5", r"Model parameters & $P$ & $<0>$ \\", (t5("model parameters P", grouped(r["P"]), "6,738,415,616"),)),
        Row("Table 5", r"LoRA parameters & $P_{\rm LoRA}$ & $<0>$ \\",
            (t5("LoRA parameters P_LoRA", grouped(r["P_lora"]), "134,217,728"),)),
        Row("Table 5", r"Pool examples & $N$ & $<0>$ \\", (t5("pool examples N", grouped(r["N"]), "197,196"),)),
        Row("Table 5", r"Processed tokens & $T$ & $<0>$ \\", (t5("processed tokens T", grouped(r["T"]), "86,343,509"),)),
        Row("Table 5", r"Response tokens & $T_r$ & $<0>$ \\", (t5("response tokens T_r", grouped(r["T_r"]), "25,430,302"),)),
        Row("Table 5", r"Vocabulary size & $|V|$ & $<0>$ \\", (t5("vocabulary size |V|", grouped(r["V"]), "32,000"),)),
        Row("Table 5", r"Hidden dimension & $d$ & $<0>$ \\", (t5("hidden dimension d", grouped(r["d"]), "4,096"),)),
        Row("Table 5", r"\LESS{} projection dimension & $D$ & $<0>$ \\",
            (t5("LESS projection dimension D", grouped(r["D"]), "8,192"),)),
        Row("Table 5", r"\LESSER{} projection dimensions & $(m,n)$ & $(<0>,<1>)$ \\",
            (t5("LESSER projection dimension m", str(r["m"]), "64"), t5("LESSER projection dimension n", str(r["n"]), "128"))),
        Row("Table 5", r"\LESS{} checkpoints & $C$ & $<0>$ \\", (t5("LESS checkpoints C", str(r["C"]), "4"),)),
    ]
    paper_t6 = {1_000: ("97.0", "99.7", "9.47"), 5_000: ("86.8", "98.5", "8.58"), 10_000: ("76.7", "97.0", "7.69")}
    for k, s in r["shares"].items():
        p = paper_t6[k]
        rows.append(Row("Table 6", f"${tex_number(grouped(k))}$" + r" & $<0>$ & $<1>$ & $<2>\times$ \\", (
            t6(k, "LESSER extraction share (%)", fixed(100 * s["lesser"], 1), p[0]),
            t6(k, "LESS extraction share (%)", fixed(100 * s["less"], 1), p[1]),
            t6(k, "extraction + training FLOP ratio (x)", fixed(s["ratio"], 2), p[2]))))

    less_shares = [s["less"] for s in r["shares"].values()]
    ratios = [s["ratio"] for s in r["shares"].values()]
    inline = [
        Claim("F_LESS / F_LESSER (x)", fixed(r["ratio"], 1), "9.7",
              "Abstract; Sec. 1; Sec. 4 Cost analysis; App. A.4; Fig. 1 is checked by cost.table1",
              (r"\frac{F_{\LESS{}}}{F_{\LESSER{}}} = <0>\times.",
               r"reduces the feature-extraction FLOP cost by $<0>\times$ for SFT",
               r"requiring $<0>\times$ fewer FLOPs on SFT",
               r"this reduces feature-extraction FLOPs by $<0>\times$")),
        Claim("24PT / F_LESSER, 6PT convention (x)", fixed(r["ratio_six_pt"], 0), "12", "App. A.4",
              (r"giving the commonly used estimate of approximately $<0>\times$",)),
        Claim("min LESS extraction share (%)", fixed(100 * min(less_shares), 1), "97.0", "App. A.4",
              (r"feature extraction accounts for at least $<0>\%$ of \LESS{}'s modeled",)),
        Claim("min LESS extraction share, Sec. 4 (%)", fixed(100 * min(less_shares), 0), "97", "Sec. 4 Cost analysis",
              (r"Feature extraction accounts for $<0>\%$ to",)),
        Claim("max LESS extraction share, Sec. 4 (%)", fixed(100 * max(less_shares), 1), "99.7", "Sec. 4 Cost analysis",
              (r"to $<0>\%$ of \LESS{}'s combined extraction and fine-tuning FLOPs",)),
        Claim("min extraction + training ratio (x)", fixed(min(ratios), 1), "7.7", "App. A.4",
              (r"the combined extraction and fine-tuning cost is $<0>$--",)),
        Claim("max extraction + training ratio (x)", fixed(max(ratios), 1), "9.5", "App. A.4",
              (r"--$<0>\times$ larger for \LESS{} than for \LESSER{}",)),
    ]
    return inline + row_claims(rows), rows


def load(data_dir: Path) -> dict:
    """Compute the SFT FLOP quantities and confirm they reproduce the recorded computation."""
    res = compute(load_counts(data_dir))
    compare_reference(res, Path(data_dir) / REFERENCE)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SFT feature-extraction FLOPs (paper Appendix A.4).")
    add_common_args(ap)
    args = ap.parse_args(argv)
    res = load(args.data_dir)
    print(summary(res))
    print(f"  reproduces {args.data_dir / REFERENCE} (the computation behind the paper)")
    claims, rows = checks(res)
    finish(claims, rows, "SFT extraction FLOPs", args.paper_dir)


if __name__ == "__main__":
    main()
