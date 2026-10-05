"""Table 1 of the paper and every Appendix A.4 cost number, regenerated from data/cost/ and checked.

Table 1 ("Output-layer gradients are much cheaper to extract", label tab:selection-cost):

    (a) SFT selector: extraction time on the 197,196-example pool, extrapolated from the matched A100
        timing (cost.timing_sft.parse_logs): LESSER 3.16 h, four-checkpoint LESS 56.2 h, 17.8x.
    (b) RL selector: added scoring time per selection round (cost.timing_rl.parse_logs): LESSER
        18.1 min, full gradients 63.3 min, 3.5x; stored feature per problem (cost.rl_flops):
        32.8 kB against 6.2 GB.

The script verifies data/cost/MANIFEST.json, prints Table 1 and the Appendix A.4 sections (SFT FLOPs,
SFT wall clock, RL FLOPs and storage, RL wall clock, GRACE timing), and compares every number
with the paper. It exits non-zero on any unexplained difference.

    python -m cost.table1                                  # print and check
    python -m cost.table1 --latex                          # also print LaTeX rows of Tables 1 and 5-8
    python -m cost.table1 --paper-dir PAPER_SOURCE         # also find every expected string in the paper source
    python -m cost.table1 --json-out cost_numbers.json     # also write every regenerated number
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from cost import rl_flops, sft_flops
from cost._common import (PAPER_REVISION, Claim, Row, add_common_args, check_paper_source, fixed, report,
                          row_claims, verify_manifest)
from cost.timing_grace import parse_timing as timing_grace
from cost.timing_rl import parse_logs as timing_rl
from cost.timing_sft import parse_logs as timing_sft

def table1(sft_t: dict, rl_t: dict, rl_f: dict) -> tuple[list[Claim], list[Row]]:
    """Table 1 rows regenerated from the SFT timing, the RL timing and the RL storage sizes."""
    kb = fixed(Fraction(rl_f["storage"]["lesser"], 10**3), 1)
    gb = fixed(Fraction(rl_f["storage"]["full"], 10**9), 1)
    rows = [
        Row("Table 1(a)", r"\LESSER{} & $<0>$ & $\mathbf{<1>}\times$ \\", (
            Claim("Table 1(a): LESSER extraction time (h)", fixed(sft_t["lesser_hours"], 2), "3.16", "Table 1(a)"),
            Claim("Table 1(a): LESSER speedup (x)", fixed(sft_t["speedup"], 1), "17.8", "Table 1(a)"))),
        Row("Table 1(a)", r"\LESS{} & $<0>$ & $1.0\times$ \\", (
            Claim("Table 1(a): LESS extraction time (h)", fixed(sft_t["less_hours"], 2), "56.16", "Table 1(a)"),)),
        Row("Table 1(b)", r"\LESSER{} & $<0>$ & $\mathbf{<1>}\times$ & $<2>$~kB \\", (
            Claim("Table 1(b): LESSER added time (min)", fixed(rl_t["minutes"]["prodsk"], 1), "18.1", "Table 1(b)"),
            Claim("Table 1(b): LESSER speedup (x)", fixed(rl_t["speedup"]["prodsk"], 1), "3.5", "Table 1(b)"),
            Claim("Table 1(b): LESSER feature size (kB)", kb, "32.8", "Table 1(b)"))),
        Row("Table 1(b)", r"Full gradients & $<0>$ & $1.0\times$ & $<1>$~GB \\", (
            Claim("Table 1(b): full-gradient added time (min)", fixed(rl_t["minutes"]["sim"], 1), "63.3", "Table 1(b)"),
            Claim("Table 1(b): full-gradient feature size (GB)", gb, "6.2", "Table 1(b)"))),
    ]
    return row_claims(rows), rows


def table1_text(sft_t: dict, rl_t: dict, rl_f: dict) -> str:
    kb = fixed(Fraction(rl_f["storage"]["lesser"], 10**3), 1)
    gb = fixed(Fraction(rl_f["storage"]["full"], 10**9), 1)
    m, s = rl_t["minutes"], rl_t["speedup"]
    return "\n".join([
        "Table 1: Output-layer gradients are much cheaper to extract (one A100 80GB)",
        "  (a) SFT selector    Time (h)   Speedup",
        f"      LESSER        {fixed(sft_t['lesser_hours'], 2):>8}   {fixed(sft_t['speedup'], 1):>6}x",
        f"      LESS          {fixed(sft_t['less_hours'], 2):>8}   {'1.0':>6}x",
        "  (b) RL selector    Added time (min)   Speedup   Feature size",
        f"      LESSER        {fixed(m['prodsk'], 1):>16}   {fixed(s['prodsk'], 1):>6}x   {kb:>9} kB",
        f"      Full gradients{fixed(m['sim'], 1):>16}   {'1.0':>6}x   {gb:>9} GB",
    ])


def jsonable(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Table 1 and the Appendix A.4 cost numbers, regenerated and checked.")
    add_common_args(ap)
    ap.add_argument("--latex", action="store_true", help="print LaTeX rows of Tables 1 and 5-8 from the regenerated values")
    ap.add_argument("--json-out", type=Path, default=None, help="write every regenerated number and check to this file")
    args = ap.parse_args(argv)

    n_files = verify_manifest(args.data_dir)
    print(f"{args.data_dir / 'MANIFEST.json'}: {n_files} files verified (size and sha256)")
    sft_f = sft_flops.load(args.data_dir)
    sft_t = timing_sft.load(args.data_dir)
    rl_t = timing_rl.load(args.data_dir)
    rl_f = rl_flops.load(args.data_dir, timing=rl_t)
    grace = timing_grace.load(args.data_dir)

    print("\n" + table1_text(sft_t, rl_t, rl_f))
    for text in (sft_flops.summary(sft_f), timing_sft.summary(sft_t), rl_flops.summary(rl_f),
                 timing_rl.summary(rl_t), timing_grace.summary(grace)):
        print("\n" + text)

    groups = [
        ("Table 1", *table1(sft_t, rl_t, rl_f)),
        ("SFT extraction FLOPs (App. A.4, Tables 5 and 6)", *sft_flops.checks(sft_f)),
        ("SFT extraction wall clock (App. A.4)", *timing_sft.checks(sft_t)),
        ("RL scoring FLOPs and storage (App. A.4, Table 7)", *rl_flops.checks(rl_f)),
        ("RL scoring wall clock (App. A.4)", *timing_rl.checks(rl_t)),
        ("GRACE extraction wall clock (App. A.4, Table 8)", *timing_grace.checks(grace)),
    ]
    bad = sum(report(claims, title) for title, claims, _ in groups)
    claims = [c for _, cs, _ in groups for c in cs]
    rows = [r for _, _, rs in groups for r in rs]
    n = {s: sum(c.status == s for c in claims) for s in ("OK", "KNOWN", "DIFF")}
    print(f"\nAll groups: {n['OK']} of {len(claims)} numbers match the paper source at {PAPER_REVISION}; "
          f"{n['KNOWN']} known paper-side discrepancy; {n['DIFF']} unexplained difference")
    if args.paper_dir is not None:
        bad += check_paper_source(claims, rows, args.paper_dir)
    if args.latex:
        print("\nLaTeX rows from the regenerated values:")
        table = None
        for r in rows:
            if r.table != table:
                print(f"% {r.table}")
                table = r.table
            print(r.tex("got"))
    if args.json_out is not None:
        out = dict(paper_revision=PAPER_REVISION, sft_flops=sft_f, sft_timing=sft_t, rl_flops=rl_f, rl_timing=rl_t,
                   grace_timing=grace, claims=[dict(name=c.name, regenerated=c.got, paper=c.paper, status=c.status,
                                                    where=c.where, known=c.known) for c in claims])
        args.json_out.write_text(json.dumps(jsonable(out), indent=1) + "\n")
        print(f"wrote {args.json_out}")
    if bad:
        raise SystemExit(f"{bad} check(s) failed")


if __name__ == "__main__":
    main()
