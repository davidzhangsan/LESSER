"""Matched RL scoring wall clock: projected LESSER, exact LESSER and full gradients on one A100.

Paper: Table 1(b); Appendix A.4 "RL scoring wall clock" (81,920 candidate and 3,200 query responses,
18.1 min against 63.3 min, 3.5x); Section 4 "Cost analysis" and Section 1 (3.5x).

The RL timing job re-scores the responses of one selection round (5,120 candidate and 200 query problems, 16 responses
each, Qwen2.5-Math-1.5B-Instruct) with the released GradAlign scorer, three times in sequence in one
job on one A100 80GB PCIe (mini-batch 2, bf16, maximum length 4,096):

    prodsk  projected output-layer feature, 64 x 128 = 8,192 values (GRADALIGN_PROD_SKETCH=1),
            the LESSER row of Table 1(b)
    prod    exact output-layer feature, |V| x d values (GRADALIGN_PROD_SKETCH=0), supplementary
    sim     full policy gradients, the full-gradient row of Table 1(b)

Each wall clock covers the whole scorer process: tokenization, model loading, scoring and writing.

Parses data/cost/timing_rl/: timing.log (start and exit line per variant with exit code,
wall_seconds and scored groups), the three variant logs (response counts, parameter count, feature
dimension, GPU, scorer mode) and agreement.json (ranking agreement between the variants and the
in-generation scores, printed for reference). Checks: exit code 0 and 5,120 scored groups per
variant; wall_seconds equals the timestamps within 1 s; 81,920 candidate and 3,200 query responses;
an A100 80GB; feature dimensions 8,192 (prodsk) and |V| d (prod); mode sim for full gradients.

    python -m cost.timing_rl.parse_logs [--paper-dir PAPER_SOURCE]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from cost._common import Claim, add_common_args, finish, fixed, grouped, read_log, require

SUBDIR = "timing_rl"
VARIANTS = ("prodsk", "prod", "sim")
LABELS = {"prodsk": "LESSER, projected", "prod": "LESSER, exact", "sim": "full gradients"}
RESPONSES_PER_PROBLEM = 16

HEADER = re.compile(r"^=== (\S+) start host=(\S+) gpu=(.+)$", re.M)
SHA = re.compile(r"^([0-9a-f]{64})  (\S+)$", re.M)
LAYOUT = re.compile(r"^layout written: (\d+) candidate responses, (\d+) query responses$", re.M)
START = re.compile(r"^=== (\S+) (prodsk|prod|sim) start \(GRADALIGN_PROD_SKETCH=(\d)\)$", re.M)
EXIT = re.compile(r"^=== (\S+) (prodsk|prod|sim) exit (-?\d+) wall_seconds=(\d+) rows=(\d+)$", re.M)


def _one(pattern: re.Pattern, text: str, what: str) -> re.Match:
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise ValueError(f"expected one {what} line, found {len(found)}")
    return found[0]


def parse_timing_log(path: Path) -> dict:
    text = read_log(path)
    header = _one(HEADER, text, "header")
    layout = _one(LAYOUT, text, "layout")
    starts = {m.group(2): m for m in START.finditer(text)}
    exits = {m.group(2): m for m in EXIT.finditer(text)}
    out = dict(gpu=header.group(3).strip(), candidates=int(layout.group(1)),
               queries=int(layout.group(2)), sha256={m.group(2): m.group(1) for m in SHA.finditer(text)},
               variants={})
    for v in VARIANTS:
        if v not in starts or v not in exits:
            raise ValueError(f"{path}: no start/exit line for {v}")
        t0 = datetime.fromisoformat(starts[v].group(1))
        t1, rc, wall, rows = (datetime.fromisoformat(exits[v].group(1)), int(exits[v].group(3)),
                              int(exits[v].group(4)), int(exits[v].group(5)))
        if rc != 0:
            raise ValueError(f"{path}: {v} exited with {rc}")
        if abs((t1 - t0).total_seconds() - wall) > 1:
            raise ValueError(f"{path}: {v} wall_seconds {wall} but timestamps differ by {(t1 - t0).total_seconds()} s")
        out["variants"][v] = dict(wall_seconds=wall, groups=rows, sketch_flag=int(starts[v].group(3)))
    return out


def parse_variant_log(path: Path) -> dict:
    text = read_log(path)
    train = re.search(r"^Loaded (\d+) train responses from (\d+) groups$", text, re.M)
    val = re.search(r"^Loaded (\d+) val responses from (\d+) groups$", text, re.M)
    params = re.search(r"Model parameters: ([\d,]+)$", text, re.M)
    gpu = re.search(r"^GPU cuda: (.+)$", text, re.M)
    mode = re.search(r" --mode (\w+) ", text)
    feature = re.search(r"PROD sketch built: V=(\d+), H=(\d+), feature_dim=(\d+)$", text, re.M)
    for what, m in (("train responses", train), ("val responses", val), ("model parameters", params),
                    ("GPU", gpu), ("scorer mode", mode)):
        if m is None:
            raise ValueError(f"{path}: no {what} line")
    if "Parallel analysis completed successfully!" not in text:
        raise ValueError(f"{path}: the scorer did not report completion")
    return dict(train_responses=int(train.group(1)), train_groups=int(train.group(2)),
                val_responses=int(val.group(1)), val_groups=int(val.group(2)),
                params=int(params.group(1).replace(",", "")), gpu=gpu.group(1).strip(), mode=mode.group(1),
                feature=None if feature is None else dict(vocab=int(feature.group(1)), hidden=int(feature.group(2)),
                                                          dim=int(feature.group(3))),
                groups_saved=len(re.findall(r"^Saved result for group \d+ ", text, re.M)))


def load(data_dir: Path) -> dict:
    """Parse and check the RL timing logs; return wall clocks and the facts the logs record."""
    d = Path(data_dir) / SUBDIR
    res = parse_timing_log(d / "timing.log")
    if "A100" not in res["gpu"] or "80GB" not in res["gpu"]:
        raise ValueError(f"timing.log: GPU {res['gpu']!r} is not an A100 80GB")
    logs = {v: parse_variant_log(d / f"{v}.log") for v in VARIANTS}
    groups = res["candidates"] // RESPONSES_PER_PROBLEM
    for v, log in logs.items():
        facts = (log["train_responses"], log["val_responses"], log["train_groups"], log["groups_saved"],
                 res["variants"][v]["groups"])
        if facts != (res["candidates"], res["queries"], groups, groups, groups):
            raise ValueError(f"{v}.log: responses/groups {facts} differ from the round layout")
        if log["gpu"] != res["gpu"]:
            raise ValueError(f"{v}.log: GPU {log['gpu']!r} differs from timing.log {res['gpu']!r}")
    if len({log["params"] for log in logs.values()}) != 1:
        raise ValueError("the three variants logged different parameter counts")
    if (logs["prodsk"]["mode"], logs["prod"]["mode"], logs["sim"]["mode"]) != ("prod", "prod", "sim"):
        raise ValueError("scorer modes differ from prodsk=prod, prod=prod, sim=sim")
    if (res["variants"]["prodsk"]["sketch_flag"], res["variants"]["prod"]["sketch_flag"]) != (1, 0):
        raise ValueError("GRADALIGN_PROD_SKETCH flags differ from prodsk=1, prod=0")
    res["logs"] = logs
    res["params"] = logs["sim"]["params"]
    res["minutes"] = {v: Fraction(res["variants"][v]["wall_seconds"], 60) for v in VARIANTS}
    full = res["variants"]["sim"]["wall_seconds"]
    res["speedup"] = {v: Fraction(full, res["variants"][v]["wall_seconds"]) for v in VARIANTS}
    res["agreement"] = json.loads(require(d / "agreement.json").read_text())
    return res


def summary(res: dict) -> str:
    groups = res["candidates"] // RESPONSES_PER_PROBLEM
    lines = [
        "RL scoring wall clock, one selection round (Appendix A.4, Table 1b)",
        f"  RL timing job, {res['gpu']}; {res['candidates']:,} candidate responses "
        f"({groups:,} problems) + {res['queries']:,} query responses; {res['params']:,} parameters",
        "  variant                     feature values   wall clock            vs full gradients",
    ]
    for v in VARIANTS:
        feat = res["logs"][v]["feature"]
        dim = f"{feat['dim']:,}" if feat else "all gradients"
        wall = res["variants"][v]["wall_seconds"]
        note = "   (exact feature: not in the compiled paper)" if v == "prod" else ""
        lines.append(f"  {LABELS[v] + ' (' + v + ')':<27} {dim:>14}   {wall:>5,} s = {fixed(res['minutes'][v], 1):>5} min"
                     f"   {fixed(res['speedup'][v], 1):>6}x{note}")
    lines.append("  ranking agreement, Spearman on live groups / top-256 overlap (agreement.json):")
    for pair, a in res["agreement"].items():
        lines.append(f"    {pair:<18} {a['spearman_live']:.4f} / {a['top256_overlap']:.3f}   ({a['live']:,} live of {a['n']:,})")
    lines.append("  timed scorer code (sha256 from timing.log):")
    lines += [f"    {digest[:16]}  {name}" for name, digest in res["sha256"].items()]
    return "\n".join(lines)


def checks(res: dict) -> tuple[list[Claim], list]:
    m, s = res["minutes"], res["speedup"]
    claims = [
        Claim("candidate responses", grouped(res["candidates"]), "81,920", "App. A.4",
              (r"On the $<0>$ candidate and",)),
        Claim("query responses", grouped(res["queries"]), "3,200", "App. A.4",
              (r"candidate and $<0>$ query responses of one round",)),
        Claim("LESSER projected scoring time (min)", fixed(m["prodsk"], 1), "18.1", "App. A.4; Sec. 4; Table 1(b)",
              (r"this pass takes $<0>$ minutes", r"to $<0>$ minutes on one A100")),
        Claim("full-gradient scoring time (min)", fixed(m["sim"], 1), "63.3", "App. A.4; Sec. 4; Table 1(b)",
              (r"against $<0>$ minutes for the full-gradient forward--backward pass",
               r"the added scoring time falls from $<0>$ to")),
        Claim("scoring-time reduction (x)", fixed(s["prodsk"], 1), "3.5", "App. A.4; Sec. 4; Sec. 1; Table 1(b)",
              (r"forward--backward pass, a $<0>\times$ reduction.", r"a \textbf{$<0>\times$} reduction.",
               r"the added scoring time per selection round by $<0>\times$ in RL")),
    ]
    return claims, []


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="RL scoring wall clock (paper Table 1b, Appendix A.4).")
    add_common_args(ap)
    args = ap.parse_args(argv)
    res = load(args.data_dir)
    print(summary(res))
    claims, rows = checks(res)
    finish(claims, rows, "RL scoring wall clock", args.paper_dir)


if __name__ == "__main__":
    main()
