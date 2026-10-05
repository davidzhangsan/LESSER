"""Matched SFT feature-extraction wall clock: LESSER against released LESS on one A100.

Paper: Table 1(a); Appendix A.4 "SFT extraction wall clock" (576 s against 2,563 s on the same
10,000-example prefix; 3.16 h against 56.2 h on the 197,196-example pool; 17.8x); Section 4
"Cost analysis" (17.8x, and 4.4x against a single LESS checkpoint); Section 1 (17.8x).

The SFT timing job extracts features for the first 10,000 pool examples with Llama-2-7B three times,
in sequence in one job on one A100 80GB PCIe, bf16:

    lesser     the LESSER extractor; it also computes the hidden-state and residual-only diagnostic
               features in the same pass, so its time is an upper bound
    less_pi8   released LESS extraction, representation/less/compute_less_embeds.py of
               dcml-lab/targeted-instruction-selection at 8ff397a, unchanged, checkpoint 316, CUDA
               projector, default projection interval 8: the paper's LESS number
    less_pi16  the same code with projection interval 16, LESS's original setting (supplementary)

Each wall clock is the whole process, from model and data loading to writing the features.

Extrapolation, as in Appendix A.4: time is linear in the number of examples, and LESS runs once
per checkpoint,

    hours(pool) = wall_seconds * N_pool / N_prefix / 3600,   LESS total = C * hours(one checkpoint).

Parses data/cost/timing_sft/: timing.log (stage start and end lines with rc and wall_seconds, the
GPU and the sha256 of the timed code), lesser_prefix.log, less_pi8.log and less_pi16.log. Checks:
rc 0; wall_seconds equals the stage timestamps within 1 s; an A100 80GB; all 10,000 examples
processed by every stage; LESS used the CUDA projector and the 134,217,728 LoRA parameters of
cost.sft_flops.

    python -m cost.timing_sft.parse_logs [--paper-dir PAPER_SOURCE]
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from cost._common import POOL_EXAMPLES, Claim, add_common_args, finish, fixed, grouped, read_log
from cost.sft_flops import LESS_CHECKPOINTS, LLAMA2_7B, LORA_MODULES, LORA_RANK, lora_params

SUBDIR = "timing_sft"
STAGES = ("lesser", "less_pi8", "less_pi16")

HEADER = re.compile(r"^START (\S+) host=(\S+) gpu=(.+)$", re.M)
SHA = re.compile(r"^([0-9a-f]{64})  (\S+)$", re.M)
STAGE_START = re.compile(r"^=== (LESSER|LESS released code (pi\d+)) prefix (\d+)\.\.(\d+) start (\S+)$", re.M)
STAGE_END = re.compile(r"^=== (LESSER|LESS (pi\d+)) rc=(-?\d+) wall_seconds=(\d+) (\S+);", re.M)
TQDM_DONE = re.compile(r"(\d+)/(\d+) \[((?:\d+:)?\d+:\d+)<")


def _stage(match: re.Match) -> str:
    return "lesser" if match.group(1) == "LESSER" else f"less_{match.group(2)}"


def parse_timing_log(path: Path) -> dict:
    text = read_log(path)
    headers = HEADER.findall(text)
    if len(headers) != 1:
        raise ValueError(f"{path}: expected one START line, found {len(headers)}")
    starts = {_stage(m): m for m in STAGE_START.finditer(text)}
    ends = {_stage(m): m for m in STAGE_END.finditer(text)}
    out = dict(gpu=headers[0][2].strip(),
               sha256={Path(p).name: d for d, p in SHA.findall(text)}, stages={})
    for s in STAGES:
        if s not in starts or s not in ends:
            raise ValueError(f"{path}: no start/end line for {s}")
        first, last = int(starts[s].group(3)), int(starts[s].group(4))
        rc, wall = int(ends[s].group(3)), int(ends[s].group(4))
        if rc != 0:
            raise ValueError(f"{path}: {s} exited with rc={rc}")
        span = (datetime.fromisoformat(ends[s].group(5)) - datetime.fromisoformat(starts[s].group(5))).total_seconds()
        if abs(span - wall) > 1:
            raise ValueError(f"{path}: {s} wall_seconds={wall} but its timestamps are {span} s apart")
        out["stages"][s] = dict(wall_seconds=wall, examples=last - first)
    return out


def parse_lesser_log(path: Path) -> dict:
    text = read_log(path)
    rates = re.findall(r"^\[shard (\d+)_(\d+)\] (\d+)/(\d+) rate=([\d.]+)/s", text, re.M)
    done = re.findall(r"^\[shard \S+\] DONE (\d+) items$", text, re.M)
    if not rates or len(done) != 1:
        raise ValueError(f"{path}: no final progress or DONE line")
    return dict(examples=int(done[0]), final_rate=float(rates[-1][4]))


def parse_less_log(path: Path) -> dict:
    text = read_log(path)
    fields = dict(examples=r"^Number of training examples: (\d+)$", projector=r"^Using (\w+Projector)$",
                  lora_params=r"^Total number of parameters that require gradients: (\d+)$",
                  saved=r"^Saved normalized train grads to: (\S+)$")
    out = {}
    for key, pattern in fields.items():
        m = re.search(pattern, text, re.M)
        if m is None:
            raise ValueError(f"{path}: no line for {key}")
        out[key] = m.group(1)
    out["examples"], out["lora_params"] = int(out["examples"]), int(out["lora_params"])
    finished = [m for m in TQDM_DONE.finditer(text) if m.group(1) == m.group(2)]
    if not finished:
        raise ValueError(f"{path}: the extraction loop did not finish")
    hms = [int(x) for x in finished[-1].group(3).split(":")]
    out["loop_examples"] = int(finished[-1].group(1))
    out["loop_seconds"] = sum(v * 60 ** i for i, v in enumerate(reversed(hms)))
    return out


def load(data_dir: Path) -> dict:
    """Parse and check the SFT timing logs; return wall clocks and the pool extrapolation."""
    d = Path(data_dir) / SUBDIR
    res = parse_timing_log(d / "timing.log")
    if "A100" not in res["gpu"] or "80GB" not in res["gpu"]:
        raise ValueError(f"timing.log: GPU {res['gpu']!r} is not an A100 80GB")
    prefix = {st["examples"] for st in res["stages"].values()}
    if len(prefix) != 1:
        raise ValueError(f"stages timed different prefixes: {prefix}")
    n_prefix = prefix.pop()

    lesser = parse_lesser_log(d / "lesser_prefix.log")
    if lesser["examples"] != n_prefix:
        raise ValueError(f"lesser_prefix.log: {lesser['examples']} examples, expected {n_prefix}")
    lora = lora_params(LORA_RANK, LLAMA2_7B["hidden"], LLAMA2_7B["layers"], LORA_MODULES)
    less = {s: parse_less_log(d / f"{s}.log") for s in ("less_pi8", "less_pi16")}
    for s, log in less.items():
        if (log["examples"], log["loop_examples"]) != (n_prefix, n_prefix):
            raise ValueError(f"{s}.log: {log['examples']} / {log['loop_examples']} examples, expected {n_prefix}")
        if log["projector"] != "CudaProjector" or log["lora_params"] != lora:
            raise ValueError(f"{s}.log: {log['projector']} with {log['lora_params']:,} LoRA parameters")

    wall = {s: res["stages"][s]["wall_seconds"] for s in STAGES}
    hours = {s: Fraction(wall[s] * POOL_EXAMPLES, n_prefix * 3600) for s in STAGES}
    res.update(n_prefix=n_prefix, n_pool=POOL_EXAMPLES, checkpoints=LESS_CHECKPOINTS, lesser_log=lesser,
               less_logs=less, wall=wall, hours=hours,
               lesser_hours=hours["lesser"], less_hours=LESS_CHECKPOINTS * hours["less_pi8"],
               speedup=LESS_CHECKPOINTS * Fraction(wall["less_pi8"], wall["lesser"]),
               speedup_one_checkpoint=Fraction(wall["less_pi8"], wall["lesser"]),
               speedup_pi16=LESS_CHECKPOINTS * Fraction(wall["less_pi16"], wall["lesser"]))
    return res


def summary(res: dict) -> str:
    C = res["checkpoints"]
    lines = [
        "SFT extraction wall clock, Llama-2-7B (Appendix A.4, Table 1a)",
        f"  SFT timing job, {res['gpu']}; the same {res['n_prefix']:,}-example prefix in one job;"
        f" pool = prefix x {res['n_pool']:,}/{res['n_prefix']:,}",
        "  stage       prefix wall clock   pool, one pass   note",
    ]
    for s in STAGES:
        if s == "lesser":
            note = f"LESSER; final rate {res['lesser_log']['final_rate']:.2f} examples/s"
        else:
            loop = res["less_logs"][s]["loop_seconds"]
            which = "LESS, one checkpoint" if s == "less_pi8" else "LESS at interval 16 (supplementary)"
            note = (f"{which}; extraction loop {loop:,} s; x{C} checkpoints ="
                    f" {fixed(C * res['hours'][s], 2)} h")
        lines.append(f"  {s:<11} {res['wall'][s]:>15,} s   {fixed(res['hours'][s], 2):>12} h   {note}")
    lines += [
        f"  LESSER {fixed(res['lesser_hours'], 2)} h vs four-checkpoint LESS {fixed(res['less_hours'], 2)} h:"
        f" {fixed(res['speedup'], 2)}x; against one checkpoint {fixed(res['speedup_one_checkpoint'], 2)}x;"
        f" LESS at interval 16: {fixed(res['speedup_pi16'], 2)}x (supplementary)",
        "  timed code (sha256 from timing.log):",
    ]
    lines += [f"    {digest[:16]}  {name}" for name, digest in res["sha256"].items()]
    return "\n".join(lines)


def checks(res: dict) -> tuple[list[Claim], list]:
    claims = [
        Claim("LESSER prefix wall clock (s)", grouped(res["wall"]["lesser"]), "576", "App. A.4",
              (r"\LESSER{} takes $<0>$ seconds",)),
        Claim("LESS one-checkpoint prefix wall clock (s)", grouped(res["wall"]["less_pi8"]), "2,563", "App. A.4",
              (r"one checkpoint of released \LESS{} takes $<0>$ seconds",)),
        Claim("prefix examples", grouped(res["n_prefix"]), "10,000", "App. A.4",
              (r"the same $<0>$-example prefix of the candidate pool",)),
        Claim("pool examples", grouped(res["n_pool"]), "197,196", "App. A.4", (r"to the full $<0>$-example pool",)),
        Claim("LESSER extrapolated to the pool (h)", fixed(res["lesser_hours"], 2), "3.16", "App. A.4; Table 1(a)",
              (r"gives $<0>$ hours for \LESSER{}",)),
        Claim("four-checkpoint LESS extrapolated (h)", fixed(res["less_hours"], 1), "56.2", "App. A.4",
              (r"and $<0>$ hours for \LESS{}",)),
        Claim("wall-clock speedup, four checkpoints (x)", fixed(res["speedup"], 1), "17.8",
              "App. A.4; Sec. 4; Sec. 1; Table 1(a)",
              (r"a $<0>\times$ difference", r"wall-clock extraction time by \textbf{$<0>\times$}",
               r"reduces feature-extraction time by $<0>\times$ in SFT")),
        Claim("wall-clock speedup, one checkpoint (x)", fixed(res["speedup_one_checkpoint"], 1), "4.4", "Sec. 4",
              (r"extraction is $<0>\times$ faster in wall clock",)),
    ]
    return claims, []


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SFT extraction wall clock (paper Table 1a, Appendix A.4).")
    add_common_args(ap)
    args = ap.parse_args(argv)
    res = load(args.data_dir)
    print(summary(res))
    claims, rows = checks(res)
    finish(claims, rows, "SFT extraction wall clock", args.paper_dir)


if __name__ == "__main__":
    main()
