"""GRACE teacher-ranking feature extraction: wall clock and peak memory, full gradients against LESSER.

Paper: Table 8 and Appendix A.4 "GRACE extraction wall clock" (32 timed responses per teacher after
three untimed warm-up responses, extrapolated to 2,048 responses per teacher; 3.9--5.4x; lower peak
memory; cosine at least 0.99 with the stored features on two responses per setting); Section 4
"Cost analysis" and Section 1 (3.9 to 5.4x).

The GRACE timing job runs a timing harness on 32 random responses per teacher together with the
stored ranking features of the same rows. The harness times the GRACE pipeline's own feature functions one
response at a time on one A100 80GB PCIe, with an fp32 student and a 512-dimensional CUDA
projector, after three untimed warm-up responses:

    full            forward and backward pass, all parameter gradients, projected
    lesser          forward pass and output-layer gradient, projected: the LESSER column
    lesser_as_run   the same plus an unused two-sided sketch, as the ranking pipeline computes it
                    (supplementary; the key name is the one timing.json uses)

Extrapolation per setting: hours = mean seconds per response * 2,048 * teachers / 3600, and the
speedup is the ratio of the mean times. Peak memory is measured (GiB), not extrapolated.

Parses data/cost/timing_grace/timing.json (per-response times) and the job log. Checks: each stored
mean equals the mean of the per-response times; the stored hours use 2,048 responses per teacher;
the job log's summary lines agree with the JSON; the correctness cosines; an A100; rc 0.

    python -m cost.timing_grace.parse_timing [--paper-dir PAPER_SOURCE]
"""
from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

from cost._common import Claim, Row, add_common_args, finish, fixed, grouped, read_log, require, row_claims

SUBDIR = "timing_grace"
LOG = "grace_timing.log"
SETTINGS = {  # key in timing.json -> Table 8 row label
    "gsm8k_llama1b": "GSM8K, Llama-3.2-1B",
    "gsm8k_olmo1b": "GSM8K, OLMo-2-1B",
    "math_llama3b": "MATH, Llama-3.2-3B",
}
SKETCH = "lesser_as_run"  # key used by the raw timing log and timing.json for the LESSER setting with the unused sketch
METHODS = ("full", "lesser", SKETCH)
RESPONSES_PER_TEACHER = 2_048
NUMBER_WORDS = {2: "two", 3: "three"}
SUMMARY_LINE = re.compile(r"^\[(\w+)\] (\w+)\s+mean ([\d.]+) s/row\s+median ([\d.]+)\s+"
                          r"peak ([\d.]+) GiB\s+-> ([\d.]+) h for (\d+) x (\d+)$", re.M)


def parse_job_log(path: Path) -> dict:
    text = read_log(path)
    start = re.search(r"^START (\S+) host=(\S+) gpus: (.+?); sha=([0-9a-f]+)$", text, re.M)
    end = re.search(r"^END (\S+) rc=(-?\d+)$", text, re.M)
    if start is None or end is None:
        raise ValueError(f"{path}: no START or END line")
    if int(end.group(2)) != 0:
        raise ValueError(f"{path}: the job exited with rc={end.group(2)}")
    lines = {(m.group(1), m.group(2)): m.groups()[2:] for m in SUMMARY_LINE.finditer(text)}
    return dict(gpu=start.group(3).strip(), code_sha256_prefix=start.group(4), summary=lines)


def load(data_dir: Path) -> dict:
    """Parse and check the GRACE timing; return per-setting hours, speedups and peak memory."""
    d = Path(data_dir) / SUBDIR
    raw = json.loads(require(d / "timing.json").read_text())
    env = raw["_env"]
    if "A100" not in env["gpu"]:
        raise ValueError(f"timing.json: GPU {env['gpu']!r} is not an A100")
    log = parse_job_log(d / LOG)
    settings = {}
    for key in SETTINGS:
        S = raw[key]
        if not S.get("complete"):
            raise ValueError(f"timing.json: setting {key} is not complete")
        per_teacher = Fraction(S["n_timed"], S["n_teachers"])
        if per_teacher.denominator != 1:
            raise ValueError(f"{key}: {S['n_timed']} timed responses for {S['n_teachers']} teachers")
        out = dict(student=S["student"], teachers=S["n_teachers"], timed_per_teacher=int(per_teacher),
                   check_cosines=[min(c["full"], c["lesser"], c[SKETCH]) for c in S["check"]], methods={})
        for method in METHODS:
            M = S["methods"][method]
            times = M["per_row_seconds"]
            if len(times) != S["n_timed"]:
                raise ValueError(f"{key}/{method}: {len(times)} per-response times, expected {S['n_timed']}")
            if abs(sum(times) / len(times) - M["s_per_row_mean"]) > 1e-5:  # per-response times are stored to 1e-5 s
                raise ValueError(f"{key}/{method}: stored mean differs from the per-response times")
            mean = Fraction(M["s_per_row_mean"])
            hours = mean * RESPONSES_PER_TEACHER * S["n_teachers"] / 3600
            if abs(float(hours) - M["extrapolated_hours_all_teachers"]) > 1e-9:
                raise ValueError(f"{key}/{method}: stored hours do not use {RESPONSES_PER_TEACHER} responses per teacher")
            logged = log["summary"].get((key, method))
            expected = (f"{M['s_per_row_mean']:.3f}", f"{M['s_per_row_median']:.3f}", f"{M['peak_alloc_gib']}",
                        f"{M['extrapolated_hours_all_teachers']:.2f}", str(S["n_teachers"]), str(RESPONSES_PER_TEACHER))
            if logged is None or tuple(logged) != expected:
                raise ValueError(f"{LOG}: summary line for {key}/{method} {logged} differs from timing.json {expected}")
            out["methods"][method] = dict(mean=mean, hours=hours, peak_gib=Fraction(str(M["peak_alloc_gib"])))
        full, lesser = out["methods"]["full"], out["methods"]["lesser"]
        if not full["peak_gib"] > lesser["peak_gib"]:
            raise ValueError(f"{key}: LESSER peak memory is not below the full-gradient peak")
        out["speedup"] = full["mean"] / lesser["mean"]
        out["speedup_sketch"] = full["mean"] / out["methods"][SKETCH]["mean"]
        settings[key] = out
    return dict(env=env, log=log, settings=settings)


def summary(res: dict) -> str:
    env, log = res["env"], res["log"]
    lines = [
        "GRACE feature extraction (Appendix A.4, Table 8)",
        f"  GRACE timing job, {log['gpu']}; torch {env['torch']}, transformers {env['transformers']},"
        f" trak {env['trak']}; {env['warm']} warm-up responses; GRACE code sha256 {log['code_sha256_prefix']}... (not shipped)",
        "  setting                    teachers  timed   s/resp full / LESSER   hours full / LESSER   speedup"
        "   peak GiB full / LESSER   speedup incl. unused sketch",
    ]
    for key, label in SETTINGS.items():
        s = res["settings"][key]
        f, l = s["methods"]["full"], s["methods"]["lesser"]
        lines.append(f"  {label:<26} {s['teachers']:>8}  {s['timed_per_teacher']:>5}   {fixed(f['mean'], 3):>7} / {fixed(l['mean'], 3):<7}"
                     f"   {fixed(f['hours'], 2):>7} / {fixed(l['hours'], 2):<8}  {fixed(s['speedup'], 2):>6}x"
                     f"   {fixed(f['peak_gib'], 1):>7} / {fixed(l['peak_gib'], 1):<12}  {fixed(s['speedup_sketch'], 2)}x")
    cosines = [c for s in res["settings"].values() for c in s["check_cosines"]]
    lines.append(f"  correctness check: min cosine with the stored ranking features {min(cosines):.6f} over {len(cosines)} responses")
    return "\n".join(lines)


def checks(res: dict) -> tuple[list[Claim], list[Row]]:
    paper = {  # Table 8: teachers, hours full, hours LESSER, speedup, peak GiB full, peak GiB LESSER
        "gsm8k_llama1b": ("14", "6.49", "1.65", "3.9", "19.6", "12.3"),
        "gsm8k_olmo1b": ("10", "5.34", "0.99", "5.4", "17.0", "7.0"),
        "math_llama3b": ("10", "14.68", "3.09", "4.8", "39.8", "24.2"),
    }
    rows = []
    for key, label in SETTINGS.items():
        s, p = res["settings"][key], paper[key]
        f, l = s["methods"]["full"], s["methods"]["lesser"]
        got = (str(s["teachers"]), fixed(f["hours"], 2), fixed(l["hours"], 2), fixed(s["speedup"], 1),
               fixed(f["peak_gib"], 1), fixed(l["peak_gib"], 1))
        names = ("teachers", "full-gradient time (h)", "LESSER time (h)", "speedup (x)",
                 "full-gradient peak memory (GiB)", "LESSER peak memory (GiB)")
        rows.append(Row("Table 8", label + r", $<0>$ teachers & $<1>$ h & $<2>$ h & $<3>\times$ & $<4>$ / $<5>$ \\",
                        tuple(Claim(f"Table 8, {label}: {n}", g, q, "Table 8") for n, g, q in zip(names, got, p))))

    settings = list(res["settings"].values())
    per_teacher = {s["timed_per_teacher"] for s in settings}
    checked = {len(s["check_cosines"]) for s in settings}
    if len(per_teacher) != 1 or len(checked) != 1:
        raise ValueError("settings differ in timed or checked responses per teacher")
    speedups = [s["speedup"] for s in settings]
    min_cos = min(c for s in settings for c in s["check_cosines"])
    inline = [
        Claim("timed responses per teacher", str(per_teacher.pop()), "32", "App. A.4; Table 8",
              (r"we randomly sample $<0>$ responses and extract their features one at a time",
               r"estimated from $<0>$ timed responses per teacher")),
        Claim("untimed warm-up responses", NUMBER_WORDS.get(res["env"]["warm"], str(res["env"]["warm"])), "three",
              "App. A.4", (r"after <0> untimed warm-up responses",)),
        Claim("responses per teacher", grouped(RESPONSES_PER_TEACHER), "2,048", "App. A.4; Table 8",
              (r"by its $<0>$ responses, then sum across teachers", r"extrapolated to all $<0>$ responses per teacher")),
        Claim("min speedup over settings (x)", fixed(min(speedups), 1), "3.9", "App. A.4; Sec. 4; Sec. 1",
              (r"\LESSER{} is $<0>$--", r"this reduces feature-extraction time by $<0>$ to",
               r"and feature-extraction time by $<0>$--")),
        Claim("max speedup over settings (x)", fixed(max(speedups), 1), "5.4", "App. A.4; Sec. 4; Sec. 1",
              (r"--$<0>\times$ faster than full-gradient extraction", r"to $<0>\times$ while also lowering peak GPU memory",
               r"--$<0>\times$ in distillation teacher selection")),
        Claim("correctness check: cosine at least", "0.99" if min_cos >= 0.99 else fixed(min_cos, 5), "0.99", "App. A.4",
              (r"cosine similarity of at least $<0>$ with the stored features",)),
        Claim("checked responses per setting", NUMBER_WORDS.get(checked.pop(), "?"), "two", "App. A.4",
              (r"on <0> sampled responses from each setting",)),
    ]
    return inline + row_claims(rows), rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="GRACE extraction timing (paper Table 8, Appendix A.4).")
    add_common_args(ap)
    args = ap.parse_args(argv)
    res = load(args.data_dir)
    print(summary(res))
    claims, rows = checks(res)
    finish(claims, rows, "GRACE extraction wall clock", args.paper_dir)


if __name__ == "__main__":
    main()
