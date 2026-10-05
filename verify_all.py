"""Regenerate every paper result from the released data and compare with the paper (CPU, about a minute).

Usage (from the repository root):
    python verify_all.py [--paper-dir PATH_TO_PAPER_LATEX_SOURCES]

Without --paper-dir, each component compares against the paper values recorded in data/*/;
with it, the regenerated tables and numbers are also checked against the LaTeX sources.
Exits non-zero if any component reports an unexplained difference. Known paper-side
discrepancies listed by a component are reported but do not fail the run.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def checks(paper):
    p = ["--paper-dir", str(paper)] if paper else []
    return [
        ("core package", ["-m", "unittest", "discover", "-s", "tests"]),
        ("SFT unit tests", ["-m", "unittest", "sft.test_sft"]),
        ("SFT tables (Tables 9-11, 3, 13; Sec. 4 and C.1 numbers)", ["-m", "sft.paper_tables", *p]),
        ("SFT figures (Figs. 3, 4, 8-10; bin statistics)", ["-m", "sft.paper_figures", *p]),
        ("RL (Figs. 5-6; F.4 RL agreement)", ["rl/figures.py", "all"]),
        ("GRACE (Table 12; F.4 GRACE statistics)", ["grace/tables.py"]),
        ("Cost (Table 1, A.4)", ["-m", "cost.table1", *p]),
        ("Analysis (Sec. 5, Fig. 7, App. E-F)", ["-m", "analysis.verify", *p]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-dir", type=Path)
    ap.add_argument("--verbose", action="store_true", help="print each component's full output")
    args = ap.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), str(ROOT / ".deps"), env.get("PYTHONPATH")]))
    failed = []
    for name, cmd in checks(args.paper_dir):
        t0 = time.time()
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        last = out.splitlines()[-1] if out else ""
        print(f"[{'ok' if proc.returncode == 0 else 'FAIL'}] {name}  ({time.time() - t0:.0f}s)  {last[:100]}")
        if args.verbose or proc.returncode:
            print("\n".join("    " + line for line in out.splitlines()))
        if proc.returncode:
            failed.append(name)
    print(f"\n{len(checks(args.paper_dir)) - len(failed)} of {len(checks(args.paper_dir))} components reproduce the paper"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
