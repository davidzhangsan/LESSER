"""Shared helpers for the cost scripts: paths, exact rounding, paper checks and the data manifest.

Every cost script regenerates its numbers from the raw data in data/cost/ and compares each one with the
string the paper prints. A ``Claim`` holds one printed number and a ``Row`` holds one LaTeX table
row. Rounding is half up on the exact rational value (``fixed``, ``sci``), so no printed digit
depends on binary floating point near a rounding boundary.

With ``--paper-dir`` a script also confirms that each expected string occurs, in its LaTeX context,
in the paper source. This guards against transcription errors in the expected strings and against
later edits of the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from sft.common import N_POOL as POOL_EXAMPLES, unwrap_cc  # SFT candidate pool size (Tulu V2, 197,196 examples)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "cost"
MANIFEST = "MANIFEST.json"

# Paper source revision the expected strings were read from (commit of the paper repository).
PAPER_REVISION = "df9fc85"
# Paper source files that state cost numbers. All of them are compiled into the paper at PAPER_REVISION.
PAPER_SOURCES = (
    "contents/abstract.tex",
    "contents/introduction.tex",
    "contents/experiments.tex",
    "contents/analysis_new.tex",
    "contents/_appendix/setup_details.tex",
)


# ----------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------
def require(path: Path) -> Path:
    """Return ``path`` if it is an existing file; raise otherwise."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing input {path} (cost/README.md lists where each input comes from)")
    return path


def read_log(path: Path) -> str:
    """Read a log as text. Carriage returns written by progress bars become line breaks."""
    return require(path).read_text(encoding="utf-8").replace("\r", "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(require(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(data_dir: Path = DATA_DIR) -> int:
    """Check the size and sha256 of every file listed in MANIFEST.json; return the file count.

    Paths in ``files`` are relative to ``data_dir``. Any missing or changed file raises.
    """
    manifest = json.loads(require(Path(data_dir) / MANIFEST).read_text())
    files = manifest["files"]
    if (manifest["n_files"], manifest["total_bytes"]) != (len(files), sum(e["bytes"] for e in files.values())):
        raise RuntimeError("MANIFEST.json: n_files or total_bytes does not match its file entries")
    for rel, entry in files.items():
        path = Path(data_dir) / rel
        size, digest = require(path).stat().st_size, sha256(path)
        if (size, digest) != (entry["bytes"], entry["sha256"]):
            raise RuntimeError(f"{path}: {size} bytes, sha256 {digest}; MANIFEST.json records "
                               f"{entry['bytes']} bytes, sha256 {entry['sha256']}")
    return len(files)


# ----------------------------------------------------------------------------------------------
# Exact rounding and the paper's number formats
# ----------------------------------------------------------------------------------------------
def fixed(x, nd: int) -> str:
    """``x`` rounded half up to ``nd`` decimals, computed on its exact value (int, Fraction or float)."""
    f = Fraction(x)
    if f < 0:
        return "-" + fixed(-f, nd)
    n = math.floor(f * 10 ** nd + Fraction(1, 2))
    if nd == 0:
        return str(n)
    digits = str(n).rjust(nd + 1, "0")
    return f"{digits[:-nd]}.{digits[-nd:]}"


def sci(x, nd: int) -> str:
    """``x`` in scientific notation with ``nd`` mantissa decimals, written like '3.11e9'."""
    f = Fraction(x)
    if f <= 0:
        raise ValueError(f"sci() needs a positive value, got {x}")
    e = math.floor(math.log10(f))
    while f >= Fraction(10) ** (e + 1):
        e += 1
    while f < Fraction(10) ** e:
        e -= 1
    mantissa = fixed(f / Fraction(10) ** e, nd)
    if Fraction(mantissa) >= 10:  # rounding carried into the next power of ten
        e += 1
        mantissa = fixed(f / Fraction(10) ** e, nd)
    return f"{mantissa}e{e}"


def grouped(n) -> str:
    """Integer with thousands separators, written like '2,563'."""
    if Fraction(n).denominator != 1:
        raise ValueError(f"grouped() needs an integer, got {n}")
    return f"{int(n):,}"


def tex_number(s: str) -> str:
    """A printed number in the paper's LaTeX: '2,563' -> '2{,}563', '3.11e9' -> '3.11\\times10^9'."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)e(-?\d+)", s)
    if m:
        exp = m.group(2)
        return rf"{m.group(1)}\times10^{exp}" if len(exp) == 1 else rf"{m.group(1)}\times10^{{{exp}}}"
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "{,}", s)


# ----------------------------------------------------------------------------------------------
# Paper claims
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Claim:
    """One number the paper prints, next to the value regenerated from data/cost/.

    got      regenerated value, formatted the way the paper prints it
    paper    the string printed in the paper source at PAPER_REVISION
    where    where the number appears in the paper
    context  LaTeX snippets around the number with '<0>' in its place; with --paper-dir, each
             snippet filled with ``paper`` must occur in the paper source
    known    a documented paper-side discrepancy: reported, but it does not fail the run
    """
    name: str
    got: str
    paper: str
    where: str
    context: tuple[str, ...] = ()
    known: str | None = None

    @property
    def status(self) -> str:
        if self.got == self.paper:
            return "OK"
        return "KNOWN" if self.known else "DIFF"


@dataclass(frozen=True)
class Row:
    """One LaTeX table row; '<i>' in the template marks the i-th cell."""
    table: str
    template: str
    cells: tuple[Claim, ...]

    def tex(self, which: str = "got") -> str:
        """The row filled with the regenerated values (``got``) or the paper's values (``paper``)."""
        return fill(self.template, [getattr(c, which) for c in self.cells])


def fill(template: str, values: Sequence[str]) -> str:
    out = template
    for i, value in enumerate(values):
        out = out.replace(f"<{i}>", tex_number(value))
    if re.search(r"<\d+>", out):
        raise ValueError(f"unfilled placeholder in {template!r}")
    return out


def row_claims(rows: Sequence[Row]) -> list[Claim]:
    return [cell for row in rows for cell in row.cells]


def report(claims: Sequence[Claim], title: str) -> int:
    """Print one line per claim; return the number of unexplained differences."""
    print(f"\n{title}: regenerated value vs the paper source at {PAPER_REVISION}")
    width = max(len(c.name) for c in claims)
    for c in claims:
        line = f"  {c.status:<5} {c.name:<{width}}  {c.got:>13}  paper {c.paper:>13}   [{c.where}]"
        if c.status == "KNOWN":
            line += f"\n        known paper-side discrepancy: {c.known}"
        elif c.status == "DIFF":
            line += "   <-- DIFFERS"
        print(line)
    n = {s: sum(c.status == s for c in claims) for s in ("OK", "KNOWN", "DIFF")}
    print(f"  {n['OK']} match, {n['KNOWN']} known paper-side discrepancy, {n['DIFF']} unexplained difference")
    return n["DIFF"]


def paper_text(paper_dir: Path) -> str:
    """PAPER_SOURCES concatenated, with LaTeX comments removed and whitespace collapsed."""
    lines = []
    for rel in PAPER_SOURCES:
        lines += require(Path(paper_dir) / rel).read_text(encoding="utf-8").splitlines()
    text = re.sub(r"\s+", " ", " ".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in lines))
    return unwrap_cc(text)



def check_paper_source(claims: Sequence[Claim], rows: Sequence[Row], paper_dir: Path) -> int:
    """Confirm that every expected string occurs, in context, in the paper source; return the number missing."""
    text = paper_text(paper_dir)
    snippets = [(c.name, fill(ctx, [c.paper])) for c in claims for ctx in c.context]
    snippets += [(r.table, r.tex("paper")) for r in rows]
    missing = [(name, s) for name, s in snippets if re.sub(r"\s+", " ", s).strip() not in text]
    print(f"\nPaper source check ({paper_dir}): {len(snippets) - len(missing)} of {len(snippets)} snippets found")
    for name, s in missing:
        print(f"  MISSING [{name}] {s}")
    return len(missing)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="directory with the cost data (default: data/cost)")
    ap.add_argument("--paper-dir", type=Path, default=None,
                    help="optional paper source checkout; confirms each expected string occurs in it")


def finish(claims: Sequence[Claim], rows: Sequence[Row], title: str, paper_dir: Path | None) -> None:
    """Report all checks; exit non-zero on any unexplained difference or missing paper snippet."""
    bad = report(claims, title)
    if paper_dir is not None:
        bad += check_paper_source(claims, rows, paper_dir)
    if bad:
        raise SystemExit(f"{bad} check(s) failed")
