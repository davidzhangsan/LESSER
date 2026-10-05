"""Regenerate every CPU result of Section 5 and Appendices E-F from ``data/analysis`` and compare with the paper.

  python -m analysis.verify                        # all checks; prints a table; exits 1 on any failure
  python -m analysis.verify --paper-dir PAPER      # also compare rendered figures pixel by pixel (needs pdftoppm)
  python -m analysis.verify manifest --write       # (re)write data/analysis/MANIFEST.json
  python -m analysis.verify payloads --paper-dir PAPER   # refresh data/analysis/reference/paper_figures

Three kinds of reference are used:
  * numbers as printed in the paper (``reference/paper_numbers.json``), compared after rounding;
  * the JSON payloads stored in the paper's figure PDFs (``reference/paper_figures``), compared to 1e-12;
  * the original analysis outputs that the paper numbers were read from (``reference/original_outputs``).
Outputs go to ``outputs/analysis`` (or ``--out``).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import batch_alignment, common, hb_decomposition, layer_curve, overlap, trajectories
from analysis.common import DATA_DIR, MODEL_KEYS, PAIRS

REF = DATA_DIR / "reference"
ORIGINAL = REF / "original_outputs"
PAPER_FIGURES = {  # regenerated figure -> ending of the same figure's file name in figures/ of the paper source
    "layer_curve": "layer_curve.pdf",
    "fig7a_alignment_base": "alignment_base.pdf",
    "fig7b_loss_share": "loss_share.pdf",
    "query_loss_curves": "query_loss_curves.pdf",
}
MANIFEST = DATA_DIR / "MANIFEST.json"
SOURCES = {  # provenance of each data/analysis subdirectory, recorded in the manifest
    "audit": "compact per-candidate audit records (analysis/audit.py compact); source files and their sha256 in audit/index.json",
    "batch_alignment/gram": "Gram matrices (analysis/batch_alignment.py gram) of the chunk-gradient sketches listed in each file's 'sources'",
    "batch_alignment/batches": "1,024-example samples (analysis/batch_alignment.py batches) of the k = 5,000 selections",
    "selections/k5000": "the randB uniform pool samples (analysis/selections.py build)",
    "trajectories/loss": "gzipped loss_trajectory.json of the 120 trajectory runs (analysis/trajectories.py pack-loss); index.json has their sha256",
    "reference/paper_figures": "JSON payloads read from the paper's figure PDFs (analysis/verify.py payloads)",
    "reference/original_outputs": "original analysis outputs that the reported numbers were read from",
    "reference/paper_numbers.json": ("numbers as printed in the paper, and the per-pair curves plotted in the Appendix F.3 figure, "
                                     "extracted from that figure's four source files (sha256 in F3_query_loss_curves.source_sha256)"),
    "reference/audit_port_cpu_check.json": "CPU check of analysis/audit.py compute against the paper's records (audit.py compare)",
}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, expected, got, ok):
        self.rows.append(dict(check=name, expected=expected, got=got, ok=bool(ok)))

    def close(self, name, expected, got, tol):
        diff = _maxdiff(expected, got)
        self.add(name, f"diff <= {tol:g}", f"max |diff| {diff:.2e}", diff <= tol)

    def rounded(self, name, expected, got, digits=2):
        self.add(name, expected, round(got, digits), round(got, digits) == expected)

    def show(self):
        width = max(len(r["check"]) for r in self.rows)
        for r in self.rows:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']:{width}s}  expected {r['expected']!s:24s} got {r['got']}")
        failed = sum(not r["ok"] for r in self.rows)
        print(f"\n{len(self.rows) - failed} of {len(self.rows)} checks passed")
        return failed


def _maxdiff(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            return float("inf")
        return max((_maxdiff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((_maxdiff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        return abs(a - b)
    return 0.0 if a == b else float("inf")


# ---------------------------------------------------------------------------------------------- manifest
def data_files():
    return sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.name not in {"MANIFEST.json"})


def source_of(rel):
    if rel in SOURCES:
        return SOURCES[rel]
    for prefix in sorted(SOURCES, key=len, reverse=True):
        if rel.startswith(prefix + "/"):
            return SOURCES[prefix]
    raise ValueError(f"no provenance recorded for data/analysis/{rel}; add it to SOURCES")


def external_files():
    """Files of other components that the analysis reads (paths relative to the repository root)."""
    from analysis import selections
    return {str(p.relative_to(common.REPO_ROOT)): p for p in selections.external_files()}


def write_manifest():
    files = {str(p.relative_to(DATA_DIR)): dict(sha256=common.sha256_file(p), bytes=p.stat().st_size,
                                                source=source_of(str(p.relative_to(DATA_DIR)))) for p in data_files()}
    external = {k: dict(sha256=common.sha256_file(p), bytes=p.stat().st_size) for k, p in external_files().items()}
    common.write_json(dict(files=files, total_bytes=sum(f["bytes"] for f in files.values()), external_inputs=external,
                           external_note="selection lists read from the SFT component (analysis/selections.py)"), MANIFEST)
    return files


def check_manifest(report):
    man = common.load_json(MANIFEST)
    present = {str(p.relative_to(DATA_DIR)) for p in data_files()}
    bad = [k for k, v in man["files"].items() if k not in present or common.sha256_file(DATA_DIR / k) != v["sha256"]]
    extra = sorted(present - set(man["files"]))
    report.add("data/analysis matches MANIFEST.json", f"{len(man['files'])} files", f"{len(bad)} changed, {len(extra)} unlisted",
               not bad and not extra)
    ext = external_files()
    changed = [k for k, v in man["external_inputs"].items() if k not in ext or common.sha256_file(ext[k]) != v["sha256"]]
    report.add("external inputs (data/sft selections) match MANIFEST.json", f"{len(man['external_inputs'])} files",
               f"{len(changed)} changed, {len(set(ext) - set(man['external_inputs']))} unlisted",
               not changed and set(ext) == set(man["external_inputs"]))


# ---------------------------------------------------------------------------------------------- payloads
def plotted_data(payload):
    """A paper figure's payload without the names of the scripts and files that drew it (their sha256 are kept)."""
    out = {k: v for k, v in payload.items() if k != "generator"}
    if "source" in out:
        out["source"] = {"sha256": out["source"]["sha256"]}
    if "sources" in out:
        out["sources"] = {k: {"sha256": v["sha256"]} for k, v in out["sources"].items()}
    return out


def paper_figure(paper_dir, name) -> Path:
    """The one PDF in figures/ of the paper source whose file name ends with PAPER_FIGURES[name]."""
    found = sorted(p for p in (Path(paper_dir) / "figures").glob("*.pdf") if p.name.endswith(PAPER_FIGURES[name]))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one paper figure ending in {PAPER_FIGURES[name]}, found {[p.name for p in found]}")
    return found[0]


def extract_payloads(paper_dir):
    from analysis.vector_figures import read_payload
    out = REF / "paper_figures"
    for name in PAPER_FIGURES:
        common.write_json(plotted_data(read_payload(paper_figure(paper_dir, name))), out / f"{name}.json")
    print("wrote payloads to", out)


def render_compare(report, paper_dir, fig_dir):
    """Pixel comparison of the regenerated figures with the paper's at 200 dpi."""
    from PIL import Image
    if shutil.which("pdftoppm") is None:
        report.add("figure rendering", "pdftoppm available", "missing", False)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in PAPER_FIGURES:
            imgs = []
            for tag, pdf in (("paper", paper_figure(paper_dir, name)), ("new", Path(fig_dir) / f"{name}.pdf")):
                stem = Path(tmp) / f"{tag}_{name}"
                subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", str(pdf), str(stem)], check=True)
                imgs.append(np.asarray(Image.open(f"{stem}.png").convert("RGB"), dtype=int))
            same_size = imgs[0].shape == imgs[1].shape
            diff = int(np.abs(imgs[0] - imgs[1]).max()) if same_size else -1
            report.add(f"figure {name}: pixels at 200 dpi", "identical", "identical" if diff == 0 else f"max diff {diff}", diff == 0)


# ---------------------------------------------------------------------------------------------- checks
def run(args):
    out = Path(args.out)
    figs = out / "figures"
    report = Report()
    check_manifest(report)
    paper = common.load_json(REF / "paper_numbers.json")
    pf = {k: common.load_json(REF / "paper_figures" / f"{k}.json") for k in PAPER_FIGURES}
    from analysis.vector_figures import read_payload

    # ---- F.1: per-example rankings and the H/B decomposition
    lc = layer_curve.compute()
    lc_path = common.write_json(lc, out / "per_example" / "layer_curve.json")
    layer_curve.figure(lc, lc_path, figs / "layer_curve.pdf")
    s = lc["summary"]
    ref_cells = common.load_json(ORIGINAL / "layer_cosine_curve.json")["cells"]
    report.close("F.1 layer curves vs original output", 0, _maxdiff(ref_cells, lc["cells"]), 0.0)
    p = read_payload(figs / "layer_curve.pdf")
    report.close("F.1 layer-curve figure payload vs paper", 0,
                 max(_maxdiff(pf["layer_curve"]["curves"], p["curves"]),
                     _maxdiff(pf["layer_curve"]["rankcorr_median_at"], p["rankcorr_median_at"])), 1e-12)
    hfr = common.load_json(ORIGINAL / "head_full_rank.json")
    report.close("F.1 per-pair output-layer rank correlation vs original output (recomputed audit)", 0,
                 max(abs(hfr[k]["rho_cos"] - v) for k, v in s["output_layer_rank_corr"].items()), 1e-4)
    report.rounded("F.1 median Spearman, output layer vs full", paper["F1_output_layer_median"], s["output_layer_rank_corr_median"])
    mm = s["output_layer_rank_corr_model_median"].values()
    report.rounded("F.1 per-model median, lowest", paper["F1_model_median_range"][0], min(mm))
    report.rounded("F.1 per-model median, highest", paper["F1_model_median_range"][1], max(mm))
    report.rounded("F.1 median with half of the blocks", paper["F1_half_blocks"], s["rankcorr_median_at"]["0.50"])
    report.rounded("F.1 median with all blocks", paper["F1_all_blocks"], s["rankcorr_median_at"]["1.00"])
    n = s["n_candidates"].values()
    report.add("F.1 candidates per pair", paper["F1_candidates_range"], [min(n), max(n)],
               [min(n), max(n)] == paper["F1_candidates_range"])
    hb = hb_decomposition.compute()
    common.write_json(hb, out / "per_example" / "hb_decomposition.json")
    report.rounded("F.1 median corr(H, B)", paper["F1_corr_HB_median"], hb["summary"]["corr_HB_median"])
    report.add("F.1 pairs with Delta_B > 0", paper["F1_dB_positive"], hb["summary"]["dB_positive"],
               hb["summary"]["dB_positive"] == paper["F1_dB_positive"])
    report.rounded("F.1 median Delta_B / Delta_H", paper["F1_ratio_dB_dH_median"], hb["summary"]["ratio_dB_dH_median"])
    ref_hb = {o["cell"]: o for o in common.load_json(ORIGINAL / "body_vs_head_on_picks.json")}
    report.close("F.1 H/B per pair vs original output (recomputed audit)", 0,
                 max(max(abs(ref_hb[k]["corr_HB"] - c["corr_HB"]), abs(ref_hb[k]["dB"] / ref_hb[k]["dH"] - c["ratio_dB_dH"]))
                     for k, c in hb["cells"].items()), 1e-4)

    # ---- F.2 / Figure 7a: batch-gradient alignment
    ba_dir = out / "batch_alignment"
    batch_alignment.read(batch_alignment.GRAM_DIR, ba_dir)
    b128 = common.load_json(ba_dir / "b128_comparison.json")
    summ = common.load_json(ba_dir / "summary.json")
    ref_b128 = common.load_json(ORIGINAL / "b128_comparison.json")
    report.close("F.2 b128 alignments vs original output (source of Fig. 7a)", 0,
                 _maxdiff([{k: r[k] for k in ("lesser_less", "rds_less", "rand_less", "less_less")} for r in ref_b128],
                          [{k: r[k] for k in ("lesser_less", "rds_less", "rand_less", "less_less")} for r in b128]), 1e-12)
    sa = common.load_json(ORIGINAL / "self_agreement.json")
    sizes = common.load_json(ba_dir / "batch_sizes.json")
    report.close("F.2 alignments at batch sizes 64-512 vs original output", 0,
                 max(abs(r[f"{a}_{nb}"] - q[f"{b}_{nb}"]) for r, q in zip(sa, sizes) for nb in batch_alignment.BATCH_SIZES
                     for a, b in (("LL", "lesser_less"), ("LESS", "less_less"), ("LESSER", "lesser_lesser"), ("RDS", "rds_less"))), 1e-12)
    raw = common.load_json(ba_dir / "raw_cosine.json")
    ref_raw = common.load_json(ORIGINAL / "uncentered_batch_cosine.json")
    report.close("F.2 raw cosines vs original output", 0, _maxdiff(ref_raw["cells"], raw["cells"]), 1e-9)
    med = summ["b128_median"]
    for key, label in (("lesser_less", "LESSER-LESS"), ("rds_less", "RDS+-LESS"), ("less_less", "LESS-LESS")):
        report.rounded(f"F.2 median alignment at 128, {label}", paper["F2_b128_median"][key], med[key])
    report.add("F.2 median alignment at 128, Random-LESS", "|x| < 0.01 (" + paper["F2_b128_median"]["rand_less"] + ")",
               round(med["rand_less"], 4), abs(med["rand_less"]) < 0.01)
    report.add("F.2 pairs with LESSER-LESS > 0", paper["F2_positive"], summ["b128_lesser_less_positive"],
               summ["b128_lesser_less_positive"] == paper["F2_positive"])
    report.rounded("F.2 raw cosine between random batches", paper["F2_raw_random"], summ["raw_cosine_random_random_median"])
    ratios = summ["ratio_lesser_less_over_less_less_median"]
    report.add("F.2 ratio LESSER-LESS / LESS-LESS, batch sizes 64-512", paper["F2_ratio_range"],
               [round(min(ratios.values()), 2), round(max(ratios.values()), 2)],
               [round(min(ratios.values()), 2), round(max(ratios.values()), 2)] == paper["F2_ratio_range"])
    report.rounded("F.2 ratio at batch size 128", paper["F2_ratio_128"], ratios["128"])
    batch_alignment.figure(ba_dir / "b128_comparison.json", figs / "fig7a_alignment_base.pdf")
    report.close("Fig. 7a payload medians vs paper", 0,
                 _maxdiff(pf["fig7a_alignment_base"]["medians"], read_payload(figs / "fig7a_alignment_base.pdf")["medians"]), 1e-12)

    # ---- F.3 / Figure 7b: training trajectories
    curves = trajectories.loss_curves()
    src = common.write_json(curves, out / "trajectories" / "loss_curves.json")
    share, counts = trajectories.loss_share_medians(curves)
    trajectories.figure_share(share, counts, src, figs / "fig7b_loss_share.pdf")
    q = trajectories.figure_curves(curves, src, figs / "query_loss_curves.pdf")
    p7b = read_payload(figs / "fig7b_loss_share.pdf")
    report.close("Fig. 7b payload vs paper", 0,
                 max(_maxdiff({k: {str(s): v for s, v in d.items()} for k, d in pf["fig7b_loss_share"]["loss_share_median"].items()},
                              {k: {str(s): v for s, v in d.items()} for k, d in p7b["loss_share_median"].items()}),
                     _maxdiff(pf["fig7b_loss_share"]["loss_share_pairs"], p7b["loss_share_pairs"])), 0.0)
    for name in ("LESSER", "LESS", "Random"):
        report.rounded(f"F.3 median share of loss decrease by step 10, {name}", paper["F3_step10"][name], share[name][10])
    report.add("F.3 pairs in the medians (LESSER, LESS, Random)", paper["F3_pairs"],
               [counts["LESSER"], counts["LESS"], counts["Random"]],
               [counts["LESSER"], counts["LESS"], counts["Random"]] == paper["F3_pairs"])
    f3 = paper["F3_query_loss_curves"]
    report.close("F.3 per-pair figure curves vs the curves plotted in the paper", 0, _maxdiff(f3["curves"], q["curves"]), 0.0)
    src_sha = {m: v["sha256"] for m, v in pf["query_loss_curves"]["sources"].items()}
    report.add("F.3 plotted curves were extracted from the paper figure's sources", "sha256 equal",
               "equal" if {m: v["sha256"] for m, v in f3["source_sha256"].items()} == src_sha else "differ",
               {m: v["sha256"] for m, v in f3["source_sha256"].items()} == src_sha)

    # ---- inputs derived from the selections: rebuilt exactly from data/sft and the stored randB samples
    from analysis import selections
    same = [batch_alignment.build_batches(m) == common.load_json(batch_alignment.BATCHES_DIR / f"{m}.json") for m in MODEL_KEYS]
    report.add("F.2 1,024-example samples rebuilt from the selections", "4 of 4 identical", f"{sum(same)} of 4 identical", all(same))
    same = [selections.load(m, t, "randB") == selections.random_b(t) for m, t in PAIRS]
    report.add("randB samples equal their rule", "20 of 20", f"{sum(same)} of 20", all(same))

    # ---- Appendix E: selection overlap
    ov = overlap.compute()
    rows = overlap.latex_rows(ov)
    report.add("App. E overlap table rows", "4 rows as printed", f"{sum(r in paper['E_overlap_rows'] for r in rows)} of 4 equal",
               rows == paper["E_overlap_rows"])
    report.rounded("App. E / Sec. 5.1 Llama-2-7B mean Jaccard", paper["E_llama2_mean_jaccard"], ov["mean_jaccard"]["llama2"], 3)
    report.add("App. E overlap vs original output", "equal",
               "equal" if ov["cells"] == common.load_json(ORIGINAL / "selection_overlap.json") else "differ",
               ov["cells"] == common.load_json(ORIGINAL / "selection_overlap.json"))

    if args.paper_dir:
        render_compare(report, args.paper_dir, figs)
    common.write_json(report.rows, out / "verify.json")
    return report.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=("run", "manifest", "payloads"))
    ap.add_argument("--out", default=str(common.DEFAULT_OUT))
    ap.add_argument("--paper-dir", help="paper source directory (figures/ holds the paper PDFs)")
    ap.add_argument("--write", action="store_true", help="manifest: rewrite MANIFEST.json")
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        if args.write:
            files = write_manifest()
            print(f"wrote {MANIFEST} ({len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e6:.1f} MB)")
        else:
            report = Report()
            check_manifest(report)
            raise SystemExit(report.show())
    elif args.cmd == "payloads":
        if not args.paper_dir:
            raise SystemExit("payloads needs --paper-dir")
        extract_payloads(args.paper_dir)
    else:
        raise SystemExit(1 if run(args) else 0)


if __name__ == "__main__":
    main()
