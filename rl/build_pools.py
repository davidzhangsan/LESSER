#!/usr/bin/env python3
"""Rebuild the problem pools of the RL experiments from the upstream Hugging Face datasets.

The GradAlign runs read these files from ``data_local/data/<name>/`` of the patched GradAlign checkout:

  cdmix4               34,000 problems: 4,000 Countdown, 20,000 WebInstruct, 10,000 DAPO (Figure 6 pool)
  countdown4_val       200 Countdown problems, the selection queries of Figure 6
  countdown4_test      500 held-out Countdown problems, verl's validation set in Figure 6
  dapo_noisy50_sub512  512 DAPO problems, 256 of them with coin-flip rewards (Figure 5 pool)
  amc22                43 AMC 2022 problems, the selection queries of Figure 5
  amc23                40 AMC 2023 problems, verl's validation set of the Figure 5 runs (never used,
                       because those runs stop after the first selection)

Each pool is rebuilt from four upstream files at pinned revisions (``SOURCES``) with the procedures that
built the originals: GradAlign's ``automated/prepare_data.py`` (WebInstruct, DAPO, AMC), the released
``automated/mix.py`` (cdmix4) and the seeded corruption step of the original runs (the DAPO pool of Figure 5;
its ``mix.py`` options are not part of the LESSER patch), the chunking of
``automated/dynamic_selection.py``, and three small procedures of the experiments reproduced here (the
Countdown splits and the 512-problem subsample). The released ``mix.py`` shuffles without a seed, so the
order of cdmix4 is shipped as metadata (``data/rl/pools/cdmix4_order.json``).

Verification (``data/rl/pools``): every JSONL file must match ``SHA256SUMS`` byte for byte. A Parquet file is
byte-identical when this script runs with the library versions that wrote the original (``pools.json``
records them: the GradAlign environment of ``third_party/GradAlign/environment.txt`` for cdmix4,
countdown4_val, dapo_noisy50_sub512 and amc22; pyarrow 21.0.0 with datasets 4.1.1 for countdown4_test and
amc23). With other versions only the Parquet encoding can differ, so its content (columns, rows and values,
ignoring the order of struct fields) must match the recorded content digest; ``--require-identical-bytes``
turns that case into an error. Any other difference raises and the file is not written.

Usage (from the repository root):

    python rl/build_pools.py --out $GRADALIGN_DIR/data_local/data        # download, build, verify
    python rl/build_pools.py --check $GRADALIGN_DIR/data_local/data      # verify existing files only
    python rl/build_pools.py --out DIR --only amc22 amc23                # a subset
    python rl/build_pools.py --out DIR --source-file webinstruct=/path/train-00000-of-00001.parquet

Upstream files are fetched with ``huggingface_hub.hf_hub_download`` (Hugging Face cache and offline mode as
usual) or taken from ``--source-file``; either way their SHA-256 must match ``SOURCES``. ``--prepared-dir``
reuses existing GradAlign ``prepare_data.py`` outputs (``<dir>/{countdown4,webinstruct,dapo}/train.jsonl``)
after checking them against their recorded SHA-256.

Requirements: pyarrow, numpy and huggingface_hub; datasets for the Parquet files that datasets wrote; pandas
for dapo_noisy50_sub512. Licenses of the upstream data: see ``SOURCES`` and rl/README.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POOLS_META = REPO_ROOT / "data" / "rl" / "pools"

SOURCES = {
    "countdown": dict(repo="Jiayi-Pan/Countdown-Tasks-3to4", revision="408f70d177020686d34a56bba5952feb45aaaee4",
                      file="data/train-00000-of-00001.parquet", bytes=2845904,
                      sha256="a1b37bea0f439c858b11b908daa682e521f13f9592f1866b08650f58b730d126",
                      license="none stated on the dataset card"),
    "webinstruct": dict(repo="TIGER-Lab/WebInstruct-verified-unfiltered", revision="ac48a536b31d3cabb346e1139fb6c69ae631abd7",
                        file="data/train-00000-of-00001.parquet", bytes=150168610,
                        sha256="14103d8dbf664fbac9e72e6e3d4894b4220139207ca1203bf14c9b825aa52ee6", license="apache-2.0"),
    "dapo": dict(repo="BytedTsinghua-SIA/DAPO-Math-17k", revision="65877096c24ffa7abc4e4fa5edb95cf3413a5674",
                 file="data/dapo-math-17k.parquet", bytes=299363855,
                 sha256="534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941", license="apache-2.0"),
    "amc": dict(repo="AI-MO/aimo-validation-amc", revision="69d78a4a2c840e82d69af6bc742bda09005f6316",
                file="data/train-00000-of-00001.parquet", bytes=19141,
                sha256="4de056fd006062cee46e3900727dd5d00094b8592ed0bfd6b5dfae2267f72dd7", license="apache-2.0"),
}
POOLS = ("cdmix4", "countdown4_val", "countdown4_test", "dapo_noisy50_sub512", "amc22", "amc23")
NEEDS = {"cdmix4": ("countdown", "webinstruct", "dapo"), "countdown4_val": ("countdown",),
         "countdown4_test": ("countdown",), "dapo_noisy50_sub512": ("dapo",), "amc22": ("amc",), "amc23": ("amc",)}

# GradAlign prepare_data.py: prompt prefix, reward style and number of rows prepared per dataset.
BOXED_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
MATH_STYLE = "rule-lighteval/MATH_v2"
PREPARE_MAX_SAMPLES = 40000
# Countdown prompt; the repository's countdown_judge.py parses numbers and target from it.
COUNTDOWN_TEMPLATE = ("Given the numbers {nums}, find a way to combine them using basic arithmetic operations (+, -, *, /) "
                      "so that the result is equal to reach the target {target}. Each number must be used exactly once. "
                      "Show your reasoning and put your final answer within \\boxed{{}}, e.g. \\boxed{{(1+2)*3=9}}.")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(rows: list, *, ascii_only: bool) -> bytes:
    """JSON lines as the original writers produced them (json.dumps default separators)."""
    return "".join(json.dumps(r, ensure_ascii=ascii_only) + "\n" for r in rows).encode("utf-8")


# ------------------------------------------------------------------------------------------- metadata
def load_metadata() -> tuple[dict, dict, list]:
    sums = {}
    for line in (POOLS_META / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest
    meta = json.loads((POOLS_META / "pools.json").read_text())
    order = json.loads((POOLS_META / "cdmix4_order.json").read_text())
    if meta["sources"] != {k: {f: v[f] for f in ("repo", "revision", "file", "bytes", "sha256")} for k, v in SOURCES.items()}:
        raise ValueError("pools.json and SOURCES disagree")
    return sums, meta, order


def library_versions() -> dict:
    versions = {}
    for name in ("pyarrow", "datasets", "pandas"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    return versions


# ------------------------------------------------------------------------------------------ upstream
def source_path(name: str, overrides: dict) -> Path:
    spec = SOURCES[name]
    if name in overrides:
        path = Path(overrides[name])
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=spec["repo"], filename=spec["file"], repo_type="dataset",
                                    revision=spec["revision"]))
    if not path.is_file() or path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"{path} is not {spec['repo']}@{spec['revision']}:{spec['file']} (size or SHA-256 differs)")
    return path


def read_rows(path: Path, columns: list, limit: int | None = None) -> list:
    """Rows of an upstream Parquet file as Python objects, in file order (as datasets.load_dataset yields them)."""
    import pyarrow.parquet as pq
    rows, handle = [], pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=8192, columns=columns):
        rows.extend(batch.to_pylist())
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


# ------------------------------------------------------------------ GradAlign prepare_data.py outputs
def verl_row(data_source: str, problem: str, answer: str, index: int) -> dict:
    """_process_to_verl of prepare_data.py; key order of a datasets 5.x row."""
    return {"data_source": data_source, "prompt": [{"role": "user", "content": BOXED_PREFIX + problem}],
            "reward_model": {"ground_truth": answer, "style": MATH_STYLE},
            "extra_info": {"split": "train", "index": index, "original_index": index}}


def prepared_webinstruct(path: Path) -> list:
    rows = read_rows(path, ["task_id", "original_question", "short_answer"], PREPARE_MAX_SAMPLES)
    return [verl_row("webinstruct", r["original_question"], str(r["short_answer"]), i) for i, r in enumerate(rows)]


def prepared_dapo(path: Path) -> list:
    rows = read_rows(path, ["prompt", "reward_model"], PREPARE_MAX_SAMPLES)
    out = []
    for i, r in enumerate(rows):
        prompt = r.get("prompt") or []
        text = prompt[0].get("content", "") if prompt and isinstance(prompt[0], dict) else ""
        problem = text.split("is the answer to the problem.\n\n")[-1].split(
            '\nRemember to put your answer on its own line after "Answer:".')[0]
        out.append(verl_row("dapo", problem, str((r.get("reward_model") or {}).get("ground_truth", "")), i))
    return out


def amc_rows(path: Path, year: int) -> list:
    """prepare_data.py --dataset amc22/amc23: the AMC 12 problems of one year, answer as an integer string."""
    rows = [r for r in read_rows(path, ["id", "problem", "answer", "url"]) if f"{year}_AMC" in r["url"]]
    return [verl_row(f"amc{year % 100}", r["problem"], str(int(r["answer"])), i) for i, r in enumerate(rows)]


# ------------------------------------------------------------------------------------ Countdown splits
def countdown_prompt(nums, target) -> str:
    return COUNTDOWN_TEMPLATE.format(nums="[" + ", ".join(str(n) for n in nums) + "]", target=target)


def countdown_row(nums, target, index: int, split: str) -> dict:
    prompt = countdown_prompt(nums, target)
    return {"data_source": "countdown", "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"ground_truth": prompt, "style": "rule-countdown"},
            "extra_info": {"split": split, "index": index, "original_index": index}}


def countdown_splits(path: Path) -> tuple[list, list, list]:
    """countdown4 (4,000), countdown4_val (200) and countdown4_test (500) from the four-number rows.

    Train and validation: the four-number rows in file order, shuffled by random.Random(42); the first 4,000
    and the next 200. Test: the distinct four-number (numbers, target) pairs not in train or validation,
    sorted, shuffled by random.Random(0); the first 500.
    """
    rows = read_rows(path, ["target", "nums"])
    four = [r for r in rows if len(r["nums"]) == 4]
    order = list(range(len(four)))
    random.Random(42).shuffle(order)
    train = [countdown_row(four[j]["nums"], four[j]["target"], i, "train") for i, j in enumerate(order[:4000])]
    val = [countdown_row(four[j]["nums"], four[j]["target"], i, "train") for i, j in enumerate(order[4000:4200])]
    used = {(tuple(four[j]["nums"]), four[j]["target"]) for j in order[:4200]}
    pool = sorted({(tuple(r["nums"]), r["target"]) for r in four} - used)
    random.Random(0).shuffle(pool)
    test = [countdown_row(list(nums), target, i, "test") for i, (nums, target) in enumerate(pool[:500])]
    return train, val, test


# ----------------------------------------------------------------------------------- mixtures
def cdmix4_rows(countdown4: list, webinstruct: list, dapo: list, order: list) -> list:
    """Released mix.py with datasets countdown4, webinstruct, dapo, counts 4,000 / 20,000 / 10,000 and line
    offsets 0: the concatenated entries in the recorded (unseeded) shuffle order, reindexed."""
    entries = countdown4[:4000] + webinstruct[:20000] + dapo[:10000]
    if sorted(order) != list(range(len(entries))):
        raise ValueError("cdmix4_order.json is not a permutation of the 34,000 entries")
    out = []
    for i, j in enumerate(order):
        entry = copy.deepcopy(entries[j])
        entry["extra_info"]["index"] = i
        entry["extra_info"]["original_index"] = i
        out.append(entry)
    return out


def dapo_noisy50_rows(dapo: list) -> list:
    """The corrupted DAPO pool of the original runs, which built it with a patched mix.py (datasets dapo, count
    40,000, --dedup --seed 0 --corrupt_fraction 0.5 --corrupt_seed 0) that the LESSER patch does not include."""
    seen, entries = set(), []
    for entry in copy.deepcopy(dapo[:PREPARE_MAX_SAMPLES]):
        key = json.dumps(entry["prompt"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    random.Random(0).shuffle(entries)
    n_corrupt = int(round(0.5 * len(entries)))
    flags = [True] * n_corrupt + [False] * (len(entries) - n_corrupt)
    random.Random(0).shuffle(flags)
    for entry, flag in zip(entries, flags):
        entry.setdefault("extra_info", {})["corrupted"] = flag
        if flag:
            entry["data_source"] = f"{entry['data_source']}_corrupted"
    for i, entry in enumerate(entries):
        entry["extra_info"]["index"] = i
        entry["extra_info"]["original_index"] = i
    return entries


def sorted_struct_fields(row: dict) -> dict:
    """Row as read back from a Parquet file whose struct fields were written in sorted order."""
    return {key: ([dict(sorted(item.items())) for item in value] if isinstance(value, list) else
                  dict(sorted(value.items())) if isinstance(value, dict) else value)
            for key, value in row.items()}


def sub512_rows(noisy: list) -> list:
    """The Figure 5 pool: 256 clean and 256 corrupted problems of the first 5,120-problem chunk.

    The chunk is dynamic_selection.py's chunk 0 of the corrupted DAPO pool with seed 42 (datasets shuffle:
    numpy default_rng(42).permutation); the sample is random.Random(0).sample of the clean, then of the
    corrupted problems in chunk order, sorted by pool index.
    """
    import numpy as np
    permutation = np.random.default_rng(42).permutation(len(noisy))
    chunk = [sorted_struct_fields(noisy[int(j)]) for j in permutation[:5120]]
    clean = [r for r in chunk if r["data_source"] == "dapo"]
    corrupted = [r for r in chunk if r["data_source"] == "dapo_corrupted"]
    rng = random.Random(0)
    sample = rng.sample(clean, 256) + rng.sample(corrupted, 256)
    sample.sort(key=lambda r: int(r["extra_info"]["index"]))
    return sample


# ------------------------------------------------------------------------------------ Parquet writers
def countdown_val_schema():
    """Arrow schema of the original countdown4_val Parquet file, which the test split was written with."""
    import pyarrow as pa
    return pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.field("element", pa.struct([("role", pa.string()), ("content", pa.string())])))),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("split", pa.string()), ("index", pa.int64()), ("original_index", pa.int64())])),
    ])


def write_parquet(rows: list, writer: str, path: Path) -> None:
    """The writer each original file came from (see pools.json)."""
    if writer == "datasets":
        import datasets
        datasets.Dataset.from_list(rows).to_parquet(str(path))
    elif writer == "pandas":
        import pandas as pd
        pd.DataFrame(rows).to_parquet(str(path))
    elif writer == "pyarrow_with_val_schema":
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows, schema=countdown_val_schema()), str(path))
    else:
        raise ValueError(f"unknown writer {writer}")


def content_digest(path: Path) -> str:
    """Columns, rows and values of a Parquet file, independent of its encoding and of struct field order."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    digest = hashlib.sha256(json.dumps(table.column_names).encode())
    for batch in table.to_batches(max_chunksize=4096):
        for row in batch.to_pylist():
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


# ------------------------------------------------------------------------------------ verification
class Verifier:
    def __init__(self, sums: dict, meta: dict, require_identical_bytes: bool):
        self.sums, self.meta, self.strict = sums, meta, require_identical_bytes
        self.versions = library_versions()
        self.notes: list = []

    def check(self, name: str, path: Path) -> str:
        """'identical' or 'content-identical'; raises on any other outcome."""
        expected = self.sums[name]
        if sha256_file(path) == expected:
            return "identical"
        if not name.endswith(".parquet"):
            raise ValueError(f"{name}: SHA-256 differs from SHA256SUMS")
        record = self.meta["files"][name]
        if content_digest(path) != record["content_sha256"]:
            raise ValueError(f"{name}: content differs from the original")
        same_versions = all(self.versions.get(lib) == version for lib, version in record["writer_versions"].items())
        if same_versions or self.strict:
            raise ValueError(f"{name}: content matches but bytes differ "
                             f"(writer versions {record['writer_versions']}, here {self.versions})")
        self.notes.append(f"{name}: content-identical; the original was written with {record['writer_versions']}")
        return "content-identical"


def install(payload_or_writer, name: str, out: Path, verifier: Verifier) -> str:
    """Write one pool file through a temporary path; keep it only if it verifies."""
    dst = out / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".build_", suffix=dst.suffix)
    os.close(fd)
    tmp = Path(tmp)
    try:
        if isinstance(payload_or_writer, bytes):
            tmp.write_bytes(payload_or_writer)
        else:
            payload_or_writer(tmp)
        status = verifier.check(name, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst)
    return status


def check_intermediate(meta: dict, name: str, payload: bytes) -> None:
    if sha256_bytes(payload) != meta["intermediates"][name]:
        raise ValueError(f"intermediate {name} differs from the one the runs used")


# ------------------------------------------------------------------------------------------ main
def build(out: Path, only: list, overrides: dict, prepared_dir: Path | None, verifier: Verifier, order: list) -> dict:
    meta, results = verifier.meta, {}
    needed = {src for pool in only for src in NEEDS[pool]}
    paths = {src: source_path(src, overrides) for src in needed
             if not (prepared_dir and src in ("webinstruct", "dapo") and (prepared_dir / src / "train.jsonl").is_file())}

    def prepared(name: str, builder):
        if prepared_dir and (prepared_dir / name / "train.jsonl").is_file():
            payload = (prepared_dir / name / "train.jsonl").read_bytes()
            check_intermediate(meta, f"{name}/train.jsonl", payload)
            return [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        rows = builder()
        check_intermediate(meta, f"{name}/train.jsonl", jsonl(rows, ascii_only=name == "countdown4"))
        return rows

    countdown = None
    if "countdown" in needed:
        train, val, test = countdown_splits(paths["countdown"])
        check_intermediate(meta, "countdown4/train.jsonl", jsonl(train, ascii_only=True))
        countdown = dict(train=train, val=val, test=test)
    if "countdown4_val" in only:
        results["countdown4_val/train.jsonl"] = install(jsonl(countdown["val"], ascii_only=True), "countdown4_val/train.jsonl", out, verifier)
        results["countdown4_val/train.parquet"] = install(lambda p: write_parquet(countdown["val"], "datasets", p),
                                                         "countdown4_val/train.parquet", out, verifier)
    if "countdown4_test" in only:
        results["countdown4_test/train.jsonl"] = install(jsonl(countdown["test"], ascii_only=True), "countdown4_test/train.jsonl", out, verifier)
        results["countdown4_test/train.parquet"] = install(
            lambda p: write_parquet(countdown["test"], "pyarrow_with_val_schema", p), "countdown4_test/train.parquet", out, verifier)
    dapo = prepared("dapo", lambda: prepared_dapo(paths["dapo"])) if "dapo" in needed else None
    if "cdmix4" in only:
        webinstruct = prepared("webinstruct", lambda: prepared_webinstruct(paths["webinstruct"]))
        rows = cdmix4_rows(countdown["train"], webinstruct, dapo, order)
        results["cdmix4/train.jsonl"] = install(jsonl(rows, ascii_only=False), "cdmix4/train.jsonl", out, verifier)
        results["cdmix4/train.parquet"] = install(lambda p: write_parquet(rows, "datasets", p), "cdmix4/train.parquet", out, verifier)
    if "dapo_noisy50_sub512" in only:
        noisy = dapo_noisy50_rows(dapo)
        check_intermediate(meta, "dapo_noisy50/train.jsonl", jsonl(noisy, ascii_only=False))
        rows = sub512_rows(noisy)
        name = "dapo_noisy50_sub512"
        results[f"{name}/train.jsonl"] = install(jsonl(rows, ascii_only=True), f"{name}/train.jsonl", out, verifier)
        results[f"{name}/train.parquet"] = install(lambda p: write_parquet(rows, "pandas", p), f"{name}/train.parquet", out, verifier)
    for year in (22, 23):
        name = f"amc{year}"
        if name not in only:
            continue
        rows = amc_rows(paths["amc"], 2000 + year)
        if year == 23:  # prepared with datasets 4.1.1, which writes struct fields in sorted order
            rows = [sorted_struct_fields(r) for r in rows]
        results[f"{name}/train.jsonl"] = install(jsonl(rows, ascii_only=False), f"{name}/train.jsonl", out, verifier)
        results[f"{name}/train.parquet"] = install(lambda p: write_parquet(rows, "datasets", p), f"{name}/train.parquet", out, verifier)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="build the pools into <out>/<name>/train.{jsonl,parquet}")
    target.add_argument("--check", type=Path, help="verify existing pools in <check>/<name>/")
    ap.add_argument("--only", nargs="+", choices=POOLS, default=list(POOLS))
    ap.add_argument("--source-file", action="append", default=[], metavar="NAME=PATH",
                    help=f"local copy of an upstream file ({', '.join(SOURCES)}); checked against its pinned SHA-256")
    ap.add_argument("--prepared-dir", type=Path, default=None,
                    help="reuse prepare_data.py outputs <dir>/{webinstruct,dapo}/train.jsonl (checked by SHA-256)")
    ap.add_argument("--require-identical-bytes", action="store_true",
                    help="fail unless every Parquet file is byte-identical (needs the writer versions of pools.json)")
    args = ap.parse_args()
    sums, meta, order = load_metadata()
    overrides = {}
    for item in args.source_file:
        name, _, path = item.partition("=")
        if name not in SOURCES or not path:
            ap.error(f"--source-file expects NAME=PATH with NAME in {sorted(SOURCES)}")
        overrides[name] = path
    verifier = Verifier(sums, meta, args.require_identical_bytes)
    only = [p for p in POOLS if p in args.only]
    if args.check:
        results = {}
        for pool in only:
            for suffix in ("jsonl", "parquet"):
                name = f"{pool}/train.{suffix}"
                path = args.check / name
                if not path.is_file():
                    raise FileNotFoundError(f"{path} is missing; build it with: python rl/build_pools.py --out {args.check}")
                results[name] = verifier.check(name, path)
    else:
        results = build(args.out, only, overrides, args.prepared_dir, verifier, order)
    for name, status in results.items():
        print(f"{status:18s} {name}")
    for note in verifier.notes:
        print("note:", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
